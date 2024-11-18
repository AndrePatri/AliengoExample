from lrhc_control.utils.sys_utils import PathsGetter
from lrhc_control.envs.lrhc_training_env_base import LRhcTrainingEnvBase

from control_cluster_bridge.utilities.shared_data.rhc_data import RobotState, RhcStatus
from control_cluster_bridge.utilities.math_utils_torch import world2base_frame, base2world_frame, w2hor_frame

import torch

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType

import os
from lrhc_control.utils.episodic_data import EpisodicData
from lrhc_control.utils.signal_smoother import ExponentialSignalSmoother
import math

from lrhc_control.utils.math_utils import check_capsize

class LinVelTrackBaseline(LRhcTrainingEnvBase):

    def __init__(self,
            namespace: str,
            actions_dim: int = 10,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1,
            use_gpu: bool = True,
            dtype: torch.dtype = torch.float32,
            debug: bool = True,
            override_agent_refs: bool = False,
            timeout_ms: int = 60000):
        
        env_name = "LinVelTrack"
        device = "cuda" if use_gpu else "cpu"

        episode_timeout_lb = 1024 # episode timeouts (including env substepping when action_repeat>1)
        episode_timeout_ub = 1024
        n_steps_task_rand_lb = 300 # agent refs randomization freq
        n_steps_task_rand_ub = 300 # lb not eq. to ub to remove correlations between episodes
        # across diff envs
        random_reset_freq = 10 # a random reset once every n-episodes (per env)
        n_preinit_steps = 1 # one steps of the controllers to properly initialize everything
        action_repeat = 1 # frame skipping (different agent action every action_repeat
        # env substeps)

        self._single_task_ref_per_episode=True # if True, the task ref is constant over the episode (ie
        # episodes are truncated when task is changed)
        self._add_prev_actions_stats_to_obs = True # add actions std, mean + last action over a horizon to obs
        self._add_contact_f_to_obs=True # add estimate vertical contact f to obs
        self._add_fail_idx_to_obs=True
        self._add_gn_rhc_loc=True
        self._use_linvel_from_rhc=False
        self._use_rhc_avrg_vel_pred=False
        self._use_vel_err_sig_smoother=False # whether to smooth vel error signal
        self._vel_err_smoother=None
        self._use_prob_based_stepping=False
        self._add_flight_info=True
        self._use_pos_control=False
        self._add_rhc_cmds_to_obs=True
        # temporarily creating robot state client to get some data
        robot_state_tmp = RobotState(namespace=namespace,
                                is_server=False, 
                                safe=False,
                                verbose=verbose,
                                vlevel=vlevel,
                                with_gpu_mirror=False,
                                with_torch_view=False)
        robot_state_tmp.run()
        rhc_status_tmp = RhcStatus(is_server=False,
                        namespace=namespace, 
                        verbose=verbose, 
                        vlevel=vlevel,
                        with_torch_view=False, 
                        with_gpu_mirror=False)
        rhc_status_tmp.run()
        n_jnts = robot_state_tmp.n_jnts()
        self._contact_names = robot_state_tmp.contact_names()
        self._n_contacts = len(self._contact_names)
        robot_state_tmp.close()
        rhc_status_tmp.close()

        # defining obs dim
        obs_dim=3 # normalized gravity vector in base frame
        obs_dim+=6 # meas twist in base frame
        obs_dim+=2*n_jnts # joint pos + vel
        if self._add_contact_f_to_obs:
            obs_dim+=3*self._n_contacts
        obs_dim+=6 # twist reference in base frame frame
        if self._add_fail_idx_to_obs:
            obs_dim+=1 # rhc controller failure index
        if self._add_gn_rhc_loc:
            obs_dim+=3
        if self._use_rhc_avrg_vel_pred:
            obs_dim+=6
        if self._add_prev_actions_stats_to_obs:
            obs_dim+=3*actions_dim# previous agent actions statistics (mean, std + last action)
        if self._add_flight_info:
            obs_dim+=self._n_contacts 
        if self._add_rhc_cmds_to_obs:
            obs_dim+=3*n_jnts 

        # health reward 
        self._health_value = 10.0

        # task tracking
        self._task_offset = 10.0
        self._task_scale = 1.5
        self._task_err_weights = torch.full((1, 6), dtype=dtype, device=device,
                            fill_value=0.0) 
        self._task_err_weights[0, 0] = 1.0
        self._task_err_weights[0, 1] = 1.0
        self._task_err_weights[0, 2] = 1.0
        self._task_err_weights[0, 3] = 1e-1
        self._task_err_weights[0, 4] = 1e-1
        self._task_err_weights[0, 5] = 1e-1

        # task pred tracking
        self._task_pred_offset = 0.0 # 10.0
        self._task_pred_scale = 1.5 # perc-based
        self._task_pred_err_weights = torch.full((1, 6), dtype=dtype, device=device,
                            fill_value=0.0) 
        self._task_pred_err_weights[0, 0] = 1.0
        self._task_pred_err_weights[0, 1] = 1.0
        self._task_pred_err_weights[0, 2] = 1.0
        self._task_pred_err_weights[0, 3] = 1e-1
        self._task_pred_err_weights[0, 4] = 1e-1
        self._task_pred_err_weights[0, 5] = 1e-1

        # fail idx
        self._rhc_fail_idx_offset = 0.0
        self._rhc_fail_idx_rew_scale = 0.0 # 1e-4
        self._rhc_fail_idx_scale=1

        # power penalty
        self._power_offset = 0.0 # 1.0
        self._power_scale = 0.0 # 10.0
        self._power_penalty_weights = torch.full((1, n_jnts), dtype=dtype, device=device,
                            fill_value=1.0)
        n_jnts_per_limb = round(n_jnts/self._n_contacts) # assuming same topology along limbs
        pow_weights_along_limb = [1.0]*n_jnts_per_limb
        for i in range(n_jnts_per_limb):
            self._power_penalty_weights[0, i*self._n_contacts:(self._n_contacts*(i+1))] = pow_weights_along_limb[i]
        self._power_penalty_weights_sum = torch.sum(self._power_penalty_weights).item()

        # jnt vel penalty 
        self._jnt_vel_offset = 0.0
        self._jnt_vel_scale = 0.0 # 0.3
        self._jnt_vel_penalty_weights = torch.full((1, n_jnts), dtype=dtype, device=device,
                            fill_value=1.0)
        jnt_vel_weights_along_limb = [1.0]*n_jnts_per_limb
        for i in range(round(n_jnts/self._n_contacts)):
            self._jnt_vel_penalty_weights[0, i*self._n_contacts:(self._n_contacts*(i+1))] = jnt_vel_weights_along_limb[i]
        self._jnt_vel_penalty_weights_sum = torch.sum(self._jnt_vel_penalty_weights).item()
        
        # task rand
        self._use_pof0 = False
        self._pof0 = 0.1
        self._twist_ref_lb = torch.full((1, 6), dtype=dtype, device=device,
                            fill_value=-0.8) 
        self._twist_ref_ub = torch.full((1, 6), dtype=dtype, device=device,
                            fill_value=0.8)
        
        # task reference parameters (specified in world frame)
        self.max_ref=1.0
        self._twist_ref_lb[0, 0] = -self.max_ref
        self._twist_ref_lb[0, 1] = -self.max_ref
        self._twist_ref_lb[0, 2] = 0.0
        self._twist_ref_ub[0, 0] = self.max_ref
        self._twist_ref_ub[0, 1] = self.max_ref
        self._twist_ref_ub[0, 2] = 0.0
        # angular vel
        self._twist_ref_lb[0, 3] = 0.0
        self._twist_ref_lb[0, 4] = 0.0
        self._twist_ref_lb[0, 5] = 0.0
        self._twist_ref_ub[0, 3] = 0.0
        self._twist_ref_ub[0, 4] = 0.0
        self._twist_ref_ub[0, 5] = 0.0

        self._twist_ref_offset = (self._twist_ref_ub + self._twist_ref_lb)/2.0
        self._twist_ref_scale = (self._twist_ref_ub - self._twist_ref_lb)/2.0

        self._this_child_path = os.path.abspath(__file__)

        super().__init__(namespace=namespace,
                    obs_dim=obs_dim,
                    actions_dim=actions_dim,
                    episode_timeout_lb=episode_timeout_lb,
                    episode_timeout_ub=episode_timeout_ub,
                    n_steps_task_rand_lb=n_steps_task_rand_lb,
                    n_steps_task_rand_ub=n_steps_task_rand_ub,
                    random_reset_freq=random_reset_freq,
                    use_random_safety_reset=True,
                    action_repeat=action_repeat,
                    env_name=env_name,
                    n_preinit_steps=n_preinit_steps,
                    verbose=verbose,
                    vlevel=vlevel,
                    use_gpu=use_gpu,
                    dtype=dtype,
                    debug=debug,
                    override_agent_refs=override_agent_refs,
                    timeout_ms=timeout_ms,
                    srew_drescaling=True,
                    use_act_mem_bf=self._add_prev_actions_stats_to_obs,
                    act_membf_size=30)

        self._is_substep_rew[0]=False
        # self._is_substep_rew[1]=True

        # custom db info 
        agent_twist_ref = self._agent_refs.rob_refs.root_state.get(data_type="twist",gpu=False)
        agent_twist_ref_data = EpisodicData("AgentTwistRefs", agent_twist_ref, 
            ["v_x", "v_y", "v_z", "omega_x", "omega_y", "omega_z"],
            ep_vec_freq=self._vec_ep_freq_metrics_db)
        self._add_custom_db_info(db_data=agent_twist_ref_data)
        rhc_fail_idx = EpisodicData("RhcFailIdx", self._rhc_fail_idx(gpu=False), ["rhc_fail_idx"],
            ep_vec_freq=self._vec_ep_freq_metrics_db)
        self._add_custom_db_info(db_data=rhc_fail_idx)

        # other static db info 
        self.custom_db_info["add_last_action_to_obs"] = self._add_prev_actions_stats_to_obs
        self.custom_db_info["use_pof0"] = self._use_pof0
        self.custom_db_info["pof0"] = self._pof0
        self.custom_db_info["action_repeat"] = self._action_repeat
        self.custom_db_info["add_flight_info"] = self._add_flight_info

    def _set_jnts_blacklist_pattern(self):
        self._jnt_q_blacklist_patterns=["wheel"]

    def _custom_post_init(self):

        device = "cuda" if self._use_gpu else "cpu"

        self._update_jnt_blacklist()

        self._n_noisy_envs=math.ceil(self._n_envs*1/100)
        if not self._use_pos_control:
            if not self._use_prob_based_stepping:
                self._is_continuous_actions[:,6:10]=False
        else:
            self._is_continuous_actions[:,6:10]=True

        # overriding parent's defaults 
        self._reward_thresh_lb[:, :]=0 # (neg rewards can be nasty, especially if they all become negative)
        self._reward_thresh_ub[:, :]=1e6
        # self._reward_thresh_lb[:, 1]=-1e6
        # self._reward_thresh_ub[:, 1]=1e6

        # obs bounds
        self._obs_threshold_lb = -1e3 # used for clipping observations
        self._obs_threshold_ub = 1e3

        # actions bounds
        v_cmd_max = 3*self.max_ref
        omega_cmd_max = 3*self.max_ref
        self._actions_lb[:, 0:3] = -v_cmd_max 
        self._actions_ub[:, 0:3] = v_cmd_max  
        self._actions_lb[:, 3:6] = -omega_cmd_max # twist cmds
        self._actions_ub[:, 3:6] = omega_cmd_max  
        if not self._use_pos_control:
            if self._use_prob_based_stepping:
                self._actions_lb[:, 6:10] = 0.0 # contact flags
                self._actions_ub[:, 6:10] = 1.0 
            else:
                self._actions_lb[:, 6:10] = -1.0 
                self._actions_ub[:, 6:10] = 1.0 
        else:
            self._actions_lb[:, 6:10] = 0.0
            self._actions_ub[:, 6:10] = 0.5
        
        # assign obs bounds
        obs_names=self._get_obs_names()
        obs_patterns=["gn",
            "linvel",
            "omega",
            "q_jnt",
            "v_jnt",
            "fc",
            "rhc_fail",
            "rhc_cmd_q",
            "rhc_cmd_v",
            "rhc_cmd_eff",
            "flight_pos"
            ]
        obs_ubs=[1.0,
            5*self.max_ref,
            5*self.max_ref,
            2*torch.pi,
            30.0,
            2.0,
            1.0,
            2*torch.pi,
            30.0,
            200.0,
            self._n_nodes_rhc.mean().item()]
        obs_lbs=[-1.0,
            -5*self.max_ref,
            -5*self.max_ref,
            -2*torch.pi,
            -30.0,
            -2.0,
            0.0,
            -2*torch.pi,
            -30.0,
            -200.0,
            0.0]
        obs_bounds = {name: (lb, ub) for name, lb, ub in zip(obs_patterns, obs_lbs, obs_ubs)}
        
        for i in range(len(obs_names)):
            obs_name=obs_names[i]
            for pattern in obs_patterns:
                if pattern in obs_name:
                    lb=obs_bounds[pattern][0]
                    ub=obs_bounds[pattern][1]
                    self._obs_lb[:, i]=lb
                    self._obs_ub[:, i]=ub
                    break
        
        # handle action memory buffer in obs
        i=0
        prev_actions_idx = next((i for i, s in enumerate(obs_names) if "_prev" in s), None)
        prev_actions_mean_idx=next((i for i, s in enumerate(obs_names) if "_avrg" in s), None)
        prev_actions_std_idx=next((i for i, s in enumerate(obs_names) if "_std" in s), None)

        self._obs_lb[:, prev_actions_idx:prev_actions_idx+self.actions_dim()]=self._actions_lb
        self._obs_ub[:, prev_actions_idx:prev_actions_idx+self.actions_dim()]=self._actions_ub
        self._obs_lb[:, prev_actions_std_idx:prev_actions_std_idx+self.actions_dim()]=0
        self._obs_ub[:, prev_actions_std_idx:prev_actions_std_idx+self.actions_dim()]=self.get_actions_scale()
        self._obs_lb[:, prev_actions_mean_idx:prev_actions_mean_idx+self.actions_dim()]=self._actions_lb
        self._obs_ub[:, prev_actions_mean_idx:prev_actions_mean_idx+self.actions_dim()]=self._actions_ub

        # some aux data to avoid allocations at training runtime
        self._rhc_twist_cmd_rhc_world=self._robot_state.root_state.get(data_type="twist",gpu=self._use_gpu).detach().clone()
        self._rhc_twist_cmd_rhc_h=self._rhc_twist_cmd_rhc_world.detach().clone()
        self._agent_twist_ref_current_w=self._rhc_twist_cmd_rhc_world.detach().clone()
        self._agent_twist_ref_current_base_loc=self._rhc_twist_cmd_rhc_world.detach().clone()
        self._substep_avrg_root_twist_base_loc=self._rhc_twist_cmd_rhc_world.detach().clone()
        self._step_avrg_root_twist_base_loc=self._rhc_twist_cmd_rhc_world.detach().clone()
        self._root_twist_avrg_rhc_base_loc=self._rhc_twist_cmd_rhc_world.detach().clone()
        self._root_twist_avrg_rhc_base_loc_next=self._rhc_twist_cmd_rhc_world.detach().clone()
        
        self._random_thresh_contacts=torch.rand((self._n_envs,self._n_contacts), device=device)
        # task aux data
        self._task_err_scaling = torch.zeros((self._n_envs, 1),dtype=self._dtype,device=device)

        self._pof1_b = torch.full(size=(self._n_envs,1),dtype=self._dtype,device=device,fill_value=1-self._pof0)
        self._bernoulli_coeffs = self._pof1_b.clone()
        self._bernoulli_coeffs[:, :] = 1.0

        if self._add_prev_actions_stats_to_obs:
            self._defaut_act_buffer_action[:, :] = (self._actions_ub+self._actions_lb)/2.0

        if self._use_vel_err_sig_smoother:
            vel_err_proxy=self._robot_state.root_state.get(data_type="twist",gpu=self._use_gpu).detach().clone()
            self._smoothing_horizon=0.1
            self._target_smoothing=0.01
            self._vel_err_smoother=ExponentialSignalSmoother(
                name="VelErrorSmoother",
                signal=vel_err_proxy, # same dimension of vel error
                update_dt=self._substep_dt,
                smoothing_horizon=self._smoothing_horizon,
                target_smoothing=self._target_smoothing,
                debug=self._is_debug,
                dtype=self._dtype,
                use_gpu=self._use_gpu)
            self.custom_db_info["smoothing_horizon"]=self._smoothing_horizon
            self.custom_db_info["target_smoothing"]=self._target_smoothing

    def get_file_paths(self):
        paths=super().get_file_paths()
        paths.append(self._this_child_path)        
        return paths

    def get_aux_dir(self):

        aux_dirs = []
        path_getter = PathsGetter()

        aux_dirs.append(path_getter.RHCDIR)
        return aux_dirs

    def _get_reward_scaling(self):
        if self._single_task_ref_per_episode:
            return self._n_steps_task_rand_ub
        else:
            return self._episode_timeout_ub
        
    def _check_sub_truncations(self):
        # overrides parent
        sub_truncations = self._sub_truncations.get_torch_mirror(gpu=self._use_gpu)
        sub_truncations[:, 0:1] = self._ep_timeout_counter.time_limits_reached()
        if self._single_task_ref_per_episode:
            sub_truncations[:, 1:2] = self._task_rand_counter.time_limits_reached()
    
    def _check_sub_terminations(self):
        # default behaviour-> to be overriden by child
        sub_terminations = self._sub_terminations.get_torch_mirror(gpu=self._use_gpu)
        robot_q_meas = self._robot_state.root_state.get(data_type="q",gpu=self._use_gpu)
        robot_q_pred = self._rhc_cmds.root_state.get(data_type="q",gpu=self._use_gpu)

        # terminate when either the real robot or the prediction from the MPC are capsized
        check_capsize(quat=robot_q_meas,max_angle=self._max_pitch_angle,
            output_t=self._is_capsized)
        check_capsize(quat=robot_q_pred,max_angle=self._max_pitch_angle,
            output_t=self._is_rhc_capsized)
        
        sub_terminations[:, 0:1] = self._rhc_status.fails.get_torch_mirror(gpu=self._use_gpu)
        sub_terminations[:, 1:2] = self._is_capsized
        sub_terminations[:, 2:3] = self._is_rhc_capsized

    def _custom_reset(self): # reset if truncated
        if self._single_task_ref_per_episode:
            return None
        else:
            return self._truncations.get_torch_mirror(gpu=self._use_gpu).cpu()
        # return self._truncations.get_torch_mirror(gpu=self._use_gpu).cpu()
    
    def reset(self):
        super().reset()

    def _pre_step(self): 
        pass

    def _custom_post_step(self,episode_finished):
        # executed after checking truncations and terminations and remote env reset
        if self._use_gpu:
            time_to_rand_or_ep_finished = torch.logical_or(self._task_rand_counter.time_limits_reached().cuda(),episode_finished)
            self.randomize_task_refs(env_indxs=time_to_rand_or_ep_finished.flatten())
        else:
            time_to_rand_or_ep_finished = torch.logical_or(self._task_rand_counter.time_limits_reached(),episode_finished)
            self.randomize_task_refs(env_indxs=time_to_rand_or_ep_finished.flatten())
        # task refs are randomized in world frame -> we rotate them in base local
        # (not super efficient, we should do it just for the finished envs)
        self._update_loc_twist_refs()

        if self._vel_err_smoother is not None: # reset smoother
            self._vel_err_smoother.reset(to_be_reset=episode_finished.flatten())

    def _custom_post_substp_pre_rew(self):
        self._update_loc_twist_refs()
        
    def _custom_post_substp_post_rew(self):
        pass
    
    def _update_loc_twist_refs(self):
        # get fresh robot orientation
        if not self._override_agent_refs:
            robot_q = self._robot_state.root_state.get(data_type="q",gpu=self._use_gpu)
            # rotate agent ref from world to robot base
            world2base_frame(t_w=self._agent_twist_ref_current_w, q_b=robot_q, 
                t_out=self._agent_twist_ref_current_base_loc)
            # write it to agent refs tensors
            self._agent_refs.rob_refs.root_state.set(data_type="twist", data=self._agent_twist_ref_current_base_loc,
                                                gpu=self._use_gpu)
        
    def _apply_actions_to_rhc(self):
        
        agent_action = self.get_actions() # see _get_action_names() to get 
        # the meaning of each component of this tensor

        rhc_latest_twist_cmd = self._rhc_refs.rob_refs.root_state.get(data_type="twist", gpu=self._use_gpu)
        rhc_latest_contact_ref = self._rhc_refs.contact_flags.get_torch_mirror(gpu=self._use_gpu)
        rhc_latest_pos_ref = self._rhc_refs.rob_refs.contact_pos.get(data_type="p_z", gpu=self._use_gpu)
        rhc_q=self._rhc_cmds.root_state.get(data_type="q",gpu=self._use_gpu) # this is always 
        # avaialble

        # reference twist for MPC is assumed to always be specified in MPC's 
        # horizontal frame, while agent actions are interpreted as in MPC's
        # base frame -> we need to rotate the actions into the horizontal frame
        base2world_frame(t_b=agent_action[:, 0:6],q_b=rhc_q,t_out=self._rhc_twist_cmd_rhc_world)
        w2hor_frame(t_w=self._rhc_twist_cmd_rhc_world,q_b=rhc_q,t_out=self._rhc_twist_cmd_rhc_h)

        rhc_latest_twist_cmd[:, 0:6] = self._rhc_twist_cmd_rhc_h
        
        # self._rhc_refs.rob_refs.root_state.set(data_type="p", data=rhc_latest_p_ref,
        #                                     gpu=self._use_gpu)
        self._rhc_refs.rob_refs.root_state.set(data_type="twist", data=rhc_latest_twist_cmd,
            gpu=self._use_gpu) 
        
        # contact flags
        if not self._use_pos_control:
            if self._use_prob_based_stepping:
                # encode actions as probs
                self._random_thresh_contacts.uniform_() # random values in-place between 0 and 1
                rhc_latest_contact_ref[:, :] = agent_action[:, 6:10] >= self._random_thresh_contacts  # keep contact with 
                # probability agent_action[:, 6:10]
            else: # just use a threshold
                rhc_latest_contact_ref[:, :] = agent_action[:, 6:10] > 0
            # actually apply actions to controller
        else:
            rhc_latest_pos_ref[:, :] = agent_action[:, 6:10]

        if self._use_gpu:
            # GPU->CPU --> we cannot use asynchronous data transfer since it's unsafe
            self._rhc_refs.rob_refs.root_state.synch_mirror(from_gpu=True,non_blocking=False) # write from gpu to cpu mirror
            self._rhc_refs.contact_flags.synch_mirror(from_gpu=True,non_blocking=False)
            self._rhc_refs.rob_refs.contact_pos.synch_mirror(from_gpu=True,non_blocking=False)

        self._rhc_refs.rob_refs.root_state.synch_all(read=False, retry=True) # write mirror to shared mem
        self._rhc_refs.contact_flags.synch_all(read=False, retry=True)
        self._rhc_refs.rob_refs.contact_pos.synch_all(read=False, retry=True)

    def _fill_obs(self,
            obs: torch.Tensor):

        # measured stuff
        robot_gravity_norm_base_loc = self._robot_state.root_state.get(data_type="gn",gpu=self._use_gpu)
        robot_twist_meas_base_loc = self._robot_state.root_state.get(data_type="twist",gpu=self._use_gpu)
        robot_jnt_q_meas = self._robot_state.jnts_state.get(data_type="q",gpu=self._use_gpu)
        if self._jnt_q_blacklist_idxs is not None: # we don't want to read joint pos from blacklist
            robot_jnt_q_meas[:, self._jnt_q_blacklist_idxs]=0.0
        robot_jnt_v_meas = self._robot_state.jnts_state.get(data_type="v",gpu=self._use_gpu)
        
        # twist estimate from mpc
        robot_twist_rhc_base_loc_next = self._rhc_cmds.root_state.get(data_type="twist",gpu=self._use_gpu)
        # cmds for jnt imp to be applied next
        robot_jnt_q_rhc_applied_next=self._rhc_cmds.jnts_state.get(data_type="q",gpu=self._use_gpu)
        robot_jnt_v_rhc_applied_next=self._rhc_cmds.jnts_state.get(data_type="v",gpu=self._use_gpu)
        robot_jnt_eff_rhc_applied_next=self._rhc_cmds.jnts_state.get(data_type="eff",gpu=self._use_gpu)

        flight_info_now = self._rhc_refs.flight_info.get(data_type="all",gpu=self._use_gpu)

        # refs
        agent_twist_ref = self._agent_refs.rob_refs.root_state.get(data_type="twist",gpu=self._use_gpu)

        next_idx=0
        obs[:, next_idx:(next_idx+3)] = robot_gravity_norm_base_loc # norm. gravity vector in base frame
        next_idx+=3
        if self._use_linvel_from_rhc:
            obs[:, next_idx:(next_idx+3)] = robot_twist_rhc_base_loc_next[:, 0:3]
        else:
            obs[:, next_idx:(next_idx+3)] = robot_twist_meas_base_loc[:, 0:3]
        next_idx+=3
        obs[:, next_idx:(next_idx+3)] = robot_twist_meas_base_loc[:, 3:6]
        next_idx+=3
        obs[:, next_idx:(next_idx+self._n_jnts)] = robot_jnt_q_meas # meas jnt pos
        next_idx+=self._n_jnts
        obs[:, next_idx:(next_idx+self._n_jnts)] = robot_jnt_v_meas # meas jnt vel
        next_idx+=self._n_jnts
        obs[:, next_idx:(next_idx+6)] = agent_twist_ref # high lev agent refs to be tracked
        next_idx+=6
        if self._add_contact_f_to_obs:
            n_forces=3*len(self._contact_names)
            obs[:, next_idx:(next_idx+n_forces)] = self._rhc_cmds.contact_wrenches.get(data_type="f",gpu=self._use_gpu)
            next_idx+=n_forces
        if self._add_fail_idx_to_obs:
            obs[:, next_idx:(next_idx+1)] = self._rhc_fail_idx(gpu=self._use_gpu)
            next_idx+=1
        if self._add_gn_rhc_loc:
            obs[:, next_idx:(next_idx+3)] = self._rhc_cmds.root_state.get(data_type="gn",gpu=self._use_gpu)
            next_idx+=3
        if self._use_rhc_avrg_vel_pred:
            self._get_avrg_rhc_root_twist(out=self._root_twist_avrg_rhc_base_loc,base_loc=True)
            obs[:, next_idx:(next_idx+6)] = self._root_twist_avrg_rhc_base_loc
            next_idx+=6
        if self._add_flight_info:
            flight_info_size=len(self._contact_names)
            obs[:, next_idx:(next_idx+flight_info_size)] = flight_info_now[:, 0:flight_info_size]
            next_idx+=flight_info_size
        if self._add_rhc_cmds_to_obs:
            obs[:, next_idx:(next_idx+self._n_jnts)] = robot_jnt_q_rhc_applied_next
            next_idx+=self._n_jnts
            obs[:, next_idx:(next_idx+self._n_jnts)] = robot_jnt_v_rhc_applied_next
            next_idx+=self._n_jnts
            obs[:, next_idx:(next_idx+self._n_jnts)] = robot_jnt_eff_rhc_applied_next
            next_idx+=self._n_jnts
        if self._add_prev_actions_stats_to_obs:
            self._prev_act_idx=next_idx
            obs[:, next_idx:(next_idx+self.actions_dim())]=self._act_mem_buffer.get(idx=0) # last obs
            next_idx+=self.actions_dim()
            obs[:, next_idx:(next_idx+self.actions_dim())]=self._act_mem_buffer.mean(clone=False)
            next_idx+=self.actions_dim()
            obs[:, next_idx:(next_idx+self.actions_dim())]=self._act_mem_buffer.std(clone=False)
            next_idx+=self.actions_dim()
        
    def _get_custom_db_data(self, 
            episode_finished):
        episode_finished = episode_finished.cpu()
        self.custom_db_data["AgentTwistRefs"].update(new_data=self._agent_refs.rob_refs.root_state.get(data_type="twist",
                                                                                            gpu=False), 
                                    ep_finished=episode_finished)
        self.custom_db_data["RhcFailIdx"].update(new_data=self._rhc_fail_idx(gpu=False), 
                                    ep_finished=episode_finished)
    
    def _drained_mech_pow(self, jnts_vel, jnts_effort):
        mech_pow_jnts=(jnts_effort*jnts_vel)*self._power_penalty_weights
        mech_pow_jnts.clamp_(0.0,torch.inf) # do not account for regenerative power
        drained_mech_pow_tot = torch.sum(mech_pow_jnts, dim=1, keepdim=True)/self._power_penalty_weights_sum
        return drained_mech_pow_tot

    def _cost_of_transport(self, jnts_vel, jnts_effort, v_ref_norm):
        drained_mech_pow=self._drained_mech_pow(jnts_vel=jnts_vel,
            jnts_effort=jnts_effort)
        robot_weight=self._rhc_robot_weight
        return drained_mech_pow/(robot_weight*(v_ref_norm+1e-3))

    def _jnt_vel_penalty(self, jnts_vel):
        task_ref = self._agent_refs.rob_refs.root_state.get(data_type="twist",gpu=self._use_gpu) # high level agent refs (hybrid twist)
        delta = 0.01 # [m/s]
        ref_norm = task_ref.norm(dim=1,keepdim=True)
        above_thresh = ref_norm >= delta
        jnts_vel_sqrd=jnts_vel*jnts_vel
        jnts_vel_sqrd[above_thresh.flatten(), :]=0 # no penalty for refs > thresh
        weighted_jnt_vel = torch.sum((jnts_vel_sqrd)*self._jnt_vel_penalty_weights, dim=1, keepdim=True)/self._jnt_vel_penalty_weights_sum
        return weighted_jnt_vel
    
    def _task_perc_err_wms(self, task_ref, task_meas, weights):
        ref_norm = task_ref.norm(dim=1,keepdim=True)
        self._task_err_scaling[:, :] = ref_norm+1e-2
        task_perc_err=self._task_err_wms(task_ref=task_ref, task_meas=task_meas, 
            scaling=self._task_err_scaling, weights=weights)
        perc_err_thresh=2.0 # no more than perc_err_thresh*100 % error on each dim
        task_perc_err.clamp_(0.0,perc_err_thresh**2) 
        return task_perc_err
    
    def _task_err_wms(self, task_ref, task_meas, scaling, weights):
        task_error = (task_meas-task_ref)
        scaled_error=task_error/scaling
        if self._vel_err_smoother is not None:
            self._vel_err_smoother.update(new_signal=scaled_error,
                ep_finished=None # reset done externally
                )
            scaled_error=self._vel_err_smoother.get()
        task_wmse = torch.sum(scaled_error*scaled_error*weights, dim=1, keepdim=True)/torch.sum(weights).item()
        return task_wmse # weighted mean square error (along task dimension)
    
    def _task_perc_err_lin(self, task_ref, task_meas, weights):
        task_wmse = self._task_perc_err_wms(task_ref=task_ref, task_meas=task_meas,
            weights=weights)
        return task_wmse.sqrt()
    
    def _task_err_lin(self, task_ref, task_meas, weights):
        self._task_err_scaling[:, :] = 1
        task_wmse = self._task_err_wms(task_ref=task_ref, task_meas=task_meas, 
            scaling=self._task_err_scaling, weights=weights)
        return task_wmse.sqrt()
    
    def _rhc_fail_idx(self, gpu: bool):
        rhc_fail_idx = self._rhc_status.rhc_fail_idx.get_torch_mirror(gpu=gpu)
        return self._rhc_fail_idx_scale*rhc_fail_idx
    
    def _compute_step_rewards(self):
        task_error_fun = self._task_perc_err_lin
        # task_error_fun= self._task_perc_err_wms
        # task_error_fun = self._task_err_lin
        sub_rewards = self._sub_rewards.get_torch_mirror(gpu=self._use_gpu)

        agent_task_ref_base_loc = self._agent_refs.rob_refs.root_state.get(data_type="twist",gpu=self._use_gpu) # high level agent refs (hybrid twist)
        self._get_avrg_step_root_twist(out=self._step_avrg_root_twist_base_loc, base_loc=True)
        task_error = task_error_fun(task_meas=self._step_avrg_root_twist_base_loc, 
            task_ref=agent_task_ref_base_loc,
            weights=self._task_err_weights)

        fail_idx=self._rhc_fail_idx(gpu=self._use_gpu)
        sub_rewards[:, 0:1] =  self._task_offset*(1-fail_idx)*torch.exp(-self._task_scale*task_error)

        if self._use_rhc_avrg_vel_pred:
            self._get_avrg_rhc_root_twist(out=self._root_twist_avrg_rhc_base_loc_next,base_loc=True) # get estimated avrg vel 
            # from MPC after stepping
            task_pred_error=task_error_fun(task_meas=self._root_twist_avrg_rhc_base_loc_next, 
                task_ref=agent_task_ref_base_loc,
                weights=self._task_pred_err_weights)
            sub_rewards[:, 1:2] = self._task_pred_offset*torch.exp(-self._task_pred_scale*task_pred_error)

        # sub_rewards[:, 0:1] = self._task_offset-self._task_scale*task_error

    def _compute_substep_rewards(self):
        task_error_fun = self._task_perc_err_lin

        # jnts_vel = self._robot_state.jnts_state.get(data_type="v",gpu=self._use_gpu)
        # jnts_effort = self._robot_state.jnts_state.get(data_type="eff",gpu=self._use_gpu)
        # ref_norm=torch.norm(agent_task_ref_base_loc, dim=1, keepdim=True)
        # CoT=self._cost_of_transport(jnts_vel=jnts_vel,jnts_effort=jnts_effort,v_ref_norm=ref_norm)
        # fail_idx=self._rhc_fail_idx(gpu=self._use_gpu)
        # if self._use_rhc_avrg_vel_pred:
        #     agent_task_ref_base_loc = self._agent_refs.rob_refs.root_state.get(data_type="twist",gpu=self._use_gpu) # high level agent refs (hybrid twist)
        #     self._get_avrg_rhc_root_twist(out=self._root_twist_avrg_rhc_base_loc_next,base_loc=True) # get estimated avrg vel 
        #     # from MPC after stepping
        #     task_pred_error=task_error_fun(task_meas=self._root_twist_avrg_rhc_base_loc_next, 
        #         task_ref=agent_task_ref_base_loc,
        #         weights=self._task_pred_err_weights)
        #     self._substep_rewards[:, 1:2] = self._task_pred_offset*torch.exp(-self._task_pred_scale*task_pred_error)

        # self._substep_rewards[:, 1:2] =self._power_offset-self._power_scale*CoT
        # self._substep_rewards[:, 2:3] = self._power_offset - self._power_scale * weighted_mech_power
        # self._substep_rewards[:, 3:4] = self._jnt_vel_offset - self._jnt_vel_scale * weighted_jnt_vel
        # self._substep_rewards[:, 4:5] = self._rhc_fail_idx_offset - self._rhc_fail_idx_rew_scale* self._rhc_fail_idx(gpu=self._use_gpu)
        # self._substep_rewards[:, 1:2] = -fail_idx # health reward
        
    def _randomize_task_refs(self,
        env_indxs: torch.Tensor = None):

        # we randomize the reference in world frame, since it's much more intuitive 
        # (it will be rotated in base frame when provided to the agent and used for rew 
        # computation)
        
        if self._use_pof0: # sample from bernoulli distribution
            torch.bernoulli(input=self._pof1_b,out=self._bernoulli_coeffs) # by default bernoulli_coeffs are 1 if not _use_pof0
        if env_indxs is None:
            random_uniform=torch.full_like(self._agent_twist_ref_current_w, fill_value=0.0)
            torch.nn.init.uniform_(random_uniform, a=-1, b=1)
            self._agent_twist_ref_current_w[:, :] = random_uniform*self._twist_ref_scale + self._twist_ref_offset
            self._agent_twist_ref_current_w[:, :] = self._agent_twist_ref_current_w*self._bernoulli_coeffs
        else:
            random_uniform=torch.full_like(self._agent_twist_ref_current_w[env_indxs, :], fill_value=0.0)
            torch.nn.init.uniform_(random_uniform, a=-1, b=1)
            self._agent_twist_ref_current_w[env_indxs, :] = random_uniform * self._twist_ref_scale + self._twist_ref_offset
            self._agent_twist_ref_current_w[env_indxs, :] = self._agent_twist_ref_current_w[env_indxs, :]*self._bernoulli_coeffs[env_indxs, :]

    def _get_obs_names(self):

        obs_names = [""] * self.obs_dim()

        # proprioceptive stream of obs
        next_idx=0
        obs_names[0] = "gn_x_base_loc"
        obs_names[1] = "gn_y_base_loc"
        obs_names[2] = "gn_z_base_loc"
        next_idx+=3
        obs_names[next_idx] = "linvel_x_base_loc"
        obs_names[next_idx+1] = "linvel_y_base_loc"
        obs_names[next_idx+2] = "linvel_z_base_loc"
        obs_names[next_idx+3] = "omega_x_base_loc"
        obs_names[next_idx+4] = "omega_y_base_loc"
        obs_names[next_idx+5] = "omega_z_base_loc"
        next_idx+=6
        jnt_names=self.get_observed_joints()
        for i in range(self._n_jnts): # jnt obs (pos):
            obs_names[next_idx+i] = f"q_jnt_{jnt_names[i]}"
        next_idx+=self._n_jnts
        for i in range(self._n_jnts): # jnt obs (v):
            obs_names[next_idx+i] = f"v_jnt_{jnt_names[i]}"
        next_idx+=self._n_jnts

        # references
        obs_names[next_idx] = "linvel_x_ref_base_loc"
        obs_names[next_idx+1] = "linvel_y_ref_base_loc"
        obs_names[next_idx+2] = "linvel_z_ref_base_loc"
        obs_names[next_idx+3] = "omega_x_ref_base_loc"
        obs_names[next_idx+4] = "omega_y_ref_base_loc"
        obs_names[next_idx+5] = "omega_z_ref_base_loc"
        next_idx+=6

        # contact forces
        if self._add_contact_f_to_obs:
            i = 0
            for contact in self._contact_names:
                obs_names[next_idx+i] = f"fc_{contact}_x_base_loc"
                obs_names[next_idx+i+1] = f"fc_{contact}_y_base_loc"
                obs_names[next_idx+i+2] = f"fc_{contact}_z_base_loc"
                i+=3        
            next_idx+=3*len(self._contact_names)

        # data directly from MPC
        if self._add_fail_idx_to_obs:
            obs_names[next_idx] = "rhc_fail_idx"
            next_idx+=1
        if self._add_gn_rhc_loc:
            obs_names[next_idx] = "gn_x_rhc_base_loc"
            obs_names[next_idx+1] = "gn_y_rhc_base_loc"
            obs_names[next_idx+2] = "gn_z_rhc_base_loc"
            next_idx+=3
        if self._use_rhc_avrg_vel_pred:
            obs_names[next_idx] = "linvel_x_avrg_rhc"
            obs_names[next_idx+1] = "linvel_y_avrg_rhc"
            obs_names[next_idx+2] = "linvel_z_avrg_rhc"
            obs_names[next_idx+3] = "omega_x_avrg_rhc"
            obs_names[next_idx+4] = "omega_y_avrg_rhc"
            obs_names[next_idx+5] = "omega_z_avrg_rhc"
            next_idx+=6
        if self._add_flight_info:
            flight_info_size=len(self._contact_names)
            for i in range(len(self._contact_names)):
                obs_names[next_idx+i] = "flight_pos_"+ self._contact_names[i]
            next_idx+=flight_info_size
        if self._add_rhc_cmds_to_obs:
            for i in range(self._n_jnts): # jnt obs (pos):
                obs_names[next_idx+i] = f"rhc_cmd_q_{jnt_names[i]}"
            next_idx+=self._n_jnts
            for i in range(self._n_jnts): # jnt obs (pos):
                obs_names[next_idx+i] = f"rhc_cmd_v_{jnt_names[i]}"
            next_idx+=self._n_jnts
            for i in range(self._n_jnts): # jnt obs (pos):
                obs_names[next_idx+i] = f"rhc_cmd_eff_{jnt_names[i]}"
            next_idx+=self._n_jnts
        # previous actions info
        if self._add_prev_actions_stats_to_obs:
            action_names = self._get_action_names()
            for prev_act_idx in range(self.actions_dim()):
                obs_names[next_idx+prev_act_idx] = action_names[prev_act_idx]+f"_prev"
            next_idx+=self.actions_dim()
            for prev_act_mean in range(self.actions_dim()):
                obs_names[next_idx+prev_act_mean] = action_names[prev_act_mean]+f"_avrg"
            next_idx+=self.actions_dim()
            for prev_act_mean in range(self.actions_dim()):
                obs_names[next_idx+prev_act_mean] = action_names[prev_act_mean]+f"_std"
            next_idx+=self.actions_dim()
        return obs_names

    def _get_action_names(self):

        action_names = [""] * self.actions_dim()
        action_names[0] = "vx_cmd" # twist commands from agent to RHC controller
        action_names[1] = "vy_cmd"
        action_names[2] = "vz_cmd"
        action_names[3] = "roll_omega_cmd"
        action_names[4] = "pitch_omega_cmd"
        action_names[5] = "yaw_omega_cmd"
        if not self._use_pos_control:
            action_names[6] = "contact_0"
            action_names[7] = "contact_1"
            action_names[8] = "contact_2"
            action_names[9] = "contact_3"
        else:
            action_names[6] = "contact_pref_0"
            action_names[7] = "contact_pref_1"
            action_names[8] = "contact_pref_2"
            action_names[9] = "contact_pref_3"

        return action_names

    def _get_rewards_names(self):

        n_rewards = 1
        reward_names = [""] * n_rewards

        reward_names[0] = "task_error"
        # reward_names[1] = "health"
        # reward_names[1] = "task_pred_error"

        # reward_names[1] = "CoT"
        # reward_names[2] = "mech_power"
        # reward_names[3] = "jnt_vel"
        # reward_names[4] = "rhc_fail_idx"
        

        return reward_names

    def _get_sub_trunc_names(self):
        sub_trunc_names = []
        sub_trunc_names.append("ep_timeout")
        if self._single_task_ref_per_episode:
            sub_trunc_names.append("task_ref_rand")
        return sub_trunc_names

    def _get_sub_term_names(self):
        # to be overridden by child class
        sub_term_names = []
        sub_term_names.append("rhc_failure")
        sub_term_names.append("robot_capsize")
        sub_term_names.append("rhc_capsize")

