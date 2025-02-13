from lrhc_control.utils.sys_utils import PathsGetter
from lrhc_control.envs.lrhc_training_env_base import LRhcTrainingEnvBase
from lrhc_control.utils.timers import PeriodicTimer

from control_cluster_bridge.utilities.shared_data.rhc_data import RobotState, RhcStatus, RhcRefs
from control_cluster_bridge.utilities.math_utils_torch import world2base_frame, base2world_frame, w2hor_frame

import torch

from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import LogType

import os
from lrhc_control.utils.episodic_data import EpisodicData
from lrhc_control.utils.signal_smoother import ExponentialSignalSmoother
import math

from lrhc_control.utils.math_utils import check_capsize

from typing import Dict
from EigenIPC.PyEigenIPC import VLevel, LogType, Journal

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
            timeout_ms: int = 60000,
            env_opts: Dict = {}):
        
        env_name = "LinVelTrack"
        device = "cuda" if use_gpu else "cpu"

        self._add_env_opt(env_opts, "srew_drescaling", 
            False)
        
        # counters settings
        self._add_env_opt(env_opts, "single_task_ref_per_episode", 
            True # if True, the task ref is constant over the episode (ie
            # episodes are truncated when task is changed) 
            )

        self._add_env_opt(env_opts, "episode_timeout_lb", 
            1024)
        self._add_env_opt(env_opts, "episode_timeout_ub", 
            1024)
        self._add_env_opt(env_opts, "n_steps_task_rand_lb", 
            512)
        self._add_env_opt(env_opts, "n_steps_task_rand_ub", 
            512)
        self._add_env_opt(env_opts, "use_random_safety_reset", 
            True)
        self._add_env_opt(env_opts, "random_reset_freq", 
            10) # a random reset once every n-episodes (per env)
        self._add_env_opt(env_opts, "use_random_trunc", 
            True)
        self._add_env_opt(env_opts, "random_trunc_freq", 
            env_opts["episode_timeout_ub"]*5) # to remove temporal correlations between envs
        self._add_env_opt(env_opts, "random_trunc_freq_delta", 
            env_opts["episode_timeout_ub"]*2)  # to randomize trunc frequency between envs
    
        if not env_opts["single_task_ref_per_episode"]:
            env_opts["random_reset_freq"]=env_opts["random_reset_freq"]/\
                round(env_opts["episode_timeout_lb"])/float(env_opts["n_steps_task_rand_lb"])
        
        self._add_env_opt(env_opts, "action_repeat", 1) # frame skipping (different agent action every action_repeat
        # env substeps)
        
        self._add_env_opt(env_opts, "n_preinit_steps", 1) # n steps of the controllers to properly initialize everything
        
        self._add_env_opt(env_opts, "vec_ep_freq_metrics_db", 1) # n eps over which debug metrics are reported
        self._add_env_opt(env_opts, "demo_envs_perc", 0.0)
        self._add_env_opt(env_opts, "max_cmd_v", 1.0) # maximum cmd v for v actions (single component)

        # action smoothing
        self._add_env_opt(env_opts, "use_action_smoothing", False)
        self._add_env_opt(env_opts, "smoothing_horizon_c", 0.01)
        self._add_env_opt(env_opts, "smoothing_horizon_d", 0.03)

        # whether to smooth vel error signal
        self._add_env_opt(env_opts, "use_track_reward_smoother", False)
        self._add_env_opt(env_opts, "smoothing_horizon_vel_err", 0.08)
        self._add_env_opt(env_opts, "track_rew_smoother", None)

        # rewards
        self._reward_map={}

        self._add_env_opt(env_opts, "add_power_reward", True)
        self._add_env_opt(env_opts, "add_CoT_reward", False)
        self._add_env_opt(env_opts, "add_action_rate_reward", True)

        self._add_env_opt(env_opts, "add_jnt_v_reward", True)

        self._add_env_opt(env_opts, "use_rhc_avrg_vel_tracking", False)

        # task tracking
        self._add_env_opt(env_opts, "use_relative_error", default=True) # use relative vel error (wrt current task norm)
        self._add_env_opt(env_opts, "directional_tracking", default=True) # whether to compute tracking rew based on reference direction

        self._add_env_opt(env_opts, "use_fail_idx_weight", default=False)
        self._add_env_opt(env_opts, "task_track_offset", default=10.0)
        self._add_env_opt(env_opts, "task_track_scale", default=3.0)
        self._add_env_opt(env_opts, "task_track_front_weight", default=1.0)
        self._add_env_opt(env_opts, "task_track_lat_weight", default=env_opts["task_track_front_weight"]/20.0)
        self._add_env_opt(env_opts, "task_track_vert_weight", default=env_opts["task_track_front_weight"]/20.0)
        self._add_env_opt(env_opts, "task_track_omega_weight", default=env_opts["task_track_front_weight"]/20.0)

        # task pred tracking
        self._add_env_opt(env_opts, "task_pred_track_offset", default=10.0)
        self._add_env_opt(env_opts, "task_pred_track_scale", default=3.0)

        # energy penalties
        self._add_env_opt(env_opts, "CoT_offset", default=1.0)
        self._add_env_opt(env_opts, "CoT_scale", default=5e-4)
        self._add_env_opt(env_opts, "power_offset", default=1.0)
        self._add_env_opt(env_opts, "power_scale", default=8e-4)

        # action rate penalty
        self._add_env_opt(env_opts, "action_rate_offset", default=1.0)
        self._add_env_opt(env_opts, "action_rate_scale", default=0.5)
        self._add_env_opt(env_opts, "action_rate_rew_d_weight", default=0.1)
        self._add_env_opt(env_opts, "action_rate_rew_c_weight", default=0.8)

        # jnt vel penalty
        self._add_env_opt(env_opts, "jnt_vel_offset", default=1.0)
        self._add_env_opt(env_opts, "jnt_vel_scale", default=2.0)

        # terminations
        self._add_env_opt(env_opts, "add_term_mpc_capsize", default=False) # add termination based on mpc capsizing prediction

        # observations
        self._add_env_opt(env_opts, "rhc_fail_idx_scale", default=1.0)
        self._add_env_opt(env_opts, "use_action_history", default=True) # whether to add information on past actions to obs
        self._add_env_opt(env_opts, "add_prev_actions_stats_to_obs", default=False) # add actions std, mean + last action over a horizon to obs (if self._use_action_history True)
        self._add_env_opt(env_opts, "actions_history_size", default=3)
        
        self._add_env_opt(env_opts, "add_mpc_contact_f_to_obs", default=True) # add estimate vertical contact f to obs
        self._add_env_opt(env_opts, "add_fail_idx_to_obs", default=True) # we need to obserse mpc failure idx to correlate it with terminations
        
        self._add_env_opt(env_opts, "use_linvel_from_rhc", default=True) # no lin vel meas available, we use est. from mpc
        self._add_env_opt(env_opts, "add_flight_info", default=True) # add feedback info on pos and length of flight phases from mpc
        self._add_env_opt(env_opts, "add_flight_settings", default=True) # add feedback info on flight params from mpc

        self._add_env_opt(env_opts, "use_prob_based_stepping", default=False) # interpret actions as stepping prob (never worked)

        self._add_env_opt(env_opts, "add_rhc_cmds_to_obs", default=True) # add the rhc cmds which are being applied now to the robot

        if not "add_periodic_clock_to_obs" in env_opts:
            # add a sin/cos clock to obs (useful if task is explicitly
            # time-dependent)
            self._add_env_opt(env_opts, "add_periodic_clock_to_obs", default=False) 

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
        rhc_refs_tmp = RhcRefs(namespace=namespace,
                            is_server=False,
                            safe=False,
                            verbose=verbose,
                            vlevel=vlevel,
                            with_gpu_mirror=False,
                            with_torch_view=False)
        rhc_refs_tmp.run()
        n_jnts = robot_state_tmp.n_jnts()
        self._contact_names = robot_state_tmp.contact_names()
        self._n_contacts = len(self._contact_names)
        self._flight_info_size=rhc_refs_tmp.flight_info.n_cols
        self._flight_setting_size=rhc_refs_tmp.flight_settings.n_cols

        robot_state_tmp.close()
        rhc_status_tmp.close()
        rhc_refs_tmp.close()

        # defining obs dimension
        obs_dim=3 # normalized gravity vector in base frame
        obs_dim+=6 # meas twist in base frame
        obs_dim+=2*n_jnts # joint pos + vel
        if env_opts["add_mpc_contact_f_to_obs"]:
            obs_dim+=3*self._n_contacts
        obs_dim+=6 # twist reference in base frame frame
        if env_opts["add_fail_idx_to_obs"]:
            obs_dim+=1 # rhc controller failure index
        if env_opts["add_term_mpc_capsize"]: 
            obs_dim+=3 # gravity vec from mpc
        if env_opts["use_rhc_avrg_vel_tracking"]:
            obs_dim+=6 # mpc avrg twist
        if env_opts["add_flight_info"]: # contact pos and len
            obs_dim+=self._flight_info_size
        if env_opts["add_flight_settings"]:
            obs_dim+=self._flight_setting_size
        if env_opts["add_rhc_cmds_to_obs"]:
            obs_dim+=3*n_jnts 
        if env_opts["use_action_history"]:
            if env_opts["add_prev_actions_stats_to_obs"]:
                obs_dim+=3*actions_dim # previous agent actions statistics (mean, std + last action)
            else: # full action history
                obs_dim+=env_opts["actions_history_size"]*actions_dim
        if env_opts["use_action_smoothing"]:
            obs_dim+=actions_dim # it's better to also add the smoothed actions as obs
        if env_opts["add_periodic_clock_to_obs"]:
            obs_dim+=2
        # Agent task reference
        self._add_env_opt(env_opts, "use_pof0", default=True) # with some prob, references will be null
        self._add_env_opt(env_opts, "pof0", default=0.01) # [0, 1] prob of null refs (from bernoulli distr)
        self._add_env_opt(env_opts, "max_ref", default=0.3)

        # ready to init base class
        self._this_child_path = os.path.abspath(__file__)
        LRhcTrainingEnvBase.__init__(self,
                    namespace=namespace,
                    obs_dim=obs_dim,
                    actions_dim=actions_dim,
                    env_name=env_name,
                    verbose=verbose,
                    vlevel=vlevel,
                    use_gpu=use_gpu,
                    dtype=dtype,
                    debug=debug,
                    override_agent_refs=override_agent_refs,
                    timeout_ms=timeout_ms,
                    env_opts=env_opts)
    
    def _add_env_opt(self,
            opts: Dict,
            name: str,
            default):

        if not name in opts:
            opts[name]=default

    def _custom_post_init(self):

        device = "cuda" if self._use_gpu else "cpu"

        self._update_jnt_blacklist() # update blacklist for joints

        self._twist_ref_lb = torch.full((1, 6), dtype=self._dtype, device=device,
                            fill_value=-1.5) 
        self._twist_ref_ub = torch.full((1, 6), dtype=self._dtype, device=device,
                            fill_value=1.5)
        
        # task reference parameters (world frame)
        # lin vel
        self._twist_ref_lb[0, 0] = -self._env_opts["max_ref"]
        self._twist_ref_lb[0, 1] = -self._env_opts["max_ref"]
        self._twist_ref_lb[0, 2] = 0.0
        self._twist_ref_ub[0, 0] = self._env_opts["max_ref"]
        self._twist_ref_ub[0, 1] = self._env_opts["max_ref"]
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

        # adding some custom db info 
        agent_twist_ref = self._agent_refs.rob_refs.root_state.get(data_type="twist",gpu=False)
        agent_twist_ref_data = EpisodicData("AgentTwistRefs", agent_twist_ref, 
            ["v_x", "v_y", "v_z", "omega_x", "omega_y", "omega_z"],
            ep_vec_freq=self._env_opts["vec_ep_freq_metrics_db"],
            store_transitions=self._full_db,
            max_ep_duration=self._max_ep_length())
        rhc_fail_idx = EpisodicData("RhcFailIdx", self._rhc_fail_idx(gpu=False), ["rhc_fail_idx"],
            ep_vec_freq=self._env_opts["vec_ep_freq_metrics_db"],
            store_transitions=self._full_db,
            max_ep_duration=self._max_ep_length())
        
        f_names=[]
        for contact in self._contact_names:
            f_names.append(f"fc_{contact}_x_base_loc")
            f_names.append(f"fc_{contact}_y_base_loc")
            f_names.append(f"fc_{contact}_z_base_loc")
        rhc_contact_f = EpisodicData("RhcContactForces", 
            self._rhc_cmds.contact_wrenches.get(data_type="f",gpu=False), 
            f_names,
            ep_vec_freq=self._env_opts["vec_ep_freq_metrics_db"],
            store_transitions=self._full_db,
            max_ep_duration=self._max_ep_length())

        self._pow_db_data=torch.full(size=(self._n_envs,2),
                dtype=self._dtype, device="cpu",
                fill_value=-1.0)
        power_db = EpisodicData("Power", 
            self._pow_db_data, 
            ["CoT", "W"],
            ep_vec_freq=self._env_opts["vec_ep_freq_metrics_db"],
            store_transitions=self._full_db,
            max_ep_duration=self._max_ep_length())
        
        self._track_error_db=torch.full_like(agent_twist_ref, fill_value=0.0)
        task_err_db = EpisodicData("TrackingError", 
            agent_twist_ref, 
            ["e_vx", "e_vy", "e_vz", "e_omegax", "e_omegay", "e_omegaz"],
            ep_vec_freq=self._env_opts["vec_ep_freq_metrics_db"],
            store_transitions=self._full_db,
            max_ep_duration=self._max_ep_length())

        self._add_custom_db_data(db_data=agent_twist_ref_data)
        self._add_custom_db_data(db_data=rhc_fail_idx)
        self._add_custom_db_data(db_data=rhc_contact_f)
        self._add_custom_db_data(db_data=power_db)
        self._add_custom_db_data(db_data=task_err_db)

        # rewards
        self._task_err_weights = torch.full((1, 6), dtype=self._dtype, device=device,
                            fill_value=0.0) 
        if self._env_opts["directional_tracking"]:
            self._task_err_weights[0, 0] = self._env_opts["task_track_front_weight"] # frontal
            self._task_err_weights[0, 1] = self._env_opts["task_track_lat_weight"] # lateral
            self._task_err_weights[0, 2] = self._env_opts["task_track_vert_weight"] # vertical
            self._task_err_weights[0, 3] = self._env_opts["task_track_omega_weight"]
            self._task_err_weights[0, 4] = self._env_opts["task_track_omega_weight"]
            self._task_err_weights[0, 5] = self._env_opts["task_track_omega_weight"]
        else:
            self._task_err_weights[0, 0] = self._env_opts["task_track_front_weight"]
            self._task_err_weights[0, 1] = self._env_opts["task_track_front_weight"]
            self._task_err_weights[0, 2] = self._env_opts["task_track_front_weight"]
            self._task_err_weights[0, 3] = self._env_opts["task_track_omega_weight"]
            self._task_err_weights[0, 4] = self._env_opts["task_track_omega_weight"]
            self._task_err_weights[0, 5] = self._env_opts["task_track_omega_weight"]
            
        self._task_pred_err_weights = torch.full((1, 6), dtype=self._dtype, device=device,
                            fill_value=0.0) 
        if self._env_opts["directional_tracking"]:
            self._task_pred_err_weights[0, 0] = self._env_opts["task_track_front_weight"]
            self._task_pred_err_weights[0, 1] = self._env_opts["task_track_lat_weight"]
            self._task_pred_err_weights[0, 2] = self._env_opts["task_track_vert_weight"]
            self._task_pred_err_weights[0, 3] = self._env_opts["task_track_omega_weight"]
            self._task_pred_err_weights[0, 4] = self._env_opts["task_track_omega_weight"]
            self._task_pred_err_weights[0, 5] = self._env_opts["task_track_omega_weight"]
        else:
            self._task_pred_err_weights[0, 0] = self._env_opts["task_track_front_weight"]
            self._task_pred_err_weights[0, 1] = self._env_opts["task_track_front_weight"]
            self._task_pred_err_weights[0, 2] = self._env_opts["task_track_front_weight"]
            self._task_pred_err_weights[0, 3] = self._env_opts["task_track_omega_weight"]
            self._task_pred_err_weights[0, 4] = self._env_opts["task_track_omega_weight"]
            self._task_pred_err_weights[0, 5] = self._env_opts["task_track_omega_weight"]

        self._power_penalty_weights = torch.full((1, self._n_jnts), dtype=self._dtype, device=device,
                            fill_value=1.0)
        self._power_penalty_weights_sum = torch.sum(self._power_penalty_weights).item()
        subr_names=self._get_rewards_names() # initializes
        # _reward_map
        
            
        # reward clipping
        self._reward_thresh_lb[:, :]=-1e6 # (neg rewards can be nasty, especially if they all become negative)
        self._reward_thresh_ub[:, :]=1e6
        # self._reward_thresh_lb[:, 1]=-1e6
        # self._reward_thresh_ub[:, 1]=1e6

        # obs bounds
        self._obs_threshold_lb = -1e3 # used for clipping observations
        self._obs_threshold_ub = 1e3

        # actions
        if not self._env_opts["use_prob_based_stepping"]:
            self._is_continuous_actions[6:10]=False

        v_cmd_max = self._env_opts["max_cmd_v"]
        omega_cmd_max = self._env_opts["max_cmd_v"]
        self._actions_lb[:, 0:3] = -v_cmd_max 
        self._actions_ub[:, 0:3] = v_cmd_max  
        self._actions_lb[:, 3:6] = -omega_cmd_max # twist cmds
        self._actions_ub[:, 3:6] = omega_cmd_max  
        if self._env_opts["use_prob_based_stepping"]:
            self._actions_lb[:, 6:10] = 0.0 # contact flags
            self._actions_ub[:, 6:10] = 1.0 
        else:
            self._actions_lb[:, 6:10] = -1.0 
            self._actions_ub[:, 6:10] = 1.0 
        
        self._default_action[:, :] = (self._actions_ub+self._actions_lb)/2.0
        self._default_action[:, ~self._is_continuous_actions] = 1.0
            
        # assign obs bounds (useful if not using automatic obs normalization)
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
            5*v_cmd_max,
            5*omega_cmd_max,
            2*torch.pi,
            30.0,
            2.0,
            1.0,
            2*torch.pi,
            30.0,
            200.0,
            self._n_nodes_rhc.mean().item()]
        obs_lbs=[-1.0,
            -5*v_cmd_max,
            -5*omega_cmd_max,
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
        if self._env_opts["use_action_history"]: # just history stats
            if self._env_opts["add_prev_actions_stats_to_obs"]:
                i=0
                prev_actions_idx = next((i for i, s in enumerate(obs_names) if "_prev_act" in s), None)
                prev_actions_mean_idx=next((i for i, s in enumerate(obs_names) if "_avrg_act" in s), None)
                prev_actions_std_idx=next((i for i, s in enumerate(obs_names) if "_std_act" in s), None)

                if prev_actions_idx is not None:
                    self._obs_lb[:, prev_actions_idx:prev_actions_idx+self.actions_dim()]=self._actions_lb
                    self._obs_ub[:, prev_actions_idx:prev_actions_idx+self.actions_dim()]=self._actions_ub
                if prev_actions_mean_idx is not None:
                    self._obs_lb[:, prev_actions_mean_idx:prev_actions_mean_idx+self.actions_dim()]=self._actions_lb
                    self._obs_ub[:, prev_actions_mean_idx:prev_actions_mean_idx+self.actions_dim()]=self._actions_ub
                if prev_actions_std_idx is not None:
                    self._obs_lb[:, prev_actions_std_idx:prev_actions_std_idx+self.actions_dim()]=0
                    self._obs_ub[:, prev_actions_std_idx:prev_actions_std_idx+self.actions_dim()]=self.get_actions_scale()
                
            else: # full history
                i=0
                first_action_mem_buffer_idx = next((i for i, s in enumerate(obs_names) if "_m1_act" in s), None)
                if first_action_mem_buffer_idx is not None:
                    action_idx_start_idx_counter=first_action_mem_buffer_idx
                    for j in range(self._env_opts["actions_history_size"]):
                        self._obs_lb[:, action_idx_start_idx_counter:action_idx_start_idx_counter+self.actions_dim()]=self._actions_lb
                        self._obs_ub[:, action_idx_start_idx_counter:action_idx_start_idx_counter+self.actions_dim()]=self._actions_ub
                        action_idx_start_idx_counter+=self.actions_dim()

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
        # aux data
        self._task_err_scaling = torch.zeros((self._n_envs, 1),dtype=self._dtype,device=device)

        self._pof1_b = torch.full(size=(self._n_envs,1),dtype=self._dtype,device=device,fill_value=1-self._env_opts["pof0"])
        self._bernoulli_coeffs = self._pof1_b.clone()
        self._bernoulli_coeffs[:, :] = 1.0

        # smoothing
        self._track_rew_smoother=None
        if self._env_opts["use_track_reward_smoother"]:
            sub_reward_proxy=self._sub_rewards.get_torch_mirror(gpu=self._use_gpu)[:, 0:1]
            smoothing_dt=self._substep_dt
            if not self._is_substep_rew[self._reward_map["task_error"]]: # assuming first reward is tracking
                smoothing_dt=self._substep_dt*self._action_repeat
            self._track_rew_smoother=ExponentialSignalSmoother(
                name=self.__class__.__name__+"VelErrorSmoother",
                signal=sub_reward_proxy, # same dimension of vel error
                update_dt=smoothing_dt,
                smoothing_horizon=self._env_opts["smoothing_horizon_vel_err"],
                target_smoothing=0.5,
                debug=self._is_debug,
                dtype=self._dtype,
                use_gpu=self._use_gpu)

        # if we need the action rete, we also need the action history
        if self._env_opts["add_action_rate_reward"]:
            if not self._env_opts["use_action_history"]:
                Journal.log(self.__class__.__name__,
                    "_custom_post_init",
                    "add_action_rate_reward is True, but ",
                    LogType.EXCEP,
                    throw_when_excep=True)
            
            history_size=self._env_opts["actions_history_size"]
            if history_size < 2:
                Journal.log(self.__class__.__name__,
                    "_custom_post_init",
                    f"add_action_rate_reward  requires actions history ({history_size}) to be >=2!",
                    LogType.EXCEP,
                    throw_when_excep=True)
        
        # add periodic timer if required
        self._periodic_clock=None
        if self._env_opts["add_periodic_clock_to_obs"]:
            self._add_env_opt(self._env_opts, "clock_period", 
                default=int(1.5*self._action_repeat*self.task_rand_timeout_bounds()[1])) # correcting with n substeps
            # (we are using the _substep_abs_counter counter)
            self._periodic_clock=PeriodicTimer(counter=self._substep_abs_counter,
                                    period=self._env_opts["clock_period"], 
                                    dtype=self._dtype,
                                    device=self._device)

    def get_file_paths(self):
        paths=LRhcTrainingEnvBase.get_file_paths(self)
        paths.append(self._this_child_path)        
        return paths

    def get_aux_dir(self):
        aux_dirs = []
        path_getter = PathsGetter()
        aux_dirs.append(path_getter.RHCDIR)
        return aux_dirs

    def _get_reward_scaling(self):
        if self._env_opts["single_task_ref_per_episode"]:
            return self._env_opts["n_steps_task_rand_ub"]
        else:
            return self._env_opts["episode_timeout_ub"]
    
    def _max_ep_length(self):
        if self._env_opts["single_task_ref_per_episode"]:
            return self._env_opts["n_steps_task_rand_ub"]
        else:
            return self._env_opts["episode_timeout_ub"]
    
    def _check_sub_truncations(self):
        # overrides parent
        sub_truncations = self._sub_truncations.get_torch_mirror(gpu=self._use_gpu)
        sub_truncations[:, 0:1] = self._ep_timeout_counter.time_limits_reached()
        if self._env_opts["single_task_ref_per_episode"]:
            sub_truncations[:, 1:2] = self._task_rand_counter.time_limits_reached()
    
    def _check_sub_terminations(self):
        # default behaviour-> to be overriden by child
        sub_terminations = self._sub_terminations.get_torch_mirror(gpu=self._use_gpu)
        
        # terminate if mpc just failed
        sub_terminations[:, 0:1] = self._rhc_status.fails.get_torch_mirror(gpu=self._use_gpu)

        # check if robot is capsizing
        robot_q_meas = self._robot_state.root_state.get(data_type="q",gpu=self._use_gpu)
        check_capsize(quat=robot_q_meas,max_angle=self._max_pitch_angle,
            output_t=self._is_capsized)
        sub_terminations[:, 1:2] = self._is_capsized
        
        if self._env_opts["add_term_mpc_capsize"]:
            # check if robot is about to capsize accordin to MPC
            robot_q_pred = self._rhc_cmds.root_state.get(data_type="q",gpu=self._use_gpu)
            check_capsize(quat=robot_q_pred,max_angle=self._max_pitch_angle,
                output_t=self._is_rhc_capsized)
            sub_terminations[:, 2:3] = self._is_rhc_capsized

    def _custom_reset(self):
        return None
    
    def reset(self):
        LRhcTrainingEnvBase.reset(self)

    def _pre_substep(self): 
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

        if self._track_rew_smoother is not None: # reset smoother
            self._track_rew_smoother.reset_all(to_be_reset=episode_finished.flatten(), 
                    value=0.0)

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
        
        self._set_rhc_refs()

        self._write_rhc_refs()

    def _set_rhc_refs(self):

        action_to_be_applied = self.get_actual_actions() # see _get_action_names() to get 
        # the meaning of each component of this tensor

        rhc_latest_twist_cmd = self._rhc_refs.rob_refs.root_state.get(data_type="twist", gpu=self._use_gpu)
        rhc_latest_contact_ref = self._rhc_refs.contact_flags.get_torch_mirror(gpu=self._use_gpu)
        rhc_latest_pos_ref = self._rhc_refs.rob_refs.contact_pos.get(data_type="p_z", gpu=self._use_gpu)
        rhc_q=self._rhc_cmds.root_state.get(data_type="q",gpu=self._use_gpu) # this is always 
        # avaialble

        # reference twist for MPC is assumed to always be specified in MPC's 
        # horizontal frame, while agent actions are interpreted as in MPC's
        # base frame -> we need to rotate the actions into the horizontal frame
        base2world_frame(t_b=action_to_be_applied[:, 0:6],q_b=rhc_q,t_out=self._rhc_twist_cmd_rhc_world)
        w2hor_frame(t_w=self._rhc_twist_cmd_rhc_world,q_b=rhc_q,t_out=self._rhc_twist_cmd_rhc_h)

        rhc_latest_twist_cmd[:, 0:6] = self._rhc_twist_cmd_rhc_h
        
        # self._rhc_refs.rob_refs.root_state.set(data_type="p", data=rhc_latest_p_ref,
        #                                     gpu=self._use_gpu)
        self._rhc_refs.rob_refs.root_state.set(data_type="twist", data=rhc_latest_twist_cmd,
            gpu=self._use_gpu) 
        
        # contact flags
        idx=self._actions_map["contact_flag_start"]
        if self._env_opts["use_prob_based_stepping"]:
            # encode actions as probs
            self._random_thresh_contacts.uniform_() # random values in-place between 0 and 1
            rhc_latest_contact_ref[:, :] = action_to_be_applied[:, idx:idx+self._n_contacts] >= self._random_thresh_contacts  # keep contact with 
            # probability action_to_be_applied[:, 6:10]
        else: # just use a threshold
            rhc_latest_contact_ref[:, :] = action_to_be_applied[:, idx:idx+self._n_contacts] > 0
        # actually apply actions to controller
        
    def _write_rhc_refs(self):

        if self._use_gpu:
            # GPU->CPU --> we cannot use asynchronous data transfer since it's unsafe
            self._rhc_refs.rob_refs.root_state.synch_mirror(from_gpu=True,non_blocking=False) # write from gpu to cpu mirror
            self._rhc_refs.contact_flags.synch_mirror(from_gpu=True,non_blocking=False)
            self._rhc_refs.rob_refs.contact_pos.synch_mirror(from_gpu=True,non_blocking=False)

        self._rhc_refs.rob_refs.root_state.synch_all(read=False, retry=True) # write mirror to shared mem
        self._rhc_refs.contact_flags.synch_all(read=False, retry=True)
        self._rhc_refs.rob_refs.contact_pos.synch_all(read=False, retry=True)
    
    def _fill_substep_obs(self,
            obs: torch.Tensor):

        # measured stuff
        robot_twist_meas_base_loc = self._robot_state.root_state.get(data_type="twist",gpu=self._use_gpu)
        robot_jnt_v_meas = self._robot_state.jnts_state.get(data_type="v",gpu=self._use_gpu)
        
        if self._env_opts["use_linvel_from_rhc"]:
            # twist estimate from mpc
            robot_twist_rhc_base_loc_next = self._rhc_cmds.root_state.get(data_type="twist",gpu=self._use_gpu)
            obs[:, self._obs_map["linvel_meas"]:(self._obs_map["linvel_meas"]+3)] = robot_twist_rhc_base_loc_next[:, 0:3]
        else:
            obs[:, self._obs_map["linvel_meas"]:(self._obs_map["linvel_meas"]+3)] = robot_twist_meas_base_loc[:, 0:3]
        obs[:, self._obs_map["omega_meas"]:(self._obs_map["omega_meas"]+3)] = robot_twist_meas_base_loc[:, 3:6]

        obs[:, self._obs_map["v_jnt"]:(self._obs_map["v_jnt"]+self._n_jnts)] = robot_jnt_v_meas 

    def _fill_step_obs(self,
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
        flight_settings_now = self._rhc_refs.flight_settings.get(data_type="all",gpu=self._use_gpu)
        
        # refs
        agent_twist_ref = self._agent_refs.rob_refs.root_state.get(data_type="twist",gpu=self._use_gpu)

        obs[:, self._obs_map["gn_base"]:(self._obs_map["gn_base"]+3)] = robot_gravity_norm_base_loc # norm. gravity vector in base frame
        
        obs[:, self._obs_map["q_jnt"]:(self._obs_map["q_jnt"]+self._n_jnts)] = robot_jnt_q_meas # meas jnt pos
        obs[:, self._obs_map["twist_ref"]:(self._obs_map["twist_ref"]+6)] = agent_twist_ref # high lev agent refs to be tracked

        if self._env_opts["add_mpc_contact_f_to_obs"]:
            n_forces=3*len(self._contact_names)
            obs[:, self._obs_map["contact_f_mpc"]:(self._obs_map["contact_f_mpc"]+n_forces)] = self._rhc_cmds.contact_wrenches.get(data_type="f",gpu=self._use_gpu)
        if self._env_opts["add_fail_idx_to_obs"]:
            obs[:, self._obs_map["rhc_fail_idx"]:(self._obs_map["rhc_fail_idx"]+1)] = self._rhc_fail_idx(gpu=self._use_gpu)
        if self._env_opts["add_term_mpc_capsize"]:
            obs[:, self._obs_map["gn_base_mpc"]:(self._obs_map["gn_base_mpc"]+3)] = self._rhc_cmds.root_state.get(data_type="gn",gpu=self._use_gpu)
        if self._env_opts["use_rhc_avrg_vel_tracking"]:
            self._get_avrg_rhc_root_twist(out=self._root_twist_avrg_rhc_base_loc,base_loc=True)
            obs[:, self._obs_map["avrg_twist_mpc"]:(self._obs_map["avrg_twist_mpc"]+6)] = self._root_twist_avrg_rhc_base_loc
        if self._env_opts["add_flight_info"]:
            obs[:, self._obs_map["flight_info"]:(self._obs_map["flight_info"]+self._flight_info_size)] = flight_info_now
        if self._env_opts["add_flight_settings"]:
            obs[:, self._obs_map["flight_settings"]:(self._obs_map["flight_settings"]+self._flight_setting_size)] = flight_settings_now

        if self._env_opts["add_rhc_cmds_to_obs"]:
            obs[:, self._obs_map["rhc_cmds_q"]:(self._obs_map["rhc_cmds_q"]+self._n_jnts)] = robot_jnt_q_rhc_applied_next
            obs[:, self._obs_map["rhc_cmds_v"]:(self._obs_map["rhc_cmds_v"]+self._n_jnts)] = robot_jnt_v_rhc_applied_next
            obs[:, self._obs_map["rhc_cmds_eff"]:(self._obs_map["rhc_cmds_eff"]+self._n_jnts)] = robot_jnt_eff_rhc_applied_next
        if self._env_opts["use_action_history"]:
            if self._env_opts["add_prev_actions_stats_to_obs"]: # just add last, std and mean to obs
                obs[:, self._obs_map["action_history_prev"]:(self._obs_map["action_history_prev"]+self.actions_dim())]=self._act_mem_buffer.get(idx=0)
                obs[:, self._obs_map["action_history_avrg"]:(self._obs_map["action_history_avrg"]+self.actions_dim())]=self._act_mem_buffer.mean(clone=False)
                obs[:, self._obs_map["action_history_std"]:(self._obs_map["action_history_std"]+self.actions_dim())]=self._act_mem_buffer.std(clone=False)
            else: # add whole memory buffer to obs
                next_idx=self._obs_map["action_history"]
                for i in range(self._env_opts["actions_history_size"]):
                    obs[:, next_idx:(next_idx+self.actions_dim())]=self._act_mem_buffer.get(idx=i) # get all (n_envs x (obs_dim x horizon))
                    next_idx+=self.actions_dim()

        if self._env_opts["use_action_smoothing"]: # adding smoothed actions
            obs[:, self._obs_map["action_smoothing"]:(self._obs_map["action_smoothing"]+self.actions_dim())]=self.get_actual_actions()
            next_idx+=self.actions_dim()
        
        if self._env_opts["add_periodic_clock_to_obs"]:
            obs[:, next_idx:(next_idx+2)]=self._periodic_clock.get()
            next_idx+=2

    def _get_custom_db_data(self, 
            episode_finished,
            ignore_ep_end):
        episode_finished = episode_finished.cpu()
        self.custom_db_data["AgentTwistRefs"].update(
                new_data=self._agent_refs.rob_refs.root_state.get(data_type="twist", gpu=False), 
                ep_finished=episode_finished,
                ignore_ep_end=ignore_ep_end)
        self.custom_db_data["RhcFailIdx"].update(new_data=self._rhc_fail_idx(gpu=False), 
                ep_finished=episode_finished,
                ignore_ep_end=ignore_ep_end)
        self.custom_db_data["RhcContactForces"].update(
                new_data=self._rhc_cmds.contact_wrenches.get(data_type="f",gpu=False), 
                ep_finished=episode_finished,
                ignore_ep_end=ignore_ep_end)
        self.custom_db_data["Power"].update(
                new_data=self._pow_db_data, 
                ep_finished=episode_finished,
                ignore_ep_end=ignore_ep_end)
        self.custom_db_data["TrackingError"].update(
                new_data=self._track_error_db, 
                ep_finished=episode_finished,
                ignore_ep_end=ignore_ep_end)
    
    def _action_rate(self):
        continuous_actions=self._is_continuous_actions
        discrete_actions=~self._is_continuous_actions
        n_c_actions=continuous_actions.sum().item()
        n_d_actions=discrete_actions.sum().item()
        actions_prev=self._act_mem_buffer.get(idx=1) 
        actions_now=self._act_mem_buffer.get(idx=0)
        actions_rate=(actions_now-actions_prev)/self.get_actions_scale() # scaling to remove dependence on actions range
        actions_rate_c=actions_rate[:, continuous_actions]
        actions_rate_d=actions_rate[:, discrete_actions]

        actions_rate_sqrd=None # assuming n_c_actions > 0 always
        actions_rate_sqrd=self._env_opts["action_rate_rew_c_weight"]*torch.sum(actions_rate_c*actions_rate_c, dim=1, keepdim=True)/n_c_actions
        if discrete_actions.any():
            actions_rate_sqrd+=self._env_opts["action_rate_rew_d_weight"]*torch.sum(actions_rate_d*actions_rate_d, dim=1, keepdim=True)/n_d_actions
        return actions_rate_sqrd

    def _mech_pow(self, jnts_vel, jnts_effort, autoscaled: bool = False, drained: bool = True):
        mech_pow_jnts=(jnts_effort*jnts_vel)*self._power_penalty_weights
        if drained:
            mech_pow_jnts.clamp_(0.0,torch.inf) # do not account for regenerative power
        mech_pow_tot = torch.sum(mech_pow_jnts, dim=1, keepdim=True)
        self._pow_db_data[:, 1:2]=mech_pow_tot.cpu()
        if autoscaled:
            mech_pow_tot=mech_pow_tot/self._power_penalty_weights_sum
        return mech_pow_tot

    def _cost_of_transport(self, jnts_vel, jnts_effort, v_ref_norm, mass_weight: bool = False):
        drained_mech_pow=self._mech_pow(jnts_vel=jnts_vel,
            jnts_effort=jnts_effort, 
            drained=True)
        CoT=drained_mech_pow/(v_ref_norm+1e-3)
        if mass_weight:
            robot_weight=self._rhc_robot_weight
            CoT=CoT/robot_weight
        
        # add to db metrics
        self._pow_db_data[:, 0:1]=CoT.cpu()
        self._pow_db_data[:, 1:2]=drained_mech_pow.cpu()

        return CoT

    def _jnt_vel_penalty(self, jnts_vel):
        weighted_jnt_vel = torch.sum(jnts_vel*jnts_vel, dim=1, keepdim=True)/self._n_jnts
        return weighted_jnt_vel
    
    def _track_relative_err_wms(self, task_ref, task_meas, weights, epsi: float = 0.0, directional: bool = False):
        ref_norm = task_ref.norm(dim=1,keepdim=True)
        self._task_err_scaling[:, :] = ref_norm+epsi
        if directional:
            task_perc_err=self._track_err_directional(task_ref=task_ref, task_meas=task_meas, 
                scaling=self._task_err_scaling, weights=weights)
        else:
            task_perc_err=self._track_err_wms(task_ref=task_ref, task_meas=task_meas, 
                scaling=self._task_err_scaling, weights=weights)
        # perc_err_thresh=2.0 # no more than perc_err_thresh*100 % error on each dim
        # task_perc_err.clamp_(0.0,perc_err_thresh**2) 
        return task_perc_err
    
    def _track_err_wms(self, task_ref, task_meas, scaling, weights):
        task_error = (task_meas-task_ref)
        # add to db metrics
        self._track_error_db[:, :]=task_error
        scaled_error=task_error/scaling
        task_wmse = torch.sum(scaled_error*scaled_error*weights, dim=1, keepdim=True)/torch.sum(weights).item()
        return task_wmse # weighted mean square error (along task dimension)
    
    def _track_err_directional(self, task_ref, task_meas, scaling, weights):
        task_error = (task_meas-task_ref)
        # add to db metrics
        self._track_error_db[:, :]=task_error
        task_error=task_error/scaling
        task_ref_xy_linvel=task_ref[:, 0:2]
        task_error_xy_linvel=task_error[:, 0:2]
        task_ref_linvel_norm=task_ref_xy_linvel.norm(dim=1,keepdim=True)
        task_ref_xy_versor=task_ref_xy_linvel/(task_ref_linvel_norm+1e-8)

        longitudinal_error_norm=torch.sum(task_error_xy_linvel*task_ref_xy_versor, dim=1, keepdim=True)
        lateral_error_norm=torch.norm(task_error_xy_linvel-longitudinal_error_norm*task_ref_xy_versor, dim=1, keepdim=True)

        # handle small refs
        below_thresh=task_ref_linvel_norm<1e-6 
        longitudinal_error_norm[below_thresh.flatten(), :]=task_meas[below_thresh.flatten(), 0:1]
        lateral_error_norm[below_thresh.flatten(), :]=task_meas[below_thresh.flatten(), 1:2]

        full_error=torch.cat((longitudinal_error_norm, lateral_error_norm, task_error[:, 2:6]), dim=1)
        task_wmse_dir = torch.sum(full_error*full_error*weights, dim=1, keepdim=True)/torch.sum(weights).item()
        return task_wmse_dir # weighted mean square error (along task dimension)
    
    def _track_relative_err_lin(self, task_ref, task_meas, weights, directional):
        task_wmse = self._track_relative_err_wms(task_ref=task_ref, task_meas=task_meas,
            weights=weights, epsi=1e-2, directional=directional)
        return task_wmse.sqrt()
    
    def _track_err_lin(self, task_ref, task_meas, weights, directional: bool = False):
        self._task_err_scaling[:, :] = 1
        if directional:
            task_wmse = self._track_err_directional(task_ref=task_ref, task_meas=task_meas, 
                scaling=self._task_err_scaling, weights=weights)
        else:
            task_wmse = self._track_err_wms(task_ref=task_ref, task_meas=task_meas, 
                scaling=self._task_err_scaling, weights=weights)
            
        return task_wmse.sqrt()
    
    def _rhc_fail_idx(self, gpu: bool):
        rhc_fail_idx = self._rhc_status.rhc_fail_idx.get_torch_mirror(gpu=gpu)
        return self._env_opts["rhc_fail_idx_scale"]*rhc_fail_idx
    
    def _compute_step_rewards(self):
        
        sub_rewards = self._sub_rewards.get_torch_mirror(gpu=self._use_gpu)

        # tracking reward
        task_error_fun = self._track_err_lin
        if self._env_opts["use_relative_error"]:
            task_error_fun = self._track_relative_err_lin

        agent_task_ref_base_loc = self._agent_refs.rob_refs.root_state.get(data_type="twist",gpu=self._use_gpu) # high level agent refs (hybrid twist)
        self._get_avrg_step_root_twist(out=self._step_avrg_root_twist_base_loc, base_loc=True)
        task_error = task_error_fun(task_meas=self._step_avrg_root_twist_base_loc, 
            task_ref=agent_task_ref_base_loc,
            weights=self._task_err_weights,
            directional=self._env_opts["directional_tracking"])

        idx=self._reward_map["task_error"]
        sub_rewards[:, idx:(idx+1)] =  self._env_opts["task_track_offset"]*torch.exp(-self._env_opts["task_track_scale"]*task_error)
        if self._env_opts["use_fail_idx_weight"]: # add weight based on fail idx
            fail_idx=self._rhc_fail_idx(gpu=self._use_gpu)
            sub_rewards[:, idx:(idx+1)]=(1-fail_idx)*sub_rewards[:, idx:(idx+1)]
        if self._track_rew_smoother is not None: # smooth reward if required
            self._track_rew_smoother.update(new_signal=sub_rewards[:, 0:1])
            sub_rewards[:, idx:(idx+1)]=self._track_rew_smoother.get()

        if self._env_opts["add_action_rate_reward"]:
            action_rate=self._action_rate()
            idx=self._reward_map["action_rate"]
            sub_rewards[:, idx:(idx+1)] = self._env_opts["action_rate_offset"]*(1-self._env_opts["action_rate_scale"]*action_rate)

        # mpc vel tracking
        if self._env_opts["use_rhc_avrg_vel_tracking"]:
            self._get_avrg_rhc_root_twist(out=self._root_twist_avrg_rhc_base_loc_next,base_loc=True) # get estimated avrg vel 
            # from MPC after stepping
            task_pred_error=task_error_fun(task_meas=self._root_twist_avrg_rhc_base_loc_next, 
                task_ref=agent_task_ref_base_loc,
                weights=self._task_pred_err_weights,
                directional=self._env_opts["directional_tracking"])
            idx=self._reward_map["rhc_avrg_vel_error"]
            sub_rewards[:, idx:(idx+1)] = self._env_opts["task_pred_track_offset"]*torch.exp(-self._env_opts["task_pred_track_scale"]*task_pred_error)
        
    def _compute_substep_rewards(self):
        
        sub_rewards = self._sub_rewards.get_torch_mirror(gpu=self._use_gpu)

        if self._env_opts["add_CoT_reward"] or self._env_opts["add_power_reward"]:
            jnts_vel = self._robot_state.jnts_state.get(data_type="v",gpu=self._use_gpu)
            jnts_effort = self._robot_state.jnts_state.get(data_type="eff",gpu=self._use_gpu)

            if self._env_opts["add_CoT_reward"]:
                agent_task_ref_base_loc = self._agent_refs.rob_refs.root_state.get(data_type="twist",gpu=self._use_gpu)
                ref_norm=torch.norm(agent_task_ref_base_loc, dim=1, keepdim=True)
                CoT=self._cost_of_transport(jnts_vel=jnts_vel,jnts_effort=jnts_effort,v_ref_norm=ref_norm, 
                    mass_weight=False # inessential scaling
                    )
                idx=self._reward_map["CoT"]
                sub_rewards[:, idx:(idx+1)] = self._env_opts["CoT_offset"]*(1-self._env_opts["CoT_scale"]*CoT)

            if self._env_opts["add_power_reward"]:
                weighted_mech_power=self._mech_pow(jnts_vel=jnts_vel,jnts_effort=jnts_effort, drained=True)
                idx=self._reward_map["mech_pow"]
                sub_rewards[:, idx:(idx+1)] = self._env_opts["power_offset"]*(1-self._env_opts["power_scale"]*weighted_mech_power)
        
        if self._env_opts["add_jnt_v_reward"]:
            jnts_vel = self._robot_state.jnts_state.get(data_type="v",gpu=self._use_gpu)
            jnt_v=self._jnt_vel_penalty(jnts_vel=jnts_vel)
            idx=self._reward_map["jnt_v"]
            sub_rewards[:, idx:(idx+1)] = self._env_opts["jnt_vel_offset"]*(1-self._env_opts["jnt_vel_scale"]*jnt_v)

    def _randomize_task_refs(self,
        env_indxs: torch.Tensor = None):

        # we randomize the reference in world frame, since it's much more intuitive 
        # (it will be rotated in base frame when provided to the agent and used for rew 
        # computation)
        
        if self._env_opts["use_pof0"]: # sample from bernoulli distribution
            torch.bernoulli(input=self._pof1_b,out=self._bernoulli_coeffs) # by default bernoulli_coeffs are 1 if not self._env_opts["use_pof0"]
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

        self._obs_map["gn_base"]=next_idx
        obs_names[0] = "gn_x_base_loc"
        obs_names[1] = "gn_y_base_loc"
        obs_names[2] = "gn_z_base_loc"
        next_idx+=3

        self._obs_map["linvel_meas"]=next_idx
        obs_names[next_idx] = "linvel_x_base_loc"
        obs_names[next_idx+1] = "linvel_y_base_loc"
        obs_names[next_idx+2] = "linvel_z_base_loc"
        next_idx+=3

        self._obs_map["omega_meas"]=next_idx
        obs_names[next_idx] = "omega_x_base_loc"
        obs_names[next_idx+1] = "omega_y_base_loc"
        obs_names[next_idx+2] = "omega_z_base_loc"
        next_idx+=3

        jnt_names=self.get_observed_joints()
        self._obs_map["q_jnt"]=next_idx
        for i in range(self._n_jnts): # jnt obs (pos):
            obs_names[next_idx+i] = f"q_jnt_{jnt_names[i]}"
        next_idx+=self._n_jnts
        
        self._obs_map["v_jnt"]=next_idx
        for i in range(self._n_jnts): # jnt obs (v):
            obs_names[next_idx+i] = f"v_jnt_{jnt_names[i]}"
        next_idx+=self._n_jnts
        
        # references
        self._obs_map["twist_ref"]=next_idx
        obs_names[next_idx] = "linvel_x_ref_base_loc"
        obs_names[next_idx+1] = "linvel_y_ref_base_loc"
        obs_names[next_idx+2] = "linvel_z_ref_base_loc"
        obs_names[next_idx+3] = "omega_x_ref_base_loc"
        obs_names[next_idx+4] = "omega_y_ref_base_loc"
        obs_names[next_idx+5] = "omega_z_ref_base_loc"
        next_idx+=6
        
        # contact forces
        if self._env_opts["add_mpc_contact_f_to_obs"]:
            i = 0
            self._obs_map["contact_f_mpc"]=next_idx
            for contact in self._contact_names:
                obs_names[next_idx+i] = f"fc_{contact}_x_base_loc"
                obs_names[next_idx+i+1] = f"fc_{contact}_y_base_loc"
                obs_names[next_idx+i+2] = f"fc_{contact}_z_base_loc"
                i+=3        
            next_idx+=3*len(self._contact_names)
            
        # data directly from MPC
        if self._env_opts["add_fail_idx_to_obs"]:
            self._obs_map["rhc_fail_idx"]=next_idx
            obs_names[next_idx] = "rhc_fail_idx"
            next_idx+=1
        if self._env_opts["add_term_mpc_capsize"]:
            self._obs_map["gn_base_mpc"]=next_idx
            obs_names[next_idx] = "gn_x_rhc_base_loc"
            obs_names[next_idx+1] = "gn_y_rhc_base_loc"
            obs_names[next_idx+2] = "gn_z_rhc_base_loc"
            next_idx+=3
        if self._env_opts["use_rhc_avrg_vel_tracking"]:
            self._obs_map["avrg_twist_mpc"]=next_idx
            obs_names[next_idx] = "linvel_x_avrg_rhc"
            obs_names[next_idx+1] = "linvel_y_avrg_rhc"
            obs_names[next_idx+2] = "linvel_z_avrg_rhc"
            obs_names[next_idx+3] = "omega_x_avrg_rhc"
            obs_names[next_idx+4] = "omega_y_avrg_rhc"
            obs_names[next_idx+5] = "omega_z_avrg_rhc"
            next_idx+=6
        
        if self._env_opts["add_flight_info"]:
            self._obs_map["flight_info"]=next_idx
            for i in range(len(self._contact_names)):
                obs_names[next_idx+i] = "flight_pos_"+ self._contact_names[i]
            next_idx+=len(self._contact_names)
            for i in range(len(self._contact_names)):
                obs_names[next_idx+i] = "flight_len_"+ self._contact_names[i]
            next_idx+=len(self._contact_names)
        
        if self._env_opts["add_flight_settings"]:
            self._obs_map["flight_settings"]=next_idx
            for i in range(len(self._contact_names)):
                obs_names[next_idx+i] = "flight_len_cmd_"+ self._contact_names[i]
            next_idx+=len(self._contact_names)
            for i in range(len(self._contact_names)):
                obs_names[next_idx+i] = "flight_apex_cmd_"+ self._contact_names[i]
            next_idx+=len(self._contact_names)
            for i in range(len(self._contact_names)):
                obs_names[next_idx+i] = "flight_end_cmd_"+ self._contact_names[i]
            next_idx+=len(self._contact_names)

        if self._env_opts["add_rhc_cmds_to_obs"]:
            self._obs_map["rhc_cmds_q"]=next_idx
            for i in range(self._n_jnts): # jnt obs (pos):
                obs_names[next_idx+i] = f"rhc_cmd_q_{jnt_names[i]}"
            next_idx+=self._n_jnts
            self._obs_map["rhc_cmds_v"]=next_idx
            for i in range(self._n_jnts): # jnt obs (pos):
                obs_names[next_idx+i] = f"rhc_cmd_v_{jnt_names[i]}"
            next_idx+=self._n_jnts
            self._obs_map["rhc_cmds_eff"]=next_idx
            for i in range(self._n_jnts): # jnt obs (pos):
                obs_names[next_idx+i] = f"rhc_cmd_eff_{jnt_names[i]}"
            next_idx+=self._n_jnts

        # previous actions info
        if self._env_opts["use_action_history"]:
            self._obs_map["action_history"]=next_idx
            action_names = self._get_action_names()
            if self._env_opts["add_prev_actions_stats_to_obs"]:
                self._obs_map["action_history_prev"]=next_idx
                for act_idx in range(self.actions_dim()):
                    obs_names[next_idx+act_idx] = action_names[act_idx]+f"_prev_act"
                next_idx+=self.actions_dim()
                self._obs_map["action_history_avrg"]=next_idx
                for act_idx in range(self.actions_dim()):
                    obs_names[next_idx+act_idx] = action_names[act_idx]+f"_avrg_act"
                next_idx+=self.actions_dim()
                self._obs_map["action_history_std"]=next_idx
                for act_idx in range(self.actions_dim()):
                    obs_names[next_idx+act_idx] = action_names[act_idx]+f"_std_act"
                next_idx+=self.actions_dim()
            else:
                for i in range(self._env_opts["actions_history_size"]):
                    for act_idx in range(self.actions_dim()):
                        obs_names[next_idx+act_idx] = action_names[act_idx]+f"_m{i+1}_act"
                    next_idx+=self.actions_dim()

        if self._env_opts["use_action_smoothing"]:
            self._obs_map["action_smoothing"]=next_idx
            for smoothed_action in range(self.actions_dim()):
                obs_names[next_idx+smoothed_action] = action_names[smoothed_action]+f"_smoothed"
            next_idx+=self.actions_dim()

        if self._env_opts["add_periodic_clock_to_obs"]:
            self._obs_map["clock"]=next_idx
            obs_names[next_idx] = "clock_cos"
            obs_names[next_idx+1] = "clock_sin"
            next_idx+=2

        return obs_names

    def _set_substep_obs(self):
        # which obs are to be averaged over substeps?

        self._is_substep_obs[self._obs_map["linvel_meas"]:self._obs_map["linvel_meas"]+3]=True
        self._is_substep_obs[self._obs_map["omega_meas"]:self._obs_map["omega_meas"]+3]=True
        self._is_substep_obs[self._obs_map["v_jnt"]:self._obs_map["v_jnt"]+self._n_jnts]=True # also good for noise

        # self._is_substep_obs[self._obs_map["contact_f_mpc"]:self._obs_map["contact_f_mpc"]+3*len(self._contact_names)]=True

    def _get_action_names(self):
        
        action_names = [""] * self.actions_dim()
        action_names[0] = "vx_cmd" # twist commands from agent to RHC controller
        action_names[1] = "vy_cmd"
        action_names[2] = "vz_cmd"
        action_names[3] = "roll_omega_cmd"
        action_names[4] = "pitch_omega_cmd"
        action_names[5] = "yaw_omega_cmd"

        next_idx=6
        
        self._actions_map["contact_flag_start"]=next_idx
        for i in range(len(self._contact_names)):
            contact=self._contact_names[i]
            action_names[next_idx] = f"contact_flag_{contact}"
            next_idx+=1

        return action_names
    
    def _set_substep_rew(self):

        # which rewards are to be computed at substeps frequency?
        self._is_substep_rew[self._reward_map["task_error"]]=False
        if self._env_opts["add_CoT_reward"]:
            self._is_substep_rew[self._reward_map["CoT"]]=True
        if self._env_opts["add_power_reward"]:
            self._is_substep_rew[self._reward_map["mech_pow"]]=True
        if self._env_opts["add_action_rate_reward"]:
            self._is_substep_rew[self._reward_map["action_rate"]]=False
        if self._env_opts["add_jnt_v_reward"]:
            self._is_substep_rew[self._reward_map["jnt_v"]]=True

        if self._env_opts["use_rhc_avrg_vel_tracking"]:
            self._is_substep_rew[self._reward_map["rhc_avrg_vel_error"]]=False

    def _get_rewards_names(self):
        
        counter=0
        reward_names = []

        # adding rewards
        reward_names.append("task_error")
        self._reward_map["task_error"]=counter
        counter+=1
        if self._env_opts["add_power_reward"] and self._env_opts["add_CoT_reward"]:
            Journal.log(self.__class__.__name__,
                    "__init__",
                    "Only one between CoT and power reward can be used!",
                    LogType.EXCEP,
                    throw_when_excep=True)
        if self._env_opts["add_CoT_reward"]:
            reward_names.append("CoT")
            self._reward_map["CoT"]=counter
            counter+=1
        if self._env_opts["add_power_reward"]:
            reward_names.append("mech_pow")
            self._reward_map["mech_pow"]=counter
            counter+=1
        if self._env_opts["add_action_rate_reward"]:
            reward_names.append("action_rate")   
            self._reward_map["action_rate"]=counter
            counter+=1   
        if self._env_opts["add_jnt_v_reward"]:
            reward_names.append("jnt_v")   
            self._reward_map["jnt_v"]=counter
            counter+=1   
        if self._env_opts["use_rhc_avrg_vel_tracking"]:
            reward_names.append("rhc_avrg_vel_error")   
            self._reward_map["rhc_avrg_vel_error"]=counter
            counter+=1   

        return reward_names

    def _get_sub_trunc_names(self):
        sub_trunc_names = []
        sub_trunc_names.append("ep_timeout")
        if self._env_opts["single_task_ref_per_episode"]:
            sub_trunc_names.append("task_ref_rand")
        return sub_trunc_names

    def _get_sub_term_names(self):
        # to be overridden by child class
        sub_term_names = []
        sub_term_names.append("rhc_failure")
        sub_term_names.append("robot_capsize")
        if self._env_opts["add_term_mpc_capsize"]:
            sub_term_names.append("rhc_capsize")

        return sub_term_names

    def _set_jnts_blacklist_pattern(self):
        # used to exclude pos measurement from wheels
        self._jnt_q_blacklist_patterns=["wheel"]

