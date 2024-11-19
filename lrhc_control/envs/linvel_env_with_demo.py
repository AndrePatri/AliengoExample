import torch
import os

from SharsorIPCpp.PySharsorIPC import VLevel, LogType, Journal

from control_cluster_bridge.utilities.math_utils_torch import base2world_frame, w2hor_frame

from lrhc_control.utils.gait_scheduler import QuadrupedGaitPatternGenerator, GaitScheduler

from lrhc_control.envs.linvel_env_baseline import LinVelTrackBaseline

class LinVelEnvWithDemo(LinVelTrackBaseline):

    def __init__(self,
            namespace: str,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1,
            use_gpu: bool = True,
            dtype: torch.dtype = torch.float32,
            debug: bool = True,
            override_agent_refs: bool = False,
            timeout_ms: int = 60000):
        
        super().__init__(namespace=namespace,
            actions_dim=10, # only contacts
            verbose=verbose,
            vlevel=vlevel,
            use_gpu=use_gpu,
            dtype=dtype,
            debug=debug,
            override_agent_refs=override_agent_refs,
            timeout_ms=timeout_ms)
        
        self._n_demo_envs_perc = 0.5 # % [0, 1]
        self._n_demo_envs=round(self._n_demo_envs_perc*self._n_envs)

        if self._use_pos_control:
            Journal.log(self.__class__.__name__,
                "__init__",
                "not supported with pos control!",
                LogType.EXCEP,
                throw_when_excep=True)
        if self._use_prob_based_stepping:
            Journal.log(self.__class__.__name__,
                "__init__",
                "not supported with probabilistic-based stepping!",
                LogType.EXCEP,
                throw_when_excep=True)
            
        if not self._n_demo_envs >0:
            Journal.log(self.__class__.__name__,
                "__init__",
                "n_imitation_envs not > 0",
                LogType.EXCEP,
                throw_when_excep=True)
        else:
            Journal.log(self.__class__.__name__,
                "__init__",
                f"Will run with {self._n_demo_envs} imitation envs.",
                LogType.INFO)
        
        self._demo_envs_idxs = torch.randperm(self._n_envs, device=self._device)[:self._n_demo_envs]
        self._demo_envs_idxs_bool=torch.full((self._n_envs, ), dtype=torch.bool, device=self._device,
                                    fill_value=False)
        self._demo_envs_idxs_bool[self._demo_envs_idxs]=True

        self._env_to_gait_sched_mapping=torch.full((self._n_envs, ), dtype=torch.int, device=self._device,
                                    fill_value=-self._n_envs+1)
        counter=0
        for i in range(self._n_envs):
            if self._demo_envs_idxs_bool[i]:
                self._env_to_gait_sched_mapping[i]=counter
                counter+=1
        
        self._init_gait_schedulers()

    def get_file_paths(self):
        paths=super().get_file_paths()
        paths.append(os.path.abspath(__file__))        
        return paths
    
    def _init_gait_schedulers(self):

        self._walk_to_trot_thresh=0.3 # [m/s]
        self._stopping_thresh=0.05

        phase_period_walk=1.5
        update_dt_walk = self._substep_dt*self._action_repeat
        self._pattern_gen_walk = QuadrupedGaitPatternGenerator(phase_period=phase_period_walk)
        gait_params_walk = self._pattern_gen_walk.get_params("walk")
        n_phases = gait_params_walk["n_phases"]
        phase_period = gait_params_walk["phase_period"]
        phase_offset = gait_params_walk["phase_offset"]
        phase_thresh = gait_params_walk["phase_thresh"]
        self._gait_scheduler_walk = GaitScheduler(
            n_phases=n_phases,
            n_envs=self._n_demo_envs,
            update_dt=update_dt_walk,
            phase_period=phase_period,
            phase_offset=phase_offset,
            phase_thresh=phase_thresh,
            use_gpu=self._use_gpu,
            dtype=self._dtype
        )

        phase_period_trot=1.0
        update_dt_trot = self._substep_dt*self._action_repeat
        self._pattern_gen_trot = QuadrupedGaitPatternGenerator(phase_period=phase_period_trot)
        gait_params_trot = self._pattern_gen_trot.get_params("trot")
        n_phases = gait_params_trot["n_phases"]
        phase_period = gait_params_trot["phase_period"]
        phase_offset = gait_params_trot["phase_offset"]
        phase_thresh = gait_params_trot["phase_thresh"]
        self._gait_scheduler_trot = GaitScheduler(
            n_phases=n_phases,
            n_envs=self._n_demo_envs,
            update_dt=update_dt_trot,
            phase_period=phase_period,
            phase_offset=phase_offset,
            phase_thresh=phase_thresh,
            use_gpu=self._use_gpu,
            dtype=self._dtype
        )

    def _custom_post_init(self):
        
        super()._custom_post_init()

        self._agent_twist_ref_h = self._rhc_twist_cmd_rhc_h.clone()
        self._agent_twist_ref_w = self._rhc_twist_cmd_rhc_h.clone()

    def _custom_post_step(self,episode_finished):
        super()._custom_post_step(episode_finished=episode_finished)
        # executed after checking truncations and terminations
        # if self._use_gpu:
        #     self._gait_scheduler_walk.reset(to_be_reset=episode_finished.cuda().flatten())
        #     self._gait_scheduler_trot.reset(to_be_reset=episode_finished.cuda().flatten())
        # else:
        #     self._gait_scheduler_walk.reset(to_be_reset=episode_finished.cpu().flatten())
        #     self._gait_scheduler_trot.reset(to_be_reset=episode_finished.cuda().flatten())

    def _apply_actions_to_rhc(self):
        
        super()._set_refs()

        # get some data
        agent_action = self.get_actions()
        agent_twist_ref_current = self._agent_refs.rob_refs.root_state.get(data_type="twist",gpu=self._use_gpu)
        rhc_latest_contact_ref = self._rhc_refs.contact_flags.get_torch_mirror(gpu=self._use_gpu)

        # overwriting agent gait actions with ones taken from gait scheduler for 
        # demonstration environments

        self._gait_scheduler_walk.step()
        self._gait_scheduler_trot.step()
        
        have_to_go_fast=agent_twist_ref_current[:, 0:3].norm(dim=1,keepdim=True)>self._walk_to_trot_thresh

        fast_and_demo=torch.logical_and(have_to_go_fast.flatten(),self._demo_envs_idxs_bool)

        have_to_stop=agent_twist_ref_current[:, 0:3].norm(dim=1,keepdim=True)<self._stopping_thresh
        stop_and_demo=torch.logical_and(have_to_stop.flatten(),self._demo_envs_idxs_bool)

        # default to walk
        agent_action[self._demo_envs_idxs, 6:10] = self._gait_scheduler_walk.get_signal(clone=True)
        # for fast enough refs, trot
        
        agent_action[fast_and_demo, 6:10] = self._gait_scheduler_trot.get_signal(clone=True)[self._env_to_gait_sched_mapping[fast_and_demo], :]
        
        # agent sets contact flags
        rhc_latest_contact_ref[self._demo_envs_idxs, :] = agent_action[self._demo_envs_idxs, 6:10] > self._gait_scheduler_walk.threshold() # keep contact if agent action > 0
        rhc_latest_contact_ref[fast_and_demo, :] = agent_action[fast_and_demo, 6:10] > self._gait_scheduler_trot.threshold() 
        rhc_latest_contact_ref[stop_and_demo, :] = True # keep contact
        
        super()._write_refs() # finally, write refs

