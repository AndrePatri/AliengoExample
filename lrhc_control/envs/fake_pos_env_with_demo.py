import torch
import os

from EigenIPC.PyEigenIPC import VLevel, LogType, Journal

from control_cluster_bridge.utilities.math_utils_torch import base2world_frame, w2hor_frame

from lrhc_control.utils.gait_scheduler import QuadrupedGaitPatternGenerator, GaitScheduler
from lrhc_control.utils.signal_smoother import ExponentialSignalSmoother

from lrhc_control.envs.fake_pos_env_baseline import FakePosEnvBaseline

class FakePosEnvWithDemo(FakePosEnvBaseline):

    def __init__(self,
            namespace: str,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1,
            use_gpu: bool = True,
            dtype: torch.dtype = torch.float32,
            debug: bool = True,
            override_agent_refs: bool = False,
            timeout_ms: int = 60000):
        
        self._full_demo=False # whether to override the full action
        self._smooth_twist_cmd=True
        self._smoothing_horizon_twist=0.08

        FakePosEnvBaseline.__init__(self,
            namespace=namespace,
            verbose=verbose,
            vlevel=vlevel,
            use_gpu=use_gpu,
            dtype=dtype,
            debug=debug,
            override_agent_refs=override_agent_refs,
            timeout_ms=timeout_ms)
        
        if self._demo_envs_idxs is None:
            Journal.log(self.__class__.__name__,
                "__init__",
                "No demo environments present. Aborting",
                LogType.EXCEP,
                throw_when_excep=True)
            
        self.switch_demo(active=True) # enable using demonstrations by default
        # (can be deactived externally)

    def get_file_paths(self):
        paths=FakePosEnvBaseline.get_file_paths(self)
        paths.append(os.path.abspath(__file__))        
        return paths
    
    def _init_demo_envs(self):
        
        self._env_to_gait_sched_mapping=torch.full((self._n_envs, ), dtype=torch.int, device=self._device,
                                    fill_value=-self._n_envs+1)
        
        counter=0
        for i in range(self._n_envs):
            if self._demo_envs_idxs_bool[i]:
                self._env_to_gait_sched_mapping[i]=counter
                counter+=1

        self._init_gait_schedulers()

        self._twist_smoother=None
        if self._smooth_twist_cmd:
            twist_proxy = self._agent_refs.rob_refs.root_state.get(data_type="twist",gpu=self._use_gpu)
            self._twist_smoother=ExponentialSignalSmoother(signal=twist_proxy[self._demo_envs_idxs, :],
                update_dt=self._substep_dt*self._action_repeat, # rate at which actions are decided by agent
                smoothing_horizon=self._smoothing_horizon_twist,
                target_smoothing=0.5, 
                debug=self._debug,
                dtype=self._dtype,
                use_gpu=self._use_gpu,
                name=self.__class__.__name__+"TwistCmdSmoother")
        
    def _init_gait_schedulers(self):
        
        self._stopping_thresh=0.01
        
        # self._walk_to_trot_thresh=0.5 # [m/s] # kyon no wheels
        # self._walk_to_trot_thresh_omega=0.5 # [m/s]
        # phase_period_walk=2.0 # kyon no wheels
        # phase_period_trot=1.5

        # self._walk_to_trot_thresh=3.0 # [m/s] # kyon wheels
        # self._walk_to_trot_thresh_omega=3.0 # [m/s]
        # phase_period_walk=2.0 # kyon wheels
        # phase_period_trot=2.0

        self._walk_to_trot_thresh=3.0 # [m/s] # centauro no wheels
        self._walk_to_trot_thresh_omega=3.0 # [m/s]
        phase_period_walk=3.5 # centauro
        phase_period_trot=1.5

        # self._walk_to_trot_thresh=3.0 # [m/s] # centauro with wheels
        # self._walk_to_trot_thresh_omega=3.0 # [m/s] 
        # phase_period_walk=3.5 # centauro
        # phase_period_trot=1.5

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
        
        FakePosEnvBaseline._custom_post_init(self)

        self._agent_twist_ref_h = self._rhc_twist_cmd_rhc_h.clone()
        self._agent_twist_ref_w = self._rhc_twist_cmd_rhc_h.clone()

    def _custom_post_step(self,episode_finished):
        FakePosEnvBaseline._custom_post_step(self,episode_finished=episode_finished)
        # executed after checking truncations and terminations
        if self.demo_active():
            finished_and_demo=torch.logical_and(episode_finished.flatten(), self._demo_envs_idxs_bool)
            finished_demo_idxs=self._env_to_gait_sched_mapping[finished_and_demo]
            terminated_and_demo=torch.logical_and(self._terminations.get_torch_mirror(gpu=self._use_gpu).flatten(), 
                                    self._demo_envs_idxs_bool)
            terminated_and_demo_idxs=self._env_to_gait_sched_mapping[terminated_and_demo]
            if self._use_gpu:
                self._gait_scheduler_walk.reset(to_be_reset=terminated_and_demo_idxs.cuda().flatten())
                self._gait_scheduler_trot.reset(to_be_reset=terminated_and_demo_idxs.cuda().flatten())
                # self._twist_smoother.reset_all(to_be_reset=finished_demo_idxs.cuda().flatten(),
                #     value=0.0)
            else:
                self._gait_scheduler_walk.reset(to_be_reset=terminated_and_demo_idxs.cpu().flatten())
                self._gait_scheduler_trot.reset(to_be_reset=terminated_and_demo_idxs.cpu().flatten())
                # self._twist_smoother.reset_all(to_be_reset=finished_demo_idxs.cpu().flatten(),
                #     value=0.0)
            
            if self._twist_smoother is not None: # smoother only reset at terminations
                
                terminated_demo_idxs=self._env_to_gait_sched_mapping[terminated_and_demo]
                if self._use_gpu:
                    self._twist_smoother.reset_all(to_be_reset=terminated_demo_idxs.cuda().flatten(),
                        value=0.0)
                else:
                    self._twist_smoother.reset_all(to_be_reset=terminated_demo_idxs.cpu().flatten(),
                        value=0.0)


    def _override_actions_with_demo(self):
        
        if self.demo_active():
            
            # get some data
            agent_action = self.get_actions()
            agent_twist_ref_current = self._agent_refs.rob_refs.root_state.get(data_type="twist",gpu=self._use_gpu)

            # overwriting agent gait actions with the ones taken from the gait schedulers for 
            # (just for demonstration environments)

            self._gait_scheduler_walk.step()
            self._gait_scheduler_trot.step()
            
            have_to_go_fast_linvel=agent_twist_ref_current[:, 0:2].norm(dim=1,keepdim=True)>self._walk_to_trot_thresh
            have_to_go_fast_omega=agent_twist_ref_current[:, 3:6].norm(dim=1,keepdim=True)>self._walk_to_trot_thresh_omega
            have_to_go_fast=torch.logical_or(have_to_go_fast_linvel, have_to_go_fast_omega)

            fast_and_demo=torch.logical_and(have_to_go_fast.flatten(),self._demo_envs_idxs_bool)
            have_to_go_slow_and_demo=~fast_and_demo

            have_to_stop_linvel=agent_twist_ref_current[:, 0:2].norm(dim=1,keepdim=True)<self._stopping_thresh
            have_to_stop_omega=agent_twist_ref_current[:, 3:6].norm(dim=1,keepdim=True)<self._stopping_thresh
            have_to_stop=torch.logical_and(have_to_stop_linvel, have_to_stop_omega)
            stop_and_demo=torch.logical_and(have_to_stop.flatten(),self._demo_envs_idxs_bool)
            
            # default to walk
            walk_signal=self._gait_scheduler_walk.get_signal(clone=True)[self._env_to_gait_sched_mapping[self._demo_envs_idxs_bool], :]
            is_contact_walk=walk_signal>self._gait_scheduler_walk.threshold()
            agent_action[self._demo_envs_idxs, 6:10] = 2.0*is_contact_walk-1.0

            if fast_and_demo.any():
                # for fast enough refs, trot
                trot_signal=self._gait_scheduler_trot.get_signal(clone=True)[self._env_to_gait_sched_mapping[fast_and_demo], :]
                is_contact_trot=trot_signal>self._gait_scheduler_trot.threshold()
                agent_action[fast_and_demo, 6:10] = 2.0*is_contact_trot-1.0
            
            if stop_and_demo.any():
                # keep contact
                agent_action[stop_and_demo, 6:10] = 1.0

            if self._full_demo:
                # overwriting agent's twist action with the identity wrt the reference (both 
                # are base frame)
                if self._twist_smoother is not None:
                    # smooth action
                    if have_to_go_slow_and_demo.any(): # higher twist ref for walking
                        agent_twist_ref_current[have_to_go_slow_and_demo, 0:6]=agent_twist_ref_current[have_to_go_slow_and_demo, 0:6]
                    self._twist_smoother.update(new_signal=
                        agent_twist_ref_current[self._demo_envs_idxs, :])
                    agent_action[self._demo_envs_idxs, 0:6]=self._twist_smoother.get()
                else:
                # agent_twist_ref_current and agent action twist are base local
                    if have_to_go_slow_and_demo.any(): # higher twist ref for walking
                        agent_twist_ref_current[have_to_go_slow_and_demo, 0:6]=agent_twist_ref_current[have_to_go_slow_and_demo, 0:6]
                    agent_action[self._demo_envs_idxs, 0:6]=agent_twist_ref_current[self._demo_envs_idxs, :]
                    
                agent_action[stop_and_demo, 0:6]=0.0
