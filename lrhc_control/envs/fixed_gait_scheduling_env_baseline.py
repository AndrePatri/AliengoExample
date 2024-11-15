import torch

from SharsorIPCpp.PySharsorIPC import VLevel, LogType
from control_cluster_bridge.utilities.math_utils_torch import world2base_frame, base2world_frame, w2hor_frame

import os

from lrhc_control.utils.gait_scheduler import QuadrupedGaitPatternGenerator, GaitScheduler

from lrhc_control.envs.linvel_env_baseline import LinVelTrackBaseline

class FixedGaitSchedEnvBaseline(LinVelTrackBaseline):

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

    def _custom_post_init(self):
        
        super()._custom_post_init()

        self._agent_twist_ref_h = self._rhc_twist_cmd_rhc_h.clone()
        self._agent_twist_ref_w = self._rhc_twist_cmd_rhc_h.clone()
        
        phase_period_walk=2.0
        update_dt_walk = self._substep_dt*self._action_repeat
        self._pattern_gen_walk = QuadrupedGaitPatternGenerator(phase_period=phase_period_walk)
        gait_params_walk = self._pattern_gen_walk.get_params("walk")
        n_phases = gait_params_walk["n_phases"]
        phase_period = gait_params_walk["phase_period"]
        phase_offset = gait_params_walk["phase_offset"]
        phase_thresh = gait_params_walk["phase_thresh"]
        self._gait_scheduler_walk = GaitScheduler(
            n_phases=n_phases,
            n_envs=self._n_envs,
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
            n_envs=self._n_envs,
            update_dt=update_dt_trot,
            phase_period=phase_period,
            phase_offset=phase_offset,
            phase_thresh=phase_thresh,
            use_gpu=self._use_gpu,
            dtype=self._dtype
        )

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
        # just override how actions are applied wrt base env

        agent_action = self.get_actions() # see _get_action_names() to get 
        # the meaning of each component of this tensor

        rhc_latest_twist_cmd = self._rhc_refs.rob_refs.root_state.get(data_type="twist", gpu=self._use_gpu)
        agent_twist_ref_current = self._agent_refs.rob_refs.root_state.get(data_type="twist",gpu=self._use_gpu)
        rhc_latest_contact_ref = self._rhc_refs.contact_flags.get_torch_mirror(gpu=self._use_gpu)
        rhc_q=self._rhc_cmds.root_state.get(data_type="q",gpu=self._use_gpu) # this is always 
        # avaialble

        # overwriting agent gait actions with gait scheduler ones
        self._gait_scheduler_walk.step()
        self._gait_scheduler_trot.step()
        walk_to_trot_thresh=0.8 # [m/s]
        stopping_thresh=0.05
        have_to_go_fast=agent_twist_ref_current[0:3].norm(dim=1,keepdim=True)>walk_to_trot_thresh
        have_to_stop=agent_twist_ref_current[0:3].norm(dim=1,keepdim=True)<stopping_thresh
        # default to walk
        agent_action[:, 6:10] = self._gait_scheduler_walk.get_signal(clone=True)
        # for fast enough refs, trot
        agent_action[have_to_go_fast.flatten(), 6:10] = \
            self._gait_scheduler_trot.get_signal(clone=True)[have_to_go_fast.flatten(), :]
        agent_action[have_to_stop.flatten(), 6:10] = 1.0
        
        # copying twist reference into action 
        agent_action[:, 0:6] = agent_twist_ref_current

        # reference twist for MPC is assumed to always be specified in MPC's 
        # horizontal frame, while agent actions are interpreted as in MPC's
        # base frame -> we need to rotate the actions into the horizontal frame
        base2world_frame(t_b=agent_action[:, 0:6],q_b=rhc_q,t_out=self._rhc_twist_cmd_rhc_world)
        w2hor_frame(t_w=self._rhc_twist_cmd_rhc_world,q_b=rhc_q,t_out=self._rhc_twist_cmd_rhc_h)

        rhc_latest_twist_cmd[:, 0:6] = self._rhc_twist_cmd_rhc_h

        self._rhc_refs.rob_refs.root_state.set(data_type="twist", data=rhc_latest_twist_cmd,
            gpu=self._use_gpu) 
        
        # agent sets contact flags
        rhc_latest_contact_ref[:, :] = agent_action[:, 6:10] > self._gait_scheduler_walk.threshold() # keep contact if agent action > 0
        rhc_latest_contact_ref[have_to_go_fast.flatten(), :] = agent_action[:, 6:10] > self._gait_scheduler_trot.threshold() 

        if self._use_gpu:
            # GPU->CPU --> we cannot use asynchronous data transfer since it's unsafe
            self._rhc_refs.rob_refs.root_state.synch_mirror(from_gpu=True,non_blocking=False) # write from gpu to cpu mirror
            self._rhc_refs.contact_flags.synch_mirror(from_gpu=True,non_blocking=False)
            self._rhc_refs.rob_refs.contact_pos.synch_mirror(from_gpu=True,non_blocking=False)

        self._rhc_refs.rob_refs.root_state.synch_all(read=False, retry=True) # write mirror to shared mem
        self._rhc_refs.contact_flags.synch_all(read=False, retry=True)
        self._rhc_refs.rob_refs.contact_pos.synch_all(read=False, retry=True)