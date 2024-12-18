from lrhc_control.utils.sys_utils import PathsGetter
from lrhc_control.envs.linvel_env_baseline import LinVelTrackBaseline

from control_cluster_bridge.utilities.shared_data.rhc_data import RobotState, RhcStatus
from control_cluster_bridge.utilities.math_utils_torch import world2base_frame, base2world_frame, w2hor_frame

import torch

from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import LogType

import os
from lrhc_control.utils.episodic_data import EpisodicData
from lrhc_control.utils.signal_smoother import ExponentialSignalSmoother
import math

from lrhc_control.utils.math_utils import check_capsize

class VariableFlightsBaseline(LinVelTrackBaseline):

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
            actions_dim=22, # twist + contact flags + flight params (length, apex, end)Xn_contacts
            verbose=verbose,
            vlevel=vlevel,
            use_gpu=use_gpu,
            dtype=dtype,
            debug=debug,
            override_agent_refs=override_agent_refs,
            timeout_ms=timeout_ms)
    
    def get_file_paths(self):
        paths=super().get_file_paths()
        paths.append(os.path.abspath(__file__))        
        return paths

    def _custom_post_init(self):
        
        super()._custom_post_init()

        # actions bounds
        if not self._use_prob_based_stepping:
            self._is_continuous_actions[6:10]=False
        v_cmd_max = 3*self.max_ref
        omega_cmd_max = 3*self.max_ref
        self._actions_lb[:, 0:3] = -v_cmd_max 
        self._actions_ub[:, 0:3] = v_cmd_max  
        self._actions_lb[:, 3:6] = -omega_cmd_max # twist cmds
        self._actions_ub[:, 3:6] = omega_cmd_max  
        if self._use_prob_based_stepping:
            self._actions_lb[:, 6:10] = 0.0 # contact flags
            self._actions_ub[:, 6:10] = 1.0 
        else:
            self._actions_lb[:, 6:10] = -1.0 
            self._actions_ub[:, 6:10] = 1.0 
        # flight params (length)
        self._actions_lb[:, 10:14]=3
        self._actions_ub[:, 10:14]=self._n_nodes_rhc.mean().item()
        # flight params (apex)
        self._actions_lb[:, 14:18]=0.0
        self._actions_ub[:, 14:18]=0.5
        # flight params (end)
        self._actions_lb[:, 18:22]=0.0
        self._actions_ub[:, 18:22]=0.5

        self._default_action[:, :] = (self._actions_ub+self._actions_lb)/2.0
        self._default_action[:, ~self._is_continuous_actions] = 1.0

    def _set_refs(self):
        super()._set_refs()

        action_to_be_applied = self.get_actual_actions()

        flight_settings = self._rhc_refs.flight_settings.get_torch_mirror(gpu=self._use_gpu)
        flight_settings[:, :]=action_to_be_applied[:, 10:(10+flight_settings.shape[1])]

    def _write_refs(self):
        super()._write_refs()
        if self._use_gpu:
            self._rhc_refs.flight_settings.synch_mirror(from_gpu=True,non_blocking=False)
        self._rhc_refs.flight_settings.synch_all(read=False, retry=True)
        
    def _get_action_names(self):

        action_names = [""] * self.actions_dim()
        action_names[0] = "vx_cmd" # twist commands from agent to RHC controller
        action_names[1] = "vy_cmd"
        action_names[2] = "vz_cmd"
        action_names[3] = "roll_omega_cmd"
        action_names[4] = "pitch_omega_cmd"
        action_names[5] = "yaw_omega_cmd"
        action_names[6] = "contact_0"
        action_names[7] = "contact_1"
        action_names[8] = "contact_2"
        action_names[9] = "contact_3"

        action_names[10] = "contact_len0"
        action_names[11] = "contact_len1"
        action_names[12] = "contact_len2"
        action_names[13] = "contact_len3"
        action_names[14] = "contact_apex0"
        action_names[15] = "contact_apex1"
        action_names[16] = "contact_apex2"
        action_names[17] = "contact_apex3"
        action_names[18] = "contact_end0"
        action_names[19] = "contact_end1"
        action_names[20] = "contact_end2"
        action_names[21] = "contact_end3"

        return action_names


