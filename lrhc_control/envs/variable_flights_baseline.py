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

from typing import Dict

class VariableFlightsBaseline(LinVelTrackBaseline):

    def __init__(self,
            namespace: str,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1,
            use_gpu: bool = True,
            dtype: torch.dtype = torch.float32,
            debug: bool = True,
            override_agent_refs: bool = False,
            timeout_ms: int = 60000,
            env_opts: Dict = {}):

        self._control_flength=False
        self._control_fapex=True
        self._control_fend=True

        actions_dim=10
        n_contacts=4
        if self._control_flength:
            actions_dim+=n_contacts
        if self._control_fapex:
            actions_dim+=n_contacts
        if self._control_fend:
            actions_dim+=n_contacts

        LinVelTrackBaseline.__init__(self,
            namespace=namespace,
            actions_dim=actions_dim,
            verbose=verbose,
            vlevel=vlevel,
            use_gpu=use_gpu,
            dtype=dtype,
            debug=debug,
            override_agent_refs=override_agent_refs,
            timeout_ms=timeout_ms,
            env_opts=env_opts)

        self.custom_db_info["control_flength"] = self._control_flength
        self.custom_db_info["control_fapex"] = self._control_fapex
        self.custom_db_info["control_fend"] = self._control_fend

    def get_file_paths(self):
        paths=LinVelTrackBaseline.get_file_paths(self)
        paths.append(os.path.abspath(__file__))        
        return paths

    def _custom_post_init(self):
        
        LinVelTrackBaseline._custom_post_init(self)

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
        LinVelTrackBaseline._set_refs(self)

        action_to_be_applied = self.get_actual_actions()
        
        start_idx_params=10
        if self._control_flength:
            flen_now=self._rhc_refs.flight_settings.get(data_type="len", gpu=self._use_gpu)
            flen_now[:, :]=action_to_be_applied[:, start_idx_params:(start_idx_params+self._n_contacts)]
            self._rhc_refs.flight_settings.set(data=flen_now, data_type="len", gpu=self._use_gpu)
            start_idx_params+=self._n_contacts

        if self._control_fapex:
            fapex_now=self._rhc_refs.flight_settings.get(data_type="apex_dpos", gpu=self._use_gpu)
            fapex_now[:, :]=action_to_be_applied[:, start_idx_params:(start_idx_params+self._n_contacts)]
            self._rhc_refs.flight_settings.set(data=fapex_now, data_type="apex_dpos", gpu=self._use_gpu)
            start_idx_params+=self._n_contacts
            
        if self._control_fend:
            fend_now=self._rhc_refs.flight_settings.get(data_type="end_dpos", gpu=self._use_gpu)
            fend_now[:, :]=action_to_be_applied[:, start_idx_params:(start_idx_params+self._n_contacts)]
            self._rhc_refs.flight_settings.set(data=fend_now, data_type="end_dpos", gpu=self._use_gpu)
            start_idx_params+=self._n_contacts

    def _write_refs(self):
        LinVelTrackBaseline._write_refs(self)
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

        next_idx=10
        if self._control_flength:
            for i in range(len(self._contact_names)):
                contact=self._contact_names[i]
                action_names[next_idx+i] = f"contact_len_{contact}"
            next_idx+=len(self._contact_names)
        if self._control_fapex:
            for i in range(len(self._contact_names)):
                contact=self._contact_names[i]
                action_names[next_idx+i] = f"contact_apex_{contact}"
            next_idx+=len(self._contact_names)
        if self._control_fend:
            for i in range(len(self._contact_names)):
                contact=self._contact_names[i]
                action_names[next_idx+i] = f"contact_end_{contact}"
            next_idx+=len(self._contact_names)

        return action_names


