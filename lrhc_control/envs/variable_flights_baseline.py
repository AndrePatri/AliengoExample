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

        self._add_env_opt(env_opts, "flength_min", default=3) # substeps

        self._add_env_opt(env_opts, "control_flength", default=True) 
        self._add_env_opt(env_opts, "control_fapex", default=True) 
        self._add_env_opt(env_opts, "control_fend", default=False) 
        
        # temporarily creating robot state client to get some data
        robot_state_tmp = RobotState(namespace=namespace,
                                is_server=False, 
                                safe=False,
                                verbose=verbose,
                                vlevel=vlevel,
                                with_gpu_mirror=False,
                                with_torch_view=False)
        robot_state_tmp.run()
        n_contacts = len(robot_state_tmp.contact_names())
        robot_state_tmp.close()
        
        actions_dim=10 # base size
        if env_opts["control_flength"]:
            actions_dim+=n_contacts
        if env_opts["control_fapex"]:
            actions_dim+=n_contacts
        if env_opts["control_fend"]:
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

    def get_file_paths(self):
        paths=LinVelTrackBaseline.get_file_paths(self)
        paths.append(os.path.abspath(__file__))        
        return paths

    def _custom_post_init(self):
        
        LinVelTrackBaseline._custom_post_init(self)

        self._add_env_opt(self._env_opts, "flength_max", default=self._n_nodes_rhc.mean().item()) # MPC steps (substeps)

        # actions bounds
        _=self._get_action_names() # also fills actions map
        
        if not self._env_opts["use_prob_based_stepping"]:
            self._is_continuous_actions[6:10]=False
        v_cmd_max = 2.5*self._env_opts["max_cmd_v"]
        omega_cmd_max = 2.5*self._env_opts["max_cmd_v"]
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
        
        # flight params (length)
        if self._env_opts["control_flength"]:
            idx=self._actions_map["flight_len_start"]
            self._actions_lb[:, idx:(idx+self._n_contacts)]=self._env_opts["flength_min"]
            self._actions_ub[:, idx:(idx+self._n_contacts)]=self._env_opts["flength_max"]
            self._is_continuous_actions[idx:(idx+self._n_contacts)]=True
        # flight params (apex)
        if self._env_opts["control_fapex"]:
            idx=self._actions_map["flight_apex_start"]
            self._actions_lb[:, idx:(idx+self._n_contacts)]=0.0
            self._actions_ub[:, idx:(idx+self._n_contacts)]=0.5
            self._is_continuous_actions[idx:(idx+self._n_contacts)]=True
        # flight params (end)
        if self._env_opts["control_fend"]:
            idx=self._actions_map["flight_end_start"]
            self._actions_lb[:, idx:(idx+self._n_contacts)]=0.0
            self._actions_ub[:, idx:(idx+self._n_contacts)]=0.5
            self._is_continuous_actions[idx:(idx+self._n_contacts)]=True

        self._default_action[:, :] = (self._actions_ub+self._actions_lb)/2.0
        self._default_action[:, ~self._is_continuous_actions] = 1.0

    def _set_rhc_refs(self):
        LinVelTrackBaseline._set_rhc_refs(self)

        action_to_be_applied = self.get_actual_actions()
        
        if self._env_opts["control_flength"]:
            idx=self._actions_map["flight_len_start"]
            flen_now=self._rhc_refs.flight_settings.get(data_type="len", gpu=self._use_gpu)
            flen_now[:, :]=action_to_be_applied[:, idx:(idx+self._n_contacts)]
            self._rhc_refs.flight_settings.set(data=flen_now, data_type="len", gpu=self._use_gpu)

        if self._env_opts["control_fapex"]:
            idx=self._actions_map["flight_apex_start"]
            fapex_now=self._rhc_refs.flight_settings.get(data_type="apex_dpos", gpu=self._use_gpu)
            fapex_now[:, :]=action_to_be_applied[:, idx:(idx+self._n_contacts)]
            self._rhc_refs.flight_settings.set(data=fapex_now, data_type="apex_dpos", gpu=self._use_gpu)
            
        if self._env_opts["control_fend"]:
            idx=self._actions_map["flight_end_start"]
            fend_now=self._rhc_refs.flight_settings.get(data_type="end_dpos", gpu=self._use_gpu)
            fend_now[:, :]=action_to_be_applied[:, idx:(idx+self._n_contacts)]
            self._rhc_refs.flight_settings.set(data=fend_now, data_type="end_dpos", gpu=self._use_gpu)

    def _write_rhc_refs(self):
        LinVelTrackBaseline._write_rhc_refs(self)
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

        next_idx=6
        self._actions_map["contact_flag_start"]=next_idx
        for i in range(len(self._contact_names)):
            contact=self._contact_names[i]
            action_names[next_idx] = f"contact_flag_{contact}"
            next_idx+=1
        if self._env_opts["control_flength"]:
            self._actions_map["flight_len_start"]=next_idx
            for i in range(len(self._contact_names)):
                contact=self._contact_names[i]
                action_names[next_idx+i] = f"flight_len_{contact}"
            next_idx+=len(self._contact_names)
        if self._env_opts["control_fapex"]:
            self._actions_map["flight_apex_start"]=next_idx
            for i in range(len(self._contact_names)):
                contact=self._contact_names[i]
                action_names[next_idx+i] = f"flight_apex_{contact}"
            next_idx+=len(self._contact_names)
        if self._env_opts["control_fend"]:
            self._actions_map["flight_end_start"]=next_idx
            for i in range(len(self._contact_names)):
                contact=self._contact_names[i]
                action_names[next_idx+i] = f"flight_end_{contact}"
            next_idx+=len(self._contact_names)

        return action_names


