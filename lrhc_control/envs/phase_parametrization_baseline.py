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

class PhaseParametrizationBaseline(LinVelTrackBaseline):

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

        self._add_env_opt(env_opts, "control_flength", default=False)
        self._add_env_opt(env_opts, "control_fapex", default=False) 
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
        
        actions_dim=6 # base size
        actions_dim+=n_contacts # frequency
        actions_dim+=n_contacts # offsets
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
        self._add_env_opt(self._env_opts, "phase_vecfreq_max", default=self._env_opts["n_steps_task_rand_ub"]*self._action_repeat) # substeps
        self._add_env_opt(self._env_opts, "flight_freq_lb_thresh", default=1.0/self._env_opts["phase_vecfreq_max"]) # substeps

        # actions bounds        
        v_cmd_max = 2.5*self._env_opts["max_cmd_v"]
        omega_cmd_max = 2.5*self._env_opts["max_cmd_v"]
        self._actions_lb[:, 0:3] = -v_cmd_max 
        self._actions_ub[:, 0:3] = v_cmd_max  
        self._actions_lb[:, 3:6] = -omega_cmd_max # twist cmds
        self._actions_ub[:, 3:6] = omega_cmd_max  

        idx=self._actions_map["flights_per_substeps_start"] # n. flights/n. substeps [0, 1.0/min_flight_length]
        self._actions_lb[:, idx:idx+self._n_contacts] = 0.0
        self._actions_ub[:, idx:idx+self._n_contacts] = 1.0/self._env_opts["flength_min"]

        idx=self._actions_map["flight_offset_start"] # [0, 1) where 1 is an offset of an offset of a full period
        self._actions_lb[:, idx:idx+self._n_contacts] = 0.0
        self._actions_ub[:, idx:idx+self._n_contacts] = 1.0

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
    
    def _custom_post_step(self,episode_finished):
        LinVelTrackBaseline._custom_post_step(self, episode_finished=episode_finished)
    
    def _set_rhc_refs(self):
        
        action_to_be_applied = self.get_actual_actions() # see _get_action_names() to get 
        # the meaning of each component of this tensor

        # twist refs
        rhc_latest_twist_cmd = self._rhc_refs.rob_refs.root_state.get(data_type="twist", gpu=self._use_gpu)
        rhc_q=self._rhc_cmds.root_state.get(data_type="q",gpu=self._use_gpu) # this is always 
        # avaialble
        rhc_latest_contact_ref = self._rhc_refs.contact_flags.get_torch_mirror(gpu=self._use_gpu)

        # reference twist for MPC is assumed to always be specified in MPC's 
        # horizontal frame, while agent actions are interpreted as in MPC's
        # base frame -> we need to rotate the actions into the horizontal frame
        base2world_frame(t_b=action_to_be_applied[:, 0:6],q_b=rhc_q,t_out=self._rhc_twist_cmd_rhc_world)
        w2hor_frame(t_w=self._rhc_twist_cmd_rhc_world,q_b=rhc_q,t_out=self._rhc_twist_cmd_rhc_h)

        rhc_latest_twist_cmd[:, 0:6] = self._rhc_twist_cmd_rhc_h
        
        self._rhc_refs.rob_refs.root_state.set(data_type="twist", data=rhc_latest_twist_cmd,
            gpu=self._use_gpu) 

        # flight settings
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
        # do not write contact flags (those are written @ the MPC freq by _pre_substep)
        if self._use_gpu:
            self._rhc_refs.rob_refs.root_state.synch_mirror(from_gpu=True,non_blocking=False) # write from gpu to cpu mirror
            self._rhc_refs.rob_refs.contact_pos.synch_mirror(from_gpu=True,non_blocking=False)
            self._rhc_refs.flight_settings.synch_mirror(from_gpu=True,non_blocking=False)

        self._rhc_refs.rob_refs.root_state.synch_all(read=False, retry=True) # write mirror to shared mem
        self._rhc_refs.rob_refs.contact_pos.synch_all(read=False, retry=True)
        self._rhc_refs.flight_settings.synch_all(read=False, retry=True)
    
    def _pre_substep(self):
        # runs at substep freq (MPC freq)
        rhc_latest_contact_ref = self._rhc_refs.contact_flags.get_torch_mirror(gpu=self._use_gpu)
        action_to_be_applied = self.get_actual_actions()

        # handle phase frequency and offset and set MPC reference accordingly
        start_freq=self._actions_map["flights_per_substeps_start"]
        zero_freq=action_to_be_applied[:, start_freq:(start_freq+self._n_contacts)]<self._env_opts["flight_freq_lb_thresh"]
        action_to_be_applied[:, start_freq:(start_freq+self._n_contacts)][zero_freq]=0.0 # set exactly zero freq
        for i in range(self._n_contacts):
            idx_freq=start_freq+i
            idx_offset=self._actions_map["flight_offset_start"]+i
            # compute period of phase for each contact [n. substeps]
            period_n_substeps_cmd=(1.0/(action_to_be_applied[:, idx_freq:(idx_freq+1)])).to(dtype=torch.int32)
            
            # compute phase offset for each contact [n. substeps]
            offset_n_substeps_cmd=(action_to_be_applied[:, idx_offset:(idx_offset+1)]*period_n_substeps_cmd).to(dtype=torch.int32)
            time_to_insert_flights=self._substep_abs_counter.time_limits_reached(limit=period_n_substeps_cmd,
                                            offset=offset_n_substeps_cmd)# robust against 0 freq since freq is always <1
            
            rhc_latest_contact_ref[:, i:i+1] =~time_to_insert_flights

        # write right away to mpc
        if self._use_gpu:
            self._rhc_refs.contact_flags.synch_mirror(from_gpu=True,non_blocking=False)
        self._rhc_refs.contact_flags.synch_all(read=False, retry=True)

    def _get_action_names(self):

        action_names = [""] * self.actions_dim()
        action_names[0] = "vx_cmd" # twist commands from agent to RHC controller
        action_names[1] = "vy_cmd"
        action_names[2] = "vz_cmd"
        action_names[3] = "roll_omega_cmd"
        action_names[4] = "pitch_omega_cmd"
        action_names[5] = "yaw_omega_cmd"

        next_idx=6
        self._actions_map["flights_per_substeps_start"]=next_idx
        for i in range(len(self._contact_names)):
            contact=self._contact_names[i]
            action_names[next_idx] = f"phase_freq_{contact}"
            next_idx+=1
        self._actions_map["flight_offset_start"]=next_idx
        for i in range(len(self._contact_names)):
            contact=self._contact_names[i]
            action_names[next_idx] = f"phase_offset_{contact}"
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


