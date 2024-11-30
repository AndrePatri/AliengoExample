import torch

from EigenIPC.PyEigenIPC import LogType, VLevel
from control_cluster_bridge.utilities.math_utils_torch import w2hor_frame

import os

from lrhc_control.envs.linvel_env_baseline import LinVelTrackBaseline

class RandomSteppingEnvBaseline(LinVelTrackBaseline):

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
            actions_dim=4, # only contacts
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
    
    def _apply_actions_to_rhc(self):
        
        agent_action = self.get_actions() # see _get_action_names() to get 
        # the meaning of each component of this tensor

        rhc_latest_twist_ref = self._rhc_refs.rob_refs.root_state.get(data_type="twist", gpu=self._use_gpu)
        agent_twist_ref_current = self._agent_refs.rob_refs.root_state.get(data_type="twist",gpu=self._use_gpu)
        rhc_latest_contact_ref = self._rhc_refs.contact_flags.get_torch_mirror(gpu=self._use_gpu)

        # overwriting agent actions with gait scheduler ones
        # default to walk
        random_uniform=torch.full_like(agent_action, fill_value=0.0)
        torch.nn.init.uniform_(random_uniform, a=-1, b=1)
        agent_action[:, :] = random_uniform
        
        # refs have to be applied in the MPC's horizontal frame
        robot_q_meas = self._robot_state.root_state.get(data_type="q",gpu=self._use_gpu)
        w2hor_frame(t_w=agent_twist_ref_current,q_b=robot_q_meas,t_out=self._agent_twist_ref_h)
        # 2D lin vel applied directly to MPC
        rhc_latest_twist_ref[:, 0:2] = self._agent_twist_ref_h[:, 0:2] # 2D lin vl
        rhc_latest_twist_ref[:, 5:6] = self._agent_twist_ref_h[:, 5:6] # yaw twist

        self._rhc_refs.rob_refs.root_state.set(data_type="twist", data=rhc_latest_twist_ref,
                                            gpu=self._use_gpu) 
        
        # agent sets contact flags
        rhc_latest_contact_ref[:, :] = agent_action[:, 0:4] > 0 # keep contact if agent action > 0

        # actually apply actions to controller
        if self._use_gpu:
            # GPU->CPU --> we cannot use asynchronous data transfer since it's unsafe
            self._rhc_refs.rob_refs.root_state.synch_mirror(from_gpu=True,non_blocking=False) # write from gpu to cpu mirror
            self._rhc_refs.contact_flags.synch_mirror(from_gpu=True,non_blocking=False)
        self._rhc_refs.rob_refs.root_state.synch_all(read=False, retry=True) # write mirror to shared mem
        self._rhc_refs.contact_flags.synch_all(read=False, retry=True)

    def _get_action_names(self):

        action_names = [""] * self.actions_dim()
        action_names[0] = "contact_0"
        action_names[1] = "contact_1"
        action_names[2] = "contact_2"
        action_names[3] = "contact_3"

        return action_names