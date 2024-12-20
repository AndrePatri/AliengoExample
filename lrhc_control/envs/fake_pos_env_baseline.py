from lrhc_control.utils.sys_utils import PathsGetter
from lrhc_control.envs.lrhc_training_env_base import LRhcTrainingEnvBase

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
from lrhc_control.envs.linvel_env_baseline import LinVelTrackBaseline

class FakePosEnvBaseline(LinVelTrackBaseline):

    def __init__(self,
            namespace: str,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1,
            use_gpu: bool = True,
            dtype: torch.dtype = torch.float32,
            debug: bool = True,
            override_agent_refs: bool = False,
            timeout_ms: int = 60000):

        self._max_distance=5.0 # [m]
        self._min_distance=0.0 
        self._max_vref=1.0 # [m/s]
        super().__init__(namespace=namespace,
            actions_dim=10, # twist + contact flags
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
        
        # position targets to be reached (wrt robot's pos at ep start)
        self._p_trgt_w=self._robot_state.root_state.get(data_type="p",gpu=self._use_gpu).detach().clone()
        self._p_delta_w=self._p_trgt_w.detach().clone()

        self._trgt_d=torch.zeros((self._n_envs, 1),dtype=self._dtype,device=self._device)
        self._trgt_theta=torch.zeros((self._n_envs, 1),dtype=self._dtype,device=self._device)

    def _update_loc_twist_refs(self):
        super()._update_loc_twist_refs()

    def _randomize_task_refs(self,
        env_indxs: torch.Tensor = None):

        # we randomize the reference in world frame

        if env_indxs is None:
            self._trgt_d.uniform_(a=self._min_distance, b=self._max_distance)
            self._trgt_theta.uniform_(a=0.0, b=2*torch.pi)

            self._p_trgt_w[:, :]=torch.cat(self._trgt_d*torch.cos(self._trgt_theta),self._trgt_d*torch.sin(self._trgt_theta), dim=1) -\
                self._robot_state.root_state.get(data_type="p",gpu=self._use_gpu)[:, 0:2]
            self._p_trgt_w[:, :]=

            self._agent_twist_ref_current_w[:, :] = random_uniform*self._twist_ref_scale + self._twist_ref_offset
            self._agent_twist_ref_current_w[:, :] = self._agent_twist_ref_current_w*self._bernoulli_coeffs
        else:
            self._trgt_d[env_indxs, :].uniform_(a=self._min_distance, b=self._max_distance)
            self._trgt_d[env_indxs, :].uniform_(a=0.0, b=2*torch.pi)

            self._agent_twist_ref_current_w[env_indxs, :] = random_uniform * self._twist_ref_scale + self._twist_ref_offset
            self._agent_twist_ref_current_w[env_indxs, :] = self._agent_twist_ref_current_w[env_indxs, :]*self._bernoulli_coeffs[env_indxs, :]

