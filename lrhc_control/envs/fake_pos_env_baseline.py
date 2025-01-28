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

from typing import Dict
class FakePosEnvBaseline(LinVelTrackBaseline):

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

        self._add_env_opt(env_opts, "max_distance", default=3.5) # [m]
        self._add_env_opt(env_opts, "min_distance", default=env_opts["max_distance"]-1.0) # [m]
        self._add_env_opt(env_opts, "max_vref", default=1.0) # [m/s]
        self._add_env_opt(env_opts, "max_dt", default=env_opts["max_distance"]/ env_opts["max_vref"])

        LinVelTrackBaseline.__init__(self, 
            namespace=namespace,
            actions_dim=actions_dim, # twist + contact flags
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
        
        # position targets to be reached (wrt robot's pos at ep start)
        self._p_trgt_w=self._robot_state.root_state.get(data_type="p",gpu=self._use_gpu)[:, 0:2].detach().clone()
        self._p_delta_w=self._p_trgt_w.detach().clone()
        self._dp_norm=torch.zeros((self._n_envs, 1),dtype=self._dtype,device=self._device)
        self._dp_versor=self._p_trgt_w.detach().clone()

        self._trgt_d=torch.zeros((self._n_envs, 1),dtype=self._dtype,device=self._device)
        self._trgt_theta=torch.zeros((self._n_envs, 1),dtype=self._dtype,device=self._device)

    def _update_loc_twist_refs(self):
        # this is called at each env substep
        
        self._compute_twist_ref_w()
    
        if not self._override_agent_refs:
            agent_p_ref_current=self._agent_refs.rob_refs.root_state.get(data_type="p",
            gpu=self._use_gpu)
            agent_p_ref_current[:, 0:2]=self._p_trgt_w

        # then convert it to base ref local for the agent
        robot_q = self._robot_state.root_state.get(data_type="q",gpu=self._use_gpu)
        # rotate agent ref from world to robot base
        world2base_frame(t_w=self._agent_twist_ref_current_w, q_b=robot_q, 
            t_out=self._agent_twist_ref_current_base_loc)
        # write it to agent refs tensors
        self._agent_refs.rob_refs.root_state.set(data_type="twist", data=self._agent_twist_ref_current_base_loc,
                                            gpu=self._use_gpu)

    def _compute_twist_ref_w(self, env_indxs: torch.Tensor = None):

        if env_indxs is None:
            # we update the position error using the current base position
            self._p_delta_w[:, :]=self._p_trgt_w-\
                self._robot_state.root_state.get(data_type="p",gpu=self._use_gpu)[:, 0:2]
                
            self._dp_norm[:, :]=self._p_delta_w.norm(dim=1,keepdim=True)+1e-6
            self._dp_versor[:, :]=self._p_delta_w/self._dp_norm

            # we compute the twist refs for the agent depending of the position error
            self._agent_twist_ref_current_w[:, 0:2]=self._dp_norm*self._dp_versor/self._env_opts["max_dt"]
        else:
            self._p_delta_w[env_indxs, :]=self._robot_state.root_state.get(data_type="p",gpu=self._use_gpu)[env_indxs, 0:2] -\
                self._p_trgt_w[env_indxs, :]
            
            self._dp_norm[env_indxs, :]=self._p_delta_w[env_indxs, :].norm(dim=1,keepdim=True)+1e-6
            self._dp_versor[env_indxs, :]=self._p_delta_w[env_indxs, :]/self._dp_norm[env_indxs, :]

            self._agent_twist_ref_current_w[env_indxs, 0:2]=self._dp_norm[env_indxs, :]*self._dp_versor[env_indxs, :]/self._env_opts["max_dt"]

    def _override_refs(self,
            env_indxs: torch.Tensor = None):
        
        # runs at every post_step
        self._agent_refs.rob_refs.root_state.synch_all(read=True,retry=True) # first read from mem
        if self._use_gpu:
            # copies latest refs to GPU 
            self._agent_refs.rob_refs.root_state.synch_mirror(from_gpu=False,non_blocking=False) 

        agent_p_ref_current=self._agent_refs.rob_refs.root_state.get(data_type="p",
                gpu=self._use_gpu)
        
        # self._p_trgt_w[:, :]=self._robot_state.root_state.get(data_type="p",gpu=self._use_gpu)[:, 0:2] + \
        #     agent_p_ref_current[:, 0:2]
        self._p_trgt_w[:, :]=agent_p_ref_current[:, 0:2]
    
    def _debug_agent_refs(self):
        if self._use_gpu:
            self._agent_refs.rob_refs.root_state.synch_mirror(from_gpu=True,non_blocking=False)
        self._agent_refs.rob_refs.root_state.synch_all(read=False, retry = True)

    def _randomize_task_refs(self,
        env_indxs: torch.Tensor = None):

        # we randomize the reference in world frame
        if env_indxs is None:
            self._trgt_d.uniform_(self._env_opts["min_distance"], self._env_opts["max_distance"])
            self._trgt_theta.uniform_(0.0, 2*torch.pi)

            self._p_trgt_w[:, :]=self._robot_state.root_state.get(data_type="p",gpu=self._use_gpu)[:, 0:2] +\
                torch.cat((self._trgt_d*torch.cos(self._trgt_theta)
                           ,self._trgt_d*torch.sin(self._trgt_theta)), dim=1)
                           
        else:
            if env_indxs.any():
                integer_idxs=torch.nonzero(env_indxs).flatten()
                
                trgt_d_selected=self._trgt_d[integer_idxs, :]
                trgt_d_selected.uniform_(self._env_opts["min_distance"], self._env_opts["max_distance"])
                self._trgt_d[integer_idxs, :]=trgt_d_selected

                trgt_theta_selected=self._trgt_theta[integer_idxs, :]
                trgt_theta_selected.uniform_(0.0, 2*torch.pi)
                self._trgt_theta[integer_idxs, :]=trgt_theta_selected

                self._p_trgt_w[integer_idxs, 0:1]=self._robot_state.root_state.get(data_type="p",gpu=self._use_gpu)[integer_idxs, 0:1] +\
                    self._trgt_d[integer_idxs, :]*torch.cos(self._trgt_theta[integer_idxs, :])
                self._p_trgt_w[integer_idxs, 1:2]=self._robot_state.root_state.get(data_type="p",gpu=self._use_gpu)[integer_idxs, 1:2] +\
                    self._trgt_d[integer_idxs, :]*torch.sin(self._trgt_theta[integer_idxs, :])
        
        self._compute_twist_ref_w(env_indxs=env_indxs)

