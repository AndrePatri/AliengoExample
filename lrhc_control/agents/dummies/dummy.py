import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal

from lrhc_control.utils.nn.normalization_utils import RunningNormalizer 

from typing import List

from EigenIPC.PyEigenIPC import LogType
from EigenIPC.PyEigenIPC import Journal
from EigenIPC.PyEigenIPC import VLevel

class DummyAgent(nn.Module):
    def __init__(self,
            obs_dim: int, 
            actions_dim: int,
            actions_ub: List[float] = None,
            actions_lb: List[float] = None,
            device:str="cuda",
            dtype=torch.float32,
            debug:bool=False):

        super().__init__()

        # self._normalize_obs = norm_obs

        self._debug = debug

        self.actor = DummyActor(obs_dim=obs_dim,
            actions_dim=actions_dim,
            actions_ub=actions_ub,
            actions_lb=actions_lb,
            device=device,
            dtype=dtype,
            layer_size=None
            )
        self.critic = None

        self._torch_device = device
        self._torch_dtype = dtype

        self.obs_running_norm = None
        # if self._normalize_obs:
        #     self.obs_running_norm = RunningNormalizer((obs_dim,), epsilon=epsilon, 
        #                             device=device, dtype=dtype, 
        #                             freeze_stats=is_eval,
        #                             debug=self._debug)
        #     self.obs_running_norm.type(dtype) # ensuring correct dtype for whole module

    def layer_width_actor(self):
        return -1

    def n_hidden_layers_actor(self):
        return -1

    def layer_width_critic(self):
        return -1

    def n_hidden_layers_critic(self):
        return -1
    
    def get_impl_path(self):
        import os 
        return os.path.abspath(__file__)
    
    def get_action(self, x):
        return self.actor.get_action(x)
    
    def get_val(self, x, a):
        return None

    def load_state_dict(self, param_dict):

        missing, unexpected = super().load_state_dict(param_dict,
            strict=False)
        if not len(missing)==0:
            Journal.log(self.__class__.__name__,
                "load_state_dict",
                f"These parameters are missing from the provided state dictionary: {str(missing)}\n",
                LogType.EXCEP,
                throw_when_excep = True)
        if not len(unexpected)==0:
            Journal.log(self.__class__.__name__,
                "load_state_dict",
                f"These parameters present in the provided state dictionary are not needed: {str(unexpected)}\n",
                LogType.WARN)

class DummyActor(nn.Module):
    def __init__(self,
            obs_dim: int, 
            actions_dim: int,
            actions_ub: List[float] = None,
            actions_lb: List[float] = None,
            device:str="cuda",
            dtype=torch.float32,
            layer_size:int=None):
        super().__init__()

        self._torch_device = device
        self._torch_dtype = dtype

        self._obs_dim = obs_dim
        self._actions_dim = actions_dim
        
        # action scale and bias
        if actions_ub is None:
            actions_ub = [1] * actions_dim
        if actions_lb is None:
            actions_lb = [-1] * actions_dim
        if (len(actions_ub) != actions_dim):
            Journal.log(self.__class__.__name__,
                "__init__",
                f"Actions ub list length should be equal to {actions_dim}, but got {len(actions_ub)}",
                LogType.EXCEP,
                throw_when_excep = True)
        if (len(actions_lb) != actions_dim):
            Journal.log(self.__class__.__name__,
                "__init__",
                f"Actions lb list length should be equal to {actions_dim}, but got {len(actions_lb)}",
                LogType.EXCEP,
                throw_when_excep = True)
        self._actions_ub = torch.tensor(actions_ub, dtype=self._torch_dtype, 
                                device=self._torch_device)
        self._actions_lb = torch.tensor(actions_lb, dtype=self._torch_dtype,
                                device=self._torch_device)
        action_scale = torch.full((actions_dim, ),
                            fill_value=0.0,
                            dtype=self._torch_dtype,
                            device=self._torch_device)
        action_scale[:] = (self._actions_ub-self._actions_lb)/2.0
        self.register_buffer(
            "action_scale", action_scale
        )
        actions_bias = torch.full((actions_dim, ),
                            fill_value=0.0,
                            dtype=self._torch_dtype,
                            device=self._torch_device)
        actions_bias[:] = (self._actions_ub+self._actions_lb)/2.0
        self.register_buffer(
            "action_bias", actions_bias
        )

    def get_n_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_impl_path(self):
        import os 
        return os.path.abspath(__file__)
    
    def forward(self, x):
        
        n_envs = x.shape[0]
        random_uniform=torch.full((n_envs, self._actions_dim), fill_value=0.0, dtype=self._torch_dtype,device=self._torch_device)

        torch.nn.init.uniform_(random_uniform, a=-1, b=1)
        
        random_actions = random_uniform*self.action_scale+self.action_bias

        return random_actions
    
    def get_action(self, x):
        action = self(x)
        return action