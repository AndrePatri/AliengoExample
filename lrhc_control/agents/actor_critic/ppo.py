
import torch.nn as nn
import torch
from torch.distributions.normal import Normal

from lrhc_control.utils.nn.normalization_utils import RunningNormalizer 

class ACAgent(nn.Module):

    def __init__(self,
            obs_dim: int, 
            actions_dim: int,
            actor_std: float = 0.01, 
            critic_std: float = 1.0,
            norm_obs: bool = True,
            device:str="cuda",
            dtype=torch.float32,
            is_eval:bool=False,
            debug:bool=False):

        self._debug = debug

        self._actor_std = actor_std
        self._critic_std = critic_std
        
        self._normalize_obs = norm_obs
        self._is_eval = is_eval

        self._torch_device = device
        self._torch_dtype = dtype

        super().__init__()
            
        self._obs_dim = obs_dim
        self._actions_dim = actions_dim
        
        size_critic = 128
        size_actor = 128
        
        self.running_norm = None
        if self._normalize_obs:
            self.running_norm = RunningNormalizer((self._obs_dim,), epsilon=1e-8, 
                                device=self._torch_device, dtype=self._torch_dtype, 
                                freeze_stats=self._is_eval,
                                debug=self._debug)

        self.critic = nn.Sequential(
            self._layer_init(layer=nn.Linear(self._obs_dim, size_critic), device=self._torch_device,dtype=self._torch_dtype),
            nn.Tanh(),
            self._layer_init(layer=nn.Linear(size_critic, size_critic), device=self._torch_device,dtype=self._torch_dtype),
            nn.Tanh(),
            self._layer_init(layer=nn.Linear(size_critic, 1), std=self._critic_std, device=self._torch_device,dtype=self._torch_dtype),
        ) # (stochastic critic)
        self.critic.type(self._torch_dtype) # ensuring correct dtype

        self.actor_mean = nn.Sequential(
            self._layer_init(layer=nn.Linear(self._obs_dim, size_actor), device=self._torch_device,dtype=self._torch_dtype),
            nn.Tanh(),
            self._layer_init(layer=nn.Linear(size_actor, size_actor), device=self._torch_device,dtype=self._torch_dtype),
            nn.Tanh(),
            self._layer_init(layer=nn.Linear(size_actor, self._actions_dim), std=self._actor_std, device=self._torch_device,dtype=self._torch_dtype),
        ) # (stochastic actor)
        self.actor_mean.type(self._torch_dtype) # ensuring correct dtype

        self.actor_logstd = nn.Parameter(torch.zeros(1, self._actions_dim, device=self._torch_device,dtype=self._torch_dtype))
    
    def get_critic_n_params(self):
        return sum(p.numel() for p in self.critic.parameters())

    def get_actor_n_params(self):
        actor_mean_size = sum(p.numel() for p in self.actor_mean.parameters())
        actor_std_size = self._actions_dim
        return actor_mean_size+actor_std_size
    
    def get_n_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_impl_path(self):
        import os 
        return os.path.abspath(__file__)

    def get_value(self, x):
        if self.running_norm is not None:
            was_training = self.running_norm.training
            self.running_norm.train(True) # freeze stats (stat update is done by actor only)
            x = self.running_norm(x)
            self.running_norm.train(was_training) # restore previous flag state
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        if self.running_norm is not None:        
            x = self.running_norm(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    
    def get_action(self, x, 
                only_mean: bool = False # useful during evaluation
                ):
        if self.running_norm is not None:
            x = self.running_norm(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if not only_mean:
            action = probs.sample()
        else:
            action = action_mean
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)

    def _layer_init(self, layer, std=torch.sqrt(torch.tensor(2)), bias_const=0.0,
            device: str = "cuda",
            dtype = torch.float32):
        # device
        layer.to(device)
        # dtype
        layer.weight.data.type(dtype)
        layer.bias.data.type(dtype)
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
