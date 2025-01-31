
import torch 
import torch.nn as nn
from torch.distributions.normal import Normal

from lrhc_control.utils.nn.normalization_utils import RunningNormalizer 
from lrhc_control.utils.nn.layer_utils import llayer_init 

from typing import List

from EigenIPC.PyEigenIPC import LogType
from EigenIPC.PyEigenIPC import Journal
from EigenIPC.PyEigenIPC import VLevel

class ACAgent(nn.Module):

    def __init__(self,
            obs_dim: int, 
            actions_dim: int,
            obs_ub: List[float] = None,
            obs_lb: List[float] = None,
            actions_ub: List[float] = None,
            actions_lb: List[float] = None,
            actor_std: float = 0.01, 
            critic_std: float = 1.0,
            rescale_obs: bool = False,
            norm_obs: bool = True,
            device:str="cuda",
            dtype=torch.float32,
            is_eval:bool=False,
            epsilon:float=1e-8,
            debug:bool=False,
            compression_ratio: float = - 1.0, # > 0; if [0, 1] compression, >1 "expansion"
            layer_width_actor:int=256,
            n_hidden_layers_actor:int=2,
            layer_width_critic:int=512,
            n_hidden_layers_critic:int=4,
            torch_compile: bool = False,
            add_weight_norm: bool = False,
            out_std_critic: float = 1.0,
            out_std_actor: float = 0.01):

        super().__init__()

        self._use_torch_compile=torch_compile

        self._layer_width_actor=layer_width_actor
        self._layer_width_critic=layer_width_critic
        self._n_hidden_layers_actor=n_hidden_layers_actor
        self._n_hidden_layers_critic=n_hidden_layers_critic

        if compression_ratio > 0.0:
            self._layer_width_actor=int(compression_ratio*obs_dim)
            self._layer_width_critic=int(compression_ratio*(obs_dim))
        
        if add_weight_norm:
            Journal.log(self.__class__.__name__,
                "__init__",
                f"Will use weight normalization reparametrization\n",
                LogType.INFO)
        
        self._normalize_obs = norm_obs
        self._rescale_obs=rescale_obs
        if self._rescale_obs and self._normalize_obs:
            Journal.log(self.__class__.__name__,
                "__init__",
                f"Both running normalization and obs rescaling is enabled. Was this intentional?",
                LogType.WARN,
                throw_when_excep = True)
        
        self._rescaling_epsi=1e-9

        self._debug = debug
        
        self._actor_std = actor_std
        self._critic_std = critic_std
        
        self._is_eval = is_eval

        self._torch_device = device
        self._torch_dtype = dtype
            
        self._obs_dim = obs_dim
        self._actions_dim = actions_dim
        
        # obs scale and bias
        if obs_ub is None:
            obs_ub = [1] * obs_dim
        if obs_lb is None:
            obs_lb = [-1] * obs_dim
        if (len(obs_ub) != obs_dim):
            Journal.log(self.__class__.__name__,
                "__init__",
                f"Observations ub list length should be equal to {obs_dim}, but got {len(obs_ub)}",
                LogType.EXCEP,
                throw_when_excep = True)
        if (len(obs_lb) != obs_dim):
            Journal.log(self.__class__.__name__,
                "__init__",
                f"Observations lb list length should be equal to {obs_dim}, but got {len(obs_lb)}",
                LogType.EXCEP,
                throw_when_excep = True)

        self._obs_ub = torch.tensor(obs_ub, dtype=self._torch_dtype, 
                                device=self._torch_device)
        self._obs_lb = torch.tensor(obs_lb, dtype=self._torch_dtype,
                                device=self._torch_device)
        obs_scale = torch.full((obs_dim, ),
                            fill_value=0.0,
                            dtype=self._torch_dtype,
                            device=self._torch_device)
        obs_scale[:] = (self._obs_ub-self._obs_lb)/2.0
        self.register_buffer(
            "obs_scale", obs_scale
        )
        obs_bias = torch.full((obs_dim, ),
                            fill_value=0.0,
                            dtype=self._torch_dtype,
                            device=self._torch_device)
        obs_bias[:] = (self._obs_ub+self._obs_lb)/2.0
        self.register_buffer(
            "obs_bias", obs_bias)
    
        if (not is_eval):
            self.critic=CriticV(obs_dim=obs_dim,
                    device=self._torch_device,
                    dtype=self._torch_dtype,
                    layer_width=self._layer_width_critic,
                    n_hidden_layers=self._n_hidden_layers_critic,
                    out_std=out_std_critic,
                    add_weight_norm=add_weight_norm)

        self.actor = Actor(obs_dim=obs_dim,
                            actions_dim=actions_dim,
                            actions_lb=actions_lb,
                            actions_ub=actions_ub,
                            device=self._torch_device,
                            dtype=self._torch_dtype,
                            layer_width=self._layer_width_actor,
                            n_hidden_layers=self._n_hidden_layers_actor,
                            add_weight_norm=add_weight_norm,
                            out_std=out_std_actor)

        self.running_norm = None
        if self._normalize_obs:
            self.running_norm = RunningNormalizer((obs_dim,), epsilon=epsilon, 
                                    device=device, dtype=dtype, 
                                    freeze_stats=True, # always start with freezed stats
                                    debug=self._debug)
            self.running_norm.type(dtype) # ensuring correct dtype for whole module
        
        if self._use_torch_compile:
            self.critic = torch.compile(self.critic)
            self.actor = torch.compile(self.actor)

        msg=f"Created PPO agent with actor [{self._layer_width_actor}, {self._n_hidden_layers_actor}]\
        and critic [{self._layer_width_critic}, {self._n_hidden_layers_critic}] sizes.\n Running normalizer: {type(self.running_norm)}"
        Journal.log(self.__class__.__name__,
            "__init__",
            msg,
            LogType.INFO)
    
    def layer_width_actor(self):
        return self._layer_width_actor

    def n_hidden_layers_actor(self):
        return self._n_hidden_layers_actor

    def layer_width_critic(self):
        return self._layer_width_critic

    def n_hidden_layers_critic(self):
        return self._n_hidden_layers_critic

    def get_impl_path(self):
        import os 
        return os.path.abspath(__file__)
    
    def update_obs_bnorm(self, x):
        self.running_norm.unfreeze()
        self.running_norm.manual_stat_update(x)
        self.running_norm.freeze()

    def _obs_scaling_layer(self, x):
        x=(x-self.obs_bias)
        x=x/(self.obs_scale+self._rescaling_epsi)
        return x
    
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
        
        # sanity check on running normalizer
        import re
        running_norm_pattern = r"running_norm"
        error=f"Found some keys in model state dictionary associated with a running normalizer. Are you running the agent with norm_obs=True?\n"
        if any(re.match(running_norm_pattern, key) for key in unexpected):
            Journal.log(self.__class__.__name__,
                "load_state_dict",
                error,
                LogType.EXCEP,
                throw_when_excep=True)
            
    def _preprocess_obs(self, x):
        if self._rescale_obs:
            x = self._obs_scaling_layer(x)
        if self.running_norm is not None:
            x = self.running_norm(x)
        return x
    
    def get_value(self, x):
        x = self._preprocess_obs(x)
        return self.critic(x)
    
    def get_action(self, x, 
                only_mean: bool = False # useful during evaluation
                ):
        x = self._preprocess_obs(x)
        return self.actor.get_action(x, only_mean=only_mean)

class CriticV(nn.Module):
    def __init__(self,
        obs_dim: int, 
        device: str = "cuda",
        dtype = torch.float32,
        layer_width: int = 512,
        n_hidden_layers: int = 4,
        out_std: float = 1.0,
        add_weight_norm: bool = False):

        super().__init__()

        self._lrelu_slope=0.01

        self._torch_device = device
        self._torch_dtype = dtype

        self._obs_dim = obs_dim
        self._v_net_dim = self._obs_dim

        self._first_hidden_layer_width=self._v_net_dim # fist layer fully connected and of same dim
        # as input

        # Input layer
        layers = [llayer_init(
            layer=nn.Linear(self._v_net_dim, self._first_hidden_layer_width),
            init_type="orthogonal",
            a_leaky_relu=self._lrelu_slope,
            device=self._torch_device,
            dtype=self._torch_dtype,
            add_weight_norm=add_weight_norm
        ), nn.Tanh()
        ]
        
        # Hidden layers
        layers.extend([
            llayer_init(
                layer=nn.Linear(self._first_hidden_layer_width, layer_width),
                init_type="orthogonal",
                device=self._torch_device,
                dtype=self._torch_dtype,
                add_weight_norm=add_weight_norm
            ),
            nn.Tanh()
        ])

        for _ in range(n_hidden_layers - 2):
            layers.extend([
                llayer_init(
                    layer=nn.Linear(layer_width, layer_width),
                    init_type="orthogonal",
                    a_leaky_relu=self._lrelu_slope,
                    device=self._torch_device,
                    dtype=self._torch_dtype,
                    add_weight_norm=add_weight_norm
                ),
                nn.Tanh()
            ])

        # Output layer
        layers.append(
            llayer_init(
                layer=nn.Linear(layer_width, 1),
                init_type="orthogonal",
                orth_init_gain=out_std,
                device=self._torch_device,
                dtype=self._torch_dtype,
                add_weight_norm=add_weight_norm
            )
        )

        # Creating the full sequential network
        self._v_net = nn.Sequential(*layers)
        self._v_net.to(self._torch_device).type(self._torch_dtype)

    def get_n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        return self._v_net(x)

class Actor(nn.Module):
        def __init__(self,
            obs_dim: int, 
            actions_dim: int,
            actions_ub: List[float] = None,
            actions_lb: List[float] = None,
            device: str = "cuda",
            dtype = torch.float32,
            layer_width: int = 256,
            n_hidden_layers: int = 2,
            add_weight_norm: bool = False,
            out_std: float = 0.01):
            
            super().__init__()

            self._lrelu_slope=0.01
            
            self._torch_device = device
            self._torch_dtype = dtype

            self._obs_dim = obs_dim
            self._actions_dim = actions_dim
            
            self._first_hidden_layer_width=self._obs_dim # fist layer fully connected and of same dim
        
            # Action scale and bias
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
                "action_bias", actions_bias)
            

            # Input layer followed by hidden layers
            layers = [llayer_init(nn.Linear(self._obs_dim, self._first_hidden_layer_width), 
                        init_type="orthogonal",
                        device=self._torch_device, 
                        dtype=self._torch_dtype,
                        add_weight_norm=add_weight_norm),
                nn.Tanh()
                ]
        
            # Hidden layers
            # first hidden optionally uses _first_hidden_layer_width
            layers.extend([
                llayer_init(nn.Linear(self._first_hidden_layer_width, layer_width), 
                    init_type="orthogonal",
                    device=self._torch_device,
                    dtype=self._torch_dtype,
                    add_weight_norm=add_weight_norm),
                nn.Tanh()
            ])
            
            for _ in range(n_hidden_layers - 2):
                layers.extend([
                    llayer_init(nn.Linear(layer_width, layer_width), 
                        init_type="orthogonal",
                        device=self._torch_device,
                        dtype=self._torch_dtype,
                        add_weight_norm=add_weight_norm),
                    nn.Tanh()
                ])
            
            # Output layer
            layers.append(
                llayer_init(
                    layer=nn.Linear(layer_width,self._actions_dim),
                    init_type="orthogonal",
                    device=self._torch_device,
                    dtype=self._torch_dtype,
                    add_weight_norm=add_weight_norm,
                    orth_init_gain=out_std
                )
            )

            # Sequential layers for the feature extractor
            self.actor_mean = nn.Sequential(*layers)
            self.actor_logstd = nn.Parameter(torch.zeros(1, self._actions_dim, 
                                                device=self._torch_device,
                                                dtype=self._torch_dtype))
     
        def get_n_params(self):
            return sum(p.numel() for p in self.parameters())
        
        def forward(self, x):
        
            return self.actor_mean(x)

        def get_action(self, x, only_mean: bool = False):
            
            action_mean = self(x)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            if not only_mean:
                action = probs.sample()
            else:
                action = action_mean
            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)
        