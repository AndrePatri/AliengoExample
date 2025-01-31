import torch 
import torch.nn as nn
from torch.distributions.normal import Normal

from lrhc_control.utils.nn.normalization_utils import RunningNormalizer 
from lrhc_control.utils.nn.layer_utils import llayer_init 

from typing import List

from EigenIPC.PyEigenIPC import LogType
from EigenIPC.PyEigenIPC import Journal
from EigenIPC.PyEigenIPC import VLevel

class SACAgent(nn.Module):
    def __init__(self,
            obs_dim: int, 
            actions_dim: int,
            obs_ub: List[float] = None,
            obs_lb: List[float] = None,
            actions_ub: List[float] = None,
            actions_lb: List[float] = None,
            rescale_obs: bool = False,
            norm_obs: bool = True,
            device:str="cuda",
            dtype=torch.float32,
            is_eval:bool=False,
            load_qf:bool=False,
            epsilon:float=1e-8,
            debug:bool=False,
            compression_ratio: float = - 1.0, # > 0; if [0, 1] compression, >1 "expansion"
            layer_width_actor:int=256,
            n_hidden_layers_actor:int=2,
            layer_width_critic:int=512,
            n_hidden_layers_critic:int=4,
            torch_compile: bool = False,
            add_weight_norm: bool = False):

        super().__init__()

        self._use_torch_compile=torch_compile

        self._layer_width_actor=layer_width_actor
        self._layer_width_critic=layer_width_critic
        self._n_hidden_layers_actor=n_hidden_layers_actor
        self._n_hidden_layers_critic=n_hidden_layers_critic

        if compression_ratio > 0.0:
            self._layer_width_actor=int(compression_ratio*obs_dim)
            self._layer_width_critic=int(compression_ratio*(obs_dim+actions_dim))
        
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

        self.actor = None
        self.qf1 = None
        self.qf1_target = None
        self.qf2 = None
        self.qf2_target = None

        self._torch_device = device
        self._torch_dtype = dtype

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
        
        self.actor = Actor(obs_dim=obs_dim,
                    actions_dim=actions_dim,
                    actions_ub=actions_ub,
                    actions_lb=actions_lb,
                    device=device,
                    dtype=dtype,
                    layer_width=self._layer_width_actor,
                    n_hidden_layers=self._n_hidden_layers_actor,
                    add_weight_norm=add_weight_norm
                    )

        if (not is_eval) or load_qf: # just needed for training or during eval
            # for debug, if enabled
            self.qf1 = CriticQ(obs_dim=obs_dim,
                    actions_dim=actions_dim,
                    device=device,
                    dtype=dtype,
                    layer_width=self._layer_width_critic,
                    n_hidden_layers=self._n_hidden_layers_critic,
                    add_weight_norm=add_weight_norm)
            self.qf1_target = CriticQ(obs_dim=obs_dim,
                    actions_dim=actions_dim,
                    device=device,
                    dtype=dtype,
                    layer_width=self._layer_width_critic,
                    n_hidden_layers=self._n_hidden_layers_critic,
                    add_weight_norm=add_weight_norm)
            
            self.qf2 = CriticQ(obs_dim=obs_dim,
                    actions_dim=actions_dim,
                    device=device,
                    dtype=dtype,
                    layer_width=self._layer_width_critic,
                    n_hidden_layers=self._n_hidden_layers_critic,
                    add_weight_norm=add_weight_norm)
            self.qf2_target = CriticQ(obs_dim=obs_dim,
                    actions_dim=actions_dim,
                    device=device,
                    dtype=dtype,
                    layer_width=self._layer_width_critic,
                    n_hidden_layers=self._n_hidden_layers_critic,
                    add_weight_norm=add_weight_norm)
        
            self.qf1_target.load_state_dict(self.qf1.state_dict())
            self.qf2_target.load_state_dict(self.qf2.state_dict())

        self.running_norm = None
        if self._normalize_obs:
            self.running_norm = RunningNormalizer((obs_dim,), epsilon=epsilon, 
                                    device=device, dtype=dtype, 
                                    freeze_stats=True, # always start with freezed stats
                                    debug=self._debug)
            self.running_norm.type(dtype) # ensuring correct dtype for whole module

        if self._use_torch_compile:
            self.actor = torch.compile(self.actor)
            self.qf1 = torch.compile(self.qf1)
            self.qf2 = torch.compile(self.qf2)
            self.qf1_target = torch.compile(self.qf1_target)
            self.qf2_target = torch.compile(self.qf2_target)
            self.running_norm=torch.compile(self.running_norm)

        msg=f"Created SAC agent with actor [{self._layer_width_actor}, {self._n_hidden_layers_actor}]\
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
    
    def _preprocess_obs(self, x):
        if self._rescale_obs:
            x = self._obs_scaling_layer(x)
        if self.running_norm is not None:
            x = self.running_norm(x)
        return x

    def get_action(self, x):
        x = self._preprocess_obs(x)
        return self.actor.get_action(x)
    
    def get_qf1_val(self, x, a):
        x = self._preprocess_obs(x)
        return self.qf1(x, a)

    def get_qf2_val(self, x, a):
        x = self._preprocess_obs(x)
        return self.qf2(x, a)
    
    def get_qf1t_val(self, x, a):
        x = self._preprocess_obs(x)
        return self.qf1_target(x, a)
    
    def get_qf2t_val(self, x, a):
        x = self._preprocess_obs(x)
        return self.qf2_target(x, a)

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

class CriticQ(nn.Module):
    def __init__(self,
        obs_dim: int, 
        actions_dim: int,
        device: str = "cuda",
        dtype = torch.float32,
        layer_width: int = 512,
        n_hidden_layers: int = 4,
        add_weight_norm: bool = False):

        super().__init__()

        self._lrelu_slope=0.01

        self._torch_device = device
        self._torch_dtype = dtype

        self._obs_dim = obs_dim
        self._actions_dim = actions_dim
        self._q_net_dim = self._obs_dim + self._actions_dim

        self._first_hidden_layer_width=self._q_net_dim # fist layer fully connected and of same dim
        # as input

        # Input layer
        layers = [llayer_init(
            layer=nn.Linear(self._q_net_dim, self._first_hidden_layer_width),
            init_type="kaiming_uniform",
            nonlinearity="leaky_relu",
            a_leaky_relu=self._lrelu_slope,
            device=self._torch_device,
            dtype=self._torch_dtype,
            add_weight_norm=add_weight_norm
        ), nn.LeakyReLU(negative_slope=self._lrelu_slope)]
        
        # Hidden layers
        layers.extend([
            llayer_init(
                layer=nn.Linear(self._first_hidden_layer_width, layer_width),
                init_type="kaiming_uniform",
                nonlinearity="leaky_relu",
                a_leaky_relu=self._lrelu_slope,
                device=self._torch_device,
                dtype=self._torch_dtype,
                add_weight_norm=add_weight_norm
            ),
            nn.LeakyReLU(negative_slope=self._lrelu_slope)
        ])

        for _ in range(n_hidden_layers - 2):
            layers.extend([
                llayer_init(
                    layer=nn.Linear(layer_width, layer_width),
                    init_type="kaiming_uniform",
                    nonlinearity="leaky_relu",
                    a_leaky_relu=self._lrelu_slope,
                    device=self._torch_device,
                    dtype=self._torch_dtype,
                    add_weight_norm=add_weight_norm
                ),
                nn.LeakyReLU(negative_slope=self._lrelu_slope)
            ])

        # Output layer
        layers.append(
            llayer_init(
                layer=nn.Linear(layer_width, 1),
                init_type="uniform",
                device=self._torch_device,
                dtype=self._torch_dtype,
                add_weight_norm=add_weight_norm
            )
        )

        # Creating the full sequential network
        self._q_net = nn.Sequential(*layers)
        self._q_net.to(self._torch_device).type(self._torch_dtype)

    def get_n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        return self._q_net(x)

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
        add_weight_norm: bool = False):
        
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
        
        # Network configuration
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5

        # Input layer followed by hidden layers
        layers = [llayer_init(nn.Linear(self._obs_dim, self._first_hidden_layer_width), 
                    init_type="kaiming_uniform",
                    nonlinearity="leaky_relu",
                    a_leaky_relu=self._lrelu_slope,
                    device=self._torch_device, 
                    dtype=self._torch_dtype,
                    add_weight_norm=add_weight_norm),
            nn.LeakyReLU(negative_slope=self._lrelu_slope)]
    
        # Hidden layers
        # first hidden optionally uses _first_hidden_layer_width
        layers.extend([
            llayer_init(nn.Linear(self._first_hidden_layer_width, layer_width), 
                init_type="kaiming_uniform",
                nonlinearity="leaky_relu",
                a_leaky_relu=self._lrelu_slope,
                device=self._torch_device,
                dtype=self._torch_dtype,
                add_weight_norm=add_weight_norm),
            nn.LeakyReLU(negative_slope=self._lrelu_slope)
        ])
        
        for _ in range(n_hidden_layers - 2):
            layers.extend([
                llayer_init(nn.Linear(layer_width, layer_width), 
                    init_type="kaiming_uniform",
                    nonlinearity="leaky_relu",
                    a_leaky_relu=self._lrelu_slope,
                    device=self._torch_device,
                    dtype=self._torch_dtype,
                    add_weight_norm=add_weight_norm),
                nn.LeakyReLU(negative_slope=self._lrelu_slope)
            ])
        
        # Sequential layers for the feature extractor
        self._fc12 = nn.Sequential(*layers)

        # Mean and log_std layers
        self.fc_mean = llayer_init(nn.Linear(layer_width, self._actions_dim), 
                        init_type="uniform",
                        device=self._torch_device, 
                        dtype=self._torch_dtype,
                        add_weight_norm=add_weight_norm)
        self.fc_logstd = llayer_init(nn.Linear(layer_width, self._actions_dim), 
                        init_type="uniform",
                        device=self._torch_device, 
                        dtype=self._torch_dtype,
                        bias_const=-1.0, # for encouraging exploration
                        add_weight_norm=add_weight_norm
                        )

        # Move all components to the specified device and dtype
        self._fc12.to(device=self._torch_device, dtype=self._torch_dtype)
        self.fc_mean.to(device=self._torch_device, dtype=self._torch_dtype)
        self.fc_logstd.to(device=self._torch_device, dtype=self._torch_dtype)

    def get_n_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, x):
        x = self._fc12(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob = log_prob - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

if __name__ == "__main__":  
    
    device = "cuda"
    dummy_obs = torch.full(size=(2, 5),dtype=torch.float32,device=device,fill_value=0) 

    sofqn = CriticQ(obs_dim=5,actions_dim=3,
            device=device,
            dtype=torch.float32)
    
    print("Db prints Q")
    print(f"N. params: {sofqn.get_n_params()}")
    
    dummy_a = torch.full(size=(2, 3),dtype=torch.float32,device=device,fill_value=0)
    q_v = sofqn.forward(x=dummy_obs,a=dummy_a)
    print(q_v)

    actor = Actor(obs_dim=5,actions_dim=3,
            actions_lb=[-1.0, -1.0, -1.0],actions_ub=[1.0, 1.0, 1.0],
            device=device,
            dtype=torch.float32)
    
    # Print the biases of the log_std layer
    print("Log_std Biases:")
    print(actor.fc_logstd.bias)  # Access the biases of the log_std layer

    print("Db prints Actor")
    print(f"N. params: {actor.get_n_params()}")
    output=actor.forward(x=dummy_obs)
    print(output)
    print(actor.get_action(x=dummy_obs))