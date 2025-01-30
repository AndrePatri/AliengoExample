import torch
import torch.nn as nn
import torch.nn.functional as F
from lrhc_control.utils.nn.normalization_utils import RunningNormalizer 

class RNDFull(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, 
            layer_width: int = 128, n_hidden_layers: int = 2, 
            device: str = "cpu",
            dtype=torch.float32,
            normalize: bool = False,
            debug:bool=False):
        
        super().__init__()
        
        self.rnd_trgt_net = RNDNetwork(input_dim=input_dim, output_dim=output_dim,
                layer_width=layer_width, n_hidden_layers=n_hidden_layers,
                target=True,
                device=device,
                dtype=dtype)
        self.rnd_predictor_net = RNDNetwork(input_dim=input_dim, output_dim=output_dim,
                layer_width=layer_width, n_hidden_layers=n_hidden_layers,
                target=False,
                device=device,
                dtype=dtype)
        
        self._normalize=normalize
        self._input_dim=input_dim

        self.running_norm = None
        if self._normalize:
            self.running_norm = RunningNormalizer((input_dim,), epsilon=1e-8, 
                                    device=device, dtype=dtype, 
                                    freeze_stats=True, # always start with freezed stats
                                    debug=debug)
            self.running_norm.type(dtype) # ensuring correct dtype for whole module
    
    def update_input_bnorm(self, x):
        self.running_norm.unfreeze()
        self.running_norm.manual_stat_update(x)
        self.running_norm.freeze()
    
    def input_dim(self):
        return self._input_dim
    
    def get_raw_bonus(self, input):
        if self.running_norm is not None:
            input = self.running_norm(input)
        return torch.mean(torch.square(self.rnd_predictor_net(input)-self.rnd_trgt_net(input)), 
                                            dim=1, 
                                            keepdim=True)

class RNDNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, 
                layer_width: int = 128, n_hidden_layers: int = 2, target: bool = False,
                device: str = "cpu",
                dtype=torch.float32):
        """
        Random Network Distillation (RND) model.

        :param input_dim: Dimension of the input (obs + action)
        :param output_dim: Dimension of the output feature space
        :param layer_width: Number of neurons per hidden layer
        :param n_hidden_layers: Number of hidden layers
        :param target: If True, this network acts as the fixed target network (no gradient updates)
        """
        self._torch_device = device
        self._torch_dtype = dtype

        super().__init__()
        
        layers = [nn.Linear(input_dim, layer_width, dtype=self._torch_dtype), nn.ReLU()]
        
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(layer_width, layer_width, dtype=self._torch_dtype))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(layer_width, output_dim, dtype=self._torch_dtype))

        self.network = nn.Sequential(*layers).to(self._torch_device, self._torch_dtype)

        if target:
            for param in self.parameters():
                param.requires_grad = False  # Freeze target network

    def forward(self, x):
        return self.network(x)
