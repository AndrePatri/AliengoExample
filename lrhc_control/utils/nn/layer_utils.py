import torch
import math

def llayer_init(layer, 
    init_type=None,
    nonlinearity="leaky_relu",
    a_leaky_relu: float=0.01,
    bias_const=0.0,
    device: str = "cuda",
    dtype = torch.float32):

        # Move to device and set dtype
        layer.to(device).type(dtype)

        # Apply weight initialization based on the init_type argument
        if init_type is not None:
            if init_type == "orthogonal":
                torch.nn.init.orthogonal_(layer.weight, gain=1.0)
                torch.nn.init.constant_(layer.bias, bias_const)
            if init_type == "uniform":
                k=1/layer.in_features
                bound=math.sqrt(k)
                torch.nn.init.uniform_(layer.weight,a=-bound,b=bound)
                torch.nn.init.uniform_(layer.bias,a=-bound,b=bound)
            elif init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity=nonlinearity, a=a_leaky_relu,mode='fan_in')
                torch.nn.init.constant_(layer.bias, bias_const)
            elif init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity, a=a_leaky_relu, mode='fan_in')
                torch.nn.init.constant_(layer.bias, bias_const)
            else:
                raise ValueError(f"Unsupported init_type: {init_type}")

        return layer