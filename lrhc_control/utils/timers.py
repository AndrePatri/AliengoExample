
import torch
from lrhc_control.utils.shared_data.training_env import SimpleCounters

class PeriodicTimer():

    def __init__(self, 
        counter: SimpleCounters,
        period: int,
        dtype: torch.dtype=torch.float32,
        device: str = "cuda"
        ):

        self._dtype=dtype
        self._torch_device=device
        
        self._counter=counter
        self._period=period
        
        self._n_envs=self._counter.n_envs()

        self._cos_sin=torch.zeros((self._n_envs, 2), dtype=self._dtype,
            device=self._torch_device)
        
        self._time_ref=self._counter.get().clone()
        self._time_ref.zero_()
        
    def get(self, clone: bool = False):

        time_now=self._counter.get()
        phase = (time_now - self._time_ref) % self._period
        self._cos_sin[:, 0:1] = torch.cos(2 * torch.pi * phase / self._period)
        self._cos_sin[:, 1:2] = torch.sin(2 * torch.pi * phase / self._period)

        if clone:
            self._cos_sin.clone()
        else:
            return self._cos_sin
    
    def reset(self, to_be_reset: torch.Tensor = None):

        self._counter.get()
