from SharsorIPCpp.PySharsorIPC import Journal
from SharsorIPCpp.PySharsorIPC import LogType

import torch
import math

from typing import List

class ExponentialSignalSmoother():
    
    # class for performing an exponential signal smoothing on a 
    # vectorized signal s
    # s_tilde(t) = (1-alpha)*s(t)+alpha*s_thilde(t-1)
    # given an horizon of h samples, 
    # s_tilde(t) = (1-alpha)*sum{k=0}^{h}{alpha*s(t-k)}+alpha^{h}*s_tilde(t-h)
    # where usually s_tilde(t-h) = s(t-h)
    # This means that the oldest signal will have influence alpha^{h} on the current smoothed signal.
    # Given a target influence percentage 0<p_t<1, a given horizon length in [s] T and update interval for
    # the signal in [s] dt, the smoothing coefficient alpha can be chosen as 
    # alpha=e^{dt/T*ln(p_t)}
    # For example, if T=1s, dt=0.03 and p_t=0.2 (i.e. 20%) -> alpha=0.953
    def __init__(self,
        signal: torch.Tensor, 
        update_dt: float, # [s]
        smoothing_horizon: float, # [s]
        target_smoothing: float, # percentage 0<pt<1
        name: str = "SignalSmoother",
        debug: bool = False,
        dtype: torch.dtype = torch.float32,
        use_gpu: bool = False):

        self._n_envs=signal.shape[0]
        self._signal_dim=signal.shape[1]

        self._name=name
        self._debug=debug
        self._dtype=dtype
        self._use_gpu=use_gpu

        self._update_dt=update_dt
        self._T=smoothing_horizon
        self._pt=target_smoothing

        self._epsi=1e-6
        if self._pt<=0:
            self._pt=self._epsi

        self._alpha=self._compute_alpha(update_dt=self._update_dt,
            smoothing_horizon=self._T,
            target_smoothing=self._pt)# no smoothing by default

        self._init_data()

    def _compute_alpha(self,
        update_dt: float = None, 
        smoothing_horizon: float = None,
        target_smoothing: float = None):
        
        # default to current vals is not provided
        if update_dt is None:
            update_dt=self._update_dt
        if smoothing_horizon is None:
            smoothing_horizon=self._T
        if target_smoothing is None:
            target_smoothing=self._pt

        return math.exp((update_dt/smoothing_horizon) * math.log(target_smoothing))

    def _init_data(self):
        # current step counter (within this episode)
        device="cuda" if self._use_gpu else "cpu"
        self._steps_counter = torch.full(size=(self._n_envs, 1), 
            fill_value=0,
            dtype=torch.int32, 
            device=device,
            requires_grad=False)
        
        self._smoothed_sig=torch.full(size=(self._n_envs, self._signal_dim), 
            fill_value=0.0,
            dtype=self._dtype, 
            device=device,
            requires_grad=False)
    
    def _check_new_data(self,new_data):
        self._check_sizes(new_data=new_data)
        self._check_finite(new_data=new_data)

    def _check_sizes(self,new_data):
        if (not new_data.shape[0] == self._n_envs) or \
            (not new_data.shape[1] == self._signal_dim):
            exception = f"Provided signal tensor shape {new_data.shape[0]}, {new_data.shape[1]}" + \
                f" does not match {self._n_envs}, {self._signal_dim}!!"
            Journal.log(self.__class__.__name__ + f"[{self._name}]",
                "_check_sizes",
                exception,
                LogType.EXCEP,
                throw_when_excep=True)
    
    def _check_finite(self,new_data):
        if (not torch.isfinite(new_data).all().item()):
            print(new_data)
            exception = f"Found non finite elements in provided data!!"
            Journal.log(self.__class__.__name__ + f"[{self._name}]",
                "_check_finite",
                exception,
                LogType.EXCEP,
                throw_when_excep=True)

    def reset_all(self):
        self._steps_counter.zero_()
        self._smoothed_sig.zero_()

    def reset(self,
        to_be_reset: torch.Tensor):
        self._steps_counter[to_be_reset, :]=0
        self._smoothed_sig[to_be_reset, :]=0.0

    def update(self, 
        new_signal: torch.Tensor,
        ep_finished: torch.Tensor = None):
        
        if self._debug:
            self._check_new_data(new_data=new_signal)

        # initialize smoothed signal with current sample if just started episode
        is_first_step=(self._steps_counter.eq(0))
        if is_first_step.any():
            self._smoothed_sig[is_first_step.flatten(), :]=new_signal[is_first_step.flatten(),:]

        self._smoothed_sig[:, :] = (1-self._alpha)*new_signal+self._alpha*self._smoothed_sig[:, :]

        self._steps_counter+=1
        if ep_finished is not None:
            self._steps_counter[ep_finished.flatten(), :] = 0
            
    def update_alpha(self,
        update_dt: float = None, 
        smoothing_horizon: float = None,
        target_smoothing: float = None):

        if not torch.all(self._steps_counter.eq(0)):
            Journal.log(self.__class__.__name__ + f"[{self._name}]",
                "update_alpha",
                "Some environments have unfinished episodes. Have you called the reset_all() method??",
                LogType.WARN,
                throw_when_excep=False)
        self._alpha=self._compute_alpha(update_dt=update_dt,
            smoothing_horizon=smoothing_horizon,
            target_smoothing=target_smoothing)

    def get(self,
        clone: bool=False):
        if clone:
            return self._smoothed_sig.detach().clone()
        else:
            return self._smoothed_sig
    
    def counter(self,
        clone: bool=False):
        if clone:
            return self._steps_counter.detach().clone()
        else:
            return self._steps_counter
    
    def horizon(self):
        return self._T
    
    def dt(self):
        return self._update_dt
    
    def alpha(self):
        return self._alpha
    
def sinusoidal_signal(t: torch.Tensor, frequency: float) -> torch.Tensor:
    """
    Generate a sinusoidal signal with unit amplitude and given frequency.
    """
    return torch.sin(2 * torch.pi * frequency * t)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Set up test parameters
    n_envs = 2
    signal_dim = 1
    update_dt = 0.03  # [s]
    smoothing_horizon = 1.0  # [s]
    target_smoothing = 0.05  # 20%
    frequency = round(100/update_dt)  # Frequency of the sinusoidal signal in Hz
    total_time = 10.0  # Total time for simulation [s]
    
    # Calculate the number of steps
    episode_length = int(round(total_time / update_dt)) + 1
    t = torch.linspace(0, total_time, episode_length)

    # Generate sinusoidal signal
    nominal_signal = sinusoidal_signal(t, frequency)

    # Initialize ExponentialSignalSmoother
    smoother = ExponentialSignalSmoother(
        signal=torch.zeros(n_envs, signal_dim, dtype=torch.float32),
        update_dt=update_dt,
        smoothing_horizon=smoothing_horizon,
        target_smoothing=target_smoothing,
        name="TestSmoother",
        debug=True,
        dtype=torch.float32,
        use_gpu=False
    )
    
    smoothed_signals = torch.zeros(episode_length, n_envs, signal_dim, dtype=torch.float32)
    for i in range(episode_length):
        new_signal = nominal_signal[i].repeat(n_envs, signal_dim)  # Shape: (n_envs, signal_dim)
        ep_finished = torch.tensor([False] * n_envs, dtype=torch.bool)
        smoother.update(new_signal, ep_finished)
        
        smoothed_signals[i] = smoother.get(clone=True)

    # Flatten the tensor for plotting
    smoothed_signals = smoothed_signals.squeeze().numpy()

    # Plot the results
    plt.figure(figsize=(14, 7))
    plt.plot(t.numpy(), nominal_signal.numpy(), label='Nominal Signal', color='blue', linestyle='--')
    plt.plot(t.numpy(), smoothed_signals[:, 0], label='Smoothed Signal', color='red')
    plt.xlabel('Time [s]')
    plt.ylabel('Signal Value')
    plt.title('Nominal VS Smoothed Signals')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print last signal and smoothed signal
    print("Last nominal signal:")
    print(nominal_signal[-1].item())
    print("Last smoothed signal:")
    print(smoothed_signals[-1, 0])
    
    # Test resetting
    smoother.reset_all()
    print("\nAfter reset_all:")
    print("smoothed")
    print(smoother.get(clone=True))
    print("counter")
    print(smoother.counter(clone=True))

    # Update alpha and test
    smoother.update_alpha(update_dt=0.02, smoothing_horizon=1.0, target_smoothing=0.5)
    print("\nAfter updating alpha:")
    print(f"New alpha: {smoother.alpha()}")
    print("Smoothed signal after alpha update:")
    print(smoother.get(clone=True))
