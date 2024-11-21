import torch
import numpy as np
from typing import List, Union, Dict

class QuadrupedGaitPatternGenerator:
    def __init__(self, 
        phase_period: float = 1.0,
        ):
        self._n_phases = 4 
        self._phase_period = phase_period
        
        self.patterns = {
            "trot": self._trot,
            "walk": self._walk,
            "pace": self._pace,
            "canter": self._canter
        }

    # Front left, back left, back right, front right

    def get_params(self, name: str) -> Dict[str, Union[float, List[float]]]:
        if name not in self.patterns:
            raise ValueError(f"Gait pattern '{name}' not recognized. Available patterns are {list(self.patterns.keys())}.")
        return self.patterns[name]()

    def _trot(self) -> Dict[str, Union[float, List[float]]]:

        phase_offset = [self._phase_period / 2, 0.0, 0.0, self._phase_period / 2]  # Diagonally opposite phases
        flight_length=self._phase_period / 4
        t_star = 3/4*self._phase_period-flight_length/2
        phase_thresh = [np.sin(2*np.pi/self._phase_period*t_star)] * self._n_phases
        
        return {
            "n_phases": self._n_phases,
            "phase_period": self._phase_period,
            "phase_offset": phase_offset,
            "phase_thresh": phase_thresh
        }

    def _walk(self) -> Dict[str, Union[float, List[float]]]:
        
        phase_offset = [0.0, self._phase_period / 4, self._phase_period / 2, 3 * self._phase_period / 4]  # Sequential phases
        # phase_thresh = [np.cos(3 * np.pi / 4)] * self._n_phases
        flight_length=self._phase_period / 32
        t_star = 3/4*self._phase_period-flight_length/2
        phase_thresh = [np.sin(2*np.pi/self._phase_period*t_star)] * self._n_phases
        return {
            "n_phases": self._n_phases,
            "phase_period": self._phase_period,
            "phase_offset": phase_offset,
            "phase_thresh": phase_thresh
        }

    # def _walk(self) -> Dict[str, Union[float, List[float]]]:
    #     phase_offset = [0.0, self._phase_period / 3, 2 * self._phase_period / 3, self._phase_period]  # Larger phase intervals
    #     flight_length = self._phase_period / 4
    #     t_star = 3/4 * self._phase_period - flight_length / 2
    #     phase_thresh = [np.sin(2 * np.pi / self._phase_period * t_star)] * self._n_phases
        
    #     return {
    #         "n_phases": self._n_phases,
    #         "phase_period": self._phase_period,
    #         "phase_offset": phase_offset,
    #         "phase_thresh": phase_thresh
    #     }

    def _pace(self) -> Dict[str, Union[float, List[float]]]:
        
        phase_offset = [0.0, self._phase_period / 2, 0.0, self._phase_period / 2]  # Lateral pairs in phase
        phase_thresh = [0.0] * self._n_phases
        
        return {
            "n_phases": self._n_phases,
            "phase_period": self._phase_period,
            "phase_offset": phase_offset,
            "phase_thresh": phase_thresh
        }

    def _canter(self) -> Dict[str, Union[float, List[float]]]:
       
        phase_offset = [0.0, self._phase_period / 3, 2 * self._phase_period / 3, self._phase_period]  # Alternating phases
        phase_thresh = [0.0] * self._n_phases
        
        return {
            "n_phases": self._n_phases,
            "phase_period": self._phase_period,
            "phase_offset": phase_offset,
            "phase_thresh": phase_thresh
        }
    
class GaitScheduler:
    def __init__(self, 
        n_phases: int,
        n_envs: int,
        update_dt: float,
        phase_period: Union[float, List[float]],
        phase_offset: Union[float, List[float]],
        phase_thresh: Union[float, List[float]],
        use_gpu: bool = False,
        dtype: torch.dtype = torch.float32):

        self._n_phases = n_phases
        self._n_envs = n_envs

        self._update_dt = update_dt

        self._use_gpu = use_gpu
        self._device = "cuda" if self._use_gpu else "cpu"
        self._torch_dtype = dtype

        self._pi = torch.tensor(np.pi, dtype=self._torch_dtype, device=self._device)

        # Initialize tensors
        self._signal = torch.full((self._n_envs, self._n_phases),
                                  dtype=self._torch_dtype,
                                  device=self._device, 
                                  fill_value=0.0)

        self._steps_counter = torch.full((self._n_envs, self._n_phases), 
                                         fill_value=0,
                                         dtype=torch.int32, 
                                         device=self._device)
        
        # Process self._phase_period
        if isinstance(phase_period, float):
            self._phase_period = torch.full((1, self._n_phases),
                                            dtype=self._torch_dtype,
                                            device=self._device, 
                                            fill_value=phase_period)
        elif isinstance(phase_period, list):
            assert len(self._phase_period) == self._n_phases, "Phase period list length must match n_phases"
            self._phase_period = torch.tensor(phase_period, 
                                               dtype=self._torch_dtype,
                                               device=self._device).unsqueeze(0)
        else:
            raise TypeError("phase_period must be a float or a list of floats")

        # Process phase_offset
        if isinstance(phase_offset, float):
            self._phase_offset = torch.full((1, self._n_phases),
                                             dtype=self._torch_dtype,
                                             device=self._device, 
                                             fill_value=phase_offset)
        elif isinstance(phase_offset, list):
            assert len(phase_offset) == self._n_phases, "Phase offset list length must match n_phases"
            self._phase_offset = torch.tensor(phase_offset, 
                                               dtype=self._torch_dtype,
                                               device=self._device).unsqueeze(0)
        else:
            raise TypeError("phase_offset must be a float or a list of floats")

        # Process phase_thresh
        if isinstance(phase_thresh, float):
            self._threshold = torch.full((1, self._n_phases),
                                          dtype=self._torch_dtype,
                                          device=self._device, 
                                          fill_value=phase_thresh)
        elif isinstance(phase_thresh, list):
            assert len(phase_thresh) == self._n_phases, "Phase threshold list length must match n_phases"
            self._threshold = torch.tensor(phase_thresh, 
                                            dtype=self._torch_dtype,
                                            device=self._device).unsqueeze(0)
        else:
            raise TypeError("phase_thresh must be a float or a list of floats")

        self.reset()
        
    def reset(self, 
        to_be_reset: torch.Tensor = None):
        if to_be_reset is None: # reset all
            self._steps_counter.zero_()
            self._signal.zero_()
        else:
            self._steps_counter[to_be_reset, :] = 0
            self._signal[to_be_reset, :] = 0.0

    def step(self):
        # Calculate the phase signal
        
        self._signal[:, :] = \
            torch.sin((self._steps_counter*self._update_dt+self._phase_offset)*2*self._pi/self._phase_period)
        
        # Increment step counter
        self._steps_counter += 1
        return self._signal[:, :] > self._threshold

    def get_signal(self, clone: bool = False):

        if clone:
            return self._signal.clone()
        else:
            return self._signal    

    def threshold(self):

        return self._threshold
    
# Testing code
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Parameters for the test
    pattern_gen = QuadrupedGaitPatternGenerator(phase_period=2.0)
    gait_params = pattern_gen.get_params("walk")
    n_phases = gait_params["n_phases"]
    n_envs = 1
    update_dt = 0.03
    phase_period = gait_params["phase_period"]
    phase_offset = gait_params["phase_offset"]
    phase_thresh = gait_params["phase_thresh"]
    
    # Initialize the GaitScheduler
    scheduler = GaitScheduler(
        n_phases=n_phases,
        n_envs=n_envs,
        update_dt=update_dt,
        phase_period=phase_period,
        phase_offset=phase_offset,
        phase_thresh=phase_thresh,
        use_gpu=False,
        dtype=torch.float32
    )

    # Run the step method for 10 iterations and collect the results
    num_steps = round(1*phase_period / update_dt)
    signals = []
    bool_signals = []
    
    env_idx = 0
    for _ in range(num_steps+1):
        bool_signals.append(scheduler.step()[env_idx,:].cpu().numpy())
        signals.append(scheduler.get_signal(clone=True)[env_idx,:].cpu().numpy())

    signals = np.array(signals)
    bool_signals = np.array(bool_signals)

    # Generate the time array in seconds
    time_array = np.arange(num_steps + 1) * update_dt

    # Plotting all signals on the same plot, with threshold lines
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot the continuous phase signals with threshold lines
    for i in range(n_phases):
        axs[0].plot(time_array, signals[:, i], label=f'Phase signal {i+1}')
        axs[0].axhline(y=phase_thresh[i], color=f'C{i}', linestyle='--', label=f'Threshold {i+1}')  # Threshold line
    axs[0].set_title('Phase Signals with Thresholds')
    axs[0].set_xlabel('Time (seconds)')
    axs[0].set_ylabel('Signal Value')
    axs[0].legend()
    axs[0].grid(True)  # Add grid

    # Plot the boolean (thresholded) phase signals
    for i in range(n_phases):
        axs[1].plot(time_array, bool_signals[:, i], label=f'Phase boolean {i+1}')
    axs[1].set_title('Phase Boolean Signals (Above Threshold)')
    axs[1].set_xlabel('Time (seconds)')
    axs[1].set_ylabel('Boolean Value')
    axs[1].legend()
    axs[1].grid(True)  # Add grid

    plt.tight_layout()
    plt.show()
