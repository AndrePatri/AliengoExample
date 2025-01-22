import torch 

from typing import List
from enum import Enum

from lrhc_control.utils.urdf_limits_parser import UrdfLimitsParser
from lrhc_control.utils.jnt_imp_cfg_parser import JntImpConfigParser

import time

from EigenIPC.PyEigenIPC import LogType
from EigenIPC.PyEigenIPC import Journal

from abc import ABC, abstractmethod

class FirstOrderFilter:

    # a class implementing a simple first order filter

    def __init__(self,
            dt: float, 
            filter_BW: float = 0.1, 
            rows: int = 1, 
            cols: int = 1, 
            device: torch.device = torch.device("cpu"),
            dtype = torch.double):
        
        self._torch_dtype = dtype

        self._torch_device = device

        self._dt = dt

        self._rows = rows
        self._cols = cols

        self._filter_BW = filter_BW

        import math 
        self._gain = 2 * math.pi * self._filter_BW

        self.yk = torch.zeros((self._rows, self._cols), device = self._torch_device, 
                                dtype=self._torch_dtype)
        self.ykm1 = torch.zeros((self._rows, self._cols), device = self._torch_device, 
                                dtype=self._torch_dtype)
        
        self.refk = torch.zeros((self._rows, self._cols), device = self._torch_device, 
                                dtype=self._torch_dtype)
        self.refkm1 = torch.zeros((self._rows, self._cols), device = self._torch_device, 
                                dtype=self._torch_dtype)
        
        self._kh2 = self._gain * self._dt / 2.0
        self._coeff_ref = self._kh2 * 1/ (1 + self._kh2)
        self._coeff_km1 = (1 - self._kh2) / (1 + self._kh2)

    def update(self, 
               refk: torch.Tensor = None):
        
        if refk is not None:
            self.refk[:, :] = refk
        self.yk[:, :] = torch.add(torch.mul(self.ykm1, self._coeff_km1), 
                            torch.mul(torch.add(self.refk, self.refkm1), 
                                        self._coeff_ref))
        self.refkm1[:, :] = self.refk
        self.ykm1[:, :] = self.yk
    
    def reset(self,
            idxs: torch.Tensor = None):

        if idxs is None:
            self.yk[:, :] = torch.zeros((self._rows, self._cols), 
                                device = self._torch_device, 
                                dtype=self._torch_dtype)
            self.ykm1[:, :] = torch.zeros((self._rows, self._cols), 
                                device = self._torch_device, 
                                dtype=self._torch_dtype)
            self.refk[:, :] = torch.zeros((self._rows, self._cols), 
                                device = self._torch_device, 
                                dtype=self._torch_dtype)
            self.refkm1[:, :] = torch.zeros((self._rows, self._cols), 
                                device = self._torch_device, 
                                dtype=self._torch_dtype)
        else:
            self.yk[idxs, :] = torch.zeros((idxs.shape[0], self._cols), 
                                device = self._torch_device, 
                                dtype=self._torch_dtype)
            self.ykm1[idxs, :] = torch.zeros((idxs.shape[0], self._cols), 
                                device = self._torch_device, 
                                dtype=self._torch_dtype)
            self.refk[idxs, :] = torch.zeros((idxs.shape[0], self._cols), 
                                device = self._torch_device, 
                                dtype=self._torch_dtype)
            self.refkm1[idxs, :] = torch.zeros((idxs.shape[0], self._cols), 
                                device = self._torch_device, 
                                dtype=self._torch_dtype)
            
    def get(self, clone: bool = False):
        out=self.yk
        if clone:
            out=out.clone()
        return out
          
if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt

    high_res_dt = 0.001  # 1 ms sampling time

    filter_dt = 0.01  # 20 ms sampling time (filtering rate)
    filter_BW = 15.0  # Filter bandwidth (Hz)
    
    rows, cols = 3, 2  # Three environments, two signal dimensions
    device = torch.device("cpu")

    # Create the filter
    filter = FirstOrderFilter(filter_dt, filter_BW, rows, cols, device)

    # Generate high-resolution time vector
    t_high_res = np.arange(0, 10, high_res_dt)  # 10 seconds of data

    # Generate original sinusoidal signals (one for each environment and column)
    signal_frequencies = [0.5, 0.7, 1.0]  # Different frequencies for each environment
    amplitudes = [1.0, 1.2, 0.8]  # Different amplitudes for each environment
    signals = []
    for i in range(rows):
        signal = amplitudes[i] * np.sin(2 * np.pi * signal_frequencies[i] * t_high_res)
        signals.append(np.stack([signal] * cols, axis=-1))  # Duplicate for 2 columns
    signals = np.stack(signals, axis=0)  # Shape: (rows, t_high_res, cols)

    # Generate high-frequency sinusoidal noise (same noise for all environments/columns)
    noise_frequency = 67.5  # Frequency of the noise (Hz)
    noise_amplitude = 0.2
    noise = noise_amplitude * np.sin(2 * np.pi * noise_frequency * t_high_res)
    noise = np.stack([noise] * cols, axis=-1)  # Duplicate for 2 columns
    noise = np.stack([noise] * rows, axis=0)  # Shape: (rows, t_high_res, cols)

    # Combine signal and noise
    noisy_signals = signals + noise  # Shape: (rows, t_high_res, cols)

    # Filter processing at 20 ms rate
    filtered_signals = []
    t_filtered = t_high_res[::int(filter_dt / high_res_dt)]  # Downsampled time vector
    for idx in range(len(t_filtered)):
        # Take noisy samples for each environment at the current downsampled index
        refk = torch.tensor(
            noisy_signals[:, idx * int(filter_dt / high_res_dt), :],
            device=device,
            dtype=filter._torch_dtype,
        )
        filter.update(refk)  # Pass the vectorized tensor directly to update
        filtered_signals.append(filter.get(clone=True).cpu().numpy())
    
    # Convert filtered signals to numpy array for plotting
    filtered_signals = np.stack(filtered_signals, axis=1)  # Shape: (rows, t_filtered, cols)

    # Plot results for each environment
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    for i in range(rows):
        for j in range(cols):
            ax = axes[i, j]
            ax.plot(t_high_res, signals[i, :, j], label="Original Signal", linestyle="--", linewidth=2)
            ax.plot(t_high_res, noisy_signals[i, :, j], label="Noisy Signal", alpha=0.7)
            ax.plot(t_filtered, filtered_signals[i, :, j], label="Filtered Signal", linewidth=2, markersize=1,marker="o")
            ax.set_title(f"Env {i + 1}, Dim {j + 1}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.legend()
            ax.grid()

    plt.tight_layout()
    plt.show()