from typing import Tuple, Union

import torch

from EigenIPC.PyEigenIPC import LogType
from EigenIPC.PyEigenIPC import Journal
from EigenIPC.PyEigenIPC import VLevel

class RunningMeanStd(object):
    def __init__(self, tensor_size, torch_device, dtype, epsilon: float = 1e-8,
                debug: bool = False):
        """
        Torch version of the same from stable_baselines3 (credits @c-rizz; Carlo Rizzardo)

        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        
        self._epsilon = epsilon
        self.mean = torch.zeros(tensor_size, device=torch_device, dtype=dtype)
        self.var = torch.ones(tensor_size, device=torch_device, dtype=dtype)
        self.count = torch.tensor(epsilon, device=torch_device, dtype=torch.float64)
        self.debug = debug

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd(tensor_size=self.mean.size(),
                                    torch_device=self.mean.device,
                                    dtype = self.mean.dtype,
                                    epsilon=self._epsilon,
                                    debug=self.debug)
        # use copy_() to avoid breaking buffer registration
        new_object.mean.copy_(self.mean.detach())
        new_object.var.copy_(self.var.detach())  
        new_object.count.copy_(self.count.detach())
        return new_object
    
    def copy_(self, src) -> None:
        """
        :return: Return a copy of the current object.
        """
        # use copy_() to avoid breaking buffer registration
        self.mean.copy_(src.mean.detach())
        self.var.copy_(src.var.detach())  
        self.count.copy_(src.count.detach())

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine witorch.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, x) -> None:
        if not x.shape[0] == 1:
            batch_mean = torch.mean(x, dim=0)
            batch_var = torch.var(x, dim=0)
            batch_size = x.size()[0]
            self.update_from_moments(batch_mean, batch_var, batch_size)
        else:
            if self.debug:
                Journal.log(self.__class__.__name__,
                    "update",
                    f"Provided batch is made of only 1 sample. Cannot update mean and std!",
                    LogType.WARN)

    def update_from_moments(self, batch_mean, batch_var, batch_size: Union[int, float]) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_size

        new_mean = self.mean + delta * batch_size / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_size
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_size / (self.count + batch_size)
        new_var = m_2 / (self.count + batch_size)

        new_count = batch_size + self.count
        
        # skip if there are infs and nans
        if torch.all(torch.isfinite(new_mean)) and torch.all(torch.isfinite(new_var)) and torch.all(torch.isfinite(new_count)):
            # use copy_() to avoid breaking buffer registration
            self.mean.copy_(new_mean)
            self.var.copy_(new_var)
            self.count.copy_(new_count)
        else:
            if self.debug:
                Journal.log(self.__class__.__name__,
                        "update_from_moments",
                        f"Detected nan/inf in mean/std tracker, skipping samples!!",
                        LogType.WARN)

class RunningNormalizer(torch.nn.Module):
    def __init__(self, shape : Tuple[int,...], dtype, device, 
                epsilon : float = 1e-8, 
                freeze_stats: bool=True,
                debug: bool = False):
        super().__init__()
        self._freeze_stats = freeze_stats
        self.register_buffer("_epsilon", torch.tensor(epsilon, device = device))
        self._running_stats = RunningMeanStd(shape, torch_device=device,dtype=dtype,
                                epsilon=epsilon,
                                debug=debug)
        self.register_buffer("vec_running_mean",  self._running_stats.mean)
        self.register_buffer("vec_running_var",   self._running_stats.var)
        self.register_buffer("vec_running_count", self._running_stats.count)

    def forward(self, x):
        
        if not (self.training or self._freeze_stats):
            self._running_stats.update(x)
        
        return (x - self._running_stats.mean)/(torch.sqrt(self._running_stats.var)+self._epsilon)

    def get_current_mean(self):
        return self._running_stats.mean.detach().clone()
    
    def get_current_std(self):
        return torch.sqrt(self._running_stats.var.detach().clone())
    
if __name__ == "__main__":  
    
    device = "cuda"
    obs_dim = 4  # Number of observation dimensions
    n_envs = 20  # Number of environments
    n_samples = 10000  # Number of iterations to update running stats

    # Set a known mean and variance for the synthetic data
    true_mean = torch.tensor([10.0, -5.0, 3.0, 7.0], dtype=torch.float32, device=device)
    true_std = torch.tensor([2.0, 0.5, 1.0, 3.0], dtype=torch.float32, device=device)

    # Create a dummy observation tensor where each observation has a known mean and std
    dummy_obs = true_mean + true_std * torch.randn(size=(n_envs, obs_dim), dtype=torch.float32, device=device)

    # Create a RunningNormalizer and set it to update running stats
    normalizer = RunningNormalizer((obs_dim,), epsilon=1e-8, device=device, dtype=torch.float32, freeze_stats=False)
    normalizer.train(False)

    # Feed the controlled data into the normalizer for multiple samples
    for i in range(n_samples):
        # Generate new batch of data with the same known mean and std
        dummy_obs = true_mean + true_std * torch.randn(size=(n_envs, obs_dim), dtype=torch.float32, device=device)
        normalizer(dummy_obs)

    # Switch to evaluation mode, meaning no more updates to running stats
    normalizer.train(True)

    # Print the computed running mean and std (they should match the true_mean and true_std)
    print("Running mean after updates")
    print(normalizer._running_stats.mean)  # Should be close to true_mean
    print("Running std after updates")
    print(torch.sqrt(normalizer._running_stats.var))  # Should be close to true_std

    # Now, test the normalization output
    test_obs = true_mean + true_std * torch.randn(size=(n_envs, obs_dim), dtype=torch.float32, device=device)
    normalized_obs = normalizer(test_obs)

    # Check if the normalized output has a mean of ~0 and std of ~1
    output_mean = normalized_obs.mean(dim=0)
    output_std = normalized_obs.std(dim=0)

    print("Output mean (should be close to 0)")
    print(output_mean)  # Should be approximately 0

    print("Output std (should be close to 1)")
    print(output_std)  # Should be approximately 1