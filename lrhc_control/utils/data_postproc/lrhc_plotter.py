import h5py
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from typing import List, Union
import re
import math
import matplotlib.lines as mlines

# Define a custom colormap (this is an example; adjust as needed)
colors = [
    "#0000FF",  # Blue
    "#00FFFF",  # Cyan
    "#00FF00",  # Green
    "#FFFF00",  # Yellow
    "#FF0000"   # Red
]
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

class LRHCPlotter:
    def __init__(self, hdf5_file_path):
        """
        Initialize the LRHCPlotter with the path to the HDF5 file.
        
        Args:
            hdf5_file_path (str): Path to the HDF5 file.
        """
        print(hdf5_file_path)
        if not os.path.exists(hdf5_file_path):
            raise FileNotFoundError(f"The file '{hdf5_file_path}' does not exist.")
        self.hdf5_file_path = hdf5_file_path
        self.data = {}
        self.attributes = {}  # Dictionary to store file-level attributes
        self.figures = []  # List to store figure objects

    def list_datasets(self):
        """
        Retrieve and return all dataset names available in the HDF5 file.
        
        Returns:
            list: A list of dataset names in the HDF5 file.
        """
        dataset_names = []
        try:
            with h5py.File(self.hdf5_file_path, 'r') as file:
                file.visititems(lambda name, obj: dataset_names.append(name) if isinstance(obj, h5py.Dataset) else None)
            return dataset_names
        except Exception as e:
            print(f"Error listing datasets: {e}")
            return dataset_names  # Return an empty list if an error occurs
    
    def list_attributes(self):
        """
        Retrieve and return all attributes from the HDF5 file.
        
        Returns:
            dict: A dictionary of attribute names and their corresponding values.
        """
        attributes = {}
        try:
            with h5py.File(self.hdf5_file_path, 'r') as file:
                for key, value in file.attrs.items():
                    attributes[key] = value
            return attributes
        except Exception as e:
            print(f"Error listing attributes: {e}")
            return attributes  # Return an empty dictionary if an error occurs

    def load_attributes(self):
        """
        Load all file-level attributes from the HDF5 file and store them in the attributes dictionary.
        """
        try:
            with h5py.File(self.hdf5_file_path, 'r') as file:
                for key, value in file.attrs.items():
                    self.attributes[key] = value
            print(f"Loaded attributes: {self.attributes}")
        except Exception as e:
            print(f"Error loading attributes: {e}")


    def load_data(self, dataset_names):
        """
        Load one or more datasets from the HDF5 file.
        
        Args:
            dataset_names (str or list): Name(s) of the datasets to load.
        """
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]  # Convert single string to list

        try:
            with h5py.File(self.hdf5_file_path, 'r') as file:
                for dataset_name in dataset_names:
                    if dataset_name in file:
                        dataset = file[dataset_name]
                        # Check if the dataset is scalar
                        if dataset.shape == ():  # Scalar datasets have an empty shape
                            self.data[dataset_name] = dataset[()]  # Use scalar access
                        else:
                            self.data[dataset_name] = dataset[:]  # Use slicing for arrays
                        print(f"Dataset '{dataset_name}' with shape {self.data[dataset_name].shape} loaded successfully.")
                    else:
                        print(f"Warning: Dataset '{dataset_name}' not found in the file.")
        except Exception as e:
            print(f"Error loading data: {e}")

    def plot_data(self, dataset_name,
            title="Plot",
            xaxis_dataset_name="", xlabel="Time", 
            ylabel="Intensity", 
            cmap="Blues",
            use_markers=False,
            data_labels = None,
            data_idxs: List[int] = None,
            distr_std = None, 
            distr_max = None, 
            distr_min = None):
        """
        Plot the data based on the number of environments in the dataset.
        
        For a single environment, plot a time series. For multiple environments,
        generate a heatmap histogram where the x-axis is time, y-axis is intensity,
        and the color gradient shows frequency. Handles non-finite values by discarding them.
        
        Args:
            dataset_name (str): Name of the dataset to plot.
            xaxis_dataset_name (str): Name of the dataset to use for the x-axis (optional).
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            cmap (str): Colormap for the heatmap (default: "Blues").
            show (bool): Whether to display the plot immediately. Default is True.
        """
        if dataset_name not in self.data:
            print(f"Dataset '{dataset_name}' not loaded. Use 'load_data' first.")
            return

        data_distr_std=None
        data_distr_min=None
        data_distr_max=None
        if distr_std is not None:
            if isinstance(distr_std, str):
                data_distr_std=self.data[distr_std]
        if distr_min is not None:
            if isinstance(distr_min, str):
                data_distr_min=self.data[distr_min]
        if distr_max is not None:
            if isinstance(distr_max, str):
                data_distr_max=self.data[distr_max]

        dataset = self.data[dataset_name]
        n_samples = 1
        n_envs = 1
        n_data = 1

        if dataset.ndim == 3:
            n_samples, n_envs, n_data = dataset.shape
        elif dataset.ndim == 2: 
            n_samples, n_data = dataset.shape
            n_envs=1
            dataset=dataset.reshape(-1, 1, n_data)
        else:
            print(f"Dataset '{dataset_name}' does not have the expected shape (n_samples x n_envs x n_data).")
            return

        # Handle optional x-axis dataset
        if xaxis_dataset_name:
            if xaxis_dataset_name not in self.data:
                print(f"X-axis dataset '{xaxis_dataset_name}' not loaded. Use 'load_data' first.")
                return
            xaxis_data = self.data[xaxis_dataset_name]
            if xaxis_data.shape != (n_samples, 1):
                print(f"X-axis dataset '{xaxis_dataset_name}' must have shape (n_samples, 1).")
                return
            xaxis = xaxis_data[:, 0]  # Extract as 1D array
        else:
            xaxis = np.arange(n_samples)  # Default x-axis is time steps

        fig, axes = None, None  # Initialize figure and axes objects

        if n_envs == 1:
            # Time series plot for single environment
            fig, ax = plt.subplots(figsize=(10, 5))
            data = dataset[:, 0, :]  # Extract single environment data
            if data_distr_std is not None and data_distr_std.ndim==3:
                data_distr_std=data_distr_std[:, 0, :]
            if data_distr_max is not None and data_distr_max.ndim==3:
                data_distr_max=data_distr_max[:, 0, :]
            if data_distr_min is not None and data_distr_min.ndim==3:
                data_distr_min=data_distr_min[:, 0, :]

            data_indexes=list(range(0, n_data)) if data_idxs is None else data_idxs
            labels=[]
            for i in range(len(data_indexes)):
                idx=data_indexes[i]
                valid_mask = np.isfinite(data[:, idx])
                label=f"Data {idx+1}" if data_labels is None else data_labels[i]
                labels.append(label)
                plt_line=None
                if use_markers:
                    plt_line, = ax.plot(xaxis[valid_mask], data[valid_mask, idx], 'o', label=label, markersize=3)
                else:
                    plt_line, = ax.plot(xaxis[valid_mask], data[valid_mask, idx], label=label)
                    if data_distr_std is not None: # add data distribution std area
                        alpha=0.2
                        ax.fill_between(xaxis[valid_mask], 
                                data[valid_mask, idx] - data_distr_std[valid_mask, idx], data[valid_mask, idx] + data_distr_std[valid_mask, idx],
                                color=plt_line.get_color(), alpha=alpha, 
                                label="± 1 std")
                    if data_distr_min is not None and data_distr_max is not None: # add min max bounds
                        alpha=0.2
                        if data_distr_std is not None:
                            alpha=0.1 # max min even more transparent
                        ax.fill_between(xaxis[valid_mask], 
                                data_distr_min[valid_mask, idx], data_distr_max[valid_mask, idx],
                                color=plt_line.get_color(), alpha=alpha, 
                                label="min/max")
                        
            ax.set_title(f"{title}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Value")
            # Create custom legend with lines instead of dots
            legend_lines = [mlines.Line2D([0], [0], color=ax.get_lines()[i].get_color(), lw=4) for i in range(len(data_indexes))]
            legend = ax.legend(legend_lines, labels, ncol=2, handlelength=2)

            # legend = ax.legend(ncol=2, markerscale=2)

            legend.set_draggable(True)  # Make the legend draggable

            ax.grid(True)
        else:
            # Heatmap histogram for multiple environments
            fig, axes = plt.subplots(n_data, 1, figsize=(10, 5 * n_data), sharex=True)
            if n_data == 1:
                axes = [axes]  # Ensure axes is iterable for n_data = 1
            
            for i in range(n_data):
                data = dataset[:, :, i]  # Extract data for all environments
                
                # Flatten and filter NaNs from data and corresponding x values
                valid_mask = xaxis > 0  # Valid (finite) mask
                
                valid_time = xaxis[valid_mask]
                valid_values = data[valid_mask, :]
                
                # Skip plotting if no valid data exists
                if valid_values.size == 0:
                    print(f"Data {i+1} contains no valid finite values. Skipping.")
                    continue
                
                tiled_values = np.zeros((valid_values.shape[0] * valid_values.shape[1],))
                for j in range(valid_values.shape[1]):
                    tiled_values[j * valid_values.shape[0]:(j * valid_values.shape[0] + valid_values.shape[0])] = valid_values[:, j]
                
                # Compute 2D histogram
                hist_2d, y_edges, x_edges = np.histogram2d(
                    tiled_values.flatten(),
                    np.tile(valid_time, valid_values.shape[1]),  # Filtered time axis
                    bins=(50, 100),  # Time bins and intensity bins
                    density=False,
                )
                
                # Plot the heatmap with white background and blue colormap
                im = axes[i].pcolormesh(
                    x_edges, 
                    y_edges, 
                    hist_2d, 
                    norm="log", # linear , symlog
                    cmap=cmap,
                    alpha=0.5,
                    shading="auto", # "nearest", "flat"
                    edgecolors="none",
                    # vmin=500,  # Set the minimum value for the colormap
                    # vmax=2000  # Set the maximum value for the colormap
                # Smooth shading for better visualization
                )
                axes[i].set_title(f"Data {i+1}")
                axes[i].set_ylabel(ylabel)
                axes[i].grid()
                
                # Add a colorbar to the plot
                fig.colorbar(im, ax=axes[i], label="Frequency")

            axes[-1].set_xlabel(xlabel)
            plt.suptitle(title)
            plt.tight_layout()

        # Store the figure in the list
        if fig is not None:
            self.figures.append(fig)

    def show(self):
        """
        Display all stored plots.
        """
        for fig in self.figures:
            fig.show()

        plt.show()

    def get_idx_matching(self,pattern_list, original_list):
        """
        Find the indices of strings in the original list that match any of the patterns.

        Args:
            pattern_list (list of str): List of regex patterns to match.
            original_list (list of str): List of strings to search in.

        Returns:
            list of int: List of indices of strings in original_list that match any pattern.
        """
        matching_indices = []
        matching_names = []
        
        for idx, string in enumerate(original_list):
            for pattern in pattern_list:
                if re.search(pattern, string):  # Check if pattern matches the string
                    matching_indices.append(idx)
                    break  # Stop checking further patterns for this string
        
        for i in range(len(matching_indices)):
            matching_names.append(original_list[matching_indices[i]])

        return matching_indices,matching_names

    def compose_datasets(self, datasets_list: List[str], name: str):
        """
        Compose multiple datasets along the data dimension (third dimension) and store the result.

        Args:
            datasets_list (List[str]): A list of dataset names to compose.
            name (str): Name for the new composed dataset.

        Raises:
            ValueError: If datasets have incompatible shapes or are not found.
        """
        # Check that all datasets are loaded
        for dataset_name in datasets_list:
            if dataset_name not in self.data:
                raise ValueError(f"Dataset '{dataset_name}' not loaded. Use 'load_data' first.")

        # Retrieve the datasets
        datasets = [self.data[dataset_name] for dataset_name in datasets_list]

        # Check compatibility (same number of samples and environments)
        base_shape = datasets[0].shape
        n_samples, n_envs = base_shape[0], base_shape[1] if len(base_shape) > 1 else 1
        for dataset, dataset_name in zip(datasets, datasets_list):
            if len(dataset.shape) == 2:  # Add singleton environment dimension if needed
                dataset = dataset[:, np.newaxis, :]
            if dataset.shape[:2] != (n_samples, n_envs):
                raise ValueError(
                    f"Dataset '{dataset_name}' has incompatible shape {dataset.shape}. "
                    f"Expected {n_samples} samples and {n_envs} environments."
                )

        # Stack datasets along the data dimension
        composed_data = np.concatenate(datasets, axis=-1)

        # Store the new dataset
        self.data[name] = composed_data
        print(f"Composed dataset '{name}' created with shape {composed_data.shape}.")

if __name__ == "__main__":
    
    # load training data
    path = "/root/training_data/d2025_01_05_h14_m44_s21-FakePosEnvBaselinedb_info.hdf5"
    plotter = LRHCPlotter(hdf5_file_path=path)
    datasets = plotter.list_datasets()
    attributes = plotter.list_attributes()
    print("Dataset names:")
    print(datasets)
    print("Attributes:")
    print(attributes)
    plotter.load_data(dataset_names=datasets)
    plotter.load_attributes()

    obs_names=list(plotter.attributes["obs_names"])

    # plot some data

    # cmd efforts
    patterns=["rhc_cmd_eff_hip"]
    selected_obs_idx,selected_obs=plotter.get_idx_matching(patterns, obs_names)
    plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - rhc cmd effort", 
        xaxis_dataset_name="n_timesteps_done",
        xlabel="n_timesteps_done",
        use_markers=True,
        data_labels=selected_obs,
        data_idxs=selected_obs_idx)
    
    # estimated contact forces
    patterns=["fc_contact"]
    selected_f_idx, selected_f=plotter.get_idx_matching(patterns, obs_names)
    plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - est. contact f", 
        xaxis_dataset_name="n_timesteps_done",
        xlabel="n_timesteps_done",
        use_markers=True,
        data_labels=selected_f,
        data_idxs=selected_f_idx)
    
    # mpc fail idx
    patterns=["fail"]
    f_idx, selected_f=plotter.get_idx_matching(patterns, obs_names)
    plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - MPC fail index", 
        xaxis_dataset_name="n_timesteps_done",
        xlabel="n_timesteps_done",
        use_markers=True,
        data_labels=selected_f,
        data_idxs=f_idx)
    
    # losses 
    plotter.compose_datasets(name="qf_loss",
        datasets_list=["qf1_loss", "qf1_loss_validation", 
                    "qf2_loss", "qf2_loss_validation"])
    plotter.plot_data(dataset_name="qf_loss", title="qf loss", 
        xaxis_dataset_name="n_timesteps_done",
        xlabel="n_timesteps_done",
        use_markers=False)
    plotter.compose_datasets(name="actor_losses",
        datasets_list=["actor_loss", "actor_loss_validation"])
    plotter.plot_data(dataset_name="actor_losses", title="actor_loss", 
        xaxis_dataset_name="n_timesteps_done",
        xlabel="n_timesteps_done",
        use_markers=True)
    plotter.compose_datasets(name="alpha_losses",
        datasets_list=["alpha_loss", "alpha_loss_validation"])
    plotter.plot_data(dataset_name="alpha_losses", title="alpha_loss", 
        xaxis_dataset_name="n_timesteps_done",
        xlabel="n_timesteps_done",
        use_markers=True)
    
    # other training data
    plotter.compose_datasets(name="qf_vals",
        datasets_list=["qf1_vals_mean", "qf2_vals_mean"])
    plotter.compose_datasets(name="qf_vals_std",
        datasets_list=["qf1_vals_std", "qf2_vals_std"])
    plotter.compose_datasets(name="qf_vals_max",
        datasets_list=["qf1_vals_max", "qf2_vals_max"])
    plotter.compose_datasets(name="qf_vals_min",
        datasets_list=["qf1_vals_min", "qf2_vals_min"])
    plotter.plot_data(dataset_name="qf_vals", 
        title="Qf mean - std - min/max", 
        xaxis_dataset_name="n_timesteps_done",
        xlabel="n_timesteps_done",
        ylabel="Q val.",
        use_markers=False,
        distr_std="qf_vals_std",
        distr_max="qf_vals_max",
        distr_min="qf_vals_min")

    # total reward
    plotter.plot_data(dataset_name="tot_rew_avrg", title="tot_rew_avrg", 
        xaxis_dataset_name="n_timesteps_done",
        xlabel="n_timesteps_done")
    plotter.plot_data(dataset_name="tot_rew_avrg_over_envs", title="tot_rew_avrg_over_envs", 
        xaxis_dataset_name="n_timesteps_done",
        xlabel="n_timesteps_done")
    plotter.plot_data(dataset_name="tot_rew_max", title="tot_rew_max", 
        xaxis_dataset_name="n_timesteps_done",
        xlabel="n_timesteps_done")
    plotter.plot_data(dataset_name="tot_rew_min", title="tot_rew_min", 
        xaxis_dataset_name="n_timesteps_done",
        xlabel="n_timesteps_done")

    # env data 
    plotter.plot_data(dataset_name="env_step_rt_factor", title="env_step_rt_factor", 
        xaxis_dataset_name="n_timesteps_done",
        xlabel="n_timesteps_done",
        use_markers=True)
    
    plotter.plot_data(dataset_name="ep_tsteps_env_distribution", title="ep_tsteps_env_distribution", 
        xaxis_dataset_name="n_timesteps_done",
        xlabel="n_timesteps_done",
        use_markers=True)
    
    plotter.plot_data(dataset_name="SubTruncations_avrg_over_envs", title="SubTruncations_avrg_over_envs", 
        xaxis_dataset_name="n_timesteps_done",
        xlabel="n_timesteps_done",
        use_markers=True)
    plotter.plot_data(dataset_name="SubTerminations_avrg_over_envs", title="SubTerminations_avrg_over_envs", 
        xaxis_dataset_name="n_timesteps_done",
        xlabel="n_timesteps_done",
        use_markers=True)
    plotter.show()  # Display all plots
    
