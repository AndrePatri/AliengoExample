import h5py
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

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

    def plot_data(self, dataset_name, xaxis_dataset_name="", title="Plot", xlabel="Time", ylabel="Intensity", cmap="Blues"):
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
        """
        if dataset_name not in self.data:
            print(f"Dataset '{dataset_name}' not loaded. Use 'load_data' first.")
            return

        dataset = self.data[dataset_name]
        if dataset.ndim != 3:
            print(f"Dataset '{dataset_name}' does not have the expected shape (n_samples x n_envs x n_data).")
            return

        n_samples, n_envs, n_data = dataset.shape

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

        if n_envs == 1:
            # Time series plot for single environment
            data = dataset[:, 0, :]  # Extract single environment data
            for i in range(n_data):
                valid_mask = np.isfinite(data[:, i])
                plt.plot(xaxis[valid_mask], data[valid_mask, i], label=f"Data {i+1}")
            plt.title(f"{title} - Single Environment")
            plt.xlabel(xlabel)
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.show()
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
                    alpha=0.8,
                    shading="auto", # "nearest", "flat"
                    edgecolors="none",
                    # vmin=500,  # Set the minimum value for the colormap
                    # vmax=2000  # Set the maximum value for the colormap
                # Smooth shading for better visualization
                )
                axes[i].set_title(f"Data {i+1} Heatmap Histogram")
                axes[i].set_ylabel(ylabel)
                axes[i].grid()
                
                # Add a colorbar to the plot
                fig.colorbar(im, ax=axes[i], label="Frequency")

            axes[-1].set_xlabel(xlabel)
            plt.suptitle(title)
            plt.tight_layout()
            plt.show()
       

if __name__ == "__main__":  
    path="/root/training_data/d2025_01_04_h13_m45_s15-FakePosEnvBaseline/d2025_01_04_h13_m45_s15-FakePosEnvBaselinedb_info.hdf5"
    plotter=LRHCPlotter(hdf5_file_path=path)
    datasets=plotter.list_datasets()
    print("Datasets names")
    print(datasets)
    plotter.load_data(dataset_names=datasets)
    # plotter.plot_data(dataset_name="tot_rew_avrg_over_envs", title="tot_rew_avrg", xaxis_dataset_name="n_timesteps_done")
    plotter.plot_data(dataset_name="tot_rew_avrg", title="tot_rew_avrg", xaxis_dataset_name="n_timesteps_done")
