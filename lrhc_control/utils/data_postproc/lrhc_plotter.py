import h5py
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from typing import List, Union, Tuple
import re
import math
import matplotlib.lines as mlines
import argparse
from matplotlib.colors import LogNorm

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
    def __init__(self, hdf5_file_path, verbose: bool = True):
        """
        Initialize the LRHCPlotter with the path to the HDF5 file.
        
        Args:
            hdf5_file_path (str): Path to the HDF5 file.
        """
        self._verbose=verbose

        if not os.path.exists(hdf5_file_path):
            raise FileNotFoundError(f"The file '{hdf5_file_path}' does not exist.")
        self.hdf5_file_path = hdf5_file_path
        self.data = {}
        self.attributes = {}  # Dictionary to store file-level attributes
        self.figures = []  # List to store figure objects

        self.map_legend_to_ax = {}  # Will map legend lines to original lines.

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
            # print(f"Loaded attributes: {self.attributes}")
        except Exception as e:
            print(f"Error loading attributes: {e}")

    def load_data(self, dataset_names, env_idx: int = None):
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
                            if env_idx is not None: # load data just for one specific env
                                if self._verbose:
                                    print(f"Dataset '{dataset_name}' will be loaded only for env {env_idx}. Original shape {self.data[dataset_name].shape[0]}, {self.data[dataset_name].shape[1]}, {self.data[dataset_name].shape[2]}")
                                self.data[dataset_name] = self.data[dataset_name][:, env_idx:env_idx+1, :]
                        if self._verbose:
                            print(f"Dataset '{dataset_name}' with shape {self.data[dataset_name].shape} loaded successfully.")
                    else:
                        if self._verbose:
                            print(f"Warning: Dataset '{dataset_name}' not found in the file.")
        except Exception as e:
            print(f"Error loading data: {e}")

    def add_datas(self, dataset_name, datas: Tuple[np.ndarray],
        avrg: bool = True):
        
        dnames=list(self.data.keys())
        if not dataset_name in dnames:
            print(f"{dataset_name} not found in available data.")
            return False

        if avrg:
            try:
                self.data[dataset_name]=self.data[dataset_name]/len(datas)
            except:
                return False

        for i in range(len(datas)):
            n_samples_base=self.data[dataset_name].shape[0]
            n_samples_incoming=datas[i].shape[0]

            # to merge data we need the same number of samples (runs may be of different length)
            n_timesteps=n_samples_base if n_samples_base<=n_samples_incoming else n_samples_incoming

            base_data=self.data[dataset_name][0:n_timesteps, :, :]
            to_be_added=datas[i][:n_timesteps, :, :]

            if not avrg:
                self.data[dataset_name]=np.concatenate((base_data,to_be_added), axis=1) # add augmented data
            else: # compute average
                self.data[dataset_name]=base_data+to_be_added/len(datas)

        return True
    
    def create_dataset(self, dataset_name, data: np.ndarray):
        print(f"Created dataset '{dataset_name}' with shape {data.shape}.")
        self.data[dataset_name]=data

    def plot_data(self, dataset_name,
            title="Plot",
            xaxis_dataset_name="", xlabel="Time", 
            ylabel="Intensity", 
            cmap="plasma", # viridis, 
            use_markers=False,
            marker_size: int = 3,
            data_labels = None,
            data_alphas = None,
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
            plt_lines=[]
            
            for i in range(len(data_indexes)):
                idx=data_indexes[i]
                valid_mask = np.logical_and(np.isfinite(data[:, idx]), xaxis[:]>=0)
                label=f"Data {idx+1}" if data_labels is None else data_labels[i]
                alpha=data_alphas[i] if data_alphas is not None else 1.0
                labels.append(label)
                plt_line=None
            
                if use_markers:
                    plt_line, = ax.plot(xaxis[valid_mask], data[valid_mask, idx], 'o', label=label, markersize=marker_size, alpha=alpha)
                else:
                    plt_line, = ax.plot(xaxis[valid_mask], data[valid_mask, idx], label=label, alpha=alpha)
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
                plt_lines.append(plt_line)

            ax.set_title(f"{title}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Value")
            # Create custom legend with lines instead of dots
            legend_lines = [mlines.Line2D([0], [0], color=ax.get_lines()[i].get_color(), lw=4) for i in range(len(data_indexes))]
            legend = ax.legend(legend_lines, labels, ncol=2, handlelength=2)
            # Set pickable property
            for line in legend_lines:
                line.set_picker(True)
            # legend = ax.legend(ncol=2, markerscale=2)

            legend.set_draggable(True)  # Make the legend draggable
            
            ax.grid(True)

            # make legends pickable
            pickradius=5
            for legend_line, ax_line in zip(legend.get_lines(), plt_lines):
                legend_line.set_picker(pickradius)  # Enable picking on the legend line.
                self.map_legend_to_ax[legend_line] = ax_line
            
            def on_pick(event):
                # On the pick event, find the original line corresponding to the legend
                # proxy line, and toggle its visibility.
                legend_line = event.artist

                # Do nothing if the source of the event is not a legend line.
                if legend_line not in self.map_legend_to_ax:
                    return

                ax_line = self.map_legend_to_ax[legend_line]
                visible = not ax_line.get_visible()
                ax_line.set_visible(visible)
                # Change the alpha on the line in the legend, so we can see what lines
                # have been toggled.
                legend_line.set_alpha(1.0 if visible else 0.2)
                fig.canvas.draw()
            fig.canvas.mpl_connect('pick_event', on_pick)
            
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
                    norm=LogNorm(), # linear , symlog
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

    def get_idx_matching(self, pattern_list, original_list):
        """
        Find the indices of strings in the original list that match any of the patterns.

        Args:
            pattern_list (list of str): List of regex patterns to match. Patterns can include '*' as a wildcard.
            original_list (list of str): List of strings to search in.

        Returns:
            list of int: List of indices of strings in original_list that match any pattern.
            list of str: List of strings in original_list that match any pattern.
        """
        def compile_pattern(pattern):
            """
            Convert a pattern with '*' into a valid regex pattern.
            For example, 'some*pattern' becomes 'some.*pattern'.
            """
            # Escape regex special characters except for '*'
            escaped_pattern = re.escape(pattern).replace(r'\*', '.*')
            return f"^{escaped_pattern}$"

        matching_indices = []
        matching_names = []

        # Precompile patterns into regex objects
        compiled_patterns = [re.compile(compile_pattern(pattern)) for pattern in pattern_list]

        for idx, string in enumerate(original_list):
            for pattern in compiled_patterns:
                if pattern.search(string):  # Check if the compiled pattern matches the string
                    matching_indices.append(idx)
                    break  # Stop checking further patterns for this string

        for i in range(len(matching_indices)):
            matching_names.append(original_list[matching_indices[i]])

        return matching_indices, matching_names
    
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
                print(f"Dataset '{dataset_name}' not loaded. Use 'load_data' first.")
                return False

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

        return True

class LRHCMultiRunPlotter():

    def __init__(self, hdf5_file_path):
        
        self._base_path=hdf5_file_path
        self._hdf5_files, self._fnames = self.check_hdf5_files(self._base_path)
        
        self._single_run_plotters=[]
        self._single_run_datasets=[]
        self._single_run_attributes=[]

        self._n_runs=len(self._hdf5_files)

        for i in range(self._n_runs):
            dataset=self._hdf5_files[i]
            verbose=False
            self._single_run_plotters.append(LRHCPlotter(hdf5_file_path=dataset, verbose=verbose))
            self._single_run_datasets.append(self._single_run_plotters[i].list_datasets())
            self._single_run_plotters[i].load_attributes()
            self._single_run_plotters[i].load_data(dataset_names=self._single_run_datasets[i])
            self._single_run_attributes.append(list(self._single_run_plotters[i].list_attributes().keys()))
        
        self._check_all_attr_are_there()
        self._check_all_datasets_are_there()

        self._dataset_names=self._single_run_datasets[0]
        self._dataset_attributes=self._single_run_attributes[0]

        self._highlight_attr_val_differences()

        self._final_plotter=self._single_run_plotters[0] # use first plotter for plotting everything
        self.data=self._final_plotter.data
        for i in range(len(self._dataset_names)-1):
            datas=()
            for j in range(self._n_runs-1):
                dset_name=self._dataset_names[i]
                datas+=(self._single_run_plotters[j+1].data[dset_name],)

            ok=self._final_plotter.add_datas(dataset_name=dset_name, datas=datas, avrg=True) # will try to merge data across runs
            # computing the average across runs
            if not ok:
                print(f"Data merge for dataset {dset_name} failed!")
            else:
                print(f"Dataset '{dset_name}' with shape {self.data[dset_name].shape} loaded successfully.")
    
    def _highlight_attr_val_differences(self):
        
        self._attr_values_across_runs={}
        self._attr_values_are_different_across_runs={}
        self._different_attrs_across_runs=[]
        attrnames='\n'.join(self._dataset_attributes)
        # print(f"Attribute list: \n {attrnames}\n")

        self.attributes={} # only attributes which are equal

        print(f"################################\n\
            The following different attributes were found:\n")

        for i in range(len(self._dataset_attributes)): # for each attr
            attr_name=self._dataset_attributes[i]
            self._attr_values_across_runs[attr_name]=[]
            self._attr_values_are_different_across_runs[attr_name]=False
            attr_value=self._single_run_plotters[0].attributes[attr_name] # init with
            if isinstance(attr_value, np.ndarray):
                attr_value=attr_value.tolist()

            self._attr_values_across_runs[attr_name].append(attr_value)
            # first run
            for j in range(len(self._single_run_plotters)-1): # for each run
                value=self._single_run_plotters[j+1].attributes[attr_name]
                if isinstance(value, np.ndarray):
                    value=value.tolist()
                if not value==attr_value:
                    self._attr_values_are_different_across_runs[attr_name]=True
                self._attr_values_across_runs[attr_name].append(value)
            if self._attr_values_are_different_across_runs[attr_name]:
                self._different_attrs_across_runs.append(attr_name)
                print(f"{attr_name}: {self._attr_values_across_runs[attr_name]}\n")
            else:
                self.attributes[attr_name]=self._single_run_plotters[0].attributes[attr_name]

        print(f"################################")
                      
    def check_hdf5_files(self, directory):
        if not os.path.isdir(directory):
            raise ValueError(f"Error: '{directory}' is not a valid directory.")

        # Get a list of all HDF5 files in the directory
        hdf5_files = glob.glob(os.path.join(directory, "*.h5")) + glob.glob(os.path.join(directory, "*.hdf5"))
        file_names = [os.path.splitext(os.path.basename(file))[0] for file in hdf5_files]

        # Check if there are no files or just one file, raise an error
        if len(hdf5_files) < 2:
            raise ValueError("Error: Less than two HDF5 files found in the directory.")
        
        fnames_db_print='\n'.join(file_names)
        print(f"\n[LRHCMultiRunPlotter] Will load runs from datasets: \n {fnames_db_print} \n")

        return hdf5_files, file_names
    
    def _check_all_attr_are_there(self):
        
        self._all_lists_equal(self._single_run_attributes)

    def _check_all_datasets_are_there(self):

        self._all_lists_equal(self._single_run_datasets)

    def _all_lists_equal(self, lst_of_lsts):
        if not lst_of_lsts:  # Handle empty input
            raise ValueError("Error: The input list is empty.")
        
        # Convert each list to a set
        first_set = set(lst_of_lsts[0])  # Take the first list as a reference
        
        for i, lst in enumerate(lst_of_lsts[1:], start=1):  # Compare with the rest
            current_set = set(lst)
            if current_set != first_set:
                missing_in_current = first_set - current_set
                extra_in_current = current_set - first_set
                raise ValueError(
                    f"Error: List at index {i} does not match the reference list.\n"
                    f"Missing elements: {missing_in_current}\n"
                    f"Extra elements: {extra_in_current}"
                )
    
        return True  # All lists are equal

    def list_datasets(self):
        return self._final_plotter.list_datasets()

    def list_attributes(self):
        return self._final_plotter.list_attributes()
    
    def load_data(self, dataset_names, env_idx: int = None):
        pass

    def load_attributes(self):
        pass

    def get_idx_matching(self, pattern_list, original_list):

        return self._final_plotter.get_idx_matching(pattern_list,original_list)

    def plot_data(self, dataset_name,
            title="Plot",
            xaxis_dataset_name="", xlabel="Time", 
            ylabel="Intensity", 
            cmap="Blues",
            use_markers=False,
            marker_size: int = 3,
            data_labels = None,
            data_alphas = None,
            data_idxs: List[int] = None,
            distr_std = None, 
            distr_max = None, 
            distr_min = None):
        
        self._final_plotter.plot_data(dataset_name=dataset_name,
            title=title,
            xaxis_dataset_name=xaxis_dataset_name,
            ylabel=ylabel,
            cmap=cmap,
            use_markers=use_markers,
            marker_size=marker_size,
            data_labels=data_labels,
            data_alphas=data_alphas,
            data_idxs=data_idxs,
            distr_std=distr_std,
            distr_max=distr_max,
            distr_min=distr_min)
    
    def compose_datasets(self, datasets_list: List[str], name: str):
        self._final_plotter.compose_datasets(datasets_list, name)

    def show(self):
        self._final_plotter.show()
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--env_db',action='store_true', help='')
    parser.add_argument('--expl',action='store_true', help='whether to plot db data from expl env (if any)')
    parser.add_argument('--demo',action='store_true', help='whether to plot db data from demo env (if any)')
    parser.add_argument('--env_idx',type=int, help='', default=None)
    parser.add_argument('--data_path',type=str, help='full path to dataset to plot')
    parser.add_argument('--multirun',action='store_true', help='plot comparative results (if env db across envs, otherwise across runs)')
    parser.add_argument('--running_obs_stats',action='store_true', help='whether to plot running stats used for obs normalization')

    args = parser.parse_args()

    path = args.data_path

    Plotter=LRHCPlotter if not args.multirun else LRHCMultiRunPlotter
    if not args.env_db:
        # load training data
        plotter = Plotter(hdf5_file_path=path)
        datasets = plotter.list_datasets()
        attributes = plotter.list_attributes()
        plotter.load_data(dataset_names=datasets, env_idx=args.env_idx)
        plotter.load_attributes()
        print("\nDataset names:")
        print(datasets)
        print("\n")
        print("\n Run attributes:")
        print(attributes)
        print("\n")

        obs_names=list(plotter.attributes["obs_names"])
        sub_trunc_names=list(plotter.attributes["sub_trunc_names"])
        sub_term_names=list(plotter.attributes["sub_term_names"])
        sub_rew_names=list(plotter.attributes["sub_reward_names"])

        n_envs=plotter.attributes["n_envs"]
        
        # fix for n_timesteps being 0 over indexes not reached during training
        where_zero=np.where(plotter.data["n_timesteps_done"]==0)[0] # getting second index (first one
        where_zero_first=where_zero[0]
        end=plotter.data["n_timesteps_done"].shape[0]
        plotter.data["n_timesteps_done"][where_zero_first:end, :]=-1 # set to invalid val
        # should be the start )
        
        substepping_dt=plotter.attributes["substep_dt"]
        action_reps=plotter.attributes["action_repeat"]
        env_step_dsec=action_reps*substepping_dt
        total_simulated_secs=plotter.data["n_timesteps_done"]*env_step_dsec
        total_simulated_vec_secs=plotter.data["n_timesteps_done"]/n_envs
        total_simulated_h=total_simulated_secs/3600.0
        total_simulated_vec_h=total_simulated_vec_secs/3600.0
        total_simulated_d=total_simulated_h/24.0
        total_simulated_vec_d=total_simulated_vec_h/24.0
        
        plotter.create_dataset(dataset_name="total_simulated_secs", 
            data=total_simulated_secs)
        plotter.create_dataset(dataset_name="total_simulated_vec_secs", 
            data=total_simulated_vec_secs)
        plotter.create_dataset(dataset_name="total_simulated_h", 
            data=total_simulated_h)
        plotter.create_dataset(dataset_name="total_simulated_vec_h", 
            data=total_simulated_vec_h)
        plotter.create_dataset(dataset_name="total_simulated_d", 
            data=total_simulated_d)
        plotter.create_dataset(dataset_name="total_simulated_vec_d", 
            data=total_simulated_vec_d)
        
        # xlabel=xlabel
        xlabel="total_simulated_vec_h"
        # xlabel="n_timesteps_done"
        xaxis_dataset_name=xlabel
        # plot some data

        marker_size=1
        
        # losses 
        compose_ok=plotter.compose_datasets(name="qf1_losses",
            datasets_list=["qf1_loss", "qf1_loss_validation"])
        plotter.plot_data(dataset_name="qf1_losses", title="qf1 loss", 
            xaxis_dataset_name=xaxis_dataset_name,
            xlabel=xlabel,
            ylabel="bellman error",
            data_labels=["qf1 training loss", "qf1 validation loss"],
            data_alphas=[0.6, 0.2],
            use_markers=False,
            marker_size=marker_size)
        plotter.compose_datasets(name="qf2_losses",
            datasets_list=["qf2_loss", "qf2_loss_validation"])
        plotter.plot_data(dataset_name="qf2_losses", title="qf2 loss", 
            xaxis_dataset_name=xaxis_dataset_name,
            xlabel=xlabel,
            ylabel="bellman error",
            data_labels=["qf2 training loss", "qf2 validation loss"],
            data_alphas=[0.6, 0.2],
            use_markers=False,
            marker_size=marker_size)
        plotter.compose_datasets(name="actor_losses",
            datasets_list=["actor_loss", "actor_loss_validation"])
        plotter.plot_data(dataset_name="actor_losses", title="actor_loss", 
            xaxis_dataset_name=xaxis_dataset_name,
            xlabel=xlabel,
            ylabel="[]",
            data_labels=["training", "validation"],
            data_alphas=[0.6, 0.2],
            use_markers=False,
            marker_size=marker_size)
        plotter.compose_datasets(name="alpha_losses",
            datasets_list=["alpha_loss", "alpha_loss_validation"])
        plotter.plot_data(dataset_name="alpha_losses", title="alpha_loss", 
            xaxis_dataset_name=xaxis_dataset_name,
            xlabel=xlabel,
            ylabel="[]",
            data_labels=["training", "validation"],
            data_alphas=[0.6, 0.2],
            use_markers=False,
            marker_size=marker_size)
        
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
            xaxis_dataset_name=xaxis_dataset_name,
            xlabel=xlabel,
            ylabel="Q val.",
            data_labels=["qf1", "qf2"],
            use_markers=False,
            marker_size=marker_size,
            distr_std="qf_vals_std",
            distr_max="qf_vals_max",
            distr_min="qf_vals_min")

        # total reward

        # distribution
        plotter.plot_data(dataset_name="tot_rew_avrg", title="scaled returns distribution across envs", 
            xaxis_dataset_name=xaxis_dataset_name,
            xlabel=xlabel)
        plotter.plot_data(dataset_name="tot_rew_max", title="max rewards distribution across envs", 
            xaxis_dataset_name=xaxis_dataset_name,
            xlabel=xlabel)
        plotter.plot_data(dataset_name="tot_rew_min", title="min rewards distribution across envs", 
            xaxis_dataset_name=xaxis_dataset_name,
            xlabel=xlabel)

        plotter.plot_data(dataset_name="tot_rew_avrg_over_envs", title="scaled returns average over envs", 
            xaxis_dataset_name=xaxis_dataset_name,
            xlabel=xlabel,
            ylabel="",
            data_labels=["return"],
            use_markers=False,
            marker_size=marker_size,
            distr_std="tot_rew_std_over_envs",
            distr_max=None, # tot_rew_max_over_envs
            distr_min=None) # tot_rew_min_over_envs
        
        # sub rewards

        for i in range(len(sub_rew_names)):
            sub_rew_name=sub_rew_names[i]
            avrg_over_envs_name=sub_rew_name+"_avrg_over_envs"
            std_over_envs_name=sub_rew_name+"_std_over_envs"
            distr_name=sub_rew_name+"_avrg"
            distr_name_max=sub_rew_name+"_max"
            distr_name_min=sub_rew_name+"_min"
            
            plotter.create_dataset(dataset_name=distr_name,
                data=plotter.data["sub_rew_avrg"][:, :, i:i+1])
            plotter.create_dataset(dataset_name=distr_name_max,
                data=plotter.data["sub_rew_max"][:, :, i:i+1])
            plotter.create_dataset(dataset_name=distr_name_min,
                data=plotter.data["sub_rew_min"][:, :, i:i+1])
            
            plotter.create_dataset(dataset_name=avrg_over_envs_name,
                data=plotter.data["sub_rew_avrg_over_envs"][:, :, i:i+1])
            plotter.create_dataset(dataset_name=std_over_envs_name,
                data=plotter.data["sub_rew_std_over_envs"][:, :, i:i+1])
            
            # distribution over envs
            plotter.plot_data(dataset_name=distr_name, title=f"scaled sub returns ({sub_rew_name}) distribution across envs", 
            xaxis_dataset_name=xaxis_dataset_name,
            xlabel=xlabel)
            plotter.plot_data(dataset_name=distr_name_max, title=f"max rewards ({sub_rew_name}) distribution across envs", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel)
            plotter.plot_data(dataset_name=distr_name_min, title=f"min rewards ({sub_rew_name}) distribution across envs", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel)

            # average over envs
            plotter.plot_data(dataset_name=avrg_over_envs_name, title=f"scaled sub returns ({sub_rew_name}) average over envs", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                ylabel="",
                data_labels=[sub_rew_name],
                use_markers=False,
                marker_size=marker_size,
                distr_std=std_over_envs_name,
                distr_max=None, # tot_rew_max_over_envs
                distr_min=None) # tot_rew_min_over_envs
        # plotter.plot_data(dataset_name="tot_rew_avrg_over_envs", title="tot_rew_avrg_over_envs", 
        #     xaxis_dataset_name=xaxis_dataset_name,
        #     xlabel=xlabel)
        # plotter.plot_data(dataset_name="tot_rew_std_over_envs", title="tot_rew_std_over_envs", 
        #     xaxis_dataset_name=xaxis_dataset_name,
        #     xlabel=xlabel)
        # plotter.plot_data(dataset_name="tot_rew_max_over_envs", title="tot_rew_max_over_envs", 
        #     xaxis_dataset_name=xaxis_dataset_name,
        #     xlabel=xlabel)
        # plotter.plot_data(dataset_name="tot_rew_min_over_envs", title="tot_rew_min_over_envs", 
        #     xaxis_dataset_name=xaxis_dataset_name,
        #     xlabel=xlabel)

        # env data 
        plotter.plot_data(dataset_name="env_step_rt_factor", title="env_step_rt_factor", 
            xaxis_dataset_name=xaxis_dataset_name,
            xlabel=xlabel,
            use_markers=True,
            marker_size=marker_size)
        
        plotter.plot_data(dataset_name="ep_tsteps_env_distr", title="ep_tsteps_env_distribution", 
            xaxis_dataset_name=xaxis_dataset_name,
            xlabel=xlabel,
            use_markers=True,
            marker_size=marker_size)
        
        plotter.plot_data(dataset_name="SubTruncations_avrg_over_envs", title="SubTruncations_avrg_over_envs", 
            xaxis_dataset_name=xaxis_dataset_name,
            xlabel=xlabel,
            ylabel="bool",
            data_labels=sub_trunc_names,
            use_markers=True,
            marker_size=marker_size)
        plotter.plot_data(dataset_name="SubTerminations_avrg_over_envs", title="SubTerminations_avrg_over_envs", 
            xaxis_dataset_name=xaxis_dataset_name,
            xlabel=xlabel,
            ylabel="bool",
            data_labels=sub_term_names,
            use_markers=True,
            marker_size=marker_size)
        
        # rnd
        if "use_rnd" in attributes:
            if attributes["use_rnd"]:
                plotter.compose_datasets(name="expl_bonus_proc",
                    datasets_list=["expl_bonus_proc_avrg", "expl_bonus_proc_std"])
                plotter.plot_data(dataset_name="expl_bonus_proc", title="expl_bonus_proc", 
                    xaxis_dataset_name=xaxis_dataset_name,
                    xlabel=xlabel,
                    use_markers=True,
                    marker_size=marker_size,
                    data_alphas=[0.3, 0.3],
                    data_labels=["expl_bonus_proc_avrg", "expl_bonus_proc_std"])
                
                plotter.compose_datasets(name="expl_bonus_raw",
                    datasets_list=["expl_bonus_raw_avrg", "expl_bonus_raw_std"])
                plotter.plot_data(dataset_name="expl_bonus_raw", title="expl_bonus_raw", 
                    xaxis_dataset_name=xaxis_dataset_name,
                    xlabel=xlabel,
                    use_markers=True,
                    marker_size=marker_size,
                    data_alphas=[0.3, 0.3],
                    data_labels=["expl_bonus_raw_avrg", "expl_bonus_raw_std"])
                
        if args.running_obs_stats:
            # obs stats
            # gravity vecs
            patterns=["gn_*"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - gravity vec", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - gravity vec", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # joint pos
            patterns=["q_jnt_*"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - meas joint q", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - meas joint q", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # joint vel
            patterns=["v_jnt_*"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - meas joint v", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - meas joint v", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # cmd efforts
            patterns=["rhc_cmd_q_*"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - rhc cmd q", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - rhc cmd q", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # cmd efforts
            patterns=["rhc_cmd_v_*"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - rhc cmd v", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - rhc cmd v", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # cmd efforts
            patterns=["rhc_cmd_eff_*"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - rhc cmd effort", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - rhc cmd effort", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # estimated contact forces
            patterns=["fc_contact*"]
            idxs, selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - est. contact f", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - est. contact f", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # mpc fail idx
            patterns=["rhc_fail*"]
            idxs, selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - MPC fail index", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - MPC fail index", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # rhc flight info
            patterns=["flight_*"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - rhc flight info", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - rhc flight info", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # linvel
            patterns=["linvel_*_base_loc"]
            idxs, selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - linvel (meas/ref)", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - linvel (meas/ref)", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # omega
            patterns=["omega_*_base_loc"]
            idxs, selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - omega (meas/ref)", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - omega (meas/ref)", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # clock (if any)
            patterns=["clock*"]
            idxs, selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - clock", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - clock", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # actions buffer stats (if used)
            patterns=["*_prev_act"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - action buffer - prev cmds", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - action buffer - prev cmds", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            patterns=["*_avrg_act"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - action buffer - mean cmds over window", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - action buffer - mean cmds over window", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            patterns=["*_std_act"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - action buffer ", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - action buffer ", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # full actions buffer (if used)
            patterns=["*_m*_act"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - action buffer ", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - action buffer ", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
                
        plotter.show()  # Display all plots

    else:
        dset_suffix="" if not args.expl else "_expl"
        if args.demo:
            dset_suffix="_demo"

        # load env db data
        plotter = Plotter(hdf5_file_path=path)
        datasets = plotter.list_datasets()
        attributes = plotter.list_attributes()
        plotter.load_data(dataset_names=datasets, env_idx=args.env_idx)
        plotter.load_attributes()
        print("\nDataset names:")
        print(datasets)
        print("\n")
        print("\n Run attributes:")
        print(attributes)
        print("\n")

        n_eps=int(plotter.attributes["ep_vec_freq"])
        obs_names=list(plotter.attributes["Obs_data_names"])
        actions_names=list(plotter.attributes["Actions_data_names"])
        contact_forces_names=list(plotter.attributes["RhcContactForces_data_names"])
        twist_refs_names=list(plotter.attributes["AgentTwistRefs_data_names"])
        rhc_refs_names=list(plotter.attributes["RhcRefsFlag_data_names"])
        sub_term_names=list(plotter.attributes["SubTerminations_data_names"])
        sub_trunc_names=list(plotter.attributes["SubTruncations_data_names"])
        sub_reward_names=list(plotter.attributes["sub_reward_names"])
        pow_names=list(plotter.attributes["Power_data_names"])
        track_err_names=list(plotter.attributes["TrackingError_data_names"])

        xlabel="env_step"
        xaxis_dataset_name=None
        marker_size=2
        for ep_idx in range(n_eps):
            ep_prefix=f"ep_{ep_idx}_"
            
            obs_datasetname=ep_prefix+"Obs"+dset_suffix
            actions_datasetname=ep_prefix+"Actions"+dset_suffix
            # gravity vec
            patterns=["gn_*"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name=obs_datasetname, 
                title=ep_prefix+"obs - gravity vec"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            plotter.plot_data(dataset_name=obs_datasetname, 
                title=ep_prefix+"obs - gravity vec"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # joint pos
            patterns=["q_jnt_*"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name=obs_datasetname, 
                title=ep_prefix+"obs - meas joint q"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
        
            # joint vel
            patterns=["v_jnt_*"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name=obs_datasetname, 
                title=ep_prefix+"obs - meas joint v"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # cmd effort
            patterns=["rhc_cmd_q_*"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name=obs_datasetname, 
                title=ep_prefix+"obs - rhc cmd q"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # cmd effort
            patterns=["rhc_cmd_v_*"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name=obs_datasetname, 
                title=ep_prefix+"obs - rhc cmd v"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # cmd effort
            patterns=["rhc_cmd_eff_*"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name=obs_datasetname, 
                title=ep_prefix+"obs - rhc cmd effort"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # estimated contact forces
            patterns=["fc_contact*"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name=obs_datasetname, 
                title=ep_prefix+"obs - est. contact f"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # mpc fail idx
            patterns=["rhc_fail*"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name=obs_datasetname, 
                title=ep_prefix+"obs - MPC fail index"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # rhc flight info
            patterns=["flight_*"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name=obs_datasetname, 
                title=ep_prefix+"obs - rhc flight info"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # linvel
            patterns=["linvel_*_base_loc"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name=obs_datasetname, 
                title=ep_prefix+"obs - linvel (meas/ref)"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # omega
            patterns=["omega_*_base_loc"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name=obs_datasetname, 
                title=ep_prefix+"obs - omega (meas/ref)"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)

            # clock (if any)
            patterns=["clock*"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name=obs_datasetname, 
                title=ep_prefix+"clock"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # actions buffer
            patterns=["*_prev_act"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name=obs_datasetname, 
                title=ep_prefix+"obs - action buffer - prev cmds"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            patterns=["*_avrg_act"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name=obs_datasetname, 
                title=ep_prefix+"obs - action buffer - mean cmds over window"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            patterns=["*_std_act"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name=obs_datasetname, 
                title=ep_prefix+"obs - action buffer "+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)

            patterns=["*_m*_act"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name=obs_datasetname, 
                title=ep_prefix+"obs - action buffer - full action history buffer"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # actions
            plotter.plot_data(dataset_name=actions_datasetname, 
                title=ep_prefix+"actions - all"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=False,
                marker_size=marker_size,
                data_labels=actions_names,
                data_idxs=None)
            # contact actions
            patterns=["*contact_flag*"]
            idxs,selected=plotter.get_idx_matching(patterns, actions_names)
            plotter.plot_data(dataset_name=actions_datasetname, 
                title=ep_prefix+"actions - contact flag actions only"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=False,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            patterns=["*phase_freq*"]
            idxs,selected=plotter.get_idx_matching(patterns, actions_names)
            plotter.plot_data(dataset_name=actions_datasetname, 
                title=ep_prefix+"actions - step frequency only [flights/mpc_steps]"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=False,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            patterns=["*phase_offset*"]
            idxs,selected=plotter.get_idx_matching(patterns, actions_names)
            plotter.plot_data(dataset_name=actions_datasetname, 
                title=ep_prefix+"actions - step offset only [mpc_steps]"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=False,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)

            patterns=["*flight_*"]
            idxs,selected=plotter.get_idx_matching(patterns, actions_names)
            plotter.plot_data(dataset_name=actions_datasetname, 
                title=ep_prefix+"actions - flight params actions only"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=False,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)

            # sub terminations
            plotter.plot_data(dataset_name=ep_prefix+"SubTerminations"+dset_suffix, 
                title=ep_prefix+"SubTerminations"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=sub_term_names,
                data_idxs=None)
            
            
            # sub terminations
            plotter.plot_data(dataset_name=ep_prefix+"SubTruncations"+dset_suffix, 
                title=ep_prefix+"SubTruncations"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=sub_trunc_names,
                data_idxs=None)
            
            # terminations
            plotter.plot_data(dataset_name=ep_prefix+"Terminations"+dset_suffix, 
                title=ep_prefix+"Terminations"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=["is_terminal"],
                data_idxs=None)
            
            # truncations
            plotter.plot_data(dataset_name=ep_prefix+"Truncations"+dset_suffix, 
                title=ep_prefix+"Truncations"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=["is_truncated"],
                data_idxs=None)
            
            # sub rewards
            plotter.plot_data(dataset_name=ep_prefix+"sub_rew"+dset_suffix, 
                title=ep_prefix+"sub_rew"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=sub_reward_names,
                data_idxs=None)
            
            # tot rewards
            plotter.plot_data(dataset_name=ep_prefix+"tot_rew"+dset_suffix, 
                title=ep_prefix+"tot reward"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=["tot_reward"],
                data_idxs=None)

            # agent twist refs
            plotter.plot_data(dataset_name=ep_prefix+"AgentTwistRefs"+dset_suffix, 
                title=ep_prefix+"agent refs"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=twist_refs_names,
                data_idxs=None)

            # other custom data

            plotter.plot_data(dataset_name=ep_prefix+"Power"+dset_suffix, 
                title=ep_prefix+"Power"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=pow_names,
                data_idxs=None)
            
            plotter.plot_data(dataset_name=ep_prefix+"TrackingError"+dset_suffix, 
                title=ep_prefix+"TrackingError"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                ylabel="m/s",
                use_markers=True,
                marker_size=marker_size,
                data_labels=track_err_names,
                data_idxs=None)

            patterns=["*z_base_loc"]
            idxs,selected=plotter.get_idx_matching(patterns, contact_forces_names)
            plotter.plot_data(dataset_name=ep_prefix+"RhcContactForces"+dset_suffix, 
                title=ep_prefix+"Vertical MPC contact f (base loc)"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=False,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            plotter.plot_data(dataset_name=ep_prefix+"RhcFailIdx"+dset_suffix, 
                title=ep_prefix+"Rhc fail idx"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels="fail_idx",
                data_idxs=None)
            
            plotter.plot_data(dataset_name=ep_prefix+"RhcRefsFlag"+dset_suffix, 
                title=ep_prefix+"Rhc refs flags"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=rhc_refs_names,
                data_idxs=None)
             
            # plotting contact phases
            from lrhc_control.utils.data_postproc.contact_visual import ContactPlotter
            patterns=["fc_contact*z*"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            vertical_contact_f=plotter.data[obs_datasetname][:, :, idxs]
            valid_mask = np.isfinite(vertical_contact_f[:, 0, 0])
            patterns=["linvel_*_ref_base_loc"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            linvel_ref=plotter.data[obs_datasetname][:, 0, idxs][valid_mask, :]
            patterns=["linvel_x_base_loc", "linvel_y_base_loc", "linvel_z_base_loc"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            linvel_meas=plotter.data[obs_datasetname][:, 0, idxs][valid_mask, :]
            valid_f=vertical_contact_f[valid_mask, 0, :]
            is_contact=valid_f>=1e-3
            contact_state=np.full_like(valid_f, fill_value=0.0)
            contact_state[is_contact]=1.0
            contact_plotter=ContactPlotter(data=contact_state.T,
                ref_vel=linvel_ref.T,meas_vel=linvel_meas.T)
            contact_plotter.plot()

        plotter.show()

    
