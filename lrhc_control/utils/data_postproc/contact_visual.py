import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class ContactPlotter:
    def __init__(self, data: np.ndarray, ref_vel: np.ndarray = None, meas_vel: np.ndarray = None):
        """
        Initializes the ContactPlotter with the input data and optional reference data.
        
        Args:
            data (np.ndarray): 2D array of shape (n_contacts, n_timesteps).
           io data over timesteps.
        """
        self.data = data
        self.n_contacts, self.n_timesteps = data.shape
        
        self.ref_vel=ref_vel
        self.meas_vel=meas_vel

        self.ref_vel_norm=np.linalg.norm(self.ref_vel, axis=0, keepdims=False).reshape(-1)
        self.meas_vel_norm=np.linalg.norm(self.meas_vel, axis=0, keepdims=False).reshape(-1)

        self._heading_ref=self.compute_heading(self.ref_vel).reshape(-1)

        self._heading_meas=self.compute_heading(self.meas_vel).reshape(-1)

    def detect_gait(self):
        """
        Detects gait based on contact patterns.

        Returns:
            list: Detected gait type for each timestep.
        """
        gait_labels = []
        for t in range(self.n_timesteps):
            contacts = self.data[:, t]
            if np.sum(contacts) == 4:
                gait_labels.append("Standing")
            elif np.sum(contacts) == 3:
                gait_labels.append("Walking")
            elif np.sum(contacts) == 2:
                gait_labels.append("Trotting")
            elif np.sum(contacts) == 1:
                gait_labels.append("Bounding")
            else:
                gait_labels.append("Flight")
        return gait_labels

    def compute_heading(self,velocity: np.ndarray) -> np.ndarray:
        """
        Computes the heading angle of the projection of a velocity vector 
        onto the xy-plane, ensuring the range is [-pi, pi).
        
        Parameters:
        velocity (np.ndarray): A 3 x n_samples array, where each column is a velocity vector [vx, vy, vz].
        
        Returns:
        np.ndarray: A 1D array of heading angles in radians, in the range [-pi, pi).
        """
        vx = velocity[0, :]
        vy = velocity[1, :]
        
        theta = np.arctan2(vy, vx)  # Range: [-pi, pi]
        
        # Ensure the range is [-pi, pi) by mapping pi to -pi
        return theta  # Smooth out discontinuities

        # return np.unwrap(theta)  # Smooth out discontinuities

    def plot(self):
        """
        Creates the contact plot with gait annotations, velocity reference, and additional data.
        """
        # Detect gait types and prepare colors
        gait_types = self.detect_gait()
        unique_gaits = list(set(gait_types))
        gait_to_color = {gait: idx for idx, gait in enumerate(unique_gaits)}
        gait_colors = [gait_to_color[gait] for gait in gait_types]

        # Custom colormap for swing ratios (lower subplot)
        clist=["#2B79D1", "#c4456d", "#972b90", "#b67d3a", "#FFA500"]
        cmap = LinearSegmentedColormap.from_list(
            "swing_ratio", clist, N=256
        )

        # Create the plot
        fig, axes = plt.subplots(4, 1, figsize=(17, 8))

        # Top plot: Contact phases
        ax = axes[0]
        # ax=axes
        for i, row in enumerate(self.data):
            for j, val in enumerate(row):
                if val > 0:  # Contact
                    ax.add_patch(plt.Rectangle((j, i), 1, 0.8, color=clist[i], alpha=0.3))
        
        ax.set_xlim(0, self.n_timesteps)
        ax.set_ylim(0, self.n_contacts)
        ax.set_yticks(range(self.n_contacts))
        ax.set_yticklabels([f"Contact {i+1}" for i in range(self.n_contacts)])
        # ax.set_xticks(np.linspace(0, self.n_timesteps, 6))
        # ax.set_xticklabels([f"{t:.1f}s" for t in np.linspace(0, self.n_timesteps / 10, 6)])
        ax.set_title("Requested contacts over episode")
        # ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Overlay velocity reference
        if self.ref_vel is not None:
            import matplotlib.lines as mlines

            ax = axes[1]
            ax.plot(range(self.n_timesteps), self.ref_vel_norm, color="blue", alpha=0.8)
            ax.plot(range(self.n_timesteps), self.meas_vel_norm, color="red", alpha=0.3)
            ax.set_ylabel("[m/s]")
            labels=["ref. velocity norm", "meas. velocity norm"]
            ax.set_xlim(0, self.n_timesteps)
            legend_lines = [mlines.Line2D([0], [0], color=ax.get_lines()[i].get_color(), lw=3) for i in range(len(ax.get_lines()))]
            legend = ax.legend(legend_lines, labels, ncol=1, handlelength=2, loc="upper right")
            # Set pickable property
            for line in legend_lines:
                line.set_picker(True)
            # legend = ax.legend(ncol=2, markerscale=2)

            legend.set_draggable(True)  # Make the legend draggable
            ax.grid()
            
            ax = axes[2]
            ax.plot(range(self.n_timesteps), self._heading_ref, '-', color="blue",linewidth=2, alpha=0.8, markersize=3)
            ax.plot(range(self.n_timesteps), self._heading_meas, 'o', color="red",linewidth=2, alpha=0.3, markersize=3)
            # Add horizontal dashed lines at -π and π
            ax.axhline(y=-np.pi, color='black', linestyle='dashed', linewidth=2, alpha=0.3)
            ax.axhline(y=np.pi, color='black', linestyle='dashed', linewidth=2, alpha=0.3)
            ax.set_ylabel("[rad]")
            ax.set_yticks([-np.pi, 0, np.pi])  # Define y-axis tick positions
            ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])  # Use LaTeX-style π notation
            labels=["ref. heading", "meas. heading"]
            ax.set_xlim(0, self.n_timesteps)

            legend_lines = [mlines.Line2D([0], [0], color=ax.get_lines()[i].get_color(), lw=3) for i in range(len(ax.get_lines()))]
            legend = ax.legend(legend_lines, labels, ncol=1, handlelength=2, loc="upper right")
            # Set pickable property
            for line in legend_lines:
                line.set_picker(True)
            # legend = ax.legend(ncol=2, markerscale=2)

            legend.set_draggable(True)  # Make the legend draggable
            ax.grid()

            ax = axes[3]
            ax.plot(range(self.n_timesteps), self.ref_vel[0, :], '-', color="blue", alpha=0.8, markersize=3, linewidth=2)
            ax.plot(range(self.n_timesteps), self.ref_vel[1, :], '-', color="red", alpha=0.8, markersize=3, linewidth=2)
            ax.plot(range(self.n_timesteps), self.ref_vel[2, :], '-', color="green", alpha=0.8, markersize=3, linewidth=2)
            ax.set_ylabel("[m/s]")
            labels=["vx ref", "vy ref", "vz ref"]
            ax.set_xlim(0, self.n_timesteps)
            
            legend_lines = [mlines.Line2D([0], [0], color=ax.get_lines()[i].get_color(), lw=3) for i in range(len(ax.get_lines()))]
            legend = ax.legend(legend_lines, labels, ncol=1, handlelength=2, loc="upper right")
            # Set pickable property
            for line in legend_lines:
                line.set_picker(True)
            # legend = ax.legend(ncol=2, markerscale=2)

            legend.set_draggable(True)  # Make the legend draggable
            ax.grid()

            # ax = axes[4]
            # ax.plot(range(self.n_timesteps), self.meas_vel[0, :], 'o', color="blue", alpha=0.8, markersize=3)
            # ax.plot(range(self.n_timesteps), self.meas_vel[1, :], 'o', color="red", alpha=0.8, markersize=3)
            # ax.plot(range(self.n_timesteps), self.meas_vel[2, :], 'o', color="green", alpha=0.8, markersize=3)
            # ax.set_ylabel("[m/s]")
            # labels=["vx meas", "vy meas", "vz meas"]

            # legend_lines = [mlines.Line2D([0], [0], color=ax.get_lines()[i].get_color(), lw=3) for i in range(len(ax.get_lines()))]
            # legend = ax.legend(legend_lines, labels, ncol=1, handlelength=2, loc="upper right")
            # # Set pickable property
            # for line in legend_lines:
            #     line.set_picker(True)
            # # legend = ax.legend(ncol=2, markerscale=2)

            # legend.set_draggable(True)  # Make the legend draggable
            # ax.grid()

        ax.set_xlabel("env steps")
        # Annotate gait types
        # for i, gait in enumerate(unique_gaits):
        #     x_positions = [t for t in range(self.n_timesteps) if gait_types[t] == gait]
        #     if x_positions:
        #         x_start, x_end = min(x_positions), max(x_positions)
        #         ax.annotate(
        #             gait,
        #             xy=((x_start + x_end) / 2, -0.5),
        #             xytext=(0, -15),
        #             textcoords="offset points",
        #             ha="center",
        #             va="top",
        #             fontsize=8,
        #             color="black",
        #             arrowprops=dict(arrowstyle="|-|", color="black", lw=0.5),
        #         )
       
        # # Final adjustments
        # plt.tight_layout()
        # plt.show()

# Test Case
if __name__ == "__main__":
    # Example data
    n_contacts = 4
    n_timesteps = 100
    data = np.random.choice([0, 1], size=(n_contacts, n_timesteps), p=[0.5, 0.5])
    velocity_reference = np.linspace(0.5, 2.5, n_timesteps)
    stepping_frequency = np.linspace(1, 4, n_timesteps)

    # Create and use the ContactPlotter
    plotter = ContactPlotter(data)
    plotter.plot()
