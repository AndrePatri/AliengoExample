import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class ContactPlotter:
    def __init__(self, data):
        """
        Initializes the ContactPlotter with the input data and optional reference data.
        
        Args:
            data (np.ndarray): 2D array of shape (n_contacts, n_timesteps).
           io data over timesteps.
        """
        self.data = data
        self.n_contacts, self.n_timesteps = data.shape
        self.velocity_reference=None
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
        fig, axes = plt.subplots(1, 1, figsize=(10, 4))

        # Top plot: Contact phases
        # ax = axes[0]
        ax=axes
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
        ax.set_xlabel("env step")
        ax.set_title("Requested contacts over episode")
        # ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Overlay velocity reference
        if self.velocity_reference is not None:
            scaled_velocity = self.velocity_reference * (self.n_contacts - 0.5) / max(self.velocity_reference)
            ax.plot(range(self.n_timesteps), scaled_velocity, color="red", label="Velocity Reference (m/s)", alpha=1.0)
            ax.legend(loc="upper right")

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
