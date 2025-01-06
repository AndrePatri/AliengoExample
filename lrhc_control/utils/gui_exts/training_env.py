from PyQt5.QtWidgets import QWidget

from control_cluster_bridge.utilities.debugger_gui.gui_exts import SharedDataWindow
from control_cluster_bridge.utilities.debugger_gui.plot_utils import RtPlotWindow
from control_cluster_bridge.utilities.debugger_gui.plot_utils import WidgetUtils

from EigenIPC.PyEigenIPC import VLevel

from lrhc_control.utils.shared_data.training_env import SharedTrainingEnvInfo
from lrhc_control.utils.shared_data.agent_refs import AgentRefs
from lrhc_control.utils.shared_data.training_env import Observations
from lrhc_control.utils.shared_data.training_env import Actions
from lrhc_control.utils.shared_data.training_env import SubRewards
from lrhc_control.utils.shared_data.training_env import TotRewards
from lrhc_control.utils.shared_data.training_env import Terminations, SubTerminations
from lrhc_control.utils.shared_data.training_env import Truncations, SubTruncations
from lrhc_control.utils.shared_data.training_env import EpisodesCounter
from lrhc_control.utils.shared_data.algo_infos import SharedRLAlgorithmInfo, QfVal, QfTrgt
from lrhc_control.utils.shared_data.training_env import SubReturns, TotReturns

import numpy as np

class TrainingEnvData(SharedDataWindow):

    def __init__(self, 
        update_data_dt: int,
        update_plot_dt: int,
        window_duration: int,
        window_buffer_factor: int = 2,
        namespace = "",
        parent: QWidget = None, 
        verbose = False,
        add_settings_tab = False,
        ):
        
        name = "TrainingEnvData"

        self.current_cluster_index = 0

        super().__init__(update_data_dt = update_data_dt,
            update_plot_dt = update_plot_dt,
            window_duration = window_duration,
            window_buffer_factor = window_buffer_factor,
            grid_n_rows = 1,
            grid_n_cols = 1,
            namespace = namespace,
            name = name,
            parent = parent, 
            verbose = verbose,
            add_settings_tab = add_settings_tab,
            )

    def _init_shared_data(self):
        
        is_server = False
        
        self.shared_data_clients.append(SharedTrainingEnvInfo(is_server=is_server,
                                            namespace=self.namespace, 
                                            verbose=True, 
                                            vlevel=VLevel.V2))
        
        self.shared_data_clients.append(SharedRLAlgorithmInfo(is_server=is_server,
                                            namespace=self.namespace, 
                                            verbose=True, 
                                            vlevel=VLevel.V2))
        
        self.shared_data_clients.append(Observations(namespace=self.namespace,
                                            is_server=False,
                                            verbose=True,
                                            vlevel=VLevel.V2,
                                            safe=False,
                                            with_gpu_mirror=False))
        
        self.shared_data_clients.append(Actions(namespace=self.namespace,
                                            is_server=False,
                                            verbose=True,
                                            vlevel=VLevel.V2,
                                            safe=False,
                                            with_gpu_mirror=False))
        
        self.shared_data_clients.append(SubRewards(namespace=self.namespace,
                                            is_server=False,
                                            verbose=True,
                                            vlevel=VLevel.V2,
                                            safe=False,
                                            with_gpu_mirror=False))
        
        self.shared_data_clients.append(TotRewards(namespace=self.namespace,
                                            is_server=False,
                                            verbose=True,
                                            vlevel=VLevel.V2,
                                            safe=False,
                                            with_gpu_mirror=False))

        self.shared_data_clients.append(Truncations(namespace=self.namespace,
                                            is_server=False,
                                            verbose=True,
                                            vlevel=VLevel.V2,
                                            safe=False,
                                            with_gpu_mirror=False))

        self.shared_data_clients.append(Terminations(namespace=self.namespace,
                                            is_server=False,
                                            verbose=True,
                                            vlevel=VLevel.V2,
                                            safe=False,
                                            with_gpu_mirror=False))

        self.shared_data_clients.append(SubTruncations(namespace=self.namespace,
                                            is_server=False,
                                            verbose=True,
                                            vlevel=VLevel.V2,
                                            safe=False,
                                            with_gpu_mirror=False))

        self.shared_data_clients.append(SubTerminations(namespace=self.namespace,
                                            is_server=False,
                                            verbose=True,
                                            vlevel=VLevel.V2,
                                            safe=False,
                                            with_gpu_mirror=False))
        
        self.shared_data_clients.append(EpisodesCounter(namespace=self.namespace,
                                            is_server=False,
                                            verbose=True,
                                            vlevel=VLevel.V2,
                                            safe=False,
                                            with_gpu_mirror=False))
        
        self.shared_data_clients.append(AgentRefs(namespace=self.namespace,
                                is_server=False,
                                with_gpu_mirror=False,
                                safe=False,
                                verbose=True,
                                vlevel=VLevel.V2))
        
        self.shared_data_clients.append(QfVal(namespace=self.namespace,
                                is_server=False,
                                safe=False,
                                verbose=True,
                                vlevel=VLevel.V2))
        
        self.shared_data_clients.append(QfTrgt(namespace=self.namespace,
                                is_server=False,
                                safe=False,
                                verbose=True,
                                vlevel=VLevel.V2))
        
        self.shared_data_clients.append(SubReturns(namespace=self.namespace,
                                            is_server=False,
                                            verbose=True,
                                            vlevel=VLevel.V2,
                                            safe=False,
                                            with_gpu_mirror=False))
        
        self.shared_data_clients.append(TotReturns(namespace=self.namespace,
                                            is_server=False,
                                            verbose=True,
                                            vlevel=VLevel.V2,
                                            safe=False,
                                            with_gpu_mirror=False))

        for client in self.shared_data_clients:
            client.run()

    def _post_shared_init(self):
        
        self.grid_n_rows = 9

        self.grid_n_cols = 2

    def _initialize(self):
        
        n_envs = self.shared_data_clients[2].n_rows

        self.rt_plotters.append(RtPlotWindow(data_dim=len(self.shared_data_clients[0].param_keys),
                                    n_data = 1,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Training env. info", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=self.shared_data_clients[0].param_keys, 
                                    ylabel=""))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=len(self.shared_data_clients[1].param_keys),
                                    n_data = 1,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"RL algorithm info", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=self.shared_data_clients[1].param_keys, 
                                    ylabel=""))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=self.shared_data_clients[2].n_cols,
                                    n_data = n_envs,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Observations", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=self.shared_data_clients[2].col_names(), 
                                    ylabel="[float]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=self.shared_data_clients[3].n_cols,
                                    n_data = n_envs,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Actions", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=self.shared_data_clients[3].col_names(), 
                                    ylabel="[float]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=len(self.shared_data_clients[4].col_names()),
                                    n_data = n_envs,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Rewards - detailed", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=self.shared_data_clients[4].col_names(), 
                                    ylabel="[float]"))

        self.rt_plotters.append(RtPlotWindow(data_dim=len(self.shared_data_clients[5].col_names()),
                                    n_data = n_envs,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Rewards", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=self.shared_data_clients[5].col_names(), 
                                    ylabel="[float]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=1,
                                    n_data = n_envs,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Truncations", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=[""], 
                                    ylabel="[bool]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=1,
                                    n_data = n_envs,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Terminations", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=[""], 
                                    ylabel="[bool]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=len(self.shared_data_clients[8].col_names()),
                                    n_data = n_envs,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"SubTruncations", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=self.shared_data_clients[8].col_names(),
                                    ylabel="[bool]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=len(self.shared_data_clients[9].col_names()),
                                    n_data = n_envs,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"SubTerminations", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=self.shared_data_clients[9].col_names(),
                                    ylabel="[bool]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=1,
                                    n_data = n_envs,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Episode counter", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=[""], 
                                    ylabel="[int]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=2,
                n_data = n_envs,
                update_data_dt=self.update_data_dt, 
                update_plot_dt=self.update_plot_dt,
                window_duration=self.window_duration, 
                parent=None, 
                base_name=f"Q value VS Q value target", 
                window_buffer_factor=self.window_buffer_factor, 
                legend_list=["qf value", "qf target"], 
                ylabel="[float]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=len(self.shared_data_clients[4].col_names()),
                                    n_data = n_envs,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Episodic returns - detailed (undiscounted)", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=self.shared_data_clients[4].col_names(), 
                                    ylabel="[float]"))

        self.rt_plotters.append(RtPlotWindow(data_dim=len(self.shared_data_clients[5].col_names()),
                                    n_data = n_envs,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Episodic returns - total (undiscounted)", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=self.shared_data_clients[5].col_names(), 
                                    ylabel="[float]"))
        
        # agent refs 
        self.rt_plotters.append(RtPlotWindow(data_dim=self.shared_data_clients[11].rob_refs.root_state.get(data_type="p").shape[1],
                    n_data = 1,
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Agent refs - root position", 
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=["p_x", "p_y", "p_z"], 
                    ylabel="[m]"))
        
        self.rt_plotters.append(RtPlotWindow(
                    data_dim=self.shared_data_clients[11].rob_refs.root_state.get(data_type="q").shape[1],
                    n_data = 1,
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Agent refs - root orientation", 
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=["q_w", "q_i", "q_j", "q_k"]))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=self.shared_data_clients[11].rob_refs.root_state.get(data_type="v").shape[1],
                    n_data = 1,
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt, 
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Agent refs - base linear vel.", 
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=["v_x", "v_y", "v_z"], 
                    ylabel="[m/s]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=self.shared_data_clients[11].rob_refs.root_state.get(data_type="omega").shape[1],
                    n_data = 1, 
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt, 
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Agent refs - base angular vel.",
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=["omega_x", "omega_y", "omega_z"], 
                    ylabel="[rad/s]"))
        
        self.grid.addFrame(self.rt_plotters[0].base_frame, 0, 0)
        self.grid.addFrame(self.rt_plotters[1].base_frame, 0, 1)
        self.grid.addFrame(self.rt_plotters[2].base_frame, 1, 0)
        self.grid.addFrame(self.rt_plotters[3].base_frame, 1, 1)
        self.grid.addFrame(self.rt_plotters[4].base_frame, 2, 0)
        self.grid.addFrame(self.rt_plotters[5].base_frame, 2, 1)
        self.grid.addFrame(self.rt_plotters[6].base_frame, 3, 0)
        self.grid.addFrame(self.rt_plotters[7].base_frame, 3, 1)
        self.grid.addFrame(self.rt_plotters[8].base_frame, 4, 0)
        self.grid.addFrame(self.rt_plotters[9].base_frame, 4, 1)
        self.grid.addFrame(self.rt_plotters[10].base_frame, 5, 0)
        self.grid.addFrame(self.rt_plotters[11].base_frame, 5, 1)
        self.grid.addFrame(self.rt_plotters[12].base_frame, 6, 0)
        self.grid.addFrame(self.rt_plotters[13].base_frame, 6, 1)

        # agent refs
        self.grid.addFrame(self.rt_plotters[14].base_frame, 7, 0)
        self.grid.addFrame(self.rt_plotters[15].base_frame, 7, 1)
        self.grid.addFrame(self.rt_plotters[16].base_frame, 8, 0)
        self.grid.addFrame(self.rt_plotters[17].base_frame, 8, 1)

    def _finalize_grid(self):
                
        widget_utils = WidgetUtils()

        settings_frames = []

        n_envs = self.shared_data_clients[2].n_rows

        node_index_slider = widget_utils.generate_complex_slider(
                        parent=None, 
                        parent_layout=None,
                        min_shown=f"{0}", min= 0, 
                        max_shown=f"{n_envs - 1}", 
                        max=n_envs - 1, 
                        init_val_shown=f"{0}", init=0, 
                        title="cluster index", 
                        callback=self._update_cluster_idx)
        
        settings_frames.append(node_index_slider)
        
        self.grid.addToSettings(settings_frames)

    def _update_cluster_idx(self,
                    idx: int):

        self.current_cluster_index = idx

        self.grid.settings_widget_list[0].current_val.setText(f'{idx}')

        for i in range(2, len(self.rt_plotters)):
            # only switching plots which have n_cluster dim
            self.rt_plotters[i].rt_plot_widget.switch_to_data(idx = self.current_cluster_index)

    def update(self,
            index: int):

        # index not used here (no dependency on cluster index)

        if not self._terminated:
            
            np_idx = np.array(index)

            # read data on shared memory
            env_data = self.shared_data_clients[0].get().flatten()
            algo_data = self.shared_data_clients[1].get().flatten()

            self.shared_data_clients[2].synch_all(read=True, retry=True) # obs
            self.shared_data_clients[3].synch_all(read=True, retry=True) # actions
            self.shared_data_clients[4].synch_all(read=True, retry=True) # sub rewards
            self.shared_data_clients[5].synch_all(read=True, retry=True) # tot reward
            self.shared_data_clients[6].synch_all(read=True, retry=True) # truncatons
            self.shared_data_clients[7].synch_all(read=True, retry=True) # terminations
            self.shared_data_clients[8].synch_all(read=True, retry=True) # subtruncatons
            self.shared_data_clients[9].synch_all(read=True, retry=True) # subterminations
            self.shared_data_clients[10].counter().synch_all(read=True, retry=True) # episodes timeout counter

            self.shared_data_clients[12].synch_all(read=True, retry=True) # q fun
            self.shared_data_clients[13].synch_all(read=True, retry=True) # q target
            
            self.shared_data_clients[14].synch_all(read=True, retry=True) # sub return
            self.shared_data_clients[15].synch_all(read=True, retry=True) # tot return 

            self.shared_data_clients[11].rob_refs.root_state.synch_all(read=True, retry=True) # agent refs

            # update plots
            self.rt_plotters[0].rt_plot_widget.update(env_data)
            self.rt_plotters[1].rt_plot_widget.update(algo_data)
            self.rt_plotters[2].rt_plot_widget.update(np.transpose(self.shared_data_clients[2].get_numpy_mirror()))
            self.rt_plotters[3].rt_plot_widget.update(np.transpose(self.shared_data_clients[3].get_numpy_mirror()))
            self.rt_plotters[4].rt_plot_widget.update(np.transpose(self.shared_data_clients[4].get_numpy_mirror()))
            self.rt_plotters[5].rt_plot_widget.update(np.transpose(self.shared_data_clients[5].get_numpy_mirror()))
            self.rt_plotters[6].rt_plot_widget.update(np.transpose(self.shared_data_clients[6].get_numpy_mirror()))
            self.rt_plotters[7].rt_plot_widget.update(np.transpose(self.shared_data_clients[7].get_numpy_mirror()))
            self.rt_plotters[8].rt_plot_widget.update(np.transpose(self.shared_data_clients[8].get_numpy_mirror()))
            self.rt_plotters[9].rt_plot_widget.update(np.transpose(self.shared_data_clients[9].get_numpy_mirror()))
            self.rt_plotters[10].rt_plot_widget.update(np.transpose(self.shared_data_clients[10].counter().get_numpy_mirror()))
            qf_val=self.shared_data_clients[12].get_numpy_mirror()
            qf_trgt=self.shared_data_clients[13].get_numpy_mirror()
            q_data_cat=np.concatenate((qf_val, qf_trgt), axis=1)
            self.rt_plotters[11].rt_plot_widget.update(np.transpose(q_data_cat))

            qf_val=self.shared_data_clients[12].get_numpy_mirror()
            qf_trgt=self.shared_data_clients[13].get_numpy_mirror()

            self.rt_plotters[12].rt_plot_widget.update(np.transpose(self.shared_data_clients[14].get_numpy_mirror()))
            self.rt_plotters[13].rt_plot_widget.update(np.transpose(self.shared_data_clients[15].get_numpy_mirror()))

            # agent refs
            self.rt_plotters[14].rt_plot_widget.update(self.shared_data_clients[11].rob_refs.root_state.get(data_type="p", robot_idxs=np_idx).flatten())
            self.rt_plotters[15].rt_plot_widget.update(self.shared_data_clients[11].rob_refs.root_state.get(data_type="q", robot_idxs=np_idx).flatten())
            self.rt_plotters[16].rt_plot_widget.update(self.shared_data_clients[11].rob_refs.root_state.get(data_type="v", robot_idxs=np_idx).flatten())
            self.rt_plotters[17].rt_plot_widget.update(self.shared_data_clients[11].rob_refs.root_state.get(data_type="omega", robot_idxs=np_idx).flatten())

