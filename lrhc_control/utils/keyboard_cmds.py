from lrhc_control.utils.shared_data.agent_refs import AgentRefs
from lrhc_control.utils.shared_data.training_env import Actions

from control_cluster_bridge.utilities.shared_data.rhc_data import RobotState
from control_cluster_bridge.utilities.math_utils import world2base_frame_twist

from EigenIPC.PyEigenIPCExt.wrappers.shared_data_view import SharedTWrapper
from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import Journal, LogType
from EigenIPC.PyEigenIPC import dtype

import math

import numpy as np

class AgentRefsFromKeyboard:

    def __init__(self, 
                namespace: str, 
                verbose = False,
                agent_refs_world: bool = False,
                env_idx: int = None):

        self._env_idx=env_idx

        self._verbose = verbose

        self._agent_refs_world=agent_refs_world
        
        self.namespace = namespace

        self._closed = False
        
        self.enable_navigation = False
        self.enable_omega = False
        self.enable_omega_roll = False
        self.enable_omega_pitch = False
        self.enable_omega_yaw = False

        self.enable_pos = False

        self.dpos = 0.1 # [m]
        self.dxy = 0.05 # [m/s]
        self.dvxyz = 0.05 # [m/s]
        self.dheading=0.05
        self._dtwist = 1.0 * math.pi / 180.0 # [rad]

        self._v_magnitude=0.0
        self._heading_lat=0.0
        self._heading_frontal=0.0
        self._heading=0.0
        self.agent_refs = None

        self.cluster_idx = -1
        self.cluster_idx_np = np.array(self.cluster_idx)

        self._twist_null = None

        self._init_shared_data()

    def _init_shared_data(self):
        
        self.env_index=None
        if self._env_idx is None:
            self.env_index = SharedTWrapper(namespace = self.namespace,
                    basename = "EnvSelector",
                    is_server = False, 
                    verbose = True, 
                    vlevel = VLevel.V2,
                    safe = False,
                    dtype=dtype.Int)
            
            self.env_index.run()
        
        self._init_rhc_ref_subscriber()
        
        self._current_twist_ref_world = np.full_like(self.agent_refs.rob_refs.root_state.get(data_type="twist", robot_idxs=self.cluster_idx_np), 
                fill_value=0.0).reshape(-1)
        self._current_twist_ref_base=np.full_like(self._current_twist_ref_world, fill_value=0.0).reshape(1, -1)

        self._current_pos_ref = np.full_like(self.agent_refs.rob_refs.root_state.get(data_type="p", robot_idxs=self.cluster_idx_np), 
                fill_value=0.0).reshape(-1)
        
        self._robot_state = RobotState(namespace=self.namespace,
                            is_server=False, 
                            safe=False,
                            verbose=True,
                            vlevel=VLevel.V2)
        self._robot_state.run()            

    def _init_rhc_ref_subscriber(self):

        self.agent_refs = AgentRefs(namespace=self.namespace,
                                is_server=False, 
                                safe=True, 
                                verbose=self._verbose,
                                vlevel=VLevel.V2,
                                with_gpu_mirror=False,
                                with_torch_view=False)

        self.agent_refs.run()

        self._twist_null = self.agent_refs.rob_refs.root_state.get(data_type="twist", robot_idxs=self.cluster_idx_np)
        self._twist_null[:]=0.0

        q0=np.full_like(self.agent_refs.rob_refs.root_state.get(data_type="q", robot_idxs=self.cluster_idx_np),fill_value=0.0)
        q0[0]=1.0
        self.agent_refs.rob_refs.root_state.set(data_type="q",data=q0,
                                        robot_idxs=self.cluster_idx_np)
        
    def __del__(self):

        if not self._closed:
            self._close()
    
    def _close(self):
        
        if self.agent_refs is not None:
            self.agent_refs.close()
        if self._robot_state is not None:
            self._robot_state.close()

        self._closed = True
    
    def _synch(self):
        
        if self.env_index is not None:
            self.env_index.synch_all(read=True, retry=True)
            env_index = self.env_index.get_numpy_mirror()
            self._env_idx=env_index[0, 0].item()
        self.cluster_idx = self._env_idx
        self.cluster_idx_np = self.cluster_idx            
    
    def _update_navigation(self, 
                    nav_type: str = "",
                    increment = True,
                    reset: bool = False,
                    refs_in_wframe: bool = False):
        
        current_twist_ref=self._current_twist_ref_world

        # randomizng in base frame if not refs_in_wframe, otherwise world
        if not reset:

            # xy vel (polar coordinates)
            if nav_type=="lateral" and not increment:
                if self._heading_lat>0:
                    self._heading_lat=self._heading_lat + self.dheading
                else:
                    self._heading_lat=self._heading_lat - self.dheading
                self._heading_frontal=self._heading_lat-np.pi/2
            if nav_type=="lateral" and increment:
                if self._heading_lat>0:
                    self._heading_lat=self._heading_lat - self.dheading
                else:
                    self._heading_lat=self._heading_lat + self.dheading
                self._heading_frontal=self._heading_lat-np.pi/2
            if nav_type=="frontal" and not increment:
                if self._heading_frontal>0:
                    self._heading_frontal=self._heading_frontal + self.dheading
                else:
                    self._heading_frontal=self._heading_frontal - self.dheading
                self._heading_lat=self._heading_frontal+np.pi/2
            if nav_type=="frontal" and increment:
                if self._heading_frontal>0:
                    self._heading_frontal=self._heading_frontal - self.dheading
                else:
                    self._heading_frontal=self._heading_frontal + self.dheading
                self._heading_lat=self._heading_frontal+np.pi/2
            
            if nav_type=="magnitude" and increment:
                self._v_magnitude=self._v_magnitude+self.dxy
            if nav_type=="magnitude" and not increment:
                self._v_magnitude=self._v_magnitude-self.dxy

            # vertical vel
            if nav_type=="vertical" and not increment:
                # frontal motion
                current_twist_ref[2] = current_twist_ref[2] - self.dvxyz
            if nav_type=="vertical" and increment:
                # frontal motion
                current_twist_ref[2] = current_twist_ref[2] + self.dvxyz

            # omega
            if nav_type=="twist_roll" and increment:
                # rotate counter-clockwise
                current_twist_ref[3] = current_twist_ref[3] + self._dtwist 
            if nav_type=="twist_roll" and not increment:
                current_twist_ref[3] = current_twist_ref[3] - self._dtwist 
            if nav_type=="twist_pitch" and increment:
                # rotate counter-clockwise
                current_twist_ref[4] = current_twist_ref[4] + self._dtwist 
            if nav_type=="twist_pitch" and not increment:
                current_twist_ref[4] = current_twist_ref[4] - self._dtwist 
            if nav_type=="twist_yaw" and increment:
                # rotate counter-omega_cmd
                current_twist_ref[5] = current_twist_ref[5] + self._dtwist
            if nav_type=="twist_yaw" and not increment:
                current_twist_ref[5] = current_twist_ref[5] - self._dtwist 

        else:

            if "lin" in nav_type:
                self._v_magnitude=0.0
                self._heading=0.0
                self._heading_frontal=0.0
                self._heading_lat=self._heading_frontal+np.pi/2
                
                current_twist_ref[0:3] = 0
                if self._agent_refs_world:
                    self._current_twist_ref_world[0:3]=0
            
            if "omega" in nav_type:
                current_twist_ref[3:] = 0
                if self._agent_refs_world:
                    self._current_twist_ref_world[3:]=0

        if self._heading_frontal>math.pi:
            self._heading_frontal=math.pi
        if self._heading_frontal<-math.pi:
            self._heading_frontal=-math.pi
        if self._heading_lat>math.pi:
            self._heading_lat=math.pi
        if self._heading_lat<-math.pi:
            self._heading_lat=-math.pi
        
        if self._v_magnitude<0:
            self._v_magnitude=0.0
            
        self._heading=self._heading_frontal

        current_twist_ref[0] = self._v_magnitude*np.cos(self._heading)
        current_twist_ref[1] = self._v_magnitude*np.sin(self._heading)

    def _update_pos(self, 
        nav_type: str = "",
        increment = True,
        reset: bool = False):
        
        current_pos_ref=self._current_pos_ref

        if not reset:
            # xy vel
            if nav_type=="lateral" and not increment:
                current_pos_ref[1]-=self.dpos
            if nav_type=="lateral" and increment:
                current_pos_ref[1]+=self.dpos
            if nav_type=="frontal" and not increment:
                current_pos_ref[0]-=self.dpos
            if nav_type=="frontal" and increment:
                current_pos_ref[0]+=self.dpos
            if nav_type=="vertical" and not increment:
                current_pos_ref[2]-=self.dpos
            if nav_type=="vertical" and increment:
                current_pos_ref[2]+=self.dpos
        else:
            robot_p = self._robot_state.root_state.get(data_type="p")[self.cluster_idx_np, :].reshape(-1)
            robot_p[2]=0.0
            current_pos_ref[:]=robot_p
        
    def _write_to_shared_mem(self):

        self.agent_refs.rob_refs.root_state.synch_all(read=True)
        self._robot_state.root_state.synch_all(read = True, retry = True) # read robot state        
        
        if self.enable_pos:
            robot_p = self._robot_state.root_state.get(data_type="p")[self.cluster_idx_np, :].reshape(-1)
            robot_p[2]=0.0
            # self.agent_refs.rob_refs.root_state.set(data_type="p",data=self._current_pos_ref-robot_p,
            #                                 robot_idxs=self.cluster_idx_np)
            self.agent_refs.rob_refs.root_state.set(data_type="p",data=self._current_pos_ref,
                                            robot_idxs=self.cluster_idx_np)
            self.agent_refs.rob_refs.root_state.synch_retry(row_index=self.cluster_idx, col_index=0, 
                                        n_rows=1, n_cols=3,
                                        read=False)
            
        if self.enable_navigation: # twist
            if self._agent_refs_world:
                # ref was set in world frame -> we need to move it in base frame before setting it to the agent
                robot_q = self._robot_state.root_state.get(data_type="q")[self.cluster_idx_np, :].reshape(1, -1)
                world2base_frame_twist(t_w=self._current_twist_ref_world.reshape(1, -1), 
                    q_b=robot_q, 
                    t_out=self._current_twist_ref_base)
                    
                self.agent_refs.rob_refs.root_state.set(data_type="twist",data=self._current_twist_ref_base,
                                                robot_idxs=self.cluster_idx_np)
            else:
                self._current_twist_ref_base[:, :]=self._current_twist_ref_world.reshape(1, -1)
            self.agent_refs.rob_refs.root_state.set(data_type="twist",data=self._current_twist_ref_base,
                                            robot_idxs=self.cluster_idx_np)
            self.agent_refs.rob_refs.root_state.synch_retry(row_index=self.cluster_idx, col_index=7, 
                                        n_rows=1, n_cols=6,
                                        read=False)
        # write to shared mem
           
   
    def _set_omega(self, 
                key):
        
        if key == "T":
            self.enable_omega = not self.enable_omega
            info = f"Twist change enabled: {self.enable_omega}"
            Journal.log(self.__class__.__name__,
                "_set_linvel",
                info,
                LogType.INFO,
                throw_when_excep = True)

        if not self.enable_omega:
            self._update_navigation(nav_type="omega",reset=True)

        if self.enable_omega and key == "x":
            self.enable_omega_roll = not self.enable_omega_roll
            info = f"Twist roll change enabled: {self.enable_omega_roll}"
            Journal.log(self.__class__.__name__,
                "_set_linvel",
                info,
                LogType.INFO,
                throw_when_excep = True)
        if self.enable_omega and key == "y":
            self.enable_omega_pitch = not self.enable_omega_pitch
            info = f"Twist pitch change enabled: {self.enable_omega_pitch}"
            Journal.log(self.__class__.__name__,
                "_set_linvel",
                info,
                LogType.INFO,
                throw_when_excep = True)
        if self.enable_omega and key == "z":
            self.enable_omega_yaw = not self.enable_omega_yaw
            info = f"Twist yaw change enabled: {self.enable_omega_yaw}"
            Journal.log(self.__class__.__name__,
                "_set_linvel",
                info,
                LogType.INFO,
                throw_when_excep = True)

        if key == "9":
            if self.enable_omega_roll:
                self._update_navigation(nav_type="twist_roll",
                                    increment = True)
            if self.enable_omega_pitch:
                self._update_navigation(nav_type="twist_pitch",
                                    increment = True)
            if self.enable_omega_yaw:
                self._update_navigation(nav_type="twist_yaw",
                                    increment = True)
        if key == "3":
            if self.enable_omega_roll:
                self._update_navigation(nav_type="twist_roll",
                                    increment = False)
            if self.enable_omega_pitch:
                self._update_navigation(nav_type="twist_pitch",
                                    increment = False)
            if self.enable_omega_yaw:
                self._update_navigation(nav_type="twist_yaw",
                                    increment = False)
        
    def _set_navigation(self,
                key):
        if key == "n":
            self.enable_navigation = not self.enable_navigation
            info = f"High level navigation enabled: {self.enable_navigation}"
            Journal.log(self.__class__.__name__,
                "_set_navigation",
                info,
                LogType.INFO,
                throw_when_excep = True)
        
        if not self.enable_navigation:
            self._update_navigation(nav_type="lin", reset = True)
            
        if key == "6" and self.enable_navigation:
            self._update_navigation(nav_type="lateral", 
                            increment = True,
                            refs_in_wframe=self._agent_refs_world)
        if key == "4" and self.enable_navigation:
            self._update_navigation(nav_type="lateral",
                            increment = False,
                            refs_in_wframe=self._agent_refs_world)
        if key == "8" and self.enable_navigation:
            self._update_navigation(nav_type="frontal",
                            increment = True,
                            refs_in_wframe=self._agent_refs_world)
        if key == "2" and self.enable_navigation:
            self._update_navigation(nav_type="frontal",
                            increment = False,
                            refs_in_wframe=self._agent_refs_world)
        if key == "+" and self.enable_navigation:
            self._update_navigation(nav_type="magnitude",
                            increment = True,
                            refs_in_wframe=self._agent_refs_world)
        if key == "-" and self.enable_navigation:
            self._update_navigation(nav_type="magnitude",
                            increment = False,
                            refs_in_wframe=self._agent_refs_world)
        if key == "p" and self.enable_navigation:
            self._update_navigation(nav_type="vertical",
                            increment = True,
                            refs_in_wframe=self._agent_refs_world)
        if key == "m" and self.enable_navigation:
            self._update_navigation(nav_type="vertical",
                            increment = False,
                            refs_in_wframe=self._agent_refs_world)
        
    def _set_position(self,key):

        if key == "P":
            self.enable_pos = not self.enable_pos
            info = f"High level pos reference change: {self.enable_pos}"
            Journal.log(self.__class__.__name__,
                "set_position",
                info,
                LogType.INFO,
                throw_when_excep = True)
        
        if not self.enable_pos:
            self._update_pos(reset = True)
            
        if key == "6" and self.enable_pos:
            self._update_pos(nav_type="lateral", 
                            increment = False)
        if key == "4" and self.enable_pos:
            self._update_pos(nav_type="lateral", 
                            increment = True)
        if key == "8" and self.enable_pos:
            self._update_pos(nav_type="frontal", 
                            increment = True)
        if key == "2" and self.enable_pos:
            self._update_pos(nav_type="frontal", 
                            increment = False)
        if key == "+" and self.enable_pos:
            self._update_pos(nav_type="vertical", 
                            increment = True)
        if key == "-" and self.enable_pos:
            self._update_pos(nav_type="vertical", 
                            increment = False)
            
    def _on_press(self, key):

        if not self._read_from_stdin:
            if hasattr(key, 'char'):
                key=key.char

        self._set_navigation(key)
        self._set_omega(key)
        self._set_position(key)

    def _on_release(self, key):
        
        if not self._read_from_stdin:
            if hasattr(key, 'char'):
                key=key.char
                
        # self._current_twist_ref_base[:, :]=0.0
        # nullify vel ref
        # self.agent_refs.rob_refs.root_state.set(data_type="twist",data=self._twist_null,
        #                     robot_idxs=self.cluster_idx_np)

    def run(self, read_from_stdin: bool = False,
        release_timeout: float = 0.1):

        info = f"Ready. Starting to listen for commands..."

        Journal.log(self.__class__.__name__,
            "run",
            info,
            LogType.INFO,
            throw_when_excep = True)
        
        self._update_navigation(reset=True)
        
        self._read_from_stdin=read_from_stdin
        
        import time

        self.agent_refs.run()
        
        if read_from_stdin:
            from control_cluster_bridge.utilities.keyboard_listener_stdin import KeyListenerStdin

            with KeyListenerStdin(on_press=self._on_press, 
                on_release=self._on_release, 
                release_timeout=release_timeout) as listener:
                
                while not listener.done:
                    self._synch() 
                    self._write_to_shared_mem()
                    time.sleep(0.01)  # Keep the main thread alive
                listener.stop()

        else:

            from pynput import keyboard
            listener=keyboard.Listener(on_press=self._on_press, 
                                on_release=self._on_release)
            listener.start()

            while True:
                try:
                    self._synch() 
                    self._write_to_shared_mem() # writes latest refs to shared mem
                    time.sleep(0.01)
                except KeyboardInterrupt:
                    break
            listener.stop()
            listener.join()

class AgentActionsFromKeyboard:

    def parse_contact_mapping(self, mapping_str):
        """
        Parses a string formatted like "0;1;3;2 " and extracts the numbers as a list of integers.
        Returns None if the input string is invalid.
        
        Args:
            mapping_str (str): The input string containing numbers separated by semicolons.
            
        Returns:
            List[int] or None: A list of integers extracted from the input string, or None if invalid.
        """
        try:
            # Strip any surrounding whitespace
            mapping_str = mapping_str.strip()
            
            # Split by semicolons and check for validity
            parts = mapping_str.split(';')
            if not all(part.isdigit() for part in parts if part):  # Ensure all parts are valid integers
                return None
            
            # Convert to integers and return as a list
            return [int(num) for num in parts if num]
        except Exception:
            # Catch any unexpected errors and return None
            return None
        
    def __init__(self, 
            namespace: str, 
            verbose = False,
            contact_mapping: str = "",
            env_idx: int = None):

        self._env_idx=env_idx

        self._contact_mapping=self.parse_contact_mapping(contact_mapping)
        if self._contact_mapping is None:
            self._contact_mapping=[0, 1, 2, 3] # key 7-9-1-3

        self._verbose = verbose

        self.namespace = namespace

        self.agent_actions = None

        self._twist_action = None
        self._contact_action = None

        self._init_shared_data()

        self._closed = False
        
        self.enable_heightchange = False
        self.height_dh = 0.02 # [m]

        self.enable_navigation = False
        self.enable_omega = False
        self.enable_omega_roll = False
        self.enable_omega_pitch = False
        self.enable_omega_yaw = False

        self.dvxyz = 0.05 # [m/s]
        self.domega = 1.0 * math.pi / 180.0 # [rad/s]

        self._phase_id_current=0
        self._contact_dpos=0.02
        self.enable_contact_pos_change= False
        self.enable_contact_pos_change_ci = [False]*4
        self.enable_contact_pos_change_xyz = [False]*3

        self._enable_flight_param_change=False
        self._d_flight_length=1
        self._d_flight_apex=0.01
        self._d_flight_end=0.01
        self._d_freq=0.005
        self._d_offset=0.01
        self._d_flength_enabled=False
        self._d_fapex_enabled=False
        self._d_fend_enabled=False
        self._d_freq_enabled=False
        self._d_offset_enabled=False

        self._d_fparam_enabled_contact_i = [False]*self._n_contacts

        self.contact_pos_change_vals = np.zeros((3, self._n_contacts))
        self.cluster_idx = -1
        self.cluster_idx_np = np.array(self.cluster_idx)

    def _init_shared_data(self):

        self.env_index=None
        if self._env_idx is None:
            self.env_index = SharedTWrapper(namespace = self.namespace,
                    basename = "EnvSelector",
                    is_server = False, 
                    verbose = True, 
                    vlevel = VLevel.V2,
                    safe = False,
                    dtype=dtype.Int)
            self.env_index.run()
        
        self.agent_actions = Actions(namespace=self.namespace+"_override",
                            is_server=False,
                            verbose=True,
                            vlevel=VLevel.V2,
                            safe=False)
        self.agent_actions.run()
        act_names=self.agent_actions.col_names()
        
        self.v_first, self.v_end = self.get_first_and_last_indices(act_names, "v")
        self.omega_first, self.omega_end = self.get_first_and_last_indices(act_names, "omega")
        self.contact_flag_first, self.contact_flag_end = self.get_first_and_last_indices(act_names, "contact_flag")
        self.flight_len_first, self.flight_len_end = self.get_first_and_last_indices(act_names, "flight_len")
        self.flight_apex_first, self.flight_apex_end = self.get_first_and_last_indices(act_names, "flight_apex")
        self.flight_end_first, self.flight_end_end = self.get_first_and_last_indices(act_names, "flight_end")

        self.flight_freq_first, self.flight_freq_end = self.get_first_and_last_indices(act_names, "phase_freq")
        self.phase_offset_first, self.phase_offset_end = self.get_first_and_last_indices(act_names, "phase_offset")

        self.contact_p_first, self.contact_p_end = self.get_first_and_last_indices(act_names, "contact_p")

        if self.contact_flag_end is not None:
            self._n_contacts=self.contact_flag_end-self.contact_flag_first+1
        else:
            self._n_contacts=self.flight_freq_end-self.flight_freq_first+1
        self._synch(read=True)
        # write defaults
        actions=self.agent_actions.get_numpy_mirror()
        if self.v_first is not None:
            actions[self.cluster_idx, self.v_first:self.v_end+1] = 0.0
        if self.omega_first is not None:
            actions[self.cluster_idx, self.omega_first:self.omega_end+1] = 0.0
        if self.contact_flag_first is not None:
            actions[self.cluster_idx, self.contact_flag_first:self.contact_flag_end+1]=1.0
        if self.flight_len_first is not None:
            actions[self.cluster_idx, self.flight_len_first:self.flight_len_end+1]=20.0
        if self.flight_apex_first is not None:
            actions[self.cluster_idx, self.flight_apex_first:self.flight_apex_end+1]=0.1
        if self.flight_end_first is not None:
            actions[self.cluster_idx, self.flight_end_first:self.flight_end_end+1]=0.0
        if self.flight_freq_first is not None:
            actions[self.cluster_idx, self.flight_freq_first:self.flight_freq_end+1]=0.0
        if self.phase_offset_first is not None:
            actions[self.cluster_idx, self.phase_offset_first:self.phase_offset_end+1]=0.0
        # actions[self.cluster_idx, self.phase_offset_first:self.phase_offset_first+1]=0.5
        # actions[self.cluster_idx, (self.phase_offset_first+3):self.phase_offset_first+4]=0.5

        self._synch(read=False)

    def get_first_and_last_indices(self, strings, substring):
        # Find indices of strings that contain the substring
        matching_indices = [i for i, s in enumerate(strings) if substring in s]
        # Check if there are any matches
        if not matching_indices:
            return None, None  # No match found
        # Return the first and last indices of matches
        return matching_indices[0], matching_indices[-1]

    def __del__(self):

        if not self._closed:

            self._close()
    
    def _close(self):
        
        if self.agent_actions is not None:

            self.agent_actions.close()

        self._closed = True
    
    def _synch(self, 
            read = True):
        
        if read:
            if self.env_index is not None:
                self.env_index.synch_all(read=True, retry=True)
                env_index = self.env_index.get_numpy_mirror()
                self._env_idx=env_index[0, 0].item()
            self.cluster_idx = self._env_idx
            self.cluster_idx_np = self.cluster_idx
        
            self.agent_actions.synch_all(read=True,retry=True)
        
        else:
            
            # write data just for target env
            self.agent_actions.synch_retry(row_index=self.cluster_idx, col_index=0, 
                    n_rows=1, n_cols=self.agent_actions.n_cols,
                    read=False)
    
    def _update_base_height(self, 
                decrement = False):
        current_actions=self.agent_actions.get_numpy_mirror()[self.cluster_idx_np, :]
        lin_v_cmd=current_actions[self.v_first:self.v_end+1]

        if decrement:
            lin_v_cmd[2] = lin_v_cmd[2] - self.dvxyz
        else:
            lin_v_cmd[2] = lin_v_cmd[2] + self.dvxyz

    def _update_navigation(self, 
                    type: str,
                    increment = True,
                    reset: bool=False):

        current_actions=self.agent_actions.get_numpy_mirror()[self.cluster_idx_np, :]
        lin_v_cmd=current_actions[self.v_first:self.v_end+1]
        omega_cmd=current_actions[self.omega_first:self.omega_end+1]

        if not reset:
            if type=="frontal_lin" and not increment:
                # frontal motion
                lin_v_cmd[0] = lin_v_cmd[0] - self.dvxyz
            if type=="frontal_lin" and increment:
                # frontal motion
                lin_v_cmd[0] = lin_v_cmd[0] + self.dvxyz
            if type=="lateral_lin" and increment:
                # lateral motion
                lin_v_cmd[1] = lin_v_cmd[1] - self.dvxyz
            if type=="lateral_lin" and not increment:
                # lateral motion
                lin_v_cmd[1] = lin_v_cmd[1] + self.dvxyz
            if type=="vertical_lin" and not increment:
                # frontal motion
                lin_v_cmd[2] = lin_v_cmd[2] - self.dvxyz
            if type=="vertical_lin" and increment:
                # frontal motion
                lin_v_cmd[2] = lin_v_cmd[2] + self.dvxyz
            if type=="twist_roll" and increment:
                # rotate counter-clockwise
                omega_cmd[0] = omega_cmd[0] + self.domega 
            if type=="twist_roll" and not increment:
                omega_cmd[0] = omega_cmd[0] - self.domega 
            if type=="twist_pitch" and increment:
                # rotate counter-clockwise
                omega_cmd[1] = omega_cmd[1] + self.domega 
            if type=="twist_pitch" and not increment:
                omega_cmd[1] = omega_cmd[1] - self.domega 
            if type=="twist_yaw" and increment:
                # rotate counter-omega_cmd
                omega_cmd[2] = omega_cmd[2] + self.domega 
            if type=="twist_yaw" and not increment:
                omega_cmd[2] = omega_cmd[2] - self.domega 
        else:
            if "twist" in type:
                omega_cmd[:]=0
            if "lin" in type:
                lin_v_cmd[:]=0

    def _update_flight_params(self, contact_idx: int, increment: bool = True):
        
        flight_params=self.agent_actions.get_numpy_mirror()[self.cluster_idx_np, :]
        
        if self._d_flength_enabled:
            contact_start=self.flight_len_first+contact_idx
            if increment:
                flight_params[contact_start:contact_start+1]=\
                    flight_params[contact_start:contact_start+1]+self._d_flight_length
            else:
                flight_params[contact_start:contact_start+1]=\
                    flight_params[contact_start:contact_start+1]-self._d_flight_length
            
        if self._d_fapex_enabled:
            contact_start=self.flight_apex_first+contact_idx
            if increment:
                flight_params[contact_start:contact_start+1]=\
                    flight_params[contact_start:contact_start+1]+self._d_flight_apex
            else:
                flight_params[contact_start:contact_start+1]=\
                    flight_params[contact_start:contact_start+1]-self._d_flight_apex
        
        if self._d_fend_enabled:
            contact_start=self.flight_end_first+contact_idx
            if increment:
                flight_params[contact_start:contact_start+1]=\
                    flight_params[contact_start:contact_start+1]+self._d_flight_end
            else:
                flight_params[contact_start:contact_start+1]=\
                    flight_params[contact_start:contact_start+1]-self._d_flight_end
        
        if self._d_freq_enabled:
            start=self.flight_freq_first+contact_idx
            if increment:
                flight_params[start:start+1]=\
                    flight_params[start:start+1]+self._d_freq
            else:
                flight_params[start:start+1]=\
                    flight_params[start:start+1]-self._d_freq
            if flight_params[start:start+1]>1/3.0:
                flight_params[start:start+1]=1/3.0
            if flight_params[start:start+1]<0.0:
                flight_params[start:start+1]=0.0

        if self._d_offset_enabled:
            start=self.phase_offset_first+contact_idx
            if increment:
                flight_params[start:start+1]=\
                    flight_params[start:start+1]+self._d_offset
            else:
                flight_params[start:start+1]=\
                    flight_params[start:start+1]-self._d_offset
            if flight_params[start:start+1]>1.0:
                flight_params[start:start+1]=1.0
            if flight_params[start:start+1]<0.0:
                flight_params[start:start+1]=0.0
                
    def _set_contacts(self,
                key,
                is_contact: bool = True):

        current_actions=self.agent_actions.get_numpy_mirror()[self.cluster_idx_np, :]
        if self.contact_flag_first is not None:
            contacts=current_actions[self.contact_flag_first:self.contact_flag_end+1]

            if key == "7":
                contacts[self._contact_mapping[0]] = 1 if is_contact else -1
            if key== "9":
                contacts[self._contact_mapping[1]] = 1 if is_contact else -1
            if key == "1":
                contacts[self._contact_mapping[2]] = 1 if is_contact else -1
            if key == "3":
                contacts[self._contact_mapping[3]] = 1 if is_contact else -1

    def _set_base_height(self,
                    key):

        if key == "h":
                    
            self.enable_heightchange = not self.enable_heightchange

            info = f"Base heightchange enabled: {self.enable_heightchange}"

            Journal.log(self.__class__.__name__,
                "_set_base_height",
                info,
                LogType.INFO,
                throw_when_excep = True)
        
        # if not self.enable_heightchange:
        #     self._update_base_height(reset=True)

        if key == "+" and self.enable_heightchange:
            self._update_base_height(decrement=False)
        
        if key == "-" and self.enable_heightchange:
            self._update_base_height(decrement=True)

    def _set_omega(self, 
                key):
        
        if key == "T":
            self.enable_omega = not self.enable_omega
            info = f"Twist change enabled: {self.enable_omega}"
            Journal.log(self.__class__.__name__,
                "_set_linvel",
                info,
                LogType.INFO,
                throw_when_excep = True)

        if not self.enable_omega:
            self._update_navigation(type="twist",reset=True)

        if self.enable_omega and key == "x":
            self.enable_omega_roll = not self.enable_omega_roll
            info = f"Twist roll change enabled: {self.enable_omega_roll}"
            Journal.log(self.__class__.__name__,
                "_set_linvel",
                info,
                LogType.INFO,
                throw_when_excep = True)
        if self.enable_omega and key == "y":
            self.enable_omega_pitch = not self.enable_omega_pitch
            info = f"Twist pitch change enabled: {self.enable_omega_pitch}"
            Journal.log(self.__class__.__name__,
                "_set_linvel",
                info,
                LogType.INFO,
                throw_when_excep = True)
        if self.enable_omega and key == "z":
            self.enable_omega_yaw = not self.enable_omega_yaw
            info = f"Twist yaw change enabled: {self.enable_omega_yaw}"
            Journal.log(self.__class__.__name__,
                "_set_linvel",
                info,
                LogType.INFO,
                throw_when_excep = True)

        if key == "+":
            if self.enable_omega_roll:
                self._update_navigation(type="twist_roll",
                                    increment = True)
            if self.enable_omega_pitch:
                self._update_navigation(type="twist_pitch",
                                    increment = True)
            if self.enable_omega_yaw:
                self._update_navigation(type="twist_yaw",
                                    increment = True)
        if key == "-":
            if self.enable_omega_roll:
                self._update_navigation(type="twist_roll",
                                    increment = False)
            if self.enable_omega_pitch:
                self._update_navigation(type="twist_pitch",
                                    increment = False)
            if self.enable_omega_yaw:
                self._update_navigation(type="twist_yaw",
                                    increment = False)

    def _set_linvel(self,
                key):

        if key == "n":
            self.enable_navigation = not self.enable_navigation
            info = f"Navigation enabled: {self.enable_navigation}"
            Journal.log(self.__class__.__name__,
                "_set_linvel",
                info,
                LogType.INFO,
                throw_when_excep = True)
        
        if not self.enable_navigation:
            self._update_navigation(type="lin",reset=True)

        if key == "8" and self.enable_navigation:
            self._update_navigation(type="frontal_lin",
                            increment = True)
        if key == "2" and self.enable_navigation:
            self._update_navigation(type="frontal_lin",
                            increment = False)
        if key == "6" and self.enable_navigation:
            self._update_navigation(type="lateral_lin", 
                            increment = True)
        if key == "4" and self.enable_navigation:
            self._update_navigation(type="lateral_lin",
                            increment = False)
        if key == "+" and self.enable_navigation:
            self._update_navigation(type="vertical_lin", 
                            increment = True)
        if key == "-" and self.enable_navigation:
            self._update_navigation(type="vertical_lin",
                            increment = False)

    def _set_contact_target_pos(self,
            key):

        if hasattr(key, 'char'):
            
            if key == "P":
                self.enable_contact_pos_change = not self.enable_contact_pos_change
                info = f"Contact pos change enabled: {self.enable_contact_pos_change}"
                Journal.log(self.__class__.__name__,
                    "_set_phase_id",
                    info,
                    LogType.INFO,
                    throw_when_excep = True)

                if not self.enable_contact_pos_change:
                    self.contact_pos_change_vals[:, :]=0
                    self.enable_contact_pos_change_xyz[0]=0
                    self.enable_contact_pos_change_xyz[1]=0
                    self.enable_contact_pos_change_xyz[2]=0
                
            if self.enable_contact_pos_change:
                if key == "x":
                    self.enable_contact_pos_change_xyz[0] = not self.enable_contact_pos_change_xyz[0]
                if key == "y":
                    self.enable_contact_pos_change_xyz[1] = not self.enable_contact_pos_change_xyz[1]
                if key == "z":
                    self.enable_contact_pos_change_xyz[2] = not self.enable_contact_pos_change_xyz[2]
        
                if key == "+":
                    self.contact_pos_change_vals[np.ix_(self.enable_contact_pos_change_xyz,
                        self.enable_contact_pos_change_ci)]+= self._contact_dpos
                if key == "-":
                    self.contact_pos_change_vals[np.ix_(self.enable_contact_pos_change_xyz,
                        self.enable_contact_pos_change_ci)]-= self._contact_dpos
                
                # not_enabled = [not x for x in self.enable_contact_pos_change_xyz]
                # self.contact_pos_change_vals[np.ix_(not_enabled,
                #         self.enable_contact_pos_change_ci)]= 0
                
        if key == Key.insert:
            self.enable_contact_pos_change_ci[0] = not self.enable_contact_pos_change_ci[0]
        if key == Key.page_up:
            self.enable_contact_pos_change_ci[1] = not self.enable_contact_pos_change_ci[1]
        if key == Key.delete:
            self.enable_contact_pos_change_ci[2] = not self.enable_contact_pos_change_ci[2]
        if key == Key.page_down:
            self.enable_contact_pos_change_ci[3] = not self.enable_contact_pos_change_ci[3]

        self.contact_pos_change_vals
    
    def _set_flight_params(self,
        key):

        if key=="F":
            self._enable_flight_param_change = not self._enable_flight_param_change
            info = f"Flight params change enabled: {self._enable_flight_param_change}"
            Journal.log(self.__class__.__name__,
                "_set_flight_params",
                info,
                LogType.INFO,
                throw_when_excep = True)
        
        if key=="L":
            self._d_flength_enabled=not self._d_flength_enabled
            info = f"Flight length change enabled: {self._d_flength_enabled}"
            Journal.log(self.__class__.__name__,
                "_set_flight_params",
                info,
                LogType.INFO,
                throw_when_excep = True)
        if key=="A":
            self._d_fapex_enabled=not self._d_fapex_enabled
            info = f"Flight apex change enabled: {self._d_fapex_enabled}"
            Journal.log(self.__class__.__name__,
                "_set_flight_params",
                info,
                LogType.INFO,
                throw_when_excep = True)
        if key=="E":
            self._d_fend_enabled=not self._d_fend_enabled
            info = f"Flight end change enabled: {self._d_fend_enabled}"
            Journal.log(self.__class__.__name__,
                "_set_flight_params",
                info,
                LogType.INFO,
                throw_when_excep = True)
        if key=="R":
            self._d_freq_enabled=not self._d_freq_enabled
            info = f"Step phase frequency change enabled: {self._d_freq_enabled}"
            Journal.log(self.__class__.__name__,
                "_set_flight_params",
                info,
                LogType.INFO,
                throw_when_excep = True)
        if key=="D":
            self._d_offset_enabled=not self._d_offset_enabled
            info = f"Step phase offset change enabled: {self._d_offset_enabled}"
            Journal.log(self.__class__.__name__,
                "_set_flight_params",
                info,
                LogType.INFO,
                throw_when_excep = True)
            
        if key=="o":
            self._d_fparam_enabled_contact_i[0]=not self._d_fparam_enabled_contact_i[0]
            info = f"Flight params change enabled: {self._d_fparam_enabled_contact_i[0]} for contact 0"
            Journal.log(self.__class__.__name__,
                "_set_flight_params",
                info,
                LogType.INFO,
                throw_when_excep = True)
        if key=="p":
            self._d_fparam_enabled_contact_i[1]=not self._d_fparam_enabled_contact_i[1]
            info = f"Flight params change enabled: {self._d_fparam_enabled_contact_i[1]} for contact 1"
            Journal.log(self.__class__.__name__,
                "_set_flight_params",
                info,
                LogType.INFO,
                throw_when_excep = True)
        if key=="k":
            self._d_fparam_enabled_contact_i[2]=not self._d_fparam_enabled_contact_i[2]
            info = f"Flight params change enabled: {self._d_fparam_enabled_contact_i[2]} for contact 2"
            Journal.log(self.__class__.__name__,
                "_set_flight_params",
                info,
                LogType.INFO,
                throw_when_excep = True)
        if key=="l":
            self._d_fparam_enabled_contact_i[3]=not self._d_fparam_enabled_contact_i[3]
            info = f"Flight params change enabled: {self._d_fparam_enabled_contact_i[3]} for contact 3"
            Journal.log(self.__class__.__name__,
                "_set_flight_params",
                info,
                LogType.INFO,
                throw_when_excep = True)
            
        if key=="+":
            if self._d_fparam_enabled_contact_i[0]:
                self._update_flight_params(contact_idx=0,
                    increment=True)
            if self._d_fparam_enabled_contact_i[1]:
                self._update_flight_params(contact_idx=1,
                    increment=True)
            if self._d_fparam_enabled_contact_i[2]:
                self._update_flight_params(contact_idx=2,
                    increment=True)
            if self._d_fparam_enabled_contact_i[3]:
                self._update_flight_params(contact_idx=3,
                    increment=True)
        if key=="-":
            if self._d_fparam_enabled_contact_i[0]:
                self._update_flight_params(contact_idx=0,
                    increment=False)
            if self._d_fparam_enabled_contact_i[1]:
                self._update_flight_params(contact_idx=1,
                    increment=False)
            if self._d_fparam_enabled_contact_i[2]:
                self._update_flight_params(contact_idx=2,
                    increment=False)
            if self._d_fparam_enabled_contact_i[3]:
                self._update_flight_params(contact_idx=3,
                    increment=False)
                
    def _on_press(self, key):
            
        self._synch(read=True) # updates data like
        # current cluster index

        if not self._read_from_stdin:
            if hasattr(key, 'char'):
                key=key.char
                # phase ids

        self._set_contacts(key=key, 
                    is_contact=False)
        # height change
        self._set_base_height(key)
        # (linear) navigation cmds
        self._set_linvel(key)
        # orientation (twist)
        self._set_omega(key)

        self._set_flight_params(key)

        self._synch(read=False)

    def _on_release(self, key):
        
        if not self._read_from_stdin:
            if hasattr(key, 'char'):
                key=key.char
                # phase ids
        self._set_contacts(key, is_contact=True)

        self._synch(read=False)

    def run(self, read_from_stdin: bool = False,
        release_timeout: float = 0.1):

        info = f"Ready. Starting to listen for commands..."

        Journal.log(self.__class__.__name__,
            "run",
            info,
            LogType.INFO,
            throw_when_excep = True)
        
        self._update_navigation(reset=True,type="lin")
        self._update_navigation(reset=True,type="twist")

        self._read_from_stdin=read_from_stdin
        if self._read_from_stdin:
            from control_cluster_bridge.utilities.keyboard_listener_stdin import KeyListenerStdin
            import time

            with KeyListenerStdin(on_press=self._on_press, 
                on_release=self._on_release, 
                release_timeout=release_timeout) as listener:
                
                while not listener.done:
                    time.sleep(0.1)  # Keep the main thread alive
                listener.stop()

        else:
            from pynput import keyboard

            with keyboard.Listener(on_press=self._on_press, 
                                on_release=self._on_release) as listener:

                listener.join()

if __name__ == "__main__":  

    keyb_cmds = AgentRefsFromKeyboard(namespace="kyon0", 
                            verbose=True)

    keyb_cmds.run()