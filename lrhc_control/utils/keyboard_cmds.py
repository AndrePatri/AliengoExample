from pynput import keyboard

from lrhc_control.utils.shared_data.agent_refs import AgentRefs
from lrhc_control.utils.shared_data.training_env import Actions

from SharsorIPCpp.PySharsor.wrappers.shared_data_view import SharedTWrapper
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import Journal, LogType
from SharsorIPCpp.PySharsorIPC import dtype

import math

import numpy as np

class AgentRefsFromKeyboard:

    def __init__(self, 
                namespace: str, 
                verbose = False):

        self._verbose = verbose

        self.namespace = namespace

        self._closed = False
        
        self.enable_navigation = False

        self.dxy = 0.05 # [m]
        self._dtwist = 1.0 * math.pi / 180.0 # [rad]

        self.agent_refs = None

        self.cluster_idx = -1
        self.cluster_idx_np = np.array(self.cluster_idx)

        self._twist_null = None

        self._init_shared_data()

    def _init_shared_data(self):

        self.env_index = SharedTWrapper(namespace = self.namespace,
                basename = "EnvSelector",
                is_server = False, 
                verbose = True, 
                vlevel = VLevel.V2,
                safe = False,
                dtype=dtype.Int)
        
        self.env_index.run()
        
        self._init_rhc_ref_subscriber()

    def _init_rhc_ref_subscriber(self):

        self.agent_refs = AgentRefs(namespace=self.namespace,
                                is_server=False, 
                                safe=False, 
                                verbose=self._verbose,
                                vlevel=VLevel.V2,
                                with_gpu_mirror=False)

        self.agent_refs.run()

        self._twist_null = self.agent_refs.rob_refs.root_state.get(data_type="twist", robot_idxs=self.cluster_idx_np)
        self._twist_null.zero_()

    def __del__(self):

        if not self._closed:

            self._close()
    
    def _close(self):
        
        if self.agent_refs is not None:

            self.agent_refs.close()

        self._closed = True
    
    def _synch(self, 
            read = True):
        
        if read:

            self.env_index.synch_all(read=True, retry=True)
            env_index = self.env_index.get_numpy_mirror()
            self.cluster_idx = env_index[0, 0].item()
            self.cluster_idx_np = self.cluster_idx
        
            self.agent_refs.rob_refs.synch_from_shared_mem()
        
        else:
            
            self.agent_refs.rob_refs.root_state.synch_retry(row_index=self.cluster_idx, col_index=0, 
                                                n_rows=1, n_cols=self.agent_refs.rob_refs.root_state.n_cols,
                                                read=False)
    
    def _update_navigation(self, 
                    type: str = "",
                    increment = True,
                    reset: bool = False):

        current_lin_v_ref = self.agent_refs.rob_refs.root_state.get(data_type="v", robot_idxs=self.cluster_idx_np)
        current_omega_ref = self.agent_refs.rob_refs.root_state.get(data_type="omega", robot_idxs=self.cluster_idx_np)

        if not reset:
            if type=="lateral_lin" and increment:
                # lateral motion
                current_lin_v_ref[1] = current_lin_v_ref[1] - self.dxy
            if type=="lateral_lin" and not increment:
                # lateral motion
                current_lin_v_ref[1] = current_lin_v_ref[1] + self.dxy
            if type=="frontal_lin" and not increment:
                # frontal motion
                current_lin_v_ref[0] = current_lin_v_ref[0] - self.dxy
            if type=="frontal_lin" and increment:
                # frontal motion
                current_lin_v_ref[0] = current_lin_v_ref[0] + self.dxy
            if type=="twist_roll" and increment:
                # rotate counter-clockwise
                current_omega_ref[0] = current_omega_ref[0] + self._dtwist 
            if type=="twist_roll" and not increment:
                current_omega_ref[0] = current_omega_ref[0] - self._dtwist 
            if type=="twist_pitch" and increment:
                # rotate counter-clockwise
                current_omega_ref[1] = current_omega_ref[1] + self._dtwist 
            if type=="twist_pitch" and not increment:
                current_omega_ref[1] = current_omega_ref[1] - self._dtwist 
            if type=="twist_yaw" and increment:
                # rotate counter-clockwise
                current_omega_ref[2] = current_omega_ref[2] + self._dtwist 
            if type=="twist_yaw" and not increment:
                current_omega_ref[2] = current_omega_ref[2] - self._dtwist 
        else:
            current_omega_ref[:] = 0
            current_lin_v_ref[:] = 0

        self.agent_refs.rob_refs.root_state.set(data_type="v",data=current_lin_v_ref,
                                    robot_idxs=self.cluster_idx_np)
        self.agent_refs.rob_refs.root_state.set(data_type="omega",data=current_omega_ref,
                                    robot_idxs=self.cluster_idx_np)

    def _set_navigation(self,
                key):
        if key.char == "n":
            self.enable_navigation = not self.enable_navigation
            info = f"High level navigation enabled: {self.enable_navigation}"
            Journal.log(self.__class__.__name__,
                "_set_navigation",
                info,
                LogType.INFO,
                throw_when_excep = True)
        
        if not self.enable_navigation:
            self._update_navigation(reset = True)
            
        if key.char == "6" and self.enable_navigation:
            self._update_navigation(type="lateral_lin", 
                            increment = True)
        if key.char == "4" and self.enable_navigation:
            self._update_navigation(type="lateral_lin",
                            increment = False)
        if key.char == "8" and self.enable_navigation:
            self._update_navigation(type="frontal_lin",
                            increment = True)
        if key.char == "2" and self.enable_navigation:
            self._update_navigation(type="frontal_lin",
                            increment = False)
                
    def _on_press(self, key):
            
        self._synch(read=True) # updates  data like
        # current cluster index

        if hasattr(key, 'char'):
            self._set_navigation(key)

        self._synch(read=False)

    def _on_release(self, key):
            
        if hasattr(key, 'char'):
            
            # nullify vel ref
            self.agent_refs.rob_refs.root_state.set(data_type="twist",data=self._twist_null,
                                robot_idxs=self.cluster_idx_np)

            if key == keyboard.Key.esc:

                self._close()

        self._synch(read=False)

    def run(self):

        info = f"Ready. Starting to listen for commands..."

        Journal.log(self.__class__.__name__,
            "run",
            info,
            LogType.INFO,
            throw_when_excep = True)
        
        self._update_navigation(reset=True)
    
        with keyboard.Listener(on_press=self._on_press, 
                               on_release=self._on_release) as listener:

            listener.join()

class AgentActionsFromKeyboard:

    def __init__(self, 
                namespace: str, 
                verbose = False):

        self._verbose = verbose

        self.namespace = namespace

        self._closed = False
        
        self.enable_heightchange = False
        self.height_dh = 0.02 # [m]

        self.enable_navigation = False
        self.enable_twist = False
        self.enable_twist_roll = False
        self.enable_twist_pitch = False
        self.enable_twist_yaw = False

        self.dvxyz = 0.05 # [m]
        self.domega = 1.0 * math.pi / 180.0 # [rad]

        self._phase_id_current=0
        self._contact_dpos=0.02
        self.enable_contact_pos_change= False
        self.enable_contact_pos_change_ci = [False]*4
        self.enable_contact_pos_change_xyz = [False]*3
        n_contacts=4
        self.contact_pos_change_vals = np.zeros((3, n_contacts))
        self.cluster_idx = -1
        self.cluster_idx_np = np.array(self.cluster_idx)

        self.agent_actions = None

        self.cluster_idx = -1
        self.cluster_idx_np = np.array(self.cluster_idx)

        self._twist_action = None
        self._contact_action = None

        self._init_shared_data()

    def _init_shared_data(self):

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
        self.contact_first, self.contact_end = self.get_first_and_last_indices(act_names, "contact")
        self.contact_p_first, self.contact_p_end = self.get_first_and_last_indices(act_names, "contact_p")

        self._synch(read=True)
        # write defaults
        actions=self.agent_actions.get_numpy_mirror()
        if self.v_first is not None:
            actions[self.cluster_idx, self.v_first:self.v_end+1] = 0.0
        if self.omega_first is not None:
            actions[self.cluster_idx, self.omega_first:self.omega_end+1] = 0.0
        if self.contact_first is not None:
            actions[self.cluster_idx, self.contact_first:self.contact_end+1]=1.0
        if self.contact_p_first is not None:
            actions[self.cluster_idx, self.contact_p_first:self.contact_p_end+1]=0.0
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

            self.env_index.synch_all(read=True, retry=True)
            env_index = self.env_index.get_numpy_mirror()
            self.cluster_idx = env_index[0, 0].item()
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
                omega_cmd[0] = omega_cmd[0] + self._domega 
            if type=="twist_roll" and not increment:
                omega_cmd[0] = omega_cmd[0] - self._domega 
            if type=="twist_pitch" and increment:
                # rotate counter-clockwise
                omega_cmd[1] = omega_cmd[1] + self._domega 
            if type=="twist_pitch" and not increment:
                omega_cmd[1] = omega_cmd[1] - self._domega 
            if type=="twist_yaw" and increment:
                # rotate counter-omega_cmd
                omega_cmd[2] = omega_cmd[2] + self._domega 
            if type=="twist_yaw" and not increment:
                omega_cmd[2] = omega_cmd[2] - self._domega 
        else:
            if "twist" in type:
                omega_cmd[:]=0
            if "lin" in type:
                lin_v_cmd[:]=0

    def _set_contacts(self,
                key,
                is_contact: bool = True):

        current_actions=self.agent_actions.get_numpy_mirror()[self.cluster_idx_np, :]
        contacts=current_actions[self.contact_first:self.contact_end+1]

        if key.char == "7":
            contacts[0] = 1 if is_contact else -1
        if key.char == "9":
            contacts[1] = 1 if is_contact else -1
        if key.char == "1":
            contacts[2] = 1 if is_contact else -1
        if key.char == "3":
            contacts[3] = 1 if is_contact else -1

    def _set_base_height(self,
                    key):

        if key.char == "h":
                    
            self.enable_heightchange = not self.enable_heightchange

            info = f"Base heightchange enabled: {self.enable_heightchange}"

            Journal.log(self.__class__.__name__,
                "_set_base_height",
                info,
                LogType.INFO,
                throw_when_excep = True)
        
        # if not self.enable_heightchange:
        #     self._update_base_height(reset=True)

        if key.char == "+" and self.enable_heightchange:
            self._update_base_height(decrement=False)
        
        if key.char == "-" and self.enable_heightchange:
            self._update_base_height(decrement=True)

    def _set_omega(self, 
                key):
        
        if key.char == "T":
            self.enable_twist = not self.enable_twist
            info = f"Twist change enabled: {self.enable_twist}"
            Journal.log(self.__class__.__name__,
                "_set_linvel",
                info,
                LogType.INFO,
                throw_when_excep = True)

        if not self.enable_twist:
            self._update_navigation(type="twist",reset=True)

        if self.enable_twist and key.char == "x":
            self.enable_twist_roll = not self.enable_twist_roll
            info = f"Twist roll change enabled: {self.enable_twist_roll}"
            Journal.log(self.__class__.__name__,
                "_set_linvel",
                info,
                LogType.INFO,
                throw_when_excep = True)
        if self.enable_twist and key.char == "y":
            self.enable_twist_pitch = not self.enable_twist_pitch
            info = f"Twist pitch change enabled: {self.enable_twist_pitch}"
            Journal.log(self.__class__.__name__,
                "_set_linvel",
                info,
                LogType.INFO,
                throw_when_excep = True)
        if self.enable_twist and key.char == "z":
            self.enable_twist_yaw = not self.enable_twist_yaw
            info = f"Twist yaw change enabled: {self.enable_twist_yaw}"
            Journal.log(self.__class__.__name__,
                "_set_linvel",
                info,
                LogType.INFO,
                throw_when_excep = True)

        if key.char == "+":
            if self.enable_twist_roll:
                self._update_navigation(type="twist_roll",
                                    increment = True)
            if self.enable_twist_pitch:
                self._update_navigation(type="twist_pitch",
                                    increment = True)
            if self.enable_twist_yaw:
                self._update_navigation(type="twist_yaw",
                                    increment = True)
        if key.char == "-":
            if self.enable_twist_roll:
                self._update_navigation(type="twist_roll",
                                    increment = False)
            if self.enable_twist_pitch:
                self._update_navigation(type="twist_pitch",
                                    increment = False)
            if self.enable_twist_yaw:
                self._update_navigation(type="twist_yaw",
                                    increment = False)

    def _set_linvel(self,
                key):

        if key.char == "n":
            self.enable_navigation = not self.enable_navigation
            info = f"Navigation enabled: {self.enable_navigation}"
            Journal.log(self.__class__.__name__,
                "_set_linvel",
                info,
                LogType.INFO,
                throw_when_excep = True)
        
        if not self.enable_navigation:
            self._update_navigation(type="lin",reset=True)

        if key.char == "8" and self.enable_navigation:
            self._update_navigation(type="frontal_lin",
                            increment = True)
        if key.char == "2" and self.enable_navigation:
            self._update_navigation(type="frontal_lin",
                            increment = False)
        if key.char == "6" and self.enable_navigation:
            self._update_navigation(type="lateral_lin", 
                            increment = True)
        if key.char == "4" and self.enable_navigation:
            self._update_navigation(type="lateral_lin",
                            increment = False)
        if key.char == "+" and self.enable_navigation:
            self._update_navigation(type="vertical_lin", 
                            increment = True)
        if key.char == "-" and self.enable_navigation:
            self._update_navigation(type="vertical_lin",
                            increment = False)

    def _set_contact_target_pos(self,
            key):

        if hasattr(key, 'char'):
            
            if key.char == "P":
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
                if key.char == "x":
                    self.enable_contact_pos_change_xyz[0] = not self.enable_contact_pos_change_xyz[0]
                if key.char == "y":
                    self.enable_contact_pos_change_xyz[1] = not self.enable_contact_pos_change_xyz[1]
                if key.char == "z":
                    self.enable_contact_pos_change_xyz[2] = not self.enable_contact_pos_change_xyz[2]
        
                if key.char == "+":
                    self.contact_pos_change_vals[np.ix_(self.enable_contact_pos_change_xyz,
                        self.enable_contact_pos_change_ci)]+= self._contact_dpos
                if key.char == "-":
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

    def _on_press(self, key):
            
        self._synch(read=True) # updates data like
        # current cluster index

        if hasattr(key, 'char'):
            self._set_contacts(key=key, 
                        is_contact=False)
            # height change
            self._set_base_height(key)
            # (linear) navigation cmds
            self._set_linvel(key)
            # orientation (twist)
            self._set_omega(key)

        self._synch(read=False)

    def _on_release(self, key):
            
        if hasattr(key, 'char'):
            self._set_contacts(key, is_contact=True)
            if key == keyboard.Key.esc:
                self._close()

        self._synch(read=False)

    def run(self):

        info = f"Ready. Starting to listen for commands..."

        Journal.log(self.__class__.__name__,
            "run",
            info,
            LogType.INFO,
            throw_when_excep = True)
        
        self._update_navigation(reset=True,type="lin")
        self._update_navigation(reset=True,type="twist")

        with keyboard.Listener(on_press=self._on_press, 
                               on_release=self._on_release) as listener:

            listener.join()

if __name__ == "__main__":  

    keyb_cmds = AgentRefsFromKeyboard(namespace="kyon0", 
                            verbose=True)

    keyb_cmds.run()



class RefsFromKeyboard:

    def __init__(self, 
        namespace: str,
        shared_refs, 
        verbose = False):

        self._verbose = verbose

        self.namespace = namespace

        self._shared_refs=shared_refs
        self._closed = False
        
        self.enable_heightchange = False
        self.height_dh = 0.02 # [m]

        self.enable_navigation = False
        self.enable_twist = False
        self.enable_twist_roll = False
        self.enable_twist_pitch = False
        self.enable_twist_yaw = False

        self.dxy = 0.05 # [m]
        self._dtwist = 1.0 * math.pi / 180.0 # [rad]

        self.enable_phase_id_change = False
        self._phase_id_current=0
        self._contact_dpos=0.02
        self.enable_contact_pos_change= False
        self.enable_contact_pos_change_ci = [False]*4
        self.enable_contact_pos_change_xyz = [False]*3
        n_contacts=4
        self.contact_pos_change_vals = np.zeros((3, n_contacts))
        self.cluster_idx = -1
        self.cluster_idx_np = np.array(self.cluster_idx)

        self._init_shared_data()

    def _init_shared_data(self):

        self.launch_keyboard_cmds = SharedTWrapper(namespace = self.namespace,
                basename = "KeyboardCmdsLauncher",
                is_server = False, 
                verbose = True, 
                vlevel = VLevel.V2,
                safe = False,
                dtype=dtype.Bool)
        
        self.launch_keyboard_cmds.run()

        self.env_index = SharedTWrapper(namespace = self.namespace,
                basename = "EnvSelector",
                is_server = False, 
                verbose = True, 
                vlevel = VLevel.V2,
                safe = False,
                dtype=dtype.Int)
        
        self.env_index.run()
        
        self._init_ref_subscriber()

    def _init_ref_subscriber(self):
        
        if not self._shared_refs.is_running():
            self._shared_refs.run()
        
        self._contact_names=self._shared_refs.rob_refs.contact_pos.contact_names

    def __del__(self):

        if not self._closed:

            self._close()
    
    def _close(self):
        
        if self._shared_refs is not None:

            self._shared_refs.close()

        self._closed = True
    
    def _synch(self, 
            read = True):
        
        if read:

            self.env_index.synch_all(read=True, retry=True)
            env_index = self.env_index.get_numpy_mirror()
            self.cluster_idx = env_index[0, 0].item()
            self.cluster_idx_np = self.cluster_idx
        
            self._shared_refs.rob_refs.synch_from_shared_mem()
            self._shared_refs.contact_flags.synch_all(read=True, retry=True)
            self._shared_refs.phase_id.synch_all(read=True, retry=True)
        
        else:
            
            self._shared_refs.rob_refs.root_state.synch_retry(row_index=self.cluster_idx, col_index=0, 
                                                n_rows=1, n_cols=self._shared_refs.rob_refs.root_state.n_cols,
                                                read=False)
            self._shared_refs.rob_refs.contact_pos.synch_retry(row_index=self.cluster_idx, col_index=0, 
                                                n_rows=1, n_cols=self._shared_refs.rob_refs.contact_pos.n_cols,
                                                read=False)
            
            self._shared_refs.contact_flags.synch_retry(row_index=self.cluster_idx, col_index=0, 
                                                n_rows=1, n_cols=self._shared_refs.contact_flags.n_cols,
                                                read=False)
            
            self._shared_refs.phase_id.synch_retry(row_index=self.cluster_idx, col_index=0, 
                                                n_rows=1, n_cols=self._shared_refs.phase_id.n_cols,
                                                read=False)
                                                
    def _update_base_height(self, 
                decrement = False):
        
        # update both base height and vel
        current_p_ref = self._shared_refs.rob_refs.root_state.get(data_type="p", robot_idxs=self.cluster_idx_np)
        if decrement:
            new_height_ref = current_p_ref[2] - self.height_dh
        else:
            new_height_ref = current_p_ref[2] + self.height_dh
        current_p_ref[2] = new_height_ref
        self._shared_refs.rob_refs.root_state.set(data_type="p",data=current_p_ref,
                                    robot_idxs=self.cluster_idx_np)
    
    def _update_navigation(self, 
                    type: str,
                    increment = True,
                    reset: bool=False):

        current_lin_v_ref = self._shared_refs.rob_refs.root_state.get(data_type="v", robot_idxs=self.cluster_idx_np)
        current_omega_ref = self._shared_refs.rob_refs.root_state.get(data_type="omega", robot_idxs=self.cluster_idx_np)

        if not reset:
            if type=="frontal_lin" and not increment:
                # frontal motion
                current_lin_v_ref[0] = current_lin_v_ref[0] - self.dxy
            if type=="frontal_lin" and increment:
                # frontal motion
                current_lin_v_ref[0] = current_lin_v_ref[0] + self.dxy
            if type=="lateral_lin" and increment:
                # lateral motion
                current_lin_v_ref[1] = current_lin_v_ref[1] - self.dxy
            if type=="lateral_lin" and not increment:
                # lateral motion
                current_lin_v_ref[1] = current_lin_v_ref[1] + self.dxy
            if type=="vertical_lin" and not increment:
                # frontal motion
                current_lin_v_ref[2] = current_lin_v_ref[2] - self.dxy
            if type=="vertical_lin" and increment:
                # frontal motion
                current_lin_v_ref[2] = current_lin_v_ref[2] + self.dxy
            if type=="twist_roll" and increment:
                # rotate counter-clockwise
                current_omega_ref[0] = current_omega_ref[0] + self._dtwist 
            if type=="twist_roll" and not increment:
                current_omega_ref[0] = current_omega_ref[0] - self._dtwist 
            if type=="twist_pitch" and increment:
                # rotate counter-clockwise
                current_omega_ref[1] = current_omega_ref[1] + self._dtwist 
            if type=="twist_pitch" and not increment:
                current_omega_ref[1] = current_omega_ref[1] - self._dtwist 
            if type=="twist_yaw" and increment:
                # rotate counter-clockwise
                current_omega_ref[2] = current_omega_ref[2] + self._dtwist 
            if type=="twist_yaw" and not increment:
                current_omega_ref[2] = current_omega_ref[2] - self._dtwist 
        else:
            if "twist" in type:
                current_omega_ref[:]=0
            if "lin" in type:
                current_lin_v_ref[:]=0

        self._shared_refs.rob_refs.root_state.set(data_type="v",data=current_lin_v_ref,
                                    robot_idxs=self.cluster_idx_np)
        self._shared_refs.rob_refs.root_state.set(data_type="omega",data=current_omega_ref,
                                    robot_idxs=self.cluster_idx_np)

    def _update_phase_id(self,
                phase_id: int = -1):

        phase_id_shared = self._shared_refs.phase_id.get_numpy_mirror()
        phase_id_shared[self.cluster_idx, :] = phase_id
        self._phase_id_current=phase_id

    def _set_contacts(self,
                key,
                is_contact: bool = True):
        contact_flags = self._shared_refs.contact_flags.get_numpy_mirror()
        if key.char == "7":
            contact_flags[self.cluster_idx, 0] = is_contact
        if key.char == "9":
            contact_flags[self.cluster_idx, 1] = is_contact
        if key.char == "1":
            contact_flags[self.cluster_idx, 2] = is_contact
        if key.char == "3":
            contact_flags[self.cluster_idx, 3] = is_contact
    
    def _set_phase_id(self,
                    key):

        if key.char == "p":
                    
            self.enable_phase_id_change = not self.enable_phase_id_change

            info = f"Phase ID change enabled: {self.enable_phase_id_change}"

            Journal.log(self.__class__.__name__,
                "_set_phase_id",
                info,
                LogType.INFO,
                throw_when_excep = True)
            
        if key.char == "0" and self.enable_phase_id_change:
            
            self._update_phase_id(phase_id = 0)

        elif key.char == "1" and self.enable_phase_id_change:

            self._update_phase_id(phase_id = 1)
        
        elif key.char == "2" and self.enable_phase_id_change:

            self._update_phase_id(phase_id = 2)
        
        elif key.char == "3" and self.enable_phase_id_change:

            self._update_phase_id(phase_id = 3)
        
        elif key.char == "4" and self.enable_phase_id_change:

            self._update_phase_id(phase_id = 4)
        
        elif key.char == "5" and self.enable_phase_id_change:

            self._update_phase_id(phase_id = 5)

        elif key.char == "6" and self.enable_phase_id_change:

            self._update_phase_id(phase_id = 6)

        elif key.char == "r" and self.enable_phase_id_change:
        
            self._update_phase_id(phase_id = -1)

    def _set_base_height(self,
                    key):

        if key.char == "h":
                    
            self.enable_heightchange = not self.enable_heightchange

            info = f"Base heightchange enabled: {self.enable_heightchange}"

            Journal.log(self.__class__.__name__,
                "_set_base_height",
                info,
                LogType.INFO,
                throw_when_excep = True)
        
        # if not self.enable_heightchange:
        #     self._update_base_height(reset=True)

        if key.char == "+" and self.enable_heightchange:
            self._update_base_height(decrement=False)
        
        if key.char == "-" and self.enable_heightchange:
            self._update_base_height(decrement=True)

    def _set_twist(self, 
                key):
        
        if key.char == "T":
            self.enable_twist = not self.enable_twist
            info = f"Twist change enabled: {self.enable_twist}"
            Journal.log(self.__class__.__name__,
                "_set_linvel",
                info,
                LogType.INFO,
                throw_when_excep = True)

        if not self.enable_twist:
            self._update_navigation(type="twist",reset=True)

        if self.enable_twist and key.char == "x":
            self.enable_twist_roll = not self.enable_twist_roll
            info = f"Twist roll change enabled: {self.enable_twist_roll}"
            Journal.log(self.__class__.__name__,
                "_set_linvel",
                info,
                LogType.INFO,
                throw_when_excep = True)
        if self.enable_twist and key.char == "y":
            self.enable_twist_pitch = not self.enable_twist_pitch
            info = f"Twist pitch change enabled: {self.enable_twist_pitch}"
            Journal.log(self.__class__.__name__,
                "_set_linvel",
                info,
                LogType.INFO,
                throw_when_excep = True)
        if self.enable_twist and key.char == "z":
            self.enable_twist_yaw = not self.enable_twist_yaw
            info = f"Twist yaw change enabled: {self.enable_twist_yaw}"
            Journal.log(self.__class__.__name__,
                "_set_linvel",
                info,
                LogType.INFO,
                throw_when_excep = True)

        if key.char == "+":
            if self.enable_twist_roll:
                self._update_navigation(type="twist_roll",
                                    increment = True)
            if self.enable_twist_pitch:
                self._update_navigation(type="twist_pitch",
                                    increment = True)
            if self.enable_twist_yaw:
                self._update_navigation(type="twist_yaw",
                                    increment = True)
        if key.char == "-":
            if self.enable_twist_roll:
                self._update_navigation(type="twist_roll",
                                    increment = False)
            if self.enable_twist_pitch:
                self._update_navigation(type="twist_pitch",
                                    increment = False)
            if self.enable_twist_yaw:
                self._update_navigation(type="twist_yaw",
                                    increment = False)
            
    def _set_linvel(self,
                key):

        if key.char == "n":
            self.enable_navigation = not self.enable_navigation
            info = f"Navigation enabled: {self.enable_navigation}"
            Journal.log(self.__class__.__name__,
                "_set_linvel",
                info,
                LogType.INFO,
                throw_when_excep = True)
        
        if not self.enable_navigation:
            self._update_navigation(type="lin",reset=True)

        if key.char == "8" and self.enable_navigation:
            self._update_navigation(type="frontal_lin",
                            increment = True)
        if key.char == "2" and self.enable_navigation:
            self._update_navigation(type="frontal_lin",
                            increment = False)
        if key.char == "6" and self.enable_navigation:
            self._update_navigation(type="lateral_lin", 
                            increment = True)
        if key.char == "4" and self.enable_navigation:
            self._update_navigation(type="lateral_lin",
                            increment = False)
        if key.char == "+" and self.enable_navigation:
            self._update_navigation(type="vertical_lin", 
                            increment = True)
        if key.char == "-" and self.enable_navigation:
            self._update_navigation(type="vertical_lin",
                            increment = False)
    
    def _set_contact_target_pos(self,
            key):

        if hasattr(key, 'char'):
            
            if key.char == "P":
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
                if key.char == "x":
                    self.enable_contact_pos_change_xyz[0] = not self.enable_contact_pos_change_xyz[0]
                if key.char == "y":
                    self.enable_contact_pos_change_xyz[1] = not self.enable_contact_pos_change_xyz[1]
                if key.char == "z":
                    self.enable_contact_pos_change_xyz[2] = not self.enable_contact_pos_change_xyz[2]
        
                if key.char == "+":
                    self.contact_pos_change_vals[np.ix_(self.enable_contact_pos_change_xyz,
                        self.enable_contact_pos_change_ci)]+= self._contact_dpos
                if key.char == "-":
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
                
        current_contact_pos_trgt = self._shared_refs.rob_refs.contact_pos.get(data_type="p", 
                                            robot_idxs=self.cluster_idx_np)
        current_contact_pos_trgt[:] = self.contact_pos_change_vals.flatten()

        self.contact_pos_change_vals

    def _on_press(self, key):
        if self.launch_keyboard_cmds.read_retry(row_index=0,
                                            col_index=0)[0]:
            
            self._synch(read=True) # updates  data like
            # current cluster index

            if hasattr(key, 'char'):
                # phase ids
                self._set_phase_id(key)
                # stepping phases (if phase id allows it)
                self._set_contacts(key=key, 
                            is_contact=False)
                # height change
                self._set_base_height(key)
                # (linear) navigation cmds
                self._set_linvel(key)
                # orientation (twist)
                self._set_twist(key)

            self._set_contact_target_pos(key)

            self._synch(read=False)

    def _on_release(self, key):

        if self.launch_keyboard_cmds.read_retry(row_index=0,
                                            col_index=0)[0]:
            
            if hasattr(key, 'char'):
                
                # print('Key {0} released.'.format(key.char))

                self._set_contacts(key=key, 
                            is_contact=True)

                if key == keyboard.Key.esc:

                    self._close()

            self._synch(read=False)

    def run(self):

        info = f"Ready. Starting to listen for commands..."

        Journal.log(self.__class__.__name__,
            "run",
            info,
            LogType.INFO,
            throw_when_excep = True)
        
        self._update_navigation(reset=True,type="lin")
        self._update_navigation(reset=True,type="twist")

        with keyboard.Listener(on_press=self._on_press, 
                    on_release=self._on_release) as listener:

            listener.join()