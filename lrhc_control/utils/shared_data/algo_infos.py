from EigenIPC.PyEigenIPCExt.wrappers.shared_data_view import SharedTWrapper
from EigenIPC.PyEigenIPC import StringTensorServer, StringTensorClient
from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import dtype as eigenipc_dtype, toNumpyDType
from EigenIPC.PyEigenIPC import Journal
from EigenIPC.PyEigenIPC import LogType

from control_cluster_bridge.utilities.shared_data.abstractions import SharedDataBase

from lrhc_control.utils.shared_data.base_data import NamedSharedTWrapper

from typing import Dict, Union, List

import numpy as np
import torch

# Training env info
class QfVal(SharedTWrapper):
                 
    def __init__(self,
        namespace = "",
        is_server = False, 
        n_envs: int = -1, 
        verbose: bool = False, 
        vlevel: VLevel = VLevel.V0,
        force_reconnection: bool = False,
        safe: bool = True):

        basename = "QFValues" 

        super().__init__(namespace = namespace,
            basename = basename,
            is_server = is_server, 
            n_rows = n_envs, 
            n_cols = 1, 
            verbose = verbose, 
            vlevel = vlevel,
            dtype=eigenipc_dtype.Float,
            fill_value=np.nan,
            safe = safe,
            force_reconnection=force_reconnection,
            with_gpu_mirror=False,
            with_torch_view=True)

class QfTrgt(SharedTWrapper):
                 
    def __init__(self,
        namespace = "",
        is_server = False, 
        n_envs: int = -1, 
        verbose: bool = False, 
        vlevel: VLevel = VLevel.V0,
        force_reconnection: bool = False,
        safe: bool = True):

        basename = "QFTrgt" 

        super().__init__(namespace = namespace,
            basename = basename,
            is_server = is_server, 
            n_rows = n_envs, 
            n_cols = 1, 
            verbose = verbose, 
            vlevel = vlevel,
            dtype=eigenipc_dtype.Float,
            fill_value=np.nan,
            safe = safe,
            force_reconnection=force_reconnection,
            with_gpu_mirror=False,
            with_torch_view=True)
        
class RLAlgorithmDebData(SharedTWrapper):
                 
    def __init__(self,
        namespace = "",
        is_server = False, 
        n_dims: int = -1, 
        verbose: bool = False, 
        vlevel: VLevel = VLevel.V0,
        force_reconnection: bool = False,
        safe: bool = True):

        basename = "RLAlgorithmDebData" 

        super().__init__(namespace = namespace,
            basename = basename,
            is_server = is_server, 
            n_rows = n_dims, 
            n_cols = 1, 
            verbose = verbose, 
            vlevel = vlevel,
            dtype=eigenipc_dtype.Float,
            fill_value=np.nan,
            safe = safe,
            force_reconnection=force_reconnection)

class DynamicRLAlgorithmNames:

    def __init__(self):

        self._keys = ["n_of_timesteps_done",
                "n_of_policy_updates", 
                "exp_to_policy_improv_ratio",
                "elapsed_hours",
                "estimated_remaining_hours",
                "env_step_sps",
                "env_step_rt_factor",
                "time_for_exp_collection",
                "policy_update_fps",
                "time_for_pol_updates",
                "is_done",
                "n_played_episodes",
                "time_for_running_stats_updates"
                ]
        self.idx_dict = dict.fromkeys(self._keys, None)

        # dynamic sim info is by convention
        # put at the start
        for i in range(len(self._keys)):
            
            self.idx_dict[self._keys[i]] = i

    def get(self):

        return self._keys

    def get_idx(self, name: str):

        return self.idx_dict[name]
    
class SharedRLAlgorithmInfo(SharedDataBase):
                           
    def __init__(self, 
                namespace: str,
                is_server = False, 
                static_params: Dict = None,
                verbose = True, 
                vlevel = VLevel.V2, 
                safe: bool = True,
                force_reconnection: bool = True):
        
        self.basename = "SharedRLAlgorithmInfo"
        self.namespace = namespace + self.basename

        self._terminate = False
        
        self.is_server = is_server

        self.init = None                                                  

        self.static_params = static_params
        self._parse_sim_dict() # applies changes if needed

        self.param_keys = []

        self.dynamic_info = DynamicRLAlgorithmNames()

        if self.is_server:

            # if client info is read on shared memory

            self.param_keys = self.dynamic_info.get() + list(self.static_params.keys())

        # actual data
        self.shared_data = RLAlgorithmDebData(namespace = self.namespace,
                    is_server = is_server, 
                    n_dims = len(self.param_keys), 
                    verbose = verbose, 
                    vlevel = vlevel,
                    safe = safe, 
                    force_reconnection = force_reconnection)
        
        # names
        if self.is_server:

            self.shared_datanames = StringTensorServer(length = len(self.param_keys), 
                                        basename = "InfoDataNames", 
                                        name_space = self.namespace,
                                        verbose = verbose, 
                                        vlevel = vlevel, 
                                        force_reconnection = force_reconnection)

        else:

            self.shared_datanames = StringTensorClient(
                                        basename = "InfoDataNames", 
                                        name_space = self.namespace,
                                        verbose = verbose, 
                                        vlevel = vlevel)
            
        self._is_running = False
    
    def _parse_sim_dict(self):

        pass
    
    def get_shared_mem(self):

        return [self.shared_data.get_shared_mem(),
            self.shared_datanames.get_shared_mem()]
    
    def is_running(self):

        return self._is_running
    
    def run(self):
        
        self.shared_datanames.run()
        
        self.shared_data.run()
            
        if self.is_server:
            
            names_written = self.shared_datanames.write_vec(self.param_keys, 0)

            if not names_written:
                
                exception = "Could not write shared names on shared memory!"

                Journal.log(self.__class__.__name__,
                    "run()",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
                            
        else:
            
            self.param_keys = [""] * self.shared_datanames.length()

            names_read = self.shared_datanames.read_vec(self.param_keys, 0)

            if not names_read:

                exception = "Could not read shared names on shared memory!"

                Journal.log(self.__class__.__name__,
                    "run()",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
            
            self.shared_data.synch_all(read=True, retry=True)
        
        self.param_values = np.full((len(self.param_keys), 1), 
                                fill_value=np.nan, 
                                dtype=toNumpyDType(eigenipc_dtype.Float))

        if self.is_server:
            
            for i in range(len(list(self.static_params.keys()))):
                
                # writing static sim info

                dyn_info_size = len(self.dynamic_info.get())

                # first m elements are custom info
                self.param_values[dyn_info_size + i, 0] = \
                    self.static_params[self.param_keys[dyn_info_size + i]]
                                        
            self.shared_data.write_retry(row_index=0,
                                    col_index=0,
                                    data=self.param_values)
            
        self._is_running = True
                          
    def write(self,
            dyn_info_name: Union[str, List[str]],
            val: Union[float, List[float]]):

        # always writes to shared memory
        
        if isinstance(dyn_info_name, list):
            
            if not isinstance(val, list):

                exception = "The provided val should be a list of values!"

                Journal.log(self.__class__.__name__,
                    "write()",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
                            
            if len(val) != len(dyn_info_name):
                
                exception = "Name list and values length mismatch!"

                Journal.log(self.__class__.__name__,
                    "write()",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
                
            for i in range(len(val)):
                
                idx = self.dynamic_info.get_idx(dyn_info_name[i])
                
                self.param_values[idx, 0] = val[i]
                
                self.shared_data.write_retry(data=self.param_values[idx, 0],
                                row_index=idx, col_index=0) 
            
        elif isinstance(dyn_info_name, str):
            
            idx = self.dynamic_info.get_idx(dyn_info_name)

            self.param_values[idx, 0] = val
        
            self.shared_data.write_retry(data=self.param_values[idx, 0],
                                row_index=idx, col_index=0) 
    
    def synch(self):

        self.shared_data.synch_all(read=True, retry = True)
    
    def get(self):

        self.synch()

        return self.shared_data.get_numpy_mirror().copy()
    
    def close(self):

        self.shared_data.close()
        self.shared_datanames.close()

    def terminate(self):

        # just an alias for legacy compatibility
        self.close()

    def __del__(self):
        
        self.close()
