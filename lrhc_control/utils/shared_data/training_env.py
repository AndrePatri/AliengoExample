from EigenIPC.PyEigenIPC.wrappers.shared_data_view import SharedTWrapper
from EigenIPC.PyEigenIPC import StringTensorServer, StringTensorClient
from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import dtype as sharsor_dtype, toNumpyDType
from EigenIPC.PyEigenIPC import Journal
from EigenIPC.PyEigenIPC import LogType

from control_cluster_bridge.utilities.shared_data.abstractions import SharedDataBase

from lrhc_control.utils.shared_data.base_data import NamedSharedTWrapper

from typing import Dict, Union, List

import numpy as np
import torch

# Training env info

class TrainingEnvDebData(SharedTWrapper):
                 
    def __init__(self,
        namespace = "",
        is_server = False, 
        n_dims: int = -1, 
        verbose: bool = False, 
        vlevel: VLevel = VLevel.V0,
        force_reconnection: bool = False,
        safe: bool = True,
        dtype: sharsor_dtype=sharsor_dtype.Float):

        basename = "TrainingEnvDebData" 

        super().__init__(namespace = namespace,
            basename = basename,
            is_server = is_server, 
            n_rows = n_dims, 
            n_cols = 1, 
            verbose = verbose, 
            vlevel = vlevel,
            dtype=dtype,
            fill_value=np.nan,
            safe = safe,
            force_reconnection=force_reconnection)

class DynamicTrainingEnvNames:

    def __init__(self):

        self._keys = ["step_rt_factor", 
                "total_rt_factor",
                "env_stepping_dt"]
        
        self.idx_dict = dict.fromkeys(self._keys, None)

        # dynamic sim info is by convention
        # put at the start
        for i in range(len(self._keys)):
            
            self.idx_dict[self._keys[i]] = i

    def get(self):

        return self._keys

    def get_idx(self, name: str):

        return self.idx_dict[name]
    
class SharedTrainingEnvInfo(SharedDataBase):
                           
    def __init__(self, 
                namespace: str,
                is_server = False, 
                training_env_params_dict: Dict = None,
                verbose = True, 
                vlevel = VLevel.V2, 
                safe: bool = True,
                force_reconnection: bool = True):
        
        self.namespace = namespace + "SharedTrainingEnvInfo"

        self._terminate = False
        
        self.is_server = is_server

        self.init = None                                                  

        self.training_env_params_dict = training_env_params_dict
        self._parse_sim_dict() # applies changes if needed

        self.param_keys = []

        self.dynamic_info = DynamicTrainingEnvNames()

        if self.is_server:

            # if client info is read on shared memory

            self.param_keys = self.dynamic_info.get() + list(self.training_env_params_dict.keys())

        # actual data
        self.shared_train_env_data = TrainingEnvDebData(namespace = self.namespace,
                    is_server = is_server, 
                    n_dims = len(self.param_keys), 
                    verbose = verbose, 
                    vlevel = vlevel,
                    safe = safe, 
                    force_reconnection = force_reconnection)
        
        # names
        if self.is_server:

            self.shared_train_env_datanames = StringTensorServer(length = len(self.param_keys), 
                                        basename = "TrainingEnvDataNames", 
                                        name_space = self.namespace,
                                        verbose = verbose, 
                                        vlevel = vlevel, 
                                        force_reconnection = force_reconnection)

        else:

            self.shared_train_env_datanames = StringTensorClient(
                                        basename = "TrainingEnvDataNames", 
                                        name_space = self.namespace,
                                        verbose = verbose, 
                                        vlevel = vlevel)
            
        self._is_running = False
    
    def _parse_sim_dict(self):

        if self.training_env_params_dict is not None:
        
            keys = list(self.training_env_params_dict.keys())

            for key in keys:
        
                if self.training_env_params_dict[key] == "cpu":
    
                    self.training_env_params_dict[key] = 0

                if self.training_env_params_dict[key] == "gpu" or \
                    self.training_env_params_dict[key] == "cuda":
        
                    self.training_env_params_dict[key] = 1
    
    def is_running(self):

        return self._is_running
    
    def get_shared_mem(self):

        return [self.shared_train_env_data.get_shared_mem(),
            self.shared_train_env_datanames.get_shared_mem()]
    
    def run(self):
        
        self.shared_train_env_datanames.run()
        
        self.shared_train_env_data.run()
            
        if self.is_server:
            
            names_written = self.shared_train_env_datanames.write_vec(self.param_keys, 0)

            if not names_written:
                
                exception = "Could not write shared train. env. names on shared memory!"

                Journal.log(self.__class__.__name__,
                    "run()",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
                            
        else:
            
            self.param_keys = [""] * self.shared_train_env_datanames.length()

            names_read = self.shared_train_env_datanames.read_vec(self.param_keys, 0)

            if not names_read:

                exception = "Could not read shared train. env. names on shared memory!"

                Journal.log(self.__class__.__name__,
                    "run()",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
            
            self.shared_train_env_data.synch_all(read=True, retry=True)
        
        self.param_values = np.full((len(self.param_keys), 1), 
                                fill_value=np.nan, 
                                dtype=toNumpyDType(sharsor_dtype.Float))

        if self.is_server:
            
            for i in range(len(list(self.training_env_params_dict.keys()))):
                
                # writing static sim info

                dyn_info_size = len(self.dynamic_info.get())

                # first m elements are custom info
                self.param_values[dyn_info_size + i, 0] = \
                    self.training_env_params_dict[self.param_keys[dyn_info_size + i]]
                                        
            self.shared_train_env_data.write_retry(row_index=0,
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
                
                self.shared_train_env_data.write_retry(data=self.param_values[idx, 0],
                                row_index=idx, col_index=0) 
            
        elif isinstance(dyn_info_name, str):
            
            idx = self.dynamic_info.get_idx(dyn_info_name)

            self.param_values[idx, 0] = val
        
            self.shared_train_env_data.write_retry(data=self.param_values[idx, 0],
                                row_index=idx, col_index=0) 
    
    def synch(self):

        self.shared_train_env_data.synch_all(read=True, retry = True)
    
    def get(self):

        self.synch()

        return self.shared_train_env_data.get_numpy_mirror().copy()
    
    def close(self):

        self.shared_train_env_data.close()
        self.shared_train_env_datanames.close()

    def terminate(self):

        # just an alias for legacy compatibility
        self.close()

    def __del__(self):
        
        self.close()

# training info

class SubRewards(NamedSharedTWrapper):

    def __init__(self,
            namespace: str,
            n_envs: int = None, 
            n_rewards: int = None, 
            reward_names: List[str] = None,
            env_names: List[str] = None,
            is_server = False, 
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            safe: bool = True,
            force_reconnection: bool = False,
            with_gpu_mirror: bool = False,
            fill_value = 0.0,
            dtype: sharsor_dtype=sharsor_dtype.Float):

        basename = "Rewards"

        super().__init__(namespace=namespace,
                    basename=basename,
                    n_rows=n_envs,
                    n_cols=n_rewards,
                    dtype=dtype,
                    col_names=reward_names,
                    row_names=env_names,
                    is_server=is_server,
                    verbose=verbose,
                    vlevel=vlevel,
                    safe=safe,
                    force_reconnection=force_reconnection,
                    with_gpu_mirror=with_gpu_mirror,
                    fill_value=fill_value)

class TotRewards(NamedSharedTWrapper):

    def __init__(self,
            namespace: str,
            n_envs: int = None, 
            reward_names: List[str] = None,
            env_names: List[str] = None,
            is_server = False, 
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            safe: bool = True,
            force_reconnection: bool = False,
            with_gpu_mirror: bool = False,
            fill_value = 0.0,
            dtype: sharsor_dtype=sharsor_dtype.Float):

        basename = "TotRewards"

        super().__init__(namespace=namespace,
                    basename=basename,
                    n_rows=n_envs,
                    n_cols=1,
                    dtype=dtype,
                    col_names=reward_names,
                    row_names=env_names,
                    is_server=is_server,
                    verbose=verbose,
                    vlevel=vlevel,
                    safe=safe,
                    force_reconnection=force_reconnection,
                    with_gpu_mirror=with_gpu_mirror,
                    fill_value=fill_value)
        
class Observations(NamedSharedTWrapper):

    def __init__(self,
            namespace: str,
            n_envs: int = None, 
            obs_dim: int = None,
            obs_names: List[str] = None,
            env_names: List[str] = None,
            is_server = False, 
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            safe: bool = True,
            force_reconnection: bool = False,
            with_gpu_mirror: bool = False,
            fill_value = 0.0,
            dtype: sharsor_dtype=sharsor_dtype.Float):

        basename = "Observations"

        super().__init__(namespace=namespace,
                    basename=basename,
                    n_rows=n_envs,
                    n_cols=obs_dim,
                    dtype=dtype,
                    col_names=obs_names,
                    row_names=env_names,
                    is_server=is_server,
                    verbose=verbose,
                    vlevel=vlevel,
                    safe=safe,
                    force_reconnection=force_reconnection,
                    with_gpu_mirror=with_gpu_mirror,
                    fill_value=fill_value)

    def get_obs_names(self):
        return self._col_names
    
class NextObservations(NamedSharedTWrapper):

    def __init__(self,
            namespace: str,
            n_envs: int = None, 
            obs_dim: int = None,
            obs_names: List[str] = None,
            env_names: List[str] = None,
            is_server = False, 
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            safe: bool = True,
            force_reconnection: bool = False,
            with_gpu_mirror: bool = False,
            fill_value = 0.0,
            dtype: sharsor_dtype=sharsor_dtype.Float):

        basename = "NextObservations"

        super().__init__(namespace=namespace,
                    basename=basename,
                    n_rows=n_envs,
                    n_cols=obs_dim,
                    dtype=dtype,
                    col_names=obs_names,
                    row_names=env_names,
                    is_server=is_server,
                    verbose=verbose,
                    vlevel=vlevel,
                    safe=safe,
                    force_reconnection=force_reconnection,
                    with_gpu_mirror=with_gpu_mirror,
                    fill_value=fill_value)
    
    def get_obs_names(self):
        return self._col_names
    
class Actions(NamedSharedTWrapper):

    def __init__(self,
            namespace: str,
            n_envs: int = None, 
            action_dim: int = None,
            action_names: List[str] = None,
            env_names: List[str] = None,
            is_server = False, 
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            safe: bool = True,
            force_reconnection: bool = False,
            with_gpu_mirror: bool = False,
            fill_value = 0.0,
            dtype: sharsor_dtype=sharsor_dtype.Float):

        basename = "Actions"

        super().__init__(namespace=namespace,
                    basename=basename,
                    n_rows=n_envs,
                    n_cols=action_dim,
                    dtype=dtype,
                    col_names=action_names,
                    row_names=env_names,
                    is_server=is_server,
                    verbose=verbose,
                    vlevel=vlevel,
                    safe=safe,
                    force_reconnection=force_reconnection,
                    with_gpu_mirror=with_gpu_mirror,
                    fill_value=fill_value)
        
class Terminations(SharedTWrapper):

    def __init__(self,
            namespace: str,
            n_envs: int = None, 
            is_server = False, 
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            safe: bool = True,
            force_reconnection: bool = False,
            with_gpu_mirror: bool = False,
            fill_value = 0):
            
            basename = "Terminations"
    
            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = 1, 
                verbose = verbose, 
                vlevel = vlevel,
                safe = safe, # boolean operations are atomic on 64 bit systems
                dtype=sharsor_dtype.Bool,
                force_reconnection=force_reconnection,
                with_gpu_mirror=with_gpu_mirror,
                with_torch_view=True,
                fill_value = fill_value)

    def reset(self):

        self.to_zero()

class Truncations(SharedTWrapper):

    def __init__(self,
            namespace: str,
            n_envs: int = None, 
            is_server = False, 
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            safe: bool = True,
            force_reconnection: bool = False,
            with_gpu_mirror: bool = False,
            fill_value = 0):
            
            basename = "Truncations"
    
            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_envs, 
                n_cols = 1, 
                verbose = verbose, 
                vlevel = vlevel,
                safe = safe, # boolean operations are atomic on 64 bit systems
                dtype=sharsor_dtype.Bool,
                force_reconnection=force_reconnection,
                with_gpu_mirror=with_gpu_mirror,
                with_torch_view=True,
                fill_value = fill_value)

    def reset(self):

        self.to_zero()

class SubTerminations(NamedSharedTWrapper):

    def __init__(self,
            namespace: str,
            n_envs: int = None, 
            n_term: int = None, 
            term_names: List[str] = None,
            env_names: List[str] = None,
            is_server = False, 
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            safe: bool = True,
            force_reconnection: bool = False,
            with_gpu_mirror: bool = False,
            fill_value = False):

        basename = "SubTerminations"

        super().__init__(namespace=namespace,
                    basename=basename,
                    n_rows=n_envs,
                    n_cols=n_term,
                    dtype=sharsor_dtype.Bool,
                    col_names=term_names,
                    row_names=env_names,
                    is_server=is_server,
                    verbose=verbose,
                    vlevel=vlevel,
                    safe=safe,
                    force_reconnection=force_reconnection,
                    with_gpu_mirror=with_gpu_mirror,
                    fill_value=fill_value)

class SubTruncations(NamedSharedTWrapper):

    def __init__(self,
            namespace: str,
            n_envs: int = None, 
            n_trunc: int = None, 
            truc_names: List[str] = None,
            env_names: List[str] = None,
            is_server = False, 
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            safe: bool = True,
            force_reconnection: bool = False,
            with_gpu_mirror: bool = False,
            fill_value = False):

        basename = "SubTruncations"

        super().__init__(namespace=namespace,
                    basename=basename,
                    n_rows=n_envs,
                    n_cols=n_trunc,
                    dtype=sharsor_dtype.Bool,
                    col_names=truc_names,
                    row_names=env_names,
                    is_server=is_server,
                    verbose=verbose,
                    vlevel=vlevel,
                    safe=safe,
                    force_reconnection=force_reconnection,
                    with_gpu_mirror=with_gpu_mirror,
                    fill_value=fill_value)
        
class SimpleCounters(SharedDataBase):
    
    class StepCounter(SharedTWrapper):

        def __init__(self,
                namespace: str,
                basename: str,
                n_envs: int = None, 
                is_server = False, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                safe: bool = True,
                force_reconnection: bool = False,
                with_gpu_mirror: bool = False,
                fill_value = 0):
                        
                super().__init__(namespace = namespace,
                    basename = basename,
                    is_server = is_server, 
                    n_rows = n_envs, 
                    n_cols = 1, 
                    verbose = verbose, 
                    vlevel = vlevel,
                    safe = safe, # boolean operations are atomic on 64 bit systems
                    dtype=sharsor_dtype.Int,
                    force_reconnection=force_reconnection,
                    with_gpu_mirror=with_gpu_mirror,
                    with_torch_view=True,
                    fill_value = fill_value)
    
    def __init__(self,
                namespace: str,
                basename: str,
                n_steps_lb: int = None,
                n_steps_ub: int = None,
                randomize_offsets_at_startup: bool=True,
                n_envs: int = None, 
                is_server = False, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                safe: bool = True,
                force_reconnection: bool = False,
                with_gpu_mirror: bool = False):

        self._using_gpu = with_gpu_mirror
        self._torch_device = "cuda" if self._using_gpu else "cpu"

        self._n_steps_lb = n_steps_lb
        self._n_steps_ub = n_steps_ub
        self._n_steps_mean = None

        self._is_server = is_server

        self._randomize_offsets_at_startup=randomize_offsets_at_startup

        if self._is_server and ((self._n_steps_lb is None) or (self._n_steps_ub is None)):
            exception = "Lower and upper step bounds have to be specified when in server mode!!!"
            Journal.log(self.__class__.__name__,
                "run()",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
        
        if (not self._n_steps_ub is None) and (not self._n_steps_lb is None):
            self._n_steps_mean = round((self._n_steps_ub+self._n_steps_lb)/2.0)

        self._n_envs = n_envs

        self._step_counter = self.StepCounter(namespace = namespace,
                    basename = basename,
                    is_server = is_server, 
                    n_envs = n_envs, 
                    verbose = verbose, 
                    vlevel = vlevel,
                    safe = safe, # boolean operations are atomic on 64 bit systems
                    force_reconnection=force_reconnection,
                    with_gpu_mirror=with_gpu_mirror)

    def _write(self):

        if self._using_gpu:
            # copy from gpu to cpu
            self._step_counter.synch_mirror(from_gpu=True,non_blocking=False)
        # copy from cpu to shared memory
        self._step_counter.synch_all(read=False, retry=True)

    def get_shared_mem(self):

        return [self._step_counter.get_shared_mem()]
    
    def is_running(self):

        return self._step_counter.is_running()

    def sync_counters(self,
        other_counter):
        self.get()[:, :]=other_counter.get() 

    def increment(self,
        to_be_incremented: torch.Tensor = None):
        
        if to_be_incremented is None:
            self.get()[:, :] = self.get() + 1
        else:
            self.get()[to_be_incremented, :] = self.get()[to_be_incremented, :] + 1

        self._write()
    
    def decrement(self,
        to_be_incremented: torch.Tensor = None):
        
        if to_be_incremented is None:
            self.get()[:, :] = self.get() - 1
        else:
            self.get()[to_be_incremented, :] = self.get()[to_be_incremented, :] - 1

        self._write()

    def get(self):

        return self._step_counter.get_torch_mirror(gpu=self._using_gpu)
    
    def counter(self):

        return self._step_counter
    
    def run(self):

        self._step_counter.run()
        if self._is_server:
            self._n_steps = torch.full((self._n_envs, 1), dtype=torch.int32, device=self._torch_device, 
                fill_value=self._n_steps_mean)
            self.reset(randomize_offsets=self._randomize_offsets_at_startup)

    def close(self):

        self._step_counter.close()

    def time_limits_reached(self):

        # to be called after increment or decrement 

        valid_counters=~(self.get()==0)
        time_l_reached = (self.get() % self._n_steps) == 0

        # counters at 0 are neglected 
        return torch.logical_and(time_l_reached,valid_counters)
    
    def steps_ub(self):
        return self._n_steps_ub

    def steps_lb(self):
        return self._n_steps_lb

    def steps_mean(self):
        return self._n_steps_mean

    def reset(self,
            to_be_reset: torch.Tensor = None,
            randomize_limits: bool = False,
            randomize_offsets: bool = False):

        if to_be_reset is None:
            # resets all counters
            self.get().zero_()
            if randomize_limits and (not self._n_steps_lb==self._n_steps_ub): # randomize counter durations upon resets
                self._n_steps[:, :] = torch.randint(low=self._n_steps_lb, high=self._n_steps_ub, size=(self._n_envs, 1),
                                            dtype=torch.int32,device=self._torch_device)
            
            if randomize_offsets:
                random_step_offsets=torch.randint(low=0, high=self._n_steps_mean, size=(self._n_envs, 1),
                                dtype=torch.int32,device=self._torch_device)
                self.get()[:, :]=self.get()+random_step_offsets # add random offsets from 0 to the mean counter duration

        else:
            n_to_be_reset = torch.sum(to_be_reset.squeeze()).item()
            if not n_to_be_reset == 0:
                self.get()[to_be_reset.squeeze() , :] = 0
                if randomize_limits and (not self._n_steps_lb==self._n_steps_ub):
                    self._n_steps[to_be_reset.squeeze() , :] = torch.randint(low=self._n_steps_lb, high=self._n_steps_ub, size=(n_to_be_reset, 1),
                                                dtype=torch.int32,device=self._torch_device)
                
                if randomize_offsets:
                    random_step_offsets=torch.randint(low=0, high=self._n_steps_mean, size=(n_to_be_reset, 1),
                                    dtype=torch.int32,device=self._torch_device)
                    self.get()[to_be_reset.squeeze(), :]=self.get()[to_be_reset.squeeze(), :]+random_step_offsets # add random offsets from 0 to the mean counter duration
        
        self._write()

class EpisodesCounter(SimpleCounters):

    def __init__(self,
                namespace: str,
                n_steps_lb: int = None,
                n_steps_ub: int = None,
                randomize_offsets_at_startup: bool = True,
                n_envs: int = None, 
                is_server = False, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                safe: bool = True,
                force_reconnection: bool = False,
                with_gpu_mirror: bool = False):

        basename = "EpisodesCounter"

        super().__init__(namespace=namespace,
                basename=basename,
                n_steps_lb=n_steps_lb,
                n_steps_ub=n_steps_ub,
                randomize_offsets_at_startup=randomize_offsets_at_startup,
                n_envs=n_envs, 
                is_server=is_server, 
                verbose=verbose, 
                vlevel=vlevel,
                safe=safe,
                force_reconnection=force_reconnection,
                with_gpu_mirror=with_gpu_mirror)

class TaskRandCounter(SimpleCounters):

    def __init__(self,
                namespace: str,
                n_steps_lb: int = None,
                n_steps_ub: int = None,
                randomize_offsets_at_startup: bool = True,
                n_envs: int = None, 
                is_server = False, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                safe: bool = True,
                force_reconnection: bool = False,
                with_gpu_mirror: bool = False):

        basename = "TaskRandCounter"

        super().__init__(namespace=namespace,
                basename=basename,
                n_steps_lb=n_steps_lb,
                n_steps_ub=n_steps_ub,
                randomize_offsets_at_startup=randomize_offsets_at_startup,
                n_envs=n_envs, 
                is_server=is_server, 
                verbose=verbose, 
                vlevel=vlevel,
                safe=safe,
                force_reconnection=force_reconnection,
                with_gpu_mirror=with_gpu_mirror)

class SafetyRandResetsCounter(SimpleCounters):

    def __init__(self,
                namespace: str,
                n_steps_lb: int = None,
                n_steps_ub: int = None,
                randomize_offsets_at_startup: bool = True,
                n_envs: int = None, 
                is_server = False, 
                verbose: bool = False, 
                vlevel: VLevel = VLevel.V0,
                safe: bool = True,
                force_reconnection: bool = False,
                with_gpu_mirror: bool = False):

        basename = "SafetyRandResetsCounter"

        super().__init__(namespace=namespace,
                basename=basename,
                n_steps_lb=n_steps_lb,
                n_steps_ub=n_steps_ub,
                randomize_offsets_at_startup=randomize_offsets_at_startup,
                n_envs=n_envs, 
                is_server=is_server, 
                verbose=verbose, 
                vlevel=vlevel,
                safe=safe,
                force_reconnection=force_reconnection,
                with_gpu_mirror=with_gpu_mirror)

if __name__ == "__main__":  

    def print_data(counter,i):
        print(f"INFO{i}")
        print("limits reached")
        print(counter.time_limits_reached())
        print("counters")
        print(counter.get())

    n_steps_lb=10
    n_steps_ub=10
    n_envs=20
    counter=SimpleCounters(namespace="Prova",
        basename="SimpleCounter",
        n_steps_lb=n_steps_lb,
        n_steps_ub=n_steps_ub,
        randomize_offsets_at_startup=True,
        n_envs=n_envs, 
        is_server=True, 
        verbose=True, 
        vlevel=VLevel.V2,
        safe=True,
        force_reconnection=True,
        with_gpu_mirror=False)
    counter.run()

    print("COUNTERS")
    print(counter.steps_lb())
    print(counter.steps_ub())
    print(counter.steps_mean())

    n_steps=30
    for i in range(n_steps):
        counter.increment()
        print_data(counter,i)
