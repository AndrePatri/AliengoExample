from EigenIPC.PyEigenIPC import Journal
from EigenIPC.PyEigenIPC import LogType

import torch

from typing import List

class MemBuffer():
    
    # memory buffer for computing runtime std and mean 
    # synchronous means that there is a unique counter for the current
    # position in the buffer
    def __init__(self,
            name: str,
            data_tensor: torch.Tensor,
            data_names: List[str] = None, 
            debug: bool = False,
            horizon: int = 2,
            dtype: torch.dtype = torch.float32,
            use_gpu: bool = False):
        
        self._name = name

        self._dtype = dtype
        self._torch_device = "cuda" if use_gpu else "cpu"

        self._horizon=horizon # number of samples to store
        if self._horizon < 2:
            exception = f"Provided horizon ({horizon}) should be at least 2!!"
            Journal.log(self.__class__.__name__ + f"[{self._name}]",
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep=True)
    
        self._membf_pos=0 #position in mem buff at which new samples with be added
      
        self._debug = debug
        self._use_constant_scaling = False # whether to use constant 
        # scaling over episodes (this is useful to log meaningful reward data). If not 
        # enabled, metrics are actually averaged over the episode's timesteps, meaning that 
        # no difference between long or short episodes can be seen
        self._scaling = None

        self._n_envs = data_tensor.shape[0]
        self._data_size = data_tensor.shape[1]
                            
        self._init_data()

        self._data_names = data_names
        if data_names is not None:
            if not len(data_names) == self._data_size:
                exception = f"Provided data names length {len(data_names)} does not match {self._data_size}!!"
                Journal.log(self.__class__.__name__,
                    "__init__",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep=True)
        else:
            self._data_names = []
            for i in range(self._data_size):
                self._data_names.append(f"{self._name}_n{i}")

    def _init_data(self):
        
        self._mem_buff=None
        # initialize a memory buffer with a given horizon
        self._mem_buff=torch.full(size=(self._n_envs, self._data_size,self._horizon), 
                fill_value=0.0,
                dtype=self._dtype, 
                device=self._torch_device,
                requires_grad=False)
        self._running_mean=torch.full(size=(self._n_envs, self._data_size), 
                fill_value=0.0,
                dtype=self._dtype, 
                device=self._torch_device,
                requires_grad=False)
        self._running_std=torch.full(size=(self._n_envs, self._data_size), 
                fill_value=0.0,
                dtype=self._dtype, 
                device=self._torch_device,
                requires_grad=False)
        
        self._membf_pos=0
    
    def reset_all(self,
           init_data:torch.Tensor=None):
        if init_data is None: # reset to 0
            self._mem_buff.zero_()
        else:
            # fill all buffer with init provided by data
            if self._debug:
                self._check_new_data(new_data=init_data)
            self._mem_buff[:, :, :]=init_data.unsqueeze(2)
        self._membf_pos=0
        self._running_mean.zero_()
        self._running_std.fill_(0.0)

    def reset(self,
        to_be_reset: torch.Tensor,
        init_data:torch.Tensor=None):

        if init_data is None: # reset to 0
            self._mem_buff[to_be_reset, :, :]=0
        else:
            # fill all buffer with init provided by data
            if self._debug:
                self._check_new_data(new_data=init_data)
            self._mem_buff[to_be_reset, :, :]=init_data[to_be_reset, :].unsqueeze(2)
        # _membf_pos kept at last one
        self._running_mean[to_be_reset, :]=0.0
        self._running_std[to_be_reset, :]=0.0
        
    def _check_new_data(self,new_data):
        self._check_sizes(new_data=new_data)
        self._check_finite(new_data=new_data)

    def _check_sizes(self,new_data):
        if (not new_data.shape[0] == self._n_envs) or \
            (not new_data.shape[1] == self._data_size):
            exception = f"Provided new_data tensor shape {new_data.shape[0]}, {new_data.shape[1]}" + \
                f" does not match {self._n_envs}, {self._data_size}!!"
            Journal.log(self.__class__.__name__ + f"[{self._name}]",
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep=True)
    
    def _check_finite(self,new_data):
        if (not torch.isfinite(new_data).all().item()):
            print(new_data)
            exception = f"Found non finite elements in provided data!!"
            Journal.log(self.__class__.__name__ + f"[{self._name}]",
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep=True)
            
    def update(self, 
        new_data: torch.Tensor):

        if self._debug:
            self._check_new_data(new_data=new_data)

        self._mem_buff[:,:,self._membf_pos]=new_data
        self._running_mean[:, :]=torch.mean(self._mem_buff,dim=2)
        self._running_std[:, :]=torch.std(self._mem_buff,dim=2)
        self._membf_pos+=1
        if self._membf_pos==self.horizon():
            self._membf_pos=0         

    def data_names(self):
        return self._data_names
    
    def horizon(self):
        return self._horizon
    
    def get(self,idx:int=None):
        if idx is None: # always get last 
            return self._mem_buff[:,:,self._membf_pos-1]
        else: # return data ad horizon idx, where 0 means latest
            # and self._horizon-1 mean oldest
            if (not idx>=0) and (idx<self._horizon):
                exception = f"Idx {idx} exceeds horizon length {self._horizon} (0-based)!"
                Journal.log(self.__class__.__name__ + f"[{self._name}]",
                    "__init__",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep=True)
            return self._mem_buff[:,:,self._membf_pos-1-idx]
    
    def get_bf(self,clone:bool=False):
        memory_buffer=self._mem_buff.detach()

        if clone:
            return memory_buffer.clone()
        else:
            return memory_buffer
        
    def std(self,clone:bool=False):
        if clone:
            return self._running_std.detach().clone()
        else:
            return self._running_std.detach()
    
    def mean(self,clone:bool=False):
        if clone:
            return self._running_mean.detach().clone()
        else:
            return self._running_mean.detach()

    def pos(self):
        return self._membf_pos

class EpisodicData():

    # class for helping log db dta from episodes over 
    # vectorized envs

    def __init__(self,
            name: str,
            data_tensor: torch.Tensor,
            data_names: List[str] = None, 
            debug: bool = False,
            dtype: torch.dtype = torch.float32,
            ep_vec_freq: int = 1,
            store_transitions: bool = False, # also store detailed transition history
            max_ep_duration: int = -1
            ):
        
        self._store_transitions=store_transitions
        self._max_ep_duration=max_ep_duration
        if self._store_transitions and self._max_ep_duration < 0:
            Journal.log(self.__class__.__name__,
                "__init__",
                "When store_transitions==True, then a positive max_ep_duration should be provided",
                LogType.EXCEP,
                throw_when_excep=True)
            
        self._keep_track=True # we generally want to propagate 
        # the current 

        self._ep_vec_freq=ep_vec_freq

        self._name = name

        self._dtype = dtype

        self._debug = debug
        
        self._use_constant_scaling = False # whether to use constant 
        # scaling over episodes (this is useful to log meaningful reward data). If not 
        # enabled, metrics are actually averaged over the episode's timesteps, meaning that 
        # no difference between long or short episodes can be seen
        self._scaling = None

        self._n_envs = data_tensor.shape[0]
        self._data_size = data_tensor.shape[1]
                            
        self._init_data()

        self._data_names = data_names
        if data_names is not None:
            if not len(data_names) == self._data_size:
                exception = f"Provided data names length {len(data_names)} does not match {self._data_size}!!"
                Journal.log(self.__class__.__name__,
                    "__init__",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep=True)
        else:
            self._data_names = []
            for i in range(self._data_size):
                self._data_names.append(f"data_n{i}")
    
    def set_constant_data_scaling(self,
                enable: bool = True,
                scaling: torch.Tensor = None):
        
        if scaling is not None:

            if (not scaling.shape[0] == self._n_envs) or \
            (not scaling.shape[1] == 1):
                exception = f"Provided scaling tensor shape {scaling.shape[0]}, {scaling.shape[1]}" + \
                    f" does not match {self._n_envs}, {1}!!"
                Journal.log(self.__class__.__name__ + f"[{self._name}]",
                    "__init__",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep=True)

            self._scaling[:, :] = scaling
        
        self._use_constant_scaling = enable

    def name(self):
        return self._name
    
    def _init_data(self):
        
        self._big_val=1e6
        # undiscounted sum of each env, during a single episode
        self._current_ep_sum = torch.full(size=(self._n_envs, self._data_size), 
                fill_value=0.0,
                dtype=self._dtype, device="cpu",
                requires_grad=False) # we don't need it on GPU
        # avrg data of each env, during a single episode, over the number of transitions
        self._current_ep_sum_scaled = torch.full(size=(self._n_envs, self._data_size), 
                fill_value=0.0,
                dtype=self._dtype, device="cpu",
                requires_grad=False)
        
        # avrg data of each env, over all the ALREADY played episodes.
        self._tot_sum_up_to_now = torch.full(size=(self._n_envs, self._data_size), 
                fill_value=0.0,
                dtype=self._dtype, device="cpu",
                requires_grad=False)
        # avrg over n of episodes (including the current one)
        self._average_over_eps = torch.full(size=(self._n_envs, self._data_size), 
                fill_value=0.0,
                dtype=self._dtype, device="cpu",
                requires_grad=False)
        self._average_over_eps_last = torch.full_like(self._average_over_eps,
                fill_value=0.0,
                requires_grad=False)
        # min max info of sub data
        self._max_over_eps = torch.full(size=(self._n_envs, self._data_size), 
                fill_value=-self._big_val,
                dtype=self._dtype, device="cpu",
                requires_grad=False)
        self._max_over_eps_last = torch.full_like(self._average_over_eps,
                fill_value=-self._big_val,
                requires_grad=False)
        self._min_over_eps = torch.full(size=(self._n_envs, self._data_size), 
                fill_value=self._big_val,
                dtype=self._dtype, device="cpu",
                requires_grad=False)
        self._min_over_eps_last = torch.full_like(self._average_over_eps,
                fill_value=self._big_val,
                requires_grad=False)
        # current episode index
        self._n_played_eps = torch.full(size=(self._n_envs, 1), 
                fill_value=0,
                dtype=torch.int32, device="cpu",
                requires_grad=False)
        self._n_played_eps_last = torch.full(size=(self._n_envs, 1), 
                fill_value=0,
                dtype=torch.int32, device="cpu",
                requires_grad=False)

        # current step counter (within this episode)
        self._steps_counter = torch.full(size=(self._n_envs, 1), 
                fill_value=0,
                dtype=torch.int32, device="cpu",
                requires_grad=False)

        self._scale_now = torch.full(size=(self._n_envs, 1), 
                fill_value=1,
                dtype=torch.int32, device="cpu",
                requires_grad=False)

        # just used if use_constant_scaling
        self._scaling = torch.full(size=(self._n_envs, 1), 
                fill_value=1,
                dtype=torch.int32, device="cpu",
                requires_grad=False) # default to scaling 1

        self._fresh_metrics_avail = torch.full(size=(self._n_envs, 1), 
            fill_value=False,
            dtype=torch.bool, device="cpu",
            requires_grad=False)
        
        self._full_data=None
        if self._store_transitions:
            self._full_data=torch.full(size=(
                            self._ep_vec_freq,
                            self._max_ep_duration,
                            self._n_envs, self._data_size), 
                fill_value=torch.nan,
                dtype=self._dtype, device="cpu",
                requires_grad=False)
            self._full_data_last=torch.full(size=(
                            self._ep_vec_freq,
                            self._max_ep_duration,
                            self._n_envs, self._data_size), 
                fill_value=torch.nan,
                dtype=self._dtype, device="cpu",
                requires_grad=False)

    def reset(self,
            keep_track: bool = None,
            to_be_reset: torch.Tensor = None):

        if to_be_reset is None: # reset all
            if keep_track is not None:
                if not keep_track:
                    self._current_ep_sum.zero_()
                    self._steps_counter.zero_()
            else:
                if not self._keep_track: # if not, we propagate ep sum and steps 
                    # from before this reset call 
                    self._current_ep_sum.zero_()
                    self._steps_counter.zero_()
            
            self._max_over_eps[:, :]=-self._big_val
            self._min_over_eps[:, :]=self._big_val
            self._current_ep_sum_scaled.zero_()
            self._tot_sum_up_to_now.zero_()
            self._average_over_eps.zero_()            
            self._n_played_eps.zero_()
            self._scale_now.fill_(1)

            if self._full_data is not None:
                self._full_data.fill_(torch.nan)
            
        else: # only reset some envs
            if keep_track is not None:
                if not keep_track:
                    self._current_ep_sum[to_be_reset, :]=0
                    self._steps_counter[to_be_reset, :]=0
            else:
                if not self._keep_track: # if not, we propagate ep sum and steps 
                    # from before this reset call 
                    self._current_ep_sum[to_be_reset, :]=0
                    self._steps_counter[to_be_reset, :]=0
            
            self._max_over_eps[to_be_reset, :]=-self._big_val
            self._min_over_eps[to_be_reset, :]=self._big_val
            self._current_ep_sum_scaled[to_be_reset, :]=0
            self._tot_sum_up_to_now[to_be_reset, :]=0
            self._average_over_eps[to_be_reset, :]=0
            self._n_played_eps[to_be_reset, :]=0
            self._scale_now[to_be_reset, :]=1
            
            if self._full_data is not None:
                self._full_data[:, :, to_be_reset, :]=torch.nan
                
    def update(self, 
        new_data: torch.Tensor,
        ep_finished: torch.Tensor,
        ignore_ep_end: torch.Tensor = None): # rewards scaled over episode length

        if (not new_data.shape[0] == self._n_envs) or \
            (not new_data.shape[1] == self._data_size):
            exception = f"Provided new_data tensor shape {new_data.shape[0]}, {new_data.shape[1]}" + \
                f" does not match {self._n_envs}, {self._data_size}!!"
            Journal.log(self.__class__.__name__ + f"[{self._name}]",
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep=True)
        
        if (not ep_finished.shape[0] == self._n_envs) or \
            (not ep_finished.shape[1] == 1):
            exception = f"Provided ep_finished boolean tensor shape {ep_finished.shape[0]}, {ep_finished.shape[1]}" + \
                f" does not match {self._n_envs}, {1}!!"
            Journal.log(self.__class__.__name__ + f"[{self._name}]",
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep=True)

        if self._debug and (not torch.isfinite(new_data).all().item()):
            print(new_data)
            exception = f"Found non finite elements in provided data!!"
            Journal.log(self.__class__.__name__ + f"[{self._name}]",
                "__init__",
                exception,
                LogType.EXCEP,
                throw_when_excep=True)
        
        self._fresh_metrics_avail[:, :]=False

        if not self._use_constant_scaling:
            # use current n of timesteps as scale
            self._scale_now[:, :] = self._steps_counter+1  
        else:
            # constant scaling (e.g. max ep length)
            self._scale_now[:, :] = self._scaling 

        # sum over the current episode
        self._current_ep_sum[:, :] = self._current_ep_sum + new_data 

        # average using the scale
        self._current_ep_sum_scaled[:, :] = self._current_ep_sum[:, :] / self._scale_now[:, :] 
        
        # sum over played episodes (stats are logged over self._ep_vec_freq episodes for each env)
        self._tot_sum_up_to_now[ep_finished.flatten(), :] += self._current_ep_sum_scaled[ep_finished.flatten(), :]

        if self._full_data is not None: # store data on single transition
            self._full_data[self._n_played_eps, self._steps_counter, # this step
                :, :]=new_data.to(dtype=self._dtype)

        self._n_played_eps[ep_finished.flatten(), 0] += 1 # an episode has been played
        
        if ignore_ep_end is not None: # ignore data if to be ignored and ep end;
            # useful to avoid introducing wrong db data when, for example, using random
            # episodic truncations
            to_be_ignored=torch.logical_and(ep_finished,ignore_ep_end)
            self._n_played_eps[to_be_ignored.flatten(), 0] -= 1 # episode never happened
            # remove data
            self._tot_sum_up_to_now[to_be_ignored.flatten(), :] -= self._current_ep_sum_scaled[to_be_ignored.flatten(), :]

        self._average_over_eps[ep_finished.flatten(), :] = \
            (self._tot_sum_up_to_now[ep_finished.flatten(), :]) / \
                self._n_played_eps[ep_finished.flatten(), :] 
        
        self._current_ep_sum[ep_finished.flatten(), :] = 0 # if finished, reset current sum
        
        self._max_over_eps[:, :]=torch.maximum(input=self._max_over_eps, other=new_data)
        self._min_over_eps[:, :]=torch.minimum(input=self._min_over_eps, other=new_data)
        # increment counters
        self._steps_counter[~ep_finished.flatten(), :] +=1 # step performed
        self._steps_counter[ep_finished.flatten(), :] =0 # reset step counters

        # automatic reset for envs when self._ep_vec_freq episodes have been played
        self._fresh_metrics_avail[:, :]=(self._n_played_eps>=self._ep_vec_freq)
        selector=self._fresh_metrics_avail.flatten()
        self._n_played_eps_last[selector, :]=\
            self._n_played_eps[selector, :]
        self._average_over_eps_last[selector, :]=\
            self._average_over_eps[selector, :]
        self._max_over_eps_last[selector, :]=\
            self._max_over_eps[selector, :]
        self._min_over_eps_last[selector, :]=\
            self._min_over_eps[selector, :]

        if self._full_data is not None:
            self._full_data_last[:, :, selector, :]=self._full_data[:, :, selector, :]
            
        EpisodicData.reset(self,to_be_reset=selector)        

    def data_names(self):
        return self._data_names
    
    def get_full_episodic_data(self, 
        env_selector: torch.Tensor = None):

        if env_selector is None:
            return self._full_data_last
        else:
            return self._full_data_last[:, :, env_selector.flatten(), :]
            
    def get_max(self, 
        env_selector: torch.Tensor = None):
        if env_selector is None:
            return self._max_over_eps_last
        else:
            return self._max_over_eps_last[env_selector.flatten(), :]
    
    def get_avrg(self, 
        env_selector: torch.Tensor = None):
        if env_selector is None:
            return self._average_over_eps_last
        else:
            return self._average_over_eps_last[env_selector.flatten(), :]
        
    def get_min(self, 
        env_selector: torch.Tensor = None):
        if env_selector is None:
            return self._min_over_eps_last
        else:
            return self._min_over_eps_last[env_selector.flatten(), :]
        
    def get_max_over_envs(self, 
        env_selector: torch.Tensor = None):
        sub_env_max_over_eps, _ = torch.max(self.get_max(env_selector=env_selector), dim=0, keepdim=True)
        return sub_env_max_over_eps
    
    def get_avrg_over_envs(self, 
        env_selector: torch.Tensor = None):
        scale=self._n_envs if env_selector is None else env_selector.flatten().shape[0]
        return torch.sum(self.get_avrg(env_selector=env_selector), dim=0, keepdim=True)/scale

    def get_min_over_envs(self, 
        env_selector: torch.Tensor = None):
        sub_env_min_over_eps, _ = torch.min(self.get_min(env_selector=env_selector), dim=0, keepdim=True)
        return sub_env_min_over_eps
    
    def get_n_played_episodes(self, 
        env_selector: torch.Tensor = None):
        if env_selector is None:
            return torch.sum(self._n_played_eps_last).item()
        else:
            return torch.sum(self._n_played_eps_last[env_selector.flatten(), :]).item()
    
    def step_counters(self, 
        env_selector: torch.Tensor = None):
        if env_selector is None:
            return self._steps_counter
        else:
            return self._steps_counter[env_selector.flatten(), :]

    def get_n_played_tsteps(self, 
        env_selector: torch.Tensor = None):
        return torch.sum(self.step_counters(env_selector=env_selector)).item()    

    def get_avrg_tstep(self, 
        env_selector: torch.Tensor = None):
        scale=self._n_envs if env_selector is None else env_selector.flatten().shape[0]
        return torch.sum(self.step_counters(env_selector=env_selector)).item()/scale  
    
    def get_min_tstep(self, 
        env_selector: torch.Tensor = None):
        return self.step_counters(env_selector=env_selector).min()
            
    def get_max_tstep(self, 
        env_selector: torch.Tensor = None):
        return self.step_counters(env_selector=env_selector).max()

if __name__ == "__main__":  

    def print_data(data,i):
        print(f"INFO{i}")
        print("max: ")
        print(test_data.get_max())
        print("avrg :")
        print(test_data.get_avrg())
        print("min: ")
        print(test_data.get_min())
        print(f"played eps {data.get_n_played_episodes()}")
        print("#################################")

    n_steps = 10
    n_envs = 2
    data_dim = 1
    ep_finished = torch.full((n_envs, n_steps),fill_value=False,dtype=torch.bool,device="cpu",
                requires_grad=False)
    new_data = torch.full((n_envs, data_dim),fill_value=0,dtype=torch.float32,device="cpu",
                requires_grad=False)
    data_scaling = torch.full((n_envs, 1),fill_value=1,dtype=torch.int32,device="cpu",
                requires_grad=False)
    data_names = ["data1"]
    test_data = EpisodicData("TestData",
                data_tensor=new_data,
                data_names=data_names,
                debug=True,
                ep_vec_freq=2)

    test_data.set_constant_data_scaling(enable=True,
                scaling=data_scaling)
    test_data.reset()
    
    ep_finished[0,  3] = True # term at tstep 
    ep_finished[0,  6] = True 


    new_data[:, 0] = 1

    print_freq=1
    for i in range(n_steps):# do some updates
        
        print("new data")
        print(new_data)
        print(ep_finished[:, i:i+1])
        test_data.update(new_data=new_data,
                    ep_finished=ep_finished[:, i:i+1])
        if (i+1)%print_freq==0:
            print_data(test_data,i)
    

    #****************MEM BUFFER********************
    # n_envs = 3
    # data_dim = 4
    # new_data = torch.full((n_envs, data_dim),fill_value=0,dtype=torch.float32,device="cuda",
                # requires_grad=False)
    # to_be_reset = torch.full((n_envs, 1),fill_value=False,dtype=torch.bool,device="cuda",
                # requires_grad=False)
    # data_names = ["okokok", "sdcsdc", "cdcsdcplpl","sacasca"]
    # new_data.fill_(1.0)
    # stds = torch.tensor([0.1, 0.2, 0.3, 0.4])  # Example standard deviations for each column

    # mem_buffer=MemBuffer(name="MemBProva",
    #         data_tensor=new_data,
    #         data_names=data_names,
    #         horizon=100000,
    #         dtype=torch.float32,
    #         use_gpu=True)
    
    # mem_buffer.reset(to_be_reset=to_be_reset.flatten())
    # # mem_buffer.reset(init_data=new_data)
    
    # for i in range(mem_buffer.horizon()+1):
    #     noise = torch.randn(n_envs, data_dim) * stds +1
    #     new_data = noise
    #     mem_buffer.update(new_data.cuda())
    #     # if i==(round(mem_buffer.horizon()/2)-1):
    #     #     to_be_reset[2,:]=True
    #     #     print(mem_buffer.get(idx=0)[2,:])
    #     #     mem_buffer.reset(to_be_reset=to_be_reset.flatten())
    #     #     print(mem_buffer.get(idx=0)[2,:])

    # print("pos")
    # print(mem_buffer.pos())
    # print("STD")
    # print(mem_buffer.std())
    # print("AVRG")
    # print(mem_buffer.mean())
    # print("AAAAA")
    # print(mem_buffer.horizon())

