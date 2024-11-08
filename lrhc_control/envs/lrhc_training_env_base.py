import torch
import math
from lrhc_control.utils.math_utils import quaternion_to_angular_velocity, quaternion_difference

from control_cluster_bridge.utilities.shared_data.rhc_data import RobotState
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcCmds, RhcPred
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcRefs
from control_cluster_bridge.utilities.shared_data.rhc_data import RhcStatus
from control_cluster_bridge.utilities.shared_data.sim_data import SharedEnvInfo
from control_cluster_bridge.utilities.shared_data.cluster_data import SharedClusterInfo

from lrhc_control.utils.shared_data.remote_stepping import RemoteStepperSrvr
from lrhc_control.utils.shared_data.remote_stepping import RemoteResetSrvr
from lrhc_control.utils.shared_data.remote_stepping import RemoteResetRequest

from lrhc_control.utils.shared_data.agent_refs import AgentRefs
from lrhc_control.utils.shared_data.training_env import SharedTrainingEnvInfo

from lrhc_control.utils.shared_data.training_env import Observations, NextObservations
from lrhc_control.utils.shared_data.training_env import TotRewards
from lrhc_control.utils.shared_data.training_env import SubRewards
from lrhc_control.utils.shared_data.training_env import Actions
from lrhc_control.utils.shared_data.training_env import Terminations, SubTerminations
from lrhc_control.utils.shared_data.training_env import Truncations, SubTruncations
from lrhc_control.utils.shared_data.training_env import EpisodesCounter,TaskRandCounter,SafetyRandResetsCounter

from lrhc_control.utils.episodic_rewards import EpisodicRewards
from lrhc_control.utils.episodic_data import EpisodicData
from lrhc_control.utils.episodic_data import MemBuffer
from lrhc_control.utils.math_utils import check_capsize
from control_cluster_bridge.utilities.math_utils_torch import world2base_frame, base2world_frame, w2hor_frame

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal

from perf_sleep.pyperfsleep import PerfSleep

from abc import abstractmethod, ABC

import os
from typing import List

class LRhcTrainingEnvBase(ABC):

    """Base class for a remote training environment tailored to Learning-based Receding Horizon Control"""

    def __init__(self,
            namespace: str,
            obs_dim: int,
            actions_dim: int,
            episode_timeout_lb: int,
            episode_timeout_ub: int,
            n_steps_task_rand_lb: int,
            n_steps_task_rand_ub: int,
            action_repeat: int = 1,
            env_name: str = "",
            n_preinit_steps: int = 1,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1,
            debug: bool = True,
            use_gpu: bool = True,
            dtype: torch.dtype = torch.float32,
            override_agent_refs: bool = False,
            timeout_ms: int = 60000,
            srew_drescaling: bool = True,
            use_act_mem_bf: bool = False,
            act_membf_size: int = 3,
            use_random_safety_reset: bool = True,
            random_reset_freq: int = None, # [n episodes]
            vec_ep_freq_metrics_db: int = 1):
        
        self._vec_ep_freq_metrics_db = vec_ep_freq_metrics_db # update single env metrics every
        # n episodes
        
        self._this_path = os.path.abspath(__file__)

        self._use_random_safety_reset = use_random_safety_reset

        self.custom_db_data = None
        
        self.custom_db_info = {}
        
        self._srew_drescaling = srew_drescaling
        
        self._action_repeat = action_repeat
        if self._action_repeat <=0: 
            self._action_repeat = 1
        self._substep_dt=1.0 # dt [s] between each substep
        try:
            self._substep_dt=self._env_opts["substep_dt"]
        except:
            pass
        sim_data = {}
        sim_info_shared = SharedEnvInfo(namespace=namespace,
                    is_server=False,
                    safe=False,
                    verbose=verbose,
                    vlevel=vlevel)
        sim_info_shared.run()
        sim_info_keys = sim_info_shared.param_keys
        sim_info_data = sim_info_shared.get().flatten()
        for i in range(len(sim_info_keys)):
            sim_data[sim_info_keys[i]] = sim_info_data[i]
        if "substepping_dt" in sim_info_keys:
            self._substep_dt=sim_data["substepping_dt"]

        self._use_act_mem_bf = use_act_mem_bf
        self._act_membf_size = round(act_membf_size/self._action_repeat) 

        self._closed = False
        self._ready=False
        self._override_agent_refs = override_agent_refs

        self._episode_timeout_lb = round(episode_timeout_lb/self._action_repeat) 
        self._episode_timeout_ub = round(episode_timeout_ub/self._action_repeat)

        self._n_steps_task_rand_lb = round(n_steps_task_rand_lb/self._action_repeat)
        self._n_steps_task_rand_ub = round(n_steps_task_rand_ub/self._action_repeat)
        
        self._random_rst_freq=random_reset_freq
        if self._random_rst_freq is None or self._random_rst_freq <=0:
            self._use_random_safety_reset=False
            self._random_rst_freq=-1

        self._namespace = namespace
        self._with_gpu_mirror = True
        self._safe_shared_mem = False
        
        self._obs_dim = obs_dim
        self._actions_dim = actions_dim

        self._use_gpu = use_gpu
        if self._use_gpu:
            self._device="cuda"
        else:
            self._device="cpu"

        self._dtype = dtype

        self._verbose = verbose
        self._vlevel = vlevel

        self._is_debug = debug

        self._env_name = env_name

        self._robot_state = None
        self._rhc_cmds = None
        self._rhc_pred = None
        self._rhc_refs = None
        self._rhc_status = None

        self._remote_stepper = None
        self._remote_resetter = None
        self._remote_reset_req = None

        self._agent_refs = None

        self._n_envs = 0

        self._n_preinit_steps = n_preinit_steps
        
        self._ep_timeout_counter = None
        self._task_rand_counter = None
        self._rand_safety_reset_counter = None

        self._obs = None
        self._next_obs = None
        self._actions = None
        self._actions_ub = None
        self._actions_lb = None
        self._tot_rewards = None
        self._sub_rewards = None
        self._sub_terminations = None
        self._sub_truncations = None
        self._terminations = None
        self._truncations = None
        self._act_mem_buffer = None

        self._episodic_rewards_metrics = None
        
        self._timeout = timeout_ms

        self._attach_to_shared_mem()

        self._init_obs()
        self._init_actions(actions_dim)
        self._init_rewards()
        self._init_terminations()
        self._init_truncations()
        self._init_infos()

        self._custom_post_init()

        self._ready=self._init_step()

    def __del__(self):

        self.close()

    def _get_this_file_path(self):
        return self._this_path
    
    def episode_timeout_bounds(self):
        return self._episode_timeout_lb, self._episode_timeout_ub
    
    def task_rand_timeout_bounds(self):
        return self._n_steps_task_rand_lb, self._n_steps_task_rand_ub
    
    def n_action_reps(self):
        return self._action_repeat
    
    def get_file_paths(self):
        from lrhc_control.utils.sys_utils import PathsGetter
        path_getter = PathsGetter()
        base_paths = []
        base_paths.append(self._get_this_file_path())
        base_paths.append(path_getter.REMOTENVPATH)
        for script_path in path_getter.SCRIPTSPATHS:
            base_paths.append(script_path)

        from SharsorIPCpp.PySharsorIPC import StringTensorClient
        from perf_sleep.pyperfsleep import PerfSleep
        shared_rhc_sahred_files = StringTensorClient(
            basename="SharedRhcFilesDropDir", 
            name_space=self._namespace,
            verbose=self._verbose, 
            vlevel=VLevel.V2)
        shared_rhc_sahred_files.run()
        shared_rhc_files_vals=[""]*shared_rhc_sahred_files.length()
        while not shared_rhc_sahred_files.read_vec(shared_rhc_files_vals, 0):
            nsecs =  1000000000 # 1 sec
            PerfSleep.thread_sleep(nsecs) # we just keep it alive
        rhc_list=[]
        for rhc_files in shared_rhc_files_vals:
            file_list = rhc_files.split(", ")
            rhc_list.extend(file_list)
        rhc_list = list(set(rhc_list)) # removing duplicates
        base_paths.extend(rhc_list)
        
        return base_paths

    def get_aux_dir(self):
        empty_list = []
        return empty_list
    
    def _init_step(self):
        
        self._check_controllers_registered(retry=True)
        self._activate_rhc_controllers()

        # just an auxiliary tensor
        initial_reset_aux = self._terminations.get_torch_mirror(gpu=self._use_gpu).clone()
        initial_reset_aux[:, :] = True # we reset all sim envs first
        init_step_ok=True
        init_step_ok=self._remote_sim_step() and init_step_ok
        if not init_step_ok:
            return False
        init_step_ok=self._remote_reset(reset_mask=initial_reset_aux) and init_step_ok
        if not init_step_ok:
            return False
            
        for i in range(self._n_preinit_steps): # perform some
            # dummy remote env stepping to make sure to have meaningful 
            # initializations (doesn't increment step counter)
            init_step_ok=self._remote_sim_step() and init_step_ok # 1 remote sim. step
            if not init_step_ok:
                return False
            init_step_ok=self._send_remote_reset_req() and init_step_ok # fake reset request 
            if not init_step_ok:
                return False
            
        self.reset()

        return init_step_ok

    def _debug(self):

        if self._use_gpu:
            # using non_blocking which is not safe when GPU->CPU
            self._obs.synch_mirror(from_gpu=True,non_blocking=True) # copy data from gpu to cpu view
            self._next_obs.synch_mirror(from_gpu=True,non_blocking=True)
            self._actions.synch_mirror(from_gpu=True,non_blocking=True)
            self._truncations.synch_mirror(from_gpu=True,non_blocking=True) 
            self._sub_truncations.synch_mirror(from_gpu=True,non_blocking=True)
            self._terminations.synch_mirror(from_gpu=True,non_blocking=True)
            self._sub_terminations.synch_mirror(from_gpu=True,non_blocking=True)
            self._tot_rewards.synch_mirror(from_gpu=True,non_blocking=True)
            self._sub_rewards.synch_mirror(from_gpu=True,non_blocking=True)
            if not self._override_agent_refs:
                self._agent_refs.rob_refs.root_state.synch_mirror(from_gpu=True,non_blocking=True) 
            # if we want reliable db data then we should synchronize data streams
            torch.cuda.synchronize()

        # copy CPU view on shared memory
        self._obs.synch_all(read=False, retry=True) 
        self._next_obs.synch_all(read=False, retry=True)
        self._actions.synch_all(read=False, retry=True) 
        self._tot_rewards.synch_all(read=False, retry=True)
        self._sub_rewards.synch_all(read=False, retry=True)
        self._truncations.synch_all(read=False, retry = True) 
        self._sub_truncations.synch_all(read=False, retry = True)
        self._terminations.synch_all(read=False, retry = True) 
        self._sub_terminations.synch_all(read=False, retry = True)
        if not self._override_agent_refs:
            self._agent_refs.rob_refs.root_state.synch_all(read=False, retry = True)

    def _remote_sim_step(self):

        self._remote_stepper.trigger() # triggers simulation + RHC
        if not self._remote_stepper.wait_ack_from(1, self._timeout):
            Journal.log(self.__class__.__name__,
            "_remote_sim_step",
            "Remote sim. env step ack not received within timeout",
            LogType.EXCEP,
            throw_when_excep = False)
            return False
        return True

    def _remote_reset(self,
                reset_mask: torch.Tensor = None):

        reset_reqs = self._remote_reset_req.get_torch_mirror()
        if reset_mask is None: # just send the signal to allow stepping, but do not reset any of
            # the remote envs
            reset_reqs[:, :] = False
        else:
            reset_reqs[:, :] = reset_mask # remotely reset envs corresponding to
            # the mask (True--> to be reset)
        self._remote_reset_req.synch_all(read=False, retry=True) # write on shared buffer
        remote_reset_ok = self._send_remote_reset_req() # process remote request

        if reset_mask is not None:
            self._synch_state(gpu=self._use_gpu) # if some env was reset, we use _obs
            # to hold the states, including resets, while _next_obs will always hold the 
            # state right after stepping the sim env
            # (could be a bit more efficient, since in theory we only need to read the envs
            # corresponding to the reset_mask)
            
            
        return remote_reset_ok
    
    def _send_remote_reset_req(self):

        self._remote_resetter.trigger()
        if not self._remote_resetter.wait_ack_from(1, self._timeout): # remote reset completed
            Journal.log(self.__class__.__name__,
                "_post_step",
                "Remote reset did not complete within the prescribed timeout!",
                LogType.EXCEP,
                throw_when_excep = False)
            return False
        return True
    
    def step(self, 
            action):

        # set action from agent
        actions = self._actions.get_torch_mirror(gpu=self._use_gpu)
        actions[:, :] = action.detach() # writes actions (detaching to avoid
        # grads prop
        
        self._pre_step()

        if self._act_mem_buffer is not None:
            self._act_mem_buffer.update(new_data=actions)

        self._apply_actions_to_rhc() # apply agent actions to rhc controller

        stepping_ok = True
        tot_rewards = self._tot_rewards.get_torch_mirror(gpu=self._use_gpu)
        sub_rewards = self._sub_rewards.get_torch_mirror(gpu=self._use_gpu)
        tot_rewards.zero_()
        sub_rewards.zero_()
        self._substep_rewards.zero_()

        for i in range(0, self._action_repeat): # remove env substepping
            stepping_ok = stepping_ok and self._check_controllers_registered(retry=False) # does not make sense to run training
            # if we lost some controllers
            stepping_ok = stepping_ok and self._remote_sim_step() # blocking, 

            # no sim substepping is allowed to fail
            self._synch_state(gpu=self._use_gpu) # read state from shared mem (done in substeps also, 
            # since substeps rewards will need updated substep obs)
            
            self._custom_post_substp_pre_rew() # custom substepping logic
            self._compute_substep_rewards()
            self._assemble_substep_rewards() # includes rewards clipping
            self._custom_post_substp_post_rew() # custom substepping logic

            if not i==(self._action_repeat-1):
                # just sends reset signal to complete remote step sequence,
                # but does not reset any remote env
                stepping_ok = stepping_ok and self._remote_reset(reset_mask=None) 
            else: # last substep
                next_obs = self._next_obs.get_torch_mirror(gpu=self._use_gpu)
                self._fill_obs(next_obs) # update next obs
                self._clamp_obs(next_obs) # good practice
                obs = self._obs.get_torch_mirror(gpu=self._use_gpu)
                obs[:, :] = next_obs # start from next observation, unless reset (handled in post_step())

                self._compute_step_rewards() # implemented by child

                tot_rewards = self._tot_rewards.get_torch_mirror(gpu=self._use_gpu)
                sub_rewards = self._sub_rewards.get_torch_mirror(gpu=self._use_gpu)
                self._clamp_rewards(sub_rewards) # clamp all sub rewards

                # compensating with action repeat to kee the same ret scale 
                tot_rewards[:, :] = torch.sum(sub_rewards, dim=1, keepdim=True)

                scale=1 # scale tot rew by the number of action repeats
                if self._srew_drescaling: # scale rewards depending on the n of subrewards
                    scale*=sub_rewards.shape[1] # n. dims rescaling
                tot_rewards.mul_(1/scale)

            if not stepping_ok:
                return False
            
        stepping_ok =  stepping_ok and self._post_step() # post sub-stepping operations
        # (if action_repeat > 1, then just the db data at the last substep is logged)
        # also, if a reset of an env occurs, obs will hold the reset state

        return stepping_ok 
    
    def _post_step(self):
        
        self._ep_timeout_counter.increment() # first increment counters
        self._task_rand_counter.increment()
        
        # check truncation and termination conditions 
        self._check_truncations() 
        self._check_terminations()
        terminated = self._terminations.get_torch_mirror(gpu=self._use_gpu)
        truncated = self._truncations.get_torch_mirror(gpu=self._use_gpu)
        
        episode_finished = torch.logical_or(terminated,
                            truncated)
        episode_finished_cpu = episode_finished.cpu()

        if self._rand_safety_reset_counter is not None:
            self._rand_safety_reset_counter.increment(to_be_incremented=episode_finished_cpu.flatten())
            # truncated[:,:] = torch.logical_or(truncated,
            #     self._rand_safety_reset_counter.time_limits_reached().cuda())

        if self._act_mem_buffer is not None:
            self._act_mem_buffer.reset(to_be_reset=episode_finished.flatten(),
                            init_data=self._defaut_act_buffer_action)

        # debug step if required (IMPORTANT: must be before remote reset so that we always db
        # actual data from the step and not after reset)
        if self._is_debug:
            self._debug() # copies db data on shared memory
            self._update_custom_db_data(episode_finished=episode_finished_cpu)
            self._episodic_rewards_metrics.update(rewards = self._sub_rewards.get_torch_mirror(gpu=False),
                            ep_finished=episode_finished_cpu)

        # remotely reset envs
        to_be_reset=self._to_be_reset()
        to_be_reset_custom=self._custom_reset()
        if to_be_reset_custom is not None:
            to_be_reset[:, :] = torch.logical_or(to_be_reset,to_be_reset_custom)
        rm_reset_ok = self._remote_reset(reset_mask=to_be_reset)
        
        self._custom_post_step(episode_finished=episode_finished) # any additional logic from child env  
        # here, before actual reset taskes place  (at this point the state is the reset one)

        # updating also prev pos and orientation in case some env was reset
        self._prev_root_p_substep[:, :]=self._robot_state.root_state.get(data_type="p",gpu=self._use_gpu)
        self._prev_root_q_substep[:, :]=self._robot_state.root_state.get(data_type="q",gpu=self._use_gpu)

        obs = self._obs.get_torch_mirror(gpu=self._use_gpu)
        self._fill_obs(obs)
        self._clamp_obs(obs)

        # updating prev step quantities
        self._prev_root_p_step[:, :]=self._robot_state.root_state.get(data_type="p",gpu=self._use_gpu)
        self._prev_root_q_step[:, :]=self._robot_state.root_state.get(data_type="q",gpu=self._use_gpu)

        # synchronize and reset counters for finished episodes
        self._ep_timeout_counter.reset(to_be_reset=episode_finished_cpu)
        self._task_rand_counter.reset(to_be_reset=episode_finished_cpu)
        # safety reset counter is only when it reches its reset interval (just to keep
        # the counter bounded)
        if self._rand_safety_reset_counter is not None:
            self._rand_safety_reset_counter.reset(to_be_reset=self._rand_safety_reset_counter.time_limits_reached())

        return rm_reset_ok
    
    def _to_be_reset(self):
        # always reset if a termination occurred or if there's a random safety reset
        # request
        terminated = self._terminations.get_torch_mirror(gpu=self._use_gpu)
        to_be_reset=terminated.cpu()

        if self._rand_safety_reset_counter is not None:
            to_be_reset[:, :]=torch.logical_or(to_be_reset,
                self._rand_safety_reset_counter.time_limits_reached())

        return to_be_reset

    def _custom_reset(self):
        # can be overridden by child
        return None
    
    def _update_custom_db_data(self,
                    episode_finished):

        # update defaults
        self.custom_db_data["RhcRefsFlag"].update(new_data=self._rhc_refs.contact_flags.get_torch_mirror(gpu=False), 
                                    ep_finished=episode_finished) # before potentially resetting the flags, get data
        self.custom_db_data["Actions"].update(new_data=self._actions.get_torch_mirror(gpu=False), 
                                    ep_finished=episode_finished)
        self.custom_db_data["SubTerminations"].update(new_data=self._sub_terminations.get_torch_mirror(gpu=False), 
                                    ep_finished=episode_finished)
        self.custom_db_data["SubTruncations"].update(new_data=self._sub_truncations.get_torch_mirror(gpu=False), 
                                    ep_finished=episode_finished)
        
        self._get_custom_db_data(episode_finished=episode_finished)

    def reset_custom_db_data(self, keep_track: bool = False):
        # to be called periodically to reset custom db data stat. collection 
        for custom_db_data in self.custom_db_data.values():
            custom_db_data.reset(keep_track=keep_track)

    def _assemble_substep_rewards(self):
        # by default assemble  substep rewards by averaging
        sub_rewards = self._sub_rewards.get_torch_mirror(gpu=self._use_gpu)
        
        # average over substeps depending on scale
        # sub_rewards[:, self._is_substep_rew] = sub_rewards[:, self._is_substep_rew] + \
        #     self._substep_rewards[:, self._is_substep_rew]/self._action_repeat
        sub_rewards[:, self._is_substep_rew] = sub_rewards[:, self._is_substep_rew] + \
            self._substep_rewards[:, self._is_substep_rew]/self._action_repeat
                    
    def randomize_task_refs(self,
                env_indxs: torch.Tensor = None):
                    
        if self._override_agent_refs:
            self._override_refs(gpu=self._use_gpu)
        else:
            self._randomize_task_refs(env_indxs=env_indxs)
            
    def reset(self):
        
        self.randomize_task_refs(env_indxs=None) # randomize all refs across envs

        self._obs.reset()
        self._actions.reset()
        self._next_obs.reset()
        self._sub_rewards.reset()
        self._tot_rewards.reset()
        self._terminations.reset()
        self._sub_terminations.reset()
        self._truncations.reset()
        self._sub_truncations.reset()

        self._ep_timeout_counter.reset(randomize_offsets=True)
        self._task_rand_counter.reset()
        self._task_rand_counter.sync_counters(other_counter=self._ep_timeout_counter)
        if self._rand_safety_reset_counter is not None:
            self._rand_safety_reset_counter.reset()

        if self._act_mem_buffer is not None:
            self._act_mem_buffer.reset_all(init_data=self._defaut_act_buffer_action)

        self._synch_state(gpu=self._use_gpu) # read obs from shared mem

        # just calling custom post step to ensure tak refs are updated 
        terminated = self._terminations.get_torch_mirror(gpu=self._use_gpu)
        truncated = self._truncations.get_torch_mirror(gpu=self._use_gpu)
        episode_finished = torch.logical_or(terminated,
                            truncated)
        self._custom_post_step(episode_finished=episode_finished)

        obs = self._obs.get_torch_mirror(gpu=self._use_gpu)
        next_obs = self._next_obs.get_torch_mirror(gpu=self._use_gpu)
        self._fill_obs(obs) # initialize observations 
        self._clamp_obs(obs) # to avoid bad things
        self._fill_obs(next_obs) # and next obs
        self._clamp_obs(next_obs)

        self.reset_custom_db_data(keep_track=False)
        self._episodic_rewards_metrics.reset(keep_track=False)

        self._prev_root_p_step[:, :]=self._robot_state.root_state.get(data_type="p",gpu=self._use_gpu)
        self._prev_root_q_step[:, :]=self._robot_state.root_state.get(data_type="q",gpu=self._use_gpu)
        self._prev_root_p_substep[:, :]=self._robot_state.root_state.get(data_type="p",gpu=self._use_gpu)
        self._prev_root_q_substep[:, :]=self._robot_state.root_state.get(data_type="q",gpu=self._use_gpu)

    def is_ready(self):
        return self._ready
    
    def close(self):
        
        if not self._closed:

            # close all shared mem. clients
            self._robot_state.close()
            self._rhc_cmds.close()
            self._rhc_pred.close()
            self._rhc_refs.close()
            self._rhc_status.close()
            
            self._remote_stepper.close()
            
            self._ep_timeout_counter.close()
            self._task_rand_counter.close()
            if self._rand_safety_reset_counter is not None:
                self._rand_safety_reset_counter.close()

            # closing env.-specific shared data
            self._obs.close()
            self._next_obs.close()
            self._actions.close()
            self._sub_rewards.close()
            self._tot_rewards.close()

            self._terminations.close()
            self._sub_terminations.close()
            self._truncations.close()
            self._sub_truncations.close()

            self._closed = True

    def get_obs(self, clone:bool=False):
        if clone:
            return self._obs.get_torch_mirror(gpu=self._use_gpu).clone().detach()
        else:
            return self._obs.get_torch_mirror(gpu=self._use_gpu).detach()

    def get_next_obs(self, clone:bool=False):
        if clone:
            return self._next_obs.get_torch_mirror(gpu=self._use_gpu).clone().detach()
        else:
            return self._next_obs.get_torch_mirror(gpu=self._use_gpu).detach()
        
    def get_actions(self, clone:bool=False):
        if clone:
            return self._actions.get_torch_mirror(gpu=self._use_gpu).clone().detach()
        else:
            return self._actions.get_torch_mirror(gpu=self._use_gpu).detach()
            
    def get_rewards(self, clone:bool=False):
        if clone:
            return self._tot_rewards.get_torch_mirror(gpu=self._use_gpu).clone().detach()
        else:
            return self._tot_rewards.get_torch_mirror(gpu=self._use_gpu).detach()
        
    def get_terminations(self, clone:bool=False):
        if clone:
            return self._terminations.get_torch_mirror(gpu=self._use_gpu).clone().detach()
        else:
            return self._terminations.get_torch_mirror(gpu=self._use_gpu).detach()
    
    def get_truncations(self, clone:bool=False):
        if clone:
            return self._truncations.get_torch_mirror(gpu=self._use_gpu).clone().detach()
        else:
            return self._truncations.get_torch_mirror(gpu=self._use_gpu).detach()
        
    def obs_dim(self):

        return self._obs_dim
    
    def actions_dim(self):

        return self._actions_dim
    
    def ep_rewards_metrics(self):

        return self._episodic_rewards_metrics
    
    def using_gpu(self):

        return self._use_gpu

    def name(self):

        return self._env_name

    def n_envs(self):

        return self._n_envs

    def dtype(self):
                                    
        return self._dtype 
    
    def obs_names(self):
        return self._get_obs_names()
    
    def action_names(self):
        return self._get_action_names()

    def sub_rew_names(self):
        return self._get_rewards_names()
    
    def sub_term_names(self):
        return self._get_sub_term_names()
    
    def sub_trunc_names(self):
        return self._get_sub_trunc_names()
    
    def _get_obs_names(self):
        # to be overridden by child class
        return None
    
    def get_robot_jnt_names(self):
        return self._robot_state.jnt_names()

    def _get_action_names(self):
        # to be overridden by child class
        return None
    
    def _get_rewards_names(self):
        # to be overridden by child class
        return None
    
    def _get_sub_term_names(self):
        # to be overridden by child class
        sub_term_names = []
        sub_term_names.append("rhc_failure")
        sub_term_names.append("robot_capsize")
        sub_term_names.append("rhc_capsize")

        return sub_term_names
    
    def _get_sub_trunc_names(self):
        # to be overridden by child class
        sub_trunc_names = []
        sub_trunc_names.append("ep_timeout")

        return sub_trunc_names
    
    def _get_custom_db_data(self, episode_finished):
        # to be overridden by child class
        pass
    
    def set_observed_joints(self):
        # ny default observe all joints available 
        return self._robot_state.jnt_names()
    
    def _set_jnts_blacklist_pattern(self):
        self._jnt_q_blacklist_patterns=[]

    def get_observed_joints(self):
        return self._observed_jnt_names
    
    def _init_obs(self):
        
        obs_threshold_default = 100.0
        self._obs_threshold_lb = -obs_threshold_default # used for clipping observations
        self._obs_threshold_ub = obs_threshold_default

        if not self._obs_dim==len(self._get_obs_names()):
            error=f"obs dim {self._obs_dim} does not match obs names length {len(self._get_obs_names())}!!"
            Journal.log(self.__class__.__name__,
                "_init_obs",
                error,
                LogType.EXCEP,
                throw_when_excep = True)

        self._obs = Observations(namespace=self._namespace,
                            n_envs=self._n_envs,
                            obs_dim=self._obs_dim,
                            obs_names=self._get_obs_names(),
                            env_names=None,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=self._use_gpu,
                            fill_value=0.0)
        
        self._next_obs = NextObservations(namespace=self._namespace,
                            n_envs=self._n_envs,
                            obs_dim=self._obs_dim,
                            obs_names=self._get_obs_names(),
                            env_names=None,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=self._use_gpu,
                            fill_value=0.0)

        self._obs.run()
        self._next_obs.run()
        
    def _init_actions(self, actions_dim: int):
        
        device = "cuda" if self._use_gpu else "cpu"
        # action scalings to be applied to agent's output
        self._actions_ub = torch.full((1, actions_dim), dtype=self._dtype, device=device,
                                        fill_value=1.0) 
        self._actions_lb = torch.full((1, actions_dim), dtype=self._dtype, device=device,
                                        fill_value=-1.0)
        self._actions_scale = (self._actions_ub - self._actions_lb)/2.0
        self._actions_offset = (self._actions_ub + self._actions_lb)/2.0

        if not self._actions_dim==len(self._get_action_names()):
            error=f"action dim {self._actions_dim} does not match action names length {len(self._get_action_names())}!!"
            Journal.log(self.__class__.__name__,
                "_init_actions",
                error,
                LogType.EXCEP,
                throw_when_excep = True)
        self._actions = Actions(namespace=self._namespace,
                            n_envs=self._n_envs,
                            action_dim=self._actions_dim,
                            action_names=self._get_action_names(),
                            env_names=None,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=self._use_gpu,
                            fill_value=0.0)

        self._actions.run()

        if self._use_act_mem_bf:
            self._act_mem_buffer=MemBuffer(name="ActionMemBuf",
                data_tensor=self._actions.get_torch_mirror(),
                data_names=self._get_action_names(),
                debug=self._debug,
                horizon=self._act_membf_size,
                dtype=self._dtype,
                use_gpu=self._use_gpu)
            self._defaut_act_buffer_action = torch.full_like(input=self.get_actions(),fill_value=0.0)

        self._random_uniform = torch.full_like(self._actions.get_torch_mirror(gpu=self._use_gpu), fill_value=0.0) # used for sampling random actions (preallocated
        # for efficiency)
        self._random_normal = torch.full_like(self._actions.get_torch_mirror(gpu=self._use_gpu), fill_value=0.0) # used for sampling random actions from a gaussian distr (preallocated
        # for efficiency)

        # default to all continuous actions (changes the way noise is added)
        self._is_continuous_actions=torch.full((1, actions_dim), 
            dtype=torch.bool, device=device,
            fill_value=True) 

    def _init_rewards(self):
        
        reward_thresh_default = 1.0
        n_sub_rewards = len(self._get_rewards_names())
        device = "cuda" if self._use_gpu else "cpu"
        self._reward_thresh_lb = torch.full((1, n_sub_rewards), dtype=self._dtype, fill_value=-reward_thresh_default, device=device) # used for clipping rewards
        self._reward_thresh_ub = torch.full((1, n_sub_rewards), dtype=self._dtype, fill_value=reward_thresh_default, device=device) 

        self._sub_rewards = SubRewards(namespace=self._namespace,
                            n_envs=self._n_envs,
                            n_rewards=n_sub_rewards,
                            reward_names=self._get_rewards_names(),
                            env_names=None,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=self._use_gpu,
                            fill_value=0.0)
        
        self._tot_rewards = TotRewards(namespace=self._namespace,
                            n_envs=self._n_envs,
                            reward_names=["total_reward"],
                            env_names=None,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=self._use_gpu,
                            fill_value=0.0)
        
        self._sub_rewards.run()
        self._tot_rewards.run()

        self._substep_rewards = self._sub_rewards.get_torch_mirror(gpu=self._use_gpu).detach().clone() 
        # used to hold substep rewards (not super mem. efficient)
        self._is_substep_rew = torch.zeros((self._substep_rewards.shape[1],),dtype=torch.bool,device=device)
        self._is_substep_rew.fill_(True) # default to all substep rewards
        self._episodic_rewards_metrics = EpisodicRewards(reward_tensor=self._sub_rewards.get_torch_mirror(),
                                        reward_names=self._get_rewards_names(),
                                        max_episode_length=self._episode_timeout_ub,
                                        ep_vec_freq=self._vec_ep_freq_metrics_db)
        self._episodic_rewards_metrics.set_constant_data_scaling(scaling=self._get_reward_scaling())
        
    def _get_reward_scaling(self):
        # to be overridden by child (default to no scaling)
        return 1
        
    def _init_infos(self):

        self.custom_db_data = {}
        # by default always log this contact data
        rhc_latest_contact_ref = self._rhc_refs.contact_flags.get_torch_mirror()
        contact_names = self._rhc_refs.rob_refs.contact_names()
        stepping_data = EpisodicData("RhcRefsFlag", rhc_latest_contact_ref, contact_names,
            ep_vec_freq=self._vec_ep_freq_metrics_db)
        self._add_custom_db_info(db_data=stepping_data)
        # log also action data
        actions = self._actions.get_torch_mirror()
        action_names = self._get_action_names()
        action_data = EpisodicData("Actions", actions, action_names,
            ep_vec_freq=self._vec_ep_freq_metrics_db)
        self._add_custom_db_info(db_data=action_data)

        # log sub-term and sub-truncations data
        t_scaling=1 # 1 so that we log an interpretable data in terms of why the episode finished
        data_scaling = torch.full((self._n_envs, 1),
                    fill_value=t_scaling,
                    dtype=torch.int32,device="cpu")
        sub_term = self._sub_terminations.get_torch_mirror()
        sub_term_names = self.sub_term_names()
        sub_term_data = EpisodicData("SubTerminations", sub_term, sub_term_names,
            ep_vec_freq=self._vec_ep_freq_metrics_db)
        sub_term_data.set_constant_data_scaling(enable=True,scaling=data_scaling)
        self._add_custom_db_info(db_data=sub_term_data)
        sub_trunc = self._sub_truncations.get_torch_mirror()
        sub_trunc_names = self.sub_trunc_names()
        sub_trunc_data = EpisodicData("SubTruncations", sub_trunc, sub_trunc_names,
            ep_vec_freq=self._vec_ep_freq_metrics_db)
        sub_trunc_data.set_constant_data_scaling(enable=True,scaling=data_scaling)
        self._add_custom_db_info(db_data=sub_trunc_data)

    def _add_custom_db_info(self, db_data: EpisodicData):
        self.custom_db_data[db_data.name()] = db_data

    def _init_terminations(self):

        # Boolean array indicating whether each environment episode has terminated after 
        # the current step. An episode termination could occur based on predefined conditions
        # in the environment, such as reaching a goal or exceeding a time limit.

        self._terminations = Terminations(namespace=self._namespace,
                            n_envs=self._n_envs,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=self._use_gpu,
                            fill_value=False) 
        self._terminations.run()

        sub_t_names = self.sub_term_names()
        n_sub_term = 1 if sub_t_names is None else len(sub_t_names)
        self._sub_terminations = SubTerminations(namespace=self._namespace,
                n_envs=self._n_envs,
                n_term=n_sub_term,
                term_names=sub_t_names,
                is_server=True,
                verbose=self._verbose,
                vlevel=self._vlevel,
                safe=True,
                force_reconnection=True,
                with_gpu_mirror=self._use_gpu,
                fill_value=False)
        self._sub_terminations.run()

        device = "cuda" if self._use_gpu else "cpu"
        self._is_capsized=torch.zeros((self._n_envs,1), 
            dtype=torch.bool, device=device)
        self._is_rhc_capsized=torch.zeros((self._n_envs,1), 
            dtype=torch.bool, device=device)
        self._max_pitch_angle=60.0*math.pi/180.0
    
    def _init_truncations(self):
        
        self._truncations = Truncations(namespace=self._namespace,
                            n_envs=self._n_envs,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=self._use_gpu,
                            fill_value=False) 
        
        self._truncations.run()

        sub_trc_names = self.sub_trunc_names()
        n_sub_trunc = 1 if sub_trc_names is None else len(sub_trc_names)

        self._sub_truncations = SubTruncations(namespace=self._namespace,
                n_envs=self._n_envs,
                n_trunc=n_sub_trunc,
                truc_names=sub_trc_names,
                is_server=True,
                verbose=self._verbose,
                vlevel=self._vlevel,
                safe=True,
                force_reconnection=True,
                with_gpu_mirror=self._use_gpu,
                fill_value=False)
        self._sub_truncations.run()
    
    def _update_jnt_blacklist(self):
        device = "cuda" if self._use_gpu else "cpu"
        all_available_jnts=self.get_observed_joints()        
        blacklist=[]
        for i in range(len(all_available_jnts)):
            for pattern in self._jnt_q_blacklist_patterns:
                if pattern in all_available_jnts[i]:
                    # stop at first pattern match
                    blacklist.append(i)
                    break
        if not len(blacklist)==0:
            self._jnt_q_blacklist_idxs=torch.tensor(blacklist, dtype=torch.int, device=device)
            
    def _attach_to_shared_mem(self):

        # runs shared mem clients for getting observation and setting RHC commands

        # remote stepping data
        self._remote_stepper = RemoteStepperSrvr(namespace=self._namespace,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            force_reconnection=True)
        self._remote_stepper.run()
        self._remote_resetter = RemoteResetSrvr(namespace=self._namespace,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            force_reconnection=True)
        self._remote_resetter.run()
        self._remote_reset_req = RemoteResetRequest(namespace=self._namespace,
                                            is_server=False, 
                                            verbose=self._verbose,
                                            vlevel=self._vlevel,
                                            safe=True)
        self._remote_reset_req.run()

        self._jnts_remapping=None
        self._jnt_q_blacklist_idxs=None

        self._robot_state = RobotState(namespace=self._namespace,
                                is_server=False, 
                                safe=self._safe_shared_mem,
                                verbose=self._verbose,
                                vlevel=self._vlevel,
                                with_gpu_mirror=self._use_gpu,
                                with_torch_view=True)
        
        self._rhc_cmds = RhcCmds(namespace=self._namespace,
                                is_server=False, 
                                safe=self._safe_shared_mem,
                                verbose=self._verbose,
                                vlevel=self._vlevel,
                                with_gpu_mirror=self._use_gpu,
                                with_torch_view=True)
        
        self._rhc_pred = RhcPred(namespace=self._namespace,
                                is_server=False, 
                                safe=self._safe_shared_mem,
                                verbose=self._verbose,
                                vlevel=self._vlevel,
                                with_gpu_mirror=self._use_gpu,
                                with_torch_view=True)

        self._rhc_refs = RhcRefs(namespace=self._namespace,
                            is_server=False,
                            safe=self._safe_shared_mem,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            with_gpu_mirror=self._use_gpu,
                            with_torch_view=True)

        self._rhc_status = RhcStatus(namespace=self._namespace,
                                is_server=False,
                                verbose=self._verbose,
                                vlevel=self._vlevel,
                                with_gpu_mirror=self._use_gpu,
                                with_torch_view=True)
        
        self._robot_state.run()
        self._n_envs = self._robot_state.n_robots()
        self._n_jnts = self._robot_state.n_jnts()
        self._n_contacts = self._robot_state.n_contacts() # we assume same n contacts for all rhcs for now

        self._rhc_cmds.run()
        self._rhc_pred.run()
        self._rhc_refs.run()
        self._rhc_status.run()
        # we read rhc info now and just this time, since it's assumed to be static 
        self._check_controllers_registered(retry=True) # blocking
        # (we need controllers to be connected to read meaningful data)

        self._rhc_status.rhc_static_info.synch_all(read=True,retry=True)
        if self._use_gpu:
            self._rhc_status.rhc_static_info.synch_mirror(from_gpu=False,non_blocking=False)
        rhc_horizons=self._rhc_status.rhc_static_info.get("horizons",gpu=self._use_gpu)
        rhc_nnodes=self._rhc_status.rhc_static_info.get("nnodes",gpu=self._use_gpu)
        rhc_dts=self._rhc_status.rhc_static_info.get("dts",gpu=self._use_gpu)
        rhc_ncontacts=self._rhc_status.rhc_static_info.get("ncontacts",gpu=self._use_gpu)
        robot_mass=self._rhc_status.rhc_static_info.get("robot_mass",gpu=self._use_gpu)
        pred_node_idxs_rhc=self._rhc_status.rhc_static_info.get("pred_node_idx",gpu=self._use_gpu)

        self._n_nodes_rhc=torch.round(rhc_nnodes) # we assume nodes are static during an env lifetime
        self._rhc_horizons=rhc_horizons
        self._rhc_dts=rhc_dts
        self._n_contacts_rhc=rhc_ncontacts
        self._rhc_robot_masses=robot_mass
        if (self._rhc_robot_masses == 0).any():
            zero_indices = torch.nonzero(self._rhc_robot_masses == 0, as_tuple=True)
            print(zero_indices)  # This will print the indices of zero elements
            Journal.log(self.__class__.__name__,
                "_attach_to_shared_mem",
                "Found at least one robot with 0 mass from RHC static info!!",
                LogType.EXCEP,
                throw_when_excep=True)

        self._rhc_robot_weight=robot_mass*9.81
        self._pred_node_idxs_rhc=pred_node_idxs_rhc
        self._pred_horizon_rhc=self._pred_node_idxs_rhc*self._rhc_dts

        # run server for agent commands
        self._agent_refs = AgentRefs(namespace=self._namespace,
                                is_server=True,
                                n_robots=self._n_envs,
                                n_jnts=self._robot_state.n_jnts(),
                                n_contacts=self._robot_state.n_contacts(),
                                contact_names=self._robot_state.contact_names(),
                                q_remapping=None,
                                with_gpu_mirror=self._use_gpu,
                                force_reconnection=True,
                                safe=False,
                                verbose=self._verbose,
                                vlevel=self._vlevel,
                                fill_value=0)
        self._agent_refs.run()

        # episode steps counters (for detecting episode truncations for 
        # time limits) 
        self._ep_timeout_counter = EpisodesCounter(namespace=self._namespace,
                            n_envs=self._n_envs,
                            n_steps_lb=self._episode_timeout_lb,
                            n_steps_ub=self._episode_timeout_ub,
                            randomize_offsets_at_startup=True,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=False) # handles step counter through episodes and through envs
        self._ep_timeout_counter.run()
        self._task_rand_counter = TaskRandCounter(namespace=self._namespace,
                            n_envs=self._n_envs,
                            n_steps_lb=self._n_steps_task_rand_lb,
                            n_steps_ub=self._n_steps_task_rand_ub,
                            randomize_offsets_at_startup=False,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=False) # handles step counter through episodes and through envs
        self._task_rand_counter.run()
        self._task_rand_counter.sync_counters(other_counter=self._ep_timeout_counter)
        if self._use_random_safety_reset:
            self._rand_safety_reset_counter=SafetyRandResetsCounter(namespace=self._namespace,
                            n_envs=self._n_envs,
                            n_steps_lb=self._random_rst_freq,
                            n_steps_ub=self._random_rst_freq,
                            randomize_offsets_at_startup=False,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=False)
            self._rand_safety_reset_counter.run()
            # self._rand_safety_reset_counter.sync_counters(other_counter=self._ep_timeout_counter)
            
        # debug data servers
        traing_env_param_dict = {}
        traing_env_param_dict["use_gpu"] = self._use_gpu
        traing_env_param_dict["debug"] = self._is_debug
        traing_env_param_dict["n_preinit_steps"] = self._n_preinit_steps
        traing_env_param_dict["n_preinit_steps"] = self._n_envs
        
        self._training_sim_info = SharedTrainingEnvInfo(namespace=self._namespace,
                is_server=True, 
                training_env_params_dict=traing_env_param_dict,
                safe=False,
                force_reconnection=True,
                verbose=self._verbose,
                vlevel=self._vlevel)
        self._training_sim_info.run()

        self._observed_jnt_names=self.set_observed_joints()
        self._set_jnts_blacklist_pattern()
        self._update_jnt_blacklist()

        self._prev_root_p_substep=self._robot_state.root_state.get(data_type="p",gpu=self._use_gpu).clone()
        self._prev_root_q_substep=self._robot_state.root_state.get(data_type="q",gpu=self._use_gpu).clone()
        self._prev_root_p_step=self._robot_state.root_state.get(data_type="p",gpu=self._use_gpu).clone()
        self._prev_root_q_step=self._robot_state.root_state.get(data_type="q",gpu=self._use_gpu).clone()
        
    def _activate_rhc_controllers(self):
        self._rhc_status.activation_state.get_torch_mirror()[:, :] = True
        self._rhc_status.activation_state.synch_all(read=False, retry=True) # activates all controllers
    
    def _synch_state(self,
            gpu: bool = True):

        # read from shared memory on CPU
        # robot state
        self._robot_state.root_state.synch_all(read = True, retry = True)
        self._robot_state.jnts_state.synch_all(read = True, retry = True)
        # rhc cmds
        self._rhc_cmds.root_state.synch_all(read = True, retry = True)
        self._rhc_cmds.jnts_state.synch_all(read = True, retry = True)
        self._rhc_cmds.contact_wrenches.synch_all(read = True, retry = True)
        # rhc pred
        self._rhc_pred.root_state.synch_all(read = True, retry = True)
        # self._rhc_pred.jnts_state.synch_all(read = True, retry = True)
        # self._rhc_pred.contact_wrenches.synch_all(read = True, retry = True)
        # refs for root link and contacts
        self._rhc_refs.rob_refs.root_state.synch_all(read = True, retry = True)
        self._rhc_refs.contact_flags.synch_all(read = True, retry = True)
        self._rhc_refs.flight_info.synch_all(read = True, retry = True)
        self._rhc_refs.rob_refs.contact_pos.synch_all(read = True, retry = True)
        # rhc cost
        self._rhc_status.rhc_cost.synch_all(read = True, retry = True)
        # rhc constr. violations
        self._rhc_status.rhc_constr_viol.synch_all(read = True, retry = True)
        # failure states
        self._rhc_status.fails.synch_all(read = True, retry = True)
        # tot cost and cnstr viol on nodes + step variable
        self._rhc_status.rhc_nodes_cost.synch_all(read = True, retry = True)
        self._rhc_status.rhc_nodes_constr_viol.synch_all(read = True, retry = True)
        self._rhc_status.rhc_fcn.synch_all(read = True, retry = True)
        self._rhc_status.rhc_fail_idx.synch_all(read = True, retry = True)
        if gpu:
            # copies data to "mirror" on GPU --> we can do it non-blocking since
            # in this direction it should be safe
            self._robot_state.root_state.synch_mirror(from_gpu=False,non_blocking=True) # copies shared data on GPU
            self._robot_state.jnts_state.synch_mirror(from_gpu=False,non_blocking=True)
            self._rhc_cmds.root_state.synch_mirror(from_gpu=False,non_blocking=True)
            self._rhc_cmds.jnts_state.synch_mirror(from_gpu=False,non_blocking=True)
            self._rhc_cmds.contact_wrenches.synch_mirror(from_gpu=False,non_blocking=True)
            self._rhc_pred.root_state.synch_mirror(from_gpu=False,non_blocking=True)
            # self._rhc_pred.jnts_state.synch_mirror(from_gpu=False,non_blocking=True)
            # self._rhc_pred.contact_wrenches.synch_mirror(from_gpu=False,non_blocking=True)
            self._rhc_refs.rob_refs.root_state.synch_mirror(from_gpu=False,non_blocking=True)
            self._rhc_refs.contact_flags.synch_mirror(from_gpu=False,non_blocking=True)
            self._rhc_refs.rob_refs.contact_pos.synch_mirror(from_gpu=False,non_blocking=True)
            self._rhc_refs.flight_info.synch_mirror(from_gpu=False,non_blocking=True)
            self._rhc_status.rhc_cost.synch_mirror(from_gpu=False,non_blocking=True)
            self._rhc_status.rhc_constr_viol.synch_mirror(from_gpu=False,non_blocking=True)
            self._rhc_status.fails.synch_mirror(from_gpu=False,non_blocking=True)
            self._rhc_status.rhc_nodes_cost.synch_mirror(from_gpu=False,non_blocking=True)
            self._rhc_status.rhc_nodes_constr_viol.synch_mirror(from_gpu=False,non_blocking=True)
            self._rhc_status.rhc_fcn.synch_mirror(from_gpu=False,non_blocking=True)
            self._rhc_status.rhc_fail_idx.synch_mirror(from_gpu=False,non_blocking=True)
            torch.cuda.synchronize() # ensuring that all the streams on the GPU are completed \
            # before the CPU continues execution
    
    def _override_refs(self,
            gpu=True):

        # just used for setting agent refs externally (i.e. from shared mem on CPU)
        self._agent_refs.rob_refs.root_state.synch_all(read=True,retry=True) # first read from mem
        if gpu:
            # copies latest refs to GPU 
            self._agent_refs.rob_refs.root_state.synch_mirror(from_gpu=False,non_blocking=False) 

    def _clamp_obs(self, 
            obs: torch.Tensor):
        if self._is_debug:
            self._check_finite(obs, "observations", False)
        torch.nan_to_num(input=obs, out=obs, nan=self._obs_threshold_ub, 
            posinf=self._obs_threshold_ub, 
            neginf=self._obs_threshold_lb) # prevent nans

        obs.clamp_(self._obs_threshold_lb, self._obs_threshold_ub)
    
    def _clamp_rewards(self, 
            rewards: torch.Tensor):
        if self._is_debug:
            self._check_finite(rewards, "rewards", False)
        torch.nan_to_num(input=rewards, out=rewards, nan=0.0, 
            posinf=None, 
            neginf=None) # prevent nans
        rewards.clamp_(self._reward_thresh_lb, self._reward_thresh_ub)

    def get_actions_lb(self):
        return self._actions_lb

    def get_actions_ub(self):
        return self._actions_ub
    
    def get_actions_scale(self):
        return self._actions_scale
    
    def get_actions_offset(self):
        return self._actions_offset
    
    def set_jnts_remapping(self, 
        remapping: List = None):

        self._jnts_remapping=remapping
        if self._jnts_remapping is not None:
            self._robot_state.set_jnts_remapping(jnts_remapping=self._jnts_remapping)
            self._rhc_cmds.set_jnts_remapping(jnts_remapping=self._jnts_remapping)
            self._rhc_pred.set_jnts_remapping(jnts_remapping=self._jnts_remapping)
            # we need to also update the list of observed joints to match
            available_joints=self._robot_state.jnt_names()
            # the remapping ordering 
            self._observed_jnt_names=[]
            for i in range(len(available_joints)):
                self._observed_jnt_names.append(available_joints[self._jnts_remapping[i]])

            self._update_jnt_blacklist()

            updated_obs_names=self._get_obs_names() # get updated obs names (should use get_observed_joints
            # internally, so that jnt names are updated)

            # also update jnt obs names on shared memory
            names_old=self._obs.get_obs_names()
            names_old_next=self._next_obs.get_obs_names()
            names_old[:]=updated_obs_names
            names_old_next[:]=updated_obs_names
            self._obs.update_names()
            self._next_obs.update_names()

    def _check_finite(self, 
                tensor: torch.Tensor,
                name: str, 
                throw: bool = False):
        if not torch.isfinite(tensor).all().item():
            exception = f"Found nonfinite elements in {name} tensor!!"            
            non_finite_idxs=torch.nonzero(~torch.isfinite(tensor))
            n_nonf_elems=non_finite_idxs.shape[0]

            if name=="observations":
                for i in range(n_nonf_elems):
                    db_msg=f"{self.obs_names()[non_finite_idxs[i,1]]} (env. {non_finite_idxs[i,0]}):" + \
                        f" {tensor[non_finite_idxs[i,0],non_finite_idxs[i,1]].item()}"
                    print(db_msg)
            if name=="rewards":
                for i in range(n_nonf_elems):
                    db_msg=f"{self.sub_rew_names()[non_finite_idxs[i,1]]} (env. {non_finite_idxs[i,0]}):" + \
                        f" {tensor[non_finite_idxs[i,0],non_finite_idxs[i,1]].item()}"
                    print(db_msg)
            print(tensor)
            Journal.log(self.__class__.__name__,
                "_check_finite",
                exception,
                LogType.EXCEP,
                throw_when_excep = throw)
            
    def _check_controllers_registered(self, 
                retry: bool = False):

        if retry:
            self._rhc_status.controllers_counter.synch_all(read=True, retry=True)
            n_connected_controllers = self._rhc_status.controllers_counter.get_torch_mirror()[0, 0].item()
            while not (n_connected_controllers == self._n_envs):
                warn = f"Expected {self._n_envs} controllers to be connected during training, " + \
                    f"but got {n_connected_controllers}. Will wait for all to be connected..."
                Journal.log(self.__class__.__name__,
                    "_check_controllers_registered",
                    warn,
                    LogType.WARN,
                    throw_when_excep = False)
                nsecs = int(2 * 1000000000)
                PerfSleep.thread_sleep(nsecs) 
                self._rhc_status.controllers_counter.synch_all(read=True, retry=True)
                n_connected_controllers = self._rhc_status.controllers_counter.get_torch_mirror()[0, 0].item()
            info = f"All {n_connected_controllers} controllers connected!"
            Journal.log(self.__class__.__name__,
                "_check_controllers_registered",
                info,
                LogType.INFO,
                throw_when_excep = False)
            return True
        else:
            self._rhc_status.controllers_counter.synch_all(read=True, retry=True)
            n_connected_controllers = self._rhc_status.controllers_counter.get_torch_mirror()[0, 0].item()
            if not (n_connected_controllers == self._n_envs):
                exception = f"Expected {self._n_envs} controllers to be connected during training, " + \
                    f"but got {n_connected_controllers}. Aborting..."
                Journal.log(self.__class__.__name__,
                    "_check_controllers_registered",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = False)
                return False
            return True
    
    def _check_truncations(self):
        
        self._check_sub_truncations()
        sub_truncations = self._sub_truncations.get_torch_mirror(gpu=self._use_gpu)
        truncations = self._truncations.get_torch_mirror(gpu=self._use_gpu)
        truncations[:, :] = torch.any(sub_truncations,dim=1,keepdim=True)

    def _check_terminations(self):
        
        self._check_sub_terminations()
        sub_terminations = self._sub_terminations.get_torch_mirror(gpu=self._use_gpu)
        terminations = self._terminations.get_torch_mirror(gpu=self._use_gpu)
        terminations[:, :] = torch.any(sub_terminations,dim=1,keepdim=True)

    def _check_sub_truncations(self):
        # default behaviour-> to be overriden by child
        sub_truncations = self._sub_truncations.get_torch_mirror(gpu=self._use_gpu)
        sub_truncations[:, 0:1]=self._ep_timeout_counter.time_limits_reached()

    def _check_sub_terminations(self):
        # default behaviour-> to be overriden by child
        sub_terminations = self._sub_terminations.get_torch_mirror(gpu=self._use_gpu)
        robot_q_meas = self._robot_state.root_state.get(data_type="q",gpu=self._use_gpu)
        robot_q_pred = self._rhc_cmds.root_state.get(data_type="q",gpu=self._use_gpu)

        # terminate when either the real robot or the prediction from the MPC are capsized
        check_capsize(quat=robot_q_meas,max_angle=self._max_pitch_angle,
            output_t=self._is_capsized)
        check_capsize(quat=robot_q_pred,max_angle=self._max_pitch_angle,
            output_t=self._is_rhc_capsized)
        
        sub_terminations[:, 0:1] = self._rhc_status.fails.get_torch_mirror(gpu=self._use_gpu)
        sub_terminations[:, 1:2] = self._is_capsized
        sub_terminations[:, 2:3] = self._is_rhc_capsized

    def is_action_continuous(self):
        return self._is_continuous_actions
    
    @abstractmethod
    def _pre_step(self):
        pass
    
    @abstractmethod
    def _custom_post_step(self,episode_finished):
        pass
    
    @abstractmethod
    def _custom_post_substp_post_rew(self):
        pass
    
    @abstractmethod
    def _custom_post_substp_pre_rew(self):
        pass

    @abstractmethod
    def _apply_actions_to_rhc(self):
        pass

    @abstractmethod
    def _compute_substep_rewards(self):
        pass

    @abstractmethod
    def _compute_step_rewards(self):
        pass

    @abstractmethod
    def _fill_obs(self,
            obs: torch.Tensor):
        pass

    @abstractmethod
    def _randomize_task_refs(self,
                env_indxs: torch.Tensor = None):
        pass

    def _custom_post_init(self):
        pass
    
    def _get_avrg_substep_root_twist(self, 
            out: torch.Tensor,
            base_loc: bool = True):
        # to be called at each substep
        robot_p_meas = self._robot_state.root_state.get(data_type="p",gpu=self._use_gpu)
        robot_q_meas = self._robot_state.root_state.get(data_type="q",gpu=self._use_gpu)

        root_v_avrg_w=(robot_p_meas-self._prev_root_p_substep)/self._substep_dt
        root_omega_avrg_w=quaternion_to_angular_velocity(q_diff=quaternion_difference(self._prev_root_q_substep,robot_q_meas),\
            dt=self._substep_dt)
        twist_w=torch.cat((root_v_avrg_w, 
            root_omega_avrg_w), 
            dim=1)
        if not base_loc:
            self._prev_root_p_substep[:, :]=robot_p_meas
            self._prev_root_q_substep[:, :]=robot_q_meas
            out[:, :]=twist_w
        world2base_frame(t_w=twist_w, q_b=self._prev_root_q_substep, t_out=out)
        self._prev_root_p_substep[:, :]=robot_p_meas
        self._prev_root_q_substep[:, :]=robot_q_meas

    def _get_avrg_step_root_twist(self, 
            out: torch.Tensor,
            base_loc: bool = True):
        # to be called after substeps of actions repeats
        robot_p_meas = self._robot_state.root_state.get(data_type="p",gpu=self._use_gpu)
        robot_q_meas = self._robot_state.root_state.get(data_type="q",gpu=self._use_gpu)

        dt=self._substep_dt*self._action_repeat # accounting for frame skipping
        root_v_avrg_w=(robot_p_meas-self._prev_root_p_step)/(dt)
        root_omega_avrg_w=quaternion_to_angular_velocity(q_diff=quaternion_difference(self._prev_root_q_step,robot_q_meas),\
            dt=dt)
        twist_w=torch.cat((root_v_avrg_w, 
            root_omega_avrg_w), 
            dim=1)
        if not base_loc:
            out[:, :]=twist_w
        world2base_frame(t_w=twist_w, q_b=self._prev_root_q_step, t_out=out)

    def _get_avrg_rhc_root_twist(self,
            out: torch.Tensor,
            base_loc: bool = True):
        
        rhc_root_p =self._rhc_cmds.root_state.get(data_type="p",gpu=self._use_gpu)
        rhc_root_q =self._rhc_cmds.root_state.get(data_type="q",gpu=self._use_gpu)
        rhc_root_p_pred =self._rhc_pred.root_state.get(data_type="p",gpu=self._use_gpu)
        rhc_root_q_pred =self._rhc_pred.root_state.get(data_type="q",gpu=self._use_gpu)

        rhc_root_v_avrg_rhc_w=(rhc_root_p_pred-rhc_root_p)/self._pred_horizon_rhc
        rhc_root_omega_avrg_rhc_w=quaternion_to_angular_velocity(q_diff=quaternion_difference(rhc_root_q,rhc_root_q_pred),\
            dt=self._pred_horizon_rhc)
    
        rhc_pred_avrg_twist_rhc_w = torch.cat((rhc_root_v_avrg_rhc_w, 
            rhc_root_omega_avrg_rhc_w), 
            dim=1)
        if not base_loc:
            out[:, :]=rhc_pred_avrg_twist_rhc_w
        # to rhc base frame (using first node as reference)
        world2base_frame(t_w=rhc_pred_avrg_twist_rhc_w, q_b=rhc_root_q, t_out=out)
