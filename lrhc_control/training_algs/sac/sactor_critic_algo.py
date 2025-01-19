from lrhc_control.agents.sactor_critic.sac import SACAgent

from lrhc_control.utils.shared_data.algo_infos import SharedRLAlgorithmInfo, QfVal, QfTrgt
from lrhc_control.utils.shared_data.training_env import SubReturns, TotReturns

import torch 
import torch.optim as optim
import torch.nn as nn

import random
import math
from typing import Dict

import os
import shutil

import time

import wandb
import h5py
import numpy as np

from EigenIPC.PyEigenIPC import LogType
from EigenIPC.PyEigenIPC import Journal
from EigenIPC.PyEigenIPC import VLevel

from abc import ABC, abstractmethod

class SActorCriticAlgoBase(ABC):

    # base class for actor-critic RL algorithms
     
    def __init__(self,
            env, 
            debug = False,
            remote_db = False,
            anomaly_detect = False,
            seed: int = 1):

        self._env = env 
        self._seed = seed

        self._eval = False
        self._det_eval = True

        self._full_env_db=False

        self._agent = None 
        
        self._debug = debug
        self._remote_db = remote_db

        self._anomaly_detect = anomaly_detect

        self._writer = None
        
        self._run_name = None
        self._drop_dir = None
        self._dbinfo_drop_fname = None
        self._model_path = None
        
        self._policy_update_db_data_dict =  {}
        self._custom_env_data_db_dict = {}
        self._hyperparameters = {}
        self._wandb_d={}

        self._episodic_reward_metrics = self._env.ep_rewards_metrics()
        
        tot_tsteps=200e6
        self._init_params(tot_tsteps=tot_tsteps)
        
        self._init_dbdata()

        self._setup_done = False

        self._verbose = False

        self._is_done = False
        
        self._shared_algo_data = None

        self._this_child_path = None
        self._this_basepath = os.path.abspath(__file__)
    
    def __del__(self):

        self.done()

    def learn(self):
  
        if not self._setup_done:
            self._should_have_called_setup()

        self._start_time = time.perf_counter()

        with torch.no_grad(): # don't need grad computation here
            for i in range(self._collection_freq):
                if not self._collect_transition():
                    return False
                self._vec_transition_counter+=1
        
        self._collection_t = time.perf_counter()
        
        if self._vec_transition_counter % self._bnorm_vecfreq == 0:
            with torch.no_grad(): # don't need grad computation here
                self._update_batch_norm(bsize=self._bnorm_bsize)

        self._policy_update_t_start = time.perf_counter()
        for i in range(self._update_freq):
            self._update_policy()
            self._update_counter+=1

        self._policy_update_t = time.perf_counter()
        
        with torch.no_grad():

            if self._vec_transition_counter % self._validation_db_vecstep_freq == 0:
                self._update_validation_losses()
            self._validation_t = time.perf_counter()

            self._post_step()
        
        return True

    def eval(self):

        if not self._setup_done:
            self._should_have_called_setup()

        self._start_time = time.perf_counter()

        if not self._collect_eval_transition():
            return False
        self._vec_transition_counter+=1

        self._collection_t = time.perf_counter()
        
        self._post_step()

        return True
    
    @abstractmethod
    def _collect_transition(self)->bool:
        pass
    
    @abstractmethod
    def _collect_eval_transition(self)->bool:
        pass

    @abstractmethod
    def _update_policy(self):
        pass
    
    @abstractmethod
    def _update_validation_losses(self):
        pass

    def setup(self,
            run_name: str,
            ns: str,
            custom_args: Dict = {},
            verbose: bool = False,
            drop_dir_name: str = None,
            eval: bool = False,
            load_qf: bool = False,
            model_path: str = None,
            n_eval_timesteps: int = None,
            comment: str = "",
            dump_checkpoints: bool = False,
            norm_obs: bool = False,
            rescale_obs: bool = True):

        self._verbose = verbose

        self._ns=ns # only used for shared mem stuff

        self._dump_checkpoints = dump_checkpoints
        
        if "full_env_db" in custom_args:
            self._full_env_db=custom_args["full_env_db"]
        
        self._eval = eval
        self._load_qf=False
        if self._eval:
            self._load_qf=load_qf
        try:
            self._det_eval=custom_args["det_eval"]
        except:
            pass
        
        self._override_agent_action=False
        if self._eval:
            self._override_agent_action=custom_args["override_agent_actions"]

        self._run_name = run_name
        from datetime import datetime
        self._time_id = datetime.now().strftime('d%Y_%m_%d_h%H_m%M_s%S')
        self._unique_id = self._time_id + "-" + self._run_name

        self._use_combined_exp_replay=False
        try:
            self._use_combined_exp_replay=self._hyperparameters["use_combined_exp_replay"]
        except:
            pass

        self._init_algo_shared_data(static_params=self._hyperparameters) # can only handle dicts with
        # numeric values

        self._hyperparameters["unique_run_id"]=self._unique_id
        self._hyperparameters.update(custom_args)

        self._torch_device = torch.device("cuda" if torch.cuda.is_available() and self._use_gpu else "cpu")

        try:
            layer_width_actor=self._hyperparameters["actor_lwidth"]
            layer_width_critic=self._hyperparameters["critic_lwidth"]
            n_hidden_layers_actor=self._hyperparameters["actor_n_hlayers"]
            n_hidden_layers_critic=self._hyperparameters["critic_n_hlayers"]
        except:
            layer_width_actor=256
            layer_width_critic=512
            n_hidden_layers_actor=2
            n_hidden_layers_critic=4
            pass

        use_torch_compile=False
        add_weight_norm=False
        compression_ratio=-1.0
        if "use_torch_compile" in self._hyperparameters and \
            self._hyperparameters["use_torch_compile"]:
            use_torch_compile=True
        if "add_weight_norm" in self._hyperparameters and \
            self._hyperparameters["add_weight_norm"]:
            add_weight_norm=True
        if "compression_ratio" in self._hyperparameters:
            compression_ratio=self._hyperparameters["compression_ratio"]
        self._agent = SACAgent(obs_dim=self._env.obs_dim(),
                    obs_ub=self._env.get_obs_ub().flatten().tolist(),
                    obs_lb=self._env.get_obs_lb().flatten().tolist(),
                    actions_dim=self._env.actions_dim(),
                    actions_ub=self._env.get_actions_ub().flatten().tolist(),
                    actions_lb=self._env.get_actions_lb().flatten().tolist(),
                    rescale_obs=rescale_obs,
                    norm_obs=norm_obs,
                    compression_ratio=compression_ratio,
                    device=self._torch_device,
                    dtype=self._dtype,
                    is_eval=self._eval,
                    load_qf=self._load_qf,
                    debug=self._debug,
                    layer_width_actor=layer_width_actor,
                    layer_width_critic=layer_width_critic,
                    n_hidden_layers_actor=n_hidden_layers_actor,
                    n_hidden_layers_critic=n_hidden_layers_critic,
                    torch_compile=use_torch_compile,
                    add_weight_norm=add_weight_norm)
        
        # loging actual widths and layers in case they were override inside agent init
        self._hyperparameters["actor_lwidth"]=self._agent.layer_width_actor()
        self._hyperparameters["actor_n_hlayers"]=self._agent.n_hidden_layers_actor()
        self._hyperparameters["critic_lwidth"]=self._agent.layer_width_critic()
        self._hyperparameters["critic_n_hlayers"]=self._agent.n_hidden_layers_critic()

        if self._agent.running_norm is not None:
            # some db data for the agent
            self._running_mean_obs = torch.full((self._db_data_size, self._env.obs_dim()), 
                        dtype=torch.float32, fill_value=0.0, device="cpu")
            self._running_std_obs = torch.full((self._db_data_size, self._env.obs_dim()), 
                        dtype=torch.float32, fill_value=0.0, device="cpu")

        # load model if necessary 
        if self._eval: # load pretrained model
            if model_path is None:
                msg = f"No model path provided in eval mode! Was this intentional? \
                    No jnt remapping will be available and a randomly init agent will be used."
                Journal.log(self.__class__.__name__,
                    "setup",
                    f"No model path provided in eval mode! Was this intentional? No jnt remapping will be avaial",
                    LogType.WARN,
                    throw_when_excep = True)
            if  n_eval_timesteps is None:
                Journal.log(self.__class__.__name__,
                    "setup",
                    f"When eval is True, n_eval_timesteps should be provided!!",
                    LogType.EXCEP,
                    throw_when_excep = True)
            # everything is ok 
            self._model_path = model_path
            if self._model_path is not None:
                self._load_model(self._model_path)

            # overwrite init params
            self._init_params(tot_tsteps=n_eval_timesteps,
                run_name=self._run_name)
        
        # adding additional db info
        self._hyperparameters["obs_names"]=self._env.obs_names()
        self._hyperparameters["action_names"]=self._env.action_names()
        self._hyperparameters["sub_reward_names"]=self._env.sub_rew_names()
        self._hyperparameters["sub_trunc_names"]=self._env._get_sub_trunc_names()
        self._hyperparameters["sub_term_names"]=self._env._get_sub_term_names()

        self._allow_expl_during_eval=self._hyperparameters["allow_expl_during_eval"]

        # reset environment
        self._env.reset()
        if self._eval:
            self._env.switch_random_reset(on=False)

        # create dump directory + copy important files for debug
        self._init_drop_dir(drop_dir_name)
        self._hyperparameters["drop_dir"]=self._drop_dir

        # add env options to hyperparameters
        self._hyperparameters.update(self._env_opts) 

        # seeding + deterministic behavior for reproducibility
        self._set_all_deterministic()
        torch.autograd.set_detect_anomaly(self._anomaly_detect)

        if not self._eval:
            self._qf_optimizer = optim.Adam(list(self._agent.qf1.parameters()) + list(self._agent.qf2.parameters()), 
                                    lr=self._lr_q)
            self._actor_optimizer = optim.Adam(list(self._agent.actor.parameters()), 
                                    lr=self._lr_policy)

            self._init_replay_buffers() # only needed when training
            if self._validate:
                self._init_validation_buffers() 
        
        if self._autotune:
            self._log_alpha = torch.zeros(1, requires_grad=True, device=self._torch_device)
            self._alpha = self._log_alpha.exp().item()
            self._a_optimizer = optim.Adam([self._log_alpha], lr=self._lr_q)
        
        if (self._debug):
            if self._remote_db:
                job_type = "evaluation" if self._eval else "training"
                full_run_config={**self._hyperparameters,**self._env.custom_db_info}
                wandb.init(
                    project="LRHControl",
                    group=self._run_name,
                    name=self._unique_id,
                    id=self._unique_id,
                    job_type=job_type,
                    # tags=None,
                    notes=comment,
                    resume="never", # do not allow runs with the same unique id
                    mode="online", # "online", "offline" or "disabled"
                    entity=None,
                    sync_tensorboard=True,
                    config=full_run_config,
                    monitor_gym=True,
                    save_code=True,
                    dir=self._drop_dir
                )
                wandb.watch((self._agent), log="all", log_freq=1000, log_graph=False)
        
        if "demo_stop_thresh" in self._hyperparameters:
            self._demo_stop_thresh=self._hyperparameters["demo_stop_thresh"]

        actions = self._env.get_actions()
        self._action_scale = self._env.get_actions_scale()
        self._action_offset = self._env.get_actions_offset()
        self._random_uniform = torch.full_like(actions, fill_value=0.0) # used for sampling random actions (preallocated
        # for efficiency)
        self._random_normal = torch.full_like(self._random_uniform,fill_value=0.0)
        # for efficiency)
        
        self._actions_override=None
        if self._override_agent_action:
            from lrhc_control.utils.shared_data.training_env import Actions
            self._actions_override = Actions(namespace=ns+"_override",
            n_envs=self._num_envs,
            action_dim=actions.shape[1],
            action_names=self._env.action_names(),
            env_names=None,
            is_server=True,
            verbose=self._verbose,
            vlevel=VLevel.V2,
            safe=True,
            force_reconnection=True,
            with_gpu_mirror=self._use_gpu,
            fill_value=0.0)
            self._actions_override.run()

        self._start_time_tot = time.perf_counter()

        self._start_time = time.perf_counter()

        self._replay_bf_full = False
        self._validation_bf_full = False
        self._bpos=0
        self._bpos_val=0

        self._is_done = False
        self._setup_done = True

    def is_done(self):

        return self._is_done 
    
    def model_path(self):

        return self._model_path

    def _init_params(self,
            tot_tsteps: int,
            run_name: str = "SACDefaultRunName"):

        self._dtype = self._env.dtype()

        self._env_opts=self._env.env_opts()

        self._num_envs = self._env.n_envs()
        self._obs_dim = self._env.obs_dim()
        self._actions_dim = self._env.actions_dim()

        self._run_name = run_name # default
        self._env_name = self._env.name()
        self._episode_timeout_lb, self._episode_timeout_ub = self._env.episode_timeout_bounds()
        self._task_rand_timeout_lb, self._task_rand_timeout_ub = self._env.task_rand_timeout_bounds()
        
        self._env_n_action_reps = self._env.n_action_reps()
        
        self._use_gpu = self._env.using_gpu()
        self._torch_device = torch.device("cpu") # defaults to cpu
        self._torch_deterministic = True

        # main algo settings

        self._collection_freq=5
        self._update_freq=25

        self._replay_buffer_size_nominal = int(4e6) # 32768
        self._replay_buffer_size_vec = self._replay_buffer_size_nominal//self._num_envs # 32768
        self._replay_buffer_size = self._replay_buffer_size_vec*self._num_envs
        self._batch_size = 8192

        self._total_timesteps = int(tot_tsteps)
        self._total_timesteps = self._total_timesteps//self._env_n_action_reps # correct with n of action reps
        self._total_timesteps_vec = self._total_timesteps // self._num_envs
        self._total_steps = self._total_timesteps_vec//self._collection_freq
        self._total_timesteps_vec = self._total_steps*self._collection_freq # correct to be a multiple of self._total_steps
        self._total_timesteps = self._total_timesteps_vec*self._num_envs # actual n transitions

        # self._warmstart_timesteps = int(5e3)
        warmstart_length_single_env=min(self._episode_timeout_lb, self._episode_timeout_ub, 
            self._task_rand_timeout_lb, self._task_rand_timeout_ub)
        self._warmstart_timesteps=warmstart_length_single_env*self._num_envs
        if self._warmstart_timesteps < self._batch_size: # ensure we collect sufficient experience before
            # starting training
            self._warmstart_timesteps=4*self._batch_size
        self._warmstart_vectimesteps = self._warmstart_timesteps//self._num_envs
        # ensuring multiple of collection_freq
        self._warmstart_timesteps = self._num_envs*self._warmstart_vectimesteps # actual

        self._lr_policy = 1e-3
        self._lr_q = 1e-3

        self._discount_factor = 0.995
        self._smoothing_coeff = 0.005

        self._policy_freq = 2
        self._trgt_net_freq = 1

        self._autotune = True
        self._trgt_avrg_entropy_per_action=-1.0 # the more negative, the more deterministic the policy
        self._target_entropy = self._trgt_avrg_entropy_per_action*self._env.actions_dim()
        self._log_alpha = None
        self._alpha = 0.2
        
        self._bnorm_bsize = 4096
        self._bnorm_vecfreq_nom = 5 # wrt vec steps
        # make sure _bnorm_vecfreq is a multiple of _collection_freq
        self._bnorm_vecfreq = (self._bnorm_vecfreq_nom//self._collection_freq)*self._collection_freq
        if self._bnorm_vecfreq == 0:
            self._bnorm_vecfreq=self._collection_freq

        self._n_expl_env_perc=0.05 # [0, 1]
        self._n_expl_envs = int(self._num_envs*self._n_expl_env_perc) # n of random envs on which noisy actions will be applied
        self._allow_expl_during_eval=False
        self._noise_freq = 50
        self._noise_freq=(self._noise_freq//self._collection_freq)*self._collection_freq
        if self._noise_freq == 0:
            self._noise_freq=self._collection_freq
        self._noise_duration = 5 # should be less than _noise_freq

        self._is_continuous_actions_bool=self._env.is_action_continuous()
        self._is_continuous_actions=torch.where(self._is_continuous_actions_bool)[0]
        self._is_discrete_actions_bool=~self._is_continuous_actions
        self._is_discrete_actions=torch.where(self._is_discrete_actions_bool)[0]
        
        self._continuous_act_expl_noise_std=0.05
        self._discrete_act_expl_noise_std=0.5

        self._a_optimizer = None
        
        # debug
        self._m_checkpoint_freq_nom = 1e6 # n totoal timesteps after which a checkpoint model is dumped
        self._m_checkpoint_freq= self._m_checkpoint_freq_nom//self._num_envs

        # default to all debug envs
        self._db_env_selector=torch.tensor(list(range(0, self._num_envs)), dtype=torch.int)
        self._db_env_selector_bool=torch.full((self._num_envs, ), 
                dtype=torch.bool, device="cpu",
                fill_value=True)

        self._expl_env_selector=None
        self._expl_env_selector_bool=torch.full((self._num_envs, ), dtype=torch.bool, device="cpu",
                fill_value=False)
        self._pert_counter=0.0

        if self._n_expl_envs>0 and ((self._num_envs-self._n_expl_envs)>0): # log data only from envs which are not altered (e.g. by exploration noise)
            # computing expl env selector
            self._expl_env_selector = torch.randperm(self._num_envs, device="cpu")[:self._n_expl_envs]
            self._expl_env_selector_bool[self._expl_env_selector]=True
            
        # and db env. selector
        self._demo_stop_thresh=None # performance metrics above which demo envs are deactivated
        # (can be overridden thorugh the provided options)
        self._demo_env_selector=self._env.demo_env_idxs()
        self._demo_env_selector_bool=self._env.demo_env_idxs(get_bool=True)
        if self._demo_env_selector_bool is None:
            self._db_env_selector_bool[:]=~self._expl_env_selector_bool
        else: # we log db data separately for env which are neither for demo nor for random exploration
            self._demo_env_selector_bool=self._demo_env_selector_bool.cpu()
            self._demo_env_selector=self._demo_env_selector.cpu()
            self._db_env_selector_bool[:]=torch.logical_and(~self._expl_env_selector_bool, ~self._demo_env_selector_bool)
                
        self._n_expl_envs = self._expl_env_selector_bool.count_nonzero()
        self._num_db_envs = self._db_env_selector_bool.count_nonzero()

        if not self._num_db_envs>0:
            Journal.log(self.__class__.__name__,
                "_init_params",
                "No indipendent db env can be computed (check your demo and expl settings)! Will use all envs.",
                LogType.EXCEP,
                throw_when_excep = False)
            self._num_db_envs=self._num_envs
            self._db_env_selector_bool[:]=True
        self._db_env_selector=torch.nonzero(self._db_env_selector_bool).flatten()
        
        self._transition_noise_freq=float(self._noise_duration)/float(self._noise_freq)
        self._env_noise_freq=float(self._n_expl_envs)/float(self._num_envs)
        self._noise_buff_freq=self._transition_noise_freq*self._env_noise_freq

        self._db_vecstep_frequency = 32 # log db data every n (vectorized) SUB timesteps
        self._db_vecstep_frequency=round(self._db_vecstep_frequency/self._env_n_action_reps) # correcting with actions reps 
        # correct db vecstep frequency to ensure it's a multiple of self._collection_freq
        self._db_vecstep_frequency=(self._db_vecstep_frequency//self._collection_freq)*self._collection_freq
        if self._db_vecstep_frequency == 0:
            self._db_vecstep_frequency=self._collection_freq

        self._env_db_checkpoints_vecfreq=250*self._db_vecstep_frequency # detailed db data from envs

        self._validate=True
        self._validation_collection_vecfreq=50 # add vec transitions to val buffer with some vec freq
        self._validation_ratio=1.0/self._validation_collection_vecfreq # [0, 1], 0.1 10% size of training buffer
        self._validation_buffer_size_nominal= int(self._replay_buffer_size_nominal*self._validation_ratio)
        self._validation_buffer_size_vec = self._validation_buffer_size_nominal//self._num_envs
        self._validation_buffer_size = self._validation_buffer_size_vec*self._num_envs
        self._validation_batch_size = int(self._batch_size*self._validation_ratio)
        self._validation_db_vecstep_freq=self._db_vecstep_frequency
        if self._eval: # no need for validation transitions during evaluation
            self._validate=False

        self._n_policy_updates_to_be_done=(self._total_steps-self._warmstart_vectimesteps)*self._update_freq #TD3 delayed update
        self._n_qf_updates_to_be_done=(self._total_steps-self._warmstart_vectimesteps)*self._update_freq # qf updated at each vec timesteps
        self._n_tqf_updates_to_be_done=(self._total_steps-self._warmstart_vectimesteps)*self._update_freq//self._trgt_net_freq

        self._exp_to_policy_grad_ratio=float(self._total_timesteps-self._warmstart_timesteps)/float(self._n_policy_updates_to_be_done)
        self._exp_to_qf_grad_ratio=float(self._total_timesteps-self._warmstart_timesteps)/float(self._n_qf_updates_to_be_done)
        self._exp_to_qft_grad_ratio=float(self._total_timesteps-self._warmstart_timesteps)/float(self._n_tqf_updates_to_be_done)

        self._db_data_size = round(self._total_timesteps_vec/self._db_vecstep_frequency)+self._db_vecstep_frequency
        # write them to hyperparam dictionary for debugging
        self._hyperparameters["n_envs"] = self._num_envs
        self._hyperparameters["obs_dim"] = self._obs_dim
        self._hyperparameters["actions_dim"] = self._actions_dim

        self._hyperparameters["seed"] = self._seed
        self._hyperparameters["using_gpu"] = self._use_gpu
        self._hyperparameters["total_timesteps_vec"] = self._total_timesteps_vec

        self._hyperparameters["collection_freq"]=self._collection_freq
        self._hyperparameters["update_freq"]=self._update_freq
        self._hyperparameters["total_steps"]=self._total_steps
        
        self._hyperparameters["n_policy_updates_when_done"] = self._n_policy_updates_to_be_done
        self._hyperparameters["n_qf_updates_when_done"] = self._n_qf_updates_to_be_done
        self._hyperparameters["n_tqf_updates_when_done"] = self._n_tqf_updates_to_be_done
        self._hyperparameters["experience_to_policy_grad_steps_ratio"] = self._exp_to_policy_grad_ratio
        self._hyperparameters["experience_to_quality_fun_grad_steps_ratio"] = self._exp_to_qf_grad_ratio
        self._hyperparameters["experience_to_trgt_quality_fun_grad_steps_ratio"] = self._exp_to_qft_grad_ratio

        self._hyperparameters["episodes timeout lb"] = self._episode_timeout_lb
        self._hyperparameters["episodes timeout ub"] = self._episode_timeout_ub
        self._hyperparameters["task rand timeout lb"] = self._task_rand_timeout_lb
        self._hyperparameters["task rand timeout ub"] = self._task_rand_timeout_ub
        
        self._hyperparameters["warmstart_timesteps"] = self._warmstart_timesteps
        self._hyperparameters["warmstart_vectimesteps"] = self._warmstart_vectimesteps
        self._hyperparameters["replay_buffer_size_nominal"] = self._replay_buffer_size_nominal
        self._hyperparameters["batch_size"] = self._batch_size
        self._hyperparameters["total_timesteps"] = self._total_timesteps
        self._hyperparameters["lr_policy"] = self._lr_policy
        self._hyperparameters["lr_q"] = self._lr_q
        self._hyperparameters["discount_factor"] = self._discount_factor
        self._hyperparameters["smoothing_coeff"] = self._smoothing_coeff
        self._hyperparameters["policy_freq"] = self._policy_freq
        self._hyperparameters["trgt_net_freq"] = self._trgt_net_freq
        self._hyperparameters["autotune"] = self._autotune
        self._hyperparameters["target_entropy"] = self._target_entropy
        self._hyperparameters["log_alpha"] = self._log_alpha
        self._hyperparameters["alpha"] = self._alpha
        self._hyperparameters["m_checkpoint_freq"] = self._m_checkpoint_freq
        self._hyperparameters["db_vecstep_frequency"] = self._db_vecstep_frequency
        self._hyperparameters["m_checkpoint_freq"] = self._m_checkpoint_freq
        
        self._hyperparameters["n_db_envs"] = self._num_db_envs
        self._hyperparameters["n_expl_envs"] = self._n_expl_envs
        self._hyperparameters["noise_freq"] = self._noise_freq
        self._hyperparameters["noise_buff_freq"] = self._noise_buff_freq
        self._hyperparameters["n_demo_envs"] = self._env.n_demo_envs()
        
        self._hyperparameters["bnorm_bsize"] = self._bnorm_bsize
        self._hyperparameters["bnorm_vecfreq"] = self._bnorm_vecfreq
        
        self._hyperparameters["validate"] = self._validate
        self._hyperparameters["validation_ratio"] = self._validation_ratio
        self._hyperparameters["validation_buffer_size_vec"] = self._validation_buffer_size_vec
        self._hyperparameters["validation_buffer_size"] = self._validation_buffer_size
        self._hyperparameters["validation_batch_size"] = self._validation_batch_size
        self._hyperparameters["validation_collection_vecfreq"] = self._validation_collection_vecfreq

        # small debug log
        info = f"\nUsing \n" + \
            f"total (vectorized) timesteps to be simulated {self._total_timesteps_vec}\n" + \
            f"total timesteps to be simulated {self._total_timesteps}\n" + \
            f"warmstart timesteps {self._warmstart_timesteps}\n" + \
            f"training replay buffer nominal size {self._replay_buffer_size_nominal}\n" + \
            f"training replay buffer size {self._replay_buffer_size}\n" + \
            f"training replay buffer vec size {self._replay_buffer_size_vec}\n" + \
            f"training batch size {self._batch_size}\n" + \
            f"validation enabled {self._validate}\n" + \
            f"validation buffer nominal size {self._validation_buffer_size_nominal}\n" + \
            f"validation buffer size {self._validation_buffer_size}\n" + \
            f"validation buffer vec size {self._validation_buffer_size_vec}\n" + \
            f"validation collection freq {self._validation_collection_vecfreq}\n" + \
            f"validation update freq {self._validation_db_vecstep_freq}\n" + \
            f"validation batch size {self._validation_batch_size}\n" + \
            f"policy update freq {self._policy_freq}\n" + \
            f"target networks freq {self._trgt_net_freq}\n" + \
            f"episode timeout max steps {self._episode_timeout_ub}\n" + \
            f"episode timeout min steps {self._episode_timeout_lb}\n" + \
            f"task rand. max n steps {self._task_rand_timeout_ub}\n" + \
            f"task rand. min n steps {self._task_rand_timeout_lb}\n" + \
            f"number of action reps {self._env_n_action_reps}\n" + \
            f"total policy updates to be performed: {self._n_policy_updates_to_be_done}\n" + \
            f"total q fun updates to be performed: {self._n_qf_updates_to_be_done}\n" + \
            f"total trgt q fun updates to be performed: {self._n_tqf_updates_to_be_done}\n" + \
            f"experience to policy grad ratio: {self._exp_to_policy_grad_ratio}\n" + \
            f"experience to q fun grad ratio: {self._exp_to_qf_grad_ratio}\n" + \
            f"experience to trgt q fun grad ratio: {self._exp_to_qft_grad_ratio}\n" + \
            f"amount of noisy transitions in replay buffer: {self._noise_buff_freq*100}% \n"
        db_env_idxs=", ".join(map(str, self._db_env_selector.tolist()))
        n_db_envs_str=f"db envs {self._num_db_envs}/{self._num_envs} \n" 
        info=info + n_db_envs_str + "Debug env. indexes are [" + db_env_idxs+"]\n"
        if self._env.demo_env_idxs() is not None:
            demo_idxs_str=", ".join(map(str, self._env.demo_env_idxs().tolist()))
            n_demo_envs_str=f"demo envs {self._env.n_demo_envs()}/{self._num_envs} \n" 
            info=info + n_demo_envs_str + "Demo env. indexes are [" + demo_idxs_str+"]\n"
        if self._expl_env_selector is not None:
            random_expl_idxs=", ".join(map(str, self._expl_env_selector.tolist()))
            n_expl_envs_str=f"expl envs {self._n_expl_envs}/{self._num_envs} \n" 
            info=info + n_expl_envs_str + "Random exploration env. indexes are [" + random_expl_idxs+"]\n"
        
        Journal.log(self.__class__.__name__,
            "_init_params",
            info,
            LogType.INFO,
            throw_when_excep = True)
        
        self._step_counter = 0
        self._vec_transition_counter = 0
        self._update_counter = 0
        self._log_it_counter = 0

    def _init_dbdata(self):

        # initalize some debug data
        self._collection_dt = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        
        self._collection_t = -1.0
        self._env_step_fps = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._env_step_rt_factor = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        
        self._batch_norm_update_dt = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        
        self._policy_update_t_start = -1.0
        self._policy_update_t = -1.0
        self._policy_update_dt = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._policy_update_fps = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        
        self._validation_t = -1.0
        self._validation_dt = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        
        self._n_of_played_episodes = torch.full((self._db_data_size, 1), 
                    dtype=torch.int32, fill_value=0, device="cpu")
        self._n_timesteps_done = torch.full((self._db_data_size, 1), 
                    dtype=torch.int32, fill_value=0, device="cpu")
        self._n_policy_updates = torch.full((self._db_data_size, 1), 
                    dtype=torch.int32, fill_value=0, device="cpu")
        self._n_qfun_updates = torch.full((self._db_data_size, 1), 
                    dtype=torch.int32, fill_value=0, device="cpu")
        self._n_tqfun_updates = torch.full((self._db_data_size, 1), 
                    dtype=torch.int32, fill_value=0, device="cpu")
        self._elapsed_min = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0, device="cpu")        
        
        self._ep_tsteps_env_distribution = torch.full((self._db_data_size, self._num_db_envs, 1), 
                    dtype=torch.int32, fill_value=-1, device="cpu")

        self._reward_names = self._episodic_reward_metrics.reward_names()
        self._reward_names_str = "[" + ', '.join(self._reward_names) + "]"
        self._n_rewards = self._episodic_reward_metrics.n_rewards()

        # db environments
        self._tot_rew_max = torch.full((self._db_data_size, self._num_db_envs, 1), 
            dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._tot_rew_avrg = torch.full((self._db_data_size, self._num_db_envs, 1), 
            dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._tot_rew_min = torch.full((self._db_data_size, self._num_db_envs, 1), 
            dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._tot_rew_max_over_envs = torch.full((self._db_data_size, 1, 1), 
            dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._tot_rew_avrg_over_envs = torch.full((self._db_data_size, 1, 1), 
            dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._tot_rew_min_over_envs = torch.full((self._db_data_size, 1, 1), 
            dtype=torch.float32, fill_value=torch.nan, device="cpu")
        
        self._sub_rew_max = torch.full((self._db_data_size, self._num_db_envs, self._n_rewards), 
            dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._sub_rew_avrg = torch.full((self._db_data_size, self._num_db_envs, self._n_rewards), 
            dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._sub_rew_min = torch.full((self._db_data_size, self._num_db_envs, self._n_rewards), 
            dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._sub_rew_max_over_envs = torch.full((self._db_data_size, 1, self._n_rewards), 
            dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._sub_rew_avrg_over_envs = torch.full((self._db_data_size, 1, self._n_rewards), 
            dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._sub_rew_min_over_envs = torch.full((self._db_data_size, 1, self._n_rewards), 
            dtype=torch.float32, fill_value=torch.nan, device="cpu")
        
        # custom data from env # (log data just from db envs for simplicity)
        self._custom_env_data = {}
        db_data_names = list(self._env.custom_db_data.keys())
        for dbdatan in db_data_names: # loop thorugh custom data
            
            self._custom_env_data[dbdatan] = {}

            max = self._env.custom_db_data[dbdatan].get_max(env_selector=self._db_env_selector).reshape(self._num_db_envs, -1)
            avrg = self._env.custom_db_data[dbdatan].get_avrg(env_selector=self._db_env_selector).reshape(self._num_db_envs, -1)
            min = self._env.custom_db_data[dbdatan].get_min(env_selector=self._db_env_selector).reshape(self._num_db_envs, -1)
            max_over_envs = self._env.custom_db_data[dbdatan].get_max_over_envs(env_selector=self._db_env_selector).reshape(1, -1)
            avrg_over_envs = self._env.custom_db_data[dbdatan].get_avrg_over_envs(env_selector=self._db_env_selector).reshape(1, -1)
            min_over_envs = self._env.custom_db_data[dbdatan].get_min_over_envs(env_selector=self._db_env_selector).reshape(1, -1)

            self._custom_env_data[dbdatan]["max"] =torch.full((self._db_data_size, 
                max.shape[0], 
                max.shape[1]), 
                dtype=torch.float32, fill_value=torch.nan, device="cpu")
            self._custom_env_data[dbdatan]["avrg"] =torch.full((self._db_data_size, 
                avrg.shape[0], 
                avrg.shape[1]), 
                dtype=torch.float32, fill_value=torch.nan, device="cpu")
            self._custom_env_data[dbdatan]["min"] =torch.full((self._db_data_size, 
                min.shape[0], 
                min.shape[1]), 
                dtype=torch.float32, fill_value=torch.nan, device="cpu")
            self._custom_env_data[dbdatan]["max_over_envs"] =torch.full((self._db_data_size, 
                max_over_envs.shape[0], 
                max_over_envs.shape[1]), 
                dtype=torch.float32, fill_value=torch.nan, device="cpu")
            self._custom_env_data[dbdatan]["avrg_over_envs"] =torch.full((self._db_data_size, 
                avrg_over_envs.shape[0], 
                avrg_over_envs.shape[1]), 
                dtype=torch.float32, fill_value=torch.nan, device="cpu")
            self._custom_env_data[dbdatan]["min_over_envs"] =torch.full((self._db_data_size, 
                min_over_envs.shape[0], 
                min_over_envs.shape[1]), 
                dtype=torch.float32, fill_value=torch.nan, device="cpu")

        # exploration envs
        if self._n_expl_envs > 0:
            self._ep_tsteps_expl_env_distribution = torch.full((self._db_data_size, self._n_expl_envs, 1), 
                    dtype=torch.int32, fill_value=-1, device="cpu")

            # also log sub rewards metrics for exploration envs
            self._sub_rew_max_expl = torch.full((self._db_data_size, self._n_expl_envs, self._n_rewards), 
                dtype=torch.float32, fill_value=torch.nan, device="cpu")
            self._sub_rew_avrg_expl = torch.full((self._db_data_size, self._n_expl_envs, self._n_rewards), 
                dtype=torch.float32, fill_value=torch.nan, device="cpu")
            self._sub_rew_min_expl = torch.full((self._db_data_size, self._n_expl_envs, self._n_rewards), 
                dtype=torch.float32, fill_value=torch.nan, device="cpu")
            self._sub_rew_max_over_envs_expl = torch.full((self._db_data_size, 1, self._n_rewards), 
                dtype=torch.float32, fill_value=torch.nan, device="cpu")
            self._sub_rew_avrg_over_envs_expl = torch.full((self._db_data_size, 1, self._n_rewards), 
                dtype=torch.float32, fill_value=torch.nan, device="cpu")
            self._sub_rew_min_over_envs_expl = torch.full((self._db_data_size, 1, self._n_rewards), 
                dtype=torch.float32, fill_value=torch.nan, device="cpu")
        
        # demo environments
        self._demo_envs_active = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._demo_perf_metric = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=torch.nan, device="cpu")
        if self._env.demo_env_idxs() is not None:
            n_demo_envs=self._env.demo_env_idxs().shape[0]

            self._ep_tsteps_demo_env_distribution = torch.full((self._db_data_size, n_demo_envs, 1), 
                    dtype=torch.int32, fill_value=-1, device="cpu")

            # also log sub rewards metrics for exploration envs
            self._sub_rew_max_demo = torch.full((self._db_data_size, n_demo_envs, self._n_rewards), 
                dtype=torch.float32, fill_value=torch.nan, device="cpu")
            self._sub_rew_avrg_demo = torch.full((self._db_data_size, n_demo_envs, self._n_rewards), 
                dtype=torch.float32, fill_value=torch.nan, device="cpu")
            self._sub_rew_min_demo = torch.full((self._db_data_size, n_demo_envs, self._n_rewards), 
                dtype=torch.float32, fill_value=torch.nan, device="cpu")
            self._sub_rew_max_over_envs_demo = torch.full((self._db_data_size, 1, self._n_rewards), 
                dtype=torch.float32, fill_value=torch.nan, device="cpu")
            self._sub_rew_avrg_over_envs_demo = torch.full((self._db_data_size, 1, self._n_rewards), 
                dtype=torch.float32, fill_value=torch.nan, device="cpu")
            self._sub_rew_min_over_envs_demo = torch.full((self._db_data_size, 1, self._n_rewards), 
                dtype=torch.float32, fill_value=torch.nan, device="cpu")
            
        # algorithm-specific db info
        self._qf1_vals_mean = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._qf2_vals_mean = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._qf1_vals_std = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._qf2_vals_std = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._qf1_vals_max = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._qf1_vals_min = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._qf2_vals_max = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._qf2_vals_min = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=torch.nan, device="cpu")
        
        self._qf1_loss = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._qf2_loss = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._actor_loss= torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._alpha_loss = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=torch.nan, device="cpu")
        if self._validate: # add db data for validation losses
            self._qf1_loss_validation = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=torch.nan, device="cpu")
            self._qf2_loss_validation = torch.full((self._db_data_size, 1), 
                        dtype=torch.float32, fill_value=torch.nan, device="cpu")
            self._actor_loss_validation= torch.full((self._db_data_size, 1), 
                        dtype=torch.float32, fill_value=torch.nan, device="cpu")
            self._alpha_loss_validation = torch.full((self._db_data_size, 1), 
                        dtype=torch.float32, fill_value=torch.nan, device="cpu")
        
        self._alphas = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=torch.nan, device="cpu")

        self._policy_entropy_mean=torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._policy_entropy_std=torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._policy_entropy_max=torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._policy_entropy_min=torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=torch.nan, device="cpu")            

    def _init_replay_buffers(self):
        
        self._bpos = 0

        self._obs = torch.full(size=(self._replay_buffer_size_vec, self._num_envs, self._obs_dim),
                        fill_value=torch.nan,
                        dtype=self._dtype,
                        device=self._torch_device,
                        requires_grad=False) 
        self._actions = torch.full(size=(self._replay_buffer_size_vec, self._num_envs, self._actions_dim),
                        fill_value=torch.nan,
                        dtype=self._dtype,
                        device=self._torch_device,
                        requires_grad=False)
        self._rewards = torch.full(size=(self._replay_buffer_size_vec, self._num_envs, 1),
                        fill_value=torch.nan,
                        dtype=self._dtype,
                        device=self._torch_device,
                        requires_grad=False)
        self._next_obs = torch.full(size=(self._replay_buffer_size_vec, self._num_envs, self._obs_dim),
                        fill_value=torch.nan,
                        dtype=self._dtype,
                        device=self._torch_device,
                        requires_grad=False) 
        self._next_terminal = torch.full(size=(self._replay_buffer_size_vec, self._num_envs, 1),
                        fill_value=False,
                        dtype=self._dtype,
                        device=self._torch_device,
                        requires_grad=False)

    def _init_validation_buffers(self):
        
        self._bpos_val = 0

        self._obs_val = torch.full(size=(self._validation_buffer_size_vec, self._num_envs, self._obs_dim),
                        fill_value=torch.nan,
                        dtype=self._dtype,
                        device=self._torch_device,
                        requires_grad=False) 
        self._actions_val = torch.full(size=(self._validation_buffer_size_vec, self._num_envs, self._actions_dim),
                        fill_value=torch.nan,
                        dtype=self._dtype,
                        device=self._torch_device,
                        requires_grad=False)
        self._rewards_val = torch.full(size=(self._validation_buffer_size_vec, self._num_envs, 1),
                        fill_value=torch.nan,
                        dtype=self._dtype,
                        device=self._torch_device,
                        requires_grad=False)
        self._next_obs_val = torch.full(size=(self._validation_buffer_size_vec, self._num_envs, self._obs_dim),
                        fill_value=torch.nan,
                        dtype=self._dtype,
                        device=self._torch_device,
                        requires_grad=False) 
        self._next_terminal_val = torch.full(size=(self._validation_buffer_size_vec, self._num_envs, 1),
                        fill_value=False,
                        dtype=self._dtype,
                        device=self._torch_device,
                        requires_grad=False)
        
    def _save_model(self,
            is_checkpoint: bool = False):

        path = self._model_path
        if is_checkpoint: # use iteration as id
            path = path + "_checkpoint" + str(self._log_it_counter)
        info = f"Saving model to {path}"
        Journal.log(self.__class__.__name__,
            "_save_model",
            info,
            LogType.INFO,
            throw_when_excep = True)
        agent_state_dict=self._agent.state_dict()
        if not self._eval: # training
            # we log the joints which were observed during training
            observed_joints=self._env.get_observed_joints()
            if observed_joints is not None:
                agent_state_dict["observed_jnts"]=self._env.get_observed_joints()

        torch.save(agent_state_dict, path) # saves whole agent state
        # torch.save(self._agent.parameters(), path) # only save agent parameters
        info = f"Done."
        Journal.log(self.__class__.__name__,
            "_save_model",
            info,
            LogType.INFO,
            throw_when_excep = True)
    
    def _dump_env_checkpoints(self):

        path = self._env_db_checkpoints_fname+str(self._log_it_counter)

        if path is not None:
            info = f"Saving env db checkpoint data to {path}"
            Journal.log(self.__class__.__name__,
                "_dump_env_checkpoints",
                info,
                LogType.INFO,
                throw_when_excep = True)
            
            with h5py.File(path+".hdf5", 'w') as hf:
                hf.create_dataset('sub_reward_names', data=self._reward_names, 
                    dtype='S20') 
                hf.create_dataset('log_iteration', data=np.array(self._log_it_counter)) 
                hf.create_dataset('n_timesteps_done', data=self._n_timesteps_done.numpy())
                hf.create_dataset('n_policy_updates', data=self._n_policy_updates.numpy())
                hf.create_dataset('elapsed_min', data=self._elapsed_min.numpy())

                # full training envs
                hf.create_dataset('sub_rew', 
                    data=self._episodic_reward_metrics.get_full_episodic_subrew(env_selector=self._db_env_selector))
                hf.create_dataset('tot_rew', 
                    data=self._episodic_reward_metrics.get_full_episodic_totrew(env_selector=self._db_env_selector))
                
                if self._n_expl_envs > 0:
                    hf.create_dataset('sub_rew_expl', 
                        data=self._episodic_reward_metrics.get_full_episodic_subrew(env_selector=self._expl_env_selector))
                    hf.create_dataset('tot_rew_expl', 
                        data=self._episodic_reward_metrics.get_full_episodic_totrew(env_selector=self._expl_env_selector))

                if self._env.n_demo_envs() > 0:
                    hf.create_dataset('sub_rew_demo', 
                        data=self._episodic_reward_metrics.get_full_episodic_subrew(env_selector=self._demo_env_selector))
                    hf.create_dataset('tot_rew_demo', 
                        data=self._episodic_reward_metrics.get_full_episodic_totrew(env_selector=self._demo_env_selector))
                
                # dump all custom env data
                db_data_names = list(self._env.custom_db_data.keys())
                for db_dname in db_data_names:
                    episodic_data=self._env.custom_db_data[db_dname]
                    var_name = db_dname
                    hf.create_dataset(var_name, 
                        data=episodic_data.get_full_episodic_data(env_selector=self._db_env_selector))
                    if self._n_expl_envs > 0:
                        hf.create_dataset(var_name+"_expl", 
                            data=episodic_data.get_full_episodic_data(env_selector=self._expl_env_selector))
                    if self._env.n_demo_envs() > 0:
                        hf.create_dataset(var_name+"_demo", 
                            data=episodic_data.get_full_episodic_data(env_selector=self._demo_env_selector))
                
            Journal.log(self.__class__.__name__,
                "_dump_env_checkpoints",
                info,
                LogType.INFO,
                throw_when_excep = True)
        
    def done(self):
        
        if not self._is_done:

            if not self._eval:
                self._save_model()
            
            self._dump_dbinfo_to_file()
            
            if self._shared_algo_data is not None:
                self._shared_algo_data.write(dyn_info_name=["is_done"],
                    val=[1.0])
                self._shared_algo_data.close() # close shared memory

            self._env.close()

            self._is_done = True

    def _dump_dbinfo_to_file(self):

        info = f"Dumping debug info at {self._dbinfo_drop_fname}"
        Journal.log(self.__class__.__name__,
            "_dump_dbinfo_to_file",
            info,
            LogType.INFO,
            throw_when_excep = True)
        
        with h5py.File(self._dbinfo_drop_fname+".hdf5", 'w') as hf:
            # hf.create_dataset('numpy_data', data=numpy_data)
            # Write dictionaries to HDF5 as attributes
            for key, value in self._hyperparameters.items():
                if value is None:
                    value = "None"
                hf.attrs[key] = value
            
            # rewards
            hf.create_dataset('sub_reward_names', data=self._reward_names, 
                dtype='S20') 
            hf.create_dataset('sub_rew_max', data=self._sub_rew_max.numpy())
            hf.create_dataset('sub_rew_avrg', data=self._sub_rew_avrg.numpy())
            hf.create_dataset('sub_rew_min', data=self._sub_rew_min.numpy())
            hf.create_dataset('sub_rew_max_over_envs', data=self._sub_rew_max_over_envs.numpy())
            hf.create_dataset('sub_rew_avrg_over_envs', data=self._sub_rew_avrg_over_envs.numpy())
            hf.create_dataset('sub_rew_min_over_envs', data=self._sub_rew_min_over_envs.numpy())

            hf.create_dataset('tot_rew_max', data=self._tot_rew_max.numpy())
            hf.create_dataset('tot_rew_avrg', data=self._tot_rew_avrg.numpy())
            hf.create_dataset('tot_rew_min', data=self._tot_rew_min.numpy())
            hf.create_dataset('tot_rew_max_over_envs', data=self._tot_rew_max_over_envs.numpy())
            hf.create_dataset('tot_rew_avrg_over_envs', data=self._tot_rew_avrg_over_envs.numpy())
            hf.create_dataset('tot_rew_min_over_envs', data=self._tot_rew_min_over_envs.numpy())
            
            hf.create_dataset('ep_tsteps_env_distribution', data=self._ep_tsteps_env_distribution.numpy())

            if self._n_expl_envs > 0:
                # expl envs
                hf.create_dataset('sub_rew_max_expl', data=self._sub_rew_max_expl.numpy())
                hf.create_dataset('sub_rew_avrg_expl', data=self._sub_rew_avrg_expl.numpy())
                hf.create_dataset('sub_rew_min_expl', data=self._sub_rew_min_expl.numpy())
                hf.create_dataset('sub_rew_max_over_envs_expl', data=self._sub_rew_max_over_envs_expl.numpy())
                hf.create_dataset('sub_rew_avrg_over_envs_expl', data=self._sub_rew_avrg_over_envs_expl.numpy())
                hf.create_dataset('sub_rew_min_over_envs_expl', data=self._sub_rew_min_over_envs_expl.numpy())

                hf.create_dataset('ep_timesteps_expl_env_distr', data=self._ep_tsteps_expl_env_distribution.numpy())
                
            hf.create_dataset('demo_envs_active', data=self._demo_envs_active.numpy())
            hf.create_dataset('demo_perf_metric', data=self._demo_perf_metric.numpy())
            
            if self._env.n_demo_envs() > 0:
                # demo envs
                hf.create_dataset('sub_rew_max_demo', data=self._sub_rew_max_demo.numpy())
                hf.create_dataset('sub_rew_avrg_demo', data=self._sub_rew_avrg_demo.numpy())
                hf.create_dataset('sub_rew_min_demo', data=self._sub_rew_min_demo.numpy())
                hf.create_dataset('sub_rew_max_over_envs_demo', data=self._sub_rew_max_over_envs_demo.numpy())
                hf.create_dataset('sub_rew_avrg_over_envs_demo', data=self._sub_rew_avrg_over_envs_demo.numpy())
                hf.create_dataset('sub_rew_min_over_envs_demo', data=self._sub_rew_min_over_envs_demo.numpy())

                hf.create_dataset('ep_timesteps_demo_env_distr', data=self._ep_tsteps_demo_env_distribution.numpy())

            # profiling data
            hf.create_dataset('env_step_fps', data=self._env_step_fps.numpy())
            hf.create_dataset('env_step_rt_factor', data=self._env_step_rt_factor.numpy())
            hf.create_dataset('collection_dt', data=self._collection_dt.numpy())
            hf.create_dataset('batch_norm_update_dt', data=self._batch_norm_update_dt.numpy())
            hf.create_dataset('policy_update_dt', data=self._policy_update_dt.numpy())
            hf.create_dataset('policy_update_fps', data=self._policy_update_fps.numpy())
            hf.create_dataset('validation_dt', data=self._validation_dt.numpy())
            
            hf.create_dataset('n_of_played_episodes', data=self._n_of_played_episodes.numpy())
            hf.create_dataset('n_timesteps_done', data=self._n_timesteps_done.numpy())
            hf.create_dataset('n_policy_updates', data=self._n_policy_updates.numpy())
            hf.create_dataset('n_qfun_updates', data=self._n_qfun_updates.numpy())
            hf.create_dataset('n_tqfun_updates', data=self._n_tqfun_updates.numpy())
            
            hf.create_dataset('elapsed_min', data=self._elapsed_min.numpy())

            # algo data 
            hf.create_dataset('qf1_vals_mean', data=self._qf1_vals_mean.numpy())
            hf.create_dataset('qf2_vals_mean', data=self._qf2_vals_mean.numpy())
            hf.create_dataset('qf1_vals_std', data=self._qf1_vals_std.numpy())
            hf.create_dataset('qf2_vals_std', data=self._qf2_vals_std.numpy())
            hf.create_dataset('qf1_vals_max', data=self._qf1_vals_max.numpy())
            hf.create_dataset('qf1_vals_min', data=self._qf1_vals_min.numpy())
            hf.create_dataset('qf2_vals_max', data=self._qf2_vals_max.numpy())
            hf.create_dataset('qf2_vals_min', data=self._qf1_vals_min.numpy())
            
            hf.create_dataset('qf1_loss', data=self._qf1_loss.numpy())
            hf.create_dataset('qf2_loss', data=self._qf2_loss.numpy())
            hf.create_dataset('actor_loss', data=self._actor_loss.numpy())
            hf.create_dataset('alpha_loss', data=self._alpha_loss.numpy())
            if self._validate:
                hf.create_dataset('qf1_loss_validation', data=self._qf1_loss_validation.numpy())
                hf.create_dataset('qf2_loss_validation', data=self._qf2_loss_validation.numpy())
                hf.create_dataset('actor_loss_validation', data=self._actor_loss_validation.numpy())
                hf.create_dataset('alpha_loss_validation', data=self._alpha_loss_validation.numpy())
                
            hf.create_dataset('alphas', data=self._alphas.numpy())
            
            hf.create_dataset('policy_entropy_mean', data=self._policy_entropy_mean.numpy())
            hf.create_dataset('policy_entropy_std', data=self._policy_entropy_std.numpy())
            hf.create_dataset('policy_entropy_max', data=self._policy_entropy_max.numpy())
            hf.create_dataset('policy_entropy_min', data=self._policy_entropy_min.numpy())
            hf.create_dataset('target_entropy', data=self._target_entropy)

            # dump all custom env data
            db_data_names = list(self._env.custom_db_data.keys())
            for db_dname in db_data_names:
                data=self._custom_env_data[db_dname]
                subnames = list(data.keys())
                for subname in subnames:
                    var_name = db_dname + "_" + subname
                    hf.create_dataset(var_name, data=data[subname])
            db_info_names = list(self._env.custom_db_info.keys())
            for db_info in db_info_names:
                hf.create_dataset(db_info, data=self._env.custom_db_info[db_info])
            
            # other data
            if self._agent.running_norm is not None:
                hf.create_dataset('running_mean_obs', data=self._running_mean_obs.numpy())
                hf.create_dataset('running_std_obs', data=self._running_std_obs.numpy())
            
        info = f"done."
        Journal.log(self.__class__.__name__,
            "_dump_dbinfo_to_file",
            info,
            LogType.INFO,
            throw_when_excep = True)

    def _load_model(self,
            model_path: str):
        
        info = f"Loading model at {model_path}"

        Journal.log(self.__class__.__name__,
            "_load_model",
            info,
            LogType.INFO,
            throw_when_excep = True)
        model_dict=torch.load(model_path, 
                    map_location=self._torch_device)
        
        observed_joints=self._env.get_observed_joints()
        if not ("observed_jnts" in model_dict):
            Journal.log(self.__class__.__name__,
            "_load_model",
            "No observed joints key found in loaded model dictionary! Let's hope joints are ordered in the same way.",
            LogType.WARN)
        else:
            required_joints=model_dict["observed_jnts"]
            self._check_observed_joints(observed_joints,required_joints)

        self._agent.load_state_dict(model_dict)
        self._switch_training_mode(False)

    def _check_observed_joints(self,
            observed_joints,
            required_joints):

        observed=set(observed_joints)
        required=set(required_joints)

        all_required_joints_avail = required.issubset(observed)
        if not all_required_joints_avail:
            missing=[item for item in required if item not in observed]
            missing_str=', '.join(missing)
            Journal.log(self.__class__.__name__,
                "_check_observed_joints",
                f"not all required joints are available. Missing {missing_str}",
                LogType.EXCEP,
                throw_when_excep = True)
        exceeding=observed-required
        if not len(exceeding)==0:
            # do not support having more joints than the required
            exc_jnts=" ".join(list(exceeding))
            Journal.log(self.__class__.__name__,
                "_check_observed_joints",
                f"more than the required joints found in the observed joint: {exc_jnts}",
                LogType.EXCEP,
                throw_when_excep = True)
        
        # here we are sure that required and observed sets match
        self._to_agent_jnt_remap=None
        if not required_joints==observed_joints:
            Journal.log(self.__class__.__name__,
                "_check_observed_joints",
                f"required jnt obs from agent have different ordering from observed ones. Will compute a remapping.",
                LogType.WARN,
                throw_when_excep = True)
            self._to_agent_jnt_remap = [observed_joints.index(element) for element in required_joints]
        
        self._env.set_jnts_remapping(remapping= self._to_agent_jnt_remap)

    def _set_all_deterministic(self):
        import random
        random.seed(self._seed)
        random.seed(self._seed) # python seed
        torch.manual_seed(self._seed)
        torch.backends.cudnn.deterministic = self._torch_deterministic
        # torch.backends.cudnn.benchmark = not self._torch_deterministic
        # torch.use_deterministic_algorithms(True)
        # torch.use_deterministic_algorithms(mode=True) # will throw excep. when trying to use non-det. algos
        import numpy as np
        np.random.seed(self._seed)

    def drop_dir(self):
        return self._drop_dir
        
    def _init_drop_dir(self,
                drop_dir_name: str = None):

        # main drop directory
        if drop_dir_name is None:
            # drop to current directory
            self._drop_dir = "./" + f"{self.__class__.__name__}/" + self._run_name + "/" + self._unique_id
        else:
            self._drop_dir = drop_dir_name + "/" + f"{self.__class__.__name__}/" + self._run_name + "/" + self._unique_id
        os.makedirs(self._drop_dir)
        
        self._env_db_checkpoints_dropdir=None
        self._env_db_checkpoints_fname=None
        if self._full_env_db>0:
            self._env_db_checkpoints_dropdir=self._drop_dir+"/env_db_checkpoints"
            self._env_db_checkpoints_fname = self._env_db_checkpoints_dropdir + \
                "/" + self._unique_id + "env_db_checkpoint"
            os.makedirs(self._env_db_checkpoints_dropdir)
        # model
        if not self._eval or (self._model_path is None):
            self._model_path = self._drop_dir + "/" + self._unique_id + "_model"
        else: # we copy the model under evaluation to the drop dir
            shutil.copy(self._model_path, self._drop_dir)

        # debug info
        self._dbinfo_drop_fname = self._drop_dir + "/" + self._unique_id + "db_info" # extension added later

        # other auxiliary db files
        aux_drop_dir = self._drop_dir + "/other"
        os.makedirs(aux_drop_dir)
        filepaths = self._env.get_file_paths() # envs implementation
        filepaths.append(self._this_basepath) # algorithm implementation
        filepaths.append(self._this_child_path)
        filepaths.append(self._agent.get_impl_path()) # agent implementation
        for file in filepaths:
            shutil.copy(file, self._drop_dir)
        aux_dirs = self._env.get_aux_dir()
        for aux_dir in aux_dirs:
            shutil.copytree(aux_dir, aux_drop_dir, dirs_exist_ok=True)
    
    def _get_performance_metric(self):
        # to be overridden
        return 0.0

    def _post_step(self):
        
        # these have to always be updated
        self._collection_dt[self._log_it_counter] += \
            (self._collection_t-self._start_time)
        self._batch_norm_update_dt[self._log_it_counter] += \
            (self._policy_update_t_start-self._collection_t)
        self._policy_update_dt[self._log_it_counter] += \
            (self._policy_update_t - self._policy_update_t_start)
        if self._validate:
            self._validation_dt[self._log_it_counter] += \
            (self._validation_t - self._policy_update_t)

        self._step_counter+=1 # counts algo steps
        
        self._demo_envs_active[self._log_it_counter]=self._env.demo_active()
        self._demo_perf_metric[self._log_it_counter]=self._get_performance_metric()
        if self._env.n_demo_envs() > 0 and (self._demo_stop_thresh is not None):
            # check if deactivation condition applies
            self._env.switch_demo(active=self._demo_perf_metric[self._log_it_counter]<self._demo_stop_thresh)
        
        if self._vec_transition_counter % self._db_vecstep_frequency == 0:
            # only log data every n timesteps 
            
            self._env_step_fps[self._log_it_counter] = (self._db_vecstep_frequency*self._num_envs)/ self._collection_dt[self._log_it_counter]
            if "substepping_dt" in self._hyperparameters:
                self._env_step_rt_factor[self._log_it_counter] = self._env_step_fps[self._log_it_counter]*self._env_n_action_reps*self._hyperparameters["substepping_dt"]

            self._n_timesteps_done[self._log_it_counter]=self._vec_transition_counter*self._num_envs
            
            self._n_policy_updates[self._log_it_counter]+=self._n_policy_updates[self._log_it_counter-1]
            self._n_qfun_updates[self._log_it_counter]+=self._n_qfun_updates[self._log_it_counter-1]
            self._n_tqfun_updates[self._log_it_counter]+=self._n_tqfun_updates[self._log_it_counter-1]

            self._policy_update_fps[self._log_it_counter] = (self._n_policy_updates[self._log_it_counter]-\
                self._n_policy_updates[self._log_it_counter-1])/self._policy_update_dt[self._log_it_counter]

            self._elapsed_min[self._log_it_counter] = (time.perf_counter() - self._start_time_tot)/60.0

            self._n_of_played_episodes[self._log_it_counter] = self._episodic_reward_metrics.get_n_played_episodes(env_selector=self._db_env_selector)

            self._ep_tsteps_env_distribution[self._log_it_counter, :]=\
                self._episodic_reward_metrics.step_counters(env_selector=self._db_env_selector)*self._env_n_action_reps

            # updating episodic reward metrics
            # debug environments
            self._tot_rew_max[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_tot_rew_max(env_selector=self._db_env_selector)
            self._tot_rew_avrg[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_tot_rew_avrg(env_selector=self._db_env_selector)
            self._tot_rew_min[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_tot_rew_min(env_selector=self._db_env_selector)
            self._tot_rew_max_over_envs[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_tot_rew_max_over_envs(env_selector=self._db_env_selector)
            self._tot_rew_avrg_over_envs[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_tot_rew_avrg_over_envs(env_selector=self._db_env_selector)
            self._tot_rew_min_over_envs[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_tot_rew_min_over_envs(env_selector=self._db_env_selector)

            self._sub_rew_max[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_sub_rew_max(env_selector=self._db_env_selector)
            self._sub_rew_avrg[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_sub_rew_avrg(env_selector=self._db_env_selector)
            self._sub_rew_min[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_sub_rew_min(env_selector=self._db_env_selector)
            self._sub_rew_max_over_envs[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_sub_rew_max_over_envs(env_selector=self._db_env_selector)
            self._sub_rew_avrg_over_envs[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_sub_rew_avrg_over_envs(env_selector=self._db_env_selector)
            self._sub_rew_min_over_envs[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_sub_rew_min_over_envs(env_selector=self._db_env_selector)
            

            # fill env custom db metrics (only for debug environments)
            db_data_names = list(self._env.custom_db_data.keys())
            for dbdatan in db_data_names:
                self._custom_env_data[dbdatan]["max"][self._log_it_counter, :, :] = self._env.custom_db_data[dbdatan].get_max(env_selector=self._db_env_selector)
                self._custom_env_data[dbdatan]["avrg"][self._log_it_counter, :, :] = self._env.custom_db_data[dbdatan].get_avrg(env_selector=self._db_env_selector)
                self._custom_env_data[dbdatan]["min"][self._log_it_counter, :, :] = self._env.custom_db_data[dbdatan].get_min(env_selector=self._db_env_selector)
                self._custom_env_data[dbdatan]["max_over_envs"][self._log_it_counter, :, :] = self._env.custom_db_data[dbdatan].get_max_over_envs(env_selector=self._db_env_selector)
                self._custom_env_data[dbdatan]["avrg_over_envs"][self._log_it_counter, :, :] = self._env.custom_db_data[dbdatan].get_avrg_over_envs(env_selector=self._db_env_selector)
                self._custom_env_data[dbdatan]["min_over_envs"][self._log_it_counter, :, :] = self._env.custom_db_data[dbdatan].get_min_over_envs(env_selector=self._db_env_selector)

            # exploration envs
            if self._n_expl_envs > 0:
                self._ep_tsteps_expl_env_distribution[self._log_it_counter, :]=\
                    self._episodic_reward_metrics.step_counters(env_selector=self._expl_env_selector)*self._env_n_action_reps

                self._sub_rew_max_expl[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_sub_rew_max(env_selector=self._expl_env_selector)
                self._sub_rew_avrg_expl[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_sub_rew_avrg(env_selector=self._expl_env_selector)
                self._sub_rew_min_expl[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_sub_rew_min(env_selector=self._expl_env_selector)
                self._sub_rew_max_over_envs_expl[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_sub_rew_max_over_envs(env_selector=self._expl_env_selector)
                self._sub_rew_avrg_over_envs_expl[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_sub_rew_avrg_over_envs(env_selector=self._expl_env_selector)
                self._sub_rew_min_over_envs_expl[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_sub_rew_min_over_envs(env_selector=self._expl_env_selector)

            # demo envs
            if self._env.n_demo_envs() > 0 and self._env.demo_active():
                # only log if demo envs are active (db data will remaing to nan if that case)
                self._ep_tsteps_demo_env_distribution[self._log_it_counter, :]=\
                    self._episodic_reward_metrics.step_counters(env_selector=self._demo_env_selector)*self._env_n_action_reps

                self._sub_rew_max_demo[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_sub_rew_max(env_selector=self._demo_env_selector)
                self._sub_rew_avrg_demo[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_sub_rew_avrg(env_selector=self._demo_env_selector)
                self._sub_rew_min_demo[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_sub_rew_min(env_selector=self._demo_env_selector)
                self._sub_rew_max_over_envs_demo[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_sub_rew_max_over_envs(env_selector=self._demo_env_selector)
                self._sub_rew_avrg_over_envs_demo[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_sub_rew_avrg_over_envs(env_selector=self._demo_env_selector)
                self._sub_rew_min_over_envs_demo[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_sub_rew_min_over_envs(env_selector=self._demo_env_selector)

            # other data
            if self._agent.running_norm is not None:
                self._running_mean_obs[self._log_it_counter, :] = self._agent.running_norm.get_current_mean()
                self._running_std_obs[self._log_it_counter, :] = self._agent.running_norm.get_current_std()

            # write some episodic db info on shared mem
            sub_returns=self._sub_returns.get_torch_mirror(gpu=False)
            sub_returns[:, :]=self._episodic_reward_metrics.get_sub_rew_avrg()
            tot_returns=self._tot_returns.get_torch_mirror(gpu=False)
            tot_returns[:, :]=self._episodic_reward_metrics.get_tot_rew_avrg_over_envs()
            self._sub_returns.synch_all(read=False)
            self._tot_returns.synch_all(read=False)
            
            self._log_info()

            self._log_it_counter+=1 

        if self._dump_checkpoints and \
            (self._vec_transition_counter % self._m_checkpoint_freq == 0):
            self._save_model(is_checkpoint=True)

        if self._full_env_db and \
            (self._vec_transition_counter % self._env_db_checkpoints_vecfreq == 0):
            self._dump_env_checkpoints()

        if self._vec_transition_counter == self._total_timesteps_vec:
            self.done()           
            
    def _should_have_called_setup(self):

        exception = f"setup() was not called!"

        Journal.log(self.__class__.__name__,
            "_should_have_called_setup",
            exception,
            LogType.EXCEP,
            throw_when_excep = True)
    
    def _log_info(self):
        
        if self._debug or self._verbose:
            elapsed_h = self._elapsed_min[self._log_it_counter].item()/60.0
            est_remaining_time_h =  elapsed_h * 1/(self._vec_transition_counter) * (self._total_timesteps_vec-self._vec_transition_counter)
            is_done=self._vec_transition_counter==self._total_timesteps_vec

            actual_tsteps_with_updates=-1
            experience_to_policy_grad_ratio=-1
            experience_to_qfun_grad_ratio=-1
            experience_to_tqfun_grad_ratio=-1
            if not self._eval:
                actual_tsteps_with_updates=(self._n_timesteps_done[self._log_it_counter].item()-self._warmstart_timesteps)
                epsi=1e-6 # to avoid div by 0
                experience_to_policy_grad_ratio=actual_tsteps_with_updates/(self._n_policy_updates[self._log_it_counter].item()-epsi)
                experience_to_qfun_grad_ratio=actual_tsteps_with_updates/(self._n_qfun_updates[self._log_it_counter].item()-epsi)
                experience_to_tqfun_grad_ratio=actual_tsteps_with_updates/(self._n_tqfun_updates[self._log_it_counter].item()-epsi)
     
        if self._debug:

            if self._remote_db: 
                # write general algo debug info to shared memory    
                info_names=self._shared_algo_data.dynamic_info.get()
                info_data = [
                    self._n_timesteps_done[self._log_it_counter].item(),
                    self._n_policy_updates[self._log_it_counter].item(),
                    experience_to_policy_grad_ratio,
                    elapsed_h,
                    est_remaining_time_h,
                    self._env_step_fps[self._log_it_counter].item(),
                    self._env_step_rt_factor[self._log_it_counter].item(),
                    self._collection_dt[self._log_it_counter].item(),
                    self._policy_update_fps[self._log_it_counter].item(),
                    self._policy_update_dt[self._log_it_counter].item(),
                    is_done,
                    self._n_of_played_episodes[self._log_it_counter].item(),
                    self._batch_norm_update_dt[self._log_it_counter].item(),
                    ]
                self._shared_algo_data.write(dyn_info_name=info_names,
                                        val=info_data)

                # write debug info to remote wandb server
                db_data_names = list(self._env.custom_db_data.keys())
                for dbdatan in db_data_names: 
                    data = self._custom_env_data[dbdatan]
                    data_names = self._env.custom_db_data[dbdatan].data_names()

                    self._custom_env_data_db_dict.update({f"env_dbdata/{dbdatan}" + "_max": 
                            wandb.Histogram(data["max"][self._log_it_counter, :, :].numpy())})
                    self._custom_env_data_db_dict.update({f"env_dbdata/{dbdatan}" + "_avrg": 
                            wandb.Histogram(data["avrg"][self._log_it_counter, :, :].numpy())})
                    self._custom_env_data_db_dict.update({f"env_dbdata/{dbdatan}" + "_min": 
                            wandb.Histogram(data["min"][self._log_it_counter, :, :].numpy())})
            
                    self._custom_env_data_db_dict.update({f"env_dbdata/{dbdatan}-{data_names[i]}" + "_max_over_envs": 
                        data["max_over_envs"][self._log_it_counter, :, i:i+1] for i in range(len(data_names))})
                    self._custom_env_data_db_dict.update({f"env_dbdata/{dbdatan}-{data_names[i]}" + "_avrg_over_envs": 
                        data["avrg_over_envs"][self._log_it_counter, :, i:i+1] for i in range(len(data_names))})
                    self._custom_env_data_db_dict.update({f"env_dbdata/{dbdatan}-{data_names[i]}" + "_min_over_envs": 
                        data["min_over_envs"][self._log_it_counter, :, i:i+1] for i in range(len(data_names))})
                
                self._wandb_d.update({'log_iteration' : self._log_it_counter})
                self._wandb_d.update(dict(zip(info_names, info_data)))

                # debug environments
                self._wandb_d.update({'correlation_db/ep_timesteps_env_distr': 
                    wandb.Histogram(self._ep_tsteps_env_distribution[self._log_it_counter, :, :].numpy())})

                self._wandb_d.update({'tot_reward/tot_rew_max': wandb.Histogram(self._tot_rew_max[self._log_it_counter, :, :].numpy()),
                    'tot_reward/tot_rew_avrg': wandb.Histogram(self._tot_rew_avrg[self._log_it_counter, :, :].numpy()),
                    'tot_reward/tot_rew_min': wandb.Histogram(self._tot_rew_min[self._log_it_counter, :, :].numpy()),
                    'tot_reward/tot_rew_max_over_envs': self._tot_rew_max_over_envs[self._log_it_counter, :, :].item(),
                    'tot_reward/tot_rew_avrg_over_envs': self._tot_rew_avrg_over_envs[self._log_it_counter, :, :].item(),
                    'tot_reward/tot_rew_min_over_envs': self._tot_rew_min_over_envs[self._log_it_counter, :, :].item()})
                # sub rewards from db envs
                self._wandb_d.update({f"sub_reward/{self._reward_names[i]}_sub_rew_max":
                        wandb.Histogram(self._sub_rew_max.numpy()[self._log_it_counter, :, i:i+1]) for i in range(len(self._reward_names))})
                self._wandb_d.update({f"sub_reward/{self._reward_names[i]}_sub_rew_avrg":
                        wandb.Histogram(self._sub_rew_avrg.numpy()[self._log_it_counter, :, i:i+1]) for i in range(len(self._reward_names))})
                self._wandb_d.update({f"sub_reward/{self._reward_names[i]}_sub_rew_min":
                        wandb.Histogram(self._sub_rew_min.numpy()[self._log_it_counter, :, i:i+1]) for i in range(len(self._reward_names))})
            
                self._wandb_d.update({f"sub_reward/{self._reward_names[i]}_sub_rew_max_over_envs":
                        self._sub_rew_max_over_envs[self._log_it_counter, :, i:i+1] for i in range(len(self._reward_names))})
                self._wandb_d.update({f"sub_reward/{self._reward_names[i]}_sub_rew_avrg_over_envs":
                        self._sub_rew_avrg_over_envs[self._log_it_counter, :, i:i+1] for i in range(len(self._reward_names))})
                self._wandb_d.update({f"sub_reward/{self._reward_names[i]}_sub_rew_min_over_envs":
                        self._sub_rew_min_over_envs[self._log_it_counter, :, i:i+1] for i in range(len(self._reward_names))})
                
                # exploration envs
                if self._n_expl_envs > 0:
                    self._wandb_d.update({'correlation_db/ep_timesteps_expl_env_distr': 
                        wandb.Histogram(self._ep_tsteps_expl_env_distribution[self._log_it_counter, :, :].numpy())})

                    # sub reward from expl envs
                    self._wandb_d.update({f"sub_reward_expl/{self._reward_names[i]}_sub_rew_max_expl":
                            wandb.Histogram(self._sub_rew_max_expl.numpy()[self._log_it_counter, :, i:i+1]) for i in range(len(self._reward_names))})
                    self._wandb_d.update({f"sub_reward_expl/{self._reward_names[i]}_sub_rew_avrg_expl":
                            wandb.Histogram(self._sub_rew_avrg_expl.numpy()[self._log_it_counter, :, i:i+1]) for i in range(len(self._reward_names))})
                    self._wandb_d.update({f"sub_reward_expl/{self._reward_names[i]}_sub_rew_min_expl":
                            wandb.Histogram(self._sub_rew_min_expl.numpy()[self._log_it_counter, :, i:i+1]) for i in range(len(self._reward_names))})
                
                    self._wandb_d.update({f"sub_reward_expl/{self._reward_names[i]}_sub_rew_max_over_envs_expl":
                            self._sub_rew_max_over_envs_expl[self._log_it_counter, :, i:i+1] for i in range(len(self._reward_names))})
                    self._wandb_d.update({f"sub_reward_expl/{self._reward_names[i]}_sub_rew_avrg_over_envs_expl":
                            self._sub_rew_avrg_over_envs_expl[self._log_it_counter, :, i:i+1] for i in range(len(self._reward_names))})
                    self._wandb_d.update({f"sub_reward_expl/{self._reward_names[i]}_sub_rew_min_over_envs_expl":
                            self._sub_rew_min_over_envs_expl[self._log_it_counter, :, i:i+1] for i in range(len(self._reward_names))})
                
                # demo envs (only log if active, otherwise we log nans which wandb doesn't like)
                if self._env.n_demo_envs() > 0:
                    if self._env.demo_active():
                        # log hystograms only if there are no nan in the data
                        self._wandb_d.update({'correlation_db/ep_timesteps_demo_env_distr':
                            wandb.Histogram(self._ep_tsteps_demo_env_distribution[self._log_it_counter, :, :].numpy())})

                        # sub reward from expl envs
                        self._wandb_d.update({f"sub_reward_demo/{self._reward_names[i]}_sub_rew_max_demo":
                                wandb.Histogram(self._sub_rew_max_demo.numpy()[self._log_it_counter, :, i:i+1]) for i in range(len(self._reward_names))})
                        self._wandb_d.update({f"sub_reward_demo/{self._reward_names[i]}_sub_rew_avrg_demo":
                                wandb.Histogram(self._sub_rew_avrg_demo.numpy()[self._log_it_counter, :, i:i+1]) for i in range(len(self._reward_names))})
                        self._wandb_d.update({f"sub_reward_demo/{self._reward_names[i]}_sub_rew_min_demo":
                                wandb.Histogram(self._sub_rew_min_demo.numpy()[self._log_it_counter, :, i:i+1]) for i in range(len(self._reward_names))})
                
                    self._wandb_d.update({f"sub_reward_demo/{self._reward_names[i]}_sub_rew_max_over_envs_demo":
                            self._sub_rew_max_over_envs_demo[self._log_it_counter, :, i:i+1] for i in range(len(self._reward_names))})
                    self._wandb_d.update({f"sub_reward_demo/{self._reward_names[i]}_sub_rew_avrg_over_envs_demo":
                            self._sub_rew_avrg_over_envs_demo[self._log_it_counter, :, i:i+1] for i in range(len(self._reward_names))})
                    self._wandb_d.update({f"sub_reward_demo/{self._reward_names[i]}_sub_rew_min_over_envs_demo":
                            self._sub_rew_min_over_envs_demo[self._log_it_counter, :, i:i+1] for i in range(len(self._reward_names))})
                    
                if self._vec_transition_counter > (self._warmstart_vectimesteps-1):
                    # algo info
                    self._policy_update_db_data_dict.update({
                        "sac_q_info/qf1_vals_mean": self._qf1_vals_mean[self._log_it_counter, 0],
                        "sac_q_info/qf2_vals_mean": self._qf2_vals_mean[self._log_it_counter, 0],
                        "sac_q_info/qf1_vals_std": self._qf1_vals_std[self._log_it_counter, 0],
                        "sac_q_info/qf2_vals_std": self._qf2_vals_std[self._log_it_counter, 0],
                        "sac_q_info/qf1_vals_max": self._qf1_vals_max[self._log_it_counter, 0],
                        "sac_q_info/qf2_vals_max": self._qf2_vals_max[self._log_it_counter, 0],
                        "sac_q_info/qf1_vals_min": self._qf1_vals_min[self._log_it_counter, 0],
                        "sac_q_info/qf2_vals_min": self._qf2_vals_min[self._log_it_counter, 0],

                        "sac_actor_info/policy_entropy_mean": self._policy_entropy_mean[self._log_it_counter, 0],
                        "sac_actor_info/policy_entropy_std": self._policy_entropy_std[self._log_it_counter, 0],
                        "sac_actor_info/policy_entropy_max": self._policy_entropy_max[self._log_it_counter, 0],
                        "sac_actor_info/policy_entropy_min": self._policy_entropy_min[self._log_it_counter, 0],
                        
                        "sac_q_info/qf1_loss": self._qf1_loss[self._log_it_counter, 0],
                        "sac_q_info/qf2_loss": self._qf2_loss[self._log_it_counter, 0],
                        "sac_actor_info/actor_loss": self._actor_loss[self._log_it_counter, 0],
                        "sac_alpha_info/alpha_loss": self._alpha_loss[self._log_it_counter, 0],

                        "sac_alpha_info/alpha": self._alphas[self._log_it_counter, 0],
                        "sac_alpha_info/target_entropy": self._target_entropy})
                    
                    if self._validate:
                        self._policy_update_db_data_dict.update({
                            "sac_q_info/qf1_loss_validation": self._qf1_loss_validation[self._log_it_counter, 0],
                            "sac_q_info/qf2_loss_validation": self._qf2_loss_validation[self._log_it_counter, 0],
                            "sac_actor_info/actor_loss_validation": self._actor_loss_validation[self._log_it_counter, 0],
                            "sac_alpha_info/alpha_loss_validation": self._alpha_loss_validation[self._log_it_counter, 0]})
                    
                    self._wandb_d.update(self._policy_update_db_data_dict)

                if self._agent.running_norm is not None:
                    # adding info on running normalizer if used
                    self._wandb_d.update({f"running_norm/mean": self._running_mean_obs[self._log_it_counter, :]})
                    self._wandb_d.update({f"running_norm/std": self._running_std_obs[self._log_it_counter, :]})
                
                self._wandb_d.update(self._custom_env_data_db_dict) 
                
                wandb.log(self._wandb_d)

        if self._verbose:
                       
            info =f"\nTotal n. timesteps simulated: {self._n_timesteps_done[self._log_it_counter].item()}/{self._total_timesteps}\n" + \
                f"N. policy updates performed: {self._n_policy_updates[self._log_it_counter].item()}/{self._n_policy_updates_to_be_done}\n" + \
                f"N. q fun updates performed: {self._n_qfun_updates[self._log_it_counter].item()}/{self._n_qf_updates_to_be_done}\n" + \
                f"N. trgt q fun updates performed: {self._n_tqfun_updates[self._log_it_counter].item()}/{self._n_tqf_updates_to_be_done}\n" + \
                f"experience to policy grad ratio: {experience_to_policy_grad_ratio}\n" + \
                f"experience to q fun grad ratio: {experience_to_qfun_grad_ratio}\n" + \
                f"experience to trgt q fun grad ratio: {experience_to_tqfun_grad_ratio}\n"+ \
                f"Warmstart completed: {self._vec_transition_counter > self._warmstart_vectimesteps or self._eval} ; ({self._vec_transition_counter}/{self._warmstart_vectimesteps})\n" +\
                f"Replay buffer full: {self._replay_bf_full}; current position {self._bpos}/{self._replay_buffer_size_vec}\n" +\
                f"Validation buffer full: {self._validation_bf_full}; current position {self._bpos_val}/{self._validation_buffer_size_vec}\n" +\
                f"Elapsed time: {self._elapsed_min[self._log_it_counter].item()/60.0} h\n" + \
                f"Estimated remaining training time: " + \
                f"{est_remaining_time_h} h\n" + \
                f"Total reward episodic data --> \n" + \
                f"max: {self._tot_rew_max_over_envs[self._log_it_counter, :, :].item()}\n" + \
                f"avg: {self._tot_rew_avrg_over_envs[self._log_it_counter, :, :].item()}\n" + \
                f"min: {self._tot_rew_min_over_envs[self._log_it_counter, :, :].item()}\n" + \
                f"Episodic sub-rewards episodic data --> \nsub rewards names: {self._reward_names_str}\n" + \
                f"max: {self._sub_rew_max_over_envs[self._log_it_counter, :]}\n" + \
                f"avg: {self._sub_rew_avrg_over_envs[self._log_it_counter, :]}\n" + \
                f"min: {self._sub_rew_min_over_envs[self._log_it_counter, :]}\n" + \
                f"N. of episodes on which episodic rew stats are computed: {self._n_of_played_episodes[self._log_it_counter].item()}\n" + \
                f"Current env. step sps: {self._env_step_fps[self._log_it_counter].item()}, time for experience collection {self._collection_dt[self._log_it_counter].item()} s\n" + \
                f"Current env (sub-stepping) rt factor: {self._env_step_rt_factor[self._log_it_counter].item()}\n" + \
                f"Current policy update fps: {self._policy_update_fps[self._log_it_counter].item()}, time for policy updates {self._policy_update_dt[self._log_it_counter].item()} s\n" + \
                f"Time spent updating batch normalizations {self._batch_norm_update_dt[self._log_it_counter].item()} s\n" + \
                f"Time spent for computing validation {self._validation_dt[self._log_it_counter].item()} s\n" + \
                f"Demo envs are active: {self._demo_envs_active[self._log_it_counter].item()}. N  demo envs if active {self._env.n_demo_envs()}\n" + \
                f"Performance metric now: {self._demo_perf_metric[self._log_it_counter].item()}/{self._demo_stop_thresh}\n"
            
            Journal.log(self.__class__.__name__,
                "_post_step",
                info,
                LogType.INFO,
                throw_when_excep = True)

    def _add_experience(self, 
            obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, 
            next_obs: torch.Tensor, 
            next_terminal: torch.Tensor) -> None:
        
        if self._validate and \
            (self._vec_transition_counter % self._validation_collection_vecfreq == 0):
            # fill validation buffer
            
            self._obs_val[self._bpos_val] = obs
            self._next_obs_val[self._bpos_val] = next_obs
            self._actions_val[self._bpos_val] = actions
            self._rewards_val[self._bpos_val] = rewards
            self._next_terminal_val[self._bpos_val] = next_terminal

            self._bpos_val += 1
            if self._bpos_val == self._validation_buffer_size_vec:
                self._validation_bf_full = True
                self._bpos_val = 0

        else: # fill normal replay buffer
            self._obs[self._bpos] = obs
            self._next_obs[self._bpos] = next_obs
            self._actions[self._bpos] = actions
            self._rewards[self._bpos] = rewards
            self._next_terminal[self._bpos] = next_terminal

            self._bpos += 1
            if self._bpos == self._replay_buffer_size_vec:
                self._replay_bf_full = True
                self._bpos = 0

    def _sample(self, size: int = None):
        
        if size is None:
            size=self._batch_size

        batched_obs = self._obs.view((-1, self._env.obs_dim()))
        batched_next_obs = self._next_obs.view((-1, self._env.obs_dim()))
        batched_actions = self._actions.view((-1, self._env.actions_dim()))
        batched_rewards = self._rewards.view(-1)
        batched_terminal = self._next_terminal.view(-1)

        # sampling from the batched buffer
        up_to = self._replay_buffer_size if self._replay_bf_full else self._bpos*self._num_envs
        shuffled_buffer_idxs = torch.randint(0, up_to,
                                        (size,)) 
        
        sampled_obs = batched_obs[shuffled_buffer_idxs]
        sampled_next_obs = batched_next_obs[shuffled_buffer_idxs]
        sampled_actions = batched_actions[shuffled_buffer_idxs]
        sampled_rewards = batched_rewards[shuffled_buffer_idxs]
        sampled_terminal = batched_terminal[shuffled_buffer_idxs]

        return sampled_obs, sampled_actions,\
            sampled_next_obs,\
            sampled_rewards, \
            sampled_terminal
    
    def _sample_validation(self, size: int = None):
        
        if size is None:
            size=self._validation_batch_size

        batched_obs = self._obs_val.view((-1, self._env.obs_dim()))
        batched_next_obs = self._next_obs_val.view((-1, self._env.obs_dim()))
        batched_actions = self._actions_val.view((-1, self._env.actions_dim()))
        batched_rewards = self._rewards_val.view(-1)
        batched_terminal = self._next_terminal_val.view(-1)

        # sampling from the batched buffer
        up_to = self._validation_buffer_size if self._validation_bf_full else self._bpos_val*self._num_envs
        shuffled_buffer_idxs = torch.randint(0, up_to,
                                        (size,)) 
        
        sampled_obs = batched_obs[shuffled_buffer_idxs]
        sampled_next_obs = batched_next_obs[shuffled_buffer_idxs]
        sampled_actions = batched_actions[shuffled_buffer_idxs]
        sampled_rewards = batched_rewards[shuffled_buffer_idxs]
        sampled_terminal = batched_terminal[shuffled_buffer_idxs]

        return sampled_obs, sampled_actions,\
            sampled_next_obs,\
            sampled_rewards, \
            sampled_terminal
    
    def _sample_random_actions(self):
        
        self._random_uniform.uniform_(-1,1)
        random_actions = self._random_uniform*self._action_scale+self._action_offset

        return random_actions
    
    def _perturb_some_actions(self,
            actions: torch.Tensor):
        
        if self._is_continuous_actions.any(): # if there are any continuous actions
            self._perturb_actions(actions,
                action_idxs=self._is_continuous_actions, 
                env_idxs=self._expl_env_selector.to(actions.device),
                normal=True, # use normal for continuous
                scaling=self._continuous_act_expl_noise_std)
        if (~self._is_continuous_actions).any(): # actions to be treated as discrete
            self._perturb_actions(actions,
                action_idxs=~self._is_continuous_actions, 
                env_idxs=self._expl_env_selector.to(actions.device),
                normal=False, # use uniform distr for discrete
                scaling=self._discrete_act_expl_noise_std)
        
        self._pert_counter+=1
        if self._pert_counter >= self._noise_duration:
            self._pert_counter=0
    
    def _perturb_actions(self, 
        actions: torch.Tensor,
        action_idxs: torch.Tensor, 
        env_idxs: torch.Tensor,
        normal: bool = True,
        scaling: float = 1.0):
        if normal: # gaussian
            # not super efficient (in theory random_normal can be made smaller in size)
            self._random_normal.normal_(mean=0, std=1)
            noise=self._random_normal
        else: # uniform
            self._random_uniform.uniform_(-1,1)
            noise=self._random_uniform
        
        env_indices = env_idxs.reshape(-1,1)  # Get indices of True environments
        action_indices = action_idxs.reshape(1,-1) # Get indices of True actions

        actions[env_indices, action_indices]=\
            actions[env_indices, action_indices]+noise[env_indices, action_indices]*self._action_scale[:,action_indices]*scaling
    
    def _update_batch_norm(self, bsize: int = None):

        if bsize is None:
            bsize=self._batch_size # same used for training

        # update obs normalization        
        # (we should sample also next obs, but if most of the transitions are not terminal, 
        # this is not an issue and is more efficient)
        if (self._agent.running_norm is not None) and \
            (not self._eval):
            batched_obs = self._obs.view((-1, self._env.obs_dim()))
            up_to = self._replay_buffer_size if self._replay_bf_full else self._bpos*self._num_envs
            shuffled_buffer_idxs = torch.randint(0, up_to,
                                            (bsize,)) 
            sampled_obs = batched_obs[shuffled_buffer_idxs]
            self._agent.update_obs_bnorm(x=sampled_obs)
            
    def _switch_training_mode(self, 
                    train: bool = True):

        self._agent.train(train)

    def _init_algo_shared_data(self,
                static_params: Dict):

        self._shared_algo_data = SharedRLAlgorithmInfo(namespace=self._ns,
                is_server=True, 
                static_params=static_params,
                verbose=self._verbose, 
                vlevel=VLevel.V2, 
                safe=False,
                force_reconnection=True)

        self._shared_algo_data.run()

        # write some initializations
        self._shared_algo_data.write(dyn_info_name=["is_done"],
                val=[0.0])
        
        # only written to if flags where enabled
        self._qf_vals=QfVal(namespace=self._ns,
            is_server=True, 
            n_envs=self._num_envs, 
            verbose=self._verbose, 
            vlevel=VLevel.V2,
            safe=False,
            force_reconnection=True)
        self._qf_vals.run()
        self._qf_trgt=QfTrgt(namespace=self._ns,
            is_server=True, 
            n_envs=self._num_envs, 
            verbose=self._verbose, 
            vlevel=VLevel.V2,
            safe=False,
            force_reconnection=True)
        self._qf_trgt.run()

        # episodic returns
        reward_names=self._episodic_reward_metrics.data_names()
        self._sub_returns=SubReturns(namespace=self._ns,
            is_server=True, 
            n_envs=self._num_envs, 
            n_rewards=len(reward_names),
            reward_names=reward_names,
            verbose=self._verbose, 
            vlevel=VLevel.V2,
            safe=False,
            force_reconnection=True)
        self._sub_returns.run()

        self._tot_returns=TotReturns(namespace=self._ns,
            is_server=True, 
            n_envs=self._num_envs, 
            verbose=self._verbose, 
            vlevel=VLevel.V2,
            safe=False,
            force_reconnection=True)
        self._tot_returns.run()
