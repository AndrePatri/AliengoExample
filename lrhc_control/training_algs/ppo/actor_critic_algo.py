from lrhc_control.agents.actor_critic.ppo import ACAgent
from lrhc_control.agents.dummies.dummy import DummyAgent

from lrhc_control.utils.shared_data.algo_infos import SharedRLAlgorithmInfo
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

class ActorCriticAlgoBase(ABC):

    # base class for actor-critic RL algorithms
     
    def __init__(self,
            env, 
            debug = False,
            remote_db = False,
            seed: int = 1):

        self._env = env 
        self._seed = seed

        self._eval = False
        self._det_eval = True

        self._full_env_db=False

        self._agent = None 
        
        self._debug = debug
        self._remote_db = remote_db

        self._writer = None
        
        self._run_name = None
        self._drop_dir = None
        self._dbinfo_drop_fname = None
        self._model_path = None
        
        self._policy_update_db_data_dict =  {}
        self._custom_env_data_db_dict = {}
        self._rnd_db_data_dict =  {}
        self._hyperparameters = {}
        self._wandb_d={}

        # get params from env
        self._get_params_from_env()

        self._torch_device = torch.device("cpu") # defaults to cpu

        self._setup_done = False

        self._verbose = False

        self._is_done = False
        
        self._shared_algo_data = None

        self._this_child_path = None
        self._this_basepath = os.path.abspath(__file__)
    
    def __del__(self):

        self.done()

    def _get_params_from_env(self):

        self._env_name = self._env.name()
        self._episodic_reward_metrics = self._env.ep_rewards_metrics()
        self._use_gpu = self._env.using_gpu()
        self._dtype = self._env.dtype()
        self._env_opts=self._env.env_opts()
        self._num_envs = self._env.n_envs()
        self._obs_dim = self._env.obs_dim()
        self._actions_dim = self._env.actions_dim()
        self._episode_timeout_lb, self._episode_timeout_ub = self._env.episode_timeout_bounds()
        self._task_rand_timeout_lb, self._task_rand_timeout_ub = self._env.task_rand_timeout_bounds()
        self._env_n_action_reps = self._env.n_action_reps()
        self._env_actions_ub=self._env.get_actions_ub()
        self._env_actions_lb=self._env.get_actions_lb()
        self._is_continuous_actions_bool=self._env.is_action_continuous()
        self._is_continuous_actions=torch.where(self._is_continuous_actions_bool)[0]
        self._is_discrete_actions_bool=self._env.is_action_discrete()
        self._is_discrete_actions=torch.where(self._is_discrete_actions_bool)[0]

    def learn(self):
        
        if not self._setup_done:
            self._should_have_called_setup()

        # first annealing the learning rate if enabled (may improve convergence)
        if self._anneal_lr:
            frac = 1.0 - (self._step_counter) / self._iterations_n
            self._lr_now_actor = frac * self._base_lr_actor
            self._lr_now_critic = frac * self._base_lr_critic
            self._optimizer.param_groups[0]["lr"] = self._lr_now_actor
            # self._optimizer.param_groups[1]["lr"] = self._lr_now_critic

        self._start_time = time.perf_counter()

        with torch.no_grad(): # don't want grad computation here

            # collect rollout experince
            rollout_ok = self._collect_rollout()
            if not rollout_ok:
                return False
            self._vec_transition_counter+=self._rollout_vec_timesteps
            self._rollout_t = time.perf_counter()
            
            # update batch normalization
            self._update_batch_norm(bsize=self._bnorm_bsize)
            self._bnorm_t = time.perf_counter()

            # generalized advantage estimation
            self._compute_returns()
            self._gae_t = time.perf_counter()

        # update policy
        self._update_policy()
        self._policy_update_t = time.perf_counter()

        with torch.no_grad():
            self._post_step()

        return not self.is_done()

    def eval(self):

        if not self._setup_done:
            self._should_have_called_setup()

        self._start_time = time.perf_counter()

        rollout_ok = self._collect_eval_rollout()
        if not rollout_ok:
            return False

        self._rollout_t = time.perf_counter()

        self._post_step()

        return not self.is_done()
    
    @abstractmethod
    def _collect_rollout(self):
        pass
    
    @abstractmethod
    def _collect_eval_rollout(self):
        pass

    @abstractmethod
    def _compute_returns(self):
       pass
    
    @abstractmethod
    def _update_policy(self):
        pass

    def setup(self,
            run_name: str,
            ns: str,
            custom_args: Dict = {},
            verbose: bool = False,
            drop_dir_name: str = None,
            eval: bool = False,
            model_path: str = None,
            n_eval_timesteps: int = None,
            comment: str = "",
            dump_checkpoints: bool = False,
            norm_obs: bool = True,
            rescale_obs: bool = False):

        tot_tsteps=int(100e6)
        if "tot_tsteps" in custom_args:
            tot_tsteps=custom_args["tot_tsteps"]

        self._verbose = verbose

        self._ns=ns # only used for shared mem stuff

        self._dump_checkpoints = dump_checkpoints
        
        self._init_algo_shared_data(static_params=self._hyperparameters) # can only handle dicts with
        # numeric values

        if "full_env_db" in custom_args:
            self._full_env_db=custom_args["full_env_db"]

        self._eval = eval

        self._override_agent_actions=False
        if "override_agent_actions" in custom_args:
            self._override_agent_actions=custom_args["override_agent_actions"]

        if self._override_agent_actions: # force evaluation mode
            Journal.log(self.__class__.__name__,
                "setup",
                "will force evaluation mode since override_agent_actions was set to true",
                LogType.INFO,
                throw_when_excep = True)
            self._eval=True
            self._det_eval=False

        self._run_name = run_name
        from datetime import datetime
        self._time_id = datetime.now().strftime('d%Y_%m_%d_h%H_m%M_s%S')
        self._unique_id = self._time_id + "-" + self._run_name

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
        
        if not self._override_agent_actions:
            self._agent = ACAgent(obs_dim=self._env.obs_dim(),
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
                            debug=self._debug,
                            layer_width_actor=layer_width_actor,
                            layer_width_critic=layer_width_critic,
                            n_hidden_layers_actor=n_hidden_layers_actor,
                            n_hidden_layers_critic=n_hidden_layers_critic,
                            torch_compile=use_torch_compile,
                            add_weight_norm=add_weight_norm,
                            out_std_actor=0.01,
                            out_std_critic=1.0)
        else: # we use a fake agent
            self._agent = DummyAgent(obs_dim=self._env.obs_dim(),
                    actions_dim=self._env.actions_dim(),
                    actions_ub=self._env.get_actions_ub().flatten().tolist(),
                    actions_lb=self._env.get_actions_lb().flatten().tolist(),
                    device=self._torch_device,
                    dtype=self._dtype,
                    debug=self._debug)
        
        # loging actual widths and layers in case they were override inside agent init
        self._hyperparameters["actor_lwidth_actual"]=self._agent.layer_width_actor()
        self._hyperparameters["actor_n_hlayers_actual"]=self._agent.n_hidden_layers_actor()
        self._hyperparameters["critic_lwidth_actual"]=self._agent.layer_width_critic()
        self._hyperparameters["critic_n_hlayers_actual"]=self._agent.n_hidden_layers_critic()

        # load model if necessary 
        if self._eval and (not self._override_agent_actions): # load pretrained model
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
                custom_args=custom_args)
        else:
            self._init_params(tot_tsteps=tot_tsteps,
                custom_args=custom_args)
        
        # adding additional db info
        self._hyperparameters["obs_names"]=self._env.obs_names()
        self._hyperparameters["action_names"]=self._env.action_names()
        self._hyperparameters["sub_reward_names"]=self._env.sub_rew_names()
        self._hyperparameters["sub_trunc_names"]=self._env.sub_trunc_names()
        self._hyperparameters["sub_term_names"]=self._env.sub_term_names()

        # reset environment
        self._env.reset()
        if self._eval:
            self._env.switch_random_reset(on=False)

        # create dump directory + copy important files for debug
        self._init_drop_dir(drop_dir_name)
        self._hyperparameters["drop_dir"]=self._drop_dir
        
        # add env options to hyperparameters
        self._hyperparameters.update(self._env_opts) 

        if not self._eval:
            self._optimizer = optim.Adam(self._agent.parameters(), 
                                    lr=self._base_lr_actor, 
                                    eps=1e-5 # small constant added to the optimization
                                    )
            # self._optimizer = optim.Adam([
            #     {'params': self._agent.actor_mean.parameters(), 'lr': self._base_lr_actor},
            #     {'params': self._agent.critic.parameters(), 'lr': self._base_lr_critic}, ],
            #     lr=self._base_lr_actor, # default to actor lr (e.g. lfor ogstd parameter)
            #     eps=1e-5 # small constant added to the optimization
            #     )
            self._init_rollout_buffers() # only needed if training
        
        self._init_dbdata()

        if (self._debug):
            if self._remote_db:
                job_type = "evaluation" if self._eval else "training"
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
                    config=self._hyperparameters,
                    monitor_gym=True,
                    save_code=True,
                    dir=self._drop_dir
                )
                wandb.watch((self._agent), log="all", log_freq=1000, log_graph=False)
                
        actions = self._env.get_actions()
        self._action_scale = self._env.get_actions_scale()
        self._action_offset = self._env.get_actions_offset()
        self._random_uniform = torch.full_like(actions, fill_value=0.0) # used for sampling random actions (preallocated
        # for efficiency)
        self._random_normal = torch.full_like(self._random_uniform,fill_value=0.0)
        # for efficiency)

        self._actions_override=None            
        if self._override_agent_actions:
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

        self._is_done = False
        self._setup_done = True

    def is_done(self):

        return self._is_done 
    
    def model_path(self):

        return self._model_path

    def _init_params(self,
            tot_tsteps: int,
            custom_args: Dict = {}):

        # policy rollout and return comp./adv estimation
        self._total_timesteps = int(tot_tsteps) # total timesteps to be collected (including sub envs)
        # self._total_timesteps = self._total_timesteps//self._env_n_action_reps # correct with n of action reps
        
        self._rollout_vec_timesteps = 128 # numer of vectorized steps (rescaled depending on env substepping) 
        # to be done per policy rollout (influences adv estimation!!!)
        self._batch_size = self._rollout_vec_timesteps * self._num_envs

        self._iterations_n = self._total_timesteps//self._batch_size # number of ppo iterations
        self._total_timesteps = self._iterations_n*self._batch_size # actual number of total tsteps to be simulated
        self._total_timesteps_vec = self._iterations_n*self._rollout_vec_timesteps

        self._bnorm_bsize = 4096 # size of batch used for batch normalization

        # policy update
        self._num_minibatches = 8
        self._minibatch_size = self._batch_size // self._num_minibatches
        
        self._base_lr_actor = 1e-3 
        self._base_lr_critic = 5e-4
        self._lr_now_actor = self._base_lr_actor
        self._lr_now_critic= self._base_lr_critic
        self._anneal_lr = False

        self._discount_factor = 0.99
        self._gae_lambda = 0.95 # λ = 1 gives an unbiased estimate of the total reward (but high variance),
        # λ < 1 gives a biased estimate, but with less variance. 0.95
        
        self._update_epochs = 10
        self._norm_adv = True
        self._clip_vloss = False
        self._clip_coef_vf = 0.2 # IMPORTANT: this clipping depends on the reward scaling (only used if clip_vloss)
        self._clip_coef = 0.2
        self._entropy_coeff = 5e-3
        self._val_f_coeff = 0.5
        self._max_grad_norm_actor = 0.5
        self._max_grad_norm_critic = 0.5
        self._target_kl = None

        self._n_policy_updates_to_be_done = self._update_epochs*self._num_minibatches*self._iterations_n
        self._n_vfun_updates_to_be_done=self._n_policy_updates_to_be_done

        self._exp_to_policy_grad_ratio=float(self._total_timesteps)/float(self._n_policy_updates_to_be_done)
        self._exp_to_vf_grad_ratio=float(self._total_timesteps)/float(self._n_vfun_updates_to_be_done)

        # debug
        self._m_checkpoint_freq_nom = 1e6 # n total timesteps after which a checkpoint model is dumped
        self._m_checkpoint_freq= self._m_checkpoint_freq_nom//self._num_envs
        self._checkpoint_nit = round(self._m_checkpoint_freq/self._rollout_vec_timesteps)
        self._m_checkpoint_freq = self._rollout_vec_timesteps*self._checkpoint_nit # ensuring _m_checkpoint_freq
        # is a multiple of self._rollout_vec_timesteps

        self._db_vecstep_frequency = 32 
        if self._db_vecstep_frequency<self._rollout_vec_timesteps:
            self._db_vecstep_frequency=self._rollout_vec_timesteps
        self._db_vecstep_freq_it = round(self._db_vecstep_frequency/self._rollout_vec_timesteps)
        self._db_vecstep_frequency = self._rollout_vec_timesteps*self._db_vecstep_freq_it # ensuring _db_vecstep_frequency
        # is a multiple of self._rollout_vec_timesteps

        self._env_db_checkpoints_vecfreq=10*self._db_vecstep_frequency

        self._db_data_size = round(self._total_timesteps_vec/self._db_vecstep_frequency)+self._db_vecstep_frequency

        # write them to hyperparam dictionary for debugging
        self._hyperparameters["n_envs"] = self._num_envs
        self._hyperparameters["obs_dim"] = self._obs_dim
        self._hyperparameters["actions_dim"] = self._actions_dim

        self._hyperparameters["seed"] = self._seed
        self._hyperparameters["using_gpu"] = self._use_gpu
        self._hyperparameters["total_timesteps"] = self._total_timesteps
        self._hyperparameters["total_timesteps_vec"] = self._total_timesteps_vec
        self._hyperparameters["n_iterations"] = self._iterations_n
        self._hyperparameters["rollout_vec_timesteps"] = self._rollout_vec_timesteps

        self._hyperparameters["n_policy_updates_per_batch"] = self._update_epochs*self._num_minibatches
        self._hyperparameters["n_policy_updates_when_done"] = self._n_policy_updates_to_be_done
        self._hyperparameters["n_vf_updates_when_done"] = self._n_vfun_updates_to_be_done
        self._hyperparameters["experience_to_policy_grad_steps_ratio"] = self._exp_to_policy_grad_ratio
        self._hyperparameters["experience_to_value_fun_grad_steps_ratio"] = self._exp_to_vf_grad_ratio

        self._hyperparameters["episodes timeout lb"] = self._episode_timeout_lb
        self._hyperparameters["episodes timeout ub"] = self._episode_timeout_ub
        self._hyperparameters["task rand timeout lb"] = self._task_rand_timeout_lb
        self._hyperparameters["task rand timeout ub"] = self._task_rand_timeout_ub

        self._hyperparameters["update_epochs"] = self._update_epochs
        self._hyperparameters["num_minibatches"] = self._num_minibatches

        self._hyperparameters["bnorm_bsize"] = self._bnorm_bsize
        self._hyperparameters["m_checkpoint_freq"] = self._m_checkpoint_freq
        self._hyperparameters["db_vecstep_frequency"] = self._db_vecstep_frequency

        self._hyperparameters["batch_size"] = self._batch_size
        self._hyperparameters["minibatch_size"] = self._minibatch_size
        self._hyperparameters["base_lr_actor"] = self._base_lr_actor
        self._hyperparameters["base_lr_critic"] = self._base_lr_critic
        self._hyperparameters["anneal_lr"] = self._anneal_lr
        self._hyperparameters["discount_factor"] = self._discount_factor
        self._hyperparameters["gae_lambda"] = self._gae_lambda
        self._hyperparameters["norm_adv"] = self._norm_adv
        self._hyperparameters["clip_coef"] = self._clip_coef
        self._hyperparameters["clip_coef_vf"] = self._clip_coef_vf
        self._hyperparameters["clip_vloss"] = self._clip_vloss
        self._hyperparameters["entropy_coeff"] = self._entropy_coeff
        self._hyperparameters["val_f_coeff"] = self._val_f_coeff
        self._hyperparameters["max_grad_norm_actor"] = self._max_grad_norm_actor
        self._hyperparameters["max_grad_norm_critic"] = self._max_grad_norm_critic
        self._hyperparameters["target_kl"] = self._target_kl

        # small debug log
        info = f"\nUsing \n" + \
            f"total (vectorized) timesteps to be simulated {self._total_timesteps_vec}\n" + \
            f"total timesteps to be simulated {self._total_timesteps}\n" + \
            f"n vec. steps per policy rollout: {self._rollout_vec_timesteps}\n" + \
            f"batch_size: {self._batch_size}\n" + \
            f"num_minibatches for policy update: {self._num_minibatches}\n" + \
            f"minibatch_size: {self._minibatch_size}\n" + \
            f"per-batch update_epochs: {self._update_epochs}\n" + \
            f"iterations_n to be done: {self._iterations_n}\n" + \
            f"total policy updates to be performed: {self._n_policy_updates_to_be_done}\n" + \
            f"total v fun updates to be performed: {self._n_vfun_updates_to_be_done}\n" + \
            f"experience to policy grad ratio: {self._exp_to_policy_grad_ratio}\n" + \
            f"experience to v fun grad ratio: {self._exp_to_vf_grad_ratio}\n" 

        Journal.log(self.__class__.__name__,
            "_init_params",
            info,
            LogType.INFO,
            throw_when_excep = True)
        
        self._step_counter = 0
        self._vec_transition_counter = 0
        self._log_it_counter = 0

    def _init_dbdata(self):

        # initalize some debug data

        # rollout phase
        self._rollout_dt = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._rollout_t = -1.0

        self._env_step_fps = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._env_step_rt_factor = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        
        self._bnorm_t = -1.0
        self._batch_norm_update_dt = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
                    
        self._gae_t = -1.0
        self._gae_dt = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")

        self._policy_update_t = -1.0
        self._policy_update_dt = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._policy_update_fps = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        
        self._n_of_played_episodes = torch.full((self._db_data_size, 1), 
                    dtype=torch.int32, fill_value=0, device="cpu")
        self._n_timesteps_done = torch.full((self._db_data_size, 1), 
                    dtype=torch.int32, fill_value=0, device="cpu")
        self._n_policy_updates = torch.full((self._db_data_size, 1), 
                    dtype=torch.int32, fill_value=0, device="cpu")
        self._n_vfun_updates = torch.full((self._db_data_size, 1), 
                    dtype=torch.int32, fill_value=0, device="cpu")
        self._elapsed_min = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0, device="cpu")
        
        self._ep_tsteps_env_distribution = torch.full((self._db_data_size, self._num_envs, 1), 
                    dtype=torch.int32, fill_value=-1, device="cpu")

        self._reward_names = self._episodic_reward_metrics.reward_names()
        self._reward_names_str = "[" + ', '.join(self._reward_names) + "]"
        self._n_rewards = self._episodic_reward_metrics.n_rewards()

        self._learning_rates = torch.full((self._db_data_size, 2), 
                    dtype=torch.float32, fill_value=0, device="cpu")

        # db environments
        self._tot_rew_max = torch.full((self._db_data_size, self._num_envs, 1), 
            dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._tot_rew_avrg = torch.full((self._db_data_size, self._num_envs, 1), 
            dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._tot_rew_min = torch.full((self._db_data_size, self._num_envs, 1), 
            dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._tot_rew_max_over_envs = torch.full((self._db_data_size, 1, 1), 
            dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._tot_rew_avrg_over_envs = torch.full((self._db_data_size, 1, 1), 
            dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._tot_rew_min_over_envs = torch.full((self._db_data_size, 1, 1), 
            dtype=torch.float32, fill_value=torch.nan, device="cpu")
        
        self._sub_rew_max = torch.full((self._db_data_size, self._num_envs, self._n_rewards), 
            dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._sub_rew_avrg = torch.full((self._db_data_size, self._num_envs, self._n_rewards), 
            dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._sub_rew_min = torch.full((self._db_data_size, self._num_envs, self._n_rewards), 
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

            max = self._env.custom_db_data[dbdatan].get_max().reshape(self._num_envs, -1)
            avrg = self._env.custom_db_data[dbdatan].get_avrg().reshape(self._num_envs, -1)
            min = self._env.custom_db_data[dbdatan].get_min().reshape(self._num_envs, -1)
            max_over_envs = self._env.custom_db_data[dbdatan].get_max_over_envs().reshape(1, -1)
            avrg_over_envs = self._env.custom_db_data[dbdatan].get_avrg_over_envs().reshape(1, -1)
            min_over_envs = self._env.custom_db_data[dbdatan].get_min_over_envs().reshape(1, -1)

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

        # algorithm-specific db info
        self._tot_loss_mean = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._value_loss_mean = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._policy_loss_mean = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._entropy_loss_mean = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._tot_loss_grad_norm_mean = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._actor_loss_grad_norm_mean = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        
        self._tot_loss_std = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._value_loss_std = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._policy_loss_std = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._entropy_loss_std = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._tot_loss_grad_norm_std = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._actor_loss_grad_norm_std = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        
        self._old_approx_kl_mean = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._approx_kl_mean = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._old_approx_kl_std = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._approx_kl_std = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        
        self._clipfrac_mean = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._clipfrac_std= torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        
        self._explained_variance = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        
        self._batch_returns_std = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._batch_returns_mean = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._batch_adv_std = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._batch_adv_mean = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._batch_val_std = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._batch_val_mean = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        
        self._running_mean_obs=None
        self._running_std_obs=None
        if self._agent.running_norm is not None and not self._eval:
            # some db data for the agent
            self._running_mean_obs = torch.full((self._db_data_size, self._env.obs_dim()), 
                        dtype=torch.float32, fill_value=0.0, device="cpu")
            self._running_std_obs = torch.full((self._db_data_size, self._env.obs_dim()), 
                        dtype=torch.float32, fill_value=0.0, device="cpu")

    def _init_rollout_buffers(self):

        self._obs = torch.full(size=(self._rollout_vec_timesteps, self._num_envs, self._obs_dim),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device) 
        self._values = torch.full(size=(self._rollout_vec_timesteps, self._num_envs, 1),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
            
        self._actions = torch.full(size=(self._rollout_vec_timesteps, self._num_envs, self._actions_dim),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._logprobs = torch.full(size=(self._rollout_vec_timesteps, self._num_envs, 1),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)

        self._next_obs = torch.full(size=(self._rollout_vec_timesteps, self._num_envs, self._obs_dim),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device) 
        self._next_values = torch.full(size=(self._rollout_vec_timesteps, self._num_envs, 1),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._next_terminal = torch.full(size=(self._rollout_vec_timesteps, self._num_envs, 1),
                        fill_value=False,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._next_done = torch.full(size=(self._rollout_vec_timesteps, self._num_envs, 1),
                        fill_value=False,
                        dtype=self._dtype,
                        device=self._torch_device)
        
        self._rewards = torch.full(size=(self._rollout_vec_timesteps, self._num_envs, 1),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
        
        
        self._advantages = torch.full(size=(self._rollout_vec_timesteps, self._num_envs, 1),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._returns = torch.full(size=(self._rollout_vec_timesteps, self._num_envs, 1),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)

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

                for key, value in self._hyperparameters.items():
                    if value is None:
                        value = "None"
        
                # full training envs
                sub_rew_full=self._episodic_reward_metrics.get_full_episodic_subrew()
                tot_rew_full=self._episodic_reward_metrics.get_full_episodic_totrew()

                if self._n_expl_envs > 0:
                    sub_rew_full_expl=self._episodic_reward_metrics.get_full_episodic_subrew(env_selector=self._expl_env_selector)
                    tot_rew_full_expl=self._episodic_reward_metrics.get_full_episodic_totrew(env_selector=self._expl_env_selector)
                if self._env.n_demo_envs() > 0:
                    sub_rew_full_demo=self._episodic_reward_metrics.get_full_episodic_subrew(env_selector=self._demo_env_selector)
                    tot_rew_full_demo=self._episodic_reward_metrics.get_full_episodic_totrew(env_selector=self._demo_env_selector)

                ep_vec_freq=self._episodic_reward_metrics.ep_vec_freq() # assuming all db data was collected with the same ep_vec_freq

                hf.attrs['sub_reward_names'] = self._reward_names
                hf.attrs['log_iteration'] = self._log_it_counter
                hf.attrs['n_timesteps_done'] = self._n_timesteps_done[self._log_it_counter]
                hf.attrs['n_policy_updates'] = self._n_policy_updates[self._log_it_counter]
                hf.attrs['elapsed_min'] = self._elapsed_min[self._log_it_counter]
                hf.attrs['ep_vec_freq'] = ep_vec_freq

                # first dump custom db data names
                db_data_names = list(self._env.custom_db_data.keys())
                for db_dname in db_data_names:
                    episodic_data_names = self._env.custom_db_data[db_dname].data_names()
                    var_name = db_dname
                    hf.attrs[var_name+"_data_names"] = episodic_data_names
                            
                for ep_idx in range(ep_vec_freq): # create separate datasets for each episode
                    ep_prefix=f'ep_{ep_idx}_'

                    # rewards
                    hf.create_dataset(ep_prefix+'sub_rew', 
                        data=sub_rew_full[ep_idx, :, :, :])
                    hf.create_dataset(ep_prefix+'tot_rew', 
                        data=tot_rew_full[ep_idx, :, :, :])
                    if self._n_expl_envs > 0:
                        hf.create_dataset(ep_prefix+'sub_rew_expl', 
                            data=sub_rew_full_expl[ep_idx, :, :, :])
                        hf.create_dataset(ep_prefix+'tot_rew_expl', 
                            data=tot_rew_full_expl[ep_idx, :, :, :])
                    if self._env.n_demo_envs() > 0:
                        hf.create_dataset(ep_prefix+'sub_rew_demo', 
                            data=sub_rew_full_demo)
                        hf.create_dataset(ep_prefix+'tot_rew_demo', 
                            data=tot_rew_full_demo[ep_idx, :, :, :])
                    
                    # dump all custom env data
                    db_data_names = list(self._env.custom_db_data.keys())
                    for db_dname in db_data_names:
                        episodic_data=self._env.custom_db_data[db_dname]
                        var_name = db_dname
                        hf.create_dataset(ep_prefix+var_name, 
                            data=episodic_data.get_full_episodic_data()[ep_idx, :, :, :])
                        if self._n_expl_envs > 0:
                            hf.create_dataset(ep_prefix+var_name+"_expl", 
                                data=episodic_data.get_full_episodic_data(env_selector=self._expl_env_selector)[ep_idx, :, :, :])
                        if self._env.n_demo_envs() > 0:
                            hf.create_dataset(ep_prefix+var_name+"_demo", 
                                data=episodic_data.get_full_episodic_data(env_selector=self._demo_env_selector)[ep_idx, :, :, :])
                
            Journal.log(self.__class__.__name__,
                "_dump_env_checkpoints",
                "done.",
                LogType.INFO,
                throw_when_excep = True)

    def done(self):
        
        if not self._is_done:

            if not self._eval:
                self._save_model()
            
            self._dump_dbinfo_to_file()
            
            if self._full_env_db:
                self._dump_env_checkpoints()

            if self._shared_algo_data is not None:
                self._shared_algo_data.write(dyn_info_name=["is_done"],
                    val=[1.0])
                self._shared_algo_data.close() # close shared memory

            self._env.close()

            self._is_done = True

    def _dump_dbinfo_to_file(self):

        import h5py

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
                dtype='S40') 
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

            # profiling data
            hf.create_dataset('env_step_fps', data=self._env_step_fps.numpy())
            hf.create_dataset('env_step_rt_factor', data=self._env_step_rt_factor.numpy())
            hf.create_dataset('rollout_dt', data=self._rollout_dt.numpy())
            hf.create_dataset('batch_norm_update_dt', data=self._batch_norm_update_dt.numpy())
            hf.create_dataset('gae_dt', data=self._gae_dt.numpy())
            hf.create_dataset('policy_update_dt', data=self._policy_update_dt.numpy())
            hf.create_dataset('policy_update_fps', data=self._policy_update_fps.numpy())
            
            hf.create_dataset('n_of_played_episodes', data=self._n_of_played_episodes.numpy())
            hf.create_dataset('n_timesteps_done', data=self._n_timesteps_done.numpy())
            hf.create_dataset('n_policy_updates', data=self._n_policy_updates.numpy())
            hf.create_dataset('n_vfun_updates', data=self._n_vfun_updates.numpy())

            hf.create_dataset('elapsed_min', data=self._elapsed_min.numpy())

            hf.create_dataset('learn_rates', data=self._learning_rates.numpy())

            # ppo iterations db data
            hf.create_dataset('tot_loss_mean', data=self._tot_loss_mean.numpy())
            hf.create_dataset('value_los_means', data=self._value_loss_mean.numpy())
            hf.create_dataset('policy_loss_mean', data=self._policy_loss_mean.numpy())
            hf.create_dataset('entropy_loss_mean', data=self._entropy_loss_mean.numpy())
            hf.create_dataset('tot_loss_grad_norm_mean', data=self._tot_loss_grad_norm_mean.numpy())
            hf.create_dataset('actor_loss_grad_norm_mean', data=self._actor_loss_grad_norm_mean.numpy())

            hf.create_dataset('tot_loss_std', data=self._tot_loss_std.numpy())
            hf.create_dataset('value_loss_std', data=self._value_loss_std.numpy())
            hf.create_dataset('policy_loss_std', data=self._policy_loss_std.numpy())
            hf.create_dataset('entropy_loss_std', data=self._entropy_loss_std.numpy())
            hf.create_dataset('tot_loss_grad_norm_std', data=self._tot_loss_grad_norm_std.numpy())
            hf.create_dataset('actor_loss_grad_norm_std', data=self._actor_loss_grad_norm_std.numpy())

            hf.create_dataset('old_approx_kl_mean', data=self._old_approx_kl_mean.numpy())
            hf.create_dataset('approx_kl_mean', data=self._approx_kl_mean.numpy())
            hf.create_dataset('old_approx_kl_std', data=self._old_approx_kl_std.numpy())
            hf.create_dataset('approx_kl_std', data=self._approx_kl_std.numpy())

            hf.create_dataset('clipfrac_mean', data=self._clipfrac_mean.numpy())
            hf.create_dataset('clipfrac_std', data=self._clipfrac_std.numpy())
            
            hf.create_dataset('explained_variance', data=self._explained_variance.numpy())

            hf.create_dataset('batch_returns_std', data=self._batch_returns_std.numpy())
            hf.create_dataset('batch_returns_mean', data=self._batch_returns_mean.numpy())
            hf.create_dataset('batch_adv_std', data=self._batch_adv_std.numpy())
            hf.create_dataset('batch_adv_mean', data=self._batch_adv_mean.numpy())
            hf.create_dataset('batch_val_std', data=self._batch_val_std.numpy())
            hf.create_dataset('batch_val_mean', data=self._batch_val_mean.numpy())

            # dump all custom env data  
            db_data_names = list(self._env.custom_db_data.keys())
            for db_dname in db_data_names:
                data=self._custom_env_data[db_dname]
                subnames = list(data.keys())
                for subname in subnames:
                    var_name = db_dname + "_" + subname
                    hf.create_dataset(var_name, data=data[subname])
            
            # other data
            if self._agent.running_norm is not None:
                if self._running_mean_obs is not None:
                    hf.create_dataset('running_mean_obs', data=self._running_mean_obs.numpy())
                if self._running_std_obs is not None:
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
                "/" + self._unique_id + "_env_db_checkpoint"
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

        self._rollout_dt[self._log_it_counter] += \
            self._rollout_t -self._start_time
        self._batch_norm_update_dt[self._log_it_counter] += \
            (self._bnorm_t-self._rollout_t)
        self._gae_dt[self._log_it_counter] += \
            self._gae_t - self._bnorm_t
        self._policy_update_dt[self._log_it_counter] += \
            self._policy_update_t - self._gae_t

        self._step_counter +=1 # counts algo steps

        if self._vec_transition_counter % self._db_vecstep_frequency== 0:
            
            self._env_step_fps[self._log_it_counter] = (self._db_vecstep_frequency*self._num_envs)/ self._rollout_dt[self._log_it_counter]
            if "substepping_dt" in self._hyperparameters:
                self._env_step_rt_factor[self._log_it_counter] = self._env_step_fps[self._log_it_counter]*self._env_n_action_reps*self._hyperparameters["substepping_dt"]
            
            self._n_timesteps_done[self._log_it_counter]=self._vec_transition_counter*self._num_envs
            
            self._update_epochs * self._num_minibatches

            self._n_policy_updates[self._log_it_counter]+=self._n_policy_updates[self._log_it_counter-1]
            self._n_vfun_updates[self._log_it_counter]+=self._n_vfun_updates[self._log_it_counter-1]
            self._policy_update_fps[self._log_it_counter] = (self._n_policy_updates[self._log_it_counter]-\
                self._n_policy_updates[self._log_it_counter-1])/self._policy_update_dt[self._log_it_counter]

            self._elapsed_min[self._log_it_counter] = (time.perf_counter() - self._start_time_tot) / 60
            
            self._learning_rates[self._log_it_counter, 0] = self._lr_now_actor
            self._learning_rates[self._log_it_counter, 0] = self._lr_now_critic

            self._n_of_played_episodes[self._log_it_counter] = self._episodic_reward_metrics.get_n_played_episodes()

            self._ep_tsteps_env_distribution[self._log_it_counter, :]=\
                self._episodic_reward_metrics.step_counters()*self._env_n_action_reps

            self._env_step_fps[self._log_it_counter] = self._db_vecstep_freq_it * self._batch_size / self._rollout_dt[self._log_it_counter]
            if "substepping_dt" in self._hyperparameters:
                self._env_step_rt_factor[self._log_it_counter] = self._env_step_fps[self._log_it_counter]*self._env_n_action_reps*self._hyperparameters["substepping_dt"] 
            self._policy_update_fps[self._log_it_counter] = self._db_vecstep_freq_it * self._update_epochs*self._num_minibatches/self._policy_update_dt[self._log_it_counter]

            # updating episodic reward metrics
            # debug environments
            self._tot_rew_max[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_tot_rew_max()
            self._tot_rew_avrg[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_tot_rew_avrg()
            self._tot_rew_min[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_tot_rew_min()
            self._tot_rew_max_over_envs[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_tot_rew_max_over_envs()
            self._tot_rew_avrg_over_envs[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_tot_rew_avrg_over_envs()
            self._tot_rew_min_over_envs[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_tot_rew_min_over_envs()

            self._sub_rew_max[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_sub_rew_max()
            self._sub_rew_avrg[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_sub_rew_avrg()
            self._sub_rew_min[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_sub_rew_min()
            self._sub_rew_max_over_envs[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_sub_rew_max_over_envs()
            self._sub_rew_avrg_over_envs[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_sub_rew_avrg_over_envs()
            self._sub_rew_min_over_envs[self._log_it_counter, :, :] = self._episodic_reward_metrics.get_sub_rew_min_over_envs()

            # fill env custom db metrics (only for debug environments)
            db_data_names = list(self._env.custom_db_data.keys())
            for dbdatan in db_data_names:
                self._custom_env_data[dbdatan]["max"][self._log_it_counter, :, :] = self._env.custom_db_data[dbdatan].get_max()
                self._custom_env_data[dbdatan]["avrg"][self._log_it_counter, :, :] = self._env.custom_db_data[dbdatan].get_avrg()
                self._custom_env_data[dbdatan]["min"][self._log_it_counter, :, :] = self._env.custom_db_data[dbdatan].get_min()
                self._custom_env_data[dbdatan]["max_over_envs"][self._log_it_counter, :, :] = self._env.custom_db_data[dbdatan].get_max_over_envs()
                self._custom_env_data[dbdatan]["avrg_over_envs"][self._log_it_counter, :, :] = self._env.custom_db_data[dbdatan].get_avrg_over_envs()
                self._custom_env_data[dbdatan]["min_over_envs"][self._log_it_counter, :, :] = self._env.custom_db_data[dbdatan].get_min_over_envs()

            # other data
            if self._agent.running_norm is not None:
                if self._running_mean_obs is not None:
                    self._running_mean_obs[self._log_it_counter, :] = self._agent.running_norm.get_current_mean()
                if self._running_std_obs is not None:
                    self._running_std_obs[self._log_it_counter, :] = self._agent.running_norm.get_current_std()

            # write some episodic db info on shared mem
            sub_returns=self._sub_returns.get_torch_mirror(gpu=False)
            sub_returns[:, :]=self._episodic_reward_metrics.get_sub_rew_avrg()
            tot_returns=self._tot_returns.get_torch_mirror(gpu=False)
            tot_returns[:, :]=self._episodic_reward_metrics.get_tot_rew_avrg()
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

        if self._vec_transition_counter==self._total_timesteps_vec:
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
            experience_to_vfun_grad_ratio=-1
            if not self._eval:
                actual_tsteps_with_updates=self._n_timesteps_done[self._log_it_counter].item()
                epsi=1e-6 # to avoid div by 0
                experience_to_policy_grad_ratio=actual_tsteps_with_updates/(self._n_policy_updates[self._log_it_counter].item()-epsi)
                experience_to_vfun_grad_ratio=actual_tsteps_with_updates/(self._n_vfun_updates[self._log_it_counter].item()-epsi)

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
                    self._rollout_dt[self._log_it_counter].item(),
                    self._policy_update_fps[self._log_it_counter].item(),
                    self._policy_update_dt[self._log_it_counter].item(),
                    is_done,
                    self._n_of_played_episodes[self._log_it_counter].item(),
                    self._batch_norm_update_dt[self._log_it_counter].item(),
                    ]
                self._shared_algo_data.write(dyn_info_name=info_names,
                                        val=info_data)
                
                ## write debug info to remote wandb server
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
                        
                # algo info
                self._policy_update_db_data_dict.update({
                        "ppo_info_losses/tot_loss_mean": self._tot_loss_mean[self._log_it_counter, 0],
                        "ppo_info_losses/tot_loss_std": self._tot_loss_std[self._log_it_counter, 0],
                        "ppo_info_losses/value_loss_mean": self._value_loss_mean[self._log_it_counter, 0],
                        "ppo_info_losses/value_loss_std": self._value_loss_std[self._log_it_counter, 0],
                        "ppo_info_losses/policy_loss_mean": self._policy_loss_mean[self._log_it_counter, 0],
                        "ppo_info_losses/policy_loss_std": self._policy_loss_std[self._log_it_counter, 0],
                        "ppo_info_losses/entropy_loss_mean": self._entropy_loss_mean[self._log_it_counter, 0],
                        "ppo_info_losses/entropy_loss_std": self._entropy_loss_std[self._log_it_counter, 0],
                        "ppo_info_other/old_approx_kl_mean": self._old_approx_kl_mean[self._log_it_counter, 0],
                        "ppo_info_other/old_approx_kl_std": self._old_approx_kl_std[self._log_it_counter, 0],
                        "ppo_info_other/approx_kl_mean": self._approx_kl_mean[self._log_it_counter, 0],
                        "ppo_info_other/approx_kl_std": self._approx_kl_std[self._log_it_counter, 0],
                        "ppo_info_other/clipfrac_mean": self._clipfrac_mean[self._log_it_counter, 0],
                        "ppo_info_other/clipfrac_std": self._clipfrac_std[self._log_it_counter, 0],
                        "ppo_info_other/explained_variance": self._explained_variance[self._log_it_counter, 0],
                        "ppo_info_val/batch_returns_mean": self._batch_returns_mean[self._log_it_counter, 0],
                        "ppo_info_val/batch_returns_std": self._batch_returns_std[self._log_it_counter, 0],
                        "ppo_info_val/batch_adv_mean": self._batch_adv_mean[self._log_it_counter, 0],
                        "ppo_info_val/batch_adv_std": self._batch_adv_std[self._log_it_counter, 0],
                        "ppo_info_val/batch_val_mean": self._batch_val_mean[self._log_it_counter, 0],
                        "ppo_info_val/batch_val_std": self._batch_val_std[self._log_it_counter, 0]

                    })

                self._wandb_d.update(self._policy_update_db_data_dict)
                
                if self._agent.running_norm is not None:
                    # adding info on running normalizer if used
                    if self._running_mean_obs is not None:
                        self._wandb_d.update({f"running_norm/mean": self._running_mean_obs[self._log_it_counter, :]})
                    if self._running_std_obs is not None:
                        self._wandb_d.update({f"running_norm/std": self._running_std_obs[self._log_it_counter, :]})
                
                self._wandb_d.update(self._custom_env_data_db_dict) 

                wandb.log(self._wandb_d)

        if self._verbose:
            info = f"\nTotal n. timesteps simulated: {self._n_timesteps_done[self._log_it_counter].item()}/{self._total_timesteps}\n" + \
                f"N. policy updates performed: {self._n_policy_updates[self._log_it_counter].item()}/{self._n_policy_updates_to_be_done}\n" + \
                f"N. v fun updates performed: {self._n_vfun_updates[self._log_it_counter].item()}/{self._n_vfun_updates_to_be_done}\n" + \
                f"N. iterations performed: {self._step_counter}/{self._iterations_n}\n" + \
                f"experience to policy grad ratio: {experience_to_policy_grad_ratio}\n" + \
                f"experience to v fun grad ratio: {experience_to_vfun_grad_ratio}\n" + \
                f"Elapsed time: {self._elapsed_min[self._log_it_counter].item()/60.0} h\n" + \
                f"Estimated remaining training time: " + \
                f"{est_remaining_time_h} h\n" + \
                "Total reward episodic data --> \n" + \
                f"max: {self._tot_rew_max_over_envs[self._log_it_counter, :, :].item()}\n" + \
                f"avg: {self._tot_rew_avrg_over_envs[self._log_it_counter, :, :].item()}\n" + \
                f"min: {self._tot_rew_min_over_envs[self._log_it_counter, :, :].item()}\n" + \
                f"Episodic sub-rewards episodic data --> \nsub rewards names: {self._reward_names_str}\n" + \
                f"max: {self._sub_rew_max_over_envs[self._log_it_counter, :]}\n" + \
                f"avg: {self._sub_rew_avrg_over_envs[self._log_it_counter, :]}\n" + \
                f"min: {self._sub_rew_min_over_envs[self._log_it_counter, :]}\n" + \
                f"N. of episodes on which episodic rew stats are computed: {self._n_of_played_episodes[self._log_it_counter].item()}\n" + \
                f"Current env. step sps: {self._env_step_fps[self._log_it_counter].item()}, time for experience collection {self._rollout_dt[self._log_it_counter].item()} s\n" + \
                f"Current env (sub-stepping) rt factor: {self._env_step_rt_factor[self._log_it_counter].item()}\n" + \
                f"Current policy update fps: {self._policy_update_fps[self._log_it_counter].item()}, time for policy updates {self._policy_update_dt[self._log_it_counter].item()} s\n" + \
                f"Time to compute GAE {self._gae_dt[self._log_it_counter].item()} s\n" + \
                f"Time spent updating batch normalizations {self._batch_norm_update_dt[self._log_it_counter].item()} s\n"
            
            Journal.log(self.__class__.__name__,
                "_post_step",
                info,
                LogType.INFO,
                throw_when_excep = True)

    def _add_experience(self, 
            pos: int,
            obs: torch.Tensor, 
            actions: torch.Tensor, 
            logprob: torch.Tensor, 
            rewards: torch.Tensor, 
            next_obs: torch.Tensor, 
            next_terminal: torch.Tensor,
            next_done: torch.Tensor) -> None:

        self._obs[pos] = obs
        self._values[pos] = self._agent.get_value(obs).view(-1, 1).detach()
        
        self._actions[pos] = actions
        self._logprobs[pos] = logprob.view(-1, 1)
        
        self._next_obs[pos] = next_obs
        self._next_values[pos] = self._agent.get_value(next_obs).view(-1, 1).detach()
        
        self._rewards[pos] = rewards

        self._next_terminal[pos] = next_terminal
        self._next_done[pos] = next_done

    def _sample(self, size: int = None):
        
        if size is None or (size >= self._batch_size):
            # get all rollout

            sampled_obs = self._obs.view((-1, self._env.obs_dim()))
            sampled_rewards = self._rewards.view(-1)
            
        else:

            batched_obs = self._obs.view((-1, self._env.obs_dim()))
            batched_rewards = self._rewards.view(-1)

            # sampling from the batched buffer
            shuffled_buffer_idxs = torch.randint(0, self._batch_size,
                                            (size,)) 
        
            sampled_obs = batched_obs[shuffled_buffer_idxs]
            sampled_rewards = batched_rewards[shuffled_buffer_idxs]

        return sampled_obs, sampled_rewards

    def _sample_random_actions(self):
        
        self._random_uniform.uniform_(-1,1)
        random_actions = self._random_uniform*self._action_scale+self._action_offset

        return random_actions
    
    def _update_batch_norm(self, bsize: int = None):

        if bsize is None:
            bsize=self._batch_size # same used for rollout
        
        # update obs normalization        
        # (we should sample also next obs, but if most of the transitions are not terminal, 
        # this is not an issue and is more efficient)
        if (self._agent.running_norm is not None) and \
            (not self._eval):

            sampled_obs, _ = self._sample(size=bsize)
            
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
