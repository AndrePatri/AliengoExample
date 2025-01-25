from lrhc_control.agents.actor_critic.ppo import ACAgent

from lrhc_control.utils.shared_data.algo_infos import SharedRLAlgorithmInfo
import torch 
import torch.optim as optim
import torch.nn as nn

import random

from typing import Dict

import os
import shutil

import time

import wandb

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
            anomaly_detect = False,
            seed: int = 1):

        self._env = env 
        self._seed = seed

        self._eval = False
        self._agent = None 
        
        self._debug = debug
        self._remote_db = remote_db

        self._anomaly_detect = anomaly_detect

        self._optimizer = None

        self._writer = None
        
        self._run_name = None
        self._drop_dir = None
        self._dbinfo_drop_fname = None
        self._model_path = None
        
        self._policy_update_db_data_dict =  {}
        self._custom_env_data_db_dict = {}
        self._hyperparameters = {}
        
        self._episodic_reward_metrics = self._env.ep_rewards_metrics()
        
        tot_tsteps = 50e6
        rollout_tsteps = 256
        self._init_params(tot_tsteps=tot_tsteps,
            rollout_tsteps=rollout_tsteps)
        
        self._init_dbdata()

        self._setup_done = False

        self._verbose = False

        self._is_done = False
        
        self._shared_algo_data = None

        self._this_child_path = None
        self._this_basepath = os.path.abspath(__file__)
    
    def __del__(self):

        self.done()

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
            norm_obs: bool = True):

        self._verbose = verbose

        self._ns=ns # only used for shared mem stuff

        self._dump_checkpoints = dump_checkpoints
        
        self._eval = eval
        try:
            self._det_eval=custom_args["det_eval"]
        except:
            pass

        self._run_name = run_name
        from datetime import datetime
        self._time_id = datetime.now().strftime('d%Y_%m_%d_h%H_m%M_s%S')
        self._unique_id = self._time_id + "-" + self._run_name
        
        self._init_algo_shared_data(static_params=self._hyperparameters) # can only handle dicts with
        # numeric values
        
        data_names={}
        data_names["obs_names"]=self._env.obs_names()
        data_names["action_names"]=self._env.action_names()
        data_names["sub_reward_names"]=self._env.sub_rew_names()

        self._hyperparameters["unique_run_id"]=self._unique_id
        self._hyperparameters.update(custom_args)
        self._hyperparameters.update(data_names)
        
        self._torch_device = torch.device("cuda" if torch.cuda.is_available() and self._use_gpu else "cpu")

        try:
            layer_size_actor=self._hyperparameters["layer_size_actor"]
            layer_size_critic=self._hyperparameters["layer_size_critic"]
        except:
            layer_size_actor=256
            layer_size_critic=256
            pass

        self._agent = ACAgent(obs_dim=self._env.obs_dim(),
                        actions_dim=self._env.actions_dim(),
                        actor_std=0.01,
                        critic_std=1.0,
                        norm_obs=norm_obs,
                        device=self._torch_device,
                        dtype=self._dtype,
                        is_eval=self._eval,
                        debug=self._debug,
                        layer_size_actor=layer_size_actor,
                        layer_size_critic=layer_size_actor)
        # self._agent.to(self._torch_device) # move agent to target device

        # load model if necessary 
        if self._eval: # load pretrained model
            if model_path is None:
                Journal.log(self.__class__.__name__,
                    "setup",
                    f"When eval is True, a model_path should be provided!!",
                    LogType.EXCEP,
                    throw_when_excep = True)
            elif n_eval_timesteps is None:
                Journal.log(self.__class__.__name__,
                    "setup",
                    f"When eval is True, n_eval_timesteps should be provided!!",
                    LogType.EXCEP,
                    throw_when_excep = True)
            else: # everything ok
                self._model_path = model_path
                # overwrite init params (recomputes n_iterations, etc...)
                self._init_params(tot_tsteps=n_eval_timesteps,
                    rollout_tsteps=self._db_vecstep_frequency_nom,
                    run_name=self._run_name)
                
            self._load_model(self._model_path)
        
        # create dump directory + copy important files for debug
        self._init_drop_dir(drop_dir_name)
        self._hyperparameters["drop_dir"]=self._drop_dir

        # seeding + deterministic behavior for reproducibility
        self._set_all_deterministic()
        torch.autograd.set_detect_anomaly(self._anomaly_detect)

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
            self._init_buffers() # only needed if training
        
        # self._env.reset()

        if (self._debug):
            if self._remote_db:
                job_type = "evaluation" if self._eval else "training"
                full_run_config={**self._hyperparameters,**self._env.env_opts()}
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
                wandb.watch(self._agent, log="all")
                # wandb.watch(self._agent.actor_mean, log="all")
                # wandb.watch(self.actor_logstd, log="all")
                
        self._setup_done = True

        self._is_done = False

        self._start_time_tot = time.perf_counter()

        self._start_time = time.perf_counter()
    
    def is_done(self):

        return self._is_done 
    
    def model_path(self):

        return self._model_path
    
    def learn(self):
        
        if not self._setup_done:
            self._should_have_called_setup()

        # annealing the learning rate if enabled (may improve convergence)
        if self._anneal_lr:
            frac = 1.0 - (self._it_counter) / self._iterations_n
            self._lr_now_actor = frac * self._base_lr_actor
            self._lr_now_critic = frac * self._base_lr_critic
            self._optimizer.param_groups[0]["lr"] = self._lr_now_actor
            # self._optimizer.param_groups[1]["lr"] = self._lr_now_critic

        self._start_time = time.perf_counter()

        with torch.no_grad(): # don't want grad computation here
            rollout_ok = self._play()
            if not rollout_ok:
                return False
        
            self._rollout_t = time.perf_counter()

            self._compute_returns()
            self._gae_t = time.perf_counter()

        self._improve_policy()
        self._policy_update_t = time.perf_counter()

        with torch.no_grad():
            self._post_step()

        return True

    def eval(self):

        if not self._setup_done:
            self._should_have_called_setup()

        self._episodic_reward_metrics.reset()

        self._start_time = time.perf_counter()

        rollout_ok = self._play()
        if not rollout_ok:
            return False

        self._rollout_t = time.perf_counter()

        self._post_step()

        return True

    @abstractmethod
    def _play(self):
        pass
    
    @abstractmethod
    def _compute_returns(self):
       pass
    
    @abstractmethod
    def _improve_policy(self):
        pass

    def _save_model(self,
            is_checkpoint: bool = False):

        path = self._model_path
        if is_checkpoint: # use iteration as id
            path = path + "_checkpoint" + str(self._it_counter)
        info = f"Saving model to {path}\n"
        Journal.log(self.__class__.__name__,
            "_save_model",
            info,
            LogType.INFO,
            throw_when_excep = True)
        torch.save(self._agent.state_dict(), path) # saves whole agent state
        # torch.save(self._agent.parameters(), path) # only save agent parameters
        info = f"Done."
        Journal.log(self.__class__.__name__,
            "done",
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
                dtype='S20') 
            hf.create_dataset('episodic_rewards', data=self._episodic_rewards.numpy())
            hf.create_dataset('episodic_sub_rewards', data=self._episodic_sub_rewards.numpy())
            hf.create_dataset('episodic_sub_rewards_env_avrg', data=self._episodic_sub_rewards_env_avrg.numpy())
            hf.create_dataset('episodic_rewards_env_avrg', data=self._episodic_rewards_env_avrg.numpy())

            # profiling data
            hf.create_dataset('rollout_dt', data=self._rollout_dt.numpy())
            hf.create_dataset('env_step_fps', data=self._env_step_fps.numpy())
            hf.create_dataset('env_step_rt_factor', data=self._env_step_rt_factor.numpy())
            hf.create_dataset('gae_dt', data=self._gae_dt.numpy())
            hf.create_dataset('policy_update_dt', data=self._policy_update_dt.numpy())
            hf.create_dataset('policy_update_fps', data=self._policy_update_fps.numpy())
            hf.create_dataset('n_of_played_episodes', data=self._n_of_played_episodes.numpy())
            hf.create_dataset('n_timesteps_done', data=self._n_timesteps_done.numpy())
            hf.create_dataset('n_policy_updates', data=self._n_policy_updates.numpy())
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
        
        self._agent.load_state_dict(torch.load(model_path, 
                            map_location=self._torch_device))
        self._switch_training_mode(False)

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

        # model
        if not self._eval:
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

    def _post_step(self):

        self._it_counter +=1 
        self._vec_transition_counter+=self._rollout_timesteps

        self._rollout_dt[self._log_it_counter] += \
            self._rollout_t -self._start_time
        self._gae_dt[self._log_it_counter] += \
            self._gae_t - self._rollout_t
        self._policy_update_dt[self._log_it_counter] += \
            self._policy_update_t - self._gae_t

        if self._vec_transition_counter % self._db_vecstep_frequency== 0:
           
            self._n_of_played_episodes[self._log_it_counter] = self._episodic_reward_metrics.get_n_played_episodes()
            self._n_timesteps_done[self._log_it_counter] = self._it_counter * self._batch_size
            self._n_policy_updates[self._log_it_counter] = self._it_counter * self._update_epochs * self._num_minibatches
            
            self._elapsed_min[self._log_it_counter] = (time.perf_counter() - self._start_time_tot) / 60
            
            self._learning_rates[self._log_it_counter, 0] = self._lr_now_actor
            self._learning_rates[self._log_it_counter, 0] = self._lr_now_critic

            self._env_step_fps[self._log_it_counter] = self._db_vecstep_freq_it * self._batch_size / self._rollout_dt[self._log_it_counter]
            if "substepping_dt" in self._hyperparameters:
                self._env_step_rt_factor[self._log_it_counter] = self._env_step_fps[self._log_it_counter]*self._env_n_action_reps*self._hyperparameters["substepping_dt"] 
            self._policy_update_fps[self._log_it_counter] = self._db_vecstep_freq_it * self._update_epochs*self._num_minibatches/self._policy_update_dt[self._log_it_counter]

            # updating episodic reward metrics
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

            # fill env custom db metrics
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
                self._running_mean_obs[self._log_it_counter, :] = self._agent.running_norm.get_current_mean()
                self._running_std_obs[self._log_it_counter, :] = self._agent.running_norm.get_current_std()

            self._log_info()
            
            self._log_it_counter+=1

        if self._dump_checkpoints and \
            (self._vec_transition_counter % self._m_checkpoint_freq == 0):
            self._save_model(is_checkpoint=True)

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
        
        if self._verbose or self._debug:
            exp_to_pol_grad_ratio=self._n_timesteps_done[self._log_it_counter].item()/self._n_policy_updates[self._log_it_counter].item()
            est_remaining_time=self._elapsed_min[self._log_it_counter].item()/60*1/self._it_counter*(self._iterations_n-self._it_counter)
            elapsed_h=self._elapsed_min[self._log_it_counter].item()/60.0
            is_done=self._vec_transition_counter==self._total_timesteps_vec

        if self._debug:
            if self._remote_db: 
                # write general algo debug info to shared memory    
                info_names=self._shared_algo_data.dynamic_info.get()
                info_data = [
                    self._n_timesteps_done[self._log_it_counter].item(),
                    self._n_policy_updates[self._log_it_counter].item(),
                    exp_to_pol_grad_ratio,
                    elapsed_h,
                    est_remaining_time,
                    self._env_step_fps[self._log_it_counter].item(),
                    self._env_step_rt_factor[self._log_it_counter].item(),
                    self._rollout_dt[self._log_it_counter].item(),
                    self._policy_update_fps[self._log_it_counter].item(),
                    self._policy_update_dt[self._log_it_counter].item(),
                    is_done,
                    self._n_of_played_episodes[self._log_it_counter].item()
                    ]
                self._shared_algo_data.write(dyn_info_name=info_names,
                                        val=info_data)
                
                # write debug info to remote wandb server

                # custom env db info
                db_data_names = list(self._env.custom_db_data.keys())
                for dbdatan in db_data_names: 
                    data = self._custom_env_data[dbdatan]
                    data_names = self._env.custom_db_data[dbdatan].data_names()

                    self._custom_env_data_db_dict.update({f"env_dbdata/{dbdatan}" + "_max": 
                            wandb.Histogram(data["max"][self._log_it_counter-1, :, :].numpy())})
                    self._custom_env_data_db_dict.update({f"env_dbdata/{dbdatan}" + "_avrg": 
                            wandb.Histogram(data["avrg"][self._log_it_counter-1, :, :].numpy())})
                    self._custom_env_data_db_dict.update({f"env_dbdata/{dbdatan}" + "_min": 
                            wandb.Histogram(data["min"][self._log_it_counter-1, :, :].numpy())})
                    
                    self._custom_env_data_db_dict.update({f"env_dbdata/{dbdatan}-{data_names[i]}" + "_max_over_envs": 
                        data["max_over_envs"][self._log_it_counter-1, :, i:i+1] for i in range(len(data_names))})
                    self._custom_env_data_db_dict.update({f"env_dbdata/{dbdatan}-{data_names[i]}" + "_avrg_over_envs": 
                        data["avrg_over_envs"][self._log_it_counter-1, :, i:i+1] for i in range(len(data_names))})
                    self._custom_env_data_db_dict.update({f"env_dbdata/{dbdatan}-{data_names[i]}" + "_min_over_envs": 
                        data["min_over_envs"][self._log_it_counter-1, :, i:i+1] for i in range(len(data_names))})
                
                wandb_d={'log_iteration' : self._log_it_counter}
                wandb_d.update(dict(zip(info_names, info_data)))
                # tot reward
                wandb_d.update({'tot_reward/tot_rew_max': wandb.Histogram(self._tot_rew_max[self._log_it_counter-1, :, :].numpy()),
                    'tot_reward/tot_rew_avrg': wandb.Histogram(self._tot_rew_avrg[self._log_it_counter-1, :, :].numpy()),
                    'tot_reward/tot_rew_min': wandb.Histogram(self._tot_rew_min[self._log_it_counter-1, :, :].numpy()),
                    'tot_reward/tot_rew_max_over_envs': self._tot_rew_max_over_envs[self._log_it_counter-1, :, :].item(),
                    'tot_reward/tot_rew_avrg_over_envs': self._tot_rew_avrg_over_envs[self._log_it_counter-1, :, :].item(),
                    'tot_reward/tot_rew_min_over_envs': self._tot_rew_min_over_envs[self._log_it_counter-1, :, :].item()})
                # sub rewards
                wandb_d.update({f"sub_reward/{self._reward_names[i]}_sub_rew_max":
                        wandb.Histogram(self._sub_rew_max.numpy()[self._log_it_counter-1, :, i:i+1]) for i in range(len(self._reward_names))})
                wandb_d.update({f"sub_reward/{self._reward_names[i]}_sub_rew_avrg":
                        wandb.Histogram(self._sub_rew_avrg.numpy()[self._log_it_counter-1, :, i:i+1]) for i in range(len(self._reward_names))})
                wandb_d.update({f"sub_reward/{self._reward_names[i]}_sub_rew_min":
                        wandb.Histogram(self._sub_rew_min.numpy()[self._log_it_counter-1, :, i:i+1]) for i in range(len(self._reward_names))})
            
                wandb_d.update({f"sub_reward/{self._reward_names[i]}_sub_rew_max_over_envs":
                        self._sub_rew_max_over_envs[self._log_it_counter-1, :, i:i+1] for i in range(len(self._reward_names))})
                wandb_d.update({f"sub_reward/{self._reward_names[i]}_sub_rew_avrg_over_envs":
                        self._sub_rew_avrg_over_envs[self._log_it_counter-1, :, i:i+1] for i in range(len(self._reward_names))})
                wandb_d.update({f"sub_reward/{self._reward_names[i]}_sub_rew_min_over_envs":
                        self._sub_rew_min_over_envs[self._log_it_counter-1, :, i:i+1] for i in range(len(self._reward_names))})
                
                wandb_d.update(self._policy_update_db_data_dict)
                wandb_d.update(self._custom_env_data_db_dict)

                wandb.log(wandb_d)

        if self._verbose:
            info = f"\nN. PPO iterations performed: {self._it_counter}/{self._iterations_n}\n" + \
                f"N. policy updates performed: {self._n_policy_updates[self._log_it_counter].item()}/" + \
                f"{self._n_policy_updates_to_be_done}\n" + \
                f"Total n. timesteps simulated: {self._n_timesteps_done[self._log_it_counter].item()}/{self._total_timesteps}\n" + \
                f"Elapsed time: {elapsed_h} h\n" + \
                f"Estimated remaining training time: " + \
                f"{est_remaining_time} h\n" + \
                f"N. of episodes on which episodic rew stats are computed: {self._n_of_played_episodes[self._log_it_counter].item()}\n" + \
                f"Total reward episodic data --> \n" + \
                f"max: {self._tot_rew_max_over_envs[self._log_it_counter, :, :].item()}\n" + \
                f"avg: {self._tot_rew_avrg_over_envs[self._log_it_counter, :, :].item()}\n" + \
                f"min: {self._tot_rew_min_over_envs[self._log_it_counter, :, :].item()}\n" + \
                f"Episodic sub-rewards episodic data --> \nsub rewards names: {self._reward_names_str}\n" + \
                f"max: {self._sub_rew_max_over_envs[self._log_it_counter, :]}\n" + \
                f"avg: {self._sub_rew_avrg_over_envs[self._log_it_counter, :]}\n" + \
                f"min: {self._sub_rew_min_over_envs[self._log_it_counter, :]}\n" + \
                f"Current rollout sps: {self._env_step_fps[self._log_it_counter].item()}, time for rollout {self._rollout_dt[self._log_it_counter].item()} s\n" + \
                f"Current rollout (sub-stepping) rt factor: {self._env_step_rt_factor[self._log_it_counter].item()}\n" + \
                f"Time to compute bootstrap {self._gae_dt[self._log_it_counter].item()} s\n" + \
                f"Current policy update fps: {self._policy_update_fps[self._log_it_counter].item()}, time for policy updates {self._policy_update_dt[self._log_it_counter].item()} s\n" + \
                f"Experience-to-policy grad ratio: {exp_to_pol_grad_ratio}\n"
            Journal.log(self.__class__.__name__,
                "_post_step",
                info,
                LogType.INFO,
                throw_when_excep = True)

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
        
        # gae computation
        self._gae_t = -1.0
        self._gae_dt = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")

        # ppo iteration
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
        self._elapsed_min = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0, device="cpu")
        self._learning_rates = torch.full((self._db_data_size, 2), 
                    dtype=torch.float32, fill_value=0, device="cpu")

        # ppo iteration db data
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
        
        # reward db data
        self._reward_names = self._episodic_reward_metrics.reward_names()
        self._reward_names_str = "[" + ', '.join(self._reward_names) + "]"
        self._n_rewards = self._episodic_reward_metrics.n_rewards()

        self._sub_rew_max = torch.full((self._db_data_size, self._num_envs, self._n_rewards), 
            dtype=torch.float32, fill_value=0.0, device="cpu")
        self._sub_rew_avrg = torch.full((self._db_data_size, self._num_envs, self._n_rewards), 
            dtype=torch.float32, fill_value=0.0, device="cpu")
        self._sub_rew_min = torch.full((self._db_data_size, self._num_envs, self._n_rewards), 
            dtype=torch.float32, fill_value=0.0, device="cpu")
        self._sub_rew_max_over_envs = torch.full((self._db_data_size, 1, self._n_rewards), 
            dtype=torch.float32, fill_value=0.0, device="cpu")
        self._sub_rew_avrg_over_envs = torch.full((self._db_data_size, 1, self._n_rewards), 
            dtype=torch.float32, fill_value=0.0, device="cpu")
        self._sub_rew_min_over_envs = torch.full((self._db_data_size, 1, self._n_rewards), 
            dtype=torch.float32, fill_value=0.0, device="cpu")

        self._tot_rew_max = torch.full((self._db_data_size, self._num_envs, 1), 
            dtype=torch.float32, fill_value=0.0, device="cpu")
        self._tot_rew_avrg = torch.full((self._db_data_size, self._num_envs, 1), 
            dtype=torch.float32, fill_value=0.0, device="cpu")
        self._tot_rew_min = torch.full((self._db_data_size, self._num_envs, 1), 
            dtype=torch.float32, fill_value=0.0, device="cpu")
        self._tot_rew_max_over_envs = torch.full((self._db_data_size, 1, 1), 
            dtype=torch.float32, fill_value=0.0, device="cpu")
        self._tot_rew_avrg_over_envs = torch.full((self._db_data_size, 1, 1), 
            dtype=torch.float32, fill_value=0.0, device="cpu")
        self._tot_rew_min_over_envs = torch.full((self._db_data_size, 1, 1), 
            dtype=torch.float32, fill_value=0.0, device="cpu")
        
        # custom data from env
        self._custom_env_data = {}
        db_data_names = list(self._env.custom_db_data.keys())
        for dbdatan in db_data_names: # loop thorugh custom data
            self._custom_env_data[dbdatan] = {}

            max = self._env.custom_db_data[dbdatan].get_max()
            avrg = self._env.custom_db_data[dbdatan].get_avrg()
            min = self._env.custom_db_data[dbdatan].get_min()
            max_over_envs = self._env.custom_db_data[dbdatan].get_max_over_envs()
            avrg_over_envs = self._env.custom_db_data[dbdatan].get_avrg_over_envs()
            min_over_envs = self._env.custom_db_data[dbdatan].get_min_over_envs()

            self._custom_env_data[dbdatan]["max"] =torch.full((self._db_data_size, 
                max.shape[0], 
                max.shape[1]), 
                dtype=torch.float32, fill_value=0.0, device="cpu")
            self._custom_env_data[dbdatan]["avrg"] =torch.full((self._db_data_size, 
                avrg.shape[0], 
                avrg.shape[1]), 
                dtype=torch.float32, fill_value=0.0, device="cpu")
            self._custom_env_data[dbdatan]["min"] =torch.full((self._db_data_size, 
                min.shape[0], 
                min.shape[1]), 
                dtype=torch.float32, fill_value=0.0, device="cpu")
            self._custom_env_data[dbdatan]["max_over_envs"] =torch.full((self._db_data_size, 
                max_over_envs.shape[0], 
                max_over_envs.shape[1]), 
                dtype=torch.float32, fill_value=0.0, device="cpu")
            self._custom_env_data[dbdatan]["avrg_over_envs"] =torch.full((self._db_data_size, 
                avrg_over_envs.shape[0], 
                avrg_over_envs.shape[1]), 
                dtype=torch.float32, fill_value=0.0, device="cpu")
            self._custom_env_data[dbdatan]["min_over_envs"] =torch.full((self._db_data_size, 
                min_over_envs.shape[0], 
                min_over_envs.shape[1]), 
                dtype=torch.float32, fill_value=0.0, device="cpu")

        # other data
        self._running_mean_obs = torch.full((self._db_data_size, self._env.obs_dim()), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._running_std_obs = torch.full((self._db_data_size, self._env.obs_dim()), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")

    def _init_params(self, 
            tot_tsteps: int,
            rollout_tsteps: int,
            run_name: str = "PPODefaultRunName"):

        self._dtype = self._env.dtype()

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

        # policy rollout and return comp./adv estimation
        self._total_timesteps = int(tot_tsteps) # total timesteps to be collected (including sub envs)
        self._total_timesteps = self._total_timesteps//self._env_n_action_reps # correct with n of action reps
        
        self._rollout_timesteps = int(rollout_tsteps) # numer of vectorized steps (rescaled depending on env substepping) 
        # to be done per policy rollout (influences adv estimation!!!)
        self._batch_size = self._rollout_timesteps * self._num_envs

        self._iterations_n = self._total_timesteps//self._batch_size # number of ppo iterations
        self._total_timesteps = self._iterations_n*self._batch_size # actual number of total tsteps to be simulated
        self._total_timesteps_vec = self._iterations_n*self._rollout_timesteps

        # policy update
        self._num_minibatches = 8
        self._minibatch_size = self._batch_size // self._num_minibatches
        
        self._base_lr_actor = 5e-4
        self._base_lr_critic = 1e-3
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

        self._n_policy_updates_to_be_done = self._update_epochs * self._num_minibatches * self._iterations_n

        self._exp_to_policy_grad_ratio=float(self._total_timesteps)/float(self._n_policy_updates_to_be_done)
        #debug
        self._m_checkpoint_freq_nom = 1e6 # n totoal timesteps after which a checkpoint model is dumped
        self._m_checkpoint_freq= self._m_checkpoint_freq_nom//self._num_envs
        
        self._db_vecstep_frequency_nom = 128 # log db data every n (vectorized) timesteps
        self._db_vecstep_frequency = 128 
        if self._db_vecstep_frequency_nom<self._rollout_timesteps:
            self._db_vecstep_frequency=self._rollout_timesteps
        else:
            self._db_vecstep_frequency=self._db_vecstep_frequency_nom
        self._checkpoint_nit = round(self._m_checkpoint_freq/self._rollout_timesteps)
        self._m_checkpoint_freq = self._rollout_timesteps*self._checkpoint_nit # ensuring _m_checkpoint_freq
        # is a multiple of self._rollout_timesteps

        self._db_vecstep_freq_it = round(self._db_vecstep_frequency/self._rollout_timesteps)
        self._db_vecstep_frequency = self._rollout_timesteps*self._db_vecstep_freq_it # ensuring _db_vecstep_frequency
        # is a multiple of self._rollout_timesteps

        self._db_data_size = round(self._total_timesteps_vec/self._db_vecstep_frequency)+self._db_vecstep_frequency
        # ensuring db_data fits 

        # write them to hyperparam dictionary for debugging
        self._hyperparameters["n_envs"] = self._num_envs
        self._hyperparameters["obs_dim"] = self._obs_dim
        self._hyperparameters["actions_dim"] = self._actions_dim
        # self._hyperparameters["critic_size"] = self._critic_size
        # self._hyperparameters["actor_size"] = self._actor_size
        self._hyperparameters["seed"] = self._seed
        self._hyperparameters["using_gpu"] = self._use_gpu
        self._hyperparameters["n_iterations"] = self._iterations_n
        self._hyperparameters["n_policy_updates_per_batch"] = self._update_epochs * self._num_minibatches
        self._hyperparameters["n_policy_updates_when_done"] = self._n_policy_updates_to_be_done
        self._hyperparameters["experience_to_policy_grad_steps_ratio"] = self._exp_to_policy_grad_ratio
        self._hyperparameters["episodes timeout lb"] = self._episode_timeout_lb
        self._hyperparameters["episodes timeout ub"] = self._episode_timeout_ub
        self._hyperparameters["task rand timeout lb"] = self._task_rand_timeout_lb
        self._hyperparameters["task rand timeout ub"] = self._task_rand_timeout_ub
        self._hyperparameters["env_n_action_reps"] = self._env_n_action_reps
        self._hyperparameters["n steps per env. rollout"] = self._rollout_timesteps
        self._hyperparameters["per-batch update_epochs"] = self._update_epochs
        self._hyperparameters["per-epoch policy updates"] = self._num_minibatches
        self._hyperparameters["total policy updates to be performed"] = self._update_epochs * self._num_minibatches * self._iterations_n
        self._hyperparameters["total_timesteps"] = self._total_timesteps
        self._hyperparameters["total_timesteps_vec"] = self._total_timesteps_vec
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
            f"n vec. steps per policy rollout: {self._rollout_timesteps}\n" + \
            f"batch_size: {self._batch_size}\n" + \
            f"num_minibatches for policy update: {self._num_minibatches}\n" + \
            f"minibatch_size: {self._minibatch_size}\n" + \
            f"per-batch update_epochs: {self._update_epochs}\n" + \
            f"iterations_n: {self._iterations_n}\n" + \
            f"episode timeout max steps: {self._episode_timeout_ub}\n" + \
            f"episode timeout min steps: {self._episode_timeout_lb}\n" + \
            f"task rand. max n steps: {self._task_rand_timeout_ub}\n" + \
            f"task rand. min n steps: {self._task_rand_timeout_lb}\n" + \
            f"number of action reps: {self._env_n_action_reps}\n" + \
            f"total policy updates to be performed: {self._n_policy_updates_to_be_done}\n" + \
            f"total_timesteps to be simulated: {self._total_timesteps}\n" + \
            f"experience to policy grad ratio: {self._exp_to_policy_grad_ratio}\n"
        Journal.log(self.__class__.__name__,
            "_init_params",
            info,
            LogType.INFO,
            throw_when_excep = True)
        
        self._it_counter = 0
        self._vec_transition_counter = 0
        self._log_it_counter = 0

    def _init_buffers(self):

        self._obs = torch.full(size=(self._rollout_timesteps, self._num_envs, self._obs_dim),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device) 
        self._values = torch.full(size=(self._rollout_timesteps, self._num_envs, 1),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
            
        self._actions = torch.full(size=(self._rollout_timesteps, self._num_envs, self._actions_dim),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._logprobs = torch.full(size=(self._rollout_timesteps, self._num_envs, 1),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)

        self._next_obs = torch.full(size=(self._rollout_timesteps, self._num_envs, self._obs_dim),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device) 
        self._next_values = torch.full(size=(self._rollout_timesteps, self._num_envs, 1),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._next_terminations = torch.full(size=(self._rollout_timesteps, self._num_envs, 1),
                        fill_value=False,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._next_dones = torch.full(size=(self._rollout_timesteps, self._num_envs, 1),
                        fill_value=False,
                        dtype=self._dtype,
                        device=self._torch_device)
        
        self._rewards = torch.full(size=(self._rollout_timesteps, self._num_envs, 1),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
        
        
        self._advantages = torch.full(size=(self._rollout_timesteps, self._num_envs, 1),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._returns = torch.full(size=(self._rollout_timesteps, self._num_envs, 1),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)

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
