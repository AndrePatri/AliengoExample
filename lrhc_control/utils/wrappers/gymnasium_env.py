from lrhc_control.utils.shared_data.training_env import Observations, NextObservations
from lrhc_control.utils.shared_data.training_env import TotRewards
from lrhc_control.utils.shared_data.training_env import Actions
from lrhc_control.utils.shared_data.training_env import Terminations
from lrhc_control.utils.shared_data.training_env import Truncations

from lrhc_control.utils.episodic_rewards import EpisodicRewards

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal
from SharsorIPCpp.PySharsorIPC import dtype as sharsor_dtype

import torch
import numpy  as np

import gymnasium as gym
from lrhc_control.utils.wrappers.env_transform_utils import DtypeObservation

class Gymnasium2LRHCEnv():

    def __init__(self,
            env_type: str,
            namespace: str = "Gymnasium2LRHCEnv",
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1,
            debug: bool = True,
            use_gpu: bool = True,
            render: bool = False,
            seed: int = 1,
            gym_env_dtype: str = np.float32,
            handle_final_obs: bool = True):
        
        # dtype mapping dictionary
        self._dtype_mapping = {
            'float32': torch.float32,
            'float64': torch.float64
        }

        self._sharsor_dtype_mapping = {
            torch.float32: sharsor_dtype.Float,
            torch.float64: sharsor_dtype.Double
        }

        self._render = render
        self._render_mode = "human" if self._render else None

        self._handle_final_obs = handle_final_obs

        self._env_type = env_type
        self._env = gym.vector.make(env_type, 
                    num_envs=args.num_envs,
                    asynchronous=True,
                    render_mode=self._render_mode)
                    # shared_memory=True,
                    # context="fork",
                    # daemon=True) # gym.make_vec is broken (pipes issue)
        # self._env = DtypeObservation(self._env, dtype=gym_env_dtype) # converting to dtype

        self._action_repeat = self._env.get_attr(name="frame_skip")[0]

        self._seed = seed

        self._namespace = namespace
        
        self._with_gpu_mirror = True
        self._safe_shared_mem = False
        
        self._closed = False

        self._obs_dim = torch.tensor(self._env.single_observation_space.shape).prod().item()
        self._actions_dim = torch.tensor(self._env.single_action_space.shape).prod().item()
        self._env_dt = 0.0
        self._use_gpu = use_gpu
        self._torch_device = "cuda" if self._use_gpu else "cpu"

        self._torch_dtype = self._dtype_mapping[str(self._env.single_observation_space.dtype)]
        
        self._verbose = verbose
        self._vlevel = vlevel
        self._is_debug = debug
        
        self._n_envs = self._env.num_envs

        # action scalings to be applied to agent's output
        self._actions_ub = torch.full((1, self._actions_dim), dtype=self._torch_dtype, device=self._torch_device,
                                        fill_value=1.0)
        self._actions_lb = torch.full((1, self._actions_dim), dtype=self._torch_dtype, device=self._torch_device,
                                        fill_value=-1.0)
        # read bounds from actions space
        self._actions_ub[:, :] = torch.from_numpy(self._env.single_action_space.high)
        self._actions_lb[:, :] = torch.from_numpy(self._env.single_action_space.low)

        reward_thresh_default = 1.0
        self._reward_thresh_lb = torch.full((1, 1), dtype=self._torch_dtype, fill_value=-reward_thresh_default, device=self._torch_device) # used for clipping rewards
        self._reward_thresh_ub = torch.full((1, 1), dtype=self._torch_dtype, fill_value=reward_thresh_default, device=self._torch_device) 

        self._obs = None
        self._next_obs = None
        self._actions = None
        self._tot_rewards = None
        self._terminations = None
        self._truncations = None
        
        self.custom_db_data = {}
        self.custom_db_info = {}

        self._episodic_rewards_metrics = None

        self._episode_duration_ub = 1

        self._episode_timeout_lb = self._episode_duration_ub
        self._episode_timeout_ub = self._episode_duration_ub
        
        self._n_steps_task_rand_lb = -1
        self._n_steps_task_rand_ub = -1

        self._init_shared_data()
        
        self.reset()

    def __del__(self):
        self.close()

    def close(self):
        
        if not self._closed:
            self._next_obs.close()
            self._obs.close()
            self._actions.close()
            self._tot_rewards.close()

            self._terminations.close()
            self._truncations.close()

            self._env.close()
            self._closed = True

    def _init_shared_data(self):
        
        self._obs = Observations(namespace=self._namespace,
                            n_envs=self._n_envs,
                            obs_dim=self._obs_dim,
                            obs_names=self.obs_names(),
                            env_names=None,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=self._use_gpu,
                            fill_value=0.0,
                            dtype=self._sharsor_dtype_mapping[self._torch_dtype])
        
        self._next_obs = NextObservations(namespace=self._namespace,
                            n_envs=self._n_envs,
                            obs_dim=self._obs_dim,
                            obs_names=self.obs_names(),
                            env_names=None,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=self._use_gpu,
                            fill_value=0.0,
                            dtype=self._sharsor_dtype_mapping[self._torch_dtype])
        
        self._actions = Actions(namespace=self._namespace,
                            n_envs=self._n_envs,
                            action_dim=self._actions_dim,
                            action_names=self.action_names(),
                            env_names=None,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=self._use_gpu,
                            fill_value=0.0,
                            dtype=self._sharsor_dtype_mapping[self._torch_dtype])
        
        self._tot_rewards = TotRewards(namespace=self._namespace,
                            n_envs=self._n_envs,
                            reward_names=self.sub_rew_names(),
                            env_names=None,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=self._use_gpu,
                            fill_value=0.0,
                            dtype=self._sharsor_dtype_mapping[self._torch_dtype])
        
        self._terminations = Terminations(namespace=self._namespace,
                            n_envs=self._n_envs,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=self._use_gpu,
                            fill_value=False) 
        
        self._truncations = Truncations(namespace=self._namespace,
                            n_envs=self._n_envs,
                            is_server=True,
                            verbose=self._verbose,
                            vlevel=self._vlevel,
                            safe=True,
                            force_reconnection=True,
                            with_gpu_mirror=self._use_gpu,
                            fill_value=False)

        self._obs.run()
        self._next_obs.run()
        self._actions.run()
        self._tot_rewards.run()
        self._truncations.run()
        self._terminations.run()
        
        self._episodic_rewards_metrics = EpisodicRewards(reward_tensor=self._tot_rewards.get_torch_mirror(),
                                        reward_names=["total_reward"],
                                        ep_vec_freq=1)
        self._episodic_rewards_metrics.set_constant_data_scaling(scaling=1)

    def gym_env(self):
        return self._env
    
    def ep_rewards_metrics(self):
        return self._episodic_rewards_metrics
    
    def dtype(self):
        return self._torch_dtype 
    
    def n_envs(self):
        return self._n_envs
    
    def obs_dim(self):
        return self._obs_dim
    
    def actions_dim(self):
        return self._actions_dim
    
    def obs_names(self):
        obs_names = [""]*self._obs_dim
        for i in range(len(obs_names)):
            obs_names[i] = f"obs_n.{i}"
        return obs_names
    
    def action_names(self):
        action_names = [""]*self._actions_dim
        for i in range(len(action_names)):
            action_names[i] = f"action_n.{i}"
        return action_names

    def sub_rew_names(self):
        return ["total_reward"]

    def get_obs(self, clone:bool=False):
        if clone:
            return self._obs.get_torch_mirror(gpu=self._use_gpu).clone()
        else:
            return self._obs.get_torch_mirror(gpu=self._use_gpu)

    def get_next_obs(self, clone:bool=False):
        if clone:
            return self._next_obs.get_torch_mirror(gpu=self._use_gpu).clone()
        else:
            return self._next_obs.get_torch_mirror(gpu=self._use_gpu)
        
    def get_actions(self, clone:bool=False):
        if clone:
            return self._actions.get_torch_mirror(gpu=self._use_gpu).clone()
        else:
            return self._actions.get_torch_mirror(gpu=self._use_gpu)
            
    def get_rewards(self, clone:bool=False):
        if clone:
            return self._tot_rewards.get_torch_mirror(gpu=self._use_gpu).clone()
        else:
            return self._tot_rewards.get_torch_mirror(gpu=self._use_gpu)

    def get_terminations(self, clone:bool=False):
        if clone:
            return self._terminations.get_torch_mirror(gpu=self._use_gpu).clone()
        else:
            return self._terminations.get_torch_mirror(gpu=self._use_gpu)
        
        return self._terminations.get_torch_mirror(gpu=self._use_gpu)

    def get_truncations(self, clone:bool=False):
        if clone:
            return self._truncations.get_torch_mirror(gpu=self._use_gpu).clone()
        else:
            return self._truncations.get_torch_mirror(gpu=self._use_gpu)
        
    def get_actions_lb(self):
        return self._actions_lb

    def get_actions_ub(self):
        return self._actions_ub

    def name(self):
        return self._namespace
    
    def episode_timeout_bounds(self):
        return self._episode_timeout_lb, self._episode_timeout_ub
    
    def task_rand_timeout_bounds(self):
        return self._n_steps_task_rand_lb, self._n_steps_task_rand_ub
    
    def n_action_reps(self):
        return self._action_repeat
    
    def using_gpu(self):
        return self._use_gpu
    
    def get_file_paths(self):
        empty_list = []
        return empty_list
    
    def get_aux_dir(self):
        empty_list = []
        return empty_list

    def reset(self):
        
        self._actions.reset()
        self._obs.reset()
        self._next_obs.reset()
        self._tot_rewards.reset()

        self._terminations.reset()
        self._truncations.reset()

        obs, _ = self._env.reset(seed=self._seed)
        self._to_torch(data=obs,output=self.get_obs())

        self.reset_custom_db_data(keep_track=False)
        self._episodic_rewards_metrics.reset(keep_track=False)

    def step(self, 
            action):
        
        stepping_ok = True
        
        actions = self._actions.get_torch_mirror(gpu=self._use_gpu)
        actions[:, :] = action.detach() # writes actions

        # step gymnasium env using the given action
        gym_env_action = actions.cpu().numpy()
        observations, rewards, terminations, truncations, infos = self._env.step(gym_env_action)

        if self._render:
            self._env.render()

        # fill transition data from gymansium env
        self._to_torch(data=observations,output=self.get_obs())
        self._to_torch(data=rewards.reshape(-1, 1),output=self.get_rewards())
        self._to_torch(data=terminations.reshape(-1, 1),output=self.get_terminations())
        self._to_torch(data=truncations.reshape(-1, 1),output=self.get_truncations())

        # handle final observations (next obs has to be the actual one)
        if self._handle_final_obs and ("final_observation" in infos):  # some sub-envs have terminated
            ep_finished = np.logical_or(terminations, truncations).flatten()
            real_next_obs = np.full_like(observations[ep_finished, :], 0.0)
            for env_idx in range(infos["final_observation"][ep_finished].size):
                real_next_obs[env_idx, :] = infos["final_observation"][ep_finished][env_idx]
            observations[ep_finished, :] = real_next_obs
        self._to_torch(data=observations,output=self.get_next_obs()) # next obs always holds the 
        # real state after stepping, even when terminations occur

        self.post_step()

        return stepping_ok
    
    def _to_torch(self, data, output):
        output[:, :]=torch.tensor(data,dtype=self._torch_dtype,device=self._torch_device)

    def _debug(self):

        if self._use_gpu:
            self._obs.synch_mirror(from_gpu=True) # copy data from gpu to cpu view
            self._next_obs.synch_mirror(from_gpu=True)
            self._actions.synch_mirror(from_gpu=True)
            self._tot_rewards.synch_mirror(from_gpu=True)
            self._truncations.synch_mirror(from_gpu=True) 
            self._terminations.synch_mirror(from_gpu=True) 

        self._obs.synch_all(read=False, retry=True) # copies data on CPU shared mem
        self._next_obs.synch_all(read=False, retry=True)
        self._actions.synch_all(read=False, retry=True) 
        self._tot_rewards.synch_all(read=False, retry=True)
        self._truncations.synch_all(read=False, retry = True)
        self._terminations.synch_all(read=False, retry = True)

    def _update_custom_db_data(self,
                    episode_finished):
        pass
    
    def reset_custom_db_data(self, keep_track: bool = True):
        # to be called periodically to reset custom db data stat. collection 
        for custom_db_data in self.custom_db_data.values():
            custom_db_data.reset(keep_track=keep_track)

    def post_step(self):
        if self._is_debug:
            terminated = self._terminations.get_torch_mirror(gpu=self._use_gpu)
            truncated = self._truncations.get_torch_mirror(gpu=self._use_gpu)
            episode_finished = torch.logical_or(terminated,
                            truncated)
            episode_finished_cpu = episode_finished.cpu()
            self._debug() # copies db data on shared memory
            self._update_custom_db_data(episode_finished=episode_finished_cpu)
            self._episodic_rewards_metrics.update(rewards = self._tot_rewards.get_torch_mirror(gpu=False),
                            ep_finished=episode_finished_cpu)

if __name__ == "__main__":  

    from lrhc_control.training_algs.sac.sac import SAC
    from lrhc_control.training_algs.ppo.ppo import PPO

    import argparse,os

    parser = argparse.ArgumentParser(description="Set CPU affinity for the script.")
    parser.add_argument('--db', action=argparse.BooleanOptionalAction, default=True, help='Whether to enable local data logging for the algorithm (reward metrics, etc..)')
    parser.add_argument('--env_db', action=argparse.BooleanOptionalAction, default=True, help='Whether to enable env db data logging on \
                            shared mem (e.g.reward metrics are not available for reading anymore)')
    parser.add_argument('--rmdb', action=argparse.BooleanOptionalAction, default=True, help='Whether to enable remote debug (e.g. data logging on remote servers)')
    parser.add_argument('--obs_norm', action=argparse.BooleanOptionalAction, default=True, help='Whether to enable the use of running normalizer in agent')
    parser.add_argument('--run_name', type=str, help='Name of training run', default="GymnasiumEnvTest")
    parser.add_argument('--drop_dir', type=str, help='Directory root where all run data will be dumped',default="/tmp")
    parser.add_argument('--comment', type=str, help='Any useful comment associated with this run',default="")
    parser.add_argument('--seed', type=int, help='seed', default=1)
    
    parser.add_argument('--eval', action=argparse.BooleanOptionalAction, default=False, help='Whether to perform an evaluation run')
    parser.add_argument('--mpath', type=str, help='Model path to be used for policy evaluation',default=None)
    parser.add_argument('--mname', type=str, help='Model name',default=None)

    parser.add_argument('--n_eval_timesteps', type=int, help='N. of evaluation timesteps to be performed', default=None)
    parser.add_argument('--dump_checkpoints', action=argparse.BooleanOptionalAction, default=False, help='Whether to dump model checkpoints during training')
    parser.add_argument('--use_cpu', action=argparse.BooleanOptionalAction, default=False, help='If set, all the training (data included) will be perfomed on CPU')

    parser.add_argument('--ns', type=str, help='Namespace to be used for shared memory', default="Gymnasium2LRHCEnv")
    parser.add_argument('--num_envs', type=int, help='seed', default=1)

    parser.add_argument('--render', action=argparse.BooleanOptionalAction, default=False, help='Whether to render environemnt')
    parser.add_argument('--handle_final_obs', action=argparse.BooleanOptionalAction, default=True, help='Whether to handle terminal obs properly')

    parser.add_argument('--sac', action=argparse.BooleanOptionalAction, default=True, help='')
    parser.add_argument('--use_cer', action=argparse.BooleanOptionalAction, default=False, help='use combined experience replay')
    
    parser.add_argument('--actor_size', type=int, help='seed', default=64)
    parser.add_argument('--critic_size', type=int, help='seed', default=64)

    parser.add_argument('--env_type', type=str, help='Name of env to be created',default="HalfCheetah-v4")

    args = parser.parse_args()
    args_dict = vars(args)
    render_mode = "human" if args.render else None
    env_type = args.env_type

    if (not args.mpath is None) and (not args.mname is None):
        mpath_full = os.path.join(args.mpath, args.mname)
    else:
        mpath_full=None

    env_wrapper = Gymnasium2LRHCEnv(env_type=env_type,
                        namespace=args.ns,
                        verbose=True,
                        vlevel=VLevel.V2,
                        debug=args.env_db,
                        use_gpu=not args.use_cpu,
                        render=args.render,
                        seed=args.seed,
                        gym_env_dtype=np.float32,
                        handle_final_obs=args.handle_final_obs)
    env_type2="training" if not args.eval else "evaluation"
    Journal.log("gymnasium_env.py",
            "wrapper_test",
            f"loading {env_type2} env {env_type}",
            LogType.INFO,
            throw_when_excep = True)
    
    custom_args_dict = {}
    if args.sac:
        algo = SAC(env=env_wrapper, 
                debug=args.db, 
                remote_db=args.rmdb,
                seed=args.seed)
        custom_args_dict["use_combined_exp_replay"]=args.use_cer
    else:
        algo = PPO(env=env_wrapper, 
                debug=args.db, 
                remote_db=args.rmdb,
                seed=args.seed)
    
    custom_args_dict.update(args_dict)
    custom_args_dict.update({"gymansium_env_type": env_type})

    algo.setup(run_name=args.run_name,
        ns=args.ns, 
        verbose=True,
        drop_dir_name=args.drop_dir,
        custom_args = custom_args_dict,
        comment=args.comment,
        eval=args.eval,
        model_path=mpath_full,
        n_eval_timesteps=args.n_eval_timesteps,
        dump_checkpoints=args.dump_checkpoints,
        norm_obs=args.obs_norm)
    
    if not args.eval:
        try:
            while not algo.is_done():
                if not algo.learn():
                    algo.done()
        except KeyboardInterrupt:
            algo.done() # in case it's interrupted by user
    else: # eval phase
        with torch.no_grad(): # no need for grad computation
            try:
                while not algo.is_done():
                    if not algo.eval():
                        algo.done()
            except KeyboardInterrupt:
                algo.done() # in case it's interrupted by userr