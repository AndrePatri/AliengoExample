from lrhc_control.utils.episodic_data import EpisodicData
import torch
from typing import List

from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import Journal
from EigenIPC.PyEigenIPC import LogType

class EpisodicRewards(EpisodicData):

    def __init__(self,
            reward_tensor: torch.Tensor,
            reward_names: List[str] = None,
            ep_vec_freq: int = None,
            store_transitions: bool = False, # also store detailed transition history
            max_ep_duration: int = -1):
        
        # separate ep data metrics for total reward
        self._tot_reward_episodic_stats=EpisodicData(name="TotReward",
            data_tensor=torch.sum(reward_tensor,dim=1, keepdim=True),
            data_names=["TotReward"],
            ep_vec_freq=ep_vec_freq,
            store_transitions=store_transitions,
            max_ep_duration=max_ep_duration)
        
        # the maximum ep length
        super().__init__(data_tensor=reward_tensor, data_names=reward_names, name="SubRewards",
                ep_vec_freq=ep_vec_freq,
                store_transitions=store_transitions,
                max_ep_duration=max_ep_duration)

        self.set_constant_data_scaling(scaling=max_ep_duration)
    
    def set_constant_data_scaling(self, scaling: int):
        # overrides parent

        scaling = torch.full((self._n_envs, 1),
                    fill_value=scaling,
                    dtype=torch.int32,device="cpu") # reward metrics are scaled using
        super().set_constant_data_scaling(enable=True,scaling=scaling)
        self._tot_reward_episodic_stats.set_constant_data_scaling(enable=True,scaling=scaling)

    def enable_timestep_scaling(self):
        super().set_constant_data_scaling(enable=False)
        self._tot_reward_episodic_stats.set_constant_data_scaling(enable=False)

    def update(self, 
        rewards: torch.Tensor,
        ep_finished: torch.Tensor,
        ignore_ep_end: torch.Tensor = None):

        super().update(new_data=rewards, ep_finished=ep_finished, ignore_ep_end=ignore_ep_end)
        self._tot_reward_episodic_stats.update(new_data=torch.sum(rewards, dim=1, keepdim=True), 
            ep_finished=ep_finished, ignore_ep_end=ignore_ep_end)

    def reward_names(self):
        return self._data_names
    
    def n_rewards(self):
        return len(self._data_names)
    
    def _init_data(self):
        # override to add functionality
        super()._init_data()
        self._tot_reward_episodic_stats._init_data()

    def reset(self,
        keep_track: bool = None,
        to_be_reset: torch.Tensor = None):
        
        super().reset(keep_track=keep_track,
            to_be_reset=to_be_reset)
        self._tot_reward_episodic_stats.reset(keep_track=keep_track,
            to_be_reset=to_be_reset)

    def get_full_episodic_subrew(self, 
        env_selector: torch.Tensor = None):
        return self.get_full_episodic_data(env_selector=env_selector)
    
    def get_full_episodic_totrew(self, 
        env_selector: torch.Tensor = None):
        return self._tot_reward_episodic_stats.get_full_episodic_data(env_selector=env_selector)
    
    # wrapping base methods for sub rewards
    def get_sub_rew_max(self, 
        env_selector: torch.Tensor = None):
        return super().get_max(env_selector=env_selector)
    
    def get_sub_rew_avrg(self, 
        env_selector: torch.Tensor = None):
        return super().get_avrg(env_selector=env_selector)

    def get_sub_rew_min(self, 
        env_selector: torch.Tensor = None):
        return super().get_min(env_selector=env_selector)

    def get_sub_rew_max_over_envs(self, 
        env_selector: torch.Tensor = None):
        return super().get_max_over_envs(env_selector=env_selector)
    
    def get_sub_rew_avrg_over_envs(self, 
        env_selector: torch.Tensor = None):
        return super().get_avrg_over_envs(env_selector=env_selector)

    def get_sub_rew_std_over_envs(self, 
        env_selector: torch.Tensor = None):
        return super().get_std_over_envs(env_selector=env_selector)

    def get_sub_rew_min_over_envs(self, 
        env_selector: torch.Tensor = None):
        return super().get_min_over_envs(env_selector=env_selector)
    
    # tot reward
    def get_tot_rew_max(self, 
        env_selector: torch.Tensor = None):
        return self._tot_reward_episodic_stats.get_max(env_selector=env_selector)
    
    def get_tot_rew_avrg(self, 
        env_selector: torch.Tensor = None):
        return self._tot_reward_episodic_stats.get_avrg(env_selector=env_selector)

    def get_tot_rew_min(self, 
        env_selector: torch.Tensor = None):
        return self._tot_reward_episodic_stats.get_min(env_selector=env_selector)

    def get_tot_rew_max_over_envs(self, 
        env_selector: torch.Tensor = None):
        return self._tot_reward_episodic_stats.get_max_over_envs(env_selector=env_selector)
    
    def get_tot_rew_avrg_over_envs(self, 
        env_selector: torch.Tensor = None):
        return self._tot_reward_episodic_stats.get_avrg_over_envs(env_selector=env_selector)

    def get_tot_rew_std_over_envs(self, 
        env_selector: torch.Tensor = None):
        return self._tot_reward_episodic_stats.get_std_over_envs(env_selector=env_selector)

    def get_tot_rew_min_over_envs(self, 
        env_selector: torch.Tensor = None):
        return self._tot_reward_episodic_stats.get_min_over_envs(env_selector=env_selector)

if __name__ == "__main__":  

    n_envs = 1
    data_dim = 3
    max_ep_length = 1
    ep_finished = torch.full((n_envs, 1),fill_value=0,dtype=torch.bool,device="cpu")
    new_data = torch.full((n_envs, data_dim),fill_value=0,dtype=torch.float32,device="cpu")
    data_names = ["okokok", "sdcsdc", "cdcsdcplpl"]
    reward_data = EpisodicRewards(reward_tensor=new_data,
                    reward_names=data_names,
                    max_episode_length=max_ep_length)
    reward_data.reset()

    ep_finished[:, :] = False
    new_data[0, 0] = 1
    new_data[0, 1] = 2
    new_data[0, 2] = 3

    reward_data.update(rewards=new_data,
                ep_finished=ep_finished)

    ep_finished[:, :] = False
    new_data+=1 

    reward_data.update(rewards=new_data,
                ep_finished=ep_finished)
    
    ep_finished[:, :] = False
    new_data+=1 

    reward_data.update(rewards=new_data,
                ep_finished=ep_finished)

    print("get_rollout_stat:")
    print(reward_data.get_avrg())

    print("get_rollout_stat_env_avrg:")
    print(reward_data.get_avrg_over_env())

    print("get_rollout_stat_comp:")
    print(reward_data.get_avrg())

    print("get_rollout_stat_comp_env_avrg:")
    print(reward_data.get_tot_avrg())