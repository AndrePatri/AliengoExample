from lrhc_control.envs.lrhc_training_env_base import LRhcTrainingEnvBase

import torch

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType

class LRhcHeightChange(LRhcTrainingEnvBase):

    def __init__(self,
            namespace: str,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1,
            use_gpu: bool = True,
            dtype: torch.dtype = torch.float32):

        obs_dim = 4
        actions_dim = 1

        episode_length = 250

        env_name = "LRhcHeightChange"

        self._epsi = 1e-6
        
        self._cnstr_viol_scale_f = 1e3

        super().__init__(namespace=namespace,
                    obs_dim=obs_dim,
                    actions_dim=actions_dim,
                    episode_length=episode_length,
                    env_name=env_name,
                    verbose=verbose,
                    vlevel=vlevel,
                    use_gpu=use_gpu,
                    dtype=dtype)
    
    def _apply_actions_to_rhc(self):
        
        agent_action = self.get_last_actions()

        rhc_current_ref = self._rhc_refs.rob_refs.root_state.get_p(gpu=self._use_gpu)
        rhc_current_ref[:, 2:3] = agent_action # overwrite z ref

        self._rhc_refs.rob_refs.root_state.set_p(p=rhc_current_ref,
                                            gpu=self._use_gpu) # write refs

        if self._use_gpu:

            self._rhc_refs.rob_refs.root_state.synch_mirror(from_gpu=self._use_gpu) # write from gpu to cpu mirror

        self._rhc_refs.rob_refs.root_state.synch_all(read=False, wait=True) # write mirror to shared mem

    def _compute_rewards(self):
        
        # task error
        h_ref = self._agent_refs.rob_refs.root_state.get_p(gpu=self._use_gpu)[:, 2:3] # getting target z ref
        robot_h = self._robot_state.root_state.get_p(gpu=self._use_gpu)[:, 2:3]
        h_error = torch.norm((h_ref - robot_h))

        # rhc penalties
        rhc_cost = self._rhc_status.rhc_cost.get_torch_view(gpu=self._use_gpu)
        rhc_const_viol = self._rhc_status.rhc_constr_viol.get_torch_view(gpu=self._use_gpu)
        rhc_fail_penalty = self._rhc_status.fails.get_torch_view(gpu=self._use_gpu)

        rewards = self._rewards.get_torch_view(gpu=self._use_gpu)
        rewards[:, 0:1] = 100 * (1 - h_error).clamp_(0, torch.inf)
        rewards[:, 1:2] = self._epsi * torch.reciprocal(rhc_cost + self._epsi)
        rewards[:, 2:3] = self._epsi * torch.reciprocal(rhc_const_viol + self._epsi)
        rewards[:, 3:4] = self._epsi * torch.reciprocal(rhc_fail_penalty + self._epsi)
        
        tot_rewards = self._tot_rewards.get_torch_view(gpu=self._use_gpu)
        tot_rewards[:, :] = torch.sum(rewards, dim=1, keepdim=True)

    def _get_observations(self):
                
        self._synch_obs(gpu=self._use_gpu)
            
        agent_h_ref = self._agent_refs.rob_refs.root_state.get_p(gpu=self._use_gpu)[:, 2:3] # getting z ref
        robot_h = self._robot_state.root_state.get_p(gpu=self._use_gpu)[:, 2:3]
        rhc_cost = self._rhc_status.rhc_cost.get_torch_view(gpu=self._use_gpu)
        rhc_const_viol = self._rhc_status.rhc_constr_viol.get_torch_view(gpu=self._use_gpu)

        obs = self._obs.get_torch_view(gpu=self._use_gpu)
        obs[:, 0:1] = robot_h
        obs[:, 1:2] = rhc_cost
        obs[:, 2:3] = self._cnstr_viol_scale_f * rhc_const_viol
        obs[: ,3:4] = agent_h_ref
    
    def _randomize_refs(self):
        
        agent_p_ref_current = self._agent_refs.rob_refs.root_state.get_p(gpu=self._use_gpu)

        ref_lb = 0.35
        ref_ub = 0.7
        agent_p_ref_current[:, 2:3] = (ref_ub-ref_lb) * torch.rand_like(agent_p_ref_current[:, 2:3]) + ref_lb # randomize h ref

        self._agent_refs.rob_refs.root_state.set_p(p=agent_p_ref_current,
                                            gpu=self._use_gpu)

        self._synch_refs(gpu=self._use_gpu)
    
    def _get_obs_names(self):

        obs_names = [""] * self.obs_dim()

        obs_names[0] = "robot_h"
        obs_names[1] = "rhc_cost"
        obs_names[2] = "rhc_const_viol"
        obs_names[3] = "agent_h_ref"

        return obs_names

    def _get_action_names(self):

        action_names = [""] * self.actions_dim()

        action_names[0] = "rhc_cmd_h"

        return action_names

    def _get_rewards_names(self):

        reward_names = [""] * 4

        reward_names[0] = "h_error"
        reward_names[1] = "rhc_cost"
        reward_names[2] = "rhc_const_viol"
        reward_names[3] = "rhc_fail_penalty"

        return reward_names