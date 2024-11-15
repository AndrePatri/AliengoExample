from lrhc_control.training_algs.sac.sactor_critic_algo import SActorCriticAlgoBase

import torch 
import torch.nn as nn
import torch.nn.functional as F

import os

import time

class SAC(SActorCriticAlgoBase):

    def __init__(self,
            env, 
            debug = False,
            remote_db = False,
            seed: int = 1):

        super().__init__(env=env, 
                    debug=debug,
                    remote_db=remote_db,
                    seed=seed)

        self._this_child_path = os.path.abspath(__file__) # overrides parent
    
    def _collect_transition(self):
        
        # experience collection
        self._switch_training_mode(train=False)

        obs = self._env.get_obs(clone=True) # also accounts for resets when envs are 
        # either terminated or truncated. CRUCIAL: we need to clone, 
        # otherwise obs is be a view and will be overridden in the call to step
        # with next_obs!!!
        if self._vec_transition_counter > (self._warmstart_vectimesteps-1):
            actions, _, _ = self._agent.get_action(x=obs)
            actions = actions.detach()
            if self._n_expl_envs>0 and \
                (self._vec_transition_counter%self._noise_freq==0 or \
                self._pert_counter>0):
                self._perturb_some_actions(actions=actions)
                
        else:
            actions = self._sample_random_actions()
                
        # perform a step of the (vectorized) env and retrieve trajectory
        env_step_ok = self._env.step(actions)
        
        # add experience to replay buffer
        self._add_experience(obs=obs,
                actions=actions,
                rewards=self._env.get_rewards(clone=False), # no need to clone 
                next_obs=self._env.get_next_obs(clone=False), # data is copied anyway
                next_terminal=self._env.get_terminations(clone=False)) 

        return env_step_ok
    
    def _collect_eval_transition(self):
        
        # experience collection
        self._switch_training_mode(train=False)

        obs = self._env.get_obs(clone=True) # also accounts for resets when envs are 
        # either terminated or truncated. CRUCIAL: we need to clone, 
        # otherwise obs is be a view and will be overridden in the call to step
        # with next_obs!!!

        if not self._override_agent_action:
            actions, _, mean = self._agent.get_action(x=obs)
            actions = actions.detach()
            
            if self._det_eval: # use mean instead of stochastic policy
                actions[:, :] = mean.detach()

        else:

            self._actions_override.synch_all(read=True,retry=True) # read from CPU
            # write on GPU
            if self._use_gpu:
                self._actions_override.synch_mirror(from_gpu=False,non_blocking=True)
            actions=self._actions_override.get_torch_mirror(gpu=self._use_gpu)

        # perform a step of the (vectorized) env and retrieve trajectory
        env_step_ok = self._env.step(actions)

        if self._load_qf:
            # get qf value for state and action using average of 
            # q networks
            qf1_v=self._agent.get_qf1_val(x=obs,a=actions)
            qf2_v=self._agent.get_qf2_val(x=obs,a=actions)
            qf_v=(qf1_v+qf2_v)/2 # use average
            qf_vals=self._qf_vals.get_torch_mirror(gpu=False)
            qf_vals[:, :]=qf_v.cpu()
            self._qf_vals.synch_all(read=False,retry=False)

            # target qf
            next_obs=self._env.get_next_obs(clone=False)
            next_action, next_log_pi, _ = self._agent.get_action(next_obs)
            qf1_v_next=self._agent.get_qf1_val(x=next_obs,a=next_action)
            qf2_v_next=self._agent.get_qf2_val(x=next_obs,a=next_action)
            min_qf_next_target = torch.min(qf1_v_next, qf2_v_next)
            rew_now=self._env.get_rewards(clone=False)
            reached_terminal_state=self._env.get_terminations(clone=False).to(torch.float32)
            qf_trgt_v=rew_now+(1 - reached_terminal_state)*self._discount_factor*min_qf_next_target
            qf_trgt=self._qf_trgt.get_torch_mirror(gpu=False)
            qf_trgt[:, :]=qf_trgt_v
            self._qf_trgt.synch_all(read=False,retry=False)

        return env_step_ok
    
    def _update_policy(self):
        
        # training phase
        if self._vec_transition_counter > (self._warmstart_vectimesteps-1):
                
            self._switch_training_mode(train=True)

            obs,actions,next_obs,rewards,next_terminal = self._sample() # sample
            # experience from replay buffer

            with torch.no_grad():
                next_action, next_log_pi, _ = self._agent.get_action(next_obs)
                qf1_next_target = self._agent.get_qf1t_val(next_obs, next_action)
                qf2_next_target = self._agent.get_qf2t_val(next_obs, next_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self._alpha * next_log_pi
                next_q_value = rewards.flatten() + (1 - next_terminal.flatten()) * self._discount_factor * (min_qf_next_target).view(-1)

            qf1_a_values = self._agent.get_qf1_val(obs, actions).view(-1)
            qf2_a_values = self._agent.get_qf2_val(obs, actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            self._qf_optimizer.zero_grad()
            qf_loss.backward()
            self._qf_optimizer.step()
            self._n_qfun_updates[self._log_it_counter]+=1

            if self._update_counter % self._policy_freq == 0:  # TD 3 Delayed update support
                # policy update
                for i in range(self._policy_freq): # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = self._agent.get_action(obs)
                    qf1_pi = self._agent.get_qf1_val(obs, pi)
                    qf2_pi = self._agent.get_qf2_val(obs, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((self._alpha * log_pi) - min_qf_pi).mean()
                    self._actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self._actor_optimizer.step()
                    if self._autotune:
                        with torch.no_grad():
                            _, log_pi, _ = self._agent.get_action(obs)
                        alpha_loss = (-self._log_alpha.exp() * (log_pi + self._target_entropy)).mean()
                        self._a_optimizer.zero_grad()
                        alpha_loss.backward()
                        self._a_optimizer.step()
                        self._alpha = self._log_alpha.exp().item()
                    self._n_policy_updates[self._log_it_counter]+=1

            # update the target networks
            if self._update_counter % self._trgt_net_freq == 0:
                for param, target_param in zip(self._agent.qf1.parameters(), self._agent.qf1_target.parameters()):
                    target_param.data.copy_(self._smoothing_coeff * param.data + (1 - self._smoothing_coeff) * target_param.data)
                for param, target_param in zip(self._agent.qf2.parameters(), self._agent.qf2_target.parameters()):
                    target_param.data.copy_(self._smoothing_coeff * param.data + (1 - self._smoothing_coeff) * target_param.data)
                self._n_tqfun_updates[self._log_it_counter]+=1

            if self._debug and \
                (self._vec_transition_counter % self._db_vecstep_frequency):
                # DEBUG INFO
        
                # current q estimates on training batch
                self._qf1_vals_mean[self._log_it_counter, 0] = qf1_a_values.mean().item()
                self._qf2_vals_mean[self._log_it_counter, 0] = qf2_a_values.mean().item()
                self._qf1_vals_std[self._log_it_counter, 0] = qf1_a_values.std().item()
                self._qf2_vals_std[self._log_it_counter, 0] = qf2_a_values.std().item()
                self._qf1_vals_max[self._log_it_counter, 0] = qf1_a_values.max().item()
                self._qf2_vals_max[self._log_it_counter, 0] = qf2_a_values.max().item()
                self._qf1_vals_min[self._log_it_counter, 0] = qf1_a_values.min().item()
                self._qf2_vals_min[self._log_it_counter, 0] = qf2_a_values.min().item()

                # q losses (~bellman error)
                self._qf1_loss[self._log_it_counter, 0] = qf1_loss.item()
                self._qf2_loss[self._log_it_counter, 0] = qf2_loss.item()
    
                if self._update_counter % self._policy_freq == 0:
                    # just log last policy update info
                    self._actor_loss[self._log_it_counter, 0] = actor_loss.item()
                    policy_entropy=-log_pi
                    self._policy_entropy_mean[self._log_it_counter, 0] = policy_entropy.mean().item()
                    self._policy_entropy_std[self._log_it_counter, 0] = policy_entropy.std().item()
                    self._policy_entropy_max[self._log_it_counter, 0] = policy_entropy.max().item()
                    self._policy_entropy_min[self._log_it_counter, 0] = policy_entropy.min().item()

                    self._alphas[self._log_it_counter, 0] = self._alpha
                    if self._autotune:
                        self._alpha_loss[self._log_it_counter, 0] = alpha_loss.item()
