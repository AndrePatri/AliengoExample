from lrhc_control.training_algs.ppo.actor_critic_algo import ActorCriticAlgoBase

import torch 
import torch.nn as nn

import os

import time

class PPO(ActorCriticAlgoBase):
     
    def __init__(self,
            env, 
            debug = True,
            remote_db = False,
            seed: int = 1):

        super().__init__(env=env, 
                    debug=debug,
                    remote_db=remote_db,
                    seed=seed)

        self._this_child_path = os.path.abspath(__file__) # overrides parent

    def _collect_rollout(self):
        
        # experience collection 
        self._switch_training_mode(train=False)

        # collect data from current policy over a number of timesteps
        for transition in range(self._rollout_vec_timesteps):
            
            obs = self._env.get_obs(clone=True) # also accounts for resets when envs are 
            # either terminated or truncated. CRUCIAL: we need to clone, 
            # otherwise obs is a view and will be overridden in the call to step
            # with next_obs!!!

            # sample actions from latest policy (actor) and state value from latest value function (critic)
            action, logprob, _ = self._agent.get_action(obs, only_mean=(self._eval and self._det_eval)) 
            action = action.detach() 
            logprob = logprob.detach()

            # perform a step of the (vectorized) env and retrieve trajectory
            env_step_ok = self._env.step(action)

            next_done=torch.logical_or(self._env.get_terminations(clone=False), 
                    self._env.get_truncations(clone=False))

            # add experience to rollout buffer
            self._add_experience(pos=transition,
                    obs=obs,
                    actions=action,
                    logprob=logprob.view(-1, 1),
                    rewards=self._env.get_rewards(clone=False),
                    next_obs=self._env.get_next_obs(clone=False),
                    next_terminal=self._env.get_terminations(clone=False),
                    next_done=next_done)

            if not env_step_ok:
                return False
        
        return env_step_ok
    
    def _collect_eval_rollout(self):
        
        # experience collection 
        self._switch_training_mode(train=False)

        # collect data from current policy over a number of timesteps
        for transition in range(self._rollout_vec_timesteps):
            
            obs = self._env.get_obs(clone=True) # also accounts for resets when envs are 
            # either terminated or truncated. CRUCIAL: we need to clone, 
            # otherwise obs is a view and will be overridden in the call to step
            # with next_obs!!!

            # sample actions from latest policy (actor) and state value from latest value function (critic)
            action, logprob, _ = self._agent.get_action(obs, only_mean=(self._eval and self._det_eval)) 
            action = action.detach() 
            logprob = logprob.detach()

            # perform a step of the (vectorized) env and retrieve trajectory
            env_step_ok = self._env.step(action)

            if not env_step_ok:
                return False
        
        return True

    def _compute_returns(self):

        # bootstrap: compute advantages and returns
        self._advantages.zero_() # reset advantages
        lastgaelam = 0
        for t in reversed(range(self._rollout_vec_timesteps)):
            # loop over state transitions
            
            nextnonterminal = 1.0 - self._next_terminal[t]
            nextnondone = 1.0 - self._next_done[t]
            
            discounted_returns = self._rewards[t] + self._discount_factor * self._next_values[t] * nextnonterminal
            td_error = discounted_returns - self._values[t] # TD error

            # compute advantages using the Generalized Advantage Estimation (GAE) 
            # GAE estimation needs successive transitions, so we need to stop when the trajectory is either 
            # truncated or terminated (that's why nextnondone is used)
            # note that longer trajectories reduce the bias in the GAE estimator
            
            self._advantages[t] = lastgaelam = td_error + self._discount_factor * self._gae_lambda * nextnondone * lastgaelam
        
        #   cumulative rewards from each time step to the end of the episode
        self._returns[:, :] = self._advantages + self._values

    def _update_policy(self):
        
        self._switch_training_mode(train=True)

        # flatten batches before policy update
        batched_obs = self._obs.view((-1, self._env.obs_dim()))
        batched_logprobs = self._logprobs.view(-1)
        batched_actions = self._actions.view((-1, self._env.actions_dim()))
        batched_advantages = self._advantages.view(-1)
        batched_returns = self._returns.view(-1)
        batched_values = self._values.view(-1)

        # optimize policy and value network
        clipfracs = []
        tot_loss_grads = []
        actor_loss_grads = []
        tloss = []
        vlosses = []
        pglosses = []
        eplosses= []
        old_approx_kls = []
        approx_kls = []

        for epoch in range(self._update_epochs):
            shuffled_batch_indxs = torch.randperm(self._batch_size) # randomizing 
            # indexes for removing temporal correlations
            for start in range(0, self._batch_size, self._minibatch_size):
                end = start + self._minibatch_size
                minibatch_inds = shuffled_batch_indxs[start:end]

                _, newlogprob, entropy, newvalue = self._agent.get_action_and_value(
                                                                    batched_obs[minibatch_inds], 
                                                                    batched_actions[minibatch_inds])
                                                                    
                logratio = newlogprob - batched_logprobs[minibatch_inds]
                ratio = logratio.exp() # ratio between the probability of taking an action
                # under the current policy and the probability of taking the same action under the
                # the previous policy (batched_actions[minibatch_inds])

                with torch.no_grad():
                    # calculate approximate KL divergence http://joschu.net/blog/kl-approx.html
                    # The KL (Kullback-Leibler) divergence is a measure of how one probability 
                    # distribution diverges from a second, expected probability distribution
                    # in PPO, this is used as a regularization term in the objective function
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self._clip_coef).float().mean().item()]

                minibatch_advantages = batched_advantages[minibatch_inds]
                if self._norm_adv: # normalizing advantages if required over minibatch
                    minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (minibatch_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -minibatch_advantages * ratio
                pg_loss2 = -minibatch_advantages * torch.clamp(ratio, 1 - self._clip_coef, 1 + self._clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self._clip_vloss:
                    v_loss_unclipped = (newvalue - batched_returns[minibatch_inds]) ** 2
                    v_clipped = batched_values[minibatch_inds] + torch.clamp(
                        newvalue - batched_values[minibatch_inds],
                        -self._clip_coef_vf,
                        self._clip_coef_vf,
                    )
                    v_loss_clipped = (v_clipped - batched_returns[minibatch_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - batched_returns[minibatch_inds]) ** 2).mean()
                
                # entropy loss
                entropy_loss = entropy.mean()

                # total loss
                loss = pg_loss - self._entropy_coeff * entropy_loss + self._val_f_coeff * v_loss

                # update policy using this minibatch
                self._optimizer.zero_grad() # reset gradients
                loss.backward() # compute backward pass
                    
                # clip gradients
                nn.utils.clip_grad_norm_(self._agent.actor.actor_mean.parameters(), 
                                    self._max_grad_norm_actor) 
                nn.utils.clip_grad_norm_(self._agent.actor.actor_logstd, 
                                    self._max_grad_norm_actor) 
                nn.utils.clip_grad_norm_(self._agent.critic.parameters(), 
                                    self._max_grad_norm_critic)
                # nn.utils.clip_grad_norm_(self._agent.parameters(), 
                #                     self._max_grad_norm_actor)
                self._optimizer.step() # update actor's (policy) parameters

                self._n_policy_updates[self._log_it_counter]+=1
                self._n_vfun_updates[self._log_it_counter]+=1

                # loss_grad_norm = loss.grad.norm()
                # pg_grad_norm = pg_loss.grad.norm()
                with torch.no_grad(): # db data
                    tloss += [loss.item()]
                    vlosses += [self._val_f_coeff * v_loss.item()]
                    pglosses += [pg_loss.item()]
                    eplosses += [- self._entropy_coeff * entropy_loss.item()]
                    # tot_loss_grads += [loss_grad_norm]
                    # actor_loss_grads += [pg_grad_norm]
                    old_approx_kls += [old_approx_kl.item()]
                    approx_kls += [approx_kl.item()]

            if self._target_kl is not None and approx_kl > self._target_kl:
                break
            
        if self._debug:
            y_pred, y_true = batched_values.cpu(), batched_returns.cpu()
            var_y = torch.var(y_true)
            explained_var = torch.nan if var_y == 0 else 1 - torch.var(y_true - y_pred) / var_y

            self._tot_loss_mean[self._log_it_counter, 0] = torch.mean(torch.tensor(tloss))
            self._value_loss_mean[self._log_it_counter, 0] = torch.mean(torch.tensor(vlosses))
            self._policy_loss_mean[self._log_it_counter, 0] = torch.mean(torch.tensor(pglosses))
            self._entropy_loss_mean[self._log_it_counter, 0] = torch.mean(torch.tensor(eplosses))
            # self._tot_loss_grad_norm_mean[self._log_it_counter, 0] = torch.mean(torch.tensor(tot_loss_grads))
            # self._actor_loss_grad_norm_mean[self._log_it_counter, 0] = torch.mean(torch.tensor(actor_loss_grads))

            self._tot_loss_std[self._log_it_counter, 0] = torch.std(torch.tensor(tloss))
            self._value_loss_std[self._log_it_counter, 0] = torch.std(torch.tensor(vlosses))
            self._policy_loss_std[self._log_it_counter, 0] = torch.std(torch.tensor(pglosses))
            self._entropy_loss_std[self._log_it_counter, 0] = torch.std(torch.tensor(eplosses))
            # self._tot_loss_grad_norm_std[self._log_it_counter, 0] = torch.std(torch.tensor(tot_loss_grads))
            # self._actor_loss_grad_norm_std[self._log_it_counter, 0] = torch.std(torch.tensor(actor_loss_grads))

            self._old_approx_kl_mean[self._log_it_counter, 0] = torch.mean(torch.tensor(old_approx_kls))
            self._approx_kl_mean[self._log_it_counter, 0] = torch.mean(torch.tensor(approx_kls))
            self._old_approx_kl_std[self._log_it_counter, 0] = torch.std(torch.tensor(old_approx_kls))
            self._approx_kl_std[self._log_it_counter, 0] = torch.std(torch.tensor(approx_kls))

            self._clipfrac_mean[self._log_it_counter, 0] = torch.mean(torch.tensor(clipfracs))
            self._clipfrac_std[self._log_it_counter, 0] = torch.std(torch.tensor(clipfracs))

            self._explained_variance[self._log_it_counter, 0] = explained_var

            self._batch_returns_std[self._log_it_counter, 0] = batched_returns.std().item() 
            self._batch_returns_mean[self._log_it_counter, 0] = batched_returns.mean().item()
            self._batch_adv_std[self._log_it_counter, 0] = batched_advantages.std().item() 
            self._batch_adv_mean[self._log_it_counter, 0] = batched_advantages.mean().item()
            self._batch_val_std[self._log_it_counter, 0] = batched_values.std().item() 
            self._batch_val_mean[self._log_it_counter, 0] = batched_values.mean().item()