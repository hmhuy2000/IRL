import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
import sys
import os
import wandb
import numpy as np
import torch.distributions as tdist
from itertools import chain

sys.path.append("..")
from base_algo import Algorithm
from buffer import RolloutBuffer_PPO_lag
from network import StateIndependentPolicy, StateFunction,StateActionFunction


def calculate_gae(values, rewards, dones, next_values, gamma, lambd):
    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # Initialize gae.
    gaes = torch.empty_like(rewards)

    # Calculate gae recursively from behind.
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)

def calculate_gae_cost(values, rewards, dones, next_values, gamma, lambd):
    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # Initialize gae.
    gaes = torch.empty_like(rewards)

    # Calculate gae recursively from behind.
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    return gaes + values, (gaes - gaes.mean())#/ (gaes.std() + 1e-8)

class PPO_continuous(Algorithm):
    def __init__(self, state_shape, action_shape, device, seed, gamma,
        buffer_size, mix, hidden_units_actor, hidden_units_critic,
        lr_actor, lr_critic,lr_cost_critic,lr_penalty, epoch_ppo, clip_eps, lambd, coef_ent, 
        max_grad_norm,reward_factor,max_episode_length,cost_limit,risk_level,
        num_envs):
        super().__init__(state_shape, action_shape, device, seed, gamma)

        # Rollout buffer.
        self.buffer = RolloutBuffer_PPO_lag(
            buffer_size=buffer_size,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            mix=mix
        )

        # Actor.
        self.actor = StateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=hidden_units_actor,
            hidden_activation=nn.ReLU()
        ).to(device)

        # Critic.
        self.critic = StateFunction(
            state_shape=state_shape,
            hidden_units=hidden_units_critic,
            hidden_activation=nn.ReLU()
        ).to(device)

        self.cost_critic = StateFunction(
            state_shape=state_shape,
            hidden_units=hidden_units_critic,
            hidden_activation=nn.ReLU()
        ).to(device)

        self.cost_critic_variance = StateFunction(
            state_shape=state_shape,
            hidden_units=hidden_units_critic,
            hidden_activation=nn.ReLU(),
            output_activation=nn.Softplus()
        ).to(device)

        self.penalty_network = StateActionFunction(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=hidden_units_critic,
            hidden_activation=nn.ReLU()
        ).to(device)

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)
        self.optim_cost_critic = Adam(self.cost_critic.parameters(), lr=lr_cost_critic)
        self.optim_cost_critic_variance = Adam(self.cost_critic_variance.parameters(), lr=lr_cost_critic)
        self.optim_penalty = Adam(self.penalty_network.parameters(), lr=lr_penalty)

        self.learning_steps_ppo = 0
        self.rollout_length = buffer_size
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm
        self.max_entropy = -1e9
        self.reward_factor = reward_factor
        self.env_length = []
        self.max_episode_length = max_episode_length
        self.rewards = []
        self.costs = []
        self.return_cost = []
        self.return_reward = []
        self.cost_limit = cost_limit
        self.num_envs = num_envs
        self.target_cost = (
            self.cost_limit * (1 - self.gamma**self.max_episode_length) / (1 - self.gamma) / self.max_episode_length
        )
        self.target_kl = 0.2
        self.risk_level = risk_level
        normal = tdist.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        self.pdf_cdf = (
            normal.log_prob(normal.icdf(torch.tensor(self.risk_level))).exp() / self.risk_level
        ) 
        print(f'pdf_cdf {self.pdf_cdf}')

        self.pdf_cdf = self.pdf_cdf.cuda()

        self.tmp_buffer = [[] for _ in range(self.num_envs)]
        self.tmp_return_cost = [0 for _ in range(self.num_envs)]
        self.tmp_return_reward = [0 for _ in range(self.num_envs)]
        self.total_cost = 0
        self.num_interact_step = 0
        print(f'target cost: {self.target_cost}')

    def is_update(self,step):
        return step % self.rollout_length == 0

    def step(self, env, state, t):
        t += 1
        action, log_pi = self.explore(state)
        next_state, reward, done, c  = env.step(action)
        for idx in range(self.num_envs):
            mask = False if t >= self.max_episode_length else done[idx]
            self.tmp_buffer[idx].append((state[idx], action[idx], reward[idx] * self.reward_factor,
            c[idx], mask, log_pi[idx], next_state[idx]))
        
            self.rewards.append(reward[idx] * self.reward_factor)
            self.costs.append(c[idx])
            self.tmp_return_cost[idx] += c[idx]
            self.tmp_return_reward[idx] += reward[idx]

            if (self.max_episode_length and t>=self.max_episode_length):
                done[idx] = True
            if done[idx]:
                for (tmp_state,tmp_action,tmp_reward,tmp_c,tmp_mask,tmp_log_pi,tmp_next_state) in self.tmp_buffer[idx]:
                    self.buffer.append(tmp_state, tmp_action, tmp_reward, tmp_c, tmp_mask, tmp_log_pi, tmp_next_state)
                self.tmp_buffer[idx] = []

                self.env_length.append(t)
                if (len(self.return_cost)>=100):
                    self.return_cost = self.return_cost[1:]
                    self.return_reward = self.return_reward[1:]
                self.return_cost.append(self.tmp_return_cost[idx])
                self.return_reward.append(self.tmp_return_reward[idx])
                self.total_cost += self.tmp_return_cost[idx]
                self.num_interact_step += self.max_episode_length
                self.tmp_return_cost[idx] = 0
                self.tmp_return_reward[idx] = 0

        if done[0]:
            next_state = env.reset()
            t = 0
        return next_state, t

    def update(self,log_info):
        self.learning_steps += 1
        states, actions, rewards, costs, dones, log_pis, next_states = \
            self.buffer.get()
        self.update_ppo(
            states, actions, rewards, costs, dones, log_pis, next_states,log_info)

    def update_ppo(self, states, actions, rewards,costs, dones, log_pis, next_states,log_info):
        with torch.no_grad():
            values = self.critic(states)
            cost_values = self.cost_critic(states)         
            cost_variances = self.cost_critic_variance(states)#.clamp(min=1e-8, max=1e8) 
            next_values = self.critic(next_states)
            next_cost_values = self.cost_critic(next_states) 
            next_cost_variances =self.cost_critic_variance(next_states)#.clamp(min=1e-8, max=1e8) 

        cvar = cost_values + self.pdf_cdf.cuda() * torch.sqrt(cost_variances)
        next_cvar = next_cost_values + self.pdf_cdf.cuda() * torch.sqrt(next_cost_variances)

        targets, gaes = calculate_gae(
            values, rewards, dones, next_values, self.gamma, self.lambd)
        cost_targets, cost_gaes = calculate_gae_cost(
            cost_values, costs, dones, next_cost_values, self.gamma, self.lambd)
        cost_cvar_targets, cost_cvar_gaes = calculate_gae_cost(
            cvar, costs, dones, next_cvar, self.gamma, self.lambd)

        target_cost_variances = (
            costs**2
            - cost_values**2
            + 2 * self.gamma * costs * next_cost_values
            + self.gamma**2 * next_cost_variances
            + self.gamma**2 * next_cost_values**2
        )
        target_cost_variances = target_cost_variances.detach().clamp(min=1e-8, max=1e8) 
        print('------------------------------------')
        print(cost_values.mean().item(),cost_targets.mean().item())
        print(cvar.mean().item(),cost_cvar_targets.mean().item())
        print(cost_variances.mean().item(),target_cost_variances.mean().item())
        print('------------------------------------')

        current_cost = cost_cvar_targets
        cost_deviation = self.target_cost - current_cost

        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            self.update_critic(states, targets, cost_targets,target_cost_variances, log_info)
        
        app_kl = 0.0
        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            if (app_kl<self.target_kl):
                app_kl = self.update_actor(states, actions, log_pis, gaes,cost_cvar_gaes, log_info)

        for _ in range(1):
            penalty = F.softplus(self.penalty_network(states=states,actions=actions))
            loss_penalty = (penalty*cost_deviation).mean()
            self.optim_penalty.zero_grad()
            loss_penalty.backward()
            nn.utils.clip_grad_norm_(self.penalty_network.parameters(), self.max_grad_norm)
            self.optim_penalty.step()

        log_info['loss/penalty_loss'] = loss_penalty.item()
        log_info['environment/total_cost_deviation'] = torch.mean(cost_deviation)
        log_info['environment/cost_variance'] = torch.mean(target_cost_variances)
        log_info['environment/CVaR_mean'] = torch.mean(cost_cvar_targets)
        log_info['environment/CVaR_max'] = torch.max(cost_cvar_targets)
        log_info['environment/CVaR_min'] = torch.min(cost_cvar_targets)
        log_info['environment/cost_rate'] = self.total_cost/self.num_interact_step
        # log_info['environment/cost_target_mean'] = torch.mean(cost_targets)
        # log_info['environment/cost_target_max'] = torch.max(cost_targets)
        # log_info['environment/cost_target_min'] = torch.min(cost_targets)
        # log_info['environment/rewards_mean'] = np.mean(self.rewards)
        # log_info['environment/rewards_max'] = np.max(self.rewards)
        # log_info['environment/rewards_min'] = np.min(self.rewards)
        # log_info['environment/cost_mean'] = np.mean(self.costs)
        # log_info['environment/cost_max'] = np.max(self.costs)
        # log_info['environment/cost_min'] = np.min(self.costs)
        log_info['environment/return_cost'] = np.mean(self.return_cost)
        log_info['environment/return_reward'] = np.mean(self.return_reward)
        self.rewards = []
        self.costs = []

        return log_info

    def update_critic(self, states, targets,cost_targets,target_cost_variances, log_info):
        loss_critic = (self.critic(states) - targets).pow_(2).mean()
        cost_variances = self.cost_critic_variance(states)#.clamp(min=1e-8, max=1e8) 
        loss_cost_critic_mean = (self.cost_critic(states) - cost_targets).pow_(2).mean()
        loss_cost_critic_variance = (
            cost_variances + target_cost_variances -  2 * torch.sqrt(cost_variances * target_cost_variances)
        ).mean()
        # loss_cost_critic_variance = (cost_variances - target_cost_variances).pow_(2).mean()
        loss_cost_critic = loss_cost_critic_mean 
        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        self.optim_cost_critic.zero_grad()
        loss_cost_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self.max_grad_norm)
        self.optim_cost_critic.step()

        self.optim_cost_critic_variance.zero_grad()
        loss_cost_critic_variance.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.cost_critic_variance.parameters(), self.max_grad_norm)
        self.optim_cost_critic_variance.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            var_1 = np.std((targets - self.critic(states)).cpu().detach().numpy())
            var_2 = np.std(targets.cpu().detach().numpy())
            # value_function = self.critic(states)

            log_info['loss/PPO-critic'] = loss_critic.item()
            log_info['loss/PPO-cost-critic'] = loss_cost_critic.item()
            # log_info['loss/PPO-cost-critic_mean'] = loss_cost_critic_mean.item()
            log_info['loss/PPO-cost-critic_variance'] = loss_cost_critic_variance.item()
            log_info['PPO-stats/residual_variance'] = var_1/var_2
            log_info['PPO-network/target_value_max'] = torch.max(targets).cpu().detach().numpy()
            log_info['PPO-network/target_value_min'] = torch.min(targets).cpu().detach().numpy()
            log_info['PPO-network/target_value_mean'] = torch.mean(targets).cpu().detach().numpy()
            log_info['PPO-network/target_cost_max'] = torch.max(cost_targets).cpu().detach().numpy()
            log_info['PPO-network/target_cost_min'] = torch.min(cost_targets).cpu().detach().numpy()
            log_info['PPO-network/target_cost_mean'] = torch.mean(cost_targets).cpu().detach().numpy()
            # log_info['PPO-network/value_max'] = torch.max(value_function).cpu().detach().numpy()
            # log_info['PPO-network/value_min'] = torch.min(value_function).cpu().detach().numpy()
            # log_info['PPO-network/value_mean'] = torch.mean(value_function).cpu().detach().numpy()

            # log_info['environment/env_length_mean'] = np.mean(self.env_length)
            # log_info['environment/env_length_max'] = np.max(self.env_length)
            # log_info['environment/env_length_min'] = np.min(self.env_length)

            self.env_length = []

    def update_actor(self, states, actions, log_pis_old, gaes, cost_gaes, log_info):

        log_pis = self.actor.evaluate_log_pi(states, actions)
        entropy = -log_pis.mean()
        self.max_entropy = max(self.max_entropy,entropy.item())
        approx_kl = (log_pis_old - log_pis).mean().item()
        ratios = (log_pis - log_pis_old).exp_()

        penalty = F.softplus(self.penalty_network(states=states,actions=actions)).detach().clamp(0,100)
        total_gae = gaes - penalty * cost_gaes
        total_gae = total_gae/(penalty+1)

        loss_actor1 = -ratios * total_gae
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * total_gae
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()

        loss_cost = ratios * cost_gaes
        loss_cost = (penalty*loss_cost).mean()

        

        final_actor_loss  = loss_actor - self.coef_ent * entropy 

        self.optim_actor.zero_grad()
        (final_actor_loss).backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        if approx_kl>=self.target_kl or self.learning_steps_ppo % self.epoch_ppo == 0:
            log_info['loss/PPO-actor'] = loss_actor.item()
            log_info['loss/PPO-actor-cost-loss'] = loss_cost.item()
            log_info['loss/PPO-actor-final-loss'] = final_actor_loss.item()
            log_info['PPO-stats/entropy'] = entropy.item()
            log_info['PPO-stats/KL'] = approx_kl
            log_info['PPO-stats/penalty_mean'] = penalty.mean()
            log_info['PPO-stats/penalty_min'] = penalty.min()
            log_info['PPO-stats/penalty_max'] = penalty.max()
            log_info['PPO-stats/relative_policy_entropy'] = entropy.item()/self.max_entropy
            log_info['PPO-network/advantage_max'] = torch.max(gaes).cpu().detach().numpy()
            log_info['PPO-network/advantage_min'] = torch.min(gaes).cpu().detach().numpy()
            log_info['PPO-network/advantage_mean'] = torch.mean(gaes).cpu().detach().numpy()

            log_info['PPO-network/advantage_cost_max'] = torch.max(cost_gaes).cpu().detach().numpy()
            log_info['PPO-network/advantage_cost_min'] = torch.min(cost_gaes).cpu().detach().numpy()
            log_info['PPO-network/advantage_cost_mean'] = torch.mean(cost_gaes).cpu().detach().numpy()
            
        return approx_kl

    def save_models(self,save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.actor.state_dict(), f'{save_dir}/actor.pth')

    def train(self):
        self.actor.train()
        self.critic.train()
        self.cost_critic.train()
        self.cost_critic_variance.train()
        self.penalty_network.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()
        self.cost_critic.eval()
        self.cost_critic_variance.eval()
        self.penalty_network.eval()

    def load_models(self,load_dir):
        if not os.path.exists(load_dir):
            raise
        self.actor.load_state_dict(torch.load(f'{load_dir}/actor.pth'))
        self.critic.load_state_dict(torch.load(f'{load_dir}/critic.pth'))

    def copyNetworksFrom(self,algo):
        self.actor.load_state_dict(algo.actor.state_dict())