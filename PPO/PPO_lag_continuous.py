import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
import sys
import os
import wandb
import numpy as np

sys.path.append("..")
from base_algo import Algorithm
from buffer import RolloutBuffer_PPO_lag
from network import StateIndependentPolicy, StateFunction


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

    return gaes + values, (gaes - gaes.mean()) #/ (gaes.std() + 1e-8)

class PPO_continuous(Algorithm):
    def __init__(self, state_shape, action_shape, device, seed, gamma,
        buffer_size, mix, hidden_units_actor, hidden_units_critic,
        lr_actor, lr_critic,lr_cost_critic,lr_penalty, epoch_ppo, clip_eps, lambd, coef_ent, 
        max_grad_norm,reward_factor,max_episode_length,cost_limit):
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
        self.penalty_param = torch.tensor(1.0,requires_grad=True).float()

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)
        self.optim_cost_critic = Adam(self.cost_critic.parameters(), lr=lr_cost_critic)
        self.optim_penalty = Adam([self.penalty_param], lr=lr_penalty)

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
        self.return_cost = [0]
        self.cost_limit = cost_limit

    def is_update(self,step):
        return step % self.rollout_length == 0

    def get_penalty(self):
        return F.softplus(self.penalty_param)

    def step(self, env, state, t):
        t += 1
        action, log_pi = self.explore(state)
        next_state, reward, done, info  = env.step(action)
        c = info['cost']
        self.return_cost[-1] += c
        mask = False if t >= self.max_episode_length else done
        self.buffer.append(state, action, reward * self.reward_factor, c, mask, log_pi, next_state)
        self.rewards.append(reward * self.reward_factor)
        self.costs.append(c)
        if (self.max_episode_length and t>=self.max_episode_length):
            done = True
        if done:
            self.env_length.append(t)
            t = 0
            next_state = env.reset()
            if (len(self.return_cost)>=100):
                self.return_cost = self.return_cost[1:]
            self.return_cost.append(0)
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
            next_values = self.critic(next_states)
            next_cost_values = self.cost_critic(next_states)            
        targets, gaes = calculate_gae(
            values, rewards, dones, next_values, self.gamma, self.lambd)
        cost_targets, cost_gaes = calculate_gae_cost(
            cost_values, costs, dones, next_cost_values, self.gamma, self.lambd)
        current_cost = np.mean(self.return_cost)
        # penalty_param = self.get_penalty()
        cost_deviation = (current_cost - self.cost_limit)
        loss_penalty = -self.penalty_param*cost_deviation
        self.optim_penalty.zero_grad()
        loss_penalty.backward()
        self.optim_penalty.step()

        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            self.update_critic(states, targets, cost_targets, log_info)
            self.update_actor(states, actions, log_pis, gaes,cost_gaes, log_info)

        log_info['loss/penalty_loss'] = loss_penalty.item()
        log_info['PPO-stats/penalty_param'] = self.penalty_param.item()
        log_info['environment/rewards_mean'] = np.mean(self.rewards)
        log_info['environment/rewards_max'] = np.max(self.rewards)
        log_info['environment/rewards_min'] = np.min(self.rewards)
        log_info['environment/rewards_std'] = np.std(self.rewards)
        log_info['environment/cost_mean'] = np.mean(self.costs)
        log_info['environment/cost_max'] = np.max(self.costs)
        log_info['environment/cost_min'] = np.min(self.costs)
        log_info['environment/cost_std'] = np.std(self.costs)
        log_info['environment/return_cost'] = np.std(self.return_cost)
        self.rewards = []
        self.costs = []

        return log_info

    def update_critic(self, states, targets,cost_targets, log_info):
        loss_critic = (self.critic(states) - targets).pow_(2).mean()
        loss_cost_critic = (self.cost_critic(states) - cost_targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        self.optim_cost_critic.zero_grad()
        loss_cost_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self.max_grad_norm)
        self.optim_cost_critic.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            var_1 = np.std((targets - self.critic(states)).cpu().detach().numpy())
            var_2 = np.std(targets.cpu().detach().numpy())
            value_function = self.critic(states)
            cost_value_function = self.cost_critic(states)

            log_info['loss/PPO-critic'] = loss_critic.item()
            log_info['loss/PPO-cost-critic'] = loss_cost_critic.item()
            log_info['PPO-stats/residual_variance'] = var_1/var_2
            log_info['PPO-network/target_value_max'] = torch.max(targets).cpu().detach().numpy()
            log_info['PPO-network/target_value_min'] = torch.min(targets).cpu().detach().numpy()
            log_info['PPO-network/target_value_mean'] = torch.mean(targets).cpu().detach().numpy()
            log_info['PPO-network/target_value_std'] = torch.std(targets).cpu().detach().numpy()
            log_info['PPO-network/value_max'] = torch.max(value_function).cpu().detach().numpy()
            log_info['PPO-network/value_min'] = torch.min(value_function).cpu().detach().numpy()
            log_info['PPO-network/value_mean'] = torch.mean(value_function).cpu().detach().numpy()
            log_info['PPO-network/value_std'] = torch.std(value_function).cpu().detach().numpy()

            log_info['PPO-network/target_cost_max'] = torch.max(cost_targets).cpu().detach().numpy()
            log_info['PPO-network/target_cost_min'] = torch.min(cost_targets).cpu().detach().numpy()
            log_info['PPO-network/target_cost_mean'] = torch.mean(cost_targets).cpu().detach().numpy()
            log_info['PPO-network/target_cost_std'] = torch.std(cost_targets).cpu().detach().numpy()
            log_info['PPO-network/cost_max'] = torch.max(cost_value_function).cpu().detach().numpy()
            log_info['PPO-network/cost_min'] = torch.min(cost_value_function).cpu().detach().numpy()
            log_info['PPO-network/cost_mean'] = torch.mean(cost_value_function).cpu().detach().numpy()
            log_info['PPO-network/cost_std'] = torch.std(cost_value_function).cpu().detach().numpy()

            log_info['environment/env_length_mean'] = np.mean(self.env_length)
            log_info['environment/env_length_max'] = np.max(self.env_length)
            log_info['environment/env_length_min'] = np.min(self.env_length)
            log_info['environment/env_length_std'] = np.std(self.env_length)

            self.env_length = []

    def update_actor(self, states, actions, log_pis_old, gaes, cost_gaes, log_info):

        log_pis = self.actor.evaluate_log_pi(states, actions)
        entropy = -log_pis.mean()
        self.max_entropy = max(self.max_entropy,entropy.item())
        approx_kl = (log_pis_old - log_pis).mean().item()
        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * gaes
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()

        loss_cost = ratios * cost_gaes
        loss_cost = loss_cost.mean()

        penalty = self.get_penalty()

        final_actor_loss  = loss_actor - penalty*loss_cost
        final_actor_loss = final_actor_loss/(1+penalty)
        final_actor_loss = -final_actor_loss

        self.optim_actor.zero_grad()
        (loss_actor - self.coef_ent * entropy).backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            log_info['loss/PPO-actor'] = loss_actor.item()
            log_info['loss/PPO-actor-cost-loss'] = - penalty*loss_cost.item()
            log_info['loss/PPO-actor-final-loss'] = final_actor_loss.item()
            log_info['PPO-stats/entropy'] = entropy.item()
            log_info['PPO-stats/KL'] = approx_kl
            log_info['PPO-stats/relative_policy_entropy'] = entropy.item()/self.max_entropy
            log_info['PPO-network/advantage_max'] = torch.max(gaes).cpu().detach().numpy()
            log_info['PPO-network/advantage_min'] = torch.min(gaes).cpu().detach().numpy()
            log_info['PPO-network/advantage_mean'] = torch.mean(gaes).cpu().detach().numpy()
            log_info['PPO-network/advantage_std'] = torch.std(gaes).cpu().detach().numpy()

            log_info['PPO-network/advantage_cost_max'] = torch.max(cost_gaes).cpu().detach().numpy()
            log_info['PPO-network/advantage_cost_min'] = torch.min(cost_gaes).cpu().detach().numpy()
            log_info['PPO-network/advantage_cost_mean'] = torch.mean(cost_gaes).cpu().detach().numpy()
            log_info['PPO-network/advantage_cost_std'] = torch.std(cost_gaes).cpu().detach().numpy()
            
    def save_models(self,save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.actor.state_dict(), f'{save_dir}/actor.pth')
        torch.save(self.critic.state_dict(), f'{save_dir}/critic.pth')

    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def load_models(self,load_dir):
        if not os.path.exists(load_dir):
            raise
        self.actor.load_state_dict(torch.load(f'{load_dir}/actor.pth'))
        self.critic.load_state_dict(torch.load(f'{load_dir}/critic.pth'))
