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
from buffer import RolloutBuffer
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


class PPO_continuous(Algorithm):
    def __init__(self, state_shape, action_shape, device, seed, gamma,
        buffer_size, mix, hidden_units_actor, hidden_units_critic,
        lr_actor, lr_critic, epoch_ppo, clip_eps, lambd, coef_ent, 
        max_grad_norm,reward_factor,max_episode_length):
        super().__init__(state_shape, action_shape, device, seed, gamma)

        # Rollout buffer.
        self.buffer = RolloutBuffer(
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

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)

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

    def is_update(self,step):
        return step % self.rollout_length == 0

    def step(self, env, state, t):
        t += 1
        action, log_pi = self.explore(state)
        next_state, reward, done, truncated, info  = env.step(action)
        mask = False if t == env._max_episode_steps else done
        self.buffer.append(state, action, reward * self.reward_factor, mask, log_pi, next_state)
        self.rewards.append(reward * self.reward_factor)
        if (self.max_episode_length and t>=self.max_episode_length):
            done = True
        if done:
            self.env_length.append(t)
            t = 0
            next_state, info = env.reset()
        return next_state, t

    def update(self,log_info):
        self.learning_steps += 1
        states, actions, rewards, dones, log_pis, next_states = \
            self.buffer.get()
        self.update_ppo(
            states, actions, rewards, dones, log_pis, next_states,log_info)

    def update_ppo(self, states, actions, rewards, dones, log_pis, next_states,log_info):
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
        targets, gaes = calculate_gae(
            values, rewards, dones, next_values, self.gamma, self.lambd)
        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            self.update_critic(states, targets, log_info)
            self.update_actor(states, actions, log_pis, gaes, log_info)
        
        log_info['environment/rewards_mean'] = np.mean(self.rewards)
        log_info['environment/rewards_max'] = np.max(self.rewards)
        log_info['environment/rewards_min'] = np.min(self.rewards)
        log_info['environment/rewards_std'] = np.std(self.rewards)
        self.rewards = []

        return log_info

    def update_critic(self, states, targets, log_info):
        loss_critic = (self.critic(states) - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            var_1 = np.std((targets - self.critic(states)).cpu().detach().numpy())
            var_2 = np.std(targets.cpu().detach().numpy())
            value_function = self.critic(states)

            log_info['loss/PPO-critic'] = loss_critic.item()
            log_info['PPO-stats/residual_variance'] = var_1/var_2
            log_info['PPO-network/target_value_max'] = torch.max(targets).cpu().detach().numpy()
            log_info['PPO-network/target_value_min'] = torch.min(targets).cpu().detach().numpy()
            log_info['PPO-network/target_value_mean'] = torch.mean(targets).cpu().detach().numpy()
            log_info['PPO-network/target_value_std'] = torch.std(targets).cpu().detach().numpy()
            log_info['PPO-network/value_max'] = torch.max(value_function).cpu().detach().numpy()
            log_info['PPO-network/value_min'] = torch.min(value_function).cpu().detach().numpy()
            log_info['PPO-network/value_mean'] = torch.mean(value_function).cpu().detach().numpy()
            log_info['PPO-network/value_std'] = torch.std(value_function).cpu().detach().numpy()
            log_info['environment/env_length_mean'] = np.mean(self.env_length)
            log_info['environment/env_length_max'] = np.max(self.env_length)
            log_info['environment/env_length_min'] = np.min(self.env_length)
            log_info['environment/env_length_std'] = np.std(self.env_length)

            self.env_length = []

    def update_actor(self, states, actions, log_pis_old, gaes,log_info):
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

        self.optim_actor.zero_grad()
        (loss_actor - self.coef_ent * entropy).backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            log_info['loss/PPO-actor'] = loss_actor.item()
            log_info['PPO-stats/entropy'] = entropy.item()
            log_info['PPO-stats/KL'] = approx_kl
            log_info['PPO-stats/relative_policy_entropy'] = entropy.item()/self.max_entropy
            log_info['PPO-network/advantage_max'] = torch.max(gaes).cpu().detach().numpy()
            log_info['PPO-network/advantage_min'] = torch.min(gaes).cpu().detach().numpy()
            log_info['PPO-network/advantage_mean'] = torch.mean(gaes).cpu().detach().numpy()
            log_info['PPO-network/advantage_std'] = torch.std(gaes).cpu().detach().numpy()
            
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
