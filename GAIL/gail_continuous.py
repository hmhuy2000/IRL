import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import sys
sys.path.append("..")

from PPO.PPO_continuous import PPO_continuous
from network import GAIL_disc


class GAIL_continuous(PPO_continuous):
    def __init__(self, exp_buffer,  state_shape, action_shape, device, seed, gamma,
        buffer_size, mix, hidden_units_actor, hidden_units_critic,
        lr_actor, lr_critic, epoch_ppo, clip_eps, lambd, coef_ent, 
        max_grad_norm,reward_factor,max_episode_length,
        units_disc, epoch_disc, lr_disc, batch_size):
        super().__init__(state_shape, action_shape, device, seed, gamma,
        buffer_size, mix, hidden_units_actor, hidden_units_critic,
        lr_actor, lr_critic, epoch_ppo, clip_eps, lambd, coef_ent, 
        max_grad_norm,reward_factor,max_episode_length)

        self.exp_buffer = exp_buffer
        self.disc = GAIL_disc(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_disc,
            hidden_activation=nn.ReLU()
        ).to(device)
        self.learning_steps_disc = 0
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc

    def update(self,log_info):
        self.learning_steps += 1
        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1
            states, actions = self.buffer.sample(self.batch_size)[:2]
            exp_states, exp_actions = \
                self.exp_buffer.sample(self.batch_size)[:2]
            self.update_disc(states, actions, exp_states, exp_actions, log_info)

        states, actions, _, dones, log_pis, next_states = self.buffer.get()
        rewards = self.disc.calculate_reward(states, actions)
        self.update_ppo(
            states, actions, rewards, dones, log_pis, next_states, log_info)
    
    def update_disc(self, states, actions, exp_states, exp_actions, log_info):
        logits_pi = self.disc(states, actions)
        exp_logits = self.disc(exp_states, exp_actions)

        loss_pi = -F.logsigmoid(-logits_pi).mean()
        exp_loss = -F.logsigmoid(exp_logits).mean()
        loss_disc = loss_pi + exp_loss

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()

        if self.learning_steps_disc % self.epoch_disc == 0:

            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                exp_acc = (exp_logits > 0).float().mean().item()
            
            log_info['disc/acc_pi'] = acc_pi
            log_info['disc/exp_acc'] = exp_acc
            log_info['loss/discriminator'] = loss_disc.item()
            