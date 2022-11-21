import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from PPO.PPO_discrete import PPO_discrete
from network.disc import AIRLDiscrim


class AIRL_discrete(PPO_discrete):
    def __init__(self, exp_buffer,  state_shape, action_shape, device, seed, gamma,
                buffer_size, mix, hidden_units_actor, hidden_units_critic,
                lr_actor, lr_critic, epoch_ppo, clip_eps, lambd, coef_ent, 
                max_grad_norm,reward_factor,max_episode_length,
                units_disc_r, units_disc_v, epoch_disc, lr_disc, batch_size):
        super().__init__(state_shape, action_shape, device, seed, gamma,
        buffer_size, mix, hidden_units_actor, hidden_units_critic,
        lr_actor, lr_critic, epoch_ppo, clip_eps, lambd, coef_ent, 
        max_grad_norm,reward_factor,max_episode_length)

        self.exp_buffer = exp_buffer
        self.disc = AIRLDiscrim(
            state_shape=state_shape,
            gamma=gamma,
            hidden_units_r=units_disc_r,
            hidden_units_v=units_disc_v,
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True)
        ).to(device)

        self.learning_steps_disc = 0
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc

    def update(self, log_info):
        self.learning_steps += 1
        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1
            states, _, _, dones, log_pis, next_states = self.buffer.sample(self.batch_size)
            states_exp, actions_exp, _, dones_exp, next_states_exp = self.exp_buffer.sample(self.batch_size)
            with torch.no_grad():
                log_pis_exp = self.actor.evaluate_log_pi(
                    states_exp, actions_exp)

            self.update_disc(
                states, dones, log_pis, next_states, states_exp,
                dones_exp, log_pis_exp, next_states_exp, log_info
            )

        states, actions, _, dones, log_pis, next_states = self.buffer.get()
        rewards = self.disc.calculate_reward(states, dones, log_pis, next_states)
        self.update_ppo(states, actions, rewards, dones, log_pis, next_states, log_info)

    def update_disc(self, states, dones, log_pis, next_states,
                    states_exp, dones_exp, log_pis_exp,
                    next_states_exp, log_info):
        logits_pi = self.disc(states, dones, log_pis, next_states)
        logits_exp = self.disc(states_exp, dones_exp, log_pis_exp, next_states_exp)

        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()

        if self.learning_steps_disc % self.epoch_disc == 0:

            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                exp_acc = (logits_exp > 0).float().mean().item()
            
            log_info['disc/acc_pi'] = acc_pi
            log_info['disc/exp_acc'] = exp_acc
            log_info['loss/discriminator'] = loss_disc.item()