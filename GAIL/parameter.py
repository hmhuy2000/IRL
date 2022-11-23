import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser()

training_group = parser.add_argument_group('training')
training_group.add_argument('--env_name',type=str,default='CartPole-v1')
training_group.add_argument('--gamma', type=float, default=0.995)
training_group.add_argument('--device_name', type=str, default='cuda')
training_group.add_argument('--seed', type=int, default=1)
training_group.add_argument('--buffer_size',type=int,default=50000)
training_group.add_argument('--mix',type=int,default=1)
training_group.add_argument('--hidden_units_actor',type=int,default=64)
training_group.add_argument('--hidden_units_critic',type=int,default=64)
training_group.add_argument('--hidden_units_disc',type=int,default=100)
training_group.add_argument('--lr_actor', type=float, default=1e-2)
training_group.add_argument('--lr_critic', type=float, default=1e-2)
training_group.add_argument('--epoch_ppo',type=int,default=50)
training_group.add_argument('--clip_eps', type=float, default=0.2)
training_group.add_argument('--lambd', type=float, default=0.97)
training_group.add_argument('--coef_ent', type=float, default=0.0)
training_group.add_argument('--max_grad_norm', type=float, default=10.0)
training_group.add_argument('--wandb_logs', type=bool, default=False)
training_group.add_argument('--num_training_step',type=int,default=int(1e7))
training_group.add_argument('--eval_interval',type=int,default=int(1e5))
training_group.add_argument('--num_eval_episodes',type=int,default=10)
training_group.add_argument('--max_episode_length',type=int,default=500)
training_group.add_argument('--reward_factor',type=float,default=1.0)
training_group.add_argument('--weight_path', type=str, default='./weights/GAIL')
training_group.add_argument('--load_dir', type=str, default='')
training_group.add_argument('--buffer_dir', type=str, default='')
training_group.add_argument('--epoch_disc',type=int,default=10)
training_group.add_argument('--lr_disc', type=float, default=1e-2)
training_group.add_argument('--batch_size',type=int,default=64)


#-------------------------------------------------------------------------------------------------#

# training
args = parser.parse_args()
gamma                                   = args.gamma
device                                  = torch.device(args.device_name)
seed                                    = args.seed
buffer_size                             = args.buffer_size
mix                                     = args.mix

hidden_units_actor                      = []
hidden_units_critic                     = []
hidden_units_disc                       = []
for _ in range(2):
    hidden_units_actor.append(args.hidden_units_actor)
    hidden_units_critic.append(args.hidden_units_critic)
    hidden_units_disc.append(args.hidden_units_disc)

lr_actor                                = args.lr_actor
lr_critic                               = args.lr_critic
epoch_ppo                               = args.epoch_ppo
clip_eps                                = args.clip_eps
lambd                                   = args.lambd
coef_ent                                = args.coef_ent
max_grad_norm                           = args.max_grad_norm
wandb_logs                              = args.wandb_logs
num_training_step                       = args.num_training_step
eval_interval                           = args.eval_interval
num_eval_episodes                       = args.num_eval_episodes
env_name                                = args.env_name
reward_factor                           = args.reward_factor
max_episode_length                      = args.max_episode_length
weight_path                             = args.weight_path
load_dir                                = args.load_dir
buffer_dir                              = args.buffer_dir
epoch_disc                              = args.epoch_disc
lr_disc                                 = args.lr_disc
batch_size                              = args.batch_size