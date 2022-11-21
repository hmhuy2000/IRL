from base_algo import sample_Algorithm
import gym
import sys
from tqdm import trange
import wandb
import matplotlib.pyplot as plt
import os

from AIRL.parameter import *
# from AIRL.airl_continuous import AIRL_continuous
from AIRL.airl_discrete import AIRL_discrete
from buffer import SerializedBuffer

def Wandb_logging(diction, step_idx, wandb_logs):
    if (wandb_logs):
        wandb.log(diction, step = step_idx)
    else:
        print(f'[INFO] {diction} step {step_idx}')

def evaluate(algo, env,log_info):
    mean_return = 0.0

    for _ in range(num_eval_episodes):
        state, info = env.reset()
        episode_return = 0.0
        done = False
        while (not done):
            action = algo.exploit(state)
            state, reward, done, truncated, info = env.step(action)
            episode_return += reward
            if (truncated):
                break
        mean_return += episode_return / num_eval_episodes
    log_info['validation/return'] = mean_return
    return mean_return

def main():
    env = gym.make(env_name,
    # render_mode="human",
    )
    test_env = gym.make(env_name,
    # render_mode="human",
    )

    state_shape=env.observation_space.shape
    action_shape=env.action_space.shape
    if ('Box' in f'{type(env.action_space)}'):
        if (len(action_shape) == 0):
            action_shape = (1,)
    else:
        # TODO: change action space, only for CartPole
        action_shape = (2,)

    if (wandb_logs):
        print('---------------------using Wandb---------------------')
        wandb.init(project=env_name, settings=wandb.Settings(_disable_stats=True), \
        group='AIRL', name=f'{seed}', entity='hmhuy')
    else:
        print('----------------------no Wandb-----------------------')

    exp_buffer = SerializedBuffer(
        path=buffer_dir,
        device=device
    )
    algo = AIRL_discrete(exp_buffer=exp_buffer, state_shape=state_shape, action_shape=action_shape,
            device=device, seed=seed, gamma=gamma,buffer_size=buffer_size,mix=mix,
            hidden_units_actor=hidden_units_actor,hidden_units_critic=hidden_units_critic,
            lr_actor=lr_actor,lr_critic=lr_critic, epoch_ppo=epoch_ppo,clip_eps=clip_eps, lambd=lambd, coef_ent=coef_ent,
            max_grad_norm=max_grad_norm,reward_factor=reward_factor,max_episode_length=max_episode_length,
            units_disc_r=hidden_units_disc_r,units_disc_v=hidden_units_disc_v,epoch_disc=epoch_disc,lr_disc=lr_disc,
            batch_size=batch_size)
    
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    
    state, info = env.reset(seed=seed)
    t = 0
    log_cnt = 0
    eval_return = -np.inf
    for step in trange(1,num_training_step+1):
        state, t = algo.step(env, state, t)
        log_info = {}
        if algo.is_update(step):
                algo.update(log_info)
                
        if step % eval_interval == 0:
            value = evaluate(algo,test_env,log_info=log_info)
            if (value > eval_return):
                eval_return = value
                algo.save_models(f'{weight_path}/{env_name}-{eval_return:.2f}')

        if (len(log_info.keys())>0):
            Wandb_logging(diction=log_info, step_idx=log_cnt,wandb_logs=wandb_logs)
            log_cnt += 1
    algo.save_models(f'{weight_path}/{env_name}-finish')
    env.close()
    test_env.close()

if __name__ == '__main__':
    main()