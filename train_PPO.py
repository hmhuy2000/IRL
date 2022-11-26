from base_algo import sample_Algorithm
from PPO.PPO_continuous import *
# from PPO.PPO_discrete import *
import gym
import safety_gym
import sys
from tqdm import trange
import wandb
import matplotlib.pyplot as plt

from PPO.parameter import *


def Wandb_logging(diction, step_idx, wandb_logs):
    if (wandb_logs):
        wandb.log(diction, step = step_idx)
    else:
        print(f'[INFO] {diction} step {step_idx}')

def evaluate(algo, env, step_idx,log_info):
    mean_return = 0.0

    for _ in range(num_eval_episodes):
        state = env.reset()
        episode_return = 0.0
        done = False
        t = 0
        while (not done):
            t += 1
            action = algo.exploit(state)
            state, reward, done, info = env.step(action)
            episode_return += reward
            if (t >= env.num_steps):
                break
            
        mean_return += episode_return / num_eval_episodes
    log_info['validation/return'] = mean_return
    return mean_return

def main_PPO():
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
        group='PPO', name=f'{seed}', entity='hmhuy')
    else:
        print('----------------------no Wandb-----------------------')

    algo = PPO_continuous(state_shape=state_shape, action_shape=action_shape,
            device=device, seed=seed, gamma=gamma,buffer_size=buffer_size,
            mix=mix, hidden_units_actor=hidden_units_actor,
            hidden_units_critic=hidden_units_critic,
            lr_actor=lr_actor,lr_critic=lr_critic, epoch_ppo=epoch_ppo,
            clip_eps=clip_eps, lambd=lambd, coef_ent=coef_ent,
            max_grad_norm=max_grad_norm,reward_factor=reward_factor,max_episode_length=env.num_steps)
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
        
    state = env.reset()
    t = 0
    log_cnt = 0
    eval_return = -np.inf
    for step in trange(1,num_training_step+1):
        state, t = algo.step(env, state, t)
        log_info = {}

        if algo.is_update(step):
                algo.update(log_info)
                
        if step % eval_interval == 0:
            value = evaluate(algo,test_env,int(step/eval_interval),log_info=log_info)
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
    main_PPO()