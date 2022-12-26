from PPO.PPO_lag_continuous import *

import gym
import safety_gym
import sys
from tqdm import trange,tqdm
import wandb
import matplotlib.pyplot as plt
from copy import deepcopy
import threading
import torch.multiprocessing as mp

from PPO.parameter import *
from vectorized_wrapper import VectorizedWrapper

max_eval_return = -np.inf


def Wandb_logging(diction, step_idx, wandb_logs):
    if (wandb_logs):
        wandb.log(diction, step = step_idx)
    print(f'[INFO] {diction} step {step_idx}')


def evaluate(algo, env,max_episode_length,log_cnt):
    print('starting evaluation...')
    global max_eval_return
    mean_return = 0.0
    mean_cost = 0.0
    log_info = {}

    for _ in trange(num_eval_episodes//num_envs_test):
        state = env.reset()
        episode_return = 0.0
        episode_cost = 0.0
        for _ in range(max_episode_length):
            action = algo.exploit(state)
            state, reward, done, cost = env.step(action)
            episode_return += np.sum(reward)
            episode_cost += np.sum(cost)
            
        mean_return += episode_return 
        mean_cost += episode_cost 

    mean_return = mean_return/num_eval_episodes
    mean_cost = mean_cost/num_eval_episodes
    log_info['validation/return'] = mean_return
    log_info['validation/cost'] = mean_cost
    Wandb_logging(diction=log_info,step_idx=log_cnt,wandb_logs=wandb_logs)
    if (mean_return > max_eval_return and mean_cost <= cost_limit):
        max_eval_return = mean_return
        algo.save_models(f'{weight_path}/{env_name}-{mean_return:.2f}')
    elif (mean_cost > cost_limit):
        print(f'violated cost limit({cost_limit:.2f}): {mean_cost:.2f} !')
    print(f'evaluated = {mean_return:.2f}, cost = {mean_cost:.2f}, maximum return = {max_eval_return:.2f}')

def train(algo,eval_algo,rank,local_num_training_step):
    env = [gym.make(env_name,) for _ in range(num_envs)]
    env = VectorizedWrapper(env)

    if (rank == 0):
        if (wandb_logs):
            print('---------------------using Wandb---------------------')
            wandb.init(project=env_name, settings=wandb.Settings(_disable_stats=True), \
            group='PPO-lag-vectorized-multi', name=f'{seed}', entity='hmhuy')
        else:
            print('----------------------no Wandb-----------------------')
        eval_thread = None
        test_env = [gym.make(env_name,) for _ in range(num_envs_test)]
        test_env = VectorizedWrapper(test_env)
        range_train = trange(1,local_num_training_step//num_envs+1)
    else:
        range_train = range(1,local_num_training_step//num_envs+1)

    state = env.reset()
    t = 0
    log_cnt = 0
    for step in range_train:
        state, t = algo.step(env, state, t)
        log_info = {}

        if algo.is_update(step*num_envs):
            algo.update(log_info)
        if (rank == 0):
            if step % (eval_interval//num_envs) == 0:
                if eval_thread is not None:
                    eval_thread.join()
                eval_algo.copyNetworksFrom(algo)
                eval_algo.eval()
                eval_thread = threading.Thread(target=evaluate, 
                args=(eval_algo,test_env,env.num_steps,log_cnt))
                eval_thread.start()
            if (len(log_info.keys())>0):
                Wandb_logging(diction=log_info, step_idx=log_cnt,wandb_logs=wandb_logs)
                log_cnt += 1

    algo.save_models(f'{weight_path}/{env_name}-finish')
    env.close()
    if (rank==0):
        test_env.close()

        

def main_PPO():
    sample_env = gym.make(env_name,
    # render_mode="human",
    )


    state_shape=sample_env.observation_space.shape
    action_shape=sample_env.action_space.shape


    algo = PPO_continuous(state_shape=state_shape, action_shape=action_shape,
            device=device, seed=seed, gamma=gamma,buffer_size=buffer_size,
            mix=mix, hidden_units_actor=hidden_units_actor,
            hidden_units_critic=hidden_units_critic,
            lr_actor=lr_actor,lr_critic=lr_critic,lr_cost_critic=lr_cost_critic,lr_penalty=lr_penalty, epoch_ppo=epoch_ppo,
            clip_eps=clip_eps, lambd=lambd, coef_ent=coef_ent,
            max_grad_norm=max_grad_norm,reward_factor=reward_factor,max_episode_length=sample_env.num_steps,
            cost_limit=cost_limit,num_envs=num_envs)
    eval_algo = deepcopy(algo)

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    mp.set_start_method('spawn')
    algo.share_memory()
    processes = []
    for rank in range(num_procs):
        p = mp.Process(target=train, args=(algo,eval_algo,rank,num_training_step//num_procs))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # train(env=env,algo=algo,worker_id=0,local_num_training_step=num_training_step//num_procs)

    return

    for step in trange(1,num_training_step//num_envs+1):
        state, t = algo.step(env, state, t)
        log_info = {}

        if algo.is_update(step*num_envs):
                algo.update(log_info)
                
        if step % (eval_interval//num_envs) == 0:
            if eval_thread is not None:
                eval_thread.join()
            eval_algo.copyNetworksFrom(algo)
            eval_algo.eval()
            eval_thread = threading.Thread(target=evaluate, 
            args=(eval_algo,test_env,sample_env.num_steps,log_cnt))
            eval_thread.start()

        if (len(log_info.keys())>0):
            Wandb_logging(diction=log_info, step_idx=log_cnt,wandb_logs=wandb_logs)
            log_cnt += 1
    algo.save_models(f'{weight_path}/{env_name}-finish')
    env.close()
    test_env.close()

if __name__ == '__main__':
    main_PPO()