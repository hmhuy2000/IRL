import torch
import os
import time

from wcsac import utils
import hydra
from wcsac.video import VideoRecorder
from tqdm import tqdm, trange
import numpy as np
from buffer import RolloutBufferwithCost
import wandb

def Wandb_logging(diction, step_idx, wandb_logs):
    if (wandb_logs):
        wandb.log(diction, step = step_idx)
    else:
        print(f'[INFO] {diction} step {step_idx}')

def evaluate(cfg,env,agent,video_recorder,work_dir,log_info,step):
    mean_reward = 0
    mean_cost = 0
    mean_goals_met = 0
    mean_hazard_touches = 0
    cost_limit_violations = 0
    for episode in range(cfg.num_eval_episodes):
        obs = env.reset()
        agent.reset()
        video_recorder.init(enabled=(episode == 0))
        done = False
        ep_reward = 0
        ep_cost = 0
        ep_goals_met = 0
        ep_hazard_touches = 0

        while not done:
            with utils.eval_mode(agent):
                action = agent.act(obs, sample=False)
            obs, reward, done, info = env.step(action)
            video_recorder.record(env)
            ep_reward += reward
            ep_cost += info.get("cost", 0)
            ep_goals_met += 1 if info.get("goal_met", False) else 0
            ep_hazard_touches += 1 if (info.get("cost_hazards", 0) > 0) else 0

        mean_reward += ep_reward
        mean_cost += ep_cost
        mean_goals_met += ep_goals_met
        mean_hazard_touches += ep_hazard_touches
        cost_limit_violations += 1 if (ep_cost > cfg.agent.params.cost_limit) else 0
        video_recorder.save(f"{step}.mp4")

    mean_reward /= cfg.num_eval_episodes
    mean_cost /= cfg.num_eval_episodes
    mean_goals_met /= cfg.num_eval_episodes
    mean_hazard_touches /= cfg.num_eval_episodes
    log_info['eval/mean_reward'] = mean_reward
    log_info['eval/mean_cost'] = mean_cost
    log_info['eval/mean_goals_met'] = mean_goals_met
    log_info['eval/hazard_touches'] = mean_hazard_touches
    log_info['eval/cost_limit_violations'] = cost_limit_violations
    agent.save(work_dir)
    agent.save_actor(os.path.join(work_dir, "model_weights"), step)

@hydra.main(config_path="./wcsac/config/train.yaml", strict=True)
def main(cfg):

    work_dir = os.getcwd()
    assert 1 >= cfg.risk_level >= 0, f"risk_level must be between 0 and 1 (inclusive), got: {cfg.risk_level}"
    assert cfg.seed != -1, f"seed must be provided, got default seed: {cfg.seed}"
    if (cfg.wandb_logs):
        print('---------------------using Wandb---------------------')
        wandb.init(project=cfg.env, settings=wandb.Settings(_disable_stats=True), \
        group='WCSAC', name=f'{cfg.seed}', entity='hmhuy')
    else:
        print('----------------------no Wandb-----------------------')
    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)
    env = utils.make_safety_env(cfg)
    eval_env = utils.make_safety_env(cfg)
    cfg.agent.params.obs_dim = int(env.observation_space.shape[0])
    cfg.agent.params.action_dim = int(env.action_space.shape[0])
    cfg.replay_buffer_capacity = int(cfg.replay_buffer_capacity)
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape

    cfg.agent.params.action_range = [
        float(env.action_space.low.min()),
        float(env.action_space.high.max()),
    ]
    agent = hydra.utils.instantiate(cfg.agent)
    utils.make_dir(work_dir, "model_weights")

    video_recorder = VideoRecorder(work_dir if cfg.save_video else None)

    replay_buffer = RolloutBufferwithCost(
        buffer_size=cfg.replay_buffer_capacity,
        state_shape=state_shape,
        action_shape=action_shape,
        device=device,
        mix=1
    )

    #--------------------------------------------------------------#
    cfg.num_train_steps = int(cfg.num_train_steps)
    episode, ep_reward, ep_cost, total_cost, done = 0, 0, 0, 0, True
    returns = []
    costs = []
    log_cnt = 0
    for step in trange(cfg.num_train_steps):
        if done:
            returns.append(ep_reward)
            costs.append(ep_cost)
            # pbar.set_description(f'return = {np.mean(returns):.2f}, cost = {np.mean(costs):.2f}')
            
            obs = env.reset()
            agent.reset()
            done = False
            ep_reward = 0
            ep_cost = 0
            ep_step = 0
            episode += 1
        log_info = {}
        if step < cfg.num_seed_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.act(obs, sample=True)
        if step >= cfg.num_seed_steps:
            agent.update(replay_buffer, log_info, step)
        next_obs, reward, done, info = env.step(action)
        cost = info.get("cost", 0)
        done = float(done)
        mask = 0 if ep_step + 1 == env.num_steps else done
        ep_reward += reward
        ep_cost += cost
        total_cost += cost
        replay_buffer.append(obs, action, reward, cost,mask, next_obs)
        obs = next_obs
        ep_step += 1
        if (step>0 and step%cfg.eval_frequency==0):
            evaluate(cfg,eval_env,agent,video_recorder,work_dir,log_info,step)
            agent.save(work_dir)
        if (len(log_info.keys())>0):
            log_info['env/return'] = np.mean(returns[-10:])
            log_info['env/cost'] = np.mean(costs[-10:])
            Wandb_logging(diction=log_info, step_idx=log_cnt,wandb_logs=cfg.wandb_logs)
            log_cnt += 1
        


if __name__ == '__main__':
    main()