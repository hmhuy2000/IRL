from base_algo import sample_Algorithm
# from PPO.PPO_continuous import *
from PPO.PPO_discrete import *
import gym
from tqdm import trange
import matplotlib.pyplot as plt
from buffer import Buffer

from PPO.parameter import *

def main_PPO():
    test_env = gym.make(env_name,
    # render_mode="human",
    )
    state_shape=test_env.observation_space.shape
    action_shape=test_env.action_space.shape
    if ('Box' in f'{type(test_env.action_space)}'):
        if (len(action_shape) == 0):
            action_shape = (1,)
    else:
        # TODO: change action space, only for CartPole
        action_shape = (2,)

    algo = PPO_discrete(state_shape=state_shape, action_shape=action_shape,
            device=device, seed=seed, gamma=gamma,buffer_size=buffer_size,
            mix=mix, hidden_units_actor=hidden_units_actor,
            hidden_units_critic=hidden_units_critic,
            lr_actor=lr_actor,lr_critic=lr_critic, epoch_ppo=epoch_ppo,
            clip_eps=clip_eps, lambd=lambd, coef_ent=coef_ent,
            max_grad_norm=max_grad_norm,reward_factor=reward_factor,max_episode_length=max_episode_length)

    if not os.path.exists(weight_path):
        print('error no path weight existed!')
        raise

    if not os.path.exists(buffer_dir):
        os.makedirs(buffer_dir)
    if not os.path.exists(f'{buffer_dir}/{env_name}'):
        os.makedirs(f'{buffer_dir}/{env_name}')

    buffer = Buffer(
        buffer_size=buffer_size,
        state_shape=state_shape,
        action_shape=action_shape,
        device=device
    )

    algo.load_models(load_dir=load_dir)
    algo.eval()
    state, info = test_env.reset(seed=seed)
    t = 0
    returns = []
    total = 0
    for step in trange(1,buffer_size+1):
        t += 1
        action = algo.exploit(state)
        next_state, reward, done, truncated, info  = test_env.step(action)
        total += reward
        mask = False if t == test_env._max_episode_steps else done
        #TODO: problem with discrete
        # buffer.append(state, action, reward, mask, next_state)
        buffer.append(state, np.asarray([action]), reward, mask, next_state)

        if (t>=max_episode_length):
            done = True
        if done:
            t = 0
            returns.append(total)
            total = 0
            if (len(returns)>0 and len(returns)%100 == 0):
                print(f'mean returns = {np.mean(returns):.2f}')
            next_state, info = test_env.reset()
        state = next_state

    print(f'final mean returns = {np.mean(returns):.2f} with std = {np.std(returns):.2f}')
    print(f'final max returns = {np.max(returns):.2f} with min returns = {np.min(returns):.2f}')
    print(f'number of expert episodes: {len(returns)}')
    test_env.close()
    buffer.save(f'{buffer_dir}/{env_name}/{buffer_size}.pth')

if __name__ == '__main__':
    main_PPO()