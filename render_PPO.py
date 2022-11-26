from base_algo import sample_Algorithm
from PPO.PPO_continuous import *
# from PPO.PPO_discrete import *
import gym
import safety_gym
from tqdm import trange
import matplotlib.pyplot as plt
from buffer import Buffer

from PPO.parameter import *

def main_PPO():

    """
    python render_PPO.py --env_name=Safexp-PointGoal1-v0 --buffer_size=100000 --seed=0 --hidden_units_actor=128 --hidden_units_critic=128
     --max_episode_length=1000 --load_dir='./weights/PPO/Safexp-PointGoal1-v0-24.87'
    """

    test_env = gym.make(env_name,
    # render_mode="human",
    )
    state_shape=test_env.observation_space.shape
    action_shape=test_env.action_space.shape

    algo = PPO_continuous(state_shape=state_shape, action_shape=action_shape,
        device=device, seed=seed, gamma=gamma,buffer_size=buffer_size,
        mix=mix, hidden_units_actor=hidden_units_actor,
        hidden_units_critic=hidden_units_critic,
        lr_actor=lr_actor,lr_critic=lr_critic, epoch_ppo=epoch_ppo,
        clip_eps=clip_eps, lambd=lambd, coef_ent=coef_ent,
        max_grad_norm=max_grad_norm,reward_factor=reward_factor,max_episode_length=test_env.num_steps)

    if not os.path.exists(weight_path):
        print('error no path weight existed!')
        raise

    algo.load_models(load_dir=load_dir)
    algo.eval()
    state = test_env.reset()

    t = 0
    returns = []
    total = 0
    for step in trange(1,10000+1):
        test_env.render()
        t += 1
        action = algo.exploit(state)
        next_state, reward, done, info  = test_env.step(action)
        total += reward
        mask = False if t == test_env.num_steps else done
        #TODO: problem with discrete
        # buffer.append(state, action, reward, mask, next_state)

        if (t>=test_env.num_steps):
            done = True
        if done:
            t = 0
            returns.append(total)
            total = 0
            if (len(returns)>0 and len(returns)%100 == 0):
                print(f'mean returns = {np.mean(returns):.2f}')
            next_state = test_env.reset()
        state = next_state

    test_env.close()
    
if __name__ == '__main__':
    main_PPO()