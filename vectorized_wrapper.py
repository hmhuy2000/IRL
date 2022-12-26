import gym
import safety_gym

import numpy as np
from multiprocessing import Process, Pipe
import os
from tqdm import tqdm
import copy
import time
import matplotlib.pyplot as plt

def worker(remote, parent_remote, env):
  '''
  Worker function which interacts with the environment over the remove connection

  Args:
    remote (multiprocessing.Connection): Worker remote connection
    parent_remote (multiprocessing.Connection): MultiRunner remote connection
    env_fn (function): Creates a environment
    planner_fn (function): Creates the planner for the environment
  '''
  parent_remote.close()

  try:
    while True:
      cmd, data = remote.recv()
      if cmd == 'step':
        res = env.step(data)
        remote.send(res)
      elif cmd == 'reset':
        obs = env.reset()
        remote.send(obs)
      elif cmd == 'close':
        remote.close()
        break
      else:
        raise NotImplementedError
  except KeyboardInterrupt:
    print('MultiRunner worker: caught keyboard interrupt')

class VectorizedWrapper(object):
    def __init__(self, envs):
        self.waiting = False
        self.closed = False
        num_envs = len(envs)
        self.num_steps = envs[0].num_steps

        self.remotes, self.worker_remotes = zip(*[Pipe() for _ in range(num_envs)])
        self.processes = [Process(target=worker, args=(worker_remote, remote, env))
                    for (worker_remote, remote, env) in zip(self.worker_remotes, self.remotes, envs)]
        self.num_processes = len(self.processes)

        for process in self.processes:
            process.daemon = True
            process.start()
        for remote in self.worker_remotes:
            remote.close()

    def reset(self):
        '''
        Reset each environment.

        Returns:
        numpy.array: Observations
        '''
        for remote in self.remotes:
            remote.send(('reset', None))

        obs = [remote.recv() for remote in self.remotes]
        obs = np.stack(obs)
        return obs

    def reset_envs(self, env_nums):
        '''
        Resets the specified environments.

        Args:
        env_nums (list[int]): The environments to be reset

        Returns:
        numpy.array: Observations
        '''

        for env_num in env_nums:
            self.remotes[env_num].send(('reset', None))
        obs = [self.remotes[env_num].recv() for env_num in env_nums]
        obs = np.stack(obs)

        return obs

    def step(self, actions):
        '''
        Step the environments synchronously.

        Args:
        actions (numpy.array): Actions to take in each environment
        auto_reset (bool): Reset environments automatically after an episode ends
        '''
        self.stepAsync(actions)
        return self.stepWait()

    def stepAsync(self, actions):
        '''
        Step each environment in a async fashion.

        Args:
        actions (numpy.array): Actions to take in each environment
        auto_reset (bool): Reset environments automatically after an episode ends
        '''

        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def stepWait(self):
        '''
        Wait until each environment has completed its next step.
        '''
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False

        res = tuple(zip(*results))

        obs, rewards, dones, infos = res
        costs = []
        for info in infos:
            costs.append(info['cost'])
        costs = tuple(costs)

        obs = np.stack(obs)
        rewards = np.stack(rewards)
        dones = np.stack(dones).astype(np.float32)
        costs = np.stack(costs).astype(np.float32)
        return obs,rewards,dones,costs

if __name__=='__main__':
    env_name='Safexp-PointGoal1-v0'
    num_envs = 5
    env = [gym.make(env_name,) for _ in range(num_envs)]
    envs = VectorizedWrapper(env)
    envs.reset()
    actions = np.stack([env[0].action_space.sample() for _ in range(num_envs)])
    states,rewards,dones,costs = envs.step(actions)
    envs.reset()