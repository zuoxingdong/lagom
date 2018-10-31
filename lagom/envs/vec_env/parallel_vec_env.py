import numpy as np

from multiprocessing import Process  # easier than threading
from multiprocessing import Pipe  # faster than Queue

from .vec_env import VecEnv
from .utils import CloudpickleWrapper


def worker(master_conn, worker_conn, make_env):
    r"""Environment worker to do working for master and send back result via Pipe connection. 
    
    Args:
        master_conn (Connection): master connection terminal
        worker_conn (Connection): worker connection terminal
        make_env (function): an argument-free function to generate an environment. 
    """
    # Close forked master connection as it is not used here
    master_conn.close()
    
    env = make_env()
    
    while True:
        cmd, data = worker_conn.recv()
        
        if cmd == 'step':
            observation, reward, done, info = env.step(data)
            
            # If episode terminates, reset the environment and send back initial observation in info
            # because terminal observation is still useful, one might needs it to train value function
            if done:
                init_observation = env.reset()
                info['init_observation'] = init_observation
            
            worker_conn.send([observation, reward, done, info])
        elif cmd == 'reset':
            observation = env.reset()
            worker_conn.send(observation)
        elif cmd == 'render':
            img = env.render(mode='rgb_array')
            worker_conn.send(img)
        elif cmd == 'close':
            env.close()
            worker_conn.close()
            break
        elif cmd == 'T':
            worker_conn.send(env.T)
        elif cmd == 'max_episode_reward':
            worker_conn.send(env.max_episode_reward)
        elif cmd == 'reward_range':
            worker_conn.send(env.reward_range)
        elif cmd == 'get_spaces':
            worker_conn.send([env.observation_space, env.action_space])


class ParallelVecEnv(VecEnv):
    r"""A vectorized environment runs in parallel. Each sub-environment uses an individual Process.
    
    For each :meth:`step` and :meth:`reset`, the command is executed for each sub-environment
    all at once in parallel. 
    
    .. note::
    
        It is recommended to use this if the simulator is very computationally expensive. In this
        case, :class:`SerialVecEnv` would be too slow. However, if the simulator is very fast, one
        should use :class:`SerialVecEnv` instead. 
        
    Example::
        
        >>> from lagom.envs import make_envs, make_gym_env
        >>> list_make_env = make_envs(make_env=make_gym_env, env_id='CartPole-v1', num_env=3, init_seed=0)
        >>> env = ParallelVecEnv(list_make_env=list_make_env)
        >>> env
        <ParallelVecEnv: CartPole-v1, n: 3>
        
        >>> env.reset()
        [array([-0.04002427,  0.00464987, -0.01704236, -0.03673052]),
         array([ 0.00854682,  0.00830137, -0.03052506,  0.03439879]),
         array([0.00025361, 0.02915667, 0.01103413, 0.04977449])]
    
    """
    def __init__(self, list_make_env):
        r"""Initialize the vectorized environment. 
        
        Args:
            list_make_env (list): a list of functions to generate environments.
        """
        self.master_conns, self.worker_conns = zip(*[Pipe() for _ in range(len(list_make_env))])
        self.list_process = [Process(target=worker, 
                                     args=[master_conn, worker_conn, CloudpickleWrapper(make_env)], 
                                     daemon=True)
                             for master_conn, worker_conn, make_env 
                             in zip(self.master_conns, self.worker_conns, list_make_env)]
        
        [process.start() for process in self.list_process]
        
        [worker_conn.close() for worker_conn in self.worker_conns]
        
        # Obtain observation and action spaces from first environment (all envs are the same for this)
        self.master_conns[0].send(['get_spaces', None])
        observation_space, action_space = self.master_conns[0].recv()
        
        super().__init__(list_make_env=list_make_env, 
                         observation_space=observation_space, 
                         action_space=action_space)
        assert len(self.master_conns) == self.num_env
        
        self.waiting = False  # If True, then workers are still working
        
    def step_async(self, actions):
        assert len(actions) == self.num_env, f'expected length {self.num_env}, got {len(actions)}'
        
        for master_conn, action in zip(self.master_conns, actions):
            master_conn.send(['step', action])

        self.waiting = True
        
    def step_wait(self):
        # Note that different worker finishes the job differently, but list comprehension
        # automatically preserve the order. This order is very important, otherwise it is a BUG !
        results = [master_conn.recv() for master_conn in self.master_conns]
        self.waiting = False
        observations, rewards, dones, infos = zip(*results)
        
        return list(observations), list(rewards), list(dones), list(infos)  # zip produces tuples
    
    def reset(self):
        [master_conn.send(['reset', None]) for master_conn in self.master_conns]
        observations = [master_conn.recv() for master_conn in self.master_conns]
        
        return observations
    
    def get_images(self):
        [master_conn.send(['render', None]) for master_conn in self.master_conns]
        imgs = [master_conn.recv() for master_conn in self.master_conns]
        
        return imgs
    
    def close_extras(self):
        if self.closed:  # all environments already closed
            return None
        
        # Waiting to receive data from all the workers if they are still working
        if self.waiting:
            [master_conn.recv() for master_conn in self.master_conns]
        
        [master_conn.send(['close', None]) for master_conn in self.master_conns]
        
        [master_conn.close() for master_conn in self.master_conns]
        
        [process.join() for process in self.list_process]
        
        self.closed = True

    @property
    def T(self):
        self.master_conns[0].send(['T', None])
        out = self.master_conns[0].recv()
        
        return out
    
    @property
    def max_episode_reward(self):
        self.master_conns[0].send(['max_episode_reward', None])
        out = self.master_conns[0].recv()
        
        return out
        
    @property
    def reward_range(self):
        self.master_conns[0].send(['reward_range', None])
        out = self.master_conns[0].recv()
        
        return out
