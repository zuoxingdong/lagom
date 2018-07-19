import numpy as np

from multiprocessing import Process  # easier than threading
from multiprocessing import Pipe  # faster than Queue

from .vec_env import VecEnv
from .utils import CloudpickleWrapper


def worker(master_conn, worker_conn, make_env):
    # Close forked master connection as it is not used here
    # It does not affect the master connection in the main process
    master_conn.close()
    
    # Create the environment
    env = make_env()
    
    # Loop until receiving close command from master
    while True:
        # Receive master command
        cmd, data = worker_conn.recv()
        
        # Do the work according to the command
        if cmd == 'step':
            observation, reward, done, info = env.step(data)
            
            # If episode terminates, reset the environment and send back initial observation in info
            if done:
                init_observation = env.reset()
                info['init_observation'] = init_observation
                
            # Send information back to master
            worker_conn.send([observation, reward, done, info])
        elif cmd == 'reset':
            # Reset environment
            observation = env.reset()
            # Send back initial observation
            worker_conn.send(observation)
        elif cmd == 'render':
            # Render the environment
            img = env.render(mode='rgb_array')
            # Send back rendered RGB image
            worker_conn.send(img)
        elif cmd == 'close':
            # Close the environment
            env.close()
            # Close the worker connection
            worker_conn.close()
            # Break the while loop
            break
        elif cmd == 'seed':
            env.seed(data)
        elif cmd == 'T':
            worker_conn.send(env.T)
        elif cmd == 'get_spaces':
            result = [env.observation_space, env.action_space]
            # Send back spaces
            worker_conn.send(result)
            

class ParallelVecEnv(VecEnv):
    """
    Run vectorized environment in parallel. Each environment is running in an individual Process.
    
    Examples:
    
        def make_env():
            env = gym.make('CartPole-v0')

            return env

        env = ParallelVecEnv([make_env]*5)
        env.reset()
        for _ in range(100):
            env.step([0]*5)
    """
    def __init__(self, list_make_env):
        """
        Args:
            list_make_env (list): list of functions to generate an environment. 
        """
        # Create Pipe connections, each for one environment worker
        self.master_conns, self.worker_conns = zip(*[Pipe() for _ in range(len(list_make_env))])
        # Create processes, each for one environment worker
        self.list_process = [Process(target=worker, 
                                     args=[master_conn, worker_conn, CloudpickleWrapper(make_env)], 
                                     daemon=True)
                             for master_conn, worker_conn, make_env 
                             in zip(self.master_conns, self.worker_conns, list_make_env)]
        # Start all the processes
        [process.start() for process in self.list_process]
        
        # Close worker connections as they are not used here, the Processes already fork them
        [worker_conn.close() for worker_conn in self.worker_conns]
        
        # Obtain observation and action spaces from all environments
        self.master_conns[0].send(['get_spaces', None])
        observation_space, action_space =  self.master_conns[0].recv()
        
        # Call parent constructor
        super().__init__(list_make_env=list_make_env, 
                         observation_space=observation_space, 
                         action_space=action_space)
        
        # Some settings
        self.waiting = False  # If True, then workers are still working
        self.closed = False  # If True, then all processes already closed
        
    def step_async(self, actions):
        # Send 'step' and action to all environment workers
        [master_conn.send(['step', action]) for master_conn, action in zip(self.master_conns, actions)]
        # Set waiting flag
        self.waiting = True
        
    def step_wait(self):
        # Receive results from all workers
        results = [master_conn.recv() for master_conn in self.master_conns]
        # Turn off waiting flag
        self.waiting = False
        # Unpack results
        observations, rewards, dones, infos = zip(*results)
        
        return observations, rewards, dones, infos
    
    def reset(self):
        # Send 'reset' to all environment workers
        [master_conn.send(['reset', None]) for master_conn in self.master_conns]
        # Receive a list of initial observations from reset() in all environment workers
        observations = [master_conn.recv() for master_conn in self.master_conns]
        
        return observations
        
    def render(self, mode='human'):
        # Send 'render' to all environment workers
        [master_conn.send(['render', None]) for master_conn in self.master_conns]
        # Receive all rendered output from all environment workers
        imgs = [master_conn.recv() for master_conn in self.master_conns]
        
        return imgs
        
    def close(self):
        if self.closed:  # all environments already closed
            return None
        
        # Waiting to receive data from all the workers if they are still working
        if self.waiting:
            [master_conn.recv() for master_conn in self.master_conns]
            
        # Send 'close' to all environment workers
        [master_conn.send(['close', None]) for master_conn in self.master_conns]
        
        # Close all master connections
        [master_conn.close() for master_conn in self.master_conns]
        
        # Join all the processes
        [process.join() for process in self.list_process]
        
        # Update closed flag
        self.closed = True
        
    def seed(self, seeds):
        # Send seeds to all environment workers
        [master_conn.send(['seed', seed]) for master_conn, seed in zip(self.master_conns, seeds)]
        
    @property
    def T(self):
        [master_conn.send(['T', None]) for master_conn in self.master_conns]
        all_T = [master_conn.recv() for master_conn in self.master_conns]
        
        return all_T
