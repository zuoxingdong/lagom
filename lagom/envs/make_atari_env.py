import numpy as np

import gym
from gym.wrappers import Monitor

from .wrappers import Wrapper
from .wrappers import GymWrapper
from .wrappers import ResizeObservation
from .wrappers import GrayScaleObservation
from .wrappers import ScaleImageObservation
from .wrappers import ClipReward
from .wrappers import FrameStack


class AtariPreprocessing(Wrapper):
    r"""Atari 2600 preprocessings. 
    
    This class follows the guidelines in 
    Machado et al. (2018), "Revisiting the Arcade Learning Environment: 
    Evaluation Protocols and Open Problems for General Agents".

    Specifically:
    
    * NoopReset: obtain initial state by taking random number of no-ops on reset. 
    * FireReset: take action on reset for environments that are fixed until firing. 
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost: turned off by default
    * Resize to a square image: 84x84 by default
    * Grayscale
    
    """
    def __init__(self, env, noop_max=30, frame_skip=4, done_on_life_loss=False):
        r"""Constructor
        
        Args:
            env (Env): environment
            noop_max (int): max number of no-ops
            frame_skip (int): the frequency at which the agent experiences the game. 
            done_on_life_loss (bool): if True, then step() returns done=True whenever a
                life is lost. 
        """
        super().__init__(env)
        assert frame_skip > 0
        
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'
        
        self.frame_skip = frame_skip
        self.done_on_life_loss = done_on_life_loss
        
        # buffer of most recent two observations for max pooling
        self.obs_buffer = [np.empty(self.env.observation_space.shape, dtype=np.uint8),
                           np.empty(self.env.observation_space.shape, dtype=np.uint8)]
        
        self.ale = self._get_ale(self.env)
        self.lives = 0
        self.real_done = False
        
    def _get_ale(self, env):
        while True:
            if isinstance(env, gym.envs.atari.AtariEnv):
                return env.ale
            else:
                env = env.unwrapped
                
    def step(self, action):
        R = 0.0
        
        for t in range(self.frame_skip):
            observation, reward, done, info = self.env.step(action)
            R += reward
            self.real_done = done
            
            if self.done_on_life_loss:
                new_lives = self.ale.lives()
                done = done or new_lives < self.lives
                self.lives = new_lives
                
            if done:
                break
            if t == self.frame_skip - 2:
                self.obs_buffer[0] = observation
            elif t == self.frame_skip - 1:
                self.obs_buffer[1] = observation
                
        if self.frame_skip > 1:  # pooling: more efficieny in-place
            np.maximum(self.obs_buffer[0], self.obs_buffer[1], out=self.obs_buffer[0])
            
        return self.obs_buffer[0], R, done, info
                
    def reset(self):
        # NoopReset
        self.env.reset()
        noops = self.env.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset()
        
        # FireReset
        action_meanings = self.env.unwrapped.get_action_meanings()
        if action_meanings[1] == 'FIRE' and len(action_meanings) >= 3:
            obs, _, done, _ = self.env.step(1)
            obs, _, done, _ = self.env.step(2)
                
        self.obs_buffer[0] = obs
        self.obs_buffer[1].fill(0)
        self.lives = self.ale.lives()
        if self.frame_skip > 1:  # pooling: more efficieny in-place
            np.maximum(self.obs_buffer[0], self.obs_buffer[1], out=self.obs_buffer[0])
            
        return self.obs_buffer[0]


def make_atari_env(env_id, seed, monitor=False, monitor_dir=None):
    r"""Create Atari environment with all necessary preprocessings. 
    
    Args:
        env_id (str): Atari game name without version, e.g. Pong, Breakout
        seed (int): random seed for the environment
        monitor (bool, optional): If ``True``, then wrap the enviroment with Monitor for video recording.  
        monitor_dir (str, optional): directory to save all data from Monitor. 
        
    Returns
    -------
    env : Env
        lagom-compatible environment
    """
    env = gym.make(env_id + 'NoFrameskip-v4')
    # remove gym TimeLimit wrapper (caps 100k frames), we want to cap 108k frames (30 min)
    env = env.env
    if monitor:
        env = Monitor(env, monitor_dir)
    env = GymWrapper(env)
    env = ResizeObservation(env, 84)
    env = GrayScaleObservation(env, keep_dim=False)
    env = AtariPreprocessing(env)
    env = ScaleImageObservation(env)
    env = ClipReward(env)
    env = FrameStack(env, 4)
    env.seed(seed)
    
    return env
