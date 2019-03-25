import numpy as np

import gym
from gym import Wrapper

from .wrappers import ResizeObservation
from .wrappers import GrayScaleObservation
from .wrappers import ScaleImageObservation
from .wrappers import FrameStack

from .wrappers import TimeLimit


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
    * Termination signal when a life is lost: turned off by default. Not recommended by Machado et al. (2018).
    * Resize to a square image: 84x84 by default
    * Grayscale
    
    Args:
        env (Env): environment
        noop_max (int): max number of no-ops
        frame_skip (int): the frequency at which the agent experiences the game. 
        done_on_life_loss (bool): if True, then step() returns done=True whenever a
            life is lost. 
    
    """
    def __init__(self, env, noop_max=30, frame_skip=4, done_on_life_loss=False):
        super().__init__(env)
        assert frame_skip > 0
        
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'
        
        self.frame_skip = frame_skip
        self.done_on_life_loss = done_on_life_loss
        
        # buffer of most recent two observations for max pooling
        self.obs_buffer = [np.empty(self.observation_space.shape, dtype=np.uint8),
                           np.empty(self.observation_space.shape, dtype=np.uint8)]
        
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
                
    def reset(self, **kwargs):
        # NoopReset
        self.env.reset(**kwargs)
        noops = self.env.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset(**kwargs)
        
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


def make_atari(name, sticky_action=True, max_episode_steps=None):
    r"""Create Atari 2600 environment and wrapped it with preprocessing guided by 
    
    Machado et al. (2018), "Revisiting the Arcade Learning Environment: 
    Evaluation Protocols and Open Problems for General Agents".
    
    Args:
        name (str): name of Atari 2600
        sticky_action (bool): whether to use sticky actions, i.e. 25% probability to persist
            the action when a new command is sent to the ALE, introducing a mild stochasticity.
        max_episode_steps (int): user-defined episode length. 
    """
    assert name is not None
    ver = 'v0' if sticky_action else 'v4'
    env_id = f'{name}NoFrameskip-{ver}'
    env = gym.make(env_id)
    env = env.env  # strip out gym TimeLimit
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps)
    else:
        env = TimeLimit(env, env.spec.max_episode_steps)
    env = ResizeObservation(env, 84)
    env = GrayScaleObservation(env, keep_dim=False)
    env = AtariPreprocessing(env)
    env = FrameStack(env, 4)
    
    return env
