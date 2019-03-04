import numpy as np

from lagom.transform import RunningMeanStd
from lagom.envs import VecEnvWrapper


class VecStandardizeReward(VecEnvWrapper):
    r"""Standardize the reward by running averages.
    
    .. warning::
    
        We do not subtract running mean from reward but only divides it by running
        standard deviation. Because subtraction by mean will alter the reward shape
        so this might degrade the performance. 
        
    .. note::
    
        Each :meth:`reset`, we do not clean up the ``self.all_returns`` buffer. Because
        of discount factor (:math:`< 1`), the running averages will converge after some iterations. 
        Therefore, we do not allow discounted factor as :math:`1.0` since it will lead to
        unbounded explosion of reward running averages. 
        
    Args:
        venv (VecEnv): a vectorized environment
        clip (float): clipping range of standardized reward, i.e. [-clip, clip]
        gamma (float): discounted factor. Note that the value 1.0 should not be used. 
        constant_std (ndarray): Constant standard deviation to standardize reward. Note that
            when it is provided, then running average will be ignored. 
    
    """
    def __init__(self, venv, clip=10., gamma=0.99, constant_std=None):
        super().__init__(venv)
        self.clip = clip
        assert gamma > 0.0 and gamma < 1.0, 'we do not allow discounted factor as 1.0. See docstring for details. '
        self.gamma = gamma
        self.constant_std = constant_std
        
        self.eps = 1e-8
        self.runningavg = RunningMeanStd()
        
        # Buffer to save discounted returns from each environment
        self.all_returns = np.zeros(self.num_env, dtype=np.float32)
        
    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        # Set discounted return buffer as zero for those episodes which terminate
        self.all_returns[dones] = 0.0
        return observations, self.process_reward(rewards), dones, infos
    
    def reset(self):
        # Reset returns buffer, because all environments are also reset
        self.all_returns.fill(0.0)
        return super().reset()
    
    def process_reward(self, rewards):
        if self.constant_std is None:
            self.all_returns = rewards + self.gamma*self.all_returns
            self.runningavg(self.all_returns)
            std = self.runningavg.sigma
        else:
            std = self.constant_std
            
        # Do NOT subtract from mean, but only divided by std
        rewards = rewards/(std + self.eps)
        rewards = np.clip(rewards, -self.clip, self.clip)
        
        return rewards
    
    @property
    def mean(self):
        return self.runningavg.mu
    
    @property
    def std(self):
        return self.runningavg.sigma
