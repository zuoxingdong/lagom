import numpy as np

from lagom.transform import RunningMeanVar
from lagom.envs import VecEnvWrapper


class VecStandardizeReward(VecEnvWrapper):
    r"""Standardize the reward by running estimation of variance.
    
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
        env (VecEnv): a vectorized environment
        clip (float): clipping range of standardized reward, i.e. [-clip, clip]
        gamma (float): discounted factor. Note that the value 1.0 should not be used. 
        constant_var (ndarray): Constant variance to standardize reward. Note that
            when it is provided, then running average will be ignored. 
    
    """
    def __init__(self, env, clip=10., gamma=0.99, constant_var=None):
        super().__init__(env)
        self.clip = clip
        assert gamma > 0.0 and gamma < 1.0, 'we do not allow discounted factor as 1.0. See docstring for details. '
        self.gamma = gamma
        self.constant_var = constant_var
        
        self.eps = 1e-8
        
        if constant_var is None:
            self.online = True
            self.running_moments = RunningMeanVar(shape=())
        else:
            self.online = False
        
        # Buffer to save discounted returns from each environment
        self.all_returns = np.zeros(len(env), dtype=np.float64)
        
    def step(self, actions):
        observations, rewards, dones, infos = self.env.step(actions)
        # Set discounted return buffer as zero for those episodes which terminate
        self.all_returns[dones] = 0.0
        return observations, self.process_reward(rewards), dones, infos
    
    def reset(self):
        # Reset returns buffer, because all environments are also reset
        self.all_returns.fill(0.0)
        return super().reset()
    
    def process_reward(self, rewards):
        # Do NOT subtract from mean, but only divided by std
        if self.online:
            self.all_returns = rewards + self.gamma*self.all_returns
            self.running_moments(self.all_returns)
            if self.running_moments.n >= 2:
                std = np.sqrt(self.running_moments.var + self.eps)
                rewards = rewards/std
        else:
            std = np.sqrt(self.constant_var + self.eps)
            rewards = rewards/std
        rewards = np.clip(rewards, -self.clip, self.clip)
        return rewards.astype(np.float32)
        
    @property
    def var(self):
        return self.running_moments.var
