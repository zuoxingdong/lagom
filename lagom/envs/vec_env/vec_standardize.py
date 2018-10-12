import numpy as np

from lagom import pickle_dump

from lagom.core.transform import RunningMeanStd

from .vec_env import VecEnvWrapper


class VecStandardize(VecEnvWrapper):
    r"""A vectorized environment wrapper that standardizes the observations and rewards
    from the environments by using running average. 
    
    Each :meth:`step_wait` and :meth:`reset`, the observation and reward are fed into
    a processor to update the running average, and then the observation is standardized
    by subtracting the running mean and devided by running standard deviation.
    
    .. note::
    
        We do not subtract running mean from reward but only divides it by running
        standard deviation. Because subtraction by mean will alter the reward shape
        so this might degrade the performance. 
        
    .. note::
    
        Each :meth:`reset`, we do not clean up the ``self.all_returns`` buffer. Because
        of discount factor (:math:`< 1`), the running averages will converge after some iterations. 
        Therefore, we do not allow discounted factor as :math:`1.0` since it will lead to
        unbounded explosion of reward running averages. 
        
    .. warning::
    
        To evaluate the agent trained on standardized environment, remember to
        save and load observation scalings, otherwise, the performance will be incorrect. 
        One could use :meth:`save_running_average`. 
    
    See :class:`RunningMeanStd` for more details about running average. 
    
    Example::
    
        >>> from lagom.envs import make_envs, make_gym_env
        >>> from lagom.envs.vec_env import SerialVecEnv
        >>> list_make_env = make_envs(make_env=make_gym_env, env_id='Pendulum-v0', num_env=2, init_seed=1)
        >>> venv = SerialVecEnv(list_make_env=list_make_env)
        >>> venv = VecStandardize(venv=venv, use_obs=True, use_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99, eps=1e-8)                                 
        >>> venv
        <VecStandardize: Pendulum-v0, n: 2>
        
        >>> venv.reset()
        array([[ 1.0000001 ,  0.99999997, -0.99999997],
               [-0.99999993, -1.00000007,  0.99999986]])
               
        >>> venv.running_averages
        {'obs_avg': {'mu': array([-0.6503495,  0.5606869,  0.4055612], dtype=float32),
          'sigma': array([0.33465797, 0.388175  , 0.23515129], dtype=float32)},
         'r_avg': {'sigma': None}}

    """
    def __init__(self,
                 venv, 
                 use_obs=True, 
                 use_reward=True, 
                 clip_obs=10., 
                 clip_reward=10., 
                 gamma=0.99, 
                 eps=1e-8, 
                 constant_obs_mean=None, 
                 constant_obs_std=None, 
                 constant_reward_std=None):
        r"""Initialize the wrapper. 
        
        Args:
            venv (VecEnv): a vectorized environment
            use_obs (bool): Whether to standardize the observation by using its running average
            use_reward (bool): Whether to standardize the reward by using its running average
                Note that running average here is computed with discounted returns iteratively. 
            clip_obs (float/ndarray): clipping range of standardized observation, i.e. [-clip_obs, clip_obs]
            clip_reward (float): clipping range of standardized reward, i.e. [-clip_reward, clip_reward]
            gamma (float): discounted factor. Note that the value 1.0 should not be used. 
                It will cause reward running average (computed with discounted returns) to exploit
                unboundly. 
            eps (float): a small epsilon for numerical stability of dividing by standard deviation. 
                e.g. when standard deviation is zero.
            constant_obs_mean (ndarray): Constant mean to standardize observation. Note that when it is
                provided, then running average will be ignored. 
            constant_obs_std (ndarray): Constant standard deviation to standardize observation. Note that
                when it is provided, then running average will be ignored.
            constant_reward_std (ndarray): Constant standard deviation to standardize reward. Note that
                when it is provided, then running average will be ignored. 
        """
        super().__init__(venv)
        
        self.obs_runningavg = RunningMeanStd()
        self.reward_runningavg = RunningMeanStd()
        
        self.use_obs = use_obs
        self.use_reward = use_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        
        self.gamma = gamma
        assert self.gamma < 1.0, 'We do not allow discounted factor as 1.0. See docstring for details. '
        self.eps = eps
        
        self.constant_obs_mean = constant_obs_mean
        self.constant_obs_std = constant_obs_std
        self.constant_reward_std = constant_reward_std
        
        # Buffer to save discounted returns from each environment
        self.all_returns = np.zeros(self.num_env).astype(np.float32)
    
    def step_wait(self):
        # Call original step_wait to get results from all environments
        observations, rewards, dones, infos = self.venv.step_wait()
        
        # Set discounted return buffer as zero for those episodes which terminate
        self.all_returns[dones] = 0.0
        
        return self.process_obs(observations), self.process_reward(rewards), dones, infos
    
    def reset(self):
        # Reset returns buffer, because all environments are also reset
        self.all_returns.fill(0.0)
        
        return self.process_obs(self.venv.reset())
    
    def close_extras(self):
        return self.venv.close()
    
    def process_reward(self, rewards):
        if self.use_reward and self.constant_reward_std is None:  # using running average
            # Compute discounted returns
            self.all_returns = rewards + self.gamma*self.all_returns
            # Update with calculated discounted returns
            self.reward_runningavg(self.all_returns)
            # Standardize the reward
            # Not subtract from mean, but only divided by std
            std = self.reward_runningavg.sigma
            if not np.allclose(std, 0.0):  # only non-zero std
                rewards = rewards/(std + self.eps)
            
            rewards = np.clip(rewards, a_min=-self.clip_reward, a_max=self.clip_reward)
            
            return rewards
        elif self.use_reward and self.constant_reward_std is not None:  # use given constant std
            rewards = rewards/(self.constant_reward_std + self.eps)
            
            rewards = np.clip(rewards, a_min=-self.clip_reward, a_max=self.clip_reward)
            
            return rewards
        else:  # return original rewards if use_reward is turned off
            return rewards
        
    def process_obs(self, obs):
        if self.use_obs and self.constant_obs_mean is None and self.constant_obs_std is None:  # use running average
            # Update with new observation
            self.obs_runningavg(obs)
            # Standardize the observation
            mean = self.obs_runningavg.mu
            std = self.obs_runningavg.sigma
            if not np.allclose(std, 0.0):  # only non-zero std
                obs = (obs - mean)/(std + self.eps)
            
            obs = np.clip(obs, a_min=-self.clip_obs, a_max=self.clip_obs)
            
            return obs
        elif self.use_obs and self.constant_obs_mean is not None and self.constant_obs_std is not None:  # use given moment
            obs = (obs - self.constant_obs_mean)/(self.constant_obs_std + self.eps)
            
            obs = np.clip(obs, a_min=-self.clip_obs, a_max=self.clip_obs)
            
            return obs
        else:  # return original observation if use_obs is turned off
            return obs
        
    @property
    def running_averages(self):
        r"""Returns the running averages for observation and reward in a dictionary. 
        
        A dictionary with keys 'obs_avg' and 'r_avg' will be created. Each key
        contains sub-keys ['mu', 'sigma'] except for reward which only contains 'sigma'. 
        
        Returns
        -------
        out : dict
            a dictionary of running averages
        """
        out = {'obs_avg': {'mu': self.obs_runningavg.mu, 
                           'sigma': self.obs_runningavg.sigma}, 
               'r_avg': {'sigma': self.reward_runningavg.sigma}}
        
        return out
        
    def save_running_average(self, f):
        r"""Save the running averages for observation and reward in a dictionary by pickling. 
        
        It saves the mean and standard deviation for observation running average and the standard deviation
        for reward running average. A dictionary with keys 'obs_avg' and 'r_avg' will be created. Each key
        contains sub-keys ['mu', 'sigma']. 
        
        Args:
            f (str): saving path
        """
        out = self.running_averages
        
        pickle_dump(obj=out, f=f, ext='.pkl')
