import numpy as np

from lagom.core.transform import RunningMeanStd

from lagom.envs.vec_env import VecEnvWrapper


class StandardizeVecEnv(VecEnvWrapper):
    """
    Standardize the observations and rewards by using running average. 
    i.e. subtract by running mean and divided by running standard deviation. 
    
    Or the user can also provide constant mean and standard deviation for standardization. 
    
    Note that we do not subtract the mean from rewards but only divided by standard deviation. 
    And the reward running average is computed by discounted returns continuously. 
    
    Also note that each `reset()` we do not clean up the `self.all_returns` buffer. 
    Because of discount factor (< 1), the running averages will be converged after some iterations. 
    Therefore, we do not allow discounted factor as 1.0, as it will lead to unbounded explosion 
    of reward running averages. 
    
    IMPORTANT: Remember to save and load observation scaling for evaluating the agent. 
    
    Examples:
    
        list_make_env = make_envs(make_env=make_gym_env, 
                                  env_id='Pendulum-v0', 
                                  num_env=2, 
                                  init_seed=1)

        venv = SerialVecEnv(list_make_env=list_make_env)

        env = StandardizeVecEnv(venv=venv, 
                                use_obs=True, 
                                use_reward=True, 
                                clip_obs=10.0, 
                                clip_reward=10.0, 
                                gamma=0.99, 
                                eps=1e-8)
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
        """
        Args:
            venv (VecEnv): vectorized environment
            use_obs (bool): Whether to standardize the observation by using its running average
            use_reward (bool): Whether to standardize the reward by using its running average
                Note that running average here is computed by discounted returns iteratively. 
            clip_obs (float/ndarray): clipping range of standardized observation, i.e. [-clip_obs, clip_obs]
            clip_reward (float): clipping range of standardized reward, i.e. [-clip_reward, clip_reward]
            gamma (float): discounted factor. Note that the value 1.0 should not be used. 
                It will let the reward running average (computed with discounted returns) exploits
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
        self.obs_runningavg = RunningMeanStd(dtype='ndarray')
        self.reward_runningavg = RunningMeanStd(dtype='ndarray')
        
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
        
        self.all_returns = np.zeros(self.num_env)
    
    def step_wait(self):
        # Call original step_wait to get results from all environments
        observations, rewards, dones, infos = self.venv.step_wait()
        
        return self.process_obs(observations), self.process_reward(rewards), dones, infos
        
    def process_reward(self, rewards):
        if self.use_reward and self.constant_reward_std is None:  # using running average
            # Compute discounted returns
            self.all_returns = rewards + self.gamma*self.all_returns
            # Update with calculated discounted returns
            self.reward_runningavg(self.all_returns)
            # Standardize the reward
            mean = self.reward_runningavg.mu
            std = self.reward_runningavg.sigma
            # Note that we do not subtract from mean, but only divided by std
            if not np.allclose(std, 0.0):  # only non-zero std
                rewards = rewards/(std + self.eps)
                
            # Clipping
            rewards = np.clip(rewards, a_min=-self.clip_reward, a_max=self.clip_reward)
            
            return rewards
        elif self.use_reward and self.constant_reward_std is not None:  # use given constant std
            rewards = rewards/(self.constant_reward_std + self.eps)
            
            # Clipping
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
            
            # Clipping
            obs = np.clip(obs, a_min=-self.clip_obs, a_max=self.clip_obs)
            
            return obs
        elif self.use_obs and self.constant_obs_mean is not None and self.constant_obs_std is not None:  # use given moment
            obs = (obs - self.constant_obs_mean)/(self.constant_obs_std + self.eps)
            
            # Clipping
            obs = np.clip(obs, a_min=-self.clip_obs, a_max=self.clip_obs)
            
            return obs
        else:  # return original observation if use_obs is turned off
            return obs
        
    def reset(self):
        return self.process_obs(self.venv.reset())
