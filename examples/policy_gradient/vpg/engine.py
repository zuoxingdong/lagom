import numpy as np

import torch

from lagom import Logger
from lagom.engine import BaseEngine
from lagom.envs import make_gym_env
from lagom.envs import make_envs
from lagom.envs.vec_env import SerialVecEnv
from lagom.envs.vec_env import StandardizeVecEnv
from lagom.runner import TrajectoryRunner


class Engine(BaseEngine):
    def train(self, n):
        # Set network as training mode
        self.agent.policy.network.train()
        
        # Collect a list of trajectories
        D = self.runner(N=self.config['train:N'], T=self.config['train:T'])
        
        # Train agent with collected data
        out_agent = self.agent.learn(D)
        # Return training output
        train_output = {}
        train_output['D'] = D
        train_output['out_agent'] = out_agent
        train_output['n'] = n
        
        return train_output
        
    def log_train(self, train_output):
        # Create training logger
        logger = Logger(name='train_logger')
        
        # Unpack training output for logging
        D = train_output['D']
        out_agent = train_output['out_agent']
        n = train_output['n']
        
        # Loggings
        # Use item() for tensor to save memory
        logger.log(key='train_iteration', val=n+1)  # iteration starts from 1
        if self.config['algo:use_lr_scheduler']:
            logger.log(key='current_lr', val=out_agent['current_lr'])

        logger.log(key='loss', val=out_agent['loss'].item())
        policy_loss = torch.stack(out_agent['batch_policy_loss']).mean().item()
        logger.log(key='policy_loss', val=policy_loss)
        entropy_loss = torch.stack(out_agent['batch_entropy_loss']).mean().item()
        logger.log(key='policy_entropy', val=-entropy_loss)  # negation of entropy loss
        value_loss = torch.stack(out_agent['batch_value_loss']).mean().item()
        logger.log(key='value_loss', val=value_loss)

        # Get some data from trajectory list
        batch_returns = [trajectory.all_returns[0] for trajectory in D]
        batch_discounted_returns = [trajectory.all_discounted_returns[0] for trajectory in D]
        num_timesteps = sum([trajectory.T for trajectory in D])

        # Log more information
        logger.log(key='num_trajectories', val=len(D))
        logger.log(key='num_timesteps', val=num_timesteps)
        logger.log(key='accumulated_trained_timesteps', val=self.agent.accumulated_trained_timesteps)
        logger.log(key='average_return', val=np.mean(batch_returns))
        logger.log(key='average_discounted_return', val=np.mean(batch_discounted_returns))
        logger.log(key='std_return', val=np.std(batch_returns))
        logger.log(key='min_return', val=np.min(batch_returns))
        logger.log(key='max_return', val=np.max(batch_returns))

        # Dump the loggings
        print('-'*50)
        logger.dump(keys=None, index=None, indent=0)
        print('-'*50)

        return logger
        
    def eval(self, n):
        # Set network as evaluation mode
        self.agent.policy.network.eval()
        
        # Create a new instance of VecEnv envrionment
        list_make_env = make_envs(make_env=make_gym_env, 
                                  env_id=self.config['env:id'], 
                                  num_env=1, 
                                  init_seed=self.config['seed'])
        env = SerialVecEnv(list_make_env)
        # Wrapper to standardize observation from training scaling of mean and standard deviation
        if self.config['env:normalize']:
            env = StandardizeVecEnv(venv=env, 
                                    use_obs=True, 
                                    use_reward=False,  # do not standardize reward, use original
                                    clip_obs=self.runner.env.clip_obs,
                                    eps=self.runner.env.eps, 
                                    constant_obs_mean=self.runner.env.obs_runningavg.mu, 
                                    constant_obs_std=self.runner.env.obs_runningavg.sigma)
        
        # Create a TrajectoryRunner
        runner = TrajectoryRunner(agent=self.agent, 
                                  env=env, 
                                  gamma=self.config['algo:gamma'])
        # Evaluate the agent for a set of trajectories
        # Retrieve pre-defined maximum episode timesteps in the environment
        T = env.T[0]  # take first one because of VecEnv
        D = runner(N=self.config['eval:N'], T=T)
        
        # Return evaluation output
        eval_output = {}
        eval_output['D'] = D
        eval_output['n'] = n
        eval_output['T'] = T
        
        return eval_output
        
    def log_eval(self, eval_output):
        # Create evaluation logger
        logger = Logger(name='eval_logger')
        
        # Unpack evaluation for logging
        D = eval_output['D']
        n = eval_output['n']
        T = eval_output['T']
        
        # Compute some metrics
        batch_returns = [sum(trajectory.all_r) for trajectory in D]
        batch_T = [trajectory.T for trajectory in D]
        
        # Loggings
        # Use item() for tensor to save memory
        logger.log(key='evaluation_iteration', val=n+1)
        logger.log(key='num_trajectories', val=len(D))
        logger.log(key='max_allowed_horizon', val=T)
        logger.log(key='average_horizon', val=np.mean(batch_T))
        logger.log(key='num_timesteps', val=np.sum(batch_T))
        logger.log(key='accumulated_trained_timesteps', val=self.agent.accumulated_trained_timesteps)
        logger.log(key='average_return', val=np.mean(batch_returns))
        logger.log(key='std_return', val=np.std(batch_returns))
        logger.log(key='min_return', val=np.min(batch_returns))
        logger.log(key='max_return', val=np.max(batch_returns))
        
        # Dump the loggings
        print('-'*50)
        logger.dump(keys=None, index=None, indent=0)
        print('-'*50)
        
        return logger
