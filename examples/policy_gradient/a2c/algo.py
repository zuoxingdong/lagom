from pathlib import Path

from itertools import count

import numpy as np
import torch

from lagom import Logger
from lagom.utils import pickle_dump
from lagom.utils import set_global_seeds

from lagom import BaseAlgorithm

from lagom.envs import make_gym_env
from lagom.envs import make_vec_env
from lagom.envs import EnvSpec
from lagom.envs.vec_env import SerialVecEnv
from lagom.envs.vec_env import VecStandardize
from lagom.envs.vec_env import VecClipAction
from lagom.envs.wrappers import TimeAwareObservation

from lagom.runner import RollingSegmentRunner

from agent import Agent
from engine import Engine


class Algorithm(BaseAlgorithm):
    def __call__(self, config, seed, device):
        set_global_seeds(seed)
        logdir = Path(config['log.dir']) / str(config['ID']) / str(seed)

        if config['env.time_aware_obs']:
            kwargs = {'extra_wrapper': [TimeAwareObservation]}
        else:
            kwargs = {}
        env = make_vec_env(SerialVecEnv, make_gym_env, config['env.id'], config['train.N'], seed, monitor=True, **kwargs)
        if config['eval.independent']:
            eval_env = make_vec_env(SerialVecEnv, make_gym_env, config['env.id'], config['eval.N'], seed)
        if config['env.clip_action']:
            env = VecClipAction(env)
            if config['eval.independent']:
                eval_env = VecClipAction(eval_env)
        if config['env.standardize']:  # running averages of observation and reward
            env = VecStandardize(venv=env, 
                                 use_obs=True, 
                                 use_reward=False,  # A2C specific 
                                 clip_obs=10., 
                                 clip_reward=10., 
                                 gamma=0.99, 
                                 eps=1e-8)    
        env_spec = EnvSpec(env)
        
        agent = Agent(config, env_spec, device)
        
        runner = RollingSegmentRunner(config, agent, env)
        
        if config['eval.independent']:
            engine = Engine(agent, runner, config, eval_env=eval_env)
        else:
            engine = Engine(agent, runner, config)
        
        train_logs = []
        eval_logs = []
        for i in count():
            if 'train.iter' in config and i >= config['train.iter']:  # enough iterations
                break
            elif 'train.timestep' in config and agent.total_T >= config['train.timestep']:  # enough timesteps
                break
            
            train_output = engine.train(i)
            
            if i == 0 or (i+1) % config['log.interval'] == 0:
                train_log = engine.log_train(train_output)
                train_logs.append(train_log)
                
                if config['eval.independent']:
                    with torch.no_grad():  # disable grad, save memory
                        eval_output = engine.eval(n=i)
                    eval_log = engine.log_eval(eval_output)
                    eval_logs.append(eval_log)

        pickle_dump(obj=train_logs, f=logdir/'train_logs', ext='.pkl')
        pickle_dump(obj=eval_logs, f=logdir/'eval_logs', ext='.pkl')
        
        return None
