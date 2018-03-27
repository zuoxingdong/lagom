from copy import deepcopy

import numpy as np

import torch.optim as optim

from lagom import BaseAlgorithm
from lagom.agents import REINFORCEAgent
from lagom.core.policies import CategoricalMLPPolicy
from lagom.core.utils import Logger
from lagom.runner import Runner
from lagom.envs import EnvSpec

from goal_sampler import SWUCBgGoalSampler

from engine import GoalEngine


class GoalSelection(BaseAlgorithm):
    def run(self, env, config, logger_queue):
        # Create a logger with algorithm name
        logger = Logger(name=f'{self.name}')
        # Create environment specification
        env_spec = EnvSpec(env)
        # Create a goal-conditional policy
        policy = CategoricalMLPPolicy(env_spec, config)
        # Create an optimzer
        optimizer = optim.Adam(policy.parameters(), lr=config['lr'])
        # Create an agent
        agent = REINFORCEAgent(policy, optimizer, config)
        # Create a Runner
        runner = Runner(agent, env, config['gamma'])
        # Create an engine (training and evaluation)
        engine = GoalEngine(agent, runner, config, logger)
            
        # Create a goal sampler
        goal_sampler = config['goal_sampler'](runner, config)
        
        for iter_num in range(config['num_goal']):
            # Sample a goal
            goal = goal_sampler.sample()
            print(f'\nSampled Goal ({iter_num+1}/{config["num_goal"]}): {goal}')
            if isinstance(goal_sampler, SWUCBgGoalSampler):
                agent_old = deepcopy(agent)
            
            # Train the sampled goal
            engine.train(iter_num, goal)
            
            # For SW-UCB-g goal sampler
            if isinstance(goal_sampler, SWUCBgGoalSampler):
                # Calculate reward from learning progress signal
                reward = goal_sampler._calculate_reward(agent_old, agent, env, config, goal)
                # Update the sampler
                goal_sampler.update(reward)
            
            # Evaluation over all goals
            engine.eval(iter_num, goal)
            
            
            print(goal_sampler.infos)
            
            
        # Put the logger into the Queue (shared memory) for Processes of Experiment
        logger_queue.put(logger)
            