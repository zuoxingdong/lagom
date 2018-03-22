import numpy as np

import torch.optim as optim

from lagom.agents import REINFORCEAgent
from lagom.core.policies import CategoricalMLPPolicy
from lagom.core.utils import Logger
from lagom.runner import Runner
from lagom.envs import EnvSpec
from lagom.algos import BaseAlgorithm
        
from engine import GoalEngine
from goal_sampler.uniform_goal_sampler import UniformGoalSampler
from goal_sampler.linear_goal_sampler import LinearGoalSampler


class GoalSelection(BaseAlgorithm):
    def run(self, env, config):
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
        # Create logger
        logger = Logger(name='goal_selection')
        # Create an engine (training and evaluation)
        engine = GoalEngine(agent, runner, config, logger)
            
        # Create a goal sampler
        goal_sampler = UniformGoalSampler(runner.env)
        #goal_sampler = LinearGoalSampler([[1, 2], [2, 2], [2, 3], [3, 3], [3, 4], [4, 4]])
        
        for iter_num in range(config['num_goal']):
            # Sample a goal
            goal = goal_sampler.sample()
            print(f'\nSampled Goal ({iter_num}/{config["num_goal"]}): {goal})
            
            # Train the sampled goal
            engine.train(iter_num, goal)
            
            # Evaluation over all goals
            engine.eval(iter_num, goal)