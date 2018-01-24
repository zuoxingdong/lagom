import os
    
import numpy as np

import torch.optim as optim

from lagom.agents import REINFORCEAgent
from lagom.core.policies import CategoricalMLPGoalPolicy
from lagom.core.utils import Logger
from lagom.runner import Runner
from lagom.envs import EnvSpec
from lagom.algos import BaseAlgorithm
from lagom.trainer import SimpleTrainer
        
from lagom.algos.goal_selection.goal_evaluator import GoalEvaluator
from lagom.algos.goal_selection.uniform_goal_sampler import UniformGoalSampler
from lagom.algos.goal_selection.linear_goal_sampler import LinearGoalSampler

        
class GoalSelection(BaseAlgorithm):
    def run(self, env, args):
        # Create environment specification
        env_spec = EnvSpec(env)
        # Create a goal-conditional policy
        policy = CategoricalMLPGoalPolicy(env_spec, fc_sizes=args.fc_sizes, predict_value=args.predict_value)
        # Create an optimzer
        optimizer = optim.Adam(policy.parameters(), lr=args.lr)
        # Create an agent
        agent = REINFORCEAgent(policy, optimizer)
        # Create a Runner
        runner = Runner(agent, env, args.gamma)
        # Create loggers
        train_logger = Logger(path='logs/', dump_mode=['screen'])
        eval_logger = Logger(path='logs/', dump_mode=[])
        
        # Create a goal sampler
        goal_sampler = UniformGoalSampler(runner.env)
        #goal_sampler = LinearGoalSampler([[2, 2], [3, 3], [4, 4]])
        
        for iter_num in range(args.num_outer_iter):
            print('\n# Outer Iteration # {:<5d}'.format(iter_num + 1))
            
            # Sample a goal
            goal = goal_sampler.sample()
            
            print('# Sampled Goal {}'.format(goal))
            
            # Set environment with sampled goal 
            runner.env.goal_states = [goal]
            
            # Training phase
            trainer = SimpleTrainer(agent, runner, args, train_logger)
            trainer.train()

            # Save all training loggings
            #train_logger.save_metriclog('ID_{:d}_train'.format(args.ID))

            # Evaluation phase
            evaluator = GoalEvaluator(agent, runner, args, eval_logger)
            evaluator.evaluate()

            all_succ_rate = list(eval_logger.metriclog['Average Return over all goals'].values())
            mean_succ_rate = np.mean(all_succ_rate)
            print('\n# Success rate over goal space: {:<}'.format(mean_succ_rate))
            
            print('# Detailed success rates: {}'.format(all_succ_rate))
            
            # Logging for success rate
            eval_logger.log_metric('Success rate over goal space', mean_succ_rate, iter_num)
        
        
        # Save the evaluation loggings
        eval_logger.save_metriclog('ID_{:d}_eval_uniform'.format(args.ID))
        
    
    
    
    

        


