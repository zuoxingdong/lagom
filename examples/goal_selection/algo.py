import os
    
import numpy as np

import torch.optim as optim

from gym_maze.envs.Astar_solver import AstarSolver

from lagom.agents import REINFORCEAgent
from lagom.core.policies import CategoricalMLPPolicy
from lagom.core.utils import Logger
from lagom.runner import Runner
from lagom.envs import EnvSpec
from lagom.algos import BaseAlgorithm
from lagom.trainer import SimpleTrainer
        
from goal_evaluator import GoalEvaluator
from uniform_goal_sampler import UniformGoalSampler
from linear_goal_sampler import LinearGoalSampler


class GoalSelection(BaseAlgorithm):
    def run(self, env, args):
        # Create environment specification
        env_spec = EnvSpec(env)
        # Create a goal-conditional policy
        policy = CategoricalMLPPolicy(env_spec, fc_sizes=args.fc_sizes, predict_value=args.predict_value)
        # Create an optimzer
        optimizer = optim.Adam(policy.parameters(), lr=args.lr)
        # Create an agent
        agent = REINFORCEAgent(policy, optimizer)
        # Create a Runner
        runner = Runner(agent, env, args.gamma)
        # Create logger
        logger = Logger(path='logs/', dump_mode=[])
        
        # Backup the allowed time steps
        T = args.T
        
        # Create a goal sampler
        goal_sampler = UniformGoalSampler(runner.env)
        #goal_sampler = LinearGoalSampler([[1, 2], [2, 2], [2, 3], [3, 3], [3, 4], [4, 4]])
        
        for iter_num in range(args.num_outer_iter):
            print(f'\n# Outer Iteration # {iter_num+1:<5d}/{args.num_outer_iter:<5d}')
            
            # Sample a goal
            goal = goal_sampler.sample()
            
            print(f'# Sampled Goal {goal}')
            
            # Set environment with sampled goal 
            runner.env.get_source_env().goal_states = [goal]
            
            # Recover to max allowed time steps
            args.T = T
            
            # Training phase
            train_logger = Logger(path='logs/', dump_mode=['screen'])
            trainer = SimpleTrainer(agent, runner, args, train_logger)
            trainer.train()
            
            # Set max time steps for optimal trajectories (consistent with A* solution)
            if args.use_optimal_T:
                args.T = self._get_optimal_steps(runner.env)
                
                print(f'optimal steps: {args.T}')
                
                # TEMP: deal with initial state, A* will give 0 step leading to numerical problem.
                if args.T == 0:
                    args.T = 2  # For initial position, optimally 2 steps needed
            
            # Evaluation phase
            eval_logger = Logger(path='logs/', dump_mode=[])
            evaluator = GoalEvaluator(agent, runner, args, eval_logger)
            evaluator.evaluate()

            all_succ_rate = list(eval_logger.metriclog['Average Return over all goals'].values())
            mean_succ_rate = np.mean(all_succ_rate)
            print(f'\n# Success rate over goal space: {mean_succ_rate:<}')
            print(f'# Average Return over all goals: {all_succ_rate}')
            
            # Logging for all success rate and its mean
            logger.log_metric('Average Return over all goals', all_succ_rate, (iter_num, tuple(goal)))
            logger.log_metric('Success rate over goal space', mean_succ_rate, (iter_num, tuple(goal)))
        
        
        # Save the loggings
        logger.save_metriclog('ID_{:d}_eval_uniform'.format(args.ID))
        
    def _get_optimal_steps(self, env):
        env.reset()
        
        # Solve maze by A* search from current state to goal
        solver = AstarSolver(env.get_source_env(), env.get_source_env().goal_states[0])
        
        if not solver.solvable():
            raise Error('The maze is not solvable given the current state and the goal state')

        num_optimal_steps = len(solver.get_actions())
        
        return num_optimal_steps
    
    

        


