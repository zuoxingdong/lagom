import numpy as np

from gym_maze.envs.Astar_solver import AstarSolver

from lagom.engine import BaseEngine

    
class GoalEngine(BaseEngine):
    def train(self, goal_iter, goal):
        # Set environment with given goal 
        self.runner.env.get_source_env().goal_states = [goal]
        
        # Set max time steps as optimal trajectories (consistent with A* solution)
        if self.config['use_optimal_T']:
            self.config['T'] = self._get_optimal_steps(self.runner.env)
            print(f'A* optimal steps: {self.config["T"]}')
        
        # Training
        for i in range(self.config['train_iter']):  # training iterations
            # Collect one batch of data from runner
            batch_data = self.runner.run(self.config['T'], self.config['train_num_epi'])
            # Update agent by learning over the batch of data
            losses = self.agent.learn(batch_data)
            
            # Useful metrics
            total_loss = losses['total_loss'].data[0]
            batch_returns = [np.sum(data['rewards']) for data in batch_data]
            batch_discounted_returns = [data['returns'][0] for data in batch_data]
            
            # Loggings
            key_goal = ('Sampled goal', goal_iter, goal)
            if i == 0 or (i + 1) % self.config['log_interval'] == 0:
                self.logger.log(self.config['ID'], 
                                [key_goal, ('Train Iteration', i+1), 'Total loss'], 
                                total_loss)
                self.logger.log(self.config['ID'], 
                                [key_goal, ('Train Iteration', i+1), 'Num Episodes'], 
                                len(batch_data))
                self.logger.log(self.config['ID'], 
                                [key_goal, ('Train Iteration', i+1), 'Average Return'], 
                                np.mean(batch_returns))
                self.logger.log(self.config['ID'], 
                                [key_goal, ('Train Iteration', i+1), 'Average Discounted Return'], 
                                np.mean(batch_discounted_returns))
                self.logger.log(self.config['ID'], 
                                [key_goal, ('Train Iteration', i+1), 'Std Return'], 
                                np.std(batch_returns))
                self.logger.log(self.config['ID'], 
                                [key_goal, ('Train Iteration', i+1), 'Min Return'], 
                                np.min(batch_returns))
                self.logger.log(self.config['ID'], 
                                [key_goal, ('Train Iteration', i+1), 'Max Return'], 
                                np.max(batch_returns))
                
                # Dump the loggings
                self.logger.dump(self.config['ID'], 
                                 [key_goal, ('Train Iteration', i+1)], 
                                 indent='')
        
    def eval(self, goal_iter, goal):
        # Get all indicies for free locations in state space
        free_space = np.where(self.runner.env.get_source_env().maze == 0)
        free_space = list(zip(free_space[0], free_space[1]))
        
        # Define goal space as free state space
        goal_space = free_space
        
        # Evaluate the performance of current agent over all feasible goals
        average_return_all_goal = []
        for g in goal_space:
            # Set environment with given goal 
            self.runner.env.get_source_env().goal_states = [g]
            
            # Set max time steps as optimal trajectories (consistent with A* solution)
            if self.config['use_optimal_T']:
                self.config['T'] = self._get_optimal_steps(self.runner.env)
                print(f'A* optimal steps: {self.config["T"]}')
            
            # Evaluate
            # Collect one batch of data from runner
            batch_data = self.runner.run(self.config['T'], self.config['eval_num_epi'])
            
            # Useful metrics
            batch_returns = [np.sum(data['rewards']) for data in batch_data]
            average_return_all_goal.append([g, np.mean(batch_returns)])
        
        # Loggings
        key_goal = ('Sampled goal', goal_iter, goal)
        success_rate_goal_space = [success_rate for g, success_rate in average_return_all_goal]
        self.logger.log(self.config['ID'], 
                        [key_goal, 'Eval', 'All success rate'], 
                        average_return_all_goal)
        self.logger.log(self.config['ID'], 
                        [key_goal, 'Eval', 'Mean success rate'], 
                        np.mean(average_return_all_goal))
        
        # Dump
        print('# Evaluation: ')
        print(f'\tMean success rate over goal space: {np.mean(success_rate_goal_space)}')
        print(f'\tAll success rate over goal space: {success_rate_goal_space}')
        
    def _get_optimal_steps(self, env):
        env.reset()
        
        # Solve maze by A* search from current state to goal
        solver = AstarSolver(env.get_source_env(), env.get_source_env().goal_states[0])
        
        if not solver.solvable():
            raise Error('The maze is not solvable given the current state and the goal state')

        num_optimal_steps = len(solver.get_actions())
        
        if num_optimal_steps == 0:  # for initial state, A* gives 0 step leading to numerical problem.
            num_optimal_steps = 2  # For initial position, optimally 2 steps needed
        
        return num_optimal_steps
    