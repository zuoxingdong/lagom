import numpy as np

from lagom.engine import BaseEngine

    
class GoalEngine(BaseEngine):
    def train(self, goal_iter, goal):
        # Set environment with given goal 
        self.runner.env.get_source_env().goal_states = [goal]
        
        # Set max time steps as optimal trajectories (consistent with A* solution)
        if self.config['use_optimal_T']:
            self.config['T'] = self.runner.env.all_steps[tuple(goal)]
        
        # Training
        for i in range(self.config['train_iter']):  # training iterations
            # Collect one batch of data from runner
            batch_data = self.runner.run(self.config['T'], 
                                         self.config['train_num_epi'], 
                                         mode='sampling')
            
            # Update agent by learning over the batch of data
            output_learn = self.agent.learn(batch_data)
            
            # Useful metrics
            loss = output_learn['loss'].item()
            batch_returns = [np.sum(episode.all_r) for episode in batch_data]
            batch_discounted_returns = [episode.returns for episode in batch_data]
            
            # Loggings
            key_goal = ('Sampled goal', goal_iter, tuple(goal))
            if i == 0 or (i + 1) % self.config['log_interval'] == 0:
                self.logger.log(self.config['ID'], 
                                [key_goal, ('Train Iteration', i+1), 'Loss'], 
                                loss)
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
        # Define goal space as free state space
        goal_space = self.runner.env.free_space
        
        # Evaluate the performance of current agent over all feasible goals
        average_return_all_goal = []
        for g in goal_space:
            # Set environment with given goal 
            self.runner.env.get_source_env().goal_states = [g]
            
            # Set max time steps as optimal trajectories (consistent with A* solution)
            if self.config['use_optimal_T']:
                self.config['T'] = self.runner.env.all_steps[tuple(g)]
            
            # Evaluate
            # Collect one batch of data from runner
            batch_data = self.runner.run(self.config['T'], 
                                         self.config['eval_num_epi'], 
                                         mode='sampling')
            
            # Useful metrics
            batch_returns = [np.sum(episode.all_r) for episode in batch_data]
            average_return_all_goal.append([g, np.mean(batch_returns)])
        
        # Loggings
        key_goal = ('Sampled goal', goal_iter, tuple(goal))
        success_rate_goal_space = [success_rate for g, success_rate in average_return_all_goal]
        mean_success_rate_goal_space = np.mean(success_rate_goal_space)
        self.logger.log(self.config['ID'], 
                        [key_goal, 'Eval', 'All success rate'], 
                        average_return_all_goal)
        self.logger.log(self.config['ID'], 
                        [key_goal, 'Eval', 'Mean success rate'], 
                        mean_success_rate_goal_space)
        
        # Dump
        print('# Evaluation: ')
        print(f'\tMean success rate over goal space: {mean_success_rate_goal_space}')
        print(f'\tAll success rate over goal space: {success_rate_goal_space}')
        
    