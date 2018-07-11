import numpy as np

from lagom.engine import BaseEngine

    
class Engine(BaseEngine):
    def train(self, train_iter):
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
            if i == 0 or (i + 1) % self.config['log_interval'] == 0:
                self.logger.log(self.config['ID'], 
                                [('Train Iteration', i+1), 'Loss'], 
                                loss)
                self.logger.log(self.config['ID'], 
                                [('Train Iteration', i+1), 'Num Episodes'], 
                                len(batch_data))
                self.logger.log(self.config['ID'], 
                                [('Train Iteration', i+1), 'Average Return'], 
                                np.mean(batch_returns))
                self.logger.log(self.config['ID'], 
                                [('Train Iteration', i+1), 'Average Discounted Return'], 
                                np.mean(batch_discounted_returns))
                self.logger.log(self.config['ID'], 
                                [('Train Iteration', i+1), 'Std Return'], 
                                np.std(batch_returns))
                self.logger.log(self.config['ID'], 
                                [('Train Iteration', i+1), 'Min Return'], 
                                np.min(batch_returns))
                self.logger.log(self.config['ID'], 
                                [('Train Iteration', i+1), 'Max Return'], 
                                np.max(batch_returns))
                
                # Dump the loggings
                self.logger.dump(self.config['ID'], 
                                 [('Train Iteration', i+1)], 
                                 indent='')
        
    def eval(self, goal_iter, goal):
        pass
        
    