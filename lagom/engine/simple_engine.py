import numpy as np

from lagom.engine import BaseEngine


class SimpleEngine(BaseEngine):
    """
    Simple Engine: 
        Iteratively:
            1. Collect one batch of data from runner
            2. Train: One learning update of the agent
               Eval: One evaluation of the agent
    """
    def __init__(self, agent, runner, config, logger):
        super().__init__(agent, runner, config, logger)
        
    def train(self):
        for i in range(self.config['train_iter']):  # training iterations
            # Collect one batch of data from runner
            batch_data = self.runner.run(self.config['T'], self.config['train_num_epi'])
            # Update agent by learning over the batch of data
            output_learn = self.agent.learn(batch_data)
            
            # Useful metrics
            total_loss = output_learn['loss'].data[0]
            batch_returns = [np.sum(data['rewards']) for data in batch_data]
            batch_discounted_returns = [data['returns'][0] for data in batch_data]
            
            # Loggings
            if i == 0 or (i + 1) % self.config['log_interval'] == 0:
                self.logger.log(self.config['ID'], 
                                [('Train Iteration', i+1), 'Total loss'], 
                                total_loss)
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
                
    def eval(self):
        for i in range(self.config['eval_iter']):  # evaluation iterations
            # Collect one batch of data from runner
            batch_data = self.runner.run(self.config['T'], self.config['eval_num_epi'])
            
            # Useful metrics
            batch_returns = [np.sum(data['rewards']) for data in batch_data]
            batch_discounted_returns = [data['returns'][0] for data in batch_data]
    
            # Loggings
            if i == 0 or (i + 1) % self.config['log_interval'] == 0:
                self.logger.log(self.config['ID'], 
                                [('Eval Iteration', i+1), 'Num Episodes'], 
                                len(batch_data))
                self.logger.log(self.config['ID'], 
                                [('Eval Iteration', i+1), 'Average Return'], 
                                np.mean(batch_returns))
                self.logger.log(self.config['ID'], 
                                [('Eval Iteration', i+1), 'Average Discounted Return'], 
                                np.mean(batch_discounted_returns))
                self.logger.log(self.config['ID'], 
                                [('Eval Iteration', i+1), 'Std Return'], 
                                np.std(batch_returns))
                self.logger.log(self.config['ID'], 
                                [('Eval Iteration', i+1), 'Min Return'], 
                                np.min(batch_returns))
                self.logger.log(self.config['ID'], 
                                [('Eval Iteration', i+1), 'Max Return'], 
                                np.max(batch_returns))
                
                # Dump the loggings
                self.logger.dump(self.config['ID'], 
                                 [('Eval Iteration', i+1)], 
                                 indent='')
