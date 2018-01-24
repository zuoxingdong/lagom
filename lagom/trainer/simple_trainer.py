import numpy as np

from lagom.trainer import BaseTrainer


class SimpleTrainer(BaseTrainer):
    """Simple training: iteratively collect data batch from runner -> update agent"""
    def __init__(self, agent, runner, args, logger=None):
        self.agent = agent
        self.runner = runner
        self.args = args
        self.logger = logger
        
        super().__init__(self.agent, self.runner, self.args, self.logger)
        
    def train(self):
        for iter_num in range(self.args.num_iter):  # training iterations
            # Collect one data batch
            data_batch = self.runner.run(self.args.T, self.args.num_episodes)
            # Update agent by learning over data batch
            losses = self.agent.learn(data_batch, standardize_r=self.args.standardize_r)
            
            # Loggings
            if self.logger is not None:
                if iter_num == 0 or (iter_num + 1) % self.args.log_interval == 0:
                    self.logger.log_metric('Loss', 
                                           losses['total_loss'].data[0], 
                                           iter_num + 1)
                    self.logger.log_metric('Num Episodes', 
                                           self.args.num_episodes,
                                           iter_num + 1)
                    
                    batch_return = [np.sum(data['rewards']) for data in data_batch]
                    self.logger.log_metric('Average Return', 
                                           np.mean(batch_return),
                                           iter_num + 1)
                    self.logger.log_metric('Std Return', 
                                           np.std(batch_return),
                                           iter_num + 1)
                    self.logger.log_metric('Min Return', 
                                           np.min(batch_return),
                                           iter_num + 1)
                    self.logger.log_metric('Max Return', 
                                           np.max(batch_return),
                                           iter_num + 1)
                    
                    # Dump all the loggings
                    self.logger.dump_metric(iter_num + 1)
    
        