from lagom.evaluator import BaseEvaluator


class SimpleEvaluator(BaseEvaluator):
    """Simple evaluation: collect one data batch from runner with the current agent"""
    def __init__(self, agent, runner, args, logger=None):
        self.agent = agent
        self.runner = runner
        self.args = args
        self.logger = logger
        
        super().__init__(self.agent, self.runner, self.args, self.logger)
        
    def evaluate(self):
        # Collect one data batch
        data_batch = self.runner.run(self.args.T, self.args.num_episodes)

        # Loggings
            if self.logger is not None:
                if iter_num == 0 or (iter_num + 1) % self.args.log_interval == 0:
                    self.logger.log_metric('Num Episodes', 
                                           [self.args.num_episodes, '{:<d}'], 
                                           iter_num + 1)
                    
                    batch_return = [np.sum(data['rewards']) for data in data_batch]
                    self.logger.log_metric('Average Return', 
                                           [np.mean(batch_return), '{:<f}'], 
                                           iter_num + 1)
                    self.logger.log_metric('Std Return', 
                                           [np.std(batch_return), '{:<f}'], 
                                           iter_num + 1)
                    self.logger.log_metric('Min Return', 
                                           [np.min(batch_return), '{:<f}'], 
                                           iter_num + 1)
                    self.logger.log_metric('Max Return', 
                                           [np.max(batch_return), '{:<f}'], 
                                           iter_num + 1)
                    
                    # Dump all the loggings
                    self.logger.dump_metric(iter_num + 1)
    