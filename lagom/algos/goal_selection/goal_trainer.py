import numpy as np

from lagom.core.utils import Logger
from lagom.trainer import BaseTrainer
from lagom.trainer import SimpleTrainer


class GoalTrainer(BaseTrainer):
    def train(self, goal):
        # Set environment with sampled goal 
        self.runner.env.goal_states = [goal]
        
        # Train
        train_logger = Logger(path=self.logger.path, dump_mode=['screen'])
        trainer = SimpleTrainer(self.agent, self.runner, self.args, train_logger)
        trainer.train()
        
        
        
        
        
        
