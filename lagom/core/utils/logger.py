import logging

import os
import shutil

from collections import OrderedDict

import numpy as np

class Logger(logging.Logger):
    def __init__(self, logger_name='logger', log_dir='logs/', use_screen=True):
        super().__init__(logger_name)
        
        self.use_screen = use_screen
        self.log_dir = log_dir
        # Create logging directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        # Create storage as key-value pairs
        self.loginfo = {}
        
        # Create storage for loggings of one training iteration
        self.log_train = OrderedDict()  # key: logging name
        
    def log_info(self, name, val):
        self.loginfo[name] = val
        
    def log_list(self, name, val):
        if name not in self.loginfo:  # initialize the list if not existed
            self.loginfo[name] = []
        # Append a new value to the list
        self.loginfo[name].append(val)
        
    def log_train_iter(self, iter_id, data_batch, losses, num_episodes):
        batch_return = [np.sum(data['rewards']) for data in data_batch]
        average_return = np.mean(batch_return)
        std_return = np.std(batch_return)
        min_return = np.min(batch_return)
        max_return = np.max(batch_return)
        loss = losses['total_loss'].data[0]
        
        iter_aligner = '\t{:<30}'  # align items in each iteration to the left
        self.log_train['Iteration'] = 'Iteration # {:<d}'.format(iter_id)
        self.log_train['Loss'] = (iter_aligner + '{:<f}').format('Loss', losses['total_loss'].data[0])
        self.log_train['Num Episodes'] = (iter_aligner + '{:<d}').format('Num Episodes', num_episodes)
        self.log_train['Average Return'] = (iter_aligner + '{:<f}').format('Average Return', average_return)
        self.log_train['Std Return'] = (iter_aligner + '{:<f}').format('Std Return', std_return)
        self.log_train['Min Return'] = (iter_aligner + '{:<f}').format('Min Return', min_return)
        self.log_train['Max Return'] = (iter_aligner + '{:<f}').format('Max Return', max_return)
        
    def dump_train_iter(self):
        if self.use_screen:
            print('{:->50}'.format(' '))  # separation line between each iteration
            for log in self.log_train.values():
                print(log)
        
        # Clear the storage for this training iteration
        self.log_train.clear()
        
    def get(self, name):
        return self.loginfo[name]
            
    def remove(self, name):
        self.loginfo.pop(name)
        
    def clear(self):
        self.loginfo.clear()
            
    def delete_log_dir(self):
        """
        Delete everything in logging directory
        """
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)