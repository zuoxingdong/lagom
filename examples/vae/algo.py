from pathlib import Path

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from lagom import BaseAlgorithm
from lagom import Logger
from lagom.utils import set_global_seeds

from engine import Engine
from network import VAE


class Algorithm(BaseAlgorithm):
    def __call__(self, config, seed, device):
        set_global_seeds(seed)
        logdir = Path(config['log.dir']) / str(config['ID']) / str(seed)
        
        train_loader, test_loader = self.make_dataset(config)
        
        model = VAE(config=config, device=device)
        
        model.train_loader = train_loader
        model.test_loader = test_loader
        model.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        engine = Engine(agent=model, runner=None, config=config)
        
        for epoch in range(config['train.num_epoch']):
            train_output = engine.train(n=epoch)
            engine.log_train(train_output, logdir=logdir, epoch=epoch)
            
            eval_output = engine.eval(n=epoch)
            engine.log_eval(eval_output, logdir=logdir, epoch=epoch)
    
        return None
    
    def make_dataset(self, config):
        train_dataset = datasets.MNIST('data/', 
                                       train=True, 
                                       download=True, 
                                       transform=transforms.ToTensor())
        test_dataset = datasets.MNIST('data/', 
                                      train=False, 
                                      transform=transforms.ToTensor())
        if config['cuda']:
            kwargs = {'num_workers': 1, 'pin_memory': True}
        else:
            kwargs = {}
        train_loader = DataLoader(train_dataset, 
                                  batch_size=config['train.batch_size'], 
                                  shuffle=True, 
                                  **kwargs)
        test_loader = DataLoader(test_dataset, 
                                 batch_size=config['eval.batch_size'], 
                                 shuffle=True, 
                                 **kwargs)
        
        return train_loader, test_loader
