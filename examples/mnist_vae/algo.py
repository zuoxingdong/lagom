from pathlib import Path

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from lagom import set_global_seeds
from lagom import BaseAlgorithm
from lagom import Logger

from engine import Engine
from network import VAE
from network import ConvVAE


class Algorithm(BaseAlgorithm):
    def __call__(self, config, seed, device_str):
        # Set random seeds
        set_global_seeds(seed)
        # Create device
        device = torch.device(device_str)
        # Use log dir for current job (run_experiment)
        logdir = Path(config['log.dir']) / str(config['ID']) / str(seed)
        
        # Create dataset for training and testing
        train_dataset = datasets.MNIST('data/', 
                                       train=True, 
                                       download=True, 
                                       transform=transforms.ToTensor())
        test_dataset = datasets.MNIST('data/', 
                                      train=False, 
                                      transform=transforms.ToTensor())
        # Define GPU-dependent keywords for DataLoader
        if config['cuda']:
            kwargs = {'num_workers': 1, 'pin_memory': True}
        else:
            kwargs = {}
        # Create data loader for training and testing
        train_loader = DataLoader(train_dataset, 
                                  batch_size=config['train.batch_size'], 
                                  shuffle=True, 
                                  **kwargs)
        test_loader = DataLoader(test_dataset, 
                                 batch_size=config['eval.batch_size'], 
                                 shuffle=True, 
                                 **kwargs)
        
        # Create the model
        if config['network.type'] == 'VAE':
            model = VAE(config=config)
        elif config['network.type'] == 'ConvVAE':
            model = ConvVAE(config=config)
        model = model.to(device)
        
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Create engine
        engine = Engine(agent=model,
                        runner=None,
                        config=config,
                        device=device,
                        optimizer=optimizer, 
                        train_loader=train_loader, 
                        test_loader=test_loader)
        
        # Training and evaluation
        for epoch in range(config['train.num_epoch']):
            train_output = engine.train(n=epoch)
            engine.log_train(train_output, logdir=logdir, epoch=epoch)
            
            eval_output = engine.eval(n=epoch)
            engine.log_eval(eval_output, logdir=logdir, epoch=epoch)
    
        return None
