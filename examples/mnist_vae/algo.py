import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from lagom import set_global_seeds
from lagom import BaseAlgorithm
from lagom.core.utils import Logger

from engine import Engine
from vae import VAE
from conv_vae import ConvVAE


class Algorithm(BaseAlgorithm):
    def __call__(self, config):
        # Set random seeds
        set_global_seeds(config['seed'])
        
        # Create device
        device = torch.device('cuda' if config['cuda'] else 'cpu')
        
        # Define GPU-dependent keywords for DataLoader
        if config['cuda']:
            kwargs = {'num_workers': 1, 'pin_memory': True}
        else:
            kwargs = {}

        # Create dataset for training and testing
        train_dataset = datasets.MNIST('data/', 
                                       train=True, 
                                       download=True, 
                                       transform=transforms.ToTensor())
        test_dataset = datasets.MNIST('data/', 
                                      train=False, 
                                      transform=transforms.ToTensor())
        # Create data loader for training and testing
        train_loader = DataLoader(train_dataset, 
                                  batch_size=config['batch_size'], 
                                  shuffle=True, 
                                  **kwargs)
        test_loader = DataLoader(test_dataset, 
                                 batch_size=config['batch_size'], 
                                 shuffle=True, 
                                 **kwargs)
        
        # Create the model
        if config['use_ConvVAE']:
            model = ConvVAE(config=None)
        else:
            model = VAE(config=None)
        model = model.to(device)
        
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Create logger
        logger = Logger(name='logger')
        
        # Create engine
        engine = Engine(model=model, 
                        optimizer=optimizer, 
                        train_loader=train_loader, 
                        test_loader=test_loader, 
                        config=config, 
                        logger=logger, 
                        device=device)
        
        # Trainning and testing
        for epoch in range(config['num_epochs']):
            print('#'*20)
            print(f'# Epoch: {epoch+1}')
            print('#'*20)
            engine.train()
            engine.eval()
            
            # Sample image from standard Gaussian noise as input to decoder
            with torch.no_grad():
                sample = torch.randn(64, 8).to(device)
                sample = model.decoder_forward(sample).cpu()
                
                save_image(sample.view(64, 1, 28, 28),
                           f'data/sample_{epoch}.png')
            
        # Save the logger
        #logger.save(name=f'{self.name}_ID_{config["ID"]}')
        
        return None