import numpy as np

import torch
from torchvision.utils import save_image

from lagom.engine import BaseEngine


class Engine(BaseEngine):
    """
    Engine for train and eval of VAE for one epoch
    """
    def __init__(self, 
                 model, 
                 optimizer, 
                 train_loader, 
                 test_loader, 
                 config, 
                 logger, 
                 device):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.logger = logger
        self.device = device
        
    def train(self):
        """
        Training for one epoch
        """
        # Set model mode to training
        self.model.train()
        
        losses = []
        
        # Iterate over batches
        for i, (data, label) in enumerate(self.train_loader):
            # Put data to defined device
            data = data.to(self.device)
            # Zero-out gradients
            self.optimizer.zero_grad()
            # Forward pass of the data
            reconstructed_x, mu, logvar = self.model(data)
            # Calculate the loss
            loss = self.model.calculate_loss(reconstructed_x, data, mu, logvar)
            losses.append(loss.item())
            # Backward pass to calcualte gradients
            loss.backward()
            # Take a gradient step
            self.optimizer.step()
            
            # Loggings
            if i == 0 or (i+1)%self.config['log_interval'] == 0:
                print(f'Training iteration #{i+1}: {loss.item()/len(data)}')
                
                """
                self.logger.log(config_ID=self.config['ID'], 
                                key_hierarchy=[('Train Iteration', i+1), 'Loss'], 
                                val=loss.item()/len(data))  # TODO: check size_average in losses
                # Dump the loggings
                self.logger.dump(config_ID=self.config['ID'], 
                                 key_hierarchy=[('Train Iteration', i+1)], 
                                 indent='')
                """
                
        print(f'====> Average loss: {np.sum(losses)/len(self.train_loader.dataset)}')
            
    def eval(self):
        """
        Evaluating for one epoch
        """
        # Set model mode to evaluation
        self.model.eval()
        
        losses = []
        
        with torch.no_grad():  # disable gradient computation to save memory
            # Iterate over batches
            for i, (data, label) in enumerate(self.test_loader):
                # Put data to defined device
                data = data.to(self.device)
                # Forward pass of the data
                reconstructed_x, mu, logvar = self.model(data)
                # Calculate the loss
                loss = self.model.calculate_loss(reconstructed_x, data, mu, logvar)
                losses.append(loss.item())
                
                # Generate reconstructed images
                if i == 0:
                    # Number of images
                    n = min(data.size(0), 8)
                    D = data[:n]
                    reconstructed_D = reconstructed_x[:n]
                    reconstructed_D = reconstructed_D.view(-1, 1, 28, 28)
                    save_image(torch.cat([D, reconstructed_D]), 
                               f'data/reconstruction.png',
                               nrow=n)
                
        print(f'====> Test set loss: {np.sum(losses)/len(self.test_loader.dataset)}')