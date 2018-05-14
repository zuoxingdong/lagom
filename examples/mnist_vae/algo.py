import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from lagom import BaseAlgorithm
from lagom.core.networks import VAE, ConvVAE
from lagom.core.utils import Logger

from engine import Engine


class Algorithm(BaseAlgorithm):
    def run(self, config):
        # Set random seeds
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        
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

        train_loader = DataLoader(train_dataset, 
                                  batch_size=config['batch_size'], 
                                  shuffle=True, 
                                  **kwargs)
        test_loader = DataLoader(test_dataset, 
                                 batch_size=config['batch_size'], 
                                 shuffle=True, 
                                 **kwargs)
        
        # Create the model
        """
        model = VAE(input_dim=784, 
                    encoder_sizes=[400], 
                    encoder_nonlinearity=F.relu, 
                    latent_dim=20, 
                    decoder_sizes=[400], 
                    decoder_nonlinearity=F.relu)
        """
        model = ConvVAE(input_channel=1,
                        input_shape=[28, 28],
                        encoder_conv_kernels=[16, 8],
                        encoder_conv_kernel_sizes=[3, 3],
                        encoder_conv_strides=[3, 2], 
                        encoder_conv_pads=[1, 1], 
                        encoder_conv_nonlinearity=F.relu, 
                        encoder_hidden_sizes=[64], 
                        encoder_hidden_nonlinearity=F.relu, 
                        latent_dim=20, 
                        decoder_hidden_sizes=[8*2*2],
                        decoder_hidden_nonlinearity=F.relu, 
                        decoder_conv_trans_input_channel=8,
                        decoder_conv_trans_input_shape=[2, 2],
                        decoder_conv_trans_kernels=[16, 8, 1],
                        decoder_conv_trans_kernel_sizes=[3, 5, 2],
                        decoder_conv_trans_strides=[2, 3, 2], 
                        decoder_conv_trans_pads=[0, 1, 1], 
                        decoder_conv_trans_nonlinearity=F.relu,
                        decoder_output_nonlinearity=None)
        
        model = model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        logger = Logger(name='logger')
        engine = Engine(model, 
                        optimizer, 
                        train_loader, 
                        test_loader, 
                        config, 
                        logger, 
                        device)
        
        for epoch in range(config['num_epochs']):
            engine.train()
            engine.eval()
            
            with torch.no_grad():
                sample = torch.randn(64, 20).to(device)
                sample = model.decode(sample).cpu()
                
                save_image(sample.view(64, 1, 28, 28),
                           f'data/sample_{epoch}.png')
            
        # Save the logger
        logger.save(name=f'{self.name}_ID_{config["ID"]}')