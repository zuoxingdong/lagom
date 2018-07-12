import torch
import torch.nn as nn
import torch.nn.functional as F

from lagom.core.networks import BaseVAE


class ConvVAE(BaseVAE):
    def make_encoder(self, config):
        conv1 = nn.Conv2d(in_channels=1, 
                          out_channels=64, 
                          kernel_size=4, 
                          stride=2)
        conv2 = nn.Conv2d(in_channels=64, 
                          out_channels=64, 
                          kernel_size=4, 
                          stride=2)
        conv3 = nn.Conv2d(in_channels=64, 
                          out_channels=64, 
                          kernel_size=4, 
                          stride=1)
        
        encoder = nn.ModuleList([conv1, conv2, conv3])

        return encoder

    def make_moment_heads(self, config):
        mu_head = nn.Linear(in_features=256, out_features=8)
        logvar_head = nn.Linear(in_features=256, out_features=8)

        return mu_head, logvar_head

    def make_decoder(self, config):
        fc1 = nn.Linear(in_features=8, out_features=256)
        trans_conv1 = nn.ConvTranspose2d(in_channels=64, 
                                         out_channels=64, 
                                         kernel_size=4, 
                                         stride=2)
        trans_conv2 = nn.ConvTranspose2d(in_channels=64, 
                                         out_channels=64, 
                                         kernel_size=4, 
                                         stride=1)
        trans_conv3 = nn.ConvTranspose2d(in_channels=64, 
                                         out_channels=64, 
                                         kernel_size=4, 
                                         stride=1)
        fc2 = nn.Linear(in_features=9216, out_features=28*28*1)

        decoder = nn.ModuleList([fc1, trans_conv1, trans_conv2, trans_conv3, fc2])

        return decoder

    def init_params(self, config):
        gain = nn.init.calculate_gain('relu')

        for module in self.encoder:
            nn.init.orthogonal_(module.weight, gain=gain)
            nn.init.constant_(module.bias, 0.0)

        nn.init.orthogonal_(self.mu_head.weight, gain=gain)
        nn.init.constant_(self.mu_head.bias, 0.0)
        nn.init.orthogonal_(self.logvar_head.weight, gain=gain)
        nn.init.constant_(self.logvar_head.bias, 0.0)

        for module in self.decoder:
            nn.init.orthogonal_(module.weight, gain=gain)
            nn.init.constant_(module.bias, 0.0)

    def encoder_forward(self, x):
        for module in self.encoder:
            x = F.relu(module(x))

        return x

    def decoder_forward(self, x):
        # Forward of first fully-connected layer
        x = F.relu(self.decoder[0](x))
        
        # Reshape as [NxCxHxW]
        x = x.view(x.size(0), 64, 2, 2)
        
        # Forward pass through transposed convolutional layer
        for module in self.decoder[1:-1]:
            x = F.relu(module(x))
            
        # Flatten to [N, D]
        x = x.view(x.size(0), -1)

        # Element-wise binary output
        x = torch.sigmoid(self.decoder[-1](x))

        return x