import torch
import torch.nn as nn
import torch.nn.functional as F

from lagom.core.networks import make_fc
from lagom.core.networks import make_cnn
from lagom.core.networks import make_transposed_cnn

from lagom.core.networks import ortho_init

from lagom.core.networks import BaseVAE


class VAE(BaseVAE):
    def make_encoder(self, config):
        last_dim = 400
        out = make_fc(input_dim=784, hidden_sizes=[last_dim])
        
        return out, last_dim

    def make_moment_heads(self, config, last_dim):
        out = {}
        
        z_dim = config['network.z_dim']
        
        out['mu_head'] = nn.Linear(in_features=last_dim, out_features=z_dim)
        out['logvar_head'] = nn.Linear(in_features=last_dim, out_features=z_dim)
        out['z_dim'] = z_dim

        return out

    def make_decoder(self, config, z_dim):
        out = make_fc(input_dim=z_dim, hidden_sizes=[self.last_dim, 784])

        return out

    def init_params(self, config):
        for layer in self.encoder:
            ortho_init(layer, nonlinearity='relu', constant_bias=0.0)
            
        ortho_init(self.mu_head, nonlinearity=None, weight_scale=0.01, constant_bias=0.0)
        ortho_init(self.logvar_head, nonlinearity=None, weight_scale=0.01, constant_bias=0.0)
        
        for layer in self.decoder:
            ortho_init(layer, nonlinearity='relu', constant_bias=0.0)

    def encoder_forward(self, x):
        # flatten input
        x = x.flatten(start_dim=1)
        
        for layer in self.encoder:
            x = F.relu(layer(x))

        return x

    def decoder_forward(self, z):
        # decoding until last layer
        for layer in self.decoder[:-1]:
            z = F.relu(layer(z))
            
        # Elementwise binary output (BCE loss)
        x = torch.sigmoid(self.decoder[-1](z))
        
        return x


class ConvVAE(BaseVAE):
    def make_encoder(self, config):
        out = make_cnn(input_channel=1, 
                       channels=[64, 64, 64], 
                       kernels=[4, 4, 4], 
                       strides=[2, 2, 1], 
                       paddings=[0, 0, 0])
        last_dim = 256
        
        return out, last_dim
        
    def make_moment_heads(self, config, last_dim):
        out = {}
        
        z_dim = config['network.z_dim']
        
        out['mu_head'] = nn.Linear(in_features=last_dim, out_features=z_dim)
        out['logvar_head'] = nn.Linear(in_features=last_dim, out_features=z_dim)
        out['z_dim'] = z_dim
        
        return out

    def make_decoder(self, config, z_dim):
        out = nn.ModuleList()
        
        out.append(nn.Linear(in_features=z_dim, out_features=self.last_dim))
        
        out.extend(make_transposed_cnn(input_channel=64, 
                                       channels=[64, 64, 64], 
                                       kernels=[4, 4, 4], 
                                       strides=[2, 1, 1], 
                                       paddings=[0, 0, 0], 
                                       output_paddings=[0, 0, 0]))
        
        out.append(nn.Linear(in_features=9216, out_features=28*28*1))
        
        return out

    def init_params(self, config):
        for layer in self.encoder:
            ortho_init(layer, nonlinearity='relu', constant_bias=0.0)
            
        ortho_init(self.mu_head, nonlinearity=None, weight_scale=0.01, constant_bias=0.0)
        ortho_init(self.logvar_head, nonlinearity=None, weight_scale=0.01, constant_bias=0.0)
        
        for layer in self.decoder:
            ortho_init(layer, nonlinearity='relu', constant_bias=0.0)

    def encoder_forward(self, x):
        for layer in self.encoder:
            x = F.relu(layer(x))

        return x

    def decoder_forward(self, z):
        # Forward of first fully-connected layer
        x = F.relu(self.decoder[0](z))
        
        # Reshape as [NxCxHxW]
        x = x.view(-1, 64, 2, 2)
        
        # Forward pass through transposed convolutional layer
        for layer in self.decoder[1:-1]:
            x = F.relu(layer(x))
            
        # Flatten to [N, D]
        x = x.flatten(start_dim=1)

        # Element-wise binary output
        x = torch.sigmoid(self.decoder[-1](x))

        return x
