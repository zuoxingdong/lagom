import torch
import torch.nn as nn
import torch.nn.functional as F

from lagom.core.networks import BaseVAE


class VAE(BaseVAE):
    def make_encoder(self, config):
        fc1 = nn.Linear(in_features=784, out_features=400)

        encoder = nn.ModuleList([fc1])

        return encoder

    def make_moment_heads(self, config):
        mu_head = nn.Linear(in_features=400, out_features=20)
        logvar_head = nn.Linear(in_features=400, out_features=20)

        return mu_head, logvar_head

    def make_decoder(self, config):
        fc1 = nn.Linear(in_features=20, out_features=400)
        fc2 = nn.Linear(in_features=400, out_features=784)

        decoder = nn.ModuleList([fc1, fc2])

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
        # Flatten input
        x = x.view(x.size(0), -1)
        
        for module in self.encoder:
            x = F.relu(module(x))

        return x

    def decoder_forward(self, x):
        for module in self.decoder[:-1]:
            x = F.relu(module(x))

        # Element-wise binary output
        x = torch.sigmoid(self.decoder[-1](x))

        return x