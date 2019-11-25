import torch
import torch.nn as nn
import torch.nn.functional as F

from lagom.networks import Module
from lagom.networks import make_fc
from lagom.networks import make_cnn
from lagom.networks import make_transposed_cnn
from lagom.networks import ortho_init


class VAE(Module):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        self.encoder = make_fc(784, [400])
        for layer in self.encoder:
            ortho_init(layer, nonlinearity='relu', constant_bias=0.0)
            
        self.mean_head = nn.Linear(400, config['nn.z_dim'])
        ortho_init(self.mean_head, weight_scale=0.01, constant_bias=0.0)
        self.logvar_head = nn.Linear(400, config['nn.z_dim'])
        ortho_init(self.logvar_head, weight_scale=0.01, constant_bias=0.0)
        
        self.decoder = make_fc(config['nn.z_dim'], [400])
        for layer in self.decoder:
            ortho_init(layer, nonlinearity='relu', constant_bias=0.0)
        self.x_head = nn.Linear(400, 784)
        ortho_init(self.x_head, nonlinearity='sigmoid', constant_bias=0.0)

    def encode(self, x):
        for layer in self.encoder:
            x = F.relu(layer(x))
        mu = self.mean_head(x)
        logvar = self.logvar_head(x)
        return mu, logvar 
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def decode(self, z):
        for layer in self.decoder:
            z = F.relu(layer(z))
        re_x = torch.sigmoid(self.x_head(z))
        return re_x
        
    def forward(self, x):
        x = x.flatten(start_dim=1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        re_x = self.decode(z)
        return re_x, mu, logvar

        
class ConvVAE(Module):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        self.encoder = make_cnn(input_channel=1, 
                                channels=[64, 64, 64], 
                                kernels=[4, 4, 4], 
                                strides=[2, 2, 1], 
                                paddings=[0, 0, 0])
        for layer in self.encoder:
            ortho_init(layer, nonlinearity='relu', constant_bias=0.0)
        
        self.mean_head = nn.Linear(256, config['nn.z_dim'])
        ortho_init(self.mean_head, weight_scale=0.01, constant_bias=0.0)
        self.logvar_head = nn.Linear(256, config['nn.z_dim'])
        ortho_init(self.logvar_head, weight_scale=0.01, constant_bias=0.0)
        
        self.decoder_fc = nn.Linear(config['nn.z_dim'], 256)
        ortho_init(self.decoder_fc, nonlinearity='relu', constant_bias=0.0)
        self.decoder = make_transposed_cnn(input_channel=64, 
                                           channels=[64, 64, 64], 
                                           kernels=[4, 4, 4], 
                                           strides=[2, 1, 1], 
                                           paddings=[0, 0, 0], 
                                           output_paddings=[0, 0, 0])
        for layer in self.decoder:
            ortho_init(layer, nonlinearity='relu', constant_bias=0.0)
        self.x_head = nn.Linear(9216, 784)
        ortho_init(self.x_head, nonlinearity='sigmoid', constant_bias=0.0)
        
    def encode(self, x):
        for layer in self.encoder:
            x = F.relu(layer(x))
        # To shape [N, D]
        x = x.flatten(start_dim=1)
        mu = self.mean_head(x)
        logvar = self.logvar_head(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        z = self.decoder_fc(z)
        z = z.view(-1, 64, 2, 2)
        for layer in self.decoder:
            z = F.relu(layer(z))
        z = z.flatten(start_dim=1)
        re_x = torch.sigmoid(self.x_head(z))
        return re_x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        re_x = self.decode(z)
        return re_x, mu, logvar
        
    
def vae_loss(re_x, x, mu, logvar, mode='BCE'):
    r"""Calculate `VAE loss function`_. 
        
    The VAE loss is the summation of reconstruction loss and KL loss. The KL loss
    is presented in Appendix B. 
        
    .. _VAE loss function:
        https://arxiv.org/abs/1312.6114
        
    Args:
        re_x (Tensor): reconstructed input returned from decoder
        x (Tensor): ground-truth input
        mu (Tensor): mean of the latent variable
        logvar (Tensor): log-variance of the latent variable
        mode (str): Type of reconstruction loss, supported ['BCE', 'MSE']
        
    Returns:
        dict: a dictionary of selected output such as loss, reconstruction loss and KL loss. 
    """
    assert mode in ['BCE', 'MSE'], f'expected either BCE or MSE, got {mode}'
    
    # shape [N, D]
    x = x.view_as(re_x)
    if mode == 'BCE':
        f = F.binary_cross_entropy
    elif mode == 'MSE':
        f = F.mse_loss
    re_loss = f(input=re_x, target=x, reduction='none')
    re_loss = re_loss.sum(1)
    KL_loss = -0.5*torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1)
    loss = re_loss + KL_loss
    
    out = {}
    out['loss'] = loss.mean()  # average over the batch
    out['re_loss'] = re_loss.mean()
    out['KL_loss'] = KL_loss.mean()
    return out
