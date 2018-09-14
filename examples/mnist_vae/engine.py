import numpy as np

import torch
from torchvision.utils import save_image

from lagom import Logger

from lagom.engine import BaseEngine


class Engine(BaseEngine):
    def train(self, n=None):
        self.agent.train()  # set to training mode
        
        # Create a logger
        train_output = Logger()
        
        # Iterate over data batches for one epoch
        for i, (data, label) in enumerate(self.train_loader):
            # Put data to device
            data = data.to(self.device)
            # Zero-out gradient buffer
            self.optimizer.zero_grad()
            # Forward pass of data
            re_x, mu, logvar = self.agent(data)
            # Calculate loss
            out = self.agent.calculate_loss(re_x=re_x, x=data, mu=mu, logvar=logvar, loss_type='BCE')
            loss = out['loss']
            # Backward pass to calcualte gradients
            loss.backward()
            # Take a gradient step
            self.optimizer.step()
            
            # Record train output
            train_output.log('epoch', n)
            train_output.log('iteration', i)
            train_output.log('train_loss', out['loss'].item())  # item() saves memory
            train_output.log('reconstruction_loss', out['re_loss'].item())
            train_output.log('KL_loss', out['KL_loss'].item())
            
            # Dump logging
            if i == 0 or (i+1) % self.config['log.interval'] == 0:
                print('-'*50)
                train_output.dump(keys=None, index=-1, indent=0)
                print('-'*50)
                
        return train_output.logs

    def log_train(self, train_output, **kwargs):
        logdir = kwargs['logdir']
        epoch = kwargs['epoch']
        
        mean_loss = np.mean(train_output['train_loss'])
        print(f'====> Average loss: {mean_loss}')
        
        # Use decoder to sample images from standard Gaussian noise
        with torch.no_grad():  # fast, disable grad
            z = torch.randn(64, self.config['network.z_dim']).to(self.device)
            re_x = self.agent.decoder_forward(z).cpu()
            save_image(re_x.view(64, 1, 28, 28), f'{logdir}/sample_{epoch}.png')

    def eval(self, n=None):
        self.agent.eval()  # set to evaluation mode
        
        # Create a logger
        eval_output = Logger()
        
        # Iterate over test batches
        for i, (data, label) in enumerate(self.test_loader):
            # Put data to device
            data = data.to(self.device)
            with torch.no_grad():  # fast, disable grad
                # Forward pass of data
                re_x, mu, logvar = self.agent(data)
                # Calculate loss
                out = self.agent.calculate_loss(re_x=re_x, x=data, mu=mu, logvar=logvar, loss_type='BCE')
            
            # Record eval output
            eval_output.log('eval_loss', out['loss'].item())
        
        return eval_output.logs

    def log_eval(self, eval_output, **kwargs):
        logdir = kwargs['logdir']
        epoch = kwargs['epoch']
        
        mean_loss = np.mean(eval_output['eval_loss'])
        print(f'====> Test set loss: {mean_loss}')
        
        # Reconstruct some test images
        data, label = next(iter(self.test_loader))  # get a random batch
        data = data.to(self.device)
        n = min(data.size(0), 8)  # number of images
        D = data[:n]
        with torch.no_grad():  # fast, disable grad
            re_x, _, _ = self.agent(D)
        compare_img = torch.cat([D.cpu(), re_x.cpu().view(-1, 1, 28, 28)])
        save_image(compare_img, f'{logdir}/reconstruction_{epoch}.png', nrow=n)
