import numpy as np

import torch
from torchvision.utils import save_image

from lagom import Logger

from lagom.engine import BaseEngine


class Engine(BaseEngine):
    def train(self, n=None):
        self.agent.train()
        
        logger = Logger()
        
        for i, (data, label) in enumerate(self.agent.train_loader):
            data = data.to(self.agent.device)
            
            self.agent.optimizer.zero_grad()
            re_x, mu, logvar = self.agent(data)
            out = self.agent.vae_loss(re_x=re_x, x=data, mu=mu, logvar=logvar, loss_type='BCE')
            loss = out['loss']
            loss.backward()
            self.agent.optimizer.step()
            
            logger('epoch', n)
            logger('iteration', i)
            logger('train_loss', out['loss'].item())
            logger('reconstruction_loss', out['re_loss'].item())
            logger('KL_loss', out['KL_loss'].item())
            
            if i == 0 or (i+1) % self.config['log.interval'] == 0:
                print('-'*50)
                logger.dump(keys=None, index=-1, indent=0)
                print('-'*50)
                
        return logger.logs

    def log_train(self, train_output, **kwargs):
        logdir = kwargs['logdir']
        epoch = kwargs['epoch']
        
        mean_loss = np.mean(train_output['train_loss'])
        print(f'====> Average loss: {mean_loss}')
        
        # Use decoder to sample images from standard Gaussian noise
        with torch.no_grad():  # fast, disable grad
            z = torch.randn(64, self.config['network.z_dim']).to(self.agent.device)
            re_x = self.agent.decoder(z).cpu()
            save_image(re_x.view(64, 1, 28, 28), f'{logdir}/sample_{epoch}.png')

    def eval(self, n=None):
        self.agent.eval()
        
        logger = Logger()
        
        for i, (data, label) in enumerate(self.agent.test_loader):
            data = data.to(self.agent.device)
            with torch.no_grad():
                re_x, mu, logvar = self.agent(data)
                out = self.agent.vae_loss(re_x=re_x, x=data, mu=mu, logvar=logvar, loss_type='BCE')
                logger('eval_loss', out['loss'].item())
        
        return logger.logs

    def log_eval(self, eval_output, **kwargs):
        logdir = kwargs['logdir']
        epoch = kwargs['epoch']
        
        mean_loss = np.mean(eval_output['eval_loss'])
        print(f'====> Test set loss: {mean_loss}')
        
        # Reconstruct some test images
        data, label = next(iter(self.agent.test_loader))  # get a random batch
        data = data.to(self.agent.device)
        n = min(data.size(0), 8)  # number of images
        D = data[:n]
        with torch.no_grad():
            re_x, _, _ = self.agent(D)
        compare_img = torch.cat([D.cpu(), re_x.cpu().view(-1, 1, 28, 28)])
        save_image(compare_img, f'{logdir}/reconstruction_{epoch}.png', nrow=n)
