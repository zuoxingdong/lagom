from time import perf_counter

import numpy as np

from lagom import Logger
from lagom import BaseEngine
from lagom.utils import color_str

import torch
from torchvision.utils import save_image

from model import vae_loss


class Engine(BaseEngine):
    def train(self, n=None, **kwargs):
        self.model.train()
        
        logger = Logger()
        for i, (data, label) in enumerate(self.train_loader):
            start_time = perf_counter()
            data = data.to(self.model.device)
            re_x, mu, logvar = self.model(data)
            out = vae_loss(re_x, data, mu, logvar, 'BCE')
            loss = out['loss']
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            logger('epoch', n)
            self.model.total_iter += 1
            logger('iteration', self.model.total_iter)
            logger('mini-batch', i)
            logger('train_loss', out['loss'].item())
            logger('reconstruction_loss', out['re_loss'].item())
            logger('KL_loss', out['KL_loss'].item())
            logger('num_seconds', round(perf_counter() - start_time, 1))
            if i == 0 or (i+1) % self.config['log.freq'] == 0:
                logger.dump(keys=None, index=-1, indent=0, border='-'*50)
        mean_loss = np.mean([logger.logs['train_loss']])
        print(f'====> Average loss: {mean_loss}')
        
        # Use decoder to sample images from standard Gaussian noise
        with torch.no_grad():  # fast, disable grad
            z = torch.randn(64, self.config['nn.z_dim']).to(self.model.device)
            re_x = self.model.decode(z).cpu()
            save_image(re_x.view(64, 1, 28, 28), f'{kwargs["logdir"]}/sample_{n}.png')
        return logger
        
    def eval(self, n=None, **kwargs):
        self.model.eval()
        
        logger = Logger()
        for i, (data, label) in enumerate(self.test_loader):
            data = data.to(self.model.device)
            with torch.no_grad():
                re_x, mu, logvar = self.model(data)
                out = vae_loss(re_x, data, mu, logvar, 'BCE')
                logger('eval_loss', out['loss'].item())
        mean_loss = np.mean(logger.logs['eval_loss'])
        print(f'====> Test set loss: {mean_loss}')
        
        # Reconstruct some test images
        data, label = next(iter(self.test_loader))  # get a random batch
        data = data.to(self.model.device)
        m = min(data.size(0), 8)  # number of images
        D = data[:m]
        with torch.no_grad():
            re_x, _, _ = self.model(D)
        compare_img = torch.cat([D.cpu(), re_x.cpu().view(-1, 1, 28, 28)])
        save_image(compare_img, f'{kwargs["logdir"]}/reconstruction_{n}.png', nrow=m)
        return logger
