import time
import numpy as np

from lagom import Logger
from lagom import BaseEngine
from lagom.utils import IntervalConditioner

import torch
from torchvision.utils import save_image

from model import vae_loss


class Engine(BaseEngine):
    def train(self, n=None, **kwargs):
        self.model.train()
        
        logger = Logger()
        cond = IntervalConditioner(interval=self.config['log.freq'], mode='accumulative')
        for i, (data, label) in enumerate(self.train_loader):
            t0 = time.perf_counter()
            data = data.to(self.config.device)
            
            re_x, mu, logvar = self.model(data)
            out = vae_loss(re_x, data, mu, logvar, 'BCE')
            loss = out['loss']
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            logger('epoch', n)
            logger('num_seconds', round(time.perf_counter() - t0, 1))
            logger('iteration', i)
            logger('total_loss', out['loss'].item())
            logger('reconstruction_loss', out['re_loss'].item())
            logger('KL_loss', out['KL_loss'].item())
            if cond(i):
                logger.dump(keys=None, index=-1, indent=0, border='-'*50)
        mean_loss = np.mean([logger.logs['total_loss']])
        print(f'====> Average loss: {mean_loss}')
        
        # Use decoder to sample images from standard Gaussian noise
        with torch.no_grad():  # fast, disable grad
            z = torch.randn(64, self.config['nn.z_dim']).to(self.config.device)
            re_x = self.model.decode(z).cpu()
            save_image(re_x.view(64, 1, 28, 28), self.config.logdir / f'sample_{n}.png')
        return logger
        
    def eval(self, n=None, **kwargs):
        self.model.eval()
        
        logger = Logger()
        for i, (data, label) in enumerate(self.test_loader):
            data = data.to(self.config.device)
            with torch.no_grad():
                re_x, mu, logvar = self.model(data)
                out = vae_loss(re_x, data, mu, logvar, 'BCE')
                logger('eval_loss', out['loss'].item())
        mean_loss = np.mean(logger.logs['eval_loss'])
        print(f'====> Test set loss: {mean_loss}')
        
        # Reconstruct some test images
        data, label = next(iter(self.test_loader))  # get a random batch
        num_img = min(data.size(0), 8)  # number of images
        data = data[:num_img]
        data = data.to(self.config.device)
        with torch.no_grad():
            re_x, _, _ = self.model(data)
        compare_img = torch.cat([data, re_x.view(-1, 1, 28, 28)]).cpu()
        save_image(compare_img, self.config.logdir / f'reconstruction_{n}.png', nrow=num_img)
        return logger
