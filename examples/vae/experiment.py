import os

import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import lagom
from lagom.utils import set_global_seeds
from lagom.utils import pickle_dump
from lagom.experiment import Grid
from lagom.experiment import Config
from lagom.experiment import Configurator
from lagom.experiment import checkpointer
from lagom.experiment import run_experiment

from engine import Engine
from model import VAE
from model import ConvVAE


configurator = Configurator(
    {'log.freq': 100, 
     
     'nn.type': Grid(['VAE', 'ConvVAE']),
     'nn.z_dim': 8,
     
     'lr': 1e-3,
     
     'train.num_epoch': 100,
     'train.batch_size': 128, 
     'eval.batch_size': 128
    }, 
    num_sample=1)


def make_dataset(config, train):
    dataset = datasets.MNIST('data/', 
                             train=train, 
                             download=True, 
                             transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=config['train.batch_size'], shuffle=True)
    return dataloader


def run(config):
    set_global_seeds(config.seed)

    train_loader = make_dataset(config, train=True)
    test_loader = make_dataset(config, train=False)
    if config['nn.type'] == 'VAE':
        model = VAE(config).to(config.device)
    elif config['nn.type'] == 'ConvVAE':
        model = ConvVAE(config).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    engine = Engine(config, 
                    model=model, 
                    optimizer=optimizer,
                    train_loader=train_loader, 
                    test_loader=test_loader)
    
    train_logs, eval_logs = [], []
    start_epoch = 0
    if config.resume_checkpointer.exists():
        out = checkpointer('load', config, model=model, optimizer=optimizer, train_logs=train_logs, eval_logs=eval_logs, start_epoch=start_epoch)
        train_logs = out['train_logs']
        eval_logs = out['eval_logs']
        start_epoch = out['start_epoch']
    
    for epoch in range(start_epoch, config['train.num_epoch']):
        train_logger = engine.train(epoch)
        train_logs.append(train_logger.logs)
        eval_logger = engine.eval(epoch)
        eval_logs.append(eval_logger.logs)
        pickle_dump(obj=train_logs, f=config.logdir/'train_logs', ext='.pkl')
        pickle_dump(obj=eval_logs, f=config.logdir/'eval_logs', ext='.pkl')
        checkpointer('save', config, model=model, optimizer=optimizer, train_logs=train_logs, eval_logs=eval_logs, start_epoch=epoch+1)
    return None


if __name__ == '__main__':
    run_experiment(run=run, 
                   configurator=configurator, 
                   seeds=lagom.SEEDS[:1],
                   log_dir='logs/default',
                   max_workers=os.cpu_count(),
                   chunksize=1, 
                   use_gpu=False,  # GPU much faster
                   gpu_ids=None)
