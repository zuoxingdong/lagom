import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import lagom
import lagom.utils as utils

from engine import Engine
from model import VAE, ConvVAE


configurator = lagom.Configurator(
    {'log.freq': 100, 
     
     'nn.type': lagom.Grid(['VAE', 'ConvVAE']),
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
    lagom.set_global_seeds(config.seed)

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
        train_logs, eval_logs, start_epoch = lagom.checkpointer('load', config, state_obj=[model, optimizer])

    for epoch in range(start_epoch, config['train.num_epoch']):
        train_logger = engine.train(epoch)
        train_logs.append(train_logger.logs)
        eval_logger = engine.eval(epoch)
        eval_logs.append(eval_logger.logs)
        utils.pickle_dump(obj=train_logs, f=config.logdir/'train_logs', ext='.pkl')
        utils.pickle_dump(obj=eval_logs, f=config.logdir/'eval_logs', ext='.pkl')
        lagom.checkpointer('save', config, obj=[train_logs, eval_logs, epoch+1], state_obj=[model, optimizer])
    return None


if __name__ == '__main__':
    lagom.run_experiment(run=run, 
                         configurator=configurator, 
                         seeds=lagom.SEEDS[:1],
                         log_dir='logs/default',
                         max_workers=None,
                         chunksize=1, 
                         use_gpu=False,  # GPU much faster
                         gpu_ids=None)
