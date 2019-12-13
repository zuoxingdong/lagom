import torch.optim as optim


def linear_lr_scheduler(optimizer, N, min_lr):
    r"""Defines a linear learning rate scheduler. 
    
    Args:
        optimizer (Optimizer): optimizer
        N (int): maximum bounds for the scheduling iteration
            e.g. total number of epochs, iterations or time steps. 
        min_lr (float): lower bound of learning rate
    """
    initial_lr = optimizer.defaults['lr']
    f = lambda n: max(min_lr/initial_lr, 1 - n/N)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
    return lr_scheduler
