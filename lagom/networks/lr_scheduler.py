import torch.optim as optim


def linear_lr_scheduler(optimizer, max_epoch, mode):
    r"""Defines a linear learning rate scheduler. 
    
    Args:
        optimizer (Optimizer): optimizer
        max_epoch (int): maximum bounds for the scheduling iteration. 
        mode (str): mode of scheduling. ['iteration-based', 'timestep-based']
    """
    assert mode in ['iteration-based', 'timestep-based']
    
    if mode == 'iteration-based':
        max_epoch = max_epoch
    elif mode == 'timestep-based':
        max_epoch = max_epoch + 1  # avoid zero lr in final iteration
        
    lambda_f = lambda epoch: 1 - epoch/max_epoch
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_f)
    
    return lr_scheduler
