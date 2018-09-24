import torch.nn as nn


def ortho_init(module, nonlinearity=None, weight_scale=1.0, constant_bias=0.0):
    r"""Applies orthogonal initialization for the parameters of a given module.
    
    Args:
        module (nn.Module): A module to apply orthogonal initialization over its parameters. 
        nonlinearity (str, optional): Nonlinearity followed by forward pass of the module. When nonlinearity
            is not ``None``, the gain will be calculated and :attr:`weight_scale` will be ignored. 
            Default: ``None``
        weight_scale (float, optional): Scaling factor to initialize the weight. Ignored when
            :attr:`nonlinearity` is not ``None``. Default: 1.0
        constant_bias (float, optional): Constant value to initialize the bias. Default: 0.0
        
    .. note::
    
        Currently, the only supported :attr:`module` are elementary neural network layers, e.g.
        nn.Linear, nn.Conv2d, nn.LSTM. The submodules are not supported.
    
    Example::
    
        >>> a = nn.Linear(2, 3)
        >>> ortho_init(a)
    
    """
    # Get the gain (scaling factor)
    if nonlinearity is not None:  # based on nonlinearity
        gain = nn.init.calculate_gain(nonlinearity)
    else:  # user provided
        gain = weight_scale
        
    # Initialization
    if isinstance(module, (nn.RNNBase, nn.RNNCellBase)):  # RNN
        # Iterate over named parameters
        for name, param in module.named_parameters():
            if 'weight' in name:  # Weight
                nn.init.orthogonal_(param, gain=gain)
            elif 'bias' in name:  # Bias
                nn.init.constant_(param, constant_bias)
    else:  # other modules with single .weight and .bias
        # Weight
        nn.init.orthogonal_(module.weight, gain=gain)
        # Bias
        nn.init.constant_(module.bias, constant_bias)
