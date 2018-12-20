import torch.nn as nn

from .ln_rnncell import LayerNormLSTMCell


def make_fc(input_dim, hidden_sizes):
    r"""Returns a ModuleList of fully connected layers. 
    
    .. note::
    
        All submodules can be automatically tracked because it uses nn.ModuleList. One can
        use this function to generate parameters in :class:`BaseNetwork`. 
    
    Example::
    
        >>> make_fc(3, [4, 5, 6])
        ModuleList(
          (0): Linear(in_features=3, out_features=4, bias=True)
          (1): Linear(in_features=4, out_features=5, bias=True)
          (2): Linear(in_features=5, out_features=6, bias=True)
        )
    
    Args:
        input_dim (int): input dimension in the first fully connected layer. 
        hidden_sizes (list): a list of hidden sizes, each for one fully connected layer. 
    
    Returns
    -------
    fc : nn.ModuleList
        A ModuleList of fully connected layers.     
    """
    assert isinstance(hidden_sizes, list), f'expected list, got {type(hidden_sizes)}'
    
    hidden_sizes = [input_dim] + hidden_sizes
    
    fc = []
    for in_features, out_features in zip(hidden_sizes[:-1], hidden_sizes[1:]):
        fc.append(nn.Linear(in_features=in_features, out_features=out_features))
    
    fc = nn.ModuleList(fc)
    
    return fc


def make_cnn(input_channel, channels, kernels, strides, paddings):
    r"""Returns a ModuleList of 2D convolution layers. 
    
    .. note::
    
        All submodules can be automatically tracked because it uses nn.ModuleList. One can
        use this function to generate parameters in :class:`BaseNetwork`. 
        
    Example::
    
        >>> make_cnn(input_channel=3, channels=[16, 32], kernels=[4, 3], strides=[2, 1], paddings=[1, 0])
        ModuleList(
          (0): Conv2d(3, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
          (1): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1))
        )
    
    Args:
        input_channel (int): input channel in the first convolution layer. 
        channels (list): a list of channels, each for one convolution layer.
        kernels (list): a list of kernels, each for one convolution layer.
        strides (list): a list of strides, each for one convolution layer. 
        paddings (list): a list of paddings, each for one convolution layer. 
    
    Returns
    -------
    cnn : nn.ModuleList
        A ModuleList of 2D convolution layers.    
    """
    N = len(channels)
    
    for item in [channels, kernels, strides, paddings]:
        assert isinstance(item, list), f'expected as list, got {type(item)}'
        assert len(item) == N, f'expected length {N}, got {len(item)}'
    
    channels = [input_channel] + channels
    
    cnn = []
    for i in range(N):
        cnn.append(nn.Conv2d(in_channels=channels[i], 
                             out_channels=channels[i+1], 
                             kernel_size=kernels[i], 
                             stride=strides[i], 
                             padding=paddings[i], 
                             dilation=1, 
                             groups=1))
    
    cnn = nn.ModuleList(cnn)
    
    return cnn


def make_transposed_cnn(input_channel, channels, kernels, strides, paddings, output_paddings):
    r"""Returns a ModuleList of 2D transposed convolution layers. 
    
    .. note::
    
        All submodules can be automatically tracked because it uses nn.ModuleList. One can
        use this function to generate parameters in :class:`BaseNetwork`. 
        
    Example::
    
        make_transposed_cnn(input_channel=3, 
                            channels=[16, 32], 
                            kernels=[4, 3], 
                            strides=[2, 1], 
                            paddings=[1, 0], 
                            output_paddings=[1, 0])
        ModuleList(
          (0): ConvTranspose2d(3, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
          (1): ConvTranspose2d(16, 32, kernel_size=(3, 3), stride=(1, 1))
        )
    
    Args:
        input_channel (int): input channel in the first transposed convolution layer. 
        channels (list): a list of channels, each for one transposed convolution layer.
        kernels (list): a list of kernels, each for one transposed convolution layer.
        strides (list): a list of strides, each for one transposed convolution layer. 
        paddings (list): a list of paddings, each for one transposed convolution layer. 
        output_paddings (list): a list of output paddings, each for one transposed convolution layer. 
    
    Returns
    -------
    transposed_cnn : nn.ModuleList
        A ModuleList of 2D transposed convolution layers.    
    """
    N = len(channels)
    
    for item in [channels, kernels, strides, paddings, output_paddings]:
        assert isinstance(item, list), f'expected as list, got {type(item)}'
        assert len(item) == N, f'expected length {N}, got {len(item)}'
    
    channels = [input_channel] + channels
    
    transposed_cnn = []
    for i in range(N):
        transposed_cnn.append(nn.ConvTranspose2d(in_channels=channels[i], 
                                                 out_channels=channels[i+1], 
                                                 kernel_size=kernels[i], 
                                                 stride=strides[i], 
                                                 padding=paddings[i], 
                                                 output_padding=output_paddings[i],
                                                 dilation=1, 
                                                 groups=1))
    
    transposed_cnn = nn.ModuleList(transposed_cnn)
    
    return transposed_cnn


def make_rnncell(cell_type, input_dim, hidden_sizes):
    r"""Returns a ModuleList of RNN Cells.
    
    .. note::
    
        All submodules can be automatically tracked because it uses nn.ModuleList. One can
        use this function to generate parameters in :class:`BaseNetwork`. 

    Example::
    
        >>> make_rnncell('LSTMCell', 3, [51, 32, 16])
        ModuleList(
          (0): LSTMCell(3, 51)
          (1): LSTMCell(51, 32)
          (2): LSTMCell(32, 16)
        )
    
    Args:
        cell_type (str): RNNCell type e.g. ['RNNCell', 'LSTMCell', 'GRUCell', 'LayerNormLSTMCell']
        input_dim (int): input dimension in the first recurrent layer. 
        hidden_sizes (list): a list of hidden sizes, each for one recurrent layer. 
    
    Returns
    -------
    rnncell : nn.ModuleList
        a ModuleList of recurrent layers (cells).
        
    """
    assert isinstance(hidden_sizes, list), f'expected as list, got {type(hidden_sizes)}'
    
    if cell_type == 'RNNCell':
        cell_f = nn.RNNCell
    elif cell_type == 'LSTMCell':
        cell_f = nn.LSTMCell
    elif cell_type == 'GRUCell':
        cell_f = nn.GRUCell
    elif cell_type == 'LayerNormLSTMCell':
        cell_f = LayerNormLSTMCell
    else:
        raise ValueError(f'expected RNNCell/LSTMCell/GRUCell/LayerNormLSTMCell, got {cell_type}')
    
    hidden_sizes = [input_dim] + hidden_sizes
    
    rnncell = []
    for input_size, hidden_size in zip(hidden_sizes[:-1], hidden_sizes[1:]):
        rnncell.append(cell_f(input_size=input_size, hidden_size=hidden_size))
    
    rnncell = nn.ModuleList(rnncell)
    
    return rnncell
