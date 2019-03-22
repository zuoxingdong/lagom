class LinearSchedule(object):
    r"""A linear scheduling from an initial to a final value over a certain timesteps, then the final
    value is fixed constantly afterwards. 
        
    .. note::

        This could be useful for following use cases:

        * Decay of epsilon-greedy: initialized with :math:`1.0` and keep with :attr:`start` time steps, then linearly
          decay to :attr:`final` over :attr:`N` time steps, and then fixed constantly as :attr:`final` afterwards.
        * Beta parameter in prioritized experience replay. 

        Note that for learning rate decay, one should use PyTorch ``optim.lr_scheduler`` instead. 
    
    Example:
    
        >>> scheduler = LinearSchedule(initial=1.0, final=0.1, N=3, start=0)
        >>> [scheduler(i) for i in range(6)]
        [1.0, 0.7, 0.4, 0.1, 0.1, 0.1]
    
    Args:
        initial (float): initial value
        final (float): final value
        N (int): number of scheduling timesteps
        start (int, optional): the timestep to start the scheduling. Default: 0
        
    """
    def __init__(self, initial, final, N, start=0):
        assert N > 0, f'expected N as positive integer, got {N}'
        assert start >= 0, f'expected start as non-negative integer, got {start}'
        
        self.initial = initial
        self.final = final
        self.N = N
        self.start = start
        
        self.x = None
    
    def __call__(self, x):
        r"""Returns the current value of the scheduling. 
        
        Args:
            x (int): the current timestep. 
            
        Returns
        -------
        out : float
            current value of the scheduling. 
        """
        assert isinstance(x, int) and x >= 0, f'expected as a non-negative integer, got {x}'
        
        if x == 0 or x < self.start:
            self.x = self.initial
        elif x >= self.start + self.N:
            self.x = self.final
        else:  # scheduling over N steps
            delta = self.final - self.initial
            ratio = (x - self.start)/self.N
            self.x = self.initial + ratio*delta
        return self.x
    
    def get_current(self):
        return self.x
