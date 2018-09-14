class BaseAlgorithm(object):
    r"""Base class for all algorithms.
    
    Any algorithm should subclass this class. 
    
    The subclass should implement at least the following:
    
    - :meth:`__call__`
    
    """
    def __init__(self, name):
        r"""Initialize the algorithm.
        
        Args:
            name (str): name of the algorithm
        """
        self.name = name
        
    def __call__(self, config, seed, device_str=None):
        r"""Run the algorithm with a configuration, a random seed and optionally a device string for PyTorch
        automatically assigned by :class:`BaseExperimentWorker`. 
        
        Args:
            config (dict): a dictionary of configuration items
            seed (int): a random seed to run the algorithm
            device_str (str, optional): a string for PyTorch device. When using :class:`BaseExperimentWorker`,
                each worker will automatically assign a device for the algorithm to run. If using CUDA is specified
                in the configuration received by the worker, then it will assign a CUDA device with its task ID
                modulus total number of GPUs, the string will be ``cuda:X`` where X is the calculated CUDA ID. 
                If configuration does not specify CUDA or specifies it as ``False``, the device string will
                be 'cpu' indicating using CPU only. In :meth:`__call__` of algorithm class, one could
                exploit the assigned device in PyTorch via ``torch.device(device_str)``. However, it is
                not restricted, even if a device string is given, one could still explicitly use other devices.
                Default: ``None``
            
        Returns
        -------
        result : object
            result of the execution. If no need to return anything, then an ``None`` should be returned. 
        """
        raise NotImplementedError
