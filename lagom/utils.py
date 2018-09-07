import torch
import numpy as np
import random

import pickle
import cloudpickle

import yaml


def set_global_seeds(seed):
    r"""Set the seed for generating random numbers.
    
    It sets the following dependencies with the given random seed:
    
    1. PyTorch
    2. Numpy
    3. Python random
    
    Args:
        seed (int): Given seed.
    """
    torch.manual_seed(seed)  # both torch and torch.cuda internally
    np.random.seed(seed)
    random.seed(seed)


class Seeder(object):
    r"""A random seed generator. 
    
    Given an initial seed, the seeder can be called continuously to sample a single
    or a batch of random seeds. 
    
    .. note::
    
        The seeder creates an independent RandomState to generate random
        numbers. It does not affect the RandomState in ``np.random``. 
    
    Example::
    
        >>> seeder = Seeder(init_seed=0)
        >>> seeder(size=5)
        [209652396, 398764591, 924231285, 1478610112, 441365315]
        
    """
    
    def __init__(self, init_seed=0):
        r"""
        Args:
            init_seed (int, optional): Initial seed for generating random seeds.
        """
        assert isinstance(init_seed, int) and init_seed >= 0, f'Seed expected to be non-negative integer, got {init_seed}'
        
        # Create a numpy RandomState with given initial seed
        # A RandomState is independent of np.random
        self.rng = np.random.RandomState(seed=init_seed)
        # Upper bound for sampling new random seeds
        self.max = np.iinfo(np.int32).max
        
    def __call__(self, size=1):
        r"""Return the sampled random seeds according to the given size. 
        
        Args:
            size (int or list): The size of random seeds to sample. 
            
        Returns:
            seeds (list): A list of sampled random seeds.
        """
        seeds = self.rng.randint(low=0, high=self.max, size=size).tolist()
        
        return seeds


def pickle_load(f):
    r"""Read a pickled data from a file. 
    
    .. note::
    
        It uses cloudpickle instead of pickle to support lambda
        function and multiprocessing. By default, the highest
        protocol is used. 
        
    .. note::
    
        Except for pure array object, it is not recommended to use
        ``np.load`` because it is often much slower. 
        
    Args:
        f (str): file path
    """
    with open(f, 'rb') as file:
        return cloudpickle.load(file)

def pickle_dump(obj, f, ext='.pkl'):
    r"""Serialize an object using pickling and save in a file. 
    
    .. note::
    
        It uses cloudpickle instead of pickle to support lambda
        function and multiprocessing. By default, the highest
        protocol is used. 
        
    .. note::
    
        Except for pure array object, it is not recommended to use
        ``np.save`` because it is often much slower. 
    
    Args:
        obj (object): a serializable object
        f (str): file path
        ext (str, optional): file extension. Default: .pkl
    """
    with open(f+ext, 'wb') as file:
        return cloudpickle.dump(obj=obj, file=file, protocol=pickle.HIGHEST_PROTOCOL)
    
def yaml_load(f):
    r"""Read the data from a YAML file. 
    
    .. note::
    
        YAML is recommended to use for a small dictionary and it is super
        human-readable. e.g. configuration settings. For saving experiment
        metrics, it is better to use :func:`pickle_load` and :func:`pickle_dump`.
        
    .. note::
    
        Except for pure array object, it is not recommended to use
        ``np.save`` because it is often much slower. 
    
    Args:
        f (str): file path
    """
    with open(f, 'r') as file:
        return yaml.load(file)
    
def yaml_dump(obj, f, ext='.yml'):
    r"""Serialize a Python object using YAML and save in a file. 
    
    .. note::
    
        YAML is recommended to use for a small dictionary and it is super
        human-readable. e.g. configuration settings. For saving experiment
        metrics, it is better to use :func:`pickle_load` and :func:`pickle_dump`.
        
    .. note::
    
        Except for pure array object, it is not recommended to use
        ``np.load`` because it is often much slower. 
        
    .. warning::
    
        It preserves the order of Python dict. Although this property is already
        supported for Python 3.7, but currently PyYAML does not. So we do a work around
        to preserve the ordering of Python dict. 
        
        * ``TODO``: remove it when PyYAML supports it officially. 
        
    Args:
        obj (object): a serializable object
        f (str): file path
        ext (str, optional): file extension. Default: .yml
    """
    # Create representer to preserve the order of dict
    # TODO: remove it when PyYAML supports it officially. 
    tag = 'tag:yaml.org,2002:map'
    representer = lambda dumper, data: dumper.represent_mapping(tag, list(data.items()))
    yaml.add_representer(dict, representer)
    
    with open(f+ext, 'w') as file:
        return yaml.dump(obj, file, default_flow_style=False)
