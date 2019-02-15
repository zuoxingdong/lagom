from collections import OrderedDict
from operator import itemgetter  # get list items with multiple indicies

from .utils import pickle_dump


class Logger(object):
    r"""Log the information in a dictionary. 
    
    If a key is logged more than once, then the new value will be appended to a list. 
    
    .. note::
    
        It uses pickle to serialize the data. Empirically, ``pickle`` is 2x faster than ``numpy.save``
        and other alternatives like ``yaml`` is too slow and ``JSON`` does not support numpy array. 
    
    .. warning::
    
        It is discouraged to store hierarchical structure, e.g. list of dict of list of ndarray.
        Because pickling such complex and large data structure is extremely slow. Put dictionary
        only at the topmost level. Large numpy array should be saved separately. 
    
    Example:
    
    * Default::
    
        >>> logger = Logger()
        >>> logger('iteration', 1)
        >>> logger('train_loss', 0.12)
        >>> logger('iteration', 2)
        >>> logger('train_loss', 0.11)
        >>> logger('iteration', 3)
        >>> logger('train_loss', 0.09)
        
        >>> logger
        OrderedDict([('iteration', [1, 2, 3]), ('train_loss', [0.12, 0.11, 0.09])])
        
        >>> logger.dump()
        Iteration: [1, 2, 3]
        Train Loss: [0.12, 0.11, 0.09]
        
    * With indentation::
    
        >>> logger.dump(indent=1)
            Iteration: [1, 2, 3]
            Train Loss: [0.12, 0.11, 0.09]
        
    * With specific keys::
    
        >>> logger.dump(keys=['iteration'])
        Iteration: [1, 2, 3]
        
    * With specific index::
    
        >>> logger.dump(index=0)
        Iteration: 1
        Train Loss: 0.12
        
    * With specific list of indices::
    
        >>> logger.dump(index=[0, 2])
        Iteration: [1, 3]
        Train Loss: [0.12, 0.09]
    
    """
    def __init__(self):
        # TODO: wait for popularity of Python 3.7 which dict preserves the order, then drop OrderedDict()
        self.logs = OrderedDict()
        
    def __call__(self, key, value):
        r"""Log the information with given key and value. 
        
        .. note::
        
            The key should be semantic and each word is separated by ``_``. 
        
        Args:
            key (str): key of the information
            value (object): value to be logged
        """
        if key not in self.logs:
            self.logs[key] = []
        
        self.logs[key].append(value)
        
    def dump(self, keys=None, index=None, indent=0, border=''):
        r"""Dump the loggings to the screen.
        
        Args:
            keys (list, optional): a list of selected keys. If ``None``, then use all keys. Default: ``None``
            index (int/list, optional): the index of logged values. It has following use cases:
                
                - ``scalar``: a specific index. If ``-1``, then use last element.
                - ``list``: a list of indicies. 
                - ``None``: all indicies.
                
            indent (int, optional): the number of tab indentation. Default: ``0``
            border (str, optional): the string to print as header and footer
        """
        if keys is None:
            keys = list(self.logs.keys())
        assert isinstance(keys, list), f'expected list, got {type(keys)}'
        
        if index is None:
            index = 'all'
        
        print(border)
        for key in keys:
            if indent > 0:
                print('\t'*indent, end='')  # do not create a new line
            
            if index == 'all':
                value = self.logs[key]
            elif isinstance(index, int):
                value = self.logs[key][index]
            elif isinstance(index, list):
                value = list(itemgetter(*index)(self.logs[key]))
            
            # Polish key string
            key = key.strip().replace('_', ' ').title()
            
            print(f'{key}: {value}')
        print(border)

    def save(self, f):
        r"""Save loggings to a file. 
        
        Args:
            f (str): file path
        """
        pickle_dump(obj=self.logs, f=f, ext='.pkl')
        
    def clear(self):
        r"""Remove all loggings in the dictionary. """
        self.logs.clear()
        
    def __repr__(self):
        return repr(self.logs)
