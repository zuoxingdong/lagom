from pathlib import Path

import pickle
import cloudpickle

import yaml


def pickle_load(f):
    r"""Read a pickled data from a file. 

    Args:
        f (str/Path): file path
    """
    if isinstance(f, Path):
        f = f.as_posix()

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
        f (str/Path): file path
        ext (str, optional): file extension. Default: .pkl
    """
    if isinstance(f, Path):
        f = f.as_posix()
    
    with open(f+ext, 'wb') as file:
        return cloudpickle.dump(obj=obj, file=file, protocol=pickle.HIGHEST_PROTOCOL)


def yaml_load(f):
    r"""Read the data from a YAML file. 

    Args:
        f (str/Path): file path
    """
    if isinstance(f, Path):
        f = f.as_posix()
    
    with open(f, 'r') as file:
        return yaml.load(file)


def yaml_dump(obj, f, ext='.yml'):
    r"""Serialize a Python object using YAML and save in a file. 
    
    .. note::
    
        YAML is recommended to use for a small dictionary and it is super
        human-readable. e.g. configuration settings. For saving experiment
        metrics, it is better to use :func:`pickle_dump`.
        
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
        f (str/Path): file path
        ext (str, optional): file extension. Default: .yml
    """
    if isinstance(f, Path):
        f = f.as_posix()
    
    # TODO: remove it when PyYAML supports it officially. 
    tag = 'tag:yaml.org,2002:map'
    representer = lambda dumper, data: dumper.represent_mapping(tag, list(data.items()))
    yaml.add_representer(dict, representer)
    
    with open(f+ext, 'w') as file:
        return yaml.dump(obj, file, default_flow_style=False)

    
class CloudpickleWrapper(object):
    r"""Uses cloudpickle to serialize contents (multiprocessing uses pickle by default)
    
    This is useful when passing lambda definition through Process arguments.
    """
    def __init__(self, x):
        self.x = x
        
    def __call__(self):
        return self.x()
    
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)
