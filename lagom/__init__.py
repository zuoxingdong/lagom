from .version import __version__

from .base_algo import BaseAlgorithm

from .logger import BaseLogger
from .logger import Logger

from .utils import set_global_seeds
from .utils import Seeder
from .utils import pickle_load
from .utils import pickle_dump
from .utils import yaml_load
from .utils import yaml_dump
from .utils import color_str
from .utils import timed

# Some macros about PyTorch compatible numpy dtype
import numpy as np
NUMPY_FLOAT = np.float32
NUMPY_INT = np.int32
