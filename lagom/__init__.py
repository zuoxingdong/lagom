from .version import __version__

from .base_algo import BaseAlgorithm

from .logger import BaseLogger
from .logger import Logger

# Some macros about PyTorch compatible numpy dtype
import numpy as np
NUMPY_FLOAT = np.float32
NUMPY_INT = np.int32
