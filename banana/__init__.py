# from jax import config

# config.update("jax_enable_x64", False)

from .core import BananaCore
from .logger import Logger, LogServer
from .io import IO
from .baseclasses import (
    BaseVariable,
    BaseModule,
    BaseUtil,
    BaseStrategy,
    BaseKernel,
    BaseAnalysis,
)
from .utils import *

# from .grid_utils import *
from .common import *

import numpy as np
import os

try:
    linewidth = os.get_terminal_size()[0]
except OSError:
    linewidth = 90

np.set_printoptions(precision=8, threshold=1000, edgeitems=3, linewidth=linewidth)
