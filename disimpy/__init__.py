# Numba warns about GPU under-utilization and, at least in version 5.7.0, importing
# numba.cuda.random leads to many deprecation warnings. Let's ignore these:

from numba.core.errors import NumbaDeprecationWarning, NumbaPerformanceWarning
import warnings

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


from . import gradients
from . import simulations
from . import substrates
from . import utils

__version__ = "0.4.0-dev"
