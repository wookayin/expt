import sys

if sys.version_info < (3, 6):
  raise RuntimeError("This library requires python 3.6+.")

__version__ = '0.3.0'

# auto-import submodules
from . import colors as colors
from . import data as data
from . import plot as plot
#
# populate common APIs
from .data import Experiment as Experiment
from .data import Hypothesis as Hypothesis
from .data import Run as Run
from .data import RunList as RunList
from .data import get_runs as get_runs
from .data import parse_run as parse_run
