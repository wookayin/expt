"""The expt module."""
# pylint: disable=useless-import-alias
# flake8: noqa

import sys

if sys.version_info < (3, 7):
  raise RuntimeError("This library requires python 3.7+.")

# See pyproject.toml
__version__ = '0.5.0.dev2'

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
from .data_loader import get_runs as get_runs
from .data_loader import get_runs_async as get_runs_async
from .data_loader import parse_run as parse_run
from .data_loader import RunLoader as RunLoader
