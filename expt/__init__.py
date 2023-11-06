"""The expt module."""
# pylint: disable=useless-import-alias
# flake8: noqa
# isort: skip_file

import sys

if sys.version_info < (3, 8):
  raise RuntimeError("This library requires python 3.8+.")

try:
  from ._version import version as __version__
  from ._version import version_tuple as __version_tuple__
except (ImportError, AttributeError) as ex:
  raise ImportError(
      "Unable to find the `expt.__version__` string. "
      "Please try reinstalling expt; "
      "or if you are on a development version, run `pip install -e .` or "
      "`python setup.py --version` and try again.") from ex

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
