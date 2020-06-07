import sys
if sys.version_info < (3, 6):
    raise RuntimeError("This library requires python 3.6+.")

__version__ = '0.1.0'

# auto-import submodules
from . import (data, plot, colors)

# populate common APIs
from .data import (
    Experiment, Hypothesis, Run, RunList,
    parse_run, get_runs,
)
