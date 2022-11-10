"""setup.py for expt. Deprecated as per PEP-621."""

import shlex
import sys
import textwrap

join = shlex.join if sys.version_info >= (3, 8) else (lambda t: ' '.join(t))

if 'pep517' in sys.argv[0]:
  pass  # invoked as a part of PEP-517 aware pip command

elif len(sys.argv) >= 2 and sys.argv[1] in ('install', 'develop'):
  sys.stderr.write("Invoked as: " + join(sys.argv) + '\n\n')
  sys.stderr.write(textwrap.dedent(
    """\
    Warning: expt no longer supports installation via `python setup.py ...`
      as per PEP-621: https://peps.python.org/pep-0621/.

    If you have cloned the repository locally, please use the following cmd:
        $ python -m pip install .       # or,
        $ python -m pip install -e .    # for editable mode

    If you are already trying `pip install` and still seeing this error,
      then build configuration in `pyproject.toml` would have some problem.
    """))  # yapf: disable
  sys.exit(1)

# See pyproject.toml for project metadata.
# These lines are only for the sake of Github package metadata.
from setuptools import setup  # noqa

setup(name='expt')
