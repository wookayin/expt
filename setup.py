"""setup.py for expt"""

import ast
import os
import shutil
import sys
import textwrap
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module='setuptools',
    message="setuptools.installer and fetch_build_eggs are deprecated."
) # yapf: disable

from setuptools import Command
from setuptools import setup

try:
  from setuptools_rust import Binding
  from setuptools_rust import build_rust
  from setuptools_rust import RustExtension
  from setuptools_rust.command import get_rust_version
except ImportError:
  sys.stderr.write(textwrap.dedent(
      """\
          Error: setuptools_rust cannot be imported.
          Runnig setup.py like `python setup.py` is deprecated.

          Please run `pip install .` or `pip install -e .[test]` instead.
      """))  # yapf: disable  # noqa
  sys.exit(1)

__PATH__ = os.path.abspath(os.path.dirname(__file__))

EXPT_DISABLE_RUST = ast.literal_eval(os.getenv("EXPT_DISABLE_RUST") or "False")


class build_rust_for_expt(build_rust):

  def run(self):
    if not EXPT_DISABLE_RUST and get_rust_version() is None:
      from distutils.errors import DistutilsPlatformError
      raise DistutilsPlatformError(
          "Rust toolchain (cargo, rustc) not found. "
          "Please install rust toolchain to build expt with the rust extension. "
          "If you would like to build expt without the extension, "
          "export EXPT_DISABLE_RUST=1 and try again.")
    return super().run()


def read_readme():
  with open('README.md', encoding='utf-8') as f:
    return f.read()


try:
  import setuptools_scm
except ImportError as ex:
  raise ImportError("setuptools_scm not found. When running setup.py directly, "
                    "setuptools_scm>=8.0 needs to be installed manually. "
                    "Or consider running `pip install -e .` instead.") from ex

install_requires = [
    'numpy>=1.16.5',
    'scipy',
    'typeguard>=2.6.1',
    'matplotlib>=3.0.0',
    'pandas>=1.3',
    'pyyaml>=6.0',
    'multiprocess>=0.70.12',
    'multiprocessing_utils==0.4',
    'typing_extensions>=4.0',
]

tests_requires = [
    'setuptools-rust',
    'mock>=2.0.0',
    'pytest>=7.0',
    'pytest-cov',
    'pytest-asyncio',
    # Optional dependencies.
    'tensorboard>=2.3',
    'fabric>=3.0',
    'paramiko>=2.8',
]


def next_semver(version: setuptools_scm.version.ScmVersion):
  """Determine next development version."""

  if version.branch and 'release' in version.branch:
    # Release branch: bump up patch versions
    return version.format_next_version(
        setuptools_scm.version.guess_next_simple_semver,
        retain=setuptools_scm.version.SEMVER_PATCH)
  else:
    # main/dev branch: bump up minor versions
    return version.format_next_version(
        setuptools_scm.version.guess_next_simple_semver,
        retain=setuptools_scm.version.SEMVER_MINOR)


setup(
    name='expt',
    # version=__version__,  # setuptools_scm sets version automatically
    use_scm_version=dict(
        write_to='expt/_version.py',
        version_scheme=next_semver,
    ),
    license='MIT',
    description='EXperiment. Plot. Tabulate.',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/wookayin/expt',
    author='Jongwook Choi',
    author_email='wookayin@gmail.com',
    #keywords='',
    classifiers=[
        # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Utilities',
        'Topic :: Scientific/Engineering',
    ],
    python_requires='>=3.8',
    packages=['expt'],
    rust_extensions=[
        RustExtension("expt._internal", binding=Binding.PyO3, \
                      debug=False  # Always use --release (optimized) build
                      ),
    ] if not EXPT_DISABLE_RUST else [],
    install_requires=install_requires,
    extras_require={'test': tests_requires},
    setup_requires=['setuptools-rust'],
    tests_require=tests_requires,
    entry_points={
        #'console_scripts': ['expt=expt:main'],
    },
    cmdclass={
        'build_rust': build_rust_for_expt,
    },  # type: ignore
    include_package_data=True,
    zip_safe=False,
)
