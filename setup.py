"""setup.py for expt"""

import sys
import os
import re
from setuptools import setup, Command

__PATH__ = os.path.abspath(os.path.dirname(__file__))


def read_readme():
    with open('README.md') as f:
        return f.read()


def read_version():
    __PATH__ = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(__PATH__, 'expt/__init__.py')) as f:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                                  f.read(), re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find __version__ string")


__version__ = read_version()


# brought from https://github.com/kennethreitz/setup.py
class DeployCommand(Command):
    description = 'Build and deploy the package to PyPI.'
    user_options = []

    def initialize_options(self): pass
    def finalize_options(self): pass

    @staticmethod
    def status(s):
        print(s)

    def run(self):
        import twine  # we require twine locally  # noqa

        assert 'dev' not in __version__, (
            "Only non-devel versions are allowed. "
            "__version__ == {}".format(__version__))

        with os.popen("git status --short") as fp:
            git_status = fp.read().strip()
            if git_status:
                print("Error: git repository is not clean.\n")
                os.system("git status --short")
                sys.exit(1)

        try:
            from shutil import rmtree
            self.status('Removing previous builds ...')
            rmtree(os.path.join(__PATH__, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution ...')
        os.system('{0} setup.py sdist'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine ...')
        ret = os.system('twine upload dist/*')
        if ret != 0:
            sys.exit(ret)

        self.status('Creating git tags ...')
        os.system('git tag v{0}'.format(__version__))
        os.system('git tag --list')
        sys.exit()


install_requires = [
    'numpy',
    'scipy',
    'dataclasses>=0.6',
    'typeguard>=2.6.1',
    'matplotlib>=3.0.0',
    'pandas>=1.0',
]

tests_requires = [
    'mock>=2.0.0',
    'pytest>=5.0',   # Python 3.5+
    'pytest-cov',
]

setup(
    name='expt',
    version=__version__,
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
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Utilities',
        'Topic :: Scientific/Engineering',
    ],
    packages=['expt'],
    install_requires=install_requires,
    extras_require={'test': tests_requires},
    setup_requires=['pytest-runner'],
    tests_require=tests_requires,
    entry_points={
        #'console_scripts': ['expt=expt:main'],
    },
    cmdclass={
        'deploy': DeployCommand,
    },
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.6',
)
