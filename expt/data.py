"""
Data structures for expt.

The "Experiment" data is structured like a 4D array, i.e.
    Experiment := [hypothesis_name, run_index, index, column]

The data is structured in the following ways (from higher to lower level):

Experiment (~= Dict[str, List[DataFrame]]):
    An experiment consists of one or multiple Hypotheses (e.g. different
    hyperparameters or algorithms) that can be compared with one another.

Hypothesis (~= List[DataFrame], or RunGroup):
    A Hypothesis consists of several `Run`s that share an
    identical experimental setups (e.g. hyperparameters).
    This usually corresponds to one single curve for each model.
    It may also contain additional metadata of the experiment.

Run (~= DataFrame == [index, column]):
    Contains a pandas DataFrame (a table-like structure, str -> Series)
    as well as more metadata (e.g. path, seed, etc.)

Note that one can also manage a collection of Experiments (e.g. the same set
of hypotheses or algorithms applied over different environments or dataset).
"""

import collections
from typing import Union, Iterable, Iterator, Optional
from typing import List, Tuple, Set, MutableMapping
from typeguard import typechecked

import sys
import types
import os.path
import pprint
from multiprocessing.pool import ThreadPool

import pandas as pd
import numpy as np
from pandas.core.accessor import CachedAccessor

from dataclasses import dataclass      # for python 3.6, backport needed

from . import plot as _plot
from .path_util import glob, exists, open, isdir


#########################################################################
# Data Classes
#########################################################################


@dataclass
class Run:
    """
    Represents a single run, containing one pd.DataFrame object as well as
    other metadata (path, etc.)
    """
    path: str
    df: pd.DataFrame

    @classmethod
    def of(cls, o):
        """A static factory method."""
        if isinstance(o, Run):
            return Run(path=o.path, df=o.df)
        elif isinstance(o, pd.DataFrame):
            return cls.from_dataframe(o)
        raise TypeError("Unknown type {}".format(type(o)))

    @classmethod
    @typechecked
    def from_dataframe(cls, df: pd.DataFrame):
        run = cls(path='', df=df)
        if hasattr(df, 'path'):
            run.path = df.path
        return run

    def __repr__(self):
        return 'Run({path}, df with {rows} rows)'.format(
            path=self.path, rows=len(self.df))

    def as_hypothesis(self):
        '''Contains a Hypothesis consisting of this run only.'''
        return Hypothesis.of(self)

    def plot(self, *args, subplots=True, **kwargs):
        return self.as_hypothesis().plot(*args, subplots=subplots, **kwargs)

    def hvplot(self, *args, subplots=True, **kwargs):
        return self.as_hypothesis().hvplot(*args, subplots=subplots, **kwargs)


@dataclass
class Hypothesis(Iterable[Run]):
    name: str
    runs: List[Run]

    def __iter__(self) -> Iterator[Run]:
        return iter(self.runs)

    @classmethod
    def of(cls, runs: Union[Run, Iterable[Run]],
           *, name: Optional[str] = None):
        """A static factory method."""
        if isinstance(runs, Run):
            runs = [runs]
            name = name or runs[0].path
        else:
            runs = list(runs)
            name = name or None
        return cls(name=name, runs=runs)

    def __getitem__(self, k):
        if isinstance(k, int):
            return self.runs[k]

        if k not in self.columns:
            raise KeyError(k)

        return pd.DataFrame({r.path: r.df[k] for r in self.runs})

    def __repr__(self) -> str:
        return (f"Hypothesis({self.name}, <{len(self.runs)} runs>)")

    def __len__(self) -> int:
        return len(self.runs)

    def __hash__(self):
        return hash(id(self))

    def describe(self):
        raise NotImplementedError

    # see module expt.plot
    plot = CachedAccessor("plot", _plot.HypothesisPlotter)
    plot.__doc__ = _plot.HypothesisPlotter.__doc__

    hvplot = CachedAccessor("hvplot", _plot.HypothesisHvPlotter)
    hvplot.__doc__ = _plot.HypothesisHvPlotter.__doc__

    @property
    def grouped(self) -> 'DataFrameGroupBy':
        return pd.concat(self._dataframes, sort=False).groupby(level=0)

    @property
    def _dataframes(self) -> List[pd.DataFrame]:
        """Get all dataframes associated with all the runs."""
        def _get_df(o):
            if isinstance(o, pd.DataFrame): return o
            else: return o.df
        return [_get_df(r) for r in self.runs]

    @property
    def columns(self) -> Iterable[str]:
        return self.grouped.mean().columns.values

    def rolling(self, *args, **kwargs):
        return self.grouped.rolling(*args, **kwargs)

    def mean(self, *args, **kwargs) -> pd.DataFrame:
        return self.grouped.mean(*args, **kwargs)

    def std(self, *args, **kwargs) -> pd.DataFrame:
        return self.grouped.std(*args, **kwargs)

    def min(self, *args, **kwargs) -> pd.DataFrame:
        return self.grouped.min(*args, **kwargs)

    def max(self, *args, **kwargs) -> pd.DataFrame:
        return self.grouped.max(*args, **kwargs)


class Experiment(Iterable[Hypothesis]):

    @typechecked
    def __init__(self,
                 title: Optional[str] = None,
                 hypotheses: Iterable[Hypothesis] = None,
                 ):
        self._title = title or "Experiment"
        self._hypotheses: MutableMapping[str, Hypothesis] = collections.OrderedDict()

        if isinstance(hypotheses, np.ndarray):
            hypotheses = list(hypotheses)

        for h in (hypotheses or []):
            if h.name in self._hypotheses:
                raise ValueError(f"Duplicate hypothesis name: `{h.name}`")
            self._hypotheses[h.name] = h

    def add_runs(self,
                 hypothesis_name: str,
                 runs: List[Union[Run,
                                  Tuple[str, pd.DataFrame],
                                  pd.DataFrame]],
                 *,
                 color=None,
                 linestyle=None,
                 ) -> Hypothesis:

        def check_runs_type(runs) -> List[Run]:
            if isinstance(runs, types.GeneratorType):
                runs = list(runs)
            if runs == []: return []
            if isinstance(runs, Run): runs = [runs]

            return [Run.of(r) for r in runs]
        runs = check_runs_type(runs)

        d = Hypothesis(name=hypothesis_name, runs=runs)
        return self.add_hypothesis(d, extend_if_conflict=True)

    @typechecked
    def add_hypothesis(self, h: Hypothesis,
                       extend_if_conflict=False,
                       ) -> Hypothesis:
        if h.name in self._hypotheses:
            if not extend_if_conflict:
                raise ValueError(f"Hypothesis named {h.name} already exists!")

            d: Hypothesis = self._hypotheses[h.name]
            d.runs.extend(h.runs)
        else:
            self._hypotheses[h.name] = h

        return self._hypotheses[h.name]

    @property
    def name(self) -> str:
        return self._title

    @property
    def title(self) -> str:
        return self._title

    def keys(self) -> Iterable[str]:
        """Return all hypothesis names."""
        return self._hypotheses.keys()

    @property
    def hypotheses(self) -> Iterable[Hypothesis]:
        return tuple(self._hypotheses.values())

    def select_top(self, criteria) -> Hypothesis:
        '''
        criteria: str (y_name) or Callable(Hypothesis -> number).
        TODO: make it more general using sorted, etc.
        '''
        if isinstance(criteria, str):
            y = str(criteria)  # make a copy for closure
            criteria = lambda h: h.mean()[y].max()   # TODO general criteria

        assert callable(criteria)
        return sorted(self.hypotheses, key=criteria, reverse=True)[0]

    def __iter__(self) -> Iterator[Hypothesis]:
        return iter(self._hypotheses.values())

    def __repr__(self) -> str:
        return (
            f"Experiment('{self.name}', {len(self._hypotheses)} hypotheses: " +
            pprint.pformat([exp for exp in self._hypotheses]) +
            ")"
        )

    def __getitem__(self, name: str) -> Hypothesis:
        return self._hypotheses[name]

    def __setitem__(self, name: str,
                    hypothesis_or_runs: Union[Hypothesis, List[Run]]) -> Hypothesis:
        """An dict-like method for adding hypothesis or runs."""
        if isinstance(hypothesis_or_runs, Hypothesis):
            if hypothesis_or_runs in self._hypotheses:
                raise ValueError(f"A hypothesis named {name} already exists")
            self._hypotheses[name] = hypothesis_or_runs
        else:
            # TODO metadata (e.g. color)
            self.add_runs(name, hypothesis_or_runs)

        return self._hypotheses[name]


    @property
    def columns(self) -> Iterable[str]:
        y: Set[str] = set()
        for h in self._hypotheses.values():
            y.update(h.columns)
        return list(y)

    def hvplot(self, *args, **kwargs):
        plot = None
        for i, (name, hypo) in enumerate(self._hypotheses.items()):
            p = hypo.hvplot(*args, label=name, **kwargs)
            plot = (plot * p) if plot else p
        return plot

    plot = CachedAccessor("plot", _plot.ExperimentPlotter)
    plot.__doc__ = _plot.ExperimentPlotter.__doc__


#########################################################################
# Data Parsing Functions
#########################################################################


def parse_run(run_folder, fillna=False,
              verbose=False) -> pd.DataFrame:
    """
    Create a pd.DataFrame object from a single directory (folder).
    """
    if verbose:
        # TODO Use python logging
        print(f"Reading {run_folder} ...",
              file=sys.stderr, flush=True)

    # make it more general (rather than being specific to progress.csv)
    # and support tensorboard eventlog files, etc.
    sources = [
        parse_run_progresscsv,
        parse_run_tensorboard,
    ]

    for fn in sources:
        try:
            df = fn(run_folder, fillna=fillna, verbose=verbose)
            if df is not None:
                break
        except (FileNotFoundError, IOError) as e:
            if verbose:
                print(f"{fn.__name__} -> {e}\n", file=sys.stderr, flush=True)
    else:
        raise pd.errors.EmptyDataError(f"Cannot handle dir: {run_folder}")

    # add some optional metadata... (might not be preserved afterwards)
    if df is not None:
        df.path = run_folder
    return df


def parse_run_progresscsv(run_folder, fillna=False,
                          verbose=False) -> pd.DataFrame:
    """
    Create a pd.DataFrame object that contains information from
    progress.csv or log.csv file (as convention)
    """
    # Try progress.csv or log.csv from folder
    detected_csv = None
    for fname in ('progress.csv', 'log.csv'):
        p = os.path.join(run_folder, fname)
        if exists(p):
            detected_csv = p
            break

    # maybe a direct file path is given instead of directory
    if detected_csv is None:
        if exists(run_folder) and not isdir(run_folder):
            detected_csv = run_folder

    if detected_csv is None:
        raise FileNotFoundError(os.path.join(run_folder, "*.csv"))

    # Read the detected file `p`
    if verbose:
        print(f"parse_run (csv): Reading {detected_csv}", file=sys.stderr, flush=True)

    with open(detected_csv, mode='r') as f:
        df = pd.read_csv(f)

    if fillna:
        df = df.fillna(0)

    return df


def parse_run_tensorboard(run_folder, fillna=False,
                          verbose=False) -> pd.DataFrame:
    """
    Create a pd.DataFrame object that contains all scalar summaries from the
    tensorboard eventfile (or its directory).
    """
    event_file = list(sorted(glob(os.path.join(run_folder, '*events.out.tfevents.*'))))

    if not event_file:  # no event file detected
        raise pd.errors.EmptyDataError(f"No event file detected in {run_folder}")
    event_file = event_file[-1]  # pick the last one
    if verbose:
        print(f"parse_run (tfevents) : Reading {event_file} ...",
              file=sys.stderr, flush=True)

    from collections import defaultdict
    from tensorflow.python.summary.summary_iterator import summary_iterator

    def iter_scalar_summary_from_event_file(event_file):
        for event in summary_iterator(event_file):
            step = int(event.step)
            if not event.HasField('summary'):
                continue
            for value in event.summary.value:
                # look for scalar values
                if value.HasField('simple_value'):
                    # value.tag, value.simple_value, etc ...
                    yield step, value

    all_data = defaultdict(dict)   # int(timestep) -> dict of columns
    for step, value in iter_scalar_summary_from_event_file(event_file):
        tag_name, scalar_value = value.tag, value.simple_value
        all_data[step][tag_name] = scalar_value

    for t in list(all_data.keys()):
        all_data[t]['global_step'] = t

    return pd.DataFrame(all_data).T


def iter_runs_serial(*path_globs, verbose=False, fillna=True) -> Iterator[Run]:
    """
    Enumerate Run objects from the given path(s).
    """
    for path_glob in path_globs:
        if verbose:
            print(f"get_runs: {str(path_glob)}", file=sys.stderr)

        paths = list(sorted(glob(path_glob)))
        for p in paths:
            try:
                df = parse_run(p, verbose=verbose, fillna=fillna)
                yield Run(p, df)
            except (pd.errors.EmptyDataError, FileNotFoundError) as e:
                print(f"[!] {p} : {e}", file=sys.stderr)


def get_runs_serial(*path_globs, verbose=False, fillna=True) -> List[Run]:
    """
    Get a list of Run objects from the given path(s).
    Works in single-thread (slow, should not used outside debugging purposes).
    """
    runs = list(iter_runs_serial(*path_globs, verbose=verbose, fillna=fillna))

    if not runs:
        for path_glob in path_globs:
            print(f"Warning: No match found for pattern {path_glob}",
                  file=sys.stderr)

    return runs


def get_runs_parallel(*path_globs, verbose=False, n_jobs=8, fillna=True,
                      ) -> List[Run]:
    """
    Get a list of Run objects from the given path(s).
    Runs in parallel.
    """

    def handle_p(p) -> Optional[Tuple[str, pd.DataFrame]]:
        try:
            df = parse_run(p, verbose=verbose, fillna=fillna)
            return p, df
        except (pd.errors.EmptyDataError, FileNotFoundError) as e:
            print(f"[!] {p} : {e}", file=sys.stderr, flush=True)
            return None

    with ThreadPool(processes=n_jobs) as pool:
        futures = []
        for path_glob in path_globs:

            paths = list(sorted(glob(path_glob)))
            if verbose and not paths:
                print(f"Warning: a glob pattern '{path_glob}' "
                       "did not match any files.", file=sys.stderr)

            for p in paths:
                future = pool.apply_async(handle_p, [p])
                futures.append(future)

        hits = 0
        result = []
        for future in futures:
            p_and_df = future.get()
            if p_and_df:
                p, df = p_and_df
                hits += 1
                result.append(Run(p, df))

    if not hits:
        for path_glob in path_globs:
            print(f"Warning: No match found for pattern {path_glob}",
                  file=sys.stderr)
    return result


get_runs = get_runs_parallel
