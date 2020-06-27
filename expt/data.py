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
import itertools
from typing import Any, Union, Dict, Callable, Optional
from typing import List, Tuple, Set, Sequence, Mapping, MutableMapping
from typing import Iterable, Iterator, Generator, TypeVar
from typeguard import typechecked

import sys
import types
import fnmatch
import os.path
import pprint
from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool as MultiprocessPool

import pandas as pd
import numpy as np
from pandas.core.accessor import CachedAccessor
from pandas.core.groupby.generic import DataFrameGroupBy

from dataclasses import dataclass      # for python 3.6, backport needed

from . import plot as _plot
from . import util
from .path_util import glob, exists, open, isdir


T = TypeVar('T')

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
        return 'Run({path!r}, df with {rows} rows)'.format(
            path=self.path, rows=len(self.df))

    @property
    def name(self) -> str:
        '''Returns the last segment of the path.'''
        path = self.path.rstrip('/')
        return os.path.basename(path)

    def to_hypothesis(self) -> 'Hypothesis':
        '''Create a new `Hypothesis` consisting of only this run.'''
        return Hypothesis.of(self)

    def plot(self, *args, subplots=True, **kwargs):
        return self.to_hypothesis().plot(*args, subplots=subplots, **kwargs)

    def hvplot(self, *args, subplots=True, **kwargs):
        return self.to_hypothesis().hvplot(*args, subplots=subplots, **kwargs)


class RunList(Sequence[Run]):
    """
    A (immutable) list of Run objects, but with some useful utility methods
    such as filtering, searching, and handy format conversion.
    """
    def __init__(self, runs: Iterable[Run]):
        runs = self._validate_type(runs)
        self._runs = list(runs)

    @classmethod
    def of(cls, runs: Iterable[Run]):
        if isinstance(runs, cls):
            return runs       # do not make a copy
        else:
            return cls(runs)  # RunList(runs)

    def _validate_type(self, runs) -> List[Run]:
        if not isinstance(runs, Iterable):
            raise TypeError("`runs` must be a {}, but given {}".format(
                Iterable, type(runs)))
        if isinstance(runs, Mapping):
            raise TypeError("`runs` should not be a dictionary, given {} "
                            " (forgot to wrap with pd.DataFrame?)".format(type(runs)))

        runs = list(runs)
        if not all(isinstance(r, Run) for r in runs):
            raise TypeError("`runs` must be a iterable of Run, "
                            "but given {}".format([type(r) for r in runs]))
        return runs

    def __getitem__(self, index_or_slice):
        o = self._runs[index_or_slice]
        if isinstance(index_or_slice, slice):
            o = RunList(o)
        return o

    def __len__(self):
        return len(self._runs)

    def __repr__(self):
        return "RunList([\n " + "\n ".join(repr(r) for r in self._runs) + "\n]"

    def extend(self, more_runs: Iterable[Run]):
        self._runs.extend(more_runs)

    def to_list(self) -> List[Run]:
        """Create a new copy of list containing all the runs."""
        return list(self._runs)

    def to_dataframe(self) -> pd.DataFrame:
        '''Return a DataFrame consisting of columns "name" and "run".'''
        return pd.DataFrame({
            'name': [r.name for r in self._runs],
            'run': self._runs,
        })

    def filter(self, fn: Union[Callable[[Run], bool], str]) -> 'RunList':
        '''Apply a filter function (Run -> bool) and return the filtered runs
        as another RunList. If a string is given, we convert it as a matcher
        function (see fnmatch) that matches run.name'''
        if isinstance(fn, str):
            pat = str(fn)
            fn = lambda run: fnmatch.fnmatch(run.name, pat)
        return RunList(filter(fn, self._runs))

    def map(self, func: Callable[[Run], Any]) -> List:
        '''Apply func for each of the runs. Return the transformation
        as a plain list.'''
        return list(map(func, self._runs))

    def to_hypothesis(self, name: str) -> 'Hypothesis':
        '''Create a new Hypothesis instance containing all the runs
        as the current RunList instance.'''
        return Hypothesis.of(self, name=name)

    def groupby(self,
                by: Callable[[Run], T], *,
                name: Callable[[T], str] = str,
                ) -> Iterator[Tuple[T, 'Hypothesis']]:
        r'''Group runs into hypotheses with the key function `by` (Run -> key).
        This will enumerate tuples (`group_key`, Hypothesis) where `group_key`
        is the result of the key function for each group, and a Hypothesis
        object (with name `name(group_key)`) will consist of all the runs
        mapped to the same group.

        Args:
            by: a key function for groupby operation. (Run -> Key)
            name: a function that maps the group (Key) into Hypothesis name (str).

        Example:
            key_func = lambda run: re.search("algo=(\w+),lr=([.0-9]+)", run.name).group(1, 2)
            for group_name, hypothesis in runs.groupby(key_func):
                ...
        '''
        series = pd.Series(self._runs)
        groupby = series.groupby(lambda i: by(series[i]))

        group: T
        for group, runs_in_group in groupby:
            yield group, Hypothesis.of(runs_in_group, name=name(group))

    def extract(self, pat: str, flags: int = 0) -> pd.DataFrame:
        r'''Extract capture groups in the regex pattern `pat` as columns.

        Example:
            >>> runs[0].name
            "ppo-halfcheetah-seed0"
            >>> df = runs.extract(r"(?P<algo>[\w]+)-(?P<env_id>[\w]+)-seed(?P<seed>[\d]+)")
            >>> assert list(df.columns) == ['algo', 'env_id', 'seed', 'run']

        '''
        df: pd.DataFrame = self.to_dataframe()
        df = df['name'].str.extract(pat, flags=flags)
        df['run'] = list(self._runs)
        return df


@dataclass
class Hypothesis(Iterable[Run]):
    name: str
    runs: RunList

    def __init__(self, name: str, runs: Union[Run, Iterable[Run]]):
        if isinstance(runs, Run) or isinstance(runs, pd.DataFrame):
            if not isinstance(runs, Run):
                runs = Run.of(runs)
            runs = [runs]  # type: ignore

        self.name = name
        self.runs = RunList(runs)

    def __iter__(self) -> Iterator[Run]:
        return iter(self.runs)

    @classmethod
    def of(cls, runs: Union[Run, Iterable[Run]],
           *, name: Optional[str] = None):
        """A static factory method."""
        if isinstance(runs, Run):
            name = name or runs.path

        return cls(name=name or '', runs=runs)

    def __getitem__(self, k):
        if isinstance(k, int):
            return self.runs[k]

        if k not in self.columns:
            raise KeyError(k)

        return pd.DataFrame({r.path: r.df[k] for r in self.runs})

    def __repr__(self) -> str:
        return (f"Hypothesis({self.name!r}, <{len(self.runs)} runs>)")

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
    def grouped(self) -> DataFrameGroupBy:
        return pd.concat(self._dataframes, sort=False).groupby(level=0)

    def empty(self) -> bool:
        sentinel = object()
        return next(iter(self.grouped), sentinel) is sentinel  # O(1)

    @property
    def _dataframes(self) -> List[pd.DataFrame]:
        """Get all dataframes associated with all the runs."""
        def _get_df(o):
            if isinstance(o, pd.DataFrame): return o
            else: return o.df
        return [_get_df(r) for r in self.runs]

    @property
    def columns(self) -> Iterable[str]:
        return util.merge_list(*[df.columns for df in self._dataframes])

    def rolling(self, *args, **kwargs):
        return self.grouped.rolling(*args, **kwargs)

    def mean(self, *args, **kwargs) -> pd.DataFrame:
        g = self.grouped
        return g.mean(*args, **kwargs)

    def std(self, *args, **kwargs) -> pd.DataFrame:
        g = self.grouped
        return g.std(*args, **kwargs)

    def min(self, *args, **kwargs) -> pd.DataFrame:
        g = self.grouped
        return g.min(*args, **kwargs)

    def max(self, *args, **kwargs) -> pd.DataFrame:
        g = self.grouped
        return g.max(*args, **kwargs)


class Experiment(Iterable[Hypothesis]):

    @typechecked
    def __init__(self,
                 name: Optional[str] = None,
                 hypotheses: Iterable[Hypothesis] = None,
                 ):
        self._name = name if name is not None else ""
        self._hypotheses: MutableMapping[str, Hypothesis] = collections.OrderedDict()

        if isinstance(hypotheses, np.ndarray):
            hypotheses = list(hypotheses)

        for h in (hypotheses or []):
            if not isinstance(h, Hypothesis):
                raise TypeError("An element of hypotheses contains a wrong type: "
                                "{} ".format(type(h)))
            if h.name in self._hypotheses:
                raise ValueError(f"Duplicate hypothesis name: `{h.name}`")
            self._hypotheses[h.name] = h

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame,
                       by: str, run: str = 'run', *,
                       name: Optional[str] = None,
                       ) -> 'Experiment':
        '''Constructs a new Experiment object from a DataFrame instance
        structured as per the convention.'''
        ex = Experiment(name=name)
        for hypothesis_name, runs_df in df.groupby(by):
            runs = RunList(runs_df[run])
            h = runs.to_hypothesis(name=hypothesis_name)
            ex.add_hypothesis(h)
        return ex

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
        _runs = check_runs_type(runs)

        d = Hypothesis.of(name=hypothesis_name, runs=_runs)
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
        return self._name

    @property
    def title(self) -> str:
        return self._name

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
            f"Experiment('{self.name}', {len(self._hypotheses)} hypotheses: [ \n " +
            '\n '.join([repr(exp) for exp in self._hypotheses]) +
            "\n])"
        )

    def __getitem__(self, key: Union[str, Tuple],
                    ) -> Union[Hypothesis, np.ndarray, Run, pd.DataFrame]:
        """Return self[key]. `key` can be one of the following:
        - str: The hypothesis's name to retrieve.
        - int: An index [0, len(self)) in all hypothesis. A numpy-style fancy
          indexing is supported.
        - Tuple(hypo_key: str|int, column: str):
          - The first axis is the same as previous (hypothesis' name or index)
          - The second one is the column name. The return value will be same
            as self[hypo_key][column].
        """
        if isinstance(key, str):
            name = key
            return self._hypotheses[name]
        elif isinstance(key, int):
            try:
                name = next(itertools.islice(self._hypotheses.keys(), key, None))
            except StopIteration:
                raise IndexError("out of range: {} (should be < {})".format(
                    key, len(self._hypotheses)))
            return self._hypotheses[name]
        elif isinstance(key, tuple):
            hypo_key, column = key
            hypos = self[hypo_key]
            if isinstance(hypos, list):
                raise NotImplementedError("2-dim fancy indexing is not implementd")
            return hypos[column]   # type: ignore
        elif isinstance(key, Iterable):
            key = list(key)
            if all(isinstance(k, bool) for k in key):
                # fancy indexing through bool
                if len(key) != len(self._hypotheses):
                    raise IndexError(
                        "boolean index did not match indexed array along"
                        " dimension 0; dimension is {} but corresponding "
                        " boolean dimension is {}".format(
                            len(self._hypotheses), len(key)))
                r = np.empty(len(key), dtype=object)
                r[:] = list(self._hypotheses.values())
                return r[key]
            else:
                # fancy indexing through int?  # TODO: support str
                hypo_keys = list(self._hypotheses.keys())
                to_key = lambda k: k if isinstance(k, str) else hypo_keys[k]
            return [self._hypotheses[to_key(k)] for k in key]
        else:
            raise ValueError("Unsupported index: {}".format(key))

    def __setitem__(self, name: str,
                    hypothesis_or_runs: Union[Hypothesis, List[Run]]) -> Hypothesis:
        """An dict-like method for adding hypothesis or runs."""
        if isinstance(hypothesis_or_runs, Hypothesis):
            if hypothesis_or_runs in self._hypotheses:
                raise ValueError(f"A hypothesis named {name} already exists")
            self._hypotheses[name] = hypothesis_or_runs
        else:
            # TODO metadata (e.g. color)
            self.add_runs(name, hypothesis_or_runs)   # type: ignore

        return self._hypotheses[name]

    @property
    def columns(self) -> Iterable[str]:
        # merge and uniquify all columns but preserving the order.
        return util.merge_list(*[h.columns for h in self._hypotheses.values()])

    @staticmethod
    def AGGREGATE_MEAN_LAST(portion: float):
        return (lambda series: series.rolling(
            max(1, int(len(series) * portion))).mean().iloc[-1])

    def summary(self, columns=None, aggregate=None) -> pd.DataFrame:
        '''Return a DataFrame that summarizes the current experiments,
        whose rows are all hypothesis.

        Args:
          columns: The list of columns to show. Defaults to `self.columns` plus
            `"index"`.
          aggregate: A function or a dict of functions ({column_name: ...})
            specifying a strategy to aggregate a `Series`. Defaults to take the
            average of the last 10% of the series.

        Example Usage:

        >>> pd.set_option('display.max_colwidth', 2000)   # hypothesis name can be long!
        >>> df = ex.summary(columns=['index', 'loss', 'return'])
        >>> df.style.background_gradient(cmap='viridis')

        '''
        columns = columns or (['index'] + list(self.columns))
        aggregate = aggregate or self.AGGREGATE_MEAN_LAST(0.1)

        df = pd.DataFrame({'hypothesis': [h.name for h in self.hypotheses]})
        hypo_means = [(h.mean() if not all(len(df) == 0 for df in h._dataframes)
                      else pd.DataFrame()) for h in self.hypotheses]

        for column in columns:
            def df_series(df: pd.DataFrame):
                if column == 'index':
                    return df.index
                if column not in df:
                    return []
                else:
                    return df[column].dropna()
            def aggregate_h(series):
                if len(series) == 0:
                    # after dropna, no numeric types to aggregate?
                    return np.nan
                aggregate_fn = aggregate
                if not callable(aggregate_fn):
                    aggregate_fn = aggregate[column]
                v = aggregate_fn(series) if column != 'index' else series.max()
                return v
            df[column] = [aggregate_h(df_series(hm)) for hm in hypo_means]
        return df

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
    from tensorflow.python.framework.dtypes import DType
    from tensorflow.core.util import event_pb2
    try:
        # avoid DeprecationWarning on tf_record_iterator
        from tensorflow.python._pywrap_record_io import RecordIterator
        def summary_iterator(path):
            for r in RecordIterator(path, ""):
                yield event_pb2.Event.FromString(r)
    except:
        from tensorflow.python.summary.summary_iterator import summary_iterator  # type: ignore

    def _read_proto_value(node, path: str):
        for p in path.split('.'):
            node = getattr(node, p, None)
            if node is None:
                return None
        return node

    def _extract_scalar_from_proto(value, step):
        if value.HasField('simple_value'):  # v1
            simple_value = _read_proto_value(value, 'simple_value')
            yield step, value.tag, simple_value
        elif value.HasField('metadata'):  # v2 eventfile
            if _read_proto_value(value, 'metadata.plugin_data.plugin_name') == 'scalars':
                t = _read_proto_value(value, 'tensor')
                if t:
                    dtype = DType(t.dtype).as_numpy_dtype
                    simple_value = np.frombuffer(t.tensor_content, dtype=dtype)[0]
                    yield step, value.tag, simple_value


    def iter_scalar_summary_from_event_file(event_file):
        for event in summary_iterator(event_file):
            step = int(event.step)
            if not event.HasField('summary'):
                continue
            for value in event.summary.value:
                yield from _extract_scalar_from_proto(value, step=step)

    # int(timestep) -> dict of columns
    all_data = defaultdict(dict)  # type: ignore
    for step, tag_name, value in iter_scalar_summary_from_event_file(event_file):
        all_data[step][tag_name] = value

    for t in list(all_data.keys()):
        all_data[t]['global_step'] = t

    df = pd.DataFrame(all_data).T

    # Reorder column names in a lexicographical order
    df = df.reindex(sorted(df.columns), axis=1)
    return df



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


def get_runs_serial(*path_globs, verbose=False, fillna=True) -> RunList:
    """
    Get a list of Run objects from the given path(s).
    Works in single-thread (slow, should not used outside debugging purposes).
    """
    runs = list(iter_runs_serial(*path_globs, verbose=verbose, fillna=fillna))

    if not runs:
        for path_glob in path_globs:
            print(f"Warning: No match found for pattern {path_glob}",
                  file=sys.stderr)

    return RunList(runs)


def _handle_path(p, verbose, fillna) -> Optional[Tuple[str, pd.DataFrame]]:
    try:
        df = parse_run(p, verbose=verbose, fillna=fillna)
        return p, df
    except (pd.errors.EmptyDataError, FileNotFoundError) as e:
        print(f"[!] {p} : {e}", file=sys.stderr, flush=True)
        return None


def get_runs_parallel(*path_globs, verbose=False, n_jobs=8, fillna=True,
                      pool_class=ThreadPool,
                      ) -> RunList:
    """
    Get a list of Run objects from the given path(s).
    Runs in parallel.
    """

    if isinstance(pool_class, str):
        Pool = {'threading': ThreadPool,
                'multiprocessing': MultiprocessPool,
                }.get(pool_class, None)
        if not Pool:
            raise ValueError("Unknown pool_class: {} ".format(pool_class) +
                             "(expected threading/multiprocessing)")
    elif callable(pool_class):
        Pool = pool_class
    else:
        raise TypeError("Unknown type for pool_class: {}".format(pool_class))

    with Pool(processes=n_jobs) as pool:   # type: ignore
        futures = []
        for path_glob in path_globs:
            paths = list(sorted(glob(path_glob)))
            if verbose and not paths:
                print(f"Warning: a glob pattern '{path_glob}' "
                       "did not match any files.", file=sys.stderr)

            for p in paths:
                future = pool.apply_async(_handle_path, [p, verbose, fillna])
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
    return RunList(result)


get_runs = get_runs_parallel
