"""Data structures for expt.

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
import fnmatch
import itertools
import os.path
import re
import sys
import types
from dataclasses import dataclass  # for python 3.6, backport needed
from multiprocessing.pool import Pool as MultiprocessPool
from multiprocessing.pool import ThreadPool
from typing import (Any, Callable, Iterable, Iterator, List, Mapping,
                    MutableMapping, Optional, Sequence, Tuple, TypeVar, Union)

import numpy as np
import pandas as pd
from pandas.core.accessor import CachedAccessor
from pandas.core.groupby.generic import DataFrameGroupBy
from scipy import interpolate
from typeguard import typechecked

from . import plot as _plot
from . import util
from .path_util import exists, glob, isdir, open

T = TypeVar('T')

try:
  from tqdm.auto import tqdm
except ImportError:
  tqdm = util.NoopTqdm

#########################################################################
# Data Classes
#########################################################################


@dataclass
class Run:
  """Represents a single run, containing one pd.DataFrame object
  as well as other metadata (path, etc.)
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
    return 'Run({path!r}, DataFrame with {rows} rows)'.format(
        path=self.path, rows=len(self.df))

  @property
  def columns(self) -> Sequence[str]:
    """Returns all column names."""
    return list(self.df.columns)  # type: ignore

  @property
  def name(self) -> str:
    """Returns the last segment of the path."""
    path = self.path.rstrip('/')
    return os.path.basename(path)

  def to_hypothesis(self) -> 'Hypothesis':
    """Create a new `Hypothesis` consisting of only this run."""
    return Hypothesis.of(self)

  def plot(self, *args, subplots=True, **kwargs):
    return self.to_hypothesis().plot(*args, subplots=subplots, **kwargs)

  def hvplot(self, *args, subplots=True, **kwargs):
    return self.to_hypothesis().hvplot(*args, subplots=subplots, **kwargs)


class RunList(Sequence[Run]):
  """A (immutable) list of Run objects, but with some useful utility
  methods such as filtering, searching, and handy format conversion."""

  def __init__(self, runs: Union[Run, Iterable[Run]]):
    runs = self._validate_type(runs)
    self._runs: List[Run] = list(runs)

  @classmethod
  def of(cls, runs: Iterable[Run]):
    if isinstance(runs, cls):
      return runs  # do not make a copy
    else:
      return cls(runs)  # RunList(runs)

  def _validate_type(self, runs) -> List[Run]:
    if not isinstance(runs, Iterable):
      raise TypeError(f"`runs` must be a Iterable, but given {type(runs)}")
    if isinstance(runs, Mapping):
      raise TypeError(f"`runs` should not be a dictionary, given {type(runs)} "
                      " (forgot to wrap with pd.DataFrame?)")

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

  def __next__(self):
    # This is a hack to prevent panda's pprint_thing() from converting
    # into a sequence of Runs.
    raise TypeError("'RunList' object is not an iterator.")

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
    """Return a DataFrame consisting of columns `name` and `run`."""
    return pd.DataFrame({
        'name': [r.name for r in self._runs],
        'run': self._runs,
    })

  def filter(self, fn: Union[Callable[[Run], bool], str]) -> 'RunList':
    """Apply a filter function (Run -> bool) and return the filtered runs
    as another RunList. If a string is given, we convert it as a matcher
    function (see fnmatch) that matches `run.name`."""
    if isinstance(fn, str):
      pat = str(fn)
      fn = lambda run: fnmatch.fnmatch(run.name, pat)
    return RunList(filter(fn, self._runs))

  def grep(self, regex: Union[str, 're.Pattern'], flags=0):
    """Apply a regex-based filter on the path of `Run`, and return the
    matched `Run`s as a RunList."""
    if isinstance(regex, str):
      regex = re.compile(regex, flags=flags)
    return self.filter(lambda r: bool(regex.search(r.path)))

  def map(self, func: Callable[[Run], Any]) -> List:
    """Apply func for each of the runs. Return the transformation
    as a plain list."""
    return list(map(func, self._runs))

  def to_hypothesis(self, name: str) -> 'Hypothesis':
    """Create a new Hypothesis instance containing all the runs
    as the current RunList instance."""
    return Hypothesis.of(self, name=name)

  def groupby(
      self,
      by: Callable[[Run], T],
      *,
      name: Callable[[T], str] = str,
  ) -> Iterator[Tuple[T, 'Hypothesis']]:
    r"""Group runs into hypotheses with the key function `by` (Run -> key).

    This will enumerate tuples (`group_key`, Hypothesis) where `group_key`
    is the result of the key function for each group, and a Hypothesis
    object (with name `name(group_key)`) will consist of all the runs
    mapped to the same group.

    Args:
      by: a key function for groupby operation. (Run -> Key)
      name: a function that maps the group (Key) into Hypothesis name (str).

    Example:
      >>> key_func = lambda run: re.search(
      >>>     "algo=(\w+),lr=([.0-9]+)", run.name).group(1, 2)
      >>> for group_name, hypothesis in runs.groupby(key_func):
      >>>   ...

    """
    series = pd.Series(self._runs)
    groupby = series.groupby(lambda i: by(series[i]))

    group: T
    for group, runs_in_group in groupby:  # type: ignore
      yield group, Hypothesis.of(runs_in_group, name=name(group))

  def extract(self, pat: str, flags: int = 0) -> pd.DataFrame:
    r"""Extract capture groups in the regex pattern `pat` as columns.

    Example:
      >>> runs[0].name
      "ppo-halfcheetah-seed0"
      >>> pattern = r"(?P<algo>[\w]+)-(?P<env_id>[\w]+)-seed(?P<seed>[\d]+)")
      >>> df = runs.extract(pattern)
      >>> assert list(df.columns) == ['algo', 'env_id', 'seed', 'run']

    """
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
  def of(cls,
         runs: Union[Run, Iterable[Run]],
         *,
         name: Optional[str] = None) -> 'Hypothesis':
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

  def __repr__(self, indent='') -> str:
    return (f"Hypothesis({self.name!r}, {len(self.runs)} runs: [\n" +
            (indent + " ") +
            (',\n' + indent + " ").join([repr(run) for run in self.runs]) +
            ",\n" + indent + "])")

  def __len__(self) -> int:
    return len(self.runs)

  def __hash__(self):
    return hash(id(self))

  def __next__(self):
    # This is a hack to prevent panda's pprint_thing() from converting
    # into a sequence of Runs.
    raise TypeError("'Hypothesis' object is not an iterator.")

  def describe(self) -> pd.DataFrame:
    """Report a descriptive statistics as a DataFrame,
    after aggregating all runs (e.g., mean)."""
    return self.mean().describe()

  def summary(self) -> pd.DataFrame:
    """Return a DataFrame that summarizes the current hypothesis."""
    return Experiment(self.name, [self]).summary()

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
      if isinstance(o, pd.DataFrame):
        return o
      else:
        return o.df

    return [_get_df(r) for r in self.runs]

  @property
  def columns(self) -> Iterable[str]:
    return util.merge_list(*[df.columns for df in self._dataframes])

  def rolling(self, *args, **kwargs):
    return self.grouped.rolling(*args, **kwargs)

  def mean(self, *args, **kwargs) -> pd.DataFrame:
    g = self.grouped
    return g.mean(*args, **kwargs)  # type: ignore

  def std(self, *args, **kwargs) -> pd.DataFrame:
    g = self.grouped
    return g.std(*args, **kwargs)  # type: ignore

  def min(self, *args, **kwargs) -> pd.DataFrame:
    g = self.grouped
    return g.min(*args, **kwargs)  # type: ignore

  def max(self, *args, **kwargs) -> pd.DataFrame:
    g = self.grouped
    return g.max(*args, **kwargs)  # type: ignore

  def interpolate(self, x_column: Optional[str] = None, *, n_samples: int):
    """Interpolate by uniform subsampling, and return a processed hypothesis.

    This is useful when the hypothesis' individual runs may have heterogeneous
    index (or the x-axis column).

    Args:
      x_column (Optional): the name of column to interpolate on. If None,
        we interpolate and subsample based on the index of dataframe.
      n_samples (int): The number of points to use when subsampling and
        interpolation. Should be greater than 2.

    Returns:
      A copy of Hypothesis with the same name, whose runs (DataFrames) consists
      of interpolated and resampled data for all the numeric columns. All
      other non-numeric columns will be dropped. Each DataFramew will have
      the specified x_column as its Index.
    """
    # For each dataframe (run), interpolate on x
    if x_column is None:
      x_series = pd.concat([pd.Series(df.index) for df in self._dataframes])
    else:
      if x_column not in self.columns:
        raise ValueError(f"Unknown column: {x_column}")
      x_series = pd.concat([df[x_column] for df in self._dataframes])

    x_min = x_series.min()
    x_max = x_series.max()
    x_samples = np.linspace(x_min, x_max, num=n_samples)  # type: ignore

    # Get interpolated dataframes.
    dfs_interp = []
    for df in self._dataframes:
      df: pd.DataFrame
      if x_column is not None:
        df = df.set_index(x_column)  # type: ignore

      # filter out non-numeric columns.
      df = df.select_dtypes(['number'])

      # yapf: disable
      def _interpolate_if_numeric(y_series: pd.Series):
        if y_series.dtype.kind in ('i', 'f'):
          # y_series might contain NaN values, but interp1d does not
          # properly deal with nan. So we filter both of x and y series.
          idx_valid = ~np.isnan(y_series)
          if idx_valid.sum() >= 2:
            return interpolate.interp1d(df.index[idx_valid],
                                        y_series[idx_valid],
                                        bounds_error=False)(x_samples)
          else:
            # Insufficient data due to a empty/crashed run.
            # Ignore the corner case! TODO: Add warning message.
            return pd.Series(np.full_like(x_samples, np.nan, dtype=float))
        else:
          # maybe impossible to interpolate (i.e. object): skip it.
          # This column will be filtered out.
          assert False, "should have been dropped earlier"
      # yapf: enable

      df_interp = df.apply(_interpolate_if_numeric)
      df_interp[x_column] = x_samples
      df_interp.set_index(x_column, inplace=True)
      dfs_interp.append(df_interp)

    assert len(dfs_interp) == len(self.runs)
    return Hypothesis(
        name=self.name,
        runs=[
            Run(r.path, df_new) for (r, df_new) in zip(self.runs, dfs_interp)
        ])


class Experiment(Iterable[Hypothesis]):

  @typechecked
  def __init__(
      self,
      name: Optional[str] = None,
      hypotheses: Iterable[Hypothesis] = None,
  ):
    self._name = name if name is not None else ""
    self._hypotheses: MutableMapping[str, Hypothesis]
    self._hypotheses = collections.OrderedDict()

    if isinstance(hypotheses, np.ndarray):
      hypotheses = list(hypotheses)

    for h in (hypotheses or []):
      if not isinstance(h, Hypothesis):
        raise TypeError("An element of hypotheses contains a wrong type: "
                        "expected {}, but given {} ".format(
                            Hypothesis, type(h)))
      if h.name in self._hypotheses:
        raise ValueError(f"Duplicate hypothesis name: `{h.name}`")
      self._hypotheses[h.name] = h

  @classmethod
  def from_dataframe(
      cls,
      df: pd.DataFrame,
      by: Optional[Union[str, List[str]]] = None,
      *,
      run_column: str = 'run',
      hypothesis_namer: Callable[..., str] = str,
      name: Optional[str] = None,
  ) -> 'Experiment':
    """Constructs a new Experiment object from a DataFrame instance
    structured as per the convention.

    Args:
      by (str, List[str]): The column name to group by. If None (default),
        it will try to automatically determine from the dataframe if there
        is only one column other than `run_column`.
      run_column (str): The column name that contains `Run` objects.
        See also `RunList.to_dataframe()`.
      hypothesis_namer: This is a mapping that transforms the group key
        (a str or tuple) that pandas groupby produces into hypothesis name.
        This function should take one positional argument for the group key.
      name: The name for the produced `Experiment`.
    """
    if by is None:
      # Automatically determine the column from df.
      by_columns = list(sorted(set(df.columns).difference([run_column])))
      if len(by_columns) != 1:
        raise ValueError("Cannot automatically determine the column to "
                         "group by. Candidates: {}".format(by_columns))
      by = next(iter(by_columns))

    ex = Experiment(name=name)
    for hypothesis_key, runs_df in df.groupby(by):
      hypothesis_name = hypothesis_namer(hypothesis_key)
      runs = RunList(runs_df[run_column])
      h = runs.to_hypothesis(name=hypothesis_name)
      ex.add_hypothesis(h)
    return ex

  def add_runs(
      self,
      hypothesis_name: str,
      runs: List[Union[Run, Tuple[str, pd.DataFrame], pd.DataFrame]],
      *,
      color=None,
      linestyle=None,
  ) -> Hypothesis:

    def check_runs_type(runs) -> List[Run]:
      if isinstance(runs, types.GeneratorType):
        runs = list(runs)
      if runs == []:
        return []
      if isinstance(runs, Run):
        runs = [runs]

      return [Run.of(r) for r in runs]

    _runs = check_runs_type(runs)

    d = Hypothesis.of(name=hypothesis_name, runs=_runs)
    return self.add_hypothesis(d, extend_if_conflict=True)

  @typechecked
  def add_hypothesis(
      self,
      h: Hypothesis,
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
  def hypotheses(self) -> Sequence[Hypothesis]:
    return tuple(self._hypotheses.values())

  def select_top(
      self,
      key,
      k=None,
      descending=True,
  ) -> Union[Hypothesis, Sequence[Hypothesis]]:
    """Choose a hypothesis that has the largest value on the specified column.

    Args:
      key: str (y_name) or Callable(Hypothesis -> number).
      k: If None, the top-1 hypothesis will be returned. Otherwise (integer),
        top-k hypotheses will be returned as a tuple.
      descending: If True, the hypothesis with largest value in key will be
        chosen. If False, the hypothesis with smallest value will be chosen.

    Returns: the top-1 hypothesis (if `k` is None) or a tuple of k hypotheses
      in the order specified by `key`.
    """
    if k is not None and k <= 0:
      raise ValueError("k must be greater than 0.")
    if k is not None and k > len(self._hypotheses):
      raise ValueError("k must be smaller than the number of "
                       "hypotheses ({})".format(len(self._hypotheses)))

    if isinstance(key, str):
      y = str(key)  # make a copy for closure
      if descending:
        key = lambda h: h.mean()[y].max()
      else:
        key = lambda h: h.mean()[y].min()
    elif callable(key):
      pass  # key: Hypothesis -> scalar.
    else:
      raise TypeError(
          f"`key` must be a str or a callable, but got: {type(key)}")

    candidates = sorted(self.hypotheses, key=key, reverse=descending)
    assert isinstance(candidates, list)
    if k is None:
      return candidates[0]
    else:
      return candidates[:k]

  def __iter__(self) -> Iterator[Hypothesis]:
    return iter(self._hypotheses.values())

  def __repr__(self) -> str:
    return (
        f"Experiment('{self.name}', {len(self._hypotheses)} hypotheses: [\n " +
        ',\n '.join([h.__repr__(indent=' ') for h in self.hypotheses]) +
        ",\n])")

  def __getitem__(
      self,
      key: Union[str, Tuple],
  ) -> Union[Hypothesis, np.ndarray, Run, pd.DataFrame]:
    """Return self[key].

    `key` can be one of the following:
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
        _keys = self._hypotheses.keys()
        name = next(itertools.islice(_keys, key, None))
      except StopIteration:
        raise IndexError("out of range: {} (should be < {})".format(
            key, len(self._hypotheses)))
      return self._hypotheses[name]
    elif isinstance(key, tuple):
      hypo_key, column = key
      hypos = self[hypo_key]
      if isinstance(hypos, list):
        raise NotImplementedError("2-dim fancy indexing is not implemented")
      return hypos[column]  # type: ignore
    elif isinstance(key, Iterable):
      key = list(key)  # type: ignore
      if all(isinstance(k, bool) for k in key):
        # fancy indexing through bool
        if len(key) != len(self._hypotheses):
          raise IndexError("boolean index did not match indexed array along"
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
      return [self._hypotheses[to_key(k)] for k in key]  # type: ignore
    else:
      raise ValueError("Unsupported index: {}".format(key))

  def __setitem__(
      self,
      name: str,
      hypothesis_or_runs: Union[Hypothesis, List[Run]],
  ) -> Hypothesis:
    """An dict-like method for adding hypothesis or runs."""
    if isinstance(hypothesis_or_runs, Hypothesis):
      if hypothesis_or_runs in self._hypotheses:
        raise ValueError(f"A hypothesis named {name} already exists")
      self._hypotheses[name] = hypothesis_or_runs
    else:
      # TODO metadata (e.g. color)
      self.add_runs(name, hypothesis_or_runs)  # type: ignore

    return self._hypotheses[name]

  @property
  def columns(self) -> Iterable[str]:
    # merge and uniquify all columns but preserving the order.
    return util.merge_list(*[h.columns for h in self._hypotheses.values()])

  @staticmethod
  def AGGREGATE_MEAN_LAST(portion: float):
    return (lambda series: series.rolling(max(1, int(len(series) * portion))
                                          ).mean().iloc[-1])  # yapf: disable

  def summary(self, columns=None, aggregate=None) -> pd.DataFrame:
    """Return a DataFrame that summarizes the current experiments,
    whose rows are all hypothesis.

    Args:
      columns: The list of columns to show. Defaults to `self.columns` plus
        `"index"`.
      aggregate: A function or a dict of functions ({column_name: ...})
        specifying a strategy to aggregate a `Series`. Defaults to take the
        average of the last 10% of the series.

    Example Usage:

      >>> pd.set_option('display.max_colwidth', 2000)   # deal with long names?
      >>> df = ex.summary(columns=['index', 'loss', 'return'])
      >>> df.style.background_gradient(cmap='viridis')

    """
    columns = columns or (['index'] + list(self.columns))
    aggregate = aggregate or self.AGGREGATE_MEAN_LAST(0.1)

    df = pd.DataFrame({'hypothesis': [h.name for h in self.hypotheses]})
    hypo_means = [
        (h.mean() if not all(len(df) == 0
                             for df in h._dataframes) else pd.DataFrame())
        for h in self.hypotheses
    ]

    for column in columns:

      def df_series(df: pd.DataFrame):
        # TODO: What if a named column is used as an index?
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
          aggregate_fn = aggregate[column]  # type: ignore
        v = aggregate_fn(series) if column != 'index' else series.max()
        return v

      df[column] = [aggregate_h(df_series(hm)) for hm in hypo_means]
    return df

  def interpolate(self, x_column: Optional[str] = None, *, n_samples: int):
    """Apply interpolation to each of the hypothesis, and return a copy
    of new Experiment (and its children Hypothesis/Run) object.

    See: Hypothesis.interpolate().
    """
    return Experiment(
        name=self.name,
        hypotheses=[
            h.interpolate(x_column, n_samples=n_samples)
            for h in self.hypotheses
        ],
    )

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


def parse_run(run_folder, fillna=False, verbose=False) -> pd.DataFrame:
  """Create a pd.DataFrame object from a single directory."""
  if verbose:
    # TODO Use python logging
    print(f"Reading {run_folder} ...", file=sys.stderr, flush=True)

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
    raise pd.errors.EmptyDataError(  # type: ignore
        f"Cannot handle dir: {run_folder}")

  # add some optional metadata... (might not be preserved afterwards)
  if df is not None:
    df.path = run_folder
  return df


def parse_run_progresscsv(run_folder,
                          fillna=False,
                          verbose=False) -> pd.DataFrame:
  """Create a pandas DataFrame object from progress.csv per convention."""
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
    print(f"parse_run (csv): Reading {detected_csv}",
          file=sys.stderr, flush=True)  # yapf: disable

  df: pd.DataFrame
  with open(detected_csv, mode='r') as f:
    df = pd.read_csv(f)  # type: ignore

  if fillna:
    df = df.fillna(0)

  return df


def parse_run_tensorboard(run_folder,
                          fillna=False,
                          verbose=False) -> pd.DataFrame:
  """Create a pandas DataFrame from tensorboard eventfile or run directory."""
  event_file = list(
      sorted(glob(os.path.join(run_folder, '*events.out.tfevents.*'))))

  if not event_file:  # no event file detected
    raise pd.errors.EmptyDataError(  # type: ignore
        f"No event file detected in {run_folder}")
  event_file = event_file[-1]  # pick the last one
  if verbose:
    print(f"parse_run (tfevents) : Reading {event_file} ...",
          file=sys.stderr, flush=True)  # yapf: disable

  from collections import defaultdict

  from tensorflow.core.util import event_pb2
  from tensorflow.python.framework.dtypes import DType
  try:
    # avoid DeprecationWarning on tf_record_iterator
    # pyright: reportMissingImports=false
    from tensorflow.python._pywrap_record_io import RecordIterator

    def summary_iterator(path):
      for r in RecordIterator(path, ""):
        yield event_pb2.Event.FromString(r)  # type: ignore
  except Exception:
    from tensorflow.python.summary.summary_iterator import summary_iterator

  def _read_proto(node, path: str):
    for p in path.split('.'):
      node = getattr(node, p, None)
      if node is None:
        return None
    return node

  def _extract_scalar_from_proto(value, step):

    if value.HasField('simple_value'):  # v1
      simple_value = _read_proto(value, 'simple_value')
      yield step, value.tag, simple_value

    elif value.HasField('metadata'):  # v2 eventfile
      plugin_name = _read_proto(value, 'metadata.plugin_data.plugin_name')
      if plugin_name == 'scalars':
        t = _read_proto(value, 'tensor')
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


def _validate_run_postprocess(run):
  if not isinstance(run, Run):
    raise TypeError("run_postprocess_fn did not return a "
                    "Run object; given {}".format(type(run)))
  return run


def iter_runs_serial(
    *path_globs,
    verbose=False,
    fillna=True,
    run_postprocess_fn=None,
) -> Iterator[Run]:
  """Enumerate Run objects from the given path(s)."""

  for path_glob in path_globs:
    if verbose:
      print(f"get_runs: {str(path_glob)}", file=sys.stderr)

    paths = list(sorted(glob(path_glob)))
    for p in paths:
      run = _handle_path(p, verbose=verbose, fillna=fillna,
                         run_postprocess_fn=run_postprocess_fn,
                         )  # yapf: disable
      if run:
        yield run


def get_runs_serial(*path_globs,
                    verbose=False,
                    fillna=True,
                    run_postprocess_fn=None) -> RunList:
  """Get a list of Run objects from the given path(s).

  This works in single-thread (very slow, should not used other than
  debugging purposes).
  """
  runs = list(
      iter_runs_serial(
          *path_globs,
          verbose=verbose,
          fillna=fillna,
          run_postprocess_fn=run_postprocess_fn))

  if not runs:
    for path_glob in path_globs:
      print(f"Warning: No match found for pattern {path_glob}",
            file=sys.stderr)  # yapf: disable

  return RunList(runs)


def _handle_path(p, verbose, fillna, run_postprocess_fn=None) -> Optional[Run]:
  try:
    df = parse_run(p, verbose=verbose, fillna=fillna)
    run = Run(path=p, df=df)
    if run_postprocess_fn:
      run = run_postprocess_fn(run)
      _validate_run_postprocess(run)
    return run
  except (pd.errors.EmptyDataError, FileNotFoundError) as e:  # type: ignore
    # Ignore empty data.
    print(f"[!] {p} : {e}", file=sys.stderr, flush=True)
    return None


def get_runs_parallel(
    *path_globs,
    verbose=False,
    n_jobs=8,
    fillna=True,
    pool_class=MultiprocessPool,
    progress_bar=True,
    run_postprocess_fn=None,
) -> RunList:
  """Get a list of Run objects from the given path(s).

    Runs in parallel.
    """

  if isinstance(pool_class, str):
    Pool = {
        'threading': ThreadPool,
        'multiprocessing': MultiprocessPool,
    }.get(pool_class, None)
    if not Pool:
      raise ValueError("Unknown pool_class: {} ".format(pool_class) +
                       "(expected: threading or multiprocessing)")
  elif callable(pool_class):
    Pool = pool_class
  else:
    raise TypeError("Unknown type for pool_class: {}".format(pool_class))

  with Pool(processes=n_jobs) as pool:  # type: ignore
    pbar = tqdm(total=1) if progress_bar else util.NoopTqdm()

    def _pbar_callback_done(run):
      del run  # unused
      pbar.update(1)
      pbar.refresh()

    def _pbar_callback_error(e):
      del e  # unused
      pbar.bar_style = 'danger'  # type: ignore

    futures = []
    for path_glob in path_globs:
      paths = list(sorted(glob(path_glob)))
      if verbose and not paths:
        print(f"Warning: a glob pattern '{path_glob}' "
              "did not match any files.", file=sys.stderr)  # yapf: disable

      for p in paths:
        future = pool.apply_async(
            _handle_path,
            args=[p, verbose, fillna, run_postprocess_fn],
            callback=_pbar_callback_done,
            error_callback=_pbar_callback_error)
        futures.append(future)

      # The total number can grow while some jobs are running.
      _completed = int(pbar.n)
      pbar.reset(total=len(futures))
      pbar.n = pbar.last_print_n = _completed
      pbar.refresh()

    hits = 0
    result = []
    for future in futures:
      run: Optional[Run] = future.get()
      if run:
        hits += 1
        result.append(run)

    # All runs have been collected, close the progress bar.
    pbar.close()

  if not hits:
    for path_glob in path_globs:
      print(f"Warning: No match found for pattern {path_glob}",
            file=sys.stderr)  # yapf: disable
  return RunList(result)


get_runs = get_runs_parallel
