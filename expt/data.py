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

from __future__ import annotations

import collections
import copy
import dataclasses
import difflib
import fnmatch
from importlib import import_module as _import
import itertools
import os.path
import re
import types
from typing import (Any, Callable, cast, Dict, Hashable, Iterable, Iterator,
                    List, Mapping, MutableMapping, Optional, overload,
                    Sequence, Tuple, TYPE_CHECKING, TypeVar, Union)
from typing_extensions import Literal

import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
import scipy.interpolate
from typeguard import typechecked

from expt import util

T = TypeVar('T')

if hasattr(types, 'EllipsisType'):
  EllipsisType = types.EllipsisType
else:
  EllipsisType = type(...)

#########################################################################
# Data Classes
#########################################################################

RunConfig = Mapping[str, Any]


@dataclasses.dataclass
class Run:
  """Represents a single run, containing one pd.DataFrame object
  as well as other metadata (path, config, etc.)
  """
  path: str
  df: pd.DataFrame
  config: Optional[RunConfig] = None

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

  def with_config(self, config: RunConfig) -> Run:
    """Create a new Run instance with the given `config`."""
    if not isinstance(config, Mapping) and callable(config):
      config = config(self)
    if not isinstance(config, Mapping):
      raise TypeError(f"`config` must be a Mapping, but given {type(config)}")
    return dataclasses.replace(self, config=config)

  def to_hypothesis(self) -> Hypothesis:
    """Create a new `Hypothesis` consisting of only this run."""
    return Hypothesis.of(self)

  @property
  def _dataframes(self) -> List[pd.DataFrame]:
    return [self.df]

  def summary(self, **kwargs) -> pd.DataFrame:
    """Return a DataFrame that summarizes the current run."""
    df = self.to_hypothesis().summary(**kwargs)
    if 'hypothesis' in df:  # e.g., when name=False
      df['hypothesis'] = df['hypothesis'].apply(lambda h: h[0])
      df = df.rename(columns={'hypothesis': 'run'})
      assert 'run' in df.columns, str(df.columns)
    return df

  def plot(self, *args, subplots=True, **kwargs):
    return self.to_hypothesis().plot(*args, subplots=subplots, **kwargs)

  def hvplot(self, *args, subplots=True, **kwargs):
    return self.to_hypothesis().hvplot(*args, subplots=subplots, **kwargs)


def _default_config_fn(run: Run) -> RunConfig:
  if run.config is not None:
    return run.config
  raise ValueError(
      f"A Run with name `{run.name}` does not have config. "
      "For those runs that do not have any config data available "
      "(see config_reader), consider using the config_fn=... parameter.")


class RunList(Sequence[Run]):
  """A (immutable) list of Run objects, but with some useful utility
  methods such as filtering, searching, and handy format conversion."""

  def __init__(self, runs: Run | Iterable[Run]):
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

  # yapf: disable
  @overload
  def __getitem__(self, index_or_slice: int) -> Run: ...
  @overload
  def __getitem__(self, index_or_slice: slice) -> RunList: ...
  # yapf: enable

  def __getitem__(self, index_or_slice) -> Run | RunList:
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

  # TODO: Make this more configurable.
  INDEX_EXCLUDE_DEFAULT = (
      'seed',
      'random_seed',
      'log_dir',
      'train_dir',
      'ckpt_dir',
      'checkpoint_dir',
  )

  def varied_config_keys(
      self,
      config_fn: Callable[[Run], RunConfig] = _default_config_fn,
      excludelist: Sequence[str] = INDEX_EXCLUDE_DEFAULT,
  ) -> Sequence[str]:
    """Get a list of config keys (or indices in to_dataframe) that have more
    than two different unique values. If all the configs are identical, the
    list will contain all the unique config keys existing in any of the runs.

    Args:
      config_fn: Config for Runs. By default, Run.config will be used.
      excludelist: Lists which keys will be excluded from the result.
    """
    return tuple(
        k for k in varied_config_keys(self, config_fn=config_fn)
        if k not in excludelist)

  def to_dataframe(
      self,
      include_config: bool = True,
      config_fn: Optional[Callable[[Run], RunConfig]] = None,
      index_keys: Optional[Sequence[str]] = None,
      index_excludelist: Sequence[str] = INDEX_EXCLUDE_DEFAULT,
      as_hypothesis: bool = False,
      hypothesis_namer: Optional[  # (run_config, runs) -> str
          Callable[[RunConfig, Sequence[Run]], str]] = None,
      include_summary: bool = False,
  ) -> pd.DataFrame:
    """Return a DataFrame of runs, with of columns `name` and `run`
    (plus some extra columns as per config_fn).

    The resulting dataframe will have a MultiIndex, consisting of all the
    column names produced by config_fn() but excluding per `index_excludelist`.
    The order of index column is the natural, sequential order found
    from the config dict (whose key is ordered).

    Args:
      include_config: If True (default), the dataframe will have run config
        as a MultiIndex.
      config_fn: A function that returns the config dict (mapping) for each
        run. This should be a function that takes a Run as the sole argument
        and returns Dict[str, number]. If not given (or None),
        `run.config` will be used instead, if available.
      config_fn: If given, this should be a function that takes a Run
        and returns Dict[str, number]. Additional series will be added
        to the dataframe from the result of this function.
      index_keys: A list of column names to include in the multi index.
        If omitted (using the default setting), all the keys from run.config
        that have at least two different unique values will be used
        (see config_fn). If there is only one run, all the columns will be
        used as the default index_keys.
      index_excludelist: A list of column names to exclude from multi index.
        Explicitly set as the empty list if you want to include the
        default excludelist names (seed, random_seed).
      as_hypothesis: If True, all the runs for each group will be merged
        as a Hypothesis. Default value is False, where individual runs will
        appear, one at a row. Only effective when config_fn is given.
      hypothesis_namer: A function that determines the name of Hypothesis
        given a group of runs. The function should takes two arguments,
        (run_config as dict, list of runs). Used when `as_hypothesis` is True.
        See also: Experiment.from_dataframe()
      include_summary: If True, include the summary statistics of each run
        or hypothesis as additional columns in the returning dataframe.
        See Hypothesis.summary().
    """
    df = pd.DataFrame({
        'name': [r.name for r in self._runs],
        'run': self._runs,
    })

    config_keys = []
    if include_config:
      if config_fn is None:
        config_fn = _default_config_fn

      if index_keys is None:  # using default index
        index_keys = varied_config_keys(self._runs, config_fn=config_fn)

      for i, run in enumerate(self._runs):
        config: Mapping[str, Any] = config_fn(run)
        if not isinstance(config, Mapping):
          raise ValueError("config_fn should return a dict-like object.")

        r_keys = index_keys
        for k in r_keys:
          if k not in config:
            raise ValueError(
                f"'{k}' not found in the config of {run}. Close matches: " +
                str(difflib.get_close_matches(k, config.keys())))
          v = config[k]
          if isinstance(v, list):
            # list is not hashable and not immutable, convert to a tuple.
            v = tuple(v)

          dtype = np.array(v).dtype
          if k not in df:
            # Create a new column if not exists
            # TODO: Consider NaN as a placeholder value rather than empty str.
            df[k] = pd.Series([''] * len(self._runs), dtype=object)
            if k not in index_excludelist:
              config_keys.append(k)

          try:
            df.at[i, k] = v
          except ValueError as e:
            raise ValueError(f"Failed to assign index for '{k}' "
                             f"with type {type(v)}") from e

    # Move 'name' and 'run' to the rightmost column.
    df.insert(len(df.columns) - 1, 'name', df.pop('name'))
    df.insert(len(df.columns) - 1, 'run', df.pop('run'))

    # Automatically set multi-index.
    if config_keys:
      df = df.set_index(config_keys, inplace=False)  # type: ignore
      df = df.sort_index(inplace=False)  # type: ignore
      assert df is not None

      # pandas groupby(dropna=...) has a bug that rows with any nan value
      # in the multi-index will be incorrectly dropped (see pandas#36060)
      # As a workaround, we fill the nan in the index with an empty string.
      df.index = pd.MultiIndex.from_frame(df.index.to_frame().fillna(""))

    # Aggregate runs into hypothesis if needed.
    if as_hypothesis:

      if not config_keys:
        raise ValueError("No config were detected. "
                         "Use config_fn or make sure Runs have config set.")

      if hypothesis_namer is None:
        # yapf: disable
        def _hypothesis_namer(config: Mapping[str, Any],
                              runs: Sequence[Run]) -> str:
          return "; ".join([f"{k}={v}" for (k, v) in config.items()])
        hypothesis_namer = _hypothesis_namer
        # yapf: enable

      def _group_to_hypothesis(g: pd.DataFrame) -> Hypothesis:
        # All runs in the group share the same keys and values.
        # Note that config lies in the (multi)index.
        group_config: Dict[str, Any] = dict(zip(g.index.names, g.index[0]))
        h_name = hypothesis_namer(group_config, g.run.values)
        return Hypothesis(h_name, g['run'].values, config=group_config)

      # yapf: disable
      df = (df
            .groupby(config_keys, dropna=False)  # dropna bug (pandas#36060)
            .apply(_group_to_hypothesis)
            .to_frame(name="hypothesis"))
      # yapf: enable

    else:
      if hypothesis_namer is not None:
        raise ValueError("When hypothesis_namer is set, "
                         "as_hypothesis must be True")

    if include_summary:
      series = df['run'] if 'run' in df else df['hypothesis']
      assert series is not None

      # TODO check name?
      df = series.apply(lambda h: h.summary(name=False).iloc[0])

    return df  # type: ignore

  def to_experiment(self, **kwargs) -> Experiment:
    return Experiment.from_runs(self, **kwargs)

  def filter(self, fn: Callable[[Run], bool] | str | re.Pattern) -> RunList:
    """Apply a filter and return the filtered runs as another RunList.

    The filter is a function (Run -> bool). Only runs that evaluates this
    function as True will be selected.

    As a special case, if a string is given as an argument, we will use it
    as the pattern for fnmatch. This can be useful for filtering runs by name.
    If a regex Pattern (re.compile) is given, all the runs whose name matches
    the pattern in part (via re.search) will be selected.
    """
    if isinstance(fn, str):
      pat = str(fn)
      fn = lambda run: fnmatch.fnmatch(run.name, pat)
    elif isinstance(fn, re.Pattern):
      pat = fn
      fn = lambda run: bool(pat.search(run.name))
    return RunList(filter(fn, self._runs))

  def grep(self, regex: str | re.Pattern, flags=0):
    """Apply a regex-based filter on the path of `Run`, and return the
    matched `Run`s as a RunList."""
    if isinstance(regex, str):
      regex = re.compile(regex, flags=flags)
    return self.filter(lambda r: bool(regex.search(r.path)))

  def map(self, func: Callable[[Run], Any]) -> List:
    """Apply func for each of the runs. Return the transformation
    as a plain list."""
    return list(map(func, self._runs))

  def to_hypothesis(self, name: str) -> Hypothesis:
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
      >>> key_func: Callable[[Run], Tuple[str, str]]
      >>> key_func = lambda run: re.search(
      >>>     "algo=(\w+),lr=([.0-9]+)", run.name).group(1, 2)
      >>> for group_name, hypothesis in runs.groupby(key_func):
      >>>   # group_name: Tuple[str, str]
      >>>   ...

    """
    series = pd.Series(self._runs)
    groupby = series.groupby(lambda i: by(series[i]))

    group: T
    for group, runs_in_group in groupby:  # type: ignore
      yield group, Hypothesis.of(runs_in_group, name=name(group))

  def extract(self, pat: str, flags: int = 0) -> pd.DataFrame:
    r"""Extract capture groups in the regex pattern `pat` as columns.

    DEPRECATED. Use to_dataframe() explicitly.

    Example:
      >>> runs[0].name
      "ppo-halfcheetah-seed0"
      >>> pattern = r"(?P<algo>[\w]+)-(?P<env_id>[\w]+)-seed(?P<seed>[\d]+)")
      >>> df = runs.extract(pattern)
      >>> assert list(df.columns) == ['algo', 'env_id', 'seed', 'run']

    """
    df: pd.DataFrame = self.to_dataframe(config_fn=lambda r: r.config or {})
    df = df['name'].str.extract(pat, flags=flags)
    df['run'] = list(self._runs)
    return df


def varied_config_keys(
    runs: Sequence[Run],
    config_fn: Callable[[Run], RunConfig] = _default_config_fn,
) -> Sequence[str]:
  """Get a list of config keys (or indices in to_dataframe) that have more than
  two different unique values. If all the configs are identical, the list
  will contain all the unique config keys existing in any of the runs."""

  key_values = collections.defaultdict(set)
  for r in runs:
    for k, v in config_fn(r).items():
      if not isinstance(v, Hashable):
        v = str(v)  # for instance, list is not hashable, e.g., "[64, 64]"
      key_values[k].add(v)

  keys = tuple(k for (k, values) in key_values.items() if len(values) > 1)
  if not keys:
    # All the runs have identical config, so use all of them
    return tuple(key_values.keys())
  return keys


@dataclasses.dataclass
class Hypothesis(Iterable[Run]):
  """Represents a single Hypothesis.

  A Hypothesis a group of runs with the same configuration; can represent a
  variant, algorithm, or such an instance with a specific set of hyperparamters.
  """
  name: str
  runs: RunList
  style: Dict[str, Any]  # TODO: Use some typing. TODO: Add tests.
  config: Optional[RunConfig] = None

  @typechecked
  def __init__(
      self,
      name: str,
      runs: Union[Run, Iterable[Run]],
      *,
      style: Optional[Dict[str, Any]] = None,
      config: Union[RunConfig, Literal['auto'], None] = 'auto',
  ):
    """Create a new Hypothesis object.

    Args:
      name: The name of the hypothesis. Should be unique within an Experiment.
      runs: The underlying runs that this hypothesis consists of.
      style: (optional) A dict that represents preferred style for plotting.
        These will be passed as kwargs to plot().
      config: A config dict that describes the configuration of the hypothesis.
        A config is optional, where `config` is explicitly set to be `None`.
        If config exists (not None), it should represent the config this
        hypothesis primarily concerns: (a subset of) shared config of all the
        underlying runs, i.e. the hypothesis' config should be compatible with
        the runs. This constraint should remain at all times, including adding
        runs in-place after instantiation. By default (`config=auto`) it will
        try to automatically determine a config by extracting from the configs
        of all the underlying runs: the intersection of the configs (w.r.t.
        keys and values). See extract_config() for more details.
    """

    if isinstance(runs, Run) or isinstance(runs, pd.DataFrame):
      if not isinstance(runs, Run):
        runs = Run.of(runs)
      runs = [runs]  # type: ignore

    self.name = name
    self.runs = RunList(runs)

    if config == 'auto':
      has_configs = [r.config is not None for r in self.runs]
      if len(self.runs) > 0 and all(has_configs):  # All the runs have a config
        config = self.extract_config(self.runs)
      elif not any(has_configs):
        config = None  # all the runs have no config
      else:
        raise ValueError("Run configs should exist for either all the runs or "
                         "none of the runs. The name of runs that do not have "
                         "config: {}".format(
                             r.name for r in self.runs if r.config is None))

    self.config = config
    self.style = {**style} if style is not None else {}

  def __iter__(self) -> Iterator[Run]:
    return iter(self.runs)

  @classmethod
  def of(
      cls,
      runs: Run | Sequence[Run],
      *,
      name: Optional[str] = None,
      config: Union[RunConfig, Literal['auto'], None] = 'auto',
  ) -> Hypothesis:
    """A static factory method."""
    if isinstance(runs, Run):
      name = name or runs.path

    return cls(name=name or '', runs=runs, config=config)

  def rename(self, name: str) -> Hypothesis:
    return dataclasses.replace(self, name=name)

  # yapf: disable
  @overload
  def __getitem__(self, k: int) -> Run: ...
  @overload
  def __getitem__(self, k: str) -> pd.DataFrame: ...
  # yapf: enable

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

  @staticmethod
  def extract_config(runs: Sequence[Run]) -> RunConfig:
    """Extract a config dict from underyling runs. It defaults to the subset
    of configs that all the runs share and have the identical value."""
    config = None

    def _intersect(dst, ref):
      keys = dst.keys() & ref.keys()
      return {k: dst[k] for k in dst.keys() if k in keys and dst[k] == ref[k]}

    for r in runs:
      if r.config is None:
        raise RuntimeError(f"A run of name `{r.name}` does not have a config.")
      if config is None:
        config = dict(r.config)
      else:
        config = _intersect(config, r.config)

    if config is None:
      raise RuntimeError("This hypothesis contains no runs.")
    return config

  def _is_compatible(self, other: Hypothesis | Run):
    if self.config and other.config:
      config: RunConfig = self.config
      rhs: RunConfig = other.config
      for k in config:
        if k not in rhs or config[k] != rhs[k]:
          return False
    elif self.config and not other.config:
      return False  # config for rhs is mandatory.
    return True

  def summary(self, **kwargs) -> pd.DataFrame:
    """Return a DataFrame that summarizes the current hypothesis."""
    return Experiment(self.name, [self]).summary(**kwargs)

  if TYPE_CHECKING:  # Provide type hint and documentation for static checker.
    import expt.plot
    plot: expt.plot.HypothesisPlotter  # type: ignore
    hvplot: expt.plot.HypothesisHvPlotter  # type: ignore

  plot = util.PropertyAccessor(  # type: ignore
      "plot", lambda self: _import("expt.plot").HypothesisPlotter(self))
  hvplot = util.PropertyAccessor(  # type: ignore
      "hvplot", lambda self: _import("expt.plot").HypothesisHvPlotter(self))

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

  def mean(self, *args, numeric_only=True, **kwargs) -> pd.DataFrame:
    g = self.grouped
    return g.mean(*args, numeric_only=numeric_only, **kwargs)

  def std(self, *args, numeric_only=True, **kwargs) -> pd.DataFrame:
    g = self.grouped
    return g.std(*args, numeric_only=numeric_only, **kwargs)

  def min(self, *args, numeric_only=True, **kwargs) -> pd.DataFrame:
    g = self.grouped
    return g.min(*args, numeric_only=numeric_only, **kwargs)

  def max(self, *args, numeric_only=True, **kwargs) -> pd.DataFrame:
    g = self.grouped
    return g.max(*args, numeric_only=numeric_only, **kwargs)

  def resample(self,
               x_column: Optional[str] = None,
               *,
               n_samples: int) -> Hypothesis:
    """Resample (usually downsample) the data by uniform subsampling.

    This is also useful when the hypothesis' individual runs may have
    heterogeneous index or the x-axis (i.e., differnet supports): when
    (linear) interpolation is applied, the resulting Hypothesis will consist of
    runs with dataframes sharing the same x axis values.

    Args:
      x_column (Optional): the name of column to sample over. If `None`,
        sampling and interpolation will be done over the index of the dataframe.
      n_samples (int): The number of points to use when subsampling.
        Should be greater than 2.
      interpolate (bool): Defaults False. If True, linear interpolation will
        be applied

    Returns:
      A copy of Hypothesis with the same name, whose runs (DataFrames) consists
      of interpolated and resampled data for all the numeric columns. All
      other non-numeric columns will be dropped. Each DataFramew will have
      the specified x_column as its Index.
    """
    return self._resample(x_column, n_samples, interpolate=False)

  def interpolate(self,
                  x_column: Optional[str] = None,
                  *,
                  n_samples: int) -> Hypothesis:
    """DEPRECATED: Use resample(..., interpolate=True) instead."""
    return self._resample(x_column, n_samples=n_samples, interpolate=True)

  def _resample(self,
                x_column: Optional[str],
                n_samples: int,
                *,
                interpolate=False) -> Hypothesis:
    # For each dataframe (run), interpolate on the `x_column` or index
    if x_column is None:
      x_series = pd.concat([pd.Series(df.index) for df in self._dataframes])
      index_name = util.ensure_unique(
          [df.index.name for df in self._dataframes])
    else:
      if x_column not in self.columns:
        raise ValueError(f"Unknown column: {x_column}")
      x_series = pd.concat([df[x_column] for df in self._dataframes])
      index_name = x_column

    x_min = x_series.min()
    x_max = x_series.max()
    x_samples = np.linspace(x_min, x_max, num=n_samples)  # type: ignore

    # Get interpolated dataframes for numerical columns.
    def _process_df_interpolate(df: pd.DataFrame) -> pd.DataFrame:
      if x_column is not None:
        df = df.set_index(x_column)  # type: ignore

      # yapf: disable
      def _interpolate_if_numeric(y_series: pd.Series):
        if y_series.dtype.kind in ('i', 'f'):
          # y_series might contain NaN values, but interp1d does not
          # properly deal with nan. So we filter both of x and y series.
          idx_valid = ~np.isnan(y_series)
          if idx_valid.sum() >= 2:
            return scipy.interpolate.interp1d(df.index[idx_valid],
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

      # filter out non-numeric columns.
      df_interp = df.select_dtypes(['number'])
      df_interp = cast(pd.DataFrame, df_interp.apply(_interpolate_if_numeric))

      df_interp[index_name] = x_samples
      df_interp.set_index(index_name, inplace=True)
      return df_interp

    def _process_df_subsample(df: pd.DataFrame) -> pd.DataFrame:
      if n_samples <= df.shape[0]:
        return df
      idx = np.linspace(0, df.shape[0] - 1, n_samples, dtype=np.int32)
      return df.loc[idx]

    if interpolate:
      processed_dfs = [_process_df_interpolate(df) for df in self._dataframes]
    else:
      processed_dfs = [_process_df_subsample(df) for df in self._dataframes]

    assert len(processed_dfs) == len(self.runs)
    return Hypothesis(
        name=self.name,
        runs=[
            Run(r.path, df_new) for (r, df_new) in zip(self.runs, processed_dfs)
        ],
        config=copy.copy(self.config),
    )

  def apply(self, fn: Callable[[pd.DataFrame], pd.DataFrame]) -> Hypothesis:
    """Apply a transformation on all underlying DataFrames.

    This returns a copy of Hypothesis and children Run objects.
    """
    return Hypothesis(
        name=self.name,
        runs=[Run(r.path, fn(r.df)) for r in self.runs],
        config=copy.copy(self.config),
    )


class Experiment(Iterable[Hypothesis]):
  """An Experiment is a collection of Hypotheses with config structure."""

  @typechecked
  def __init__(
      self,
      name: Optional[str] = None,
      hypotheses: Sequence[Hypothesis] = (),  # TODO Support pd.DataFrame.
      *,
      config_keys: Optional[Sequence[str]] = None,
      summary_columns: Optional[Sequence[str]] = None,  # TODO test this
  ):
    """Create a new Experiment object.

    Args:
      name: The name.
      hypotheses: A collection of hypotheses to initialize with.
    """
    self._name = name if name is not None else ""
    self._hypotheses: Dict[str, Hypothesis] = collections.OrderedDict()

    if isinstance(config_keys, str):
      config_keys = [config_keys]
    self._config_keys = list(config_keys) if config_keys is not None else []

    if isinstance(summary_columns, str):
      summary_columns = [summary_columns]
    self._summary_columns = tuple(summary_columns) \
      if summary_columns is not None else None

    # The internal pd.DataFrame representation that backs Experiment.
    # index   = [*config_keys, name: str]  (a MultiIndex)
    # columns = [hypothesis: Hypothesis, *summary_keys]

    if isinstance(hypotheses, np.ndarray):
      hypotheses = list(hypotheses)

    for h in hypotheses:
      self.add_hypothesis(h, extend_if_conflict=False)

  def _replace(self, **kwargs) -> Experiment:
    ex = Experiment(
        name=kwargs.pop('name', self._name),
        hypotheses=list(self._hypotheses.values()),
        config_keys=kwargs.pop('config_keys', self._config_keys),
        summary_columns=kwargs.pop('summary_columns', self._summary_columns),
    )
    if kwargs:
      raise ValueError("Unknown fields: {}".format(list(kwargs.keys())))
    return ex

  @property
  def _df(self) -> pd.DataFrame:
    hypotheses: List[Hypothesis] = list(self._hypotheses.values())

    df = pd.DataFrame({
        'name': [h.name for h in hypotheses],
        'hypothesis': hypotheses,
        **{  # config keys (will be index)
            k: [(h.config or {}).get(k) for h in hypotheses]
            for k in self._config_keys
        },
    })

    def _append_summary(summary_columns):
      nonlocal df
      df = pd.concat([
          df,
          pd.DataFrame({
              k: [
                  # TODO: h.summary is expensive and slow, cache it
                  h.summary(columns=summary_columns).loc[0, k]
                  for h in self._hypotheses.values()
              ] for k in summary_columns
          }),
      ], axis=1)  # yapf: disable

    if self._summary_columns is not None:
      _append_summary(summary_columns=self._summary_columns)
    else:
      _append_summary(summary_columns=self.columns)

    # Need to sort index w.r.t the multi-index level hierarchy, because
    # the order of hypotheses being added is not guaranteed
    df = df.set_index([*self._config_keys, 'name']).sort_index()
    return df

  @classmethod
  def from_runs(
      cls,
      runs: RunList,
      *,
      config_fn: Callable[[Run], RunConfig] = _default_config_fn,
      config_keys: Optional[Sequence[str]] = None,
      summary_columns: Optional[Sequence[str]] = None,
      name: Optional[str] = None,
  ) -> Experiment:
    """Construct a new Experiment object directly from a RunList."""

    df = runs.to_dataframe(
        include_config=True,
        config_fn=config_fn,
        index_keys=config_keys,
        as_hypothesis=True,
        include_summary=True,
        hypothesis_namer=None,  # TODO use more general hypothesis_factory.
    )

    if summary_columns:
      try:
        df = df[['hypothesis', *summary_columns]]
      except KeyError:
        # summary_columns often have many mistakes or missing keys,
        # so let the error message more informative.
        missing_cols = [col for col in summary_columns if col not in df.keys()]
        suggestions = [
            difflib.get_close_matches(col, df.keys()) for col in missing_cols
        ]
        linesep = '\n'
        raise KeyError(
            f"Some columns do not exist in the dataframe: {missing_cols}. "
            f"Close matches = {linesep.join(str(s) for s in suggestions)}"
        ) from None

    return cls.from_dataframe(cast(pd.DataFrame, df), name=name)

  @classmethod
  def from_dataframe(
      cls,
      df: pd.DataFrame,
      by: None | str | Sequence[str] = None,
      *,
      run_column: str = 'run',
      hypothesis_namer: Optional[  # (run_config, runs) -> str
          Callable[[RunConfig, Sequence[Run]], str]] = None,
      name: Optional[str] = None,
  ) -> Experiment:
    """Constructs a new Experiment object from a DataFrame instance,
    that is structured as per the convention. The DataFrame objects are
    usually constructed from `RunList.to_dataframe()`.

    Args:
      df: The dataframe to create an Experiment from. This should contain
        a column `run` of `Run` objects, or a column `hypothesis` of
        `Hypothesis` objects.
      by (str, List[str]): The column name to group by, when the dataframe
        consists of Runs (rather than Hypotheses). If None (default),
        it will try to automatically determine from the dataframe if there
        is only one column other than `run_column`.
      run_column (str): The column name that contains `Run` objects.
        See also `RunList.to_dataframe()`.
      hypothesis_namer: This is a mapping that transforms the group key
        (a dict) into hypothesis name. The function should take two arguments,
        (group_key as dict, list of runs).
      name: The name for the produced `Experiment`.
    """
    if by is None:
      # Automatically determine the column from df.
      by_columns = list(sorted(set(df.columns).difference([run_column])))
      if 'hypothesis' in by_columns:
        # TODO test this behavior
        by = 'hypothesis'
      elif len(by_columns) == 1:
        by = next(iter(by_columns))
      else:
        raise ValueError("Cannot automatically determine the column to "
                         "group by. Candidates: {}".format(by_columns))

    if isinstance(df.index, pd.MultiIndex):
      config_keys = tuple(k for k in df.index.names if k != 'name')
    elif by not in ('hypothesis', 'run'):
      config_keys = by
    else:
      config_keys = None

    def _aslist(x: None | str | Sequence[str]) -> List[str]:
      if x is None:
        return []
      if isinstance(x, str):
        return [x]
      return list(x)

    summary_columns = list(sorted(
        set(df.columns).difference([run_column, 'hypothesis', *_aslist(by)])
    ))  # yapf: disable
    ex = Experiment(
        name=name,
        config_keys=config_keys,
        summary_columns=summary_columns if summary_columns else None)

    # Special case: already grouped (see RunList:to_dataframe(as_hypothesis=True))
    # TODO: This groupby feature needs to be incorporated by to_dataframe().
    if by == 'hypothesis':
      hypotheses = list(df['hypothesis'])
      if not hypotheses:
        raise ValueError("The dataframe contains no Hypotheses, seems empty.")
      if not isinstance(hypotheses[0], Hypothesis):
        raise ValueError("The column 'hypothesis' does not contain "
                         "a Hypothesis object.")
      for h in hypotheses:
        # TODO test this behavior on h_namer
        ex.add_hypothesis(h)

    else:
      if hypothesis_namer is None:
        hypothesis_namer = lambda keys, _: str(keys)

        if run_column not in df.columns:
          raise ValueError(
              f"The dataframe does not have a column `{run_column}`.")

      # A column made of Runs...
      for hypothesis_key, runs_df in df.groupby(by):
        if isinstance(hypothesis_key, tuple):
          hypothesis_key = dict(zip(cast(List[str], by), hypothesis_key))
        runs = RunList(runs_df[run_column])
        hypothesis_name = hypothesis_namer(hypothesis_key, runs)
        h = runs.to_hypothesis(name=hypothesis_name)
        ex.add_hypothesis(h)

    return ex

  def add_runs(
      self,
      hypothesis_name: str,
      runs: RunList | List[Run | Tuple[str, pd.DataFrame] | pd.DataFrame],
  ) -> Hypothesis:
    util.warn_deprecated(
        "add_runs() is deprecated. Use add_hypothesis() instead.")

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
      *,
      extend_if_conflict=False,
  ) -> Hypothesis:

    if h.name in self._hypotheses:
      if not extend_if_conflict:
        raise ValueError(f"Hypothesis named {h.name} already exists!")

      util.warn_deprecated(
          "add_hypothesis(extend_if_conflict=True) is deprecated, because "
          "it can result in an inconsistent state with config. "
          "This argument will be removed in a future version.")

      # Validate if the config (hierarchical index) doesn't match.
      # this assumes the index of hierarhical index matches hypothesis config.
      d: Hypothesis = self._hypotheses[h.name]
      for r in h.runs:
        if not d._is_compatible(r):
          raise ValueError(
              f"Run {r.name} is not compatible with the existing config.")

      d.runs.extend(h.runs)
    else:
      self._hypotheses[h.name] = h  # add into the collection.

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
  ) -> Hypothesis | Sequence[Hypothesis]:
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

  def select(self, expr: str | Callable[[Hypothesis], bool]) -> Experiment:
    """Select a subset of Hypothesis matching the given criteria."""

    if isinstance(expr, str):
      df = self._df.query(expr)
      name = self.name + "[" + expr + "]"

    elif callable(expr):  # Hypothesis -> bool
      df = self._df
      mask = df['hypothesis'].apply(expr)
      if mask.dtype != bool:
        raise TypeError("The filter function must return bool, but unexpected "
                        "data type found: {}".format(mask.dtype))
      df = df[mask]
      name = self.name  # TODO: Customize? repr(fn)?

    else:
      raise TypeError(  # ...
          "`expr` must be a str or Callable, but given {}".format(type(expr)))

    return Experiment.from_dataframe(df, name=name)

  def __iter__(self) -> Iterator[Hypothesis]:
    return iter(self._hypotheses.values())

  def __repr__(self) -> str:
    return (
        f"Experiment('{self.name}', {len(self._hypotheses)} hypotheses: [\n " +
        ',\n '.join([h.__repr__(indent=' ') for h in self.hypotheses]) +
        ",\n])")

  def _repr_html_(self, include_hypothesis=False, include_name=True):

    # TODO: more fine-grained control of style.
    df = self._df
    hypotheses = df['hypothesis']

    if not include_hypothesis:
      df = df.drop(columns=['hypothesis'], errors='ignore')
    if not include_name:
      df = df.drop(columns=['name'], errors='ignore')

    # TODO: the lower the better in some cases.
    df_styler = df.style.background_gradient().set_table_styles([
        {
            "selector": "td, th",
            "props": list({
                "border": "1px solid grey !important",
                "word-break": "break-word",
            }.items()),
        },
        {
            "selector": "td.data, th.row_heading",
            "props": list({
                "word-break": "keep-all",
            }.items()),
        },
    ])  # yapf: disable

    if not df_styler:
      return None

    # Add tooltip for individual run information (path)
    ttips = pd.DataFrame(index=df.index, columns=df.columns)
    for i, h in enumerate(hypotheses):
      h = cast(Hypothesis, h)
      h_summary: pd.DataFrame = h.summary(individual_runs=True)
      for j, column in enumerate(df.columns):
        if column in h_summary:
          ttips.iloc[i, j] = r'\A'.join(
              f' {r_val:.6g} ::: {r.path}' \
              for r, r_val in zip(h.runs, h_summary[column])
          )

    df_styler = df_styler.set_tooltips(ttips)

    return ''.join([
        '<div>',
        '''<style scoped>
          .experiment-name { font-weight: bold; font-size: 14pt; }
          .pd-t { background-color: #ffe066 !important; color: black !important; padding: 2px; white-space: pre; }
        </style>''',
        f'<div class="experiment-name">{self.name}</div>',
        df_styler.to_html(),
        '</div>',
    ])

  # yapf: disable
  @overload
  def __getitem__(self, key: str) -> Hypothesis: ...
  @overload
  def __getitem__(self, key: int) -> Hypothesis: ...
  @overload
  def __getitem__(self, key: Tuple | List | np.ndarray) -> np.ndarray: ...
  @overload
  def __getitem__(self, key: Tuple) -> pd.DataFrame: ...
  # yapf: enable

  def __getitem__(
      self,
      key: int | str | Tuple | List | np.ndarray,
  ) -> Hypothesis | np.ndarray | Run | pd.DataFrame:
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

  @property
  def columns(self) -> Sequence[str]:
    # merge and uniquify all columns but preserving the order.
    return util.merge_list(*[h.columns for h in self._hypotheses.values()])

  @staticmethod
  def AGGREGATE_MEAN_LAST(portion: float):
    return (lambda series: series.rolling(max(1, int(len(series) * portion))
                                          ).mean().iloc[-1])  # yapf: disable

  def summary(
      self,
      *,
      name=True,
      individual_runs: bool = False,
      columns: Sequence[str] | None = None,
      aggregate=None,
  ) -> pd.DataFrame:
    """Return a DataFrame that summarizes the current experiments,
    whose rows are all hypothesis.

    Args:
      name: If True, include the column 'name'. Otherwise include 'hypothesis'.
      individual_runs: If True, show the summary of metrics for each of the
        individual runs. If False (default), show the summary of aggregated,
        averaged hypothesis.
      columns: The list of columns to show. Defaults to `self.columns` plus
        `"index"` (or `df.index.name` if exists).
      aggregate: A function or a dict of functions ({column_name: ...})
        specifying a strategy to aggregate a `Series`. Defaults to take the
        average of the last 10% of the series.

    Example Usage:

      >>> pd.set_option('display.max_colwidth', 2000)   # deal with long names?
      >>> df = ex.summary(columns=['index', 'loss', 'return'])
      >>> df.style.background_gradient(cmap='viridis')

    """

    def _entries():
      if individual_runs:
        for h in self.hypotheses:
          yield from h.runs
      else:
        yield from self.hypotheses

    entries: List[Run | Hypothesis] = list(_entries())

    if name:
      df = pd.DataFrame({'name': [h.name for h in entries]})
    else:
      col = 'runs' if individual_runs else 'hypothesis'
      df = pd.DataFrame({'hypothesis': entries})

    def _mean(h) -> pd.DataFrame:
      if isinstance(h, Hypothesis):
        return h.mean()
      else:
        return h.df

    rows: List[pd.DataFrame] = [
        (_mean(h) if not all(len(df) == 0 for df in h._dataframes) \
                  else pd.DataFrame())
        for h in entries
    ]

    index_name: str = (
        util.ensure_unique({cast(str, h.index.name) for h in rows})
        or 'index')  # yapf: disable  # noqa: W503

    if columns is None:
      if index_name not in self.columns:
        columns = [index_name] + list(self.columns)
      else:
        columns = list(self.columns)
    else:
      columns = list(columns)

    aggregate = aggregate or self.AGGREGATE_MEAN_LAST(0.1)

    def make_summary_series(column: str) -> pd.Series:

      def df_series(df: pd.DataFrame):
        # TODO: What if a named column is used as an index?
        if column == index_name:
          return df.index
        if column not in df:
          return []
        else:
          return df[column].dropna()

      def aggregate_h(series):
        if len(series) == 0:
          # after dropna, no numeric types to aggregate?
          return np.nan
        if series.dtype.kind == 'O':  # object, not a numeric type
          return np.nan
        aggregate_fn = aggregate
        if not callable(aggregate_fn):
          aggregate_fn = aggregate[column]  # type: ignore
        v = aggregate_fn(series) if column != index_name else series.max()
        return v

      return pd.Series(
          name=column,
          data=[aggregate_h(df_series(hm)) for hm in rows],
      )

    df = pd.concat(
        [df] +  # ... index and hypothesis
        [make_summary_series(column) for column in columns],
        axis=1)

    assert len(df.columns) == len(set(df.columns.values)), (
        f"The columns of summary DataFrame must be unique. Found: {df.columns}")
    return df

  @typechecked
  def with_config_keys(
      self,
      new_config_keys: Sequence[Union[str, EllipsisType]],  # type: ignore
  ) -> Experiment:
    """Create a new Experiment with the same set of Hypotheses, but a different
    config keys in the multi-index (usually reordering).

    Note that the underlying hypothesis objects in the new Experiment object
    won't change, e.g., their name, config, etc. would remain the same.

    Args:
      new_config_keys: The new list of config keys. This can contain `...`
        (Ellipsis) as the last element, which refers to all the other keys
        in the current Experiment that was not included in the list.
    """

    if new_config_keys[-1] is ...:
      keys_requested = [x for x in new_config_keys if x is not ...]
      keys_appended = [x for x in self._config_keys if x not in keys_requested]
      new_config_keys = keys_requested + keys_appended

    for key in new_config_keys:
      if not isinstance(key, str):
        raise TypeError(f"Invalid config key: {type(key)}")
      for h in self._hypotheses.values():
        if h.config is None:
          raise ValueError(f"{h} does not have a config.")
        if key not in h.config.keys():
          raise ValueError(f"'{key}' not found in the config of {h}. "
                           "Close matches: " +
                           str(difflib.get_close_matches(key, h.config.keys())))

    return self._replace(config_keys=new_config_keys)

  def resample(self, *, n_samples: int) -> Experiment:
    """Resample data uniformly (equidistantly) for each of the hypotheses,
    and return a copy of new Experiment objet.

    See: Hypothesis.resample().
    """
    return Experiment(
        name=self.name,
        hypotheses=[h.resample(n_samples=n_samples) for h in self.hypotheses],
        config_keys=self._config_keys,
        summary_columns=self._summary_columns,
    )

  def interpolate(self,
                  x_column: Optional[str] = None,
                  *,
                  n_samples: int) -> Experiment:
    """Apply resampling and interpolation to each of the hypothesis,
    and return a copy of new Experiment (with its children Hypothesis/Run)
    object.

    See: Hypothesis.interpolate().
    """
    return Experiment(
        name=self.name,
        hypotheses=[
            h.interpolate(x_column, n_samples=n_samples)
            for h in self.hypotheses
        ],
        config_keys=self._config_keys,
        summary_columns=self._summary_columns,
    )

  def apply(self, fn: Callable[[pd.DataFrame], pd.DataFrame]) -> Experiment:
    """Apply a transformation on all underlying DataFrames.

    This returns a copy of Experiment and children Hypothesis objects.
    """
    return Experiment(
        name=self.name,
        hypotheses=[h.apply(fn) for h in self.hypotheses],
        config_keys=self._config_keys,
        summary_columns=self._summary_columns,
    )

  def hvplot(self, *args, **kwargs):
    plot = None
    for i, (name, hypo) in enumerate(self._hypotheses.items()):
      p = hypo.hvplot(*args, label=name, **kwargs)
      plot = (plot * p) if plot else p  # type: ignore
    return plot

  if TYPE_CHECKING:  # Provide type hint and documentation for static checker.
    import expt.plot
    plot: expt.plot.ExperimentPlotter

  plot = util.PropertyAccessor(  # type: ignore
      "plot", lambda self: _import("expt.plot").ExperimentPlotter(self))


__all__ = (
    'Run',
    'RunList',
    'Hypothesis',
    'Experiment',
)
