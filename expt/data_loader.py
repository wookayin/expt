"""Data Loader for expt."""

import abc
import multiprocessing.pool
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import multiprocess.pool
import numpy as np
import pandas as pd

from . import path_util, util
from .data import Experiment, Hypothesis, Run, RunList

try:
  from tqdm.auto import tqdm
except ImportError:
  tqdm = util.NoopTqdm

#########################################################################
# Individual Run Parsers
#########################################################################


class LogParser:
  """Interface for log parsing."""

  def __init__(self, log_dir):
    self._log_dir = log_dir

  @abc.abstractmethod
  def read(self, verbose=False) -> 'LogParser':
    """Load or reload the data from the directory."""
    return self

  @abc.abstractmethod
  def parse(self) -> pd.DataFrame:
    """Return the log data read as a DataFrame."""

  def __repr__(self):
    return f"<{type(self).__name__}, log_dir={self._log_dir}>"


def parse_run(log_dir, fillna=False, verbose=False) -> pd.DataFrame:
  """Create a pd.DataFrame object from a single directory."""
  if verbose:
    # TODO Use python logging
    print(f"Reading {log_dir} ...", file=sys.stderr, flush=True)

  # make it more general (rather than being specific to progress.csv)
  # and support tensorboard eventlog files, etc.
  sources = [
      parse_run_progresscsv,
      parse_run_tensorboard,
  ]

  for fn in sources:
    try:
      df = fn(log_dir, fillna=fillna, verbose=verbose)
      if df is not None:
        break
    except (FileNotFoundError, IOError) as e:
      if verbose:
        print(f"{fn.__name__} -> {e}\n", file=sys.stderr, flush=True)
  else:
    raise pd.errors.EmptyDataError(  # type: ignore
        f"Cannot handle dir: {log_dir}")

  # add some optional metadata... (might not be preserved afterwards)
  if df is not None:
    df.path = log_dir
  return df


class CSVLogParser(LogParser):
  """Parse log data from progress.csv per convention."""

  def __init__(self, log_dir):
    super().__init__(log_dir=log_dir)
    self._df = None

    # Find the target CSV file.
    # Try progress.csv or log.csv from folder
    detected_csv = None
    for fname in ('progress.csv', 'log.csv'):
      p = os.path.join(self._log_dir, fname)
      if path_util.exists(p):
        detected_csv = p
        break

    # maybe a direct file path is given instead of directory
    if detected_csv is None:
      f = self._log_dir
      if path_util.exists(f) and not path_util.isdir(f):
        detected_csv = f

    if detected_csv is None:
      raise FileNotFoundError(os.path.join(self._log_dir, "*.csv"))

    self._csv_path = detected_csv

  def read(self, verbose=False):
    # Read the detected file `p`
    if verbose:
      print(f"parse_run (csv): Reading {self._csv_path}",
            file=sys.stderr, flush=True)  # yapf: disable

    df: pd.DataFrame
    with open(self._csv_path, mode='r', encoding='utf-8') as f:
      df = pd.read_csv(f)  # type: ignore

    self._df = df
    return self

  def parse(self, fillna=False) -> pd.DataFrame:
    df = self._df
    if df is None:
      raise pd.errors.EmptyDataError(  # type: ignore
          f"No rows found in {self._csv_path}")
    if fillna:
      df = df.fillna(0)
    return df


class TensorboardLogParser(LogParser):
  """Log parser for tensorboard run directory."""

  def __init__(self, log_dir):
    super().__init__(log_dir=log_dir)

    # Initialize the resources.
    event_glob = os.path.join(log_dir, '*events.out.tfevents.*')

    # TODO: When a new event file is added?
    self._event_files = list(sorted(path_util.glob(event_glob)))
    if not self._event_files:  # no event file detected
      raise FileNotFoundError(f"No event file detected in {self._log_dir}")

    # Initialize the internal data structure.
    # int(timestep) -> dict: {columns -> ...}
    self._all_data: Dict[int, Dict[str, Any]] = defaultdict(dict)
    self.read()

  def _extract_scalar_from_proto(self, value, step):
    from tensorflow.python.framework.dtypes import DType

    def _read_proto(node, path: str):
      for p in path.split('.'):
        node = getattr(node, p, None)
        if node is None:
          return None
      return node

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

  def _iter_scalar_summary_from_event_file(self, event_file):
    from tensorflow.core.util import event_pb2

    try:
      # avoid DeprecationWarning on tf_record_iterator
      # pyright: reportMissingImports=false
      from tensorflow.python._pywrap_record_io import RecordIterator

      def summary_iterator(path):
        for r in RecordIterator(path, ""):
          yield event_pb2.Event.FromString(r)  # type: ignore
    except Exception:
      from tensorflow.python.summary.summary_iterator import summary_iterator

    for event in summary_iterator(event_file):
      step = int(event.step)
      if not event.HasField('summary'):
        continue
      for value in event.summary.value:
        yield from self._extract_scalar_from_proto(value, step=step)

  def read(self, verbose=False) -> 'TensorboardLogParser':
    for event_file in self._event_files:
      if verbose:
        print(f"parse_run (tfevents) : Reading {event_file} ...",
              file=sys.stderr, flush=True)  # yapf: disable

      for step, tag_name, value in \
          self._iter_scalar_summary_from_event_file(event_file):
        self._all_data[step][tag_name] = value

    for t in list(self._all_data.keys()):
      self._all_data[t]['global_step'] = t

    return self

  def parse(self) -> pd.DataFrame:
    df = pd.DataFrame(self._all_data).T

    # Reorder column names in a lexicographical order
    df = df.reindex(sorted(df.columns), axis=1)
    return df


def parse_run_progresscsv(log_dir, fillna=False, verbose=False) -> pd.DataFrame:
  return CSVLogParser(log_dir).read(verbose=verbose).parse(fillna=fillna)


def parse_run_tensorboard(log_dir, fillna=False, verbose=False) -> pd.DataFrame:
  """Create a pandas DataFrame from tensorboard eventfile or run directory."""
  return TensorboardLogParser(log_dir).read(verbose=verbose).parse()


def _get_parser_for(log_dir):
  for parser_cls in (
      CSVLogParser,
      TensorboardLogParser,
  ):
    try:
      parser = parser_cls(log_dir)
      return parser
    except (FileNotFoundError, IOError):
      # When log_dir is not supported by the parser,
      # an expected exception is thrown. Try the next one.
      pass

  # TODO: Use some appropriate exception type.
  raise FileNotFoundError(f"Cannot read {log_dir} using known log parsers.")


#########################################################################
# Run Loader Functions
#########################################################################


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

    paths = list(sorted(path_util.glob(path_glob)))
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
    pool_class=multiprocess.pool.Pool,
    progress_bar=True,
    run_postprocess_fn=None,
) -> RunList:
  """Get a list of Run objects from the given path glob patterns.

  This runs in parallel, using the multiprocess library (a fork of python
  standard library multiprocessing) which is more friendly with ipython
  and serializing non-picklable objects.
  """

  if isinstance(pool_class, str):
    Pool = {
        'threading': multiprocess.pool.ThreadPool,
        'multiprocess': multiprocess.pool.Pool,  # third-party (works better)
        'multiprocessing': multiprocessing.pool.Pool,  # python stdlib
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
      paths = list(sorted(path_util.glob(path_glob)))
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

#########################################################################
# Run Loader Objects
#########################################################################


class RunLoader:
  """A manager that supports parallel and incremental loading of runs."""

  def __init__(self, run_dirs):
    pass

  def reload(self):
    pass

  def get_runs(self) -> RunList:
    pass

  def close(self):
    pass


__all__ = (
    'get_runs',
    'get_runs_serial',
    'get_runs_parallel',
    'iter_runs_serial',
    'parse_run',
)
