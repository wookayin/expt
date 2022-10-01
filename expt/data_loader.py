"""Data Loader for expt."""

import abc
import asyncio
import atexit
import dataclasses
import functools
import itertools
import multiprocessing.pool
import os
import sys
import threading
import warnings
from collections import Counter, defaultdict, namedtuple
from pathlib import Path
from typing import (Any, Callable, Dict, Iterator, List, Mapping, Optional,
                    Tuple, TypeVar, Union)

import multiprocess.pool
import numpy as np
import pandas as pd

from expt import path_util, util
from expt.data import Experiment, Hypothesis, Run, RunList

try:
  from tqdm.auto import tqdm
except ImportError:
  tqdm = util.NoopTqdm

#########################################################################
# Individual Run Reader
#########################################################################

LogReaderContext = TypeVar('LogReaderContext')


class LogReader:
  """Interface for reading logs.

  LogReaders should maintain its internal state stored in a context object,
  so that it can work even in a forked multiprocess worker.
  """

  def __init__(self, log_dir):
    self._log_dir = log_dir

  @property
  def log_dir(self) -> str:
    return self._log_dir

  @abc.abstractmethod
  def new_context(self) -> LogReaderContext:  # type: ignore
    """Create a new, empty context. This context can be used to store
    all the internal states or data for reading logs. Or it can be a Process
    object if you want a daemon reader process running in the background.
    The context object will be serialized for multiprocess IPC, when being
    passed to a worker process.

    IMPORTANT: Serialization for multiprocessing is often very slow,
    outweighing the benefit of parallelization. To avoid expensive memory copy,
    be sure to use shared memory or serialization-free data structure."""
    return dict()

  @abc.abstractmethod
  def read(self, context: LogReaderContext, verbose=False) -> LogReaderContext:
    """Load the remaining portion (or all) of the data."""
    return context

  @abc.abstractmethod
  def result(self, context: LogReaderContext) -> pd.DataFrame:
    """Return the log data read as a DataFrame."""
    del context
    raise NotImplementedError

  def __repr__(self):
    return f"<{type(self).__name__}, log_dir={self.log_dir}>"


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


class CSVLogReader(LogReader):
  """Parse log data from progress.csv per convention."""

  def __init__(self, log_dir):
    super().__init__(log_dir=log_dir)

    # Find the target CSV file.
    # Try progress.csv or log.csv from folder
    detected_csv = None
    for fname in ('progress.csv', 'log.csv'):
      p = os.path.join(self.log_dir, fname)
      if path_util.exists(p):
        detected_csv = p
        break

    # maybe a direct file path is given instead of directory
    if detected_csv is None:
      f = self.log_dir
      if path_util.exists(f) and not path_util.isdir(f):
        detected_csv = f

    if detected_csv is None:
      raise FileNotFoundError(os.path.join(self.log_dir, "*.csv"))

    self._csv_path = detected_csv

  def read(self, context, verbose=False):
    del context  # unused, this implementation always read the data in full.

    # Read the detected file `p`
    if verbose:
      print(f"parse_run (csv): Reading {self._csv_path}",
            file=sys.stderr, flush=True)  # yapf: disable

    df: pd.DataFrame
    with open(self._csv_path, mode='r', encoding='utf-8') as f:
      df = pd.read_csv(f)  # type: ignore

    return df

  def result(self, context, fillna=False) -> pd.DataFrame:
    df = context
    assert isinstance(df, pd.DataFrame)
    if fillna:
      df = df.fillna(0)
    return df


class TensorboardLogReader(LogReader):
  """Log reader for tensorboard run directory."""

  def __init__(self, log_dir):
    super().__init__(log_dir=log_dir)

    # Initialize the resources.
    event_glob = os.path.join(log_dir, '*events.out.tfevents.*')

    # TODO: When a new event file is added?
    self._event_files = list(sorted(path_util.glob(event_glob)))
    if not self._event_files:  # no event file detected
      raise FileNotFoundError(f"No event file detected in {self.log_dir}")

    # Initialize tensorflow earlier otherwise the forked processes will
    # need to do it again.
    import tensorflow as tf  # type: ignore

  @dataclasses.dataclass
  class Context:  # LogReaderContext
    rows_read: Counter = dataclasses.field(default_factory=Counter)
    data: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)
    last_read_rows: int = 0

  def new_context(self) -> 'Context':
    return self.Context()

  # Helper function
  def _extract_scalar_from_proto(self, value, step):

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
          from tensorflow.python.framework.dtypes import DType
          dtype = DType(t.dtype).as_numpy_dtype
          simple_value = np.frombuffer(t.tensor_content, dtype=dtype)[0]
          yield step, value.tag, simple_value

  def _iter_scalar_summary_from(self, event_file, *,
                                skip=0, limit=None,  # per event_file
                                rows_callback):  # yapf: disable
    from tensorflow.core.util.event_pb2 import Event
    from tensorflow.python.lib.io import tf_record

    def summary_iterator(path, skip=0):
      from tensorflow.python.util import deprecation as tf_deprecation
      with tf_deprecation.silence():  # pylint: disable=not-context-manager
        # compatible with TF 1.x, 2.x (although deprecated)
        eventfile_tfrecord = tf_record.tf_record_iterator(path)
        eventfile_tfrecord = itertools.islice(
            eventfile_tfrecord,
            skip,
            skip + limit if limit is not None else None,
        )
      for serialized_pb in eventfile_tfrecord:
        yield Event.FromString(serialized_pb)  # type: ignore

    rows_read = 0

    for event in summary_iterator(event_file, skip=skip):
      rows_read += 1
      step = int(event.step)
      if not event.HasField('summary'):
        continue
      for value in event.summary.value:
        yield from self._extract_scalar_from_proto(value, step=step)

    rows_callback(rows_read)

  def read(self, context: 'Context', verbose=False):
    context.last_read_rows = 0

    chunk = defaultdict(dict)  # tag_name -> step -> value
    for event_file in self._event_files:
      if verbose:
        print(f"parse_run (tfevents) : Reading {event_file} ...",
              file=sys.stderr, flush=True)  # yapf: disable

      def _callback(rows_read: int):
        context.rows_read.update({event_file: rows_read})
        context.last_read_rows += rows_read

      for global_step, tag_name, value in \
          self._iter_scalar_summary_from(
              event_file, skip=context.rows_read[event_file],
              rows_callback=_callback,
          ):  # noqa: E125
        # Make sure global_step is always stored as integer.
        global_step = int(global_step)
        chunk[tag_name][global_step] = value

    df_chunk = pd.DataFrame(chunk)
    df_chunk['global_step'] = df_chunk.index.astype(int)

    # Merge the previous dataframe and the new one that was read.
    # The current chunk will overwrite any existing previous row.
    df = df_chunk.combine_first(context.data)
    context.data = df

    return context

  def result(self, context) -> pd.DataFrame:
    # Reorder column names in a lexicographical order
    df = context.data
    df = df.reindex(sorted(df.columns), axis=1)
    return df


def parse_run_progresscsv(log_dir, fillna=False, verbose=False) -> pd.DataFrame:
  parser = CSVLogReader(log_dir)
  ctx = parser.read(parser.new_context(), verbose=verbose)
  return parser.result(ctx, fillna=fillna)


def parse_run_tensorboard(log_dir, fillna=False, verbose=False) -> pd.DataFrame:
  del fillna  # unused
  parser = TensorboardLogReader(log_dir)
  ctx = parser.read(parser.new_context(), verbose=verbose)
  return parser.result(ctx)


def _get_reader_for(log_dir):
  for reader_cls in (
      CSVLogReader,
      TensorboardLogReader,
  ):
    try:
      reader = reader_cls(log_dir)
      return reader
    except (FileNotFoundError, IOError):
      # When log_dir is not supported by the reader,
      # an expected exception is thrown. Try the next one.
      pass

  # TODO: Use some appropriate exception type.
  raise FileNotFoundError(f"Cannot read {log_dir} using known log readers.")


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


# TODO: Deprecate in favor of RunLoader.
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

  def __init__(
      self,
      *path_globs,
      verbose: bool = False,
      progress_bar: bool = True,
      run_postprocess_fn: Optional[Callable[[Run], Run]] = None,
      n_jobs: int = 8,
  ):
    self._readers = []
    self._reader_contexts = []

    self._verbose = verbose
    self._progress_bar = progress_bar
    self._run_postprocess_fn = run_postprocess_fn

    self.add_paths(*path_globs)

    # Initialize multiprocess pool.
    if n_jobs > 1:
      self._pool = multiprocess.pool.Pool(processes=n_jobs)
    else:
      self._pool = None

    atexit.register(self.close)

  def close(self):
    if self._pool:
      self._pool.close()
      self._pool = None

  def add_paths(self, *path_globs):
    for path_glob in path_globs:

      if isinstance(path_glob, (list, tuple)):
        self.add_paths(*path_glob)
        continue

      path_glob: str
      paths = list(sorted(path_util.glob(path_glob)))
      if self._verbose and not paths:
        print(f"Warning: a glob pattern '{path_glob}' "
              "did not match any files.", file=sys.stderr)  # yapf: disable

      # TODO: Creating LogReader/context for each path can be expensive
      # and therefore still needs to be parallelized (as in get_runs_parallel)
      for log_dir in paths:
        # TODO: When any one of them fails or not ready? ignore, or raise?
        self.add_log_dir(log_dir)

  def add_log_dir(self, log_dir: Union[Path, str]) -> LogReader:
    reader = _get_reader_for(log_dir)
    self.add_reader(reader)
    return reader

  def add_reader(self, reader: LogReader):
    self._readers.append(reader)
    self._reader_contexts.append(reader.new_context())

  @staticmethod
  def _worker_handler(
      reader: LogReader,
      context: LogReaderContext,
      run_postprocess_fn: Optional[Callable[[Run], Run]] = None,
  ) -> Tuple[Optional[Run], LogReaderContext]:
    """The job function to be executed in a "forked" worker process."""
    try:
      context = reader.read(context)
      df = reader.result(context)
      run = Run(path=reader.log_dir, df=df)
      if run_postprocess_fn:
        run = run_postprocess_fn(run)
        _validate_run_postprocess(run)
      return run, context
    except (pd.errors.EmptyDataError, FileNotFoundError) as e:  # type: ignore
      # Ignore empty data.
      print(f"[!] {reader.log_dir} : {e}", file=sys.stderr, flush=True)
      return None, context

  def get_runs(self, parallel=True, *, tqdm_bar=None) -> RunList:
    """Refresh and get all the runs from all the log directories.

    This will work as incremental reading: the portion of data that was read
    from the last get_runs() call will be cached.
    """
    if not self._readers:
      return RunList([])  # special case, no matches

    # TODO: What if there is a new directory that matches the glob pattern
    # but created *after* initialization of log readers? How to add them?

    # Reload the data (incrementally or read from scratch) and return runs.
    if self._pool is None or not parallel:
      return self._get_runs_serial()

    else:
      pool = self._pool
      if self._progress_bar:
        pbar = tqdm(total=len(self._readers)) if tqdm_bar is None else tqdm_bar
      else:
        pbar = util.NoopTqdm()

      def _pbar_callback_done(run):
        del run  # unused
        pbar.update(1)
        pbar.refresh()

      def _pbar_callback_error(e):
        del e  # unused
        pbar.bar_style = 'danger'  # type: ignore

      futures = []
      for reader, context in zip(self._readers, self._reader_contexts):
        future = pool.apply_async(
            self._worker_handler,
            # Note: Serialization of context can be EXTREMELY slow
            # depending on the data type of context objects.
            args=[reader, context],
            kwds=dict(run_postprocess_fn=self._run_postprocess_fn),
            callback=_pbar_callback_done,
            error_callback=_pbar_callback_error,
        )
        futures.append(future)

      # The total number can grow while some jobs are running.
      _completed = int(pbar.n)
      pbar.reset(total=len(futures))
      pbar.n = pbar.last_print_n = _completed
      pbar.refresh()

      result = []
      for j, future in enumerate(futures):
        reader = self._readers[j]
        run, self._reader_contexts[j] = future.get()

        # TODO: better deal with failed runs.
        if run is not None:
          result.append(run)

      # All runs have been collected, close the progress bar.
      pbar.close()

    return RunList(result)

  def _get_runs_serial(self) -> RunList:
    """Non-parallel version of get_runs().

    It might be useful for debugging, but also might be faster for smaller
    numbers of runs and the rows in the log data, than the parallel version
    due to overhead in multiprocess, TF initialization, and serialization
    (which can often take up to 1~2 seconds).
    """
    runs = []
    pbar = tqdm(total=len(self._readers)) \
      if self._progress_bar else util.NoopTqdm()  # noqa: E127

    for j, reader in enumerate(self._readers):
      run, new_context = self._worker_handler(
          reader=reader,
          context=self._reader_contexts[j],
          run_postprocess_fn=self._run_postprocess_fn)
      self._reader_contexts[j] = new_context

      # TODO: better deal with failed runs.
      if run is not None:
        runs.append(run)
      pbar.update(1)

    return RunList(runs)

  # Asynchronous execution in a thread.
  _get_runs_async = util.wrap_async(get_runs)

  async def get_runs_async(self, polling_interval=0.5, **kwargs):
    """Asynchronous version of get_runs().

    This wraps get_runs() to be executed in a thread, so that the UI does not
    block while the loader is fetching and processing runs. Specifically,
    the tqdm progress bar could be updated when shown as a jupyter widget.
    """
    pbar = tqdm(total=len(self._readers))
    loop = asyncio.get_event_loop()
    future = loop.create_task(self._get_runs_async(tqdm_bar=pbar, **kwargs))
    while not future.done():
      await asyncio.sleep(polling_interval)
      pbar.update(0)

    return future.result()


__all__ = (
    'get_runs',
    'get_runs_serial',
    'get_runs_parallel',
    'iter_runs_serial',
    'parse_run',
)
