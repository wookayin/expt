"""Data Loader for expt."""

import abc
import asyncio
import atexit
from collections import Counter
from collections import defaultdict
import contextlib
import dataclasses
import itertools
import multiprocessing.pool
import os
import pathlib
from pathlib import Path
import sys
import tempfile
from typing import (Any, Callable, Dict, Generic, Iterator, NamedTuple,
                    Optional, Sequence, Tuple, Type, TYPE_CHECKING, TypeVar,
                    Union)

import multiprocess.pool
import numpy as np
import pandas as pd

from expt import path_util
from expt import util
from expt.data import Run
from expt.data import RunList

try:
  from tqdm.auto import tqdm
except ImportError:
  tqdm = util.NoopTqdm

if TYPE_CHECKING:
  from tqdm import tqdm as tqdm_std
  ProgressBar = Union[tqdm, tqdm_std]
else:
  ProgressBar = Any

try:
  import tensorboard
except ImportError:
  tensorboard = None

#########################################################################
# Individual Run Reader
#########################################################################

LogReaderContext = TypeVar('LogReaderContext')


class LogReader(abc.ABC, Generic[LogReaderContext]):
  """Interface for reading logs.

  LogReaders should maintain its internal state stored in a context object,
  so that it can work even in a forked multiprocess worker.

  Note that LogReaders must be forkable (i.e., serialize and deserialize)
  and its context object must be serializable so as to be sent to multiprocess.

  Note that the constructor of LogReader implementations will throw
  an exception of type `CannotHandleException`, if the given `log_dir`
  cannot be handeled by the LogReader. E.g., when there is no tensorboard
  eventfile for a TensorboardLogReader.
  """

  def __init__(self, log_dir):
    if not isinstance(log_dir, (pathlib.Path, str)):
      raise TypeError(f"`log_dir` must be a `str` or `Path`,"
                      f" but given {type(log_dir)}")
    self._log_dir = str(log_dir)

    if not path_util.isdir(log_dir):
      raise FileNotFoundError(log_dir)

  @property
  def log_dir(self) -> str:
    return self._log_dir

  @abc.abstractmethod
  def new_context(self) -> LogReaderContext:
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

  def read_once(self, verbose=False) -> pd.DataFrame:
    ctx = self.read(self.new_context(), verbose=verbose)
    return self.result(ctx)

  def __repr__(self):
    return f"<{type(self).__name__}, log_dir={self.log_dir}>"


class CannotHandleException(RuntimeError):
  """Raised when LogReader cannot handle a given directory."""

  def __init__(self,
               log_dir,
               reader: Optional[LogReader] = None,
               reason: str = ""):
    msg = f"log_dir `{log_dir}` cannot be handled"
    if reader:
      msg += f" by {type(reader).__name__}"
    if reason:
      msg += f" (reason: {reason})"
    msg += "."
    super().__init__(msg)


class CSVLogReader(LogReader[pd.DataFrame]):
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
      raise CannotHandleException(log_dir, self,
                                  "Does not contain progress.csv or log.csv")

    self._csv_path = detected_csv

  def new_context(self) -> pd.DataFrame:
    return pd.DataFrame()

  def read(self, context: pd.DataFrame, verbose=False) -> pd.DataFrame:
    # unused, this implementation always ignore the previous context
    # and read the data in full (no incremental reading).
    del context

    # Read the detected file `p`
    if verbose:
      print(f"CSVLogReader: Reading {self._csv_path}",
            file=sys.stderr, flush=True)  # yapf: disable

    df: pd.DataFrame
    with path_util.open(self._csv_path, mode='r') as f:
      df = pd.read_csv(f)  # type: ignore

    return df

  def result(self, context: pd.DataFrame) -> pd.DataFrame:
    df = context
    assert isinstance(df, pd.DataFrame)
    return df


if tensorboard or TYPE_CHECKING:
  from tensorboard.backend.event_processing import event_file_loader

  class RemoteAwareEventFileLoader(event_file_loader.LegacyEventFileLoader):
    """An iterator that yields parsed Event protos over remote a file.

    This supports other protocols such as sftp:// like expt.path_util do;
    tensorboard's event file loader supports local and gcs:// paths only.

    Currently, the GFile backend (PyRecordReader_New in tensorboard's TF stub
    or the tensorflow.io core module) can support ONLY local and GCS files,
    no way to open a file through a python IO (io.IOBase) or an unix socket.
    The only way to read from a remote file (via SSH/SFTP) is actually to
    download and stream the remote file into a local, temporary file
    (for the sake of maximum throughput).

    Note: .close() must be called otherwise a resource/connection will leak.
    """

    def __init__(self, path: str):
      if path_util.SFTPPathUtil.supports(path):
        # TODO: Verbose logging of file path, etc.
        # TODO: This is blocking in a thread. Add an asynchronous version.
        self._tmpdir = tempfile.TemporaryDirectory(prefix="expt")
        self._local_file = path_util.SFTPPathUtil().download_local(
            path, self._tmpdir.name)
        super().__init__(self._local_file)
      else:
        self._tmpdir = None
        super().__init__(path)

    def close(self):
      with contextlib.suppress(IOError):
        if self._tmpdir:
          self._tmpdir.cleanup()


class TensorboardLogReader(  # ...
    LogReader['TensorboardLogReader.Context']):  # type: ignore  # noqa
  """A (python-based) Log reader for tensorboard run directory.

  Note that this implementation relies on `tf.io.GFile` which uses some
  native code to iterate event protobufs, or on tensorboard's fallback
  in the absence of tensorflow. However, both are very slow, so an use of
  RustTensorboardLogReader is recommended.
  """

  def __init__(self, log_dir):
    super().__init__(log_dir=log_dir)

    # Initialize the resources.
    event_glob = os.path.join(log_dir, '*events.out.tfevents.*')

    # TODO: When a new event file is added?
    self._event_files = list(sorted(path_util.glob(event_glob)))
    if not self._event_files:  # no event file detected
      raise CannotHandleException(log_dir, self, "No event file detected")

    # Ensure tensorboard is imported, so forked workers don't need to load again
    # pylint: disable-next=all
    import tensorboard.backend.event_processing

  @dataclasses.dataclass
  class Context:  # LogReaderContext
    rows_read: Counter = dataclasses.field(default_factory=Counter)
    data: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)
    last_read_rows: int = 0

  def new_context(self) -> 'Context':
    return self.Context()

  # Helper function
  def _extract_scalar_from_proto(self, value, step):
    from tensorboard.compat.tensorflow_stub.dtypes import DType

    if value.HasField('simple_value'):  # v1
      simple_value = value.simple_value
      yield step, value.tag, simple_value

    elif value.HasField('metadata'):  # v2 eventfile
      plugin_name = None
      with contextlib.suppress(AttributeError):
        plugin_name = value.metadata.plugin_data.plugin_name
      if plugin_name == 'scalars':
        if hasattr(value, 'tensor') and value.tensor:
          t = value.tensor
          dtype = DType(t.dtype).as_numpy_dtype
          if hasattr(t, 'float_val') and t.float_val:
            simple_value = t.float_val[0]
          else:
            simple_value = np.frombuffer(t.tensor_content, dtype=dtype)[0]
          yield step, value.tag, simple_value

  def _iter_scalar_summary_from(self, event_file, *,
                                skip=0, limit=None,  # per event_file
                                rows_callback):  # yapf: disable
    if not TYPE_CHECKING:
      from tensorboard.compat.proto.event_pb2 import Event
    else:

      class Event(NamedTuple):  # a type checking stub, see event_pb2.Event
        step: int
        HasField: Callable[[str], bool]
        summary: Any

    def summary_iterator(path, skip=0) -> Iterator[Event]:
      # Requires tensorboard >= 2.3.0
      reader = RemoteAwareEventFileLoader(path)
      eventfile_iterator = reader.Load()

      eventfile_iterator = itertools.islice(
          eventfile_iterator,
          skip,
          skip + limit if limit is not None else None,
      )
      return eventfile_iterator  # type: ignore

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
        print(f"TensorboardLogReader: Reading {event_file} ...",
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
    df_chunk.index.name = 'global_step'
    df_chunk['global_step'] = df_chunk.index.astype(int)

    # Merge the previous dataframe and the new one that was read.
    # The current chunk will overwrite any existing previous row.
    df = df_chunk.combine_first(context.data)
    context.data = df

    return context

  def result(self, context: 'Context') -> pd.DataFrame:
    # Reorder column names in a lexicographical order
    df = context.data
    df = df.reindex(sorted(df.columns), axis=1)
    return df


class RustTensorboardLogReader(LogReader[Dict]):
  """Log reader for tensorboard run directory, backed by a rust extension.

  This rust-based implementation should be x15 ~ x20 faster than the old
  TensorboardLogReader written in Python.
  """

  def __init__(self, log_dir):
    super().__init__(log_dir=log_dir)

    # Import early, so that multiprocess workers do not need to import again
    # pylint: disable-next=unused-import
    try:
      import expt._internal  # noqa: F401  # type: ignore
    except ImportError as ex:
      raise CannotHandleException(
          log_dir, self, "expt's rust extension is not installed.") from ex

    event_glob = os.path.join(log_dir, '*events.out.tfevents.*')

    # TODO: Cannot handle a case where new eventfile is created afterwards.
    self._eventfiles = path_util.glob(event_glob)
    if not self._eventfiles:  # no event file detected
      raise CannotHandleException(log_dir, self, "No event file detected")

    self._is_remote = path_util.SFTPPathUtil.supports(self.log_dir)

  def new_context(self) -> Dict:
    return {}

  def read(self, context: Dict, verbose=False):
    if self._is_remote:
      return self._read_remote(context, verbose=verbose)
    else:
      return self._read_local(context, verbose=verbose)

  def _read_remote(self, context: Dict, verbose=False):
    import expt._internal
    del context  # No context is used, always read in full.

    tmp_prefix = f"expt-{os.path.basename(self.log_dir)[:64]}-"
    with tempfile.TemporaryDirectory(prefix=tmp_prefix) as tmpdir:
      tmpdir = os.path.join(tmpdir, os.path.basename(self.log_dir))
      os.makedirs(tmpdir, exist_ok=False)

      if verbose:
        print(f"RustTensorboardLogReader: Downloading to {tmpdir}")
      _download_local = path_util.SFTPPathUtil().download_local

      for remote_file in self._eventfiles:
        local_file = _download_local(remote_file, tmpdir=tmpdir)
        assert os.path.exists(local_file)

      # TODO: Create the reader object once, and reuse it.
      # To make this possible, we should either make it serializable or
      # improve the pickle-for-multiprocess structure of LogReaders.
      # TODO: Serialization overhead is quite heavy; slower in multiprocess.
      # pylint: disable-next=c-extension-no-member,protected-access
      reader = expt._internal.TensorboardEventFileReader(tmpdir)
      return reader.get_data()

  def _read_local(self, context: Dict, verbose=False):
    import expt._internal
    del context  # No context is used, always read in full.

    # pylint: disable-next=c-extension-no-member,protected-access
    reader = expt._internal.TensorboardEventFileReader(self.log_dir)
    return reader.get_data()

  def result(self, context: Dict) -> pd.DataFrame:
    # context: Dict[str, List[ Tuple[Step, Value] ]]
    df = pd.DataFrame(context)
    df.index.name = 'global_step'
    df['global_step'] = df.index.astype(int)
    df = df.reindex(sorted(df.columns), axis=1)
    return df


DEFAULT_READER_CANDIDATES = (
    CSVLogReader,
    RustTensorboardLogReader,
    TensorboardLogReader,
)


def _get_reader_for(log_dir,
                    *,
                    candidates: Optional[Sequence[Type[LogReader]]] = None,
                    verbose=False) -> LogReader:
  if candidates is None:
    candidates = DEFAULT_READER_CANDIDATES

  for reader_cls in candidates:
    if not issubclass(reader_cls, LogReader):
      raise TypeError(f"`{reader_cls}` is not a subtype of LogReader.")

    try:
      reader: LogReader = reader_cls(log_dir)
      return reader
    except CannotHandleException as ex:
      # When log_dir is not supported by the reader,
      # an expected exception is thrown. Try the next one.
      if verbose:
        print(str(ex), file=sys.stderr)
        sys.stderr.flush()

  tried: str = ', '.join(t.__name__ for t in candidates)
  raise CannotHandleException(log_dir, None,
                              f"No available readers, tried: {tried}")


#########################################################################
# parse_run methods (deprecated)
#########################################################################


def parse_run(
    log_dir: Union[str, Path],
    *,
    reader_cls: Optional[Type[LogReader]] = None,
    verbose=False,
) -> pd.DataFrame:
  """(Deprecated) Create a pd.DataFrame object from a single directory."""

  util.warn_deprecated(
      "Use of parse_run is deprecated, and it is no longer used internally "
      "except for testing. Instead, please use LogReader class directly.")

  if verbose:
    # TODO Use python logging
    print(f"Reading {log_dir} ...", file=sys.stderr, flush=True)

  reader = _get_reader_for(
      log_dir,
      candidates=[reader_cls] if reader_cls else None,
      verbose=verbose,
  )

  ctx = reader.read(reader.new_context(), verbose=verbose)
  df = reader.result(ctx)

  # add some optional metadata... (might not be preserved afterwards)
  if df is not None:
    df.path = log_dir
  return df


def parse_run_progresscsv(log_dir, verbose=False) -> pd.DataFrame:
  return parse_run(log_dir, reader_cls=CSVLogReader,
                   verbose=verbose)  # yapf: disable


def parse_run_tensorboard(log_dir, verbose=False) -> pd.DataFrame:
  return parse_run(log_dir, reader_cls=RustTensorboardLogReader,
                   verbose=verbose)  # yapf: disable


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
    run_postprocess_fn=None,
) -> Iterator[Run]:
  """Enumerate Run objects from the given path(s)."""

  loader = RunLoader(
      *path_globs,
      verbose=verbose,
      run_postprocess_fn=run_postprocess_fn,
  )
  # pylint: disable-next=protected-access
  yield from loader._iter_runs_serial()


def get_runs_serial(*path_globs,
                    verbose=False,
                    run_postprocess_fn=None) -> RunList:
  """Get a list of Run objects from the given path(s).

  This works in single-thread (very slow, should not used other than
  debugging purposes).
  """
  runs = list(
      iter_runs_serial(
          *path_globs, verbose=verbose, run_postprocess_fn=run_postprocess_fn))

  if not runs:
    for path_glob in path_globs:
      print(f"Warning: No match found for pattern {path_glob}",
            file=sys.stderr)  # yapf: disable

  return RunList(runs)


def get_runs_parallel(
    *path_globs,
    verbose=False,
    n_jobs=8,
    pool_class=multiprocess.pool.Pool,
    progress_bar=True,
    run_postprocess_fn=None,
) -> RunList:
  """Get a list of Run objects from the given path glob patterns.

  This runs in parallel, using the multiprocess library (a fork of python
  standard library multiprocessing) which is more friendly with ipython
  and serializing non-picklable objects.

  Deprecated: Use expt.RunLoader instead.
  """

  loader = RunLoader(
      *path_globs,
      verbose=verbose,
      progress_bar=progress_bar,
      run_postprocess_fn=run_postprocess_fn,
      n_jobs=n_jobs,
      pool_class=pool_class,
  )
  try:
    return loader.get_runs()
  finally:
    loader.close()


get_runs = get_runs_parallel


async def get_runs_async(
    *path_globs,
    verbose=False,
    n_jobs=8,
    pool_class=multiprocess.pool.Pool,
    progress_bar=True,
    run_postprocess_fn=None,
) -> RunList:
  """An asynchronous version of get_runs."""

  loader = RunLoader(
      *path_globs,
      verbose=verbose,
      progress_bar=progress_bar,
      run_postprocess_fn=run_postprocess_fn,
      n_jobs=n_jobs,
      pool_class=pool_class,
  )
  try:
    return await loader.get_runs_async()
  finally:
    loader.close()


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
      pool_class=multiprocess.pool.Pool,
      reader_cls: Union[None, Type[LogReader],  # ...
                        Sequence[Type[LogReader]]] = None,
  ):
    self._readers = []
    self._reader_contexts = []

    self._verbose = verbose
    self._progress_bar = progress_bar
    self._run_postprocess_fn = run_postprocess_fn

    if isinstance(reader_cls, Type):
      reader_cls = [reader_cls]
    self._reader_cls: Optional[Sequence[Type[LogReader]]] = reader_cls

    self.add_paths(*path_globs)

    # Initialize multiprocess pool.
    if n_jobs > 1:
      if isinstance(pool_class, str):
        Pool = {
            'threading': multiprocess.pool.ThreadPool,
            'multiprocess': multiprocess.pool.Pool,  # third-party (better)
            'multiprocessing': multiprocessing.pool.Pool,  # python stdlib
        }.get(pool_class, None)
        if not Pool:
          raise ValueError("Unknown pool_class: {} ".format(pool_class) +
                           "(expected: threading or multiprocessing)")
      elif callable(pool_class):
        Pool = pool_class
      else:
        raise TypeError("Unknown type for pool_class: {}".format(pool_class))
      self._pool = Pool(processes=n_jobs)
    else:
      self._pool = None

    atexit.register(self.close)

  def close(self):
    if self._pool:
      self._pool.close()
      self._pool = None

  @path_util.session_wrap
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
    reader = _get_reader_for(log_dir, candidates=self._reader_cls)
    self.add_reader(reader)
    return reader

  def add_reader(self, reader: LogReader):
    self._readers.append(reader)
    self._reader_contexts.append(reader.new_context())

  @staticmethod
  def _worker_handler(
      reader: LogReader,  # pickled and passed into a worker
      context: LogReaderContext,
      run_postprocess_fn: Optional[Callable[[Run], Run]] = None,
  ) -> Tuple[Optional[Run], LogReaderContext]:
    """The job function to be executed in a "forked" worker process."""
    try:
      with path_util.session():
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

  def get_runs(self,
               *,
               parallel=True,
               tqdm_bar: Optional[ProgressBar] = None) -> RunList:
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
      return RunList(self._iter_runs_serial(tqdm_bar=tqdm_bar))

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

  def _iter_runs_serial(self, tqdm_bar: Optional[ProgressBar] = None):
    """Non-parallel version of get_runs().

    It might be useful for debugging, but also might be faster for smaller
    numbers of runs and the rows in the log data, than the parallel version
    due to overhead in multiprocess, TF initialization, and serialization
    (which can often take up to 1~2 seconds).
    """
    pbar = tqdm_bar
    if pbar is None:
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
        yield run
      pbar.update(1)

    pbar.close()

  # Asynchronous execution in a thread.
  _get_runs_async = util.wrap_async(get_runs)

  async def get_runs_async(self,
                           polling_interval=0.5,
                           tqdm_bar: Optional[ProgressBar] = None,
                           **kwargs):
    """Asynchronous version of get_runs().

    This wraps get_runs() to be executed in a thread, so that the UI does not
    block while the loader is fetching and processing runs. Specifically,
    the tqdm progress bar could be updated when shown as a jupyter widget.
    """
    if tqdm_bar is None:
      pbar = tqdm(total=len(self._readers))
    else:
      pbar = tqdm_bar

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
    'get_runs_async',
    'LogReader',
    'CannotHandleException',
    'CSVLogReader',
    'TensorboardLogReader',
    'RustTensorboardLogReader',
    'RunLoader',
)
