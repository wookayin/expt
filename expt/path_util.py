"""Path (local- and remote-) related utilities."""

import ast
import contextlib
import fnmatch
import functools
from glob import glob as local_glob
import io
import os
import os.path
from pathlib import Path
from pathlib import PurePath
from pathlib import PurePosixPath
import shlex
import shutil
import socket
import stat
import subprocess
import sys
from typing import (Any, Callable, Dict, List, Sequence, Tuple, TYPE_CHECKING,
                    Union)
from typing_extensions import Protocol
import urllib.parse

import multiprocessing_utils

PathType = Union[str, os.PathLike, PurePath]

if TYPE_CHECKING:
  import fabric
  import paramiko


class PathUtilInterface(Protocol):
  """Interface for path utils."""

  @staticmethod
  def supports(path: PathType) -> bool:
    raise NotImplementedError

  def glob(self, pattern: PathType) -> Sequence[str]:
    raise NotImplementedError

  def exists(self, path: PathType) -> bool:
    raise NotImplementedError

  def isdir(self, path: PathType) -> bool:
    raise NotImplementedError

  def open(self, path: PathType, *, mode='r'):
    raise NotImplementedError


def _to_path_string(path: PathType) -> str:
  if isinstance(path, (PurePath, Path)):
    return str(path)
  elif isinstance(path, str):
    return path
  else:
    raise TypeError(str(type(path)))


# ---------------------------------------------------------------------------
# Local Files
# ---------------------------------------------------------------------------


class LocalPathUtil(PathUtilInterface):

  @staticmethod
  def supports(path: PathType) -> bool:
    return True  # TODO Exclude other protocols

  def glob(self, pattern: PathType) -> Sequence[str]:
    pattern = _to_path_string(pattern)
    return local_glob(pattern)

  def exists(self, path: PathType) -> bool:
    path = _to_path_string(path)
    return os.path.exists(path)

  def isdir(self, path: PathType) -> bool:
    path = _to_path_string(path)
    return os.path.isdir(path)

  def open(self, path: PathType, *, mode='r'):
    path = _to_path_string(path)
    return io.open(path, mode=mode, encoding='utf-8')


# ---------------------------------------------------------------------------
# SSH / SFTP / SCP
# ---------------------------------------------------------------------------


class SFTPPathUtil(PathUtilInterface):

  @staticmethod
  def supports(path: PathType) -> bool:
    path = _to_path_string(path)
    if path.startswith('sftp://') or path.startswith('scp://'):
      try:
        import fabric  # pylint: disable=unused-import  # noqa
        import paramiko  # pylint: disable=unused-import  # noqa
      except ImportError as ex:
        raise ImportError("To use sftp://... or scp:// paths, "
                          "'fabric' and 'paramiko' are required. " +
                          str(ex)) from ex
      return True
    else:
      return False

  # TODO: Make verbosity control / logging (to see what's going on)

  # **Thread-local and multiprocess-safe** storage for per-host SFTP session,
  # when reusing SSH connection and SFTP session with keep_alive().
  _local_storage = multiprocessing_utils.local()

  @classmethod
  @contextlib.contextmanager
  def session(cls):
    S: Any = cls._local_storage

    # Create a thread-local (process-local) cache for per-host SSH connections.
    empty_cache: Dict[Tuple[str, str, int],  # (hostname, username, port)
                      Tuple['paramiko.SFTPClient',  # (sftp, conn)
                            'fabric.connection.Connection']] = {}

    if getattr(S, 'sftp_cache', None) is None:
      S.sftp_cache = empty_cache
      S.count = 1
    else:
      # nested session, reuse the cache
      S.count += 1

    try:
      yield
    finally:
      # Clean up the session: close all the cached resources and
      # free up the thread-local storage.
      S.count -= 1
      if S.count == 0:
        for _, (sftp, conn) in (S.sftp_cache or {}).items():
          with contextlib.suppress(IOError):
            sftp.close()
          with contextlib.suppress(IOError):
            conn.close()
          del sftp, conn

        S.sftp_cache = None

  @classmethod
  @contextlib.contextmanager
  def _establish(cls, path: PathType):
    path = _to_path_string(path)
    uri = urllib.parse.urlparse(path)

    import fabric.connection
    conn: 'fabric.connection.Connection'
    sftp: 'paramiko.SFTPClient'

    # Reuse a keepalive connection if already established before.
    sftp_cache = getattr(cls._local_storage, 'sftp_cache', None)
    sftp_cache_key = (uri.hostname, uri.username, uri.port)
    should_close = False

    if sftp_cache is not None and sftp_cache_key in sftp_cache:
      sftp, conn = sftp_cache[sftp_cache_key]
    else:
      # Make a new sftp connection if needed.
      conn = fabric.connection.Connection(
          host=uri.hostname,
          user=uri.username,
          port=uri.port or 22,
          connect_timeout=5.0,
      )
      try:
        sftp = conn.sftp()
      except socket.gaierror as ex:
        raise IOError(
            f"Cannot establish SSH connection to `{uri.netloc}`: {str(ex)}"
        ) from ex
      except TimeoutError as ex:
        raise TimeoutError(
            f"Cannot establish SSH connection to `{uri.netloc}`: {str(ex)}"
        ) from ex

      if sftp_cache is not None:
        sftp_cache[sftp_cache_key] = (sftp, conn)
      else:
        should_close = True

    try:
      remote_path = uri.path[1:] if uri.path.startswith('/') else uri.path
      yield sftp, uri, remote_path
    finally:
      if should_close:  # a new Connetion was created here, but not cached
        with contextlib.suppress(IOError):
          sftp.close()
        with contextlib.suppress(IOError):
          conn.close()
        del sftp, conn

  def glob(self, pattern: PathType) -> Sequence[str]:
    if '**' in _to_path_string(pattern):
      raise NotImplementedError("Globbing with double stars is not supported.")

    with self._establish(pattern) as (sftp, uri, remote_path):

      def walk(context: PurePosixPath, path_parts: Sequence[str]):
        if len(path_parts) == 0:
          yield str(context)
          return

        head, *tail = path_parts
        dir_ls = sftp.listdir(str(context))
        for f in dir_ls:
          if fnmatch.fnmatch(f, head):
            yield from walk(context / f, tail)

      prefix = "".join([
          uri.scheme + '://',
          uri.netloc,  # username + hostname + port
          '/',
      ])

      path_parts = PurePosixPath(remote_path).parts
      if not path_parts:
        return []
      if path_parts[0] == '/':
        # absolute path
        context = PurePosixPath('/')
        path_parts = path_parts[1:]
      else:
        # relative path from $HOME (or default cwd in SFTP)
        context = PurePosixPath('.')

      return [prefix + p for p in walk(context, path_parts)]

  def exists(self, path: PathType) -> bool:
    with self._establish(path) as (sftp, _, remote_path):
      try:
        sftp.stat(remote_path)
        return True
      except FileNotFoundError:
        return False

  def isdir(self, path: PathType) -> bool:
    with self._establish(path) as (sftp, _, remote_path):
      try:
        lstat = sftp.stat(remote_path)
        return stat.S_ISDIR(lstat.st_mode)  # type: ignore
      except FileNotFoundError:
        return False

  @contextlib.contextmanager
  def open(self, path: PathType, *, mode='r'):
    # Open a remote file, e.g., `with open(...) as f:`
    with self._establish(path) as (sftp, _, remote_path):
      yield sftp.open(remote_path)

  def download_local(self, path: PathType, tmpdir: str) -> str:
    """Fetches and download the remote file into a local temporary directory.

    Note that the temp directory to download into is an unique directory,
    so it does not get messed up with other eventfiles from other log_dirs.
    Callers should be responsible to ensure this and remove the directory
    after the use of file is complete.

    Returns the local path on the tmpdir as str.
    """
    if not os.path.isdir(tmpdir):
      raise FileNotFoundError(f"Not a directory: `{tmpdir}")

    basename = os.path.basename(path)
    local_path = os.path.join(tmpdir, basename)

    with self._establish(path) as (sftp, _, remote_path):
      # TODO: This is blocking, very slow. Can we make it async?
      sftp.get(remote_path, local_path, prefetch=True)

    return local_path


# yapf: disable
if TYPE_CHECKING:
  class TempFile(Protocol):  # see tempfile._TemporaryFileWrapper
    name: str
    def close(self): ...
else:
  TempFile = Any
# yapf: enable

# ---------------------------------------------------------------------------
# Google Cloud
# ---------------------------------------------------------------------------

# Options for gsutil. By default, gsutil is disabled as tf.io.gfile is
# **much** faster than gsutil commands (for globbing and listing).
USE_GSUTIL = bool(ast.literal_eval(os.environ.get('USE_GSUTIL', 'True')))
IS_GSUTIL_AVAILABLE = bool(shutil.which("gsutil"))
GSUTIL_NO_MATCHES = 'One or more URLs matched no objects'


class GCloudPathUtil(PathUtilInterface):

  @staticmethod
  def supports(path: PathType) -> bool:
    return _to_path_string(path).startswith('gs://')

  def glob(self, pattern: PathType) -> Sequence[str]:
    pattern = _to_path_string(pattern)

    # Bug: GCP glob does not match any directory on trailing slashes.
    # https://github.com/GoogleCloudPlatform/gsutil/issues/444
    pattern = pattern.rstrip('/')
    if USE_GSUTIL:
      try:
        if pattern.endswith('/*'):
          # A small optimization: 'gsutil ls foo/' is much faster
          # than 'gsutil ls -d foo/*' (around 3x~5x)
          return gsutil('ls', pattern.rstrip('*'))
        else:
          return gsutil('ls', '-d', pattern)
      except GsCommandException as e:
        if GSUTIL_NO_MATCHES in str(e):
          return []
        raise
    else:
      # A small optimization: when the pattern itself is a dir (no glob
      # was used) or the globbing '*' is used only at the last segment,
      # we can just simply list dir which is *much* faster than globbing.
      if self.isdir(pattern):
        return [pattern.rstrip('/')]
      elif pattern.endswith('/*') and self.isdir(pattern.rstrip('*')):
        dir_base = pattern.rstrip('*')
        ls = _import_gfile().listdir(dir_base)
        return [os.path.join(dir_base, entry) for entry in ls]
      else:
        return _import_gfile().glob(pattern)  # noqa

  def exists(self, path: PathType) -> bool:
    path = _to_path_string(path)

    if USE_GSUTIL:
      try:
        return bool(gsutil('ls', '-d', path))
      except GsCommandException as e:
        if GSUTIL_NO_MATCHES in str(e):
          return False
        raise
    else:
      return _import_gfile().exists(path)  # noqa

  def isdir(self, path: PathType) -> bool:
    path = _to_path_string(path)
    path = path.rstrip('/')
    return _import_gfile().isdir(path)  # noqa

  def open(self, path: PathType, *, mode='r'):
    path = _to_path_string(path)
    return _import_gfile().GFile(path, mode=mode)  # noqa


def _import_gfile():
  # Reference: https://www.tensorflow.org/api_docs/python/tf/io/gfile
  try:
    # pylint: disable-next=all
    import tensorflow.io.gfile as gfile  # type: ignore
    return gfile
  except ImportError as ex:
    raise RuntimeError("To be able to read GCP path (gs://...), "
                       "tensorflow should be installed. "
                       "(Cannot import tensorflow.io.gfile)") from ex


class GsCommandException(RuntimeError):
  pass


def gsutil(*args) -> List[str]:
  """Execute gsutil command synchronously and return the result as a list."""
  # TODO: Handle 401 exceptions.
  cmd = ['gsutil'] + list(args)
  p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

  lines = []
  assert p.stdout is not None
  for line in p.stdout:
    line = line.rstrip()
    if not isinstance(line, str):
      line = line.decode('utf-8')
    if line:
      lines.append(line)

  retcode = p.wait()
  if retcode != 0:
    stderr = p.stderr.read().decode('utf8')  # type: ignore
    if stderr.startswith('CommandException: '):
      raise GsCommandException(stderr[len('CommandException: '):])

    sys.stderr.write(stderr)
    sys.stderr.flush()
    raise subprocess.CalledProcessError(
        retcode, cmd=' '.join(shlex.quote(s) for s in cmd))
  return lines


def use_gsutil(value: bool):
  """Configure path_util to use gsutil."""
  global USE_GSUTIL
  if value:
    if not IS_GSUTIL_AVAILABLE:
      raise RuntimeError("gsutil is not available.")
    USE_GSUTIL = True
  else:
    USE_GSUTIL = False


# ---------------------------------------------------------------------------
# Public Module Interface
# ---------------------------------------------------------------------------

BACKENDS = [
    GCloudPathUtil(),
    SFTPPathUtil(),
    LocalPathUtil(),
]


def _choose_backend(path: PathType) -> PathUtilInterface:
  for backend in BACKENDS:
    if backend.supports(path):
      return backend
  raise ValueError(f"The path or URL `{path}` is not supported.")


def session():
  """Open a new session (as a context manager), within which expensive
  remote connection resources (e.g., SSH) can be cached and reused."""
  return SFTPPathUtil.session()


def session_wrap(fn: Callable):
  """Wrap a function call with a session() contextmanager."""

  @functools.wraps(fn)
  def _wrapped(*args, **kwargs):
    with session():
      return fn(*args, **kwargs)

  return _wrapped


def glob(pattern: PathType) -> Sequence[str]:
  """A glob function, returning a list of paths matching a pathname pattern.

  It supports local file path (i.e., glob.glob) and Google Cloud Storage
  (i.e., gs://...) path via gfile.glob(...).
  """
  return _choose_backend(pattern).glob(pattern)


def exists(path: PathType) -> bool:
  """Similar to os.path.exists(path), but supports both local path and
  remote path (Google Cloud Storage, gs://...) via gfile.exists(...).
  """
  return _choose_backend(path).exists(path)


def isdir(path: PathType):
  """Similar to os.path.isdir(path), but supports both local path and
  remote path (Google Cloud Storage, gs://...) via gfile.isdir(...).
  """
  return _choose_backend(path).isdir(path)


# pylint: disable-next=redefined-builtin
def open(path: PathType, *, mode='r'):
  """Similar to built-in open(...), but supports Google Cloud Storage
  (i.e., gs://...) path via gfile.GFile(...) as well as local path.
  """
  return _choose_backend(path).open(path, mode=mode)
