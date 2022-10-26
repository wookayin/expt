"""Path (local- and remote-) related utilities."""

import ast
from distutils.spawn import find_executable
from glob import glob as local_glob
import io
import os
import os.path
from pathlib import Path
import shlex
import subprocess
import sys
from typing import List, Sequence, Union
from typing_extensions import Protocol

PathType = Union[str, os.PathLike]


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
  if isinstance(path, Path):
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
# Google Cloud
# ---------------------------------------------------------------------------

# Options for gsutil. By default, gsutil is disabled as tf.io.gfile is
# **much** faster than gsutil commands (for globbing and listing).
USE_GSUTIL = bool(ast.literal_eval(os.environ.get('USE_GSUTIL', 'True')))
IS_GSUTIL_AVAILABLE = find_executable("gsutil")
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
    LocalPathUtil(),
]


def _choose_backend(path: PathType) -> PathUtilInterface:
  for backend in BACKENDS:
    if backend.supports(path):
      return backend
  raise ValueError(f"The path or URL `{path}` is not supported.")


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
