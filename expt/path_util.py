"""
Path (local- and remote-) related utilities.
"""

import io
import os.path
import shlex
import subprocess
import sys
from glob import glob as local_glob
from typing import List

from distutils.spawn import find_executable


IS_GSUTIL_AVAILABLE = find_executable("gsutil")
GSUTIL_NO_MATCHES = 'One or more URLs matched no objects'


def _import_gfile():
    # Reference: https://www.tensorflow.org/api_docs/python/tf/io/gfile
    try:
        import tensorflow.io.gfile as gfile
        return gfile
    except:
        pass

    raise RuntimeError("To be able to read GCP path (gs://...), "
                        "tensorflow should be installed. "
                        "(Cannot import tensorflow.io.gfile)")


class GsCommandException(RuntimeError):
   pass


def gsutil(*args) -> List[str]:
    """Execute gsutil command synchronously and return the result as a list."""
    # TODO: Handle 401 exceptions.
    cmd = ['gsutil'] + list(args)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    retcode = p.wait()
    if retcode != 0:
        stderr = p.stderr.read().decode('utf8')  # type: ignore
        if stderr.startswith('CommandException: '):
            raise GsCommandException(stderr[len('CommandException: '):])

        sys.stderr.write(stderr)
        sys.stderr.flush()
        raise subprocess.CalledProcessError(
            retcode, cmd=' '.join(shlex.quote(s) for s in cmd))
    return p.stdout.read().decode('utf8').rstrip('\n').split('\n')  # type: ignore


def glob(pattern):
    """
    A glob function, returning a list of paths matching a pathname pattern.

    It supports local file path (i.e., glob.glob) and Google Cloud Storage
    (i.e., gs://...) path via gfile.glob(...).
    """
    if pattern.startswith('gs://'):
        # Bug: GCP glob does not match any directory on trailing slashes.
        # https://github.com/GoogleCloudPlatform/gsutil/issues/444
        pattern = pattern.rstrip('/')
        if IS_GSUTIL_AVAILABLE:
            try:
                return gsutil('ls', '-d', pattern)
            except GsCommandException as e:
                if GSUTIL_NO_MATCHES in str(e):
                    return []
                raise
        else:
            # Note: gfile.glob is extremely slow (way slower than gsutil)
            return _import_gfile().glob(pattern)  # noqa
    else:
        return local_glob(pattern)


def exists(path) -> bool:
    """
    Similar to os.path.exists(path), but supports both local path and
    remote path (Google Cloud Storage, gs://...) via gfile.exists(...).
    """
    if path.startswith('gs://'):
        if IS_GSUTIL_AVAILABLE:
            try:
                return bool(gsutil('ls', '-d', path))
            except GsCommandException as e:
                if GSUTIL_NO_MATCHES in str(e):
                    return False
                raise
        else:
            return _import_gfile().exists(path)  # noqa
    else:
        return os.path.exists(path)


def isdir(path):
    """
    Similar to os.path.isdir(path), but supports both local path and
    remote path (Google Cloud Storage, gs://...) via gfile.isdir(...).
    """
    if path.startswith('gs://'):
        # Bug: GCP glob does not match any directory on trailing slashes.
        # https://github.com/GoogleCloudPlatform/gsutil/issues/444
        path = path.rstrip('/')
        return _import_gfile().isdir(path)  # noqa
    else:
        return os.path.isdir(path)


def open(path, *, mode='r'):
    """
    Similar to built-in open(...), but supports Google Cloud Storage
    (i.e., gs://...) path via gfile.GFile(...) as well as local path.
    """
    if path.startswith('gs://'):
        return _import_gfile().GFile(path, mode=mode)  # noqa
    else:
        return io.open(path, mode=mode)
