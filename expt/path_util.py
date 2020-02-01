"""
Path (local- and remote-) related utilities.
"""

import io
import os.path

from glob import glob as local_glob


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
        return _import_gfile().glob(pattern)  # noqa
    else:
        return local_glob(pattern)


def exists(path):
    """
    Similar to os.path.exists(path), but supports both local path and
    remote path (Google Cloud Storage, gs://...) via gfile.exists(...).
    """
    if path.startswith('gs://'):
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
