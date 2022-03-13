"""Utilities for expt."""

import collections
import contextlib
from typing import List

from typeguard import typechecked


class PropertyAccessor:
  """A property-like object."""

  def __init__(self, name: str, accessor):
    self._name = name
    self._accessor = accessor

  def __get__(self, obj, cls):
    if obj is None:
      return self._accessor
    accessor_obj = self._accessor(obj)
    object.__setattr__(obj, self._name, accessor_obj)
    return accessor_obj


@typechecked
def prettify_labels(labels: List[str]) -> List[str]:
  """Apply a sensible default so that the plot does not look ugly."""

  # truncate length
  def _truncate(label: str) -> str:
    if len(label) >= 20:
      return '...' + label[-17:]
    return label

  labels = [_truncate(label) for label in labels]

  return labels


def merge_list(*lists):
  """Merge given lists into one without duplicated entries. Orders are
  preserved as in the order of each of the flattened elements appear."""

  merged = {}
  for item in lists:
    merged.update(collections.OrderedDict.fromkeys(item))
  return list(merged)


# Fallback for tqdm.
class NoopTqdm:
  """A dummy, no-op tqdm used when tqdm is missing."""

  def __init__(self, *args, **kwargs):
    del args, kwargs  # unused
    self.total = self.max = 0
    self.n = self.last_print_n = 0

  def noop(self, *args, **kwargs):  # pylint: disable=C0116
    del args, kwargs  # unused

  def __getattr__(self, _):
    return self.noop


@contextlib.contextmanager
def timer(name=""):
  import time
  _start = time.time()
  yield
  _end = time.time()
  print("[%s] Elapsed time : %.3f sec" % (name, _end - _start))
