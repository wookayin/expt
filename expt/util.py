"""Utilities for expt."""

import asyncio
import collections
import concurrent.futures
import contextlib
import functools
from typing import Callable, Collection, List, Optional, TypeVar
import warnings

from typeguard import typechecked

T = TypeVar('T')

# Make DeprecationWarning within expt printed, but only once
warnings.filterwarnings(
    "once", "", category=DeprecationWarning, module=r"^expt\.")


def warn_deprecated(msg):
  warnings.warn(msg, DeprecationWarning, stacklevel=2)


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


def ensure_unique(items: Collection[T]) -> T:
  s = set(items)
  if len(s) == 0:
    raise ValueError("`items` is empty.")
  elif len(s) == 1:
    return next(iter(s))

  raise ValueError("`items` contains non-unique values: {s}")


def ensure_notNone(x: Optional[T]) -> T:
  assert x is not None
  return x


# Fallback for tqdm.
class NoopTqdm:
  """A dummy, no-op tqdm used when tqdm is missing."""

  def __init__(self, *args, **kwargs):
    del args, kwargs  # unused
    self.total = self.max = 0
    self.n = self.last_print_n = 0
    self.leave = False

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


def wrap_async(blocking_fn: Callable):
  """A wrapper that wraps a synchronous function into an asynchornous
  function."""

  executor = concurrent.futures.ThreadPoolExecutor()

  @functools.wraps(blocking_fn)
  async def wrapped(*args, **kwargs):
    loop = asyncio.get_event_loop()
    func = functools.partial(blocking_fn, *args, **kwargs)
    result = await loop.run_in_executor(executor, func)
    return result

  wrapped._executor = executor
  return wrapped
