"""Utilities for expt."""

import collections
from typing import List

from typeguard import typechecked


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
