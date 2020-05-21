from typing import List
import collections

import numpy as np
from scipy import interpolate
from typeguard import typechecked


@typechecked
def prettify_labels(labels: List[str]) -> List[str]:
    """
    Apply a sensible default so that the plot does not look ugly.
    """

    # truncate length
    def _truncate(l: str) -> str:
        if len(l) >= 20:
            return '...' + l[-17:]
        return l
    labels = [_truncate(l) for l in labels]

    return labels


def merge_list(*lst):
    """Merge given lists into one without duplicated entries. Orders are
    preserved as in the order of each of the flattened elements appear."""

    merged = {}
    for l in lst:
        merged.update(collections.OrderedDict.fromkeys(l))
    return list(merged)
