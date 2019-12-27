import numpy as np
from scipy import interpolate

from typing import List
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
