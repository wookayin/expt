"""
Tests for expt.plot
"""

import sys
import numpy as np
import pandas as pd
import pytest

import expt.data
from expt.data import Experiment, Hypothesis, Run, RunList

try:
    from rich.console import Console
    console = Console(markup=False)
    print = console.log
except ImportError:
    pass


def V(x):
    """Print the object and return it."""
    kwargs = dict(_stack_offset=2) if print.__name__ == 'log' else {}
    print(x, **kwargs)
    return x


class TestHypothesisPlot:

    def setup_method(self, method):
        sys.stdout.write("\n")

    def _fixture(self) -> Hypothesis:
        data = dict()
        data['step'] = np.arange(1000)
        data['loss'] = np.exp((-data['step'] + 700) / 1000.0)
        data['accuracy'] = data['step'] / 1000.0
        data['lr'] = np.ones(1000) * 0.001
        runs = [Run("seed0", pd.DataFrame(data))]
        h = Hypothesis(name="hypo_plot", runs=runs)
        return h

    # TODO: This test suite is incomplete. Add more tests.

    def testGridSpec(self):
        hypothesis = self._fixture()

        # single y
        g = hypothesis.plot(y="loss")
        assert V(g.axes.shape) == (1, 1)
        assert V(g.axes_active.shape) == (1, )

        # two y's   [2x1]
        g = hypothesis.plot(y=["loss", "accuracy"])
        assert V(g.axes.shape) == (1, 2)

        # three y's [2x2]
        g = hypothesis.plot(y=["loss", "accuracy", "lr"])
        assert V(g.axes.shape) == (2, 2)
        assert V(g.axes_active.shape) == (3, )

    def testWhenFigAxesAreGiven(self):
        hypothesis = self._fixture()
        import matplotlib.pyplot as plt

        # single y given a single axesplot
        fig, ax = plt.subplots()
        g = hypothesis.plot(y="loss", ax=ax)
        assert g.axes.shape == (1, 1)
        assert g.axes_active[0] is ax       # we need to reuse the given ax
        assert g.figure is fig

        # length mismatch?
        with pytest.raises(ValueError) as ex:
            g = hypothesis.plot(y=["loss", "accuracy"], ax=ax)
        assert "The length of `ax` and `y` must be equal" in str(ex.value)

        # two y's given two axesplots
        fig, axes = plt.subplots(2, 1, squeeze=False)
        g = hypothesis.plot(y=["loss", "accuracy"], ax=axes)
        assert g.axes.shape == (2, 1)
        assert g.axes_active[0] is axes.flat[0]
        assert g.axes_active[1] is axes.flat[1]
