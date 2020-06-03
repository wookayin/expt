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

    @staticmethod
    def _fixture() -> Hypothesis:
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

        # two y's, given a 1D array of axesplots: error
        fig, axes = plt.subplots(1, 2)
        assert axes.shape == (2,)
        with pytest.raises(ValueError) as ex:
            g = hypothesis.plot(y=["loss", "accuracy"], ax=axes)
            #assert g.axes.shape == (1, 2)
        assert "the rank should be 2 (but given 1)" in str(ex.value)

        # two y's, given 2D array of axesplots
        fig, axes = plt.subplots(2, 1, squeeze=False)
        g = hypothesis.plot(y=["loss", "accuracy"], ax=axes)
        assert g.axes.shape == (2, 1)
        assert g.axes_active[0] is axes.flat[0]
        assert g.axes_active[1] is axes.flat[1]


class TestExperimentPklot:

    def setup_method(self, method):
        sys.stdout.write("\n")

    @staticmethod
    def _fixture() -> Experiment:
        # Note different column names (some are shared, some are not)
        h0 = Hypothesis("hyp0", Run('r0', pd.DataFrame(
            {"a": [1, 2, 3], "b0": [10, 9, 8]})))
        h1 = Hypothesis("hyp1", Run('r1', pd.DataFrame(
            {"a": [4, 5, 6], "b1": [7, 6, 5]})))
        ex = Experiment(title="ex", hypotheses=[h0, h1])
        return ex

    def testGridPlotBasic(self):   # TODO: Add more complex scenario.
        ex = self._fixture()

        # plot all known columns
        g = V(ex.plot())
        assert g.axes.shape == (2, 2)
        assert len(g.axes_active) == 3  # a, b0, b1

        # __getitem__ (str)
        # TODO: validate contents
        assert g['a'] is g.axes_active[0]
        assert g['b0'] is g.axes_active[1]
        assert g['b1'] is g.axes_active[2]
        for y in ('a', 'b0', 'b1'):
            assert g[y].get_title() == y

        with pytest.raises(ValueError) as ex:
            g['b']
        assert 'Unknown index: b' in V(str(ex.value))
        assert "['b1', 'b0']" in str(ex.value) or \
               "['b0', 'b1']" in str(ex.value)

        # __getitem__ (int)
        assert g[0] is g.axes_active[0]
        assert g[1] is g.axes_active[1]
        assert g[-1] is g.axes_active[2]  # we have 3 active axes
