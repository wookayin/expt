"""
Tests for expt.plot
"""

import sys
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

import expt.plot
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

    def testSuptitle(self):
        hypothesis = self._fixture()

        # when given, it should be set
        g = hypothesis.plot(suptitle="super!")
        assert g.fig._suptitle.get_text() == "super!"

        # default behavior: hypothesis name?
        g = hypothesis.plot()
        assert g.fig._suptitle.get_text() == hypothesis.name

        # default behavior: if ax or grid is given, do not set suptitle
        fig, ax = plt.subplots()
        g = hypothesis.plot(ax=ax, y='loss')
        assert g.fig._suptitle is None or g.fig._suptitle.get_text() == ""

        g = expt.plot.GridPlot(y_names=['loss'])
        g = hypothesis.plot(grid=g, y='loss')
        assert g.fig._suptitle is None or g.fig._suptitle.get_text() == ""

    def testSingleHypothesisLegend(self):
        hypothesis = self._fixture()

        # default behavior: no legend
        g = hypothesis.plot()
        assert g.figure.legends == []
        for ax in g.axes_active: assert ax.get_legend() is None

        # boolean (True/False)
        g = hypothesis.plot(legend=False)
        for ax in g.axes_active: assert ax.get_legend() is None
        g = hypothesis.plot(legend=True)
        for ax in g.axes_active:
            assert len(ax.get_legend().texts) == 1

        # int/str
        g = hypothesis.plot(legend='loss')
        for ax in g.axes_active:
            assert bool(ax.get_legend()) == (ax.get_title() == 'loss'), str(ax.get_title())

        # dict
        g = hypothesis.plot(legend=dict(ax=-1, loc='center'))  # the last one: lr
        for ax in g.axes_active:
            assert bool(ax.get_legend()) == (ax.get_title() == 'lr'), str(ax.get_title())
        # https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.legend.html
        assert ax.get_legend()._loc in (10, 'center')


class TestExperimentPlot:

    def setup_method(self, method):
        sys.stdout.write("\n")

    @staticmethod
    def _fixture() -> Experiment:
        # Note different column names (some are shared, some are not)
        h0 = Hypothesis("hyp0", Run('r0', pd.DataFrame(
            {"a": [1, 2, 3], "b0": [10, 9, 8]})))
        h1 = Hypothesis("hyp1", Run('r1', pd.DataFrame(
            {"a": [4, 5, 6], "b1": [7, 6, 5]})))
        ex = Experiment(name="ex", hypotheses=[h0, h1])
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

    # we can reuse the same test suite,
    # replacing `hypothesis` with `ex` (backed by self._fixture())
    # TODO: Use inheritance.
    testWhenFigAxesAreGiven = TestHypothesisPlot.testWhenFigAxesAreGiven
    testSuptitle = TestHypothesisPlot.testSuptitle

    def testExMultiHypothesisLegend(self):
        ex = self._fixture()

        # default behavior: a legend on the first subplot (assume >1 subplots)
        g = ex.plot()
        assert g.figure.legends == []
        for k, ax in enumerate(g.axes_active):
            assert bool(ax.get_legend()) == bool(k == 0)
            if ax.get_legend():
                assert len(ax.get_legend().texts) == len(ex.hypotheses)

        # boolean (True/False)
        g = ex.plot(legend=False)
        for ax in g.axes_active: assert ax.get_legend() is None
        g = ex.plot(legend=True)
        for ax in g.axes_active:
            assert len(ax.get_legend().texts) == len(ex.hypotheses)  # 2

        # int/str
        g = ex.plot(legend='a')
        for ax in g.axes_active:
            assert bool(ax.get_legend()) == (ax.get_title() == 'a'), str(ax.get_title())

        # dict
        g = ex.plot(legend=dict(ax=None))  # on the figure:
        assert g.figure.legends
        for ax in g.axes_active: assert ax.get_legend() is None
        # Do we have labels for all of 2 hypotheses?
        assert len(g.figure.legends[0].texts) == len(ex.hypotheses)

        # For single y ...
        g = ex.plot(y="a", legend=False)
        assert g.figure.legends == []
        for ax in g.axes_active: assert ax.get_legend() is None
        # subplots=False cases
        g = ex.plot(y=["a", "b0"], subplots=False, legend=False)
        assert g.figure.legends == []
        for ax in g.axes_active: assert ax.get_legend() is None
