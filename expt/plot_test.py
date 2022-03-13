"""Tests for expt.plot"""
import contextlib
import sys
from typing import cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import expt.colors
import expt.data
import expt.plot
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


@contextlib.contextmanager
def matplotlib_rcparams(kwargs: dict):
  old_values = {k: matplotlib.rcParams[k] for k in kwargs.keys()}
  try:
    for k in kwargs.keys():
      matplotlib.rcParams[k] = kwargs[k]
    yield
  finally:
    for k in kwargs.keys():
      matplotlib.rcParams[k] = old_values[k]


# -----------------------------------------------------------------------------
# Fixtures


@pytest.fixture
def hypothesis() -> Hypothesis:
  """Make a sample Hypothesis."""

  def _make_data(max_accuracy=1.0):
    data = dict()
    data['step'] = np.arange(10, 10000 + 10, 10)
    data['loss'] = np.exp((-data['step'] + 700) / 1000.0)
    data['loss'] += np.random.normal(0, 0.1, size=1000)
    data['accuracy'] = (data['step'] / data['step'].max()) * max_accuracy
    data['lr'] = np.ones(1000) * 0.001
    return data

  runs = [
      Run("seed0", pd.DataFrame(_make_data(max_accuracy=0.9))),
      Run("seed1", pd.DataFrame(_make_data(max_accuracy=1.0))),
  ]
  h = Hypothesis(name="hypo_plot", runs=runs)
  return h


@pytest.fixture(name="ex")
def experiment() -> Experiment:
  """Make a sample experiment."""

  # Note different column names (some are shared, some are not)
  h0 = Hypothesis("hyp0",
                  Run('r0', pd.DataFrame({
                      "a": [1, 2, 3],
                      "b0": [10, 9, 8]
                  })))
  h1 = Hypothesis("hyp1",
                  Run('r1', pd.DataFrame({
                      "a": [4, 5, 6],
                      "b1": [7, 6, 5]
                  })))
  ex = Experiment(name="ex", hypotheses=[h0, h1])
  return ex


# -----------------------------------------------------------------------------
# Tests


class TestGridPlot:

  def test_layout(self):
    g = expt.plot.GridPlot(y_names=["single"])
    assert g.axes.shape == (1, 1)

    g = expt.plot.GridPlot(y_names=[str(i) for i in range(16)])
    assert g.axes.shape == (4, 4)

    g = expt.plot.GridPlot(y_names=[str(i) for i in range(28)])
    assert g.axes.shape == (5, 6)  # 5 rows, 6 columns = 30

    g = expt.plot.GridPlot(y_names=["a", "b", "c", "d", "e"], layout=(1, -1))
    assert g.axes.shape == (1, 5)

    g = expt.plot.GridPlot(y_names=["a", "b", "c", "d", "e"], layout=(-1, 1))
    assert g.axes.shape == (5, 1)

    g = expt.plot.GridPlot(y_names=["a", "b", "c", "d", "e"], layout=(2, -1))
    assert g.axes.shape == (2, 3)  # 2 * 3 = 6 > 5


class TestHypothesisPlot:

  def setup_method(self, method):
    sys.stdout.write("\n")

  def teardown_method(self, method):
    plt.close("all")

  # TODO: This test suite is incomplete. Add more tests.

  def test_grid_spec(self, hypothesis: Hypothesis):
    # single y
    g = hypothesis.plot(y="loss")
    assert V(g.axes.shape) == (1, 1)
    assert V(g.axes_active.shape) == (1,)

    # two y's   [2x1]
    g = hypothesis.plot(y=["loss", "accuracy"])
    assert V(g.axes.shape) == (1, 2)

    # three y's [2x2]
    g = hypothesis.plot(y=["loss", "accuracy", "lr"])
    assert V(g.axes.shape) == (2, 2)
    assert V(g.axes_active.shape) == (3,)

  def test_when_fig_axes_are_given(self, hypothesis: Hypothesis):
    # single y given a single axesplot
    fig, ax = plt.subplots()
    g = hypothesis.plot(y="loss", ax=ax)
    assert g.axes.shape == (1, 1)
    assert g.axes_active[0] is ax  # we need to reuse the given ax
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
      # assert g.axes.shape == (1, 2)
    assert "the rank should be 2 (but given 1)" in str(ex.value)

    # two y's, given 2D array of axesplots
    fig, axes = plt.subplots(2, 1, squeeze=False)
    g = hypothesis.plot(y=["loss", "accuracy"], ax=axes)
    assert g.axes.shape == (2, 1)
    assert g.axes_active[0] is axes.flat[0]
    assert g.axes_active[1] is axes.flat[1]

  def test_suptitle(self, hypothesis: Hypothesis):
    from matplotlib.text import Text

    def _ensureText(t) -> Text:
      assert isinstance(t, Text)
      return cast(Text, t)

    # when given, it should be set
    g = hypothesis.plot(suptitle="super!")
    assert _ensureText(g.fig._suptitle).get_text() == "super!"

    # default behavior: hypothesis name?
    g = hypothesis.plot()
    assert _ensureText(g.fig._suptitle).get_text() == hypothesis.name

    # default behavior: if ax or grid is given, do not set suptitle
    fig, ax = plt.subplots()
    g = hypothesis.plot(ax=ax, y='loss')
    assert g.fig._suptitle is None or g.fig._suptitle.get_text() == ""

    g = expt.plot.GridPlot(y_names=['loss'])
    g = hypothesis.plot(grid=g, y='loss')
    assert g.fig._suptitle is None or g.fig._suptitle.get_text() == ""

  def test_single_hypothesis_legend(self, hypothesis: Hypothesis):
    # default behavior: no legend
    g = hypothesis.plot()
    assert g.figure.legends == []
    for ax in g.axes_active:
      assert ax.get_legend() is None

    # boolean (True/False)
    g = hypothesis.plot(legend=False)
    for ax in g.axes_active:
      assert ax.get_legend() is None
    g = hypothesis.plot(legend=True)
    for ax in g.axes_active:
      assert len(ax.get_legend().texts) == 1

    # int/str
    g = hypothesis.plot(legend='loss')
    for ax in g.axes_active:
      assert bool(ax.get_legend()) == (ax.get_title() == 'loss'), str(
          ax.get_title())

    # dict
    g = hypothesis.plot(legend=dict(ax=-1, loc='center'))  # the last one: lr
    for ax in g.axes_active:
      assert bool(ax.get_legend()) == (ax.get_title() == 'lr'), str(
          ax.get_title())
    # https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.legend.html
    assert ax.get_legend()._loc in (10, 'center')  # type: ignore

    # Custom override with labels=... argument
    g = hypothesis.plot()
    g.add_legend(ax=0, labels=["custom label"])
    leg = g.axes_active[0].get_legend()
    assert len(leg.texts) == 1
    assert leg.texts[0]._text == 'custom label'

    with pytest.raises(ValueError, match='should have length'):
      g.add_legend(ax=0, labels=["custom label", "too many args"])

  def test_error_range_averaging(self, hypothesis: Hypothesis):
    # show individual runs
    g = hypothesis.plot(x='step', y=['loss', 'accuracy'], rolling=10)
    assert len(g['accuracy'].get_lines()) == 1 + len(hypothesis.runs)
    for line in g['accuracy'].get_lines():
      assert line.get_xdata().max() >= 9900

    # fill 1-std range.
    # TODO: validate color, alpha, etc.
    g = hypothesis.plot(
        x='step', y=['loss', 'accuracy'], err_style='fill', rolling=10)
    assert g['accuracy'].collections, "no filled area found"
    for fill in g['accuracy'].collections:
      assert fill.get_paths()[0].vertices.max() >= 9900

    # TODO: what if some runs are truncated?

  def test_error_range_custom_fn(self, hypothesis: Hypothesis):
    """Tests plot(err_fn=...)"""

    def err_fn(h: Hypothesis) -> pd.DataFrame:
      return cast(pd.DataFrame, h.grouped.std()).applymap(lambda x: 5000)

    # without interpolation
    g = hypothesis.plot(x='step', y='loss', err_style='fill', err_fn=err_fn)
    assert g['loss'].collections[0].get_paths()[0].vertices[-1][1] >= 5000

    # with interpolation
    g = hypothesis.plot(
        x='step', y='loss', err_style='fill', n_samples=100,
        err_fn=err_fn)  # Note: err_style='fill'
    assert g['loss'].collections[0].get_paths()[0].vertices[-1][1] >= 5000

    g = hypothesis.plot(
        x='step', y='loss', err_style='runs', n_samples=100,
        err_fn=err_fn)  # Note: with err_style='runs', err_fn is not useful..?


class TestExperimentPlot:

  def setup_method(self, method):
    sys.stdout.write("\n")

  def test_gridplot_basic(self, ex: Experiment):
    # TODO: Add more complex scenario.

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

    with pytest.raises(ValueError) as e:
      g['b']
    assert 'Unknown index: b' in V(str(e.value))
    assert "['b1', 'b0']" in str(e.value) or \
           "['b0', 'b1']" in str(e.value)

    # __getitem__ (int)
    assert g[0] is g.axes_active[0]
    assert g[1] is g.axes_active[1]
    assert g[-1] is g.axes_active[2]  # we have 3 active axes

  # we can reuse the same test suite,
  # replacing `hypothesis` with `ex` (backed by self._fixture())
  # TODO: Use inheritance and avoid wrong types.
  def test_when_fig_axes_are_given(self, ex: Experiment):
    TestHypothesisPlot.test_when_fig_axes_are_given(self, ex)  # type: ignore

  def test_suptitle(self, ex: Experiment):
    TestHypothesisPlot.test_suptitle(self, ex)  # type: ignore

  def test_multi_hypothesis_legend(self, ex: Experiment):
    # default behavior: a legend on the first subplot (assume >1 subplots)
    g = ex.plot()
    assert g.figure.legends == []
    for k, ax in enumerate(g.axes_active):
      assert bool(ax.get_legend()) == bool(k == 0)
      if ax.get_legend():
        assert len(ax.get_legend().texts) == len(ex.hypotheses)

    # boolean (True/False)
    g = ex.plot(legend=False)
    for ax in g.axes_active:
      assert ax.get_legend() is None
    g = ex.plot(legend=True)
    for ax in g.axes_active:
      assert len(ax.get_legend().texts) == len(ex.hypotheses)  # 2

    # int/str
    g = ex.plot(legend='a')
    for ax in g.axes_active:
      assert bool(ax.get_legend()) == (ax.get_title() == 'a'), str(
          ax.get_title())

    # dict
    g = ex.plot(legend=dict(ax=None))  # on the figure:
    assert g.figure.legends
    for ax in g.axes_active:
      assert ax.get_legend() is None
    # Do we have labels for all of 2 hypotheses?
    assert len(g.figure.legends[0].texts) == len(ex.hypotheses)

    # For single y ...
    g = ex.plot(y="a", legend=False)
    assert g.figure.legends == []
    for ax in g.axes_active:
      assert ax.get_legend() is None
    # subplots=False cases
    g = ex.plot(y=["a", "b0"], subplots=False, legend=False)
    assert g.figure.legends == []
    for ax in g.axes_active:
      assert ax.get_legend() is None

  def test_color_kwargs(self, ex: Experiment):
    import cycler
    assert len(ex.hypotheses) == 2

    # Given kwargs
    with pytest.raises(ValueError, match=r'should have the same number'):
      g = ex.plot(colors=["red"])

    g = ex.plot(colors=["magenta", "blue"])
    for ax in g.axes_active:
      assert ax.get_lines()[0].get_color() == 'magenta'
      assert ax.get_lines()[1].get_color() == 'blue'

    # By default, should respect matplotlib's config
    color = cycler.cycler(color=['green', 'cyan'])
    with matplotlib_rcparams({'axes.prop_cycle': color}):
      g = ex.plot()
      for ax in g.axes_active:
        assert ax.get_lines()[0].get_color() == 'green'
        assert ax.get_lines()[1].get_color() == 'cyan'

    # if color is missing, we fall back to a default color palette
    axprop = cycler.cycler(linestyle=['-', '--'])
    default_colors = expt.colors.DefaultColors
    with matplotlib_rcparams({'axes.prop_cycle': axprop}):
      g = ex.plot()
      for ax in g.axes_active:
        assert ax.get_lines()[0].get_color() != ax.get_lines()[1].get_color()
        assert ax.get_lines()[0].get_color() == default_colors[0]
        assert ax.get_lines()[1].get_color() == default_colors[1]
