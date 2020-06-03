"""expt.widgets: Interactive widgets for jupyterlab.

This is still a working-in-progress and experimental implementation,
and things might be unstable or changing rapidly.
"""

import asyncio
import functools
import inspect
import sys
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

import ipywidgets as widgets
import matplotlib.pyplot as plt
from cached_property import cached_property
from IPython.display import clear_output
from IPython.display import display as ipython_display
from typeguard import typechecked

import expt
import expt.data_loader
from expt.data import Experiment, Hypothesis, Run, RunList

try:
  import nest_asyncio
  nest_asyncio.apply()
except ImportError as ex:
  raise ImportError(
      "To make asynchronous events work properly on jupyter, "
      "We require nest_asyncio. Please install `nest_asyncio` manually."
  ) from ex


def Collapsible(widget: widgets.Widget,
                title: str = "",
                is_open: bool = True) -> widgets.Widget:
  """Create a Collapsible widget."""
  ac = widgets.Accordion([widget])  # type: ignore
  ac.selected_index = 0 if is_open else None
  ac.set_title(0, title)
  return ac


def callback_syncwrap(fn: Callable[..., Awaitable]):
  """A callback function for Widget, e.g. Button.on_click(), that wraps
    an *asynchronous function* to run synchronously in the event loop."""

  def _f(this: widgets.Widget):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(fn())

  return _f


class WidgetBase:
  pass


class RunsPanel(WidgetBase):
  """An experimental RunsPanel for jupyter."""

  @typechecked
  def __init__(
      self,
      paths: Sequence[str],
      description: str = "",
      *,
      run_postprocess_fn: Optional[Callable[[Run], Run]] = None,
      load_data: bool = True,
      display: bool = False,
  ):
    """An interactive widget for a typical expt routine.

    Args:
      paths: paths to load by get_runs.
      description: Description or title for expt.Experiment.
      load_data: If True, an async coroutine task will be launched to
        load the data once as soon as the widget is created.
      display: If True (defaults False), the widget will be displayed
        as soon as the instance is created while the cell execution hasn't
        been completed yet.
      run_postprocess_fn: post-processing of Run objects when loading data.
    """

    self._run_loader = expt.data_loader.RunLoader(
        *paths, run_postprocess_fn=run_postprocess_fn)
    self._runs: RunList = RunList([])
    self._description = description or ""
    self._layout: widgets.Widget = self.create_layout()
    if display:
      ipython_display(self)

    self._ex_factory = None
    self._summary_fn = Experiment.summary
    self._plot_fn = Experiment.plot

    if load_data:
      asyncio.ensure_future(self.reload_data())

  def create_layout(self) -> widgets.Widget:
    """Create a layout for the dashboard.

    Selectbox (_multiselect)
    Button (_refresh)    ||  Quick statusline (_statusline)
    + Data Loading
    + Summary
    + Plot
    """
    self._multiselect = widgets.SelectMultiple(
        options={},
        rows=1,
        description="",
        layout=dict(width='initial'),
    )
    self._multiselect.observe(lambda change: self.invalidate(), 'value')

    # control panel
    self._button_refresh = widgets.Button(description="Refresh")
    self._button_refresh.on_click(callback_syncwrap(self.refresh))
    self._statusline = widgets.Output(
        layout=dict(width='100%', height='auto', overflow='hidden'))
    self._control_panel = widgets.HBox([
        self._button_refresh,
        self._statusline,
    ])

    # output views
    self._out_data = widgets.Output()
    self._out_summary = widgets.Output()
    self._out_plot = widgets.Output()

    return widgets.VBox([
        self._multiselect,
        self._control_panel,
        Collapsible(self._out_data, title="Data Loading"),
        Collapsible(self._out_summary, title="Summary (Table)"),
        Collapsible(self._out_plot, title="Plot"),
    ])

  def _repr_mimebundle_(self, *args, **kwargs):  # IPython >= 6.1
    return self._layout._repr_mimebundle_(*args, **kwargs)  # type: ignore

  @property
  def description(self) -> str:
    return self._description

  @property
  def ex_factory(self):
    return self._ex_factory or self._ex_factory_default

  @ex_factory.setter
  def ex_factory(self, ex_factory):
    self._ex_factory = ex_factory  # TODO validate types

  def _ex_factory_default(self, runs: RunList) -> Experiment:
    # How to construct ex? We use the factory default one,
    # using RunList.to_dataframe() and Experiment.from_dataframe().
    df = runs.to_dataframe(as_hypothesis=True)
    ex = Experiment.from_dataframe(df, name=self.description)
    ex.runs_df = df  # type: ignore
    return ex

  @cached_property
  def ex(self) -> Experiment:
    """A cached property based on the current selection/filter status."""
    # apply filters if necessary
    runs = self._runs
    if runs is None:
      raise RuntimeError("Runs have never been loaded.")

    if self._multiselect.value:
      runs = runs.filter(lambda r: r.path in self._multiselect.value)

    print(f"Creating a new Experiment from {len(runs)} runs.\n")

    ex = self.ex_factory(runs)
    if not isinstance(ex, Experiment):
      raise TypeError(f"{self._ex_factory} did not return an expt.Experiment")
    return ex

  def invalidate(self):
    """Invalidates all cache (self.ex)."""
    if 'ex' in self.__dict__:  # TODO thread lock?
      del self.__dict__['ex']

  @property
  def summary_fn(self):
    return self._summary_fn

  @summary_fn.setter
  def summary_fn(self, summary_fn):
    self._summary_fn = summary_fn  # TODO validate type

  def summary(self, *args, **kwargs):
    self._summary_fn = lambda ex: ex.summary(*args, **kwargs)

  @property
  def plot_fn(self):
    return self._plot_fn

  @plot_fn.setter
  def plot_fn(self, plot_fn):
    self._plot_fn = plot_fn  # TODO validate type

  def plot(self, *args, **kwargs):
    self._plot_fn = lambda ex: ex.plot(*args, **kwargs)

  async def reload_data(self):
    # TODO: preserve selection. Draw without data fetching.

    with self._out_data:
      clear_output()
      self.log("Reloading run data ...")
      print("Time (reload_data):", datetime.utcnow().astimezone())
      self._runs = await self._run_loader.get_runs_async()
      self.log(" Done!", append=True)

      self._multiselect.options = {
          f"[{i}] {run.path} ({len(run.df)} rows)": run.path  # label: value
          for i, run in enumerate(self._runs)
      }
      self._multiselect.rows = min(24, len(self._runs) + 1)

  def draw_summary(self):
    """Describe the Experiment."""
    with self._out_summary:
      clear_output()
      self.log("Generating tabular summary ...")
      if hasattr(self.ex, 'runs_df'):
        ipython_display(self.ex.runs_df)  # type: ignore

      df = self._summary_fn(self.ex)  # type: ignore
      self._summary = df
      ipython_display(df.style.background_gradient(cmap='viridis'))
    return self._summary

  def draw_plot(self, *args, **kwargs):
    """Plot the current Experiment."""
    with self._out_plot:
      clear_output()
      self.log(f"Drawing plots with {len(self._multiselect.value)} runs ...")
      self._plot = self._plot_fn(self.ex)
      self.log(" Done!", append=True)
      plt.show()
    return self._plot

  def log(self, msg, append=False):
    with self._statusline:
      if not append:
        clear_output(wait=True)
      print(msg, end='', flush=True)

  async def refresh(self):
    await self.reload_data()
    self.draw_summary()
    self.draw_plot()
