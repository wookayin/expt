"""
expt.widgets -- Interactive widgets for jupyterlab.

This is still a working-in-progress and experimental implementation,
and things might be flaky or unstable.
"""

from typing import Dict, Any
import sys
import inspect
import functools
import asyncio

from cached_property import cached_property
from typeguard import typechecked
from IPython.display import display as ipython_display
from IPython.display import clear_output
import ipywidgets as widgets
import matplotlib.pyplot as plt

from expt.data import Run, RunList, Hypothesis, Experiment

try:
    import nest_asyncio; nest_asyncio.apply()
except ImportError as ex:
    raise ImportError(
        "To make asynchronous events work properly on jupyter, "
        "We require nest_asyncio. Please install `nest_asyncio` manually."
    ) from ex



def Collapsible(widget, title="", is_open=True) -> widgets.Widget:
    """Create a Collapsible widget."""
    ac = widgets.Accordion([widget])
    if not is_open:
        ac.selected_index = None
    ac.set_title(0, title)
    return ac


def callback_syncwrap(fn):
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
    def __init__(self, runs: RunList,
                 description: str = "", *,
                 display=True,
                 groupby=None,
                 ):
        self._runs: RunList = runs
        self._description = description or ""
        self._layout: widgets.Widget = self.create_layout()
        if display:
            ipython_display(self._layout)

        self.groupby = groupby
        self._summary_kwargs = dict()
        self._plot_kwargs = dict()

    def create_layout(self) -> widgets.Widget:
        """Create a layout for the dashboard.
        """
        self._multiselect = widgets.SelectMultiple(
            options={
                f"[{i}] {run.path} ({len(run.df)} rows)": run.path  # label: value
                for i, run in enumerate(self._runs) },
            rows=min(24, len(self._runs)),
            description="Select to plot: ",
            layout=dict(width='initial'),
        )
        self._multiselect.observe(lambda change: self.invalidate(), 'value')

        # control panel
        self._button_refresh = widgets.Button(description="Refresh")
        self._button_refresh.on_click(callback_syncwrap(self.refresh))
        self._control_panel = widgets.HBox([
            self._button_refresh,
        ])

        # output views
        self._out_summary = widgets.Output()
        self._out_plot = widgets.Output()

        return widgets.VBox([
            self._multiselect,
            self._control_panel,
            Collapsible(self._out_summary, title="Summary"),
            Collapsible(self._out_plot, title="Plot"),
        ])

    def _ipython_display_(self):
        return self._layout._ipython_display_()

    @property
    def description(self) -> str:
        return self._description

    @cached_property
    def ex(self) -> Experiment:
        """A cached property based on the current selection/filter status."""
        # apply filters if necessary
        runs = self._runs
        if self._multiselect.value:
            runs = runs.filter(lambda r: r.path in self._multiselect.value)

        sys.stderr.write(f"Creating a new Experiment from {len(runs)} runs.\n")
        ex = Experiment(name=self.description)
        groupby_fn = self.groupby or (lambda run: run.name)

        for _, hypo in runs.groupby(groupby_fn):
            ex.add_hypothesis(hypo)
        return ex

    def invalidate(self):
        """Invalidates all cache."""
        if 'ex' in self.__dict__:   # TODO thread lock?
            del self.__dict__['ex']

    def summary(self, *args, **kwargs):
        if args:
            raise RuntimeError("No positional arguments allowed.")
        self._summary_kwargs = kwargs.copy()

    def plot(self, *args, **kwargs):
        if args:
            raise RuntimeError("No positional arguments allowed.")
        self._plot_kwargs = kwargs.copy()

    def draw_summary(self):
        """Describe the Experiment."""
        with self._out_summary:
            clear_output()
            df = self.ex.summary(**self._summary_kwargs)
            self._summary = df
            ipython_display(df.style.background_gradient(cmap='viridis'))
        return self._summary

    def draw_plot(self, *args, **kwargs):
        """Plot the current Experiment."""
        with self._out_plot:
            sys.stderr.write("Drawing plots ...\n")
            self._plot = self.ex.plot(**self._plot_kwargs)
            clear_output()
            plt.show()
        return self._plot

    async def refresh(self):
        self.draw_summary()
        self.draw_plot()
