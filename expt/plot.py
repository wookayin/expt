"""Plotting behavior (matplotlib, hvplot, etc.) for expt.data"""

from __future__ import annotations

import difflib
import itertools
from typing import (Any, Callable, cast, Dict, Iterable, List, Optional,
                    overload, Sequence, Tuple, Union)
import warnings

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.legend
import matplotlib.ticker
import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy

from expt import util
from expt.data import Experiment
from expt.data import Hypothesis

# yapf: disable
warnings.filterwarnings("ignore", category=UserWarning,
                        message='Creating legend with loc="best"')
# yapf: enable

HypothesisSummaryFn = Callable[  # see HypothesisPlotter
    [Hypothesis], pd.DataFrame]
HypothesisSummaryErrFn = Callable[  # see HypothesisPlotter
    [Hypothesis], Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]]


class GridPlot:
  """Multi-plot grid subplots.

  This class provides more object-oriented methods to manipulate matplotlib
  grid subplots (even after creation). For example, add_legend()."""

  def __init__(
      self,
      *,
      fig: Optional[Figure] = None,
      axes: Optional[Union[np.ndarray, Axes]] = None,
      y_names: List[str],
      layout: Optional[Tuple[int, int]] = None,
      figsize: Optional[Tuple[int, int]] = None,
  ):
    """Initialize the matplotlib figure and GridPlot object.

    Args:
    """

    if isinstance(y_names, str):
      raise TypeError("`y_names` must be a List[str], "
                      "but given {}".format(type(y_names)))
    y_names = list(y_names)
    if not y_names:
      raise ValueError("y_names should be a non-empty array.")
    self.n_plots = len(y_names)
    self._y_names = y_names

    # Compute the grid shape
    if layout is None:
      rows = int((self.n_plots + 1)**0.5)
      cols = int(np.ceil(self.n_plots / rows))
      layout = (rows, cols)
    else:
      rows, cols = layout
      if rows == -1 and cols == -1:
        raise ValueError("Invalid layout: {}".format(layout))
      elif rows < 0:
        rows = int(np.ceil(self.n_plots / cols))
      elif cols < 0:
        cols = int(np.ceil(self.n_plots / rows))

    # Calculate the base figure size
    if figsize is None:
      # default, each axes subplot has some reasonable size
      # TODO: Handle cols = -1 or rows = -1 case
      figsize = (cols * 4, rows * 3)

    # Initialize the subplot grid
    if fig is None and axes is None:
      subplots_kwargs: Dict[str, Any] = dict(
          figsize=figsize,
          squeeze=False,
      )
      import matplotlib.pyplot as plt  # lazy load
      fig, axes = plt.subplots(rows, cols, **subplots_kwargs)
    elif fig is None and axes is not None:
      if isinstance(axes, np.ndarray):
        if len(axes.shape) != 2:
          raise ValueError("When axes is a ndarray of Axes, the rank should be "
                           "2 (but given {})".format(len(axes.shape)))
      else:
        # ensure axes is a 2D array of Axes
        axes = np.asarray(axes, dtype=object).reshape([1, 1])
      fig = axes.flat[0].get_figure()
    else:
      raise ValueError("If fig is given, axes should be given as well")

    self._fig = cast(Figure, fig)
    self._axes = cast(np.ndarray, axes)

    for yi, ax in zip(self._y_names, self.axes_active):
      ax.set_title(yi)
    for yi, ax in zip(self._y_names, self.axes_inactive):
      ax.axis('off')

    if len(self._axes.flat) < len(y_names):
      raise ValueError("Grid size is %d * %d = %d, smaller than len(y) = %d" %
                       (rows, cols, len(self._axes.flat), len(y_names)))

  @property
  def fig(self) -> Figure:
    return self._fig

  @property
  def figure(self) -> Figure:
    return self._fig

  def _ipython_display_(self):
    from IPython.display import display
    return display(self.figure)

  @property
  def axes(self) -> np.ndarray:  # array[Axes]
    """Return a grid of subplots (always a 2D ndarray)."""
    return self._axes

  @property
  def axes_active(self) -> np.ndarray:  # array[Axes]
    """Return a flattened ndarray of active subplots (excluding that are
    turned off), whose length equals `self.n_plots`."""
    return self.axes.flat[:self.n_plots]

  @property
  def axes_inactive(self) -> np.ndarray:
    """Return a flattened ndarray of inactive subplots, whose length
    equals `prod(self.axes.shape) - self.n_plots`."""
    return self.axes.flat[self.n_plots:]

  # yapf: disable
  @overload
  def __getitem__(self, key: str) -> Axes: ...
  @overload
  def __getitem__(self, key: int) -> Axes: ...
  # yapf: enable

  def __getitem__(self, key) -> Union[Axes, np.ndarray]:
    # Note: see add_legend(ax) as well
    if isinstance(key, str):
      # find axes by name
      try:
        index = self._y_names.index(key)
      except (ValueError, IndexError) as e:
        raise ValueError(
            "Unknown index: {}. Close matches: {}".format(
                key, difflib.get_close_matches(key, self._y_names))
        ) from e  # yapf: disable

      # TODO: support fancy indeing (multiple keys)
      return self.axes_active[index]
    elif isinstance(key, int):
      # find axes by index
      index = key
      return self.axes_active[index]
    else:
      raise TypeError("Unsupported index : {}".format(type(key)))

  def set(self, **kwargs):
    """Set attributes on each subplot Axes."""
    for ax in self.axes.flat:
      ax.set(**kwargs)
    return self

  def savefig(self, *args, **kwargs):
    """Save the figure."""
    kwargs = kwargs.copy()
    kwargs.setdefault("bbox_inches", "tight")
    self.fig.savefig(*args, **kwargs)

  def clear_legends(self):
    """Clear all legends on the figure itself and all the subplot axes."""
    self.fig.legends[:] = []
    for ax in self.axes.flat:
      legend = ax.get_legend()
      if legend:
        legend.remove()
    return self

  def add_legend(self,
                 ax: Optional[Union[Axes, str, int]] = None,
                 loc=None,
                 labels: Optional[Sequence[str]] = None,
                 **kwargs):
    """Put a legend for all objects aggregated over the axes.

    If ax is given, a legend will be put at the specified Axes. If it
    is str or int (say `k`), `self[k]` will be used instead. Otherwise
    (i.e., ax=None), the legend will be put on figure-level (if there are
    more than two Axes), or the unique axis (if there is only one Axes).

    Args:
      loc: str or pair of floats (see matplotlib.pyplot.legend)
      labels: If given, this will override label names.
      kwargs: Additional kwargs passed to ax.legend().
          e.g., ncol=4

    Returns:
      The matplotlib legend object added.
    """
    target: Union[Figure, Axes]
    if ax is None:
      # automatic: figure legend or the (unique) axes
      if self.n_plots >= 2:
        target = self.fig
      else:
        target = self.axes_active[0]
    else:
      if isinstance(ax, (int, str)):  # see __getitem__
        target = self[ax]  # type: ignore
      else:
        target = ax

    # TODO: Customize how to sort legend items.
    legend_handles, legend_labels = zip(
        *[(h, label) for (label, h) in sorted(self._collect_legend().items())])
    if labels is not None:
      if len(labels) != len(legend_labels):
        raise ValueError(
            f"labels {labels} should have length {len(legend_labels)} "
            f"but was given {len(labels)}")
      legend_labels = list(labels)
    legend = target.legend(legend_handles, legend_labels, loc=loc, **kwargs)

    if isinstance(target, Axes) and not target.lines:
      target.axis('off')

    return legend

  def _collect_legend(self) -> Dict[str, Any]:
    """Collect all legend labels from the subplot axes."""
    legend_labels = dict()
    for ax in self.axes_active:
      for handle, label in zip(*ax.get_legend_handles_labels()):  # type: ignore
        if label in legend_labels:
          # TODO: Add warning/exception if color conflict is found
          pass
        else:
          legend_labels[label] = handle
    return legend_labels


class HypothesisPlotter:
  """Implements expt.data.Hypothesis.plot()."""

  def __init__(self, hypothesis: Hypothesis):  # type: ignore
    self._parent = hypothesis

  def __repr__(self):
    return "<HypothesisPlotter<{}>".format(self._parent)

  @property
  def name(self):
    return self._parent.name

  @property
  def runs(self):
    return self._parent.runs

  @property
  def grouped(self) -> DataFrameGroupBy:
    return self._parent.grouped

  @property
  def _dataframes(self) -> List[pd.DataFrame]:
    return self._parent._dataframes

  KNOWN_ERR_STYLES = (None, False, 'band', 'fill', 'runs', 'unit_traces')

  def __call__(self,
               *args,
               subplots=True,
               err_style="runs",
               err_fn: Optional[HypothesisSummaryErrFn] = None,
               representative_fn: Optional[HypothesisSummaryFn] = None,
               std_alpha=0.2,
               runs_alpha=0.2,
               n_samples=None,
               rolling: Union[int, dict, None] = None,
               ignore_unknown: bool = False,
               legend: Union[bool, int, str, Dict[str, Any]] = False,
               prettify_labels: bool = False,
               suptitle: Optional[str] = None,
               grid: Optional[GridPlot] = None,
               ax: Optional[Union[Axes, np.ndarray]] = None,
               tight_layout: Union[bool, Dict[str, Any]] = True,
               rasterized: bool = False,
               **kwargs) -> GridPlot:
    '''Hypothesis.plot based on matplotlib.

    This can work in two different modes:
    (1) Plot (several or all) columns as separate subplots (subplots=True)
    (2) Plot (several or all) columns in a single axesplot (subplots=False)

    Additional keyword arguments:
      - rolling (int, dict, or None): If given (i.e. not None), use a rolling
        window for smoothing. By default the rolling window is centered. The
        integer value can represent the window size for rolling and smoothing.
        If it's a dict, it will be kwargs to the DataFrame.rolling();
        e.g. rolling = {"center": True, "window": 10}
      - n_samples (int): If given, we subsample using n_samples number of
          equidistant points over the x axis. Values will be interpolated.
      - legend (bool, int, str, or dict):
          If True, a legend will be added to each of the subplots.
          Default is False (no legend). If a dict is given, it will be
          passed as kwargs to GridPlot.add_legend(). Please see the
          documentation of `GridPlot.add_legend`.
          If int or str is given, same meaning as `dict(ax=...)`.
      - prettify_labels (bool): If True, apply a sensible default
          prettifier to the legend labels, truncating long names.
          Default is False.
      - suptitle (str): suptitle for the figure. Defaults to the name
          of the hypothesis. Use empty string("") to disable suptitle.
      - err_style (str): How to show individual runs (traces) or
          confidence interval as shaded area.
          Possible values: (None, 'runs', 'unit_traces', 'band', 'fill')
            (i) runs, unit_traces: Show individual runs/traces (see runs_alpha)
            (ii) band, fill: Show as shaded area (see std_alpha)
            (iii) None or False: do not display any errors
      - err_fn (Callable: Hypothesis -> pd.DataFrame | Tuple):
          A strategy to compute the error range when err_style is band or fill.
          Defaults to "standard deviation.", i.e. `hypothosis.grouped.std()`.
          This function may return either:
            (i) a single DataFrame, representing the standard error,
              which must have the same column and index as the hypothesis; or
            (ii) a tuple of two DataFrames, representing the error range
              (lower, upper). Both DataFrames must also have the same
              column and index as the hypothesis.
          In the case of (i), we assume that a custom `representative_fn` is
          NOT being used, but the representative value of the hypothesis is
          the grouped mean of the Hypothesis, i.e., `hypothesis.mean()`.
          (Example) To use standard error for the bands, you can use either
          `err_fn=lambda h: h.grouped.sem()` or
          `err_fn=lambda h: (h.grouped.mean() - h.grouped.sem(),
                             h.grouped.mean() + h.grouped.sem())`.
      - representative_fn (Callable: Hypothesis -> pd.DataFrame):
          A strategy to compute the representative value (usually drawn
          in a thicker line) when plotting.
          This function should return a DataFrame that has the same column
          and index as the hypothesis.
          Defaults to "sample mean.", i.e., `hypothesis.mean()`
          For instance, to use median instead of mean, use
          `representative_fn=lambda h: h.grouped.median()`
      - std_alpha (float): If not None, will show the error band as a
          shaded area. Defaults 0.2,
      - runs_alpha (float): If not None, will draw an individual line
          for each run. Defaults 0.2.
      - ax (Axes or ndarray of Axes): Subplot axes to draw plots on.
          Note that this is mutually exclusive with `grid`.
      - grid (GridPlot): A `GridPlot` instance (optional) to use for
          matplotlib figure and axes. If this is given, `ax` must be None.
      - tight_layout (bool | dict): Applies to Figure. see fig.tight_layout()
      - rasterized (bool): Applies to Figure. If true, rasterize vector
          graphics into raster images, when drawing the plot to produce smaller
          file size when the number of data points is large (e.g., >= 10K),
          which could be loaded much faster when saved as a PDF image (savefig).
          Highly recommend to use fig.set_dpi(...) for a good image quality.

    All other kwargs is passed to DataFrame.plot(). For example, you may
    find the following parameters useful:
        `x`, `y`, `layout`, `figsize`, `ax`, `legend`, etc.

    Returns:
        A `GridPlot` instance. If you need to access subplot fig and axes,
        use `g.fig` and `g.axes`, etc.
    '''
    if len(self.runs) == 0:
      raise ValueError("No data to plot, hypothesis.run is empty.")

    if self._parent.empty():
      # nothing to draw (no rows)
      raise ValueError("No data to plot, all runs have empty DataFrame.")

    #
    ## STEP 1. Prepare data (mean ± std)
    #

    def _representative_and_err(h: Hypothesis) -> Tuple[
        pd.DataFrame,  # representative (mean)
        Tuple[pd.DataFrame, pd.DataFrame]  # error band range (stderr)
    ]:  # yapf: disable
      """Evaluate representative_fn and err_fn."""

      representative: pd.DataFrame = (
          representative_fn(h) if representative_fn \
          else cast(pd.DataFrame, h.mean(numeric_only=True))
      )
      err_range: Tuple[pd.DataFrame, pd.DataFrame]
      std = err_fn(h) if err_fn else h.std(numeric_only=True)

      # Condition check: when representative_fn is given,
      # err_fn should return a range (i.e., tuple)
      if representative_fn and err_fn and not isinstance(std, tuple):
        raise ValueError(
            "When representative_fn is given, err_fn must return a range "
            "(tuple of pd.DataFrame) representing the lower and upper value "
            "of the error band. Pass err_fn=None to use the default one, "
            "or try: lambda h: (h.mean() + h.std(), h.mean() - h.std()). "
            f"err_fn returned: {std}")

      if isinstance(std, pd.DataFrame):
        mean = h.mean(numeric_only=True)
        err_range = (mean - std, mean + std)
        return representative, err_range

      elif (isinstance(std, tuple) and len(std) == 2 and
            isinstance(std[0], pd.DataFrame) and
            isinstance(std[1], pd.DataFrame)):
        err_range = (std[0], std[1])
        return representative, err_range  # type: ignore

      raise ValueError("err_fn must return either a tuple of "
                       "two DataFrames or a single DataFrame, but "
                       f"got {type(std)}")

    NULL = pd.DataFrame()
    representative: pd.DataFrame = NULL
    err: Tuple[pd.DataFrame, pd.DataFrame] = (NULL, NULL)
    _h_interpolated: Optional[Hypothesis] = None

    if 'x' not in kwargs:
      # index (same across runs) being x value, so we can simply average
      representative, err = _representative_and_err(self._parent)
    else:
      # might have different x values --- we need to interpolate.
      # (i) check if the x-column is consistent?
      x = kwargs['x']
      if n_samples is None and np.any(self._parent.grouped[x].nunique() > 1):
        warnings.warn(
            f"The x value (column `{x}`) is not consistent "
            "over different runs. Automatically falling back to the "
            "subsampling and interpolation mode (n_samples=10000). "
            "Explicitly setting the `n_samples` parameter is strongly "
            "recommended.", UserWarning)
        n_samples = 10000
      else:
        representative, err = _representative_and_err(self._parent)

    if n_samples is not None:
      # subsample by interpolation, then average.
      _h_interpolated = self._parent.interpolate(
          x_column=kwargs.get('x', None), n_samples=n_samples)
      representative, err = _representative_and_err(_h_interpolated)

      # Now that the index of group-averaged dataframes are the x samples
      # we interpolated on, we can let DataFrame.plot use them as index
      if 'x' in kwargs:
        del kwargs['x']

    if not isinstance(representative, pd.DataFrame):
      raise TypeError("representative_fn should return a pd.DataFrame, "
                      f"but got {type(err)}")

    #
    ## STEP 2. Determine y columns
    #

    # determine which columns to draw (i.e. y) before smoothing.
    # should only include numerical values
    _use_default_y: bool = 'y' not in kwargs
    y: List[str] = kwargs.get('y', representative.columns)
    if isinstance(y, str):
      y = [y]
    y = list(y)
    # Always exclude the x-axis (if non-index)
    if 'x' in kwargs:
      y = [yi for yi in y if yi != kwargs['x']]

    if any(not isinstance(yi, str) for yi in y):
      raise TypeError("`y` contains one or more non-str argument(s).")

    if ignore_unknown:
      # Add extra columns even if do not exist in the dataframe, if y is given.
      # This is a hack to handle non-homogeneous column names in a single
      # Experiment. see test_gridplot_basic() for a scenario.
      extra_y = set(y) - set(representative.columns)
      for yi in extra_y:
        representative[yi] = np.nan
        err[0][yi] = np.nan
        err[1][yi] = np.nan

    def _should_include_column(col_name: str) -> bool:
      if not col_name:  # empty name
        return False

      # include only numeric values (integer or float).
      # (check from the originial hypothesis dataframe, not from representative)
      for df in self._dataframes:
        if col_name in df and df[col_name].dtype.kind not in ('i', 'f'):
          if not _use_default_y:
            raise ValueError(f"Invalid y: the column `{col_name}` "
                             f"has a non-numeric type: {df[col_name].dtype}.")
          return False

      # unknown column in the DataFrame
      # Note that additional extra_y columns are also accepted
      if col_name not in representative.columns:
        if ignore_unknown:
          return False  # just ignore, no error
        else:
          raise ValueError(
              f"Unknown column name '{col_name}'. " +
              f"Available columns: {list(representative.columns)}; " +
              "Use ignore_unknown=True to ignore unknown columns.")

      return True

    # Exclude non-numeric types that cannot be plotted or interpolated
    y = [yi for yi in y if _should_include_column(yi)]
    _columns_to_keep = [*y]
    if 'x' in kwargs:
      _columns_to_keep.append(kwargs['x'])
    representative = cast(pd.DataFrame, representative[_columns_to_keep])
    err = (
        cast(pd.DataFrame, err[0][_columns_to_keep]),
        cast(pd.DataFrame, err[1][_columns_to_keep]),
    )

    #
    ## STEP 3. Processing of data, such as interpolation and smoothing
    #

    # There might be many NaN values if each column is being logged
    # at a different period. We fill in the missing values.
    representative = util.ensure_notNone(representative.interpolate())
    assert representative is not None
    err = (util.ensure_notNone(err[0].interpolate()),
           util.ensure_notNone(err[1].interpolate()))

    if rolling:
      rolling_kwargs = _rolling_kwargs(rolling)
      representative = representative.rolling(**rolling_kwargs).mean()
      err = (err[0].rolling(**rolling_kwargs).mean(),
             err[1].rolling(**rolling_kwargs).mean())

    # suptitle: defaults to hypothesis name if ax/grid was not given
    if suptitle is None and (ax is None and grid is None):
      suptitle = self._parent.name

    # Merge with hypothesis's default style
    if self._parent.style:
      kwargs = {**self._parent.style, **kwargs}

    return self._do_plot(
        y,
        representative,  # type: ignore
        err,  # type: ignore
        _h_interpolated=_h_interpolated,
        n_samples=n_samples,
        subplots=subplots,
        rolling=rolling,
        err_style=err_style,
        std_alpha=std_alpha,
        runs_alpha=runs_alpha,
        legend=legend,
        prettify_labels=prettify_labels,
        suptitle=suptitle,
        grid=grid,
        ax=ax,
        tight_layout=tight_layout,
        rasterized=rasterized,
        args=args,  # type: ignore
        kwargs=kwargs)  # type: ignore

  def _validate_ax_with_y(self, ax, y):
    assert not isinstance(y, str)
    if not isinstance(ax, (Axes, np.ndarray)):
      raise TypeError("`ax` must be a single Axes or a ndarray of Axes, "
                      "but given {}".format(ax))
    if isinstance(ax, Axes):
      ax = np.asarray([[ax]], dtype=object)
    if len(ax.ravel()) != len(y):
      raise ValueError("The length of `ax` and `y` must be equal: "
                       "ax = {}, y = {}".format(len(ax), len(y)))

  def _do_plot(
      self,
      y: List[str],
      representative: pd.DataFrame,  # usually mean
      err_range: Tuple[pd.DataFrame, pd.DataFrame],  # usually mean ± stderr
      *,
      _h_interpolated: Optional[Hypothesis] = None,  # type: ignore
      n_samples: Optional[int],
      subplots: bool,
      rolling: Union[int, dict, None],
      err_style: Optional[str],
      std_alpha: float,
      runs_alpha: float,
      legend: Union[bool, int, str, Dict[str, Any]],
      prettify_labels: bool = True,
      suptitle: Optional[str],
      grid: Optional[GridPlot] = None,
      ax: Optional[Union[Axes, np.ndarray]] = None,
      tight_layout: Union[bool, Dict[str, Any]] = True,
      rasterized: bool = False,
      args: List,
      kwargs: Dict,
  ) -> GridPlot:

    # Fill-in sensible defaults.
    if subplots:  # mode (1) -- separate subplots
      title = list(y)
      color = [kwargs.get('color', '#1f77b4')] * len(y)
      label = [self.name or ''] * len(y)
      if prettify_labels:
        label = util.prettify_labels(label)

      # TODO: layout -> grid
      # Prepare kwargs for DataFrame.plot() call
      kwargs.update(dict(y=y, title=title, color=color, label=label))

      if grid is None:
        if ax is not None:
          # TODO: ignore figsize, layout, etc.
          self._validate_ax_with_y(ax, y)
          grid = GridPlot(y_names=y, axes=ax)
        else:
          grid = GridPlot(y_names=y,
                          layout=kwargs.get('layout', None),
                          figsize=kwargs.get('figsize', None))  # yapf: disable
      else:  # (*)
        if ax is not None:
          raise ValueError("Either one of `grid` and `ax` can be given.")
        self._validate_ax_with_y(grid.axes_active, y)

      # we have made sure that ax is always associated with grid.
      ax = grid.axes_active

    else:  # mode (2) -- everything in a single axesplot
      # TODO: This mode is deprecated, will not work anymore
      if not kwargs.get('title', None):
        kwargs['title'] = self.name

      if grid is None:
        grid = GridPlot(y_names=[kwargs['title']],
                        layout=(1, 1),
                        axes=ax,
                        figsize=kwargs.get('figsize', None))  # yapf: disable
      else:  # (*)
        if ax is not None:
          raise ValueError("Either one of `grid` and `ax` can be given.")
        self._validate_ax_with_y(grid.axes_active, y)

      ax = grid.axes_active[0]

    if isinstance(ax, np.ndarray) and 'layout' in kwargs:
      # avoid matplotlib warning: when multiple axes are passed,
      # layout are ignored.
      del kwargs['layout']

    # Legend
    if not isinstance(legend, bool) and isinstance(legend, (int, str)):
      legend = dict(ax=legend)
    if isinstance(legend, dict):
      kwargs['legend'] = False  # to use aggregated legend
    else:
      kwargs['legend'] = bool(legend)

    axes = representative.plot(
        *args, subplots=subplots, ax=ax, rasterized=rasterized, **kwargs)

    if err_style not in self.KNOWN_ERR_STYLES:
      raise ValueError(f"Unknown err_style '{err_style}', "
                       f"expected one of {self.KNOWN_ERR_STYLES}")

    if err_style in ('band', 'fill'):
      # show shadowed range of 1-std errors
      for ax, yi in zip(np.asarray(axes).flat, y):  # type: ignore
        if yi not in err_range[0] or yi not in err_range[1]:
          continue
        ax = cast(Axes, ax)
        mean_line = ax.get_lines()[-1]
        x = kwargs.get('x', None)
        x_values = representative[x].values if x else representative[yi].index
        ax.fill_between(x_values,
                        err_range[0][yi].values,
                        err_range[1][yi].values,
                        color=mean_line.get_color(),
                        alpha=std_alpha,
                        rasterized=rasterized,
                        )  # yapf: disable

    elif err_style in ('runs', 'unit_traces') and len(self.runs) > 1:
      # show individual runs
      for ax, yi in zip(np.asarray(axes).flat, y):  # type: ignore
        ax = cast(Axes, ax)
        x = kwargs.get('x', None)
        color = ax.get_lines()[-1].get_color()

        if n_samples:
          df_individuals = _h_interpolated._dataframes  # type: ignore
        else:
          df_individuals = self._dataframes

        for df in df_individuals:
          if yi not in df:
            continue
          if rolling:
            df = df.rolling(**_rolling_kwargs(rolling)).mean()

          # Note: the series may have nan (missing) values.
          df_yi = df[[x, yi]] if x is not None else df[yi]
          cast(pd.DataFrame, df_yi).dropna().plot(
              ax=ax, x=x, legend=False, label='',
              color=color, alpha=runs_alpha)  # yapf: disable

    # some sensible styling (grid, tight_layout) AFTER calling plot()
    # Note: 'kwargs[grid]' is for axes and 'grid' means GridPlot
    # TODO: Fix the name conflict.
    if kwargs.get('grid', True):  # this is always true?
      for ax in grid.axes_active:
        ax = cast(Axes, ax)
        ax.grid(which='both', alpha=0.5)

    # after data is available, configure x axis
    if kwargs.get('xaxis_formatter', True):
      for ax in grid.axes_active:
        ax = cast(Axes, ax)
        autoformat_xaxis(ax)

    # Add legend by default
    if legend:
      if isinstance(legend, dict):
        grid.add_legend(**legend)

    # add figure title
    fig = grid.figure
    if suptitle is not None:
      _add_suptitle(fig, suptitle)

    if tight_layout:
      tight_layout = {} if isinstance(tight_layout, bool) \
                        else tight_layout
      fig.tight_layout(**tight_layout)
    return grid


class HypothesisHvPlotter(HypothesisPlotter):

  def __repr__(self):
    return "<HypothesisHvPlotter<{}>".format(self._parent)

  def _do_plot(
      self,
      y: List[str],
      representative: pd.DataFrame,
      err_range: Tuple[pd.DataFrame, pd.DataFrame],  # usually mean ± stderr
      *,
      _h_interpolated: Optional[Hypothesis] = None,
      n_samples: Optional[int],
      subplots: bool,
      rolling: Optional[int],
      err_style: Optional[str],
      std_alpha: float,
      runs_alpha: float,
      legend: Union[bool, int, str, Dict[str, Any]],
      prettify_labels: bool = False,
      suptitle: Optional[str],
      ax=None,
      grid=None,
      tight_layout: bool = True,
      rasterized: bool = True,
      args: List,
      kwargs: Dict,
  ):
    if not hasattr(representative, 'hvplot'):
      import hvplot.pandas

    if subplots:
      kwargs['legend'] = False

      # TODO implement various options for hvplot.
      kwargs.update(dict(y=y))
      p = representative.hvplot(shared_axes=False, subplots=True, **kwargs)

      # Display a single legend without duplication
      if legend and isinstance(p.data, dict):
        if isinstance(legend, bool):
          for overlay in p.data.values():
            overlay.opts('Curve', show_legend=legend)
        elif isinstance(legend, int):
          for overlay in itertools.islice(p.data.values(), legend, legend + 1):
            overlay.opts('Curve', show_legend=True)
        elif isinstance(legend, str):
          for k in p.data.keys():
            yi = k if isinstance(k, str) else k[0]
            if yi == legend:
              p[yi].opts('Curve', show_legend=True)
        else:
          raise TypeError(f"Unsupported type for legend : "
                          f"{type(legend)}")  # yapf: disable

    else:
      # TODO implement this version
      raise NotImplementedError

    if err_style in ('band', 'fill') and std_alpha:
      # TODO
      band_lower, band_upper = err_range
      band_lower['_facet'] = 'lower'
      band_upper['_facet'] = 'upper'
      band = pd.concat([band_lower.add_suffix('.min'),
                        band_upper.add_suffix('.max')], axis=1)  # yapf: disable
      x_col = kwargs.get('x', None)
      if x_col:
        x_col += '.min'

      # for each subplot (one for each y variable), display error band
      from holoviews.core.overlay import Overlayable

      def _overlay_area(yi):
        return band.hvplot.area(
            x=x_col,
            y=yi + '.min',
            y2=yi + '.max',
            legend=False,
            line_width=0,
            hover=False,
            alpha=std_alpha,
        )

      if isinstance(p.data, pd.DataFrame):
        # single plot rather than multiple subplots
        p = p * _overlay_area(y[0])

      else:
        # subplots (see y for the list of subplots)
        for k in p.data.keys():
          yi = k if isinstance(k, str) else k[0]
          assert isinstance(yi, str)
          if yi not in y:
            continue

          area_fill = band.hvplot.area(
              x=x_col,
              y=yi + '.min',
              y2=yi + '.max',
              legend=False,
              line_width=0,
              hover=False,
              alpha=std_alpha,
          ).opts(show_legend=False)

          assert isinstance(p.data[k], Overlayable), \
              str(type(p.data[k]))
          p.data[k] = p.data[k] * area_fill  # overlay 1-std err range

    # TODO: runs_alpha and rolling

    p = p.opts(shared_axes=False)
    return p


class LegendSpec(dict):
  """A frozen contianer that holds keyward arguments to matplotlib.legend()."""

  def __repr__(self):
    return f"LegendSpec({super().__repr__()})"

  def _immutable(self, *args, **kwargs):
    raise RuntimeError("LegendSpec is a frozen dictionary.")

  __setitem__ = __delitem__ = pop = popitem = _immutable
  clear = setdefault = _immutable  # type: ignore

  def __call__(self, **kwargs) -> LegendSpec:
    # Return an updated copy if it is called.
    return LegendSpec({**self, **kwargs})

  update = __call__  # type: ignore
  with_ = __call__


class LegendPreset:
  """Some commonly-used configurations for placing legend.

  See the documentation of matplotlib.legend().
  """
  FIRST_AXIS = LegendSpec(ax=0)
  RIGHT = LegendSpec(loc='upper left', bbox_to_anchor=(1.03, 0.95), ncol=1)
  BOTTOM = LegendSpec(loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=4)
  BOTTOM_1COL = BOTTOM(ncol=1)
  TOP = LegendSpec(loc='lower center', bbox_to_anchor=(0.5, 0.95), ncol=4)

  @classmethod
  def AUTO(cls, labels: List[str]):
    """Default, automatic behavior of placing the legend.

    Put a legend on the right side of the figure (outside the grid),
    or on the first axes if #hypothesis is small enough (<=5).
    """
    # only one: put in the upper center like a title.
    if len(labels) == 1:
      return cls.TOP

    # It can be lengthy; Use side layout on the right.
    if len(labels) > 5 or any(len(label) > 20 for label in labels):
      return cls.RIGHT

    # compact enough; show in the first axes.
    return cls.FIRST_AXIS

  def __new__(cls):
    raise TypeError("This class cannot be instantiated.")


class ExperimentPlotter:

  def __init__(self, experiment: Experiment):
    self._parent = experiment
    self.LegendPreset = LegendPreset

  def __repr__(self):
    return "<ExperimentPlotter<{}>".format(self._parent)

  @property
  def _hypotheses(self):
    return self._parent._hypotheses

  @property
  def _columns(self) -> Iterable[str]:
    if self._parent._summary_columns:
      return self._parent._summary_columns
    else:
      return self._parent.columns

  def __call__(
      self,
      *args,
      suptitle: Optional[str] = None,
      legend: Union[bool, int, str, Dict[str, Any], Callable[[List[str]], Any],
                    LegendSpec] = LegendPreset.AUTO,
      grid=None,
      colors=None,
      tight_layout: Union[bool, Dict[str, Any]] = True,
      **kwargs,
  ) -> GridPlot:
    '''Experiment.plot.

    @see DataFrame.plot()

    Args:
      subplots:
      y: (Str, Iterable[Str])
      suptitle (str):
      legend: Whether or how to put legend of hypothesis names on the GridPlot.
        - True: Put legend on every individual axes.
        - False: No legend on any individual axes.
        - dict: It will be passed as kwargs to GridPlot.add_legend().
          Examples:
            `dict(ax=0)` means to put legend on the first subplot axes,
            `dict(ax=None)` means to put legend on the figure itself,
            `dict(ax='loss')` means to put legend on the subplot axes
               whose title is `loss`.
            `dict(bbox_to_anchor=(0, 1.1), loc='lower left')`:
               put a large legend on the figure (e.g., above the title)
        - int: Have a same meaning as dict(ax=...).
        - str: Have a same meaning as dict(ax=...).
        - or a callable (List[str] -> ...), which dynamically determines
          the argument value given a list of labels.
        - (See also) ex.plot.LegendPreset for some common presets.
      colors: Iterable[Str]
    '''
    # Prepare kwargs for Hypothesis.plot().

    # TODO extract x-axis
    y = kwargs.get('y', None)
    if y is None:
      # draw all known columns by default
      y = list(self._columns)
      if 'x' in kwargs:
        y = [yi for yi in y if yi != kwargs['x']]
      kwargs['y'] = y

    # Assign line style for each hypothesis
    # TODO: avoid conflicts as much as we can against hypothosis.style
    axes_cycle = matplotlib.rcParams['axes.prop_cycle']()  # type: ignore
    axes_props = list(itertools.islice(axes_cycle, len(self._hypotheses)))
    for key in list(axes_props[0].keys()):
      # explicit kwargs passed to plot() should take a precedence.
      if key in kwargs:
        for prop in axes_props:
          del prop[key]

    if 'color' not in axes_props[0].keys():
      # axes.prop_cycle does not have color. Fall back to default colors
      from .colors import ExptSensible17
      color_it = itertools.cycle(ExptSensible17)
      for prop, c in zip(axes_props, color_it):
        prop['color'] = c

    if colors is not None:
      if len(colors) != len(self._hypotheses):
        raise ValueError("`colors` should have the same number of elements as "
                         "Hypotheses ({}), but the given length is {}.".format(
                             len(self._hypotheses), len(colors)))
      for prop, given_color in zip(axes_props, colors):
        prop['color'] = given_color

    hypothesis_labels = [name for name, _ in self._hypotheses.items()]
    if kwargs.get('prettify_labels', False):
      hypothesis_labels = util.prettify_labels(hypothesis_labels)

    # Legend (applies a sensible default)
    if isinstance(legend, LegendSpec):
      pass
    elif callable(legend):
      # resolve legend args on-the-fly.
      legend = legend(hypothesis_labels)

    if not isinstance(legend, bool) and isinstance(legend, (int, str)):
      legend = dict(ax=legend)
    if isinstance(legend, dict):
      kwargs['legend'] = False  # to use aggregated legend
    else:
      kwargs['legend'] = bool(legend)

    given_ax_or_grid = ('ax' in kwargs) or (grid is not None)

    for i, (name, hypo) in enumerate(self._hypotheses.items()):
      h_kwargs = kwargs.copy()
      if grid is not None:
        h_kwargs.pop('ax', None)  # i=0: ax, i>0: grid

      if isinstance(y, str):
        # display different hypothesis over subplots:
        h_kwargs['label'] = hypothesis_labels[i]
        h_kwargs['subplots'] = False
        if 'title' not in kwargs:
          h_kwargs['title'] = y  # column name

      else:
        # display multiple columns over subplots:
        if y is not None:
          h_kwargs['label'] = [f'{y_i} ({name})' for y_i in y]
          if h_kwargs.get('prettify_labels', False):
            h_kwargs['label'] = util.prettify_labels(h_kwargs['label'])
        h_kwargs['subplots'] = True

      h_kwargs.update(axes_props[i])  # e.g. color, linestyle, etc.

      # Hypothesis' own style should take more priority
      h_kwargs.update(hypo.style)

      # exclude the hypothesis if it has no runs in it
      if hypo.empty():
        warnings.warn(f"Hypothesis `{hypo.name}` has no data, "
                      "ignoring it", UserWarning)
        continue

      h_kwargs['tight_layout'] = False
      h_kwargs['ignore_unknown'] = True
      h_kwargs['suptitle'] = ''  # no suptitle for each hypo

      grid = hypo.plot(*args, grid=grid, **h_kwargs)  # on the same ax(es)?
      assert grid is not None

    assert grid is not None  # True if len(hypothesis) > 0

    # corner case: if there is only one column, use it as a label
    if len(grid.axes_active) == 1 and isinstance(y, str):
      grid.axes_active[0].set_ylabel(y)

    # Add legend by default
    if legend:
      if isinstance(legend, dict):
        grid.add_legend(**legend)

    # title, etc.
    if suptitle is None:
      ex = self._parent
      suptitle = ex.name
    if not given_ax_or_grid:  # do not add suptitle if ax or grid were given
      _add_suptitle(grid.figure, suptitle)

    # adjust figure after all hypotheses have been drawn,
    # but just only once (see tight_layout on HypothesisPlotter)
    if tight_layout:
      tight_layout = {} if isinstance(tight_layout, bool) else tight_layout
      grid.figure.tight_layout(**tight_layout)

    return grid


# Protected Utilities for plotting
def _add_suptitle(fig, suptitle, fontsize='x-large', y=1.02, **kwargs):
  if suptitle is None:
    raise ValueError("suptitle should not be None")

  if isinstance(suptitle, str):
    # For default y value, refer to https://stackoverflow.com/questions/8248467
    fig.suptitle(suptitle, fontsize=fontsize, y=y, **kwargs)
  elif isinstance(suptitle, dict):
    fig.suptitle(**suptitle)
  else:
    raise TypeError("Expected str or dict for suptitle: {}".format(suptitle))


def _rolling_kwargs(rolling: Union[int, Dict[str, Any]]) -> Dict[str, Any]:
  defaults = dict(min_periods=1, center=True)
  if isinstance(rolling, dict):
    if 'window' not in rolling:
      raise ValueError("rolling.window must not be None")
    return {**defaults, **rolling}
  else:  # int
    return {'window': rolling, **defaults}


# yapf: disable
FORMATTER_MEGA = matplotlib.ticker.FuncFormatter(lambda x, pos: '{0:g}M'.format(x / 1e6))
FORMATTER_KILO = matplotlib.ticker.FuncFormatter(lambda x, pos: '{0:g}K'.format(x / 1e3))
# yapf: enable


def autoformat_xaxis(ax: Axes, scale: Optional[float] = None):
  if scale is None:
    scale = ax.xaxis.get_data_interval()[1]

  assert scale is not None
  if scale >= 1e6:
    ticks = FORMATTER_MEGA
  elif scale >= 1e3:
    ticks = FORMATTER_KILO
  else:
    ticks = None

  if ticks is not None:
    ax.xaxis.set_major_formatter(ticks)
  return ticks


def make_legend_fig(legend: matplotlib.legend.Legend) -> Figure:
  """Create a new matplotlib figure that contains a copy of the legend."""

  # Get the dimensions (in inches) of the legend's bounding box
  legend_inches = legend.get_window_extent().transformed(
      cast(Figure, legend.figure).dpi_scale_trans.inverted())

  fig = Figure(
      figsize=(
          legend_inches.width + 0.05,
          legend_inches.height + 0.05,
      ))
  fig.add_axes([0, 0, 1, 1]).axis('off')

  fig.legend(
      legend.legendHandles,
      [t.get_text() for t in legend.texts],
      ncol=legend._ncols,
      loc='center',
      bbox_to_anchor=(0.5, 0.5),
  )
  return fig


HypothesisPlotter.__doc__ = HypothesisPlotter.__call__.__doc__
HypothesisHvPlotter.__doc__ = HypothesisHvPlotter.__call__.__doc__
ExperimentPlotter.__doc__ = ExperimentPlotter.__call__.__doc__
