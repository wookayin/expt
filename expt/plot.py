"""
Plotting behavior (matplotlib, hvplot, etc.) for expt.data
"""

import difflib
import itertools
import warnings
from typing import Union, Iterable, Iterator, Optional
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from scipy import interpolate

import matplotlib.ticker
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from . import util
from . import data


warnings.filterwarnings("ignore", category=UserWarning,
                        message='Creating legend with loc="best"')


class GridPlot:
    """Multi-plot grid subplots.

    This class provides more object-oriented methods to manipulate matplotlib
    grid subplots (even after creation). For example, add_legend()."""

    def __init__(self, *,
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
            rows = int((self.n_plots + 1) ** 0.5)
            cols = int(np.ceil(self.n_plots / rows))
            layout = (rows, cols)
        else:
            rows, cols = layout

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
            import matplotlib.pyplot as plt   # lazy load
            fig, axes = plt.subplots(rows, cols, **subplots_kwargs)
        elif fig is None and axes is not None:
            if isinstance(axes, np.ndarray):
                if len(axes.shape) != 2:
                    raise ValueError(
                        "When axes is a ndarray of Axes, the rank should be "
                        "2 (but given {})".format(len(axes.shape)))
            else:
                # ensure axes is a 2D array of Axes
                axes = np.asarray(axes, dtype=object).reshape([1, 1])
            fig = axes.flat[0].get_figure()
        else:
            raise ValueError("If fig is given, axes should be given as well")

        self._fig: Figure = fig
        self._axes: np.ndarray = axes

        for yi, ax in zip(self._y_names, self.axes_active):
            ax.set_title(yi)
        for yi, ax in zip(self._y_names, self.axes_inactive):
            ax.axis('off')

        if len(self._axes.flat) < len(y_names):
            raise ValueError(
                "Grid size is %d * %d = %d, smaller than len(y) = %d" % (
                    rows, cols, len(self._axes.flat), len(y_names)))

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

    def __getitem__(self, key) -> Union[Axes, np.ndarray]:
        # Note: see add_legend(ax) as well
        if isinstance(key, str):
            # find axes by name
            try:
                index = self._y_names.index(key)
            except (ValueError, IndexError) as e:
                raise ValueError(
                    "Unknown index: {}. Close matches: {}".format(
                        key, difflib.get_close_matches(key, self._y_names)
                    )) from e
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
                   loc=None, **kwargs):
        """Put a legend for all objects aggregated over the axes.

        If ax is given, a legend will be put at the specified Axes. If it
        is str or int (say `k`), `self[k]` will be used instead. Otherwise
        (i.e., ax=None), the legend will be put on figure-level (if there are
        more than two Axes), or the unique axis (if there is only one Axes).

        Args:
            loc: str or pair of floats (see matplotlib.pyplot.legend)
            kwargs: Additional kwargs passed to ax.legend().
              e.g., ncol=4
        """
        if ax is None:
            # automatic: figure legend or the (unique) axes
            if self.n_plots >= 2:
                target = self.fig
            else:
                target = self.axes_active[0]
        else:
            if isinstance(ax, (int, str)):   # see __getitem__
                target = self[ax]
            else:
                target = ax

        legend_labels = zip(*[(h, l) for (l, h) in
                              sorted(self._collect_legend().items())])
        target.legend(*legend_labels, loc=loc, **kwargs)

        if isinstance(target, Axes) and not target.lines:
            target.axis('off')
        return self

    def _collect_legend(self) -> Dict[str, Any]:
        """Collect all legend labels from the subplot axes."""
        legend_labels = dict()
        for ax in self.axes_active:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                if label in legend_labels:
                    # TODO: Add warning/exception if color conflict is found
                    pass
                else:
                    legend_labels[label] = handle
        return legend_labels


class HypothesisPlotter:

    def __init__(self, hypothesis: 'Hypothesis'):
        self._parent = hypothesis

    @property
    def name(self):
        return self._parent.name

    @property
    def runs(self):
        return self._parent.runs

    @property
    def grouped(self):
        return self._parent.grouped

    @property
    def _dataframes(self) -> List[pd.DataFrame]:
        return self._parent._dataframes

    def interpolate_and_average(
        self,
        df_list: List[pd.DataFrame],
        n_samples: int,
        x_column: Optional[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        # for each df, interpolate on 'x'
        x_series = pd.concat([
            (pd.Series(df.index) if x_column is None else df[x_column])
            for df in df_list])
        x_min = x_series.min()
        x_max = x_series.max()
        x_samples = np.linspace(x_min, x_max, num=n_samples)

        # get interpolated dataframes
        df_interp_list = []
        for df in df_list:
            if x_column is not None:
                df = df.set_index(x_column)

            def _interpolate_if_numeric(y_series):
                if y_series.dtype.kind in ('i', 'f'):
                    return interpolate.interp1d(df.index, y_series, bounds_error=False)(x_samples)
                else:
                    # maybe impossible to interpolate (i.e. object), skip it.
                    # (this column will be filtered out later on)
                    return pd.Series(np.empty_like(x_samples, dtype=object))
            df_interp = df.apply(_interpolate_if_numeric)
            df_interp[x_column] = x_samples
            df_interp.set_index(x_column, inplace=True)
            df_interp_list.append(df_interp)

        # have individual interpolation data for each run cached (for the last call).
        self._df_interp_list = df_interp_list

        grouped = pd.concat(df_interp_list, sort=False).groupby(level=0)
        mean, std = grouped.mean(), grouped.std()

        return mean, std


    KNOWN_ERR_STYLES = (None, False, 'band', 'fill', 'runs', 'unit_traces')

    def __call__(self, *args,
                 subplots=True,
                 err_style="runs",
                 std_alpha=0.2, runs_alpha=0.2,
                 n_samples=None,
                 rolling=None,
                 ignore_unknown: bool = False,
                 legend: Union[bool, int, str, Dict[str, Any]] = False,
                 prettify_labels: bool = False,
                 suptitle: Optional[str] = None,
                 grid: Optional[GridPlot] = None,
                 ax: Optional[Union[Axes, np.ndarray]] = None,
                 tight_layout: bool = True,
                 **kwargs) -> GridPlot:
        '''
        Hypothesis.plot based on matplotlib.

        This can work in two different modes:
        (1) Plot (several or all) columns as separate subplots (subplots=True)
        (2) Plot (several or all) columns in a single axesplot (subplots=False)

        Additional keyword arguments:
            - rolling (int): A window size for rolling and smoothing.
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
               (i) runs, unit_traces: Show individual runs/traces (see runs_alpha).
               (ii) band, fill: Show as shaded area (see std_alpha).
               (iii) None or False: do not display any errors.
            - std_alpha (float): If not None, will show the 1-std range as a
              shaded area. Defaults 0.2,
            - runs_alpha (float): If not None, will draw an individual line
              for each run. Defaults 0.2.
            - ax (Axes or ndarray of Axes): Subplot axes to draw plots on.
              Note that this is mutually exclusive with `grid`.
            - grid (GridPlot): A `GridPlot` instance (optional) to use for
              matplotlib figure and axes. If this is given, `ax` must be None.

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
            raise ValueError("No data to plot, all runs have a empty DataFrame.")

        grouped = self.grouped

        if 'x' not in kwargs:
            # index (same across runs) being x value, so we can simply average
            mean, std = grouped.mean(), grouped.std()
        else:
            # might have different x values --- we need to interpolate.
            # (i) check if the x-column is consistent?
            if n_samples is None and np.any(grouped.nunique()[kwargs['x']] > 1):
                warnings.warn(
                    f"The x value (column `{kwargs['x']}`) is not consistent "
                    "over different runs. Automatically falling back to the "
                    "subsampling and interpolation mode (n_samples=10000). "
                    "Explicitly setting the `n_samples` parameter is strongly "
                    "recommended.", UserWarning
                )
                n_samples = 10000
            else:
                mean, std = grouped.mean(), grouped.std()

        if n_samples is not None:
            # subsample by interpolation, then average.
            mean, std = self.interpolate_and_average(
                self._dataframes, n_samples=n_samples,
                x_column=kwargs.get('x', None),
            )
            # Now that the index of group-averaged dataframes are the x samples
            # we interpolated on, we can let DataFrame.plot use them as index
            if 'x' in kwargs:
                del kwargs['x']

        # there might be many NaN values if each column is being logged
        # at a different period. We fill in the missing values.
        mean = mean.interpolate()
        std = std.interpolate()

        # determine which columns to draw (i.e. y) before smoothing.
        # should only include numerical values
        y: Iterable[str] = kwargs.get('y', None) or mean.columns
        if isinstance(y, str): y = [y]
        if 'x' in kwargs:
            y = [yi for yi in y if yi != kwargs['x']]

        if ignore_unknown:
            # TODO(remove): this is hack to handle homogeneous column names
            # over different hypotheses in a single of experiment, because it
            # will end up adding dummy columns instead of ignoring unknowns.
            extra_y = set(y) - set(mean.columns)
            for yi in extra_y:
                mean[yi] = np.nan

        def _should_include_column(col_name: str) -> bool:
            if not col_name:   # empty name
                return False

            # unknown column in the DataFrame
            if col_name not in mean.dtypes:
                if ignore_unknown:
                    return False   # just ignore, no error
                else:
                    raise ValueError(
                        f"Unknown column name '{col_name}'. " +
                        f"Available columns: {list(mean.columns)}; " +
                        "Use ignore_unknown=True to ignore unknown columns.")

            # include only numeric values (integer or float)
            if not (mean.dtypes[col_name].kind in ('i', 'f')):
                return False
            return True
        y = [yi for yi in y if _should_include_column(yi)]

        if rolling:
            mean = mean.rolling(rolling, min_periods=1, center=True).mean()
            std = std.rolling(rolling, min_periods=1, center=True).mean()

        # suptitle: defaults to hypothesis name if ax/grid was not given
        if suptitle is None and (ax is None and grid is None):
            suptitle = self._parent.name

        return self._do_plot(y, mean, std, n_samples=n_samples,
                             subplots=subplots, rolling=rolling,
                             err_style=err_style,
                             std_alpha=std_alpha, runs_alpha=runs_alpha,
                             legend=legend,
                             prettify_labels=prettify_labels,
                             suptitle=suptitle,
                             grid=grid, ax=ax,
                             tight_layout=tight_layout,
                             args=args, kwargs=kwargs)  # type: ignore

    def _validate_ax_with_y(self, ax, y):
        assert not isinstance(y, str)
        if not isinstance(ax, (Axes, np.ndarray)):
            raise TypeError("`ax` must be a single Axes or a ndarray of Axes, "
                            "but given {}".format(ax))
        if isinstance(ax, Axes):
            ax = np.asarray([ax], dtype=object)
        if len(ax) != len(y):
            raise ValueError("The length of `ax` and `y` must be equal: "
                             "ax = {}, y = {}".format(len(ax), len(y)))

    def _do_plot(self,
                 y: List[str],
                 mean: pd.DataFrame,
                 std: pd.DataFrame,
                 *,
                 n_samples: Optional[int],
                 subplots: bool,
                 rolling: Optional[int],
                 err_style: Optional[str],
                 std_alpha: float,
                 runs_alpha: float,
                 legend: Union[bool, int, str, Dict[str, Any]],
                 prettify_labels: bool = True,
                 suptitle: Optional[str],
                 grid: Optional[GridPlot] = None,
                 ax: Optional[Union[Axes, np.ndarray]] = None,
                 tight_layout: Union[bool, Dict[str, Any]] = True,
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
            kwargs.update(dict(y=y, title=title,
                               color=color, label=label))

            if grid is None:
                if ax is not None:
                    # TODO: ignore figsize, layout, etc.
                    self._validate_ax_with_y(ax, y)
                    grid = GridPlot(y_names=y, axes=ax)
                else:
                    grid = GridPlot(y_names=y,
                                    layout=kwargs.get('layout', None),
                                    figsize=kwargs.get('figsize', None))
            else: # (*)
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
                                layout=(1, 1), axes=ax,
                                figsize=kwargs.get('figsize', None))
            else: # (*)
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
            kwargs['legend'] = False      # to use aggregated legend
        else:
            kwargs['legend'] = bool(legend)

        axes = mean.plot(*args, subplots=subplots, ax=ax, **kwargs)

        if err_style not in self.KNOWN_ERR_STYLES:
            raise ValueError(f"Unknown err_style '{err_style}', "
                             f"expected one of {self.KNOWN_ERR_STYLES}")

        if err_style in ('band', 'fill'):
            # show shadowed range of 1-std errors
            for ax, yi in zip(np.asarray(axes).flat, y):
                mean_line = ax.get_lines()[-1]
                ax.fill_between((mean - std)[yi].index,
                                (mean - std)[yi].values,
                                (mean + std)[yi].values,
                                color=mean_line.get_color(),
                                alpha=std_alpha)

        elif err_style in ('runs', 'unit_traces') and len(self.runs) > 1:
            # show individual runs
            for ax, yi in zip(np.asarray(axes).flat, y):
                x = kwargs.get('x', None)
                color = ax.get_lines()[-1].get_color()

                df_individuals = self._df_interp_list if n_samples \
                    else self._dataframes

                for df in df_individuals:
                    if not yi in df:
                        continue
                    if rolling:
                        df = df.rolling(rolling, min_periods=1, center=True).mean()
                    df.plot(ax=ax, x=x, y=yi, legend=False, label='',
                            color=color, alpha=runs_alpha)

        # some sensible styling (grid, tight_layout) AFTER calling plot()
        # Note: 'kwargs[grid]' is for axes and 'grid' means GridPlot
        # TODO: Fix the name conflict.
        if kwargs.get('grid', True):   # this is always true?
            for ax in grid.axes_active:
                ax.grid(which='both', alpha=0.5)

        # after data is available, configure x axis
        if kwargs.get('xaxis_formatter', True):
            for ax in grid.axes_active:
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
            tight_layout = {} if isinstance(tight_layout, bool) else tight_layout
            fig.tight_layout(**tight_layout)
        return grid


class HypothesisHvPlotter(HypothesisPlotter):

    def _do_plot(self,
                 y: List[str],
                 mean: pd.DataFrame,
                 std: pd.DataFrame,
                 *,
                 n_samples: Optional[int],
                 subplots: bool,
                 rolling: Optional[int],
                 err_style: Optional[str],
                 std_alpha: float,
                 runs_alpha: float,
                 legend: Union[bool, int, str, Dict[str, Any]],
                 prettify_labels: bool = False,
                 suptitle: Optional[str],
                 ax = None,
                 grid = None,
                 tight_layout: bool = True,
                 args: List,
                 kwargs: Dict,
                 ):
        if not hasattr(mean, 'hvplot'):
            import hvplot.pandas

        if subplots:
            kwargs['legend'] = False

            # TODO implement various options for hvplot.
            kwargs.update(dict(y=y))
            p = mean.hvplot(shared_axes=False, subplots=True,
                            **kwargs)

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
                    raise TypeError("Unsupported type for legend : {}".format(type(legend)))

        else:
            # TODO implement this version
            raise NotImplementedError

        if err_style in ('band', 'fill') and std_alpha:
            band_lower = mean - std
            band_lower['_facet'] = 'lower'
            band_upper = mean + std
            band_upper['_facet'] = 'upper'
            band = pd.concat([band_lower.add_suffix('.min'),
                              band_upper.add_suffix('.max')], axis=1)
            x_col = kwargs.get('x', None)
            if x_col:
                x_col += '.min'

            # for each subplot (one for each y variable), display error band
            from holoviews.core.overlay import Overlayable
            def _overlay_area(yi):
                return band.hvplot.area(
                    x=x_col,
                    y=yi + '.min', y2=yi + '.max',
                    legend=False, line_width=0, hover=False,
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
                        y=yi + '.min', y2=yi + '.max',
                        legend=False, line_width=0, hover=False,
                        alpha=std_alpha,
                    ).opts(show_legend=False)

                    assert isinstance(p.data[k], Overlayable), str(type(p.data[k]))
                    p.data[k] = p.data[k] * area_fill   # overlay 1-std err range

        # TODO: runs_alpha and rolling

        p = p.opts(shared_axes=False)
        return p


class ExperimentPlotter:

    def __init__(self, experiment: 'Experiment'):
        self._parent = experiment

    @property
    def _hypotheses(self):
        return self._parent._hypotheses

    @property
    def _columns(self) -> List[str]:
        return self._parent.columns

    DEFAULT_LEGEND_SPEC = dict(loc='upper left', bbox_to_anchor=(1.03, 1))

    def _determine_default_legend_spec(self, labels: List[str]):
        # Put a legend on the right side of the figure (outside the grid),
        # or on the first axes if #hypothesis is small enough (<=5)
        if len(labels) > 5:
            return self.DEFAULT_LEGEND_SPEC
        if any(len(l) > 20 for l in labels):
            return self.DEFAULT_LEGEND_SPEC
        return dict(ax=0)

    def __call__(self, *args,
                 suptitle: Optional[str] = None,
                 legend: Union[bool, int, str, Dict[str, Any]
                               ] = DEFAULT_LEGEND_SPEC,
                 grid=None,
                 colors=None,
                 tight_layout: Union[bool, Dict[str, Any]] = True,
                 **kwargs) -> GridPlot:
        '''
        Experiment.plot.

        @see DataFrame.plot()

        Args:
            subplots:
            y: (Str, Iterable[Str])
            suptitle (str):
            legend (bool or dict): Whether to put legend on the GridPlot.
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

        if colors is None:
            # len(colors) should equal hypothesis. or have as attributes
            from .colors import get_standard_colors
            colors = get_standard_colors(num_colors=len(self._hypotheses))

        hypothesis_labels = [name for name, _ in self._hypotheses.items()]
        if kwargs.get('prettify_labels', False):
            hypothesis_labels = util.prettify_labels(hypothesis_labels)

        # Legend (applies a sensible conditional default)
        if legend == self.DEFAULT_LEGEND_SPEC:
            legend = self._determine_default_legend_spec(hypothesis_labels)

        if not isinstance(legend, bool) and isinstance(legend, (int, str)):
            legend = dict(ax=legend)
        if isinstance(legend, dict):
            kwargs['legend'] = False      # to use aggregated legend
        else:
            kwargs['legend'] = bool(legend)

        given_ax_or_grid = ('ax' in kwargs) or (grid is not None)

        for i, (name, hypo) in enumerate(self._hypotheses.items()):
            if isinstance(y, str):
                # display different hypothesis over subplots:
                kwargs['label'] = hypothesis_labels[i]
                kwargs['subplots'] = False
                if 'title' not in kwargs:
                    kwargs['title'] = y    # column name

            else:
                # display multiple columns over subplots:
                if y:
                    kwargs['label'] = [f'{y_i} ({name})' for y_i in y]
                    if kwargs.get('prettify_labels', False):
                        kwargs['label'] = util.prettify_labels(kwargs['label'])
                kwargs['subplots'] = True
            kwargs['color'] = colors[i]

            # exclude the hypothesis if it has no runs in it
            if hypo.empty():
                warnings.warn(f"Hypothesis `{hypo.name}` has no data, "
                              "ignoring it", UserWarning)
                continue

            kwargs['tight_layout'] = False
            kwargs['ignore_unknown'] = True
            kwargs['suptitle'] = ''   # no suptitle for each hypo

            grid = hypo.plot(*args, grid=grid, **kwargs)    # on the same ax(es)?
            assert grid is not None

            kwargs.pop('ax', None)      # From now on, grid.axes will be used

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
        if not given_ax_or_grid:   # do not add suptitle if ax or grid were given
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


FORMATTER_MEGA = matplotlib.ticker.FuncFormatter(
    lambda x, pos: '{0:g}M'.format(x / 1e6))
FORMATTER_KILO = matplotlib.ticker.FuncFormatter(
    lambda x, pos: '{0:g}K'.format(x / 1e3))


def autoformat_xaxis(ax: Axes, scale: Optional[float] = None):
    if scale is None:
        scale = ax.xaxis.get_data_interval()[1]

    if scale >= 1e6:
        ticks = FORMATTER_MEGA
    elif scale >= 1e3:
        ticks = FORMATTER_KILO
    else:
        ticks = None

    if ticks is not None:
        ax.xaxis.set_major_formatter(ticks)
    return ticks


HypothesisPlotter.__doc__ = HypothesisPlotter.__call__.__doc__
HypothesisHvPlotter.__doc__ = HypothesisHvPlotter.__call__.__doc__
ExperimentPlotter.__doc__ = ExperimentPlotter.__call__.__doc__
