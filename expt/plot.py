"""
Plotting behavior (matplotlib, hvplot, etc.) for expt.data
"""

import warnings
from typing import Union, Iterable, Iterator, Optional
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from scipy import interpolate

from matplotlib.axes import Axes

from . import util
from . import data


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
                 prettify_labels: bool = True,
                 **kwargs):
        '''
        Hypothesis.plot based on matplotlib.

        This can work in two different modes:
        (1) Plot (several or all) columns as separate subplots (subplots=True)
        (2) Plot (several or all) columns in a single axesplot (subplots=False)

        Additional keyword arguments:
            - rolling (int): A window size for rolling and smoothing.
            - n_samples (int): If given, we subsample using n_samples number of
              equidistant points over the x axis. Values will be interpolated.
            - prettify_labels (bool): If True (default), apply a sensible
              default prettifier to the legend labels, truncating long names.
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

        All other kwargs is passed to DataFrame.plot(). For example, you may
        find the following parameters useful:
            `x`, `y`, `layout`, `figsize`, `ax`, `legend`, etc.

        '''
        if len(self.runs) == 0:
            # nothing to plot, do nothing
            return

        if 'x' not in kwargs:
            # index (same across runs) being x value, so we can simply average
            mean, std = self.grouped.mean(), self.grouped.std()
        else:
            # might have different x values --- we need to interpolate.
            # (i) check if the x-column is consistent?
            if n_samples is None and np.any(self.grouped.nunique()[kwargs['x']] > 1):
                warnings.warn(
                    f"The x value (column `{kwargs['x']}`) is not consistent "
                    "over different runs. Automatically falling back to the "
                    "subsampling and interpolation mode (n_samples=10000). "
                    "Explicitly setting the `n_samples` parameter is strongly "
                    "recommended.", UserWarning
                )
                n_samples = 10000
            else:
                mean, std = self.grouped.mean(), self.grouped.std()

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

        # determine which columns to draw (i.e. y) before smoothing.
        # should only include numerical values
        y: Iterable[str] = kwargs.get('y', None) or mean.columns
        if isinstance(y, str): y = [y]
        if 'x' in kwargs:
            y = [yi for yi in y if yi != kwargs['x']]

        def _should_include_column(col_name: str) -> bool:
            if not col_name:   # empty name
                return False

            # unknown column in the DataFrame
            if col_name not in mean.dtypes:
                if ignore_unknown:
                    return False   # just ignore, no error
                else:
                    raise ValueError(f"Unknown column name '{col_name}'. "
                                     "Available columns: {}".format(
                                         list(mean.columns)))

            # include only numeric values (integer or float)
            if not (mean.dtypes[col_name].kind in ('i', 'f')):
                return False
            return True
        y = [yi for yi in y if _should_include_column(yi)]

        if rolling:
            mean = mean.rolling(rolling, min_periods=1, center=True).mean()
            std = std.rolling(rolling, min_periods=1, center=True).mean()

        return self._do_plot(y, mean, std, n_samples=n_samples,
                             subplots=subplots, rolling=rolling,
                             err_style=err_style,
                             std_alpha=std_alpha, runs_alpha=runs_alpha,
                             prettify_labels=prettify_labels,
                             args=args, kwargs=kwargs)

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
                 prettify_labels: bool = True,
                 args: List,
                 kwargs: Dict,
                 ):
        # Fill-in sensible defaults.
        if subplots:  # mode (1) -- separate subplots
            title = list(y)
            color = [kwargs.get('color', '#1f77b4')] * len(y)
            label = [self.name or ''] * len(y)
            if prettify_labels:
                label = util.prettify_labels(label)

            # TODO: layout -> grid
            kwargs.update(dict(y=y, title=title,
                               color=color, label=label))

            if 'ax' in kwargs and isinstance(kwargs['ax'], np.ndarray):
                kwargs['ax'] = kwargs['ax'].flatten()[:len(y)]

            # automatic grid-like layout
            if 'layout' not in kwargs:
                rows = int((len(y) + 1) ** 0.5)
                cols = int(np.ceil(len(y) / rows))
                kwargs['layout'] = (rows, cols)
            else:
                rows, cols = kwargs['layout']

            if 'figsize' not in kwargs:
                # default, each axes subplot has some reasonable size
                kwargs['figsize'] = (cols * 4, rows * 3)

        else:  # mode (2) -- everything in a single axesplot
            if not kwargs.get('title', None):
                kwargs['title'] = self.name

        if isinstance(kwargs.get('ax', None), np.ndarray) and 'layout' in kwargs:
            # avoid matplotlib warning: when multiple axes are passed,
            # layout are ignored.
            del kwargs['layout']
        axes = mean.plot(*args, subplots=subplots, **kwargs)

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
                    if rolling:
                        df = df.rolling(rolling, min_periods=1, center=True).mean()
                    df.plot(ax=ax, x=x, y=yi, legend=False, label='',
                            color=color, alpha=runs_alpha)


        # some sensible styling (grid, tight_layout)
        axes_arr = np.asarray(axes).flat
        if kwargs.get('grid', True):
            for ax in axes_arr:
                ax.grid(which='both', alpha=0.5)
        fig = axes_arr[0].get_figure()
        fig.tight_layout()
        return axes


class HypothesisHvPlotter(HypothesisPlotter):

    def _do_plot(self,
                 y: List[str],
                 mean: pd.DataFrame,
                 std: pd.DataFrame,
                 n_samples: Optional[int],
                 subplots: bool,
                 rolling: Optional[int],
                 std_alpha: Optional[float],
                 runs_alpha: Optional[float],
                 args: List,
                 kwargs: Dict,
                 ):

        if subplots:
            kwargs['legend'] = False

            # TODO implement various options for hvplot.
            kwargs.update(dict(y=y))
            p = mean.hvplot(shared_axes=False, subplots=True,
                            **kwargs)

            # Display a single legend without duplication
            if isinstance(p.data, dict):
                next(iter(p.data.values())).opts('Curve', show_legend=True)
        else:
            # TODO implement this version
            raise NotImplementedError

        if std_alpha is not None:
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
                # single plot rather than multiple subplots:sp
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
        return p


class ExperimentPlotter:

    def __init__(self, experiment: 'Experiment'):
        self._parent = experiment

    @property
    def _hypotheses(self):
        return self._parent._hypotheses

    def __call__(self, *args, ax=None,
                 colors=None, **kwargs):
        '''
        Experiment.plot.

        @see DataFrame.plot()

        Args:
            subplots:
            y: (Str, Iterable[Str])
            colors: Iterable[Str]
        '''
        # TODO extract x-axis
        y = kwargs.get('y', None)
        if y is None:
            # draw all known columns by default
            y = set()
            for h in self._hypotheses.values():
                y.update(h.columns)
            y = list(y)
            if 'x' in kwargs:
                y = [yi for yi in y if yi != kwargs['x']]
            kwargs['y'] = y

        if colors is None:
            # len(colors) should equal hypothesis. or have as attributes
            from .colors import get_standard_colors
            colors = get_standard_colors(num_colors=len(self._hypotheses))

        hypothesis_labels = [name for name, _ in self._hypotheses.items()]
        if kwargs.get('prettify_labels', True):
            hypothesis_labels = util.prettify_labels(hypothesis_labels)

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
                    if kwargs.get('prettify_labels', True):
                        kwargs['label'] = util.prettify_labels(kwargs['label'])
                kwargs['subplots'] = True
            kwargs['color'] = colors[i]
            ax = hypo.plot(*args, ax=ax, **kwargs)    # on the same ax(es)?

        if isinstance(ax, Axes) and isinstance(y, str):
            ax.set_ylabel(kwargs['y'])

        return ax


HypothesisPlotter.__doc__ = HypothesisPlotter.__call__.__doc__
HypothesisHvPlotter.__doc__ = HypothesisHvPlotter.__call__.__doc__
ExperimentPlotter.__doc__ = ExperimentPlotter.__call__.__doc__
