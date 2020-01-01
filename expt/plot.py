"""
Plotting behavior (matplotlib, hvplot, etc.) for expt.data
"""

import warnings
from typing import Union, Iterable, Iterator, Optional, List

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
    ) -> pd.DataFrame:

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
            df_interp = df.apply(lambda y_series: interpolate.interp1d(df.index, y_series, bounds_error=False)(x_samples))
            df_interp[x_column] = x_samples
            df_interp.set_index(x_column, inplace=True)
            df_interp_list.append(df_interp)

        # have individual interpolation data for each run cached (for the last call).
        self._df_interp_list = df_interp_list

        grouped = pd.concat(df_interp_list, sort=False).groupby(level=0)
        mean, std = grouped.mean(), grouped.std()

        return mean, std

    def __call__(self, *args,
                 subplots=True,
                 std_alpha=0.2, runs_alpha=None,
                 n_samples=None,
                 rolling=None,
                 **kwargs):
        '''
        Hypothesis.plot based on matplotlib.
        see DataFrame.plot() for available keyword arguments.

        This can work in two different modes:
        (1) Plot (several or all) columns as separate subplots (subplots=True)
        (2) Plot (several or all) columns in a single axesplot (subplots=False)

        Additional keyword arguments:
            - rolling (int): A window size for rolling and smoothing.
            - n_samples (int): If given, we subsample using n_samples number of
              equidistant points over the x axis. Values will be interpolated.
            - std_alpha (float): If not None, will show the 1-std range as a
              shaded area. Defaults 0.2 (enabled).
            - runs_alpha (float): If not None, will draw an individual line
              for each run. Defaults None (disabled), recommend value 0.2.
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
        y: Iterable[str] = kwargs.get('y', mean.columns)
        if isinstance(y, str): y = [y]
        if 'x' in kwargs:
            y = [yi for yi in y if yi != kwargs['x']]
        y = [yi for yi in y if mean.dtypes[yi].kind in ('i', 'f')]

        if rolling:
            mean = mean.rolling(rolling, min_periods=1, center=True).mean()
            std = std.rolling(rolling, min_periods=1, center=True).mean()

        # Fill-in sensible defaults.
        if subplots:  # mode (1) -- separate subplots
            title = list(y)
            color = [kwargs.get('color', '#1f77b4')] * len(y)
            label = util.prettify_labels([self.name or ''] * len(y))

            # TDOO: layout -> grid
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

        if std_alpha is not None:
            # show shadowed range of 1-std errors
            for ax, yi in zip(np.asarray(axes).flat, y):
                mean_line = ax.get_lines()[-1]
                ax.fill_between((mean - std)[yi].index,
                                (mean - std)[yi].values,
                                (mean + std)[yi].values,
                                color=mean_line.get_color(),
                                alpha=std_alpha)

        if runs_alpha and len(self.runs) > 1:
            # show individual runs
            for ax, yi in zip(np.asarray(axes).flat, y):
                x = kwargs.get('x', None)
                color = ax.get_lines()[-1].get_color()

                df_individuals = self._df_interp_list if n_samples \
                    else self._dataframes

                for df in df_individuals:
                    if rolling:
                        df = df.rolling(rolling, min_periods=1, center=True).mean()
                    df.plot(ax=ax, x=x, y=yi, legend=False, color=color,
                            alpha=runs_alpha)

        # some sensible styling (grid, tight_layout)
        axes_arr = np.asarray(axes).flat
        for ax in axes_arr:
            ax.grid(which='both', alpha=0.5)
        fig = axes_arr[0].get_figure()
        fig.tight_layout()
        return axes


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

        hypothesis_labels = util.prettify_labels(
            [name for name, _ in self._hypotheses.items()]
        )
        for i, (name, hypo) in enumerate(self._hypotheses.items()):
            if isinstance(y, str):
                # display different hypothesis over subplots:
                kwargs['label'] = hypothesis_labels[i]
                kwargs['subplots'] = False
                kwargs['title'] = y    # column name

            else:
                # display multiple columns over subplots:
                if y:
                    kwargs['label'] = util.prettify_labels(
                        [f'{y_i} ({name})' for y_i in y]
                    )
                kwargs['subplots'] = True
            kwargs['color'] = colors[i]
            ax = hypo.plot(*args, ax=ax, **kwargs)    # on the same ax(es)?

        if isinstance(ax, Axes) and isinstance(y, str):
            ax.set_ylabel(kwargs['y'])

        return ax
