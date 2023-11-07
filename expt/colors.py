"""Utility for color."""

import sys

_thismodule = sys.modules[__name__]

# populate all matplotlib named colors as module attributes.
# https://matplotlib.org/examples/color/named_colors.html
from matplotlib import colors as mcolors

_colors = {}
_colors.update(mcolors.BASE_COLORS)
_colors.update(mcolors.CSS4_COLORS)
_colors.update({
    k.replace(':', '_'): v  # e.g., tab:blue -> tab_blue
    for k, v in mcolors.TABLEAU_COLORS.items()
})

for color_name in _colors:
  setattr(_thismodule, color_name, color_name)

del sys
del _thismodule

# Matplotlib default cycle
MatplotlibDefault = (
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf',
)

# Sensible default color sets that are more distinguishable: 17 colors
# https://matplotlib.org/3.1.0/gallery/color/named_colors.html
ExptSensible17 = (
    'dimgray',
    'dodgerblue',
    'limegreen',
    'coral',
    'gold',
    'lightpink',
    'brown',
    'red',
    'purple',
    'green',
    'cadetblue',
    'burlywood',
    'royalblue',
    'violet',
    'lightseagreen',
    'yellowgreen',
    'sandybrown',
)

# 20 Distinct Colors Palette by Sasha Trubetskoy: 17 colors
# (Except for white, gray, and black that are quite invisible)
# https://sashamaps.net/docs/tools/20-colors/
Trubetskoy17 = (
    '#800000',  # Maroon (99.99%)
    '#4363d8',  # Blue (99.99%)
    '#ffe119',  # Yellow (99.99%)
    '#e6beff',  # Lavender (99.99%)
    '#f58231',  # Orange (99.99%)
    '#3cb44b',  # Green (99%)
    '#000075',  # Navy (99.99%)
    '#e6194b',  # Red (99%)
    '#46f0f0',  # Cyan (99%)
    '#f032e6',  # Magenta (99%)
    '#9a6324',  # Brown (99%)
    '#008080',  # Teal (99%)
    '#911eb4',  # Purple (95%*)
    '#aaffc3',  # Mint (99%)
    '#ffd8b1',  # Apiroct (95%)
    '#bcf60c',  # Lime (95%)
    '#fabed4',  # Pink (99%)
    '#808000',  # Olive (95%)
    '#fffac8',  # Beige (99%)
    #'#a9a9a9',
    #'#ffffff',
    #'#000000'
)

try:
  # pandas 1.2+
  from pandas.plotting._matplotlib.style import get_standard_colors
except ImportError:
  get_standard_colors = None

if get_standard_colors is None:
  try:
    # pandas >=1.0, <1.2
    from pandas.plotting._matplotlib.style import \
        _get_standard_colors  # type: ignore
  except ImportError:
    # pandas <1.0
    from pandas.plotting._style import _get_standard_colors  # type: ignore

  get_standard_colors = _get_standard_colors
