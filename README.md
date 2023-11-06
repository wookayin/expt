# expt

Experiment. Plot. Tabulate.

<p align="center">
  <img src="https://raw.githubusercontent.com/wookayin/expt/master/examples/demo-figure.png" width="640" />
</p>

`expt` is a small Python library that helps you draw time-series plots from a tabular data.
Typical use cases are drawing learning curves for machine learning experiments.
`expt` aim to provide a minimal API for doing most of common plotting needs with sensible defaults and useful batteries included.


### Features:

- Parsing of CSV or TensorBoard eventfiles, in parallel
- Easy plotting of individual runs, hypotheses, or experiments.


Usage
-----

See [a demo ipython notebook][demo-notebook], until an user guide and API documentation arrive.

[demo-notebook]: https://github.com/wookayin/expt/blob/master/examples/quick-tour.ipynb

Installation:
```
pip install expt
pip install git+https://github.com/wookayin/expt@master        # Latest development version
```


License
-------

MIT License (c) 2019-2023 Jongwook Choi
