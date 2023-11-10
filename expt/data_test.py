"""Tests for expt.data"""
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access

import itertools
import re
import sys
from typing import Any, cast, Dict

import numpy as np
import pandas as pd
import pytest

import expt.data
from expt.data import Experiment
from expt.data import Hypothesis
from expt.data import Run
from expt.data import RunList

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


class _TestBase:

  def setup_method(self, method):
    sys.stdout.write("\n")


@pytest.fixture
def runs_gridsearch() -> RunList:
  """12 runs, 6 hypotheses."""
  runs = []
  ALGO, ENV_ID = ["ppo", "sac"], ["halfcheetah", "hopper", "humanoid"]
  for (algo, env_id) in itertools.product(ALGO, ENV_ID):
    for seed in [0, 1]:
      data: Dict[str, Any] = {
          "reward": np.arange(100),
          "global_step": np.arange(0, 1000, 10),
          # all the hypotheses may not have the identical columns.
          f"{algo}_loss": np.zeros(100),
      }
      if seed == 0:
        data.update({
            f"{algo}_loss": np.zeros(100),
        })
      df = pd.DataFrame(data).set_index('global_step')
      df['global_step'] = list(df.index.values)
      run = Run(f"{algo}-{env_id}-seed{seed}", df)
      runs.append(run)
  return RunList(runs)


def _runs_gridsearch_config_fn(r: Run):
  config = {}
  config['algo'], config['env_id'], config['seed'] = r.name.split('-')
  config['common_hparam'] = 1
  return config


class TestRun(_TestBase):

  def test_creation(self):
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    r = Run("foo/bar", df=df)

    r2 = r.with_config(config=dict(conf_foo='bar'))
    assert r is not r2  # This is not in-place substitution.

    assert r2.name == r.name
    assert r2.path == r.path
    assert r2.config == dict(conf_foo='bar')

  def test_run_properties(self):
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})

    r = Run("/tmp/some-run", df=df)
    assert r.path == "/tmp/some-run"
    assert r.name == "some-run"
    assert list(r.columns) == ["a", "b"]

    config = dict(learning_rate=0.1, seed=1)
    r = Run("/tmp/some-run-with-config", df=df, config=config)
    assert r.config == config

  def test_run_summary(self):
    r = Run("foo", pd.DataFrame({"y": np.arange(100)}))
    df = r.summary()

    print(df)
    assert len(df) == 1
    assert df['name'][0] == 'foo'

    # name=False: 'run'


class TestRunList(_TestBase):

  @staticmethod
  def _fixture() -> RunList:
    return RunList(
        Run(
            "r%d" % i,
            pd.DataFrame({"x": [i]}),
            config={
                "type": "runlist_fixture",
                "i": i
            },
        ) for i in range(16))

  def test_basic_operations(self):
    # instantiation
    r0 = Run("r0", pd.DataFrame({"y": [1, 2, 3]}))
    r1 = Run("r1", pd.DataFrame({"y": [1, 4, 9]}))
    runs = RunList([r0, r1])
    print("runs =", runs)

    # unpacking, indexing
    r0_, r1_ = runs
    assert r0_ is r0 and r1_ is r1
    assert runs[0] is r0
    assert runs[-1] is r1

    # iteration
    assert list(runs) == [r0, r1]
    for i, r in enumerate(runs):
      print("r from iterable :", r)
      assert r == [r0, r1][i]

    assert RunList.of(runs) is runs  # no copy should be made

    # list-like operations: extend
    r2 = Run("r1", pd.DataFrame({"y": [2, 2, 2]}))
    runs.extend([r2])
    assert len(runs) == 3

  def test_slice(self):
    runs = self._fixture()  # [r0 ... r15]

    # slice (sublist) should be a RunList
    # pylint: disable=not-an-iterable
    o = V(runs[:5])
    assert isinstance(o, RunList)
    assert [r.name for r in o] == ["r0", "r1", "r2", "r3", "r4"]

    o = V(runs[-2:5:-2])
    assert isinstance(o, RunList)
    assert [r.name for r in o] == ["r14", "r12", "r10", "r8", "r6"]
    # pylint: enable=not-an-iterable

  def test_filter(self):
    runs = self._fixture()

    # basic operations
    filtered = V(runs.filter(lambda run: run.name in ["r1", "r7"]))
    assert len(filtered) == 2
    assert list(filtered) == [runs[1], runs[7]]

    # invalid argument (fn)
    with pytest.raises(TypeError):
      runs.filter([lambda run: True])  # type: ignore

    # filter by string (special case)
    filtered = V(runs.filter("r1*"))  # 1, 10-15
    assert len(filtered) == 7

    # filter by regex (search, not match)
    filtered = V(runs.filter(re.compile("(1|2)$")))
    assert [r.name for r in filtered] == ['r1', 'r2', 'r11', 'r12']

  def test_grep(self):
    runs = self._fixture()
    assert len(runs.grep("1")) == 7
    assert len(runs.grep("^1$")) == 0
    assert len(runs.grep("r[369]")) == 3
    assert list(runs.grep("0")) == [runs[0], runs[10]]
    assert len(runs.grep(re.compile(".*13$"))) == 1
    assert len(runs.grep("R", flags=re.IGNORECASE)) == 16

  def test_map(self):
    runs = self._fixture()
    t = V(runs.map(lambda run: run.name))
    assert t == ["r%d" % i for i in range(16)]

  def test_to_hypothesis(self):
    runs = self._fixture()
    h = V(runs.to_hypothesis(name="runlist"))
    assert isinstance(h, Hypothesis)
    assert h.name == "runlist"
    assert not (h.runs is runs)  # should make a new instance

    for i in range(16):
      assert h.runs[i] is runs[i]

    assert h.config == {"type": "runlist_fixture"}

  def test_groupby(self):
    runs = self._fixture()
    groups = dict(runs.groupby(lambda run: int(run.name[1:]) % 5))
    V(groups)

    assert len(groups) == 5
    assert isinstance(groups[0], Hypothesis)
    assert groups[0].runs.map(lambda run: run.name) == [
        "r0", "r5", "r10", "r15"
    ]

  def test_extract(self, runs_gridsearch: RunList):
    print(runs_gridsearch)
    df = runs_gridsearch.extract(
        r"(?P<algo>[\w]+)-(?P<env_id>[\w]+)-seed(?P<seed>[\d]+)")
    print(df)
    assert set(df.columns) == set(['algo', 'env_id', 'seed', 'run'])
    assert list(df['algo'].unique()) == ["ppo", "sac"]
    assert list(df['env_id'].unique()) == ["halfcheetah", "hopper", "humanoid"]  # yapf: disable
    assert list(df['seed'].unique()) == ['0', '1']

  def test_varied_config_keys(self, runs_gridsearch: RunList):
    runs = runs_gridsearch

    def config_fn(r: Run):
      config = {}
      config['algo'], config['env_id'], config['seed'] = r.name.split('-')
      config['common_opts'] = 'shared'
      config['common_opts_2'] = [64, 64]  # list is unhashable
      config['r_id'] = id(r)
      return config

    varied_config_keys = runs.varied_config_keys(
        config_fn=config_fn, excludelist=['seed'])

    print(f"varied_config_keys = {varied_config_keys}")
    assert 'foo' not in varied_config_keys
    assert 'seed' not in varied_config_keys  # excludelist
    assert varied_config_keys == ('algo', 'env_id', 'r_id')

  def test_to_dataframe_multiindex(self, runs_gridsearch: RunList):
    runs = runs_gridsearch

    # when there is no run.config: error
    with pytest.raises(ValueError, match="does not have config"):
      df = runs.to_dataframe()  # include_config=True

    # when there is no run.config but no include_config: OK, Use 0..N index
    df = runs.to_dataframe(include_config=False)
    print(df)
    assert list(df.index) == list(range(len(runs)))
    assert list(df.columns) == ['name', 'run']

    # with custom config_fn
    def _config_fn(run: Run):
      algorithm, env, seed = run.name.split('-')
      config: Dict[str, Any] = dict(
          algorithm=algorithm,
          env=env,
          seed=seed,
          common="common",
      )
      config['hidden_dims'] = ([64, 64, 64] if env == 'humanoid' \
                               else [64, 64])
      return config

    df = runs.to_dataframe(config_fn=_config_fn)
    print(df)
    # index: should exclude 'common'
    assert df.index.names == ['algorithm', 'env', 'hidden_dims']
    assert list(df.columns) == ['seed', 'name', 'run']  # in order!
    assert isinstance(df.run[0], Run)

    # additional option: as_hypothesis
    df = runs.to_dataframe(config_fn=_config_fn, as_hypothesis=True)
    print(df)
    assert list(df.columns) == ['hypothesis']  # no 'name'
    assert isinstance(df.hypothesis[0], Hypothesis)
    # Hypothesis.config exists?
    assert df.hypothesis[0].config == dict(
        algorithm='ppo',
        env='halfcheetah',
        hidden_dims=(64, 64),  # Note: list converted to tuple
    )
    assert df.hypothesis[0].name == (
        'algorithm=ppo; env=halfcheetah; hidden_dims=(64, 64)')

    # additional option: include_summary (reward)
    df = runs.to_dataframe(
        config_fn=_config_fn, as_hypothesis=True, include_summary=True)

    # Note: global_step is the name of the index in runs_gridsearch
    assert list(df.columns) == [
        'hypothesis', 'reward', 'ppo_loss', 'global_step', 'sac_loss'
    ]
    assert df.reward[0] == df.hypothesis[0].summary()['reward'][0]

    # Tests index_keys and index_excludelist
    df = runs.to_dataframe(config_fn=_config_fn, \
                           index_keys=['algorithm', 'common'])
    assert df.index.names == ['algorithm', 'common']
    df = runs.to_dataframe(config_fn=_config_fn, index_excludelist=['env'])
    assert df.index.names == ['algorithm', 'seed', 'hidden_dims']

  def test_to_dataframe_singlerun(self, runs_gridsearch: RunList):
    run = runs_gridsearch[0]
    run.config = {
        'foo': 1,
        'bar': 2,
    }

    runs = RunList([run])
    df = runs.to_dataframe()
    assert df.index.names == ['foo', 'bar']

  @pytest.mark.xfail
  def test_pivot_table(self, runs_gridsearch: RunList):
    runs = runs_gridsearch

    # TODO: Add config.
    df = runs.pivot_table()
    print(df)


class TestHypothesis(_TestBase):

  @staticmethod
  def _fixture(named_index=False):
    # yapf: disable
    h = Hypothesis.of(name="h", runs=[
        # represents y = 2x curve
        Run("r0", pd.DataFrame({
            "x": [0, 2, 4, 6, 8],
            "y": [0, 4, 8, 12, 16],
            "z": [{"v": 1}, {"v": 2}, {"v": 3}, {"v": 4}, {"v": 5}],
        })),
        # represents y = 2x + 1 curve
        Run("r1", pd.DataFrame({
            "x": [1, 3, 5, 7, 9],
            "y": [3, 7, 11, 15, 19],
            "z": [{"v": -1}, {"v": -2}, {"v": -3}, {"v": -4}, {"v": -5}],
        })),
    ])

    if named_index:
      for r in h.runs:
        # Duplicate 'x' (like global_step) as the index and a column.
        r.df['_index'] = r.df['x']
        r.df = r.df.set_index('_index')
        r.df.index.name = 'x'

    # yapf: enable
    return h

  def test_creation(self):
    # instance creation (with auto type conversion)
    h = Hypothesis("hy0", [])
    print(h)
    assert h.name == 'hy0'
    assert isinstance(h.runs, RunList)

    # factory
    r0 = Run("r0", pd.DataFrame({"y": [1, 2, 3]}))
    h = Hypothesis.of([r0])
    print(h)
    assert h.runs.to_list() == [r0]

    h = Hypothesis.of(r0)
    print(h)
    assert h.name == 'r0'
    assert h.runs.to_list() == [r0]

    def generator():
      for i in ["a", "b", "c"]:
        yield Run(i, df=pd.DataFrame())

    h = Hypothesis.of(generator(), name="generator")
    print(h)
    assert h.name == 'generator'
    assert len(h) == 3

  def test_plot_method(self):
    import expt.plot
    h = Hypothesis("h", [])
    assert h.plot.__doc__ == expt.plot.HypothesisPlotter.__doc__

  def test_properties(self):
    # test gropued, columns, mean, std, min, max, etc.
    pass

  def test_config(self):
    """Tests Hypothesis.config."""

    # Automatic mode
    r0 = Run("r0", pd.DataFrame(), config={"a": 1, "b": 2, "c": [3], "r0": 0})
    r1 = Run("r1", pd.DataFrame(), config={"a": 1, "b": 4, "c": [3], "r1": 1})
    h = Hypothesis.of(name="hello", runs=[r0, r1])
    assert h.config == {"a": 1, "c": [3]}  # see Hypothesis.extract_config
    assert cast(expt.data.RunConfig, r0.config)['r0'] == 0

    h = Hypothesis.of([r0])
    assert h.config == r0.config
    assert not (h.config is r0.config)

    h = Hypothesis.of([])
    assert h.config is None

    # Manual mode
    h = Hypothesis.of(name="hello", runs=[r0, r1], config=None)
    assert h.config is None

    h = Hypothesis.of(name="hello", runs=[r0, r1], config={"foo": "bar"})
    assert h.config == {"foo": "bar"}

  def test_summary(self):
    # (1) with unnamed index.
    r0 = Run("r0", pd.DataFrame({"x": np.arange(100), "y": np.arange(100)}))
    r1 = Run("r1", pd.DataFrame({"x": np.zeros(100), "y": np.arange(100)}))
    h = Hypothesis.of(name="hello", runs=[r0, r1])
    df = h.summary()

    print(df)
    assert len(df) == 1
    assert df['name'][0] == 'hello'
    # the summary columns should include "index"
    assert list(df.columns) == ['name', 'index', 'x', 'y']

    # The default behavior is average of last 10% rows.
    assert df['x'][0] == np.mean(range(90, 100)) / 2.0
    assert df['y'][0] == np.mean(range(90, 100))
    assert df['index'][0] == 99  # max(index)

    # (2) with named index ('global_step') with a same column name
    h: Hypothesis = self._fixture(named_index=True)
    assert h.columns == ['x', 'y', 'z']
    df = h.summary()
    print(df)
    assert len(df) == 1
    assert df['name'][0] == 'h'
    # the summary columns should include the name of index (`x`)
    # as well as other columns in the correct order
    assert list(df.columns) == ['name', 'x', 'y', 'z']
    assert np.isnan(df['z']).all()

    # (3) individual_runs mode
    df = h.summary(individual_runs=True)
    print(df)
    assert len(df) == len(h.runs)
    assert list(df.columns) == ['name', 'x', 'y', 'z']
    # summary for each of the individual runs
    # "x": [0, 2, 4, 6, 8]  or [1, 3, 5, 7, 9]
    np.testing.assert_array_almost_equal(df['x'].values, [8, 9])
    np.testing.assert_array_almost_equal(df['y'].values, [16, 19])
    assert np.isnan(df['z']).all()

  def test_resample(self):
    h: Hypothesis = self._fixture()
    h.config = {"kind": "resample"}

    h2 = h.resample("x", n_samples=91)

    # it should preserve other metadata (e.g., config)
    assert h2.name == h.name
    assert h2.config == h.config

    # all the dataframes should have the same length
    assert all(len(df) == 91 for df in h2._dataframes)

    # non-numeric columns will be preserved.
    # x is not a index but a normal column?
    assert h2.columns == ['x', 'y', 'z']

    # TODO: Errorneous case: when n_samples > #data points

  def test_interpolate(self):
    """Tests interpolate and subsampling when runs have different support."""
    h: Hypothesis = self._fixture()
    h.config = {"kind": "interpolate"}

    # (1) Test a normal case.
    h_interpolated = h.interpolate("x", n_samples=91)
    assert h_interpolated.name == h.name

    dfs_interp = h_interpolated._dataframes
    for df_interp in dfs_interp:
      assert df_interp.index.name == "x"

    # all the dataframes should have the same length
    assert len(dfs_interp[0]) == len(dfs_interp[1]) == 91
    # for the first dataframe, x=[8..9] should be missing; for the second one, x=[0..1]
    assert np.isnan(dfs_interp[0]['y'][dfs_interp[0].index > 8]).all()
    assert np.isnan(dfs_interp[1]['y'][dfs_interp[0].index < 1]).all()
    # non-numeric column (e.g., z) should be dropped
    assert h_interpolated.columns == ['y']
    if False:  # DEBUG
      print(list(zip(dfs_interp[0].index, dfs_interp[0]['y'])))

    # validate interpolation result
    np.testing.assert_allclose(
        np.array(dfs_interp[0].loc[dfs_interp[0].index <= 8, 'y']),
        np.array(dfs_interp[0].index[dfs_interp[0].index <= 8]) * 2)
    np.testing.assert_allclose(
        np.array(dfs_interp[1].loc[dfs_interp[1].index >= 1, 'y']),
        np.array(dfs_interp[1].index[dfs_interp[1].index >= 1]) * 2 + 1)

    # two individual runs now have the same support on the x-axis
    assert dfs_interp[0].index.min() == dfs_interp[1].index.min() == 0.0
    assert dfs_interp[0].index.max() == dfs_interp[1].index.max() == 9.0

    # it should preserve other metadata (e.g., config)
    assert h_interpolated.config == h.config

    # (2) Interpolate with index, no x parameters given
    h = h.apply(lambda df: df.set_index("x"))
    assert h.runs[0].df.index.name == 'x'
    assert h.runs[1].df.index.name == 'x'
    h_interpolated = h.interpolate(n_samples=91)

    # non-numeric column (e.g., z) should be dropped
    assert h_interpolated.columns == ['y']

    # index name should be preserved
    assert h_interpolated.runs[0].df.index.name == 'x'
    assert h_interpolated.runs[1].df.index.name == 'x'

    # (3) Invalid use
    with pytest.raises(ValueError, match="Unknown column"):
      h_interpolated = h.interpolate("unknown_index", n_samples=1000)

  def test_apply(self):
    h: Hypothesis = self._fixture()
    h.config = {"kind": "apply"}

    def _transform(df: pd.DataFrame) -> pd.DataFrame:
      df = df.copy()
      df['y'] = df['y'] / 4
      return df

    h2 = h.apply(_transform)
    assert h.name == h2.name
    print(h[0].df)
    print(h2[0].df)
    np.testing.assert_array_equal(h[0].df['y'], [0, 4, 8, 12, 16])
    np.testing.assert_array_equal(h2[0].df['y'], [0, 1, 2, 3, 4])

    # it should preserve other metadata (e.g., config)
    assert h2.config == h.config


class TestExperiment(_TestBase):

  def test_create_simple(self):
    ex = Experiment("only_name")
    assert len(ex.hypotheses) == 0
    assert ex.name == "only_name"

    h = TestHypothesis._fixture()
    ex = Experiment("one_hypo", [h])
    assert len(ex.hypotheses) == 1
    assert ex.name == "one_hypo"

    assert ex._summary_columns is None
    assert list(ex._df.columns) == ["hypothesis", "x", "y", "z"]

  def test_create_from_dataframe_run(self, runs_gridsearch: RunList):
    """Tests Experiment.from_dataframe with the minimal defaults."""

    df = runs_gridsearch.to_dataframe(include_config=False)
    ex = V(Experiment.from_dataframe(df))
    # Each run becomes an individual hypothesis.
    assert len(ex.hypotheses) == 12

  def test_create_from_dataframe_run_multicolumns(self,
                                                  runs_gridsearch: RunList):
    """Tests Experiment.from_dataframe where the dataframe consists of
      `runs` and multiple other columns (via RunList.extract)."""

    # Reuse the fixture data from testRunListExtract.
    df = runs_gridsearch.extract(
        r"(?P<algo>[\w]+)-(?P<env_id>[\w]+)-seed(?P<seed>[\d]+)")
    assert 'run' in df.columns and 'hypothesis' not in df.columns
    assert set(df.columns) == set(["algo", "env_id", "seed", "run"])

    # If `by` is missing, cannot be automatically determined.
    with pytest.raises(
        ValueError, match=re.escape("Candidates: ['algo', 'env_id', 'seed']")
    ):  # yapf: disable
      Experiment.from_dataframe(df)

    # implicit groupby via from_dataframe
    # yapf: disable
    ex = Experiment.from_dataframe(df, by="algo", name="Exp.foobar")
    assert len(ex.hypotheses) == 2
    assert list(V(ex["ppo"].runs)) == [r for r in runs_gridsearch if "ppo" in r.name]
    assert list(V(ex["sac"].runs)) == [r for r in runs_gridsearch if "sac" in r.name]
    assert ex.name == "Exp.foobar"
    assert list(ex._df.index.names) == ['algo', 'name']  # Note the order
    # yapf: enable

    # All other columns than by = "algo"
    assert ex._summary_columns == ("env_id", "seed")

    # by: not exists?
    with pytest.raises(KeyError):  # TODO: Improve exception
      ex = Experiment.from_dataframe(df, by="unknown", name="Exp.foobar")

  def test_create_from_dataframe_run_general(self, runs_gridsearch: RunList):
    """Experiment.from_dataframe() from a DataFrame of `run`s, by grouping
       the runs into Hypothesis by (algorithm, env), i.e. only averaging
       over random seed."""

    df = runs_gridsearch.extract(
        r"(?P<algo>[\w]+)-(?P<env_id>[\w]+)-seed(?P<seed>[\d]+)")
    assert 'run' in df.columns and 'hypothesis' not in df.columns

    # Default: group by ['algo', 'env_id']
    ex = V(Experiment.from_dataframe(df, by=["algo", "env_id"], name="A"))
    assert len(ex.hypotheses) == 6
    assert isinstance(ex.hypotheses[0].name, str)
    assert ex.hypotheses[0].name == repr({
        'algo': 'ppo',
        'env_id': 'halfcheetah'
    })

    # Group by ['env_id', 'algo'] and use a custom namer
    namer = lambda t, _: f"{t['env_id']}-{t['algo']}"
    ex = V(Experiment.from_dataframe(
        df, by=["env_id", "algo"], hypothesis_namer=namer, name="B"))  # yapf: disable
    assert ex.hypotheses[0].name == 'halfcheetah-ppo'
    assert len(ex.hypotheses) == 6

  def test_create_from_dataframe_hypothesis(self, runs_gridsearch: RunList):
    """Create Experiment from a dataframe that has a `hypothesis` column."""

    def config_fn(r: Run):
      config = {}
      config['algo'], config['env_id'], config['seed'] = r.name.split('-')
      return config

    df = runs_gridsearch.to_dataframe(
        as_hypothesis=True,
        config_fn=config_fn,
        include_config=True,
    )
    assert 'hypothesis' in df.columns and 'run' not in df.columns
    assert df.shape[0] == 6  # 6 hypotheses

    # use default parameters to create ex
    ex = V(Experiment.from_dataframe(df))
    self._validate_ex_gridsearch(ex)

    # an incorrect use of `by`?
    with pytest.raises(ValueError, match='does not have a column'):
      ex = Experiment.from_dataframe(df, by="algo")

  def test_create_from_runs(self, runs_gridsearch: RunList):
    """Tests Experiment.from_runs with the minimal defaults."""

    # Uses default for config_keys: see varied_config_keys.
    ex = Experiment.from_runs(
        runs_gridsearch,
        config_fn=_runs_gridsearch_config_fn,
        name="ex_from_runs",
    )
    assert ex.name == "ex_from_runs"
    assert ex._config_keys == ['algo', 'env_id']  # no 'common_hparam'
    self._validate_ex_gridsearch(ex)

  def _validate_ex_gridsearch(self, ex: Experiment):
    assert len(ex.hypotheses) == 6
    hypothesis_names = [
        # Note that the default hypothesis_namer is used
        'algo=ppo; env_id=halfcheetah',
        'algo=ppo; env_id=hopper',
        'algo=ppo; env_id=humanoid',
        'algo=sac; env_id=halfcheetah',
        'algo=sac; env_id=hopper',
        'algo=sac; env_id=humanoid',
    ]
    assert [ex.hypotheses[i].name for i in range(6)] == hypothesis_names

    # Use custom name for some hypotheses
    for h in ex.hypotheses:
      assert h.config is not None
      h.name = f"{h.config['algo']} ({h.config['env_id']})"

    # Because df already has a multi-index, so should ex.
    assert ex._df.index.names == ['algo', 'env_id', 'name']  # Note the order
    assert list(ex._df.index.get_level_values('name')) == [
        'ppo (halfcheetah)',
        'ppo (hopper)',
        'ppo (humanoid)',
        'sac (halfcheetah)',
        'sac (hopper)',
        'sac (humanoid)',
    ]
    assert list(ex._df.index.get_level_values('algo')) == (  # ...
        ['ppo'] * 3 + ['sac'] * 3)
    assert list(ex._df.index.get_level_values('env_id')) == (
        ['halfcheetah', 'hopper', 'humanoid'] * 2)

  def test_add_inplace(self, runs_gridsearch: RunList):
    """Tests Experiment.add_runs() and Experiment.add_hypothesis()."""
    ex = Experiment("dummy")
    ex.add_runs("halfcheetah", [runs_gridsearch[0], runs_gridsearch[1]])
    ex.add_runs("hopper", runs_gridsearch[2:4])

    assert len(ex.hypotheses) == 2
    assert isinstance(ex['halfcheetah'], Hypothesis)
    assert isinstance(ex['hopper'], Hypothesis)
    assert ex[0].name == 'halfcheetah'
    assert ex[1].name == 'hopper'

    ex = Experiment("dummy")
    h0 = Hypothesis.of(runs_gridsearch[0:2], name='halfcheetah')
    h1 = Hypothesis.of(runs_gridsearch[2:4], name='hopper')
    ex.add_hypothesis(h0)
    ex.add_hypothesis(h1)

    assert len(ex.hypotheses) == 2
    assert isinstance(ex['halfcheetah'], Hypothesis)
    assert isinstance(ex['hopper'], Hypothesis)
    assert ex[0].name == 'halfcheetah'
    assert ex[1].name == 'hopper'

  def test_indexing(self):
    """Tests __getitem__."""

    # pylint: disable=unsubscriptable-object
    h0 = Hypothesis("hyp0", Run('r0', pd.DataFrame({"a": [1, 2, 3]})))
    h1 = Hypothesis("hyp1", Run('r1', pd.DataFrame({"a": [4, 5, 6]})))

    ex = Experiment(name="ex", hypotheses=[h0, h1])

    # get hypothesis by name or index
    assert V(ex[0]) == h0
    assert V(ex[1]) == h1
    with pytest.raises(IndexError):
      V(ex[2])
    assert V(ex["hyp0"]) == h0
    with pytest.raises(KeyError):
      V(ex["hyp0-not"])

    # nested index
    r = V(ex["hyp0", 'a'])
    assert isinstance(r, pd.DataFrame)
    # assert r is ex["hyp0"]['a']             # TODO
    assert list(r['r0']) == [1, 2, 3]

    # fancy index
    # ----------------
    # (1) by integer index
    r = V(ex[[0, 1]])
    assert r[0] is h0 and r[1] is h1
    # (2) by name?
    r = V(ex[['hyp1', 'hyp0']])
    assert r[0] is h1 and r[1] is h0
    # (3) boolean index (select)
    r = V(ex[[False, True]])
    assert len(r) == 1 and r[0] is h1
    with pytest.raises(IndexError):
      ex[[False, True, False]]
    # (4) non-standard iterable
    r = V(ex[pd.Series([1, 0])])
    assert r[0] is h1 and r[1] is h0

    with pytest.raises(NotImplementedError):  # TODO
      r = V(ex[['hyp1', 'hyp0'], 'a'])
    # pylint: enable=unsubscriptable-object

  def test_with_config_keys(self, runs_gridsearch: RunList):
    ex_base = Experiment.from_runs(
        runs_gridsearch,
        config_fn=_runs_gridsearch_config_fn,
        config_keys=['algo', 'env_id', 'common_hparam'],
    )

    assert ex_base._config_keys == ['algo', 'env_id', 'common_hparam']
    assert ex_base._df.index.get_level_values( \
        'algo').tolist() == ['ppo', 'ppo', 'ppo', 'sac', 'sac', 'sac']
    assert ex_base._df.index.get_level_values( \
        'env_id').tolist() == ['halfcheetah', 'hopper', 'humanoid'] * 2

    def validate_env_id_algo(ex):
      assert ex._summary_columns == ex_base._summary_columns
      assert set(ex._hypotheses.values()) == set(ex_base._hypotheses.values())

      # rows should be sorted according to the new multiindex
      assert ex._df.index.get_level_values('env_id').tolist() == [
          'halfcheetah', 'halfcheetah', \
          'hopper', 'hopper', \
          'humanoid', 'humanoid',
      ]
      assert ex._df.index.get_level_values('common_hparam').tolist() == [1] * 6
      assert ex._df.index.get_level_values('algo').tolist() == [
          'ppo', 'sac', 'ppo', 'sac', 'ppo', 'sac'
      ]

    # (1) full reordering
    ex1 = ex_base.with_config_keys(['env_id', 'common_hparam', 'algo'])
    assert ex1._config_keys == ['env_id', 'common_hparam', 'algo']
    validate_env_id_algo(ex1)

    # (2) partial with ellipsis
    ex2 = ex_base.with_config_keys(['env_id', ...])
    assert ex2._config_keys == ['env_id', 'algo', 'common_hparam']
    validate_env_id_algo(ex2)

    # (3) partial subset. TODO: Things to decide:
    # - To reduce or not to reduce?
    # - Hypothesis objects should remain the same or changes in
    #   name, config, etc.?

    # (4) not existing keys: error
    with pytest.raises(ValueError, \
                       match="'foo' not found in the config of") as e:
      ex_base.with_config_keys(['env_id', 'foo', 'algo'])

  def test_select_top(self):
    # yapf: disable
    hypos = [
        Hypothesis(f"hyp{i}", Run(f'r{i}', pd.DataFrame({
            "score": [i * 100],  # the greater, the better (argmax: 4)
            "loss": -i,          # the lower,   the better (argmax: 4)
        })))
        for i in range(5)
    ]
    ex = Experiment(name="ex", hypotheses=hypos)
    # yapf: enable

    # top-1
    assert ex.select_top("score") is hypos[4]
    assert ex.select_top("loss", descending=False) is hypos[4]

    # top-K as list
    assert ex.select_top("score", k=2) == [hypos[4], hypos[3]]
    assert ex.select_top("score", descending=False, k=3,
                         ) == [hypos[0], hypos[1], hypos[2]]  # yapf: disable

    # invalid inputs
    with pytest.raises(ValueError, match='k must be greater than 0'):
      ex.select_top("score", k=0)
    with pytest.raises(ValueError, match='k must be smaller than the number of hypotheses'):  # yapf: disable
      ex.select_top("score", k=6)

  def test_resample(self):
    h: Hypothesis = TestHypothesis._fixture()
    h.config = {"kind": "subsample"}

    ex = Experiment(
        name="interp_test", hypotheses=[h], summary_columns=('x', 'y'))
    ex2 = ex.resample(n_samples=91)
    assert ex2.hypotheses[0]._dataframes[0].__len__() == 91
    assert ex2.hypotheses[0]._dataframes[1].__len__() == 91

    assert ex2._config_keys == ex._config_keys
    assert ex2._summary_columns == ex._summary_columns

  def test_interpolate(self):
    h: Hypothesis = TestHypothesis._fixture()
    h.config = {"kind": "interpolate"}

    ex = Experiment(
        name="interp_test", hypotheses=[h], summary_columns=('x', 'y'))
    ex_interpolated = ex.interpolate("x", n_samples=91)
    assert ex_interpolated.hypotheses[0]._dataframes[0].index.name == 'x'
    assert ex_interpolated.hypotheses[0]._dataframes[1].index.name == 'x'
    assert ex_interpolated.hypotheses[0]._dataframes[0].__len__() == 91
    assert ex_interpolated.hypotheses[0]._dataframes[1].__len__() == 91

    assert ex_interpolated._config_keys == ex._config_keys
    assert ex_interpolated._summary_columns == ex._summary_columns == ('x', 'y')

  def test_select_query(self, runs_gridsearch: RunList):
    """Tests Experiment.select()"""

    df = runs_gridsearch.to_dataframe(
        as_hypothesis=True,
        config_fn=_runs_gridsearch_config_fn,
        include_config=True,
    )
    base_ex = Experiment.from_dataframe(df)
    assert len(base_ex.hypotheses) == 6  # [ppo, sac] * (3 envs)

    # Create a sub-view of Experiment by applying the query.
    ex = base_ex.select('env_id == "halfcheetah"')
    assert len(ex.hypotheses) == 2
    assert ex.hypotheses[0].config['env_id'] == "halfcheetah"  # type: ignore
    assert ex.hypotheses[1].config['env_id'] == "halfcheetah"  # type: ignore

    assert ex._df.index.names == base_ex._df.index.names
    assert list(ex._df.index.get_level_values('name')) == [
        'algo=ppo; env_id=halfcheetah',
        'algo=sac; env_id=halfcheetah',
    ]
    # the underlying hypothesis objects must be the same.
    assert set(ex.hypotheses) == set(h for h in base_ex.hypotheses if \
                                     'halfcheetah' in h.name)

  def test_select_fn(self, runs_gridsearch: RunList):
    """Tests Experiments.select()"""

    df = runs_gridsearch.to_dataframe(
        as_hypothesis=True,
        config_fn=_runs_gridsearch_config_fn,
        include_config=True,
    )
    base_ex = Experiment.from_dataframe(df)
    assert len(base_ex.hypotheses) == 6  # [ppo, sac] * (3 envs)

    ex = base_ex.select(lambda h: 'humanoid' in h.name)
    assert len(ex.hypotheses) == 2
    assert ex.hypotheses[0].config['env_id'] == "humanoid"  # type: ignore
    assert ex.hypotheses[1].config['env_id'] == "humanoid"  # type: ignore

    ex = base_ex.select(lambda h: True)
    assert len(ex.hypotheses) == len(base_ex.hypotheses)

    ex = base_ex.select(lambda h: h.name == 'algo=sac; env_id=halfcheetah')
    assert len(ex.hypotheses) == 1

    # error case
    with pytest.raises(
        TypeError,
        match=('The filter function must return bool, '
               'but unexpected data type found: float64')):
      ex = base_ex.select(
          lambda h: np.asarray(1.0, dtype=np.float64))  # type: ignore

  def test_plot_method(self):
    import expt.plot
    ex = Experiment("ex", [])
    assert ex.plot.__doc__ == expt.plot.ExperimentPlotter.__doc__


if __name__ == '__main__':
  sys.exit(pytest.main(["-s", "-v"] + sys.argv))
