"""Tests for expt.data"""

import itertools
import re
import sys

import numpy as np
import pandas as pd
import pytest

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


class _TestBase:

  def setup_method(self, method):
    sys.stdout.write("\n")


@pytest.fixture
def runs_gridsearch() -> RunList:
  runs = []
  ALGO, ENV_ID = ["ppo", "sac"], ["halfcheetah", "hopper", "humanoid"]
  for (algo, env_id) in itertools.product(ALGO, ENV_ID):
    for seed in [0, 1]:
      df = pd.DataFrame({"reward": np.arange(100)})
      runs.append(Run(f"{algo}-{env_id}-seed{seed}", df))
  return RunList(runs)


class TestRun(_TestBase):

  def testRunProperties(self):
    r = Run("/tmp/some-run",
            pd.DataFrame({
                "a": [1, 2, 3, 4],
                "b": [5, 6, 7, 8],
            }))

    assert r.path == "/tmp/some-run"
    assert r.name == "some-run"
    assert list(r.columns) == ["a", "b"]


class TestRunList(_TestBase):

  def _fixture(self) -> RunList:
    return RunList(Run("r%d" % i, pd.DataFrame({"x": [i]})) for i in range(16))

  def testRunList(self):
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

  def testRunListSlice(self):
    runs = self._fixture()

    # slice (sublist) should be a RunList
    o = V(runs[:5])
    assert isinstance(o, RunList)
    assert [r.name for r in o] == ["r0", "r1", "r2", "r3", "r4"]

    o = V(runs[-2:5:-2])
    assert isinstance(o, RunList)
    assert [r.name for r in o] == ["r14", "r12", "r10", "r8", "r6"]

  def testRunListFilter(self):
    runs = self._fixture()

    # basic operations
    filtered = V(runs.filter(lambda run: run.name in ["r1", "r7"]))
    assert len(filtered) == 2
    assert list(filtered) == [runs[1], runs[7]]

    # invalid argument (fn)
    with pytest.raises(TypeError):
      runs.filter([lambda run: True])

    # filter by string (special case)
    filtered = V(runs.filter("r1*"))  # 1, 10-15
    assert len(filtered) == 7

  def testRunListGrep(self):
    runs = self._fixture()
    assert len(runs.grep("1")) == 7
    assert len(runs.grep("^1$")) == 0
    assert len(runs.grep("r[369]")) == 3
    assert list(runs.grep("0")) == [runs[0], runs[10]]
    assert len(runs.grep(re.compile(".*13$"))) == 1
    assert len(runs.grep("R", flags=re.IGNORECASE)) == 16

  def testRunListMap(self):
    runs = self._fixture()
    t = V(runs.map(lambda run: run.name))
    assert t == ["r%d" % i for i in range(16)]

  def testRunListToHypothesis(self):
    runs = self._fixture()
    h = V(runs.to_hypothesis(name="runlist"))
    assert isinstance(h, Hypothesis)
    assert h.name == "runlist"
    assert not (h.runs is runs)  # should make a new instance

    for i in range(16):
      assert h.runs[i] is runs[i]

  def testRunListGroupby(self):
    runs = self._fixture()
    groups = dict(runs.groupby(lambda run: int(run.name[1:]) % 5))
    V(groups)

    assert len(groups) == 5
    assert isinstance(groups[0], Hypothesis)
    assert groups[0].runs.map(lambda run: run.name) == [
        "r0", "r5", "r10", "r15"
    ]

  def testRunListExtract(self, runs_gridsearch: RunList):
    print(runs_gridsearch)
    df = runs_gridsearch.extract(
        r"(?P<algo>[\w]+)-(?P<env_id>[\w]+)-seed(?P<seed>[\d]+)")
    print(df)
    assert set(df.columns) == set(['algo', 'env_id', 'seed', 'run'])
    assert list(df['algo'].unique()) == ["ppo", "sac"]
    assert list(df['env_id'].unique()) == ["halfcheetah", "hopper", "humanoid"]  # yapf: disable
    assert list(df['seed'].unique()) == ['0', '1']


class TestHypothesis(_TestBase):

  def testHypothesisCreation(self):
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

  def testHypothesisData(self):
    # test gropued, columns, mean, std, min, max, etc.
    pass

  def testHypothesisInterpolate(self):
    """Tests interpolate and subsampling when runs have different support."""
    # yapf: disable
    h = Hypothesis.of(name="h", runs=[
        # represents y = 2x curve
        Run("r0", pd.DataFrame({"x": [0, 2, 4, 6, 8],
                                "y": [0, 4, 8, 12, 16],
                                "z": [{}, {}, {}, {}, {}],
                                })),
        # represents y = 2x + 1 curve
        Run("r1", pd.DataFrame({"x": [1, 3, 5, 7, 9],
                                "y": [3, 7, 11, 15, 19],
                                "z": [{}, {}, {}, {}, {}],
                                })),
    ])
    # yapf: enable

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
    assert h_interpolated.columns == ["y"]
    if False:  # DEBUG
      print(list(zip(dfs_interp[0].index, dfs_interp[0]['y'])))

    # validate interpolation result
    np.testing.assert_allclose(
        np.array(dfs_interp[0].loc[dfs_interp[0].index <= 8, 'y']),
        np.array(dfs_interp[0].index[dfs_interp[0].index <= 8]) * 2)
    np.testing.assert_allclose(
        np.array(dfs_interp[1].loc[dfs_interp[1].index >= 1, 'y']),
        np.array(dfs_interp[1].index[dfs_interp[1].index >= 1]) * 2 + 1)

    # (2) Invalid use
    with pytest.raises(ValueError, match="Unknown column"):
      h_interpolated = h.interpolate("unknown_index", n_samples=1000)


class TestExperiment(_TestBase):

  def testFromDataFrameDefault(self, runs_gridsearch: RunList):
    """Tests Experiment.from_dataframe with the minimal defaults."""
    df = runs_gridsearch.to_dataframe()
    ex = Experiment.from_dataframe(df)
    # Each run becomes an individual hypothesis.
    assert len(ex.hypotheses) == 12

  def testFromDataFrameMulticolumns(self, runs_gridsearch: RunList):
    """Tests Experiment.from_dataframe where the dataframe consists of
        multiple columns (usually parsed via RunList.extract)."""
    # Reuse the fixture data from testRunListExtract.
    df = runs_gridsearch.extract(
        r"(?P<algo>[\w]+)-(?P<env_id>[\w]+)-seed(?P<seed>[\d]+)")
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
    # yapf: enable

  def testFromDataFrameGeneral(self, runs_gridsearch: RunList):
    """Organize a RunList into Hypothesis grouped by (algorithm, env),
        i.e. only averaging over random seed."""
    df = runs_gridsearch.extract(
        r"(?P<algo>[\w]+)-(?P<env_id>[\w]+)-seed(?P<seed>[\d]+)")

    # Default: group by ['algo', 'env_id']
    ex = V(Experiment.from_dataframe(df, by=["algo", "env_id"], name="A"))
    assert len(ex.hypotheses) == 6
    assert isinstance(ex.hypotheses[0].name, str)
    assert ex.hypotheses[0].name == repr(('ppo', 'halfcheetah'))

    # Group by ['env_id', 'algo'] and use a custom namer
    namer = lambda t: f"{t[0]}-{t[1]}"
    ex = V(Experiment.from_dataframe(
            df, by=["env_id", "algo"], hypothesis_namer=namer, name="B"))  # yapf: disable
    assert ex.hypotheses[0].name == 'halfcheetah-ppo'
    assert len(ex.hypotheses) == 6

  def testExperimentIndexing(self):
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

  def testTopK(self):
    # yapf: disable
    hypos = [
        Hypothesis(f"hyp{i}", Run(f'r{i}', pd.DataFrame({
            "score": [i * 100],  # the greater, the better (argmax: 4)
            "loss" : -i,         # the lower,   the better (argmax: 4)
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


if __name__ == '__main__':
  sys.exit(pytest.main(["-s", "-v"] + sys.argv))
