"""Tests for expt.data_loader."""
# pylint: disable=protected-access

import functools
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import NamedTuple, Type
import urllib.request
import warnings

import numpy as np
import pandas as pd
import pytest

from expt import data
from expt import data_loader
from expt import util

__PATH__ = os.path.abspath(os.path.dirname(__file__))
FIXTURE_PATH = Path(os.path.join(__PATH__, '../fixtures'))

# Suppress and silence TF warnings.
# such as, cpu_utils.cc: Failed to get CPU frequency: 0 Hz
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def _setup_fixture():
  """Set up example fixture data."""
  URL = "https://tensorboard.dev/static/log_set-2021-03-11.zip"

  if (FIXTURE_PATH / "lr_1E-03,conv=1,fc=2").exists():
    return

  print(f"Downloading and extracting from {URL} ...")
  with urllib.request.urlopen(URL) as z:
    with tempfile.NamedTemporaryFile() as tfile:
      tfile.write(z.read())
      tfile.seek(0)
      shutil.unpack_archive(tfile.name, FIXTURE_PATH, format='zip')


@pytest.mark.filterwarnings('ignore:.*parse_run.*:DeprecationWarning')
class TestParseRun:

  @classmethod
  def setup_class(cls):
    _setup_fixture()

  @pytest.fixture
  def path_tensorboard(self) -> Path:
    return FIXTURE_PATH / "lr_1E-03,conv=1,fc=2"

  @pytest.fixture
  def path_csv(self) -> Path:
    return FIXTURE_PATH / "sample_csv"

  def test_parse_cannot_handle(self, path_csv, path_tensorboard, tmp_path):
    """Tests incompatible logdir format and parser."""

    with pytest.raises(data_loader.CannotHandleException):
      data_loader.CSVLogReader(path_tensorboard)

    with pytest.raises(data_loader.CannotHandleException):
      data_loader.TensorboardLogReader(path_csv)

    with pytest.raises(data_loader.CannotHandleException):
      data_loader.RustTensorboardLogReader(path_csv)

    with pytest.raises(data_loader.CannotHandleException):
      data_loader.parse_run_tensorboard(path_csv)

    with pytest.raises(data_loader.CannotHandleException):
      data_loader.parse_run_progresscsv(path_tensorboard)

    with pytest.raises(data_loader.CannotHandleException):
      data_loader.parse_run(tmp_path, verbose=True)

  def test_logreaders_throw_when_path_not_exist(self):
    # Any LogReaders should raise FileNotFoundError, not CannotHandleException
    for cls in data_loader.DEFAULT_READER_CANDIDATES:
      with pytest.raises(FileNotFoundError):
        data_loader.parse_run(FIXTURE_PATH / "DOES-NOT-EXIST", reader_cls=cls)

  def test_parser_detection(self, path_csv, path_tensorboard, tmp_path):
    """Tests automatic parser resolution."""

    p = data_loader._get_reader_for(path_tensorboard)
    assert isinstance(p, (data_loader.TensorboardLogReader,
                          data_loader.RustTensorboardLogReader))
    assert p.log_dir == str(path_tensorboard)

    p = data_loader._get_reader_for(path_csv)
    assert isinstance(p, data_loader.CSVLogReader)
    assert p.log_dir == str(path_csv)

    with pytest.raises(data_loader.CannotHandleException):
      p = data_loader._get_reader_for(tmp_path)

  def test_parse_progresscsv(self, path_csv):
    df: pd.DataFrame = data_loader.parse_run(path_csv)

    assert len(df) >= 50
    np.testing.assert_array_equal(df.columns, [
        'initial_reset_time',
        'episode_rewards',
        'episode_lengths',
        'episode_end_times',
    ])

    # DataFrame from CSV has a consecutive, unnamed index.
    assert df.index.name is None
    assert list(df.index.values) == list(range(df.shape[0]))

  def test_parse_tensorboard_py(self, path_tensorboard):
    # via either rust or python tensorboard
    df: pd.DataFrame = data_loader.parse_run(
        path_tensorboard, reader_cls=data_loader.TensorboardLogReader)
    assert len(df) >= 400
    np.testing.assert_array_equal(
        df.columns,
        ['accuracy/accuracy', 'global_step', 'xent/xent_1'],
    )

    # index: This tensorboard log has a period of 5, so
    # the DataFrame's index is not necessarily of consecutive integers.
    # Note: should have the same result as the rust version.
    assert df.index.name == 'global_step'
    np.testing.assert_array_equal(
        df.index,
        np.arange(0, 2000 + 1, 5),
    )

  def test_parse_tensorboard_incremental_read(self, path_tensorboard):
    r = data_loader.TensorboardLogReader(path_tensorboard)

    ctx = r.new_context()
    print("ctx[0]: last_read_rows =", ctx.last_read_rows)
    assert ctx.last_read_rows == 0

    # Assume as if data were streaming.
    r._iter_scalar_summary_from = functools.partial(  # type: ignore
        r._iter_scalar_summary_from, limit=300)

    ctx = r.read(ctx)
    print("ctx[1]: last_read_rows =", ctx.last_read_rows)
    assert ctx.last_read_rows == 304  # 300 + 2 + 2 from three eventfiles

    # Incremental read: Subsequent read() call should have no effect and fast.
    ctx = r.read(ctx)
    print("ctx[2]: last_read_rows =", ctx.last_read_rows)
    assert ctx.last_read_rows == 103  # Remaning reads

    df = r.result(ctx)
    assert len(df) >= 400
    np.testing.assert_array_equal(
        df.columns, ['accuracy/accuracy', 'global_step', 'xent/xent_1'])

    # The result should be identical as non-incremental read.
    df_ref = data_loader.TensorboardLogReader(path_tensorboard).read_once()
    assert np.all(df == df_ref)

  @pytest.mark.skipif(
      importlib.util.find_spec("expt._internal") is None,
      reason="The rust extension is not available")
  def test_parse_tensorboard_fast_with_rust(self):
    parser = data_loader.RustTensorboardLogReader(
        Path(FIXTURE_PATH) / "lr_1E-03,conv=1,fc=2")
    ctx = parser.read(parser.new_context())
    df: pd.DataFrame = parser.result(ctx)

    assert len(df) >= 400
    np.testing.assert_array_equal(
        sorted(df.columns),
        sorted(['accuracy/accuracy', 'global_step', 'xent/xent_1']),
    )

    # Note: should have the same result as the python version.
    assert df.index.name == 'global_step'
    np.testing.assert_array_equal(
        df.index,
        np.arange(0, 2000 + 1, 5),
    )


class TestGetRunsRemote:
  """Tests reading runs from a remote machine over SSH/SFTP.

  A test SSH server is assumed to have a (git clone) copy of the `expt` repo
  at home: i.e., the path being `~/expt`. Examples of path:

    - sftp://{hostname}/expt/fixtures/sample_csv
    - sftp://{hostname}/expt/fixtures/lr_1E-03,conv=1,fc=2
  """

  @property
  def paths(self):
    hostname = os.environ.get("EXPT_SSH_HOST")
    if not hostname:
      pytest.skip("Requires SSH host setup for test")

    # Absolute and normalized path, which is assumed to be accessible over ssh.
    p = FIXTURE_PATH.resolve()

    # see _setup_fixture()
    return {
        "sftp": f"sftp://{hostname}/{p}/sample_csv/",
        "scp": f"scp://{hostname}/{p}/lr_1E-03,conv=1,fc=2",
        "not_found": f"scp://{hostname}/{p}/DOES-NOT-EXIST",
    }

  def setup_method(self, method):
    print("")

  @pytest.mark.parametrize("cls", [
      data_loader.TensorboardLogReader,
      data_loader.RustTensorboardLogReader,
  ])
  def test_parse_tensorboard_ssh(self, cls):
    # Note that this directory contains multiple eventfiles
    # (including an empty eventfile), so need to download all of them
    # into the SAME directory in order for (Rust)TensorboardLogReader to work.
    df = data_loader.parse_run(self.paths["scp"], verbose=True, reader_cls=cls)
    print(df)
    assert len(df) > 200

  def test_parse_pandas_ssh(self):
    df = data_loader.parse_run_progresscsv(self.paths["sftp"], verbose=True)
    print(df)
    assert len(df) >= 50

  def test_parse_filenotfound_ssh(self):
    with pytest.raises(FileNotFoundError):
      data_loader.parse_run(self.paths["not_found"], verbose=True)

    # TODO: RunLoader ignores non-existent directories rather than raising
    # FileNotFoundError. This behavior will be changed in the future.
    # with pytest.raises(FileNotFoundError):
    if True:  # pylint: disable=using-constant-test
      data_loader.get_runs(self.paths["not_found"], verbose=True)

  def test_get_runs_ssh(self):
    # Can it handle or ignore non-existent paths?
    runs = data_loader.get_runs(*self.paths.values(), n_jobs=1)
    print(runs)
    assert len(runs) == 2
    assert runs[0].path.rstrip('/') == self.paths["sftp"].rstrip('/')
    assert runs[1].path.rstrip('/') == self.paths["scp"].rstrip('/')


class TestConfigReader:
  """Tests ConfigReader."""

  def test_config_yaml(self):
    r = data_loader.YamlConfigReader(config_filename="config.yaml")
    cfg = r(FIXTURE_PATH / "sample_csv")
    assert isinstance(cfg, dict)

    assert cfg['name'] == 'sample_csv'
    assert cfg['param1'] == 'foo'
    assert cfg['param2'] == 'bar'

    # the behavior of nested config key is undefined at the moment,
    # but yaml reads the config file as nested dicts which is quite natural
    assert cfg['nested'] == {'foo': 1, 'bar': 2}

    # config.yaml not exists?
    with pytest.raises(FileNotFoundError):
      r(FIXTURE_PATH / "lr_1E-03,conv=1,fc=2")


class TestRunLoader:
  """Tests loading of multiple runs at once, in parallel."""

  paths = [
      FIXTURE_PATH / "lr_1E-03,conv=1,fc=2",
      FIXTURE_PATH / "lr_1E-03,conv=2,fc=2",
      FIXTURE_PATH / "lr_1E-04,conv=1,fc=2",
      FIXTURE_PATH / "lr_1E-04,conv=2,fc=2",
      FIXTURE_PATH / "sample_csv",  # note: this has config.yaml
  ]

  @classmethod
  def setup_class(cls):
    _setup_fixture()

  def test_get_runs_parallel(self):
    """Tests expt.get_runs()"""
    runs = data_loader.get_runs_parallel(*self.paths, n_jobs=4)
    print(runs)
    assert len(runs) == len(self.paths)

  def test_get_runs_serial(self):
    """Tests expt.get_run_serial()"""
    runs = data_loader.get_runs_serial(*self.paths)
    print(runs)
    assert len(runs) == len(self.paths)

  @pytest.mark.asyncio
  async def test_get_runs_async(self):
    """Tests expt.get_run_async() with parallel."""
    runs = await data_loader.get_runs_async(*self.paths)
    print(runs)
    assert len(runs) == len(self.paths)

  def test_run_loader_args(self):
    path_csv = self.paths[4]

    with pytest.raises(data_loader.CannotHandleException):
      data_loader.RunLoader(path_csv, reader_cls=())

    with pytest.raises(data_loader.CannotHandleException):
      data_loader.RunLoader(
          path_csv, reader_cls=data_loader.TensorboardLogReader)

    # OK
    data_loader.RunLoader(path_csv, reader_cls=(
        data_loader.TensorboardLogReader,
        data_loader.CSVLogReader,
    ))  # yapf: disable

  def test_run_loader_serial(self):
    paths = [self.paths[0], self.paths[4]]

    def postprocess(run: data.Run) -> data.Run:
      run.df['new_column'] = 0.0
      return run

    loader = data_loader.RunLoader(
        *paths, run_postprocess_fn=postprocess, n_jobs=1)
    runs = loader.get_runs(parallel=False)

    assert len(runs) == len(paths)
    for r, p in zip(runs, paths):
      assert r.path == str(p)  # type: ignore

    # validate postprocess_fn
    assert 'new_column' in runs[0].df  # type: ignore

  def test_run_loader_parallel(self):
    print("")
    loader = data_loader.RunLoader(*self.paths, n_jobs=4)

    with util.timer("First reading"):
      runs = loader.get_runs()
      print(runs)
    assert len(runs) == len(self.paths)

    for r, p in zip(runs, self.paths):
      assert r.path == str(p)  # type: ignore
    assert len(runs[0].df) == 401  # type: ignore

    # incremental loading test (should be much faster)
    with util.timer("Incremental reading"):
      runs_2 = loader.get_runs()

    # it should yield exactly the same result.
    for r, r2 in zip(runs, runs_2):
      assert r.path == r2.path
      np.testing.assert_array_equal(r.df, r2.df)

  def test_run_loader_config(self):
    # default config_reader
    loader = data_loader.RunLoader(self.paths[-1])
    run = loader.get_runs()[0]
    assert run.path.endswith('sample_csv')
    assert run.config is not None

    # default config_reader, that has no config
    loader = data_loader.RunLoader(self.paths[0])
    run = loader.get_runs()[0]
    assert run.path.index('conv=1,fc=2') > 0
    assert run.config is None

    # disabled config_reader
    loader = data_loader.RunLoader(self.paths[-1], config_reader=())
    run = loader.get_runs()[0]
    assert run.path.endswith('sample_csv')
    assert run.config is None

    # custom config_reader
    dummy_reader = lambda _: dict(dummy="config")
    loader = data_loader.RunLoader(
        self.paths[-1], config_reader=dummy_reader)  # type: ignore
    run = loader.get_runs()[0]
    assert run.path.endswith('sample_csv')
    assert run.config == {'dummy': 'config'}

  @pytest.mark.asyncio
  @pytest.mark.parametrize("parallel_mode", ['parallel', 'serial'])
  async def test_run_loader_async(self, parallel_mode):
    loader = data_loader.RunLoader(*self.paths)
    runs = await loader.get_runs_async(parallel=(parallel_mode == 'parallel'))
    assert len(runs) == len(self.paths)
    assert [r.path for r in runs] == [str(p) for p in self.paths]


@pytest.mark.benchmark
@pytest.mark.skipif(
    importlib.util.find_spec("expt._internal") is None,
    reason="The rust extension is not available")
class TestLargeDataBenchmark:
  """Benchmark and performance test for loading large, real-world run data.

  Test data fixtures need to be downloaded or prepared in advance.
  See ^/fixtures/README.md for how to set up benchmark fixtures and
  the information about the dataset.

  To run this test, try:

  $ pytest --run-slow -s -k TestLargeDataBenchmark
  """

  class BenchmarkData:
    """Local clone of gs://tensorboard-bench-logs/edge_cgan/."""

    class Logdir(NamedTuple):
      num_rows: int
      path: str = '.'  # type: ignore

    lifull_001 = Logdir(path="./tb", num_rows=119345)
    lifull_002 = Logdir(path="./tb", num_rows=332740)
    lifull_003 = Logdir(path="./tb", num_rows=389632)
    lifull_004 = Logdir(path="./tb", num_rows=463309)
    lifull_005 = Logdir(path="./tb", num_rows=386142)
    egraph_edge_cgan_001 = Logdir(path=".", num_rows=3999766)
    egraph_edge_cgan_002 = Logdir(path=".", num_rows=1601450)
    egraph_edge_cgan_003 = Logdir(path=".", num_rows=2136493)

    def __init__(self, path: Path):
      for f, val in type(self).__dict__.items():
        if isinstance(val, self.Logdir):
          setattr(self, f, val._replace(path=str(path / f / val.path)))

    def __getitem__(self, name) -> 'Logdir':
      return getattr(self, name)

  @pytest.fixture
  def edge_cgan(self) -> 'BenchmarkData':
    path = Path(FIXTURE_PATH) / 'edge_cgan'
    if not path.exists():
      pytest.skip("`fixture/edge_cgan` does not exist; "
                  "Skipping performance test.")
    return self.BenchmarkData(path=path)

  @pytest.fixture
  def profiling(self, request: pytest.FixtureRequest):
    if request.node.get_closest_marker("slow"):
      print("(a slow test)", end=' ', flush=True)
    _start = time.time()
    yield
    _end = time.time()
    print("\n[%s] Elapsed time: %.3f sec" % (request.node.name, _end - _start))

  compare_tensorboard_and_rustboard = pytest.mark.parametrize("reader", [
      pytest.param(data_loader.RustTensorboardLogReader, id="rustboard"),
      pytest.param(data_loader.TensorboardLogReader, id="tensorboard",
                   marks=pytest.mark.slow),
  ])  # yapf: disable

  # ---------------------------------------------------------------------------

  @pytest.mark.parametrize("logdir", [
      pytest.param('lifull_001'),
      pytest.param('lifull_002'),
      pytest.param('lifull_003', marks=pytest.mark.slow),
      pytest.param('lifull_004', marks=pytest.mark.slow),
      pytest.param('lifull_005', marks=pytest.mark.slow),
      # 'egraph_edge_cgan_001',
      # 'egraph_edge_cgan_002',
      # 'egraph_edge_cgan_003',
  ])  # yapf: disable
  @compare_tensorboard_and_rustboard
  def test_single_logdir(self, profiling, logdir,
                         reader: Type[data_loader.LogReader],
                         edge_cgan: BenchmarkData):
    df = data_loader.parse_run(
        edge_cgan[logdir].path,
        reader_cls=reader,
    )
    assert len(df) == edge_cgan[logdir].num_rows
    del profiling

  @compare_tensorboard_and_rustboard
  def test_parallel_lifull_all(self, profiling,
                               reader: Type[data_loader.LogReader],
                               edge_cgan: BenchmarkData):
    logdirs = [
        edge_cgan.lifull_001,
        edge_cgan.lifull_002,
        edge_cgan.lifull_003,
        edge_cgan.lifull_004,
        edge_cgan.lifull_005,
    ]

    print("")  # Avoid tqdm overwriting the line
    runs: data.RunList = data_loader.RunLoader(
        *[log.path for log in logdirs],
        reader_cls=reader,
        n_jobs=5,
    ).get_runs()

    for run, logdir in zip(runs, logdirs):
      assert len(run.df) == logdir.num_rows
    del profiling


if __name__ == '__main__':
  sys.exit(pytest.main(["-s", "-v"] + sys.argv))
