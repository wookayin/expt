"""Tests for expt.data_loader."""
# pylint: disable=protected-access

import functools
import os
from pathlib import Path
import shutil
import sys
import tempfile
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

  def test_parse_tensorboard(self, path_tensorboard):
    r = data_loader.parse_run(path_tensorboard)
    assert len(r) >= 400
    np.testing.assert_array_equal(
        r.columns, ['accuracy/accuracy', 'global_step', 'xent/xent_1'])

  def test_parse_progresscsv(self, path_csv):
    r = data_loader.parse_run(path_csv)

    assert len(r) >= 50
    np.testing.assert_array_equal(r.columns, [
        'initial_reset_time',
        'episode_rewards',
        'episode_lengths',
        'episode_end_times',
    ])

  def test_parse_cannot_handle(self, path_csv, path_tensorboard, tmp_path):
    """Tests incompatible logdir format and parser."""

    with pytest.raises(data_loader.CannotHandleException):
      data_loader.CSVLogReader(path_tensorboard)

    with pytest.raises(data_loader.CannotHandleException):
      data_loader.TensorboardLogReader(path_csv)

    with pytest.raises(data_loader.CannotHandleException):
      data_loader.parse_run_tensorboard(path_csv)

    with pytest.raises(data_loader.CannotHandleException):
      data_loader.parse_run_progresscsv(path_tensorboard)

    with pytest.raises(data_loader.CannotHandleException):
      data_loader.parse_run(tmp_path, verbose=True)

  def test_parser_detection(self, path_csv, path_tensorboard, tmp_path):
    """Tests automatic parser resolution."""

    p = data_loader._get_reader_for(path_tensorboard)
    assert isinstance(p, data_loader.TensorboardLogReader)
    assert p.log_dir == str(path_tensorboard)

    p = data_loader._get_reader_for(path_csv)
    assert isinstance(p, data_loader.CSVLogReader)
    assert p.log_dir == str(path_csv)

    with pytest.raises(data_loader.CannotHandleException):
      p = data_loader._get_reader_for(tmp_path)

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

  def test_parse_tensorboard_ssh(self):
    df = data_loader.parse_run_tensorboard(self.paths["scp"], verbose=True)
    print(df)
    assert len(df) > 200

  def test_parse_pandas_ssh(self):
    df = data_loader.parse_run_progresscsv(self.paths["sftp"], verbose=True)
    print(df)
    assert len(df) >= 50

  def test_parse_filenotfound_ssh(self):
    with pytest.raises(pd.errors.EmptyDataError):
      data_loader.parse_run(self.paths["not_found"], verbose=True)


class TestRunLoader:
  """Tests loading of multiple runs at once, in parallel."""

  paths = [
      FIXTURE_PATH / "lr_1E-03,conv=1,fc=2",
      FIXTURE_PATH / "lr_1E-03,conv=2,fc=2",
      FIXTURE_PATH / "lr_1E-04,conv=1,fc=2",
      FIXTURE_PATH / "lr_1E-04,conv=2,fc=2",
      FIXTURE_PATH / "sample_csv",
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

  @pytest.mark.asyncio
  @pytest.mark.parametrize("parallel_mode", ['parallel', 'serial'])
  async def test_run_loader_async(self, parallel_mode):
    loader = data_loader.RunLoader(*self.paths)
    runs = await loader.get_runs_async(parallel=(parallel_mode == 'parallel'))
    assert len(runs) == len(self.paths)
    assert [r.path for r in runs] == [str(p) for p in self.paths]


if __name__ == '__main__':
  sys.exit(pytest.main(["-s", "-v"] + sys.argv))
