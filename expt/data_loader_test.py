import functools
import os
import shutil
import sys
import tempfile
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from expt import data, data_loader, util

__PATH__ = os.path.abspath(os.path.dirname(__file__))
FIXTURE_PATH = os.path.join(__PATH__, '../fixtures')

# Suppress and silence TF warnings.
# such as, cpu_utils.cc: Failed to get CPU frequency: 0 Hz
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def _setup_fixture():
  """Set up example fixture data."""
  URL = "https://tensorboard.dev/static/log_set-2021-03-11.zip"

  if (Path(FIXTURE_PATH) / "lr_1E-03,conv=1,fc=2").exists():
    return

  print(f"Downloading and extracting from {URL} ...")
  with urllib.request.urlopen(URL) as z:
    with tempfile.NamedTemporaryFile() as tfile:
      tfile.write(z.read())
      tfile.seek(0)
      shutil.unpack_archive(tfile.name, FIXTURE_PATH, format='zip')


class TestGetRuns:

  @classmethod
  def setup_class(cls):
    _setup_fixture()

  def test_parse_tensorboard(self):
    r = data_loader.parse_run(Path(FIXTURE_PATH) / "lr_1E-03,conv=1,fc=2")

    assert len(r) >= 400
    np.testing.assert_array_equal(
        r.columns, ['accuracy/accuracy', 'global_step', 'xent/xent_1'])

  def test_parse_progresscsv(self):
    r = data_loader.parse_run(Path(FIXTURE_PATH) / "sample_csv")

    assert len(r) >= 50
    np.testing.assert_array_equal(r.columns, [
        'initial_reset_time',
        'episode_rewards',
        'episode_lengths',
        'episode_end_times',
    ])

  def test_parse_cannot_handle(self):
    # incompatible logdir format and parser
    with pytest.raises(FileNotFoundError):
      data_loader.CSVLogReader(Path(FIXTURE_PATH) / "lr_1E-03,conv=1,fc=2")
    with pytest.raises(FileNotFoundError):
      data_loader.TensorboardLogReader(Path(FIXTURE_PATH) / "sample_csv")

  def test_parser_detection(self):

    log_dir = Path(FIXTURE_PATH) / "lr_1E-03,conv=1,fc=2"
    p = data_loader._get_reader_for(log_dir)
    assert isinstance(p, data_loader.TensorboardLogReader)
    assert p._log_dir == log_dir

    log_dir = Path(FIXTURE_PATH) / "sample_csv"
    p = data_loader._get_reader_for(log_dir)
    assert isinstance(p, data_loader.CSVLogReader)
    assert p._log_dir == log_dir

  def test_parse_tensorboard_incremental_read(self):
    path = Path(FIXTURE_PATH) / "lr_1E-03,conv=1,fc=2"
    r = data_loader.TensorboardLogReader(path)

    print(path)
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
    def _read(reader) -> pd.DataFrame:
      ctx = reader.new_context()
      ctx = reader.read(ctx)
      return reader.result(ctx)

    df_ref = _read(data_loader.TensorboardLogReader(path))
    assert np.all(df == df_ref)


class TestRunLoader:
  """Tests loading of multiple runs at once, in parallel."""

  paths = [
      Path(FIXTURE_PATH) / "lr_1E-03,conv=1,fc=2",
      Path(FIXTURE_PATH) / "lr_1E-03,conv=2,fc=2",
      Path(FIXTURE_PATH) / "lr_1E-04,conv=1,fc=2",
      Path(FIXTURE_PATH) / "lr_1E-04,conv=2,fc=2",
      Path(FIXTURE_PATH) / "sample_csv",
  ]

  @classmethod
  def setup_class(cls):
    _setup_fixture()

  def test_get_runs_parallel(self):
    runs = data_loader.get_runs_parallel(*self.paths, n_jobs=4)
    print(runs)
    assert len(runs) == len(self.paths)

  def test_run_loader_serial(self):
    paths = [self.paths[0], self.paths[4]]

    def postprocess(run: data.Run) -> data.Run:
      run.df['new_column'] = 0.0
      return run

    loader = data_loader.RunLoader(
        *paths, run_postprocess_fn=postprocess, n_jobs=4)
    runs = loader.get_runs()

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
  async def test_run_loader_async(self):
    loader = data_loader.RunLoader(*self.paths)
    runs = await loader.get_runs_async(parallel=False)
    assert len(runs) == len(self.paths)
    assert [r.path for r in runs] == [str(p) for p in self.paths]


if __name__ == '__main__':
  sys.exit(pytest.main(["-s", "-v"] + sys.argv))
