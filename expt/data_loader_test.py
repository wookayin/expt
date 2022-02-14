import os
import shutil
import sys
import tempfile
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pytest

from expt import data_loader

__PATH__ = os.path.abspath(os.path.dirname(__file__))
FIXTURE_PATH = os.path.join(__PATH__, '../fixtures')


class TestGetRuns:

  @classmethod
  def setup_class(cls):
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
      data_loader.CSVLogParser(Path(FIXTURE_PATH) / "lr_1E-03,conv=1,fc=2")
    with pytest.raises(FileNotFoundError):
      data_loader.TensorboardLogParser(Path(FIXTURE_PATH) / "sample_csv")

  def test_parser_detection(self):

    log_dir = Path(FIXTURE_PATH) / "lr_1E-03,conv=1,fc=2"
    p = data_loader._get_parser_for(log_dir)
    assert isinstance(p, data_loader.TensorboardLogParser)
    assert p._log_dir == log_dir

    log_dir = Path(FIXTURE_PATH) / "sample_csv"
    p = data_loader._get_parser_for(log_dir)
    assert isinstance(p, data_loader.CSVLogParser)
    assert p._log_dir == log_dir


if __name__ == '__main__':
  sys.exit(pytest.main(["-s", "-v"] + sys.argv))
