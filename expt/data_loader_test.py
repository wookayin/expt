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
    with (urllib.request.urlopen(URL) as z,
          tempfile.NamedTemporaryFile() as tfile):
      tfile.write(z.read())
      tfile.seek(0)
      shutil.unpack_archive(tfile.name, FIXTURE_PATH, format='zip')

  def test_parse_tensorboard(self):
    r = data_loader.parse_run(Path(FIXTURE_PATH) / "lr_1E-03,conv=1,fc=2")
    print(r)

    assert len(r) >= 400
    np.testing.assert_array_equal(
        r.columns,
        ['accuracy/accuracy', 'global_step', 'xent/xent_1']
    )


if __name__ == '__main__':
  sys.exit(pytest.main(["-s", "-v"] + sys.argv))
