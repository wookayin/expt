"""Tests for expt.path_util."""

import json
import os
from pathlib import Path
import sys

import pytest

import expt.path_util as P

try:
  from rich.console import Console
  console = Console(markup=False)
  print = console.log
except ImportError:
  pass

__PATH__ = os.path.abspath(os.path.dirname(__file__))
FIXTURE_PATH = Path(os.path.join(__PATH__, '../fixtures'))


def V(x):
  """Print the object and return it."""
  kwargs = dict(_stack_offset=2) if print.__name__ == 'log' else {}
  print(x, **kwargs)
  return x


os.environ['NO_GCE_CHECK'] = 'true'

# tf.io.gfile is quite slow; see tensorflow#57939
P.use_gsutil(True)


def test_local_file():
  """Tests local file."""
  sys.stdout.write("local\n")

  PROGRESS_CSV = "sample_csv/progress.csv"

  glob = V(P.glob(FIXTURE_PATH / "**/*.csv"))
  assert str(FIXTURE_PATH / PROGRESS_CSV) in glob

  assert P.exists(FIXTURE_PATH / PROGRESS_CSV)

  assert P.isdir(FIXTURE_PATH / "sample_csv")
  assert P.isdir(FIXTURE_PATH / "sample_csv/")
  assert not P.isdir(FIXTURE_PATH / PROGRESS_CSV)

  with P.open(FIXTURE_PATH / PROGRESS_CSV) as f:
    line = f.readline()
    assert 'episode_rewards' in line


@pytest.mark.parametrize("protocol", ["sftp", "scp"])
def test_ssh(protocol: str):
  """Tests sftp:// files."""

  hostname = os.environ.get("EXPT_SSH_HOST")
  if not hostname:
    pytest.skip("Requires SSH host setup for test")

  port = os.environ.get("EXPT_SSH_PORT", "22")
  uri_base = f"{protocol}://{hostname}:{port}"

  bashrc = uri_base + "/.bashrc"
  directory = uri_base + "/.ssh"
  not_exist = uri_base + "/__NOT_EXIST__"

  sys.stdout.write("ssh\n")

  assert P.exists(bashrc) is True
  assert P.exists(not_exist) is False

  with P.open(bashrc) as f:
    lines = f.readlines()
  assert (lines.__len__() > 0)

  assert P.isdir(bashrc) is False
  assert P.isdir(directory) is True
  assert P.isdir(not_exist) is False

  glob = V(P.glob(uri_base + "/.*bash*"))
  assert bashrc in glob

  glob = V(P.glob(uri_base + "//etc/*bashrc*"))
  assert glob

  glob = V(P.glob(uri_base + "/.ss*/*s*"))


@pytest.mark.slow
def test_gcloud():
  """Tests gs:// files.

  URL: https://console.cloud.google.com/storage/browser/tfds-data
  """
  sys.stdout.write("gcloud\n")
  TFDATA_JSONL: str = "gs://tfds-data/community-datasets-list.jsonl"

  assert V(P.glob("gs://tfds-data/*.jsonl")) == [TFDATA_JSONL]

  assert P.exists(TFDATA_JSONL)
  assert not P.exists("gs://tfds-data/404.txt")

  assert P.isdir("gs://tfds-data/datasets/mnist")
  assert P.isdir("gs://tfds-data/datasets/mnist/")
  assert not P.isdir(TFDATA_JSONL)

  with P.open(TFDATA_JSONL) as f:
    line = f.readline()
    assert 'name' in json.loads(line)  # is a valid JSON?


if __name__ == '__main__':
  sys.exit(pytest.main(["-s", "-v", "--run-slow"] + sys.argv))
