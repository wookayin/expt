from typing import List

import pytest


def pytest_addoption(parser):
  parser.addoption(
      "--run-slow",
      action="store_true",
      default=False,
      help="Run slow tests as well",
  )


def pytest_collection_modifyitems(config, items: List[pytest.Item]):
  # Skip all tests with the 'slow' marker unless '--run-slow'
  if not config.getoption("--run-slow"):
    skip = pytest.mark.skip(reason="Run only when --run-slow is given")
    for item in items:
      if item.get_closest_marker("slow"):
        item.add_marker(skip)
