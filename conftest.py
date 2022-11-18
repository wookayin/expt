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


def pytest_report_teststatus(report: pytest.TestReport, config: pytest.Config):
  # Report duration time for @pytest.mark.benchmark tests.
  if report.when == 'call' and report.keywords.get('benchmark'):
    if report.outcome == 'passed':
      summary = "PASSED ({:.2f} sec)".format(report.duration)
      return report.outcome, 'B', (summary)
