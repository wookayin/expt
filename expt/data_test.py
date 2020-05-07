import sys

import pandas as pd
import pytest

from expt.data import Run, RunList
from expt.data import Hypothesis


class TestDataStructure:

    def setup_method(self, method):
        print("")

    def testRunList(self):
        r0 = Run("r0", pd.DataFrame({"y" : [1, 2, 3]}))
        r1 = Run("r1", pd.DataFrame({"y" : [1, 4, 9]}))
        runs = RunList([r0, r1])

        r0_, r1_ = runs  # unpacking
        assert r0_ is r0 and r1_ is r1

        print("runs =", runs)
        assert list(runs) == [r0, r1]
        assert runs[0] is r0

        for i, r in enumerate(runs):
            print("r from iterable :", r)
            assert r == [r0, r1][i]

        assert RunList.of(runs) is runs  # no copy!

        r2 = Run("r1", pd.DataFrame({"y" : [2, 2, 2]}))
        runs.extend([r2])
        assert len(runs) == 3


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
        assert h.runs.as_list() == [r0]

        h = Hypothesis.of(r0)
        print(h)
        assert h.name == 'r0'
        assert h.runs.as_list() == [r0]

        h = Hypothesis.of(
            (Run(i, df=pd.DataFrame()) for i in ["a", "b", "c"]),
            name="generator")
        print(h)
        assert h.name == 'generator'
        assert len(h) == 3


if __name__ == '__main__':
    sys.exit(pytest.main(["-s", "-v"] + sys.argv))
