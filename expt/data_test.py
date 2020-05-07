import sys

import numpy as np
import pandas as pd
import pytest

from expt.data import Run, RunList
from expt.data import Hypothesis
from expt.data import Experiment


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

    def testExperimentIndexing(self):
        h0 = Hypothesis("hyp0", Run('r0', pd.DataFrame({"a": [1, 2, 3]})))
        h1 = Hypothesis("hyp1", Run('r1', pd.DataFrame({"a": [4, 5, 6]})))

        ex = Experiment(title="ex", hypotheses=[h0, h1])
        def V(x):
            print(x)
            return x

        # get hypothesis by name or index
        assert V(ex[0]) == h0
        assert V(ex[1]) == h1
        with pytest.raises(IndexError): V(ex[2])
        assert V(ex["hyp0"]) == h0
        with pytest.raises(KeyError): V(ex["hyp0-not"])

        # nested index
        r = V(ex["hyp0", 'a'])
        assert isinstance(r, pd.DataFrame)
        # assert r is ex["hyp0"]['a']             # TODO
        assert list(r['r0']) == [1, 2, 3]

        # fancy index
        r = V(ex[[0, 1]])
        assert r[0] is h0 and r[1] is h1
        r = V(ex[['hyp1', 'hyp0']])
        assert r[0] is h1 and r[1] is h0

        with pytest.raises(NotImplementedError):  # TODO
            r = V(ex[['hyp1', 'hyp0'], 'a'])


if __name__ == '__main__':
    sys.exit(pytest.main(["-s", "-v"] + sys.argv))
