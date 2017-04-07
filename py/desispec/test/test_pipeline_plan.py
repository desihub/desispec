"""
tests desispec.pipeline.core
"""

import os
import unittest
import shutil
import time
import numpy as np

from desispec.pipeline.common import *
from desispec.pipeline.graph import *
from desispec.pipeline.plan import *
import desispec.io as io

from . import pipehelpers as ph


class TestPipelinePlan(unittest.TestCase):

    def setUp(self):
        self.prod = "test"
        self.raw = ph.fake_raw()
        self.redux = ph.fake_redux(self.prod)
        ph.fake_env(self.raw, self.redux, self.prod, self.prod)

    def tearDown(self):
        if os.path.exists(self.raw):
            shutil.rmtree(self.raw)
        if os.path.exists(self.redux):
            shutil.rmtree(self.redux)
        ph.fake_env_clean()


    def test_select_nights(self):
        allnights = [
            "20150102",
            "20160204",
            "20170103",
            "20170211"
        ]
        checkyear = [
            "20170103",
            "20170211"
        ]
        checkmonth = [
            "20160204",
            "20170211"
        ]
        selyear = select_nights(allnights, r"2017.*")
        self.assertTrue(selyear == checkyear)
        selmonth = select_nights(allnights, r"[0-9]{4}02[0-9]{2}")
        self.assertTrue(selmonth == checkmonth)


    def test_graph_night(self):
        grph, expcnt, bricks = graph_night(ph.fake_night())


    def test_create_load_prod(self):
        grph, expcnt, bricks = graph_night(ph.fake_night())
        expnightcnt, bricks = create_prod()
        fullgrph = load_prod()
        self.assertTrue(grph == fullgrph)



#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
