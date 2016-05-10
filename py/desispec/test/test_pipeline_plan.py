"""
tests desispec.pipeline.core
"""

import os
import unittest
from uuid import uuid4
import shutil
import time
import numpy as np

from desispec.pipeline.plan import *
import desispec.io as io

#- TODO: override log level to quiet down error messages that are supposed
#- to be there from these tests

class TestPlanCmd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.testraw = 'test-'+uuid4().hex
        os.mkdir(cls.testraw)

        cls.night = time.strftime('%Y%m%d', time.localtime(time.time()-12*3600))
        cls.nightdir = os.path.join(cls.testraw, cls.night)
        os.mkdir(cls.nightdir)

        for expid in [0, 1]:
            fibermap = io.fibermap.empty_fibermap(10)
            for key in fibermap.dtype.names:
                column = fibermap[key]
                fibermap[key] = np.random.random(column.shape).astype(column.dtype)
            hdr = {'flavor': 'flat'}
            fmfile = os.path.join(cls.nightdir, "fibermap-{:08d}.fits".format(expid))
            io.write_fibermap(fmfile, fibermap, header=hdr)

        for expid in [2, 3]:
            fibermap = io.fibermap.empty_fibermap(10)
            for key in fibermap.dtype.names:
                column = fibermap[key]
                fibermap[key] = np.random.random(column.shape).astype(column.dtype)
            hdr = {'flavor': 'arc'}
            fmfile = os.path.join(cls.nightdir, "fibermap-{:08d}.fits".format(expid))
            io.write_fibermap(fmfile, fibermap, header=hdr)

        for expid in [4, 5]:
            fibermap = io.fibermap.empty_fibermap(10)
            for key in fibermap.dtype.names:
                column = fibermap[key]
                fibermap[key] = np.random.random(column.shape).astype(column.dtype)
            hdr = {'flavor': 'science'}
            fmfile = os.path.join(cls.nightdir, "fibermap-{:08d}.fits".format(expid))
            io.write_fibermap(fmfile, fibermap, header=hdr)

        for expid in range(6):
            for band in ['b', 'r', 'z']:
                for spec in range(10):
                    cam = "{}{}".format(band, spec)
                    pixfile = os.path.join(cls.nightdir, "pix-{}-{:08d}.fits".format(cam, expid))
                    with open(pixfile, 'w') as p:
                        p.write("")


    @classmethod
    def tearDownClass(cls):
        #shutil.rmtree(cls.testraw)
        pass


    def test_graph_names(self):
        pass
    
    def test_graph(self):
        grph = graph_night(self.testraw, self.night)
        with open(os.path.join(self.testraw, "{}.dot".format(self.night)), 'w') as f:
            graph_dot(grph, f)
        graph_write(os.path.join(self.testraw, "{}_graph.yml".format(self.night)), grph)


    def test_graph_slice_spec(self):
        grph = graph_night(self.testraw, self.night)
        grph4 = graph_slice_spec(grph, [4])
        with open(os.path.join(self.testraw, "{}-spec4.dot".format(self.night)), 'w') as f:
            graph_dot(grph4, f)


    def test_graph_make_fail(self):
        grph = graph_night(self.testraw, self.night)
        grph4 = graph_slice_spec(grph, [4])
        graph_mark(grph4, graph_name(self.night, "pix-r4-{:08d}".format(3)), 'fail', descend=True)
        with open(os.path.join(self.testraw, "{}-spec4_fail.dot".format(self.night)), 'w') as f:
            graph_dot(grph4, f)


    def test_graph_slice(self):
        grph = graph_night(self.testraw, self.night)
        grph4 = graph_slice_spec(grph, [4])
        exgraph = graph_slice(grph4, types=['frame'], deps=True)
        with open(os.path.join(self.testraw, "{}-spec4_extract.dot".format(self.night)), 'w') as f:
            graph_dot(exgraph, f)

    

#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()           
