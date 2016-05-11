"""
tests desispec.pipeline.core
"""

import os
import unittest
from uuid import uuid4
import shutil
import time
import numpy as np

from desispec.pipeline.run import *
import desispec.io as io

#- TODO: override log level to quiet down error messages that are supposed
#- to be there from these tests

class TestPipelineRun(unittest.TestCase):

    def setUp(self):
        self.uid = uuid4().hex
        self.testraw = "test_raw-{}".format(self.uid)
        os.mkdir(self.testraw)
        self.testprod = "test_redux-{}".format(self.uid)
        os.mkdir(self.testprod)

        self.night = time.strftime('%Y%m%d', time.localtime(time.time()-12*3600))
        self.rawnightdir = os.path.join(self.testraw, self.night)
        os.mkdir(self.rawnightdir)

        self.cal2d = os.path.join(self.testprod, 'calib2d')
        if not os.path.isdir(self.cal2d):
            os.makedirs(self.cal2d)

        self.calpsf = os.path.join(self.cal2d, 'psf')
        if not os.path.isdir(self.calpsf):
            os.makedirs(self.calpsf)

        self.calpsfnight = os.path.join(self.calpsf, self.night)
        if not os.path.isdir(self.calpsfnight):
            os.makedirs(self.calpsfnight)

        self.expdir = os.path.join(self.testprod, 'exposures')
        if not os.path.isdir(self.expdir):
            os.makedirs(self.expdir)

        self.expnight = os.path.join(self.expdir, self.night)
        if not os.path.isdir(self.expnight):
            os.makedirs(self.expnight)

        self.brkdir = os.path.join(self.testprod, 'bricks')
        if not os.path.isdir(self.brkdir):
            os.makedirs(self.brkdir)

        self.logdir = os.path.join(self.testprod, 'logs')
        if not os.path.isdir(self.logdir):
            os.makedirs(self.logdir)

        self.faildir = os.path.join(self.testprod, 'failed')
        if not os.path.isdir(self.faildir):
            os.makedirs(self.faildir)

        self.scriptdir = os.path.join(self.testprod, 'scripts')
        if not os.path.isdir(self.scriptdir):
            os.makedirs(self.scriptdir)

        for expid in [0, 1]:
            fibermap = io.fibermap.empty_fibermap(10)
            for key in fibermap.dtype.names:
                column = fibermap[key]
                fibermap[key] = np.random.random(column.shape).astype(column.dtype)
            hdr = {'flavor': 'flat'}
            fmfile = os.path.join(self.rawnightdir, "fibermap-{:08d}.fits".format(expid))
            io.write_fibermap(fmfile, fibermap, header=hdr)

        for expid in [2, 3]:
            fibermap = io.fibermap.empty_fibermap(10)
            for key in fibermap.dtype.names:
                column = fibermap[key]
                fibermap[key] = np.random.random(column.shape).astype(column.dtype)
            hdr = {'flavor': 'arc'}
            fmfile = os.path.join(self.rawnightdir, "fibermap-{:08d}.fits".format(expid))
            io.write_fibermap(fmfile, fibermap, header=hdr)

        for expid in [4, 5]:
            fibermap = io.fibermap.empty_fibermap(10)
            for key in fibermap.dtype.names:
                column = fibermap[key]
                fibermap[key] = np.random.random(column.shape).astype(column.dtype)
            hdr = {'flavor': 'science'}
            fmfile = os.path.join(self.rawnightdir, "fibermap-{:08d}.fits".format(expid))
            io.write_fibermap(fmfile, fibermap, header=hdr)

        for expid in range(6):
            for band in ['b', 'r', 'z']:
                for spec in range(10):
                    cam = "{}{}".format(band, spec)
                    pixfile = os.path.join(self.rawnightdir, "pix-{}-{:08d}.fits".format(cam, expid))
                    with open(pixfile, 'w') as p:
                        p.write("")

        self.grph = graph_night(self.testraw, self.night)
        graph_write(os.path.join(self.testraw, "{}_graph.yml".format(self.night)), self.grph)


    def tearDown(self):
        shutil.rmtree(self.testraw)
        shutil.rmtree(self.testprod)
        pass

    
    def test_options(self):
        options = default_options()
        dump = os.path.join(self.testraw, "opts.yml")
        write_options(dump, options)


    def test_failpkl(self):
        options = default_options()
        #run_step('bootcalib', self.testraw, self.testprod, self.grph, options)
    

#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()           
