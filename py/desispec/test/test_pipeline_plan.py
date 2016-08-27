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

class TestPipelinePlan(unittest.TestCase):

    def setUp(self):
        self.proddir = 'test-prod-'+uuid4().hex
        self.testraw = 'test-'+uuid4().hex
        os.mkdir(self.testraw)

        self.night = time.strftime('%Y%m%d', time.localtime(time.time()-12*3600))
        self.nightdir = os.path.join(self.testraw, self.night)
        os.mkdir(self.nightdir)

        for expid in [0, 1]:
            fibermap = io.fibermap.empty_fibermap(10)
            for key in fibermap.dtype.names:
                column = fibermap[key]
                fibermap[key] = np.random.random(column.shape).astype(column.dtype)
            hdr = {'flavor': 'flat'}
            fmfile = os.path.join(self.nightdir, "fibermap-{:08d}.fits".format(expid))
            io.write_fibermap(fmfile, fibermap, header=hdr)

        for expid in [2, 3]:
            fibermap = io.fibermap.empty_fibermap(10)
            for key in fibermap.dtype.names:
                column = fibermap[key]
                fibermap[key] = np.random.random(column.shape).astype(column.dtype)
            hdr = {'flavor': 'arc'}
            fmfile = os.path.join(self.nightdir, "fibermap-{:08d}.fits".format(expid))
            io.write_fibermap(fmfile, fibermap, header=hdr)

        for expid in [4, 5]:
            fibermap = io.fibermap.empty_fibermap(10)
            for key in fibermap.dtype.names:
                column = fibermap[key]
                fibermap[key] = np.random.random(column.shape).astype(column.dtype)
            hdr = {'flavor': 'science'}
            fmfile = os.path.join(self.nightdir, "fibermap-{:08d}.fits".format(expid))
            io.write_fibermap(fmfile, fibermap, header=hdr)

        for expid in range(6):
            for band in ['b', 'r', 'z']:
                for spec in range(10):
                    cam = "{}{}".format(band, spec)
                    pixfile = os.path.join(self.nightdir, "pix-{}-{:08d}.fits".format(cam, expid))
                    with open(pixfile, 'w') as p:
                        p.write("")

    def tearDown(self):
        shutil.rmtree(self.testraw)
        if os.path.exists(self.proddir):
            shutil.rmtree(self.proddir)


    def test_graph_names(self):
        pass


    def test_graph(self):
        (grph, expcount, bricks) = graph_night(self.testraw, self.night)        
        with open(os.path.join(self.testraw, "{}.dot".format(self.night)), 'w') as f:
            graph_dot(grph, f)        
        graph_write(os.path.join(self.testraw, "{}_graph.yml".format(self.night)), grph)


    def test_graph_slice_spec(self):
        (grph, expcount, bricks) = graph_night(self.testraw, self.night)
        grph4 = graph_slice_spec(grph, [4])
        with open(os.path.join(self.testraw, "{}-spec4.dot".format(self.night)), 'w') as f:
            graph_dot(grph4, f)


    def test_graph_make_fail(self):
        (grph, expcount, bricks) = graph_night(self.testraw, self.night)
        grph4 = graph_slice_spec(grph, [4])
        graph_mark(grph4, graph_name(self.night, "pix-r4-{:08d}".format(3)), 'fail', descend=True)
        with open(os.path.join(self.testraw, "{}-spec4_fail.dot".format(self.night)), 'w') as f:
            graph_dot(grph4, f)


    def test_graph_slice(self):
        (grph, expcount, bricks) = graph_night(self.testraw, self.night)
        grph4 = graph_slice_spec(grph, [4])
        exgraph = graph_slice(grph4, types=['frame'], deps=True)
        with open(os.path.join(self.testraw, "{}-spec4_extract.dot".format(self.night)), 'w') as f:
            graph_dot(exgraph, f)

    def test_create_prod(self):
        create_prod(self.testraw, self.proddir)
        for subdir in [
            'bricks',
            'calib2d/psf',
            'calib2d/'+self.night,
            'exposures/'+self.night,
            'plan/',
            'run/logs',
            'run/scripts',
            'run/failed',
            ]:
            self.assertTrue(os.path.exists(os.path.join(self.proddir, subdir)), 'missing '+subdir)        

        graph = graph_read_prod(self.proddir, nightstr=self.night)

    def test_graph_path(self):
        x = graph_path(self.testraw, self.proddir, '{}_fibermap-{:08d}'.format(self.night, 1), 'fibermap')
        x = graph_path(self.testraw, self.proddir, '{}_pix-b1-{:08d}'.format(self.night, 3), 'pix')
        x = graph_path(self.testraw, self.proddir, '{}_psfboot-b1'.format(self.night), 'psfboot')
        x = graph_path(self.testraw, self.proddir, '{}_psf-b1-{:08d}'.format(self.night, 3), 'psf')
        x = graph_path(self.testraw, self.proddir, '{}_psfnight-r2'.format(self.night), 'psfnight')
        x = graph_path(self.testraw, self.proddir, '{}_frame-z3-{:08d}'.format(self.night, 3), 'frame')
        x = graph_path(self.testraw, self.proddir, '{}_cframe-z3-{:08d}'.format(self.night, 3), 'cframe')
        x = graph_path(self.testraw, self.proddir, '{}_fiberflat-z3-{:08d}'.format(self.night, 3), 'fiberflat')
        x = graph_path(self.testraw, self.proddir, '{}_sky-r4-{:08d}'.format(self.night, 3), 'sky')
        x = graph_path(self.testraw, self.proddir, '{}_stdstars-5-{:08d}'.format(self.night, 3), 'stdstars')
        x = graph_path(self.testraw, self.proddir, '{}_calib-r4-{:08d}'.format(self.night, 3), 'calib')
        x = graph_path(self.testraw, self.proddir, 'brick-b-brickname', 'brick')
        x = graph_path(self.testraw, self.proddir, 'zbest-b-brickname', 'zbest')


#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()           
