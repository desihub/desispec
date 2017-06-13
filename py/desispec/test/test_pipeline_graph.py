"""
tests desispec.pipeline.graph
"""

import os
import unittest
import shutil
import time
import copy

import numpy as np

from desispec.pipeline.common import *
from desispec.pipeline.graph import *
from desispec.pipeline.plan import *
import desispec.io as io

from . import pipehelpers as ph


class TestPipelineGraph(unittest.TestCase):

    def setUp(self):
        self.prod = "test"
        self.raw = ph.fake_raw()
        self.redux = ph.fake_redux(self.prod)
        ph.fake_env(self.raw, self.redux, self.prod, self.redux)
        self.specs = [ x for x in range(10) ]

    def tearDown(self):
        if os.path.exists(self.raw):
            shutil.rmtree(self.raw)
        if os.path.exists(self.redux):
            shutil.rmtree(self.redux)
        ph.fake_env_clean()

    def test_graph_name(self):
        check = graph_name("foo", "bar", "blat")
        self.assertTrue(check == "_".join(["foo", "bar", "blat"]))


    def test_graph_night_split(self):
        in1 = "blah"
        nt, obj = graph_night_split(in1)
        self.assertTrue(nt == "")
        self.assertTrue(obj == in1)

        in2 = "10101010"
        nt, obj = graph_night_split(in2)
        self.assertTrue(nt == "10101010")
        self.assertTrue(obj == "")

        in3 = "10101010{}blah".format("_")
        nt, obj = graph_night_split(in3)
        self.assertTrue(nt == "10101010")
        self.assertTrue(obj == "blah")


    def test_graph_name_split(self):
        band = "b"
        spec = 6
        expid = 12345678
        brk = "3411p200"
        nside = 64
        pix = 123

        typ = "fibermap"
        obj = "{}-{}".format(typ, expid)
        check = graph_name_split(obj)
        self.assertTrue(check[0], typ)
        self.assertTrue(check[1], expid)

        typ = "pix"
        obj = "{}-{}{}-{}".format(typ, band, spec, expid)
        check = graph_name_split(obj)
        self.assertTrue(check[0], typ)
        self.assertTrue(check[1], band)
        self.assertTrue(check[2], spec)
        self.assertTrue(check[3], expid)

        typ = "psfboot"
        obj = "{}-{}{}".format(typ, band, spec)
        check = graph_name_split(obj)
        self.assertTrue(check[0], typ)
        self.assertTrue(check[1], band)
        self.assertTrue(check[2], spec)

        typ = "psf"
        obj = "{}-{}{}-{}".format(typ, band, spec, expid)
        check = graph_name_split(obj)
        self.assertTrue(check[0], typ)
        self.assertTrue(check[1], band)
        self.assertTrue(check[2], spec)
        self.assertTrue(check[3], expid)

        typ = "psfnight"
        obj = "{}-{}{}".format(typ, band, spec)
        check = graph_name_split(obj)
        self.assertTrue(check[0], typ)
        self.assertTrue(check[1], band)
        self.assertTrue(check[2], spec)

        typ = "frame"
        obj = "{}-{}{}-{}".format(typ, band, spec, expid)
        check = graph_name_split(obj)
        self.assertTrue(check[0], typ)
        self.assertTrue(check[1], band)
        self.assertTrue(check[2], spec)
        self.assertTrue(check[3], expid)

        typ = "fiberflat"
        obj = "{}-{}{}-{}".format(typ, band, spec, expid)
        check = graph_name_split(obj)
        self.assertTrue(check[0], typ)
        self.assertTrue(check[1], band)
        self.assertTrue(check[2], spec)
        self.assertTrue(check[3], expid)

        typ = "sky"
        obj = "{}-{}{}-{}".format(typ, band, spec, expid)
        check = graph_name_split(obj)
        self.assertTrue(check[0], typ)
        self.assertTrue(check[1], band)
        self.assertTrue(check[2], spec)
        self.assertTrue(check[3], expid)

        typ = "stdstars"
        obj = "{}-{}-{}".format(typ, spec, expid)
        check = graph_name_split(obj)
        self.assertTrue(check[0], typ)
        self.assertTrue(check[1], spec)
        self.assertTrue(check[2], expid)

        typ = "calib"
        obj = "{}-{}{}-{}".format(typ, band, spec, expid)
        check = graph_name_split(obj)
        self.assertTrue(check[0], typ)
        self.assertTrue(check[1], band)
        self.assertTrue(check[2], spec)
        self.assertTrue(check[3], expid)

        typ = "cframe"
        obj = "{}-{}{}-{}".format(typ, band, spec, expid)
        check = graph_name_split(obj)
        self.assertTrue(check[0], typ)
        self.assertTrue(check[1], band)
        self.assertTrue(check[2], spec)
        self.assertTrue(check[3], expid)

        typ = "spectra"
        obj = "{}-{}-{}".format(typ, nside, pix)
        check = graph_name_split(obj)
        self.assertTrue(check[0], typ)
        self.assertTrue(int(check[1]), nside)
        self.assertTrue(int(check[2]), pix)

        typ = "zbest"
        obj = "{}-{}-{}".format(typ, nside, pix)
        check = graph_name_split(obj)
        self.assertTrue(check[0], typ)
        self.assertTrue(int(check[1]), nside)
        self.assertTrue(int(check[2]), pix)


    def test_graph_path(self):
        grph, expcnt, allpix = graph_night(ph.fake_night(), self.specs, False)
        for key, val in grph.items():
            path = graph_path(key)
            self.assertTrue(path != "")
            if val["type"] == "night":
                self.assertTrue(os.path.abspath(path) == os.path.abspath(os.path.join(self.redux, self.prod, "exposures", key)))


    def test_graph_prune(self):
        grph, expcnt, allpix = graph_night(ph.fake_night(), self.specs, False)
        # prune one science exposure, check that everything from stdstars onward
        # is cut
        graph_prune(grph, "{}_pix-r0-00000002".format(ph.fake_night()), descend=True)
        for key, val in grph.items():
            self.assertTrue(val["type"] != "stdstars")
            self.assertTrue(val["type"] != "calib")
            self.assertTrue(val["type"] != "cframe")


    def test_graph_slice(self):
        grph, expcnt, bricks = graph_night(ph.fake_night(), self.specs, False)
        newgrph = graph_slice(grph, types=["frame"], deps=True)
        # the new graph should have frame objects and their immediate
        # dependencies.
        allowed = ["frame", "psfnight", "pix", "fibermap"]
        for key, val in newgrph.items():
            self.assertTrue(val["type"] in allowed)


    def test_graph_set_recursive(self):
        grph, expcnt, bricks = graph_night(ph.fake_night(), self.specs, False)
        # change state of one science exposure
        graph_set_recursive(grph, "{}_pix-r0-00000002".format(ph.fake_night()), "fail")
        for key, val in grph.items():
            if val["type"] == "stdstars":
                self.assertTrue(val["state"] == "fail")
            if val["type"] == "calib":
                self.assertTrue(val["state"] == "fail")
            if val["type"] == "cframe":
                self.assertTrue(val["state"] == "fail")



#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
