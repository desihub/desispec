# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desispec.io.
"""

import unittest
import os
import shutil
from importlib import resources



from desispec.calibfinder import CalibFinder

class TestCalibFinder(unittest.TestCase):
    """Test desispec.calibfinder
    """
    def tearDown(self):
        pass
        if os.path.isdir(self.calibdir) :
            shutil.rmtree(self.calibdir)

    def setUp(self):    
        #- Create temporary calib directory
        self.calibdir  = os.path.join(os.environ['HOME'], 'preproc_unit_test')
        if not os.path.exists(self.calibdir): os.makedirs(self.calibdir)
        #- Copy test calibration-data.yaml file 
        specdir=os.path.join(self.calibdir,"spec/sp0")
        if not os.path.isdir(specdir) :
            os.makedirs(specdir)
        for c in "brz" :
            shutil.copy(str(resources.files('desispec').joinpath(f'test/data/ql/{c}0.yaml')), os.path.join(specdir,f"{c}0.yaml"))
        #- Set calibration environment variable    
        os.environ["DESI_SPECTRO_CALIB"] = self.calibdir
        
    
    def test_init(self):
        """Cleanup test files if they exist.
        """
        
        pheader={"DATE-OBS":'2018-11-30T12:42:10.442593-05:00',"DOSVER":'SIM'}
        header={"DETECTOR":'SIM',"CAMERA":'b0      ',"FEEVER":'SIM'}
        cfinder = CalibFinder([pheader,header])
        print(cfinder.value("DETECTOR"))
        if cfinder.haskey("BIAS") :
            print(cfinder.findfile("BIAS"))
