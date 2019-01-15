# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Test desispec.io.
"""

import unittest

from desispec.calibfinder import CalibFinder

class TestCalibFinder(unittest.TestCase):
    """Test desispec.calibfinder
    """
    
    def test_init(self):
        """Cleanup test files if they exist.
        """
        
        pheader={"DATE-OBS":'2018-11-30T12:42:10.442593-05:00',"DOSVER":'winlight'}
        header={"DETECTOR":'sn22794   ',"CAMERA":'b2      ',"FEEVER":'v20160312'}
        cfinder = CalibFinder([pheader,header])
        print(cfinder.value("DETECTOR"))
        if cfinder.haskey("BIAS") :
            print(cfinder.findfile("BIAS"))
        

if __name__ == '__main__':
    unittest.main()
