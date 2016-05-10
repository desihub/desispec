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

class TestRunCmd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.testraw = 'test-'+uuid4().hex
        os.mkdir(cls.testraw)

        cls.night = time.strftime('%Y%m%d', time.localtime(time.time()-12*3600))
        cls.nightdir = os.path.join(cls.testraw, cls.night)
        os.mkdir(cls.nightdir)


    @classmethod
    def tearDownClass(cls):
        #shutil.rmtree(cls.testraw)
        pass

    
    def test_options(self):
        options = default_options()
        dump = os.path.join(self.testraw, "opts.yml")
        write_options(dump, options)
    

#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()           
