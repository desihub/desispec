import os
import unittest
from uuid import uuid4

import numpy as np

from desispec.resolution import Resolution
from desispec.spectra import Spectra
from desispec.fiberflat import FiberFlat
from desispec import io
from desispec.pipeline.core import runcmd

class TestRunCmd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.nspec = 5
        cls.nwave = 20
        id = uuid4().hex
        cls.framefile = 'frame-'+id
        cls.fiberflatfile = 'fiberflat-'+id
        cls.skyfile = 'sky-'+id

    @classmethod
    def tearDownClass(cls):
        """Cleanup in case tests crashed and left files behind"""
        for filename in [cls.framefile, cls.fiberflatfile, cls.skyfile]:
            if os.path.exists(filename):
                os.remove(filename)

    def test_compute_fiberflat(self):
        #- Make a fake frame
        wave = 5000+np.arange(self.nwave)
        flux = np.ones((self.nspec, self.nwave))
        ivar = np.ones((self.nspec, self.nwave))
        Rdata = np.ones((self.nspec, 1, self.nwave))
        frame = Spectra(wave, flux, ivar, Rdata)
        io.write_frame(self.framefile, frame)
        
        #- run the command and confirm error code = 0
        cmd = 'desi_compute_fiberflat.py --infile {} --outfile {}'.format(
            self.framefile, self.fiberflatfile)
        err = runcmd(cmd, [self.framefile,], [self.fiberflatfile,], clobber=True)
        self.assertEqual(err, 0)
        
        #- Confirm that the output file can be read as a fiberflat
        ff = io.read_fiberflat(self.fiberflatfile)
        
#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()           

        
        