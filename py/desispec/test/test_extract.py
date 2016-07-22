from __future__ import absolute_import, division, print_function, unicode_literals

try:
    from specter.psf import load_psf
    nospecter = False
except ImportError:
    from desispec.log import get_logger
    log = get_logger()
    log.error('specter not installed; skipping extraction tests')
    nospecter = True

import unittest
import uuid
import os
from pkg_resources import resource_filename

import desispec.image
import desispec.io
import desispec.scripts.extract

from astropy.io import fits
import numpy as np

class TestExtract(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        testhash = uuid.uuid4()
        cls.imgfile = 'test-img-{}.fits'.format(testhash)
        cls.outfile = 'test-out-{}.fits'.format(testhash)
        cls.fibermapfile = 'test-fibermap-{}.fits'.format(testhash)
        cls.psffile = resource_filename('specter', 'test/t/psf-monospot.fits')
        # cls.psf = load_psf(cls.psffile)
        
        pix = np.random.normal(0, 3.0, size=(500,500))
        ivar = np.ones_like(pix) / 3.0**2
        mask = np.zeros(pix.shape, dtype=np.uint32)
        mask[200] = 1
        img = desispec.image.Image(pix, ivar, mask, camera='z0')
        desispec.io.write_image(cls.imgfile, img)
        
        fibermap = desispec.io.empty_fibermap(100)
        desispec.io.write_fibermap(cls.fibermapfile, fibermap)

    def setUp(self):
        for filename in (self.outfile, ):
            if os.path.exists(filename):
                os.remove(filename)
        
    @classmethod
    def tearDownClass(cls):
        for filename in (cls.imgfile, cls.outfile, cls.fibermapfile):
            if os.path.exists(filename):
                os.remove(filename)
           
    @unittest.skipIf(nospecter, 'specter not installed; skipping extraction test')
    def test_extract(self):
        template = "desi_extract_spectra -i {} -p {} -w 7500,7600,0.75 -f {} -s 0 -n 4 --bundlesize 2 -o {}"
        
        cmd = template.format(self.imgfile, self.psffile, self.fibermapfile, self.outfile)
        opts = cmd.split(" ")[1:]
        args = desispec.scripts.extract.parse(opts)
        desispec.scripts.extract.main(args)

        self.assertTrue(os.path.exists(self.outfile))
        frame1 = desispec.io.read_frame(self.outfile)
        os.remove(self.outfile)
        
        desispec.scripts.extract.main_mpi(args, comm=None)
        self.assertTrue(os.path.exists(self.outfile))
        frame2 = desispec.io.read_frame(self.outfile)
        
        self.assertTrue(np.all(frame1.flux[0:4] == frame2.flux[0:4]))
        self.assertTrue(np.all(frame1.ivar[0:4] == frame2.ivar[0:4]))
        self.assertTrue(np.all(frame1.mask[0:4] == frame2.mask[0:4]))
        self.assertTrue(np.all(frame1.chi2pix[0:4] == frame2.chi2pix[0:4]))
        self.assertTrue(np.all(frame1.resolution_data[0:4] == frame2.resolution_data[0:4]))        

if __name__ == '__main__':
    unittest.main()
