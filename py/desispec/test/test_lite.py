"""
tests minimal dependency basic functionality
"""

import unittest
import os, sys, tempfile

import numpy as np

from desitarget.targetmask import desi_mask

import desispec.io
from desispec.spectra import Spectra, stack
import desispec.coaddition

class TestLite(unittest.TestCase):

    #- Create unique test filename in a subdirectory
    def setUp(self):
        self.specfile = os.path.expandvars('$DESI_ROOT/spectro/redux/iron/tiles/cumulative/1000/20210517/coadd-0-1000-thru20210517.fits')

    def test_filter_stack_coadd(self):
        sp1 = desispec.io.read_spectra(self.specfile)
        sp2 = desispec.io.read_spectra(self.specfile)
        keep = (sp1.fibermap['DESI_TARGET'] & desi_mask.QSO) != 0
        sp1 = sp1[keep]
        sp2 = sp2[keep]

        #- stack two sets of spectra, should double length
        sp = stack([sp1, sp2])
        self.assertEqual(len(sp.fibermap), 2*len(sp1.fibermap))

        #- in place coaddition; back to a single set of targets
        desispec.coaddition.coadd(sp)
        self.assertEqual(len(sp.fibermap), len(sp1.fibermap))

        #- write the coadd to a new file
        with tempfile.TemporaryDirectory() as tempdir:
            outfile = f'{tempdir}/coadd.fits'
            desispec.io.write_spectra(outfile, sp)
            self.assertTrue(os.path.exists(outfile))


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)

if __name__ == '__main__':
    unittest.main()
