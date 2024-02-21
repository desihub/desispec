"""
tests minimal dependency basic functionality
"""

import unittest
import os, sys, tempfile
import shutil

import numpy as np
from astropy.table import Table

from desitarget.targetmask import desi_mask

import desispec.io
from desispec.spectra import Spectra, stack
import desispec.coaddition
from desispec.specscore import tophat_wave

class TestLite(unittest.TestCase):

    #- Create unique test filename in a subdirectory
    @classmethod
    def setUpClass(cls):
        cls.testdir = tempfile.mkdtemp()
        cls.infile = f'{cls.testdir}/input-coadd.fits'
        cls.outfile = f'{cls.testdir}/output-coadd.fits'

        # create spectra, but don't use desispec.test.util.get_blank_spectra
        # and desispec.io.fibermap.empty_fibermap to avoid desimodel dependency
        nspec = 5
        bands = ('b', 'r', 'z')
        flux = dict()
        ivar = dict()
        wave = dict()
        rdat = dict()
        for i, band in enumerate(bands):
            wave[band] = np.arange(tophat_wave[band][0]-10, tophat_wave[band][1]+10)
            nwave = len(wave[band])
            flux[band] = np.ones( (nspec, nwave) )
            ivar[band] = np.ones( (nspec, nwave) )
            wave[band] = 5000 + i*nwave//2 + np.arange(nwave)
            rdat[band] = np.ones( (nspec, 1, nwave) )

        fibermap = Table()
        fibermap['TARGETID'] = np.arange(nspec)
        fibermap['FIBER'] = np.arange(nspec, dtype='i4')
        fibermap['TILEID'] = 1000 * np.ones(nspec, dtype='i4')
        fibermap['FIBERSTATUS'] = np.zeros(nspec, dtype='i4')
        fibermap['DESI_TARGET'] = np.arange(nspec, dtype='i8')

        cls.spectra = Spectra(bands=bands, wave=wave, flux=flux, ivar=ivar,
                              resolution_data=rdat, fibermap=fibermap)
        desispec.io.write_spectra(cls.infile, cls.spectra)

    def tearDown(self):
        if os.path.exists(self.outfile):
            os.remove(self.outfile)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.testdir)

    def test_filter_stack_coadd(self):
        sp1 = desispec.io.read_spectra(self.infile)
        sp2 = desispec.io.read_spectra(self.infile)
        keep = (sp1.fibermap['DESI_TARGET'] & desi_mask.QSO) != 0

        sp1 = sp1[keep]
        sp2 = sp2[keep]

        #- stack two sets of spectra, should double length
        sp = stack([sp1, sp2])
        self.assertEqual(len(sp.fibermap), 2*len(sp1.fibermap))

        #- in place coaddition; back to a single set of targets
        desispec.coaddition.coadd(sp)
        self.assertEqual(len(sp.fibermap), len(sp1.fibermap))

        #- coadd across cameras
        sp = desispec.coaddition.coadd_cameras(sp)

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
