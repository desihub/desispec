import unittest, os
from uuid import uuid1

import numpy as np

from desispec.spectra import Spectra
from desispec.fiberflat import FiberFlat
from desispec.sky import SkyModel
import desispec.io
from astropy.io import fits

class TestIO(unittest.TestCase):
    
    #- Create unique test filename in a subdirectory
    def setUp(self):
        self.testfile = 'test-{uuid}/test-{uuid}.fits'.format(uuid=uuid1())
        
    #- Cleanup test files if they exist
    def tearDown(self):
        if os.path.exists(self.testfile):
            os.remove(self.testfile)
            testpath = os.path.normpath(os.path.dirname(self.testfile))
            if testpath != '.':
                os.removedirs(testpath)
    
    def test_fitsheader(self):
        #- None is ok; just returns blank Header
        header = desispec.io.util.fitsheader(None)
        self.assertTrue(isinstance(header, fits.Header))
        self.assertEqual(len(header), 0)

        #- input is dict
        hdr = dict()
        hdr['BLAT'] = 'foo'
        hdr['BAR'] = (1, 'biz bat')
        header = desispec.io.util.fitsheader(hdr)
        self.assertTrue(isinstance(header, fits.Header))
        self.assertEqual(header['BLAT'], 'foo')
        self.assertEqual(header['BAR'], 1)
        self.assertEqual(header.comments['BAR'], 'biz bat')
        
        #- input header as a list, get a fits.Header back
        hdr = list()
        hdr.append( ('BLAT', 'foo') )
        hdr.append( ('BAR', (1, 'biz bat')) )
        header = desispec.io.util.fitsheader(hdr)
        self.assertTrue(isinstance(header, fits.Header))
        self.assertEqual(header['BLAT'], 'foo')
        self.assertEqual(header['BAR'], 1)
        self.assertEqual(header.comments['BAR'], 'biz bat')

        #- fits.Header -> fits.Header
        header = desispec.io.util.fitsheader(header)
        self.assertTrue(isinstance(header, fits.Header))
        self.assertEqual(header['BLAT'], 'foo')
        self.assertEqual(header['BAR'], 1)
        self.assertEqual(header.comments['BAR'], 'biz bat')
        
        #- Can't convert and int into a fits Header
        self.assertRaises(ValueError, desispec.io.util.fitsheader, (1,))
        
    def test_frame_rw(self):
        nspec, nwave, ndiag = 5, 10, 3
        flux = np.random.uniform(size=(nspec, nwave))
        ivar = np.random.uniform(size=(nspec, nwave))
        mask = np.zeros((nspec, nwave), dtype=int)
        wave = np.arange(nwave)
        R = np.random.uniform( size=(nspec, ndiag, nwave) )
        spx = Spectra(wave, flux, ivar, mask, R)
                
        desispec.io.write_frame(self.testfile, spx)
        spectra = desispec.io.read_frame(self.testfile)

        self.assertTrue(np.all(flux == spectra.flux))
        self.assertTrue(np.all(ivar == spectra.ivar))
        self.assertTrue(np.all(wave == spectra.wave))
        self.assertTrue(np.all(mask == spectra.mask))
        self.assertTrue(np.all(R == spectra.resolution_data))
        self.assertTrue(spectra.resolution_data.dtype.isnative)
        
    def test_sky_rw(self):
        nspec, nwave = 5,10 
        wave = np.arange(nwave)
        flux = np.random.uniform(size=(nspec, nwave))
        ivar = np.random.uniform(size=(nspec, nwave))
        mask = np.zeros(shape=(nspec, nwave), dtype=int)

        # skyflux,skyivar,skymask,cskyflux,cskyivar,wave
        sky = SkyModel(wave, flux, ivar, mask)
        desispec.io.write_sky(self.testfile, sky)
        xsky = desispec.io.read_sky(self.testfile)
                
        self.assertTrue(np.all(sky.wave  == xsky.wave))
        self.assertTrue(np.all(sky.flux  == xsky.flux))
        self.assertTrue(np.all(sky.ivar  == xsky.ivar))
        self.assertTrue(np.all(sky.mask  == xsky.mask))
        self.assertTrue(xsky.flux.dtype.isnative)
                
    # fiberflat,fiberflat_ivar,fiberflat_mask,mean_spectrum,wave
    def test_fiberflat_rw(self):
        nspec, nwave, ndiag = 10, 20, 3
        flat = np.random.uniform(size=(nspec, nwave))
        ivar = np.random.uniform(size=(nspec, nwave))
        mask = np.zeros(shape=(nspec, nwave), dtype=int)
        meanspec = np.random.uniform(size=(nwave,))
        wave = np.arange(nwave)

        ff = FiberFlat(wave, flat, ivar, mask, meanspec)

        desispec.io.write_fiberflat(self.testfile, ff)
        xff = desispec.io.read_fiberflat(self.testfile)
                
        self.assertTrue(np.all(ff.fiberflat == xff.fiberflat))
        self.assertTrue(np.all(ff.ivar == xff.ivar))
        self.assertTrue(np.all(ff.mask == xff.mask))
        self.assertTrue(np.all(ff.meanspec == xff.meanspec))
        self.assertTrue(np.all(ff.wave == xff.wave))

        self.assertTrue(xff.fiberflat.dtype.isnative)
        self.assertTrue(xff.ivar.dtype.isnative)
        self.assertTrue(xff.mask.dtype.isnative)
        self.assertTrue(xff.meanspec.dtype.isnative)
        self.assertTrue(xff.wave.dtype.isnative)
                
    def test_fibermap_rw(self):
        fibermap = desispec.io.fibermap.empty_fibermap(10)
        for key in fibermap.dtype.names:
            column = fibermap[key]
            fibermap[key] = np.random.random(column.shape).astype(column.dtype)
            
        desispec.io.write_fibermap(self.testfile, fibermap)
        
        #- Read without and with header
        fm = desispec.io.read_fibermap(self.testfile)
        self.assertTrue(isinstance(fm, np.ndarray))

        fm, hdr = desispec.io.read_fibermap(self.testfile, header=True)
        self.assertTrue(isinstance(fm, np.ndarray))
        self.assertTrue(isinstance(hdr, fits.Header))
                
        self.assertEqual(set(fibermap.dtype.names), set(fm.dtype.names))
        for key in fibermap.dtype.names:
            c1 = fibermap[key]
            c2 = fm[key]
            #- Endianness may change, but kind, size, shape, and values are same
            self.assertEqual(c1.dtype.kind, c2.dtype.kind)
            self.assertEqual(c1.dtype.itemsize, c2.dtype.itemsize)
            self.assertEqual(c1.shape, c2.shape)
            self.assertTrue(np.all(c1 == c2))
                
    def test_native_endian(self):
        for dtype in ('>f8', '<f8', '<f4', '>f4', '>i4', '<i4', '>i8', '<i8'):
            data1 = np.arange(100).astype(dtype)
            data2 = desispec.io.util.native_endian(data1)
            self.assertTrue(data2.dtype.isnative, dtype+' is not native endian')
            self.assertTrue(np.all(data1 == data2))

#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()           
