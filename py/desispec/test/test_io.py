import unittest, os
from uuid import uuid1

import numpy as np

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
        nspec, nwave, ndiag = 10, 20, 3
        flux = np.random.uniform(size=(nspec, nwave))
        ivar = np.random.uniform(size=(nspec, nwave))
        wave = np.arange(nwave)
        R = np.random.uniform( size=(nspec, ndiag, nwave) )
                
        desispec.io.write_frame(self.testfile, flux, ivar, wave, R)
        xflux, xivar, xwave, xR, hdr = desispec.io.read_frame(self.testfile)

        self.assertTrue(np.all(flux == xflux))
        self.assertTrue(np.all(ivar == xivar))
        self.assertTrue(np.all(wave == xwave))
        self.assertTrue(np.all(R == xR))
        self.assertTrue(R.dtype.isnative)
        
    def test_sky_rw(self):
        nspec, nwave, ndiag = 10, 20, 3
        flux = np.random.uniform(size=(nspec, nwave))
        ivar = np.random.uniform(size=(nspec, nwave))
        mask = np.zeros(shape=(nspec, nwave))
        cflux = np.random.uniform(size=(nspec, nwave))
        civar = np.random.uniform(size=(nspec, nwave))
        wave = np.arange(nwave)

        # skyflux,skyivar,skymask,cskyflux,cskyivar,wave
        desispec.io.write_sky(self.testfile, flux, ivar, mask, cflux, civar, wave)
        xflux, xivar, xmask, xcflux, xcivar, xwave, hdr = desispec.io.read_sky(self.testfile)
                
        self.assertTrue(np.all(flux == xflux))
        self.assertTrue(np.all(ivar == xivar))
        self.assertTrue(np.all(cflux == xcflux))
        self.assertTrue(np.all(civar == xcivar))
        self.assertTrue(np.all(mask == xmask))
        self.assertTrue(np.all(wave == xwave))
        self.assertTrue(flux.dtype.isnative)
                
    # fiberflat,fiberflat_ivar,fiberflat_mask,mean_spectrum,wave
    def test_fiberflat_rw(self):
        nspec, nwave, ndiag = 10, 20, 3
        flat = np.random.uniform(size=(nspec, nwave))
        ivar = np.random.uniform(size=(nspec, nwave))
        mask = np.zeros(shape=(nspec, nwave))
        meanspec = np.random.uniform(size=(nspec, nwave))
        wave = np.arange(nwave)

        desispec.io.write_fiberflat(self.testfile, flat, ivar, mask, meanspec, wave)
        xflat, xivar, xmask, xmeanspec, xwave, hdr = desispec.io.read_fiberflat(self.testfile)
                
        self.assertTrue(np.all(flat == xflat))
        self.assertTrue(np.all(ivar == xivar))
        self.assertTrue(np.all(mask == xmask))
        self.assertTrue(np.all(meanspec == xmeanspec))
        self.assertTrue(np.all(wave == xwave))
        self.assertTrue(flat.dtype.isnative)
                
    def test_fibermap_rw(self):
        fibermap = desispec.io.fibermap.empty_fibermap(10)
        for key in fibermap.dtype.names:
            column = fibermap[key]
            fibermap[key] = np.random.random(column.shape).astype(column.dtype)
            
        desispec.io.write_fibermap(self.testfile, fibermap)
        fm, hdr = desispec.io.read_fibermap(self.testfile)
                
        self.assertEqual(set(fibermap.dtype.names), set(fm.dtype.names))
        for key in fibermap.dtype.names:
            c1 = fibermap[key]
            c2 = fm[key]
            #- Endianness may change, but kind, size, and values are same
            self.assertEqual(c1.dtype.kind, c2.dtype.kind)
            self.assertEqual(c1.dtype.itemsize, c2.dtype.itemsize)
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
