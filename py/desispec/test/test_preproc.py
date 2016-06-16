from __future__ import absolute_import, division, print_function

import unittest
import os
import os.path
from astropy.io import fits
import numpy as np

import desispec.scripts.preproc
from desispec.preproc import preproc, _parse_sec_keyword, _clipped_std_bias
from desispec import io

def xy2hdr(xyslice):
    '''
    convert 2D slice into IRAF style [a:b,c:d] header value
    
    e.g. xyslice2header(np.s_[0:10, 5:20]) -> '[6:20,1:10]'
    '''
    yy, xx = xyslice
    value = '[{}:{},{}:{}]'.format(xx.start+1, xx.stop, yy.start+1, yy.stop)
    return value

class TestPreProc(unittest.TestCase):
    
    def tearDown(self):
        for filename in [self.calibfile, self.rawfile, self.pixfile]:
            if os.path.exists(filename):
                os.remove(filename)
    
    def setUp(self):
        self.calibfile = 'test-calib-askjapqwhezcpasehadfaqp.fits'
        self.rawfile = 'test-raw-askjapqwhezcpasehadfaqp.fits'
        self.pixfile = 'test-pix-askjapqwhezcpasehadfaqp.fits'
        hdr = dict()
        hdr['CAMERA'] = 'b0'
        hdr['DATE-OBS'] = '2018-09-23T08:17:03.988'

        #- [x,y] 1-indexed for FITS; in reality the amps will be symmetric
        #- but the header definitions don't require that to make sure we are
        #- getting dimensions correct

        #- Dimensions per amp, not full 4-quad CCD
        self.ny = ny = 500
        self.nx = nx = 400
        self.noverscan = nover = 50

        #- BIASSEC = overscan region in raw image
        #- DATASEC = data region in raw image
        #- CCDSEC = where should this go in output

        hdr['BIASSEC1'] = xy2hdr(np.s_[0:ny, nx:nx+nover])
        hdr['DATASEC1'] = xy2hdr(np.s_[0:ny, 0:nx])
        hdr['CCDSEC1'] = xy2hdr(np.s_[0:ny, 0:nx])
        
        hdr['BIASSEC2'] = xy2hdr(np.s_[0:ny, nx+nover:nx+2*nover])
        hdr['DATASEC2'] = xy2hdr(np.s_[0:ny, nx+2*nover:nx+2*nover+nx])
        hdr['CCDSEC2'] =  xy2hdr(np.s_[0:ny, nx:nx+nx])

        hdr['BIASSEC3'] = xy2hdr(np.s_[ny:ny+ny, nx:nx+nover])
        hdr['DATASEC3'] = xy2hdr(np.s_[ny:ny+ny, 0:nx])
        hdr['CCDSEC3'] = xy2hdr(np.s_[ny:ny+ny, 0:nx])
        
        hdr['BIASSEC4'] = xy2hdr(np.s_[ny:ny+ny, nx+nover:nx+2*nover])
        hdr['DATASEC4'] = xy2hdr(np.s_[ny:ny+ny, nx+2*nover:nx+2*nover+nx])
        hdr['CCDSEC4'] =  xy2hdr(np.s_[ny:ny+ny, nx:nx+nx])
        
        hdr['NIGHT'] = '20150102'
        hdr['EXPID'] = 1
        
        self.header = hdr
        self.rawimage = np.zeros((2*self.ny, 2*self.nx+2*self.noverscan))
        self.offset = {'1':100.0, '2':100.5, '3':50.3, '4':200.4}
        self.gain = {'1':1.0, '2':1.5, '3':0.8, '4':1.2}
        self.rdnoise = {'1':2.0, '2':2.2, '3':2.4, '4':2.6}
        
        self.quad = {
            '1': np.s_[0:ny, 0:nx], '2': np.s_[0:ny, nx:nx+nx],
            '3': np.s_[ny:ny+ny, 0:nx], '4': np.s_[ny:ny+ny, nx:nx+nx],
        }
        
        for amp in ('1', '2', '3', '4'):
            self.header['GAIN'+amp] = self.gain[amp]
            self.header['RDNOISE'+amp] = self.rdnoise[amp]
            
            xy = _parse_sec_keyword(hdr['BIASSEC'+amp])
            shape = [xy[0].stop-xy[0].start, xy[1].stop-xy[1].start]
            self.rawimage[xy] += self.offset[amp]
            self.rawimage[xy] += np.random.normal(scale=self.rdnoise[amp], size=shape)/self.gain[amp]
            xy = _parse_sec_keyword(hdr['DATASEC'+amp])
            shape = [xy[0].stop-xy[0].start, xy[1].stop-xy[1].start]
            self.rawimage[xy] += self.offset[amp]
            self.rawimage[xy] += np.random.normal(scale=self.rdnoise[amp], size=shape)/self.gain[amp]

        #- raw data are integers, not floats
        self.rawimage = self.rawimage.astype(np.int32)

        #- Confirm that all regions were correctly offset
        assert not np.any(self.rawimage == 0.0)
            
    def test_preproc(self):
        image = preproc(self.rawimage, self.header)
        self.assertEqual(image.pix.shape, (2*self.ny, 2*self.nx))
        self.assertTrue(np.all(image.ivar <= 1/image.readnoise**2))
        for amp in ('1', '2', '3', '4'):
            pix = image.pix[self.quad[amp]]
            rdnoise = np.median(image.readnoise[self.quad[amp]])
            npixover = self.ny * self.noverscan
            self.assertAlmostEqual(np.mean(pix), 0.0, delta=3*rdnoise/np.sqrt(npixover))
            self.assertAlmostEqual(np.std(pix), self.rdnoise[amp], delta=0.2)
            self.assertAlmostEqual(rdnoise, self.rdnoise[amp], delta=0.2)

    def test_bias(self):
        image = preproc(self.rawimage, self.header, bias=False)
        bias = np.zeros(self.rawimage.shape)
        image = preproc(self.rawimage, self.header, bias=bias)
        fits.writeto(self.calibfile, bias)
        image = preproc(self.rawimage, self.header, bias=self.calibfile)
        with self.assertRaises(ValueError):
            image = preproc(self.rawimage, self.header, bias=bias[0:10, 0:10])

    def test_pixflat(self):
        image = preproc(self.rawimage, self.header, pixflat=False)
        pixflat = np.ones_like(image.pix)
        image = preproc(self.rawimage, self.header, pixflat=pixflat)
        fits.writeto(self.calibfile, pixflat)
        image = preproc(self.rawimage, self.header, pixflat=self.calibfile)
        with self.assertRaises(ValueError):
            image = preproc(self.rawimage, self.header, pixflat=pixflat[0:10, 0:10])

    def test_mask(self):
        image = preproc(self.rawimage, self.header, mask=False)
        mask = np.random.randint(0, 2, size=image.pix.shape)
        image = preproc(self.rawimage, self.header, mask=mask)
        self.assertTrue(np.all(image.mask == mask))
        fits.writeto(self.calibfile, mask)
        image = preproc(self.rawimage, self.header, mask=self.calibfile)
        self.assertTrue(np.all(image.mask == mask))
        with self.assertRaises(ValueError):
            image = preproc(self.rawimage, self.header, mask=mask[0:10, 0:10])

    def test_pixflat_mask(self):
        from desispec.maskbits import ccdmask
        pixflat = np.ones((2*self.ny, 2*self.nx))
        pixflat[0:10, 0:10] = 0.0
        pixflat[10:20, 10:20] = 0.05
        image = preproc(self.rawimage, self.header, pixflat=pixflat)
        self.assertTrue(np.all(image.mask[0:10,0:10] & ccdmask.PIXFLATZERO))
        self.assertTrue(np.all(image.mask[10:20,10:20] & ccdmask.PIXFLATLOW))

    def test_io(self):
        io.write_raw(self.rawfile, self.rawimage, self.header, camera='b0')
        io.write_raw(self.rawfile, self.rawimage, self.header, camera='r1')
        io.write_raw(self.rawfile, self.rawimage, self.header, camera='z9')
        b0 = io.read_raw(self.rawfile, 'b0')
        r1 = io.read_raw(self.rawfile, 'r1')
        z9 = io.read_raw(self.rawfile, 'z9')
        
    def test_32_64(self):
        '''
        64-bit integers aren't supported for compressed HDUs;
        make sure we handle that gracefully
        '''
        data64 = np.linspace(0, 2**60, 10, dtype=np.int64)
        datasmall64 = np.linspace(0, 2**30, 10, dtype=np.int64)
        data32 = np.linspace(0, 2**30, 10, dtype=np.int32)
        data16 = np.linspace(0, 2**10, 10, dtype=np.int16)

        #- Primary HDU should be blank
        #- Should be written as vanilla ImageHDU
        io.write_raw(self.rawfile, data64, self.header, camera='b0')
        #- Should be written as vanilla ImageHDU
        io.write_raw(self.rawfile, data64, self.header, camera='b1')
        #- Should be converted to 32-bit CompImageHDU
        io.write_raw(self.rawfile, datasmall64, self.header, camera='b2')
        #- Should be 32-bit CompImageHDU
        io.write_raw(self.rawfile, data32, self.header, camera='b3')
        #- Should be 16-bit CompImageHDU
        io.write_raw(self.rawfile, data16, self.header, camera='b4')
        
        fx = fits.open(self.rawfile)
                
        #- Blank PrimaryHDU should have been inserted
        self.assertTrue(isinstance(fx[0], fits.PrimaryHDU))
        self.assertTrue(fx[0].data == None)
        #- 64-bit image written uncompressed after blank HDU
        self.assertTrue(isinstance(fx[1], fits.ImageHDU))
        self.assertEqual(fx[1].data.dtype, np.dtype('>i8'))
        self.assertEqual(fx[1].header['EXTNAME'], 'B0')
        
        #- 64-bit image written uncompressed
        self.assertTrue(isinstance(fx[2], fits.ImageHDU))
        self.assertEqual(fx[2].data.dtype, np.dtype('>i8'))
        self.assertEqual(fx[2].header['EXTNAME'], 'B1')
        
        #- 64-bit image with small numbers converted to 32-bit compressed
        self.assertTrue(isinstance(fx[3], fits.CompImageHDU))
        self.assertEqual(fx[3].data.dtype, np.int32)
        self.assertEqual(fx[3].header['EXTNAME'], 'B2')
        
        #- 32-bit image written compressed
        self.assertTrue(isinstance(fx[4], fits.CompImageHDU))
        self.assertEqual(fx[4].data.dtype, np.int32)
        self.assertEqual(fx[4].header['EXTNAME'], 'B3')

        #- 16-bit image written compressed
        self.assertTrue(isinstance(fx[5], fits.CompImageHDU))
        self.assertEqual(fx[5].data.dtype, np.int16)
        self.assertEqual(fx[5].header['EXTNAME'], 'B4')

    def test_keywords(self):
        for keyword in self.header.keys():
            #- Missing GAIN* and RDNOISE* are warnings but not errors
            if keyword.startswith('GAIN') or keyword.startswith('RDNOISE'):
                continue
            
            #- DATE-OBS, NIGHT, and EXPID are also optional
            #- (but maybe they should be required...)
            if keyword in ('DATE-OBS', 'NIGHT', 'EXPID'):
                continue

            if os.path.exists(self.rawfile):
                os.remove(self.rawfile)
            value = self.header[keyword]
            del self.header[keyword]
            with self.assertRaises(KeyError):
                io.write_raw(self.rawfile, self.rawimage, self.header)
            self.header[keyword] = value
            
        dateobs = self.header

    #- striving for 100% coverage...
    def test_pedantic(self):
        with self.assertRaises(ValueError):
            _parse_sec_keyword('blat')
        #- should log a warning about large readnoise
        rawimage = self.rawimage + np.random.normal(scale=2, size=self.rawimage.shape)
        image = preproc(rawimage, self.header)
        #- should log an error about huge readnoise
        rawimage = self.rawimage + np.random.normal(scale=10, size=self.rawimage.shape)
        image = preproc(rawimage, self.header)
        #- should log a warning about small readnoise
        rdnoise = 0.7 * np.mean(self.rdnoise.values())
        rawimage = np.random.normal(scale=rdnoise, size=self.rawimage.shape)
        image = preproc(rawimage, self.header)
        #- should log a warning about tiny readnoise
        rdnoise = 0.01 * np.mean(self.rdnoise.values())
        rawimage = np.random.normal(scale=rdnoise, size=self.rawimage.shape)
        image = preproc(rawimage, self.header)
        #- Missing expected RDNOISE keywords shouldn't be fatal
        hdr = self.header.copy()
        del hdr['RDNOISE1']
        del hdr['RDNOISE2']
        del hdr['RDNOISE3']
        del hdr['RDNOISE4']
        image = preproc(self.rawimage, hdr)
        #- Missing expected GAIN keywords should log error but not crash
        hdr = self.header.copy()
        del hdr['GAIN1']
        del hdr['GAIN2']
        del hdr['GAIN3']
        del hdr['GAIN4']
        image = preproc(self.rawimage, hdr)

    def test_preproc_script(self):
        io.write_raw(self.rawfile, self.rawimage, self.header, camera='b0')
        io.write_raw(self.rawfile, self.rawimage, self.header, camera='b1')
        args = ['--infile', self.rawfile, '--cameras', 'b1',
                '--pixfile', self.pixfile]
        if os.path.exists(self.pixfile):
            os.remove(self.pixfile)            
        desispec.scripts.preproc.main(args)
        img = io.read_image(self.pixfile)        
        self.assertEqual(img.pix.shape, (2*self.ny, 2*self.nx))

    def test_clipped_std_bias(self):
        '''Compare to www.wolframalpha.com integrals'''
        self.assertAlmostEqual(_clipped_std_bias(1), 0.53956, places=5)
        self.assertAlmostEqual(_clipped_std_bias(2), 0.879626, places=6)
        self.assertAlmostEqual(_clipped_std_bias(3), 0.986578, places=6)
        np.random.seed(1)
        x = np.random.normal(size=1000000)
        biased_std = np.std(x[np.abs(x)<3])
        self.assertAlmostEqual(biased_std, _clipped_std_bias(3), places=3)

    #- Not implemented yet, but flag these as expectedFailures instead of
    #- successful tests of raising NotImplementedError
    @unittest.expectedFailure
    def test_default_bias(self):
        image = preproc(self.rawimage, self.header, bias=True)

    @unittest.expectedFailure
    def test_default_pixflat(self):
        image = preproc(self.rawimage, self.header, pixflat=True)

    @unittest.expectedFailure
    def test_default_mask(self):
        image = preproc(self.rawimage, self.header, mask=True)
        
                
if __name__ == '__main__':
    unittest.main()
