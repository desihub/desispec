from __future__ import absolute_import, division, print_function

import unittest
import os.path
from astropy.io import fits
import numpy as np

from desispec.preproc import preproc, _parse_sec_keyword
from desispec import io

class TestPreProc(unittest.TestCase):
    
    def tearDown(self):
        for filename in [self.calibfile, self.rawfile]:
            if os.path.exists(filename):
                os.remove(filename)
    
    def setUp(self):
        self.calibfile = 'test-calib-askjapqwhezcpasehadfaqp.fits'
        self.rawfile = 'test-raw-askjapqwhezcpasehadfaqp.fits'
        hdr = dict()
        hdr['CAMERA'] = 'b0'
        hdr['DATE-OBS'] = '2018-09-23T08:17:03.988'

        #- [x,y] 1-indexed for FITS; in reality the amps will be symmetric
        #- but the header definitions don't require that to make sure we are
        #- getting dimensions correct

        hdr['CCDSEC'] = '[1:190,1:150]'     #- dimensions of output

        hdr['BIASSECA'] = '[101:150,1:80]'  #- overscan region in raw image
        hdr['DATASECA'] = '[1:100,1:80]'    #- data region in raw image
        hdr['CCDSECA'] =  '[1:100,1:80]'    #- where should this go in output
        
        hdr['BIASSECB'] = '[151:200,1:80]'
        hdr['DATASECB'] = '[201:290,1:80]'
        hdr['CCDSECB'] =  '[101:190,1:80]'

        hdr['BIASSECC'] = '[151:200,81:150]'
        hdr['DATASECC'] = '[201:290,81:150]'
        hdr['CCDSECC'] =  '[101:190,81:150]'

        hdr['BIASSECD'] = '[101:150,81:150]'
        hdr['DATASECD'] = '[1:100,81:150]'
        hdr['CCDSECD'] =  '[1:100,81:150]'
        
        self.header = hdr
        self.ny = 150
        self.nx = 190
        self.noverscan = 50
        self.rawimage = np.zeros((self.ny, self.nx+2*self.noverscan))
        self.offset = dict(A=100.0, B=100.5, C=50.3, D=200.4)
        self.gain = dict(A=1.0, B=1.5, C=0.8, D=1.2)
        self.rdnoise = dict(A=2.0, B=2.2, C=2.4, D=2.6)
        
        self.quad = dict(
            A = np.s_[0:80, 0:100], B = np.s_[0:80, 100:190],
            C = np.s_[80:150, 100:190], D = np.s_[80:150, 0:100],
        )
        
        for amp in ('A', 'B', 'C', 'D'):
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

        #- Confirm that all regions were correctly offset
        assert not np.any(self.rawimage == 0.0)
            
    def test_preproc(self):
        image = preproc(self.rawimage, self.header)
        self.assertEqual(image.pix.shape, (self.ny, self.nx))
        self.assertTrue(np.all(image.ivar <= 1/image.readnoise**2))
        for amp in ('A', 'B', 'C', 'D'):
            pix = image.pix[self.quad[amp]]
            rdnoise = np.median(image.readnoise[self.quad[amp]])
            self.assertAlmostEqual(np.median(pix), 0.0, delta=0.2)
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
        pixflat = np.ones((self.ny, self.nx))
        image = preproc(self.rawimage, self.header, pixflat=pixflat)
        fits.writeto(self.calibfile, pixflat)
        image = preproc(self.rawimage, self.header, pixflat=self.calibfile)
        with self.assertRaises(ValueError):
            image = preproc(self.rawimage, self.header, pixflat=pixflat[0:10, 0:10])

    def test_mask(self):
        image = preproc(self.rawimage, self.header, mask=False)
        mask = np.random.randint(0, 2, size=(self.ny, self.nx))
        image = preproc(self.rawimage, self.header, mask=mask)
        self.assertTrue(np.all(image.mask == mask))
        fits.writeto(self.calibfile, mask)
        image = preproc(self.rawimage, self.header, mask=self.calibfile)
        self.assertTrue(np.all(image.mask == mask))
        with self.assertRaises(ValueError):
            image = preproc(self.rawimage, self.header, mask=mask[0:10, 0:10])

    def test_pixflat_mask(self):
        from desispec.maskbits import ccdmask
        pixflat = np.ones((self.ny, self.nx))
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

    def test_keywords(self):
        for keyword in self.header.keys():
            #- Missing GAIN* and RDNOISE* are warnings but not errors
            if keyword.startswith('GAIN') or keyword.startswith('RDNOISE'):
                continue

            print('--', keyword, '--')

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
        del hdr['RDNOISEA']
        del hdr['RDNOISEB']
        del hdr['RDNOISEC']
        del hdr['RDNOISED']
        image = preproc(self.rawimage, hdr)
        #- Missing expected GAIN keywords should log error but not crash
        hdr = self.header.copy()
        del hdr['GAINA']
        del hdr['GAINB']
        del hdr['GAINC']
        del hdr['GAIND']
        image = preproc(self.rawimage, hdr)

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
