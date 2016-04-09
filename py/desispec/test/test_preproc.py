import unittest
import os.path
from astropy.io import fits
import numpy as np

from desispec.preproc import preproc, _parse_sec_keyword

class TestPreProc(unittest.TestCase):
    
    def tearDown(self):
        if os.path.exists(self.calibfile):
            os.remove(self.calibfile)
    
    def setUp(self):
        self.calibfile = 'test-calib-askjapqwhezcpasehadfaqp.fits'
        hdr = dict()
        hdr['CAMERA'] = 'b0'
        hdr['DATE-OBS'] = '2018-09-23T08:17:03.988'
        hdr['CCDSEC'] = '[1:200,1:150]'
        hdr['BIASSECA'] = '[1:20,1:80]'
        hdr['DATASECA'] = '[21:110,1:80]'
        hdr['CCDSECA'] =  '[1:90,1:80]'
        hdr['BIASSECB'] = '[221:240,1:80]'
        hdr['DATASECB'] = '[111:220,1:80]'
        hdr['CCDSECB'] =  '[91:200,1:80]'
        hdr['BIASSECC'] = '[1:20,81:150]'
        hdr['DATASECC'] = '[21:110,81:150]'
        hdr['CCDSECC'] =  '[1:90,81:150]'
        hdr['BIASSECD'] = '[221:240,81:150]'
        hdr['DATASECD'] = '[111:220,81:150]'
        hdr['CCDSECD'] =  '[91:200,81:150]'
        self.header = hdr
        self.ny = 150
        self.nx = 200
        self.noverscan = 20
        self.rawimage = np.zeros((self.ny, self.nx+2*self.noverscan))
        self.offset = dict(A=100.0, B=100.5, C=50.3, D=200.4)
        self.gain = dict(A=1.0, B=1.5, C=0.8, D=1.2)
        self.rdnoise = dict(A=2.0, B=2.2, C=2.4, D=2.6)
        
        self.quad = dict(
            A = np.s_[0:80, 0:90], B = np.s_[0:80, 90:200],
            C = np.s_[80:150, 0:90], D = np.s_[80:150, 90:200],
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
