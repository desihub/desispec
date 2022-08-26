from __future__ import absolute_import, division, print_function

import unittest
import os
import os.path
import warnings
from astropy.io import fits
import numpy as np
import shutil
from pkg_resources import resource_filename

import desispec.scripts.preproc
from desispec.preproc import preproc, parse_sec_keyword, _clipped_std_bias
from desispec.preproc import get_amp_ids, get_readout_mode
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
        pass
        if os.path.isdir(self.calibdir) :
            shutil.rmtree(self.calibdir)

    def setUp(self):
        #- catch specific warnings so that we can find and fix
        # warnings.filterwarnings("error", ".*did not parse as fits unit.*")

        #- Create temporary calib directory
        self.calibdir  = os.path.join(os.environ['HOME'], 'preproc_unit_test')
        if not os.path.exists(self.calibdir): os.makedirs(self.calibdir)
        #- Copy test calibration-data.yaml file
        specdir=os.path.join(self.calibdir,"spec/sp0")
        if not os.path.isdir(specdir) :
            os.makedirs(specdir)
        for c in "brz" :
            shutil.copy(resource_filename('desispec', 'test/data/ql/{}0.yaml'.format(c)),os.path.join(specdir,"{}0.yaml".format(c)))
        #- Set calibration environment variable
        os.environ["DESI_SPECTRO_CALIB"] = self.calibdir

        self.calibfile = os.path.join(self.calibdir,'test-calib-askjapqwhezcpasehadfaqp.fits')
        self.rawfile   = os.path.join(self.calibdir,'desi-raw-askjapqwhezcpasehadfaqp.fits')
        self.pixfile   = os.path.join(self.calibdir,'test-pix-askjapqwhezcpasehadfaqp.fits')

        primary_hdr = dict()
        primary_hdr['DATE-OBS'] = '2018-09-23T08:17:03.988'
        primary_hdr['DOSVER']   = 'SIM' # ICS version

        hdr = dict()
        hdr['CAMERA'] = 'b0'
        hdr['DETECTOR'] = 'SIM' # CCD chip identifier
        hdr['FEEVER']   = 'SIM' # readout electronic

        #- [x,y] 1-indexed for FITS; in reality the amps will be symmetric
        #- but the header definitions don't require that to make sure we are
        #- getting dimensions correct

        #- Dimensions per amp, not full 4-quad CCD
        self.ny = ny = 500
        self.nx = nx = 400
        self.noverscan = nover = 50
        self.noverscan_row = nover_row = 50

        #- ORSEC = overscan region in raw image (rows)
        #- BIASSEC = overscan region in raw image (columns)
        #- DATASEC = data region in raw image
        #- CCDSEC = where should this go in output

        hdr['ORSECA'] = xy2hdr(np.s_[ny:ny+nover_row, 0:nx])
        hdr['BIASSECA'] = xy2hdr(np.s_[0:ny, nx:nx+nover])
        hdr['DATASECA'] = xy2hdr(np.s_[0:ny, 0:nx])
        hdr['CCDSECA'] = xy2hdr(np.s_[0:ny, 0:nx])

        hdr['ORSECB'] = xy2hdr(np.s_[ny:ny+nover_row, nx+2*nover:nx+2*nover+nx])
        hdr['BIASSECB'] = xy2hdr(np.s_[0:ny, nx+nover:nx+2*nover])
        hdr['DATASECB'] = xy2hdr(np.s_[0:ny, nx+2*nover:nx+2*nover+nx])
        hdr['CCDSECB'] =  xy2hdr(np.s_[0:ny, nx:nx+nx])

        hdr['ORSECC'] = xy2hdr(np.s_[ny+nover_row:ny+2*nover_row, 0:nx])
        hdr['BIASSECC'] = xy2hdr(np.s_[ny+2*nover_row:ny+ny+2*nover_row, nx:nx+nover])
        hdr['DATASECC'] = xy2hdr(np.s_[ny+2*nover_row:ny+ny+2*nover_row, 0:nx])
        hdr['CCDSECC'] = xy2hdr(np.s_[ny:ny+ny, 0:nx])

        hdr['ORSECD'] = xy2hdr(np.s_[ny+nover_row:ny+2*nover_row, nx+2*nover:nx+2*nover+nx])
        hdr['BIASSECD'] = xy2hdr(np.s_[ny+2*nover_row:ny+ny+2*nover_row, nx+nover:nx+2*nover])
        hdr['DATASECD'] = xy2hdr(np.s_[ny+2*nover_row:ny+ny+2*nover_row, nx+2*nover:nx+2*nover+nx])
        hdr['CCDSECD'] =  xy2hdr(np.s_[ny:ny+ny, nx:nx+nx])

        hdr['NIGHT'] = '20150102'
        hdr['EXPID'] = 1
        hdr['EXPTIME'] = 10.0

        # add to header the minimal set of keywords needed to
        # identify the config in the ccd_calibration.yaml file

        self.primary_header = primary_hdr
        self.header = hdr
        self.rawimage = np.zeros((2*self.ny+2*self.noverscan_row, 2*self.nx+2*self.noverscan))
        self.offset = {'A':100.0, 'B':100.5, 'C':50.3, 'D':200.4}
        self.offset_row = {'A':50.0, 'B':50.5, 'C':20.3, 'D':40.4}
        self.gain = {'A':1.0, 'B':1.5, 'C':0.8, 'D':1.2}
        self.rdnoise = {'A':2.0, 'B':2.2, 'C':2.4, 'D':2.6}

        self.quad = {
            'A': np.s_[0:ny, 0:nx], 'B': np.s_[0:ny, nx:nx+nx],
            'C': np.s_[ny:ny+ny, 0:nx], 'D': np.s_[ny:ny+ny, nx:nx+nx],
        }

        for amp in ('A', 'B', 'C', 'D'):
            self.header['GAIN'+amp] = self.gain[amp]
            self.header['RDNOISE'+amp] = self.rdnoise[amp]

            # Overscan row
            xy = parse_sec_keyword(hdr['ORSEC'+amp])
            shape = [xy[0].stop-xy[0].start, xy[1].stop-xy[1].start]
            self.rawimage[xy] += self.offset_row[amp]
            self.rawimage[xy] += self.offset[amp]
            self.rawimage[xy] += np.random.normal(scale=self.rdnoise[amp], size=shape)/self.gain[amp]
            # Overscan col
            xy = parse_sec_keyword(hdr['BIASSEC'+amp])
            shape = [xy[0].stop-xy[0].start, xy[1].stop-xy[1].start]
            self.rawimage[xy] += self.offset[amp]
            self.rawimage[xy] += np.random.normal(scale=self.rdnoise[amp], size=shape)/self.gain[amp]
            # Extend into the row region
            xy_row = parse_sec_keyword(hdr['ORSEC'+amp])
            xy = (xy_row[0], xy[1])
            shape = [xy[0].stop-xy[0].start, xy[1].stop-xy[1].start]
            self.rawimage[xy] += self.offset[amp]
            self.rawimage[xy] += np.random.normal(scale=self.rdnoise[amp], size=shape)/self.gain[amp]
            # Data
            xy = parse_sec_keyword(hdr['DATASEC'+amp])
            shape = [xy[0].stop-xy[0].start, xy[1].stop-xy[1].start]
            self.rawimage[xy] += self.offset[amp]
            #self.rawimage[xy] += self.offset_row[amp]
            self.rawimage[xy] += np.random.normal(scale=self.rdnoise[amp], size=shape)/self.gain[amp]

        #- raw data are integers, not floats
        self.rawimage = self.rawimage.astype(np.int32)

        #- Confirm that all regions were correctly offset
        assert not np.any(self.rawimage == 0.0)

    def test_preproc_no_orsec(self):
        # Strip out ORSEC
        old_header = self.header.copy()
        old_image = self.rawimage.copy()
        for amp in ('A', 'B', 'C', 'D'):
            old_header.pop('ORSEC{}'.format(amp))
            xy = parse_sec_keyword(self.header['DATASEC'+amp])
            #old_image[xy] -= np.int32(self.offset_row[amp]) -- OR_SEC now zero
        #
        image = preproc(old_image, old_header, primary_header = self.primary_header)
        self.assertEqual(image.pix.shape, (2*self.ny, 2*self.nx))
        self.assertTrue(np.all(image.ivar <= 1/image.readnoise**2))
        for amp in ('A', 'B', 'C', 'D'):
            pix = image.pix[self.quad[amp]]
            rdnoise = np.median(image.readnoise[self.quad[amp]])
            npixover = self.ny * self.noverscan
            self.assertAlmostEqual(np.mean(pix), 0.0, delta=1)  # Using np.int32 pushed this to 1
            self.assertAlmostEqual(np.std(pix), self.rdnoise[amp], delta=0.2)
            self.assertAlmostEqual(rdnoise, self.rdnoise[amp], delta=0.2)

    def test_preproc(self):
        image = preproc(self.rawimage, self.header, primary_header = self.primary_header)
        self.assertEqual(image.pix.shape, (2*self.ny, 2*self.nx))
        self.assertTrue(np.all(image.ivar <= 1/image.readnoise**2))
        for amp in ('A', 'B', 'C', 'D'):
            pix = image.pix[self.quad[amp]]
            rdnoise = np.median(image.readnoise[self.quad[amp]])
            npixover = self.ny * self.noverscan
            self.assertAlmostEqual(np.mean(pix), 0.0, delta=5*rdnoise/np.sqrt(npixover)) # JXP increased this
            self.assertAlmostEqual(np.std(pix), self.rdnoise[amp], delta=0.2)
            self.assertAlmostEqual(rdnoise, self.rdnoise[amp], delta=0.2)

    def test_preproc1234(self):
        """Should also work with old amp names 1-4 instead of A-D"""
        hdr = self.header.copy()
        for prefix in ('ORSEC', 'BIASSEC', 'DATASEC', 'CCDSEC'):
            for amp, ampnum in zip(('A','B','C','D'), ('1','2','3','4')):
                if prefix+amp in hdr:
                    hdr[prefix+ampnum] = hdr[prefix+amp]
                    del hdr[prefix+amp]

        image = preproc(self.rawimage, hdr, primary_header=self.primary_header)

    def test_amp_ids(self):
        """Test auto-detection of amp names"""
        hdr = dict(
            BIASSECA=self.header['BIASSECA'],
            BIASSECB=self.header['BIASSECB'],
            BIASSECC=self.header['BIASSECC'],
            BIASSECD=self.header['BIASSECD'],
            )
        self.assertEqual(get_amp_ids(hdr), ['A', 'B', 'C', 'D'])

        hdr = dict(
            BIASSEC1=self.header['BIASSECA'],
            BIASSEC2=self.header['BIASSECB'],
            BIASSEC3=self.header['BIASSECC'],
            BIASSEC4=self.header['BIASSECD'],
            )
        self.assertEqual(get_amp_ids(hdr), ['1', '2', '3', '4'])

        hdr = dict()
        with self.assertRaises(KeyError):
            get_amp_ids(hdr)

    def test_readout_mode(self):
        """Test 2-amp vs. 4-amp detection"""
        hdr = dict(
            BIASSECA=self.header['BIASSECA'],
            BIASSECB=self.header['BIASSECB'],
            BIASSECC=self.header['BIASSECC'],
            BIASSECD=self.header['BIASSECD'],
            )
        self.assertEqual(get_readout_mode(hdr), "4Amp")

        hdr = dict(
            BIASSECA=self.header['BIASSECA'],
            BIASSECB=self.header['BIASSECB'],
            )
        self.assertEqual(get_readout_mode(hdr), "2AmpLeftRight")

        hdr = dict(
            BIASSECC=self.header['BIASSECC'],
            BIASSECD=self.header['BIASSECD'],
            )
        self.assertEqual(get_readout_mode(hdr), "2AmpLeftRight")

        hdr = dict(
            BIASSECA=self.header['BIASSECA'],
            BIASSECC=self.header['BIASSECC'],
            )
        self.assertEqual(get_readout_mode(hdr), "2AmpUpDown")

        hdr = dict(
            BIASSECB=self.header['BIASSECB'],
            BIASSECD=self.header['BIASSECD'],
            )
        self.assertEqual(get_readout_mode(hdr), "2AmpUpDown")

        #- (very) old numbers instead of letters for amps
        hdr = dict(
            BIASSEC1=self.header['BIASSECA'],
            BIASSEC2=self.header['BIASSECB'],
            )
        self.assertEqual(get_readout_mode(hdr), "2AmpLeftRight")

        hdr = dict(
            BIASSEC3=self.header['BIASSECC'],
            BIASSEC4=self.header['BIASSECD'],
            )
        self.assertEqual(get_readout_mode(hdr), "2AmpLeftRight")

        hdr = dict(
            BIASSEC1=self.header['BIASSECA'],
            BIASSEC3=self.header['BIASSECC'],
            )
        self.assertEqual(get_readout_mode(hdr), "2AmpUpDown")

        hdr = dict(
            BIASSEC2=self.header['BIASSECB'],
            BIASSEC4=self.header['BIASSECD'],
            )
        self.assertEqual(get_readout_mode(hdr), "2AmpUpDown")

        #- bad combos raise exception
        with self.assertRaises(ValueError):
            hdr = dict(
                BIASSECA=self.header['BIASSECA'],
                BIASSECD=self.header['BIASSECD'],
                )
            get_readout_mode(hdr)

    def test_bias(self):
        image = preproc(self.rawimage, self.header, primary_header = self.primary_header, bias=False)
        bias = np.zeros(self.rawimage.shape)
        image = preproc(self.rawimage, self.header, primary_header = self.primary_header, bias=bias)
        fits.writeto(self.calibfile, bias)
        image = preproc(self.rawimage, self.header, primary_header = self.primary_header, bias=self.calibfile)
        with self.assertRaises(ValueError):
            image = preproc(self.rawimage, self.header, primary_header = self.primary_header, bias=bias[0:10, 0:10])

    def test_pixflat(self):
        image = preproc(self.rawimage, self.header, primary_header = self.primary_header, pixflat=False)
        pixflat = np.ones_like(image.pix)
        image = preproc(self.rawimage, self.header, primary_header = self.primary_header, pixflat=pixflat)
        fits.writeto(self.calibfile, pixflat)
        image = preproc(self.rawimage, self.header, primary_header = self.primary_header, pixflat=self.calibfile)
        with self.assertRaises(ValueError):
            image = preproc(self.rawimage, self.header, primary_header = self.primary_header, pixflat=pixflat[0:10, 0:10])

    def test_mask(self):
        image = preproc(self.rawimage, self.header, primary_header = self.primary_header, mask=False)
        mask = np.random.randint(0, 2, size=image.pix.shape)
        image = preproc(self.rawimage, self.header, primary_header = self.primary_header, mask=mask)
        self.assertTrue(np.all(image.mask == mask))
        fits.writeto(self.calibfile, mask)
        image = preproc(self.rawimage, self.header, primary_header = self.primary_header, mask=self.calibfile)
        self.assertTrue(np.all(image.mask == mask))
        with self.assertRaises(ValueError):
            image = preproc(self.rawimage, self.header, primary_header = self.primary_header, mask=mask[0:10, 0:10])

    def test_pixflat_mask(self):
        from desispec.maskbits import ccdmask
        pixflat = np.ones((2*self.ny, 2*self.nx))
        pixflat[0:10, 0:10] = 0.0
        pixflat[10:20, 10:20] = 0.05
        image = preproc(self.rawimage, self.header, primary_header = self.primary_header, pixflat=pixflat)
        self.assertTrue(np.all(image.mask[0:10,0:10] & ccdmask.PIXFLATZERO))
        self.assertTrue(np.all(image.mask[10:20,10:20] & ccdmask.PIXFLATLOW))

    def test_io(self):
        io.write_raw(self.rawfile, self.rawimage, self.header, primary_header = self.primary_header, camera='b0')
        io.write_raw(self.rawfile, self.rawimage, self.header, primary_header = self.primary_header, camera='R1')
        io.write_raw(self.rawfile, self.rawimage, self.header, primary_header = self.primary_header, camera='z9')
        self.header['CAMERA'] = 'B3'
        io.write_raw(self.rawfile, self.rawimage, self.header, primary_header = self.primary_header)

        b0 = io.read_raw(self.rawfile, 'b0')
        #b1 = io.read_raw(self.rawfile, 'b1')
        #r1 = io.read_raw(self.rawfile, 'r1')
        #z9 = io.read_raw(self.rawfile, 'Z9')

        self.assertEqual(b0.meta['CAMERA'], 'b0')
        #self.assertEqual(b1.meta['CAMERA'], 'b1')
        #self.assertEqual(r1.meta['CAMERA'], 'r1')
        #self.assertEqual(z9.meta['CAMERA'], 'z9')

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
        io.write_raw(self.rawfile, data64, self.header, primary_header = self.primary_header, camera='b0')
        #- Should be written as vanilla ImageHDU
        io.write_raw(self.rawfile, data64, self.header, primary_header = self.primary_header, camera='b1')
        #- Should be converted to 32-bit CompImageHDU
        io.write_raw(self.rawfile, datasmall64, self.header, primary_header = self.primary_header, camera='b2')
        #- Should be 32-bit CompImageHDU
        io.write_raw(self.rawfile, data32, self.header, primary_header = self.primary_header, camera='b3')
        #- Should be 16-bit CompImageHDU
        io.write_raw(self.rawfile, data16, self.header, primary_header = self.primary_header, camera='b4')

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


        # not a very useful test :
        # it is tested by the other tests
        #def test_keywords(self):
        #for keyword in self.header:
        #- Missing GAIN* and RDNOISE* are warnings but not errors
        #    if keyword.startswith('GAIN') or keyword.startswith('RDNOISE'):
        #        continue

        #- DATE-OBS, NIGHT, and EXPID are also optional
        #- (but maybe they should be required...)
        #   if keyword in ('DATE-OBS', 'NIGHT', 'EXPID'):
        #       continue

        #  if os.path.exists(self.rawfile):
        #      os.remove(self.rawfile)
        #      value = self.header[keyword]

        #       del self.header[keyword]

        #       with self.assertRaises(KeyError):
        #           io.write_raw(self.rawfile, self.rawimage, self.header, primary_header = self.primary_header)

        #      self.header[keyword] = value

        #dateobs = self.header

    #- striving for 100% coverage...
    def test_pedantic(self):
        with self.assertRaises(ValueError):
            parse_sec_keyword('blat')
        #- should log a warning about large readnoise
        rawimage = self.rawimage + np.random.normal(scale=2, size=self.rawimage.shape)
        image = preproc(rawimage, self.header, primary_header = self.primary_header)
        #- should log an error about huge readnoise
        rawimage = self.rawimage + np.random.normal(scale=10, size=self.rawimage.shape)
        image = preproc(rawimage, self.header, primary_header = self.primary_header)
        #- should log a warning about small readnoise
        rdnoise = 0.7 * np.mean(list(self.rdnoise.values()))
        rawimage = np.random.normal(scale=rdnoise, size=self.rawimage.shape)
        image = preproc(rawimage, self.header, primary_header = self.primary_header)
        #- should log a warning about tiny readnoise
        rdnoise = 0.01 * np.mean(list(self.rdnoise.values()))
        rawimage = np.random.normal(scale=rdnoise, size=self.rawimage.shape)
        image = preproc(rawimage, self.header, primary_header = self.primary_header)
        #- Missing expected RDNOISE keywords shouldn't be fatal
        hdr = self.header.copy()
        del hdr['RDNOISEA']
        del hdr['RDNOISEB']
        del hdr['RDNOISEC']
        del hdr['RDNOISED']
        image = preproc(self.rawimage, hdr, primary_header = self.primary_header)
        #- Missing expected GAIN keywords should log error but not crash
        hdr = self.header.copy()
        del hdr['GAINA']
        del hdr['GAINB']
        del hdr['GAINC']
        del hdr['GAIND']
        image = preproc(self.rawimage, hdr, primary_header = self.primary_header)

    def test_preproc_script(self):
        io.write_raw(self.rawfile, self.rawimage, self.header, primary_header = self.primary_header, camera='b0')
        io.write_raw(self.rawfile, self.rawimage, self.header, primary_header = self.primary_header, camera='b1')
        args = ['--infile', self.rawfile, '--cameras', 'b0',
                '--outfile', self.pixfile]
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
    def test_default_bias(self):
        image = preproc(self.rawimage, self.header, primary_header = self.primary_header, bias=True)

    def test_default_pixflat(self):
        image = preproc(self.rawimage, self.header, primary_header = self.primary_header, pixflat=True)

    def test_default_mask(self):
        image = preproc(self.rawimage, self.header, primary_header = self.primary_header, mask=True)


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)

if __name__ == '__main__':
    unittest.main()
