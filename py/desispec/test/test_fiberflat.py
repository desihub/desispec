"""
test desispec.fiberflat
"""

from __future__ import division

import unittest
import copy
import os
import tempfile
import shutil
from uuid import uuid1

import numpy as np
import scipy.sparse
from astropy.table import Table

from desispec.maskbits import specmask
from desispec.resolution import Resolution
from desispec.frame import Frame
from desispec.fiberflat import FiberFlat
from desispec.fiberflat import compute_fiberflat, apply_fiberflat
from desispec.fiberflat import average_fiberflat, autocalib_fiberflat, gradient_correction
from desiutil.log import get_logger
from desispec.io import write_frame
import desispec.io as io

from desispec.scripts import fiberflat as ffscript


def _get_data():
    """
    Return basic test data:
      - 1D wave[nwave]
      - 2D flux[nspec, nwave]
      - 2D ivar[nspec, nwave]
    """
    nspec = 10
    nwave = 100
    wave = np.linspace(0, np.pi, nwave)
    y = np.sin(wave)
    flux = np.tile(y, nspec).reshape(nspec, nwave)
    ivar = np.ones(flux.shape)
    mask = np.zeros(flux.shape, dtype=int)

    return wave, flux, ivar, mask

def _get_fibermap(petal, nspec):
    fibermap = Table()
    fibermap['FIBER'] = petal*500 + np.arange(nspec)
    alpha = 2*np.pi/10  # angle of one petal in radians
    theta = np.random.uniform(petal*alpha, (petal+1)*alpha, size=nspec)
    r = np.random.uniform(0,400, size=nspec)
    fibermap['FIBERASSIGN_X'] = r*np.cos(theta)
    fibermap['FIBERASSIGN_Y'] = r*np.sin(theta)
    fibermap['FIBERSTATUS'] = 0

    return fibermap


class TestFiberFlat(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.origdir = os.getcwd()
        cls.testdir = tempfile.mkdtemp()
        os.chdir(cls.testdir)

    def setUp(self):
        os.chdir(self.testdir)
        id = uuid1()
        self.testfibermap = 'test_fibermap_{}.fits'.format(id)
        self.testframe = 'test_frame_{}.fits'.format(id)
        self.testflat = 'test_fiberflat_{}.fits'.format(id)

    def tearDown(self):
        os.chdir(self.testdir)
        if os.path.isfile(self.testframe):
            os.unlink(self.testframe)
        if os.path.isfile(self.testflat):
            os.unlink(self.testflat)
        if os.path.isfile(self.testfibermap):
            os.unlink(self.testfibermap)

    @classmethod
    def tearDownClass(cls):
        #- Remove testdir only if it was created by tempfile.mkdtemp
        if cls.testdir.startswith(tempfile.gettempdir()) and os.path.exists(cls.testdir):
            shutil.rmtree(cls.testdir)

        os.chdir(cls.origdir)


    def test_interface(self):
        """
        Basic test that interface works and identical inputs result in
        identical outputs
        """
        wave, flux, ivar, mask = _get_data()
        nspec, nwave = flux.shape

        #- Setup data for a Resolution matrix
        sigma = 4.0
        ndiag = 11
        xx = np.linspace(-(ndiag-1)/2.0, +(ndiag-1)/2.0, ndiag)
        Rdata = np.zeros( (nspec, ndiag, nwave) )
        kernel = np.exp(-xx**2/(2*sigma))
        kernel /= sum(kernel)
        for i in range(nspec):
            for j in range(nwave):
                Rdata[i,:,j] = kernel

        #- Run the code
        frame = Frame(wave, flux, ivar, mask, Rdata, spectrograph=0, meta=dict(CAMERA='x0'))
        ff = compute_fiberflat(frame)

        #- Check shape of outputs
        self.assertEqual(ff.fiberflat.shape, flux.shape)
        self.assertEqual(ff.ivar.shape, flux.shape)
        self.assertEqual(ff.mask.shape, flux.shape)
        self.assertEqual(len(ff.meanspec), nwave)

        #- Identical inputs should result in identical ouputs
        for i in range(1, nspec):
            self.assertTrue(np.all(ff.fiberflat[i] == ff.fiberflat[0]))
            self.assertTrue(np.all(ff.ivar[i] == ff.ivar[0]))

    def test_resolution(self):
        """
        Test that identical spectra convolved with different resolutions
        results in identical fiberflats
        """
        wave, flux, ivar, mask = _get_data()
        nspec, nwave = flux.shape

        #- Setup a Resolution matrix that varies with fiber and wavelength
        #- Note: this is actually the transpose of the resolution matrix
        #- I wish I was creating, but as long as we self-consistently
        #- use it for convolving and solving, that shouldn't matter.
        sigma = np.linspace(2, 10, nwave*nspec)
        ndiag = 21
        xx = np.linspace(-ndiag/2.0, +ndiag/2.0, ndiag)
        Rdata = np.zeros( (nspec, len(xx), nwave) )
        for i in range(nspec):
            for j in range(nwave):
                kernel = np.exp(-xx**2/(2*sigma[i*nwave+j]**2))
                kernel /= sum(kernel)
                Rdata[i,:,j] = kernel

        #- Convolve the data with the resolution matrix
        convflux = np.empty_like(flux)
        for i in range(nspec):
            convflux[i] = Resolution(Rdata[i]).dot(flux[i])

        #- Run the code
        frame = Frame(wave, convflux, ivar, mask, Rdata, spectrograph=0, meta=dict(CAMERA='x0'))
        ff = compute_fiberflat(frame)

        #- These fiber flats should all be ~1
        self.assertTrue( np.all(np.abs(ff.fiberflat-1) < 0.001) )

    def test_throughput(self):
        """
        Test that spectra with different throughputs but the same resolution
        produce a fiberflat mirroring the variations in throughput
        """
        wave, flux, ivar, mask = _get_data()
        nspec, nwave = flux.shape

        #- Setup data for a Resolution matrix
        sigma = 4.0
        ndiag = 21
        xx = np.linspace(-(ndiag-1)/2.0, +(ndiag-1)/2.0, ndiag)
        Rdata = np.zeros( (nspec, ndiag, nwave) )
        kernel = np.exp(-xx**2/(2*sigma))
        kernel /= sum(kernel)
        for i in range(nspec):
            for j in range(nwave):
                Rdata[i,:,j] = kernel

        #- Vary the input flux prior to calculating the fiber flat
        flux[1] *= 1.1
        flux[2] *= 1.2
        flux[3] *= 0.8

        #- Convolve with the (common) resolution matrix
        convflux = np.empty_like(flux)
        for i in range(nspec):
            convflux[i] = Resolution(Rdata[i]).dot(flux[i])

        frame = Frame(wave, convflux, ivar, mask, Rdata, spectrograph=0, meta=dict(CAMERA='x0'))
        ff = compute_fiberflat(frame)

        #- flux[1] is brighter, so should fiberflat[1].  etc.
        self.assertTrue(np.allclose(ff.fiberflat[0], ff.fiberflat[1]/1.1))
        self.assertTrue(np.allclose(ff.fiberflat[0], ff.fiberflat[2]/1.2))
        self.assertTrue(np.allclose(ff.fiberflat[0], ff.fiberflat[3]/0.8))

    def test_throughput_resolution(self):
        """
        Test that spectra with different throughputs and different resolutions
        result in fiberflat variations that are only due to throughput.
        """
        wave, flux, ivar, mask = _get_data()
        nspec, nwave = flux.shape

        #- Setup a Resolution matrix that varies with fiber and wavelength
        #- Note: this is actually the transpose of the resolution matrix
        #- I wish I was creating, but as long as we self-consistently
        #- use it for convolving and solving, that shouldn't matter.
        sigma = np.linspace(2, 10, nwave*nspec)
        ndiag = 21
        xx = np.linspace(-ndiag/2.0, +ndiag/2.0, ndiag)
        Rdata = np.zeros( (nspec, len(xx), nwave) )
        for i in range(nspec):
            for j in range(nwave):
                kernel = np.exp(-xx**2/(2*sigma[i*nwave+j]**2))
                kernel /= sum(kernel)
                Rdata[i,:,j] = kernel

        #- Vary the input flux prior to calculating the fiber flat
        flux[1] *= 1.1
        flux[2] *= 1.2
        flux[3] /= 1.1
        flux[4] /= 1.2

        #- Convolve the data with the varying resolution matrix
        convflux = np.empty_like(flux)
        for i in range(nspec):
            convflux[i] = Resolution(Rdata[i]).dot(flux[i])

        #- Run the code
        frame = Frame(wave, convflux, ivar, mask, Rdata, spectrograph=0, meta=dict(CAMERA='x0'))
        #- Set an accuracy for this
        accuracy=1.e-9
        ff = compute_fiberflat(frame,accuracy=accuracy)

        #- Compare variation with middle fiber
        mid = ff.fiberflat.shape[0] // 2

        diff = (ff.fiberflat[1]/1.1 - ff.fiberflat[mid])
        self.assertLess(np.max(np.abs(diff)), accuracy)

        diff = (ff.fiberflat[2]/1.2 - ff.fiberflat[mid])
        self.assertLess(np.max(np.abs(diff)), accuracy)

        diff = (ff.fiberflat[3]*1.1 - ff.fiberflat[mid])
        self.assertLess(np.max(np.abs(diff)), accuracy)

        diff = (ff.fiberflat[4]*1.2 - ff.fiberflat[mid])
        self.assertLess(np.max(np.abs(diff)), accuracy)

        #- Add outliers and ensure that it doesn't significantly change
        frame.flux[0][0] = 1000
        frame.flux[1][10] = 2000
        frame.flux[2][20] = 3000
        ff2 = compute_fiberflat(frame,accuracy=accuracy)
        self.assertTrue(np.allclose(ff.fiberflat, ff2.fiberflat))


    def test_apply_fiberflat(self):
        '''test apply_fiberflat interface and changes to flux and mask'''
        wave = np.arange(5000, 5050)
        nwave = len(wave)
        nspec = 3
        flux = np.random.uniform(size=(nspec, nwave))
        ivar = np.ones_like(flux)
        frame = Frame(wave, flux, ivar, spectrograph=0, meta=dict(CAMERA='x0'))
        frame.meta['HELIOCOR'] = 1.0  #- breaks test due to interpolation over bad ff

        fiberflat = np.ones_like(flux)
        ffivar = 2*np.ones_like(flux)
        ffmask = np.zeros_like(flux, dtype=np.uint32)
        fiberflat[0] *= 0.8
        fiberflat[1] *= 1.2
        fiberflat[2, 0:10] = 0  #- bad fiberflat
        ffivar[2, 10:20] = 0    #- bad fiberflat
        ffmask[2, 20:30] = 1    #- bad fiberflat

        ff = FiberFlat(wave, fiberflat, ffivar, mask=ffmask)

        self.assertTrue(np.all(ff.fiberflat == fiberflat))
        self.assertTrue(np.all(ff.ivar == ffivar))
        self.assertTrue(np.all(ff.mask == ffmask))

        #- Test applying fiberflat
        origframe = copy.deepcopy(frame)
        apply_fiberflat(frame, ff)

        #- did mask get set for bad fiberflat?
        ii = (ff.fiberflat == 0)
        self.assertTrue(np.all((frame.mask[ii] & specmask.BADFIBERFLAT) != 0))
        ii = (ff.ivar == 0)
        self.assertTrue(np.all((frame.mask[ii] & specmask.BADFIBERFLAT) != 0))
        ii = (ff.mask != 0)
        self.assertTrue(np.all((frame.mask[ii] & specmask.BADFIBERFLAT) != 0))

        #- was fiberflat applied for non-masked pixels?
        ok = (frame.mask & specmask.BADFIBERFLAT) == 0
        self.assertTrue(np.all(frame.flux[0][ok[0]] == origframe.flux[0][ok[0]]/0.8))
        self.assertTrue(np.all(frame.flux[1][ok[1]] == origframe.flux[1][ok[1]]/1.2))
        self.assertTrue(np.all(frame.flux[2][ok[2]] == origframe.flux[2][ok[2]]))

        #- Should fail if frame and ff don't have a common wavelength grid
        frame.wave = frame.wave + 0.1
        with self.assertRaises(ValueError):
            apply_fiberflat(frame, ff)


    def test_apply_fiberflat_ivar(self):
        '''test error propagation in apply_fiberflat'''
        wave = np.arange(5000, 5010)
        nwave = len(wave)
        nspec = 3
        flux = np.random.uniform(0.9, 1.0, size=(nspec, nwave))
        ivar = np.ones_like(flux)
        origframe = Frame(wave, flux, ivar, spectrograph=0, meta=dict(CAMERA='x0'))

        fiberflat = np.ones_like(flux)
        ffmask = np.zeros_like(flux, dtype=np.uint32)
        fiberflat[0] *= 0.5
        fiberflat[1] *= 1.5

        #- ff with essentially no error
        ffivar = 1e20 * np.ones_like(flux)
        ff = FiberFlat(wave, fiberflat, ffivar, mask=ffmask)
        frame = copy.deepcopy(origframe)
        apply_fiberflat(frame, ff)
        self.assertTrue(np.allclose(frame.ivar, fiberflat**2))

        #- ff with large error
        ffivar = np.ones_like(flux)
        ff = FiberFlat(wave, fiberflat, ffivar)
        frame = copy.deepcopy(origframe)
        apply_fiberflat(frame, ff)

        #- c = a/b
        #- (sigma_c/c)^2 = (sigma_a/a)^2 + (sigma_b/b)^2
        var = frame.flux**2 * (1.0/(origframe.ivar * origframe.flux**2) + \
                               1.0/(ff.ivar * ff.fiberflat**2))
        self.assertTrue(np.allclose(frame.ivar, 1/var))

        #- ff.ivar=0 should result in frame.ivar=0, even if ff.fiberflat=0 too
        ffivar = np.ones_like(flux)
        ffivar[0, 0:5] = 0.0
        fiberflat[0, 0:5] = 0.0
        ff = FiberFlat(wave, fiberflat, ffivar)
        frame = copy.deepcopy(origframe)
        apply_fiberflat(frame, ff)

        self.assertTrue(np.all(frame.ivar[0, 0:5] == 0.0))

    def test_main(self):
        """
        Test the main program.
        """
        # generate the frame data
        wave, flux, ivar, mask = _get_data()
        nspec, nwave = flux.shape

        #- Setup data for a Resolution matrix
        sigma = 4.0
        ndiag = 11
        xx = np.linspace(-(ndiag-1)/2.0, +(ndiag-1)/2.0, ndiag)
        Rdata = np.zeros( (nspec, ndiag, nwave) )
        kernel = np.exp(-xx**2/(2*sigma))
        kernel /= sum(kernel)
        for i in range(nspec):
            for j in range(nwave):
                Rdata[i,:,j] = kernel

        #- Convolve the data with the resolution matrix
        convflux = np.empty_like(flux)
        for i in range(nspec):
            convflux[i] = Resolution(Rdata[i]).dot(flux[i])

        # create a fake fibermap
        fibermap = io.empty_fibermap(nspec, nwave)
        for i in range(0, nspec):
            fibermap['OBJTYPE'][i] = 'BAD'
        io.write_fibermap(self.testfibermap, fibermap)

        #- write out the frame
        frame = Frame(wave, convflux, ivar, mask, Rdata, spectrograph=0, fibermap=fibermap,
                      meta=dict(FLAVOR='flat', CAMERA='x0'))
        write_frame(self.testframe, frame, fibermap=fibermap)

        # set program arguments
        argstr = [
            '--infile', self.testframe,
            '--outfile', self.testflat
        ]

        # run it
        args = ffscript.parse(options=argstr)
        ffscript.main(args)


class TestFiberFlatObject(unittest.TestCase):

    def setUp(self):
        self.nspec = 5
        self.nwave = 10
        self.wave = np.arange(self.nwave)
        self.fiberflat = np.random.uniform(size=(self.nspec, self.nwave))
        self.ivar = np.ones(self.fiberflat.shape)
        self.mask = np.zeros(self.fiberflat.shape, dtype=np.uint32)
        self.meanspec = np.random.uniform(size=self.nwave)
        self.header = dict(blat=1, foo=2)
        self.ff = FiberFlat(self.wave, self.fiberflat, self.ivar, self.mask, self.meanspec, header=self.header)

    def test_init(self):
        for key in ('wave', 'fiberflat', 'ivar', 'mask', 'meanspec'):
            x = self.ff.__getattribute__(key)
            y = self.__getattribute__(key)
            self.assertTrue(np.all(x == y), key)

        self.assertEqual(self.nspec, self.ff.nspec)
        self.assertEqual(self.nwave, self.ff.nwave)

    def test_dimensions(self):
        #- check dimensionality mismatches
        with self.assertRaises(ValueError):
            FiberFlat(self.wave, self.wave, self.ivar, self.mask, self.meanspec)

        with self.assertRaises(ValueError):
            FiberFlat(self.wave, self.wave, self.ivar, self.mask, self.meanspec)

        with self.assertRaises(ValueError):
            FiberFlat(self.wave, self.fiberflat, self.ivar, self.mask, self.fiberflat)

        with self.assertRaises(ValueError):
            FiberFlat(self.wave, self.fiberflat[0:2], self.ivar, self.mask, self.meanspec)

        with self.assertRaises(ValueError):
            FiberFlat(self.fiberflat, self.fiberflat, self.ivar, self.mask, self.meanspec)

        with self.assertRaises(ValueError):
            FiberFlat(self.wave, self.fiberflat, self.wave, self.mask, self.meanspec)

        with self.assertRaises(ValueError):
            FiberFlat(self.wave, self.fiberflat, self.ivar, self.mask[0:2, :], self.meanspec)

        fibers = np.arange(self.nspec)
        FiberFlat(self.wave, self.fiberflat, self.ivar, self.mask, self.meanspec, fibers=fibers)
        with self.assertRaises(ValueError):
            FiberFlat(self.wave, self.fiberflat, self.ivar, self.mask, self.meanspec, fibers=fibers[1:])

    def test_slice(self):
        x = self.ff[1]
        x = self.ff[1:2]
        x = self.ff[[1,2,3]]
        x = self.ff[self.ff.fibers<3]

    def test_average_fiberflat(self):
        ff = average_fiberflat([self.ff, self.ff])
        self.assertTrue(np.allclose(ff.fiberflat, self.ff.fiberflat))
        self.assertTrue(np.allclose(ff.ivar, self.ff.ivar*2))

        ff = average_fiberflat([self.ff, self.ff, self.ff, self.ff])
        self.assertTrue(np.allclose(ff.fiberflat, self.ff.fiberflat))
        self.assertTrue(np.allclose(ff.ivar, self.ff.ivar*4*2/np.pi))  # 2/pi due to median vs. mean penalty

        #- boundary cases of 1 or 0 inputs
        ff = average_fiberflat([self.ff,])
        self.assertIs(ff, self.ff)

        with self.assertRaises(ValueError):
            ff = average_fiberflat([])

    def test_autocalib_fiberflat(self):
        fiberflats = list()
        expid = 1000
        for petal in range(10):
            fibermap = _get_fibermap(petal, self.nspec)
            for i in range(3):
                ff = copy.deepcopy(self.ff)
                ff.header['EXPID'] = expid
                expid += 1
                ff.header['CAMERA'] = f'r{petal}'
                ff.fibermap = fibermap
                fiberflats.append(ff)

        ff = autocalib_fiberflat(fiberflats)

    def test_gradient_correction(self):
        ref_fiberflats = dict()
        tilted_fiberflats = dict()
        for petal in range(10):
            fibermap = _get_fibermap(petal, self.nspec)
            camera = f'r{petal}'

            ff = copy.deepcopy(self.ff)
            ff.fiberflat[:,:] = 1.0
            ff.header['CAMERA'] = f'r{petal}'
            ff.fibermap = fibermap
            ref_fiberflats[camera] = ff

            ff = copy.deepcopy(self.ff)
            ff.fiberflat[:,:] = 1.0
            ff.header['CAMERA'] = f'r{petal}'
            ff.fibermap = fibermap
            #- add +/- 5% tilt edge-to-edge
            tilt = 1 + 0.05*fibermap['FIBERASSIGN_X']/400
            print(f'DEBUG: {petal=}  {np.min(tilt)=}  {np.max(tilt)=}')
            print(f'DEBUG: {petal=}  {np.min(ff.fiberflat)=}  {np.max(ff.fiberflat)=}')
            for i in range(ff.fiberflat.shape[0]):
                ff.fiberflat[i] *= tilt[i]

            tilted_fiberflats[camera] = ff

        try:
            final_fiberflats = gradient_correction(tilted_fiberflats, ref_fiberflats)
        except Exception as err:
            for petal in range(10):
                camera = f'r{petal}'
                ffratio = tilted_fiberflats[camera].fiberflat / ref_fiberflats[camera].fiberflat
                print(f'DEBUG: {camera=} {ffratio=}')

            raise err

        for cam in final_fiberflats:
            self.assertTrue(np.allclose(final_fiberflats[cam].fiberflat, ref_fiberflats[cam].fiberflat))
