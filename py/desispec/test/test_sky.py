"""
tests desispec.sky
"""

import unittest

import numpy as np
from astropy.table import Table
from desispec.sky import compute_sky, subtract_sky, SkyModel, get_sky_fibers
from desispec.resolution import Resolution
from desispec.frame import Frame
from desispec.maskbits import fibermask
import desispec.io

import desispec.scripts.sky as skyscript


class TestSky(unittest.TestCase):

    #- Create unique test filename in a subdirectory
    def setUp(self):
        #- Create a fake sky
        self.nspec = 20
        self.wave = np.arange(4000, 4502)
        self.nwave = len(self.wave)
        self.flux = np.zeros(self.nwave)
        for i in range(0, self.nwave, 20):
            self.flux[i] = i
        self.ivar = np.ones(self.flux.shape)
        self.mask = np.zeros(self.flux.shape, dtype=np.int32)
        self.tpcorr = np.linspace(0.9, 1.1, self.nspec)
        self.add_variance = True

    def _get_spectra(self,with_gradient=False):
        #- Setup data for a Resolution matrix
        sigma2 = 4.0
        ndiag = 21
        xx = np.linspace(-(ndiag-1)/2.0, +(ndiag-1)/2.0, ndiag)
        Rdata = np.zeros( (self.nspec, ndiag, self.nwave) )
        for i in range(self.nspec):
            kernel = np.exp(-(xx+float(i)/self.nspec*0.3)**2/(2*sigma2))
            #kernel = np.exp(-xx**2/(2*sigma2))
            kernel /= sum(kernel)
            for j in range(self.nwave):
                Rdata[i,:,j] = kernel

        flux = np.zeros((self.nspec, self.nwave),dtype=float)
        ivar = np.ones((self.nspec, self.nwave),dtype=float)
        # Add a random component
        for i in range(self.nspec) :
            ivar[i] += 0.4*np.random.uniform(size=self.nwave)

        mask = np.zeros((self.nspec, self.nwave), dtype=int)

        fibermap = desispec.io.empty_fibermap(self.nspec, 1500)
        fibermap['OBJTYPE'][0::2] = 'SKY'

        x=fibermap["FIBERASSIGN_X"]
        y=fibermap["FIBERASSIGN_Y"]
        x = x-np.mean(x)
        y = y-np.mean(y)
        if np.std(x)>0 : x /= np.std(x)
        if np.std(y)>0 : y /= np.std(y)
        w = (self.wave-self.wave[0])/(self.wave[-1]-self.wave[0])*2.-1
        for i in range(self.nspec):
            R = Resolution(Rdata[i])

            if with_gradient :
                scale = 1.+(0.1*x[i]+0.2*y[i])*(1+0.4*w)
                flux[i] = R.dot(scale*self.flux)
            else :
                flux[i] = R.dot(self.flux)

        return Frame(self.wave, flux, ivar, mask, Rdata, spectrograph=2, fibermap=fibermap)

    def test_uniform_resolution(self):
        #- Setup data for a Resolution matrix
        spectra = self._get_spectra()

        sky = compute_sky(spectra,add_variance=self.add_variance)
        self.assertEqual(sky.flux.shape, spectra.flux.shape)
        self.assertEqual(sky.ivar.shape, spectra.ivar.shape)
        self.assertEqual(sky.mask.shape, spectra.mask.shape)

        delta=spectra.flux[0]-sky.flux[0]
        d=np.inner(delta,delta)
        self.assertAlmostEqual(d,0.)

        delta=spectra.flux[-1]-sky.flux[-1]
        d=np.inner(delta,delta)
        self.assertAlmostEqual(d,0.)

    def test_subtract_sky(self):
        spectra = self._get_spectra()
        sky = compute_sky(spectra,add_variance=self.add_variance)
        subtract_sky(spectra, sky)
        #- allow some slop in the sky subtraction
        self.assertTrue(np.allclose(spectra.flux, 0, rtol=1e-5, atol=1e-6))

    def test_sky_slice(self):
        flux = np.tile(self.flux, self.nspec).reshape(self.nspec, self.nwave)
        ivar = np.tile(self.ivar, self.nspec).reshape(self.nspec, self.nwave)
        mask = np.tile(self.mask, self.nspec).reshape(self.nspec, self.nwave)
        sky1 = SkyModel(self.wave, flux, ivar, mask, nrej=10,
                stat_ivar=ivar*100, throughput_corrections=self.tpcorr)

        sky2 = sky1[2]
        self.assertEqual(sky2.flux.shape, (1,self.nwave))
        self.assertEqual(sky2.ivar.shape, (1,self.nwave))
        self.assertEqual(sky2.mask.shape, (1,self.nwave))
        self.assertEqual(sky2.stat_ivar.shape, (1,self.nwave))
        self.assertEqual(sky2.throughput_corrections.shape, (1,))
        self.assertEqual(sky2.nrej, sky1.nrej)

        sky2 = sky1[2:3]
        self.assertEqual(sky2.flux.shape, (1,self.nwave))
        self.assertEqual(sky2.ivar.shape, (1,self.nwave))
        self.assertEqual(sky2.mask.shape, (1,self.nwave))
        self.assertEqual(sky2.stat_ivar.shape, (1,self.nwave))
        self.assertEqual(sky2.throughput_corrections.shape, (1,))
        self.assertEqual(sky2.nrej, sky1.nrej)

        sky2 = sky1[2:4]
        self.assertEqual(sky2.flux.shape, (2,self.nwave))
        self.assertEqual(sky2.ivar.shape, (2,self.nwave))
        self.assertEqual(sky2.mask.shape, (2,self.nwave))
        self.assertEqual(sky2.stat_ivar.shape, (2,self.nwave))
        self.assertEqual(sky2.throughput_corrections.shape, (2,))
        self.assertEqual(sky2.nrej, sky1.nrej)

        sky2 = sky1[[1,2,3]]
        self.assertEqual(sky2.flux.shape, (3,self.nwave))
        self.assertEqual(sky2.ivar.shape, (3,self.nwave))
        self.assertEqual(sky2.mask.shape, (3,self.nwave))
        self.assertEqual(sky2.stat_ivar.shape, (3,self.nwave))
        self.assertEqual(sky2.throughput_corrections.shape, (3,))
        self.assertEqual(sky2.nrej, sky1.nrej)

        ii = np.arange(sky1.nspec, dtype=int)%2 == 0
        n = np.sum(ii)
        sky2 = sky1[ii]
        self.assertEqual(sky2.flux.shape, (n,self.nwave))
        self.assertEqual(sky2.ivar.shape, (n,self.nwave))
        self.assertEqual(sky2.mask.shape, (n,self.nwave))
        self.assertEqual(sky2.stat_ivar.shape, (n,self.nwave))
        self.assertEqual(sky2.throughput_corrections.shape, (n,))
        self.assertEqual(sky2.nrej, sky1.nrej)

        #- stat_ivar and throughput_corrections are optional, but shouldn't break slicing
        sky1.stat_ivar = None
        sky1.throughput_corrections = None
        sky2 = sky1[2:4]
        self.assertEqual(sky2.flux.shape, (2,self.nwave))
        self.assertEqual(sky2.ivar.shape, (2,self.nwave))
        self.assertEqual(sky2.mask.shape, (2,self.nwave))
        self.assertEqual(sky2.stat_ivar, None)
        self.assertEqual(sky2.throughput_corrections, None)
        self.assertEqual(sky2.nrej, sky1.nrej)

    def test_get_sky_fibers(self):
        """Test desispec.sky.get_sky_fibers"""
        fibermap = Table()
        fibermap['TARGETID'] = np.arange(5)
        fibermap['OBJTYPE'] = ['TGT', 'SKY', 'TGT', 'SKY', 'TGT']
        fibermap['FIBERSTATUS'] = 0

        #- Test default behavior
        skyfibers = list(get_sky_fibers(fibermap))
        self.assertEqual(skyfibers, [1,3])

        #- Test overrides
        skyfibers = list(get_sky_fibers(fibermap, override_sky_targetids=[0,1]))
        self.assertEqual(skyfibers, [0,1])

        skyfibers = list(get_sky_fibers(fibermap, exclude_sky_targetids=[0,1]))
        self.assertEqual(skyfibers, [3,])

        #- SKY fibers with VARIABLETHRU should be excluded
        fibermap['FIBERSTATUS'][3] = fibermask.VARIABLETHRU
        skyfibers = list(get_sky_fibers(fibermap))
        self.assertEqual(skyfibers, [1,])

    def test_main(self):
        pass

    def runTest(self):
        pass
