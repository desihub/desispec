"""
tests desispec.sky
"""

import unittest

import numpy as np
from desispec.sky import compute_sky, subtract_sky
from desispec.resolution import Resolution
from desispec.frame import Frame
import desispec.io

import desispec.scripts.sky as skyscript


class TestSky(unittest.TestCase):
    
    #- Create unique test filename in a subdirectory
    def setUp(self):
        #- Create a fake sky
        self.nspec = 40
        self.wave = np.arange(4000, 4500)
        self.nwave = len(self.wave)
        self.flux = np.zeros(self.nwave)
        for i in range(0, self.nwave, 20):
            self.flux[i] = i
            
        self.ivar = np.ones(self.flux.shape)
                    
    def _get_spectra(self,with_gradient=False):
        #- Setup data for a Resolution matrix
        sigma = 4.0
        ndiag = 21
        xx = np.linspace(-(ndiag-1)/2.0, +(ndiag-1)/2.0, ndiag)
        Rdata = np.zeros( (self.nspec, ndiag, self.nwave) )
        for i in range(self.nspec):
            for j in range(self.nwave):
                kernel = np.exp(-xx**2/(2*sigma))
                kernel /= sum(kernel)
                Rdata[i,:,j] = kernel
                
        flux = np.zeros((self.nspec, self.nwave),dtype=float)
        ivar = np.ones((self.nspec, self.nwave),dtype=float)
        mask = np.zeros((self.nspec, self.nwave), dtype=int)
        
        fibermap = desispec.io.empty_fibermap(self.nspec, 1500)
        fibermap['OBJTYPE'][0::2] = 'SKY'
        x=fibermap["X_TARGET"]
        y=fibermap["Y_TARGET"]
        x = x-np.mean(x)
        y = y-np.mean(y)
        if np.std(x)>0 : x /= np.std(x)
        if np.std(y)>0 : y /= np.std(y)
        
        for i in range(self.nspec):
            R = Resolution(Rdata[i])
            if with_gradient :
                scale = 1.+0.1*x[i]+0.2*y[i]
                flux[i] = scale*R.dot(self.flux)
                
            else :
                flux[i] = R.dot(self.flux)

        
        

        return Frame(self.wave, flux, ivar, mask, Rdata, spectrograph=2, fibermap=fibermap)
                    
    def test_uniform_resolution(self):        
        #- Setup data for a Resolution matrix
        spectra = self._get_spectra()
                        
        sky = compute_sky(spectra,add_variance=True)
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
        sky = compute_sky(spectra,add_variance=True)
        subtract_sky(spectra, sky)
        #- allow some slop in the sky subtraction
        self.assertTrue(np.allclose(spectra.flux, 0, rtol=1e-5, atol=1e-6))

    def test_subtract_sky_with_gradient(self):
        spectra = self._get_spectra(with_gradient=True)
        sky = compute_sky(spectra,fp_corr_deg=1,add_variance=True)
        #import astropy.io.fits as pyfits
        #h=pyfits.HDUList([pyfits.PrimaryHDU(spectra.flux),pyfits.ImageHDU(sky.flux)])
        #h.writeto("toto.fits",overwrite=True)
        subtract_sky(spectra, sky)
        
        #- allow some slop in the sky subtraction
        self.assertTrue(np.allclose(spectra.flux, 0, rtol=1e-4, atol=1e-4))

    def test_main(self):
        pass
        
        
    def runTest(self):
        pass
                
def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)

if __name__ == '__main__':
    unittest.main()
