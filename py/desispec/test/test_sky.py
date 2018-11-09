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
        self.nspec = 20
        self.wave = np.arange(4000, 4502)
        self.nwave = len(self.wave)
        self.flux = np.zeros(self.nwave)
        for i in range(0, self.nwave, 20):
            self.flux[i] = i
        self.ivar = np.ones(self.flux.shape)
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
        
        x=fibermap["DESIGN_X"]
        y=fibermap["DESIGN_Y"]
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

    def test_subtract_sky_with_gradient_using_compute_polynomial_times_sky(self):
        spectra = self._get_spectra(with_gradient=True)
        sky = compute_sky(spectra,angular_variation_deg=1,chromatic_variation_deg=1,add_variance=self.add_variance)
        subtract_sky(spectra, sky)
        
        #- allow some slop in the sky subtraction
        self.assertTrue(np.allclose(spectra.flux, 0, rtol=1e-3, atol=1.)) # this is not exact, it's an iterative fit
        
    
    def test_subtract_sky_with_gradient_using_compute_non_uniform_sky(self):
        spectra = self._get_spectra(with_gradient=True)
        sky = compute_sky(spectra,angular_variation_deg=1,chromatic_variation_deg=-1,add_variance=self.add_variance)
        subtract_sky(spectra, sky)
        
        #- allow some slop in the sky subtraction
        self.assertTrue(np.allclose(spectra.flux, 0, rtol=1e-3, atol=1e-3))
    
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
