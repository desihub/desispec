"""
tests desispec.fibercrosstalk
"""

import unittest

import numpy as np
from pkg_resources import resource_filename

from desispec.fibercrosstalk import correct_fiber_crosstalk
from desispec.frame import Frame
from desispec.io import empty_fibermap,read_xytraceset
from desispec.resolution import Resolution

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
        self.psffile = resource_filename('specter', 'test/t/psf-monospot.fits')

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

        fibermap = empty_fibermap(self.nspec, 1500)
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
        meta={"camera":"r2"}
        return Frame(self.wave, flux, ivar, mask, Rdata, spectrograph=2, fibermap=fibermap,meta=meta)

    def test_fiber_crosstalk(self):
        #- Just a functional test
        spectra = self._get_spectra()

        xyset = read_xytraceset(self.psffile)
        correct_fiber_crosstalk(spectra,xyset=xyset)

if __name__ == '__main__':
    unittest.main()
