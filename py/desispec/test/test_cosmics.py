"""
test desispec.cosmics
"""

import unittest
import numpy as np
from desispec.image import Image
from desispec.cosmics import reject_cosmic_rays_ala_sdss
from desispec.log import get_logger

#- Create a DESI logger at level WARNING to quiet down the fiberflat calc
import logging
log = get_logger(logging.WARNING)


class TestCosmics(unittest.TestCase):

    def test_rejection_ala_sdss(self):
        """
        Very basic test of reject_cosmic_rays_ala_sdss
        """
        #- Can it find a cosmic trace?
        pix=np.zeros((53,50))
        ivar=np.ones(pix.shape)
        for i in range(12,20) :
            pix[i,i]=100

        image = Image(pix,ivar)
        rejected=reject_cosmic_rays_ala_sdss(image,dilate=False)
        diff=np.sum(np.abs((pix>0).astype(int) - rejected.astype(int)))
        self.assertTrue(diff==0)

        #- Does it not find a PSF-like object?
        psfpix = np.zeros(pix.shape)
        for i in range(35,45):
            for j in range(35,45):
                r2 = i**2 + j**2
                psfpix[i,j] = np.exp(-r2/2.0)

        image = Image(psfpix,ivar)
        rejected=reject_cosmic_rays_ala_sdss(image,dilate=False)
        diff=np.sum(np.abs((pix>0).astype(int) - rejected.astype(int)))
        self.assertTrue(np.all(psfpix*(rejected==0) == psfpix))

        #- Can it find one and not the other?
        image = Image(pix+psfpix,ivar)
        rejected=reject_cosmic_rays_ala_sdss(image,dilate=False)
        diff=np.sum(np.abs((pix>0).astype(int) - rejected.astype(int)))
        self.assertTrue(np.all(psfpix*(rejected==0) == psfpix))
        diff=np.sum(np.abs((pix>0).astype(int) - rejected.astype(int)))
        self.assertTrue(diff==0)
        
#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()           
