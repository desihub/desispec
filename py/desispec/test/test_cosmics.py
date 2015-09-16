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
        pix=np.zeros((53,50))
        ivar=np.ones(pix.shape)
        for i in range(12,20) :
            pix[i,i]=100
        image = Image(pix,ivar)        
        rejected=reject_cosmic_rays_ala_sdss(image,dilate=False)
        diff=np.sum(np.abs((pix>0).astype(int) - rejected.astype(int)))
        self.assertTrue(diff==0)
        
#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()           
