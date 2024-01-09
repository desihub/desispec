"""
test desispec.cosmics
"""

import unittest
import numpy as np
from desispec.image import Image
from desispec.cosmics import reject_cosmic_rays_ala_sdss, reject_cosmic_rays
from desiutil.log import get_logger
from desispec.maskbits import ccdmask

#- Create a DESI logger at level WARNING to quiet down the fiberflat calc
import logging
log = get_logger(logging.WARNING)


class TestCosmics(unittest.TestCase):

    def setUp(self):
        #- pixels with a cosmic ray
        self.pix = np.zeros((53,50))
        self.ivar = np.ones(self.pix.shape)
        for i in range(12,20) :
            self.pix[i,i]=100

        #- pixels with a PSF-like object
        self.psfpix = np.zeros(self.pix.shape)
        for i in range(35,45):
            for j in range(35,45):
                r2 = (i-40)**2 + (j-40)**2
                self.psfpix[i,j] = np.exp(-r2/2.0)

        #- a bad pixel mask that goes through the PSF-like object
        self.badmask = np.zeros(self.pix.shape, dtype=np.uint32)
        self.badmask[:, 39] = ccdmask.BAD

    def test_rejection_ala_sdss(self):
        """
        Very basic test of reject_cosmic_rays_ala_sdss
        """
        #- Does it reject a diagonal cosmic ray?
        image = Image(self.pix, self.ivar, camera="r0")
        rejected=reject_cosmic_rays_ala_sdss(image,dilate=False)
        diff=np.sum(np.abs((self.pix>0).astype(int) - rejected.astype(int)))
        self.assertTrue(diff==0)

        #- Does it not find a PSF-like object?
        image = Image(self.psfpix, self.ivar, camera="r0")
        rejected=reject_cosmic_rays_ala_sdss(image,dilate=False)
        diff=np.sum(np.abs((self.pix>0).astype(int) - rejected.astype(int)))
        self.assertTrue(np.all(self.psfpix*(rejected==0) == self.psfpix))

        #- Can it find one and not the other?
        image = Image(self.pix+self.psfpix, self.ivar, camera="r0")
        rejected=reject_cosmic_rays_ala_sdss(image,dilate=False)
        diff=np.sum(np.abs((self.pix>0).astype(int) - rejected.astype(int)))
        self.assertTrue(np.all(self.psfpix*(rejected==0) == self.psfpix))
        diff=np.sum(np.abs((self.pix>0).astype(int) - rejected.astype(int)))
        self.assertTrue(diff==0)

    def test_psf_with_bad_column(self):
        '''test a PSF-like spot with a masked column going through it'''
        image = Image(self.psfpix, self.ivar, mask=self.badmask, camera="r0")
        image.pix[self.badmask>0] = 0
        rejected=reject_cosmic_rays_ala_sdss(image,dilate=False)
        self.assertTrue(not np.any(rejected))

    def test_different_cameras(self):
        '''test a PSF-like spot with a masked column going through it'''
        for camera in ('b0', 'r1', 'z2'):
            image = Image(self.pix, self.ivar, mask=self.badmask, camera=camera)
            rejected = reject_cosmic_rays_ala_sdss(image,dilate=False)
            cosmic = image.pix > 0
            self.assertTrue(np.all(rejected[cosmic]))

        #- camera must be valid
        with self.assertRaises(KeyError):
            image = Image(self.pix, self.ivar, mask=self.badmask, camera='a0')
            rejected = reject_cosmic_rays_ala_sdss(image,dilate=False)

    def test_reject_cosmics(self):
        """
        Test that the generic cosmics interface updates the mask
        """
        image = Image(self.pix, self.ivar, camera="r0")
        reject_cosmic_rays(image)
        cosmic = (image.pix > 0)
        self.assertTrue(np.all(image.mask[cosmic] & ccdmask.COSMIC))
