"""Some simple unit tests of qproc/rowbyrowextract.py"""

from pkg_resources import resource_filename
import numpy as np
from specter.psf import load_psf
from desispec import io
from desispec.qproc import rowbyrowextract


def test_rowbyrowextract():
    psf = load_psf(resource_filename("specter.test", "t/psf-gausshermite2.fits"))
    shape = (psf.npix_y, psf.npix_x)
    pix = np.zeros(shape, dtype='f4')
    ivar = np.ones(shape, dtype='f4')
    image = io.image.Image(pix, ivar)

    # test #1: an all zero image should give all zero extracted spectra
    frame = rowbyrowextract.extract(image, psf, nspec=25)
    assert np.allclose(frame.flux, 0)

    # I don't immediately see how to construct "simple" PSFs appropriate for
    # unit tests from specter.  i.e., pure Gaussian, straight traces, ....
    # so these tests rapidly become more complicated than I would like.

    fibers = np.arange(25)
    # note: we're zeroing the tails since these are usually zero in DESI,
    # and so that's the default in extract(...) below.  But we could have
    # made the opposite choice.
    profiles = rowbyrowextract.onedprofile_gh(
        psf, np.arange(psf.npix_y), ispec=fibers, tails=False)
    profilex, profile, wave = profiles
    norm = np.sum(profile, axis=0)
    assert np.all(norm < 1.1)  # profiles shouldn't sum to much more than one?
    assert np.all(norm > 0.2)  # profiles shouldn't be too far from one?
    # note that the test PSF in specter is pretty ugly near the ends of the
    # wavelength bounds and needs a lot of tolerance!
    assert np.abs(np.median(norm) - 1) < 0.01
    # at least the median is well behaved.

    profwidth = profile.shape[0]
    assert np.all(profile[profwidth // 2, :, :] > profile[0, :, :])
    # profiles should be at least vaguely peaky.

    zeropix = rowbyrowextract.model(frame, profile, profilex, shape)
    assert np.allclose(zeropix, 0)

    frame.flux[...] = 1
    onefluxpix = rowbyrowextract.model(frame, profile, profilex, shape)
    image2 = io.image.Image(onefluxpix, ivar)
    frame2 = rowbyrowextract.extract(image2, psf, nspec=25)
    assert np.allclose(frame2.flux, 1)

    np.random.seed(1)
    frame.flux[...] = np.random.randn(*frame.flux.shape) + 10
    randomfluxpix = rowbyrowextract.model(frame, profile, profilex, shape)
    imagerandom = io.image.Image(randomfluxpix, ivar)
    framerandom = rowbyrowextract.extract(imagerandom, psf, nspec=25)
    assert np.allclose(framerandom.flux, frame.flux)

    noise = np.random.randn(*shape) * ivar**(-0.5)
    image3 = io.image.Image(onefluxpix + noise, ivar)
    frame3 = rowbyrowextract.extract(image3, psf, nspec=25)
    chi = (frame3.flux - 1) * frame3.ivar**0.5
    assert np.all(chi**2 < 6**2)
    assert np.abs(np.mean(chi**2) - 1) < 0.01
