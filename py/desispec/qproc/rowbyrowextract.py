"""
desispec.qproc.rowbyrowextract
==============================

Row-by-row extraction routines (Horne 1986).
"""

import numpy as np
import numba
from scipy import special, linalg
from specter.util import custom_hermitenorm, custom_erf
from desispec.qproc.qframe import QFrame
from desispec import qproc, io
from desiutil.log import log


@numba.jit(nopython=True, cache=False)
def pgh(x, m=0, xc=0.0, sigma=1.0):
    """
    Pixel-integrated (probabilist) Gauss-Hermite function.
    Arguments:
      x: pixel-center baseline array
      m: order of Hermite polynomial multiplying Gaussian core
      xc: sub-pixel position of Gaussian centroid relative to x baseline
      sigma: sigma parameter of Gaussian core in units of pixels
    Uses the relationship
    Integral{ H_k(x) exp(-0.5 x^2) dx} = -H_{k-1}(x) exp(-0.5 x^2) + const
    Written: Adam S. Bolton, U. of Utah, fall 2010
    Adapted for efficiency by S. Bailey while dropping generality
    modified from the orig _pgh to enable jit-compiling
    --> no longer passing in self, calling custom numba functions in util
    E. Schlafly added some broadcasting bits to make it play more nicely
    with large arrays.  This should probably go to specter instead.
    """

    # Evaluate H[m-1] at half-pixel offsets above and below x
    dx = x-xc-0.5
    u = np.concatenate((dx, dx[-1:]+1.0)) / sigma
    origshape = u.shape
    u = u.reshape(-1)

    if m > 0:
        y = (-custom_hermitenorm(m-1, u) * np.exp(-0.5 * u**2) /
             np.sqrt(2. * np.pi))
        y = y.reshape(origshape)
        return (y[1:] - y[0:-1])
    else:
        y = custom_erf(u/np.sqrt(2.))
        y = y.reshape(origshape)
        return 0.5 * (y[1:] - y[0:-1])


def onedprofile_gaussian(psf, ypix, ispec, nsig=4):
    """Compute the cross-dispersion profile for a Gaussian.

    This was warm-up code for the Gauss-Hermite implementation below,
    but may have independent value?  It will work with any specter PSF,
    albeit using a Gaussian approximation to the cross-dispersion profile.

    Parameters
    ----------
    psf : specter.psf.PSF
        psf
    ypix : np.ndarray
        rows for which to compute profile
    ispec : np.ndarray
        indices of spectra for which to compute profiles
    nsig : np.ndarray
        how many sigma out to compute the Gaussian; profile is treated
        as zero beyond roughly here.

    Returns
    -------
    profilepix, profile, lambda
    profilepix : np.ndarray
        the x-coordinates corresponding to profile on the detector
    profile : np.ndarray
        the cross dispersion profiles
    lambda : np.ndarray
        the wavelengths of the spectra at the requested locations
    """
    lams = []
    xsigs = []
    xcens = []
    for ispec0 in ispec:
        lam = psf.wavelength(ispec=ispec0, y=ypix)
        lams.append(lam)
        xsigs.append(psf.xsigma(ispec0, lam))
        xcens.append(psf.x(ispec0, wavelength=lam))

    hwidth = int(np.ceil(np.max([nsig * np.max(xsig0) for xsig0 in xsigs])))
    xcens = np.array(xcens)
    xpixs = (np.round(xcens)[None, :, :].astype('i4') +
             np.arange(-hwidth, hwidth+1)[:, None, None])
    xsigs = np.array(xsigs)
    lams = np.array(lams)
    uu = (xpixs - 0.5 - xcens[None, ...]) / xsigs[None, ...]
    erfs = special.erf(uu / np.sqrt(2))
    uulast = (xpixs[-1] + 0.5 - xcens) / xsigs
    erflast = special.erf(uulast / np.sqrt(2))
    profiles = np.diff(np.vstack([erfs, erflast[None, :]]), axis=0)
    profiles /= np.sum(np.clip(profiles, 0., np.inf), axis=0, keepdims=True)
    return xpixs, profiles, lams


def tail1d(core, xx, index):
    """Compute the cross-dispersion profile of the tail model.

    Parameters
    ----------
    core : np.ndarray
        core scale (pixels)
    xx : np.ndarray
        distance in x from center
    index : np.ndarray
        power law index of tail model

    Returns
    -------
    np.ndarray
    tail amplitude at given location
    """
    tail = (np.sqrt(np.pi) * (core**2 + xx**2) ** (-0.5 * (1 + index))
            * (core**2 + index * xx**2) / index
            * special.gamma((index - 1) / 2) / special.gamma(index / 2))
    return tail


def onedprofile_gh(psf, ypix, ispec, tails=True):
    """Compute cross-dispersion profile for Gauss-Hermite PSFs

    Parameters
    ----------
    psf : specter.psf.gausshermite.GaussHermitePSF
        psf
    ypix : np.ndarray
        rows for which to compute profile
    ispec : np.ndarray
        indices of spectra for which to compute profiles

    Returns
    -------
    profilepix, profile, lambda
    profilepix : np.ndarray
        the x-coordinates corresponding to profile on the detector
    profile : np.ndarray
        the cross dispersion profiles
    lambda : np.ndarray
        the wavelengths of the spectra at the requested locations
    """
    lams = []
    sigx1s = []
    xcens = []
    coeffs = {}
    degx1 = psf._polyparams['GHDEGX']
    for ispec0 in ispec:
        lam = psf.wavelength(ispec0, y=ypix)
        lams.append(lam)
        sigx1s.append(psf.coeff['GHSIGX'].eval(ispec0, lam))
        xcens.append(psf._x.eval(ispec0, lam))
        for k in range(degx1 + 1):
            coeffs[k] = coeffs.get(k, list())
            coeffs[k].append(psf.coeff[f'GH-{k}-0'].eval(ispec0, lam))

    hwidth = psf._polyparams['HSIZEX']
    xcens = np.array(xcens)
    sigx1s = np.array(sigx1s)
    lams = np.array(lams)
    xpixs = (np.round(xcens)[None, :, :].astype('i4') +
             np.arange(-hwidth, hwidth+1)[:, None, None])
    xfunc1 = np.zeros((degx1 + 1,) + xpixs.shape)
    for i in range(degx1 + 1):
        xfunc1[i, ...] = pgh(xpixs, i, xcens, sigx1s)
    profiles = np.zeros(xpixs.shape, dtype='f4')
    for k in range(degx1+1):
        profiles += np.array(coeffs[k]) * xfunc1[k, ...]
    if tails:
        # add on the contribution from the tails model.
        for i, ispec0 in enumerate(ispec):
            # math in tail1d assumes TAILXSCA = 1
            # it looks like that's a usual value
            # tailxsca = psf.coeff['TAILXSCA'].eval(ispec0, lams[i])
            # math in tail1d assumes TAILYSCA = 1
            # tailysca = psf.coeff['TAILYSCA'].eval(ispec, lams[i])
            tailamp = psf.coeff['TAILAMP'].eval(ispec0, lams[i])
            tailcore = psf.coeff['TAILCORE'].eval(ispec0, lams[i])
            tailinde = psf.coeff['TAILINDE'].eval(ispec0, lams[i])
            profiles[:, i, :] += tailamp * tail1d(
                tailcore, xpixs[:, i, :] - xcens[i], tailinde)

    profiles /= np.sum(np.clip(profiles, 0., np.inf), axis=0, keepdims=True)
    return xpixs, profiles, lams


def model(frame, profile, profilepix, shape):
    """Create model image for frame.

    Parameters
    ----------
    frame : desispec.qproc.qframe.QFrame
        frame for which to create model
    profile : np.ndarray
        cross-dispersion profile at each row for each spectrum
    profilepix : np.ndarray
        x-coordinates in image corresponding to each entry in profile
    shape : (int, int)
        shape of output image
    """
    out = np.zeros(shape, dtype='f4')
    nspec, npix = frame.flux.shape
    ipix = np.arange(npix)
    for ispec in range(nspec):
        out[ipix, profilepix[:, ispec, ipix]] += (
            profile[:, ispec, :] * frame.flux[ispec, :])
    return out


@numba.jit(nopython=True, cache=False)
def build_atcinva(atcinva, xstart, xend, profiles, ivar):
    """Build A^T C^-1 A for the row-by-row extraction.

    Modifies input atcinva in place.  Assumes spectra are ordered,
    so xstart[i+1] > xstart[i].

    Parameters
    ----------
    atcinvb : np.ndarray
        matrix to fill in (modified in place)
    xstart : int
        x coordinate of start of each profile
    xend : int
        x coordinate of end of each profile
    profiles : np.ndarray
        cross-dispersion profile for each trace
    ivar : np.ndarray
        ivar for appropriate row of image
    """
    npix = profiles.shape[0]
    nspec = profiles.shape[1]
    for i1 in range(nspec):
        for i2 in range(i1, nspec):  # just the upper half
            noverlap = xend[i1] - xstart[i2] + 1
            if noverlap <= 0:
                break  # no overlap between spectra
                # we can break rather than continue since if these two
                # don't overlap, no more widely separated spectra will
                # overlap either.
            for j in range(noverlap):
                atcinva[i1, i2] += (
                    profiles[npix - noverlap + j, i1] *
                    ivar[xstart[i2] + j] *
                    profiles[j, i2])
            if i1 != i2:  # fill in the bottom half of the matrix
                atcinva[i2, i1] = atcinva[i1, i2]


@numba.jit(nopython=True, cache=False)
def build_atcinvb(atcinvb, xstart, profiles, data):
    """Build A^T C^-1 b for the row-by-row extraction.

    Modifies input atcinvb in place.

    Parameters
    ----------
    atcinvb : np.ndarray
        matrix to fill in (modified in place)
    xstart : int
        x coordinate of start of each profile
    profiles : np.ndarray
        cross-dispersion profile for each trace
    data : np.ndarray
        flux * ivar for appropriate row of image
    """
    npix = profiles.shape[0]
    nspec = profiles.shape[1]
    for i1 in range(nspec):
        for j in range(npix):
            atcinvb[i1] += profiles[j, i1] * data[xstart[i1] + j]


def extract(image, psf, blocksize=25, fibermap=None, nspec=500,
            return_model=False, tails=False):
    """Extract spectra from image using an optimal row-by-row extraction.

    The algorithm extracts the spectra row-by-row, finding the fluxes
    associated with the 1D PSF that best explain the data.

    This only works for Gauss-Hermite PSFs at present, since it does
    some trickery to efficiently get 1D PSFs for large groups of wavelengths
    and spectra.

    Parameters
    ----------
    image : Image
        image from which to extract the spectra.
    psf : specter.psf.gausshermite.GaussHermitePSF
        PSF to use
    blocksize : int
        number of spectra to fit simultaneously.  Spectra in
        different blocks will be fit independently.
    fibermap : astropy.Table
        fibermap for image
    nspec : int
        total number of spectra to extract
    return_model : bool
        if True, return also the model of the image, the profiles,
        and the locations in the image corresponding to the profiles.
    tails : bool
        include PSF tails in the modeling.  These are usually identically
        zero for DESI and so default to off.

    Returns
    -------
    frame : desispec.qproc.qframe.QFrame
        The 1D extracted spectra as a QFrame
    """
    allispec = np.arange(nspec)
    ny, nx = image.pix.shape
    outspec = np.zeros((nspec, ny), dtype='f4')
    outvar = np.zeros((nspec, ny), dtype='f4')
    startfiber = 0
    wave = np.zeros((nspec, ny), dtype='f4')
    outprofile = []
    outprofilepix = []
    # avoid singular matrices, have some huge uncertainties instead
    ivar = np.clip(image.ivar, 1e-10, np.inf)
    data = image.pix * ivar
    log.info(f'Extracting {nspec} fibers...')
    while startfiber < nspec:
        # generate all 1d profiles
        ispec = allispec[startfiber:startfiber + blocksize]
        # 34% of time.
        # tails = False since these aren't actually used at present,
        # and we compute a lot to get zero.
        xxa, profilesa, wavea = onedprofile_gh(psf, np.arange(ny), ispec,
                                               tails=tails)
        outprofile.append(profilesa)
        outprofilepix.append(xxa)
        wave[ispec, :] = wavea
        atcinva = np.zeros((len(ispec), len(ispec)), dtype='f4')
        atcinvb = np.zeros(len(ispec), dtype='f4')
        for ty in np.arange(ny):  # ~60% of time in this loop
            xx = xxa[:, :, ty]
            minx = xx[0, 0]
            maxx = xx[-1, -1]
            xstart = xx[0, :] - minx
            xend = xx[-1, :] - minx
            profiles = profilesa[:, :, ty]
            ivar0 = ivar[ty, minx:maxx + 1]
            data0 = data[ty, minx:maxx + 1]
            atcinva *= 0
            atcinvb *= 0
            build_atcinva(atcinva, xstart, xend, profiles, ivar0)
            build_atcinvb(atcinvb, xstart, profiles, data0)

            # this takes ~45% of the time.  It's presumably possible to speed
            # up at least the variance computation---we don't need all of the
            # off diagonal elements, and presumably part of solving involves
            # computing something like the LU factorization from which we could
            # maybe efficiently get the variances.  But I stared a little at
            # cholesky decompositions and couldn't immediately find a win here.
            par = linalg.solve(atcinva, atcinvb)
            var = np.diag(linalg.inv(atcinva))

            outspec[ispec, ty] = par
            outvar[ispec, ty] = var
        startfiber += blocksize

    if fibermap is None:
        log.warning("setting up a fibermap to save the FIBER identifiers")
        fibermap = io.fibermap.empty_fibermap(nspec)  # 2% of time
        fibermap["FIBER"] = np.arange(nspec)
        if (image.meta is not None) and ('CAMERA' in image.meta) and (image.meta['CAMERA'] != 'unknown'):
            petal = int(image.meta['CAMERA'][1])
            fibermap["FIBER"] += petal*500
    else:
        fibermap = fibermap[:nspec]

    out = QFrame(wave, outspec, 1/outvar, mask=None,
                              fibers=fibermap['FIBER'],
                              meta=image.meta, fibermap=fibermap)
    if return_model:
        outprofile = np.concatenate(outprofile, axis=1)
        outprofilepix = np.concatenate(outprofilepix, axis=1)
        # 3% of time
        outmodel = model(out, outprofile, outprofilepix, image.pix.shape)
        out = (out, outmodel, outprofile, outprofilepix)
    return out
