"""
desispec.skygradpca
============

Class with sky model gradient templates
"""

import os
import glob
import numpy as np
import multiprocessing.pool
from scipy.signal import medfilt
from scipy.ndimage import map_coordinates
from astropy.table import Table
import astropy.stats
import desispec.io
import desispec.calibfinder


class SkyGradPCA(object):
    def __init__(self, wave, flux, header=None):
        """Create SkyGradPCA object

        Args:
            wave  : 1D[nwave] wavelength in Angstroms
            flux  : 2D[ntemplate, nwave] template fluxes (unitless)
            header : (optional) header from FITS file HDU0
        All input arguments become attributes
        """
        assert wave.ndim == 1
        assert flux.ndim == 2
        self.nspec, self.nwave = flux.shape
        self.wave = wave
        self.flux = flux
        self.header = header


def configure_for_xyr(skygradpca, x, y, R, skyfibers=None):
    """Add additional information to a SkyGradPCA object for fitting.

    A SkyGradPCA object contains the templates used to fit out sky gradients.
    When these templates are actually used in fitting, it's helpful to
    additionally know things like the locations of the fibers on the particular
    exposure and the mean resolution matrix for the exposure, etc.  This
    function mutates the skygradpca object, adding new attributes for fitting.

    Args:
        skygradpca: SkyGradPCA object to extend
        x: 1D[nfiber] array of x fiber coordinates on exposure (mm)
        y: 1D[nfiber] array of y fiber coordinates on exposure (mm)
        R: list of resolution matrices for each fiber (unitless)
        skyfibers: (optional) indices of sky fibers in list of all fibers
    """
    skygradpca.x = x
    skygradpca.y = y
    skygradpca.dx = x - np.mean(x)
    skygradpca.dy = y - np.mean(y)
    meanR = np.sum(R) / len(R)
    deconvskygradpca = np.zeros_like(skygradpca.flux)
    for i in range(skygradpca.nspec):
        deconvskygradpca[i] = np.linalg.solve(
            meanR.T.dot(meanR).toarray(),
            meanR.T.dot(skygradpca.flux[i]))
    skygradpca.deconvflux = deconvskygradpca
    skygradpca.skyfibers = skyfibers


def evaluate_model(skygradpca, R, coeff, mean=None):
    """Evaluate a skygradpca object for a frame.

    The skygradpca object must first be configured for the frame with
    configure_for_xyr.

    Args:
        skygradpca: SkyGradPCA instance, configured with configure_for_xyr
        R: resolution matrices for frame
        coeff (array, 1D): coefficients for eigenvectors
        mean (optional): mean sky spectrum

    Returns:
        sky spectrum (array[nfiber, nwave]) given mean spectrum, per-fiber
        resolution matrices, and gradient amplitudes
    """

    nfiber = len(skygradpca.x)
    sky = np.zeros((nfiber, skygradpca.nwave), dtype='f8')
    if mean is not None:
        sky += mean[None, :]
    for i in range(nfiber):
        for j in range(skygradpca.nspec):
            sky[i, :] += (
                coeff[2*j+0]*skygradpca.deconvflux[j]*skygradpca.dx[i] +
                coeff[2*j+1]*skygradpca.deconvflux[j]*skygradpca.dy[i])
        sky[i, :] = R[i].dot(sky[i, :])
    return sky


def gather_skies(fn=None, camera='r', petal=0, n=np.inf, heliocor=True,
                 specprod='daily'):
    """Gather sframes for performing bulk analyses; e.g., PCA.

    This function gathers residual sky spectra together for many exposures,
    bundling it together for later analysis.

    Args:
        fn: (list[str], optional) sframe file names to gather
            If not set, all of the sframes in the `specprod` product in the
            `camera` camera and `petal` petal will be gathered.
        camera: (str, optional) camera to gather
        petal: (int, optional) petal to gather
        n: (int, optional) only gather at most this many exposures worth of
            sky spectra
        heliocor: (bool, optional) if True, apply correction to sframe files
            to bring them to earth frame

    Returns:
        ndarray with EXPID, FIBER, CAMERA, X, Y, WAVE, FLUX, IVAR for each
        sky fiber from sframe files.
    """
    if fn is None:
        fn = glob.glob(os.path.join(
            os.environ['DESI_ROOT'], 'spectro', 'redux', specprod,
            'exposures', '*', '*', f'sframe-{camera}{petal}-*'))
        if len(fn) > n:
            fn = fn[:n]
    out = []
    for i, fn0 in enumerate(fn):
        try:
            frame = desispec.io.read_frame(fn0)
        except Exception as e:
            print(e)
            continue
        if frame.meta['EXPTIME'] < 100:
            continue
        mfib = ((frame.fibermap['OBJTYPE'] == 'SKY') &
                (~np.all(frame.mask != 0, axis=1)) &
                (~np.all(frame.ivar == 0, axis=1)))
        flux = frame.flux[mfib, :]
        medflux = medfilt(flux, kernel_size=[1, 11])
        m = frame.mask[mfib, :] != 0
        flux[m] = medflux[m]
        ivar = frame.ivar[mfib, :]
        ivar[m] = 0
        wave = frame.wave
        if heliocor:
            wavegeo = wave / frame.meta['HELIOCOR']
            rownum = (np.arange(flux.shape[0])[:, None] *
                      np.ones((1, flux.shape[1])))
            colnum = np.interp(wave, wavegeo, np.arange(len(wavegeo)))
            colnum = colnum[None, :]*np.ones((flux.shape[0], 1))
            flux = map_coordinates(flux, [rownum, colnum], mode='nearest')
            var = 1/ivar
            var = map_coordinates(var, [rownum, colnum], mode='nearest',
                                  order=1)
            ivar = 1/var
            ivar[~np.isfinite(ivar)] = 0
        nwave = flux.shape[1]
        res = np.zeros(np.sum(mfib), dtype=[
            ('EXPID', 'i4'), ('FIBER', 'i4'), ('CAMERA', 'U1'),
            ('X', 'f4'), ('Y', 'f4'),
            ('WAVE', f'{nwave}f4'),
            ('FLUX', f'{nwave}f4'), ('IVAR', f'{nwave}f4')])
        res['X'] = frame.fibermap['FIBERASSIGN_X'][mfib]
        res['Y'] = frame.fibermap['FIBERASSIGN_Y'][mfib]
        res['FLUX'] = flux
        res['IVAR'] = ivar
        res['CAMERA'] = frame.meta['CAMERA'][0]
        res['FIBER'] = frame.fibermap['FIBER'][mfib]
        res['EXPID'] = frame.meta['EXPID']
        res['WAVE'] = wave
        out.append(res)
    return np.concatenate(out)


def make_all_pcs_wrapper(args):
    """Wrapper function; gathers skies for cam/petal combinations."""
    return gather_skies(fn=args[0], camera=args[1], petal=args[2], **args[3])


def compute_pcs(skies, topn=6, niter=5):
    """Computes principal components of sky spectra.

    This computes the PCs used for the skygradpca product.

    Args:
        skies: output of gather skies; the sky spectra to do PCA on
        topn: (optional int) number of PCA to compute
        niter: (optional int) number of iterations to do, rejecting
            discrepant pixels.

    Returns:
        Output of np.linalg.svd on the clipped, mean-subtracted sky spectra.
    """
    flux = skies['FLUX'].copy()
    nmed = 101
    masklimit = nmed / 1.5
    fluxmed = medfilt(flux, [1, nmed])
    ivar = skies['IVAR'].copy()
    res = astropy.stats.sigma_clip(flux, axis=0)
    flux[res.mask] = fluxmed[res.mask]
    ivar[res.mask] = 0
    nmask = np.sum(res.mask, axis=1)
    flux = flux[nmask < masklimit, :]
    skies = skies[nmask < masklimit]
    ivar = ivar[nmask < masklimit, :]
    chi2 = np.zeros(len(skies), dtype='f4')

    for i in range(niter):
        mn, med, sd = astropy.stats.sigma_clipped_stats(chi2)
        nsd = 2
        keep = ~np.all(ivar == 0, axis=1) & (chi2 <= mn + nsd*sd)
        dvec = flux[keep] - np.mean(flux[keep], axis=0, keepdims=True)
        uu, ss, vv = np.linalg.svd(dvec)
        ss2 = ss.copy()
        ss2[topn:] = 0
        ind = np.arange(min([uu.shape[0], vv.shape[0]]))
        ss2mat = np.zeros((uu.shape[0], vv.shape[0]), dtype='f4')
        ss2mat[ind, ind] = ss2
        recon = np.dot(uu, np.dot(ss2mat, vv))
        chi = (dvec - recon)*np.sqrt(ivar[keep])
        chi2[keep] = np.sum(chi**2, axis=1)

    return uu, ss, vv


def make_all_pcs(specprod, minnight=20200101,
                 cameras='brz', petals=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], **kw):
    """Makes SkyGradPCA objects derived from residual skies from a specprod.

    This grabs all of the sframe files passing certain criteria using
    gather_skies and runs a PCA on them, producing SkyGradPCA objects for
    each camera/petal.  It uses 30 processes, one for each camera/petal
    combination.

    Args:
        specprod (str): specprod to use
        minnight (int, optional): use only spectra taken after this date
        cameras (str, optional): string containing some of 'brz'
        petals (list[int], optional): petals to use
        **kw (dict, optional): additional keywords passed to gather_skies.

    Returns:
        dictionary[camera, petal] containing the corresponding SkyGradPCA
        object.
    """
    exps = Table.read(os.path.join(
        os.environ['DESI_SPECTRO_REDUX'], specprod,
        f'exposures-{specprod}.csv'))
    m = ((exps['NIGHT'] >= minnight) &
         (exps['SKY_MAG_R_SPEC'] < 19.5) &
         (exps['EXPTIME'] > 60) &
         (exps['FAPRGRM'] != 'BACKUP') &
         (exps['EFFTIME_SPEC'] > 0))
    exps = exps[m]
    pool = multiprocessing.pool.Pool(len(cameras)*len(petals))
    combos = [[c, p] for c in cameras for p in petals]
    fnall = []
    for c, p in combos:
        fn = [os.path.join(os.environ['DESI_SPECTRO_REDUX'],
                           specprod, 'exposures', f'{e["NIGHT"]:08d}',
                           f'{e["EXPID"]:08d}',
                           f'sframe-{c}{p}-{e["EXPID"]:08d}.fits')
              for e in exps]
        fnall.append(fn)

    out = pool.map(make_all_pcs_wrapper,
                   [[f]+c+[kw] for f, c in zip(fnall, combos)])
    res = pool.map(compute_pcs, out)
    pcs = dict()
    for (c, p), r, o in zip(combos, res, out):
        pcs[c, p] = SkyGradPCA(o['WAVE'][0], r[2][:2, :].copy())
    return pcs


def make_all_pcs_by_filter(**kw):
    """Wrapper for make_all_pcs, doing one filter at a time."""
    pcs = dict()
    for f in 'brz':
        newpcs = make_all_pcs(cameras=f, **kw)
        pcs.update(newpcs)
    return pcs


def write_pcs(pcs):
    """Writes SkyGradPCA objects to DESI_SPECTRO_CALIB.

    Takes the output of make_all_pcs and writes it to DESI_SPECTRO_CALIB.

    Args:
        pcs (dict[camera, petal] of SkyGradPCA): output of make_all_pcs
    """
    for camera, petal in pcs:
        sm = desispec.calibfinder.sp2sm(petal)
        pc = pcs[camera, petal]
        smstr = 'sm%d' % sm
        camstr = '%s%d' % (camera, petal)
        calibfile = os.path.join(
            os.environ['DESI_SPECTRO_CALIB'], 'spec', smstr,
            'skygradpca-%s-%s.fits' % (smstr, camstr))
        desispec.io.write_skygradpca(calibfile, pc)


def doall(specprod='fuji'):
    """Perfoms and writes out analysis of sframe sky residuals.

    Gathers all of the sframe residual sky spectra for each camera/petal,
    runs a PCA on them, makes the appropriate SkyGradPCA objects, and
    writes them out to DESI_SPECTRO_CALIB.

    Args:
        specprod (str, optional): specprod to run on
    """
    pcs = make_all_pcs_by_filter(specprod)
    write_pcs(pcs)
