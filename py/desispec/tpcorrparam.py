"""
desispec.tpcorrparam
====================

This module implements a model for how the throughput of a fiber varies.

It has the following components:

1. We find that the fibers don't follow the nighly flat field exactly, and
   apply a fixed mean correction from the nightly flat field.  This is thought
   to reflect the different range of angles entering the fiber from the flat
   field relative to the sky.
2. We find that fibers' throughput depends modestly on the fibers' location
   within their patrol radii.  We apply a cubic polynomial in (x, y) within
   the patrol disk to correct for this.
3. We find additional coherent variations in throughput from exposure to
   exposure.  These are modeled with a PCA and fit exposure-by-exposure.

The TPCorrParam object has mean, spatial, and pca attributes describing
each of these three components.
"""


import os
import numpy as np
from astropy.io import fits
from desiutil.log import get_logger
import desispec.calibfinder


def cubic(p, x, y):
    """Returns a cubic polynomial with coefficients p in x & y.

    Args:
        p (list[float]): 10 element floating point list with poly coefficients
        x (ndarray[float]): x coordinates
        y (ndarray[float]): y coordinates

    Returns:
        cubic polynomial in two variables with coefficients p evaluated at x, y
    """
    return (p[0] + p[1]*x + p[2]*y + p[3]*x**2 + p[4]*x*y + p[5]*y**2 +
            p[6]*x**3 + p[7]*x**2*y + p[8]*x*y**2 + p[9]*y**3)


def tpcorrspatialmodel(param, xfib, yfib):
    """Evaluate spatial tpcorr throughput model at xfib, yfib.

    Evaluates the throughput model at xfib, yfib.  param carries the
    polynomial coefficients of the model as well as the central fiber
    location.

    Args:
        param: ndarray describing fit.  Contains at least the following fields:
            PARAM: cubic polynomial coefficients
            X: central fiber X location (mm)
            Y: central fiber Y location (mm)
        xfib: x location at which model is to be evaluated (mm)
        yfib: y location at which model is to be evaluated (mm)

    Returns:
        throughput model evaluated at xfib, yfib
    """
    return cubic(param['PARAM'].T, (xfib-param['X'])/6, (yfib-param['Y'])/6)


def tpcorrmodel(tpcorrparam, xfib, yfib, pcacoeff=None):
    """Evaluates full TPCORR throughput model.

    The full model consistents of a PCA and a spatial within-patrol-radius
    component.  These two components are evaluated and summed.

    Args:
        tpcorrparam: TPCorrParam object describing the model
        xfib: x coordinates of fibers (mm)
        yfib: y coordinates of fibers (mm)
        pcacoeff (optional): PCA coefficients of throughput model
            If not set, no TPCORR PCA terms are fit.

    Returns:
        Throughput model evaluated at xfib, yfib with pca coefficients
        pcacoeff.
    """
    xfib = xfib.copy()
    yfib = yfib.copy()
    m = ~np.isfinite(xfib) | ~np.isfinite(yfib) | (xfib == 0) | (yfib == 0)
    xfib[m] = tpcorrparam.spatial['X'][m]
    yfib[m] = tpcorrparam.spatial['Y'][m]
    res = tpcorrparam.mean + tpcorrspatialmodel(
        tpcorrparam.spatial, xfib, yfib) - 1
    if pcacoeff is not None:
        for cc, vv in zip(pcacoeff, tpcorrparam.pca):
            res += cc * vv
    return res


class TPCorrParam:
    def __init__(self, mean, spatial, pca):
        """Create TPCorrParam object.

        Args:
            mean: mean throughput difference between flat and sky for each
                fiber
            spatial (ndarray): parameters of spatial model describing variation
                of throughput within patrol radius.  Contains the following
                fields:
                PARAM: cubic polynomial coefficients
                X: central X coordinate of fiber (mm)
                Y: central Y coordinate of fiber (mm)
            pca (ndarray[float]): principal components of throughput variations
                between fibers in an exposure.
        """
        self.mean = mean
        self.spatial = spatial
        self.pca = pca


def gather_tpcorr(recs):
    """Gather TPCORR measurements for exposures.

    Args:
        recs (ndarray): ndarray array containing information about exposures
            from which to gather TPCORR information.  Includes at least the
            following fields:
            NIGHT (int): night on which exposure was observed
            EXPID (int): exposure id

    Returns: ndarray with the following fields for each exposure
        X: x coordinate of fiber (mm)
        Y: y coordinate of fiber (mm)
        TPCORR: tpcorr of fiber
        CAMERA: camera for exposure
        NIGHT: night for exposure
        EXPID: expid of exposure
    """
    out = np.zeros(3*len(recs), dtype=[
        ('X', '5000f4'), ('Y', '5000f4'), ('TPCORR', '5000f4'),
        ('CAMERA', 'U1'), ('NIGHT', 'i4'), ('EXPID', 'i4')])
    count = -1
    for rec in recs:
        for camera in 'brz':
            count += 1
            out['CAMERA'][count] = camera
            out['NIGHT'][count] = rec['NIGHT']
            out['EXPID'][count] = rec['EXPID']
            for petal in range(10):
                address = (rec['NIGHT'], rec['EXPID'], f'{camera}{petal}')
                try:
                    sky = desispec.io.read_sky(address)
                    frame = desispec.io.read_frame(address)
                except Exception as e:
                    print(e)
                    continue
                fiber = frame.fibermap['FIBER']
                out['TPCORR'][count, fiber] = sky.throughput_corrections
                out['X'][count, fiber] = frame.fibermap['FIBER_X']
                out['Y'][count, fiber] = frame.fibermap['FIBER_Y']
    return out


def gather_dark_tpcorr(specprod=None):
    """Gathers TPCORR for dark exposures expected to be well behaved.

    This wraps gather_tpcorr, making a sensible selection of dark
    exposures from the given specprod.

    Args:
        specprod (str): specprod to use

    Returns:
        ndarray from gather_tpcorr with tpcorr information from selected
        exposures.
    """
    if specprod is None:
        specprod = os.environ.get('SPECPROD', 'daily')
    expfn = os.path.join(os.environ['DESI_SPECTRO_REDUX'],
                         specprod, 'exposures-{specprod}.fits')
    exps = fits.getdata(expfn)
    m = ((exps['EXPTIME'] > 300) & (exps['SKY_MAG_R_SPEC'] > 20.5) &
         (exps['SKY_MAG_R_SPEC'] < 30) & (exps['FAPRGRM'] == 'dark'))
    if specprod == 'daily':
        exps = exps[exps['NIGHT'] >= 20210901]
    return gather_tpcorr(exps[m])


def pca_tpcorr(tpcorr):
    """Run a PCA on TPCORR measurements.

    This uses the tpcorr gathered by gather_tpcorr and runs a PCA on them.

    Args:
        tpcorr (ndarray): result of gather_tpcorr

    Returns:
        dict: A dictionary containing:

        * dict[filter] of (tpcorrmed, pc_info, exp_info), describing the results
          of the fit.
        * tpcorrmed: the median TPCORR of the whole sample; i.e., how
          different the nightly flat usually is from the sky.
        * pc_info: ndarray with:

          - pca: the principal components
          - amplitude: their corresponding singular values

        * exp_info: ndarray with:

          - expid: exposure id
          - coeff: pc component coefficients for this exposure
    """
    from astropy.stats import mad_std
    out = dict()
    for f in 'brz':
        # z5 is ugly in 1st PC; what would we want to do here?
        # r7, 3994 in mean, 1st PC
        # 4891 is a mystery and shows very large variability
        m = ((tpcorr['CAMERA'] == f) &
             (np.sum(tpcorr['TPCORR'] == 0, axis=1) < 100))
        tpcorr0 = tpcorr['TPCORR'][m]
        tpcorr0[tpcorr0 == 0] = 1
        rms = mad_std(tpcorr0, axis=1)
        tpcorr0[rms > 0.025, :] = 1
        rms = mad_std(tpcorr0, axis=0)
        tpcorr0[:, rms > 0.1] = 1  # zero 4891
        if f == 'r':
            # zero bad CTE region on r4
            tpcorr0[:, 2250:2272] = 1
        tpcorr0 = np.clip(tpcorr0, 0.9, 1/0.9)
        tpcorrmed = np.median(tpcorr0, axis=0)
        tpcorr0 -= tpcorrmed[None, :]
        uu, ss, vv = np.linalg.svd(tpcorr0)
        eid = tpcorr['EXPID'][m]
        resvec = np.zeros(vv.shape[0], dtype=[
            ('pca', '%df4' % vv.shape[1]),
            ('amplitude', 'f4')])
        resvec['pca'] = vv
        if len(ss) < len(resvec):
            ss = np.concatenate([ss, np.zeros(len(resvec)-len(ss))])
        resvec['amplitude'] = ss
        resexp = np.zeros(uu.shape[0], dtype=[
            ('expid', 'i4'),
            ('coeff', '%df4' % uu.shape[1])])
        resexp['expid'] = eid
        resexp['coeff'] = uu
        out[f] = tpcorrmed, resvec, resexp
    return out


def fit_tpcorr_per_fiber(tpcorr):
    """Fit the spatial throughput variations for each fiber.

    Args:
        tpcorr (ndarray): output from gather_tpcorr with tpcorr
            measurements and related metadata for each fiber

    Returns:
        dict[camera] of ndarray including the following fields:
        X (float): central X position of each fiber used in fit (mm)
        Y (float): central Y position of each fiber used in fit (mm)
        FIBER (int): fiber number
        PARAM (ndarray[float], length 10): 2D cubic polynomial coefficients
    """
    from scipy.optimize import least_squares
    guess = np.zeros(10, dtype='f4')
    out = dict()
    log = get_logger()
    for camera in 'brz':
        res = np.zeros(5000, dtype=[
            ('FIBER', 'f4'), ('X', 'f4'), ('Y', 'f4'), ('PARAM', '10f4')])
        m = (tpcorr['CAMERA'] == camera)
        log.info('Starting spatial tpcorr analysis for %s camera' % camera)
        tpcorr0 = tpcorr[m]
        for i in range(5000):
            if (i % 1000) == 0:
                log.info('Fiber %d of 5000' % i)
            res['FIBER'][i] = i
            xx = tpcorr0['X'][:, i]
            yy = tpcorr0['Y'][:, i]
            mok = np.isfinite(xx) & np.isfinite(yy) & (xx != 0) & (yy != 0)
            if np.sum(mok) == 0:
                res['X'][i] = 0
                res['Y'][i] = 0
                res['PARAM'][i, :] = 0
                res['PARAM'][i, 0] = 1
                log.info(
                    'Dummy entry for fiber %d with no good measurements.' % i)
                continue
            res['X'][i] = np.median(xx[mok])
            res['Y'][i] = np.median(yy[mok])
            xx = (xx - res['X'][i]) / 6  # 6 mm fiber patrol radius
            yy = (yy - res['Y'][i]) / 6
            bb = tpcorr0['TPCORR'][:, i]
            m = (bb != 0) & (bb > 0.5) & (bb < 2) & mok
            # a few fibers (e.g., 1680) have some weird historical TPCORR
            # data.  Don't use those horrible points.
            # a large chunk of fibers on r4 are bad (250 - 272).
            # This is the bad CTE region.  We probably don't want to use
            # these in the mean adjustment.  See if these have IVAR=0 or
            # something?
            xx = xx[m]
            yy = yy[m]
            bb = bb[m]
            def model(param):
                return cubic(param, xx, yy)
            def chi(param):
                return (bb - model(param))
            fit = least_squares(chi, guess, loss='soft_l1')
            res['PARAM'][i, :] = fit.x
            # do something sane for stationary fibers
            cov = np.cov(xx*6, yy*6)
            det = np.linalg.det(cov)
            # normal positioner has a determinant of ~60
            # this flags stuck ones, or ones that have been stuck for
            # a very long time.
            # covariance does a bit better than stdev for occasional
            # positioners that have been stuck in a few different positions
            if det < 10:
                res['PARAM'][i, 0] = np.median(bb)
                res['PARAM'][i, 1:] = 0
        out[camera] = res
    return out


# note! one worry: in the current PR, the meaning of TPCORR will change
# moving forward, and we won't have any of the current throughput measurements
# for computing these coefficients in the future.  We'd have to develop
# an alternative scheme for computing these coefficients or would have
# to change this PR so that the new and old TPCORR have similar meanings.
# That option would mean passing calculate_throughput_corrections some
# goofy sky model that doesn't include the new model TPCORR from this PR.
# Or we add another model extension with the new TPCORR from this PR,
# and then this routine would require the product of the model TPCORR and
# the measured TPCORR.
def make_tpcorrparam_files(tpcorr=None, spatial=None, dirname=None):
    """Gather, fit, and write tpcorrparam files to output directory.

    This function is intended for producing tpcorrparam files to
    DESI_SPECTRO_CALIB.  It can gather and fit tpcorr measurements and
    write the results.

    Args:
        tpcorr (optional): result of gather_tpcorr.  If None, this is
            populated with the result of gather_dark_tpcorr.
        spatial (optional): spatial fit results.  If None, this is
            populated with fit_tpcorr_per_fiber(tpcorr).
        dirname (optional): directory to which to write files.  Defaults
            to DESI_SPECTRO_CALIB.
    """
    if dirname is None:
        dirname = os.environ['DESI_SPECTRO_CALIB']
    if tpcorr is None:
        tpcorr = gather_dark_tpcorr()
    if spatial is None:
        spatial = fit_tpcorr_per_fiber(tpcorr)
    tpcorrspatmod = np.zeros_like(tpcorr['TPCORR'])
    for f in 'brz':
        m = tpcorr['CAMERA'] == f
        tpcorrspatmod[m] = tpcorrspatialmodel(
            spatial[f], tpcorr['X'][m], tpcorr['Y'][m])
    m = ((tpcorr['X'] == 0) | (tpcorr['Y'] == 0) |
         ~np.isfinite(tpcorr['X']) | ~np.isfinite(tpcorr['Y']) |
         ~np.isfinite(tpcorr['TPCORR']))
    tpcorrresid = tpcorr.copy()
    tpcorrresid['TPCORR'] -= tpcorrspatmod
    tpcorrresid['TPCORR'] += 1
    tpcorrresid['TPCORR'][m] = 0  # invalid value
    pca = pca_tpcorr(tpcorrresid)
    for f in 'brz':
        constant = spatial[f]['PARAM'][:, 0].copy()
        spatial[f]['PARAM'][:, 0] = 1
        for i in range(10):
            sm = desispec.calibfinder.sp2sm(i)
            sm = 'sm%d' % sm
            fn = os.path.join(dirname, 'spec', sm,
                              f'tpcorrparam-{sm}-{f}{i}.fits')
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            fits.writeto(fn, spatial[f][i*500:(i+1)*500],
                         fits.Header(dict(extname='SPATIAL')))
            fits.append(fn, constant[i*500:(i+1)*500],
                        fits.Header(dict(extname='MEAN')))
            fits.append(fn, pca[f][1]['pca'][:2, i*500:(i+1)*500],
                        fits.Header(dict(extname='PCA')))
