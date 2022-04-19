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


# let's gather a lot of skies and do a PCA
def gather_skies(fn=None, camera='r', petal=0, n=np.inf, heliocor=True,
                 tpcorr_power=1, include_mean_sky=True,
                 specprod='daily'):
    if fn is None:
        fn = glob.glob(os.path.join(
            os.environ['DESI_ROOT'], 'spectro', 'redux', 'fuji',
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
    return gather_skies(fn=args[0], camera=args[1], petal=args[2], **args[3])


def compute_pcs(skies, topn=6, niter=5):
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
    pcs = dict()
    for f in 'brz':
        newpcs = make_all_pcs(cameras=f, **kw)
        pcs.update(newpcs)
    return pcs


def write_pcs(pcs):
    for camera, petal in pcs:
        sm = desispec.calibfinder.sp2sm(petal)
        pc = pcs[camera, petal]
        smstr = 'sm%d' % sm
        camstr = '%s%d' % (camera, petal)
        calibfile = os.path.join(
            os.environ['DESI_SPECTRO_CALIB'], 'spec', smstr,
            'skygradpca-%s-%s.fits' % (smstr, camstr))
        desispec.io.write_skygradpca(calibfile, pc)


def doall():
    pcs = make_all_pcs_by_filter(specprod='fuji')
    write_pcs(pcs)
