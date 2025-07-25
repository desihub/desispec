#!/usr/bin/env python

# Compute the per-fiber redshift success rates and save them to disk
# Usage:
# salloc -N 1 -C cpu -t 04:00:00 -q interactive ./desi_per_fiber_qa_stats.py -i /global/cfs/cdirs/desicollab/users/rongpu/tmp/zcatalog/v0.4/main/ -o per_fiber_qa_stats.fits

from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack, join
import fitsio
# from astropy.io import fits

from scipy.stats import kstest
from scipy.interpolate import interp1d
from desitarget.targetmask import desi_mask, bgs_mask

import argparse

import time
time_start = time.time()


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--indir', type=str, help="input directory path", required=True)
parser.add_argument('-o', '--output', type=str, help="output file path", required=True)
args = parser.parse_args()

indir = args.indir
output_fn = args.output

ks_test = True
pvalue_threshold = 1e-4

for tracer in ['LRG', 'ELG_LOP', 'ELG_VLO', 'QSO', 'BGS_BRIGHT', 'BGS_FAINT']:

    print(tracer)

    if tracer in ['LRG', 'ELG', 'QSO', 'ELG_LOP', 'ELG_VLO', 'BGS_ANY']:
        fn = os.path.join(indir, 'ztile-main-dark-cumulative.fits')
        fn1 = os.path.join(indir, 'ztile-main-dark-cumulative-extra.fits')
        cat = Table(fitsio.read(fn))
        cat1 = Table(fitsio.read(fn1, columns=['Z', 'ZWARN', 'Z_QSO', 'GOOD_Z_LRG', 'GOOD_Z_ELG', 'GOOD_Z_QSO']))
        cat = hstack([cat, cat1], join_type='exact')
        mask = np.where(cat['DESI_TARGET'] & desi_mask[tracer] > 0)[0]
        cat = cat[mask]
    else:
        fn = os.path.join(indir, 'ztile-main-bright-cumulative.fits')
        fn1 = os.path.join(indir, 'ztile-main-bright-cumulative-extra.fits')
        cat = Table(fitsio.read(fn))
        cat1 = Table(fitsio.read(fn1, columns=['Z', 'ZWARN', 'GOOD_Z_BGS']))
        cat = hstack([cat, cat1], join_type='exact')
        mask = np.where(cat['BGS_TARGET'] & bgs_mask[tracer] > 0)[0]
        cat = cat[mask]

    print(tracer, 'zcat', len(cat))
    stats_zcat = Table()
    stats_zcat['FIBER'], stats_zcat[tracer.lower()+'_zcat_n_tot'] = np.unique(cat['FIBER'], return_counts=True)
    # fill in the missing fibers
    tmp = Table()
    tmp['FIBER'] = np.arange(5000)
    stats_zcat = join(stats_zcat, tmp, keys='FIBER', join_type='outer').filled(0)

    # Remove FIBERSTATUS!=0 fibers
    mask = cat['COADD_FIBERSTATUS']==0
    print('FIBERSTATUS   ', np.sum(~mask), np.sum(mask), np.sum(~mask)/len(mask))
    cat = cat[mask]

    # Remove "no data" fibers
    mask = cat['ZWARN'] & 2**9==0
    print('No data   ', np.sum(~mask), np.sum(mask), np.sum(~mask)/len(mask))
    cat = cat[mask]

    # Require a minimum depth
    if tracer in ['BGS_ANY', 'BGS_BRIGHT', 'BGS_FAINT']:
        min_depth = 160
        mask = cat['EFFTIME_SPEC']>min_depth
    else:
        min_depth = 800.
        mask = cat['EFFTIME_SPEC']>min_depth
    print('Min depth   ', np.sum(~mask), np.sum(mask), np.sum(~mask)/len(mask))
    cat = cat[mask]

    # Apply masks
    if tracer=='LRG':
        tmp1 = Table(fitsio.read(os.path.join('/dvs_ro/cfs/cdirs/desi/users/rongpu/targets/dr9.0/1.1.1/resolve/dr9_lrg_1.1.1_basic.fits'), columns=['TARGETID']))
        tmp2 = Table(fitsio.read(os.path.join('/dvs_ro/cfs/cdirs/desi/users/rongpu/targets/dr9.0/1.1.1/resolve/dr9_lrg_1.1.1_lrgmask_v1.1.fits.gz')))
        lrgmask = hstack([tmp1, tmp2])
        lrgmask = lrgmask[lrgmask['lrg_mask']==0]
        mask = np.in1d(cat['TARGETID'], lrgmask['TARGETID'])
        print('Mask', np.sum(~mask), np.sum(mask), np.sum(~mask)/len(mask))
        cat = cat[mask]
    elif tracer in ['ELG', 'ELG_LOP', 'ELG_VLO']:
        tmp1 = Table(fitsio.read(os.path.join('/dvs_ro/cfs/cdirs/desi/users/rongpu/targets/dr9.0/1.1.1/resolve/dr9_elg_1.1.1_basic.fits'), columns=['TARGETID']))
        tmp2 = Table(fitsio.read(os.path.join('/dvs_ro/cfs/cdirs/desi/users/rongpu/targets/dr9.0/1.1.1/resolve/dr9_elg_1.1.1_elgmask_v1.fits.gz')))
        elgmask = hstack([tmp1, tmp2])
        elgmask = elgmask[elgmask['elg_mask']==0]
        mask = np.in1d(cat['TARGETID'], elgmask['TARGETID'])
        print('Mask', np.sum(~mask), np.sum(mask), np.sum(~mask)/len(mask))
        cat = cat[mask]

    if tracer=='QSO':
        mask = cat['PRIORITY']==3400
        print('Remove QSO reobservations', np.sum(~mask), np.sum(mask), np.sum(~mask)/len(mask))
        cat = cat[mask]
        z_col = 'Z_QSO'
    else:
        z_col = 'Z'

    print(tracer, len(cat))

    good_z_col = 'GOOD_Z_' + tracer.split('_')[0]
    print(tracer, 'average failure rate', np.sum(~cat[good_z_col])/len(cat))

    stats = Table()
    stats['FIBER'], stats[tracer.lower()+'_n_tot'] = np.unique(cat['FIBER'], return_counts=True)
    stats.sort(tracer.lower()+'_n_tot')
    tt = Table()
    tt['FIBER'], tt[tracer.lower()+'_n_fail'] = np.unique(cat['FIBER'][~cat[good_z_col]], return_counts=True)
    stats = join(stats, tt, keys='FIBER', join_type='outer').filled(0)
    stats[tracer.lower()+'_frac_fail'] = stats[tracer.lower()+'_n_fail']/stats[tracer.lower()+'_n_tot']
    error_floor = True
    n, p = stats[tracer.lower()+'_n_tot'].copy(), stats[tracer.lower()+'_frac_fail'].copy()
    if error_floor:
        p1 = np.maximum(p, 1/n)  # error floor
    else:
        p1 = p
    stats[tracer.lower()+'_frac_fail_err'] = np.clip(np.sqrt(n * p * (1-p))/n, np.sqrt(n * p1 * (1-p1))/n, 1)

    if ks_test:

        for apply_good_z_cut in [True, False]:

            outliers = []

            for ii in range(3):  # 3 iterations
                print('iteration', ii+1)
                mask = ~np.in1d(cat['FIBER'], outliers)
                if apply_good_z_cut:
                    mask &= cat[good_z_col]
                allz = np.sort(np.array(cat[z_col][mask]))
                x = allz.copy()
                y = np.linspace(0, 1, len(x))
                cdf = interp1d(x, y, fill_value=(0, 1), bounds_error=False)

                pvalues = np.zeros(len(stats))
                for index, fiber in enumerate(stats['FIBER']):
                    mask = cat['FIBER']==stats['FIBER'][index]
                    if apply_good_z_cut:
                        mask &= cat[good_z_col]
                        if np.sum(mask)==0:
                            pvalues[index] = -99
                            continue
                    pvalues[index] = kstest(cat[z_col][mask], cdf).pvalue

                mask_outlier = (pvalues<pvalue_threshold) & (pvalues!=-99)
                outliers = np.array(np.sort(stats['FIBER'][mask_outlier]))
                print('{} outlier fibers:'.format(len(outliers)), list(outliers))

            if apply_good_z_cut:
                stats[tracer.lower()+'_ks_pvalue_goodz'] = pvalues
            else:
                stats[tracer.lower()+'_ks_pvalue_allz'] = pvalues

    stats = join(stats_zcat, stats, keys='FIBER', join_type='outer').filled(-99)
    print()

stats.sort('FIBER')
stats.write(output_fn, overwrite=False)

print('Done!', time.strftime('%H:%M:%S', time.gmtime(time.time() - time_start)))
