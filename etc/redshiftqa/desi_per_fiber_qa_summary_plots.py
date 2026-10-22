#!/usr/bin/env python

# This is an step 2 of the redshift QA generation procedure
# Make per-tracer summary plots:
#   redshift vs fiber (2D density), failure rate vs fiber, focal plane failure rate, and number of objects vs fiber
# The PNG files are saved in $QA_DIR/png/summary/

# Example usage:
# export ZCAT_MAIN_DIR=/dvs_ro/cfs/cdirs/desi/spectro/redux/nevis/zcatalog/v2/main
# export QA_DIR=/global/cfs/cdirs/desicollab/users/rongpu/redshift_qa/nevis
# srun -N 1 -C cpu -t 04:00:00 -q interactive ./desi_per_fiber_qa_summary_plots.py -i $ZCAT_MAIN_DIR --stats $QA_DIR/per_fiber_qa_stats.fits -o $QA_DIR
# 
# To make the plots for selected tracers only, e.g., add: --tracer LRG QSO

from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack, join
import fitsio
# from astropy.io import fits

from desitarget.targetmask import desi_mask, bgs_mask
from desimodel.io import load_focalplane

import argparse

import time


time_start = time.time()

all_tracers = ['LRG', 'LGE', 'ELG_LOP', 'ELG_VLO', 'QSO', 'BGS_BRIGHT', 'BGS_FAINT']
pvalue_threshold = 1e-4
min_nobs = 50  # do not plot fibers with fewer objects than this
params = {'figure.facecolor': 'w'}
plt.rcParams.update(params)

# Redshift bin size and maximum redshift of the redshift vs fiber plots
z_bin_size_dict = {'LRG': 0.025, 'LGE': 0.025, 'ELG_LOP': 0.025, 'ELG_VLO': 0.025, 'QSO': 0.05, 'BGS_BRIGHT': 0.01, 'BGS_FAINT': 0.01}
z_max_dict = {'LRG': 1.5, 'LGE': 1.5, 'ELG_LOP': 1.8, 'ELG_VLO': 1.8, 'QSO': 4.5, 'BGS_BRIGHT': 0.7, 'BGS_FAINT': 0.7}
# Failure rate range of the failure rate vs fiber plot and the focal plane plot
frac_fail_range_dict = {'LRG': (0., 0.1), 'LGE': (0., 0.1), 'ELG_LOP': (0.2, 0.5), 'ELG_VLO': (0., 0.2), 'QSO': (0.2, 0.5),
                        'BGS_BRIGHT': (0., 0.1), 'BGS_FAINT': (0., 0.1)}


def make_plots(tracer, stats, indir, png_dir, fp):
    '''
    Make the summary plots of a tracer.

    Args:
        tracer: tracer name, e.g., LRG (str)
        stats: per-fiber stats table
        indir: input zcatalog directory (str)
        png_dir: output directory of the PNG files (str)
        fp: table of fiber positions with columns FIBER, FIBERASSIGN_X, and FIBERASSIGN_Y
    '''

    print(tracer)

    # Fibers with p-values below the threshold; ELG_LOP (all z) is not used because of its large number of outliers
    mask_outlier = stats[tracer.lower()+'_ks_pvalue_goodz']<pvalue_threshold
    if tracer!='ELG_LOP':
        mask_outlier |= (stats[tracer.lower()+'_ks_pvalue_allz']<pvalue_threshold)
    mask_outlier &= (stats[tracer.lower()+'_n_tot']>=min_nobs)
    outliers = np.array(stats['FIBER'][mask_outlier])
    print('{} outlier fibers:'.format(len(outliers)), list(outliers))

    ################################## Plots made with the per-fiber stats ##################################

    # Number of objects vs fiber
    fig, ax = plt.subplots(10, 1, figsize=(16, 20))
    for index in range(10):
        fiber_min, fiber_max = index*500-0.5, (index+1)*500-0.5
        mask = (stats['FIBER']>fiber_min) & (stats['FIBER']<fiber_max)
        mask &= stats[tracer.lower()+'_n_tot']>0
        n_good = stats[tracer.lower()+'_n_tot'][mask] - stats[tracer.lower()+'_n_fail'][mask]
        ax[index].bar(stats['FIBER'][mask], stats[tracer.lower()+'_n_tot'][mask], width=1., label='all z')
        ax[index].bar(stats['FIBER'][mask], n_good, width=1., label='good z')
        if index==0:
            ax[index].legend(loc='upper right')
        ax[index].set_xlim(fiber_min-3, fiber_max+3)
        ax[index].annotate('PETAL_LOC {}'.format(index), (fiber_min+3, 0.85), xycoords=('data', 'axes fraction'), fontsize='large')
        ax[index].set_ylabel('{} N'.format(tracer))
    plt.tight_layout()
    plt.savefig(os.path.join(png_dir, '{}_n_vs_fiber.png'.format(tracer.lower())))
    plt.close()

    # Failure rate vs fiber
    ymin, ymax = frac_fail_range_dict[tracer]
    fig, ax = plt.subplots(10, 1, figsize=(16, 20))
    for index in range(10):
        fiber_min, fiber_max = index*500-0.5, (index+1)*500-0.5
        mask = (stats['FIBER']>fiber_min) & (stats['FIBER']<fiber_max)
        mask &= (stats[tracer.lower()+'_n_tot']>=min_nobs)
        mask_bad = mask & np.isin(stats['FIBER'], outliers)
        mask_good = mask & (~mask_bad)
        mask_off_scale = mask & (stats[tracer.lower()+'_frac_fail']>ymax)
        for mask1, color in [(mask_good, 'C0'), (mask_bad, 'C3')]:
            frac_fail = stats[tracer.lower()+'_frac_fail'][mask1]
            frac_fail_err = stats[tracer.lower()+'_frac_fail_err'][mask1]
            ax[index].errorbar(stats['FIBER'][mask1], frac_fail, yerr=(np.clip(frac_fail_err, None, frac_fail), frac_fail_err),
                               color=color, fmt='.', ms=3, elinewidth=1)
        for bad_index in np.where(mask_off_scale)[0]:
            ax[index].arrow(stats['FIBER'][bad_index], ymax-(ymax-ymin)*0.22, 0, (ymax-ymin)/10, head_width=3, head_length=(ymax-ymin)/10, fc='C3', ec='C3')
        ax[index].grid(alpha=0.5)
        ax[index].set_ylim(ymin-(ymax-ymin)*0.03, ymax)
        ax[index].set_xlim(fiber_min-3, fiber_max+3)
        ax[index].annotate('PETAL_LOC {}'.format(index), (fiber_min+3, 0.85), xycoords=('data', 'axes fraction'), fontsize='large')
        ax[index].set_ylabel('{} failure rate'.format(tracer))
    plt.tight_layout()
    plt.savefig(os.path.join(png_dir, '{}_failure_rate_vs_fiber.png'.format(tracer.lower())))
    plt.close()

    # Focal plane failure rate
    fp_stats = join(stats, fp, keys='FIBER', join_type='inner')
    mask = fp_stats[tracer.lower()+'_n_tot']>=min_nobs
    plt.figure(figsize=(12, 11.5))
    plt.plot(fp_stats['FIBERASSIGN_X'][~mask], fp_stats['FIBERASSIGN_Y'][~mask], 'x', color='0.7', ms=3)
    plt.scatter(fp_stats['FIBERASSIGN_X'][mask], fp_stats['FIBERASSIGN_Y'][mask], c=1-fp_stats[tracer.lower()+'_frac_fail'][mask],
                s=45, vmin=1-ymax, vmax=1-ymin, cmap='viridis')
    plt.axis([-420, 420, -420, 420])
    plt.gca().set_aspect('equal')
    plt.xlabel('FIBERASSIGN_X (mm)')
    plt.ylabel('FIBERASSIGN_Y (mm)')
    plt.title('{} redshift success rate (fibers with N<{} not shown)'.format(tracer, min_nobs))
    plt.colorbar(fraction=0.04, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(png_dir, '{}_focal_plane_success_rate.png'.format(tracer.lower())))
    plt.close()

    print(tracer, 'stats plots done!', time.strftime('%H:%M:%S', time.gmtime(time.time() - time_start)))

    ################################## Plots made with the redshift catalog ##################################

    if tracer in ['LRG', 'LGE', 'ELG', 'QSO', 'ELG_LOP', 'ELG_VLO', 'BGS_ANY']:
        fn = os.path.join(indir, 'ztile-main-dark-cumulative.fits')
        fn1 = os.path.join(indir, 'ztile-main-dark-cumulative-extra.fits')
        cat = Table(fitsio.read(fn, columns=['DESI_TARGET', 'FIBER', 'COADD_FIBERSTATUS', 'EFFTIME_SPEC', 'TARGETID', 'PRIORITY']))
        cat1 = Table(fitsio.read(fn1, columns=['Z', 'ZWARN', 'Z_QSO', 'GOOD_Z_LRG', 'GOOD_Z_ELG', 'GOOD_Z_QSO']))
        cat = hstack([cat, cat1], join_type='exact')
        mask = np.where(cat['DESI_TARGET'] & desi_mask[tracer] > 0)[0]
        cat = cat[mask]
    else:
        fn = os.path.join(indir, 'ztile-main-bright-cumulative.fits')
        fn1 = os.path.join(indir, 'ztile-main-bright-cumulative-extra.fits')
        cat = Table(fitsio.read(fn, columns=['BGS_TARGET', 'FIBER', 'COADD_FIBERSTATUS', 'EFFTIME_SPEC', 'TARGETID', 'PRIORITY']))
        cat1 = Table(fitsio.read(fn1, columns=['Z', 'ZWARN', 'GOOD_Z_BGS']))
        cat = hstack([cat, cat1], join_type='exact')
        mask = np.where(cat['BGS_TARGET'] & bgs_mask[tracer] > 0)[0]
        cat = cat[mask]

    print(tracer, 'zcat', len(cat))

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

    # Apply masks (no imaging mask is available for LGE)
    if tracer=='LRG':
        lrgmask_fn = os.path.expandvars('$DESI_ROOT_READONLY/survey/catalogs/DA2/main/LSS/LRGtargetsDR9v1.1.1_lrgimask.fits')
        lrgmask = Table(fitsio.read(lrgmask_fn))
        lrgmask = lrgmask[lrgmask['lrg_mask']==0]
        mask = np.isin(cat['TARGETID'], lrgmask['TARGETID'])
        print('Mask', np.sum(~mask), np.sum(mask), np.sum(~mask)/len(mask))
        cat = cat[mask]
    elif tracer in ['ELG', 'ELG_LOP', 'ELG_VLO']:
        elgmask_fn = os.path.expandvars('$DESI_ROOT_READONLY/survey/catalogs/DA2/main/LSS/ELGtargetsDR9v1.1.1_elgimask.fits')
        elgmask = Table(fitsio.read(elgmask_fn))
        elgmask = elgmask[elgmask['elg_mask']==0]
        mask = np.isin(cat['TARGETID'], elgmask['TARGETID'])
        print('Mask', np.sum(~mask), np.sum(mask), np.sum(~mask)/len(mask))
        cat = cat[mask]

    # Remove QSO reobservations
    if tracer=='QSO':
        mask = cat['PRIORITY']==3400
        print('Remove QSO reobservations', np.sum(~mask), np.sum(mask), np.sum(~mask)/len(mask))
        cat = cat[mask]
        z_col = 'Z_QSO'
    else:
        z_col = 'Z'

    if tracer=='LGE':
        good_z_col = 'GOOD_Z_LRG'  # LGE and LRG share the same redshift quality cuts
    else:
        good_z_col = 'GOOD_Z_' + tracer.split('_')[0]
    print(tracer, 'average failure rate', np.sum(~cat[good_z_col])/len(cat))

    z_bin_size = z_bin_size_dict[tracer]
    z_bins = np.arange(-z_bin_size/2, z_max_dict[tracer], z_bin_size)

    # Redshift vs fiber
    for apply_good_z_cut in [True, False]:

        if apply_good_z_cut:
            cat1 = cat[cat[good_z_col]]
            suffix = 'goodz'
        else:
            cat1 = cat
            suffix = 'allz'

        # Weight each object by the inverse of the number of objects in the fiber
        # so that the fibers are normalized to the median number of objects per fiber
        fibers, n_tot = np.unique(cat1['FIBER'], return_counts=True)
        n_median = np.median(n_tot[n_tot>=min_nobs])
        weight_per_fiber = np.zeros(5000)
        weight_per_fiber[fibers] = n_median/n_tot
        weight = weight_per_fiber[cat1['FIBER']]
        n_tot_per_fiber = np.zeros(5000, dtype=int)
        n_tot_per_fiber[fibers] = n_tot
        mask_enough = n_tot_per_fiber[cat1['FIBER']]>=min_nobs

        for normalize in [False, True]:

            # Compute the 2D histograms first to get a common color scale for the panels
            hists = []
            for index in range(10):
                fiber_min, fiber_max = index*500-0.5, (index+1)*500-0.5
                mask = (cat1['FIBER']>fiber_min) & (cat1['FIBER']<fiber_max)
                fiber_bins = np.linspace(fiber_min, fiber_max, 501)
                if normalize:
                    mask &= mask_enough  # do not plot fibers with too few objects
                    hist = np.histogram2d(cat1['FIBER'][mask], cat1[z_col][mask], weights=weight[mask], bins=[fiber_bins, z_bins])[0]
                else:
                    hist = np.histogram2d(cat1['FIBER'][mask], cat1[z_col][mask], bins=[fiber_bins, z_bins])[0]
                hists.append(hist)
            vmax = np.percentile(np.concatenate([hist[hist>0] for hist in hists]), 99.5)

            fig, ax = plt.subplots(10, 1, figsize=(16, 20))
            for index in range(10):
                fiber_min, fiber_max = index*500-0.5, (index+1)*500-0.5
                fiber_bins = np.linspace(fiber_min, fiber_max, 501)
                ax[index].pcolormesh(fiber_bins, z_bins, hists[index].T, vmin=0, vmax=vmax, rasterized=True)
                ax[index].set_xlim(fiber_min, fiber_max)
                ax[index].annotate('PETAL_LOC {}'.format(index), (fiber_min+3, 0.85), xycoords=('data', 'axes fraction'), fontsize='large', color='w')
                ax[index].set_ylabel('{} redshift'.format(tracer))
            plt.tight_layout()
            if normalize:
                output_fn = os.path.join(png_dir, '{}_z_vs_fiber_{}_norm.png'.format(tracer.lower(), suffix))
            else:
                output_fn = os.path.join(png_dir, '{}_z_vs_fiber_{}.png'.format(tracer.lower(), suffix))
            plt.savefig(output_fn)
            plt.close()

    del cat, cat1
    gc.collect()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indir', type=str, help="input directory path", required=True)
    parser.add_argument('-o', '--output', type=str, help="output directory path", required=True)
    parser.add_argument('--tracer', type=str, nargs='+', help="tracer type(s), e.g., LRG; default: all tracers", default=all_tracers)
    parser.add_argument('--stats', type=str, help="per-fiber stats path", required=True)
    args = parser.parse_args()

    stats = Table(fitsio.read(args.stats))
    stats.sort('FIBER')

    png_dir = os.path.join(args.output, 'png', 'summary')
    os.makedirs(png_dir, exist_ok=True)

    # Fiber positions of the neutral focal plane
    fp = load_focalplane()[0]
    fp = fp[fp['DEVICE_TYPE']=='POS']
    fp = Table(fp['FIBER', 'OFFSET_X', 'OFFSET_Y'])
    fp.rename_column('OFFSET_X', 'FIBERASSIGN_X')
    fp.rename_column('OFFSET_Y', 'FIBERASSIGN_Y')

    for tracer in args.tracer:
        make_plots(tracer, stats, args.indir, png_dir, fp)

    print('Done!', time.strftime('%H:%M:%S', time.gmtime(time.time() - time_start)))


if __name__ == '__main__':
    main()
