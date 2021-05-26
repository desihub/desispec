'''
Generate Master TSNR ensemble DFLUX files.  See doc. 4723.  Note: in this
instance, ensemble avg. of flux is written, in order to efficiently generate
tile depths.
'''
import os
import sys
import argparse
import numpy as np
from pkg_resources import resource_filename

import astropy.io.fits as fits
from astropy.table import Table, join

import matplotlib.pyplot as plt

from desiutil.log import get_logger
from desispec.tsnr import template_ensemble

def parse(options=None):
    parser = argparse.ArgumentParser(description="Generate a sim. template ensemble stack of given type and write it to disk at --outdir.")
    parser.add_argument('-i','--infile', type = str, required=True,
                        help='tsnr-ensemble fits filename')
    parser.add_argument('--tsnr-table-filename', type=str, required=True,
                        help='TSNR afterburner file, with TSNR2_TRACER.')
    parser.add_argument('--plot', action='store_true',
                        help='plot the fit.')
    parser.add_argument('--exclude', type=str, default=None, required=False,
                        help='comma separated list of expids to ignore.')    
    parser.add_argument('-o','--outfile', type = str, required=False,
                        help='tsnr-ensemble fits output')

    args = None

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args

def tsnr_efftime(exposures_table_filename, tsnr_table_filename, tracer, plot=True, exclude=None):
    '''
    Given an external calibration, e.g.
    /global/cfs/cdirs/desi/survey/observations/SV1/sv1-exposures.fits

    with e.g. EFFTIME_DARK and

    a tsnr afterburner run, e.g.
    /global/cfs/cdirs/desi/spectro/redux/cascades/tsnr-cascades.fits

    Compute linear coefficient to convert TSNR2_TRACER_BRZ to EFFTIME_DARK
    or EFFTIME_BRIGHT.
    '''

    log = get_logger()

    tsnr_col  = 'TSNR2_{}'.format(tracer.upper())

    ext_calib = Table.read(exposures_table_filename)
    
    # Quality cuts.
    ext_calib = ext_calib[(ext_calib['EXPTIME'] > 0.)]

    log.info('Calibrating tracer {}'.format(tracer))
    
    if tracer in ['bgs', 'mws' , 'gpbbright']:
        ext_col   = 'EFFTIME_BRIGHT'

        # Expected BGS exposure is 180s nominal.
        #ext_calib = ext_calib[(ext_calib['EFFTIME_BRIGHT'] > 120.)]
        ext_calib = ext_calib[(ext_calib['TARGETS']=='BGS+MWS')]

    elif tracer in ['elg', 'lrg', 'qso', 'lya', 'gpbdark']:
        ext_col   = 'EFFTIME_DARK'

        # Expected dark exposure is 900s nominal.
        ext_calib = ext_calib[(ext_calib['TARGETS']=='ELG')
                              |(ext_calib['TARGETS']=='QSO+ELG')
                              |(ext_calib['TARGETS']=='QSO+LRG')]
    else:
        log.critical('External calibration for tracer {} is not defined.'.format(tracer))
        
    tsnr_run  = Table.read(tsnr_table_filename)

    # TSNR == 0.0 if exposure was not successfully reduced.
    tsnr_run  = tsnr_run[tsnr_run[tsnr_col] > 0.0]

    # External excluded exposures.
    exclude   = np.array(exclude)

    if exclude is not None:
        log.info('Excluding exposure list of: {}'.format(exclude))
    
    tsnr_run  = tsnr_run[~np.isin(tsnr_run['EXPID'], exclude)]
    
    # Keep common exposures.
    ref_extexpids = ext_calib['EXPID']
    
    ext_calib = ext_calib[np.isin(ext_calib['EXPID'], tsnr_run['EXPID'])]
    tsnr_run  = tsnr_run[np.isin(tsnr_run['EXPID'], ext_calib['EXPID'])]

    ref_frac  = np.mean(np.isin(ref_extexpids, ext_calib['EXPID']))
    
    log.info('Found {:.6f} of ref. exposures in provided tsnr'.format(ref_frac))

    if ref_frac < 0.7:
        log.critical('Provided TSNR is likely missing exposures required to calibrate (!)')
    
    tsnr_run  = join(tsnr_run, ext_calib['EXPID', ext_col], join_type='left', keys='EXPID')
    with_reference_tsnr = (tsnr_col in ext_calib.dtype.names)
    if with_reference_tsnr :
        tsnr_run[tsnr_col+"_REF"] = ext_calib[tsnr_col]
    else :
        log.warning("no {} column in ref, cannot calibrate it".format(tsnr_col))

    tsnr_run.sort(ext_col)

    tsnr_run.pprint()

    log.info('Calibrating {} [{}, {}] against {} [{}, {}].'.format(tsnr_col, tsnr_run[tsnr_col].min(), tsnr_run[tsnr_col].max(), ext_col, tsnr_run[ext_col].min(), tsnr_run[ext_col].max()))
    
    slope_efftime  = np.sum(tsnr_run[ext_col] * tsnr_run[tsnr_col]) / np.sum(tsnr_run[tsnr_col]**2.)

    if with_reference_tsnr :
        slope_tsnr2    = np.sum(tsnr_run[tsnr_col+"_REF"] * tsnr_run[tsnr_col]) / np.sum(tsnr_run[tsnr_col]**2.)
    else :
        slope_tsnr2    = 1.

    if plot:       
        plt.figure("efftime-vs-tsnr2-{}".format(tracer))
        plt.plot(tsnr_run[tsnr_col], tsnr_run[ext_col], c='k', marker='.', lw=0.0, markersize=1)
        plt.plot(tsnr_run[tsnr_col], slope_efftime*tsnr_run[tsnr_col], c='k', lw=0.5)
        plt.title('{} = {:.3f} x {} (ref. frac. {:.2f})'.format(ext_col, slope_efftime, tsnr_col, ref_frac))
        plt.xlabel("new "+tsnr_col)
        plt.ylabel("SV1 reference "+ext_col)
        plt.grid()
        
        plt.figure("efftime-vs-tsnr2-{}-ratio".format(tracer))
        plt.plot(slope_efftime*tsnr_run[tsnr_col], tsnr_run[ext_col] / (slope_efftime*tsnr_run[tsnr_col]), c='k', lw=0.0, marker='.', markersize=2)
        plt.axhline(1.1, c='c', alpha=0.75, lw=0.5)
        plt.axhline(0.9, c='c', alpha=0.75, lw=0.5)
        plt.axhline(1.2, c='b', alpha=0.50, lw=0.5)
        plt.axhline(0.8, c='b', alpha=0.50, lw=0.5)
        plt.title('{} = {:.3f} x {} (ref. frac. {:.2f})'.format(ext_col, slope_efftime, tsnr_col, ref_frac))
        plt.xlabel("efftime "+tsnr_col)
        plt.ylabel("SV1 reference "+ext_col+"/"+"new efftime "+tsnr_col)
        plt.ylim(0.0, 2.5)
        plt.grid()
        
        if with_reference_tsnr :
            plt.figure("tsnr2-vs-tsnr2-{}".format(tracer))
            plt.plot(tsnr_run[tsnr_col], tsnr_run[tsnr_col+"_REF"], c='k', marker='.', lw=0.0, markersize=1)
            plt.plot(tsnr_run[tsnr_col], slope_tsnr2*tsnr_run[tsnr_col], c='k', lw=0.5)
            plt.title('{} = {:.3f} x {} (ref. frac. {:.2f})'.format(tsnr_col+"_REF", slope_tsnr2, tsnr_col, ref_frac))
            plt.xlabel("new "+tsnr_col)
            plt.ylabel("SV1 reference "+tsnr_col)
            plt.grid()
        
        plt.show()

    return  slope_efftime , slope_tsnr2



def main(args):
    log = get_logger()

    effective_time_calibration_table_filename = resource_filename('desispec', 'data/tsnr/sv1-exposures.csv')


    ens = fits.open(args.infile)
    hdr = ens[0].header

    tracer = hdr["TRACER"].strip().lower()
    log.info("tracer = {}".format(tracer))

    if args.exclude is not None:
        args.exclude = [np.int(x) for x in args.exclude.split(',')]
    
    slope_efftime,slope_tsnr2 = tsnr_efftime(exposures_table_filename=effective_time_calibration_table_filename, tsnr_table_filename=args.tsnr_table_filename, tracer=tracer,plot=args.plot, exclude=args.exclude)

    # TSNR2_REF = slope_tsnr2 * TSNR2_CURRENT
    # so I have to apply to delta_flux a scale:
    flux_scale = np.sqrt(slope_tsnr2)
    # EFFTIME_REF = slope_efftime * TSNR2_CURRENT = slope_efftime/slope_tsnr2 * TSNR2_REF
    slope_efftime_calib = slope_efftime/slope_tsnr2

    if 'FLUXSCAL' in hdr :
        # need to account for previous flux scale if exists because used to compute the TSNR values
        old_flux_scale = hdr['FLUXSCAL']
        new_flux_scale = old_flux_scale * flux_scale
    else :
        new_flux_scale = flux_scale

    if args.outfile :
        log.info('appending SNR2TIME coefficient of {:.6f} to {}'.format(slope_efftime, args.infile))
        hdr['FLUXSCAL'] = ( new_flux_scale , "flux scale factor")
        hdr['SNR2TIME'] = ( slope_efftime_calib , "eff. time factor")
        hdr['TIMEFILE']    = os.path.basename(effective_time_calibration_table_filename)
        hdr['TSNRFILE']    = os.path.basename(args.tsnr_table_filename)
        ens.writeto(args.outfile, overwrite=True)
        log.info("wrote {}".format(args.outfile))
    else :
        log.info('fitted slope efftime vs tsnr2 = {:.6f}'.format(slope_efftime))
        log.info('fitted slope tsnr2(ref) vs tsnr2(current) = {:.6f}'.format(slope_tsnr2))
        log.warning("the calibration has not been saved (use option -o to write the result)")

if __name__ == '__main__':
    print("please run desi_calibrate_tsnr_ensemble")
