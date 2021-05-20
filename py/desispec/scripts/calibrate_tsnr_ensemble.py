'''
Generate Master TSNR ensemble DFLUX files.  See doc. 4723.  Note: in this
instance, ensemble avg. of flux is written, in order to efficiently generate
tile depths.

Currently assumes redshift and mag ranges derived from FDR, but uniform in both.
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
    parser.add_argument('--dry-run', action='store_true',
                        help='do not save the result in the file')

    args = None

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args

def tsnr_efftime(exposures_table_filename, tsnr_table_filename, tracer, plot=True):
    '''
    Given an external calibration, e.g.
    /global/cfs/cdirs/desi/survey/observations/SV1/sv1-exposures.fits

    with e.g. EFFTIME_DARK and

    a tsnr afterburner run, e.g.
    /global/cfs/cdirs/desi/spectro/redux/cascades/tsnr-cascades.fits

    Compute linear coefficient to convert TSNR2_TRACER_BRZ to EFFTIME_DARK
    or EFFTIME_BRIGHT.
    '''

    tsnr_col  = 'TSNR2_{}'.format(tracer.upper())

    ext_calib = Table.read(exposures_table_filename)

    # Quality cuts.
    ext_calib = ext_calib[(ext_calib['EXPTIME'] > 60.)]

    if tracer in ['bgs', 'mws']:
        ext_col   = 'EFFTIME_BRIGHT'

        # Expected BGS exposure is 180s nominal.
        ext_calib = ext_calib[(ext_calib['EFFTIME_BRIGHT'] > 120.)]

    else:
        ext_col   = 'EFFTIME_DARK'

        # Expected BGS exposure is 900s nominal.
        ext_calib = ext_calib[(ext_calib['EFFTIME_DARK'] > 450.)]

    tsnr_run  = Table.read(tsnr_table_filename)

    # TSNR == 0.0 if exposure was not successfully reduced.
    tsnr_run  = tsnr_run[tsnr_run[tsnr_col] > 0.0]

    # Keep common exposures.
    ext_calib = ext_calib[np.isin(ext_calib['EXPID'], tsnr_run['EXPID'])]
    tsnr_run  = tsnr_run[np.isin(tsnr_run['EXPID'], ext_calib['EXPID'])]

    tsnr_run  = join(tsnr_run, ext_calib['EXPID', ext_col], join_type='left', keys='EXPID')
    tsnr_run.sort(ext_col)

    tsnr_run.pprint()

    # from   scipy  import stats
    # res       = stats.linregress(tsnr_run[ext_col], tsnr_run[tsnr_col])
    # slope     = res.slope
    # intercept = res.intercept

    slope     = np.sum(tsnr_run[ext_col] * tsnr_run[tsnr_col]) / np.sum(tsnr_run[tsnr_col]**2.)

    if plot:
        plt.figure("efftime-vs-tsnr-{}".format(tracer))
        plt.plot(tsnr_run[tsnr_col], tsnr_run[ext_col], c='k', marker='.', lw=0.0, markersize=1)
        plt.plot(tsnr_run[tsnr_col], slope*tsnr_run[tsnr_col], c='k', lw=0.5)
        plt.title('{} = {:.3f} x {}'.format(ext_col, slope, tsnr_col))
        plt.xlabel(tsnr_col)
        plt.ylabel(ext_col)
        plt.grid()
        plt.show()

    return  slope



def main(args):
    log = get_logger()

    effective_time_calibration_table_filename = resource_filename('desispec', 'data/tsnr/sv1-exposures.csv')


    ens = fits.open(args.infile)
    hdr = ens[0].header

    tracer = hdr["TRACER"].strip().lower()
    log.info("tracer = {}".format(tracer))

    slope = tsnr_efftime(exposures_table_filename=effective_time_calibration_table_filename, tsnr_table_filename=args.tsnr_table_filename, tracer=tracer,plot=args.plot)

    if not args.dry_run :
        log.info('appending TSNR2TOEFFTIME coefficient of {:.6f} to {}'.format(slope, args.infile))

        hdr['TSNR2TOEFFTIME'] = slope
        hdr['EFFTIMEFILE']    = os.path.basename(effective_time_calibration_table_filename)
        hdr['TSNRRUNFILE']    = os.path.basename(args.tsnr_table_filename)
        ens.writeto(args.infile, overwrite=True)
        log.info("wrote {}".format(args.infile))
    else :
        log.info('fitted slope = {:.6f}'.format(slope))
        log.warning("fid not overwrite the file {} (because of option --dry-run)".format(args.infile))

if __name__ == '__main__':
    print("please run desi_calibrate_tsnr_ensemble")
