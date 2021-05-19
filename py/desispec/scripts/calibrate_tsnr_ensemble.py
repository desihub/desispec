'''
Generate Master TSNR ensemble DFLUX files.  See doc. 4723.  Note: in this
instance, ensemble avg. of flux is written, in order to efficiently generate
tile depths.

Currently assumes redshift and mag ranges derived from FDR, but uniform in both.
'''
import os
import sys
import yaml
import desiutil
import fitsio
import desisim
import argparse
import os.path                       as     path
import numpy                         as     np
import astropy.io.fits               as     fits
import matplotlib.pyplot             as     plt

from   desiutil                      import depend
from   astropy.convolution           import convolve, Box1DKernel
from   pathlib                       import Path
from   desiutil.dust                 import mwdust_transmission
from   desiutil.log                  import get_logger
from   pkg_resources                 import resource_filename
from   scipy.interpolate             import interp1d
from   astropy.table                 import Table, join


from tsnr import template_ensemble

np.random.seed(seed=314)

# AR/DK DESI spectra wavelengths
# TODO:  where are brz extraction wavelengths defined?  https://github.com/desihub/desispec/issues/1006.
wmin, wmax, wdelta = 3600, 9824, 0.8
wave               = np.round(np.arange(wmin, wmax + wdelta, wdelta), 1)
cslice             = {"b": slice(0, 2751), "r": slice(2700, 5026), "z": slice(4900, 7781)}

def parse(options=None):
    parser = argparse.ArgumentParser(description="Generate a sim. template ensemble stack of given type and write it to disk at --outdir.")
    parser.add_argument('-i','--infile', type = str, required=True,
                        help='tsnr-ensemble fits filename')
    parser.add_argument('--tsnr-table-filename', type=str, required=True,
                        help='TSNR afterburner file, with TSNR2_TRACER.')

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
        plt.plot(tsnr_run[ext_col], tsnr_run[tsnr_col], c='k', marker='.', lw=0.0, markersize=1)
        plt.plot(tsnr_run[ext_col], intercept + slope*tsnr_run[ext_col], c='k', lw=0.5)
        plt.title('{} = {:.3f} x {} + {:.3f}'.format(tsnr_col, slope, ext_col, intercept))
        plt.xlabel(ext_col)
        plt.ylabel(tsnr_col)
        plt.show()

    return  slope



def main():
    log = get_logger()

    args = parse()

    effective_time_calibration_table_filename = resource_filename('desispec', 'data/tsnr/sv1-exposures.csv')

    slope = tsnr_efftime(effective_time_calibration_table_filename, args.tsnr_table_filename)

    log.info('Appending TSNR2TOEFFTIME coefficient of {:.6f} to {}'.format(slope, args.infile))

    ens = fits.open(args.infile)
    hdr = ens[0].header

    hdr['TSNR2TOEFFTIME'] = slope
    hdr['EFFTIMEFILE']    = args.external_calib.replace('/global/cfs/cdirs/desi/survey', '$DESISURVEYOPS')
    hdr['TSNRRUNFILE']    = args.tsnr_run.replace('/global/cfs/cdirs/desi/spectro/redux',        '$REDUX')

    depend.setdep(hdr, 'desisim',  desisim.__version__)
    depend.setdep(hdr, 'desiutil', desiutil.__version__)

    ens.writeto(args.infile, overwrite=True)

    log.info('Done.')

if __name__ == '__main__':
    main()
