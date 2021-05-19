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

np.random.seed(seed=314)

# AR/DK DESI spectra wavelengths
# TODO:  where are brz extraction wavelengths defined?  https://github.com/desihub/desispec/issues/1006.
wmin, wmax, wdelta = 3600, 9824, 0.8
wave               = np.round(np.arange(wmin, wmax + wdelta, wdelta), 1)
cslice             = {"b": slice(0, 2751), "r": slice(2700, 5026), "z": slice(4900, 7781)}

def parse(options=None):
    parser = argparse.ArgumentParser(description="Generate a sim. template ensemble stack of given type and write it to disk at --outdir.")
    parser.add_argument('--nmodel', type = int, default = 2000, required=False,
                        help='Number of galaxies in the ensemble.')
    parser.add_argument('--tracer', type = str, default = 'bgs', required=True,
                        help='Tracer to generate of [bgs, lrg, elg, qso].')
    parser.add_argument('--configdir', type = str, default = None, required=False,
                        help='Directory to config files if not desispec repo.')
    parser.add_argument('--smooth', type=float, default=100., required=False,
                        help='Smoothing scale [A] for DFLUX calc.')
    parser.add_argument('--Nz', action='store_true',
                        help = 'Apply tracer Nz weighting in stacking of ensemble.')
    parser.add_argument('--outdir', type = str, default = 'bgs', required=True,
			help='Directory to write to.')
    args = None

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args


class Config(object):
    def __init__(self, cpath):
        with open(cpath) as f:
            d = yaml.load(f, Loader=yaml.FullLoader)

        for key in d:
            setattr(self, key, d[key])


def main():
    log = get_logger()

    args = parse()

    rads = template_ensemble(args.outdir, tracer=args.tracer, nmodel=args.nmodel, log=log, configdir=args.configdir, Nz=args.Nz, smooth=args.smooth)

    effective_time_calibration_table_filename = resource_filename('desispec', 'data/tsnr/sv1-exposures.csv')

    slope = tsnr_efftime(effective_time_calibration_table_filename, args.tsnr_run, args.tracer)

    log.info('Appending TSNR2TOEFFTIME coefficient of {:.6f} to {}/tsnr-ensemble-{}.fits.'.format(slope, args.outdir, args.tracer))

    ens = fits.open('{}/tsnr-ensemble-{}.fits'.format(args.outdir, args.tracer))
    hdr = ens[0].header

    hdr['TSNR2TOEFFTIME'] = slope
    hdr['EFFTIMEFILE']    = args.external_calib.replace('/global/cfs/cdirs/desi/survey', '$DESISURVEYOPS')
    hdr['TSNRRUNFILE']    = args.tsnr_run.replace('/global/cfs/cdirs/desi/spectro/redux',        '$REDUX')

    depend.setdep(hdr, 'desisim',  desisim.__version__)
    depend.setdep(hdr, 'desiutil', desiutil.__version__)

    ens.writeto('{}/tsnr-ensemble-{}.fits'.format(args.outdir, args.tracer), overwrite=True)

    log.info('Done.')

if __name__ == '__main__':
    main()
