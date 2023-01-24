'''
desispec.scripts.compute.tsnr_ensemble
======================================

Generate Master TSNR ensemble DFLUX files.  See doc. 4723.  Note: in this
instance, ensemble avg. of flux is written, in order to efficiently generate
tile depths.
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

from desispec.tsnr import template_ensemble, gfa_template_ensemble

np.random.seed(seed=314)

# AR/DK DESI spectra wavelengths
# TODO:  where are brz extraction wavelengths defined?  https://github.com/desihub/desispec/issues/1006.
wmin, wmax, wdelta = 3600, 9824, 0.8
wave               = np.round(np.arange(wmin, wmax + wdelta, wdelta), 1)
cslice             = {"b": slice(0, 2751), "r": slice(2700, 5026), "z": slice(4900, 7781)}

def parse(options=None):
    parser = argparse.ArgumentParser(description="Generate a sim. template ensemble stack of given type and write it to disk at --outdir.")
    parser.add_argument('--nmodel', type = int, default = 1000, required=False,
                        help='Number of galaxies in the ensemble.')
    parser.add_argument('--tracer', type = str, default = 'bgs', required=True,
                        help='Tracer to generate of [bgs, lrg, elg, qso].')
    parser.add_argument('--smooth', type=float, default=100., required=False,
                        help='Smoothing scale [A] for DFLUX calc.')
    parser.add_argument('--config-filename', type = str, default = None, required=False,
			help='path to config filename (default is from python package desispec/data/tsnr/tsnr-config-{tracer}.yaml)')
    parser.add_argument('--nz-filename', type = str, default = None, required=False,
			help='path to n(z) filename (default is from $DESIMODEL/data/targets/nz_{tracer}.dat)')
    parser.add_argument('--outdir', type = str, default = 'bgs', required=True,
			help='Directory to write to.')
    parser.add_argument('--no-nz-convolution', action='store_true',
			help='Dont convolve each template dF^2 with redshift distribution')
    parser.add_argument('--mag-range', action='store_true',
			help='Monte Carlo the full mag range (given in config file) instead of using the same effective mag for all templates')
    args = None

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args

def main(args):
    if args.tracer == 'gpb':
        templates = gfa_template_ensemble()
        templates.compute()
        templates.plot()
        templates.write(dirname=args.outdir)

    elif args.tracer in ['bgs', 'lrg', 'elg', 'lya', 'qso']:
        templates = template_ensemble(tracer=args.tracer,config_filename=args.config_filename)
        templates.compute(nmodel=args.nmodel, smooth=args.smooth, nz_table_filename=args.nz_filename,
                      convolve_to_nz=(not args.no_nz_convolution), single_mag=(not args.mag_range))
        filename = "{}/tsnr-ensemble-{}.fits".format(args.outdir,args.tracer)
        templates.write(filename)
    else:
        raise ValueError('Unknown tracer {} to compute.'.format(args.tracer))

if __name__ == '__main__':
    print("please run desi_compute_tsnr_ensemble")
