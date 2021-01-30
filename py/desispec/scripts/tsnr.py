"""
This script computes the template SNR
"""

import argparse
from astropy.table import Table

from desispec.io import read_frame
from desispec.io import read_fiberflat
from desispec.io import read_sky
from desispec.io.fluxcalibration import read_flux_calibration
from desispec.tsnr import calc_tsnr



def parse(options=None):
    parser = argparse.ArgumentParser(description="Apply fiberflat, sky subtraction and calibration.")
    parser.add_argument('-i','--infile', type = str, default = None, required=True,
                        help = 'path of DESI exposure frame fits file')
    parser.add_argument('--fiberflat', type = str, default = None, required=True,
                        help = 'path of DESI fiberflat fits file')
    parser.add_argument('--sky', type = str, default = None, required=True,
                        help = 'path of DESI sky fits file')
    parser.add_argument('--calib', type = str, default = None, required=True,
                        help = 'path of DESI calibration fits file')
    parser.add_argument('-o','--outfile', type = str, default = None, required=True,
                        help = 'path to output table csv or fits file')

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args):

    frame     = read_frame(args.infile)
    fiberflat = read_fiberflat(args.fiberflat)
    skymodel  = read_sky(args.sky)
    fluxcalib = read_flux_calibration(args.calib)
    results   = calc_tsnr(frame, fiberflat=fiberflat, skymodel=skymodel, fluxcalib=fluxcalib)

    table=Table(frame.fibermap)
    for k in results.keys() :
        table[k]=results[k]
    table.write(args.outfile,overwrite=True)
    print("wrote {}".format(args.outfile))
