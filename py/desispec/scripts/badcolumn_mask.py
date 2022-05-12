#!/usr/bin/env python

import os,sys
import argparse
import numpy as np
import fitsio
import scipy.ndimage

from astropy.table import Table

from desiutil.log import get_logger
from desispec.io import read_xytraceset,read_frame,write_frame
from desispec.badcolumn import add_badcolumn_mask

def parse(options=None):

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Read traceset and badcolumns table and mask spectra in frame")
    parser.add_argument('-i','--infile', type = str, default = None, required = True,
                        help = 'input frame file')
    parser.add_argument('-o','--outfile', type = str, default = None, required = False,
                        help = 'output frame file (or use --overwrite option)')
    parser.add_argument('--overwrite', action = 'store_true',
                        help = 'overwrite mask in frame file (or use --outframe)')
    parser.add_argument('--psf', type = str, default = None, required = True,
                        help = 'input psf file for the traceset')
    parser.add_argument('--badcolumns', type = str, default = None, required = True,
                        help = 'input table with bad columns with at least rows COLUMN and VALUE')
    parser.add_argument('--threshold-elec-per-sec', type = float, default = 0.005, required = False,
                        help = 'threshold in electrons per sec')
    parser.add_argument('--frac-threshold', type = float, default = 0.4, required = False,
                        help = 'mask fibers with a number of masked spectral value exceeding this threshold')

    args = parser.parse_args(options)

    return args

def main(args=None):

    log=get_logger()

    if not isinstance(args, argparse.Namespace):
        args = parse(args)
    
    frame   = read_frame(args.infile)
    xyset   = read_xytraceset(args.psf)
    badcols = Table.read(args.badcolumns)

    add_badcolumn_mask(frame=frame,xyset=xyset,badcolumns_table=badcols,threshold_value=args.threshold_elec_per_sec,threshold_specfrac=args.frac_threshold)

    if args.overwrite  :
        write_frame(args.infile,frame)
        log.info("overwrote "+args.infile)
    else :
        write_frame(args.outfile,frame)
        log.info("wrote "+args.outfile)

    return 0
