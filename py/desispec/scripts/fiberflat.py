"""
desispec.scripts.fiberflat
==========================

Utility functions to compute a fiber flat correction and apply it
We try to keep all the (fits) io separated.
"""
from __future__ import absolute_import, division
import time

import numpy as np

from desispec.io import read_frame
from desispec.io import write_fiberflat
from desispec.fiberflat import compute_fiberflat
from desiutil.log import get_logger
from desispec.cosmics import reject_cosmic_rays_1d

import argparse

def parse(options=None):
    parser = argparse.ArgumentParser(description="Compute the fiber flat field correction from a DESI continuum lamp frame")
    parser.add_argument('-i','--infile', type = str, default = None, required=True,
                        help = 'path of DESI frame fits file corresponding to a continuum lamp exposure')
    parser.add_argument('-o','--outfile', type = str, default = None, required=True,
                        help = 'path of DESI fiberflat fits file')
    parser.add_argument('--nsig', type = float, default = 10, required=False,
                        help = 'nsigma clipping')
    parser.add_argument('--acc', type = float, default = 5.e-4, required=False,
                        help = 'required accuracy (iterative loop)')
    parser.add_argument('--smoothing-resolution', type = float, default = 5., required=False,
                        help = 'resolution for spline fit to reject outliers')
    parser.add_argument('--cosmics-nsig', type = float, default = 0, required=False,
                        help = 'n sigma rejection for cosmics in 1D (default, no rejection)')
    
    args = parser.parse_args(options)

    return args


def main(args=None) :

    if not isinstance(args, argparse.Namespace):
        args = parse(args)

    log=get_logger()
    log.info("starting at {}".format(time.asctime()))

    # Process
    frame = read_frame(args.infile)
    
    if args.cosmics_nsig>0 : # Reject cosmics         
        reject_cosmic_rays_1d(frame,args.cosmics_nsig)
    
    fiberflat = compute_fiberflat(frame,nsig_clipping=args.nsig,accuracy=args.acc,smoothing_res=args.smoothing_resolution)

    # Write
    write_fiberflat(args.outfile, fiberflat, frame.meta)
    log.info("successfully wrote %s"%args.outfile)
    log.info("done at {}".format(time.asctime()))
