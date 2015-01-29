#!/usr/bin/env python
#
# See top-level LICENSE file for Copyright information
#
# -*- coding: utf-8 -*-

"""
This script processes an exposure by applying fiberflat, sky subtraction, spectro-photometric calibration depending on input.
"""

from desispec.io.frame import read_frame,write_frame
from desispec.io.fiberflat import read_fiberflat
from desispec.io.sky import read_sky
from desispec.fiberflat import apply_fiberflat
from desispec.sky import subtract_sky

import argparse
import os
import os.path
import numpy as np
import sys
from astropy.io import fits


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--infile', type = str, default = None,
                    help = 'path of DESI exposure frame fits file')
parser.add_argument('--fiberflat', type = str, default = None,
                    help = 'path of DESI fiberflat fits file')
parser.add_argument('--sky', type = str, default = None,
                    help = 'path of DESI sky fits file')
parser.add_argument('--outfile', type = str, default = None,
                    help = 'path of DESI sky fits file')
# add calibration here when exists

args = parser.parse_args()

if args.infile is None:
    print('Missing input')
    parser.print_help()
    sys.exit(12)

if args.fiberflat is None and args.sky is None:
    print('Nothing to do ??')
    parser.print_help()
    sys.exit(12)
 
if args.outfile is None:
    print('Missing output')
    parser.print_help()
    sys.exit(12)


head = fits.getheader(args.infile)
flux,ivar,wave,resol = read_frame(args.infile)

if args.fiberflat!=None :
    print "apply fiberflat"
    # read fiberflat
    fiberflat,ffivar,ffmask,ffmeanspec,ffwave = read_fiberflat(args.fiberflat)

    # apply fiberflat to sky fibers
    apply_fiberflat(flux=flux,ivar=ivar,wave=wave,fiberflat=fiberflat,ffivar=ffivar,ffmask=ffmask,ffwave=ffwave)


if args.sky!=None :
    print "subtract sky"
    # read sky
    skyflux,sivar,smask,cskyflux,csivar,swave=read_sky(args.sky)
    # subtract sky
    subtract_sky(flux=flux,ivar=ivar,resolution_data=resol,wave=wave,skyflux=skyflux,convolved_skyivar=csivar,skymask=smask,skywave=swave)
  

# save output
write_frame(args.outfile,head,flux,ivar,wave,resol)

print "successfully wrote",args.outfile
