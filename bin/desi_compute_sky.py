#!/usr/bin/env python
#
# See top-level LICENSE file for Copyright information
#
# -*- coding: utf-8 -*-

"""
This script computes the fiber flat field correction from a DESI continuum lamp frame.
"""

from desispec.io.frame import read_frame
from desispec.io.fibermap import read_fibermap
from desispec.io.fiberflat import read_fiberflat
from desispec.io.sky import write_sky
from desispec.fiberflat import apply_fiberflat
from desispec.sky import compute_sky
import argparse
import os
import os.path
import numpy as np
import sys
from astropy.io import fits


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--infile', type = str, default = None,
                    help = 'path of DESI exposure frame fits file')
parser.add_argument('--fibermap', type = str, default = None,
                    help = 'path of DESI exposure frame fits file')
parser.add_argument('--fiberflat', type = str, default = None,
                    help = 'path of DESI fiberflat fits file')
parser.add_argument('--outfile', type = str, default = None,
                    help = 'path of DESI sky fits file')


args = parser.parse_args()

if args.infile is None:
    print('Missing input')
    parser.print_help()
    sys.exit(12)

if args.fibermap is None:
    print('Missing fibermap')
    parser.print_help()
    sys.exit(12)

if args.fiberflat is None:
    print('Missing fiberflat')
    parser.print_help()
    sys.exit(12)
    
if args.outfile is None:
    print('Missing output')
    parser.print_help()
    sys.exit(12)

# read exposure to load data and get range of spectra
head = fits.getheader(args.infile)
specmin=head["SPECMIN"]
specmax=head["SPECMAX"]

flux,ivar,wave,resol = read_frame(args.infile)

# read fibermap to locate sky fibers
table=read_fibermap(args.fibermap)
selection=np.where((table["OBJTYPE"]=="SKY")&(table["FIBER"]>=specmin)&(table["FIBER"]<=specmax))[0]
if selection.size == 0 :
    print("Error, no sky fiber !")
    sys.exit(12)

# read fiberflat
fiberflat,ffivar,ffmask,ffmeanspec,ffwave = read_fiberflat(args.fiberflat)

# apply fiberflat to sky fibers
apply_fiberflat(flux=flux,ivar=ivar,wave=wave,fiberflat=fiberflat,ffivar=ffivar,ffmask=ffmask,ffwave=ffwave)

# compute sky model
skyflux,skyivar,skymask = compute_sky(wave,flux[selection],ivar[selection],resol[selection])

# write result
write_sky(args.outfile,head,skyflux,skyivar,skymask,wave)

print "successfully wrote",args.outfile
