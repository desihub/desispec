#!/usr/bin/env python
#
# See top-level LICENSE file for Copyright information
#
# -*- coding: utf-8 -*-

"""
This script computes the fiber flat field correction from a DESI continuum lamp frame.
"""

from desispec.io.frame import read_frame
from desispec.io.fiberflat import write_fiberflat
from desispec.fiberflat import compute_fiberflat
import argparse
import os
import os.path
import numpy as np
import sys
from astropy.io import fits


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--infile', type = str, default = None,
                    help = 'path of DESI frame fits file corresponding to a continuum lamp exposure')
parser.add_argument('--outfile', type = str, default = None,
                    help = 'path of DESI fiberflat fits file')


args = parser.parse_args()

if args.infile is None:
    print('Missing input')
    parser.print_help()
    sys.exit(12)
    
if args.outfile is None:
    print('Missing output')
    parser.print_help()
    sys.exit(12)

head = fits.getheader(args.infile)
flux,ivar,wave,resol = read_frame(args.infile)
fiberflat,fiberflat_ivar,fiberflat_mask,mean_spectrum = compute_fiberflat(wave,flux,ivar,resol)
write_fiberflat(args.outfile,head,fiberflat,fiberflat_ivar,fiberflat_mask,mean_spectrum,wave)

print "successfully wrote",args.outfile
