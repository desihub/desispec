#!/usr/bin/env python
#
# See top-level LICENSE file for Copyright information
#
# -*- coding: utf-8 -*-

"""
This script computes the fiber flat field correction from a DESI continuum lamp frame.
"""

from desispec.io import read_frame
from desispec.io import write_fiberflat
from desispec.fiberflat import compute_fiberflat
from desispec.log import get_logger
import argparse
import os
import os.path
import numpy as np
import sys
def main() :

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--infile', type = str, default = None,
                        help = 'path of DESI frame fits file corresponding to a continuum lamp exposure')
    parser.add_argument('--outfile', type = str, default = None,
                        help = 'path of DESI fiberflat fits file')


    args = parser.parse_args()
    log=get_logger()

    if args.infile is None:
        log.critical('Missing input')
        parser.print_help()
        sys.exit(12)

    if args.outfile is None:
        log.critical('Missing output')
        parser.print_help()
        sys.exit(12)

    log.info("starting")

    spectra = read_frame(args.infile)
    fiberflat = compute_fiberflat(spectra)
    write_fiberflat(args.outfile, fiberflat, spectra.header)

    log.info("successfully wrote %s"%args.outfile)


if __name__ == '__main__':
    main()
