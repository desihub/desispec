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
import sys

def main() :

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--infile', type = str, default = None, required=True,
                        help = 'path of DESI frame fits file corresponding to a continuum lamp exposure')
    parser.add_argument('--outfile', type = str, default = None, required=True,
                        help = 'path of DESI fiberflat fits file')


    args = parser.parse_args()
    log=get_logger()
    
    log.info("starting")

    frame = read_frame(args.infile)
    fiberflat = compute_fiberflat(frame)
    write_fiberflat(args.outfile, fiberflat, frame.meta)

    log.info("successfully wrote %s"%args.outfile)


if __name__ == '__main__':
    main()
