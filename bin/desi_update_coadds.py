#!/usr/bin/env python
#
# See top-level LICENSE file for Copyright information
#
# -*- coding: utf-8 -*-
"""
Update co-adds for a single brick.
"""

import argparse
import os.path

import numpy as np

import desispec.io

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', action = 'store_true',
        help = 'Provide verbose reporting of progress.')
    parser.add_argument('--brick', type = str, default = None, metavar = 'NAME',
        help = 'Name of brick to process')
    parser.add_argument('--specprod', type = str, default = None, metavar = 'PATH',
        help = 'Override default path ($DESI_SPECTRO_REDUX/$PRODNAME) to processed data.')
    args = parser.parse_args()

    if args.brick is None:
        print 'Missing required brick argument.'
        return -1

    # Open the combined coadd file for this brick, for updating.
    coadd_all_path = desispec.io.meta.findfile('coadd_all',brickid = args.brick,specprod = args.specprod)
    coadd_all = desispec.io.brick.Brick(coadd_all_path,mode = 'update')

    # Loop over bands for this brick.
    for band in 'brz':
        # Open this band's brick file for reading.
        brick_path = desispec.io.meta.findfile('brick',brickid = args.brick,band = band,specprod = args.specprod)
        if not os.path.exists(brick_path):
            print 'Skipping non-existent brick file',brick_path
            continue
        brick = desispec.io.brick.Brick(brick_path,mode = 'readonly')
        num_objects = brick.get_num_objects()
        if args.verbose:
            print 'Processing %s with %d objects...' % (brick_path,num_objects)
        # Open this band's coadd file for updating.
        coadd_path = desispec.io.meta.findfile('coadd',brickid = args.brick,band = band,specprod = args.specprod)
        coadd = desispec.io.brick.Brick(coadd_path,mode = 'update')

        # Close files for this band.
        coadd.close()
        brick.close()

    # Close the combined coadd file.
    coadd_all.close()

if __name__ == '__main__':
    main()
