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
import desispec.coaddition

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
    coadd_all_file = desispec.io.brick.CoAddedBrick(coadd_all_path,mode = 'update')

    # Initialize a dictionary of objects we have processed for each band.
    objects = dict(b = { },r = { },z = { })

    # Loop over bands for this brick.
    for band in 'brz':
        # Open this band's brick file for reading.
        brick_path = desispec.io.meta.findfile('brick',brickid = args.brick,band = band,specprod = args.specprod)
        if not os.path.exists(brick_path):
            print 'Skipping non-existent brick file',brick_path
            continue
        brick_file = desispec.io.brick.Brick(brick_path,mode = 'readonly')
        num_objects = brick_file.get_num_objects()
        flux,ivar,wlen,resolution = (brick_file.hdu_list[0].data,brick_file.hdu_list[1].data,
            brick_file.hdu_list[2].data,brick_file.hdu_list[3].data)
        if args.verbose:
            print 'Processing %s with %d objects...' % (brick_path,num_objects)
        # Open this band's coadd file for updating.
        coadd_path = desispec.io.meta.findfile('coadd',brickid = args.brick,band = band,specprod = args.specprod)
        coadd_file = desispec.io.brick.CoAddedBrick(coadd_path,mode = 'update')
        # Loop over objects in the brick file.
        for index,info in enumerate(brick_file.hdu_list[4].data):
            resolution_matrix = desispec.io.frame.resolution_data_to_sparse_matrix(resolution[index])
            spectrum = desispec.coaddition.Spectrum(wlen,flux[index],ivar[index],resolution_matrix)
            target_id = info['TARGETID']
            if target_id in objects[band]:
                objects[band][target_id] += spectrum
            else:
                objects[band][target_id] = spectrum
        # Save the coadded spectra for this band.
        for target_id in objects[band]:
            print 'Coadded target ID',target_id
            #objects[band][target_id]._finalize()
        # Close files for this band.
        coadd_file.close()
        brick_file.close()

    # Close the combined coadd file.
    coadd_all_file.close()

if __name__ == '__main__':
    main()
