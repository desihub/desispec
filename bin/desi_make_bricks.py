#!/usr/bin/env python
#
# See top-level LICENSE file for Copyright information
#
# -*- coding: utf-8 -*-
"""
Read fibermap and cframe files for all exposures of a single night and update or create brick files.
"""

import argparse
import os.path

import numpy as np

import desispec.io

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', action = 'store_true',
        help = 'Provide verbose reporting of progress.')
    parser.add_argument('--night', type = str, default = None, metavar = 'YYYYMMDD',
        help = 'Night to process in the format YYYYMMDD')
    parser.add_argument('--specprod', type = str, default = None, metavar = 'PATH',
        help = 'Override default path ($DESI_SPECTRO_REDUX/$PRODNAME) to processed data.')
    args = parser.parse_args()

    if args.night is None:
        print 'Missing required night argument.'
        return -1

    # Initialize a dictionary of Brick objects indexed by '<band>_<brick-id>' strings.
    bricks = { }
    try:
        # Loop over exposures available for this night.
        for exposure in desispec.io.get_exposures(args.night,specprod = args.specprod):
            # Ignore exposures with no fibermap, assuming they are calibration data.
            fibermap_path = desispec.io.findfile(filetype = 'fibermap',night = args.night,
                expid = exposure,specprod = args.specprod)
            if not os.path.exists(fibermap_path):
                if args.verbose:
                    print 'Skipping exposure %08d with no fibermap.' % exposure
                continue
            # Open the fibermap.
            fibermap_data,fibermap_hdr = desispec.io.read_fibermap(fibermap_path)
            brick_ids = set(fibermap_data['BRICKNAME'])
            # Loop over per-camera cframes available for this exposure.
            cframes = desispec.io.get_files(filetype = 'cframe',night = args.night,
                expid = exposure,specprod = args.specprod)
            if args.verbose:
                print 'Exposure %08d covers %d bricks and has cframes for %s.' % (
                    exposure,len(brick_ids),','.join(cframes.keys()))
            for camera,cframe_path in cframes.iteritems():
                band,spectro_id = camera[0],int(camera[1:])
                this_camera = (fibermap_data['SPECTROID'] == spectro_id)
                # Read this cframe file.
                flux,ivar,wave,resolution,hdr = desispec.io.read_frame(cframe_path)
                # Loop over bricks.
                for brick_id in brick_ids:
                    # Lookup the fibers belong to this brick.
                    this_brick = (fibermap_data['BRICKNAME'] == brick_id)
                    brick_data = fibermap_data[this_camera & this_brick]
                    fibers = np.mod(brick_data['FIBER'],500)
                    if len(fibers) == 0:
                        continue
                    brick_key = '%s_%s' % (band,brick_id)
                    # Open the brick file if this is the first time we are using it.
                    if brick_key not in bricks:
                        brick_path = desispec.io.findfile('brick',brickid = brick_id,band = band)
                        bricks[brick_key] = desispec.io.brick.Brick(brick_path,mode = 'update')
                    # Add these fibers to the brick file. Note that the wavelength array is
                    # not per-fiber, so we do not slice it before passing it to add_objects().
                    bricks[brick_key].add_objects(flux[fibers],ivar[fibers],
                        wave,resolution[fibers],brick_data,args.night,exposure)
        # Close all brick files.
        for brick in bricks.itervalues():
            if args.verbose:
                print 'Brick %s now contains %d spectra for %d targets.' % (
                    brick.path,brick.get_num_spectra(),brick.get_num_targets())
            brick.close()

    except RuntimeError,e:
        print str(e)
        return -2

if __name__ == '__main__':
    main()
