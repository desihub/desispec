
"""
Read fibermap and cframe files for all exposures of a single night and update or create brick files.
"""

import argparse
import os.path

import numpy as np

import desispec.io
from desispec.log import get_logger, DEBUG


def parse(options=None):
    parser = argparse.ArgumentParser(description="Update or create brick files.")
    parser.add_argument('--verbose', action = 'store_true',
        help = 'Provide verbose reporting of progress.')
    parser.add_argument('--night', type = str, default = None, metavar = 'YYYYMMDD',
        help = 'Night to process in the format YYYYMMDD')
    parser.add_argument('--specprod', type = str, default = None, metavar = 'PATH',
        help = 'Override default path ($DESI_SPECTRO_REDUX/$PRODNAME) to processed data.')

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args):

    if args.verbose:
        log = get_logger(DEBUG)
    else:
        log = get_logger()

    if args.night is None:
        log.critical('Missing required night argument.')
        return -1

    # Initialize a dictionary of Brick objects indexed by '<band>_<brick-id>' strings.
    bricks = { }
    try:
        # Loop over exposures available for this night.
        for exposure in desispec.io.get_exposures(args.night, specprod_dir = args.specprod):
            # Ignore exposures with no fibermap, assuming they are calibration data.
            fibermap_path = desispec.io.findfile(filetype = 'fibermap',night = args.night,
                expid = exposure, specprod_dir = args.specprod)
            if not os.path.exists(fibermap_path):
                log.debug('Skipping exposure %08d with no fibermap.' % exposure)
                continue
            # Open the fibermap.
            fibermap_data = desispec.io.read_fibermap(fibermap_path)
            brick_names = set(fibermap_data['BRICKNAME'])
            # Loop over per-camera cframes available for this exposure.
            cframes = desispec.io.get_files(
                    filetype = 'cframe', night = args.night,
                    expid = exposure, specprod_dir = args.specprod)
            log.debug('Exposure %08d covers %d bricks and has cframes for %s.' % (
                exposure,len(brick_names),','.join(cframes.keys())))
            for camera,cframe_path in cframes.iteritems():
                band,spectro_id = camera[0],int(camera[1:])
                this_camera = (fibermap_data['SPECTROID'] == spectro_id)
                # Read this cframe file.
                frame = desispec.io.read_frame(cframe_path)
                # Loop over bricks.
                for brick_name in brick_names:
                    # Lookup the fibers belong to this brick.
                    this_brick = (fibermap_data['BRICKNAME'] == brick_name)
                    brick_data = fibermap_data[this_camera & this_brick]
                    fibers = np.mod(brick_data['FIBER'],500)
                    if len(fibers) == 0:
                        continue
                    brick_key = '%s_%s' % (band,brick_name)
                    # Open the brick file if this is the first time we are using it.
                    if brick_key not in bricks:
                        brick_path = desispec.io.findfile('brick',brickname = brick_name,band = band)
                        header = dict(BRICKNAM=(brick_name, 'Imaging brick name'),
                                      CHANNEL=(band, 'Spectrograph channel [b,r,z]'), )
                        bricks[brick_key] = desispec.io.brick.Brick(brick_path,mode = 'update',header = header)
                    # Add these fibers to the brick file. Note that the wavelength array is
                    # not per-fiber, so we do not slice it before passing it to add_objects().
                    bricks[brick_key].add_objects(frame.flux[fibers], frame.ivar[fibers],
                        frame.wave, frame.resolution_data[fibers], brick_data,args.night,exposure)
        # Close all brick files.
        for brick in bricks.itervalues():
            log.debug('Brick %s now contains %d spectra for %d targets.' % (
                brick.path,brick.get_num_spectra(),brick.get_num_targets()))
            brick.close()

    except RuntimeError as e:
        log.critical(str(e))
        return -2

