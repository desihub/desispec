#!/usr/bin/env python

import os
import argparse
from desispec.io.raw import process_raw, read_raw_primary_header
import numpy as np
from astropy.io import fits

import astropy.io.fits as pyfits
from astropy.table import Table,vstack

from desiutil.log import get_logger

from desispec.ccdcalib import compute_dark_file, run_preproc_dark
from desispec.util import parse_nights, get_night_range
from desispec.io.util import get_speclog,erow_to_goodcamword,decode_camword
from desispec.io import findfile
from desispec.workflow.tableio import load_table
from desispec.scripts.compute_dark import get_stacked_dark_exposure_table


def preproc_darks_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Compute a preprocs for dark exposures used for dark creation",
                                     epilog='''
                                     Input is a list of raw dark images, possibly with various exposure times.
                                     Raw images are preprocessed without dark,mask correction.
                                     However gains are applied so the output is in electrons/sec.
                                     ''')

    parser.add_argument('-e','--expids', type = str, default = None, required = False, nargs="*",
                        help = 'exposures to process, can be a list of expids or a single expid')
    parser.add_argument('-n', '--night', type=int, default = None, required=False,
                        help='YEARMMDD night defining the darks to run through preproc')
    parser.add_argument('-c','--camword',type = str, required = True,
                        help = 'header HDU (int or string)')
    parser.add_argument('--bias', type = str, default = None, required=False,
                         help = 'specify a bias image calibration file (standard preprocessing calibration is turned off)')
    parser.add_argument('--nocosmic', action = 'store_true',
                        help = 'do not perform comic ray subtraction (much slower, but more accurate because median can leave traces)')
    parser.add_argument('--specprod', type=str, default=None, required=False,
                        help='Specify specprod to use nightly bias files and the exposure tables. Default is $SPECPROD if it is defined, otherwise will use the bias in DESI_SPECTRO_CALIB.')
    parser.add_argument('--preproc-dark-dir', type=str, default=None, required=False,
                        help='Specify alternate specprod directory where preprocessed dark frame images are saved. Default is same input specprod')
    parser.add_argument('--dry-run', action='store_true', help="Print which images would be used, but don't compute dark")

    return parser


def parse(options=None):
    # parse the command line arguments
    parser = preproc_darks_parser()
    
    #- uses sys.argv if options=None
    args = parser.parse_args(options)

    return args


def main(args=None):

    if not isinstance(args, argparse.Namespace):
        args = parse(args)

    log  = get_logger()

    # Methods to specify what images to use
    #   1. --images
    #   2. --nights [--first-expid, --last-expid, [--reference-night | --reference-expid]]
    #   3. --reference-night [--before, --after, [--first-expid, --last-expid]]

    if args.specprod is not None :
        os.environ["SPECPROD"] = args.specprod

    # check consistency of input options
    if args.night is not None and args.expids is not None :
        log.error("Cannot specify both --night and --expids arguments")
        return 1

    if args.expids is None and args.night is None:
        log.error("Need to specify input using --expids or --night")
        return 1

    ## get the requested cameras from the camword
    requested_cameras = decode_camword(args.camword)

    # first find the exposures if they are not given in input
    exptablename = findfile("exposure_table", specprod=args.specprod, night=args.night)
    exptable = load_table(tablename=exptablename, tabletype="exposure_table")
    if exptable is None or len(exptable) == 0:
        log.error("No valid exposures found for dark frame computation.")
        return 1
    
    if args.expids is not None:
        exptable = exptable[np.isin(exptable["EXPID"], args.expids)]

    # assemble corresponding images
    expids, cameras_list, files = [], [], []
    for row in exptable:
        filename = findfile("raw",night=row["NIGHT"],expid=row["EXPID"])
        if os.path.exists(filename):
            files.append(filename)
            cameras_list.append(decode_camword(row["CAMERA"]))
            expids.append(row["EXPID"])
        else:
            log.error(f'Skipping missing file {filename}')

    if args.dry_run:
        image_str = ' '.join(files)
        log.info(f'Input images: {image_str}')
        log.info('--dry-run mode, exiting before running preproc_darks')
        return 0

    if args.bias is None:
        thisbias = True
    else:
        thisbias = args.bias
    for expid, cameras, filename in zip(expids, cameras_list, files):
        fx = fits.open(filename, memmap=False)
        for camera in cameras:
            if camera in requested_cameras:
                if camera.upper() not in fx:
                    log.error(f'Camera {camera} not in {filename}')
                    continue
                ## process the image
                log.info(f'Processing {filename} for camera {camera} (expid={expid})')   
                primary_header = read_raw_primary_header(filename)
                rawimage = fx[camera.upper()].data
                header = fx[camera.upper()].header
                  
                if args.preproc_dark_dir is not None :
                    preproc_filename = findfile("preproc_for_dark",night=night,expid=expid,camera=camera,specprod_dir=preproc_dark_dir)
                else:
                    preproc_filename = findfile("preproc_for_dark",night=args.night,expid=expid,camera=camera)
                img = process_raw(primary_header, rawimage, header, camera=camera, bias=thisbias, nocosmic=nocosmic,
                      mask=False, dark=False, pixflat=False, fallback_on_dark_not_found=True

                # is saved in preproc_dark_dir if not None
                io.write_image(preproc_filename,img)
                log.info(f"Wrote {preproc_filename}")
        
    return 0
