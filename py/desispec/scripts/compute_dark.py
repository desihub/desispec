#!/usr/bin/env python

import os
import argparse
import numpy as np


import astropy.io.fits as pyfits
from astropy.table import Table,vstack

from desiutil.log import get_logger

from desispec.ccdcalib import compute_dark_file
from desispec.util import parse_nights, get_night_range
from desispec.io.util import get_speclog,erow_to_goodcamword,decode_camword
from desispec.io import findfile
from desispec.workflow.tableio import load_table


def compute_dark_baseparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Compute a master dark",
                                     epilog='''
                                     Input is a list of raw dark images, possibly with various exposure times.
                                     Raw images are preprocessed without dark,mask correction.
                                     However gains are applied so the output is in electrons/sec.
                                     We first compute a masked median of the preprocessed images divided by their exposure time.
                                     Then mask outlier pixels, and then compute the dark with optimal weights (propto. exptime).
                                     We use for this the keyword EXPREQ in the raw image primary header, or EXPTIME if the former is absent.''')

    parser.add_argument('-n','--nights', type=str, default = None, required=False,
                        help='YEARMMDD nights to use (comma separated or with : to define a range. integers that do not correspond to valid dates are ignored)')
    parser.add_argument('-r', '--reference-night', type=int, default = None, required=False,
                        help='YEARMMDD reference night defining the hardware state for this dark frame (default is most recent, option cannot be set at the same time as reference-expid)')
    parser.add_argument('-b', '--before', type=int, default=30, required=False,
                        help = 'Number of nights before reference-night to include')
    parser.add_argument('-a', '--after', type=int, default=15, required=False,
                        help = 'Number of nights after reference-night to include')
    parser.add_argument('--reference-expid', type=int, default = None, required=False,
                        help='reference expid defining the hardware state for this dark frame (default is most recent, option cannot be set at the same time as reference-night)')
    parser.add_argument('--first-expid', type=int, default = None, required=False,
                        help='First EXPID to include (to use in combination with --nights option)')
    parser.add_argument('--last-expid', type=int, default = None, required=False,
                        help='Last EXPID to include (to use in combination with --nights option)')
    parser.add_argument('--min-exptime', type=float, default = 500,
                        help='minimal exposure time to consider')
    parser.add_argument('--max-exptime', type=float, default = None,
                        help='maximal exposure time to consider')
    parser.add_argument('--max-temperature-diff', type=float, default = 4. ,
                        help='maximal difference of CCD temperature to consider')
    parser.add_argument('--bias', type = str, default = None, required=False,
                         help = 'specify a bias image calibration file (standard preprocessing calibration is turned off)')
    parser.add_argument('--nocosmic', action = 'store_true',
                        help = 'do not perform comic ray subtraction (much slower, but more accurate because median can leave traces)')
    parser.add_argument('--min-hours-since-vccd-on', type=float, default=4., required=False,
                        help='Minimum time (in hours) since voltages were turned on')
    parser.add_argument('--specprod', type=str, default=None, required=False,
                        help='Specify specprod to use nightly bias files and the exposure tables. Default is $SPECPROD if it is defined, otherwise will use the bias in DESI_SPECTRO_CALIB.')
    parser.add_argument('--save-preproc', action='store_true', help='save intermediate preproc files')
    parser.add_argument('--preproc-dark-dir', type=str, default=None, required=False,
                        help='Specify alternate specprod directory where preprocessed dark frame images are saved. Default is same input specprod')
    parser.add_argument('--dry-run', action='store_true', help="If dry_run, print which images would be used, but don't compute dark.")
    parser.add_argument('--max-dark-exposures', type=int, default=300, required=False,
                        help='Maximum number of dark exposures to use. Default is 300. If more than this number of exposures are found, ' \
                        'the script will downselect to the closest exposures in time up to this limit.')
    parser.add_argument('--skip-camera-check', action='store_true', help="If True, doesn't check if camera exists for an exposure ahead of time.")
    parser.add_argument('--dont-search-filesystem', action='store_true', help="If True, doesn't search filesystem for exposures.")

    return parser


def compute_dark_parser():
    """
    Takes the "base" argparser that is also used by desi_compute_dark_night and add specific args
    """
    parser = compute_dark_baseparser()
    parser.add_argument('-i','--images', type = str, default = None, required = False, nargs="*",
                        help = 'path of raws image fits files (or use --nights or --reference-night)')
    parser.add_argument('-c','--camera',type = str, required = True,
                        help = 'header HDU (int or string)')
    parser.add_argument('-o','--outfile', type = str, default = None, required = True,
                        help = 'output median image filename')
    return parser


def parse(options=None):
    # parse the command line arguments
    parser = compute_dark_parser()
    
    #- uses sys.argv if options=None
    args = parser.parse_args(options)

    return args

def get_stacked_dark_exposure_table(args):
    """
    Get the exposure table for the dark exposures to be used.
    If --nights is specified, it will return the exposures for those nights.
    If --reference-night is specified, it will return the exposures around that night.
    If --images is specified, it will return the exposures corresponding to those images.
    """
    log  = get_logger()
    # check all required environment variables, then return error if any are missing
    envok = True
    for k in ["DESI_SPECTRO_DATA","DESI_SPECTRO_REDUX","SPECPROD"] :
        if k not in os.environ :
            envok = False
            log.error(f"args.nights/referene_night specified but variable {k} is not set so we cannot find the exposures.")
            if k=="SPECPROD" :
                log.error("consider using argument --specprod.")

    if not envok:
        return None

    if args.nights is None:
        nights = get_night_range(args.reference_night, args.before, args.after)
    else:
        nights = parse_nights(args.nights)

    log.info(f"Will look for dark exposures in nights: {nights}")
    tables = []
    missing_nights = []
    for night in nights :
        filename = findfile("exposure_table",night=night, readonly=True)
        if os.path.isfile(filename) :
            tmp_table=load_table(filename, suppress_logging=True)
            if len(tmp_table)==0 : continue

            # keep only valid exposures
            keep = (tmp_table['LASTSTEP'] != 'ignore')
            tmp_table = tmp_table[keep]
            if len(tmp_table)==0 : continue

            # only keep useful rows to avoid issues with columns
            table = tmp_table['NIGHT', 'EXPID', 'MJD-OBS', 'OBSTYPE', 'EXPTIME']
            tables.append(table)
        else :
            log.warning(f"No exposure table for {night}")
            nightdir=os.path.join(os.environ["DESI_SPECTRO_DATA"],str(night))
            if not os.path.isdir(nightdir) :
                log.warning(f"No data directory {nightdir}")
                continue
            missing_nights.append(night)
    if len(missing_nights)>0 and args.dont_search_filesystem:
        log.info(f"Found nights without exposure tables ({missing_nights}) "
                 + "but told not to search for missing nights, so continuing without them.")
    elif len(missing_nights)>0:
        log.info(f"Computing speclog for nights without exposure tables ({missing_nights})")
        table = get_speclog(missing_nights)
        if len(tmp_table)>0 :
            table = table[["NIGHT","EXPID","MJD-OBS","OBSTYPE","EXPTIME"]]
            for i in range(len(table)) :
                table["OBSTYPE"][i]=table["OBSTYPE"][i].lower()
            tables.append(table)
    if len(tables)>0 :
        exptable=vstack(tables)
    else :
        log.error(f"empty list of dark exposures")
        return None

    valid=(exptable["OBSTYPE"]=="dark")
    log.info(f"{np.sum(valid)} dark exposures")
    valid &= (exptable["EXPTIME"]>=args.min_exptime)
    log.info(f"{np.sum(valid)} dark exposures with EXPTIME>={args.min_exptime}")
    if args.first_expid is not None :
        valid &= (exptable["EXPID"]>=args.first_expid)
        log.info(f"{np.sum(valid)} dark exposures with EXPID>={args.first_expid}")
    if args.last_expid is not None :
        valid &= (exptable["EXPID"]<=args.last_expid)
        log.info(f"{np.sum(valid)} dark exposures with EXPID<={args.last_expid}")

    # trim to valid exposures
    exptable = exptable[valid]
    exptable.sort('EXPID')
    log.info(f"{len(exptable)} valid dark exposures found.")

    # assemble corresponding images
    args.images = []
    file_exists = np.ones(len(exptable), dtype=bool)
    if not args.dont_search_filesystem:
        for e in range(len(exptable)):
            filename = findfile("raw",night=exptable["NIGHT"][e],expid=exptable["EXPID"][e])
            if not os.path.exists(filename):
                # "Missing" files can occur due to a mismatch between the NIGHT header keyword
                # and the directory in which the file is found, e.g. 20250620/00298589/desi-00298589.fits.fz
                # has header NIGHT=20250619, but also FLAVOR=science instead of FLAVOR=dark
                file_exists[e] = False
                log.error(f'Skipping missing file {filename}')

    if not np.all(file_exists):
        exptable = exptable[file_exists]
        log.info(f"{len(exptable)} exposures will be used to build the {args.camera} dark")
        log.info(exptable)

    return exptable


def main(args=None, exptable=None):

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
    if args.nights is not None and args.images is not None :
        log.error("Cannot specify both --nights and --image arguments")
        return 1

    if args.nights is None and args.images is None and args.reference_night is None:
        log.error("Need to specify input using --images, --nights, or --reference-night")
        return 1

    if args.reference_night is not None and args.reference_expid is not None :
        log.error("Cannot use --reference-night and --reference-expid at the same time.")
        return 1

    # first find the exposures if they are not given in input
    if args.images is None:
        if exptable is None:
            # if no images are given, we need to find the exposures
            exptable = get_stacked_dark_exposure_table(args)
            keep = np.repeat(True,len(exptable))
            for i,entry in enumerate(exptable) :
                keep[i] &= ( args.camera in decode_camword(erow_to_goodcamword(entry, suppress_logging=True, exclude_badamps=True)) )
            exptable = exptable[keep]

            if exptable is None or len(exptable) == 0:
                log.error("No valid exposures found for dark frame computation.")
                return 1

        # assemble corresponding images
        args.images = []
        for row in exptable:
            filename = findfile("raw",night=row["NIGHT"],expid=row["EXPID"])
            if os.path.exists(filename):
                args.images.append(filename)
            else:
                log.error(f'Skipping missing file {filename}')
                return 1

    # find the most recent exposure with the camera and read its header
    # unless reference_expid or reference_night is set
    reference_header = None
    reference_header_possible_filenames = []
    if args.reference_expid is not None :
        selection=np.where(exptable["EXPID"]==args.reference_expid)[0]
        if selection.size == 0 :
            log.error(f"Reference expid {args.reference_expid} is not in the list of input darks")
            return 1
        reference_header_possible_filenames.append(args.images[selection[0]])
    elif args.reference_night is not None :
        selection=np.where(exptable["NIGHT"]==args.reference_night)[0]
        if selection.size == 0 :
            log.error(f"No dark during reference night {args.reference_night} in input list")
            return 1
        indices=np.argsort(exptable["EXPID"][selection])[::-1]
        for index in indices :
            reference_header_possible_filenames.append(args.images[selection[index]])
    else :
        if exptable is not None:
            indices=np.argsort(exptable["EXPID"])[::-1]
            for index in indices :
                reference_header_possible_filenames.append(args.images[index])
        else:
            reference_header_possible_filenames.extend(args.images)

    for filename in reference_header_possible_filenames :
        fitsfile=pyfits.open(filename)
        if not args.camera in fitsfile :
            fitsfile.close()
            continue
        reference_header = fitsfile[args.camera].header
        fitsfile.close()
        break
    if reference_header is None :
        log.critical(f"No exposure has the camera {args.camera}.")
        return 1

    assert args.camera.upper() == reference_header['CAMERA'].upper()

    log.info(f"Use for hardware state reference EXPID={reference_header['EXPID']} NIGHT={reference_header['NIGHT']} CAMERA={reference_header['CAMERA']} DETECTOR={reference_header['DETECTOR']}")

    if args.dry_run:
        image_str = ' '.join(args.images)
        log.info(f'Input images: {image_str}')
        log.info('--dry-run mode, exiting before generating dark')
        return 0

    compute_dark_file(args.images, args.outfile, camera=args.camera, bias=args.bias,
                      nocosmic=args.nocosmic,
                      min_vccdsec=(args.min_hours_since_vccd_on * 3600.),
                      max_temperature_diff=args.max_temperature_diff,
                      reference_header=reference_header,
                      save_preproc=args.save_preproc,
                      preproc_dark_dir=args.preproc_dark_dir,
                      max_dark_exposures=args.max_dark_exposures)
    return 0
