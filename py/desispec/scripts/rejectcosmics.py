"""
desispec.scripts.rejectcosmics
==============================

This script finds cosmics in a pre-processed image and write the result in the mask extension of an output image
(output can be same as input).
"""

from desispec.io import image
from desispec.maskbits import ccdmask
from desispec.cosmics import reject_cosmic_rays
from desiutil.log import get_logger
import argparse
import numpy as np


def parse(options=None):
    parser = argparse.ArgumentParser(description="Find and mask cosmics.")
    parser.add_argument('--infile', type = str, default = None, required=True,
                        help = 'path of DESI exposure image fits file')
    parser.add_argument('--outfile', type = str, default = None,
                        help = 'path of DESI output exposure image fits file (default is overwriting input with new mask)')
    parser.add_argument('--ignore_cosmic_ccdmask', action = 'store_true',
                        help = 'ignore pre-existing bitmask ccdmask.COSMIC (for development tests)')

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args) :

    if args.outfile is not None :
        outfile=args.outfile
    else :
        outfile=args.infile

    log = get_logger()
    log.info("starting finding cosmics in %s"%args.infile)

    img=image.read_image(args.infile)

    if args.ignore_cosmic_ccdmask :
        log.warning("ignore cosmic ccdmask for test")
        log.debug("ccdmask.COSMIC = %d"%ccdmask.COSMIC)
        cosmic_ray_prexisting_mask = img.mask & ccdmask.COSMIC
        img._mask &= ~ccdmask.COSMIC  #- turn off cosmic mask

    reject_cosmic_rays(img)

    log.info("writing data and new mask in %s"%outfile)
    image.write_image(outfile, img, meta=img.meta)

    log.info("done")
