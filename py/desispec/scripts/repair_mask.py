"""
desispec.scripts.repair_mask
============================

Patch holes in spectrograph 2D cosmic-ray masks using morphological binary
image analysis.
"""

import os
import argparse

from desispec.joincosmics import RepairMask

def parse(options=None):
    p = argparse.ArgumentParser(description='Patch holes in cosmic-ray masks')
    p.add_argument('-i', '--infiles', type=str, required=True, nargs='+',
                   help='Input preproc files (FITS format).')
    p.add_argument('-l', '--loglevel', type=str, default='info',
                   choices=['debug','info','warning','error','critical'],
                   help='Set verbosity of the logger.')

    if options is None:
        args = p.parse_args()
    else:
        args = p.parse_args(options)
    return args

def main(args):
    from astropy.io import fits
    import logging

    # Set up the logger.
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: {}'.format(args.loglevel))
    logging.basicConfig(level=numeric_level)

    for infile in args.infiles:
        hdus = fits.open(infile)
        img = hdus['IMAGE'].data
        mask = hdus['MASK'].data

        rep = RepairMask()
        logging.info('Repairing cosmic mask for {}.'.format(infile))
        newmask = rep.repair(mask)

        # Set up image file prefix.
        prefix = os.path.basename(infile)
        prefix = prefix.replace('.fits', '')

        # Output initial image, mask, and patched mask.
        logging.info('Plotting IMG, MASK, NEWMASK.')
        rep.plot(img, mask, newmask, prefix=prefix)

        # Output image chunks for easier visual inspection.
        logging.info('Plotting image chunks.')
        rep.plot_chunks(img, mask, newmask, prefix=prefix)
