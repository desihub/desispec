#!/usr/bin/env python
#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-

"""
This script computes the fiber flat field correction from a DESI continuum lamp frame.
"""

import pdb
from desispec.io import read_frame
from desispec.io import write_fiberflat
from desispec.fiberflat import compute_fiberflat
from desispec.log import get_logger
from desispec.io.qa import load_qa_frame
from desispec.io import write_qa_frame
import argparse

from xastropy.xutils import xdebug as xdb

def main() :

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--infile', type = str, default = None, required=True,
                        help = 'path of DESI frame fits file corresponding to a continuum lamp exposure')
    parser.add_argument('--outfile', type = str, default = None, required=True,
                        help = 'path of DESI fiberflat fits file')
    parser.add_argument('--qafile', type=str, default=None, required=False,
                        help='path of QA file.')


    args = parser.parse_args()
    log=get_logger()

    log.info("starting")

    frame = read_frame(args.infile)
    fiberflat = compute_fiberflat(frame)

    # QA
    if (args.qafile is not None):
        log.info("performing fiberflat QA")
        # Load
        qaframe = load_qa_frame(args.qafile, frame, flavor='science')
        # Run
        qaframe.run_qa('FIBERFLAT', (frame, fiberflat))
        # Write
        if args.qafile is not None:
            write_qa_frame(args.qafile, qaframe)
            log.info("successfully wrote {:s}".format(args.qafile))

    # Write
    write_fiberflat(args.outfile, fiberflat, frame.meta)
    log.info("successfully wrote %s"%args.outfile)


if __name__ == '__main__':
    main()
