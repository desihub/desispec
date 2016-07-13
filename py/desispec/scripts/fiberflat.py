"""
desispec.fiberflat
==================

Utility functions to compute a fiber flat correction and apply it
We try to keep all the (fits) io separated.
"""
from __future__ import absolute_import, division

import numpy as np

from desispec.io import read_frame
from desispec.io import write_fiberflat
from desispec.fiberflat import compute_fiberflat
from desispec.log import get_logger
from desispec.io.qa import load_qa_frame
from desispec.io import write_qa_frame
from desispec.qa import qa_plots
import argparse


def parse(options=None):
    parser = argparse.ArgumentParser(description="Compute the fiber flat field correction from a DESI continuum lamp frame")
    parser.add_argument('--infile', type = str, default = None, required=True,
                        help = 'path of DESI frame fits file corresponding to a continuum lamp exposure')
    parser.add_argument('--outfile', type = str, default = None, required=True,
                        help = 'path of DESI fiberflat fits file')
    parser.add_argument('--qafile', type=str, default=None, required=False,
                        help='path of QA file')
    parser.add_argument('--qafig', type = str, default = None, required=False,
                        help = 'path of QA figure file')
    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args) :

    log=get_logger()
    log.info("starting")

    # Process
    frame = read_frame(args.infile)
    fiberflat = compute_fiberflat(frame)

    # QA
    if (args.qafile is not None):
        log.info("performing fiberflat QA")
        # Load
        qaframe = load_qa_frame(args.qafile, frame, flavor=frame.meta['FLAVOR'])
        # Run
        qaframe.run_qa('FIBERFLAT', (frame, fiberflat))
        # Write
        if args.qafile is not None:
            write_qa_frame(args.qafile, qaframe)
            log.info("successfully wrote {:s}".format(args.qafile))
        # Figure(s)
        if args.qafig is not None:
            qa_plots.frame_fiberflat(args.qafig, qaframe, frame, fiberflat)

    # Write
    write_fiberflat(args.outfile, fiberflat, frame.meta)
    log.info("successfully wrote %s"%args.outfile)


