
from __future__ import absolute_import, division

from desispec.io import read_frame
from desispec.io import read_fiberflat
from desispec.io import write_sky
from desispec.io.qa import load_qa_frame
from desispec.io import write_qa_frame
from desispec.fiberflat import apply_fiberflat
from desispec.sky import compute_sky
from desispec.qa import qa_plots
from desispec.log import get_logger
import argparse
import numpy as np


def parse(options=None):
    parser = argparse.ArgumentParser(description="Compute the sky model.")

    parser.add_argument('--infile', type = str, default = None, required=True,
                        help = 'path of DESI exposure frame fits file')
    parser.add_argument('--fiberflat', type = str, default = None, required=True,
                        help = 'path of DESI fiberflat fits file')
    parser.add_argument('--outfile', type = str, default = None, required=True,
                        help = 'path of DESI sky fits file')
    parser.add_argument('--qafile', type = str, default = None, required=False,
                        help = 'path of QA file. Will calculate for Sky Subtraction')
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

    # read exposure to load data and get range of spectra
    frame = read_frame(args.infile)
    specmin, specmax = np.min(frame.fibers), np.max(frame.fibers)

    # read fiberflat
    fiberflat = read_fiberflat(args.fiberflat)

    # apply fiberflat to sky fibers
    apply_fiberflat(frame, fiberflat)

    # compute sky model
    skymodel = compute_sky(frame)

    # QA
    if (args.qafile is not None) or (args.qafig is not None):
        log.info("performing skysub QA")
        # Load
        qaframe = load_qa_frame(args.qafile, frame, flavor=frame.meta['FLAVOR'])
        # Run
        qaframe.run_qa('SKYSUB', (frame, skymodel))
        # Write
        if args.qafile is not None:
            write_qa_frame(args.qafile, qaframe)
            log.info("successfully wrote {:s}".format(args.qafile))
        # Figure(s)
        if args.qafig is not None:
            qa_plots.frame_skyres(args.qafig, frame, skymodel, qaframe)

    # write result
    write_sky(args.outfile, skymodel, frame.meta)
    log.info("successfully wrote %s"%args.outfile)


