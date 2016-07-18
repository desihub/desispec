# Script for generating QA from a Production run
from __future__ import absolute_import, division

from desispec.qa import QA_Prod
from desispec.log import get_logger
import argparse
import numpy as np


def parse(options=None):
    parser = argparse.ArgumentParser(description="Generate Production Level QA")

    parser.add_argument('--specprod_dir', type = str, default = None, required=True,
                        help = 'Path containing the exposures/directory to use')
    parser.add_argument('--remake_frame', type = int, default = 0,
                        help = 'Bitwise flag to control remaking the QA files (1) and figures (2) for each frame in the production')
    parser.add_argument('--slurp', type = bool, default = False,
                        help = 'slurp production QA files into one?')
    #parser.add_argument('--qafile', type = str, default = None, required=False,
    #                    help = 'path of QA file. Will calculate for Sky Subtraction')
    #parser.add_argument('--qafig', type = str, default = None, required=False,
    #                    help = 'path of QA figure file')

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args) :

    log=get_logger()

    log.info("starting")

    qa_prod = QA_Prod(args.specprod_dir)

    # Remake Frame QA?
    if args.remake_frame > 0:
        log.info("(re)generating QA related to frames")
        if (args.remake_frame % 4) >= 2:
            remake_plots = True
        else:
            remake_plots = False
        # Run
        qa_prod.remake_frame_qa(remake_plots=remake_plots)

    # Slurp?
    if args.slurp:
        qa_prod.slurp(remake=(args.remake_frame > 0))

