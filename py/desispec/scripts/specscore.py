

"""
Compute some information scores on spectra in frames
"""


import argparse

import numpy as np
from astropy.io import fits

from desispec import io
from desiutil.log import get_logger
from desispec.specscore import compute_and_append_frame_scores
from desispec.io.util import write_bintable
from desispec.io  import write_frame

def parse(options=None):
    parser = argparse.ArgumentParser(description="Add or modify SCORES HDU in a frame or cframe fits.")
    parser.add_argument('-i', '--infile', type = str, default = None, required=True, nargs='*',
                        help = 'list of path to DESI frame fits files')
    parser.add_argument('--overwrite', action="store_true",
                        help = 'The HDU SCORES is overwritten if it already exists in the file')
    parser.add_argument('--calibrated', action="store_true",
                        help = 'Are the fluxes calibrated')
    parser.add_argument('--suffix', type = str, default = None, required=False,
                        help = 'suffix added to the column name in the SCORES table to describe the level of processing of the spectra in the frame. For instance "RAW","FFLAT","SKYSUB","CALIB"')
                       
    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args

def main(args) :

    log = get_logger()

    for filename in args.infile :
        
        log.info("reading %s"%filename)
        frame=io.read_frame(filename)        
        scores,comments=compute_and_append_frame_scores(frame,suffix=args.suffix,calibrated=args.calibrated,overwrite=args.overwrite)
        log.info("Adding or replacing SCORES extention with {} in {}".format(scores.keys(),filename))
        write_bintable(filename,data=scores,comments=comments,extname="SCORES",clobber=True)        
        #write_frame(filename,frame) # an alternative 
