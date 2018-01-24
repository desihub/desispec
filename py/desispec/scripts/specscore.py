

"""
Compute some information scores on spectra in frames
"""


import argparse

import numpy as np
from astropy.io import fits

from desispec import io
from desiutil.log import get_logger
from desispec.specscore import compute_frame_scores,append_scores_and_write_frame

def parse(options=None):
    parser = argparse.ArgumentParser(description="Add or modify SCORES HDU to a frame or cframe fits file with simple scores on the spectra.")
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

        # is it a frame or a cframe ... this depends on the units

        
        new_scores,new_comments=compute_frame_scores(frame,suffix=args.suffix,calibrated=args.calibrated)
        
        append_scores_and_write_frame(frame,filename,new_scores,new_comments,args.overwrite)
        
        
