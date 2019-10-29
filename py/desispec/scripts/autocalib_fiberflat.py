"""
desispec.scripts.average_fiberflat
==================================

"""
from __future__ import absolute_import, division
import time

import numpy as np
from desiutil.log import get_logger
from desispec.io import read_fiberflat,write_fiberflat
from desispec.fiberflat import autocalib_fiberflat
import argparse


def parse(options=None):
    parser = argparse.ArgumentParser(description="Merge fiber flats from different calibration lamps")
    parser.add_argument('-i','--infile', type = str, default = None, required=True, nargs="*")
    parser.add_argument('--prefix', type = str, required=False, default="./fiberflat-", help = "output filename prefix, including directory (one file per spectrograph)")
    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args

def main(args) :

    log=get_logger()
    log.info("starting at {}".format(time.asctime()))
    inputs=[]
    for filename in args.infile :
        inputs.append(read_fiberflat(filename))
    fiberflats = autocalib_fiberflat(inputs)
    for spectro in fiberflats.keys() :
        ofilename="{}{}-autocal.fits".format(args.prefix,spectro)
        write_fiberflat(ofilename,fiberflats[spectro])
        log.info("successfully wrote %s"%ofilename)
    
