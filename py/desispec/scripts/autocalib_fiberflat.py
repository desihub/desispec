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
from desispec.io import findfile

import argparse


def parse(options=None):
    parser = argparse.ArgumentParser(description="Merge fiber flats from different calibration lamps")
    parser.add_argument('-i','--infile', type = str, default = None, required=True, nargs="*")
    parser.add_argument('--prefix', type = str, required=False, default=None, help = "output filename prefix, including directory (one file per spectrograph), default is findfile('fiberflatnight',night,...,cam)")
    parser.add_argument('--night', type = str, required=False, default=None)
    parser.add_argument('--arm', type = str, required=False, default=None, help="b, r or z")
    
    
    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args

def main(args) :
    
    log=get_logger()
    if ( args.night is None or args.arm is None ) and args.prefix is None :
        log.error("ERROR in arguments, need night and arm or prefix for output file names")
        return
    
    log=get_logger()
    log.info("starting at {}".format(time.asctime()))
    inputs=[]
    for filename in args.infile :
        inputs.append(read_fiberflat(filename))
    fiberflats = autocalib_fiberflat(inputs)
    for spectro in fiberflats.keys() :
        if args.prefix :
            ofilename="{}{}-autocal.fits".format(args.prefix,spectro)
        else :
            camera="{}{}".format(args.arm,spectro)
            ofilename=findfile('fiberflatnight', args.night, 0 , camera)
        write_fiberflat(ofilename,fiberflats[spectro])
        log.info("successfully wrote %s"%ofilename)
    
