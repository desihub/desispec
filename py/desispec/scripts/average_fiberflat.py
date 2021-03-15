"""
desispec.scripts.average_fiberflat
==================================

"""
from __future__ import absolute_import, division
import time

import numpy as np
from desiutil.log import get_logger
from desispec.io import read_fiberflat,write_fiberflat
from desispec.fiberflat import average_fiberflat
import argparse


def parse(options=None):
    parser = argparse.ArgumentParser(description="Average fiber flats")
    parser.add_argument('-i','--infile', type = str, default = None, required=True, nargs="*")
    parser.add_argument('-o','--outfile', type = str, default = None, required=True)
    parser.add_argument('--program', type = str, default = None, required=False, help="only input with this program")
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
        inflat=read_fiberflat(filename)
        if args.program is not None :
            if args.program != inflat.header["PROGRAM"] :
                log.info("skip {}".format(filename))
                continue
                
        inputs.append(read_fiberflat(filename))
    fiberflat = average_fiberflat(inputs)
    write_fiberflat(args.outfile,fiberflat)
    log.info("successfully wrote %s"%args.outfile)
    
