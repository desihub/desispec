#!/usr/bin/env python
#
# See top-level LICENSE file for Copyright information
#
# -*- coding: utf-8 -*-

"""
This script computes the fiber flat field correction from a DESI continuum lamp frame.
"""

from desispec.io import read_frame
from desispec.io import read_fibermap
from desispec.io import read_fiberflat
from desispec.io import write_sky
from desispec.fiberflat import apply_fiberflat
from desispec.sky import compute_sky
from desispec.log import get_logger
import argparse
import os
import os.path
import numpy as np
import sys

def main() :

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--infile', type = str, default = None,
                        help = 'path of DESI exposure frame fits file')
    parser.add_argument('--fibermap', type = str, default = None,
                        help = 'path of DESI exposure frame fits file')
    parser.add_argument('--fiberflat', type = str, default = None,
                        help = 'path of DESI fiberflat fits file')
    parser.add_argument('--outfile', type = str, default = None,
                        help = 'path of DESI sky fits file')


    args = parser.parse_args()

    if args.infile is None:
        print('Missing input')
        parser.print_help()
        sys.exit(12)

    if args.fibermap is None:
        print('Missing fibermap')
        parser.print_help()
        sys.exit(12)

    if args.fiberflat is None:
        print('Missing fiberflat')
        parser.print_help()
        sys.exit(12)

    if args.outfile is None:
        print('Missing output')
        parser.print_help()
        sys.exit(12)

    log=get_logger()

    log.info("starting")

    # read exposure to load data and get range of spectra
    flux,ivar,wave,resol,head = read_frame(args.infile)
    specmin=head["SPECMIN"]
    specmax=head["SPECMAX"]

    # read fibermap to locate sky fibers
    table,fmheader=read_fibermap(args.fibermap)
    selection=np.where((table["OBJTYPE"]=="SKY")&(table["FIBER"]>=specmin)&(table["FIBER"]<=specmax))[0]
    if selection.size == 0 :
        log.error("no sky fiber in fibermap %s"%args.fibermap)
        sys.exit(12)

    # read fiberflat
    fiberflat,ffivar,ffmask,ffmeanspec,ffwave,ffhdr = read_fiberflat(args.fiberflat)

    # apply fiberflat to sky fibers
    apply_fiberflat(flux=flux,ivar=ivar,wave=wave,fiberflat=fiberflat,ffivar=ffivar,ffmask=ffmask,ffwave=ffwave)

    # compute sky model
    skyflux,skyivar,skymask,cskyflux,cskyivar = compute_sky(wave,flux[selection],ivar[selection],resol[selection])

    # write result
    write_sky(args.outfile,skyflux,skyivar,skymask,cskyflux,cskyivar,wave,head)

    log.info("successfully wrote %s"%args.outfile)


if __name__ == '__main__':
    main()
