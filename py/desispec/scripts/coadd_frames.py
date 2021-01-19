"""
Coadd spectra
"""

from __future__ import absolute_import, division, print_function

import os
import numpy as np
from astropy.table import Table

from desiutil.log import get_logger
from desispec.io import read_frame,write_frame
from desispec.coaddition import coadd,coadd_cameras,resample_spectra_lin_or_log


def parse(options=None):
    import argparse

    parser = argparse.ArgumentParser("Coadd frames")
    parser.add_argument("-i","--infile", type=str, nargs='+', help="input frame files")
    parser.add_argument("-o","--outfile", type=str,  help="output frame file")

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args

def main(args=None):

    log = get_logger()

    if args is None:
        args = parse()


    log.info("reading frames ...")

    frames = []
    for filename in args.infile :
        frame = read_frame(filename)
        if len(frames)==0 :
            frames.append(frame)
            continue

        if "CAMERA" in frames[0].meta and (frame.meta["CAMERA"] != frames[0].meta["CAMERA"]) :
            log.error("ignore {} because not same camera".format(filename))
            continue
        if frame.wave.size != frames[0].wave.size or np.max(np.abs(frame.wave-frames[0].wave))>0.01 :
            log.error("ignore {} because not same wave".format(filename))
            continue
        if frames[0].fibermap is not None :
            if not np.all(frame.fibermap["TARGETID"] == frames[0].fibermap["TARGETID"]) :
                log.error("ignore {} because not same targets".format(filename))
                continue
        frames.append(frame)


    ivar = frames[0].ivar * (frames[0].mask == 0)
    ivarflux = frames[0].ivar * (frames[0].mask == 0) * frames[0].flux
    mask = frames[0].mask
    for frame in frames[1:] :
        ivar += frame.ivar * (frame.mask == 0)
        ivarflux += frame.ivar * (frame.mask == 0) * frame.flux
        mask &= frame.mask # and mask

    coadd = frames[0]
    coadd.ivar = ivar
    coadd.flux = ivarflux/(ivar+(ivar==0))
    coadd.mask = mask

    log.info("writing {} ...".format(args.outfile))
    write_frame(args.outfile,coadd)

    log.info("done")
