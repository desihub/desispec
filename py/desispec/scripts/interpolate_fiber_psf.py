#!/usr/bin/env python

import argparse
import numpy as np

import astropy.io.fits as pyfits

from desiutil.log import get_logger
from desispec.io import read_xytraceset,write_xytraceset
from desispec.xytraceset import XYTraceSet
from desispec.util import parse_fibers

def parse(options=None):

    parser = argparse.ArgumentParser(description="Interpolate the trace and PSF parameters from neighboring fibers. This is to get an approximate trace and PSF model for fibers that have 0/low throughput or dark/hot CCD columns along their trace.")

    parser.add_argument('-i','--infile', type = str, default = None, required=True, help = 'input psf fits file')
    parser.add_argument('-o','--outfile', type = str, default = None, required=True, help = 'output psf fits file')
    parser.add_argument('--fibers', type = str, default = None, required=True, help = 'fiber indices (i, or i:j or i,j,k) (for i, a fiber will be inserted between index i-1 and i in the file)')

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args

def main(args=None):

    log=get_logger()

    if args is None :
        args = parse()

    bad_fibers = parse_fibers(args.fibers)

    neighboring_fiber = None

    h = pyfits.open(args.infile)

    nfibers = h["XTRACE"].data.shape[0]

    # coordinates of the middle of the traces
    x = h["XTRACE"].data[:,0]
    y = h["YTRACE"].data[:,0]

    # compute offsets
    dx_in = []
    dx_out = []
    for b in range(20) :
        dx_in.append(np.median(x[b*25+1:(b+1)*25]-x[b*25:(b+1)*25-1]))
        if b>0 :
            dx_out.append(x[b*25]-x[b*25-1])
    dx_in = np.median(dx_in)
    dx_out = np.median(dx_out)

    log.info("dx_in={:.3f}".format(dx_in))
    log.info("dx_out={:.3f}".format(dx_out))

    for bad_fiber in bad_fibers :

        # find a neigboring fiber

        neighboring_fiber = None

        x_of_bad=x[bad_fiber]

        bundle = bad_fiber//25

        if bad_fiber%25 < 12 : # fiber is to the left of the bundle, so choose a neighbor to the right
            step = 1
        else :
            step = -1

        neighboring_fiber = None

        neighboring_fiber_right = bad_fiber+1
        while neighboring_fiber_right in bad_fibers :
            neighboring_fiber_right += 1
        bundle_right=neighboring_fiber_right//25

        neighboring_fiber_left = bad_fiber-1
        while neighboring_fiber_left in bad_fibers :
            neighboring_fiber_left -= 1
        bundle_left=neighboring_fiber_left//25

        # one or the other is off range
        if neighboring_fiber_left<0 :
            if neighboring_fiber_right < nfibers :
                neighboring_fiber = neighboring_fiber_right
            else :
                log.error("sorry, didn't find a good neighbor for fiber {}".format(bad_fiber))
                continue
        else :
            if neighboring_fiber_right >= nfibers :
                neighboring_fiber = neighboring_fiber_left

        # both are in the same bundle
        if neighboring_fiber is None and bundle_right==bundle and bundle_left==bundle :
            if bad_fiber%25 < 12 : # fiber is to the left of the bundle, so choose a neighbor to the right
                neighboring_fiber = neighboring_fiber_right
            else :
                neighboring_fiber = neighboring_fiber_left

        # pick the one that is in the same bundle
        if neighboring_fiber is None :
            if bundle_right==bundle :
                neighboring_fiber = neighboring_fiber_right
            elif bundle_left==bundle :
                neighboring_fiber = neighboring_fiber_left
            else :
                # none is in the same bundle, pick the nearest
                if np.abs(bad_fiber-neighboring_fiber_right) < np.abs(bad_fiber-neighboring_fiber_left) :
                    neighboring_fiber = neighboring_fiber_right
                else :
                    neighboring_fiber = neighboring_fiber_left



        # set default values
        if "PSF" in h :
            h["PSF"].data["COEFF"][:,bad_fiber,:] =  h["PSF"].data["COEFF"][:,neighboring_fiber,:]
        h["XTRACE"].data[bad_fiber] = h["XTRACE"].data[neighboring_fiber]
        h["YTRACE"].data[bad_fiber] = h["YTRACE"].data[neighboring_fiber]

        # adjust x value

        delta_out = bad_fiber//25 - neighboring_fiber//25
        delta_in  = (bad_fiber - neighboring_fiber)-delta_out
        x_of_bad  = x[neighboring_fiber] + delta_in*dx_in + delta_out*dx_out
        h["XTRACE"].data[bad_fiber,0] = x_of_bad

        # adjust y value
        ii=(np.abs(x-x_of_bad)<dx_in*10)&(x!=x_of_bad)
        if np.sum(ii)>2 :
            pol = np.poly1d(np.polyfit(x[ii],y[ii],2))
            y_of_bad = pol(x_of_bad)
            h["YTRACE"].data[bad_fiber,0] = y_of_bad

        log.info("Fixed fiber {} using fiber {} as reference.".format(bad_fiber,neighboring_fiber))

    h.writeto(args.outfile,overwrite=True)
    log.info("wrote {}".format(args.outfile))

    return 0
