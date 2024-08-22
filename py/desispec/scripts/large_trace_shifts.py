"""
desispec.scripts.large_trace_shifts
===================================

"""

import os, sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from desispec.io.xytraceset import read_xytraceset
from desispec.io import read_image
from desiutil.log import get_logger
from desispec.large_trace_shifts import detect_spots_in_image,match_same_system
from desispec.trace_shifts import write_traces_in_psf

def parse(options=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="""Find large trace shifts by matching arc lamp spots in preprocessed images.""")
    parser.add_argument('--ref-image', type = str, default = None, required=True,
                        help = 'path of DESI reference preprocessed arc lamps fits image')
    parser.add_argument('-i','--image', type = str, default = None, required=True,
                        help = 'path of DESI preprocessed arc lamps fits image')
    parser.add_argument('--ref-psf', type = str, default = None, required=False,
                        help = 'path of DESI psf fits file corresponding to the reference image')
    parser.add_argument('-o','--output-psf', type = str, default = None, required=False,
                        help = 'path of output shifted psf file')
    parser.add_argument('--plot',action='store_true', help="plot spots")

    args = parser.parse_args(options)
    return args

def main(args=None) :

    log= get_logger()

    if not isinstance(args, argparse.Namespace):
        args = parse(args)


    ref_image = read_image(args.ref_image)
    xref,yref = detect_spots_in_image(ref_image)

    in_image  = read_image(args.image)
    xin,yin = detect_spots_in_image(in_image)

    indices,distances = match_same_system(xref,yref,xin,yin,remove_duplicates=True)

    ok=(indices>=0)

    rmsdist = 1.4*np.median(distances[ok])
    ok &= (distances<5*rmsdist)

    nmatch = np.sum(ok)
    if nmatch<10 :
        message = "too few matches: {}. Aborting.".format(nmatch)
        log.error(message)
        sys.exit(12)
    xref=xref[ok]
    yref=yref[ok]
    xin=xin[indices[ok]]
    yin=yin[indices[ok]]


    delta_x = np.median(xin-xref)
    delta_y = np.median(yin-yref)
    log.info("First delta_x = {:.2f} delta_y = {:.2f}".format(delta_x,delta_y))

    distances = (xin-xref-delta_x)**2+(yin-yref-delta_y)**2
    rmsdist = 1.4*np.median(distances)
    ok = (distances<5*rmsdist)
    nmatch = np.sum(ok)
    if nmatch<10 :
        message = "too few matches: {}. Aborting.".format(nmatch)
        log.error(message)
        sys.exit(12)

    xref=xref[ok]
    yref=yref[ok]
    xin=xin[ok]
    yin=yin[ok]
    delta_x = np.median(xin-xref)
    delta_y = np.median(yin-yref)
    distances = (xin-xref-delta_x)**2+(yin-yref-delta_y)**2
    rms_dist = np.sqrt(np.mean(distances**2))
    log.info("Refined delta_x = {:.2f} delta_y = {:.2f} rms dist = {:.2f}".format(delta_x,delta_y,rms_dist))


    if args.ref_psf is not None :
        log.info("Read traceset in {}".format(args.ref_psf))
        tset = read_xytraceset(args.ref_psf)
        tset.x_vs_wave_traceset._coeff[:,0] += delta_x
        tset.y_vs_wave_traceset._coeff[:,0] += delta_y

        if args.output_psf is not None :
            log.info("Write modified traceset in {}".format(args.output_psf))
            write_traces_in_psf(args.ref_psf,args.output_psf,tset)

    if args.plot :
        if 0 :
            plt.figure()
            plt.plot(xref,yref,".")
            plt.plot(xin,yin,".")

        plt.figure()
        plt.subplot(221)
        plt.plot(xref,xin-xref,".")
        plt.axhline(delta_x,linestyle="--")
        plt.xlabel("X")
        plt.ylabel("dX")
        plt.subplot(222)
        plt.plot(yref,xin-xref,".")
        plt.axhline(delta_x,linestyle="--")
        plt.xlabel("Y")
        plt.ylabel("dX")
        plt.subplot(223)
        plt.plot(xref,yin-yref,".")
        plt.axhline(delta_y,linestyle="--")
        plt.xlabel("X")
        plt.ylabel("dY")
        plt.subplot(224)
        plt.plot(yref,yin-yref,".")
        plt.axhline(delta_y,linestyle="--")
        plt.xlabel("Y")
        plt.ylabel("dY")

        plt.figure()
        plt.plot(xref,yref,"X",color="C0")
        plt.plot(xin,yin,".",color="red",alpha=0.7)
        plt.plot(xin-delta_x,yin-delta_y,".",color="C1")

        plt.show()
