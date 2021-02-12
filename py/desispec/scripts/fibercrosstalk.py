"""
desispec.scripts.trace_shifts
=============================
"""
from __future__ import absolute_import, division

import sys
import argparse
import numpy as np
from scipy.signal import fftconvolve
from astropy.table import Table
import matplotlib.pyplot as plt

from desiutil.log import get_logger
from desispec.io import read_frame,read_xytraceset
from desispec.interpolation import resample_flux
from desispec.calibfinder import findcalibfile
from desispec.fibercrosstalk import compute_fiber_crosstalk_kernels

def parse(options=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="""Measures the optical crosstalk between adjucent fibers using sky subtracted frame files.""")

    parser.add_argument('-i','--infile', type = str, default = None, required=True, nargs = "*",
                        help = 'path to sky subtracted (s)frame files')
    parser.add_argument('-o','--outfile', type = str, default = None, required=False,
                        help = 'output fits file')
    parser.add_argument('--fiber-dpix', type = float, default = 7.3, required=False,
                        help = 'distance between adjacent fiber traces in pixels')
    parser.add_argument('--plot', action = "store_true",
                        help = 'plot results')

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args

def main(args) :

    log= get_logger()


    # precompute convolution kernels
    kernels = compute_fiber_crosstalk_kernels()

    A = None
    B = None

    # one measurement per fiber bundle
    nfiber_per_bundle = 25
    nbundles=20
    previous_psf_filename = None
    for filename in args.infile :

        # read a frame and fiber the sky fibers
        frame = read_frame(filename)
        skyfibers = np.where(frame.fibermap["OBJTYPE"]=="SKY")[0]
        log.info("{} sky fibers in {}".format(skyfibers.size,filename))

        frame.ivar *= (frame.mask==0)

        # also open trace set to determine the shift
        # to apply to adjacent spectra
        psf_filename = findcalibfile([frame.meta,],"PSF")

        # only reread if necessary
        if previous_psf_filename is None or previous_psf_filename != psf_filename :
            tset  = read_xytraceset(psf_filename)
            previous_psf_filename = psf_filename

        # will use this y
        central_y = tset.npix_y//2

        mwave = np.mean(frame.wave)

        if A is None :
            A = np.zeros((nbundles,dfiber.size,dfiber.size,frame.wave.size))
            B = np.zeros((nbundles,dfiber.size,frame.wave.size))
        else :
            assert(A.shape[3]==frame.wave.size)

        for skyfiber in skyfibers :
            cflux = np.zeros((dfiber.size,frame.wave.size))
            skyfiberbundle=skyfiber//nfiber_per_bundle

            skyfiber_central_wave = tset.wave_vs_y(skyfiber,central_y)

            for i,df in enumerate(dfiber) :
                otherfiber = df+skyfiber
                if otherfiber<0 : continue
                if otherfiber>=frame.nspec : continue
                if otherfiber//nfiber_per_bundle != skyfiberbundle : continue # not same bundle

                medsnr=np.median(np.sqrt(frame.ivar[otherfiber])*(frame.mask[otherfiber]==0)*frame.flux[otherfiber])
                if medsnr<3. : continue # need good SNR to model cross talk

                # interpolate over masked array and shift
                good=(frame.ivar[otherfiber]>0)&(frame.mask[otherfiber]==0)
                if np.sum(good)==0 : continue

                # account for change of wavelength for same y coordinate
                otherfiber_central_wave = tset.wave_vs_y(otherfiber,central_y)
                flux = np.interp(frame.wave+(otherfiber_central_wave-skyfiber_central_wave),frame.wave[good],frame.flux[otherfiber][good])
                kern=kernels[np.abs(df)]
                cflux[i] = fftconvolve(flux,kern,mode="same")
                B[skyfiberbundle,i] += frame.ivar[skyfiber]*cflux[i]*frame.flux[skyfiber]
                for j in range(i+1) :
                    A[skyfiberbundle,i,j] += frame.ivar[skyfiber]*cflux[i]*cflux[j]

    for bundle in range(nbundles) :
        for i in range(dfiber.size) :
            for j in range(i) :
                A[bundle,j,i] = A[bundle,i,j]

    # now solve
    tmp = np.zeros((nbundles,dfiber.size,frame.wave.size))
    tmp_ivar = np.zeros((nbundles,dfiber.size,frame.wave.size))
    for bundle in range(nbundles) :
        for j in range(frame.wave.size) :
            try:
                Ai=np.linalg.inv(A[bundle,:,:,j])
                tmp[bundle,:,j]=Ai.dot(B[bundle,:,j])
                tmp_ivar[bundle,:,j]=1./np.diag(Ai)
            except np.linalg.LinAlgError as e :
                pass
    # rebin
    wave=np.linspace(frame.wave[0],frame.wave[-1],40)
    crosstalk      = np.zeros((nbundles,dfiber.size,wave.size))
    crosstalk_ivar = np.zeros((nbundles,dfiber.size,wave.size))

    for bundle in range(nbundles) :
        for i,df in enumerate(dfiber) :
            crosstalk[bundle,i],crosstalk_ivar[bundle,i]=resample_flux(wave,frame.wave,tmp[bundle,i],tmp_ivar[bundle,i])


    table=Table()
    table["WAVELENGTH"]=wave
    for bundle in range(nbundles) :
        for i,df in enumerate(dfiber) :
            key="CROSSTALK-B{:02d}-F{:+d}".format(bundle,df)
            table[key]=crosstalk[bundle,i]
            key="CROSSTALKIVAR-B{:02d}-F{:+d}".format(bundle,df)
            table[key]=crosstalk_ivar[bundle,i]
    table.write(args.outfile,overwrite=True)
    log.info("wrote {}".format(args.outfile))

    if args.plot :
        for bundle in range(nbundles) :
            for i,df in enumerate(dfiber) :
                err=1./np.sqrt(crosstalk_ivar[bundle,i]+(crosstalk_ivar[bundle,i]==0))
                plt.errorbar(wave,crosstalk[bundle,i],err,fmt="o-",label="bundle = {:02d} dfiber = {:+d}".format(bundle,df))
        plt.grid()
        plt.legend()
        plt.show()
