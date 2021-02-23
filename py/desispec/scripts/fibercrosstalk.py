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
from desispec.fibercrosstalk import compute_crosstalk_kernels
from desispec.maskbits import specmask

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

    skylines=np.array([5578.94140625,5656.60644531,6302.08642578,6365.50097656,6618.06054688,7247.02929688,7278.30273438,7318.26904297,7342.94482422,7371.50878906,8346.74902344,8401.57617188,8432.42675781,8467.62011719,8770.36230469,8780.78027344,8829.57421875,8838.796875,8888.29394531,8905.66699219,8922.04785156,8960.48730469,9326.390625,9378.52734375,9442.37890625,9479.48242188,9569.9609375,9722.59082031,9793.796875])
    skymask=None

    # precompute convolution kernels
    kernels = compute_crosstalk_kernels()

    A = None
    B = None

    dfiber=np.array([-2,-1,1,2])

    # one measurement per fiber bundle
    nfiber_per_bundle = 25
    nbundles=20
    previous_psf_filename = None
    for filename in args.infile :

        # read a frame and fiber the sky fibers
        frame = read_frame(filename)

        if skymask is None :
            skymask=np.ones(frame.wave.size)
            for line in skylines :
                skymask[np.abs(frame.wave-line)<2]=0

        skyfibers = np.where(frame.fibermap["OBJTYPE"]=="SKY")[0]
        log.info("{} sky fibers in {}".format(skyfibers.size,filename))

        frame.ivar *= ((frame.mask==0)|(frame.mask==specmask.BADFIBER)) # ignore BADFIBER which is a statement on the positioning
        frame.ivar *= skymask

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
            fA = np.zeros((dfiber.size,dfiber.size,frame.wave.size))
            fB = np.zeros((dfiber.size,frame.wave.size))
            ninput=np.zeros((nbundles,dfiber.size))
        else :
            assert(A.shape[3]==frame.wave.size)

        for skyfiber in skyfibers :
            cflux = np.zeros((dfiber.size,frame.wave.size))
            skyfiberbundle=skyfiber//nfiber_per_bundle

            nbad=np.sum(frame.ivar[skyfiber]==0)
            if nbad>200 :
                if nbad<2000:
                    log.warning("ignore skyfiber {} from {} with {} masked pixel".format(skyfiber,filename,nbad))
                continue

            skyfiber_central_wave = tset.wave_vs_y(skyfiber,central_y)

            should_consider = False
            must_exclude    = False
            fA *= 0.
            fB *= 0.

            for i,df in enumerate(dfiber) :
                otherfiber = df+skyfiber
                if otherfiber<0 : continue
                if otherfiber>=frame.nspec : continue
                if otherfiber//nfiber_per_bundle != skyfiberbundle : continue # not same bundle


                snr=np.sqrt(frame.ivar[otherfiber])*frame.flux[otherfiber]
                medsnr=np.median(snr)
                if medsnr>2. :  # need good SNR to model cross talk
                    should_consider = True # in which case we need all of the contaminants to the sky fiber ...

                nbad=np.sum(snr==0)
                if nbad>200 :
                    if nbad<2000:
                        log.warning("ignore fiber {} from {} with {} masked pixel".format(otherfiber,filename,nbad))
                    must_exclude = True # because 1 bad fiber
                    break

                if np.any(snr>1000.) :
                    log.error("signal to noise is suspiciously too high in fiber {} from {}".format(otherfiber,filename))
                    must_exclude = True # because 1 bad fiber
                    break

                # interpolate over masked pixels or low snr pixels and shift
                medivar=np.median(frame.ivar[otherfiber])
                good=(frame.ivar[otherfiber]>0.01*medivar)# interpolate over brigh sky lines

                # account for change of wavelength for same y coordinate
                otherfiber_central_wave = tset.wave_vs_y(otherfiber,central_y)
                flux = np.interp(frame.wave+(otherfiber_central_wave-skyfiber_central_wave),frame.wave[good],frame.flux[otherfiber][good])
                kern=kernels[np.abs(df)]
                cflux[i] = fftconvolve(flux,kern,mode="same")
                fB[i] = frame.ivar[skyfiber]*cflux[i]*frame.flux[skyfiber]
                for j in range(i+1) :
                    fA[i,j] = frame.ivar[skyfiber]*cflux[i]*cflux[j]

            if should_consider and ( not must_exclude ) :
                for i in range(dfiber.size) :
                    ninput[skyfiberbundle,i] += int(np.sum(fB[i])!=0) # to monitor
                B[skyfiberbundle] += fB
                A[skyfiberbundle] += fA

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
            key="NINPUT-B{:02d}-F{:+d}".format(bundle,df)
            table[key]=np.repeat(ninput[bundle,i],wave.size)

    table.write(args.outfile,overwrite=True)
    log.info("wrote {}".format(args.outfile))

    log.info("number of sky fibers used per bundle:")
    for bundle in range(nbundles) :
        log.info("bundle {}: {}".format(bundle,ninput[bundle]))




    if args.plot :
        for bundle in range(nbundles) :
            for i,df in enumerate(dfiber) :
                err=1./np.sqrt(crosstalk_ivar[bundle,i]+(crosstalk_ivar[bundle,i]==0))
                plt.errorbar(wave,crosstalk[bundle,i],err,fmt="o-",label="bundle = {:02d} dfiber = {:+d}".format(bundle,df))
        plt.grid()
        plt.legend()
        plt.show()
