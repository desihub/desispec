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
import scipy.ndimage

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

    # precompute convolution kernels
    kernels = compute_crosstalk_kernels()

    A = None
    B = None
    out_wave = None

    dfiber=np.array([-2,-1,1,2])
    #dfiber=np.array([-1,1])

    npar=dfiber.size
    with_cst = True # to marginalize over residual background (should not change much)
    if with_cst :
        npar += 1

    # one measurement per fiber bundle
    nfiber_per_bundle = 25
    nbundles = 500//nfiber_per_bundle

    xtalks=[]

    previous_psf_filename = None
    for filename in args.infile :

        # read a frame and fiber the sky fibers
        frame = read_frame(filename)

        if out_wave is None :
            dwave=(frame.wave[-1]-frame.wave[0])/40
            out_wave=np.linspace(frame.wave[0]+dwave/2,frame.wave[-1]-dwave/2,40)

        skyfibers = np.where((frame.fibermap["OBJTYPE"]=="SKY")&(frame.fibermap["FIBERSTATUS"]==0))[0]
        log.info("{} sky fibers in {}".format(skyfibers.size,filename))

        frame.ivar *= ((frame.mask==0)|(frame.mask==specmask.BADFIBER)) # ignore BADFIBER which is a statement on the positioning

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
            A = np.zeros((nbundles,npar,npar,out_wave.size))
            B = np.zeros((nbundles,npar,out_wave.size))
            fA = np.zeros((npar,npar,out_wave.size))
            fB = np.zeros((npar,out_wave.size))
            ninput=np.zeros((nbundles,dfiber.size))

        for skyfiber in skyfibers :
            cflux = np.zeros((dfiber.size,out_wave.size))
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

            use_median_filter = False # not needed
            median_filter_width=30
            skyfiberflux,skyfiberivar = resample_flux(out_wave,frame.wave,frame.flux[skyfiber],frame.ivar[skyfiber])

            if use_median_filter :
                good=(skyfiberivar>0)
                skyfiberflux=np.interp(out_wave,out_wave[good],skyfiberflux[good])
                skyfiberflux=scipy.ndimage.filters.median_filter(skyfiberflux,median_filter_width,mode='constant')


            for i,df in enumerate(dfiber) :
                otherfiber = df+skyfiber
                if otherfiber<0 : continue
                if otherfiber>=frame.nspec : continue
                if otherfiber//nfiber_per_bundle != skyfiberbundle : continue # not same bundle


                snr=np.sqrt(frame.ivar[otherfiber])*frame.flux[otherfiber]
                medsnr=np.median(snr)
                if medsnr>2 :  # need good SNR to model cross talk
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
                if use_median_filter :
                    flux = scipy.ndimage.filters.median_filter(flux,median_filter_width,mode='constant')
                kern=kernels[np.abs(df)]
                tmp = fftconvolve(flux,kern,mode="same")
                cflux[i] = resample_flux(out_wave,frame.wave,tmp)

                fB[i] = skyfiberivar*cflux[i]*skyfiberflux
                for j in range(i+1) :
                    fA[i,j] = skyfiberivar*cflux[i]*cflux[j]

            if should_consider and ( not must_exclude ) :

                scflux=np.sum(cflux,axis=0)
                mscflux=np.sum(skyfiberivar*scflux)/np.sum(skyfiberivar)
                if mscflux < 100 :
                    continue

                if with_cst :
                    i=dfiber.size
                    fA[i,i] = skyfiberivar # constant term
                    fB[i] = skyfiberivar*skyfiberflux
                    for j in range(i) :
                        fA[i,j] = skyfiberivar*cflux[j]

                # just stack all wavelength to get 1 number for this fiber
                scflux=np.sum(cflux[np.abs(dfiber)==1],axis=0)
                a=np.sum(skyfiberivar*scflux**2)
                b=np.sum(skyfiberivar*scflux*skyfiberflux)
                xtalk=b/a
                err=1./np.sqrt(a)
                msky=np.sum(skyfiberivar*skyfiberflux)/np.sum(skyfiberivar)
                ra=frame.fibermap["TARGET_RA"][skyfiber]
                dec=frame.fibermap["TARGET_DEC"][skyfiber]

                if np.abs(xtalk)>0.02 and np.abs(xtalk)/err>5 :
                    log.warning("discard skyfiber = {}, xtalk = {:4.3f} +- {:4.3f}, ra = {:5.4f} , dec = {:5.4f}, sky fiber flux= {:4.3f}, cont= {:4.3f}".format(skyfiber,xtalk,err,ra,dec,msky,mscflux))
                    continue

                if err<0.01/5. :
                    xtalks.append(xtalk)

                for i in range(dfiber.size) :
                    ninput[skyfiberbundle,i] += int(np.sum(fB[i])!=0) # to monitor
                B[skyfiberbundle] += fB
                A[skyfiberbundle] += fA


    for bundle in range(nbundles) :
        for i in range(npar) :
            for j in range(i) :
                A[bundle,j,i] = A[bundle,i,j]

    # now solve
    crosstalk = np.zeros((nbundles,dfiber.size,out_wave.size))
    crosstalk_ivar = np.zeros((nbundles,dfiber.size,out_wave.size))
    for bundle in range(nbundles) :
        for j in range(out_wave.size) :
            try:
                Ai=np.linalg.inv(A[bundle,:,:,j])
                if with_cst :
                    crosstalk[bundle,:,j]=Ai.dot(B[bundle,:,j])[:-1] # last coefficient is constant
                    crosstalk_ivar[bundle,:,j]=1./np.diag(Ai)[:-1]
                else :
                    crosstalk[bundle,:,j]=Ai.dot(B[bundle,:,j])
                    crosstalk_ivar[bundle,:,j]=1./np.diag(Ai)

            except np.linalg.LinAlgError as e :
                pass

    table=Table()
    table["WAVELENGTH"]=out_wave
    for bundle in range(nbundles) :
        for i,df in enumerate(dfiber) :
            key="CROSSTALK-B{:02d}-F{:+d}".format(bundle,df)
            table[key]=crosstalk[bundle,i]
            key="CROSSTALKIVAR-B{:02d}-F{:+d}".format(bundle,df)
            table[key]=crosstalk_ivar[bundle,i]
            key="NINPUT-B{:02d}-F{:+d}".format(bundle,df)
            table[key]=np.repeat(ninput[bundle,i],out_wave.size)

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
