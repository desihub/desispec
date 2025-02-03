"""
desispec.scripts.trace_shifts
=============================

"""

import os, sys
import argparse
import numpy as np
from numpy.linalg.linalg import LinAlgError
import astropy.io.fits as pyfits
from numpy.polynomial.legendre import legval,legfit
from importlib import resources

import specter.psf

from desispec.xytraceset import XYTraceSet
from desispec.io.xytraceset import read_xytraceset
from desispec.io import read_image
from desispec.io import shorten_filename
from desiutil.log import get_logger
from desispec.trace_shifts import write_traces_in_psf,compute_dx_from_cross_dispersion_profiles,compute_dy_from_spectral_cross_correlation,monomials,polynomial_fit,compute_dy_using_boxcar_extraction,compute_dx_dy_using_psf,shift_ycoef_using_external_spectrum,recompute_legendre_coefficients,recompute_legendre_coefficients_for_x,recompute_legendre_coefficients_for_y



def parse(options=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="""Measures trace shifts from a preprocessed image and an input psf, and writes the modified trace coordinates in an output psf file to be used for extractions.
Two methods are implemented.
1) cross-correlation : dx shifts are measured from cross-dispersion profiles of traces.
   dy shifts (wavelength calibration) are measured in two steps, an internal calibration determined from the cross-correlation of fiber spectra obtained from a resampled boxcar extraction with their median, and the final wavelength calibration is obtained from the  cross-correlation of the median fiber spectrum (after a second boxcar extraction) with an external spectrum (given with --spectrum option).
   This method is efficient for measuring trace shifts on science exposures with a large sky background.
2) forward model : dy,dy shifts are determined simultaneously by a forward modeling of the image around a given external list of lines (--lines option).
   This method is in principle statistically optimal, but it is slow and cannot be applied to blended and broad sky lines. It is useful to shift traces from arc lamp images (though specex does the same thing in C++).""")

    parser.add_argument('-i','--image', type = str, default = None, required=True,
                        help = 'path of DESI preprocessed fits image')
    parser.add_argument('--psf', type = str, default = None, required=True,
                        help = 'path of DESI psf fits file')
    parser.add_argument('--lines', type = str, default = None, required=False,
                        help = 'path of lines ASCII file. Using this option changes the fit method.')
    parser.add_argument('--spectrum', type = str, default = None, required=False,
                        help = 'path to an spectrum ASCII file for external wavelength calibration')
    parser.add_argument('--sky', action = 'store_true',
                        help = 'use sky spectrum desispec/data/spec-sky.dat for external wavelength calibration')
    parser.add_argument('--arc-lamps', action = 'store_true',
                        help = 'use arc lamps spectrum desispec/data/spec-arc-lamps.dat for external wavelength calibration')
    parser.add_argument('-o','--outpsf', type = str, default = None, required=True,
                        help = 'path of output PSF with shifted traces')
    parser.add_argument('--outoffsets', type = str, default = None, required=False,
                        help = 'path of output ASCII file with measured offsets for QA')
    parser.add_argument('--degxx', type = int, default = 2, required=False,
                        help = 'polynomial degree for x shifts along x')
    parser.add_argument('--degxy', type = int, default = 2, required=False,
                        help = 'polynomial degree for x shifts along y')
    parser.add_argument('--degyx', type = int, default = 2, required=False,
                        help = 'polynomial degree for y shifts along x')
    parser.add_argument('--degyy', type = int, default = 2, required=False,
                        help = 'polynomial degree for y shifts along y')
    parser.add_argument('--continuum', action='store_true',
                        help = 'only fit shifts along x for continuum input image')
    parser.add_argument('--auto', action='store_true',
                        help = 'choose best method (sky,continuum or just internal calib) from the FLAVOR keyword in the input image header')

    parser.add_argument('--nfibers', type = int, default = None, required=False,
                        help = 'limit the number of fibers for debugging')
    parser.add_argument('--max-error', type = float, default = 0.05 , required=False,
                        help = "max statistical error threshold to automatically lower polynomial degree")
    parser.add_argument('--width', type = int, default = 7 , required=False,
                        help = "width of cross-dispersion profile")
    parser.add_argument('--ccd-rows-rebin', type = int, default = 4 , required=False,
                        help = "rebinning of CCD rows to run faster")

    args = parser.parse_args(options)

    return args


def read_specter_psf(filename) :
    hdulist = pyfits.open(filename)
    head=hdulist[0].header
    if "PSFTYPE" not in head :
        raise KeyError("No PSFTYPE in PSF header, cannot load this PSF in specter")
    psftype=head["PSFTYPE"]
    hdulist.close()

    if psftype=="GAUSS-HERMITE" :
        psf = specter.psf.GaussHermitePSF(filename)
    elif psftype=="SPOTGRID" :
        psf = specter.psf.SpotGridPSF(filename)
    else :
        raise ValueError("Unknown PSFTYPE='{}'".format(psftype))
    return psf

def fit_xcoeff_ycoeff(dx,ex,x_for_dx,y_for_dx,degxx,degxy,
                      dy,ey,x_for_dy,y_for_dy,degyx,degyy,tset,max_error) :

    log     = get_logger()
    xcoef   = tset.x_vs_wave_traceset._coeff
    ycoef   = tset.y_vs_wave_traceset._coeff
    nfibers = xcoef.shape[0]

    n = 0
    nloops = max(degxx, degyx) + max(degxy, degyy)
    while True: # loop because polynomial degrees could be reduced

        # Try fitting offsets.
        log.info("polynomial fit of measured offsets with degx=(%d,%d) degy=(%d,%d)"%(degxx,degxy,degyx,degyy))
        try :
            dx_coeff,dx_coeff_covariance,dx_errorfloor,dx_mod,dx_mask=polynomial_fit(z=dx,ez=ex,xx=x_for_dx,yy=y_for_dx,degx=degxx,degy=degxy)
            dy_coeff,dy_coeff_covariance,dy_errorfloor,dy_mod,dy_mask=polynomial_fit(z=dy,ez=ey,xx=x_for_dy,yy=y_for_dy,degx=degyx,degy=degyy)

            log.info("dx dy error floor = %4.3f %4.3f pixels"%(dx_errorfloor,dy_errorfloor))

            #log.info("check fit uncertainties are ok on edge of CCD")

            merr=0.
            for fiber in [0,nfibers-1] :
                for rw in [-1,1] :
                    tx = legval(rw,xcoef[fiber])
                    ty = legval(rw,ycoef[fiber])
                    m=monomials(tx,ty,degxx,degxy)
                    tdx=np.inner(dx_coeff,m)
                    tsx=np.sqrt(np.inner(m,dx_coeff_covariance.dot(m)))
                    m=monomials(tx,ty,degyx,degyy)
                    tdy=np.inner(dy_coeff,m)
                    tsy=np.sqrt(np.inner(m,dy_coeff_covariance.dot(m)))
                    merr=max(merr,tsx)
                    merr=max(merr,tsy)
            log.info("max edge shift error = %4.3f pixels"%merr)
            if degxx==0 and degxy==0 and degyx==0 and degyy==0 :
                break

        except ( LinAlgError , ValueError ) :
            log.warning("polynomial fit failed with degx=(%d,%d) degy=(%d,%d)"%(degxx,degxy,degyx,degyy))
            if degxx==0 and degxy==0 and degyx==0 and degyy==0 :
                log.error("polynomial degrees are already 0. we can't fit the offsets")
                raise RuntimeError("polynomial degrees are already 0. we can't fit the offsets")
            merr = 100000. # this will lower the pol. degree.

        if merr > max_error :
            if merr != 100000. :
                log.warning("max edge shift error = %4.3f pixels is too large, reducing degrees"%merr)

            if (degxy>0 or degyy>0) and (degxy>degxx or degyy>degyx): # first along wavelength
                if degxy>0 : degxy-=1
                if degyy>0 : degyy-=1
            else :                                                  # then along fiber
                if degxx>0 : degxx-=1
                if degyx>0 : degyx-=1
        else :
            # error is ok, so we quit the loop
            break

        # Sanity check to ensure looping is not infinite.
        n += 1
        if n > nloops:
            raise RuntimeError(f'Maximum fit iterations {nloops} exceeded.')
    # print central shift
    mx=np.median(x_for_dx)
    my=np.median(y_for_dx)
    m=monomials(mx,my,degxx,degxy)
    mdx=np.inner(dx_coeff,m)
    mex=np.sqrt(np.inner(m,dx_coeff_covariance.dot(m)))

    mx=np.median(x_for_dy)
    my=np.median(y_for_dy)
    m=monomials(mx,my,degyx,degyy)
    mdy=np.inner(dy_coeff,m)
    mey=np.sqrt(np.inner(m,dy_coeff_covariance.dot(m)))
    log.info("central shifts dx = %4.3f +- %4.3f dy = %4.3f +- %4.3f "%(mdx,mex,mdy,mey))

    new_xcoeff , new_ycoeff = recompute_legendre_coefficients(xcoef=tset.x_vs_wave_traceset._coeff,
                                                              ycoef=tset.y_vs_wave_traceset._coeff,
                                                              wavemin=tset.wavemin,
                                                              wavemax=tset.wavemax,
                                                              degxx=degxx,degxy=degxy,degyx=degyx,degyy=degyy,
                                                              dx_coeff=dx_coeff,dy_coeff=dy_coeff)

    return new_xcoeff , new_ycoeff

def fit_xcoeff(dx,ex,x_for_dx,y_for_dx,degxx,degxy,tset,max_error) :

    log     = get_logger()
    xcoef   = tset.x_vs_wave_traceset._coeff
    ycoef   = tset.x_vs_wave_traceset._coeff
    nfibers = xcoef.shape[0]

    n = 0
    nloops = degxx + degxy
    while True: # loop because polynomial degrees could be reduced

        # Try fitting offsets.
        log.debug("polynomial fit of measured offsets with degx=(%d,%d)"%(degxx,degxy))
        try :
            dx_coeff,dx_coeff_covariance,dx_errorfloor,dx_mod,dx_mask=polynomial_fit(z=dx,ez=ex,xx=x_for_dx,yy=y_for_dx,degx=degxx,degy=degxy)
            if dx_errorfloor>0.1 :
                log.info("dx error floor = %4.3f pixels"%(dx_errorfloor))

            log.debug("check fit uncertainties are ok on edge of CCD")

            merr=0.
            for fiber in [0,nfibers-1] :
                for rw in [-1,1] :
                    tx = legval(rw,xcoef[fiber])
                    ty = legval(rw,ycoef[fiber])
                    m=monomials(tx,ty,degxx,degxy)
                    tdx=np.inner(dx_coeff,m)
                    tsx=np.sqrt(np.inner(m,dx_coeff_covariance.dot(m)))
                    merr=max(merr,tsx)
            log.debug("max edge shift error = %4.3f pixels"%merr)
            if degxx==0 and degxy==0  :
                break

        except ( LinAlgError , ValueError ) :
            log.warning("polynomial fit failed with degx=(%d,%d)"%(degxx,degxy))
            if degxx==0 and degxy==0  :
                log.error("polynomial degrees are already 0. we can't fit the offsets")
                raise RuntimeError("polynomial degrees are already 0. we can't fit the offsets")
            merr = 100000. # this will lower the pol. degree.

        if merr > max_error :
            if merr != 100000. :
                log.warning("max edge shift error = %4.3f pixels is too large, reducing degrees"%merr)

            if degxy>0 and degxy>degxx : # first along wavelength
                if degxy>0 : degxy-=1
            else :                                                  # then along fiber
                if degxx>0 : degxx-=1
        else :
            # error is ok, so we quit the loop
            break

        # Sanity check to ensure looping is not infinite.
        n += 1
        if n > nloops:
            raise RuntimeError(f'Maximum fit iterations {nloops} exceeded.')
    # print central shift
    mx=np.median(x_for_dx)
    my=np.median(y_for_dx)
    m=monomials(mx,my,degxx,degxy)
    mdx=np.inner(dx_coeff,m)
    mex=np.sqrt(np.inner(m,dx_coeff_covariance.dot(m)))
    log.info("central shifts dx = %4.3f +- %4.3f"%(mdx,mex))

    new_xcoeff = recompute_legendre_coefficients_for_x(xcoef=tset.x_vs_wave_traceset._coeff,
                                                       ycoef=tset.y_vs_wave_traceset._coeff,
                                                       wavemin=tset.wavemin,
                                                       wavemax=tset.wavemax,
                                                       degxx=degxx,degxy=degxy,
                                                       dx_coeff=dx_coeff)

    return new_xcoeff

def fit_ycoeff(dy,ey,x_for_dy,y_for_dy,degyx,degyy,tset,max_error) :

    log     = get_logger()
    xcoef   = tset.x_vs_wave_traceset._coeff
    ycoef   = tset.y_vs_wave_traceset._coeff
    nfibers = xcoef.shape[0]

    n = 0
    nloops = degyx + degyy
    while True: # loop because polynomial degrees could be reduced

        # Try fitting offsets.
        log.info("polynomial fit of measured offsets with degy=(%d,%d)"%(degyx,degyy))
        try :
            dy_coeff,dy_coeff_covariance,dy_errorfloor,dy_mod,dy_mask=polynomial_fit(z=dy,ez=ey,xx=x_for_dy,yy=y_for_dy,degx=degyx,degy=degyy)

            if dy_errorfloor>0.1 :
                log.info("dx error floor = %4.3f pixels"%(dy_errorfloor))

            #log.info("check fit uncertainties are ok on edge of CCD")

            merr=0.
            for fiber in [0,nfibers-1] :
                for rw in [-1,1] :
                    tx = legval(rw,xcoef[fiber])
                    ty = legval(rw,ycoef[fiber])
                    m=monomials(tx,ty,degyx,degyy)
                    tdy=np.inner(dy_coeff,m)
                    tsy=np.sqrt(np.inner(m,dy_coeff_covariance.dot(m)))
                    merr=max(merr,tsy)
            log.info("max edge shift error = %4.3f pixels"%merr)
            if degyx==0 and degyy==0 :
                break

        except ( LinAlgError , ValueError ) :
            log.warning("polynomial fit failed with degy=(%d,%d)"%(degyx,degyy))
            if degyx==0 and degyy==0 :
                log.error("polynomial degrees are already 0. we can't fit the offsets")
                raise RuntimeError("polynomial degrees are already 0. we can't fit the offsets")
            merr = 100000. # this will lower the pol. degree.

        if merr > max_error :
            if merr != 100000. :
                log.warning("max edge shift error = %4.3f pixels is too large, reducing degrees"%merr)

            if degyy>0 and degyy>degyx: # first along wavelength
                if degyy>0 : degyy-=1
            else :                                                  # then along fiber
                if degyx>0 : degyx-=1
        else :
            # error is ok, so we quit the loop
            break

        # Sanity check to ensure looping is not infinite.
        n += 1
        if n > nloops:
            raise RuntimeError(f'Maximum fit iterations {nloops} exceeded.')
    # print central shift
    mx=np.median(x_for_dy)
    my=np.median(y_for_dy)
    m=monomials(mx,my,degyx,degyy)
    mdy=np.inner(dy_coeff,m)
    mey=np.sqrt(np.inner(m,dy_coeff_covariance.dot(m)))
    log.info("central shifts dy = %4.3f +- %4.3f "%(mdy,mey))

    new_ycoeff = recompute_legendre_coefficients_for_y(xcoef=tset.x_vs_wave_traceset._coeff,
                                                       ycoef=tset.y_vs_wave_traceset._coeff,
                                                       wavemin=tset.wavemin,
                                                       wavemax=tset.wavemax,
                                                       degyx=degyx,degyy=degyy,
                                                       dy_coeff=dy_coeff)

    return new_ycoeff

def fit_trace_shifts(image, args):
    """
    Perform the fitting of shifts of spectral traces
    This consists of two steps, one is internal, by
    cross-correlating spectra to themselves, and then
    cross-correlating to external (ususally sky) spectrum

    Return updated traceset and two dictionaies with offset information
    to be written in the PSF file
    """
    global psfs

    log=get_logger()

    log.info("starting")

    tset = read_xytraceset(args.psf)
    wavemin = tset.wavemin
    wavemax = tset.wavemax
    xcoef   = tset.x_vs_wave_traceset._coeff
    ycoef   = tset.y_vs_wave_traceset._coeff

    nfibers=xcoef.shape[0]
    log.info("read PSF trace with xcoef.shape = {} , ycoef.shape = {} , and wavelength range {}:{}".format(xcoef.shape,ycoef.shape,int(wavemin),int(wavemax)))

    lines=None
    if args.lines is not None :
        log.info("We will fit the image using the psf model and lines")

        # read lines
        lines=np.loadtxt(args.lines,usecols=[0])
        ok=(lines>wavemin)&(lines<wavemax)
        log.info("read {} lines in {}, with {} of them in traces wavelength range".format(len(lines),args.lines,np.sum(ok)))
        lines=lines[ok]


    else :
        log.info("We will do an internal calibration of trace coordinates without using the psf shape in a first step")


    internal_wavelength_calib    = (not args.continuum)

    if args.auto :
        log.debug("read flavor of input image {}".format(args.image))
        hdus = pyfits.open(args.image)
        if "FLAVOR" not in hdus[0].header :
            log.error("no FLAVOR keyword in image header, cannot run with --auto option")
            raise KeyError("no FLAVOR keyword in image header, cannot run with --auto option")
        flavor = hdus[0].header["FLAVOR"].strip().lower()
        hdus.close()
        log.info("Input is a '{}' image".format(flavor))
        if flavor == "flat" :
            internal_wavelength_calib = False
        elif flavor == "arc" :
            internal_wavelength_calib = True
            args.arc_lamps = True
        else :
            internal_wavelength_calib = True
            args.sky = True
        log.info("wavelength calib, internal={}, sky={} , arc_lamps={}".format(internal_wavelength_calib,args.sky,args.arc_lamps))

    spectrum_filename = args.spectrum
    continuum_subtract = False
    if args.sky :
        continuum_subtract = True
        srch_file = "data/spec-sky.dat"
        if not resources.files('desispec').joinpath(srch_file).is_file():
            log.error("Cannot find sky spectrum file {:s}".format(srch_file))
            raise RuntimeError("Cannot find sky spectrum file {:s}".format(srch_file))
        spectrum_filename = resources.files('desispec').joinpath(srch_file)
    elif args.arc_lamps :
        srch_file = "data/spec-arc-lamps.dat"
        if not resources.files('desispec').joinpath(srch_file).is_file():
            log.error("Cannot find arc lamps spectrum file {:s}".format(srch_file))
            raise RuntimeError("Cannot find arc lamps spectrum file {:s}".format(srch_file))
        spectrum_filename = resources.files('desispec').joinpath(srch_file)
    if spectrum_filename is not None :
        log.info("Use external calibration from cross-correlation with {}".format(spectrum_filename))

    if args.nfibers is not None :
        nfibers = args.nfibers # FOR DEBUGGING

    fibers=np.arange(nfibers)
    internal_offset_info = None
    specter_psf = None

    if lines is not None :

        # use a forward modeling of the image
        # it's slower and works only for individual lines
        # it's in principle more accurate
        # but gives systematic residuals for complex spectra like the sky

        if specter_psf is None :
            specter_psf = read_specter_psf(args.psf)

        x,y,dx,ex,dy,ey,fiber_xy,wave_xy=compute_dx_dy_using_psf(specter_psf,image,fibers,lines)
        x_for_dx=x
        y_for_dx=y
        fiber_for_dx=fiber_xy
        wave_for_dx=wave_xy
        x_for_dy=x
        y_for_dy=y
        fiber_for_dy=fiber_xy
        wave_for_dy=wave_xy

    else :

        # internal calibration method that does not use the psf
        # nor a prior set of lines. this method is much faster

        degxx=args.degxx
        degxy=args.degxy
        degyx=args.degyx
        degyy=args.degyy

        for iteration in range(10) :
            # measure x shifts
            x_for_dx,y_for_dx,dx,ex,fiber_for_dx,wave_for_dx = compute_dx_from_cross_dispersion_profiles(xcoef,ycoef,wavemin,wavemax, image=image, fibers=fibers, width=args.width, deg=args.degxy,image_rebin=args.ccd_rows_rebin)

            if iteration == 0 :
                # if any quadrant is masked, reduce to a single offset
                hy = image.pix.shape[0] // 2
                hx = image.pix.shape[1] // 2
                allgood = True
                for curxop in [np.less, np.greater]:
                    for curyop in [np.less, np.greater]:
                        allgood &= np.any(curxop(x_for_dx, hx) & curyop(y_for_dx, hy))
                        # some data in this quadrant
                if not allgood :
                    log.warning("No shift data for at least one quadrant of the CCD, falls back to deg=0 shift")
                    degxx=0
                    degxy=0
                    degyx=0
                    degyy=0

            new_xcoeff = fit_xcoeff(dx,ex,x_for_dx,y_for_dx,degxx,degxy,tset,args.max_error)
            maxdx = np.max(np.abs(new_xcoeff[:,0]-xcoef[:,0]))
            tset.x_vs_wave_traceset._coeff = new_xcoeff
            xcoef = tset.x_vs_wave_traceset._coeff
            if maxdx<0.001 :
                break

        if internal_wavelength_calib and (degyx>0 or degyy>0) :

            for iteration in range(10) :
                # measure y shifts
                x_for_dy,y_for_dy,dy,ey,fiber_for_dy,wave_for_dy,dwave,dwave_err  = compute_dy_using_boxcar_extraction(tset, image=image, fibers=fibers, width=args.width, continuum_subtract=continuum_subtract)
                mdy = np.median(dy)
                log.info("Subtract median(dy)={}".format(mdy))
                dy -= mdy # remove median, because this is an internal calibration

                new_ycoeff = fit_ycoeff(dy,ey,x_for_dy,y_for_dy,degyx,degyy,tset,args.max_error)

                meandy = np.mean(new_ycoeff[:,0]-ycoef[:,0])
                maxdy = np.max(np.abs(new_ycoeff[:,0]-ycoef[:,0]-meandy))  # remove mean because of the median dy correction above
                log.info("iteration = {} maxdy={:0.4f}".format(iteration,maxdy))
                tset.y_vs_wave_traceset._coeff = new_ycoeff
                ycoef = tset.y_vs_wave_traceset._coeff

                if maxdy<0.001 :
                    break

                internal_offset_info = dict(wave=wave_for_dy,
                                            fiber=fiber_for_dy,
                                            dwave=dwave,
                                            dwave_err=dwave_err)
        else :
            # duplicate dx results with zero shift to avoid write special case code below
            x_for_dy = x_for_dx.copy()
            y_for_dy = y_for_dx.copy()
            dy       = np.zeros(dx.shape)
            ey       = 1.e-6*np.ones(ex.shape)
            fiber_for_dy = fiber_for_dx.copy()
            wave_for_dy  = wave_for_dx.copy()

    # write this for debugging
    if args.outoffsets :
        file=open(args.outoffsets,"w")
        file.write("# axis wave fiber x y delta error polval (axis 0=y axis1=x)\n")
        for e in range(dy.size) :
            file.write("0 %f %d %f %f %f %f %f\n"%(wave_for_dy[e],fiber_for_dy[e],x_for_dy[e],y_for_dy[e],dy[e],ey[e],dy_mod[e]))
        for e in range(dx.size) :
            file.write("1 %f %d %f %f %f %f %f\n"%(wave_for_dx[e],fiber_for_dx[e],x_for_dx[e],y_for_dx[e],dx[e],ex[e],dx_mod[e]))
        file.close()
        log.info("wrote offsets in ASCII file %s"%args.outoffsets)



    #tset.x_vs_wave_traceset._coeff,tset.y_vs_wave_traceset._coeff = fit_xcoeff_ycoeff(dx,ex,x_for_dx,y_for_dx,degxx,degxy,
    #                                                                                 dy,ey,x_for_dy,y_for_dy,degyx,degyy,tset,args.max_error)

    # compute x y to record max deviations
    wave = np.linspace(tset.wavemin,tset.wavemax,5)
    x0 = np.zeros((tset.nspec,wave.size))
    y0 = np.zeros((tset.nspec,wave.size))
    for s in range(tset.nspec) :
        x0[s]=tset.x_vs_wave(s,wave)
        y0[s]=tset.y_vs_wave(s,wave)



    # use an input spectrum as an external calibration of wavelength
    if spectrum_filename is not None :
        # the psf is used only to convolve the input spectrum
        # the traceset of the psf is not used here
        psf = read_specter_psf(args.psf)
        (tset.y_vs_wave_traceset._coeff,
         wave_external, dwave_external, dwave_err_external) = shift_ycoef_using_external_spectrum(psf=psf, xytraceset=tset,
                                                                             image=image, fibers=fibers,
                                                                             spectrum_filename=spectrum_filename,
                                                                             degyy=args.degyy, width=7)
        external_offset_info = dict(wave=wave_external,
                                    dwave=dwave_external,
                                    dwave_err=dwave_err_external)
    else:
        external_offset_info = None
    x = np.zeros(x0.shape)
    y = np.zeros(x0.shape)
    for s in range(tset.nspec) :
        x[s]=tset.x_vs_wave(s,wave)
        y[s]=tset.y_vs_wave(s,wave)
    dx = x-x0
    dy = y-y0
    if tset.meta is None : tset.meta = dict()
    tset.meta["MEANDX"]=np.mean(dx)
    tset.meta["MINDX"]=np.min(dx)
    tset.meta["MAXDX"]=np.max(dx)
    tset.meta["MEANDY"]=np.mean(dy)
    tset.meta["MINDY"]=np.min(dy)
    tset.meta["MAXDY"]=np.max(dy)

    return tset, internal_offset_info, external_offset_info

def main(args=None) :

    log= get_logger()

    if not isinstance(args, argparse.Namespace):
        args = parse(args)

    log.info("degxx={} degxy={} degyx={} degyy={}".format(args.degxx,args.degxy,args.degyx,args.degyy))

    # read preprocessed image
    image=read_image(args.image)
    log.info("read image {}".format(args.image))
    if image.mask is not None :
        image.ivar *= (image.mask==0)

    if np.all(image.ivar == 0.0):
        log.critical(f"Entire {os.path.basename(args.image)} image is masked; can't fit traceshifts")
        sys.exit(1)

    tset, internal_offset_info, external_offset_info = fit_trace_shifts(image=image, args=args)
    tset.meta['IN_PSF'] = shorten_filename(args.psf)
    tset.meta['IN_IMAGE'] = shorten_filename(args.image)

    if args.outpsf is not None :
        write_traces_in_psf(args.psf,args.outpsf,tset, internal_offset_info=internal_offset_info,
                            external_offset_info=external_offset_info)
        log.info("wrote modified PSF in %s"%args.outpsf)
