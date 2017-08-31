
from __future__ import absolute_import, division


import argparse
import numpy as np
from numpy.linalg.linalg import LinAlgError
import astropy.io.fits as pyfits
from numpy.polynomial.legendre import legval,legfit
from desispec.interpolation import resample_flux

from desispec.io import read_image
from desiutil.log import get_logger
from desispec.linalg import cholesky_solve,cholesky_solve_and_invert
from desispec.interpolation import resample_flux
from desispec.trace_shifts import read_psf,read_traces_in_psf,write_traces_in_psf,boxcar_extraction,compute_dx_from_cross_dispersion_profiles,compute_dy_from_spectral_cross_correlation,monomials,polynomial_fit,compute_fiber_bundle_trace_shifts_using_psf,_u,compute_dy_using_boxcar_extraction,compute_dx_dy_using_psf



def parse(options=None):
    parser = argparse.ArgumentParser(description="Measure trace shifts.")
    
    parser.add_argument('--image', type = str, default = None, required=True,
                        help = 'path of DESI preprocessed fits file')
    parser.add_argument('--psf', type = str, default = None, required=True,
                        help = 'path of DESI psf fits file')
    parser.add_argument('--fit-lines', type = str, default = None, required=False,
                        help = 'path of lines ASCII file. Using this option changes the fit method.')
    parser.add_argument('--outpsf', type = str, default = None, required=True,
                        help = 'path of output PSF with shifted traces')
    parser.add_argument('--outoffsets', type = str, default = None, required=False,
                        help = 'path of output ASCII file with measured offsets')
    parser.add_argument('--degxx', type = int, default = 2, required=False,
                        help = 'polynomial degree for x shifts along x')
    parser.add_argument('--degxy', type = int, default = 2, required=False,
                        help = 'polynomial degree for x shifts along y')
    parser.add_argument('--degyx', type = int, default = 2, required=False,
                        help = 'polynomial degree for y shifts along x')
    parser.add_argument('--degyy', type = int, default = 2, required=False,
                        help = 'polynomial degree for y shifts along y')
    parser.add_argument('--nfibers', type = int, default = None, required=False,
                        help = 'limit the number of fibers for debugging')
    
    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args) :

    global psfs

    log=get_logger()

    log.info("starting")
    
    
    # read preprocessed image
    image=read_image(args.image)
    log.info("read image {}".format(args.image))
    if image.mask is not None :
        image.ivar *= (image.mask==0)

    xcoef=None
    ycoef=None
    psf=None
    wavemin=None
    wavemax=None
    nfibers=None
    lines=None

    xcoef,ycoef,wavemin,wavemax = read_traces_in_psf(args.psf)
    nfibers=xcoef.shape[0]
    log.info("read PSF trace coef with {} fibers and wavelength range {}:{}".format(nfibers,int(wavemin),int(wavemax)))
    
    if args.fit_lines is not None :
        log.info("We will fit the image using the psf model and lines")
        # load the psf model
        psf = read_psf(args.psf)
        
        # read lines
        lines=np.loadtxt(args.fit_lines,usecols=[0])
        ok=(lines>wavemin)&(lines<wavemax)
        log.info("read {} lines in {}, with {} of them in traces wavelength range".format(len(lines),args.fit_lines,np.sum(ok)))
        lines=lines[ok]
        

    else :
        log.info("We will do an internal calibration of trace coordinates without using the psf shape in a first step")
        
        
        
    if args.nfibers is not None :
        nfibers = args.nfibers # FOR DEBUGGING

    fibers=np.arange(nfibers)

    if lines is not None :

        # use a forward modeling of the image
        # it's slower and works only for individual lines
        # it's in principle more accurate
        # but gives systematic residuals for complex spectra like the sky
        
        x,y,dx,ex,dy,ey,fiber_xy,wave_xy=compute_dx_dy_using_psf(psf,image,fibers,lines)
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
        
        # measure x shifts
        x_for_dx,y_for_dx,dx,ex,fiber_for_dx,wave_for_dx = compute_dx_from_cross_dispersion_profiles(xcoef,ycoef,wavemin,wavemax, image=image, fibers=fibers, width=7)
        # measure y shifts
        x_for_dy,y_for_dy,dy,ey,fiber_for_dy,wave_for_dy = compute_dy_using_boxcar_extraction(xcoef,ycoef,wavemin,wavemax, image=image, fibers=fibers, width=7)

    print(wave_for_dx.shape)
    print(fiber_for_dx.shape)
    
                
    # write this for debugging
    if args.outoffsets :
        file=open(args.outoffsets,"w")
        file.write("# axis f x y delta error\n")
        for e in range(dy.size) :
            file.write("0 %f %d %f %f %f %f\n"%(wave_for_dy[e],fiber_for_dy[e],x_for_dy[e],y_for_dy[e],dy[e],ey[e]))
        for e in range(dx.size) :
            file.write("1 %f %d %f %f %f %f\n"%(wave_for_dx[e],fiber_for_dx[e],x_for_dx[e],y_for_dx[e],dx[e],ex[e]))
            
        file.close()
        log.info("wrote offsets in ASCII file %s"%args.outoffsets)
    
    
    
    
    degxx=args.degxx
    degxy=args.degxy
    degyx=args.degyx
    degyy=args.degyy
    
    while(True) : # loop because polynomial degrees could be reduced
        
        log.info("polynomial fit of measured offsets with degx=(%d,%d) degy=(%d,%d)"%(degxx,degxy,degyx,degyy))
        try :
            dx_coeff,dx_coeff_covariance,dx_errorfloor,dx_mod,dx_mask=polynomial_fit(dd=dx,sdd=ex,xx=x_for_dx,yy=y_for_dx,degx=degxx,degy=degxy)
            dy_coeff,dy_coeff_covariance,dy_errorfloor,dy_mod,dy_mask=polynomial_fit(dd=dy,sdd=ey,xx=x_for_dy,yy=y_for_dy,degx=degyx,degy=degyy)
            
            log.info("dx dy error floor = %4.3f %4.3f pixels"%(dx_errorfloor,dy_errorfloor))

            log.info("check fit uncertainties are ok on edge of CCD")
            
            merr=0.
            for fiber in [0,nfibers-1] :
                for wave in [wavemin,wavemax] :
                    tx = legval(_u(wave,wavemin,wavemax),xcoef[fiber])
                    ty = legval(_u(wave,wavemin,wavemax),ycoef[fiber])
                    m=monomials(tx,ty,degxx,degxy)
                    tdx=np.inner(dx_coeff,m)
                    tsx=np.sqrt(np.inner(m,dx_coeff_covariance.dot(m)))
                    #print(x,y,m,dx)
                    m=monomials(tx,ty,degyx,degyy)
                    tdy=np.inner(dy_coeff,m)
                    tsy=np.sqrt(np.inner(m,dy_coeff_covariance.dot(m)))
                    merr=max(merr,tdx)
                    merr=max(merr,tdy)
                    #log.info("fiber=%d wave=%dA x=%d y=%d dx=%4.3f+-%4.3f dy=%4.3f+-%4.3f"%(fiber,int(wave),int(x),int(y),dx,sx,dy,sy))
            log.info("max edge shift error = %4.3f pixels"%merr)
            if degxx==0 and degxy==0 and degyx==0 and degyy==0 :
                break
        
        except LinAlgError :
            log.warning("polynomial fit failed with degx=(%d,%d) degy=(%d,%d)"%(degxx,degxy,degyx,degyy))
            if degxx==0 and degxy==0 and degyx==0 and degyy==0 :
                log.error("polynomial degrees are already 0. we can fit the offsets")
                raise RuntimeError("polynomial degrees are already 0. we can fit the offsets")
            merr = 100000. # this will lower the pol. degree.
        
        if merr > 0.05 :
            if merr != 100000. :
                log.warning("max edge shift error = %4.3f pixels is too large, reducing degrees"%merr)
            
            if degxy>0 and degyy>0 and degxy>degxx and degyy>degyx : # first along wavelength
                degxy-=1
                degyy-=1
            else : # then along fiber
                degxx-=1
                degyx-=1
        else :
            # error is ok, so we quit the loop
            break
    
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
    
    # for each fiber, apply offsets and recompute legendre polynomial
    log.info("for each fiber, apply offsets and recompute legendre polynomial")    
    for fiber in range(nfibers) :
        wave=np.linspace(wavemin,wavemax,100)
        x = legval(_u(wave,wavemin,wavemax),xcoef[fiber])
        y = legval(_u(wave,wavemin,wavemax),ycoef[fiber])
                
        m=monomials(x,y,degxx,degxy)
        dx=m.T.dot(dx_coeff)
        rwave=_u(wave,wavemin,wavemax)
        xcoef[fiber]=legfit(rwave,x+dx,deg=xcoef.shape[1]-1)
        
        m=monomials(x,y,degyx,degyy)
        dy=m.T.dot(dy_coeff)
        rwave=_u(wave,wavemin,wavemax)
        ycoef[fiber]=legfit(rwave,y+dy,deg=ycoef.shape[1]-1)
    
    # write the modified PSF
    write_traces_in_psf(args.psf,args.outpsf,xcoef,ycoef,wavemin,wavemax)
    
    log.info("wrote modified PSF in %s"%args.outpsf)
        
        
        
