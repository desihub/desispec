
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


def read_traces_in_psf(psf_filename) :
    """Reads traces in PSF file
    
    Args:
        psf_filename : Path to input fits file which has to contain XTRACE and YTRACE HDUs
    Returns:
        xtrace : 2D np.array of shape (nfibers,ncoef) containing Legendre coefficents for each fiber to convert wavelenght to XCCD
        ytrace : 2D np.array of shape (nfibers,ncoef) containing Legendre coefficents for each fiber to convert wavelenght to YCCD
        wavemin : float
        wavemax : float. wavemin and wavemax are used to define a reduced variable u(wave,wavemin,wavemax)=2*(wave-wavemin)/(wavemax-wavemin)-1
                  used to compute the traces, xccd=legval(xtrace[fiber],u(wave,wavemin,wavemax))
    
    """

    log=get_logger()
    
    xtrace=None
    ytrace=None
    wavemin=None
    wavemax=None
    wavemin2=None
    wavemax2=None
    
    psf=pyfits.open(psf_filename)
    
    if "XTRACE" in psf :
        xtrace=psf["XTRACE"].data
        ytrace=psf["YTRACE"].data
        wavemin=psf["XTRACE"].header["WAVEMIN"]
        wavemax=psf["XTRACE"].header["WAVEMAX"]
        wavemin2=psf["YTRACE"].header["WAVEMIN"]
        wavemax2=psf["YTRACE"].header["WAVEMAX"]
        
    else :
       psftype=psf[0].header["PSFTYPE"]
       log.info("psf is a '%s'"%psftype)
       if psftype == "bootcalib" :    
           wavemin = psf[0].header["WAVEMIN"]
           wavemax = psf[0].header["WAVEMAX"]
           xcoef   = psf[0].data
           ycoef   = psf[1].data
           xsig    = psf[2].data
       elif psftype == "GAUSS-HERMITE" :
           table=psf[1].data        
           i=np.where(table["PARAM"]=="X")[0][0]
           wavemin=table["WAVEMIN"][i]
           wavemax=table["WAVEMAX"][i]
           xtrace=table["COEFF"][i]
           i=np.where(table["PARAM"]=="Y")[0][0]
           ytrace=table["COEFF"][i]
           wavemin2=table["WAVEMIN"][i]
           wavemax2=table["WAVEMAX"][i]
    
    if xtrace is None or ytrace is None :
        raise ValueError("could not find XTRACE and YTRACE in psf file %s"%psf_filename)
    if wavemin != wavemin2 :
        raise ValueError("XTRACE and YTRACE don't have same WAVEMIN %f %f"%(wavemin,wavemin2))
    if wavemax != wavemax2 :
        raise ValueError("XTRACE and YTRACE don't have same WAVEMAX %f %f"%(wavemax,wavemax2))
    if xtrace.shape[0] != ytrace.shape[0] :
        raise ValueError("XTRACE and YTRACE don't have same number of fibers %d %d"%(xtrace.shape[0],ytrace.shape[0]))
        
    
    psf.close()
    return xtrace,ytrace,wavemin,wavemax
    
def write_traces_in_psf(input_psf_filename,output_psf_filename,xcoef,ycoef,wavemin,wavemax) :
    
    psf_fits=pyfits.open(input_psf_filename)

    psftype=psf_fits[0].header["PSFTYPE"]
    if psftype=="GAUSS-HERMITE" :             
        i=np.where(psf_fits[1].data["PARAM"]=="X")[0][0]
        psf_fits[1].data["COEFF"][i][:xcoef.shape[0]]=xcoef
        i=np.where(psf_fits[1].data["PARAM"]=="Y")[0][0]
        psf_fits[1].data["COEFF"][i][:ycoef.shape[0]]=ycoef
    
    if "XTRACE" in psf_fits :
        psf_fits["XTRACE"].data = xcoef
        psf_fits["XTRACE"].header["WAVEMIN"] = wavemin
        psf_fits["XTRACE"].header["WAVEMAX"] = wavemax
    if "YTRACE" in psf_fits :
        psf_fits["YTRACE"].data = ycoef
        psf_fits["YTRACE"].header["WAVEMIN"] = wavemin
        psf_fits["YTRACE"].header["WAVEMAX"] = wavemax
    
    psf_fits.writeto(output_psf_filename,clobber=True)
    
    

def _u(wave,wavemin,wavemax) :
    return 2.*(wave-wavemin)/(wavemax-wavemin)-1.


def boxcar_extraction(xcoef,ycoef,wavemin,wavemax, image, fibers=None, width=5) :
    
    """Find and returns  wavelength  spectra and inverse variance
    """
    log=get_logger()
    log.info("Starting boxcar extraction...")
    
    if fibers is None :
        fibers = np.arange(psf.nspec)
    
    log.info("wavelength range : [%f,%f]"%(wavemin,wavemax))
    
    if image.mask is not None :
        image.ivar *= (image.mask==0)
    
    #  Applying a mask that keeps positive value to get the Variance by inversing the inverse variance.
    var=np.zeros(image.ivar.size)
    ok=image.ivar.ravel()>0
    var[ok] = 1./image.ivar.ravel()[ok]
    var=var.reshape(image.ivar.shape)

    badimage=(image.ivar==0)
    
    n0 = image.pix.shape[0]
    n1 = image.pix.shape[1]
    
    frame_flux = np.zeros((fibers.size,n0))
    frame_ivar = np.zeros((fibers.size,n0))
    frame_wave = np.zeros((fibers.size,n0))
    xx         = np.tile(np.arange(n1),(n0,1))
    hw = width//2
    
    ncoef=ycoef.shape[1]
    twave=np.linspace(wavemin, wavemax, ncoef+2)
    
    for f,fiber in enumerate(fibers) :
        log.info("extracting fiber #%03d"%fiber)
        y_of_wave     = legval(_u(twave, wavemin, wavemax), ycoef[fiber])
        coef          = legfit(_u(y_of_wave, 0, n0), twave, deg=ncoef) # add one deg
        frame_wave[f] = legval(_u(np.arange(n0).astype(float), 0, n0), coef)
        x_of_y        = np.floor( legval(_u(frame_wave[f], wavemin, wavemax), xcoef[fiber]) + 0.5 ).astype(int)
        mask=((xx.T>=x_of_y-hw)&(xx.T<=x_of_y+hw)).T
        frame_flux[f]=image.pix[mask].reshape((n0,width)).sum(-1)
        tvar=var[mask].reshape((n0,width)).sum(-1)
        frame_ivar[f]=(tvar>0)/(tvar+(tvar==0))
        bad=(badimage[mask].reshape((n0,width)).sum(-1))>0
        frame_ivar[f,bad]=0.
    
    return frame_flux, frame_ivar, frame_wave

def compute_dx(xcoef,ycoef,wavemin,wavemax, image, fibers=None, width=5,deg=2) :
    
    
    log=get_logger()
    log.info("Starting perpendicular boxcar extraction...")
    
    if fibers is None :
        fibers = np.arange(psf.nspec)
    
    log.info("wavelength range : [%f,%f]"%(wavemin,wavemax))
    
    if image.mask is not None :
        image.ivar *= (image.mask==0)

        
    #   Variance based on inverse variance's size
    var    = np.zeros(image.ivar.shape)

    #   Applying a mask that keeps positive value to get the Variance by inversing the inverse variance.
    
    n0 = image.pix.shape[0]
    n1 = image.pix.shape[1]
    
    y  = np.arange(n0)
    xx = np.tile(np.arange(n1),(n0,1))
    hw = width//2
    
    ncoef=ycoef.shape[1]
    twave=np.linspace(wavemin, wavemax, ncoef+2)

    ox=np.array([])
    oy=np.array([])
    odx=np.array([])
    oex=np.array([])
    of=np.array([])
    ol=np.array([])
    
    for f,fiber in enumerate(fibers) :
        log.info("computing dx for fiber #%03d"%fiber)
        y_of_wave     = legval(_u(twave, wavemin, wavemax), ycoef[fiber])
        coef          = legfit(_u(y_of_wave, 0, n0), twave, deg=ncoef) # add one deg
        twave         = legval(_u(np.arange(n0).astype(float), 0, n0), coef)
        x_of_y        = legval(_u(twave, wavemin, wavemax), xcoef[fiber])
        x_of_y_int    = np.floor(x_of_y+0.5).astype(int)
        dx            = (xx.T-x_of_y).T
        mask=((xx.T>=x_of_y_int-hw)&(xx.T<=x_of_y_int+hw)).T
        swdx           = (dx[mask] * image.ivar[mask] * image.pix[mask] ).reshape((n0,width)).sum(-1)
        sw            = (image.ivar[mask] * image.pix[mask]).reshape((n0,width)).sum(-1)
        swy           = sw*y
        swx           = sw*x_of_y
        swl           = sw*twave

        # rebin
        rebin = 100
        sw  = sw[:(n0//rebin)*rebin].reshape(n0//rebin,rebin).sum(-1)
        swdx = swdx[:(n0//rebin)*rebin].reshape(n0//rebin,rebin).sum(-1)
        swx = swx[:(n0//rebin)*rebin].reshape(n0//rebin,rebin).sum(-1)
        swy = swy[:(n0//rebin)*rebin].reshape(n0//rebin,rebin).sum(-1)
        swl = swl[:(n0//rebin)*rebin].reshape(n0//rebin,rebin).sum(-1)


        sw[sw<0]       = 0        
        fdx            = swdx/(sw+(sw==0))
        fx             = swx/(sw+(sw==0))
        fy             = swy/(sw+(sw==0))
        fl             = swl/(sw+(sw==0))
        fex            = 1./np.sqrt(sw+(sw==0))
        
        good_fiber=True
        for loop in range(10) :

            if np.sum(sw>0) < deg+2 :
                good_fiber=False
                break

            try :
                c             = np.polyfit(fy,fdx,deg,w=sw)
                pol           = np.poly1d(c)
                chi2          = sw*(fdx-pol(fy))**2
                mchi2         = np.median(chi2[sw>0])
                sw /= mchi2
                bad           = chi2>25.*mchi2
                nbad          = np.sum(bad)
                sw[bad]       = 0.
            except LinAlgError :
                good_fiber=False
                break
            
            if nbad==0 :
                break
        
        
        # we return the original sample of offset values
        if good_fiber :
            ox  = np.append(ox,fx[sw>0])
            oy  = np.append(oy,fy[sw>0])
            odx = np.append(odx,fdx[sw>0])
            oex = np.append(oex,fex[sw>0])
            of = np.append(of,fiber*np.ones(fy[sw>0].size))
            ol = np.append(ol,fl[sw>0])
    
    return ox,oy,odx,oex,of,ol
        
    


def compute_dy_from_spectral_cross_correlation(flux,refwave,refflux,ivar=None,hw=3.) :

    error_floor=0.001 #A
    
    if ivar is None :
        ivar=np.ones(flux.shape)
    dwave=refwave[1]-refwave[0]
    ihw=int(hw/dwave)+1
    chi2=np.zeros((2*ihw+1))
    for i in range(2*ihw+1) :
        d=i-ihw
        b=ihw+d
        e=-ihw+d
        if e==0 :
            e=refwave.size
        chi2[i] = np.sum(ivar[ihw:-ihw]*(flux[ihw:-ihw]-refflux[b:e])**2)
    i=np.argmin(chi2)
    b=i-1
    e=i+2
    if b<0 : 
        b=0
        e=b+3
    if e>2*ihw+1 :
        e=2*ihw+1
        b=e-3
    x=dwave*(np.arange(b,e)-ihw)
    c=np.polyfit(x,chi2[b:e],2)
    if c[0]>0 :
        delta=-c[1]/(2.*c[0])
        sigma=np.sqrt(1./c[0] + error_floor**2)
    else :
        delta=0.
        sigma=100.
    return delta,sigma


def monomials(x,y,degx,degy) :
    M=[]
    for i in range(degx+1) :
        for j in range(degy+1) :
            M.append(x**i*y**j)
    return np.array(M)
    
def polynomial_fit_of_offsets(dd,sdd,xx,yy,degx,degy) :
   
    M=monomials(x=xx,y=yy,degx=degx,degy=degy)
    
    a_large_error = 1.e4
    sdd[sdd>1]= a_large_error # totally deweight unmeasured data
    
    error_floor=0.002 # pix
    
    npar=M.shape[0]
    A=np.zeros((npar,npar))
    B=np.zeros((npar))
    
    mask=(sdd<a_large_error)
    for loop in range(100) : # loop to increase errors
        
        w=1./(sdd**2+error_floor**2)
        w[mask==0]=0.
        
        A *= 0.
        B *= 0.
        for k in range(npar) :
            B[k]=np.sum(w*dd*M[k])
            for l in range(k+1) :
                A[k,l]=np.sum(w*M[k]*M[l])
                if l!=k : A[l,k]=A[k,l]
        coeff=cholesky_solve(A,B)
        mod = M.T.dot(coeff)
        
        # compute rchi2 with median
        ndata=np.sum(w>0)
        rchi2=1.4826*np.median(np.sqrt(w)*np.abs(dd-mod))*ndata/float(ndata-npar)
        # std chi2 
        rchi2_std = np.sum(w*(dd-mod)**2)/(ndata-npar)
        #print("#%d rchi2=%f rchi2_std=%f ngood=%d nbad=%d error floor=%f"%(loop,rchi2,rchi2_std,ndata,np.sum(w==0),error_floor))
        
        # reject huge outliers
        nbad=0
        rvar=w*(dd-mod)**2
        worst=np.argmax(rvar)
        if rvar[worst] > 25*max(rchi2,1.2) : # cap rchi2 if starting point is very bad
            #print("remove one bad measurement at %2.1f sigmas"%np.sqrt(rvar[worst]))
            mask[worst]=0
            nbad=1
        
        if rchi2>1 :
            if nbad==0 or loop>5 :
                error_floor+=0.002
        
        if rchi2<=1. and nbad==0 :
            break
    
    # rerun chol. solve to get covariance
    coeff,covariance=cholesky_solve_and_invert(A,B)
        
        
    return coeff,covariance,error_floor,mod,mask

def parse(options=None):
    parser = argparse.ArgumentParser(description="Compute the sky model.")

    parser.add_argument('--image', type = str, default = None, required=True,
                        help = 'path of DESI preprocessed fits file')
    parser.add_argument('--psf', type = str, default = None, required=True,
                        help = 'path of DESI psf fits file')
    #parser.add_argument('--lines', type = str, default = None, required=True,
    #                    help = 'path of lines ASCII file')
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
    #parser.add_argument('--ncpu', type = int, default = 1, required=False,
    #                    help = 'number of cpu for multiprocessing')
    
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
    
    # read psf
    xcoef,ycoef,wavemin,wavemax = read_traces_in_psf(args.psf)
    nfibers=xcoef.shape[0]
    log.info("read PSF with {} fibers and wavelength range {}:{}".format(nfibers,int(wavemin),int(wavemax)))
    
    # read preprocessed image
    image=read_image(args.image)
    log.info("read image {}".format(args.image))
    if image.mask is not None :
        image.ivar *= (image.mask==0)
    
    #nfibers = 20 # FOR DEBUGGING

    # measure x shifts
    x_for_dx,y_for_dx,dx,ex,fiber_for_dx,wave_for_dx = compute_dx(xcoef,ycoef,wavemin,wavemax, image, fibers=np.arange(nfibers), width=7)
    
    # boxcar extraction to measure y shifts
    log.info("boxcar extraction")
    fibers=np.arange(nfibers)
    boxcar_flux, boxcar_ivar, boxcar_wave = boxcar_extraction(xcoef,ycoef,wavemin,wavemax, image, fibers=fibers, width=7)
    
    log.info("resampling with oversampling")
    wave=boxcar_wave[nfibers//2]
    dwave=np.median(np.gradient(boxcar_wave))/2 # oversampling
    wave=np.linspace(wave[0],wave[-1],int((wave[-1]-wave[0])/dwave))
    nwave=wave.size
    nfibers=fibers.size
    flux=np.zeros((nfibers,nwave))
    ivar=np.zeros((nfibers,nwave))
    for i in range(nfibers) :
        log.info("resampling fiber #%03d"%fibers[i])
        flux[i],ivar[i] = resample_flux(wave, boxcar_wave[i],boxcar_flux[i],boxcar_ivar[i])
    mflux=np.median(flux,axis=0)

    '''
    import matplotlib.pyplot as plt
    for i in range(nfibers) :        
        plt.plot(wave[ivar[i]>0],flux[i][ivar[i]>0])
    plt.plot(wave,mflux,lw=2)
    plt.show()
    '''
    
    # measure y shifts 
    nblocks = args.degyy+2

    
    x_for_dy=np.array([])
    y_for_dy=np.array([])
    dy=np.array([])
    ey=np.array([])
    fiber_for_dy=np.array([])
    wave_for_dy=np.array([])
    
    for i,fiber in enumerate(fibers) :
        log.info("computing dy for fiber #%03d"%fiber)
        
        for b in range(nblocks) :
            wmin=wave[0]+((wave[-1]-wave[0])/nblocks)*b
            if b<nblocks-1 :
                wmax=wave[0]+((wave[-1]-wave[0])/nblocks)*(b+1)
            else :
                wmax=wave[-1]
            ok=(wave>=wmin)&(wave<=wmax)
            sw=np.sum(ivar[i,ok]*flux[i,ok]*(flux[i,ok]>0))
            if sw<=0 :
                continue
            dwave,err = compute_dy_from_spectral_cross_correlation(flux[i,ok],wave[ok],mflux[ok],ivar=ivar[i,ok],hw=3.)
            block_wave = np.sum(ivar[i,ok]*flux[i,ok]*(flux[i,ok]>0)*wave[ok])/sw
            if err > 1 :
                continue

            tx = legval(_u(block_wave,wavemin,wavemax),xcoef[fiber])
            ty = legval(_u(block_wave,wavemin,wavemax),ycoef[fiber])
            eps=0.1
            yp = legval(_u(block_wave+eps,wavemin,wavemax),ycoef[fiber])
            dydw = (yp-ty)/eps
            tdy = dwave*dydw
            tey = err*dydw
            
            x_for_dy=np.append(x_for_dy,tx)
            y_for_dy=np.append(y_for_dy,ty)
            dy=np.append(dy,tdy)
            ey=np.append(ey,tey)
            fiber_for_dy=np.append(fiber_for_dy,fiber)
            wave_for_dy=np.append(wave_for_dy,block_wave)
    


    

    
            
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
            dx_coeff,dx_coeff_covariance,dx_errorfloor,dx_mod,dx_mask=polynomial_fit_of_offsets(dd=dx,sdd=ex,xx=x_for_dx,yy=y_for_dx,degx=degxx,degy=degxy)
            dy_coeff,dy_coeff_covariance,dy_errorfloor,dy_mod,dy_mask=polynomial_fit_of_offsets(dd=dy,sdd=ey,xx=x_for_dy,yy=y_for_dy,degx=degyx,degy=degyy)
            
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
        
        
        
