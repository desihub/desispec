
from __future__ import absolute_import, division


import argparse
import numpy as np
from numpy.linalg.linalg import LinAlgError
import astropy.io.fits as pyfits
from numpy.polynomial.legendre import legval,legfit

import specter.psf
from desispec.io import read_image
from desiutil.log import get_logger
from desispec.linalg import cholesky_solve,cholesky_solve_and_invert
from desispec.interpolation import resample_flux

def read_psf(psf_filename) :
    try :
        psftype=pyfits.open(psf_filename)[0].header["PSFTYPE"]
    except KeyError :
        psftype=""
    if psftype=="GAUSS-HERMITE" :
        return specter.psf.GaussHermitePSF(psf_filename)
    elif psftype=="SPOTGRID" :
        return specter.psf.SpotGridPSF(psf_filename)

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

def compute_dx_from_cross_dispersion_profiles(xcoef,ycoef,wavemin,wavemax, image, fibers=None, width=5,deg=2) :
    
    
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
    
def polynomial_fit(dd,sdd,xx,yy,degx,degy) :
   
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

def compute_fiber_bundle_trace_shifts_using_psf(fibers,line,psf,image,maxshift=2.) :
    
    log=get_logger()
    #log.info("compute_fiber_bundle_offsets fibers={} line={}".format(fibers,line))

    # get central coordinates of bundle for interpolation of offsets on CCD
    x,y = psf.xy([int(np.median(fibers)),],line)
    

    try : 
        nfibers=len(fibers)
        
        # compute stamp coordinates
        xstart=None
        xstop=None
        ystart=None
        ystop=None
        xs=[]
        ys=[]
        pix=[]
        xx=[]
        yy=[]
        
        for fiber in fibers :
            txs,tys,tpix = psf.xypix(fiber,line)
            xs.append(txs)
            ys.append(tys)
            pix.append(tpix)
            if xstart is None :
                xstart =txs.start
                xstop  =txs.stop
                ystart =tys.start
                ystop  =tys.stop
            else :
                xstart =min(xstart,txs.start)
                xstop  =max(xstop,txs.stop)
                ystart =min(ystart,tys.start)
                ystop  =max(ystop,tys.stop)

        # load stamp data, with margins to avoid problems with shifted psf
        margin=int(maxshift)+1
        stamp=np.zeros((ystop-ystart+2*margin,xstop-xstart+2*margin))
        stampivar=np.zeros(stamp.shape)
        stamp[margin:-margin,margin:-margin]=image.pix[ystart:ystop,xstart:xstop]
        stampivar[margin:-margin,margin:-margin]=image.ivar[ystart:ystop,xstart:xstop]


        # will use a fixed footprint despite changes of psf stamps
        # so that chi2 always based on same data set
        footprint=np.zeros(stamp.shape)   
        for i in range(nfibers) :
            footprint[margin-ystart+ys[i].start:margin-ystart+ys[i].stop,margin-xstart+xs[i].start:margin-xstart+xs[i].stop]=1

        #plt.imshow(footprint) ; plt.show() ; sys.exit(12)

        # define grid of shifts to test
        res=0.5
        nshift=int(maxshift/res)
        dx=res*np.tile(np.arange(2*nshift+1)-nshift,(2*nshift+1,1))
        dy=dx.T
        original_shape=dx.shape
        dx=dx.ravel()
        dy=dy.ravel()
        chi2=np.zeros(dx.shape)

        A=np.zeros((nfibers,nfibers))
        B=np.zeros((nfibers))
        mods=np.zeros(np.zeros(nfibers).shape+stamp.shape)
        
        debugging=False

        if debugging : # FOR DEBUGGING KEEP MODELS            
            models=[]


        # loop on possible shifts
        # refit fluxes and compute chi2
        for d in range(len(dx)) :
            # print(d,dx[d],dy[d])
            A *= 0
            B *= 0
            mods *= 0

            for i,fiber in enumerate(fibers) :

                # apply the PSF shift
                psf._cache={} # reset cache !!
                psf.coeff['X']._coeff[fiber][0] += dx[d]
                psf.coeff['Y']._coeff[fiber][0] += dy[d]

                # compute pix and paste on stamp frame
                xx, yy, pix = psf.xypix(fiber,line)
                mods[i][margin-ystart+yy.start:margin-ystart+yy.stop,margin-xstart+xx.start:margin-xstart+xx.stop]=pix

                # undo the PSF shift
                psf.coeff['X']._coeff[fiber][0] -= dx[d]
                psf.coeff['Y']._coeff[fiber][0] -= dy[d]

                B[i] = np.sum(stampivar*stamp*mods[i])
                for j in range(i+1) :
                    A[i,j] = np.sum(stampivar*mods[i]*mods[j]) 
                    if j!=i :
                        A[j,i] = A[i,j]
            Ai=np.linalg.inv(A)
            flux=Ai.dot(B)
            model=np.zeros(stamp.shape)
            for i in range(nfibers) :
                model += flux[i]*mods[i]
            chi2[d]=np.sum(stampivar*(stamp-model)**2)
            if debugging :
                models.append(model)
            
        if debugging :
            schi2=chi2.reshape(original_shape).copy() # FOR DEBUGGING
            sdx=dx.copy()
            sdy=dy.copy()
        
        # find minimum chi2 grid point
        k   = chi2.argmin()
        j,i = np.unravel_index(k, ((2*nshift+1),(2*nshift+1)))
        #print("node dx,dy=",dx.reshape(original_shape)[j,i],dy.reshape(original_shape)[j,i])
        
        # cut a region around minimum
        delta=1
        istart=max(0,i-delta)
        istop=min(2*nshift+1,i+delta+1)
        jstart=max(0,j-delta)
        jstop=min(2*nshift+1,j+delta+1)
        chi2=chi2.reshape(original_shape)[jstart:jstop,istart:istop].ravel()
        dx=dx.reshape(original_shape)[jstart:jstop,istart:istop].ravel()
        dy=dy.reshape(original_shape)[jstart:jstop,istart:istop].ravel()    
        # fit 2D polynomial of deg2
        m = np.array([dx*0+1, dx, dy, dx**2, dy**2, dx*dy ]).T
        c, r, rank, s = np.linalg.lstsq(m, chi2)
        if c[3]>0 and c[4]>0 :
            # get minimum
            # dchi2/dx=0 : c[1]+2*c[3]*dx+c[5]*dy = 0
            # dchi2/dy=0 : c[2]+2*c[4]*dy+c[5]*dx = 0
            a=np.array([[2*c[3],c[5]],[c[5],2*c[4]]])
            b=np.array([c[1],c[2]])
            t=-np.linalg.inv(a).dot(b)
            dx=t[0]
            dy=t[1]
            sx=1./np.sqrt(c[3])
            sy=1./np.sqrt(c[4])
            #print("interp dx,dy=",dx,dy)
            
            if debugging : # FOR DEBUGGING
                import matplotlib.pyplot as plt
                plt.figure()
                plt.subplot(2,2,1,title="chi2")
                plt.imshow(schi2,extent=(-nshift*res,nshift*res,-nshift*res,nshift*res),origin=0,interpolation="nearest")            
                plt.plot(dx,dy,"+",color="white",ms=20)
                plt.xlabel("x")
                plt.ylabel("y")
                plt.subplot(2,2,2,title="data")
                plt.imshow(stamp*footprint,origin=0,interpolation="nearest")
                plt.grid()
                k0=np.argmin(sdx**2+sdy**2)
                plt.subplot(2,2,3,title="original psf")
                plt.imshow(models[k0],origin=0,interpolation="nearest")            
                plt.grid()
                plt.subplot(2,2,4,title="shifted psf")
                plt.imshow(models[k],origin=0,interpolation="nearest")
                plt.grid()
                plt.show()
                
        else :
            log.warning("fit failed (bad chi2 surf.) for fibers [%d:%d] line=%dA"%(fibers[0],fibers[-1]+1,int(line)))
            dx=0.
            dy=0.
            sx=10.
            sy=10.
    except LinAlgError :
        log.warning("fit failed (masked or missing data) for fibers [%d:%d] line=%dA"%(fibers[0],fibers[-1]+1,int(line)))
        dx=0.
        dy=0.
        sx=10.
        sy=10.
    
    return x,y,dx,dy,sx,sy


def compute_dy_using_boxcar_extraction(xcoef,ycoef,wavemin,wavemax, image, fibers, width=7, degyy=2) :

    log=get_logger()
    
    log.info("boxcar extraction")
    boxcar_flux, boxcar_ivar, boxcar_wave = boxcar_extraction(xcoef,ycoef,wavemin,wavemax, image, fibers=fibers, width=7)
    
    log.info("resampling with oversampling")
    nfibers=len(fibers)
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
    nblocks = degyy+2

    
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
    

    return x_for_dy,y_for_dy,dy,ey,fiber_for_dy,wave_for_dy
    
def compute_dx_dy_using_psf(psf,image,fibers,lines) :

    log = get_logger()
    
    nlines=len(lines)
    nfibers=len(fibers)
    
    log.info("computing spots coordinates and define bundles")
    x=np.zeros((nfibers,nlines))
    y=np.zeros((nfibers,nlines))
    
    # load expected spots coordinates 
    for fiber in range(nfibers) :
        for l,line in enumerate(lines) :
            x[fiber,l],y[fiber,l] = psf.xy(fiber,line)

    bundle_fibers=[]
    bundle_xmin=[]
    bundle_xmax=[]
    xwidth=9.
    bundle_xmin.append(x[0,nlines//2]-xwidth/2)
    bundle_xmax.append(x[0,nlines//2]+xwidth/2)
    bundle_fibers.append([0,])
    
    
    for fiber in range(1,nfibers) :
        tx=x[fiber,nlines//2]
        found=False
        for b in range(len(bundle_fibers)) :
            if tx+xwidth/2 >= bundle_xmin[b] and tx-xwidth/2 <= bundle_xmax[b] :
                found=True
                bundle_fibers[b].append(fiber)
                bundle_xmin[b]=min(bundle_xmin[b],tx-xwidth/2)
                bundle_xmax[b]=max(bundle_xmax[b],tx+xwidth/2)
                break
        if not found :
            bundle_fibers.append([fiber,])
            bundle_xmin.append(tx-xwidth/2)
            bundle_xmax.append(tx+xwidth/2)
    
    log.info("measure offsets dx dy per bundle ({}) and spectral line ({})".format(len(bundle_fibers),len(lines)))

    wave_xy=np.array([])  # line
    fiber_xy=np.array([])  # central fiber in bundle
    x=np.array([])  # central x in bundle at line wavelength
    y=np.array([])  # central x in bundle at line wavelength
    dx=np.array([]) # measured offset along x
    dy=np.array([]) # measured offset along y
    ex=np.array([]) # measured offset uncertainty along x
    ey=np.array([]) # measured offset uncertainty along y
    
    for b in range(len(bundle_fibers)) :
        for l,line in enumerate(lines) :
            tx,ty,tdx,tdy,tex,tey = compute_fiber_bundle_trace_shifts_using_psf(fibers=bundle_fibers[b],psf=psf,image=image,line=line)
            log.info("fibers [%d:%d] %dA dx=%4.3f+-%4.3f dy=%4.3f+-%4.3f"%(bundle_fibers[b][0],bundle_fibers[b][-1]+1,int(line),tdx,tex,tdy,tey))
            if tex<1. and tey<1. :                
                wave_xy=np.append(wave_xy,line)
                fiber_xy=np.append(fiber_xy,int(np.median(bundle_fibers[b])))
                x=np.append(x,tx)
                y=np.append(y,ty)
                dx=np.append(dx,tdx)
                dy=np.append(dy,tdy)
                ex=np.append(ex,tex)
                ey=np.append(ey,tey)
    return x,y,dx,ex,dy,ey,fiber_xy,wave_xy

    
