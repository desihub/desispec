
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
    psf=pyfits.open(psf_filename)
    try :
        xtrace=psf["XTRACE"].data
        ytrace=psf["YTRACE"].data
    except : 
        raise ValueError("did not find XTRACE or YTRACE HDU in %s"%psf_filename)
    wavemin=psf["XTRACE"].header["WAVEMIN"]
    wavemax=psf["XTRACE"].header["WAVEMAX"]

    wavemin2=psf["YTRACE"].header["WAVEMIN"]
    wavemax2=psf["YTRACE"].header["WAVEMAX"]
    
    if wavemin != wavemin2 :
        raise ValueError("XTRACE and YTRACE don't have same WAVEMIN %f %f"%(wavemin,wavemin2))
    if wavemax != wavemax2 :
        raise ValueError("XTRACE and YTRACE don't have same WAVEMAX %f %f"%(wavemax,wavemax2))
    if xtrace.shape[0] != ytrace.shape[0] :
        raise ValueError("XTRACE and YTRACE don't have same number of fibers %d %d"%(xtrace.shape[0],ytrace.shape[0]))

    psf.close()
    return xtrace,ytrace,wavemin,wavemax
    
def _u(wave,wavemin,wavemax) :
    return 2.*(wave-wavemin)/(wavemax-wavemin)-1.

def compute_fiber_bundle_offsets(fibers,line,psf,image,maxshift=2.) :
    
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

def monomials(x,y,degx,degy) :
    M=[]
    for i in range(degx+1) :
        for j in range(degy+1) :
            M.append(x**i*y**j)
    return np.array(M)
    
def polynomial_fit_of_offsets(dd,sdd,xx,yy,degx,degy) :
   
    M=monomials(x=xx,y=yy,degx=degx,degy=degy)
    
    sdd[sdd>1]=10000. # totally deweight unmeasured data
    
    error_floor=0. # pix
    
    npar=M.shape[0]
    A=np.zeros((npar,npar))
    B=np.zeros((npar))
        
    for loop in range(10) : # loop to increase errors
        w=1./(sdd**2+error_floor**2)
        A *= 0.
        B *= 0.
        for k in range(npar) :
            B[k]=np.sum(w*dd*M[k])
            for l in range(k+1) :
                A[k,l]=np.sum(w*M[k]*M[l])
                if l!=k : A[l,k]=A[k,l]
        coeff=cholesky_solve(A,B)
        mod = M.T.dot(coeff)
        rchi2 = np.sum(w*(dd-mod)**2)/(dd.size-npar)
        #print("#%d rchi2=%f"%(loop,rchi2))
        if rchi2>1 :
            error_floor+=0.002
        else :
            break
    # rerun chol. solve to get covariance
    coeff,covariance=cholesky_solve_and_invert(A,B)
        
        
    return coeff,covariance,error_floor,mod

def parse(options=None):
    parser = argparse.ArgumentParser(description="Compute the sky model.")

    parser.add_argument('--image', type = str, default = None, required=True,
                        help = 'path of DESI preprocessed fits file')
    parser.add_argument('--psf', type = str, default = None, required=True,
                        help = 'path of DESI psf fits file')
    parser.add_argument('--lines', type = str, default = None, required=True,
                        help = 'path of lines ASCII file')
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
    
    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def main(args) :

    log=get_logger()

    log.info("starting")
    
    # read psf
    psf = read_psf(args.psf)
    wavemin = psf.wmin
    wavemax = psf.wmax
    nfibers = psf.nspec
    
    # nfibers = 3 # FOR DEBUGGING

    log.info("read PSF with {} fibers and wavelength range {}:{}".format(nfibers,int(wavemin),int(wavemax)))
    
    # read lines
    lines=np.loadtxt(args.lines,usecols=[0])
    ok=(lines>wavemin)&(lines<wavemax)
    log.info("read {} lines in {}, with {} of them in traces wavelength range".format(len(lines),args.lines,np.sum(ok)))
    lines=lines[ok]
    nlines=len(lines)
    
    # read preprocessed image
    image=read_image(args.image)
    log.info("read image {}".format(args.image))
    if image.mask is not None :
        image.ivar *= (image.mask==0)
    
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
    ndata=len(bundle_fibers)*len(lines)
    bb=np.zeros(ndata)  # bundle
    ll=np.zeros(ndata)  # line
    ff=np.zeros(ndata)  # central fiber in bundle
    xx=np.zeros(ndata)  # central x in bundle at line wavelength
    yy=np.zeros(ndata)  # central x in bundle at line wavelength
    dxx=np.zeros(ndata) # measured offset along x
    dyy=np.zeros(ndata) # measured offset along y
    sxx=np.zeros(ndata) # measured offset uncertainty along x
    syy=np.zeros(ndata) # measured offset uncertainty along y
    i=0
    for b in range(len(bundle_fibers)) :
        for l,line in enumerate(lines) :
            x,y,dx,dy,sx,sy = compute_fiber_bundle_offsets(fibers=bundle_fibers[b],psf=psf,image=image,line=line)
            log.info("fibers [%d:%d] %dA dx=%4.3f+-%4.3f dy=%4.3f+-%4.3f"%(bundle_fibers[b][0],bundle_fibers[b][-1]+1,int(line),dx,sx,dy,sy))
            bb[i]=b
            ll[i]=line
            ff[i]=int(np.median(bundle_fibers[b]))
            xx[i]=x
            yy[i]=y
            dxx[i]=dx
            dyy[i]=dy
            sxx[i]=sx
            syy[i]=sy
            i+=1
    
    # write this for debugging
    if args.outoffsets :
        file=open(args.outoffsets,"w")
        file.write("# b l f x y dx dy sx sy\n")
        for e in range(len(dxx)) :
            file.write("%d %f %d %f %f %f %f %f %f\n"%(bb[e],ll[e],ff[e],xx[e],yy[e],dxx[e],dyy[e],sxx[e],syy[e]))
        file.close()
        log.info("wrote offsets in ASCII file %s"%args.outoffsets)
    
        
    degxx=args.degxx
    degxy=args.degxy
    degyx=args.degyx
    degyy=args.degyy
    
    while(True) : # loop because polynomial degrees could be reduced
        
        log.info("polynomial fit of %d measured offsets with degx=(%d,%d) degy=(%d,%d)"%(xx.size,degxx,degxy,degyx,degyy))
        try :
            dx_coeff,dx_coeff_covariance,dx_errorfloor,dx_mod=polynomial_fit_of_offsets(dd=dxx,sdd=sxx,xx=xx,yy=yy,degx=degxx,degy=degxy)
            dy_coeff,dy_coeff_covariance,dy_errorfloor,dy_mod=polynomial_fit_of_offsets(dd=dyy,sdd=syy,xx=xx,yy=yy,degx=degyx,degy=degyy)
            
            log.info("dx dy error floor = %4.3f %4.3f pixels"%(dx_errorfloor,dy_errorfloor))

            log.info("check fit uncertainties are ok on edge of CCD")


            merr=0.
            for fiber in [0,nfibers-1] :
                for wave in [psf.wmin_all,psf.wmax_all] :
                    x,y=psf.xy(fiber,wave)
                    m=monomials(x,y,degxx,degxy)
                    dx=np.inner(dx_coeff,m)
                    sx=np.sqrt(np.inner(m,dx_coeff_covariance.dot(m)))
                    #print(x,y,m,dx)
                    m=monomials(x,y,degyx,degyy)
                    dy=np.inner(dy_coeff,m)            
                    sy=np.sqrt(np.inner(m,dy_coeff_covariance.dot(m)))
                    merr=max(merr,sx)
                    merr=max(merr,sy)
                    #log.info("fiber=%d wave=%dA x=%d y=%d dx=%4.3f+-%4.3f dy=%4.3f+-%4.3f"%(fiber,int(wave),int(x),int(y),dx,sx,dy,sy))
            log.info("max shift error = %4.3f pixels"%merr)
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
                log.warning("max shift error = %4.3f pixels is too large, reducing degrees"%merr)
            
            if degxy>0 and degyy>0 and degxy>degxx and degyy>degyx : # first along wavelength
                degxy-=1
                degyy-=1
            else : # then along fiber
                degxx-=1
                degyx-=1
        else :
            # error is ok, so we quit the loop
            break
    
        

    log.info("for each fiber, apply offsets and recompute legendre polynomial")
    
    
    # find legendre degree wavemin wavemax in PSF file 
    # that depends on the PSF type ...
    x_legendre_wavemin  = None
    x_legendre_wavemax  = None
    x_legendre_ncoeff   = None
    y_legendre_wavemin  = None
    y_legendre_wavemax  = None
    y_legendre_ncoeff   = None
    
    # for GAUSS-HERMITE
    x_table_index = None
    y_table_index = None
    
    psf_fits=pyfits.open(args.psf)
    psftype=psf_fits[0].header["PSFTYPE"]
    if psftype=="GAUSS-HERMITE" :
        table=psf_fits[1].data
        i=np.where(table["PARAM"]=="X")[0]
        if i.size != 1 :
            raise RuntimeError("Cannot understand the %s PSF format in file %s"%(psftype,args))
        x_table_index=i[0]
        x_legendre_wavemin = table["WAVEMIN"][x_table_index]
        x_legendre_wavemax = table["WAVEMAX"][x_table_index]
        x_legendre_ncoeff = table["COEFF"][x_table_index].shape[1]
        i=np.where(table["PARAM"]=="Y")[0]
        if i.size != 1 :
            raise RuntimeError("Cannot understand the %s PSF format in file %s"%(psftype,args))
        y_table_index=i[0]
        y_legendre_wavemin = table["WAVEMIN"][y_table_index]
        y_legendre_wavemax = table["WAVEMAX"][y_table_index]
        y_legendre_ncoeff = table["COEFF"][y_table_index].shape[1]
    else :
        raise RuntimeError("Sorry, modifications of trace shifts for PSF type %s is not yet implemented"%psftype)
    
    

    xcoeff=np.zeros((nfibers,x_legendre_ncoeff))
    ycoeff=np.zeros((nfibers,y_legendre_ncoeff))
    
    
    for fiber in range(nfibers) :
        wave=np.linspace(psf.wmin,psf.wmax,100)
        x,y=psf.xy(fiber,wave)
        
        m=monomials(x,y,degxx,degxy)
        dx=m.T.dot(dx_coeff)
        rwave=_u(wave,x_legendre_wavemin,x_legendre_wavemax)
        xcoeff[fiber]=legfit(rwave,x+dx,deg=x_legendre_ncoeff-1)
        
        m=monomials(x,y,degyx,degyy)
        dy=m.T.dot(dy_coeff)
        rwave=_u(wave,y_legendre_wavemin,y_legendre_wavemax)
        ycoeff[fiber]=legfit(rwave,y+dy,deg=y_legendre_ncoeff-1)
    
    # now we have to save this
    if psftype=="GAUSS-HERMITE" :
        psf_fits[1].data["COEFF"][x_table_index][:nfibers]=xcoeff
        psf_fits[1].data["COEFF"][y_table_index][:nfibers]=ycoeff
        # also save this in XTRACE and YTRACE HDUs if exist
        if "XTRACE" in psf_fits :
            psf_fits["XTRACE"].data = xcoeff
            psf_fits["XTRACE"].header["WAVEMIN"] = x_legendre_wavemin
            psf_fits["XTRACE"].header["WAVEMAX"] = x_legendre_wavemax
        if "YTRACE" in psf_fits :
            psf_fits["YTRACE"].data = ycoeff
            psf_fits["YTRACE"].header["WAVEMIN"] = y_legendre_wavemin
            psf_fits["YTRACE"].header["WAVEMAX"] = y_legendre_wavemax
    else :
        raise RuntimeError("Sorry, modifications of trace shifts for PSF type %s is not yet implemented"%psftype)

    # write the modified PSF    
    psf_fits.writeto(args.outpsf,clobber=True)
    log.info("wrote modified PSF in %s"%args.outpsf)
    
        
        
        
