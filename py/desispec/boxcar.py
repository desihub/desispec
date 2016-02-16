"""
boxcar extraction for Spectra from Desi Image
"""
import numpy as np

def do_boxcar(image,band,psf,camera,boxwidth=2.5,dw=0.5,nspec=500):
    """ 
    Args:  
         image  : desispec.image object
         band: band [r,b,z]
         psf: psf object respective of given band
         camera : camera ID
         boxwidth: HW box size in pixels
         dw: wavelength binning in output spectra, Default= 0.5A
         nspec: number of spectra to extract from the given image object

    Returns a desispec.frame object
    """
    from desispec.frame import Frame
    import math

    if band == "r":
        wmin=5625
        wmax=7741
        waves=np.arange(wmin,wmax,0.25)
        mask=np.zeros((4114,4128))
    elif band == "b":
        wmin=3569
        wmax=5949
        waves=np.arange(wmin,wmax,0.25)
        mask=np.zeros((4096,4096))
    elif band == "z":
        wmin=7435
        wmax=9834
        waves=np.arange(wmin,wmax,0.25)
        mask=np.zeros((4114,4128))
    else:
        print "Band can be r z or b"
        return None

    xs=psf.x(None,waves)
    ys=psf.y(None,waves)
    maxx,maxy=mask.shape
    maxx=maxx-1
    maxy=maxy-1
    ranges=np.zeros((mask.shape[1],xs.shape[0]+1),dtype=int)
    for bin in xrange(0,len(waves)):
        ixmaxold=0
        for spec in xrange(0,xs.shape[0]):
            xpos=xs[spec][bin]
            ypos=int(ys[spec][bin])
            if xpos<0 or xpos>maxx or ypos<0 or ypos>maxy : 
                continue 
            xmin=xpos-boxwidth
            xmax=xpos+boxwidth
            ixmin=int(math.floor(xmin))
            ixmax=int(math.floor(xmax))
            if ixmin <= ixmaxold:
                print "Error Box width overlaps,",xpos,ypos,ixmin,ixmaxold
                return None,None
            ixmaxold=ixmax
            if mask[int(xpos)][ypos]>0 :
                continue
        # boxing in x vals
            if ixmin < 0: #int value is less than 0
                ixmin=0
                rxmin=1.0
            else:# take part of the bin depending on real xmin
                rxmin=1.0-xmin+ixmin
            if ixmax>maxx:# xmax is bigger than the image
                ixmax=maxx
                rxmax=1.0
            else: # take the part of the bin depending on real xmax
                rxmax=xmax-ixmax
            ranges[ypos][spec+1]=math.ceil(xmax)#end at next column
            if  ranges[ypos][spec]==0:
                ranges[ypos][spec]=ixmin
            mask[ixmin][ypos]=rxmin
            for x in xrange(ixmin+1,ixmax): mask[x][ypos]=1.0
            mask[ixmax][ypos]=rxmax
    for ypos in xrange(ranges.shape[0]):
        lastval=ranges[ypos][0]
        for sp in xrange(1,ranges.shape[1]):
            if  ranges[ypos][sp]==0:
                ranges[ypos][sp]=lastval
            lastval=ranges[ypos][sp]
    

    maskedimg=(image.pix*mask.T)
    flux=np.zeros((maskedimg.shape[0],ranges.shape[1]-1))
    for r in xrange(flux.shape[0]):
        row=np.add.reduceat(maskedimg[r],ranges[r])[:-1]
        flux[r]=row
    from desispec.interpolation import resample_flux
    wtarget=np.arange(wmin,wmax+dw/2.0,dw)
    fflux=np.zeros((500,len(wtarget)))
    ivar=np.zeros((500,len(wtarget)))
    resolution=np.zeros((500,21,len(wtarget)))
    #TODO get the approximate resolution matrix. Like in specsim?
    for spec in xrange(flux.shape[1]):
        ww=psf.wavelength(spec)
        fflux[spec,:]=resample_flux(wtarget,ww,flux[:,spec])
        ivar[spec,:]=1./(fflux[spec,:]+image.readnoise)
    dwave=np.gradient(wtarget)
    fflux/=dwave
    ivar*=dwave**2
    #Extracted the full image but write frame in [nspec,nwave]
            
    return Frame(wtarget,fflux[:nspec],ivar[:nspec],resolution_data=resolution[:nspec],spectrograph=camera)
