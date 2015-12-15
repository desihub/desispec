"""
Pipeline Preprocessing algorithms for Quicklook
"""

import numpy as np
import os
from desispec.qlpipeline import PAs

#rimage=read_image(filename) # image object 
#dark=read_dark(filename) # dark object

#def dark_subtract(rawimage,darkimage):
from desispec.qlpipeline import QLExceptions
class DarkSubtraction(PAs.PipelineAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Dark Subtraction"
        from desispec.image import Image as im
        PAs.PipelineAlg.__init__(self,name,im,im,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise QLExceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise QLExceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        if "DarkImage" not in kwargs:
            raise QLExceptions.ParameterException("Need Dark Image")
        return self.do_darkSubtract(args[0],**kwargs)
    def get_default_config(self):
        return {("DarkImage","%%DarkImage","Dark image to subtract")}
        
    def do_darkSubtract(self,rawimage,**kwargs):
        """ 
        rawimage: raw DESI object Should come from Read_rawimage
        darkimage: DESI dark onject Should come from Read_darkframe
        """
    #rimage=fits.open(rawimage) # this should be read from desispec.io routine once exist
    #dark=fits.open(darkimage)

        # subtract pixel by pixel dark # may need to subtract separately each quadrant. For now assuming 
        # a single set.
        darkimage=kwargs["DarkImage"]
        rimage=rawimage.pix
        dimage=darkimage.pix

        rx=rimage.shape[0]
        ry=rimage.shape[1]
        value=np.zeros((rx,ry))
    # check dimensionality:
        assert rx ==dimage.shape[0]
        assert ry==dimage.shape[1]
    #for ii in range(rx):
    #    for jj in range(ry):
        value=rimage-dimage
        dknoise=dimage.std()  # or some values read from somewhere?
        ivar_new=1./(rimage+dimage+dknoise**2)
        mask=rawimage.mask
        from desispec.image import Image as im
        return im(value,ivar_new,mask)
    # Should save as an image?       

class PixelFlattening(PAs.PipelineAlg):
    from desispec.image import Image as im
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Pixel Flattening"
        from desispec.image import Image as im
        PAs.PipelineAlg.__init__(self,name,im,im,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise QLExceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise QLExceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        if "PixelFlat" not in kwargs:
            raise QLExceptions.ParameterException("Need Pixel Flat image")
        return self.do_pixelFlat(args[0],**kwargs)
    def get_default_config(self):
        return {("PixelFlat","%%PixelFlat","Pixel Flat image to apply")}
    def do_pixelFlat(self,image, **kwargs):

        """
        image: image object typically after dark subtraction)
        pixelflat: a pixel flat object
        """
    #pflat=read_image(pixelflat)
        pixelFlat=kwargs["PixelFlat"]
        pflat=pixelFlat.pix # should be read from desispec.io
        rx=pflat.shape[0]
        ry=pflat.shape[1]
        
    # check dimensionality
        assert rx ==image.pix.shape[0]
        assert ry==image.pix.shape[1]
        
        
    #value=np.zeros((rx,ry))
        
    #for ii in range(rx):
    #    for jj in range(ry):
        value=image.pix/pflat
        ivar=image.ivar 
        #TODO ivar from pixel flat need to be propagated
        mask=image.mask 
        from desispec.image import Image as im
        return im(value,ivar,mask)

class BiasSubtraction(PAs.PipelineAlg):
    from desispec.image import Image as im
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Bias Subtraction"
        from desispec.image import Image as im
        PAs.PipelineAlg.__init__(self,name,im,im,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise QLExceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise QLExceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        if "BiasImage" not in kwargs:
            raise QLExceptions.ParameterException("Need Bias image")
        return self.do_biasSubtract(args[0],**kwargs)
    def get_default_config(self):
        return {("BiasImage","%%BiasImage","Bias image to subtract")}
    def do_biasSubtract(self,rawimage,**kwargs):
        """ rawimage: rawimage object
        bias: bias object
        Should this be similar to BOSS 4 quadrants?
        """
        # subtract pixel by pixel dark # may need to subtract separately each quadrant. For now assuming 
        # a single set.
        biasimage=kwargs["BiasImage"]
        rimage=rawimage.pix
        bimage=biasimage.pix

        rx=rimage.shape[0]
        ry=rimage.shape[1]
        value=np.zeros((rx,ry))
    # check dimensionality:
        assert rx ==bimage.shape[0]
        assert ry==bimage.shape[1]
    #for ii in range(rx):
    #    for jj in range(ry):
        value=rimage-bimage
        dknoise=bimage.std()  # or some values read from somewhere?
        ivar_new=1./(rimage+bimage+dknoise**2)
        mask=rawimage.mask
        from desispec.image import Image as im
        return im(value,ivar_new,mask)
        #return rawimage

class BoxcarExtraction(PAs.PipelineAlg):
    from desispec.image import Image as im
    from  desispec.frame import Frame as fr
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Boxcar Extraction"
        from  desispec.frame import Frame as fr
        from desispec.image import Image as im
        PAs.PipelineAlg.__init__(self,name,im,fr,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise QLExceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise QLExceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        if "Band" not in kwargs:
            raise QLExceptions.ParameterException("Need Band name")
        if "Spectrograph" not in kwargs:
            raise QLExceptions.ParameterException("Need Spectrograph ID")
        if "PSFFile" not in kwargs:
            raise QLExceptions.ParameterException("Need PSF File")
        return self.do_boxcar(args[0],**kwargs)
    def get_default_config(self):
        return {("FiberMap","%%FiberMap","FiberMap object to map spectra to images"),
                ("Band","r","Which band to work on [r,b,z]"),
                ("Spectrograph",0,"Spectrograph to use [0-9]"),
                ("BoxWidth",2.5,"Boxcar halfwidth"),
                ("PSFFile","%%PSFFile","PSFFile to use"),
                ("Wmin",5625,"Lower limit of extrapolation wavelength"),
                ("Wmax",7741,"Upper limit of extrapolation wavelength"),
                ("DeltaW",0.5,"Binwidth of extrapolated wavelength array")
                }
    def do_boxcar(self,image,**kwargs):
        """ image  : desispec.image
        band: band [r,b,z]
        camera : camera ID
        """
        from desispec.frame import Frame
        import math
        psf=kwargs["PSFFile"]
        band=kwargs["Band"] #must arg
        camera=kwargs["Spectrograph"] #must arg
        boxWidth=2.5
        fiberMap=None
        if "FiberMap" in kwargs:fiberMap=kwargs["FiberMap"]
        if "BoxWidth" in kwargs:boxWidth=kwargs["BoxWidth"]
        dw=0.5
        if "DeltaW" in kwargs:dw=kwargs["DeltaW"]
        if band == "r":
            #if psffile is None:psffile=os.getenv('DESIMODEL')+"/data/specpsf/psf-r.fits"
            wmin=5625
            wmax=7741
            waves=np.arange(wmin,wmax,0.25)
            mask=np.zeros((4114,4128))
        elif band == "b":
            #if psffile is None:psffile=os.getenv('DESIMODEL')+"/data/specpsf/psf-b.fits"
            wmin=3569
            wmax=5949
            waves=np.arange(wmin,wmax,0.25)
            mask=np.zeros((4096,4096))
        elif band == "z":
            #if psffile is None:psffile=os.getenv('DESIMODEL')+"/data/specpsf/psf-z.fits"
            wmin=7435
            wmax=9834
            waves=np.arange(wmin,wmax,0.25)
            mask=np.zeros((4114,4128))
        else:
            print "Band can be r z or b"
            return None

        xs=psf.x(None,waves)
        ys=psf.y(None,waves)
        maxX,maxY=mask.shape
        maxX=maxX-1
        maxY=maxY-1
        ranges=np.zeros((mask.shape[1],xs.shape[0]+1),dtype=int)
        for bin in xrange(0,len(waves)):
            ixmaxOld=0
            for spec in xrange(0,xs.shape[0]):
                xpos=xs[spec][bin]
                ypos=int(ys[spec][bin])
                if xpos<0 or xpos>maxX or ypos<0 or ypos>maxY : 
                    continue 
                xmin=xpos-boxWidth
                xmax=xpos+boxWidth
                ixmin=int(math.floor(xmin))
                ixmax=int(math.floor(xmax))
                if ixmin <= ixmaxOld:
                    print "Error Box width overlaps,",xpos,ypos,ixmin,ixmaxOld
                    return None,None
                ixmaxOld=ixmax
                if mask[int(xpos)][ypos]>0 :
                    continue
            # boxing in x vals
                if ixmin < 0: #int value is less than 0
                    ixmin=0
                    rxmin=1.0
                else:# take part of the bin depending on real xmin
                    rxmin=1.0-xmin+ixmin
                if ixmax>maxX:# xmax is bigger than the image
                    ixmax=maxX
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
    
        if "Wmin" in kwargs:wmin=kwargs["Wmin"]
        if "Wmax" in kwargs:wmax=kwargs["Wmax"]
        if "DeltaW" in kwargs:dw=kwargs["DeltaW"]

        maskedImg=(image.pix*mask.T)
        flux=np.zeros((maskedImg.shape[0],ranges.shape[1]-1))
        for r in xrange(flux.shape[0]):
            row=np.add.reduceat(maskedImg[r],ranges[r])[:-1]
            flux[r]=row
        from desispec.interpolation import resample_flux
        wtarget=np.arange(wmin,wmax+dw/2.0,dw)
        Flux=np.zeros((500,len(wtarget)))
        ivar=np.zeros((500,len(wtarget)))
        resolution=np.zeros((500,21,len(wtarget)))
        #TODO get the approximate resolution matrix. Like in quicksim?
        for spec in xrange(flux.shape[1]):
            ww=psf.wavelength(spec)
            Flux[spec,:]=resample_flux(wtarget,ww,flux[:,spec])
            ivar[spec,:]=1./(Flux[spec,:]+image.readnoise)
        dwave=np.gradient(wtarget)
        Flux/=dwave
        ivar*=dwave**2
        #Extracted the full image but write frame in [nspec,nwave]
        nspec=500 # keeping all 500 spectra for now
            
        return Frame(wtarget,Flux[:nspec],ivar[:nspec],resolution_data=resolution[:nspec],spectrograph=camera)
