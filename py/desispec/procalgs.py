"""
Pipeline Preprocessing algorithms for Quicklook
"""

import numpy as np
import os
from desispec.quicklook import pas

#rimage=read_image(filename) # image object 
#dark=read_dark(filename) # dark object

#def dark_subtract(rawimage,darkimage):
from desispec.quicklook import qlexceptions
class DarkSubtraction(pas.PipelineAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Dark Subtraction"
        from desispec.image import Image as im
        pas.PipelineAlg.__init__(self,name,im,im,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        if "DarkImage" not in kwargs:
            raise qlexceptions.ParameterException("Need Dark Image")
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

class PixelFlattening(pas.PipelineAlg):
    from desispec.image import Image as im
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Pixel Flattening"
        from desispec.image import Image as im
        pas.PipelineAlg.__init__(self,name,im,im,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        if "PixelFlat" not in kwargs:
            raise qlexceptions.ParameterException("Need Pixel Flat image")
        return self.do_pixelFlat(args[0],**kwargs)
    def get_default_config(self):
        return {("PixelFlat","%%PixelFlat","Pixel Flat image to apply")}
    def do_pixelFlat(self,image, **kwargs):

        """
        image: image object typically after dark subtraction)
        pixelflat: a pixel flat object
        """
    #pflat=read_image(pixelflat)
        pixelflat=kwargs["PixelFlat"]
        pflat=pixelflat.pix # should be read from desispec.io
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

class BiasSubtraction(pas.PipelineAlg):
    from desispec.image import Image as im
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Bias Subtraction"
        from desispec.image import Image as im
        pas.PipelineAlg.__init__(self,name,im,im,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        if "BiasImage" not in kwargs:
            raise qlexceptions.ParameterException("Need Bias image")
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

class BoxcarExtraction(pas.PipelineAlg):
    from desispec.image import Image as im
    from  desispec.frame import Frame as fr
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Boxcar Extraction"
        from  desispec.frame import Frame as fr
        from desispec.image import Image as im
        pas.PipelineAlg.__init__(self,name,im,fr,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        if "Band" not in kwargs:
            raise qlexceptions.ParameterException("Need Band name")
        if "Spectrograph" not in kwargs:
            raise qlexceptions.ParameterException("Need Spectrograph ID")
        if "PSFFile" not in kwargs:
            raise qlexceptions.ParameterException("Need PSF File")
        return self.boxcar_extract(args[0],**kwargs)
    def get_default_config(self):
        return {("Band","r","Which band to work on [r,b,z]"),
                ("Spectrograph",0,"Spectrograph to use [0-9]"),
                ("BoxWidth",2.5,"Boxcar halfwidth"),
                ("PSFFile","%%PSFFile","PSFFile to use"),
                ("Wmin",5625,"Lower limit of extrapolation wavelength"),
                ("Wmax",7741,"Upper limit of extrapolation wavelength"),
                ("DeltaW",0.5,"Binwidth of extrapolated wavelength array")
                }
    def boxcar_extract(self,image,**kwargs):
        from desispec.boxcar import do_boxcar
        from specter.psf import load_psf
        psf=kwargs["PSFFile"]
        #psf=load_psf(psffile)
        band=kwargs["Band"]
        camera=kwargs["Spectrograph"]
        boxwidth=kwargs["BoxWidth"]
        
        return do_boxcar(image,band,psf,camera,boxwidth=2.5,dw=0.5,nspec=500)


