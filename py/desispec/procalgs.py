"""
Pipeline Preprocessing algorithms for Quicklook
"""

import numpy as np
import os
from desispec.quicklook import pas
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
        value=rimage-dimage
        dknoise=dimage.std()  # or some values read from somewhere?
        ivar_new=1./(rimage+dimage+dknoise**2)
        mask=rawimage.mask
        from desispec.image import Image as im
        return im(value,ivar_new,mask)

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
        pixelflat=kwargs["PixelFlat"]
        pflat=pixelflat.pix # should be read from desispec.io
        rx=pflat.shape[0]
        ry=pflat.shape[1]
        
    # check dimensionality
        assert rx ==image.pix.shape[0]
        assert ry==image.pix.shape[1]
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
        value=rimage-bimage
        dknoise=bimage.std()  # or some values read from somewhere?
        ivar_new=1./(rimage+bimage+dknoise**2)
        mask=rawimage.mask
        from desispec.image import Image as im
        return im(value,ivar_new,mask)

class BoxcarExtraction(pas.PipelineAlg):
    from desispec.image import Image as im
    from desispec.frame import Frame as fr
    from desispec.boxcar import do_boxcar

    
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
                ("DeltaW",0.5,"Binwidth of extrapolated wavelength array")
                }
    def boxcar_extract(self,image,**kwargs):
        from desispec.boxcar import do_boxcar
        from specter.psf import load_psf
        psf=kwargs["PSFFile"] # TODO This is confusing, PSFFile looks like a file not an object. 
        band=kwargs["Band"]
        camera=kwargs["Spectrograph"]
        boxwidth=kwargs["BoxWidth"]
       
        return do_boxcar(image,band,psf,camera,boxwidth=2.5,dw=0.5,nspec=500)

class SubtractSky(pas.PipelineAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Sky Subtraction"
        from  desispec.frame import Frame as fr
        from desispec.image import Image as im
        pas.PipelineAlg.__init__(self,name,im,fr,config,logger)
    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        if "FiberMap" not in kwargs:
            raise qlexceptions.ParameterException("Need Fibermap file")
        if "FiberFlat" not in kwargs:
            raise qlexceptions.ParameterException("Need Fiberflat file")
        fiber_flat=self.fiberflat(args[0],**kwargs)
        sky_model=self.computesky(args[0],**kwargs)
        return self.subtractsky(args[0],fiber_flat,sky_model)

    
    def fiberflat(self,frame,**kwargs):
        from desispec.io import read_fiberflat
        fiberflat=read_fiberflat(kwargs["FiberFlat"]) # need a fiberflat file
        return fiberflat
        #apply_fiberflat(frame,fiberflat)
    
    def computesky(self,frame,**kwargs):
        from desispec.sky import compute_sky
        skymodel=compute_sky(frame,kwargs["FiberMap"])
        return skymodel
    
    def subtractsky(self,frame,fiberflat,skymodel):
        from desispec.sky import subtract_sky
        from desispec.fiberflat import apply_fiberflat
        
        apply_fiberflat(frame,fiberflat)  
        subtract_sky(frame,skymodel)
        return frame
