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
        camera=rawimage.camera
        meta=rawimage.meta
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
        return im(value,ivar_new,mask,camera=camera,meta=meta)

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
        
        #-export kwargs 
        input_image=args[0]
        pixelflat=kwargs["PixelFlat"]
        
        return self.run_pa(input_image,pixelflat)

    def run_pa(self,input_image,pixelflat): #- explicit arguments
        """
           input_image: image to apply pixelflat to
           pixelflat object
        """ 
        return self.do_pixelFlat(input_image,pixelflat)

    def get_default_config(self):
        return {("PixelFlat","%%PixelFlat","Pixel Flat image to apply")}

    def do_pixelFlat(self,image, pixelflat):

        """
        image: image object typically after dark subtraction)
        pixelflat: a pixel flat object
        """
        pflat=pixelflat.pix # should be read from desispec.io
        camera=image.camera
        meta=image.meta
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
        return im(value,ivar,mask,camera=camera,meta=meta)

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

        input_image=args[0]
        biasimage=kwargs["BiasImage"]

        return self.run_pa(input_image,biasimage)

    def run_pa(self,input_image,biasimage):
        return self.do_biasSubtract(input_image,biasimage)

    def get_default_config(self):
        return {("BiasImage","%%BiasImage","Bias image to subtract")}

    def do_biasSubtract(self,rawimage,biasimage):
        """ rawimage: rawimage object
        bias: bias object
        Should this be similar to BOSS 4 quadrants?
        """
        # subtract pixel by pixel dark # may need to subtract separately each quadrant. For now assuming 
        # a single set.

        rimage=rawimage.pix
        camera=rawimage.camera
        meta=rawimage.meta
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
        return im(value,ivar_new,mask,camera=camera,meta=meta)


class BootCalibration(pas.PipelineAlg):
    from desispec import bootcalib as desiboot
    
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Boot Calibration"
        from desispec.frame import Frame as fr
        from desispec.frame import Image as im
        pas.PipelineAlg.__init__(self,name,im,fr,config,logger)
        
    def run(self,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if 'FiberFlatImage' not in kwargs:
            raise qlexceptions.ParameterException("Need FiberFlatImage")
        if 'ArcLampImage' not in kwargs: 
            raise qlexceptions.ParameterException("Need ArcLampImage")

        if "Deg" not in kwargs:
            deg=5 #- 5th order legendre polynomial
        else:
            deg=kwargs["Deg"]

        flatimage=kwargs["FiberFlatImage"]
        arcimage=kwargs["ArcLampImage"]
        outputfile=kwargs["outputFile"]

        return self.run_pa(deg,flatimage,arcimage,outputfile)


    def run_pa(self,deg,flatimage,arcimage,outputfile):
        from desispec import bootcalib as desiboot
        xfit,fdicts,gauss,all_wv_soln=desiboot.bootcalib(deg,flatimage,arcimage)

        desiboot.write_psf(outputfile, xfit, fdicts, gauss,all_wv_soln)

    

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
        if "PSFFile" not in kwargs:
            raise qlexceptions.ParameterException("Need PSF File")

        input_image=args[0]
        psf=kwargs["PSFFile"]
        boxwidth=kwargs["BoxWidth"]
        dw=kwargs["DeltaW"]
        nspec=kwargs["Nspec"]

        return self.run_pa(input_image,psf,boxwidth,dw,nspec)

    def run_pa(self, input_image, psf, boxwidth, dw, nspec):
        from desispec.boxcar import do_boxcar
        return do_boxcar(input_image, psf, boxwidth=boxwidth, dw=dw, nspec=nspec)

  
    def get_default_config(self):
        return {("BoxWidth",2.5,"Boxcar halfwidth"),
                ("PSFFile","%%PSFFile","PSFFile to use"),
                ("DeltaW",0.5,"Binwidth of extrapolated wavelength array")
                ("Nspec",500,"number of spectra to extract")
                }

# TODO 2d extraction runs fine as well. Will need more testing of the setup.

class Extraction_2d(pas.PipelineAlg):
    """ 
       Offline 2D extraction for offline QuickLook
    """
    from desispec.image import Image as im
    from desispec.frame import Frame as fr

    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="2D Extraction" # using specter.extract.ex2d
        from  desispec.frame import Frame as fr
        from desispec.image import Image as im
        pas.PipelineAlg.__init__(self,name,im,fr,config,logger)
 
    def run(self,*args,**kwargs):

        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        if "PSFFile_sp" not in kwargs:
            raise qlexceptions.ParameterException("Need PSF File")
        from specter.psf import load_psf

        input_image=args[0]
        psffile=kwargs["PSFFile_sp"]
        psf=load_psf(psffile)

        if "Wavelength" not in kwargs:
            wstart = np.ceil(psf.wmin_all)
            wstop = np.floor(psf.wmax_all)
            dw = 0.5
        else: 
            wavelength=kwargs["Wavelength"]
            if kwargs["Wavelength"] is not None: #- should be in wstart,wstop,dw format                
                wstart, wstop, dw = map(float, wavelength.split(','))
            else: 
                wstart = np.ceil(psf.wmin_all)
                wstop = np.floor(psf.wmax_all)
                dw = 0.5            
        wave = np.arange(wstart, wstop+dw/2.0, dw)

        if "Specmin" not in kwargs:
            specmin=0
        else:
            specmin=kwargs["Specmin"]
            if kwargs["Specmin"] is None:
               specmin=0

        if "Nspec" not in kwargs:
            nspec = psf.nspec
        else:
            nspec=kwargs["Nspec"]
            if nspec is None:
                nspec=psf.nspec

        specmax = specmin + nspec

        camera = input_image.meta['CAMERA']     #- b0, r1, .. z9
        spectrograph = int(camera[1])
        fibermin = spectrograph*500 + specmin
  
        if "FiberMap" not in kwargs:
            fibermap = None
            fibers = np.arange(fibermin, fibermin+nspec, dtype='i4')
        else:
            fibermap=kwargs["FiberMap"]
            fibermap = fibermap[fibermin:fibermin+nspec]
            fibers = fibermap['FIBER']
        if "Regularize" in kwargs:
            regularize=kwargs["Regularize"]
        else:
            regularize=False
        bundlesize=25 #- hard coded
      
        if "Outfile" in kwargs:
            outfile=kwargs["Outfile"]
        else:
            outfile=None

        if "Nwavestep" in kwargs:
            wavesize=kwargs["Nwavestep"]
        else:
            wavesize=50       

        return self.run_pa(input_image,psf,specmin,nspec,wave,regularize=regularize,bundlesize=bundlesize, wavesize=wavesize,outfile=outfile,fibers=fibers,fibermap=fibermap)

    def run_pa(self,input_image,psf,specmin,nspec,wave,regularize=None,ndecorr=True,bundlesize=25,wavesize=50, outfile=None,fibers=None,fibermap=None):
        import specter
        from specter.extract import ex2d
        from desispec.frame import Frame as fr

        flux,ivar,Rdata=ex2d(input_image.pix,input_image.ivar*(input_image.mask==0),psf,specmin,nspec,wave,regularize=regularize,ndecorr=ndecorr,bundlesize=bundlesize,wavesize=wavesize)

        #- Augment input image header for output
        input_image.meta['NSPEC']   = (nspec, 'Number of spectra')
        input_image.meta['WAVEMIN'] = (wave[0], 'First wavelength [Angstroms]')
        input_image.meta['WAVEMAX'] = (wave[-1], 'Last wavelength [Angstroms]')
        input_image.meta['WAVESTEP']= (wave[1]-wave[0], 'Wavelength step size [Angstroms]')
        input_image.meta['SPECTER'] = (specter.__version__, 'https://github.com/desihub/specter')
        #input_image.meta['IN_PSF']  = (_trim(psf_file), 'Input spectral PSF')
        #input_image.meta['IN_IMG']  = (_trim(input_file), 'Input image')

        frame = fr(wave, flux, ivar, resolution_data=Rdata,fibers=fibers, meta=input_image.meta, fibermap=fibermap)
        
        if outfile is not None:  #- writing to a frame file if needed.
            from desispec import io
            io.write_frame(outfile,frame)
            
        return frame

#TODO: Everything below and beyond extraction is to be tested. Adding PA steps as placeholder. Not configured yet.
 
class FiberFlat(pas.PipelineAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Fiber Flatfield"
        from desispec.frame import Frame as fr
        from desispec.image import Image as im
        pas.PipelineAlg.__init__(self,name,fr,fr,config,logger)

    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        if "FiberFlatFrame" not in kwargs:
            raise qlexceptions.ParameterException("Need Fiberflat frame file")
        
        input_frame=args[0]
        fiberflat=kwargs["FiberFlatFrame"]
        
        return self.run_pa(input_frame,fiberflat)

    def run_pa(self,input_frame,friberflat): 
     
        from desispec.fiberflat import compute_fiberflat,apply_fiberflat
        fiberflat=compute_fiberflat(input_frame) 
        return apply_fiberflat(input_frame,fiberflat)


class SubtractSky(pas.PipelineAlg):

    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Sky Subtraction"
        from  desispec.frame import Frame as fr
        from desispec.image import Image as im
        pas.PipelineAlg.__init__(self,name,fr,fr,config,logger)

    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        
        input_frame=args[0]
           
        return self.run_pa(input_frame)
    
    def run_pa(self,input_frame):
        from desispec.sky import subtract_sky
        skymodel=compute_sky(frame)
        return subtract_sky(frame,skymodel)
