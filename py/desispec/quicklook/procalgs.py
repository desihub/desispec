"""
Pipeline Preprocessing algorithms for Quicklook
"""

import numpy as np
import os,sys
from desispec.quicklook import pas
from desispec.quicklook import qlexceptions,qllogger

qlog=qllogger.QLLogger("QuickLook",20)
log=qlog.getlog()

class Preproc_test(pas.PipelineAlg):

    #- this is not real preproc, but only for testing QAs, now input taking pix = output pix

    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Preproc Test"
        from desispec.image import Image as im
        pas.PipelineAlg.__init__(self,name,im,im,config,logger)

    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        input_image=args[0]
        
        return self.run_pa(input_image)

    def run_pa(self,image):
        return image

    def get_default_config(self):
        return {}


class BootCalibration(pas.PipelineAlg):
    from desispec import bootcalib as desiboot
    
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Boot Calibration"
        from desispec.frame import Frame as fr
        from desispec.image import Image as im
        pas.PipelineAlg.__init__(self,name,im,fr,config,logger)
        
    def run(self,*args,**kwargs):
        if len(args) == 0 : #- args[0] should be the fiberflat image
            raise qlexceptions.ParameterException("Missing input parameter")
        if 'ArcLampImage' not in kwargs: 
            raise qlexceptions.ParameterException("Need ArcLampImage")

        if "Deg" not in kwargs:
            deg=5 #- 5th order legendre polynomial
        else:
            deg=kwargs["Deg"]

        flatimage=args[0]
        arcimage=kwargs["ArcLampImage"]
        outputfile=kwargs["outputFile"]

        return self.run_pa(deg,flatimage,arcimage,outputfile)


    def run_pa(self,deg,flatimage,arcimage,outputfile):
        from desispec import bootcalib as desiboot
        xfit,fdicts,gauss,all_wv_soln=desiboot.bootcalib(deg,flatimage,arcimage)

        desiboot.write_psf(outputfile, xfit, fdicts, gauss,all_wv_soln)
        log.info("PSF file wrtten. Exiting Quicklook for this configuration.") #- File written no need to go further
        sys.exit(0)   

    

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
        nspec=kwargs["Nspec"]

        if "Wavelength" not in kwargs:
            wstart = np.ceil(psf.wmin)
            wstop = np.floor(psf.wmax)
            dw = 0.5
        else: 
            wavelength=kwargs["Wavelength"]
            if kwargs["Wavelength"] is not None: #- should be in wstart,wstop,dw format                
                wstart, wstop, dw = map(float, wavelength.split(','))
            else: 
                wstart = np.ceil(psf.wmin)
                wstop = np.floor(psf.wmax)
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

        camera = input_image.meta['CAMERA'].lower()     #- b0, r1, .. z9
        spectrograph = int(camera[1])
        fibermin = spectrograph*500 + specmin
        if "FiberMap" not in kwargs:
            fibermap = None
            fibers = np.arange(fibermin, fibermin+nspec, dtype='i4')
        else:
            fibermap=kwargs["FiberMap"]
            fibermap = fibermap[fibermin:fibermin+nspec]
            fibers = fibermap['FIBER']
        if "Outfile" in kwargs:
            outfile=kwargs["Outfile"]
        else:
            outfile=None

        return self.run_pa(input_image,psf,wave,boxwidth,nspec,fibers=fibers,fibermap=fibermap,outfile=outfile)


    def run_pa(self, input_image, psf, outwave, boxwidth, nspec,fibers=None, fibermap=None,outfile=None):
        from desispec.boxcar import do_boxcar
        from desispec.frame import Frame as fr
        
        flux,ivar,Rdata=do_boxcar(input_image, psf, outwave, boxwidth=boxwidth, 
nspec=nspec)

        #- write to a frame object
        
        frame = fr(outwave, flux, ivar, resolution_data=Rdata,fibers=fibers, meta=input_image.meta, fibermap=fibermap)
        
        if outfile is not None:  #- writing to a frame file if needed.
            from desispec import io
            io.write_frame(outfile,frame)
            log.info("wrote frame output file  %s"%outfile)

        return frame

  
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

        camera = input_image.meta['CAMERA'].lower()     #- b0, r1, .. z9
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
        if "ndecorr" in kwargs:
            ndecorr=ndecorr
        else: 
            ndecorr=True

        bundlesize=25 #- hard coded
      
        if "Outfile" in kwargs:
            outfile=kwargs["Outfile"]
        else:
            outfile=None

        if "Nwavestep" in kwargs:
            wavesize=kwargs["Nwavestep"]
        else:
            wavesize=50       

        return self.run_pa(input_image,psf,specmin,nspec,wave,regularize=regularize,ndecorr=ndecorr, bundlesize=bundlesize, wavesize=wavesize,outfile=outfile,fibers=fibers,fibermap=fibermap)

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
            log.info("wrote frame output file  %s"%outfile)

        return frame


class ComputeFiberflat(pas.PipelineAlg):
    """ PA to compute fiberflat field correction from a DESI continuum lamp frame
    """
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Compute Fiberflat"
        from desispec.frame import Frame as fr
        from desispec.image import Image as im
        pas.PipelineAlg.__init__(self,name,fr,fr,config,logger)

    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        input_frame=args[0] #- frame object to calculate fiberflat from
        if "outputFile" not in kwargs:
            raise qlexceptions.ParameterException("Need output file name to write fiberflat File")
        outputfile=kwargs["outputFile"]            

        return self.run_pa(input_frame,outputfile)
    
    def run_pa(self,input_frame,outputfile):
        from desispec.fiberflat import compute_fiberflat
        import desispec.io.fiberflat as ffIO
        fiberflat=compute_fiberflat(input_frame)
        ffIO.write_fiberflat(outputfile,fiberflat,header=input_frame.meta)
        log.info("Fiberflat file wrtten. Exiting Quicklook for this configuration") #- File written no need to go further
        sys.exit(0) 
 
class ApplyFiberFlat(pas.PipelineAlg):
    """
       PA to Apply the fiberflat field to the given frame
    """
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Apply FiberFlat"
        from desispec.frame import Frame as fr
        from desispec.image import Image as im
        pas.PipelineAlg.__init__(self,name,fr,fr,config,logger)

    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        if "FiberFlatFile" not in kwargs:
            raise qlexceptions.ParameterException("Need Fiberflat file")
        
        input_frame=args[0]
        fiberflat=kwargs["FiberFlatFile"]
        
        return self.run_pa(input_frame,fiberflat)

    def run_pa(self,input_frame,fiberflat): 
     
        from desispec.fiberflat import apply_fiberflat 
        apply_fiberflat(input_frame,fiberflat)
        return input_frame

class ApplyFiberFlat_QL(pas.PipelineAlg):
    """
       PA to Apply the fiberflat field (QL) to the given frame
    """
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Apply FiberFlat"
        from desispec.frame import Frame as fr
        from desispec.image import Image as im
        pas.PipelineAlg.__init__(self,name,fr,fr,config,logger)

    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        if "FiberFlatFile" not in kwargs:
            raise qlexceptions.ParameterException("Need Fiberflat file")
        
        input_frame=args[0]
        fiberflat=kwargs["FiberFlatFile"]
        
        return self.run_pa(input_frame,fiberflat)

    def run_pa(self,input_frame,fiberflat): 
     
        from desispec.quicklook.quickfiberflat import apply_fiberflat 
        fframe=apply_fiberflat(input_frame,fiberflat)
        return fframe

class ComputeSky(pas.PipelineAlg):
    """ PA to compute sky model from a DESI frame
    """
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Compute Sky"
        from desispec.frame import Frame as fr
        from desispec.image import Image as im
        pas.PipelineAlg.__init__(self,name,fr,fr,config,logger)

    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        if "FiberFlatFile" not in kwargs: #- need this as fiberflat has to apply to frame first
            raise qlexceptions.ParameterException("Need Fiberflat frame file")
        input_frame=args[0] #- frame object to calculate sky from
        if "FiberMap" in kwargs:
            fibermap=kwargs["FiberMap"]
        if "Outfile" not in kwargs:
            raise qlexceptions.ParameterException("Need output file name to write skymodel")
        fiberflat=kwargs["FiberFlatFile"]
        outputfile=kwargs["Outfile"]
        return self.run_pa(input_frame,fiberflat,outputfile)
    
    def run_pa(self,input_frame,fiberflat,outputfile):
        from desispec.fiberflat import apply_fiberflat
        from desispec.sky import compute_sky
        from desispec.io.sky import write_sky

        #- First apply fiberflat to sky fibers
        apply_fiberflat(input_frame,fiberflat)

        #- calculate the model
        skymodel=compute_sky(input_frame)
        write_sky(outputfile,skymodel,input_frame.meta)
        log.info("Sky Model file wrtten. Exiting pipeline for this configuration")
        sys.exit(0)


class ComputeSky_QL(pas.PipelineAlg):
    """ PA to compute sky model from a DESI frame
    """
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Compute Sky"
        from desispec.frame import Frame as fr
        from desispec.image import Image as im
        pas.PipelineAlg.__init__(self,name,fr,fr,config,logger)

    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        input_frame=args[0] #- frame object to calculate sky from. Should be fiber flat corrected
        if "FiberMap" in kwargs:
            fibermap=kwargs["FiberMap"]
        else: fibermap=None

        if "Outfile" not in kwargs:
            raise qlexceptions.ParameterException("Need output file name to write skymodel")

        outputfile=kwargs["Outfile"]
        return self.run_pa(input_frame,outputfile,fibermap=fibermap)
    
    def run_pa(self,input_frame,outputfile,fibermap=None): #- input frame should be already fiberflat fielded
        from desispec.io.sky import write_sky
        from desispec.quicklook.quicksky import compute_sky
       
        skymodel=compute_sky(input_frame,fibermap)                
        
        write_sky(outputfile,skymodel,input_frame.meta)
        log.info("Sky Model file wrtten. Exiting the pipeline for this configuration")
        sys.exit(0)

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
        if "SkyFile" not in kwargs:
            raise qlexceptions.ParameterException("Need Skymodel file")

        input_frame=args[0] #- this must be flat field applied before sky subtraction in the pipeline
        skyfile=kwargs["SkyFile"]    #- Read sky model file itself from an argument
        from desispec.io.sky import read_sky
        skymodel=read_sky(skyfile)
                   
        return self.run_pa(input_frame,skymodel)
    
    def run_pa(self,input_frame,skymodel):
        from desispec.sky import subtract_sky
        subtract_sky(input_frame,skymodel)
        return input_frame

class SubtractSky_QL(pas.PipelineAlg):
    """
       This is for QL Sky subtraction. The input frame object should be fiber flat corrected.
       Unlike offline, if no skymodel file is given as input, a sky compute method is called
       to create a skymodel object and then subtraction is performed. Outputing that skymodel
       to a file is optional and can be configured.
    """
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

        input_frame=args[0] #- this must be flat field applied before sky subtraction in the pipeline

        if "SkyFile" in kwargs:
            from desispec.io.sky import read_sky
            skyfile=kwargs["SkyFile"]    #- Read sky model file itself from an argument
            log.info("Using given sky file %s for subtraction"%skyfile)

            skymodel=read_sky(skyfile)

        else:
            if "Outskyfile" in kwargs:
                outskyfile=kwargs["Outskyfile"]
            else: outskyfile=None

            log.info("No sky file given. Computing sky first")
            from desispec.quicklook.quicksky import compute_sky
            fibermap=input_frame.fibermap
            skymodel=compute_sky(input_frame,fibermap)
            if outskyfile is not None:
                from desispec.io.sky import write_sky
                log.info("writing an output sky model file %s "%outskyfile)
                write_sky(outputfile,skymodel,input_frame.meta)

        #- now do the subtraction                   
        return self.run_pa(input_frame,skymodel)
    
    def run_pa(self,input_frame,skymodel):
        from desispec.quicklook.quicksky import subtract_sky
        sframe=subtract_sky(input_frame,skymodel)
        return sframe

