"""
Pipeline Preprocessing algorithms for Quicklook
"""

import numpy as np
import os,sys
from desispec.quicklook import pas
from desispec.quicklook import qlexceptions,qllogger

qlog=qllogger.QLLogger("QuickLook",20)
log=qlog.getlog()


class Initialize(pas.PipelineAlg):
    """
    This is particularly needed to run some QAs before preprocessing. 
    It reads rawimage and does input = output. e.g QA to run after this PA: bias from overscan etc"
    """

    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Ready"
        import astropy
        rawtype=astropy.io.fits.hdu.hdulist.HDUList
        pas.PipelineAlg.__init__(self,name,rawtype,rawtype,config,logger)

    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        input_raw=args[0]
            
        return self.run_pa(input_raw)

    def run_pa(self,raw):
        """ 
        We don't need to dump the raw file again here, so skipping"
        """
        return raw

    def get_default_config(self):
        return {}


class Preproc(pas.PipelineAlg):
    #- TODO: currently io itself seems to have the preproc inside it. And preproc does bias, pi
     # xelflat, etc in one step. 

    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Preproc"
        import astropy
        #- No raw object (like image or frame object) exists yet so,
        #- type from reading raw: eg: raw=fits.open('desi-00000002.fits.fz',memmap=False)
        #- rawtype=type(raw)
        rawtype=astropy.io.fits.hdu.hdulist.HDUList
        from desispec.image import Image as im
        pas.PipelineAlg.__init__(self,name,rawtype,im,config,logger)

    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        input_raw=args[0]

        dumpfile=None
        if "dumpfile" in kwargs:
            dumpfile=kwargs["dumpfile"]

        if 'camera' not in kwargs: 
            raise qlexceptions.ParameterException("Need Camera to run preprocess on raw files")
        else: 
            camera=kwargs["camera"]
        if camera.upper() not in input_raw:
            raise IOError('Camera {} not in raw input'.format(camera))
        if "Bias" in kwargs:
            bias=kwargs["Bias"]
        else: bias=False
    
        if "Pixflat" in kwargs:
            pixflat=kwargs["Pixflat"]
        else: pixflat=False

        if "Mask" in kwargs:
            mask=kwargs["Mask"]
        else: mask=False

        return self.run_pa(input_raw,camera,bias=bias,pixflat=pixflat,mask=mask,dumpfile=dumpfile)

    def run_pa(self,input_raw,camera,bias=False,pixflat=False,mask=False,dumpfile=None):
        import desispec.preproc

        rawimage=input_raw[camera.upper()].data
        header=input_raw[camera.upper()].header
        primary_header=input_raw[0].header
        if 'INHERIT' in header and header['INHERIT']:
            h0 = input_raw[0].header
            for key in h0:
                if key not in header:
                    header[key] = h0[key]
        #- WARNING!!!This is a hack for QL to run on old raw images for QLF to be working on old set of data
        if "PROGRAM" not in header:
            log.info("WARNING!!! Temporary hack for QL to add header key PROGRAM. Only to facilitate QLF to work on their dataset. Remove this after some time and run with new data set")
            header["PROGRAM"]= 'dark'
        if header["FLAVOR"] not in [None,'bias','arc','flat','science']:
            header["FLAVOR"] = 'science'        

        img = desispec.preproc.preproc(rawimage,header,primary_header,bias=bias,pixflat=pixflat,mask=mask)
        if dumpfile is not None:
            from desispec import io
            night = img.meta['NIGHT']
            expid = img.meta['EXPID']
            io.write_image(dumpfile, img)
            log.info("Wrote intermediate file %s after %s"%(dumpfile,self.name))
        return img


class BootCalibration(pas.PipelineAlg):
    from desispec import bootcalib as desiboot
    
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Boot Calibration"
        from desispec.frame import Frame as fr
        from desispec.image import Image as im
        pas.PipelineAlg.__init__(self,name,im,fr,config,logger)
        
    def run(self,*args,**kwargs):
        if 'ArcLampImage' not in kwargs: 
            raise qlexceptions.ParameterException("Need ArcLampImage")
        if 'FlatImage' not in kwargs:
            raise qlexceptions.ParameterException("Need FlatImage")
        if 'outputFile' not in kwargs:
            raise qlexceptions.ParameterException("Need outputFile")

        if "Deg" not in kwargs:
            deg=5 #- 5th order legendre polynomial
        else:
            deg=kwargs["Deg"]

        flatimage=kwargs["FlatImage"]
        arcimage=kwargs["ArcLampImage"]
        outputfile=kwargs["outputFile"]

        return self.run_pa(deg,flatimage,arcimage,outputfile,args)

    def run_pa(self,deg,flatimage,arcimage,outputfile,args):
        from desispec.util import runcmd
        cmd = "desi_bootcalib --arcfile {} --fiberflat {} --outfile {}".format(arcimage,flatimage,outputfile)
        if runcmd(cmd) !=0:
            raise RuntimeError('desi_bootcalib failed, psfboot not written')

        img=args[0]
        return img


class BoxcarExtract(pas.PipelineAlg):
    from desispec.image import Image as im
    from desispec.frame import Frame as fr
    from desispec.boxcar import do_boxcar

    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="BoxcarExtract"
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

        dumpfile=None
        if "dumpfile" in kwargs:
            dumpfile=kwargs["dumpfile"]

        psf=kwargs["PSFFile"]
        boxwidth=kwargs["BoxWidth"]
        nspec=kwargs["Nspec"]
        if "usesigma" in kwargs:
             usesigma=kwargs["usesigma"]
        else: usesigma = False

        if "Wavelength" not in kwargs:
            wstart = np.ceil(psf.wmin)
            wstop = np.floor(psf.wmax)
            dw = 0.5
        else: 
            wavelength=kwargs["Wavelength"]
            if kwargs["Wavelength"] is not None: #- should be in wstart,wstop,dw format                
                wstart, wstop, dw = [float(w) for w in wavelength]
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
        maskFile=None
        if "MaskFile" in kwargs:
            maskFile=kwargs['MaskFile']

        #- Add some header keys relevant for this extraction
        input_image.meta['NSPEC']   = (nspec, 'Number of spectra')
        input_image.meta['WAVEMIN'] = (wstart, 'First wavelength [Angstroms]')
        input_image.meta['WAVEMAX'] = (wstop, 'Last wavelength [Angstroms]')
        input_image.meta['WAVESTEP']= (dw, 'Wavelength step size [Angstroms]')

        return self.run_pa(input_image,psf
                           ,wave,boxwidth,nspec,
                           fibers=fibers,fibermap=fibermap,
                           dumpfile=dumpfile,maskFile=maskFile,usesigma=usesigma)

    def run_pa(self, input_image, psf, outwave, boxwidth, nspec,
               fibers=None, fibermap=None,dumpfile=None,
               maskFile=None,usesigma=False):
        from desispec.boxcar import do_boxcar
        from desispec.frame import Frame as fr
        import desispec.psf
        if fibermap['OBJTYPE'][0] == 'ARC':
            psf=desispec.psf.PSF(psf)
        flux,ivar,Rdata=do_boxcar(input_image, psf, outwave, boxwidth=boxwidth, 
                                  nspec=nspec,maskFile=maskFile,usesigma=usesigma)

        #- write to a frame object
        frame = fr(outwave, flux, ivar, resolution_data=Rdata,fibers=fibers, meta=input_image.meta, fibermap=fibermap)

        if dumpfile is not None:
            from desispec import io
            night = frame.meta['NIGHT']
            expid = frame.meta['EXPID']
            io.write_frame(dumpfile, frame)
            log.info("Wrote intermediate file %s after %s"%(dumpfile,self.name))

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
                wstart, wstop, dw = [float(w) for w in wavelength]
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
            name="ComputeFiberflat"
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

class ComputeFiberflat_QL(pas.PipelineAlg):
    """ PA to compute fiberflat field correction from a DESI continuum lamp frame
    """
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="ComputeFiberflat"
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
    
    def run_pa(self,frame,outputfile):
        from desispec.fiberflat import FiberFlat
        import desispec.io.fiberflat as ffIO
        from desispec.linalg import cholesky_solve
        nwave=frame.nwave
        nfibers=frame.nspec
        wave = frame.wave  #- this will become part of output too
        flux = frame.flux
        sumFlux=np.zeros((nwave))
        realFlux=np.zeros(flux.shape)
        ivar = frame.ivar*(frame.mask==0)
        #deconv
        for fib in range(nfibers):
            Rf=frame.R[fib].todense()
            B=flux[fib]
            realFlux[fib]=cholesky_solve(Rf,B)
            sumFlux+=realFlux[fib]
        #iflux=nfibers/sumFlux
        flat = np.zeros(flux.shape)
        flat_ivar=np.zeros(ivar.shape)
        avg=sumFlux/nfibers
        for fib in range(nfibers):
            Rf=frame.R[fib]
            # apply and reconvolute
            M=Rf.dot(avg)
            M0=(M==0)
            flat[fib]=(~M0)*flux[fib]/(M+M0) +M0
            flat_ivar[fib]=ivar[fib]*M**2
        fibflat=FiberFlat(frame.wave.copy(),flat,flat_ivar,frame.mask.copy(),avg)
        
        #fiberflat=compute_fiberflat(input_frame)
        ffIO.write_fiberflat(outputfile,fiberflat,header=input_frame.meta)
 
class ApplyFiberFlat(pas.PipelineAlg):
    """
       PA to Apply the fiberflat field to the given frame
    """
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="ApplyFiberFlat"
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

        dumpfile=None
        if "dumpfile" in kwargs:
            dumpfile=kwargs["dumpfile"]

        fiberflat=kwargs["FiberFlatFile"]
        
        return self.run_pa(input_frame,fiberflat,dumpfile=dumpfile)

    def run_pa(self,input_frame,fiberflat,dumpfile=None): 
     
        from desispec.quicklook.quickfiberflat import apply_fiberflat 
        fframe=apply_fiberflat(input_frame,fiberflat)

        if dumpfile is not None:
            from desispec import io
            night = fframe.meta['NIGHT']
            expid = fframe.meta['EXPID']
            io.write_frame(dumpfile, fframe)
            log.info("Wrote intermediate file %s after %s"%(dumpfile,self.name))

        return fframe

class ComputeSky(pas.PipelineAlg):
    """ PA to compute sky model from a DESI frame
    """
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="ComputeSky"
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


class ComputeSky_QL(pas.PipelineAlg):
    """ PA to compute sky model from a DESI frame
    """
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="ComputeSky_QL"
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

        if "Apply_resolution" in kwargs:
            apply_resolution=kwargs["Apply_resolution"]

        if "Outfile" not in kwargs:
            raise qlexceptions.ParameterException("Need output file name to write skymodel")

        outputfile=kwargs["Outfile"]
        return self.run_pa(input_frame,outputfile,fibermap=fibermap,apply_resolution=apply_resolution)
    
    def run_pa(self,input_frame,outputfile,fibermap=None,apply_resolution=False): #- input frame should be already fiberflat fielded
        from desispec.io.sky import write_sky
        from desispec.quicklook.quicksky import compute_sky
       
        skymodel=compute_sky(input_frame,fibermap,apply_resolution=apply_resolution)                
        
        write_sky(outputfile,skymodel,input_frame.meta)

class SkySub(pas.PipelineAlg):

    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="SkySub"
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
        return (input_frame, skymodel)

class SkySub_QL(pas.PipelineAlg):
    """
       This is for QL Sky subtraction. The input frame object should be fiber flat corrected.
       Unlike offline, if no skymodel file is given as input, a sky compute method is called
       to create a skymodel object and then subtraction is performed. Outputing that skymodel
       to a file is optional and can be configured.
    """
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="SkySub_QL"
        from  desispec.frame import Frame as fr
        from desispec.image import Image as im
        pas.PipelineAlg.__init__(self,name,fr,type(tuple),config,logger)

    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))

        input_frame=args[0] #- this must be flat field applied before sky subtraction in the pipeline

        dumpfile=None
        if "dumpfile" in kwargs:
            dumpfile=kwargs["dumpfile"]

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
            if "Apply_resolution" in kwargs:
                apply_resolution=kwargs["Apply_resolution"]
                log.info("Apply fiber to fiber resolution variation in computing sky")
            else: apply_resolution = False
            fibermap=input_frame.fibermap
            skymodel=compute_sky(input_frame,fibermap,apply_resolution=apply_resolution)
            if outskyfile is not None:
                from desispec.io.sky import write_sky
                log.info("writing an output sky model file %s "%outskyfile)
                write_sky(outskyfile,skymodel,input_frame.meta)

        #- now do the subtraction                   
        return self.run_pa(input_frame,skymodel,dumpfile=dumpfile)
    
    def run_pa(self,input_frame,skymodel,dumpfile=None):
        from desispec.quicklook.quicksky import subtract_sky
        sframe=subtract_sky(input_frame,skymodel)

        if dumpfile is not None:
            from desispec import io
            night = sframe.meta['NIGHT']
            expid = sframe.meta['EXPID']
            io.write_frame(dumpfile, sframe)
            log.info("Wrote intermediate file %s after %s"%(dumpfile,self.name))

        return (sframe,skymodel)

class ResolutionFit(pas.PipelineAlg):

    """
    Fitting of Arc lines on extracted arc spectra, polynomial expansion of the fitted sigmas, and updating
    the coefficients to the new PSF file
    """ 
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="ResolutionFit"
        from  desispec.frame import Frame as fr
        pas.PipelineAlg.__init__(self,name,fr,fr,config,logger)

    def run(self,*args,**kwargs):
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        if not kwargs["PSFbootfile"]:
             raise qlexceptions.ParameterException("Missing psfbootfile in the arguments")

        if "PSFoutfile" not in kwargs:
             raise qlexceptions.ParameterException("Missing psfoutfile in the arguments")

        psfoutfile=kwargs["PSFoutfile"]
        psfbootfile=kwargs["PSFbootfile"] 

        if "usesigma" in kwargs:
             usesigma=kwargs["usesigma"]
        else: usesigma = False

        from desispec.psf import PSF
        psfboot=PSF(psfbootfile)
        domain=(psfboot.wmin,psfboot.wmax)

        input_frame=args[0]

        linelist=None
        if "Linelist" in kwargs:
            linelist=kwargs["Linelist"]

        npoly=2
        if "NPOLY" in kwargs:
            npoly=kwargs["NPOLY"]
        nbins=2
        if "NBINS" in kwargs:
            nbins=kwargs["NBINS"]

        return self.run_pa(input_frame,psfbootfile,psfoutfile,usesigma,linelist=linelist,npoly=npoly,nbins=nbins,domain=domain)
    
    def run_pa(self,input_frame,psfbootfile,outfile,usesigma,linelist=None,npoly=2,nbins=2,domain=None):
        from desispec.quicklook.arcprocess import process_arc,write_psffile
        from desispec.quicklook.palib import get_resolution
        from desispec.psf import PSF

        wcoeffs=process_arc(input_frame,linelist=linelist,npoly=npoly,nbins=nbins,domain=domain)

        #- write out the psf outfile
        wstep=input_frame.meta["WAVESTEP"]
        write_psffile(psfbootfile,wcoeffs,outfile,wavestepsize=wstep)
        log.info("Wrote psf file {}".format(outfile))

        #- update the arc frame resolution from new coeffs
        newpsf=PSF(outfile)
        input_frame.resolution_data=get_resolution(input_frame.wave,input_frame.nspec,newpsf,usesigma=usesigma)
 
        return input_frame

