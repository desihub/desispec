"""
Pipeline Preprocessing algorithms for Quicklook
"""

import numpy as np
import os,sys
import astropy
import astropy.io.fits as fits 
from desispec import io
from desispec.io import read_raw,read_image
from desispec.io.meta import findfile
from desispec.io.fluxcalibration import read_average_flux_calibration
from desispec.calibfinder import findcalibfile
from desispec.quicklook import pas
from desispec.quicklook import qlexceptions,qllogger
from desispec.image import Image as im
from desispec.frame import Frame as fr
from desispec.io.xytraceset import read_xytraceset
from desispec.maskbits import ccdmask

qlog=qllogger.QLLogger("QuickLook",20)
log=qlog.getlog()


class Initialize(pas.PipelineAlg):
    """
    This PA takes information from the fibermap and raw header
    and adds it to the general info section of the merged dictionary
    """
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Ready"
        rawtype=astropy.io.fits.hdu.hdulist.HDUList
        pas.PipelineAlg.__init__(self,name,rawtype,rawtype,config,logger)

    def run(self,*args,**kwargs):
        if len(args) == 0 :
            log.critical("Missing input parameter!")
            sys.exit()
        if not self.is_compatible(type(args[0])):
            log.critical("Incompatible input!")
            sys.exit("Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))

        raw=args[0]
        flavor=kwargs['Flavor']
        peaks=None
        fibermap=None
        if flavor != 'bias' and flavor != 'dark':
            fibermap=kwargs['FiberMap']
            peaks=kwargs['Peaks']
        camera=kwargs['Camera']

        return self.run_pa(raw,fibermap,camera,peaks,flavor)

    def run_pa(self,raw,fibermap,camera,peaks,flavor):
        import pytz
        import datetime
        from desitarget.targetmask import desi_mask
        from desispec.fluxcalibration import isStdStar

        #- Create general info dictionary to be sent to merged json
        general_info={}

        #- Get information from raw header
        general_info['PROGRAM']=program=raw[0].header['PROGRAM'].upper()
        calibs=['arcs','flat','bias','dark']

        if not flavor in calibs:
            general_info['AIRMASS']=raw[0].header['AIRMASS']
            general_info['SEEING']=raw[0].header['SEEING']

            #- Get information from fibermap
    
            #- Limit flux info to fibers in camera
            minfiber=int(camera[1])*500
            maxfiber=minfiber+499
            fibermags=[]
            for flux in ['FLUX_G','FLUX_R','FLUX_Z']:
                fibermags.append(22.5-2.5*np.log10(fibermap[flux][minfiber:maxfiber+1]))

            #- Set sky/no flux fibers to 30 mag
            for i in range(3):
                skyfibs=np.where(fibermags[i]==0.)[0]
                noflux=np.where(fibermags[i]==np.inf)[0]
                badmags=np.array(list(set(skyfibs) | set(noflux)))
                fibermags[i][badmags]=30.

            general_info['FIBER_MAGS']=fibermags
    
            #- Limit RA and DEC to 5 decimal places
            targetra=fibermap['TARGET_RA'][minfiber:maxfiber+1]
            general_info['RA']=[float("%.5f"%ra) for ra in targetra]
            targetdec=fibermap['TARGET_DEC'][minfiber:maxfiber+1]
            general_info['DEC']=[float("%.5f"%dec) for dec in targetdec]
    
            #- Find fibers in camera per target type
            elgfibers=np.where((fibermap['DESI_TARGET']&desi_mask.ELG)!=0)[0]
            general_info['ELG_FIBERID']=[elgfib for elgfib in elgfibers if minfiber <= elgfib <= maxfiber]
            lrgfibers=np.where((fibermap['DESI_TARGET']&desi_mask.LRG)!=0)[0]
            general_info['LRG_FIBERID']=[lrgfib for lrgfib in lrgfibers if minfiber <= lrgfib <= maxfiber]
            qsofibers=np.where((fibermap['DESI_TARGET']&desi_mask.QSO)!=0)[0]
            general_info['QSO_FIBERID']=[qsofib for qsofib in qsofibers if minfiber <= qsofib <= maxfiber]
            skyfibers=np.where((fibermap['DESI_TARGET']&desi_mask.SKY)!=0)[0]
            general_info['SKY_FIBERID']=[skyfib for skyfib in skyfibers if minfiber <= skyfib <= maxfiber]
            general_info['NSKY_FIB']=len(general_info['SKY_FIBERID'])
            stdfibers=np.where(isStdStar(fibermap))[0]
            general_info['STAR_FIBERID']=[stdfib for stdfib in stdfibers if minfiber <= stdfib <= maxfiber]

        general_info['EXPTIME']=raw[0].header['EXPTIME']
#        general_info['FITS_DESISPEC_VERION']=raw[0].header['FITS_DESISPEC_VERSION']
#        general_info['PROC_DESISPEC_VERION']=raw[0].header['PROC_DESISPEC_VERSION']
#        general_info['PROC_QuickLook_VERION']=raw[0].header['PROC_QuickLook_VERSION']

        #- Get peaks from configuration file
        if not flavor != 'arcs' and flavor in calibs:
            general_info['B_PEAKS']=peaks['B_PEAKS']
            general_info['R_PEAKS']=peaks['R_PEAKS']
            general_info['Z_PEAKS']=peaks['Z_PEAKS']

        #- Get current time information
        general_info['QLrun_datime_UTC']=datetime.datetime.now(tz=pytz.utc).isoformat()

        return (raw,general_info)


class Preproc(pas.PipelineAlg):
    #- TODO: currently io itself seems to have the preproc inside it. And preproc does bias, pi
     # xelflat, etc in one step. 
    from desispec.maskbits import ccdmask
    
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Preproc"

        rawtype=astropy.io.fits.hdu.hdulist.HDUList
        pas.PipelineAlg.__init__(self,name,rawtype,im,config,logger)

    def run(self,*args,**kwargs):
        if len(args) == 0 :
            #raise qlexceptions.ParameterException("Missing input parameter")
            log.critical("Missing input parameter!")
            sys.exit()

        if not self.is_compatible(type(args[0])):
            #raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
            log.critical("Incompatible input!")
            sys.exit("Was expecting {} got {}".format(type(self.__inpType__),type(args[0])))

        input_raw=args[0][0]

        dumpfile=None
        if "dumpfile" in kwargs:
            dumpfile=kwargs["dumpfile"]

        if 'camera' not in kwargs: 
            #raise qlexceptions.ParameterException("Need Camera to run preprocess on raw files")
            log.critical("Need Camera to run preprocess on raw files")
            sys.exit()

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

    def run_pa(self,input_raw,camera,bias=False,pixflat=False,mask=True,dumpfile='ttt1.fits'):
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
        #if "PROGRAM" not in header:
        #    log.warning("Temporary hack for QL to add header key PROGRAM. Only to facilitate QLF to work on their dataset. Remove this after some time and run with new data set")
        #    header["PROGRAM"]= 'dark'
        #if header["FLAVOR"] not in [None,'bias','arc','flat','science']:
        #    header["FLAVOR"] = 'science'        

        img = desispec.preproc.preproc(rawimage,header,primary_header,bias=bias,pixflat=pixflat,mask=mask)
                
        
        if img.mask is not None :
            img.pix *= (img.mask==0)
        
        
        if dumpfile is not None:
            night = img.meta['NIGHT']
            expid = img.meta['EXPID']
            io.write_image(dumpfile, img)
            log.debug("Wrote intermediate file %s after %s"%(dumpfile,self.name))
        

        return img


class Flexure(pas.PipelineAlg):
    """ Use desi_compute_trace_shifts to output modified psf file
    """
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Flexure"
        pas.PipelineAlg.__init__(self,name,im,fr,config,logger)

    def run(self,*args,**kwargs):
        if 'preprocFile' not in kwargs:
            #raise qlexceptions.ParameterException("Must provide preproc file for desi_compute_trace_shifts")
            log.critical("Must provide preproc file for desi_compute_trace_shifts")
            sys.exit()

        if 'inputPSFFile' not in kwargs:
            #raise qlexceptions.ParameterException("Must provide input psf file desi_compute_trace_shifts")
            log.critical("Must provide input psf file desi_compute_trace_shifts")
            sys.exit()
        
        if 'outputPSFFile' not in kwargs:
            #raise qlexceptions.ParameterException("Must provide output psf file")
            log.critical("Must provide output psf file")
            sys.exit()

        preproc_file=kwargs["preprocFile"]
        input_file=kwargs["inputPSFFile"]
        output_file=kwargs["outputPSFFile"]

        return self.run_pa(preproc_file,input_file,output_file,args)

    def run_pa(self,preproc_file,input_file,output_file,args):
        from desispec.util import runcmd
        #- Generate modified psf file
        cmd="desi_compute_trace_shifts --image {} --psf {} --outpsf {}".format(preproc_file,input_file,output_file)
        if runcmd(cmd) !=0:
            raise RuntimeError('desi_compute_trace_shifts failed, psftrace not written')

        #- return image object to pass to boxcar for extraction
        img=args[0]
        return img


class BoxcarExtract(pas.PipelineAlg):
    from desispec.quicklook.qlboxcar import do_boxcar
    from desispec.maskbits import ccdmask
    
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="BoxcarExtract"
        pas.PipelineAlg.__init__(self,name,im,fr,config,logger)

    def run(self,*args,**kwargs):

        if len(args) == 0 :
            #raise qlexceptions.ParameterException("Missing input parameter")
            log.critical("Missing input parameter")
            sys.exit()

        if not self.is_compatible(type(args[0])):
            #raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
            log.critical("Incompatible input!")
            sys.exit("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))

        if "PSFFile" not in kwargs:
            #raise qlexceptions.ParameterException("Need PSF File")
            log.critical("Need PSF File")
            sys.exit()

        input_image=args[0]

        dumpfile=None
        if "dumpfile" in kwargs:
            dumpfile=kwargs["dumpfile"]

        flavor=kwargs["Flavor"]

        psf_filename=kwargs["PSFFile"]
        #psf = PSF(psf_filename)
        tset = read_xytraceset(psf_filename)
        boxwidth=kwargs["BoxWidth"]
        nspec=kwargs["Nspec"]
        quickRes=kwargs["QuickResolution"] if "QuickResolution" in kwargs else False
        if "usesigma" in kwargs:
             usesigma=kwargs["usesigma"]
        else: usesigma = False
        
        if "Wavelength" not in kwargs:
            wstart = np.ceil(tset.wavemin)
            wstop = np.floor(tset.wavemax)
            dw = 0.5
        else: 
            wavelength=kwargs["Wavelength"]
            if kwargs["Wavelength"] is not None: #- should be in wstart,wstop,dw format                
                wstart, wstop, dw = [float(w) for w in wavelength]
            else: 
                wstart = np.ceil(tset.wavemin)
                wstop = np.floor(tset.wavemax)
                dw = 0.5            
        wave = np.arange(wstart, wstop+dw/2.0, dw)
        if "Specmin" not in kwargs:
            specmin=0
        else:
            specmin=kwargs["Specmin"]
            if kwargs["Specmin"] is None:
               specmin=0

        if "Nspec" not in kwargs:
            nspec = tset.nspec
        else:
            nspec=kwargs["Nspec"]
            if nspec is None:
                nspec=tset.nspec

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

        return self.run_pa(input_image,flavor,tset,wave,boxwidth,nspec,
                           fibers=fibers,fibermap=fibermap,dumpfile=dumpfile,
                           maskFile=maskFile,usesigma=usesigma,quick_resolution=quickRes)

    def run_pa(self,input_image,flavor,tset,outwave,boxwidth,nspec,
               fibers=None,fibermap=None,dumpfile=None,
               maskFile=None,usesigma=False,quick_resolution=False):
        from desispec.quicklook.qlboxcar import do_boxcar
        #import desispec.tset
        flux,ivar,Rdata=do_boxcar(input_image, tset, outwave, boxwidth=boxwidth, 
                                  nspec=nspec,maskFile=maskFile,usesigma=usesigma,
                                  quick_resolution=quick_resolution)

        #- write to a frame object
        qndiag=21
        wsigma=None
        if quick_resolution:
            log.warning("deprecated, please use QFrame format to store sigma values")
            wsigma=np.zeros(flux.shape)
            if tset.ysig_vs_wave_traceset is not None :
                dw = np.gradient(outwave)
                for i in range(nspec):
                    ysig = tset.ysig_vs_wave(i,outwave)
                    y    = tset.y_vs_wave(i,outwave)
                    dydw = np.gradient(y)/dw
                    wsigma[i] = ysig/dydw # in A
        frame = fr(outwave, flux, ivar, resolution_data=Rdata,fibers=fibers, 
                   meta=input_image.meta, fibermap=fibermap,
                   wsigma=wsigma,ndiag=qndiag)

        if dumpfile is not None:
            night = frame.meta['NIGHT']
            expid = frame.meta['EXPID']
            io.write_frame(dumpfile, frame)
            log.debug("Wrote intermediate file %s after %s"%(dumpfile,self.name))

        return frame
    
    def get_default_config(self):
        return {("BoxWidth",2.5,"Boxcar halfwidth"),
                ("PSFFile","%%PSFFile","PSFFile to use"),
                ("DeltaW",0.5,"Binwidth of extrapolated wavelength array"),
                ("Nspec",500,"number of spectra to extract")
                }


# TODO 2d extraction runs fine as well. Will need more testing of the setup.

class Extraction_2d(pas.PipelineAlg):
    """ 
       Offline 2D extraction for offline QuickLook
    """
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="2D Extraction" # using specter.extract.ex2d
        pas.PipelineAlg.__init__(self,name,im,fr,config,logger)
 
    def run(self,*args,**kwargs):

        if len(args) == 0 :
            #raise qlexceptions.ParameterException("Missing input parameter")
            log.critical("Missing input parameter")
            sys.exit()

        if not self.is_compatible(type(args[0])):
            #raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
            log.critical("Incompatible input!")
            sys.exit("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))

        if "PSFFile_sp" not in kwargs:
            #raise qlexceptions.ParameterException("Need PSF File")
            log.critical("Need PSF File")
            sys.exit()

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
            io.write_frame(outfile,frame)
            log.debug("wrote frame output file  %s"%outfile)

        return frame


class ComputeFiberflat(pas.PipelineAlg):
    """ PA to compute fiberflat field correction from a DESI continuum lamp frame
    """
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="ComputeFiberflat"
        pas.PipelineAlg.__init__(self,name,fr,fr,config,logger)

    def run(self,*args,**kwargs):
        if len(args) == 0 :
            #raise qlexceptions.ParameterException("Missing input parameter")
            log.critical("Missing input parameter")
            sys.exit()

        if not self.is_compatible(type(args[0])):
            #raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
            log.critical("Incompatible input!")
            sys.exit("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))

        input_frame=args[0] #- frame object to calculate fiberflat from
        if "outputFile" not in kwargs:
            #raise qlexceptions.ParameterException("Need output file name to write fiberflat File")
            log.critical("Need output file name to write fiberflat File")
            sys.exit()

        outputfile=kwargs["outputFile"]            

        return self.run_pa(input_frame,outputfile)
    
    def run_pa(self,input_frame,outputfile):
        from desispec.fiberflat import compute_fiberflat
        import desispec.io.fiberflat as ffIO
        fiberflat=compute_fiberflat(input_frame)
        ffIO.write_fiberflat(outputfile,fiberflat,header=input_frame.meta)
        log.debug("Fiberflat file wrtten. Exiting Quicklook for this configuration") #- File written no need to go further
        # !!!!! SAMI to whoever wrote this
        # PA's or any other components *CANNOT* call sys.exit()!! this needs to be fixed!!!!!
        sys.exit(0) 

class ComputeFiberflat_QL(pas.PipelineAlg):
    """ PA to compute fiberflat field correction from a DESI continuum lamp frame
    """
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="ComputeFiberflat"
        pas.PipelineAlg.__init__(self,name,fr,fr,config,logger)

    def run(self,*args,**kwargs):
        if len(args) == 0 :
            #raise qlexceptions.ParameterException("Missing input parameter")
            log.critical("Missing input parameter")
            sys.exit()

        if not self.is_compatible(type(args[0])):
            #raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
            log.critical("Incompatible input!")
            sys.exit("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        
        input_frame=args[0] #- frame object to calculate fiberflat from
        if "outputFile" not in kwargs:
            #raise qlexceptions.ParameterException("Need output file name to write fiberflat File")
            log.critical("Need output file name to write fiberflat File")
            sys.exit()
        
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
            try:
                realFlux[fib]=cholesky_solve(Rf,B)
            except:
                log.warning("cholesky_solve failed for fiber {}, using numpy.linalg.solve instead.".format(fib))
                realFlux[fib]=np.linalg.solve(Rf,B)
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
        ffIO.write_fiberflat(outputfile,fibflat,header=frame.meta)
        log.info("Wrote fiberflat file {}".format(outputfile))

        fflatfile = ffIO.read_fiberflat(outputfile)

        return fflatfile

class ApplyFiberFlat(pas.PipelineAlg):
    """
       PA to Apply the fiberflat field to the given frame
    """
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="ApplyFiberFlat"
        pas.PipelineAlg.__init__(self,name,fr,fr,config,logger)

    def run(self,*args,**kwargs):
        if len(args) == 0 :
            #raise qlexceptions.ParameterException("Missing input parameter")
            log.critical("Missing input parameter")
            sys.exit()

        if not self.is_compatible(type(args[0])):
            #raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
            log.critical("Incompatible input!")
            sys.exit("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))

        if "FiberFlatFile" not in kwargs:
            #raise qlexceptions.ParameterException("Need Fiberflat file")
            log.critical("Need Fiberflat file")
            sys.exit()

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
        pas.PipelineAlg.__init__(self,name,fr,fr,config,logger)

    def run(self,*args,**kwargs):
        if len(args) == 0 :
            #raise qlexceptions.ParameterException("Missing input parameter")
            log.critical("Missing input parameter!")
            sys.exit()

        if not self.is_compatible(type(args[0])):
            #raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
            log.critical("Incompatible input!")
            sys.exit("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))

        if "FiberFlatFile" not in kwargs:
            #raise qlexceptions.ParameterException("Need Fiberflat file")
            log.critical("Need Fiberflat file")
            sys.exit()

        
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
            night = fframe.meta['NIGHT']
            expid = fframe.meta['EXPID']
            io.write_frame(dumpfile, fframe)
            log.debug("Wrote intermediate file %s after %s"%(dumpfile,self.name))

        return fframe


class ComputeSky(pas.PipelineAlg):
    """ PA to compute sky model from a DESI frame
    """
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="ComputeSky"
        pas.PipelineAlg.__init__(self,name,fr,fr,config,logger)

    def run(self,*args,**kwargs):
        if len(args) == 0 :
            #raise qlexceptions.ParameterException("Missing input parameter")
            log.critical("Missing input parameter!")
            sys.exit()
    
        if not self.is_compatible(type(args[0])):
            #raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
            log.critical("Incompatible input!")
            sys.exit("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))

        if "FiberFlatFile" not in kwargs: #- need this as fiberflat has to apply to frame first
            #raise qlexceptions.ParameterException("Need Fiberflat frame file")
            log.critical("Need Fiberflat frame file!")
            sys.exit()

        input_frame=args[0] #- frame object to calculate sky from
        if "FiberMap" in kwargs:
            fibermap=kwargs["FiberMap"]
        if "Outfile" not in kwargs:
            #raise qlexceptions.ParameterException("Need output file name to write skymodel")
            log.critical("Incompatible input!")
            sys.exit("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))

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
        log.debug("Sky Model file wrtten. Exiting pipeline for this configuration")
        sys.exit(0)

class ComputeSky_QL(pas.PipelineAlg):
    """ PA to compute sky model from a DESI frame
    """
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="ComputeSky_QL"
        pas.PipelineAlg.__init__(self,name,fr,fr,config,logger)

    def run(self,*args,**kwargs):
        if len(args) == 0 :
            #raise qlexceptions.ParameterException("Missing input parameter")
            log.critical("Missing input parameter!")
            sys.exit()

        if not self.is_compatible(type(args[0])):
            #raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
            log.critical("Incompatible input!")
            sys.exit("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))

        input_frame=args[0] #- frame object to calculate sky from. Should be fiber flat corrected
        if "FiberMap" in kwargs:
            fibermap=kwargs["FiberMap"]
        else: fibermap=None

        if "Apply_resolution" in kwargs:
            apply_resolution=kwargs["Apply_resolution"]

        if "Outfile" not in kwargs:
            #raise qlexceptions.ParameterException("Need output file name to write skymodel")
            log.critical("Need output file name to write skymodel!")
            sys.exit()

        outputfile=kwargs["Outfile"]
        return self.run_pa(input_frame,outputfile,fibermap=fibermap,apply_resolution=apply_resolution)
    
    def run_pa(self,input_frame,outputfile,fibermap=None,apply_resolution=False): #- input frame should be already fiberflat fielded
        from desispec.io.sky import write_sky
        from desispec.quicklook.quicksky import compute_sky
       
        skymodel=compute_sky(input_frame,fibermap,apply_resolution=apply_resolution)                
        
        write_sky(outputfile,skymodel,input_frame.meta)
        # SEE ABOVE COMMENT!!!!
        log.debug("Sky Model file wrtten. Exiting the pipeline for this configuration")
        sys.exit(0)

class SkySub(pas.PipelineAlg):

    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="SkySub"
        pas.PipelineAlg.__init__(self,name,fr,fr,config,logger)

    def run(self,*args,**kwargs):
        if len(args) == 0 :
            #raise qlexceptions.ParameterException("Missing input parameter")
            log.critical("Missing input parameter!")
            sys.exit()

        if not self.is_compatible(type(args[0])):
            #raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
            log.critical("Incompatible input!")
            sys.exit("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))

        if "SkyFile" not in kwargs:
            #raise qlexceptions.ParameterException("Need Skymodel file")
            log.critical("Need Skymodel file!")
            sys.exit()


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
        pas.PipelineAlg.__init__(self,name,fr,type(tuple),config,logger)

    def run(self,*args,**kwargs):
        if len(args) == 0 :
            #raise qlexceptions.ParameterException("Missing input parameter")
            log.critical("Missing input parameter!")
            sys.exit()

        if not self.is_compatible(type(args[0])):
            #raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
            log.critical("Incompatible input!")
            sys.exit("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))

        input_frame=args[0] #- this must be flat field applied before sky subtraction in the pipeline

        dumpfile=None
        if "dumpfile" in kwargs:
            dumpfile=kwargs["dumpfile"]

        if "SkyFile" in kwargs:
            from desispec.io.sky import read_sky
            skyfile=kwargs["SkyFile"]    #- Read sky model file itself from an argument
            log.debug("Using given sky file %s for subtraction"%skyfile)

            skymodel=read_sky(skyfile)

        else:
            if "Outskyfile" in kwargs:
                outskyfile=kwargs["Outskyfile"]
            else: outskyfile=None

            log.debug("No sky file given. Computing sky first")
            from desispec.quicklook.quicksky import compute_sky
            if "Apply_resolution" in kwargs:
                apply_resolution=kwargs["Apply_resolution"]
                log.debug("Apply fiber to fiber resolution variation in computing sky")
            else: apply_resolution = False
            fibermap=input_frame.fibermap
            skymodel=compute_sky(input_frame,fibermap,apply_resolution=apply_resolution)
            if outskyfile is not None:
                from desispec.io.sky import write_sky
                log.debug("writing an output sky model file %s "%outskyfile)
                write_sky(outskyfile,skymodel,input_frame.meta)

        #- now do the subtraction                   
        return self.run_pa(input_frame,skymodel,dumpfile=dumpfile)
    
    def run_pa(self,input_frame,skymodel,dumpfile=None):
        from desispec.quicklook.quicksky import subtract_sky
        sframe=subtract_sky(input_frame,skymodel)

        if dumpfile is not None:
            night = sframe.meta['NIGHT']
            expid = sframe.meta['EXPID']
            io.write_frame(dumpfile, sframe)
            log.debug("Wrote intermediate file %s after %s"%(dumpfile,self.name))

        return (sframe,skymodel)

class ApplyFluxCalibration(pas.PipelineAlg):
    """PA to apply flux calibration to the given sframe
    """
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Apply Flux Calibration"
        pas.PipelineAlg.__init__(self,name,fr,fr,config,logger)

    def run(self,*args,**kwargs):
        if len(args) == 0 :
            log.critical("Missing input parameter!")
            sys.exit()

        if not self.is_compatible(type(args[0][0])):
            log.critical("Incompatible input!")
            sys.exit("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0][0])))

        input_frame=args[0][0]

        if "outputfile" in kwargs:
            outputfile=kwargs["outputfile"]
        else:
            log.critical("Must provide output file to write cframe")
            sys.exit()

        return self.run_pa(input_frame,outputfile=outputfile)

    def run_pa(self,frame,outputfile=None):
        night=frame.meta['NIGHT']
        camera=frame.meta['CAMERA']
        expid=frame.meta['EXPID']

        rawfile=findfile('raw',night,expid,rawdata_dir=os.environ['QL_SPEC_DATA'])
        rawfits=fits.open(rawfile)
        primary_header=rawfits[0].header
        image=read_raw(rawfile,camera)

        fluxcalib_filename=findcalibfile([image.meta,primary_header],"FLUXCALIB")

        fluxcalib = read_average_flux_calibration(fluxcalib_filename)
        log.info("read average calib in {}".format(fluxcalib_filename))
        seeing  = frame.meta["SEEING"]
        airmass = frame.meta["AIRMASS"]
        exptime = frame.meta["EXPTIME"]
        exposure_calib = fluxcalib.value(seeing=seeing,airmass=airmass)
        for q in range(frame.nspec) :
            fiber_calib=np.interp(frame.wave[q],fluxcalib.wave,exposure_calib)*exptime
            inv_calib = (fiber_calib>0)/(fiber_calib + (fiber_calib==0))
            frame.flux[q] *= inv_calib
            frame.ivar[q] *= fiber_calib**2*(fiber_calib>0)

        write_qframe(outputfile,frame)
        log.info("Wrote flux calibrated frame file %s after %s"%(outputfile,self.name))

        return frame

class ResolutionFit(pas.PipelineAlg):

    """
    Fitting of Arc lines on extracted arc spectra, polynomial expansion of the fitted sigmas, and updating
    the coefficients to the new traceset file
    """ 
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="ResolutionFit"
        pas.PipelineAlg.__init__(self,name,fr,fr,config,logger)

    def run(self,*args,**kwargs):
        if len(args) == 0 :
            #raise qlexceptions.ParameterException("Missing input parameter")
            log.critical("Missing input parameter!")
            sys.exit()

        if not self.is_compatible(type(args[0])):
            #raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
            log.critical("Incompatible input!")
            sys.exit("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))

        if "PSFoutfile" not in kwargs:
            #raise qlexceptions.ParameterException("Missing psfoutfile in the arguments")
            log.critical("Missing psfoutfile in the arguments!")
            sys.exit()

        psfoutfile=kwargs["PSFoutfile"]
        psfinfile=kwargs["PSFinputfile"] 

        if "usesigma" in kwargs:
             usesigma=kwargs["usesigma"]
        else: usesigma = False

        tset = read_xytraceset(psfinfile)
        domain=(tset.wavemin,tset.wavemax)

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

        return self.run_pa(input_frame,psfinfile,psfoutfile,usesigma,linelist=linelist,npoly=npoly,nbins=nbins,domain=domain)
    
    def run_pa(self,input_frame,psfinfile,outfile,usesigma,linelist=None,npoly=2,nbins=2,domain=None):
        from desispec.quicklook.arcprocess import process_arc,write_psffile
        from desispec.quicklook.palib import get_resolution
        
        wcoeffs,wavemin,wavemax =process_arc(input_frame,linelist=linelist,npoly=npoly,nbins=nbins,domain=domain)
        write_psffile(psfinfile,wcoeffs,wavemin,wavemax,outfile)
        log.debug("Wrote xytraceset file {}".format(outfile))
        
        #- update the arc frame resolution from new coeffs
        tset = read_xytraceset(outfile)
        input_frame.resolution_data=get_resolution(input_frame.wave,input_frame.nspec,tset,usesigma=usesigma)
 
        return (tset,input_frame)


# =======================
# qproc algorithms
# =======================

from desispec.sky import SkyModel
from desispec.qproc.io import write_qframe
from desispec.qproc.qextract import qproc_boxcar_extraction
from desispec.qproc.qfiberflat import qproc_apply_fiberflat 
from desispec.qproc.qsky import qproc_sky_subtraction

class Extract_QP(pas.PipelineAlg):

    
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Extract_QP"
        pas.PipelineAlg.__init__(self,name,im,fr,config,logger)

    def run(self,*args,**kwargs):

        if len(args) == 0 :
            #raise qlexceptions.ParameterException("Missing input parameter")
            log.critical("Missing input parameter!")
            sys.exit()
       
        if not self.is_compatible(type(args[0])):
            #raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
            log.critical("Incompatible input!")
            sys.exit("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))

        if "PSFFile" not in kwargs:
            #raise qlexceptions.ParameterException("Need PSF File")
            log.critical("Need PSF file!")
            sys.exit()

        input_image=args[0]

        dumpfile=None
        if "dumpfile" in kwargs:
            dumpfile=kwargs["dumpfile"]
        
        psf_filename=kwargs["PSFFile"]
        print("psf_filename=",psf_filename)

        traceset = read_xytraceset(psf_filename)
        
        width=kwargs["FullWidth"]
        nspec=kwargs["Nspec"]
        if "Wavelength" not in kwargs:
            wstart = np.ceil(traceset.wavemin)
            wstop = np.floor(traceset.wavemax)
            dw = 0.5
        else: 
            wavelength=kwargs["Wavelength"]
            print('kwargs["Wavelength"]=',kwargs["Wavelength"])
            if kwargs["Wavelength"] is not None: #- should be in wstart,wstop,dw format                
                wstart, wstop, dw = [float(w) for w in wavelength]
            else: 
                wstart = np.ceil(traceset.wmin)
                wstop = np.floor(traceset.wmax)
                dw = 0.5            
        wave = np.arange(wstart, wstop+dw/2.0, dw)
        if "Specmin" not in kwargs:
            specmin=0
        else:
            specmin=kwargs["Specmin"]
            if kwargs["Specmin"] is None:
               specmin=0

        if "Nspec" not in kwargs:
            nspec = traceset.nspec
        else:
            nspec=kwargs["Nspec"]
            if nspec is None:
                nspec=traceset.nspec

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
        

        
        return self.run_pa(input_image,traceset,wave,width,nspec,
                           fibers=fibers,fibermap=fibermap,dumpfile=dumpfile,
                           maskFile=maskFile)

    def run_pa(self,input_image,traceset,outwave,width,nspec,
               fibers=None,fibermap=None,dumpfile=None,
               maskFile=None):
        
        qframe = qproc_boxcar_extraction(traceset,input_image,fibers=fibers, width=width, fibermap=fibermap)
        
        if dumpfile is not None:
            write_qframe(dumpfile, qframe, fibermap=fibermap)
            log.debug("Wrote intermediate file %s after %s"%(dumpfile,self.name))
        
        return qframe
        
  
    def get_default_config(self):
        return {("FullWidth",7,"Boxcar full width"),
                ("PSFFile","%%PSFFile","PSFFile to use"),
                ("DeltaW",0.5,"Binwidth of extrapolated wavelength array"),
                ("Nspec",500,"number of spectra to extract")
                }

class ComputeFiberflat_QP(pas.PipelineAlg):
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="ComputeFiberflat"
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

    def run_pa(self,qframe,outputfile):
        from desispec.qproc.qfiberflat import qproc_compute_fiberflat
        import desispec.io.fiberflat as ffIO

        fibflat=qproc_compute_fiberflat(qframe)

        ffIO.write_fiberflat(outputfile,fibflat,header=qframe.meta)
        log.info("Wrote fiberflat file {}".format(outputfile))

        fflatfile = ffIO.read_fiberflat(outputfile)

        return fflatfile

class ApplyFiberFlat_QP(pas.PipelineAlg):
    """
       PA to Apply the fiberflat field (QP) to the given qframe
    """
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="Apply FiberFlat"
        pas.PipelineAlg.__init__(self,name,fr,fr,config,logger)

    def run(self,*args,**kwargs):
        if len(args) == 0 :
            #raise qlexceptions.ParameterException("Missing input parameter")
            log.critical("Missing input parameter!")
            sys.exit()

        if not self.is_compatible(type(args[0])):
            #raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
            log.critical("Incompatible input!")
            sys.exit("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))

        if "FiberFlatFile" not in kwargs:
            #raise qlexceptions.ParameterException("Need Fiberflat file")
            log.critical("Need Fiberflat file!")
            sys.exit()

        input_qframe=args[0]

        dumpfile=None
        if "dumpfile" in kwargs:
            dumpfile=kwargs["dumpfile"]

        fiberflat=kwargs["FiberFlatFile"]
        
        return self.run_pa(input_qframe,fiberflat,dumpfile=dumpfile)

    def run_pa(self,qframe,fiberflat,dumpfile=None): 

        
        qproc_apply_fiberflat(qframe,fiberflat)
        
        if dumpfile is not None:
            night = qframe.meta['NIGHT']
            expid = qframe.meta['EXPID']
            write_qframe(dumpfile, qframe)
            log.debug("Wrote intermediate file %s after %s"%(dumpfile,self.name))
        
        return qframe

class SkySub_QP(pas.PipelineAlg):
    """
       Sky subtraction. The input frame object should be fiber flat corrected.
       No sky model is saved for now
    """
    def __init__(self,name,config,logger=None):
        if name is None or name.strip() == "":
            name="SkySub_QP"
        pas.PipelineAlg.__init__(self,name,fr,type(tuple),config,logger)

    def run(self,*args,**kwargs):
        if len(args) == 0 :
            #raise qlexceptions.ParameterException("Missing input parameter")
            log.critical("Missing input parameter!")
            sys.exit()
        if not self.is_compatible(type(args[0])):
            #raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
            log.critical("Incompatible input!")
            sys.exit("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))

        input_qframe=args[0] #- this must be flat field applied before sky subtraction in the pipeline

        dumpfile=None
        if "dumpfile" in kwargs:
            dumpfile=kwargs["dumpfile"]

        #- now do the subtraction                   
        return self.run_pa(input_qframe,dumpfile=dumpfile)
    
    def run_pa(self,qframe,dumpfile=None):

        skymodel = qproc_sky_subtraction(qframe,return_skymodel=True)
        #qproc_sky_subtraction(qframe)
        
        if dumpfile is not None:
            night = qframe.meta['NIGHT']
            expid = qframe.meta['EXPID']
            write_qframe(dumpfile, qframe)
            log.debug("Wrote intermediate file %s after %s"%(dumpfile,self.name))
        
        # convert for QA
#        sframe=qframe.asframe()
#        tmpsky=np.interp(sframe.wave,qframe.wave[0],skymodel[0])
#        skymodel = SkyModel(sframe.wave,np.tile(tmpsky,(sframe.nspec,1)),np.ones(sframe.flux.shape),np.zeros(sframe.flux.shape,dtype="int32"))
        
        return (qframe,skymodel)
