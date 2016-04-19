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

        return self.do_bootcalib(**kwargs)

    def do_bootcalib(self,**kwargs):
        import numpy as np
        from desispec import bootcalib as desiboot
        from desiutil import funcfits as dufits
        from desispec.io import read_image
        if "Deg" not in kwargs:
            deg=5 #- 5th order legendre polynomial
        else:
            deg=kwargs["Deg"]
        flatimage=kwargs["FiberFlatImage"]
        camera=flatimage.camera
        flat=flatimage.pix
        ny=flat.shape[0]
        #- Somewhat inherited from desispec/bin/desi_bootcalib.py directly as needed

        xpk,ypos,cut=desiboot.find_fiber_peaks(flat)
        xset,xerr=desiboot.trace_crude_init(flat,xpk,ypos)
        xfit,fdicts=desiboot.fit_traces(xset,xerr)
        gauss=desiboot.fiber_gauss(flat,xfit,xerr)

        #- Also need wavelength solution not just trace
        arcimage=kwargs["ArcLampImage"]
        arc=arcimage.pix
        all_spec=desiboot.extract_sngfibers_gaussianpsf(arc,xfit,gauss)
        llist=desiboot.load_arcline_list(camera)
        dlamb,wmark,gd_lines,line_guess=desiboot.load_gdarc_lines(camera)
        # Solve for wavelengths
        all_wv_soln=[]
        all_dlamb=[]
        for ii in range(all_spec.shape[1]):
            spec=all_spec[:,ii]
            pixpk=desiboot.find_arc_lines(spec)
            id_dict=desiboot.id_arc_lines(pixpk,gd_lines,dlamb,wmark,line_guess=line_guess)
            id_dict['fiber']=ii
            # Find the other good ones
            if camera == 'z':
                inpoly = 3  # The solution in the z-camera has greater curvature
            else:
                inpoly = 2
            desiboot.add_gdarc_lines(id_dict, pixpk, gd_lines, inpoly=inpoly)
            #- Now the rest
            desiboot.id_remainder(id_dict, pixpk, llist)
            # Final fit wave vs. pix too
            final_fit, mask = dufits.iter_fit(np.array(id_dict['id_wave']), np.array(id_dict['id_pix']), 'polynomial', 3, xmin=0., xmax=1.)
            rms = np.sqrt(np.mean((dufits.func_val(np.array(id_dict['id_wave'])[mask==0],final_fit)-np.array(id_dict['id_pix'])[mask==0])**2))
            final_fit_pix,mask2 = dufits.iter_fit(np.array(id_dict['id_pix']), np.array(id_dict['id_wave']),'legendre',deg, niter=5)

            id_dict['final_fit'] = final_fit
            id_dict['rms'] = rms
            id_dict['final_fit_pix'] = final_fit_pix
            id_dict['wave_min'] = dufits.func_val(0,final_fit_pix)
            id_dict['wave_max'] = dufits.func_val(ny-1,final_fit_pix)
            id_dict['mask'] = mask
            all_wv_soln.append(id_dict)

        desiboot.write_psf(kwargs["outputFile"], xfit, fdicts, gauss,all_wv_soln)

    

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
        from desispec.boxcar import do_boxcar
        if len(args) == 0 :
            raise qlexceptions.ParameterException("Missing input parameter")
        if not self.is_compatible(type(args[0])):
            raise qlexceptions.ParameterException("Incompatible input. Was expecting %s got %s"%(type(self.__inpType__),type(args[0])))
        if "PSFFile" not in kwargs:
            raise qlexceptions.ParameterException("Need PSF File")
        return do_boxcar(args[0],kwargs["PSFFile"],boxwidth=kwargs["BoxWidth"],dw=kwargs["DeltaW"],nspec=kwargs["Nspec"])

    def get_default_config(self):
        return {("BoxWidth",2.5,"Boxcar halfwidth"),
                ("PSFFile","%%PSFFile","PSFFile to use"),
                ("DeltaW",0.5,"Binwidth of extrapolated wavelength array")
                ("Nspec",500,"number of spectra to extract")
                }

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
        from desispec.fiberflat import compute_fiberflat,apply_fiberflat
        fiberflat=compute_fiberflat(args[0],kwargs["FiberFlatFrame"])
        return apply_fiberflat(args[0],fiberflat)


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

        return self.do_sky_subtract(args[0],**kwargs)
    
    def do_sky_subtract(self,frame,**kwargs):
        from desispec.sky import subtract_sky
        skymodel=compute_sky(frame)
        return subtract_sky(frame,skymodel)
