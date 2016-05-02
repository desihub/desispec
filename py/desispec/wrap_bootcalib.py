'''
Not sure if this should be here and even the name of this file. 
This is trying to address the comment: from sbailey on quicklook PR 159
"desispec/bin/desi_bootcalib.py should be refactored into a lightweight script wrappering an algorithm, such that quicklook.BootCalibration.do_bootcalib can call that algorithm instead of having to replicate much of that script"

Offline pipeline also seems to be using the same so far. So may be used for offline also.

'''

import numpy as np
from desispec import bootcalib as desiboot
from desiutil import funcfits as dufits
from desispec.io import read_image


def wrap_bootcalib(deg,flatimage,arcimage):

    """
       deg: Legendre polynomial degree to use to fit
       flatimage: desispec.image.Image object of flatfield
       arcimage: desispec.image.Image object of arc

    #- Mostly inherited from desispec/bin/desi_bootcalib directly as needed

      returns xfit
              fdicts
              gauss
              all_wave_soln ; as defined in desispec/bin/desi_bootcalib and/or desispec.scripts.bootcalib.py
   """    

    camera=flatimage.camera
    flat=flatimage.pix
    ny=flat.shape[0]

    xpk,ypos,cut=desiboot.find_fiber_peaks(flat)
    xset,xerr=desiboot.trace_crude_init(flat,xpk,ypos)
    xfit,fdicts=desiboot.fit_traces(xset,xerr)
    gauss=desiboot.fiber_gauss(flat,xfit,xerr)

    #- Also need wavelength solution not just trace

    arc=arcimage.pix
    all_spec=desiboot.extract_sngfibers_gaussianpsf(arc,xfit,gauss)
    llist=desiboot.load_arcline_list(camera)
    dlamb,wmark,gd_lines,line_guess=desiboot.load_gdarc_lines(camera)
        
    #- Solve for wavelengths
    all_wv_soln=[]
    all_dlamb=[]
    for ii in range(all_spec.shape[1]):
        spec=all_spec[:,ii]
        pixpk=desiboot.find_arc_lines(spec)
        id_dict=desiboot.id_arc_lines(pixpk,gd_lines,dlamb,wmark,line_guess=line_guess)
        id_dict['fiber']=ii
        #- Find the other good ones
        if camera == 'z':
            inpoly = 3  # The solution in the z-camera has greater curvature
        else:
            inpoly = 2
        desiboot.add_gdarc_lines(id_dict, pixpk, gd_lines, inpoly=inpoly)
        #- Now the rest
        desiboot.id_remainder(id_dict, pixpk, llist)
        #- Final fit wave vs. pix too
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

    return xfit, fdicts, gauss,all_wv_soln


