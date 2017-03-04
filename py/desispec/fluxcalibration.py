"""
desispec.fluxcalibration
========================

Flux calibration routines.
"""
from __future__ import absolute_import
import numpy as np
from .resolution import Resolution
from .linalg import cholesky_solve, cholesky_solve_and_invert, spline_fit
from .interpolation import resample_flux
from .log import get_logger
from .io.filters import load_filter
from desispec import util
import scipy, scipy.sparse, scipy.ndimage
import sys
import time
from astropy import units
import multiprocessing
#import scipy.interpolate

#rebin spectra into new wavebins. This should be equivalent to desispec.interpolation.resample_flux. So may not be needed here
#But should move from here anyway.

def rebinSpectra(spectra,oldWaveBins,newWaveBins):
    tck=scipy.interpolate.splrep(oldWaveBins,spectra,s=0,k=1)
    specnew=scipy.interpolate.splev(newWaveBins,tck,der=0)
    return specnew

def applySmoothingFilter(flux) :
    # it was checked that the width of the median_filter has little impact on best fit stars
    # smoothing the ouput (with a spline for instance) does not improve the fit
    return scipy.ndimage.filters.median_filter(flux,200)
#
# Import some global constants.
#
# Why not use astropy constants?
#
# This is VERY inconvenient when trying to build documentation!
# The documentation may be build in an environment that does not have
# scipy installed.  There is no obvious reason why this has to be a module-level
# calculation.
#
import scipy.constants as const
h=const.h
pi=const.pi
e=const.e
c=const.c
erg=const.erg
try:
    hc = const.h/const.erg*const.c*1.e10  # (in units of ergsA)
except TypeError:
    hc = 1.9864458241717586e-08

def resample_template(data_wave_per_camera,resolution_data_per_camera,template_wave,template_flux,template_id) :
    output_flux=np.array([])
    for cam in data_wave_per_camera :
        flux1=resample_flux(data_wave_per_camera[cam],template_wave,template_flux) # this is slow
        flux2=Resolution(resolution_data_per_camera[cam]).dot(flux1) # this is slow
        norme=applySmoothingFilter(flux2) # this is fast
        flux3=flux2/(norme+(norme==0))
        output_flux = np.append(output_flux,flux3)
    return template_id,output_flux

def _func(arg) :
    return resample_template(**arg)


def redshift_fit(wave, flux, ivar, resolution_data, stdwave, stdflux, z_max=0.005, z_res=0.00005, template_error=0.):
    """ Redshift fit of a single template

    Args:
        wave : A dictionary of 1D array of vacuum wavelengths [Angstroms]. Example below.
        flux : A dictionary of 1D observed flux for the star
        ivar : A dictionary 1D inverse variance of flux
        resolution_data: resolution corresponding to the star's fiber
        stdwave : 1D standard star template wavelengths [Angstroms]
        stdflux : 1D[nwave] template flux        
        
    Returns:
        redshift : redshift of standard star
        

    Notes:
      - wave and stdwave can be on different grids that don't
        necessarily overlap
      - wave does not have to be uniform or monotonic.  Multiple cameras
        can be supported by concatenating their wave and flux arrays
    """
    cameras = list(flux.keys())
    log = get_logger()
    log.debug(time.asctime())

    # resampling on a log wavelength grid
    #####################################
    # need to go fast so we resample both data and model on a log grid

    # define grid
    minwave = 100000.
    maxwave = 0.
    for cam in cameras :
        minwave=min(minwave,np.min(wave[cam]))
        maxwave=max(maxwave,np.max(wave[cam]))
    # ala boss
    lstep=np.log10(1+z_res)
    margin=int(np.log10(1+z_max)/lstep)+1
    minlwave=np.log10(minwave)
    maxlwave=np.log10(maxwave) # desired, but readjusted
    nstep=(maxlwave-minlwave)/lstep
    
    resampled_lwave=minlwave+lstep*np.arange(nstep)
    resampled_wave=10**resampled_lwave

    # map data on grid
    resampled_data={}
    resampled_ivar={}
    resampled_model={}
    for cam in cameras :
        tmp_flux,tmp_ivar=resample_flux(resampled_wave,wave[cam],flux[cam],ivar[cam])
        resampled_data[cam]=tmp_flux

        
        resampled_ivar[cam]=tmp_ivar

        # we need to have the model on a larger grid than the data wave for redshifting
        dwave=wave[cam][-1]-wave[cam][-2]
        npix=int((wave[cam][-1]*z_max)/dwave+2)
        extended_cam_wave=np.append( wave[cam][0]+dwave*np.arange(-npix,0) ,  wave[cam])
        extended_cam_wave=np.append( extended_cam_wave, wave[cam][-1]+dwave*np.arange(1,npix+1))
        # ok now we also need to increase the resolution
        tmp_res=np.zeros((resolution_data[cam].shape[0],resolution_data[cam].shape[1]+2*npix))
        tmp_res[:,:npix] = np.tile(resolution_data[cam][:,0],(npix,1)).T
        tmp_res[:,npix:-npix] = resolution_data[cam]
        tmp_res[:,-npix:] = np.tile(resolution_data[cam][:,-1],(npix,1)).T
        # resampled model at camera resolution, with margin
        tmp=resample_flux(extended_cam_wave,stdwave,stdflux)
        tmp=Resolution(tmp_res).dot(tmp)
        # map on log lam grid
        resampled_model[cam]=resample_flux(resampled_wave,extended_cam_wave,tmp)

        # we now normalize both model and data
        tmp=applySmoothingFilter(resampled_data[cam])
        resampled_data[cam]/=(tmp+(tmp==0))
        resampled_ivar[cam]*=tmp**2
        
        if template_error>0 :
            ok=np.where(resampled_ivar[cam]>0)[0]
            if ok.size > 0 :
                resampled_ivar[cam][ok] = 1./ ( 1/resampled_ivar[cam][ok] + template_error**2 )
                
        tmp=applySmoothingFilter(resampled_model[cam])
        resampled_model[cam]/=(tmp+(tmp==0))
        resampled_ivar[cam]*=(tmp!=0)

    # fit the best redshift
    chi2=np.zeros((2*margin+1))
    ndata=np.zeros((2*margin+1))
    for i in range(-margin,margin+1) :
        for cam in cameras :
            ndata[i+margin] += np.sum(resampled_ivar[cam][margin:-margin]>0)
            if i<margin :
                chi2[i+margin] += np.sum(resampled_ivar[cam][margin:-margin]*(resampled_data[cam][margin:-margin]-resampled_model[cam][margin+i:-margin+i])**2)                
            else :
                chi2[i+margin] += np.sum(resampled_ivar[cam][margin:-margin]*(resampled_data[cam][margin:-margin]-resampled_model[cam][margin+i:])**2)
    
    i=np.argmin(chi2)-margin
    z=10**(i*lstep)-1
    log.debug("Best z=%f"%z)
    return z
   

def _compute_coef(w,coords) :
    """ Function used by interpolate_on_parameter_grid

    Args:
        w : 1D[npar] array of weights, one for each parameter axis
        coords : 2D[ntemplates,npar] array of coordinates of template nodes on cube for interpolation
        
    Returns:
        coefficients : 1D[ntemplates] linear coefficient of each template (sum=1)
        derivative   : 2D[npar,ntemplates] derivative of linear coefficient wrt w

    Notes :
        the linear coefficient of each template is a product of the form, for instance, w[0]*(1-w[1])*w[2]
    """
    ns=coords.shape[0]
    npar=coords.shape[1]
    coef=np.ones(ns)
    dcoefdw=np.ones((npar,ns))
    for s in range(ns) :
            coef[s]=1.
            for a in range(npar) :
                if coords[s,a]==0 :
                    coef[s] *= w[a]
                else :
                    coef[s] *= (1-w[a])
        
    for a in range(npar) :
        for s in range(ns) :
            dcoefdw[a,s]=1.
            for a2 in range(npar) :
                if(a2!=a) :
                    if coords[s,a2]==0 :
                        dcoefdw[a,s] *= w[a2]
                    else :
                        dcoefdw[a,s] *= (1-w[a2]) 
                else :
                    if coords[s,a]==1 :
                        dcoefdw[a,s] *= -1
    return coef,dcoefdw

def _compute_model(coef,templates) :
    """ Function used by interpolate_on_parameter_grid
        Computes model from coefficients (used several times in routine)
    Args:
        coef : 1D[ntemplates] array of coefficients
        templates : 2D[ntemplates,nwave] array of template spectra
        
    Returns:
        model : 1D[nwave]
    """
    model = np.zeros(templates[0].size)
    for c,m in zip(coef,templates) :
        model += c*m
    return model
    
def interpolate_on_parameter_grid(data_wave, data_flux, data_ivar, template_flux, teff, logg, feh, template_chi2) :
    """ 3D Interpolation routine among templates based on a grid of parameters teff, logg, feh.
        The tricky part is to define a cube on the parameter grid populated with templates, and it is not always possible.
        The routine never extrapolates, so that we stay in the range of input parameters.

    Args:
        data_wave : 1D[nwave] array of wavelength (concatenated list of input wavelength of different cameras and exposures)
        data_flux : 1D[nwave] array of normalized flux = (input flux)/median_filter(input flux) (concatenated list)
        data_ivar : 1D[nwave] array of inverse variance of normalized flux
        template_flux : 2D[ntemplates,nwave] array of normalized flux of templates (after resample, convolution and division by median_filter)
        teff : 1D[ntemplates]
        logg : 1D[ntemplates]
        feh  : 1D[ntemplates]
        template_chi2 : 1D[ntemplatess] array of precomputed chi2 = sum(data_ivar*(data_flux-template_flux)**2)
    
    Returns:
       coefficients : best fit coefficient of linear combination of templates (only 8 of them around the one with best template_chi2 
                      are potentially non null
       chi2 : chi2 of the linear combination
    """
    
    log = get_logger()
    log.debug("starting interpolation on grid")

    best_model_id = np.argmin(template_chi2)
    ndata=np.sum(data_ivar>0)
    
    log.debug("best model id=%d chi2/ndata=%f teff=%d logg=%2.1f feh=%2.1f"%(best_model_id,template_chi2[best_model_id]/ndata,teff[best_model_id],logg[best_model_id],feh[best_model_id]))
    
    ntemplates=template_flux.shape[0]
    
    # physical parameters define axes
    npar=3
    param=np.zeros((npar,ntemplates))
    param[0]=teff
    param[1]=logg
    param[2]=feh

    # grid nodes coordinates (unique values of the parameters)
    uparam=[]
    for a in range(npar) : 
        uparam.append(np.unique(param[a]))
    for a in range(npar) : 
        log.debug("param %d : %s"%(a,str(uparam[a])))
    
    # the parameters of the fit are npar interpolation coefficients w
    w=np.zeros(npar)
    eps=0. # could start with an offset

    # index on grid of the two best models along each axis of the grid, one of which being the best match
    desired_node_cube_coords=np.zeros((npar,2)).astype(int)    
    desired_node_grid_coords=np.zeros((npar,2)).astype(int)    
       
    for a in range(npar) : # a is an axis 
        
        # this is the coordinate on axis 'a' of the best node
        ibest=np.where(uparam[a]==param[a,best_model_id])[0][0]
        
        # selection of all grid nodes with templates on the same "line" of direction 'a'
        line_selection=np.ones(ntemplates).astype(bool)
        for a2 in range(npar) :
            if a2 != a : line_selection &= (param[a2]==param[a2,best_model_id])

        if np.sum(line_selection)<2 :
            log.warning("Cannot interpolate; the best fit is in a corner of the populated parameter grid")
            final_coefficients = np.zeros(ntemplates)
            final_coefficients[best_model_id] = 1.
            return final_coefficients,template_chi2[best_model_id]
            
        left_selection  = line_selection & (param[a]<param[a,best_model_id])
        right_selection = line_selection & (param[a]>param[a,best_model_id])

        # now we have to decide which direction to go from here to set one edge of the cube
        if np.sum(right_selection)==0 : # no choise
            desired_node_grid_coords[a,0]=np.where(uparam[a]==np.max(param[a][left_selection]))[0][0]
            desired_node_grid_coords[a,1]=ibest            
            w[a]=eps # model = w*left+(1-w)*right, here best in on the right (eps~0)
        elif np.sum(left_selection)==0 : # no choise
            desired_node_grid_coords[a,0]=ibest
            desired_node_grid_coords[a,1]=np.where(uparam[a]==np.min(param[a][right_selection]))[0][0]
            w[a]=(1-eps)  # model = w*left+(1-w)*right, here best in on the left
        else :
            # choice based on chi2
            im=np.where(uparam[a]==np.max(param[a][left_selection]))[0][0] # grid coord
            ip=np.where(uparam[a]==np.min(param[a][right_selection]))[0][0] # grid coord
            # need to get template id to read chi2
            jm=np.where(line_selection&(param[a]==uparam[a][im]))[0][0]
            jp=np.where(line_selection&(param[a]==uparam[a][ip]))[0][0]
            if template_chi2[jm]<template_chi2[jp] : # we go left
                desired_node_grid_coords[a,0]=im
                desired_node_grid_coords[a,1]=ibest
                w[a]=eps
                
            else : # we go right
                desired_node_grid_coords[a,0]=ibest
                desired_node_grid_coords[a,1]=ip
                w[a]=(1-eps)
        #log.debug("desired_node_grid_coords[%d]=%s"%(a,str(desired_node_grid_coords[a])))

    # interpolation scheme based on a cube where the best node is one corner
    # we have 2**npar nodes which are the corners of a cube
    ns=8
    node_template_ids=-1*np.ones(ns).astype(int) 
    node_cube_coords=np.zeros((ns,3)).astype(int) # 0 or 1
    node_grid_coords=np.zeros((ns,3)).astype(int)
    
    s=0

    
    # distance weighting
    dweights=np.zeros(npar)
    for a in range(npar) :
       dweights[a] = 1. / np.std(param[a])**2  

    for k0 in range(2) :        
        for k1 in range(2) :             
            for k2 in range(2) : 
                node_cube_coords[s,0]=k0
                node_cube_coords[s,1]=k1
                node_cube_coords[s,2]=k2
                i0=desired_node_grid_coords[0,k0]
                i1=desired_node_grid_coords[1,k1]
                i2=desired_node_grid_coords[2,k2]
                p0=uparam[0][i0]
                p1=uparam[1][i1]
                p2=uparam[2][i2]
                #log.debug("star %d [%d,%d,%d];[%d,%d,%d];[%f,%f,%f]"%(s,k0,k1,k2,i0,i1,i2,p0,p1,p2))

                
                
                # the grid is rectangular, but not fully filled, so we cant always find exact match
                dist=(dweights[0]*(param[0]-p0)**2+dweights[1]*(param[1]-p1)**2+dweights[2]*(param[2]-p2)**2)
                ii=np.argsort(dist)
                for i in ii :
                    if i in node_template_ids[:s] : # already there
                        continue
                    node_template_ids[s]=i
                    break
                if node_template_ids[s]==-1 :
                    log.error("didn't find a node for this axis")
                    sys.exit(12)                
                
                # update indices
                log.debug("node %d [%d,%d,%d];[%d,%d,%d];[%f,%f,%f] id=%d"%(s,k0,k1,k2,i0,i1,i2,p0,p1,p2,node_template_ids[s]))
                s+=1

    # we are done with the indexing and choice of template nodes
    # node_flux
    node_template_flux = template_flux[node_template_ids]
    
    # compute all weighted scalar products among templates
    HA=np.zeros((ns,ns))
    for s in range(ns) :
        for s2 in range(ns) :
            if HA[s2,s] != 0 :
                HA[s,s2] = HA[s2,s]
            else :
                HA[s,s2] = np.sum(data_ivar*node_template_flux[s]*node_template_flux[s2])            

    # initial state
    coef , dcoefdw = _compute_coef(w,node_cube_coords)
    log.debug("init coef=%s"%coef)
    model=_compute_model(coef,node_template_flux)
    chi2=np.sum(data_ivar*(data_flux-model)**2)
    log.debug("init w=%s chi2/ndata=%f"%(w,chi2/ndata))
    
    # now we have to do the fit
    # fitting one axis at a time (simultaneous fit of 3 axes was tested and found inefficient : rapidly stuck on edges)
    # it has to be iterative because the model is a non-linear combination of parameters w, ex: w[0]*(1-w[1])*(1-w[2])
    for loop in range(50) :
        
        previous_chi2=chi2.copy()
        previous_w=w.copy()

        for a in range(npar) :
            previous_chi2_a=chi2.copy()
            
            # updated coef and derivative (non linear)
            coef , dcoefdw = _compute_coef(w,node_cube_coords)
            # checking
            # must be = 1
            #log.debug("sum(coef)=%f"%np.sum(coef))
            # must be = 0
            #log.debug("sum(dcoefdw)=%s"%np.sum(dcoefdw,axis=1))
            
            # current model
            model=_compute_model(coef,node_template_flux)
            # residuals
            res=data_flux-model
            
            A=0.
            B=0.
            for s in range(ns) :
                for s2 in range(ns) :
                    A += dcoefdw[a,s]*dcoefdw[a,s2]*HA[s,s2]
                B += dcoefdw[a,s]*np.sum(data_ivar*res*node_template_flux[s])
            dw=B/A
            # dw is the best fit increment of w 
            #log.debug("dw=%f"%dw)

            # test if this improves chi2
            tmp_w=w.copy()
            tmp_w[a]+=dw
            # apply bounds
            if tmp_w[a]>1 :
                tmp_w[a]=1
            elif tmp_w[a]<0 :
                tmp_w[a]=0
            
            tmp_coef , junk = _compute_coef(tmp_w,node_cube_coords)
            tmp_model=_compute_model(tmp_coef,node_template_flux)
            tmp_res=data_flux-tmp_model
            tmp_chi2=np.sum(data_ivar*tmp_res**2)
            
            #log.debug("loop #%d a=%d w=%s chi2/ndata=%f"%(loop,a,w,chi2/data_flux.size))
            if tmp_chi2<previous_chi2_a :
                w=tmp_w
                chi2=tmp_chi2
            elif tmp_chi2>previous_chi2_a+1e-12 :
                log.warning("the fitted dw gave a larger chi2 ??? %f"%(tmp_chi2-previous_chi2_a))
            
        log.debug("loop #%d w=%s chi2/ndata=%f (-dchi2_loop=%f -dchi2_tot=%f)"%(loop,w,chi2/ndata,previous_chi2-chi2,template_chi2[best_model_id]-chi2))
        diff=np.max(np.abs(w-previous_w))
        if diff < 0.001 :
            break

    '''
    # useful debugging plot
    import matplotlib.pyplot as plt
    plt.figure()        
    ok=np.where(data_ivar>0)[0]
    plt.errorbar(data_wave[ok],data_flux[ok],1./np.sqrt(data_ivar[ok]),fmt="o")
    ii=np.argsort(data_wave)
    plt.plot(data_wave[ii],model[ii],"-",c="r")
    plt.show()
    '''
    
    input_number_of_templates=template_flux.shape[0]
    final_coefficients=np.zeros(input_number_of_templates)
    final_coefficients[node_template_ids]=coef
    
    return final_coefficients,chi2
        

def match_templates(wave, flux, ivar, resolution_data, stdwave, stdflux, teff, logg, feh, ncpu=1, z_max=0.005, z_res=0.00002, template_error=0):
    """For each input spectrum, identify which standard star template is the closest
    match, factoring out broadband throughput/calibration differences.

    Args:
        wave : A dictionary of 1D array of vacuum wavelengths [Angstroms]. Example below.
        flux : A dictionary of 1D observed flux for the star
        ivar : A dictionary 1D inverse variance of flux
        resolution_data: resolution corresponding to the star's fiber
        stdwave : 1D standard star template wavelengths [Angstroms]
        stdflux : 2D[nstd, nwave] template flux
        teff : 1D[nstd] effective model temperature
        logg : 1D[nstd] model surface gravity
        feh : 1D[nstd] model metallicity
        ncpu : number of cpu for multiprocessing

    Returns:
        coef : numpy.array of linear coefficient of standard stars        
        redshift : redshift of standard star
        chipdf : reduced chi2

    Notes:
      - wave and stdwave can be on different grids that don't
        necessarily overlap
      - wave does not have to be uniform or monotonic.  Multiple cameras
        can be supported by concatenating their wave and flux arrays
    """
    # I am treating the input arguments from three frame files as dictionary. For example
    # wave{"r":rwave,"b":bwave,"z":zwave}
    # Each data(3 channels) is compared to every model.
    # flux should be already flat fielded and sky subtracted.



    cameras = list(flux.keys())
    log = get_logger()
    log.debug(time.asctime())

    
    # normalize data and store them in single array
    data_wave=np.array([])
    data_flux=np.array([])
    data_ivar=np.array([])
    for cam in cameras :
        data_wave=np.append(data_wave,wave[cam])
        tmp=applySmoothingFilter(flux[cam]) # this is fast
        data_flux=np.append(data_flux,flux[cam]/(tmp+(tmp==0)))
        data_ivar=np.append(data_ivar,ivar[cam]*tmp**2)
    
    # mask potential cosmics
    ok=np.where(data_ivar>0)[0]
    if ok.size>0 :
        data_ivar[ok] *= (data_flux[ok]<1.+3/np.sqrt(data_ivar[ok]))

    # add error propto to flux to account for model error
    if template_error>0  :
        ok=np.where(data_ivar>0)[0]
        if ok.size>0 :
            data_ivar[ok] = 1./ ( 1./data_ivar[ok] + template_error**2 )

    # mask sky lines
    # in vacuum
    # mask blue lines that can affect fit of Balmer series
    # line at 5577 has a stellar line close to it !
    # line at 7853. has a stellar line close to it !
    # line at 9790.5 has a stellar line close to it !
    # all of this is based on analysis of a few exposures of BOSS data
    skylines=np.array([4358,5461,5577,5889.5,5895.5,6300,7821.5,7853.,7913.,9790.5])    
    hw=4. # A
    for line in skylines :
        data_ivar[(data_wave>=(line-hw))&(data_wave<=(line+hw))]=0.

    ndata = np.sum(data_ivar>0)
    

    # start looking at models
    
    # find canonical f-type model: Teff=6000, logg=4, Fe/H=-1.5
    canonical_model=np.argmin((teff-6000.0)**2+(logg-4.0)**2+(feh+1.5)**2)
    
    # fit redshift on canonical model
    # we use the original data to do this
    # because we resample both the data and model on a logarithmic grid in the routine
    z = redshift_fit(wave, flux, ivar, resolution_data, stdwave, stdflux[canonical_model], z_max, z_res)
    
        
    # now we go back to the model spectra , redshift them, resample, apply resolution, normalize and chi2 match
    
    ntemplates=stdflux.shape[0]

    # here we take into account the redshift once and for all
    shifted_stdwave=stdwave/(1+z)
    
    func_args = []
    # need to parallelize the model resampling
    for template_id in range(ntemplates) :
        arguments={"data_wave_per_camera":wave,
                   "resolution_data_per_camera":resolution_data,
                   "template_wave":shifted_stdwave,
                   "template_flux":stdflux[template_id],
                   "template_id":template_id}
        func_args.append( arguments )
    
    
    if ncpu > 1:
        log.debug("creating multiprocessing pool with %d cpus"%ncpu); sys.stdout.flush()
        pool = multiprocessing.Pool(ncpu)
        log.debug("Running pool.map() for {} items".format(len(func_args))); sys.stdout.flush()
        results  =  pool.map(_func, func_args)
        log.debug("Finished pool.map()"); sys.stdout.flush()
        pool.close()
        pool.join()
        log.debug("Finished pool.join()"); sys.stdout.flush()
    else:
        log.debug("Not using multiprocessing for {} cpus".format(ncpu))
        results = [_func(x) for x in func_args]
        log.debug("Finished serial loop over compute_chi2")

    # collect results
    # in case the exit of the multiprocessing pool is not ordered as the input
    # we returned the template_id
    template_flux=np.zeros((ntemplates,data_flux.size))
    for result in results :
        template_id = result[0]
        template_flux[template_id] = result[1]

    # compute model chi2
    template_chi2=np.zeros(ntemplates)
    for template_id in range(ntemplates) :
        template_chi2[template_id] = np.sum(data_ivar*(data_flux-template_flux[template_id])**2)
    
    best_model_id=np.argmin(template_chi2) 
    best_chi2=template_chi2[best_model_id]
    log.debug("selected best model {} chi2/ndf {}".format(best_model_id, best_chi2/ndata))
    
    # interpolate around best model using parameter grid
    coef,chi2 = interpolate_on_parameter_grid(data_wave, data_flux, data_ivar, template_flux, teff, logg, feh, template_chi2)
    log.debug("after interpolation chi2/ndf {}".format(chi2/ndata))
    
    return coef,z,chi2/ndata


def normalize_templates(stdwave, stdflux, mags, filters):
    """Returns spectra normalized to input magnitudes.

    Args:
        stdwave : 1D array of standard star wavelengths [Angstroms]
        stdflux : 1D observed flux
        mags : 1D array of observed AB magnitudes
        filters : list of filter names for mags, e.g. ['SDSS_r', 'DECAM_g', ...]

    Returns:
        stdwave : same as input
        normflux : normalized flux array

    Only SDSS_r band is assumed to be used for normalization for now.
    """
    log = get_logger()

    nstdwave=stdwave.size
    normflux=np.array(nstdwave)

    fluxunits = 1e-17 * units.erg / units.s / units.cm**2 / units.Angstrom

    for i,v in enumerate(filters):
        #Normalizing using only SDSS_R band magnitude
        if v.upper() == 'SDSS_R' or v.upper() =='DECAM_R' or v.upper()=='DECAM_G' :
            #-TODO: Add more filters for calibration. Which one should be used if multiple mag available?
            refmag=mags[i]
            filter_response=load_filter(v)
            apMag=filter_response.get_ab_magnitude(stdflux*fluxunits,stdwave)
            log.info('scaling {} mag {:f} to {:f}.'.format(v, apMag,refmag))
            scalefac=10**((apMag-refmag)/2.5)
            normflux=stdflux*scalefac

            break  #- found SDSS_R or DECAM_R; we can stop now
        count=0
        for k,f in enumerate(['SDSS_R','DECAM_R','DECAM_G']):
            ii,=np.where((np.asarray(filters)==f))
            count=count+ii.shape[0]
        if (count==0):
            log.error("No magnitude given for SDSS_R, DECAM_R or DECAM_G filters")
            sys.exit(0)
    return normflux

def compute_flux_calibration(frame, input_model_wave,input_model_flux,input_model_fibers, nsig_clipping=4.,debug=False):
    """Compute average frame throughput based on data frame.(wave,flux,ivar,resolution_data)
    and spectro-photometrically calibrated stellar models (model_wave,model_flux).
    Wave and model_wave are not necessarily on the same grid

    Args:
      frame : Frame object with attributes wave, flux, ivar, resolution_data
      input_model_wave : 1D[nwave] array of model wavelengths
      input_model_flux : 2D[nstd, nwave] array of model fluxes
      input_model_fibers : 1D[nstd] array of model fibers
      nsig_clipping : (optional) sigma clipping level

    Returns:
         desispec.FluxCalib object
         calibration: mean calibration (without resolution)

    Notes:
      - we first resample the model on the input flux wave grid
      - then convolve it to the data resolution (the input wave grid is supposed finer than the spectral resolution)
      - then iteratively
        - fit the mean throughput (deconvolved, this is needed because of sharp atmospheric absorption lines)
        - compute broad band correction to fibers (to correct for small mis-alignement for instance)
        - perform outlier rejection

     There is one subtelty with the relation between calibration and resolution.
      - The input frame flux is on average flux^frame_fiber = R_fiber*C*flux^true where C is the true calibration (or throughput)
        which is a function of wavelength. This is the system we solve.
      - But we want to return a calibration vector per fiber C_fiber defined by flux^cframe_fiber = flux^frame_fiber/C_fiber,
        such that flux^cframe can be compared with a convolved model of the truth, flux^cframe_fiber = R_fiber*flux^true,
        i.e. (R_fiber*C*flux^true)/C_fiber = R_fiber*true_flux, giving C_fiber = (R_fiber*C*flux^true)/(R_fiber*flux^true)
      - There is no solution for this for all possible input specta. The solution for a flat spectrum is returned,
        which is very close to C_fiber = R_fiber*C (but not exactly).

    """

    log=get_logger()
    log.info("starting")

    #- Pull out just the standard stars for convenience, but keep the
    #- full frame of spectra around because we will later need to convolved
    #- the calibration vector for each fiber individually
    stdfibers = np.intersect1d( np.where(frame.fibermap['OBJTYPE'] == 'STD')[0] , input_model_fibers)
    stdstars = frame[stdfibers]

    nwave=stdstars.nwave
    nstds=stdstars.flux.shape[0]

    # resample model to data grid and convolve by resolution
    model_flux=np.zeros((nstds, nwave))
    convolved_model_flux=np.zeros((nstds, nwave))
    for fiber in range(model_flux.shape[0]) :
        model_flux[fiber]=resample_flux(stdstars.wave,input_model_wave,input_model_flux[fiber])
        convolved_model_flux[fiber]=stdstars.R[fiber].dot(model_flux[fiber])

    # iterative fitting and clipping to get precise mean spectrum
    current_ivar=stdstars.ivar.copy()

    #- Start with a first pass median rejection
    calib = (convolved_model_flux!=0)*(stdstars.flux/(convolved_model_flux + (convolved_model_flux==0)))
    median_calib = np.median(calib, axis=0)
    chi2 = stdstars.ivar * (stdstars.flux - convolved_model_flux*median_calib)**2
    bad=(chi2>nsig_clipping**2)
    current_ivar[bad] = 0

    smooth_fiber_correction=np.ones((stdstars.flux.shape))
    chi2=np.zeros((stdstars.flux.shape))

    # chi2 = sum w ( data_flux - R*(calib*model_flux))**2
    # chi2 = sum (sqrtw*data_flux -diag(sqrtw)*R*diag(model_flux)*calib)

    sqrtw=np.sqrt(current_ivar)
    #sqrtwmodel=np.sqrt(current_ivar)*convolved_model_flux # used only for QA
    sqrtwflux=np.sqrt(current_ivar)*stdstars.flux

    # diagonal sparse matrices
    D1=scipy.sparse.lil_matrix((nwave,nwave))
    D2=scipy.sparse.lil_matrix((nwave,nwave))

    # test
    # nstds=20
    nout_tot=0
    for iteration in range(20) :

        # fit mean calibration
        A=scipy.sparse.lil_matrix((nwave,nwave)).tocsr()
        B=np.zeros((nwave))

        # loop on fiber to handle resolution
        for fiber in range(nstds) :
            if fiber%10==0 :
                log.info("iter %d fiber %d"%(iteration,fiber))

            R = stdstars.R[fiber]

            # diagonal sparse matrix with content = sqrt(ivar)*flat
            D1.setdiag(sqrtw[fiber]*smooth_fiber_correction[fiber])
            D2.setdiag(model_flux[fiber])
            sqrtwmodelR = D1.dot(R.dot(D2)) # chi2 = sum (sqrtw*data_flux -diag(sqrtw)*smooth_fiber_correction*R*diag(model_flux)*calib )

            A = A+(sqrtwmodelR.T*sqrtwmodelR).tocsr()
            B += sqrtwmodelR.T*sqrtwflux[fiber]

        #- Add a weak prior that calibration = median_calib
        #- to keep A well conditioned
        minivar = np.min(current_ivar[current_ivar>0])
        log.debug('min(ivar[ivar>0]) = {}'.format(minivar))
        epsilon = minivar/10000
        A = epsilon*np.eye(nwave) + A   #- converts sparse A -> dense A
        B += median_calib*epsilon

        log.info("iter %d solving"%iteration)
        ### log.debug('cond(A) {:g}'.format(np.linalg.cond(A)))
        #calibration=cholesky_solve(A, B)
        w = np.diagonal(A)>0
        A_pos_def = A[w,:]
        A_pos_def = A_pos_def[:,w]
        calibration = B*0
        try:
            calibration[w]=cholesky_solve(A_pos_def, B[w])
        except np.linalg.linalg.LinAlgError:
            log.info('cholesky fails in iteration {}, trying svd'.format(iteration))
            calibration[w] = np.linalg.lstsq(A_pos_def,B[w])[0]

        log.info("iter %d fit smooth correction per fiber"%iteration)
        # fit smooth fiberflat and compute chi2
        for fiber in range(nstds) :
            if fiber%10==0 :
                log.info("iter %d fiber %d(smooth)"%(iteration,fiber))

            M = stdstars.R[fiber].dot(calibration*model_flux[fiber])
            
            try:
                pol=np.poly1d(np.polyfit(stdstars.wave,stdstars.flux[fiber]/(M+(M==0)),deg=1,w=current_ivar[fiber]*M**2))
            except:
                current_ivar[fiber]=0.
            smooth_fiber_correction[fiber]=pol(stdstars.wave)
            chi2[fiber]=current_ivar[fiber]*(stdstars.flux[fiber]-smooth_fiber_correction[fiber]*M)**2

        log.info("iter {0:d} rejecting".format(iteration))

        nout_iter=0
        if iteration<1 :
            # only remove worst outlier per wave
            # apply rejection iteratively, only one entry per wave among fibers
            # find waves with outlier (fastest way)
            nout_per_wave=np.sum(chi2>nsig_clipping**2,axis=0)
            selection=np.where(nout_per_wave>0)[0]
            for i in selection :
                worst_entry=np.argmax(chi2[:,i])
                current_ivar[worst_entry,i]=0
                sqrtw[worst_entry,i]=0
                #sqrtwmodel[worst_entry,i]=0
                sqrtwflux[worst_entry,i]=0
                nout_iter += 1

        else :
            # remove all of them at once
            bad=(chi2>nsig_clipping**2)
            current_ivar *= (bad==0)
            sqrtw *= (bad==0)
            #sqrtwmodel *= (bad==0)
            sqrtwflux *= (bad==0)
            nout_iter += np.sum(bad)

        nout_tot += nout_iter

        sum_chi2=float(np.sum(chi2))
        ndf=int(np.sum(chi2>0)-nwave-nstds*2)
        chi2pdf=0.
        if ndf>0 :
            chi2pdf=sum_chi2/ndf

        # normalize to get a mean fiberflat=1
        mean=np.nanmean(smooth_fiber_correction,axis=0)
        smooth_fiber_correction /= mean

        log.info("iter #%d chi2=%f ndf=%d chi2pdf=%f nout=%d mean=%f"%(iteration,sum_chi2,ndf,chi2pdf,nout_iter,np.mean(mean)))

        if nout_iter == 0 and np.max(np.abs(mean-1))<0.005 :
            break

    # smooth_fiber_correction does not converge exactly to one on average, so we apply its mean to the calibration
    # (tested on sims)
    calibration /= mean

    log.info("nout tot=%d"%nout_tot)

    # solve once again to get deconvolved variance
    #calibration,calibcovar=cholesky_solve_and_invert(A.todense(),B)
    calibcovar=np.linalg.inv(A)
    calibvar=np.diagonal(calibcovar)
    log.info("mean(var)={0:f}".format(np.mean(calibvar)))

    calibvar=np.array(np.diagonal(calibcovar))
    # apply the mean (as in the iterative loop)
    calibvar *= mean**2
    calibivar=(calibvar>0)/(calibvar+(calibvar==0))

    # we also want to save the convolved calibration and a calibration variance
    # first compute average resolution
    mean_res_data=np.mean(frame.resolution_data,axis=0)
    R = Resolution(mean_res_data)
    # compute convolved calib
    ccalibration = np.zeros(frame.flux.shape)
    for i in range(frame.nspec):
        norme = frame.R[i].dot(np.ones(calibration.shape))
        ok=np.where(norme>0)[0]
        if ok.size :
            ccalibration[i][ok]=frame.R[i].dot(calibration)[ok]/norme[ok]
        
    # Use diagonal of mean calibration covariance for output.
    ccalibcovar=R.dot(calibcovar).dot(R.T.todense())
    ccalibvar=np.array(np.diagonal(ccalibcovar))

    # apply the mean (as in the iterative loop)
    ccalibvar *= mean**2
    ccalibivar=(ccalibvar>0)/(ccalibvar+(ccalibvar==0))

    # convert to 2D
    # For now this is the same for all fibers; in the future it may not be
    ccalibivar = np.tile(ccalibivar, frame.nspec).reshape(frame.nspec, frame.nwave)

    # need to do better here
    mask = (ccalibivar==0).astype(np.int32)

    # return calibration, calibivar, mask, ccalibration, ccalibivar
    return FluxCalib(stdstars.wave, ccalibration, ccalibivar, mask, R.dot(calibration))



class FluxCalib(object):
    def __init__(self, wave, calib, ivar, mask, meancalib=None):
        """Lightweight wrapper object for flux calibration vectors

        Args:
            wave : 1D[nwave] input wavelength (Angstroms)
            calib: 2D[nspec, nwave] calibration vectors for each spectrum
            ivar : 2D[nspec, nwave] inverse variance of calib
            mask : 2D[nspec, nwave] mask of calib (0=good)
            meancalib : 1D[nwave] mean convolved calibration (optional)

        All arguments become attributes, plus nspec,nwave = calib.shape

        The calib vector should be such that

            [1e-17 erg/s/cm^2/A] = [photons/A] / calib
        """
        assert wave.ndim == 1
        assert calib.ndim == 2
        assert calib.shape == ivar.shape
        assert calib.shape == mask.shape
        assert np.all(ivar >= 0)

        self.nspec, self.nwave = calib.shape
        self.wave = wave
        self.calib = calib
        self.ivar = ivar
        self.mask = util.mask32(mask)
        self.meancalib = meancalib

def apply_flux_calibration(frame, fluxcalib):
    """
    Applies flux calibration to input flux and ivar

    Args:
        frame: Spectra object with attributes wave, flux, ivar, resolution_data
        fluxcalib : FluxCalib object with wave, calib, ...

    Modifies frame.flux and frame.ivar
    """
    log=get_logger()
    log.info("starting")

    # check same wavelength, die if not the case
    mval=np.max(np.abs(frame.wave-fluxcalib.wave))
    #if mval > 0.00001 :
    if mval > 0.001 :
        log.error("not same wavelength (should raise an error instead)")
        sys.exit(12)

    nwave=frame.nwave
    nfibers=frame.nspec

    """
    F'=F/C
    Var(F') = Var(F)/C**2 + F**2*(  d(1/C)/dC )**2*Var(C)
    = 1/(ivar(F)*C**2) + F**2*(1/C**2)**2*Var(C)
    = 1/(ivar(F)*C**2) + F**2*Var(C)/C**4
    = 1/(ivar(F)*C**2) + F**2/(ivar(C)*C**4)
    """
    # for fiber in range(nfibers) :
    #     C = fluxcalib.calib[fiber]
    #     flux[fiber]=frame.flux[fiber]*(C>0)/(C+(C==0))
    #     ivar[fiber]=(ivar[fiber]>0)*(civar[fiber]>0)*(C>0)/(   1./((ivar[fiber]+(ivar[fiber]==0))*(C**2+(C==0))) + flux[fiber]**2/(civar[fiber]*C**4+(civar[fiber]*(C==0)))   )

    C = fluxcalib.calib
    frame.flux = frame.flux * (C>0) / (C+(C==0))
    frame.ivar *= (fluxcalib.ivar>0) * (C>0)
    for i in range(nfibers) :
        ok=np.where(frame.ivar[i]>0)[0]        
        if ok.size>0 :
            frame.ivar[i,ok] = 1./( 1./(frame.ivar[i,ok]*C[i,ok]**2)+frame.flux[i,ok]**2/(fluxcalib.ivar[i,ok]*C[i,ok]**4)  )


def ZP_from_calib(wave, calib):
    """ Calculate the ZP in AB magnitudes given the calibration and the wavelength arrays
    Args:
        wave:  1D array (A)
        calib:  1D array (converts erg/s/A to photons/s/A)

    Returns:
      ZP_AB: 1D array of ZP values in AB magnitudes

    """
    ZP_flambda = 1e-17 / calib  # erg/s/cm^2/A
    ZP_fnu = ZP_flambda * wave**2 / (2.9979e18)  # c in A/s
    # Avoid 0 values
    ZP_AB = np.zeros_like(ZP_fnu)
    gdZ = ZP_fnu > 0.
    ZP_AB[gdZ] = -2.5 * np.log10(ZP_fnu[gdZ]) - 48.6
    # Return
    return ZP_AB


def qa_fluxcalib(param, frame, fluxcalib):
    """
    Args:
        param: dict of QA parameters
        frame: Frame
        fluxcalib: FluxCalib
        model_tuple : tuple of model data for standard stars (read from stdstars-...fits)

    Returns:
        qadict: dict of QA outputs
          Need to record simple Python objects for yaml (str, float, int)

    """
    log = get_logger()
    qadict = {}

    # Unpack model

    # Standard stars
    stdfibers = np.where((frame.fibermap['OBJTYPE'] == 'STD'))[0]
    stdstars = frame[stdfibers]
    nstds = len(stdfibers)
    #try:
    #    assert np.array_equal(frame.fibers[stdfibers], input_model_fibers)
    #except AssertionError:
    #    log.error("Bad indexing in standard stars")

    # Calculate ZP for mean spectrum
    #medcalib = np.median(fluxcalib.calib,axis=0)
    medcalib = np.median(fluxcalib.calib[stdfibers],axis=0)
    ZP_AB = ZP_from_calib(fluxcalib.wave, medcalib)  # erg/s/cm^2/A

    # ZP at fiducial wavelength (AB mag for 1 photon/s/A)
    iZP = np.argmin(np.abs(fluxcalib.wave-param['ZP_WAVE']))
    qadict['ZP'] = float(np.median(ZP_AB[iZP-10:iZP+10]))

    # Unpack star data
    #sqrtwmodel, sqrtwflux, current_ivar, chi2 = indiv_stars

    # RMS
    qadict['NSTARS_FIBER'] = int(nstds)
    ZP_fiducial = np.zeros(nstds)

    for ii in range(nstds):
        # Good pixels
        gdp = stdstars.ivar[ii, :] > 0.
        icalib = fluxcalib.calib[stdfibers[ii]][gdp]
        i_wave = fluxcalib.wave[gdp]
        # ZP
        ZP_stars = ZP_from_calib(i_wave, icalib)
        iZP = np.argmin(np.abs(i_wave-param['ZP_WAVE']))
        ZP_fiducial[ii] = float(np.median(ZP_stars[iZP-10:iZP+10]))
    #import pdb; pdb.set_trace()
    qadict['RMS_ZP'] = float(np.std(ZP_fiducial))

    # MAX ZP Offset
    #stdfibers = np.where(frame.fibermap['OBJTYPE'] == 'STD')[0]
    ZPoffset = ZP_fiducial-qadict['ZP']
    imax = np.argmax(np.abs(ZPoffset))
    qadict['MAX_ZP_OFF'] = [float(ZPoffset[imax]),
                            int(stdfibers[np.argmax(ZPoffset)])]
    if qadict['MAX_ZP_OFF'][0] > param['MAX_ZP_OFF']:
        log.warning("Bad standard star ZP {:g}, in fiber {:d}".format(
                qadict['MAX_ZP_OFF'][0], qadict['MAX_ZP_OFF'][1]))
    # Return
    return qadict
