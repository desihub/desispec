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
from desiutil.log import get_logger
from .io.filters import load_filter
from desispec import util
import scipy, scipy.sparse, scipy.ndimage
import sys
import time
from astropy import units
import multiprocessing
from pkg_resources import resource_exists, resource_filename

def isStdStar(desi_target, bright=None):
    """
    Determines if target(s) are standard stars

    Args:
        desi_target: int or array of DESI_TARGET targeting bit mask(s)

    Optional:
        bright: if True, only bright time standards; if False, only darktime, otherwise both

    Returns bool or array of bool

    TODO: move out of scripts/stdstars.py
    """
    from desitarget.targetmask import desi_mask
    yes = (desi_target & desi_mask.STD_WD) != 0
    if bright is None:
        yes |= (desi_target & desi_mask.mask('STD_WD|STD_FAINT|STD_BRIGHT')) != 0
    elif bright:
        yes |= (desi_target & desi_mask.mask('STD_WD|STD_BRIGHT')) != 0
    else:
        yes |= (desi_target & desi_mask.mask('STD_WD|STD_FAINT')) != 0

    return yes

def applySmoothingFilter(flux,width=200) :
    """ Return a smoothed version of the input flux array using a median filter

    Args:
        flux  : 1D array of flux 
        width : size of the median filter box
            
    Returns:
        smooth_flux : median filtered flux of same size as input
    """

    # it was checked that the width of the median_filter has little impact on best fit stars
    # smoothing the ouput (with a spline for instance) does not improve the fit
    return scipy.ndimage.filters.median_filter(flux,width,mode='constant')
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
    """Resample a spectral template on the data wavelength grid. Then convolve the spectra by the resolution
    for each camera. Also returns the result of applySmoothingFilter. This routine is used internally in
    a call to multiprocessing.Pool. 
    
    Args:
        data_wave_per_camera : A dictionary of 1D array of vacuum wavelengths [Angstroms], one entry per camera and exposure.
        resolution_data_per_camera :  A dictionary of resolution corresponding for the fiber, one entry per camera and exposure.
        template_wave : 1D array, input spectral template wavelength [Angstroms] (arbitrary spacing).
        template_flux : 1D array, input spectral template flux density.
        template_id   : int, template identification index, used to ensure matching of input/output after a multiprocessing run.
    
    Returns:
        template_id   : int, template identification index, same as input.
        output_wave   : A dictionary of 1D array of vacuum wavelengths
        output_flux   : A dictionary of 1D array of output template flux
        output_norm   : A dictionary of 1D array of output template smoothed flux
    """
    output_wave=np.array([])
    output_flux=np.array([])
    output_norm=np.array([])
    sorted_keys = list(data_wave_per_camera.keys())
    sorted_keys.sort() # force sorting the keys to agree with data (found unpredictable ordering in tests)
    for cam in sorted_keys :
        flux1=resample_flux(data_wave_per_camera[cam],template_wave,template_flux) # this is slow
        flux2=Resolution(resolution_data_per_camera[cam]).dot(flux1) # this is slow
        norme=applySmoothingFilter(flux2) # this is fast
        flux3=flux2/(norme+(norme==0))
        output_flux = np.append(output_flux,flux3)
        output_norm = np.append(output_norm,norme)
        output_wave = np.append(output_wave,data_wave_per_camera[cam]) # need to add wave to avoid wave/flux matching errors
    return template_id,output_wave,output_flux,output_norm


def _func(arg) :
    """ Used for multiprocessing.Pool """
    return resample_template(**arg)

def _smooth_template(template_id,camera_index,template_flux) :
    """ Used for multiprocessing.Pool """
    norme = applySmoothingFilter(template_flux)
    return template_id,camera_index,norme

def _func2(arg) :
    """ Used for multiprocessing.Pool """
    return _smooth_template(**arg)

def redshift_fit(wave, flux, ivar, resolution_data, stdwave, stdflux, z_max=0.005, z_res=0.00005, template_error=0.):
    """ Redshift fit of a single template

    Args:
        wave : A dictionary of 1D array of vacuum wavelengths [Angstroms]. Example below.
        flux : A dictionary of 1D observed flux for the star
        ivar : A dictionary 1D inverse variance of flux
        resolution_data: resolution corresponding to the star's fiber
        stdwave : 1D standard star template wavelengths [Angstroms]
        stdflux : 1D[nwave] template flux        
        z_max : float, maximum blueshift and redshift in scan, has to be positive
        z_res : float, step of of redshift scan between [-z_max,+z_max]
        template_error : float, assumed template flux relative error

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
    import matplotlib.pyplot as plt
    
    i=np.argmin(chi2)-margin
    z=10**(-i*lstep)-1
    log.debug("Best z=%f"%z)
    '''
    log.debug("i=%d"%i)
    log.debug("lstep=%f"%lstep)
    log.debug("margin=%d"%margin)
    plt.figure()
    #plt.plot(chi2)
    for cam in cameras :
        ok=np.where(resampled_ivar[cam]>0)[0]
        #plt.plot(resampled_wave[ok],resampled_data[cam][ok],"o",c="gray")
        plt.errorbar(resampled_wave[ok],resampled_data[cam][ok],1./np.sqrt(resampled_ivar[cam][ok]),fmt="o",color="gray")        
        plt.plot(resampled_wave[margin:-margin],resampled_model[cam][margin+i:-margin+i],"-",c="r")
    plt.show()
    '''
    return z
   

def _compute_coef(coord,node_coords) :
    """ Function used by interpolate_on_parameter_grid2

    Args:
        coord : 1D array of coordinates of size n_axis
        node_coords : 2D array of coordinates of nodes, shape = (n_nodes,n_axis)

    Returns:
        coef : 1D array of linear coefficients for each node, size = n_nodes   
    """
    
    n_nodes=node_coords.shape[0]
    npar=node_coords.shape[1]
    coef=np.ones(n_nodes)
    for s in range(n_nodes) :
        coef[s]=1.
        for a in range(npar) :
            dist=np.abs(node_coords[s,a]-coord[a]) # distance between model point and node along axis a
            
            # piece-wise linear version
            if dist>1 :
                coef[s]=0.
                break
            coef[s] *= (1.-dist)
            
            # we could alternatively have used b-spline of higher order
    
    norme=np.sum(coef)
    if norme<=0 : # we are outside of valid grid
        return np.zeros(coef.shape) # will be detected in fitter
    coef /= norme
    return coef

  
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
        coefficients : best fit coefficient of linear combination of templates
        chi2 : chi2 of the linear combination
    """
    
    log = get_logger()
    log.debug("starting interpolation on grid")

    best_model_id = np.argmin(template_chi2)
    ndata=np.sum(data_ivar>0)
    
    log.debug("best model id=%d chi2/ndata=%f teff=%d logg=%2.1f feh=%2.1f"%(best_model_id,template_chi2[best_model_id]/ndata,teff[best_model_id],logg[best_model_id],feh[best_model_id]))
    
    ntemplates=template_flux.shape[0]

    log_linear = False # if True , model = exp( sum_i a_i * log(template_flux_i) ), else model = sum_i a_i * template_flux_i
    
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
    #for a in range(npar) : 
    #    log.debug("param %d : %s"%(a,str(uparam[a])))
    
   
    node_grid_coords=np.zeros((npar,3)).astype(int)   
    for a in range(npar) : # a is an axis         
        # this is the coordinate on axis 'a' of the best node
        i=np.where(uparam[a]==param[a,best_model_id])[0][0]
        node_grid_coords[a]=np.array([i-1,i,i+1])
        log.debug("node_grid_coords[%d]=%s"%(a,node_grid_coords[a]))
    
    # we don't always have a template on all nodes
    node_template_ids=[] 
    node_cube_coords=[] 
    for i0,j0 in zip(node_grid_coords[0],[-1,0,1]) :
        for i1,j1 in zip(node_grid_coords[1],[-1,0,1]) :
            for i2,j2 in zip(node_grid_coords[2],[-1,0,1]) :
                
                # check whether coord is in grid
                in_grid = (i0>=0)&(i0<uparam[0].size)&(i1>=0)&(i1<uparam[1].size)&(i2>=0)&(i2<uparam[2].size)
                if not in_grid :
                    continue
                # check whether there is a template on this node
                selection=np.where((param[0]==uparam[0][i0])&(param[1]==uparam[1][i1])&(param[2]==uparam[2][i2]))[0]
                if selection.size == 0 : # no template on node
                    log.debug("not template for params = %f,%f,%f"%(uparam[0][i0],uparam[1][i1],uparam[2][i2]))
                    continue
                # we have one
                node_cube_coords.append([j0,j1,j2])
                node_template_ids.append(selection[0])
    node_template_ids=np.array(node_template_ids).astype(int)
    node_cube_coords=np.array(node_cube_coords).astype(int)
    
    # the parameters of the fit are npar coordinates in the range [-1,1] centered on best fit node
    coord=np.zeros(npar)
    
    n_templates = node_template_ids.size
    
    # we are done with the indexing and choice of template nodes
    node_template_flux = template_flux[node_template_ids]
        
    # compute all weighted scalar products among templates (only works if linear combination, not the log version)
    HB=np.zeros(n_templates)
    HA=np.zeros((n_templates,n_templates))
    for t in range(n_templates) :
        HB[t] = np.sum(data_ivar*data_flux*node_template_flux[t])
        for t2 in range(n_templates) :
            if HA[t2,t] != 0 :
                HA[t,t2] = HA[t2,t]
            else :
                HA[t,t2] = np.sum(data_ivar*node_template_flux[t]*node_template_flux[t2])
    
    chi2_0 = np.sum(data_ivar*data_flux**2)
    
    # chi2  =  np.sum(data_ivar*(data_flux-model)**2)
    #       =  chi2_0 - 2*np.sum(data_ivar*data_flux*model) + np.sum(data_ivar*model**2)
    # model = sum_i coef_i model_i
    # chi2  =  chi2_0 - 2* sum_i coef_i * HB[i] + sum_ij coef_i * coef_j * HA[i,j]
    # chi2  =  chi2_0 - 2*np.inner(coef,HB) + np.inner(coef,HA.dot(coef))
    
    
    # initial state
    coef = _compute_coef(coord,node_cube_coords)
    chi2 = chi2_0 - 2*np.inner(coef,HB) + np.inner(coef,HA.dot(coef))
    log.debug("init coord=%s chi2/ndata=%f"%(coord,chi2/ndata))
    
    # now we have to do the fit
    # fitting one axis at a time (simultaneous fit of 3 axes was tested and found inefficient : rapidly stuck on edges)
    # it has to be iterative because the model is a non-linear combination of parameters w, ex: w[0]*(1-w[1])*(1-w[2])
    for loop in range(50) :
        
        previous_chi2=chi2.copy()
        previous_coord=coord.copy()
        
        for a in range(npar) :
            previous_chi2_a=chi2.copy()
            
            # it's a linear combination of templates, but the model is non-linear function of coordinates
            # so there is no gain in trying to fit robustly with Gauss-Newton, we simply do a scan
            # it is converging rapidely (need however to iterate on axes)
            xcoord=coord.copy()
            xx=np.linspace(-1,1,41) # keep points on nodes , 41 is the resolution, 0.05 of node inter-distance
            chi2=np.zeros(xx.shape)
            for i,x in enumerate(xx) :
                xcoord[a]=x
                coef = _compute_coef(xcoord,node_cube_coords)
                if np.sum(coef)==0 : # outside valid range
                    chi2[i]=1e20
                else :
                    chi2[i] = chi2_0 - 2*np.inner(coef,HB) + np.inner(coef,HA.dot(coef))
            
            ibest=np.argmin(chi2)
            chi2=chi2[ibest]
            coord[a]=xx[ibest]
        
        log.debug("loop #%d coord=%s chi2/ndata=%f (-dchi2_loop=%f -dchi2_tot=%f)"%(loop,coord,chi2/ndata,previous_chi2-chi2,template_chi2[best_model_id]-chi2))
        diff=np.max(np.abs(coord-previous_coord))
        if diff < 0.001 :
            break
    
    # finally perform an exact best fit per axis 
    for loop in range(50) :
        previous_chi2=chi2.copy()
        previous_coord=coord.copy()
        for a in range(npar) :
            if coord[a]==-1 or coord[a]==1 :
                continue # we are on edge, no gain in refitting
            xcoord=coord.copy()
            coef_minus = _compute_coef(xcoord,node_cube_coords)
            eps=0.001
            xcoord[a] += eps
            coef_plus  = _compute_coef(xcoord,node_cube_coords)
            dcoef_dcoord = (coef_plus-coef_minus)/eps # do a numeric derivative
            #log.debug("dcoef_dcoord=%s"%dcoef_dcoord)
            B = np.inner(dcoef_dcoord,HB) - np.inner(dcoef_dcoord,HA.dot(coef_minus))        
            A = np.inner(dcoef_dcoord,HA.dot(dcoef_dcoord))
            if A>0 :
                dcoord=B/A
                #log.debug("dcoord=%f"%dcoord)
                tmp_coord=coord.copy()
                tmp_coord[a] += dcoord
                if tmp_coord[a]<-1 or tmp_coord[a]>1 :
                    #log.debug("do not allow extrapolations")
                    continue            
                coef = _compute_coef(tmp_coord,node_cube_coords)
                tmp_chi2 = chi2_0 - 2*np.inner(coef,HB) + np.inner(coef,HA.dot(coef))
                if tmp_chi2 < chi2 :
                    log.debug("Improved chi2 by %f with a shift along %d of %f"%(chi2-tmp_chi2,a,dcoord))
                    coord=tmp_coord
                    chi2 = tmp_chi2
        diff=np.max(np.abs(coord-previous_coord))
        if diff < 0.001 :
            break    
    
    coef = _compute_coef(coord,node_cube_coords)
    chi2 = chi2_0 - 2*np.inner(coef,HB) + np.inner(coef,HA.dot(coef))
    
    input_number_of_templates=template_flux.shape[0]
    final_coefficients=np.zeros(input_number_of_templates)
    final_coefficients[node_template_ids]=coef
    
    log.debug("COORD=%s"%coord)
    log.debug("COEF=%s"%coef)    
    #for i in np.where(final_coefficients>0)[0] :
    #    log.debug("TEFF[%d]=%f"%(i,teff[i]))
    #    log.debug("LOGG[%d]=%f"%(i,logg[i]))
    #    log.debug("FEH[%d]=%f"%(i,feh[i]))
    log.debug("TEFF=%f"%np.inner(final_coefficients,teff))
    log.debug("LOGG=%f"%np.inner(final_coefficients,logg))
    log.debug("FEH=%f"%np.inner(final_coefficients,feh))
    log.debug("Contributing template Ids=%s"%np.where(final_coefficients!=0)[0])
    
    '''
    # useful debugging plot
    import matplotlib.pyplot as plt
    plt.figure()        
    ok=np.where(data_ivar>0)[0]
    ii=np.argsort(data_wave[ok])
    twave=data_wave[ok][ii]
    tflux=data_flux[ok][ii]
    tivar=data_ivar[ok][ii]
    #plt.errorbar(twave,tflux,1./np.sqrt(tivar),fmt="o")
    plt.plot(twave,tflux,".",c="gray",alpha=0.2)
    dw=np.min(twave[twave>twave[0]+0.5]-twave[0])
    bins=np.linspace(twave[0],twave[-1],(twave[-1]-twave[0])/dw+1)
    sw,junk=np.histogram(twave,bins=bins,weights=tivar)
    swx,junk=np.histogram(twave,bins=bins,weights=tivar*twave)
    swy,junk=np.histogram(twave,bins=bins,weights=tivar*tflux)
    tflux=swy[sw>0]/sw[sw>0]
    twave2=swx[sw>0]/sw[sw>0]
    terr=1./np.sqrt(sw[sw>0])
    plt.errorbar(twave2,tflux,terr,fmt="o",alpha=0.5)
    model = np.zeros(data_flux.shape)
    for c,t in zip(coef,node_template_flux) :
        model += c*t
    plt.plot(twave,model[ok][ii],"-",c="r")
    plt.show()
    '''

    
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

    # fit continuum and save it
    continuum={}
    for cam in wave.keys() :
        tmp=applySmoothingFilter(flux[cam]) # this is fast
        continuum[cam] = tmp
            
    # mask out wavelength that could bias the fit
    
    log.debug("mask potential cosmics (3 sigma positive fluctuations)")
    for cam in wave.keys() :
        ok=np.where((ivar[cam]>0))[0]
        if ok.size>0 :
            ivar[cam][ok] *= (flux[cam][ok]<(continuum[cam][ok]+3/np.sqrt(ivar[cam][ok])))
    
    
    log.debug("mask sky lines")
    # in vacuum
    # mask blue lines that can affect fit of Balmer series
    # line at 5577 has a stellar line close to it !
    # line at 7853. has a stellar line close to it !
    # mask everything above 8270A because it can bias the star redshift 
    # all of this is based on analysis of a few exposures of BOSS data
    # in vacuum
    skylines=np.array([4047.5,4359.3,5462.3,5578.9,5891.3,5897.3,6301.8,6365.4,7823.3,7855.2])
    
    hw=6. # A
    for cam in wave.keys() :
        for line in skylines :
            ivar[cam][(wave[cam]>=(line-hw))&(wave[cam]<=(line+hw))]=0.
        ivar[cam][wave[cam]>8270]=0.
    
    # mask telluric lines
    srch_filename = "data/arc_lines/telluric_lines.txt"
    if not resource_exists('desispec', srch_filename):
        log.error("Cannot find telluric mask file {:s}".format(srch_filename))
        raise Exception("Cannot find telluric mask file {:s}".format(srch_filename))
    telluric_mask_filename = resource_filename('desispec', srch_filename)
    telluric_features = np.loadtxt(telluric_mask_filename)
    log.debug("Masking telluric features from file %s"%telluric_mask_filename)
    for cam in wave.keys() :
        for feature in telluric_features :
            ivar[cam][(wave[cam]>=feature[0])&(wave[cam]<=feature[1])]=0.
    
    

    # add error propto to flux to account for model error
    if template_error>0  :
        for cam in wave.keys() :
            ok=np.where(ivar[cam]>0)[0]
            if ok.size>0 :
                ivar[cam][ok] = 1./ ( 1./ivar[cam][ok] + (template_error*continuum[cam][ok] )**2 )
    
    # normalize data and store them in single array
    data_wave=np.array([])
    data_flux=np.array([])
    data_continuum=np.array([])
    data_ivar=np.array([])
    data_index=np.array([])
    sorted_keys = list(wave.keys())
    sorted_keys.sort() # force sorting the keys to agree with models (found unpredictable ordering in tests)
    for index,cam in enumerate(sorted_keys) :
        data_index=np.append(data_index,np.ones(wave[cam].size)*index)
        data_wave=np.append(data_wave,wave[cam])
        data_flux=np.append(data_flux,flux[cam]/(continuum[cam]+(continuum[cam]==0)))
        data_continuum=np.append(data_continuum,continuum[cam])
        data_ivar=np.append(data_ivar,ivar[cam]*continuum[cam]**2)
    data_index=data_index.astype(int)
    
    ndata = np.sum(data_ivar>0)
    
    
    # start looking at models
    
    # find canonical f-type model: Teff=6000, logg=4, Fe/H=-1.5
    canonical_model=np.argmin((teff-6000.0)**2+(logg-4.0)**2+(feh+1.5)**2)
    
    # fit redshift on canonical model
    # we use the original data to do this
    # because we resample both the data and model on a logarithmic grid in the routine
    
    if True : # mask Ca H&K lines. Present in ISM, can bias the stellar redshift fit
        log.debug("Mask ISM lines for redshift")
        ismlines=np.array([3934.77,3969.59])
        hw=6. # A
        for cam in wave.keys() :
            for line in ismlines :
                ivar[cam][(wave[cam]>=(line-hw))&(wave[cam]<=(line+hw))]=0.
    
    z = redshift_fit(wave, flux, ivar, resolution_data, stdwave, stdflux[canonical_model], z_max, z_res)
            
    # now we go back to the model spectra , redshift them, resample, apply resolution, normalize and chi2 match
    
    ntemplates=stdflux.shape[0]

    # here we take into account the redshift once and for all
    shifted_stdwave=stdwave*(1+z)
        
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
        log.debug("Finished serial loop")

    # collect results
    # in case the exit of the multiprocessing pool is not ordered as the input
    # we returned the template_id
    template_flux=np.zeros((ntemplates,data_flux.size))
    template_norm=np.zeros((ntemplates,data_flux.size))
    for result in results :
        template_id       = result[0]
        template_tmp_wave = result[1]
        template_tmp_flux = result[2]
        template_tmp_norm = result[3]
        mdiff=np.max(np.abs(data_wave-template_tmp_wave)) # just a safety check
        if mdiff>1.e-5 :
            log.error("error indexing of wave and flux somewhere above, checking if it's just an ordering issue, max diff=%f"%mdiff)
            raise ValueError("wavelength array difference cannot be fixed with reordering, ordered max diff=%f"%mdiff)
        template_flux[template_id] = template_tmp_flux
        template_norm[template_id] = template_tmp_norm

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
    
    log.debug("use best fit to derive calibration and apply it to the templates before refitting the star ...")
    # the division by the median filtered spectrum leaves some imprint of the input transmission
    # so we will apply calibration to the model and redo the whole fit
    # to make sure this is not driving the stellar model selection.

    
    log.debug("remultiply template by their norme")
    template_flux *= template_norm 
    
    log.debug("compute best fit model")
    model=np.zeros(data_wave.size)
    for c,t in zip(coef,template_flux) :
        if c>0 : model += c*t


    func_args=[]    
    for index in np.unique(data_index) :
        log.debug("compute calib for cam index %d"%index)
        ii=np.where(data_index==index)[0]
        calib = (data_flux[ii]*data_continuum[ii])/(model[ii]+(model[ii]==0))
        scalib = applySmoothingFilter(calib,width=400)
        
        min_scalib=0.
        bad=scalib<=min_scalib
        if np.sum(bad)>0 :
            scalib[bad]=min_scalib
            
        log.debug("multiply templates by calib for cam index %d"%index)
        template_flux[:,ii] *= scalib
        
        # apply this to all the templates and recompute median filter
        for t in range(template_flux.shape[0]) :
            arguments={"template_id":t,"camera_index":index,"template_flux":template_flux[t][ii]}
            func_args.append(arguments)
    
    if ncpu > 1:
        log.debug("divide templates by median filters using multiprocessing.Pool of ncpu=%d"%ncpu)
        pool = multiprocessing.Pool(ncpu)
        results  =  pool.map(_func2, func_args)
        log.debug("finished pool.map()"); sys.stdout.flush()
        pool.close()
        pool.join()
        log.debug("finished pool.join()"); sys.stdout.flush()
    else :
        log.debug("divide templates serially")
        results = [_func2(x) for x in func_args]
        log.debug("Finished serial loop")
    
    # collect results
    for result in results :
        template_id = result[0]
        index  = result[1]
        template_flux[template_id][data_index==index] /= (result[2] + (result[2]==0))
    
    log.debug("refit the model ...")
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


def normalize_templates(stdwave, stdflux, mag, filter_name):
    """Returns spectra normalized to input magnitudes.

    Args:
        stdwave : 1D array of standard star wavelengths [Angstroms]
        stdflux : 1D observed flux
        mag : float desired magnitude
        filter_name : filter_name, e.g. DECAM_G, DECAM_R

    Returns:
        stdwave : same as input
        normflux : normalized flux array

    Only SDSS_r band is assumed to be used for normalization for now.
    """
    log = get_logger()
    fluxunits = 1e-17 * units.erg / units.s / units.cm**2 / units.Angstrom
    filter_response=load_filter(filter_name)
    apMag=filter_response.get_ab_magnitude(stdflux*fluxunits,stdwave)
    scalefac=10**((apMag-mag)/2.5)
    log.debug('scaling mag {:.3f} to {:.3f} using scalefac {:.3f}'.format(apMag,mag, scalefac))
    normflux=stdflux*scalefac

    return normflux

def compute_flux_calibration(frame, input_model_wave,input_model_flux,input_model_fibers, nsig_clipping=4.,deg=2,debug=False):
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
    stdfibers = np.where(isStdStar(frame.fibermap['DESI_TARGET']))[0]
    assert len(stdfibers) > 0

    if not np.all(np.in1d(stdfibers, input_model_fibers)):
        bad = set(input_model_fibers) - set(stdfibers)
        log.error('Discarding input_model_fibers that are not standards: {}'.format(bad))
        stdfibers = np.intersect1d(stdfibers, input_model_fibers)

    stdstars = frame[stdfibers]

    nwave=stdstars.nwave
    nstds=stdstars.flux.shape[0]

    dwave=(stdstars.wave-np.mean(stdstars.wave))/(stdstars.wave[-1]-stdstars.wave[0]) # normalized wave for polynomial fit

    # resample model to data grid and convolve by resolution
    model_flux=np.zeros((nstds, nwave))
    convolved_model_flux=np.zeros((nstds, nwave))

    for fiber in range(model_flux.shape[0]) :
        model_flux[fiber]=resample_flux(stdstars.wave,input_model_wave,input_model_flux[fiber])
        convolved_model_flux[fiber]=stdstars.R[fiber].dot(model_flux[fiber])

    # iterative fitting and clipping to get precise mean spectrum
    current_ivar=stdstars.ivar*(stdstars.mask==0)
    
    #- Start with a first pass median rejection
    calib = (convolved_model_flux!=0)*(stdstars.flux/(convolved_model_flux + (convolved_model_flux==0)))
    median_calib = np.median(calib, axis=0)

    # First fit of smooth correction per fiber, and 10% model error to variance,  and perform first outlier rejection
    smooth_fiber_correction=np.ones((stdstars.flux.shape))
    chi2=np.zeros((stdstars.flux.shape))  

    for fiber in range(nstds) :
        M = median_calib*stdstars.R[fiber].dot(model_flux[fiber])
        
        try:
            pol=np.poly1d(np.polyfit(dwave,stdstars.flux[fiber]/(M+(M==0)),deg=deg,w=current_ivar[fiber]*M**2))
            smooth_fiber_correction[fiber]=pol(dwave)
        except ValueError :
            log.warning("polynomial fit for fiber %d failed"%fiber)
            current_ivar[fiber]=0.

        chi2[fiber]=current_ivar[fiber]*(stdstars.flux[fiber]-smooth_fiber_correction[fiber]*M)**2
        
    
    bad=(chi2>nsig_clipping**2)
    current_ivar[bad] = 0
    
    sqrtw=np.sqrt(current_ivar)
    sqrtwflux=np.sqrt(current_ivar)*stdstars.flux

    # diagonal sparse matrices
    D1=scipy.sparse.lil_matrix((nwave,nwave))
    D2=scipy.sparse.lil_matrix((nwave,nwave))

    
    nout_tot=0
    previous_mean=0.
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

        if np.sum(current_ivar>0)==0 :
            log.error("null ivar, cannot calibrate this frame")
            raise ValueError("null ivar, cannot calibrate this frame")

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
                pol=np.poly1d(np.polyfit(dwave,stdstars.flux[fiber]/(M+(M==0)),deg=deg,w=current_ivar[fiber]*M**2))
                smooth_fiber_correction[fiber]=pol(dwave)
            except ValueError :
                log.warning("polynomial fit for fiber %d failed"%fiber)
                current_ivar[fiber]=0.
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

        if nout_iter == 0 and np.max(np.abs(mean-previous_mean))<0.0001 :
            break
        previous_mean = mean
    
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
    mask = frame.mask.copy()

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

        self.meta = dict(units='photons/(erg/s/cm^2)')

    def __repr__(self):
        txt = '<{:s}: nspec={:d}, nwave={:d}, units={:s}'.format(
            self.__class__.__name__, self.nspec, self.nwave, self.meta['units'])

        # Finish
        txt = txt + '>'
        return (txt)


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


def ZP_from_calib(exptime, wave, calib):
    """ Calculate the ZP in AB magnitudes given the calibration and the wavelength arrays
    Args:
        exptime:  float;  exposure time in seconds
        wave:  1D array (A)
        calib:  1D array (converts erg/s/A to photons/s/A)

    Returns:
      ZP_AB: 1D array of ZP values in AB magnitudes

    """
    ZP_flambda = 1e-17 / (calib/exptime)  # erg/s/cm^2/A
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

    Returns:
        qadict: dict of QA outputs
          Need to record simple Python objects for yaml (str, float, int)

    """
    log = get_logger()
    qadict = {}

    # Unpack model
    exptime = frame.meta['EXPTIME']

    # Standard stars
    stdfibers = np.where(isStdStar(frame.fibermap['DESI_TARGET']))[0]
    stdstars = frame[stdfibers]
    nstds = len(stdfibers)
    #try:
    #    assert np.array_equal(frame.fibers[stdfibers], input_model_fibers)
    #except AssertionError:
    #    log.error("Bad indexing in standard stars")

    # Calculate ZP for mean spectrum
    #medcalib = np.median(fluxcalib.calib,axis=0)
    medcalib = np.median(fluxcalib.calib[stdfibers],axis=0)
    ZP_AB = ZP_from_calib(exptime, fluxcalib.wave, medcalib)  # erg/s/cm^2/A

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
        ZP_stars = ZP_from_calib(exptime, i_wave, icalib)
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
