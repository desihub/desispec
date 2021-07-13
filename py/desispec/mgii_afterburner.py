### Authors: Edmond CHAUSSIDON (CEA Saclay), Corentin RAVOUX (CEA Saclay)
### Contact: edmond.chaussidon@cea.fr, corentin.ravoux@cea.fr
###
### Note:
### Please see https://desi.lbl.gov/trac/attachment/wiki/TargetSelectionWG/TSTeleconMinutes/DESITS210113/QSO_completeness_Ravoux_13_01_2021_v2.pdf for details about the MgII fitter
###
### the "optimal"/chosen parameters are (can be modify when desi_qso_mgii_afterburner is called):
### lambda_width      = 250
### max_sigma         = 200
### min_sigma         = 10
### min_deltachi2     = 16
### min_signifiance_A = 3
### min_A             = 0

import sys

import numpy as np
from astropy.table import Table
from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter

import redrock.templates

from desispec.io import read_spectra
from desispec.coaddition import coadd_cameras
from desispec.interpolation import resample_flux

import logging
logger = logging.getLogger("mgii_afterburner")


def load_redrock_templates(template_dir=None):
    '''
    < COPY from prospect.plotframes to avoid to load prospect in desispec >

    Load redrock templates; redirect stdout because redrock is chatty
    '''
    saved_stdout = sys.stdout
    sys.stdout = open('/dev/null', 'w')
    try:
        templates = dict()
        for filename in redrock.templates.find_templates(template_dir=template_dir):
            tx = redrock.templates.Template(filename)
            templates[(tx.template_type, tx.sub_type)] = tx
    except Exception as err:
        sys.stdout = saved_stdout
        raise(err)
    sys.stdout = saved_stdout
    return templates


def create_model(spectra, redshifts,
                 archetype_fit=False,
                 archetypes_dir=None,
                 template_dir=None):
    '''
    < COPY from prospect.plotframes to avoid to load prospect in desispec >

    Returns model_wave[nwave], model_flux[nspec, nwave], row matched to redshifts,
    which can be in a different order than spectra.
    - redshifts must be entry-matched to spectra.
    '''

    if archetype_fit:
        from redrock.archetypes import All_archetypes

    if np.any(redshifts['TARGETID'] != spectra.fibermap['TARGETID']) :
        raise RuntimeError('zcatalog and spectra do not match (different targetids)')

    templates = load_redrock_templates(template_dir=template_dir)

    #- Empty model flux arrays per band to fill
    model_flux = dict()
    for band in spectra.bands:
        model_flux[band] = np.zeros(spectra.flux[band].shape)

    for i in range(len(redshifts)):
        zb = redshifts[i]

        if archetype_fit:
          archetypes = All_archetypes(archetypes_dir=archetypes_dir).archetypes
          archetype  = archetypes[zb['SPECTYPE']]
          coeff      = zb['COEFF']

          for band in spectra.bands:
              wave                = spectra.wave[band]
              wavehash            = hash((len(wave), wave[0], wave[1], wave[-2], wave[-1], spectra.R[band].data.shape[0]))
              dwave               = {wavehash: wave}
              mx                  = archetype.eval(zb['SUBTYPE'], dwave, coeff, wave, zb['Z'])
              model_flux[band][i] = spectra.R[band][i].dot(mx)

        else:
          tx    = templates[(zb['SPECTYPE'], zb['SUBTYPE'])]
          coeff = zb['COEFF'][0:tx.nbasis]
          model = tx.flux.T.dot(coeff).T

          for band in spectra.bands:
              mx                  = resample_flux(spectra.wave[band], tx.wave*(1+zb['Z']), model)
              model_flux[band][i] = spectra.R[band][i].dot(mx)

    #- Now combine, if needed, to a single wavelength grid across all cameras
    if spectra.bands == ['brz'] :
        model_wave = spectra.wave['brz']
        mflux = model_flux['brz']

    elif np.all([ band in spectra.bands for band in ['b','r','z'] ]) :
        br_split = 0.5*(spectra.wave['b'][-1] + spectra.wave['r'][0])
        rz_split = 0.5*(spectra.wave['r'][-1] + spectra.wave['z'][0])
        keep = dict()
        keep['b'] = (spectra.wave['b'] < br_split)
        keep['r'] = (br_split <= spectra.wave['r']) & (spectra.wave['r'] < rz_split)
        keep['z'] = (rz_split <= spectra.wave['z'])
        model_wave = np.concatenate( [
            spectra.wave['b'][keep['b']],
            spectra.wave['r'][keep['r']],
            spectra.wave['z'][keep['z']],
        ] )
        mflux = np.concatenate( [
            model_flux['b'][:, keep['b']],
            model_flux['r'][:, keep['r']],
            model_flux['z'][:, keep['z']],
        ], axis=1 )
    else :
        raise RuntimeError("create_model: Set of bands for spectra not supported")

    return model_wave, mflux


def get_spectra(spectra_name, redrock_name, lambda_width, index_to_fit,
                template_dir=None, archetypes_dir=None):
    """
    Get the spectra and the best fit model from a given spectra and redrock file.
    Args:
        spectra_name (str): The name of the spectra file.
        redrock_name (str): The name of the redrock file associated to the spectra
                          file.
        lambda_width (float): The width in wavelength (in Angstrom) considered
                              for the fitting arount the MgII peak
        index_to_fit (boolean numpy array): boolean array of size 500 specifing which spectra have to be used
        add_linear_term (boolean): Add a linear term to the Gaussian peak term
                                   to fit the continuum.
        template_dir (str): If the redrock template variable is not loaded by
                            the desi environment, specify the template path
        archetypes_dir (str): If not None, use the archetypes templates in the
                              path specified
    Returns:
        target_id (numpy array): Array containing target id of the the object
                                 to fit
        redshift_redrock (numpy array): Array containing the redshift of the the
                                        object to fit
        flux (numpy array): Array containing the full flux arrays of every
                            object to fit
        ivar_flux (numpy array): Array containing the inverse variance arrays
                                 of every object to fit
        model_flux (numpy array): Array containing the best fit redrock model
                                  for every object to fit
        wavelength (numpy array): Array containing the wavelength
        index_with_fit (boolean numpy array): boolean array of index_to_fit size masking index where mgII fitter is not apply
    """
    spectra = read_spectra(spectra_name)
    spectra = spectra.select(targets=spectra.fibermap["TARGETID"][index_to_fit])

    if 'brz' not in spectra.bands:
        spectra = coadd_cameras(spectra)

    redshifts = Table.read(redrock_name, 'REDSHIFTS')[index_to_fit]
    if archetypes_dir is not None:
        model_wave, model_flux = create_model(spectra,
                                              redshifts,
                                              archetype_fit=True,
                                              archetypes_dir=archetypes_dir,
                                              template_dir=template_dir)
    else:
        model_wave, model_flux = create_model(spectra,
                                              redshifts,
                                              archetype_fit=False,
                                              archetypes_dir=None,
                                              template_dir=template_dir)

    redshift_redrock = redshifts["Z"]
    wavelength = spectra.wave['brz']

    mgii_peak_1, mgii_peak_2 = 2803.5324, 2796.3511
    mean_mgii_peak = (mgii_peak_1 + mgii_peak_2)/2
    non_visible_peak = (redshift_redrock+1) * mean_mgii_peak < np.min(wavelength) + lambda_width/2
    non_visible_peak |= (redshift_redrock+1) * mean_mgii_peak > np.max(wavelength) - lambda_width/2

    index_with_fit = ~non_visible_peak

    target_id = spectra.fibermap["TARGETID"][index_with_fit]
    redshift_redrock = redshift_redrock[index_with_fit]
    flux = spectra.flux['brz'][index_with_fit]
    ivar_flux = spectra.ivar['brz'][index_with_fit]
    model_flux = model_flux[index_with_fit]

    return target_id, redshift_redrock, flux, ivar_flux, model_flux, wavelength, index_with_fit


def fit_mgii_line(target_id,
                  redshift_redrock,
                  flux,
                  ivar_flux,
                  model_flux,
                  wavelength,
                  lambda_width,
                  add_linear_term=False,
                  gaussian_smoothing_fit=None,
                  mask_mgii=None):
    """
    Fitting routine. Fit a Gaussian peak on preselected spectra and return the
    main parameters of the fit including parameter errors.
    Args:
        target_id (numpy array): Array containing target id of the the object
                                 to fit
        redshift_redrock (numpy array): Array containing the redshift of the the
                                        object to fit
        flux (numpy array): Array containing the full flux arrays of every
                            object to fit
        ivar_flux (numpy array): Array containing the inverse variance arrays
                                 of every object to fit
        model_flux (numpy array): Array containing the best fit redrock model
                                  for every object to fit
        wavelength (numpy array): Array containing the wavelength
        lambda_width (float): The width in wavelength (in Angstrom) considered
                              for the fitting arount the MgII peak
        add_linear_term (boolean): Add a linear term to the Gaussian peak term
                                   to fit the continuum.
        gaussian_smoothing_fit (float): If not None, the spectra is smoothed by
                                        the given value before the fit
        mask_mgii (float): If not None, mask a region of near the MgII peak with
                           the given witdh to fit double MgII peak (in progress)
    Returns:
        fit_results (numpy array): Array containing the parameters of the fit
    """
    mgii_peak_1 = 2803.5324
    mgii_peak_2 = 2796.3511
    mean_mgii_peak = (mgii_peak_1 + mgii_peak_2)/2
    mgii_peak_observed_frame = (redshift_redrock+1) * mean_mgii_peak

    if(add_linear_term):
        fit_results = np.zeros((target_id.shape[0],11))
    else:
        fit_results = np.zeros((target_id.shape[0],9))

    for i in range(len(flux)):
        centered_wavelenght = wavelength - mgii_peak_observed_frame[i]
        mask_wave = np.abs(centered_wavelenght) < lambda_width/2

        if(mask_mgii) is not None:
            mask_wave &= np.abs(centered_wavelenght) > mask_mgii/2

        flux_centered = flux[i][mask_wave]
        model_flux_centered = model_flux[i][mask_wave]
        with np.errstate(divide='ignore',invalid='ignore'): # if zero division --> sigma = np.inf --> in curve_fit and the rest we only have 1/sigma --> 1/inf = 0.0
            sigma_flux_centered = (1 /np.sqrt(ivar_flux[i]))[mask_wave]

        if(gaussian_smoothing_fit is not None):
            flux_centered = gaussian_filter(flux_centered, gaussian_smoothing_fit)

        if(add_linear_term):
            fit_function = lambda x, A, sigma, B, C : A * np.exp(-1.0 * (x)**2 / (2 * sigma**2)) + B + C * x
            try:
                popt, pcov = curve_fit(fit_function,
                                       xdata=centered_wavelenght[mask_wave],
                                       ydata=flux_centered,
                                       sigma=sigma_flux_centered,
                                       p0=[1.0,lambda_width/2,np.mean(flux_centered),0.0],
                                       bounds=([-np.inf,-np.inf,-np.inf,-0.01], [np.inf,np.inf,np.inf,0.01]))
            except RuntimeError:
                print("Fit not converged")
                popt = np.full((4),0)
                pcov = np.full((4,4),0)
            fit_results[i][1:4] = popt[0:3]
            fit_results[i][4:7] = np.diag(pcov)[0:3]
            fit_results[i][7] = popt[3]
            fit_results[i][8] = np.diag(pcov)[3]
        else:
            fit_function = lambda x, A, sigma, B : A * np.exp(-1.0 * (x)**2 / (2 * sigma**2)) + B
            try:
                popt, pcov = curve_fit(fit_function,
                                       xdata=centered_wavelenght[mask_wave],
                                       ydata=flux_centered,
                                       sigma=sigma_flux_centered,
                                       p0=[1.0, lambda_width/2, np.mean(flux_centered)])
            except RuntimeError:
                print("Fit not converged")
                popt = np.full((3),0)
                pcov = np.full((3,3),0)
            fit_results[i][1:4] = popt
            fit_results[i][4:7] = np.diag(pcov)

        chi2_gauss = (np.sum(((flux_centered - fit_function(centered_wavelenght[mask_wave], *popt))/sigma_flux_centered)**2))
        chi2_RR = (np.sum(((flux_centered - model_flux_centered)/sigma_flux_centered)**2))

        fit_results[i][0] = chi2_RR - chi2_gauss

    return fit_results


def create_mask_fit(fit_results,
                    max_sigma=None,
                    min_sigma=None,
                    min_deltachi2=None,
                    min_A=None,
                    min_signifiance_A=None):
    """
    Create a mask based on the results of the MgII fit and some parameters which
    constraints these parameters
    Args:
        fit_results (numpy array): Array containing the parameters of the fit
        max_sigma (float): Maximal value for the standard deviation of the
                           fitted Gaussian peak
        min_sigma (float): Minimal value for the standard deviation of the
                           fitted Gaussian peak
        min_deltachi2 (float): Minimal value for the difference of chi2 between
                               redrock fit and MgII fitter over the lambda_width
                               interval considered
        min_A (float): Minimal value for the amplitude of the fitted Gaussian peak
        min_signifiance_A (float): Minimal signifiance for the amplitude of the
                                   fitted Gaussian peak. The signifiance is here
                                   define as the ratio between peak amplitude
                                   and error on the peak amplitude.
    Returns:
        mask (boolean numpy array): mask with the same length than
                                    fit_results which indicates the objects
                                    validating parameter constraints
    """
    mask = np.full(fit_results.shape[0], True)
    if(max_sigma is not None):
        mask &= np.abs(fit_results[:,2]) < max_sigma # sigma < max_sigma
    if(min_sigma is not None):
        mask &= np.abs(fit_results[:,2]) > min_sigma # sigma > min_sigma
    if(min_deltachi2 is not None):
        mask &= np.abs(fit_results[:,0]) > min_deltachi2 # deltachi2 > min_deltachi2
    if(min_A is not None):
        mask &= fit_results[:,1] > min_A # A > min_A
    if(min_signifiance_A is not None):
        mask &= np.abs(fit_results[:,1]) > min_signifiance_A * np.sqrt(np.abs(fit_results[:,4])) # A > min_signifiance_A * sigma(A)
    return mask


def mgii_fitter(spectra_name,
                redrock_name,
                index_to_fit,
                lambda_width,
                add_linear_term=False,
                gaussian_smoothing_fit=None,
                template_dir=None,
                archetypes_dir=None,
                max_sigma=None,
                min_sigma=None,
                min_deltachi2=None,
                min_A=None,
                min_signifiance_A=None):
    """
    MgII fitter afterburner main function. For a given spectra file and its
    associated redrock file (redrock output), returns a numpy mask which indicates
    the objects fitted by redrock as galaxies and considered as QSO by the
    MgII fitter. The mask is determined following input parameters.
    Args:
        spectra_name (str): The name of the spectra file.
        redrock_name (str): The name of the redrock file associated to the spectra
                          file.
        lambda_width (float): The width in wavelength (in Angstrom) considered
                              for the fitting arount the MgII peak
        index_to_fit (boolean numpy array): boolean array of size 500 specifing which spectra have to be used
        add_linear_term (boolean): Add a linear term to the Gaussian peak term
                                   to fit the continuum.
        template_dir (str): If the redrock template variable is not loaded by
                            the desi environment, specify the template path
        archetypes_dir (str): If not None, use the archetypes templates in the
                              path specified
        max_sigma (float): Maximal value for the standard deviation of the
                           fitted Gaussian peak
        min_sigma (float): Minimal value for the standard deviation of the
                           fitted Gaussian peak
        min_deltachi2 (float): Minimal value for the difference of chi2 between
                               redrock fit and MgII fitter over the lambda_width
                               interval considered
        min_A (float): Minimal value for the amplitude of the fitted Gaussian peak
        min_signifiance_A (float): Minimal signifiance for the amplitude of the
                                   fitted Gaussian peak. The signifiance is here
                                   define as the ratio between peak amplitude
                                   and error on the peak amplitude.
    Returns:
        mask_fit (boolean numpy array): mask with the same length than
                                        index_with_fit which indicates object
                                        considered by MgII fitter as QSO

        fit_results (numpy array): Array containing the parameters of the fit to save them

        index_with_fit (boolean numpy array): boolean array of index_to_fit size masking index where mgII fitter is not apply
    """

    (target_id,
     redshift_redrock,
     flux,
     ivar_flux,
     model_flux,
     wavelength,
     index_with_fit) = get_spectra(spectra_name,
                                   redrock_name,
                                   lambda_width,
                                   index_to_fit,
                                   template_dir=template_dir,
                                   archetypes_dir=archetypes_dir)

    fit_results = fit_mgii_line(target_id,
                                redshift_redrock,
                                flux,
                                ivar_flux,
                                model_flux,
                                wavelength,
                                lambda_width,
                                add_linear_term=add_linear_term,
                                gaussian_smoothing_fit=gaussian_smoothing_fit)

    mask_fit = create_mask_fit(fit_results,
                               max_sigma = max_sigma,
                               min_sigma = min_sigma,
                               min_deltachi2 = min_deltachi2,
                               min_A = min_A,
                               min_signifiance_A = min_signifiance_A)

    return mask_fit, fit_results, index_with_fit


### Modifier ces fonctions la pour quelle soit utilisable !

def read_fit(name_fits):
    h = fitsio.FITS(name_fits,"r")[1]
    target_id = h["TARGET_ID"][:]
    fit_results = np.zeros((target_id.shape[0],9))
    fit_results[:,0] = h["CHI2_GAUSS"][:]
    fit_results[:,1] = h["CHI2_RR"][:]
    fit_results[:,2] = h["DELTA_CHI2"][:]
    fit_results[:,3] = h["A"][:]
    fit_results[:,6] = h["VAR_A"][:]
    fit_results[:,4] = h["SIGMA"][:]
    fit_results[:,7] = h["VAR_SIGMA"][:]
    fit_results[:,5] = h["B"][:]
    fit_results[:,8] = h["VAR_B"][:]
    if(len(h) > 9):
        fit_results[:,9] = h["C"]
        fit_results[:,10] = h["VAR_C"]
    return(target_id,fit_results)
