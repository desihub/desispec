"""
desispec.coaddition
===================

Coadd spectra.
"""

from __future__ import absolute_import, division, print_function
import os, sys, time

import numpy as np

import scipy.sparse
import scipy.linalg
import scipy.sparse.linalg

from astropy.table import Table, Column

import multiprocessing

from desiutil.log import get_logger

from desispec.interpolation import resample_flux
from desispec.spectra import Spectra
from desispec.resolution import Resolution
from desispec.fiberbitmasking import get_all_fiberbitmask_with_amp, get_all_nonamp_fiberbitmask_val, get_justamps_fiberbitmask
from desispec.specscore import compute_coadd_scores
from desispec.util import ordered_unique

#- Fibermap columns that come from targeting or MTL files
fibermap_target_cols = (
    'TARGETID',
    'TARGET_RA',
    'TARGET_DEC',
    'PMRA',
    'PMDEC',
    'REF_EPOCH',
    'LAMBDA_REF',
    'FA_TARGET',
    'FA_TYPE',
    'OBJTYPE',
    'NUMTARGET',
    'OBSCONDITIONS',
    'MORPHTYPE',
    'FLUX_G',
    'FLUX_R',
    'FLUX_Z',
    'FLUX_IVAR_G',
    'FLUX_IVAR_R',
    'FLUX_IVAR_Z',
    'REF_ID',
    'REF_CAT',
    'GAIA_PHOT_G_MEAN_MAG',
    'GAIA_PHOT_BP_MEAN_MAG',
    'GAIA_PHOT_RP_MEAN_MAG',
    'PARALLAX',
    'EBV',
    'FLUX_W1',
    'FLUX_W2',
    'FIBERFLUX_G',
    'FIBERFLUX_R',
    'FIBERFLUX_Z',
    'FIBERTOTFLUX_G',
    'FIBERTOTFLUX_R',
    'FIBERTOTFLUX_Z',
    'MASKBITS',
    'SERSIC',
    'SHAPE_R', 'SHAPE_E1', 'SHAPE_E2',
    'PHOTSYS',
    'PRIORITY_INIT',
    'NUMOBS_INIT',
    'RELEASE',
    'BRICKID',
    'BRICKNAME', 'BRICK_OBJID',
    'BLOBDIST',
    'FIBERFLUX_IVAR_G',
    'FIBERFLUX_IVAR_R',
    'FIBERFLUX_IVAR_Z',
    'CMX_TARGET',
    'SV0_TARGET',
    'SV1_TARGET',
    'SV2_TARGET',
    'SV3_TARGET',
    'DESI_TARGET',
    'BGS_TARGET',
    'MWS_TARGET',
    'SCND_TARGET',
    'HPXPIXEL',
)

#- PRIORITY could change between tiles; SUBPRIORITY won't but keep 'em together
fibermap_mtl_cols = (
    'PRIORITY',
    'SUBPRIORITY',
)

#- Fibermap columns that were added by fiberassign
#- ... same per target, regardless of assignment
fibermap_fiberassign_target_cols = (
    'FA_TARGET', 'FA_TYPE', 'OBJTYPE',
    )
#- ... varies per assignment
fibermap_fiberassign_cols = (
    'PETAL_LOC', 'DEVICE_LOC', 'LOCATION', 'FIBER', 'FIBERSTATUS',
    'FIBERASSIGN_X', 'FIBERASSIGN_Y',
    'LAMBDA_REF', 'PLATE_RA', 'PLATE_DEC',
    )

#- Fibermap columns added frome the platemaker coordinates file
fibermap_coords_cols = (
    'NUM_ITER', 'FIBER_X', 'FIBER_Y', 'DELTA_X', 'DELTA_Y',
    'FIBER_RA', 'FIBER_DEC',
    )

#- Fibermap columns with exposure metadata
fibermap_exp_cols = (
    'NIGHT', 'EXPID', 'MJD', 'TILEID', 'EXPTIME'
    )

#- Fibermap columns added by flux calibration
fibermap_cframe_cols = (
    'PSF_TO_FIBER_SPECFLUX',
    )

#- Columns to include in the per-exposure EXP_FIBERMAP
fibermap_perexp_cols = \
    ('TARGETID',) + \
    fibermap_mtl_cols + \
    fibermap_exp_cols + \
    fibermap_fiberassign_cols + \
    fibermap_coords_cols + fibermap_cframe_cols + \
    ('IN_COADD_B', 'IN_COADD_R', 'IN_COADD_Z')

def calc_mean_std_ra_dec(ras, decs):
    """
    Calculate mean/std of ras, decs accounting for RA wraparound and cos(dec)

    Args:
        ras (array): input RA values in degrees
        decs (array): input declination values in degrees

    Returns: mean_ra, std_ra, mean_dec, std_dec

    where the means are in degrees and the standard deviations are in arcsec,
    including cos(dec) correction.

    For efficiency, this does not try to handle dec= +/-90 poles correctly,
    nor arbitrarily large spreads of angles.  i.e. this is ok for a mean
    of fiber positions scattered about a single target, but not for e.g.
    a science analysis of the central location of a cluster of galaxies.
    """
    ras = np.asarray(ras)
    decs = np.asarray(decs)
    if np.max(ras) - np.min(ras) > 180:
        offset = 180.0
        ras = (ras + offset) % 360
    else:
        offset = 0.0

    mean_dec = np.mean(decs)
    std_dec = np.std(decs) * 3600

    mean_ra = (np.mean(ras) - offset + 360) % 360
    std_ra = np.std(ras) * np.cos(np.radians(mean_dec)) * 3600

    return mean_ra, std_ra, mean_dec, std_dec

def use_for_coadd(fiberstatus, band):
    """
    Determine which exposures should be used for a per-camera coadd

    Args:
        fiberstatus (array): FIBERSTATUS bitmasks, one per exposure
        band (str): camera band, 'b', 'r', or 'z'

    Returns: boolean array of whether exposure should be used in coadd or not

    This is factored into a separate function because it is used in
    `coadd`, `coadd_cameras`, and `coadd_fibermap`
    """
    if band not in ('b', 'r', 'z'):
        raise ValueError(f'band={band} should be b, r, or z')

    fiberstatus_bits = get_all_fiberbitmask_with_amp(band)
    good_fiberstatus = ( (fiberstatus & fiberstatus_bits) == 0 )
    return good_fiberstatus


def coadd_fibermap(fibermap, onetile=False):
    """
    Coadds fibermap

    Args:
        fibermap (Table or ndarray): fibermap of individual exposures

    Options:
        onetile (bool): this is a coadd of a single tile, not across tiles

    Returns: (coadded_fibermap, exp_fibermap) Tables


    coadded_fibermap contains the coadded_fibermap for the columns that can
    be coadded, while exp_fibermap is the subset of columns of the original
    fibermap that can't be meaningfully coadded because they are per-exposure
    quantities like FIBER_X.

    If onetile is True, the coadded_fibermap includes additional columns like
    MEAN_FIBER_X that are meaningful if coadding a single tile, but not if
    coadding across tiles.
    """

    log = get_logger()
    log.debug("'coadding' fibermap")

    if onetile:
        ntile = len(np.unique(fibermap['TILEID']))
        if ntile != 1:
            msg = f'input has {ntile} tiles, but onetile=True option'
            log.error(msg)
            raise ValueError(msg)

    #- make a copy of input fibermap that we can modify with new columns.
    #- This will become the per-exposure fibermap for the EXP_FIBERMAP HDU
    exp_fibermap = Table(fibermap, copy=True)

    #- Remove the "fibermap" input from the current namespace so that we don't accidentally use it
    #- NOTE: does not actually delete/modify the original input
    del fibermap

    #- Get TARGETIDs, preserving order in which they first appear
    #- tfmap = "Target Fiber Map", i.e. one row per target instead of one row per exposure
    targets, ii = ordered_unique(exp_fibermap["TARGETID"], return_index=True)
    tfmap = exp_fibermap[ii]
    assert np.all(targets == tfmap['TARGETID'])
    ntarget = targets.size

    #- New columns to fill in for whether exposure was used in coadd
    exp_fibermap['IN_COADD_B'] = np.zeros(len(exp_fibermap), dtype=bool)
    exp_fibermap['IN_COADD_R'] = np.zeros(len(exp_fibermap), dtype=bool)
    exp_fibermap['IN_COADD_Z'] = np.zeros(len(exp_fibermap), dtype=bool)

    #- initialize NUMEXP=-1 to check that they all got filled later
    tfmap['COADD_NUMEXP'] = np.zeros(len(tfmap), dtype=np.int16) - 1
    tfmap['COADD_EXPTIME'] = np.zeros(len(tfmap), dtype=np.float32) - 1
    tfmap['COADD_NUMNIGHT'] = np.zeros(len(tfmap), dtype=np.int16) - 1
    tfmap['COADD_NUMTILE'] = np.zeros(len(tfmap), dtype=np.int16) - 1

    # some cols get combined into mean or rms;
    # MJD handled separately to get min/max/mean
    mean_cols = [
        'DELTA_X', 'DELTA_Y',
        'PSF_TO_FIBER_SPECFLUX',
        ]
    # Note: treat the fiber coordinates separately because of missing coordinate problem
    # that require an additional "good_coords" condition relative to other mean cols

    if onetile:
        mean_cols.extend(['FIBER_X', 'FIBER_Y'])

    #- rms_cols and std_cols must also be in mean_cols
    rms_cols = ['DELTA_X', 'DELTA_Y']
    std_cols = []  # currently none; RA/dec handled separately

    #- Add other MEAN/RMS/STD columns
    for k in mean_cols:
        if k in exp_fibermap.colnames :
            dtype = np.float32
            if k in mean_cols:
                xx = Column(np.zeros(ntarget, dtype=dtype))
                tfmap.add_column(xx,name='MEAN_'+k)
            if k in rms_cols:
                xx = Column(np.zeros(ntarget, dtype=np.float32))
                tfmap.add_column(xx,name='RMS_'+k)
            if k in std_cols:
                xx = Column(np.zeros(ntarget, dtype=np.float32))
                tfmap.add_column(xx,name='STD_'+k)

            tfmap.remove_column(k)

    #- FIBER_RA/DEC handled differently due to RA wraparound and cos(dec)
    if 'FIBER_RA' in exp_fibermap.colnames and 'FIBER_DEC' in exp_fibermap.colnames:
        tfmap.add_column(Column(np.zeros(ntarget, dtype=np.float64)), name='MEAN_FIBER_RA')
        tfmap.add_column(Column(np.zeros(ntarget, dtype=np.float32)), name='STD_FIBER_RA')
        tfmap.add_column(Column(np.zeros(ntarget, dtype=np.float64)), name='MEAN_FIBER_DEC')
        tfmap.add_column(Column(np.zeros(ntarget, dtype=np.float32)), name='STD_FIBER_DEC')
        tfmap.remove_column('FIBER_RA')
        tfmap.remove_column('FIBER_DEC')

    #- MIN_, MAX_, MEAN_MJD over exposures used in coadd
    if 'MJD' in exp_fibermap.colnames :
        dtype = np.float64
        if not 'MIN_MJD' in tfmap.dtype.names :
            xx = Column(np.zeros(ntarget, dtype=dtype))
            tfmap.add_column(xx,name='MIN_MJD')
        if not 'MAX_MJD' in tfmap.dtype.names :
            xx = Column(np.zeros(ntarget, dtype=dtype))
            tfmap.add_column(xx,name='MAX_MJD')
        if not 'MEAN_MJD' in tfmap.dtype.names :
            xx = Column(np.zeros(ntarget, dtype=dtype))
            tfmap.add_column(xx,name='MEAN_MJD')
    
    if 'FIBERSTATUS' in tfmap.dtype.names :
        tfmap.rename_column('FIBERSTATUS', 'COADD_FIBERSTATUS')
    if not  'COADD_FIBERSTATUS' in tfmap.dtype.names :
        raise KeyError("no COADD_FIBERSTATUS column in tfmap")

    if  'FIBERSTATUS' in exp_fibermap.dtype.names :
        fiberstatus_key='FIBERSTATUS'
    elif  'COADD_FIBERSTATUS' in exp_fibermap.dtype.names :
         fiberstatus_key='COADD_FIBERSTATUS'
    else :
         raise KeyError("no FIBERSTATUS nor COADD_FIBERSTATUS column in fibermap")

    for i,tid in enumerate(targets) :
        jj = exp_fibermap["TARGETID"]==tid

        #- Only a subset of "good" FIBERSTATUS flags are included in the coadd
        targ_fibstatuses = exp_fibermap[fiberstatus_key][jj]
        in_coadd_b = use_for_coadd(targ_fibstatuses, 'b')
        in_coadd_r = use_for_coadd(targ_fibstatuses, 'r')
        in_coadd_z = use_for_coadd(targ_fibstatuses, 'z')
        good_coadds = (in_coadd_b | in_coadd_r | in_coadd_z)
        coadd_numexp = np.count_nonzero(good_coadds)
        tfmap['COADD_NUMEXP'][i] = coadd_numexp
        exp_fibermap['IN_COADD_B'][jj] = in_coadd_b
        exp_fibermap['IN_COADD_R'][jj] = in_coadd_r
        exp_fibermap['IN_COADD_Z'][jj] = in_coadd_z

        # Check if there are some good coadds to compute aggregate quantities;
        # Otherwise just use all the (bad) exposures; will still count NUM on good_coadds
        if coadd_numexp>0:
            compute_coadds = good_coadds
            # coadded FIBERSTATUS = bitwise AND of input FIBERSTATUS
            tfmap['COADD_FIBERSTATUS'][i] = np.bitwise_and.reduce(exp_fibermap[fiberstatus_key][jj][good_coadds])
        else:
            compute_coadds = ~good_coadds
            # if all inputs were bad, COADD_FIBERSTATUS is OR of inputs instead of AND
            tfmap['COADD_FIBERSTATUS'][i] = np.bitwise_or.reduce(exp_fibermap[fiberstatus_key][jj])
        
        #- For FIBER_RA/DEC quantities, only average over good coordinates.
        #  There is a bug that some "missing" coordinates were set to FIBER_RA=FIBER_DEC=0
        #  (we are assuming there are not valid targets at exactly 0,0; only missing coords)
        if 'FIBER_RA' in exp_fibermap.colnames and 'FIBER_DEC' in exp_fibermap.colnames:
            good_coords = (exp_fibermap['FIBER_RA'][jj]!=0)|(exp_fibermap['FIBER_DEC'][jj]!=0)
            
            #- Check whether entries with good coordinates exist (if not use all coordinates)
            if np.count_nonzero(good_coords)>0:
                compute_coords = good_coords
            else:
                compute_coords = ~good_coords
                
            #- Check for edge case where good_coadds and good_coords do not overlap:
            #  if they overlap, use both conditions; otherwise compute coordinates over good_coords
            if np.count_nonzero(compute_coadds&compute_coords)>0:
                compute_coadds_coords = compute_coadds&compute_coords
            else:
                #TODO - decide if it's worth adding the following Warning message to the log
                #print(f"Warning: TARGETID lacks overlap between good_coadds and good_coords: {tid}")
                compute_coadds_coords = compute_coords
                        
        # Note: NIGHT and TILEID may not be present when coadding previously
        # coadded spectra.
        if 'NIGHT' in exp_fibermap.colnames:
            tfmap['COADD_NUMNIGHT'][i] = len(np.unique(exp_fibermap['NIGHT'][jj][good_coadds]))
        if 'TILEID' in exp_fibermap.colnames:
            tfmap['COADD_NUMTILE'][i] = len(np.unique(exp_fibermap['TILEID'][jj][good_coadds]))
        if 'EXPTIME' in exp_fibermap.colnames :
            tfmap['COADD_EXPTIME'][i] = np.sum(exp_fibermap['EXPTIME'][jj][good_coadds])

        # Calc MEAN_* and RMS_* columns using the same exposures as the coadd
        # Note RA/DEC/MJD are handled separately
        for k in mean_cols:
            if k in exp_fibermap.colnames :
                vals=exp_fibermap[k][jj][compute_coadds]
                tfmap['MEAN_'+k][i] = np.mean(vals)

        for k in rms_cols:
            if k in exp_fibermap.colnames :
                vals=exp_fibermap[k][jj][compute_coadds]
                # RMS includes mean offset, not same as STD
                tfmap['RMS_'+k][i] = np.sqrt(np.mean(vals**2)).astype(np.float32)

        #- STD of FIBER_RA, FIBER_DEC in arcsec, handling cos(dec) and RA wrap
        if 'FIBER_RA' in exp_fibermap.colnames and 'FIBER_DEC' in exp_fibermap.colnames:
            decs = exp_fibermap['FIBER_DEC'][jj][compute_coadds_coords]
            ras = exp_fibermap['FIBER_RA'][jj][compute_coadds_coords]
            mean_ra, std_ra, mean_dec, std_dec = calc_mean_std_ra_dec(ras, decs)
            tfmap['MEAN_FIBER_RA'][i] = mean_ra
            tfmap['STD_FIBER_RA'][i] = np.float32(std_ra)
            tfmap['MEAN_FIBER_DEC'][i] = mean_dec
            tfmap['STD_FIBER_DEC'][i] = np.float32(std_dec)

        #- future proofing possibility of other STD cols
        for k in std_cols:
            if k in exp_fibermap.colnames:
                vals=exp_fibermap[k][jj][compute_coadds]
                # STD removes mean offset, not same as RMS
                tfmap['STD_'+k][i] = np.std(vals).astype(np.float32)
                        
        # MIN_, MAX_MJD over exposures used in the coadd
        if 'MJD' in exp_fibermap.colnames :
            vals=exp_fibermap['MJD'][jj][compute_coadds]
            tfmap['MIN_MJD'][i] = np.min(vals)
            tfmap['MAX_MJD'][i] = np.max(vals)
            tfmap['MEAN_MJD'][i] = np.mean(vals)

        # Error propagation of IVAR values when taking an unweighted MEAN 
        #- (Note 1: IVAR will be 0.0 if any of ivar[compute_coadds]=0)
        #- (Note 2: these columns are place-holder for possible future use)    
        for k in ['FIBER_RA_IVAR', 'FIBER_DEC_IVAR',
                  'DELTA_X_IVAR', 'DELTA_Y_IVAR'] :
            if k in exp_fibermap.colnames :
                tfmap[k][i]=1./np.mean(1./exp_fibermap[k][jj][compute_coadds])

        #- Targeting bits can evolve, so use bitwise OR of any input bits set
        #- See Sec 5.1 of https://ui.adsabs.harvard.edu/abs/2023AJ....165...50M/abstract
        for targetcol in ('CMX_TARGET',
                          'SV1_DESI_TARGET', 'SV1_BGS_TARGET', 'SV1_MWS_TARGET', 'SV1_SCND_TARGET',
                          'SV2_DESI_TARGET', 'SV2_BGS_TARGET', 'SV2_MWS_TARGET', 'SV2_SCND_TARGET',
                          'SV3_DESI_TARGET', 'SV3_BGS_TARGET', 'SV3_MWS_TARGET', 'SV3_SCND_TARGET',
                          'DESI_TARGET', 'BGS_TARGET', 'MWS_TARGET', 'SCND_TARGET'):
            if targetcol in tfmap.colnames:
                tfmap[targetcol][i] = np.bitwise_or.reduce(exp_fibermap[targetcol][jj],axis=0)

    #- Remove some columns that apply to individual exp but not coadds
    #- (even coadds of the same tile)
    for k in ['NIGHT', 'EXPID', 'MJD', 'EXPTIME', 'NUM_ITER',
            'PSF_TO_FIBER_SPECFLUX']:
        if k in tfmap.colnames:
            tfmap.remove_column(k)

    #- Remove columns that don't apply to coadds across tiles
    if not onetile:
        for k in ['TILEID', 'FIBER', 'FIBER_X', 'FIBER_Y',
                'PRIORITY', 'PETAL_RA', 'PETAL_DEC', 'LAMBDA_REF',
                'FIBERASSIGN_X', 'FIBERASSIGN_Y',
                'PETAL_LOC', 'DEVICE_LOC', 'LOCATION'
                ]:
            if k in tfmap.colnames:
                tfmap.remove_column(k)

    #- keep exposure-specific columns that are present in the input fibermap
    ii = np.isin(fibermap_perexp_cols, exp_fibermap.dtype.names)
    keepcols = tuple(np.array(fibermap_perexp_cols)[ii])
    exp_fibermap = exp_fibermap[keepcols]

    return tfmap, exp_fibermap

def _mask_cosmics(wave, flux, ivar, mask, subset, ivarjj_masked,
                  tid=None, cosmics_nsig=None):
    """
    Mask cosmics for in multiple spectra by updating ivar array inplace
    
    Args:
    wave (numpy.ndarray): 1d array of wavelengths
    flux (numpy.ndarray): 2d array of fluxes (from Spectra object)
    ivar (numpy.ndarray): 2d array of ivars (from Spectra)
    mask (numpy.ndarray or None): 2d array of masks (from Spectra) or None
    subset (numpy.ndarray): 1d int array of indices of individual spectra that we are coadding
    ivarjj_masked (numpy.ndarray): 2d array ivars that is being updated inplace
    tid (int): targetid (used for logging)
    cosmics_nsig (float): threshold for cosmic ray rejection
    """

    # interpolate over bad measurements
    # to be able to compute gradient next
    # to a bad pixel and identify outlier
    # many cosmics residuals are on edge
    # of cosmic ray trace, and so can be
    # next to a masked flux bin
    log = get_logger()
    grad = []
    gradvar = []
    for j in subset  :
        if mask is not None :
            ttivar0 = ivar[j] * (mask[j] == 0)
        else :
            ttivar0 = ivar[j]
        good = (ttivar0 > 0)
        bad  = ~good
        if np.sum(good) == 0 :
            continue
        nbad = np.sum(bad)
        ttflux = flux[j].copy()
        if nbad > 0 :
            ttflux[bad] = np.interp(wave[bad], wave[good], ttflux[good])
        ttivar = ivar[j].copy()
        if nbad > 0 :
            ttivar[bad] = np.interp(wave[bad], wave[good], ttivar[good])
        ttvar = 1./(ttivar + (ttivar == 0))
        ttflux[1:] = ttflux[1:] - ttflux[:-1]
        ttvar[1:] = ttvar[1:] + ttvar[:-1]
        ttflux[0] = 0
        grad.append(ttflux)
        gradvar.append(ttvar)
    grad, gradvar = np.array(grad), np.array(gradvar)
    gradivar = (gradvar > 0 ) /np.array(gradvar + (gradvar == 0))
    nspec = grad.shape[0]
    sgradivar = np.sum(gradivar)
    if sgradivar > 0 :
        meangrad = np.sum(gradivar * grad,axis=0) / sgradivar
        deltagrad = grad - meangrad
        chi2 = np.sum(gradivar * deltagrad**2, axis=0) / (nspec - 1)
        bad  = (chi2 > cosmics_nsig**2)
        nbad = np.sum(bad)
        if nbad > 0 :
            log.info("masking {} values for targetid={}".format(nbad, tid))
            badindex = np.where(bad)[0]
            for bi in badindex  :
                k = np.argmax(gradivar[:, bi] * deltagrad[:, bi]**2)
                ivarjj_masked[k, bi] = 0.
                log.debug("masking spec {} wave={}".format(k, wave[bi]))
    return 

def coadd(spectra, cosmics_nsig=None, onetile=False) :
    """
    Coadd spectra for each target and each camera, modifying input spectra obj.

    Args:
       spectra: desispec.spectra.Spectra object

    Options:
       cosmics_nsig: float, nsigma clipping threshold for cosmics rays (default 4)
       onetile: bool, if True, inputs are from a single tile

    Notes: if `onetile` is True, additional tile-specific columns
       like LOCATION and FIBER are included the FIBERMAP; otherwise
       these are only in the EXP_FIBERMAP since for the same target they could
       be different on different tiles.
    """
    log = get_logger()
    targets = ordered_unique(spectra.fibermap["TARGETID"])
    ntarget=targets.size
    log.debug("number of targets= {}".format(ntarget))

    #- Use "None" as default -> 4.0 so that scripts can use args.nsig
    #- with default None, which lets this function be the sole "owner" of
    #- the true numeric default
    if cosmics_nsig is None:
        cosmics_nsig = 4.0   # Note: if you change this, also change docstring

    if cosmics_nsig > 0:
        log.info(f'Clipping cosmics with {cosmics_nsig=}')
    else:
        log.info(f'Not performing cosmics sigma clipping ({cosmics_nsig=})')

    for b in spectra.bands :
        log.debug("coadding band '{}'".format(b))
        nwave=spectra.wave[b].size
        tflux=np.zeros((ntarget,nwave),dtype=spectra.flux[b].dtype)
        tivar=np.zeros((ntarget,nwave),dtype=spectra.ivar[b].dtype)
        if spectra.mask is not None :
            tmask=np.zeros((ntarget,nwave),dtype=spectra.mask[b].dtype)
        else :
            tmask=None
        trdata=np.zeros((ntarget,spectra.resolution_data[b].shape[1],nwave),dtype=spectra.resolution_data[b].dtype)

        if 'FIBERSTATUS' in spectra.fibermap.dtype.names:
            fiberstatus = spectra.fibermap['FIBERSTATUS']
        else:
            fiberstatus = spectra.fibermap['COADD_FIBERSTATUS']

        good_fiberstatus = use_for_coadd(fiberstatus, b)

        for i,tid in enumerate(targets) :
            jj=np.where( (spectra.fibermap["TARGETID"]==tid) & good_fiberstatus )[0]

            #- if all spectra were flagged as bad (FIBERSTATUS != 0), contine
            #- to next target, leaving tflux and tivar=0 for this target
            if len(jj) == 0:
                continue

            # here we keek original variance array
            # that will not be modified
            # and ivarjj_masked which will be modified by
            # cosmic rays check and mask>0 check 
            ivarjj_orig = spectra.ivar[b][jj].copy()
            if spectra.mask is not None :
                ivarjj_masked = spectra.ivar[b][jj] * (spectra.mask[b][jj] == 0)
            else :
                ivarjj_masked = ivarjj_orig.copy()
            
            if cosmics_nsig is not None and cosmics_nsig > 0 and len(jj)>2 :
                _mask_cosmics(spectra.wave[b], spectra.flux[b],
                                              spectra.ivar[b],
                                              spectra.mask[b] if spectra.mask is not None else None,
                                              jj, ivarjj_masked, cosmics_nsig=cosmics_nsig,
                              tid=tid)

            tivar[i] = np.sum(ivarjj_masked,axis=0)
            tflux[i] = np.sum(ivarjj_masked * spectra.flux[b][jj],axis=0)
            ww = spectra.resolution_data[b].shape[1]//2
            # resolution kernel width
            npix = spectra.resolution_data[b].shape[2]
            # indices of the corresponding variance point
            # that needs to be used for ivar weights
            res_indices = (np.arange(npix)[None, :]
                           + np.arange(-ww, ww+1)[:, None]) % npix
            res_whts = np.array([_[res_indices] for _ in ivarjj_masked])
            res_norm = res_whts.sum(axis=0)
            trdata[i, :, : ] = np.sum(res_whts *
                                    spectra.resolution_data[b][jj, : , : ],
                                               axis=0) / (res_norm + (res_norm == 0))
            bad=(tivar[i]==0)
            if np.sum(bad)>0 :
                tivar[i][bad] = np.sum(ivarjj_orig[:,bad], axis=0)
                # if all masked, keep original ivar
                tflux[i][bad] = np.sum(ivarjj_orig[:,bad] *
                                       spectra.flux[b][jj][:,bad], axis=0)
            ok=(tivar[i]>0)
            if np.sum(ok)>0 :
                tflux[i][ok] /= tivar[i][ok]
            if spectra.mask is not None :
                tmask[i]      = np.bitwise_and.reduce(spectra.mask[b][jj],axis=0)
        spectra.flux[b] = tflux
        spectra.ivar[b] = tivar
        if spectra.mask is not None :
            spectra.mask[b] = tmask
        spectra.resolution_data[b] = trdata

    if spectra.scores is not None:
        orig_scores = Table(spectra.scores.copy())
        orig_scores['TARGETID'] = spectra.fibermap['TARGETID']
    else:
        orig_scores = None

    spectra.fibermap, exp_fibermap = coadd_fibermap(spectra.fibermap, onetile=onetile)
    spectra.exp_fibermap = exp_fibermap
    spectra.scores=None
    compute_coadd_scores(spectra, orig_scores, update_coadd=True)


def coadd_cameras(spectra, cosmics_nsig=0., onetile=False) :
    """
    Return coadd across both exposures and cameras

    Args:
       spectra: desispec.spectra.Spectra object

    Options:
       cosmics_nsig: float, nsigma clipping threshold for cosmics rays
       onetile: bool, if True, inputs are from a single tile

    If `onetile` is True, additional tile-specific columns
    like LOCATION and FIBER are included the FIBERMAP; otherwise
    these are only in the EXP_FIBERMAP since for the same target they could
    be different on different tiles.

    Note: unlike `coadd`, this does not modify the input spectra object
    """

    #check_alignement_of_camera_wavelength(spectra)

    log = get_logger()

    # ordering
    mwave=[np.mean(spectra.wave[b]) for b in spectra.bands]
    sbands=np.array(spectra.bands)[np.argsort(mwave)] # bands sorted by inc. wavelength
    log.debug("wavelength sorted cameras= {}".format(sbands))

    # create wavelength array
    wave=None
    tolerance=0.0001 #A , tolerance
    for b in sbands :
        if wave is None :
            wave=spectra.wave[b]
        else :
            wave=np.append(wave,spectra.wave[b][spectra.wave[b]>wave[-1]+tolerance])
    nwave=wave.size

    # check alignment, caching band wavelength grid indices as we go
    windict = {}
    number_of_overlapping_cameras=np.zeros(nwave)
    for b in spectra.bands :
        imin = np.argmin(np.abs(spectra.wave[b][0] - wave))
        windices = np.arange(imin, imin+len(spectra.wave[b]), dtype=int)
        dwave = spectra.wave[b] - wave[windices]
        if np.any(np.abs(dwave) > tolerance):
            msg = "Input wavelength grids (band '{}') are not aligned. Use --lin-step or --log10-step to resample to a common grid.".format(b)
            log.error(msg)
            raise ValueError(msg)
        number_of_overlapping_cameras[windices] += 1
        windict[b] = windices

    # targets
    targets = ordered_unique(spectra.fibermap["TARGETID"])
    ntarget=targets.size
    log.debug("number of targets= {}".format(ntarget))


    # ndiag = max of all cameras
    ndiag=0
    if spectra.resolution_data is not None:
        for b in sbands :
            ndiag=max(ndiag,spectra.resolution_data[b].shape[1])
    log.debug("ndiag=%d", ndiag)

    b = sbands[0]
    flux=np.zeros((ntarget,nwave),dtype=spectra.flux[b].dtype)
    ivar=np.zeros((ntarget,nwave),dtype=spectra.ivar[b].dtype)
    if spectra.mask is not None :
        ivar_unmasked=np.zeros((ntarget,nwave),dtype=spectra.ivar[b].dtype)
        mask=np.zeros((ntarget,nwave),dtype=spectra.mask[b].dtype)
    else :
        ivar_unmasked=ivar.copy()
        mask=None

    if spectra.resolution_data is not None:
        rdata=np.zeros((ntarget,ndiag,nwave),dtype=spectra.resolution_data[b].dtype)
    else:
        rdata = None

    for b in spectra.bands :
        log.debug("coadding band '{}'".format(b))

        # indices
        windices = windict[b]

        if spectra.resolution_data is not None:
            band_ndiag = spectra.resolution_data[b].shape[1]
        else:
            band_ndiag = 0

        if 'FIBERSTATUS' in spectra.fibermap.dtype.names:
            fiberstatus = spectra.fibermap['FIBERSTATUS']
        else:
            fiberstatus = spectra.fibermap['COADD_FIBERSTATUS']

        good_fiberstatus = use_for_coadd(fiberstatus, b)

        for i,tid in enumerate(targets) :
            jj=np.where( (spectra.fibermap["TARGETID"]==tid) & good_fiberstatus )[0]

            #- if all spectra were flagged as bad (FIBERSTATUS != 0), contine
            #- to next target, leaving tflux and tivar=0 for this target
            if len(jj) == 0:
                continue

            ivar_unmasked[i,windices] += np.sum(spectra.ivar[b][jj],axis=0)

            if spectra.mask is not None :
                ivarjj=spectra.ivar[b][jj]*(spectra.mask[b][jj]==0)
            else :
                ivarjj=spectra.ivar[b][jj]
            if cosmics_nsig is not None and cosmics_nsig > 0 and len(jj)>2  :
                _mask_cosmics(spectra.wave[b], spectra.flux[b],
                                              spectra.ivar[b],
                                              spectra.mask[b] if spectra.mask is not None else None,
                                              jj, ivarjj, cosmics_nsig=cosmics_nsig,
                              tid=tid)

            ivar[i,windices] += np.sum(ivarjj,axis=0)
            flux[i,windices] += np.sum(ivarjj*spectra.flux[b][jj],axis=0)
            if spectra.resolution_data is not None:
                for r in range(band_ndiag) :
                    rdata[i,r+(ndiag-band_ndiag)//2,windices] += np.sum((spectra.ivar[b][jj]*spectra.resolution_data[b][jj,r]),axis=0)
            if spectra.mask is not None :
                # this deserves some attention ...

                tmpmask=np.bitwise_and.reduce(spectra.mask[b][jj],axis=0)

                # directly copy mask where no overlap
                jj=(number_of_overlapping_cameras[windices]==1)
                mask[i,windices[jj]] = tmpmask[jj]

                # 'and' in overlapping regions
                jj=(number_of_overlapping_cameras[windices]>1)
                mask[i,windices[jj]] = mask[i,windices[jj]] & tmpmask[jj]


    for i,tid in enumerate(targets) :
        ok=(ivar[i]>0)
        if np.sum(ok)>0 :
            flux[i][ok] /= ivar[i][ok]
        ok=(ivar_unmasked[i]>0)
        if np.sum(ok)>0 and rdata is not None:
            rdata[i][:,ok] /= ivar_unmasked[i][ok]

    if 'COADD_NUMEXP' in spectra.fibermap.colnames:
        fibermap = spectra.fibermap
        exp_fibermap = spectra.exp_fibermap
    else:
        fibermap, exp_fibermap = coadd_fibermap(spectra.fibermap, onetile=onetile)

    bands=""
    for b in sbands :
        bands+=b

    if spectra.mask is not None :
        dmask={bands:mask,}
    else :
        dmask=None

    if rdata is not None:
        rdata = {bands:rdata}

    res = Spectra(
            bands=[bands,],
            wave={bands:wave,},
            flux={bands:flux,},
            ivar={bands:ivar,},
            mask=dmask,
            resolution_data=rdata,
            fibermap=fibermap,
            exp_fibermap=exp_fibermap,
            meta=spectra.meta,
            extra=spectra.extra,
            scores=None)

    if spectra.scores is not None:
        orig_scores = spectra.scores.copy()
        orig_scores['TARGETID'] = spectra.fibermap['TARGETID']
    else:
        orig_scores = None

    compute_coadd_scores(res, orig_scores, update_coadd=True)

    return res

def get_resampling_matrix(global_grid,local_grid,sparse=False):
    """Build the rectangular matrix that linearly resamples from the global grid to a local grid.

    The local grid range must be contained within the global grid range.

    Args:
        global_grid(numpy.ndarray): Sorted array of n global grid wavelengths.
        local_grid(numpy.ndarray): Sorted array of m local grid wavelengths.

    Returns:
        numpy.ndarray: Array of (m,n) matrix elements that perform the linear resampling.
    """
    assert np.all(np.diff(global_grid) > 0),'Global grid is not strictly increasing.'
    assert np.all(np.diff(local_grid) > 0),'Local grid is not strictly increasing.'
    # Locate each local wavelength in the global grid.
    global_index = np.searchsorted(global_grid,local_grid)

    assert local_grid[0] >= global_grid[0],'Local grid extends below global grid.'
    assert local_grid[-1] <= global_grid[-1],'Local grid extends above global grid.'

    # Lookup the global-grid bracketing interval (xlo,xhi) for each local grid point.
    # Note that this gives xlo = global_grid[-1] if local_grid[0] == global_grid[0]
    # but this is fine since the coefficient of xlo will be zero.
    global_xhi = global_grid[global_index]
    global_xlo = global_grid[global_index-1]
    # Create the rectangular interpolation matrix to return.
    alpha = (local_grid - global_xlo)/(global_xhi - global_xlo)
    local_index = np.arange(len(local_grid),dtype=int)
    matrix = np.zeros((len(local_grid),len(global_grid)))
    matrix[local_index,global_index] = alpha
    matrix[local_index,global_index-1] = 1-alpha

    # turn into a sparse matrix
    return scipy.sparse.csc_matrix(matrix)



def decorrelate_divide_and_conquer(Cinv,Cinvf,wavebin,flux,ivar,rdata) :
    """Decorrelate an inverse covariance using the matrix square root.

    Implements the decorrelation part of the spectroperfectionism algorithm described in
    Bolton & Schlegel 2009 (BS) http://arxiv.org/abs/0911.2689.

    with the divide and conquer approach, i.e. per diagonal block of the matrix, with an
    overlapping 'skin' from one block to another.

    Args:
        Cinv: Square 2D array: input inverse covariance matrix
        Cinvf: 1D array: input
        wavebin: minimal size of wavelength bin in A, used to define the core and skin size
        flux: 1D array: output flux (has to be allocated)
        ivar: 1D array: output flux inverse variance (has to be allocated)
        rdata: 2D array: output resolution matrix per diagonal (has to be allocated)
    """

    chw=max(10,int(50/wavebin)) #core is 2*50+1 A
    skin=max(2,int(10/wavebin)) #skin is 10A
    nn=Cinv.shape[0]
    nstep=nn//(2*chw+1)+1
    Lmin=1e-15/np.mean(np.diag(Cinv)) # Lmin is scaled with Cinv values
    ndiag=rdata.shape[0]
    dd=np.arange(ndiag,dtype=int)-ndiag//2

    for c in range(chw,nn+(2*chw+1),(2*chw+1)) :
        b=max(0,c-chw-skin)
        e=min(nn,c+chw+skin+1)
        b1=max(0,c-chw)
        e1=min(nn,c+chw+1)
        bb=max(0,b1-b)
        ee=min(e-b,e1-b)
        if e<=b : continue
        L,X = scipy.linalg.eigh(Cinv[b:e,b:e],overwrite_a=False,turbo=True)
        nbad = np.count_nonzero(L < Lmin)
        if nbad > 0:
            #log.warning('zeroing {0:d} negative eigenvalue(s).'.format(nbad))
            L[L < Lmin] = Lmin
        Q = X.dot(np.diag(np.sqrt(L)).dot(X.T))
        s = np.sum(Q,axis=1)

        b1x=max(0,c-chw-3)
        e1x=min(nn,c+chw+1+3)

        tR = (Q/s[:,np.newaxis])
        tR_it = scipy.linalg.inv(tR.T)
        tivar = s**2

        flux[b1:e1] = (tR_it.dot(Cinvf[b:e])/tivar)[bb:ee]
        ivar[b1:e1] = (s[bb:ee])**2
        for j in range(b1,e1) :
            k=(dd>=-j)&(dd<nn-j)
            # k is the diagonal index
            # j is the wavelength index
            # it could be the transposed, I am following what it is specter.ex2d, L209
            rdata[k,j] = tR[j-b+dd[k],j-b]

def spectroperf_resample_spectrum_singleproc(spectra,target_index,wave,wavebin,resampling_matrix,ndiag,flux,ivar,rdata) :
    cinv = None
    for b in spectra.bands :
        twave=spectra.wave[b]
        jj=(twave>=wave[0])&(twave<=wave[-1])
        twave=twave[jj]
        tivar=spectra.ivar[b][target_index][jj]
        diag_ivar = scipy.sparse.dia_matrix((tivar,[0]),(twave.size,twave.size))
        RR = Resolution(spectra.resolution_data[b][target_index][:,jj]).dot(resampling_matrix[b])
        tcinv  = RR.T.dot(diag_ivar.dot(RR))
        tcinvf = RR.T.dot(tivar*spectra.flux[b][target_index][jj])
        if cinv is None :
            cinv  = tcinv
            cinvf = tcinvf
        else :
            cinv  += tcinv
            cinvf += tcinvf
    cinv = cinv.todense()
    decorrelate_divide_and_conquer(cinv,cinvf,wavebin,flux[target_index],ivar[target_index],rdata[target_index])

# for multiprocessing, with shared memory buffers
def spectroperf_resample_spectrum_multiproc(shm_in_wave,shm_in_flux,shm_in_ivar,shm_in_rdata,in_nwave,in_ndiag,in_bands,target_indices,wave,wavebin,resampling_matrix,ndiag,ntarget,shm_flux,shm_ivar,shm_rdata) :

    nwave = wave.size

    # manipulate shared memory as np arrays

    # input shared memory
    in_wave = list()
    in_flux = list()
    in_ivar  = list()
    in_rdata  = list()

    nbands = len(shm_in_wave)
    for b in range(nbands) :
        in_wave.append( np.array(shm_in_wave[b],copy=False).reshape(in_nwave[b]) )
        in_flux.append( np.array(shm_in_flux[b],copy=False).reshape((ntarget,in_nwave[b])) )
        in_ivar.append( np.array(shm_in_ivar[b],copy=False).reshape((ntarget,in_nwave[b])) )
        in_rdata.append( np.array(shm_in_rdata[b],copy=False).reshape((ntarget,in_ndiag[b],in_nwave[b])) )


    # output shared memory

    flux  = np.array(shm_flux,copy=False).reshape(ntarget,nwave)
    ivar  = np.array(shm_ivar,copy=False).reshape(ntarget,nwave)
    rdata = np.array(shm_rdata,copy=False).reshape(ntarget,ndiag,nwave)

    for target_index in target_indices :

        cinv = None
        for b in range(nbands) :
            twave=in_wave[b]
            jj=(twave>=wave[0])&(twave<=wave[-1])
            twave=twave[jj]
            tivar=in_ivar[b][target_index][jj]
            diag_ivar = scipy.sparse.dia_matrix((tivar,[0]),(twave.size,twave.size))
            RR = Resolution(in_rdata[b][target_index][:,jj]).dot(resampling_matrix[in_bands[b]])
            tcinv  = RR.T.dot(diag_ivar.dot(RR))
            tcinvf = RR.T.dot(tivar*in_flux[b][target_index][jj])
            if cinv is None :
                cinv  = tcinv
                cinvf = tcinvf
            else :
                cinv  += tcinv
                cinvf += tcinvf
        cinv = cinv.todense()
        decorrelate_divide_and_conquer(cinv,cinvf,wavebin,flux[target_index],ivar[target_index],rdata[target_index])


def spectroperf_resample_spectra(spectra, wave, nproc=1) :
    """
    Resampling of spectra file using the spectrophotometic approach

    Args:
       spectra: desispec.spectra.Spectra object
       wave: 1D numy array with new wavelenght grid

    Returns:
       desispec.spectra.Spectra object
    """

    log = get_logger()
    log.debug("resampling to wave grid of size {}: {}".format(wave.size,wave))

    b=spectra.bands[0]
    ntarget=spectra.flux[b].shape[0]
    nwave=wave.size

    if spectra.mask is not None :
        mask = np.zeros((ntarget,nwave),dtype=spectra.mask[b].dtype)
    else :
        mask = None
    # number of diagonals is the max of the number of diagonals in the
    # input spectra cameras
    ndiag = 0
    for b in spectra.bands :
        ndiag = max(ndiag,spectra.resolution_data[b].shape[1])


    dw=np.gradient(wave)
    wavebin=np.min(dw[dw>0.]) # min wavelength bin size
    log.debug("min wavelength bin= {:2.1f} A; ndiag= {:d}".format(wavebin,ndiag))
    log.debug("compute resampling matrices")
    resampling_matrix=dict()
    for b in spectra.bands :
        twave=spectra.wave[b]
        jj=np.where((twave>=wave[0])&(twave<=wave[-1]))[0]
        twave=spectra.wave[b][jj]
        resampling_matrix[b] = get_resampling_matrix(wave,twave)


    if nproc==1 :

        # allocate array
        flux  = np.zeros((ntarget,nwave),dtype=float)
        ivar  = np.zeros((ntarget,nwave),dtype=float)
        rdata = np.zeros((ntarget,ndiag,nwave),dtype=float)

        # simply loop on targets
        for target_index in range(ntarget) :
            log.debug("resampling {}/{}".format(target_index+1,ntarget))
            t0=time.time()
            spectroperf_resample_spectrum_singleproc(spectra,target_index,wave,wavebin,resampling_matrix,ndiag,flux,ivar,rdata)
            t1=time.time()
            log.debug("done one spectrum in {} sec".format(t1-t0))
    else :

        log.debug("allocate shared memory")

        # input
        shm_in_wave = list()
        shm_in_flux = list()
        shm_in_ivar  = list()
        shm_in_rdata  = list()
        in_nwave = list()
        in_ndiag = list()
        for b in spectra.bands :
            shm_in_wave.append( multiprocessing.Array('d',spectra.wave[b],lock=False) )
            shm_in_flux.append( multiprocessing.Array('d',spectra.flux[b].ravel(),lock=False) )
            shm_in_ivar.append( multiprocessing.Array('d',spectra.ivar[b].ravel(),lock=False) )
            shm_in_rdata.append( multiprocessing.Array('d',spectra.resolution_data[b].ravel(),lock=False) )
            in_nwave.append(spectra.wave[b].size)
            in_ndiag.append(spectra.resolution_data[b].shape[1])

        # output
        shm_flux=multiprocessing.Array('d',ntarget*nwave,lock=False)
        shm_ivar=multiprocessing.Array('d',ntarget*nwave,lock=False)
        shm_rdata=multiprocessing.Array('d',ntarget*ndiag*nwave,lock=False)

        # manipulate shared memory as np arrays
        flux  = np.array(shm_flux,copy=False).reshape(ntarget,nwave)
        ivar  = np.array(shm_ivar,copy=False).reshape(ntarget,nwave)
        rdata = np.array(shm_rdata,copy=False).reshape(ntarget,ndiag,nwave)

        # split targets per process
        target_indices = np.array_split(np.arange(ntarget),nproc)

        # loop on processes
        procs=list()
        for proc_index in range(nproc) :
            log.debug("starting process #{}".format(proc_index+1))
            proc = multiprocessing.Process(target=spectroperf_resample_spectrum_multiproc,
                                           args=(shm_in_wave,shm_in_flux,shm_in_ivar,shm_in_rdata,
                                                 in_nwave,in_ndiag,spectra.bands,
                                                 target_indices[proc_index],wave,wavebin,
                                                 resampling_matrix,ndiag,ntarget,
                                                 shm_flux,shm_ivar,shm_rdata))
            proc.start()
            procs.append(proc)

        # wait for the processes to finish
        log.info("waiting for the {} processes to finish ...".format(nproc))
        for proc in procs :
            proc.join()
        log.info("all done!")

    bands=""
    for b in spectra.bands : bands += b

    if spectra.mask is not None :
        dmask={bands:mask,}
    else :
        dmask=None
    res=Spectra(bands=[bands,],wave={bands:wave,},flux={bands:flux,},ivar={bands:ivar,},mask=dmask,resolution_data={bands:rdata,},
                fibermap=spectra.fibermap,meta=spectra.meta,extra=spectra.extra,scores=spectra.scores)
    return res


def fast_resample_spectra(spectra, wave) :
    """
    Fast resampling of spectra file.
    The output resolution = Id. The neighboring
    flux bins are correlated.

    Args:
       spectra: desispec.spectra.Spectra object
       wave: 1D numy array with new wavelenght grid

    Returns:
       desispec.spectra.Spectra object, resolution data=Id
    """

    log = get_logger()
    log.debug("Resampling to wave grid: {}".format(wave))


    nwave=wave.size
    b=spectra.bands[0]
    ntarget=spectra.flux[b].shape[0]
    nres=spectra.resolution_data[b].shape[1]
    ivar=np.zeros((ntarget,nwave),dtype=spectra.flux[b].dtype)
    flux=np.zeros((ntarget,nwave),dtype=spectra.ivar[b].dtype)
    if spectra.mask is not None :
        mask = np.zeros((ntarget,nwave),dtype=spectra.mask[b].dtype)
    else :
        mask = None
    rdata=np.ones((ntarget,1,nwave),dtype=spectra.resolution_data[b].dtype) # pointless for this resampling
    bands=""
    for b in spectra.bands :
        if spectra.mask is not None :
            tivar=spectra.ivar[b]*(spectra.mask[b]==0)
        else :
            tivar=spectra.ivar[b]
        for i in range(ntarget) :
            ivar[i]  += resample_flux(wave,spectra.wave[b],tivar[i])
            flux[i]  += resample_flux(wave,spectra.wave[b],tivar[i]*spectra.flux[b][i])
        bands += b
    for i in range(ntarget) :
        ok=(ivar[i]>0)
        flux[i,ok]/=ivar[i,ok]
    if spectra.mask is not None :
        dmask={bands:mask,}
    else :
        dmask=None
    res=Spectra(bands=[bands,],wave={bands:wave,},flux={bands:flux,},ivar={bands:ivar,},mask=dmask,resolution_data={bands:rdata,},
                fibermap=spectra.fibermap,meta=spectra.meta,extra=spectra.extra,scores=spectra.scores)
    return res

def resample_spectra_lin_or_log(spectra, linear_step=0, log10_step=0, fast=False, wave_min=None, wave_max=None, nproc=1) :
    """
    Resampling of spectra file.


    Args:
       spectra: desispec.spectra.Spectra object
       linear_step: if not null the ouput wavelenght grid will be linear with this step
       log10_step: if not null the ouput wavelenght grid will be logarthmic with this step

    Options:
       fast: simple resampling. fast but at the price of correlated output flux bins and no information on resolution
       wave_min: if set, use this min wavelength
       wave_max: if set, use this max wavelength

    Returns:
       desispec.spectra.Spectra object
    """

    wmin=None
    wmax=None
    for b in spectra.bands :
        if wmin is None :
            wmin=spectra.wave[b][0]
            wmax=spectra.wave[b][-1]
        else :
            wmin=min(wmin,spectra.wave[b][0])
            wmax=max(wmax,spectra.wave[b][-1])

    if wave_min is not None :
        wmin = wave_min
    if wave_max is not None :
        wmax = wave_max

    if linear_step>0 :
        nsteps=int((wmax-wmin)/linear_step) + 1
        wave=wmin+np.arange(nsteps)*linear_step
    elif log10_step>0 :
        lwmin=np.log10(wmin)
        lwmax=np.log10(wmax)
        nsteps=int((lwmax-lwmin)/log10_step) + 1
        wave=10**(lwmin+np.arange(nsteps)*log10_step)
    if fast :
        return fast_resample_spectra(spectra=spectra,wave=wave)
    else :
        return spectroperf_resample_spectra(spectra=spectra,wave=wave,nproc=nproc)
