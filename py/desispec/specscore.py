"""
desispec.specscore
==================

Spectral scores routines.
"""
from __future__ import absolute_import
import numpy as np
from desiutil.log import get_logger

# Definition of hard-coded top hat filter sets for each camera B,R,Z.
# The top hat filters have to be fully contained in the sensitive region of each camera.
tophat_wave={"b":[4000,5800],"r":[5800,7600],"z":[7600,9800]}

def _auto_detect_camera(frame) :
    mwave=np.mean(frame.wave)
    if mwave<=tophat_wave["b"][1] : return "b"
    elif mwave>=tophat_wave["z"][0] : return "z"
    else : return "r"

def compute_frame_scores(frame,band=None,suffix=None,flux_per_angstrom=None) :
    """Computes scores in spectra of a frame.

    The scores are sum,mean,medians in a predefined and fixed wavelength range
    for each DESI camera arm, or band, b, r or z.
    The band argument is optional because it can be automatically chosen
    from the wavelength range in the frame.
    The suffix is added to the key name in the output dictionnary, for
    instance 'RAW', 'SKYSUB', 'CALIB' ...
    The boolean argument flux_per_angstrom is needed if there is no
    'BUNIT' keyword in frame.meta (frame fits header)
    
    Parameters
    ----------
    frame : :class:`~desispec.frame.Frame`
        A Frame object.
    band : :class:`str`, optional
        Spectrograph band, ``b``, ``r``, ``z``, autodetected by default.
    suffix : :class:`str`, optional
        Character string added to the keywords in the output dictionary,
        for instance suffix='RAW'
    flux_per_angstrom : :class:`bool`, optional
        If ``True`` the spectra are assumed flux_per_angstrom, *i.e.* flux
        densities. If ``False``, the spectra are assumed to be counts or
        photo-electrons per bin. ``None`` by default in which case the
        ``frame.units`` string is read to find out whether the flux quantity is
        per unit wavelenght or per bin.

    Returns
    -------
    :func:`tuple`
        A tuple containg a :class:`dict` of 1D arrays of size = number of 
        spectra in frame and a :class:`dict` of string with comments
        on the type of scores.
    """
    log=get_logger()
    if band is not None :
        if not band.lower() in tophat_wave.keys() :
            message="'{}' is not an allowed camera arm (has to be in {}, upper orlower case)".format(band,tophat_wave.keys())
            log.error(message)
            raise KeyError(message)
    else :
        band = _auto_detect_camera(frame)
            
    ii=(frame.wave>=tophat_wave[band][0])&(frame.wave<tophat_wave[band][1])
    if np.sum(ii)==0 :
        message="no intersection of frame wavelenght and tophat range {}".format(tophat_wave[band])
        log.error(message)
        raise ValueError(message)
    
    scores = dict()
    comments = dict()
    ivar = frame.ivar
    ivar[ivar<0] *= 0. # make sure it's not negative    
    dwave = np.gradient(frame.wave)

    if suffix is None :
        suffix="_"
    else :
        suffix="_%s_"%suffix

    if flux_per_angstrom is None :

        units=None
        if frame.meta is not None :
            if "BUNIT" in frame.meta :
                units=frame.meta["BUNIT"]
        if units is None :
            log.error("Cannot interpret the flux units because no BUNIT information in frame.meta, and the flux_per_angstrom argument is None. Returning empty dicts.")
            # return empty dicts
            scores=dict()
            comments=dict()
            return scores,comments

        denominator=units.strip().split("/")[-1]
        if denominator.find("A")>=0 :
            flux_per_angstrom=True
        elif denominator.find("bin")>=0 :
            flux_per_angstrom=False
        else :
            log.error("Cannot understand in the flux unit '%s' whether it is per Angstrom or per bin. Returning empty dicts.")
            # return empty dicts
            scores=dict()
            comments=dict()
            return scores,comments
        
        
    if flux_per_angstrom :
        # we need to integrate the flux accounting for the wavelength bin
        k="INTEG%sFLUX_%s"%(suffix,band.upper())
        scores[k]       = np.sum(frame.flux[:,ii]*dwave[ii],axis=1)
        comments[k]     = "integ. flux in wave. range {},{}A".format(tophat_wave[band][0],tophat_wave[band][1])
        # simple median
        k="MEDIAN%sFLUX_%s"%(suffix,band.upper())
        scores[k]       = np.median(frame.flux[:,ii],axis=1) # already per angstrom
        comments[k]     = "median flux in wave. range {},{}A".format(tophat_wave[band][0],tophat_wave[band][1])        
    else :
        # simple sum of counts
        k="SUM%sCOUNT_%s"%(suffix,band.upper())
        scores[k]       = np.sum(frame.flux[:,ii],axis=1)
        comments[k]     = "sum counts in wave. range {},{}A".format(tophat_wave[band][0],tophat_wave[band][1])
        # median count per A
        k="MEDIAN%sCOUNT_%s"%(suffix,band.upper())
        scores[k]       = np.median(frame.flux[:,ii]/dwave[ii],axis=1) # per angstrom
        comments[k]     = "median counts/A in wave. range {},{}A".format(tophat_wave[band][0],tophat_wave[band][1])
        
    # the signal to noise scales with sqrt(integration wavelength range) (same for uncalibrated or calibrated data)
    k="MEDIAN%sSNR_%s"%(suffix,band.upper())
    scores[k]    = np.median(np.sqrt(ivar[:,ii])*frame.flux[:,ii]/np.sqrt(dwave[ii]),axis=1)
    comments[k]  = "median SNR/sqrt(A) in wave. range {},{}A".format(tophat_wave[band][0],tophat_wave[band][1])
    return scores,comments

def append_frame_scores(frame,new_scores,new_comments,overwrite) :

    log = get_logger()
    
    if frame.scores is not None :
        
        scores = dict()
        comments = dict()

        # frame.scores can be a 
        frame_scores_keys = None
        if isinstance(frame.scores, np.recarray) :
            frame_scores_keys = frame.scores.columns.names
        elif isinstance(frame.scores, dict) :
            frame_scores_keys = frame.scores.keys()
        else :
            log.error("I don't know how to handle the frame.scores class '%s'"%frame.scores.__class__)
            raise ValueError("I don't know how to handle the frame.scores class '%s'"%frame.scores.__class__)
        
        for k in frame_scores_keys :
            scores[k]=frame.scores[k]
            comments[k]="" 
        
        if frame.scores_comments is not None :
            for k in comments.keys() :
                comments[k]= frame.scores_comments[k]
        
        for k in new_scores.keys() :
            if k in scores.keys() and not overwrite :
                log.warning("do not overwrite score {}".format(k))
            else :
                scores[k]   = new_scores[k]
                comments[k] = new_comments[k]

    else :
        scores   = new_scores
        comments = new_comments
        
    frame.scores = scores
    frame.scores_comments = comments

    return scores,comments

    
def compute_and_append_frame_scores(frame,band=None,suffix=None,flux_per_angstrom=False,overwrite=True) :
    new_scores,new_comments = compute_frame_scores(frame,band=band,suffix=suffix,flux_per_angstrom=flux_per_angstrom)
    return append_frame_scores(frame,new_scores,new_comments,overwrite=overwrite)

