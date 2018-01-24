"""
desispec.specscore
========================

Spectral scores routines.
"""
from __future__ import absolute_import
import numpy as np
from desiutil.log import get_logger
from desispec.io.util import write_bintable #,add_columns

# Definition of hard-coded top hat filter sets for each camera B,R,Z.
# The top hat filters have to be fully contained in the sensitive region of each camera.
tophat_wave={"b":[4000,5800],"r":[5800,7600],"z":[7600,9800]}

def auto_detect_camera(frame) :
    mwave=np.mean(frame.wave)
    if mwave<=tophat_wave["b"][1] : return "b"
    elif mwave>=tophat_wave["z"][0] : return "z"
    else : return "r"

def compute_frame_scores(frame,band=None,suffix=None,calibrated=False) :
    
    log=get_logger()
    if band is not None :
        if not band.lower() in tophat_wave.keys() :
            message="'{}' is not an allowed camera arm (has to be in {}, upper orlower case)".format(band,tophat_wave.keys())
            log.error(message)
            raise KeyError(message)
    else :
        band = auto_detect_camera(frame)
            
    ii=(frame.wave>=tophat_wave[band][0])&(frame.wave<tophat_wave[band][1])
    if np.sum(ii)==0 :
        message="no intersection of frame wavelenght and tophat range {}".format(tophat_wave[band])
        log.Error(message)
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
    
    if calibrated :
        # we need to integrate the flux accounting for the wavelength bin
        k="INTEG%sFLUX_%s"%(suffix,band.upper())
        scores[k]       = np.sum(frame.flux[:,ii]*dwave[ii],axis=1)
        comments[k]     = "integ. flux in wave. range {},{}A".format(tophat_wave[band][0],tophat_wave[band][1])
        # simple median
        k="MEDIAN%sFLUX_%s"%(suffix,band.upper())
        scores[k]       = np.median(frame.flux[:,ii]/dwave[ii],axis=1) # per angstrom
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

def append_scores_and_write_frame(frame,filename,new_scores,new_comments,overwrite) :

    log = get_logger()
    
    if frame.scores is not None :
        
        scores = dict()
        comments = dict()
        for k in frame.scores.columns.names :
            scores[k]=frame.scores[k]
            comments[k]="" 
        
        if frame.scores_comments is not None :
            for k in comments.keys() :
                comments[k]= frame.scores_comments[k]
        
        log.info("Appending the scores that were already present in {}".format(filename))
        for k,v in comments.items() :
            log.debug("Already present: {}, {}".format(k,v))

        for k in new_scores.keys() :
            if k in scores.keys() and not overwrite :
                log.warning("do not overwrite score {} in {}".format(k,filename))
            else :
                scores[k]   = new_scores[k]
                comments[k] = new_comments[k]

    else :
        scores   = new_scores
        comments = new_comments
    
    log.info("Adding or replacing SCORES extention with {} in {}".format(scores.keys(),filename))
    write_bintable(filename,data=scores,comments=comments,extname="SCORES",clobber=True)
    
    
    
