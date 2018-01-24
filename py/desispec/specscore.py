"""
desispec.specscore
========================

Spectral scores routines.
"""
from __future__ import absolute_import
import numpy as np
from desiutil.log import get_logger
from desispec.io.util import write_bintable,add_columns

# Definition of hard-coded top hat filter sets for each camera B,R,Z.
# The top hat filters have to be fully contained in the sensitive region of each camera.
tophat_wave={"b":[4000,5800],"r":[5800,7600],"z":[7600,9800]}

def auto_detect_camera(frame) :
    mwave=np.mean(frame.wave)
    if mwave<=tophat_wave["b"][1] : return "b"
    elif mwave>=tophat_wave["z"][0] : return "z"
    else : return "r"

def compute_frame_scores(frame,camera_arm=None) :

    log=get_logger()
    if camera_arm is not None :
        if not camera_arm.lower() in tophat_wave.keys() :
            message="'{}' is not an allowed camera arm (has to be in {}, upper orlower case)".format(camera_arm,tophat_wave.keys())
            log.error(message)
            raise KeyError(message)
    else :
        camera_arm = auto_detect_camera(frame)
            
    ii=(frame.wave>=tophat_wave[camera_arm][0])&(frame.wave<tophat_wave[camera_arm][1])
    if np.sum(ii)==0 :
        message="no intersection of frame wavelenght and tophat range {}".format(tophat_wave[camera_arm])
        log.Error(message)
        raise ValueError(message)
    
    scores = dict()
    comments = dict()
    ivar = frame.ivar
    ivar[ivar<0] *= 0. # make sure it's not negative    
    dwave = np.gradient(frame.wave)

    # there is limited space for comments in fits ...
    scores["SUM_COUNTS"]       = np.sum(frame.flux[:,ii],axis=1)
    comments["SUM_COUNTS"]     = "sum counts in wave. range {},{}A".format(tophat_wave[camera_arm][0],tophat_wave[camera_arm][1])
    
    scores["MEDIAN_COUNTS"]    = np.median(frame.flux[:,ii]/dwave[ii],axis=1) # per angstrom
    comments["MEDIAN_COUNTS"]  = "median counts/A in wave. range {},{}A".format(tophat_wave[camera_arm][0],tophat_wave[camera_arm][1])

    # the signal to noise scales with sqrt(integration wavelength range)
    scores["MEDIAN_SNR"]       = np.median(np.sqrt(ivar[:,ii])*frame.flux[:,ii]/np.sqrt(dwave[ii]),axis=1)
    comments["MEDIAN_SNR"]     = "median SNR/sqrt(A) in wave. range {},{}A".format(tophat_wave[camera_arm][0],tophat_wave[camera_arm][1])
    return scores,comments


def append_scores_and_write_frame(frame,filename,new_scores,new_comments,overwrite) :

    log = get_logger()
    
    if frame.scores is not None :
        scores = dict()
        for k in frame.scores.columns.names :
            scores[k]=frame.scores[k]

        if frame.scores_comments is not None :
            comments = frame.scores_comments
        else :
            comments = dict()
            for k in scores.keys() :
                comments[k]=""

        log.info("Appending the scores that were already present in {}".format(filename))
        for k,v in comments.items() :
            log.debug("Already present: {}, {}".format(k,v))

        for k,v in new_scores.items() :
            if k in scores.keys() :
                if overwrite :
                    log.debug("overwriting score {} in {}".format(k,filename))
                    scores[k]=v
                else :
                    log.warning("do not overwrite score {} in {}".format(k,filename))
            else :
                scores = add_columns(scores,k,v)
                comments[k] = new_comments[k]
    else :
        scores   = new_scores
        comments = new_comments

    # who do we know which camera arm it is?
    log.info("Adding or replacing SCORES extention with {} in {}".format(scores.keys(),filename))
    write_bintable(filename,data=scores,comments=comments,extname="SCORES",clobber=True)
    
    
    
