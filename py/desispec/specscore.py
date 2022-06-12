"""
desispec.specscore
==================

Spectral scores routines.
"""
from __future__ import absolute_import
import numpy as np
import astropy.table
from desispec.frame import Frame
from desiutil.log import get_logger

from desispec.util import ordered_unique

# Definition of hard-coded top hat filter sets for each camera B,R,Z.
# The top hat filters have to be fully contained in the sensitive region of each camera.
tophat_wave={"b":[4000,5800],"r":[5800,7600],"z":[7600,9800]}

def _auto_detect_camera(frame) :
    mwave=np.mean(frame.wave)
    if mwave<=tophat_wave["b"][1] : return "b"
    elif mwave>=tophat_wave["z"][0] : return "z"
    else : return "r"

def compute_coadd_scores(coadd, specscores=None, update_coadd=True):
    """Compute scores for a coadded Spectra object

    Args:
        coadd: a Spectra object from a coadd

    Options:
        update_coadd: if True, update coadd.scores
        specscores: scores Table from the uncoadded spectra including a TARGETID column

    Returns tuple of dictionaries (scores, comments); see compute_frame_scores

    ``specscores`` is used to update TSNR2 scores by summing inputs
    """
    log = get_logger()
    scores = dict()
    comments = dict()

    scores['TARGETID'] = coadd.target_ids()
    comments['TARGETID'] = 'DESI Unique Target ID'

    #- if coadd.fibermap doesn't have FIBER, create dummy for Frame
    if 'FIBER' in coadd.fibermap.dtype.names:
        fibers = coadd.fibermap['FIBER']
    else:
        fibers = -np.arange(len(coadd.fibermap))

    if coadd.bands == ['brz']:
        #- i.e. this is a coadd across cameras
        fr = Frame(coadd.wave['brz'], coadd.flux['brz'], coadd.ivar['brz'],
                    fibermap=coadd.fibermap, fibers=fibers, meta=coadd.meta,
                    resolution_data=coadd.resolution_data['brz'])
        for band in ['b', 'r', 'z']:
            bandscores, bandcomments = compute_frame_scores(fr, band=band,
                    suffix='COADD', flux_per_angstrom=True)
            scores.update(bandscores)
            comments.update(bandcomments)
    else:
        #- otherwise try individual bands, upper or lowercase
        for band in ['b', 'r', 'z', 'B', 'R', 'Z']:
            if band in coadd.bands:
                fr = Frame(coadd.wave[band], coadd.flux[band], coadd.ivar[band],
                        fibermap=coadd.fibermap, fibers=fibers, meta=coadd.meta,
                        resolution_data=coadd.resolution_data[band])
                bandscores, bandcomments = compute_frame_scores(fr, band=band,
                        suffix='COADD', flux_per_angstrom=True)
                scores.update(bandscores)
                comments.update(bandcomments)

    if specscores is not None:
        tsnrscores, tsnrcomments = compute_coadd_tsnr_scores(specscores)
        scores.update(tsnrscores)
        comments.update(tsnrcomments)

    #- convert to float32
    for col in scores.keys():
        if scores[col].dtype == np.float64:
            scores[col] = scores[col].astype(np.float32)

    if update_coadd:
        if hasattr(coadd, 'scores') and coadd.scores is not None:
            for key in scores:
                coadd.scores[key] = scores[key]
                coadd.scores_comments[key] = comments[key]
        else:
            coadd.scores = scores
            coadd.scores_comments = comments

    return scores, comments

def compute_coadd_tsnr_scores(specscores):
    """
    Compute coadded TSNR2 scores (TSNR2=Template Signal-to-Noise squared)

    Args:
        specscores : uncoadded scores with TSNR2* columns (dict or Table-like)

    Returns (tsnrscores, comments) tuple of dictionaries
    """
    log = get_logger()

    targetids = ordered_unique(specscores['TARGETID'])
    num_targets = len(targetids)

    tsnrscores = dict()
    comments = dict()
    tsnrscores['TARGETID'] = targetids

    #- Derive which TSNR2_XYZ_[BRZ] columns exist
    tsnrkeys = list()
    tsnrtypes = list()

    if isinstance(specscores, dict):
        _colnames = specscores.keys()
    else:
        _colnames = specscores.dtype.names

    for colname in _colnames:
        if colname.startswith('TSNR2_'):
            parts = colname.split('_')

            # Ignore brz coadded values as handled independently by adding b,r,z.
            if parts[-1] in ['b', 'r', 'z', 'B', 'R', 'Z']:
                _, targtype, band = parts

                tsnrscores[colname] = np.zeros(num_targets, dtype=np.float32)
                comments[colname] = f'{targtype} {band} template (S/N)^2'
                tsnrkeys.append(colname)

                if targtype not in tsnrtypes:
                    tsnrtypes.append(targtype)

    if len(tsnrkeys) == 0:
        log.warning('No TSNR2_* scores found to coadd')
    else:
        #- Add TSNR2_*_B/R/Z columns summed across exposures
        for i, tid in enumerate(targetids):
            jj = specscores['TARGETID'] == tid
            for colname in tsnrkeys:
                tsnrscores[colname][i] = np.sum(specscores[colname][jj])

        #- Additionally sum across B/R/Z
        for targtype in tsnrtypes:
            col = f'TSNR2_{targtype}'
            tsnrscores[col] = np.zeros(num_targets, dtype=np.float32)
            comments[col] = f'{targtype} template (S/N)^2 summed over B,R,Z'
            for band in ['B', 'R', 'Z']:
                colbrz = f'TSNR2_{targtype}_{band}'

                #- Missing cameras can result in missing columns, which
                #- should be treated as SNR=0 but not crash
                if colbrz in tsnrscores.keys():
                    tsnrscores[col] += tsnrscores[colbrz]

    return tsnrscores, comments


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
    frame : :class:`~desispec.frame.Frame` or :class:`~desispec.frame.QFrame`
        A Frame or a QFrame object.
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
            
    is_a_frame = (len(frame.wave.shape)==1)
    
    mask=(frame.wave>=tophat_wave[band][0])*(frame.wave<tophat_wave[band][1])
    
    if np.sum(mask)==0 :
        message="no intersection of frame wavelenght and tophat range {}".format(tophat_wave[band])
        log.error(message)
        raise ValueError(message)
    
    scores = dict()
    comments = dict()
    ivar = frame.ivar
    ivar[ivar<0] *= 0. # make sure it's not negative  
    if is_a_frame : 
        dwave = np.gradient(frame.wave)
    else : # a qframe
        dwave = np.gradient(frame.wave,axis=1)

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
        
    nspec=frame.flux.shape[0]
    if flux_per_angstrom :
        # we need to integrate the flux accounting for the wavelength bin
        k="INTEG%sFLUX_%s"%(suffix,band.upper())
        if is_a_frame :
            scores[k] = np.sum(frame.flux[:,mask]*dwave[mask],axis=1)
        else :
            scores[k] = np.array([np.sum(frame.flux[i,mask[i]]*dwave[i,mask[i]]) for i in range(nspec)])
        comments[k]     = "integ. flux in wave. range {},{}A".format(tophat_wave[band][0],tophat_wave[band][1])
        # simple median
        k="MEDIAN%sFLUX_%s"%(suffix,band.upper())
        if is_a_frame :
            scores[k] = np.median(frame.flux[:,mask],axis=1) # already per angstrom
        else :
            scores[k] = np.array([np.median(frame.flux[i,mask[i]]) for i in range(nspec)])
        comments[k]     = "median flux in wave. range {},{}A".format(tophat_wave[band][0],tophat_wave[band][1])        
    else :
        # simple sum of counts
        k="SUM%sCOUNT_%s"%(suffix,band.upper())
        if is_a_frame :
            scores[k]       = np.sum(frame.flux[:,mask],axis=1)
        else :
            scores[k] = np.array([np.sum(frame.flux[i,mask[i]]) for i in range(nspec)])
        comments[k]     = "sum counts in wave. range {},{}A".format(tophat_wave[band][0],tophat_wave[band][1])
        # median count per A
        k="MEDIAN%sCOUNT_%s"%(suffix,band.upper())
        if is_a_frame :
            scores[k] = np.median(frame.flux[:,mask]/dwave[mask],axis=1) # per angstrom
        else :
            scores[k] = np.array([np.median(frame.flux[i,mask[i]]/dwave[i,mask[i]]) for i in range(nspec)])
        comments[k]     = "median counts/A in wave. range {},{}A".format(tophat_wave[band][0],tophat_wave[band][1])

    # the signal to noise scales with sqrt(integration wavelength range) (same for uncalibrated or calibrated data)
    k="MEDIAN%sSNR_%s"%(suffix,band.upper())
    if is_a_frame :
        scores[k]    = np.median((np.sqrt(ivar[:,mask])*frame.flux[:,mask]/np.sqrt(dwave[mask])),axis=1)
    else :
        scores[k] = np.array([np.median(np.sqrt(ivar[i,mask[i]])*frame.flux[i,mask[i]]/np.sqrt(dwave[i,mask[i]])) for i in range(nspec)])
    comments[k]  = "median SNR/sqrt(A) in wave. range {},{}A".format(tophat_wave[band][0],tophat_wave[band][1])

    return scores,comments

def append_frame_scores(frame,new_scores,new_comments,overwrite) :

    log = get_logger()
    
    if frame.scores is not None :
        
        scores = dict()
        comments = dict()

        # frame.scores can be a 
        frame_scores_keys = None
        if isinstance(frame.scores, (np.ndarray, astropy.table.Table, np.recarray)) :
            frame_scores_keys = frame.scores.dtype.names
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

