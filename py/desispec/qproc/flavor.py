"""
desispec.qproc.flavor
=====================

Please add module-level documentation.
"""
import numpy as np

from desiutil.log import get_logger

# tool to check the flavor of a qframe

def check_qframe_flavor(qframe,input_flavor=None):
    """
    Tool to check the flavor of a qframe

    Args:
      qframe : DESI QFrame object

    Optional:
         input_flavor

    Return:
         flavor string
    """
    log = get_logger()

    log.debug("Checking qframe flavor...")

    # resample
    mwave=np.mean(qframe.wave,axis=0)
    rflux=np.zeros(qframe.flux.shape)
    for i in range(rflux.shape[0]) :
        jj=(qframe.ivar[i]>0)
        if np.sum(jj)>0 :
            rflux[i]=np.interp(mwave,qframe.wave[i,jj],qframe.flux[i,jj],left=0,right=0)

    # median of resampled spectra
    median_spec=np.median(rflux,axis=0)

    # final scores
    median_of_median_spec = np.median(median_spec)
    max_of_median_spec    = np.max(median_spec)
    log.info("Median of median spectrum = {}".format(median_of_median_spec))
    log.info("Max    of median spectrum = {}".format(max_of_median_spec))

    # a very crude guess
    if median_of_median_spec > 1000 :
        #- Very bright median = FLAT
        guessed_flavor = "FLAT"
    elif median_of_median_spec < 100 and max_of_median_spec > 1000 :
        #- Peaks are much bigger than continuum = ARC
        guessed_flavor = "ARC"
    elif max_of_median_spec < 100 :
        if qframe.meta['EXPTIME'] > 0:
            #- non-zero exposure time but no detected signal = DARK
            guessed_flavor = "DARK"
        else:
            #- EXPTIME=0 sure sounds like a ZERO...
            guessed_flavor = "ZERO"
    else :
        guessed_flavor = "SKY"


    if input_flavor is not None and ( input_flavor.upper() == "ZERO" or input_flavor.upper() == "DARK" ) :
        if guessed_flavor != "ZERO" :
            log.warning("Keep original flavor '{}' despite guess = '{}'".format(input_flavor.upper(),guessed_flavor))
        guessed_flavor = input_flavor.upper()

    log.info("FLAVOR INPUT='{}' GUESSED='{}'".format(input_flavor,guessed_flavor))

    return guessed_flavor
