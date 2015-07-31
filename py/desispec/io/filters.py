import numpy as np
import scipy.interpolate

# reading filter quantum efficiency
#- TODO: merge this with desisim filter class
def read_filter_response(given_filter,basepath):
    """
    Read requested filter (SDSS_r, DECAM_g, ...) from files in basepath dir
    
    Returns tuple of wavelength, throughput arrays
    """
    filterNameMap={}

    filttype=str.split(given_filter,'_')
    if filttype[0]=='SDSS':
        filterNameMap=given_filter.lower()+"0.txt"
    else: #if breakfilt[0]=='DECAM':
        filterNameMap=given_filter.lower()+".txt"
    filter_response={}
    fileName=basepath+filterNameMap
    wave, throughput = np.loadtxt(fileName, unpack=True)
    ## tck=scipy.interpolate.splrep(wave, throughput, s=0)
    ## filter_response=(wave, throughput, tck)
    ## return filter_response
    return wave, throughput
