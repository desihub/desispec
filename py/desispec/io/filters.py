import numpy as np
import scipy.interpolate

# reading filter quantum efficiency
#- TODO: merge this with desisim filter class
def read_filter_response(given_filter,basepath):
    """
    Read requested filter (SDSS_r, DECAM_g, ...) from files in basepath dir
    
    Returns tuple of wavelength, throughput arrays
    This is for reading files from desisim/data etc.
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

def load_filter_response(given_filter):
    """ 
    Uses speclite.filters to load the filter transmission
    Returns wavelength and throughput

    Args:
        given_filter: given filter for which the qe is to be loaded. Desi templates   
        have them in uppercase, so it should be in upper case like SDSS, DECAM or WISE.
        Speclite has lower case so are mapped here.
    """

    import speclite
    filternamemap={}
    filttype=str.split(given_filter,'_')
    if filttype[0]=='SDSS':
        filternamemap=filttype[0].lower()+'2010-'+filttype[1].lower()
    if filttype[0]=='DECAM':
        if filttype[1]=='Y':
            filternamemap=filttype[0].lower()+'2014-'+filttype[1]
        else: filternamemap=filttype[0].lower()+'2014-'+filttype[1].lower()
    if filttype[0]=='WISE':
        filternamemap=filttype[0].lower()+'2010-'+filttype[1]
    
    filter_response=speclite.filters.load_filter(filternamemap)
    wavelength=filter_response._wavelength
    throughput=filter_response.response
    return wavelength,throughput

    
           
