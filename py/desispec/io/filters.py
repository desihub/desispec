import numpy as np
import speclite

def load_filter(given_filter):
    """ 
    Uses speclite.filters to load the filter transmission
    Returns speclite.filters.FilterResponse object

    Args:
        given_filter: given filter for which the qe is to be loaded. Desi templates/   
        files have them in uppercase, so it should be in upper case like SDSS, DECAM or 
        WISE. Speclite has lower case so are mapped here.
    """

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
    return filter_response

    
           
