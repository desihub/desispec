"""
desispec.io.filters
===================

"""
import numpy as np
import speclite.filters

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

def load_legacy_survey_filter(band,photsys) :
    """
    Uses speclite.filters to load the filter transmission
    Returns speclite.filters.FilterResponse object

    Args:
        band: filter pass-band in "G","R","Z","W1","W2"
        photsys: "N" or "S" for North (BASS+MzLS) or South (CTIO/DECam)
    """
    filternamemap=None
    if band[0].upper()=="W" : # it's WISE
        filternamemap = "wise2010-{}".format(band.upper())
    elif band.upper() in ["G","R","Z"] :
        if photsys=="N" :
            if band.upper() in ["G","R"] :
                filternamemap="BASS-{}".format(band.lower())
            else :
                filternamemap="MzLS-z"
        elif photsys=="S" :
            filternamemap="decam2014-{}".format(band.lower())
        else :
            raise ValueError("unknown photsys '{}', known ones are 'N' and 'S'".format(photsys))
    else :
        raise ValueError("unknown band '{}', known ones are 'G','R','Z','W1' and 'W2'".format(photsys))

    filter_response=speclite.filters.load_filter(filternamemap)
    return filter_response

def load_gaia_filter(band,dr=2):
    """
    Uses speclite.filters to load the filter transmission
    Returns speclite.filters.FilterResponse object

    Args:
        band: filter pass-band in "G","BP","RP"
        dr: 2 or 3
    """
    if band.upper() not in ["G","BP","RP"]:
        raise ValueError("unknown band '{}'".format(band))
    if dr!=2:
        raise ValueError("currently only DR2 is supported")
    filternamemap = f'gaiadr{dr}-{band}'
    filter_response=speclite.filters.load_filter(filternamemap)
    return filter_response
