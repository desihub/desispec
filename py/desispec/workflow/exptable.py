#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from astropy.table import Table
from astropy.io import fits
## Import some helper functions, you can see their definitions by uncomenting the bash shell command
from desispec.workflow.utils import define_variable_from_environment, pathjoin, get_json_dict
from desiutil.log import get_logger

#############################################
##### Exposure Table Column Definitions #####
#############################################
## To eventually being turned into a full-fledged data model. For now a brief description.
# EXPID, int, the exposure ID.
# EXPTIME, float, the exposure time.
# OBSTYPE, string, the obstype as defined by ICS.
# SPECTROGRAPHS, string, the spectrographs as defined by ICS.
# CAMWORD, string, typically 'a'+ str(spectrographs) meaning all cameras of the available spectrographs.
# TILEID, int, the TILEID of the tile the exposure observed.
# NIGHT, int, the night of the observation.
# EXPFLAG, int, A 0 signifies that it is a good observation. 1 or greater indicates an issue.
#               Each number will be assigned a meaning. Currently 1 just means "don't use".
# HEADERERR, np.ndarray, Given as a "|" separated list of key=value pairs describing columns in the table that should
#                        be corrected. The workflow transforms these into an array of strings.
#                        NOTE: This will be used to change the given key/value pairs in the production table.
# SURVEY, int, a numeric ID for a given observing run, e.g. CMX is 0. SV0 is 1, etc.
# SEQNUM, int, The number of the current exposure in a sequence of exposures, as given by ICS. If not a sequence, SEQNUM is 1.
# SEQTOT, int, The total number of exposures taken in the current sequence, as given by ICS. If not a sequence, SEQTOT is 1.
# PROGRAM, string, The program as given by ICS.
# MJD-OBS, float, The MJD-OBS as given by ICS. Modified Julian Date of the observation.
# REQRA, float, The REQRA as given by ICS. The requested RA.
# REQDEC, float, The REQDEC as given by ICS. The requested DEC.
# TARGTRA, float, The TARGTRA as given by ICS. The RA of the target.
# TARGTDEC, float, The TARGTDEC as given by ICS. The DEC of the target.
# COMMENTS, np.ndarray, In the csv given as either a "|" separated list of comments or one long comment. When loaded it
#                       is a numpy array of the strings.These are not used by the workflow but useful for humans
#                       to put notes for other humans.
##################################################

def get_exposure_table_column_defs(return_default_values=False):
    """
    Contains the column names, data types, and default row values for a DESI Exposure table. It returns
    the names and datatypes with the defaults being given with an optional flag. Returned as 2 (or 3) lists.

    Args:
        return_default_values, bool. True if you want the default values returned.

    Returns:
        colnames, list. List of column names for an exposure table.
        coldtypes, list. List of column datatypes for the names in colnames.
        coldeflts, list. Optionally returned if return_default_values is True. List of default values for the
                         corresponding colnames.
    """
    ## Define the column names for the exposure table and their respective datatypes, split in two
    ##     only for readability's sake
    colnames1 = ['EXPID', 'EXPTIME', 'OBSTYPE', 'SPECTROGRAPHS', 'CAMWORD', 'TILEID']
    coltypes1 = [int, float, 'S8', 'S10', 'S30', int]
    coldeflt1 = [-99, 0.0, 'unknown', '0123456789', 'a09123456789', -99]

    colnames2 = ['NIGHT', 'EXPFLAG', 'HEADERERR', 'SURVEY', 'SEQNUM', 'SEQTOT', 'PROGRAM', 'MJD-OBS']
    coltypes2 = [int, int, np.ndarray, int, int, int, 'S30', float]
    coldeflt2 = [20000101, 0, np.array([], dtype=str), 0, 1, 1, 'unknown', 50000.0]

    colnames3 = ['REQRA', 'REQDEC', 'TARGTRA', 'TARGTDEC', 'COMMENTS']
    coltypes3 = [float, float, float, float, np.ndarray]
    coldeflt3 = [-99.99, -89.99, -99.99, -89.99, np.array([], dtype=str)]

    colnames = colnames1 + colnames2 + colnames3
    coldtypes = coltypes1 + coltypes2 + coltypes3
    coldeflts = coldeflt1 + coldeflt2 + coldeflt3

    if return_default_values:
        return colnames, coldtypes, coldeflts
    else:
        return colnames, coldtypes

def default_exptypes_for_exptable():
    """
    Defines the exposure types to be recognized by the workflow and saved in the exposure table by default.

    Returns:
        list. A list of default obstypes to be included in an exposure table.
    """
    ## Define the science types to be included in the exposure table (case insensitive)
    return ['arc','flat','twilight','science','sci','dither','dark','bias','zero']

def get_survey_definitions():
    """
    Defines a numeric value to a 'survey', which in this context is a duration of of observing nights that
    shared some common theme. Examples: minisv2, SV0, SV1, commissioning, etc. Currently a placeholder for
    future development.

    Returns:
        survey_def, dict. A dictionary with keys corresponding to the numeric representation of a particular survey,
                          with values being a tuple of ints. The first int is the first valid night and the
                          second int is the last valid night of the survey.
    """
    ## Create a rudimentary way of assigning "SURVEY keywords based on what date range a night falls into"
    survey_def = {0: (20200201, 20200315), 1: (
        20201201, 20210401)}  # 0 is CMX, 1 is SV1, 2 is SV2, ..., 99 is any testing not in these timeframes
    return survey_def

def get_surveynum(night, survey_definitions=None):
    """
    Given a night and optionally the survey definitions (which are looked up if not given), this returns the
    proper numeric survey ID for the night.

    Args:
        night, int or str. The night of observations for which you want to know the numeric survey ID.
        survey_definitions, dict. A dictionary with keys corresponding to the numeric representation of a particular survey,
                          with values being a tuple of ints. The first int is the first valid night and the
                          second int is the last valid night of the survey.

    Returns:
        int. The numerical ID corresponding to the survey in which the given night took place.
    """
    night = int(night)
    if survey_definitions is None:
        survey_definitions = get_survey_definitions()

    for survey, (low, high) in survey_definitions.items():
        if night >= low and night <= high:
            return survey
    return 99

def night_to_month(night):
    """
    Trivial function that returns the month portion of a night. Can be given a string or int.

    Args:
        night, int or str. The night you want the month of.

    Returns:
        str. The zero-padded (length two) string representation of the month corresponding to the input month.
    """
    return str(night)[:-2]

def get_exposure_table_name(night, extension='csv'):
    """
    Defines the default exposure name given the night of the observations and the optional extension.

    Args:
        night, int or str. The night of the observations going into the exposure table.
        extension, str. The extension (and therefore data format) without a leading period  of the saved table.
                        Default is 'csv'.

    Returns:
        str. The exposure table name given the input night and extension.
    """
    # if night is None and 'PROD_NIGHT' in os.environ:
    #     night = os.environp['PROD_NIGHT']
    return f'exposure_table_{night}.{extension}'

def get_exposure_table_path(night=None):
    """
    Defines the default path to save an exposure table. If night is given, it saves it under a monthly directory
    to reduce the number of files in a large production directory.

    Args:
        night, int or str or None. The night corresponding to the exposure table. If None, no monthly subdirectory is used.

    Returns:
         str. The full path to the directory where the exposure table should be written (or is already written). This
              does not including the filename.
    """
    # if night is None and 'PROD_NIGHT' in os.environ:
    #     night = os.environp['PROD_NIGHT']
    spec_redux = define_variable_from_environment(env_name='DESI_SPECTRO_REDUX',
                                                          var_descr="The exposure table path")
    # subdir = define_variable_from_environment(env_name='USER', var_descr="Username for unique exposure table directories")
    subdir = define_variable_from_environment(env_name='SPECPROD', var_descr="Use SPECPROD for unique exposure table directories")
    if night is None:
        return pathjoin(spec_redux,subdir,'exposure_tables')
    else:
        month = night_to_month(night)
        path = pathjoin(spec_redux,subdir,'exposure_tables',month)
        return path

def get_exposure_table_pathname(night, extension='csv'):#base_path,prodname
    """
    Defines the default pathname to save an exposure table.

    Args:
        night, int or str or None. The night corresponding to the exposure table.

    Returns:
         str. The full pathname where the exposure table should be written (or is already written). This
              includes the filename.
    """
    # if night is None and 'PROD_NIGHT' in os.environ:
    #     night = os.environp['PROD_NIGHT']
    path = get_exposure_table_path(night)
    table_name = get_exposure_table_name(night, extension)
    return pathjoin(path,table_name)

def instantiate_exposure_table(colnames=None, coldtypes=None, rows=None):
    """
    Create an empty exposure table with proper column names and datatypes. If rows is given, it inserts the rows
    into the table, otherwise it returns a table with no rows.

    Args:
        colnames, list. List of column names for an exposure table.
        coldtypes, list. List of column datatypes for the names in colnames.
        rows, list or np.array of Table.Rows or dicts. An iterable set of Table.Row's or dicts with keys/colnames and value
                                                       pairs that match the default column names and data types of the
                                                       default exposure table.

    Returns:
          exposure_table, Table. An astropy Table with the column names and data types for a DESI workflow exposure
                                 table. If the input rows was not None, it contains those rows, otherwise it has no rows.
    """
    if colnames is None or coldtypes is None:
       colnames, coldtypes = get_exposure_table_column_defs()

    exposure_table = Table(names=colnames,dtype=coldtypes)
    if rows is not None:
        for row in rows:
            exposure_table.add_row(row)
    return exposure_table



def summarize_exposure(raw_data_dir, night, exp, obstypes=None, surveynum=None, colnames=None, coldefaults=None, verbosely=False):
    """
    Given a raw data directory and exposure information, this searches for the raw DESI data files for that
    exposure and loads in relevant information for that flavor+obstype. It returns a dictionary if the obstype
    is one of interest for the exposure table, a string if the exposure signifies the end of a calibration sequence,
    and None if the exposure is not in the given obstypes.

    Args:
        raw_data_dir, str. The path to where the raw data is stored. It should be the upper level directory where the
                           nightly subdirectories reside.
        night, str or int. Used to know what nightly subdirectory to look for the given exposure in.
        exp, str or int or float. The exposure number of interest.
        obstypes, list or np.array of str's. The list of 'OBSTYPE' keywords to match to. If a match is found, the
                                             information about that exposure is taken and returned for the exposure
                                             table. Otherwise None is returned (or str if it is an end-of-cal manifest).
                                             If None, the default list in default_exptypes_for_exptable() is used.
        surveynum, int. The numeric ID of the survey that the night corresponds to. If none, it is looked up from the
                        default in get_surveynum().
        colnames, list or np.array. List of column names for an exposure table. If None, the defaults are taken from
                                    get_exposure_table_column_defs().
        coldefaults, list or np.array. List of default values for the corresponding colnames. If None, the defaults
                                       are taken from get_exposure_table_column_defs().
        verbosely, bool. Whether to print more detailed output (True) or more succinct output (False).

    Returns:
        outdict, dict. Dictionary with keys corresponding to the column names of an exposure table. Values are
                       taken from the data when found, otherwise the values are the corresponding default given in
                       coldefaults.
        OR
        str. If the exposures signifies the end of a calibration sequence, it returns a string describing the type of
             sequence that ended. Either "(short|long|arc) calib complete".
        OR
        NoneType. If the exposure obstype was not in the requested types (obstypes).
    """
    log = get_logger()

    ## Make sure the inputs are in the right format
    if type(exp) is not str:
        exp = int(exp)
        exp = f'{exp:08d}'
    night = str(night)

    ## Use defaults if things aren't defined
    if obstypes is None:
        obstypes = default_exptypes_for_exptable()
    if surveynum is None:
        surveynum = get_surveynum(night)
    if colnames is None or coldefaults is None:
        cnames, cdtypes, cdflts = get_exposure_table_column_defs(return_default_values=True)
        if colnames is None:
            colnames = cnames
        if coldefaults is None:
            coldefaults = cdflts

    ## Give a header for the exposure
    if verbosely:
        log.info(f'\n\n###### Summarizing exposure: {exp} ######\n')
    else:
        log.info(f'Summarizing exposure: {exp}')
    ## Request json file is first used to quickly identify science exposures
    ## If a request file doesn't exist for an exposure, it shouldn't be an exposure we care about
    reqpath = pathjoin(raw_data_dir, night, exp, f'request-{exp}.json')
    if not os.path.isfile(reqpath):
        if verbosely:
            log.info(f'{reqpath} did not exist!')
        else:
            log.info(f'{exp}: skipped  -- request not found')
        return None

    ## Load the json file in as a dictionary
    req_dict = get_json_dict(reqpath)

    ## Check to see if it is a manifest file for calibrations
    if "SEQUENCE" in req_dict and req_dict["SEQUENCE"].lower() == "manifest":
        if 'PROGRAM' in req_dict:
            prog = req_dict['PROGRAM'].lower()
            if 'calib' in prog and 'done' in prog:
                if 'short' in prog:
                    return "short calib complete"
                elif 'long' in prog:
                    return "long calib complete"
                elif 'arc' in prog:
                    return 'arc calib complete'
                else:
                    pass

    ## If FLAVOR is wrong or no obstype is defines, skip it
    if 'FLAVOR' not in req_dict.keys():
        if verbosely:
            log.info(f'WARNING: {reqpath} -- flavor not given!')
        else:
            log.info(f'{exp}: skipped  -- flavor not given!')
        return None

    flavor = req_dict['FLAVOR'].lower()
    if flavor != 'science' and 'dark' not in obstypes and 'zero' not in obstypes:
        ## If FLAVOR is wrong
        if verbosely:
            log.info(f'ignoring: {reqpath} -- {flavor} not a flavor we care about')
        else:
            log.info(f'{exp}: skipped  -- not science')
        return None

    if 'OBSTYPE' not in req_dict.keys():
        ## If no obstype is defines, skip it
        if verbosely:
            log.info(f'ignoring: {reqpath} -- {flavor} flavor but obstype not defined')
        else:
            log.info(f'{exp}: skipped  -- obstype not given')
        return None
    else:
        if verbosely:
            log.info(f'using: {reqpath}')

    ## If obstype isn't in our list of ones we care about, skip it
    obstype = req_dict['OBSTYPE'].lower()
    if obstype in obstypes:
        ## Look for the data. If it's not there, say so then move on
        datapath = pathjoin(raw_data_dir, night, exp, f'desi-{exp}.fits.fz')
        if not os.path.exists(datapath):
            if verbosely:
                log.info(f'could not find {datapath}! It had obstype={obstype}. Skipping')
            else:
                log.info(f'{exp}: skipped  -- data not found')
            return None
        else:
            if verbosely:
                log.info(f'using: {datapath}')

        ## Raw data, so ensure it's read only and close right away just to be safe
        hdulist = fits.open(datapath, mode='readonly')
        # log.debug(hdulist.info())

        if 'SPEC' in hdulist:
            hdu = hdulist['SPEC']
            if verbosely:
                log.info("SPEC found")
        elif 'SPS' in hdulist:
            hdu = hdulist['SPS']
            if verbosely:
                log.info("SPS found")
        else:
            log.info(f'{exp}: skipped  -- "SPEC" HDU not found!!')
            hdulist.close()
            return None

        header, specs = dict(hdu.header).copy(), hdu.data.copy()
        hdulist.close()
        # log.debug(header)
        # log.debug(specs)

        ## Define the column values for the current exposure in a dictionary
        outdict = {}
        for key,default in zip(colnames,coldefaults):
            if key in header.keys():
                val = header[key]
                if type(val) is str:
                    outdict[key] = val.lower()
                else:
                    outdict[key] = val
            else:
                outdict[key] = default

        ## For now assume that all 3 cameras were good for all operating spectrographs
        outdict['SPECTROGRAPHS'] = ''.join([str(spec) for spec in np.sort(specs)])
        outdict['CAMWORD'] = 'a' + outdict['SPECTROGRAPHS']

        ## Survey number befined in upper loop based on night
        outdict['SURVEY'] = surveynum

        ## As an example of future flag possibilites, flag science exposures are
        ##    garbage if less than 60 seconds
        if header['OBSTYPE'].lower() == 'science' and float(header['EXPTIME']) < 60:
            outdict['EXPFLAG'] = 2
        else:
            outdict['EXPFLAG'] = 0

        ## For Things defined in both request and data, if they don't match, flag in the
        ##     output file for followup/clarity
        for check in ['EXPTIME', 'OBSTYPE', 'FLAVOR']:
            rval, hval = req_dict[check], header[check]
            if rval != hval:
                if verbosely:
                    log.info(f'{rval}\t{hval}')
                outdict['EXPFLAG'] = 1
                outdict['HEADERERR'] += f'req:{rval} but hdu:{hval} | '
            else:
                if verbosely:
                    log.info(f'{check} checks out')

        #outdict['COMMENTS'] += '|'
        #outdict['HEADERERR'] += '|'

        #cnames,ctypes,cdefs = get_exposure_table_column_defs(return_default_values=True)
        #for nam,typ,deflt in zip(cnames,ctypes,cdefs):
        #    if nam not in outdict.keys():
        #        outdict[nam] = deflt
                
        log.info(f'Done summarizing exposure: {exp}')
        return outdict

