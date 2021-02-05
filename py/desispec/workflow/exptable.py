#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from astropy.table import Table
from astropy.io import fits
## Import some helper functions, you can see their definitions by uncomenting the bash shell command
from desispec.workflow.utils import define_variable_from_environment, pathjoin, get_json_dict
from desispec.workflow.desi_proc_funcs import load_raw_data_header, cameras_from_raw_data
from desiutil.log import get_logger
from desispec.util import header2night
from desispec.io.util import create_camword, parse_badamps

#############################################
##### Exposure Table Column Definitions #####
#############################################
## To eventually being turned into a full-fledged data model. For now a brief description.
# EXPID, int, the exposure ID.
# EXPTIME, float, the exposure time.
# OBSTYPE, string, the obstype as defined by ICS.
# CAMWORD, string, typically 'a'+ str(spectrographs) defined by ICS unless specified by command line argument
# TILEID, int, the TILEID of the tile the exposure observed.
# TARGTRA, float, The TARGTRA as given by ICS. The RA of the target.
# TARGTDEC, float, The TARGTDEC as given by ICS. The DEC of the target.
# NIGHT, int, the night of the observation.
# PURPOSE, str, The purpose of the exposure as defined by ICS.
# FA_SURV, str, The survey designated/used by fiberassign
# SEQNUM, int, The number of the current exposure in a sequence of exposures, as given by ICS. If not a sequence, SEQNUM is 1.
# SEQTOT, int, The total number of exposures taken in the current sequence, as given by ICS. If not a sequence, SEQTOT is 1.
# PROGRAM, string, The program as given by ICS.
# MJD-OBS, float, The MJD-OBS as given by ICS. Modified Julian Date of the observation.
# BADCAMWORD, string, camword defining the bad cameras that should not be processed.
# BADAMPS, string, semicolon separated list of "{camera}{petal}{amp}", i.e. "[brz][0-9][ABCD]". Example: 'b7D;z8A'
# LASTSTEP, string, the last step the pipeline should run through for the given exposure. Inclusive of last step.
# EXPFLAG, np.ndarray, set of flags that describe that describe the exposure.
# HEADERERR, np.ndarray, In the csv given as a "|" separated list of key=value pairs describing columns in the table that should
#                        be corrected. The workflow transforms these into an array of strings.
#                        NOTE: This will be used to change the given key/value pairs in the production table.
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
        coltypes, list. List of column datatypes for the names in colnames.
        coldeflts, list. Optionally returned if return_default_values is True. List of default values for the
                         corresponding colnames.
    """
    ## Define the column names for the exposure table and their respective datatypes, split in two
    ##     only for readability's sake
    colnames  = ['EXPID', 'EXPTIME', 'OBSTYPE', 'CAMWORD'    , 'TILEID', 'TARGTRA', 'TARGTDEC', 'NIGHT' ]
    coltypes  = [int    ,  float   , 'S8'     , 'S30'        ,  int    ,  float   ,  float    ,  int     ]
    coldeflt  = [-99    ,  0.0     , 'unknown', 'a0123456789',  -99    ,  89.99   ,  -89.99   ,  20000101]

    colnames += ['PURPOSE', 'FA_SURV', 'SEQNUM', 'SEQTOT', 'PROGRAM', 'MJD-OBS']
    coltypes += ['S30'    , 'S10'    ,  int    ,  int    , 'S60'    ,  float   ]
    coldeflt += [''       , ''       ,  1      ,  1      , 'unknown',  50000.0 ]

    colnames += ['BADCAMWORD', 'BADAMPS', 'LASTSTEP', 'EXPFLAG'  , 'HEADERERR', 'COMMENTS']
    coltypes += ['S30'       , 'S30'    , 'S30'     , np.ndarray , np.ndarray , np.ndarray]
    coldeflt += [''          , ''       , 'all'     , np.array([], dtype=str), np.array([], dtype=str), np.array([], dtype=str)]

    if return_default_values:
        return colnames, coltypes, coldeflt
    else:
        return colnames, coltypes

def default_exptypes_for_exptable():
    """
    Defines the exposure types to be recognized by the workflow and saved in the exposure table by default.

    Returns:
        list. A list of default obstypes to be included in an exposure table.
    """
    ## Define the science types to be included in the exposure table (case insensitive)
    return ['arc','flat','twilight','science','sci','dither','dark','bias','zero']

def get_exposure_flags():
    """
    Defines the exposure flags that can be saved in the exposure table.

    Returns:
        list. A list of exposure flags that can be included in an exposure table.
    """
    return [
            'good',
            'extra_cal', # if more than one series of cals are run, it would be nice to flag and skip others in some circumstances

            ## Might potentially crash, but nothing fundamentally wrong
            'low_flux',
            'short_exposure',
            'aborted',

            ## Missing or incorrect data
            'metadata_missing', # important header keywords or fiberassign values missing
            'metadata_mismatch', # the raw data header and accompanying files disagree on something

            ## Hardware issues
            'misconfig_cal', # cal lamps weren't on, etc.
            'misconfig_petal', # positioners in wrong places, etc.

            ## Targeting issues
            'off_target',  # telescope wasn't pointed, etc.
            'no_stdstars',  # data is missing standard stars

            ## Others
            'test', # a test exposure that shouldn't be processed
            'corrupted', # data is corrupted
            'junk',

            ## No explanation, but don't use
            'bad'
           ]

def get_last_step_options():
    """
    Defines the LASTSTEP options that can be saved in the exposure table that will be understood by the pipeline.

    Returns:
        list. A list of LASTSTEP's that can be included in an exposure table.
    """
    return ['ignore', 'skysub', 'stdstarfit', 'fluxcal', 'all']

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

def get_exposure_table_path(night=None, usespecprod=True):
    """
    Defines the default path to save an exposure table. If night is given, it saves it under a monthly directory
    to reduce the number of files in a large production directory.

    Args:
        night, int or str or None. The night corresponding to the exposure table. If None, no monthly subdirectory is used.
        usespecprod, bool. Whether to use the master version in the exposure table repo or the version in a specprod.

    Returns:
         str. The full path to the directory where the exposure table should be written (or is already written). This
              does not including the filename.
    """
    # if night is None and 'PROD_NIGHT' in os.environ:
    #     night = os.environp['PROD_NIGHT']
    if usespecprod:
        basedir = define_variable_from_environment(env_name='DESI_SPECTRO_REDUX',
                                                      var_descr="The specprod path")
        # subdir = define_variable_from_environment(env_name='USER', var_descr="Username for unique exposure table directories")
        subdir = define_variable_from_environment(env_name='SPECPROD', var_descr="Use SPECPROD for unique exposure table directories")
        basedir = pathjoin(basedir, subdir)
    else:
        basedir = define_variable_from_environment(env_name='DESI_SPECTRO_LOG',
                                                   var_descr="The exposure table repository path")
    if night is None:
        return pathjoin(basedir,'exposure_tables')
    else:
        month = night_to_month(night)
        path = pathjoin(basedir,'exposure_tables',month)
        return path

def get_exposure_table_pathname(night, usespecprod=True, extension='csv'):#base_path,specprod
    """
    Defines the default pathname to save an exposure table.

    Args:
        night, int or str or None. The night corresponding to the exposure table.
        usespecprod, bool. Whether to use the master version or the version in a specprod.

    Returns:
         str. The full pathname where the exposure table should be written (or is already written). This
              includes the filename.
    """
    path = get_exposure_table_path(night, usespecprod=usespecprod)
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

def keyval_change_reporting(keyword, original_val, replacement_val):
    """
    Creates a reporting string to be saved in the HEADERERR or COMMENTS column of the exposure table. Give the keyword,
    the original value, and the value that it was replaced with.

    Args:
        keyword, str. Keyword in the exposure table that the values correspond to.
        original_val, str. The value typically saved for that keyword, except when it deviates.
        replacement_val, str. The value that was saved instead in that keyword column.

    Returns:
        str. Of the format ' keyword:original->replacement '
    """
    if original_val is None:
        original_val = "None"
    return f'{keyword}:{original_val}->{replacement_val}'

def deconstruct_keyval_reporting(entry):
    """
    Takes a reporting of the form '{colname}:{oldval}->{newval}' and returns colname, oldval, newval.

    Args:
        entry, str. A string of the form '{colname}:{oldval}->{newval}'. colname should be an all upper case column name.
                    oldval and newval can include any string characters except the specific combination "->".

    Returns:
        key, str. The string that precedes the initial colon. The name of the column being reported.
        val1, str. The string after the initial colon and preceding the '->'. The original value of the column.
        val2, str. The string after the '->'. The value that the original was changed to.
    """
    ## Ensure that the rudimentary characteristics are there
    if ':' not in entry or '->' not in entry:
        raise ValueError("Entry must be of the form {key}:{oldval}->{newval}. Exiting")
    ## Get the key left of colon
    entries = entry.split(':')
    key = entries[0]
    ## The values could potentially have colon's. This allows for that
    values = ':'.join(entries[1:])
    ## Two values should be separated by text arrow
    val1,val2 = values.split("->")
    return key, val1, val2

def validate_badamps(badamps,joinsymb=';'):
    """
    Checks and transforms badamps string for consistency with the for need in an exposure or processing table
    for use in the Pipeline. Specifically ensure they come in (camera,petal,amplifier) sets,
    with appropriate checking of those values to make sure they're valid. Returns the input string
    except removing whitespace and replacing potential character separaters with joinsymb (default ';').

    Args:
        badamps, str. A string of {camera}{petal}{amp} entries separated by symbol given with joinsymb (semicolon
                      by default). I.e. [brz][0-9][ABCD]. Example: 'b7D;z8A'.
        joinsymb, str. The symbol separating entries in the str list given by badamps.

    Returns:
        badamps, str. Input badamps string of {camera}{petal}{amp} entries separated by symbol given with
                      joinsymb (semicolon by default). I.e. [brz][0-9][ABCD]. Example: 'b7D;z8A'.
                      Differs from input in that other symbols used to separate terms are replaaced by joinsymb
                      and whitespace is removed.

    """
    badamps = badamps.replace(' ', '').strip()
    for symb in [',', ':', '|', '.']:
        badamps = badamps.replace(symb, joinsymb)
    ## test that the string can be parsed
    throw = parse_badamps(badamps, joinsymb=joinsymb)
    return badamps

def summarize_exposure(raw_data_dir, night, exp, obstypes=None, colnames=None, coldefaults=None, verbosely=False):
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
    if colnames is None or coldefaults is None:
        cnames, cdtypes, cdflts = get_exposure_table_column_defs(return_default_values=True)
        if colnames is None:
            colnames = cnames
        if coldefaults is None or len(coldefaults)!=len(colnames):
            coldefaults = cdflts
    colnames,coldefaults = np.asarray(colnames),np.asarray(coldefaults,dtype=object)

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
        ## standardize the naming of end of arc/flats as best we can
        if int(night) < 20200310:
            pass
        elif int(night) < 20200801:
            if 'PROGRAM' in req_dict:
                prog = req_dict['PROGRAM'].lower()
                if 'calib' in prog and 'done' in prog:
                    if 'short' in prog:
                        return "endofshortflats"
                    elif 'long' in prog:
                        return 'endofflats'
                    elif 'arc' in prog:
                        return 'endofarcs'
        else:
            if 'MANIFEST' in req_dict:
                manifest = req_dict['MANIFEST']
                if 'name' in manifest:
                    name = manifest['name'].lower()
                    if name in ['endofarcs', 'endofflats', 'endofshortflats']:
                        return name

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
            log.info(f'{exp}: skipped  -- {flavor} not a relevant flavor')
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
    if obstype not in obstypes:
        ## If obstype is wrong
        if verbosely:
            log.info(f'ignoring: {reqpath} -- {obstype} not an obstype we care about')
        else:
            log.info(f'{exp}: skipped  -- {obstype} not relevant obstype')
        return None

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
    # log.debug(hdulist.info())

    header,fx = load_raw_data_header(pathname=datapath, return_filehandle=True)
    # log.debug(header)
    # log.debug(specs)

    ## Define the column values for the current exposure in a dictionary
    outdict = {}
    ## Set HEADERERR and EXPFLAG before loop because they may be set if other columns have missing information
    outdict['HEADERERR'] = coldefaults[colnames == 'HEADERERR'][0]
    outdict['EXPFLAG'] = coldefaults[colnames == 'EXPFLAG'][0]
    ## Loop over columns and fill in the information. If unavailable report/flag if necessary and assign default
    for key,default in zip(colnames,coldefaults):
        ## These are dealt with separately
        if key in ['NIGHT','HEADERERR','EXPFLAG']:
            continue
        ## These just need defaults, as they are user defined (except FA_SURV which comes from the request.json file
        elif key in ['CAMWORD', 'FA_SURV', 'BADCAMWORD', 'BADAMPS', 'LASTSTEP', 'COMMENTS']:
            outdict[key] = default
        ## Try to find the key in the raw data header
        elif key in header.keys():
            val = header[key]
            if type(val) is str:
                outdict[key] = val.lower()
            else:
                outdict[key] = val
        ## If key not in the header, identify that and place a default value
        ## If obstype isn't arc or flat, don't worry about seqnum or seqtot
        elif key in ['SEQNUM','SEQTOT'] and obstype not in ['arc','flat']:
            outdict[key] = default
        ## If tileid or TARGT and not science, just replace with default
        elif key in ['TILEID','TARGTRA','TARGTDEC'] and obstype not in ['science']:
            outdict[key] = default
        ## if something else, flag as missing metadata and replace with default
        else:
            if 'metadata_missing' not in outdict['EXPFLAG']:
                outdict['EXPFLAG'] = np.append(outdict['EXPFLAG'], 'metadata_missing')
            outdict[key] = default
            if np.isscalar(default):
                reporting = keyval_change_reporting(key, '', default)
                outdict['HEADERERR'] = np.append(outdict['HEADERERR'], reporting)

    ## Make sure that the night is defined:
    try:
        outdict['NIGHT'] = int(header['NIGHT'])
    except (KeyError, ValueError, TypeError):
        if 'metadata_missing' not in outdict['EXPFLAG']:
            outdict['EXPFLAG'] = np.append(outdict['EXPFLAG'], 'metadata_missing')
        outdict['NIGHT'] = header2night(header)
        try:
            orig = str(header['NIGHT'])
        except (KeyError, ValueError, TypeError):
            orig = ''
        reporting = keyval_change_reporting('NIGHT',orig,outdict['NIGHT'])
        outdict['HEADERERR'] = np.append(outdict['HEADERERR'],reporting)

    ## Get the cameras available in the raw data and summarize with camword
    cams = cameras_from_raw_data(fx)
    camword = create_camword(cams)
    outdict['CAMWORD'] = camword
    fx.close()

    ## Add the fiber assign survey, if it doesn't exist use the pre-defined one
    if "FA_SURV" in req_dict and "FA_SURV" in colnames:
        outdict['FA_SURV'] = req_dict['FA_SURV']

    ## Flag the exposure based on PROGRAM information
    if 'system test' in outdict['PROGRAM'].lower():
        outdict['LASTSTEP'] = 'ignore'
        outdict['EXPFLAG'] = np.append(outdict['EXPFLAG'], 'test')
        log.info(f"Exposure {exp} identified as system test. Not processing.")
    elif obstype == 'science' and float(outdict['EXPTIME']) < 59.0:
        outdict['LASTSTEP'] = 'skysub'
        outdict['EXPFLAG'] = np.append(outdict['EXPFLAG'], 'short_exposure')
        log.info(f"Science exposure {exp} with EXPTIME less than 59s. Processing through sky subtraction.")
    elif obstype == 'science' and 'undither' in outdict['PROGRAM']:
        outdict['LASTSTEP'] = 'fluxcal'
        log.info(f"Science exposure {exp} identified as undithered. Processing through flux calibration.")
    elif obstype == 'science' and 'dither' in outdict['PROGRAM']:
        outdict['LASTSTEP'] = 'skysub'
        log.info(f"Science exposure {exp} identified as dither. Processing through sky subtraction.")

    ## For Things defined in both request and data, if they don't match, flag in the
    ##     output file for followup/clarity
    for check in ['OBSTYPE']:#, 'FLAVOR']:
        rval, hval = req_dict[check], header[check]
        if rval != hval:
            log.warning(f'In keyword {check}, request and data header disagree: req:{rval}\tdata:{hval}')
            if 'metadata_mismatch' not in outdict['EXPFLAG']:
                outdict['EXPFLAG'] = np.append(outdict['EXPFLAG'], 'metadata_mismatch')
            outdict['COMMENTS'] = np.append(outdict['COMMENTS'],f'For {check}: req={rval} but hdu={hval}')
        else:
            if verbosely:
                log.info(f'{check} checks out')

    ## Special logic for EXPTIME because of real-world variance on order 10's - 100's of ms
    check = 'EXPTIME'
    rval, hval = req_dict[check], header[check]
    if np.abs(float(rval)-float(hval))>0.5:
        log.warning(f'In keyword {check}, request and data header disagree: req:{rval}\tdata:{hval}')
        if 'aborted' not in outdict['EXPFLAG']:
            outdict['EXPFLAG'] = np.append(outdict['EXPFLAG'], 'aborted')
        outdict['COMMENTS'] = np.append(outdict['COMMENTS'],f'For {check}: req={rval} but hdu={hval}')
    else:
        if verbosely:
            log.info(f'{check} checks out')

    log.info(f'Done summarizing exposure: {exp}')
    return outdict

