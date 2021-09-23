#!/usr/bin/env python
# coding: utf-8

import os
import glob
import numpy as np
from astropy.table import Table
from astropy.io import fits
## Import some helper functions, you can see their definitions by uncomenting the bash shell command
from desispec.workflow.utils import define_variable_from_environment, get_json_dict
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
# BADAMPS, string, comma separated list of "{camera}{petal}{amp}", i.e. "[brz][0-9][ABCD]". Example: 'b7D,z8A'
#                  in the csv this is saved as a semicolon separated list
# LASTSTEP, string, the last step the pipeline should run through for the given exposure. Inclusive of last step.
# EXPFLAG, np.ndarray, set of flags that describe that describe the exposure.
# HEADERERR, np.ndarray, In the csv given as a "|" separated list of key=value pairs describing columns in the table that should
#                        be corrected. The workflow transforms these into an array of strings.
#                        NOTE: This will be used to change the given key/value pairs in the production table.
# COMMENTS, np.ndarray, In the csv given as either a "|" separated list of comments or one long comment. When loaded it
#                       is a numpy array of the strings.These are not used by the workflow but useful for humans
#                       to put notes for other humans.
##################################################

def exposure_table_column_defs():
    """
    Contains the column names, data types, and default row values for a DESI Exposure table. It returns
    the names, datatypes, and defaults as a list of 3-tuples.

    Args:
        None.

    Returns:
        columns, list. List of tuples (name, type, default).
    """
    columns = [
                ('EXPID', int, -99),
                ('OBSTYPE', 'S8', 'unknown'),
                ('TILEID', int, -99),
                ('LASTSTEP', 'S30', 'all'),
                ('CAMWORD', 'S30', 'a0123456789'),
                ('BADCAMWORD', 'S30', ''),
                ('BADAMPS', 'S30', ''),
                ('EXPTIME', float, 0.0),
                ('EFFTIME_ETC', float, -99.),
                ('SURVEY', 'S10', 'unknown'),
                ('FA_SURV', 'S10', 'unknown'),
                ('FAPRGRM', 'S10', 'unknown'),
                ('GOALTIME', float, -99.),
                ('GOALTYPE', 'S10', 'unknown'),
                ('EBVFAC', float, 1.0),
                ('AIRMASS', float, 1.0),
                ('SPEED', float, -99.0),
                ('TARGTRA', float, 89.99),
                ('TARGTDEC', float, -89.99),
                ('SEQNUM', int, 1),
                ('SEQTOT', int, 1),
                ('PROGRAM', 'S60', 'unknown'),
                ('PURPOSE', 'S30', 'unknown'),
                ('MJD-OBS', float, 50000.0),
                ('NIGHT', int, 20000101),
                ('HEADERERR', np.ndarray, np.array([], dtype=str)),
                ('EXPFLAG', np.ndarray, np.array([], dtype=str)),
                ('COMMENTS', np.ndarray, np.array([], dtype=str))
              ]
    return columns

def default_obstypes_for_exptable():
    """
    Defines the exposure types to be recognized by the workflow and saved in the exposure table by default.

    Returns:
        list. A list of default obstypes to be included in an exposure table.
    """
    ## Define the science types to be included in the exposure table (case insensitive)
    return ['arc','flat','twilight','science','dark','zero']

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
            'low_sn',
            'low_speed',
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
    columns = exposure_table_column_defs()

    colnames, coltypes, coldeflt = [], [], []
    for colname, coltype, coldef in columns:
        colnames.append(colname)
        coltypes.append(coltype)
        coldeflt.append(coldef)

    if return_default_values:
        return colnames, coltypes, coldeflt
    else:
        return colnames, coltypes

def get_exposure_table_column_names():
    """
    Contains the column names, data types, and default row values for a DESI Exposure table. It returns
    the names as a list.

    Args:
        None

    Returns:
        colnames, list. List of column names for an exposure table.
    """
    columns  = exposure_table_column_defs()

    colnames = []
    for colname, coltype, coldef in columns:
        colnames.append(colname)

    return colnames

def get_exposure_table_column_types(asdict=True):
    """
    Returns the datatypes values for each column entry of a DESI pipeline Exposure table. It returns
    the names as keys and datatypes as values in a dictionary if asdict=True. If False, the datatypes
    are returned as a list (in the same order as the names returned by get_exposure_table_column_names().

    Args:
        asdict, bool. True if you want the types as values in a dictionary with keys as the names of the
                      columns. If False, an ordered list is returned. Default is True.

    Returns:
        coltypes, dict or list. If asdict, a dict is returned with column datatypes as the values and columns
                                names as the keys. If False, a list of datatypes in the same order as the names
                                returned from get_exposure_table_column_names().
    """
    columns = exposure_table_column_defs()

    if asdict:
        coltypes = {}
        for colname, coltype, coldef in columns:
            coltypes[colname] = coltype
    else:
        coltypes = []
        for colname, coltype, coldef in columns:
            coltypes.append(coltype)

    return coltypes

def get_exposure_table_column_defaults(asdict=True):
    """
    Returns the default values for each column entry of a DESI pipeline Exposure table. It returns
    the names as keys and defaults as values in a dictionary if asdict=True. If False, the defaults
    are returned as a list (in the same order as the names returned by get_exposure_table_column_names().

    Args:
        asdict, bool. Default is True. If you want the defaults as values in a dictionary with keys as the names of the
                      columns. If False, an ordered list is returned.

    Returns:
        coldefs, dict or list. If asdict, a dict is returned with column defaults as the values and columns
                                names as the keys. If False, a list of defaults in the same order as the names
                                returned from get_exposure_table_column_names().
    """
    columns = exposure_table_column_defs()

    if asdict:
        coldefs = {}
        for colname, coltype, coldef in columns:
            coldefs[colname] = coldef
    else:
        coldefs = []
        for colname, coltype, coldef in columns:
            coldefs.append(coldef)

    return coldefs

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
        extension, str. The extension (and therefore data format) without a leading period of the saved table.
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
        basedir = os.path.join(basedir, subdir)
    else:
        basedir = define_variable_from_environment(env_name='DESI_SPECTRO_LOG',
                                                   var_descr="The exposure table repository path")
    if night is None:
        return os.path.join(basedir,'exposure_tables')
    else:
        month = night_to_month(night)
        path = os.path.join(basedir,'exposure_tables',month)
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
    return os.path.join(path,table_name)

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

def validate_badamps(badamps,joinsymb=','):
    """
    Checks (and transforms) badamps string for consistency with the for need in an exposure or processing table
    for use in the Pipeline. Specifically ensure they come in (camera,petal,amplifier) sets,
    with appropriate checking of those values to make sure they're valid. Returns the input string
    except removing whitespace and replacing potential character separaters with joinsymb (default ',').
    Returns None if None is given.

    Args:
        badamps, str. A string of {camera}{petal}{amp} entries separated by symbol given with joinsymb (comma
                      by default). I.e. [brz][0-9][ABCD]. Example: 'b7D,z8A'.
        joinsymb, str. The symbol separating entries in the str list given by badamps.

    Returns:
        newbadamps, str. Input badamps string of {camera}{petal}{amp} entries separated by symbol given with
                      joinsymb (comma by default). I.e. [brz][0-9][ABCD]. Example: 'b7D,z8A'.
                      Differs from input in that other symbols used to separate terms are replaaced by joinsymb
                      and whitespace is removed.

    """
    if badamps is None:
        return badamps

    log = get_logger()
    ## Possible other joining symbols to automatically replace
    symbs = [';', ':', '|', '.', ',','-','_']

    ## Not necessary, as joinsymb would just be replaced with itself, but this is good better form
    if joinsymb in symbs:
        symbs.remove(joinsymb)

    ## Remove whitespace and replace possible joining symbols with the designated one.
    newbadamps = badamps.replace(' ', '').strip()
    for symb in symbs:
        newbadamps = newbadamps.replace(symb, joinsymb)

    ## test that the string can be parsed. Raises exception if it fails to parse
    throw = parse_badamps(newbadamps, joinsymb=joinsymb)

    ## Inform user of the result
    if badamps == newbadamps:
        log.info(f'Badamps given as: {badamps} verified to work')
    else:
        log.info(f'Badamps given as: {badamps} verified to work with modifications to: {newbadamps}') 
    return newbadamps

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
                                             If None, the default list in default_obstypes_for_exptable() is used.
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
    exp = int(exp)
    expstr = f'{exp:08d}'

    night = str(night)

    ## Give a header for the exposure
    if verbosely:
        log.info(f'\n\n###### Summarizing exposure: {exp} ######\n')
    else:
        log.info(f'Summarizing exposure: {exp}')

    ## Use defaults if things aren't defined
    if obstypes is None:
        obstypes = default_obstypes_for_exptable()

    ## Figure out what columns to fill from the data
    coldefault_dict = get_exposure_table_column_defaults(asdict=True)
    if colnames is None and coldefaults is not None:
        log.warning("Can't interpret coldefaults without corresponding colnames. Ignoring user specified coldefaults.")
    elif colnames is not None and coldefaults is not None:
        if len(colnames) != len(coldefaults):
            log.warning("Lengths of colnames and coldefaults must be equal. Ignoring user specified values.")
        else:
            log.info("Using user specified colnames and coldefaults.")
            coldefault_dict = {}
            for name,default in zip(colnames,coldefaults):
                coldefault_dict[name] = default
    elif colnames is not None:
        for name in colnames:
            if name not in coldefault_dict:
                log.warning(f"User specified {name} not in available colnames {coldefault_dict.keys()}.")
        for key in list(coldefault_dict.keys()):
            if key not in colnames:
                coldefault_dict.pop(key)

    ## Define the pathnames to the various data products
    ## TODO: tie these back in with desispec.io.meta
    manpath = os.path.join(raw_data_dir, night, expstr, f'manifest_{expstr}.json')
    reqpath = os.path.join(raw_data_dir, night, expstr, f'request-{expstr}.json')
    datpath = os.path.join(raw_data_dir, night, expstr, f'desi-{expstr}.fits.fz')
    etcpath = os.path.join(raw_data_dir, night, expstr, f'etc-{expstr}.json')

    ## If there is a manifest file, open it and see what it says
    if os.path.isfile(manpath):
        log.info(f'Found manifest file: {manpath}')
        ## Load the json file in as a dictionary
        manifest_dict = get_json_dict(manpath)
        if int(night) < 20200309:
            log.error(f"Manifest found on night {night} prior to invention of manifest. Expid: {exp}.")
        elif int(night) < 20200801:
            if 'PROGRAM' in manifest_dict:
                prog = manifest_dict['PROGRAM'].lower()
                if 'calib' in prog and 'done' in prog:
                    if 'short' in prog:
                        return "endofshortflats"
                    elif 'long' in prog:
                        return 'endofflats'
                    elif 'arc' in prog:
                        return 'endofarcs'
                else:
                    log.warning(f"Couldn't parse program name {prog} in manifest.")
        else:
            ## Starting Fall of 2020 the manifest should have standardized language, so no program parsing
            if 'MANIFEST' in manifest_dict:
                name = manifest_dict['MANIFEST'].lower()
                if name in ['endofarcs', 'endofflats', 'endofshortflats']:
                    return name
                elif name in ['endofzeros']:
                    log.info(f"Found {name} flag. Not using that information.")
                else:
                    log.error(f"Couldn't understand manifest name: {name}.")
            else:
                log.error(f"Couldn't find MANIFEST in manifest file. Available keys: {manifest_dict.keys()}.")

    ## Request json file can also be used to identify the end of calibrations
    ## It also has information on what we wanted the exposure to be, which is useful to check against what we got
    if os.path.isfile(reqpath):
        log.info(f"Found request file: {reqpath}")
        ## Load the json file in as a dictionary
        req_dict = get_json_dict(reqpath)
    else:
        log.error(f"Couldn't find request file: {reqpath}.")
        req_dict = {}

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

    ## Look for the data. If it's not there, say so then move on
    if not os.path.exists(datpath):
        if 'OBSTYPE' not in req_dict:
            logtype = log.error
        elif req_dict['OBSTYPE'].lower() in ['science','arc','flat']:
            logtype = log.error
        else:
            logtype = log.info
        if verbosely:
            logtype(f"Couldn't find {datpath} for exposure {exp}! Skipping")
        else:
            logtype(f'{exp}: skipped  -- data not found')
        return None
    else:
        log.info(f'Found raw data file: {datpath}')
        dat_header, fx = load_raw_data_header(pathname=datpath, return_filehandle=True)

    ## If FLAVOR is wrong or no obstype is defines, skip it
    if 'FLAVOR' not in dat_header:
        if verbosely:
            log.info(f'WARNING: {reqpath} -- flavor not given!')
        else:
            log.info(f'{exp}: skipped  -- flavor not given!')
        return None

    flavor = dat_header['FLAVOR'].lower()
    if flavor != 'science' and 'dark' not in obstypes and 'zero' not in obstypes:
        ## If FLAVOR is wrong
        if verbosely:
            log.info(f'ignoring: {reqpath} -- {flavor} not a flavor we care about')
        else:
            log.info(f'{exp}: skipped  -- {flavor} not a relevant flavor')
        return None

    if 'OBSTYPE' not in dat_header:
        ## If no obstype is defines, skip it
        if verbosely:
            log.info(f'ignoring: {reqpath} -- {flavor} flavor but obstype not defined')
        else:
            log.info(f'{exp}: skipped  -- obstype not given')
        return None

    ## If obstype isn't in our list of ones we care about, skip it
    obstype = dat_header['OBSTYPE'].lower()
    if obstype not in obstypes:
        ## If obstype is wrong
        if verbosely:
            log.info(f'ignoring: {reqpath} -- {obstype} not an obstype we care about')
        else:
            log.info(f'{exp}: skipped  -- {obstype} not relevant obstype')
        return None
    else:
        log.info(f"Exposure {exp} has obstype: {obstype}")

    ## Define the column values for the current exposure in a dictionary
    outdict = coldefault_dict.copy()

    ## Get the cameras available in the raw data and summarize with camword
    cams = cameras_from_raw_data(fx)
    outdict['CAMWORD'] = create_camword(cams)
    fx.close()

    ## Loop over columns and fill in the information. If unavailable report/flag if necessary and assign default
    for key,default in coldefault_dict.items():
        ## These are dealt with separately
        if key in ['EFFTIME_ETC', 'CAMWORD', 'NIGHT', 'SURVEY', 'FA_SURV', 'FAPRGRM', 'GOALTIME', 'GOALTYPE', 'SPEED',
                   'EBVFAC', 'LASTSTEP', 'BADCAMWORD', 'BADAMPS', 'EXPFLAG', 'HEADERERR', 'COMMENTS']:
            continue
        ## Try to find the key in the raw data header
        elif key in dat_header:
            val = dat_header[key]
            if isinstance(val, str):
                outdict[key] = val.lower().strip()
            else:
                outdict[key] = val
        ## If key not in the dat_header, identify that and place a default value
        ## If obstype isn't arc or flat, don't worry about seqnum or seqtot
        elif key in ['SEQNUM','SEQTOT'] and obstype not in ['arc','flat']:
            continue
        ## If tileid or TARGT and not science, just replace with default
        elif key in ['TILEID','TARGTRA','TARGTDEC'] and obstype not in ['science']:
            continue
        ## If trying to assign purpose and it's before that was defined, just give default
        elif key in ['PURPOSE'] and int(night) < 20201201:
            continue
        ## if something else, flag as missing metadata and replace with default
        else:
            log.warning(f"Missing header keyword: {key}. Using default {default}")
            if 'metadata_missing' not in outdict['EXPFLAG']:
                outdict['EXPFLAG'] = np.append(outdict['EXPFLAG'], 'metadata_missing')
            if np.isscalar(default):
                reporting = keyval_change_reporting(key, '', default)
                outdict['HEADERERR'] = np.append(outdict['HEADERERR'], reporting)

    ## Make sure that the night is defined:
    try:
        outdict['NIGHT'] = int(dat_header['NIGHT'])
    except (KeyError, ValueError, TypeError):
        log.error(f"int(dat_header['NIGHT']) failed for exp={exp}")
        if 'metadata_missing' not in outdict['EXPFLAG']:
            outdict['EXPFLAG'] = np.append(outdict['EXPFLAG'], 'metadata_missing')
        try:
            outdict['NIGHT'] = header2night(dat_header)
        except (KeyError, ValueError, TypeError):
            log.error(f"Even header2night failed for exp={exp}")
            outdict['NIGHT'] = night
        try:
            orig = str(dat_header['NIGHT'])
        except (KeyError, ValueError, TypeError):
            orig = ''
        reporting = keyval_change_reporting('NIGHT',orig,outdict['NIGHT'])
        outdict['HEADERERR'] = np.append(outdict['HEADERERR'],reporting)

    ## Verify we agree on what we're looking at
    if exp != outdict['EXPID']:
        log.error(f"Input exposure id doesn't match that derived from the header! {exp}!={outdict['EXPID']}")
    if int(night) != outdict['NIGHT']:
        log.error(f"Input night doesn't match that derived from the header! {night}!={outdict['NIGHT']}")

    ## For Things defined in both request and data, if they don't match, flag in the
    ##     output file for followup/clarity
    for check in ['OBSTYPE']:#, 'FLAVOR']:
        if check in req_dict and check in dat_header:
            rval, hval = req_dict[check], dat_header[check]
            if rval != hval:
                log.warning(f'In keyword {check}, request and data header disagree: req:{rval}\tdata:{hval}')
                if 'metadata_mismatch' not in outdict['EXPFLAG']:
                    outdict['EXPFLAG'] = np.append(outdict['EXPFLAG'], 'metadata_mismatch')
                outdict['COMMENTS'] = np.append(outdict['COMMENTS'],f'For {check}: req={rval} but hdu={hval}')
            else:
                if verbosely:
                    log.info(f'{check} checks out')
        else:
            if check not in dat_header:
                log.warning(f'{check} not found in header of exp {exp}')
            else:
                log.warning(f'{check} not found in request file of exp {exp}')

    ## Special logic for EXPTIME because of real-world variance on order 10's - 100's of ms
    # check = 'EXPTIME'
    # rval, hval = req_dict[check], dat_header[check]
    # if np.abs(float(rval)-float(hval))>0.5:
    #     log.warning(f'In keyword {check}, request and data header disagree: req:{rval}\tdata:{hval}')
    #     if 'aborted' not in outdict['EXPFLAG']:
    #         outdict['EXPFLAG'] = np.append(outdict['EXPFLAG'], 'aborted')
    #     outdict['COMMENTS'] = np.append(outdict['COMMENTS'],f'For {check}: req={rval} but hdu={hval}')
    # else:
    #     if verbosely:
    #         log.info(f'{check} checks out')

    ## Now for science exposures,
    if obstype == 'science':
        ## fiberassign used to be uncompressed, check the new format first but try old if necessary
        fbapath = os.path.join(raw_data_dir, night, expstr, f"fiberassign-{outdict['TILEID']:06d}.fits.gz")
        altfbapath = fbapath.replace('.fits.gz', '.fits')
        if not os.path.isfile(fbapath) and os.path.isfile(altfbapath):
            fbapath = altfbapath

        ## Load fiberassign file. If not available return empty dict
        if os.path.isfile(fbapath):
            log.info(f"Found fiberassign file: {fbapath}.")
            fba = fits.open(fbapath)
            extra_in_fba = ('EXTRA' in fba)
            fba_header = fba['PRIMARY'].header
            fba.close()
        else:
            log.error(f"Couldn't find fiberassign file: {fbapath}.")
            fba_header = {}
            extra_in_fba = False

        ## Add the fiber assign info. Try fiberassign file first, then raw data, then req
        for name in ["SURVEY","FA_SURV","FAPRGRM","GOALTIME","GOALTYPE","EBVFAC"]:
            for location in [fba_header,dat_header,req_dict]:
                if name in location:
                    val = location[name]
                    if isinstance(val,str):
                        val = val.lower().strip()
                    outdict[name] = val
                    break

        ## Load etc json file. If not available return empty dict
        if os.path.isfile(etcpath):
            log.info(f"Found etc file: {etcpath}.")
            etc_dict = get_json_dict(etcpath)
        else:
            log.warning(f"Couldn't find etc file: {etcpath}.")
            etc_dict = {}

        ## If EBVFAC wasn't found above, look in etc dict
        ## Default if both fail is 1 (already set)
        if outdict['EBVFAC'] == coldefault_dict['EBVFAC'] and 'fassign' in etc_dict:
            if 'EBVFAC' in etc_dict['fassign']:
                outdict['EBVFAC'] = etc_dict['fassign']['EBVFAC']
            elif 'MW_transp' in etc_dict['fassign']:
                outdict['EBVFAC'] = 1.0 / etc_dict['fassign']['MW_transp']

        ## Get EFFTIME from etc if available, then check in raw data.
        ## If ETCTEFF is then available and it can be transoformed to a float, use it
        ## And for data before June 2021, check for ACTTEFF.
        ## Default if all fail is -99 (already set)
        if 'expinfo' in etc_dict and 'efftime' in etc_dict['expinfo']:
            outdict['EFFTIME_ETC'] = etc_dict['expinfo']['efftime']
        elif 'ETCTEFF' in dat_header:
            try:
                outdict['EFFTIME_ETC'] = float(dat_header['ETCTEFF'])
            except:
                try:
                    orig = str(dat_header['ETCTEFF'])
                except:
                    orig = ''
                reporting = keyval_change_reporting('ETCTEFF', orig, outdict['EFFTIME_ETC'])
                outdict['HEADERERR'] = np.append(outdict['HEADERERR'], reporting)
                log.error(f"Couldn't convert ETCTEFF with value {orig} to float.")
        elif int(outdict['NIGHT']) < 20210614 and 'ACTTEFF' in dat_header:
            try:
                outdict['EFFTIME_ETC'] = float(dat_header['ACTTEFF'])
            except:
                try:
                    orig = str(dat_header['ACTTEFF'])
                except:
                    orig = ''
                reporting = keyval_change_reporting('ACTTEFF', orig, outdict['EFFTIME_ETC'])
                outdict['HEADERERR'] = np.append(outdict['HEADERERR'], reporting)
                log.error(f"Couldn't convert ACTTEFF with value {orig} to float.")

        ## Get the airmass factor from the etc. If unavailable, try to calculate from the airmass in the raw data
        ## Default if both fail is 1 (already set)
        if outdict['AIRMASS']==coldefault_dict['AIRMASS'] and 'expinfo' in etc_dict and 'AIRMASS' in etc_dict['expinfo']:
            outdict['AIRMASS'] = etc_dict['expinfo']['AIRMASS']

        ## If main survey data, report when varibles weren't available
        if int(night) > 20210500:
            for name in ["FA_SURV","FAPRGRM","GOALTIME","GOALTYPE",'AIRMASS','EBVFAC']:#,'EFFTIME_ETC']:
                if outdict[name] == coldefault_dict[name]:
                    log.warning(f"Couldn't find or derive {name}, so leaving {name} with default value " +
                                "of {outdict[name]}")

        if outdict['SURVEY'] == 'main':
            ## If defined, use effective speed. Otherwise set to very high value so we pass the relevant cuts
            if outdict['EFFTIME_ETC'] > 0.:
                efftime = outdict['EFFTIME_ETC']
            else:
                log.warning("No EFFTIME_ETC found. Not performing speed cut.")
                efftime = 1.0E5

            ## Define survey speed for QA
            ## Keep historical cuts accurate by only using new survey speed for exposures taken after 2021 shutdown
            ## Speed ref: https://desi.lbl.gov/trac/wiki/SurveyOps/SurveySpeed
            time_ratio = (efftime / outdict['EXPTIME'])
            ebvfac2 = outdict['EBVFAC'] ** 2
            if int(night) < 20210900:
                airfac2 = airmass_to_airfac(outdict['AIRMASS']) ** 2
                speed = time_ratio * ebvfac2 * airfac2
            else:
                aircorr = airmass_to_aircorr(outdict['AIRMASS'])
                speed = time_ratio * ebvfac2 * aircorr
            outdict['SPEED'] = speed

                    
        ## Flag the exposure based on PROGRAM information
        ## Define thresholds
        threshold_exptime = 60.
        if 'system test' in outdict['PROGRAM'].lower():
            outdict['LASTSTEP'] = 'ignore'
            outdict['EXPFLAG'] = np.append(outdict['EXPFLAG'], 'test')
            log.warning(f"LASTSTEP CHANGE. Exposure {exp} identified as system test. Not processing.")
        elif obstype == 'science' and 'undither' in outdict['PROGRAM']:
            outdict['LASTSTEP'] = 'fluxcal'
            log.warning(f"LASTSTEP CHANGE. Science exposure {exp} identified as undithered. Processing through " +
                        "flux calibration.")
            outdict['COMMENTS'] = np.append(outdict['COMMENTS'], 'undithered dither')
        elif obstype == 'science' and 'dither' in outdict['PROGRAM']:
            outdict['LASTSTEP'] = 'skysub'
            outdict['COMMENTS'] = np.append(outdict['COMMENTS'], 'dither')
            log.warning(f"LASTSTEP CHANGE. Science exposure {exp} identified as dither. Processing " +
                        "through sky subtraction.")
        ## Otherwise flag exposure based on "extra" hdu being in the fiberassign file
        elif extra_in_fba:
            outdict['LASTSTEP'] = 'skysub'
            outdict['COMMENTS'] = np.append(outdict['COMMENTS'], 'dither')
            log.warning(f"LASTSTEP CHANGE. Science exposure {exp} identified as dither. Processing " +
                        "through sky subtraction.")
        ## Otherwise check that the data meets quality standards
        ## Cut on signal:
        elif float(outdict['EXPTIME']) < threshold_exptime:
            outdict['LASTSTEP'] = 'skysub'
            outdict['EXPFLAG'] = np.append(outdict['EXPFLAG'], 'short_exposure')
            outdict['COMMENTS'] = np.append(outdict['COMMENTS'], f'EXPTIME={outdict["EXPTIME"]:.1f}s lt {threshold_exptime:.1f}')
            log.warning(f"LASTSTEP CHANGE. Science exposure {exp} with EXPTIME={outdict['EXPTIME']} less" +
                        f" than {threshold_exptime}s. Processing through sky subtraction.")
        elif outdict['SURVEY'] == 'main':
            ## If defined, use GOALTIME. Otherwise set to 0 so that we always pass the relevant cuts
            if outdict['GOALTIME'] > 0.:
                goaltime = outdict['GOALTIME']
            else:
                log.warning("No GOALTIME found. Not performing S/N cut.")
                goaltime = 0.

            ## If defined, use effective speed. Otherwise set to very high value so we pass the relevant cuts
            if outdict['EFFTIME_ETC'] > 0.:
                efftime = outdict['EFFTIME_ETC']
            else:
                log.warning("No EFFTIME_ETC found. Not performing speed cut.")
                efftime = 1.0E5

            ## Define thresholds
            threshold_percent_goal = 0.05
            threshold_speed_dark = 1/5.  # = 0.5*(1/2.5) = half the survey threshold
            threshold_speed_bright = 1/12.  # = 0.5*(1/6) = half the survey threshold
            threshold_efftime = threshold_percent_goal * goaltime  # 0.0 if GOALTIME not defined in headers

            threshold_speed = 0.
            if outdict['GOALTYPE'] == 'dark':
                threshold_speed = threshold_speed_dark
            elif outdict['GOALTYPE'] == 'bright':
                threshold_speed = threshold_speed_bright
            elif outdict['GOALTYPE'] == 'backup':
                pass
            elif outdict['GOALTYPE'] != coldefault_dict['GOALTYPE']:
                log.warning(f"Couldn't understand GOALTYPE={outdict['GOALTYPE']}")

            ## Perform the data quality cuts
            ## Cut on S/N:
            if efftime < threshold_efftime:
                outdict['LASTSTEP'] = 'skysub'
                outdict['EXPFLAG'] = np.append(outdict['EXPFLAG'], 'low_sn')
                outdict['COMMENTS'] = np.append(outdict['COMMENTS'], f'efftime={outdict["EFFTIME_ETC"]:.1f}s lt {threshold_efftime:.1f}')
                log.warning(f"LASTSTEP CHANGE. Science exposure {exp} with EFFTIME={outdict['EFFTIME_ETC']} " +
                            f"less than {threshold_percent_goal}% GOALTIME ({outdict['GOALTIME']}) = " +
                            f"{threshold_efftime:.4f}. Processing through sky subtraction.")
            ## Cut on Speed:
            elif speed < threshold_speed:
                outdict['LASTSTEP'] = 'skysub'
                outdict['EXPFLAG'] = np.append(outdict['EXPFLAG'], 'low_speed')
                outdict['COMMENTS'] = np.append(outdict['COMMENTS'], f'speed={speed:.4f} lt {threshold_speed:.4f}')
                log.warning(f"LASTSTEP CHANGE. Science exposure {exp} with speed={speed:.4f} less than threshold " +
                            f"speed={threshold_speed:.4f}. Processing through sky subtraction.")

    log.info(f'Done summarizing exposure: {exp}')
    return outdict

def airfac_to_airmass(airfac, k=0.114):
    """
    Transforms an "AIRFAC" term of survey speed to airmass:
    AIRFAC = 10^[k*(X-1)/2.5]
    https://desi.lbl.gov/trac/wiki/SurveyOps/SurveySpeed
    """
    X = 1+((2.5/k)*np.log10(airfac))
    return X

def airmass_to_airfac(airmass, k=0.114):
    """
    Transforms an airmass to "AIRFAC":
    AIRFAC = 10^[k*(X-1)/2.5]
    https://desi.lbl.gov/trac/wiki/SurveyOps/SurveySpeed
    """
    airfac = 10**(k*(airmass-1)/2.5)
    return airfac

def airmass_to_aircorr(airmass):
    """
    Transforms an airmass to "air correction" term of survey speed:
    AIRCORR = X^1.75
    https://desi.lbl.gov/trac/wiki/SurveyOps/SurveySpeed
    """
    aircorr = np.power(airmass,1.75)
    return aircorr

def aircorr_to_airmass(aircorr):
    """
    Transforms an "air correction" term of survey speed to airmass:
    AIRCORR = X^1.75
    https://desi.lbl.gov/trac/wiki/SurveyOps/SurveySpeed
    """
    airmass = np.power(aircorr,1/1.75)
    return airmass

def airfac_to_aircorr(airfac):
    """
    Transforms an "AIRFAC" term of survey speed to an "air correction" term of survey speed
    https://desi.lbl.gov/trac/wiki/SurveyOps/SurveySpeed
    """
    return airmass_to_aircorr(airfac_to_airmass(airfac))
