"""
desispec.workflow.proctable
===========================

Please add module-level documentation.
"""

import numpy as np
import os

from astropy.table import Table, vstack
from collections import OrderedDict

## Import some helper functions, you can see their definitions by uncomenting the bash shell command
from desispec.workflow.exptable import default_obstypes_for_exptable

from desispec.workflow.utils import define_variable_from_environment, pathjoin
from desispec.io.util import difference_camwords, parse_badamps, create_camword, decode_camword
from desiutil.log import get_logger

###############################################
##### Processing Table Column Definitions #####
###############################################
## To eventually being turned into a full-fledged data model. For now a brief description.
# EXPID, int, the exposure ID's assosciate with the job. Always a np.array, even if a single exposure.
# OBSTYPE, string, the obstype as defined by ICS.
# TILEID, int, the TILEID of the tile the exposure observed.
# NIGHT, int, the night of the observation.
# BADAMPS, string, comma list of "{camera}{petal}{amp}", i.e. "[brz][0-9][ABCD]". Example: 'b7D,z8A'
#                  in the csv this is saved as a semicolon separated list
# LASTSTEP, string, the last step the pipeline should run through for the given exposure. Inclusive of last step.
# EXPFLAG, np.ndarray, set of flags that describe that describe the exposure.
# PROCCAMWORD, string, The result of difference_camword(CAMWORD,BADCAMWWORD) from those exposure table entries.
#                      This summarizes the cameras that should be processed for the given exposure/job
# CALIBRATOR, int, A 0 signifies that the job is not assosciated with a calibration exposure. 1 means that it is.
# INTID, int, an internally generated ID for a single job within a production. Only unique within a production and
#             not guaranteed will not necessarily be the same between different production runs (e.g. between a daily
#             run and a large batch reprocessing run).
# OBSDESC, string, describes the observation in more detail than obstype. Currently only used for DITHER on dither tiles.
# JOBDESC, string, described the job that the row defines. For a single science exposure that could be 'prestdstar' or
#                  'poststdstar'. For joint science that would be 'stdstarfit'. For individual arcs it is 'arc', for
#                  joint arcs it is 'psfnight'. For individual flats it is 'flat', for joint fits it is 'psfnightly'.
# LATEST_QID, int, the most recent Slurm ID assigned to the submitted job.
# SUBMIT_DATE, int, the 'unix time' of the job submission in seconds (int(time.time())).
# STATUS, string, the most recent Slurm status of the job. See docstring of desispec.workflow.queue.get_resubmission_states
#                 for a list and description.
# SCRIPTNAME, string, the name of the script submitted to Slurm. Due to astropy table constraints, this is truncated
#                     to a maximum of 40 characters.
# INT_DEP_IDS, np.array, internal ID's of all jobs that are dependencies for the current row. I.e. inputs to the current job.
# LATEST_DEP_QID,  np.array, the most recent Slurm ID's for the dependencies jobs uniquely identified by internal ID's
#                            in INT_DEP_IDS
# ALL_QIDS, np.array, a list of all Slurm ID's assosciated with submissions of this job. Useful if multiple submissions
#                     were made because of node failures or any other issues that were later resolved (or not resolved).
##################################################

def get_processing_table_column_defs(return_default_values=False, overlap_only=False, unique_only=False):
    """
    Contains the column names, data types, and default row values for a DESI processing table. It returns
    the names and datatypes with the defaults being given with an optional flag. Returned as 2 (or 3) lists.

    Args:
        return_default_values, bool. True if you want the default values returned.
        overlap_only, bool. Only return the columns that are common to both processing and exposure tables.
        unique_only, bool. Only return columns that are not found in an exposure table.

    Returns:
        colnames, list. List of column names for an processing table.
        coldtypes, list. List of column datatypes for the names in colnames.
        coldeflts, list. Optionally returned if return_default_values is True. List of default values for the corresponding colnames.
    """
    ## Define the column names for the internal production table and their respective datatypes, split in two
    ##     only for readability's sake

    colnames1 = ['EXPID'                        , 'OBSTYPE', 'TILEID', 'NIGHT' ]
    coltypes1 = [np.ndarray                     , 'S10'    , int     , int     ]
    coldeflt1 = [np.ndarray(shape=0).astype(int), 'unknown', -99     , 20000101]

    colnames1 += ['BADAMPS', 'LASTSTEP', 'EXPFLAG'               ]
    coltypes1 += ['S30'    , 'S30'     ,  np.ndarray             ]
    coldeflt1 += [''       , 'all'     ,  np.array([], dtype=str)]

    colnames2 = [ 'PROCCAMWORD'    ,'CALIBRATOR', 'INTID', 'OBSDESC', 'JOBDESC', 'LATEST_QID']
    coltypes2 = [ 'S40'            , np.int8    ,  int   , 'S16'    , 'S12'    , int         ]
    coldeflt2 = [ 'a0123456789'    , 0          ,  -99   , ''       , 'unknown', -99         ]

    colnames2 += [ 'SUBMIT_DATE', 'STATUS', 'SCRIPTNAME']
    coltypes2 += [  int         , 'S14'   , 'S40'       ]
    coldeflt2 += [ -99          , 'U'     , ''   ]

    colnames2 += ['INT_DEP_IDS'                  , 'LATEST_DEP_QID'               , 'ALL_QIDS'                     ]
    coltypes2 += [np.ndarray                     , np.ndarray                     , np.ndarray                     ]
    coldeflt2 += [np.ndarray(shape=0).astype(int), np.ndarray(shape=0).astype(int), np.ndarray(shape=0).astype(int)]

    colnames = colnames1 + colnames2
    coldtypes = coltypes1 + coltypes2
    coldeflts = coldeflt1 + coldeflt2

    if return_default_values:
        if overlap_only:
            return colnames1, coltypes1, coldeflt1
        elif unique_only:
            return colnames2, coltypes2, coldeflt2
        else:
            return colnames, coldtypes, coldeflts
    else:
        if overlap_only:
            return colnames1, coltypes1
        elif unique_only:
            return colnames2, coltypes2
        else:
            return colnames, coldtypes

def default_obstypes_for_proctable():
    """
    Defines the exposure types to be recognized by the workflow and saved in the processing table by default.

    Returns:
        list. A list of default obstypes to be included in a processing table.
    """
    ## Define the science types to be included in the exposure table (case insensitive)
    return ['bias', 'dark', 'arc', 'flat', 'science', 'twilight', 'sci', 'dither']

def get_processing_table_name(specprod=None, prodmod=None, extension='csv'):
    """
    Defines the default processing name given the specprod of the production and the optional extension.

    Args:
        specprod, str or None. The name of the production. If None, it will be taken from the environment variable.
        prodmod, str. Additional str that can be added to the production table name to further differentiate it.
            Used in daily workflow to add the night to the name and make it unique from other nightly tables.
        extension, str. The extension (and therefore data format) without a leading period of the saved table.
            Default is 'csv'.

    Returns:
        str. The processing table name given the input night and extension.
    """
    if specprod is None:
        specprod = define_variable_from_environment(env_name='SPECPROD',
                                                    var_descr="Use SPECPROD for unique processing table directories")

    if prodmod is not None:
        prodname_modifier = '-' + str(prodmod)
    elif 'SPECPROD_MOD' in os.environ:
        prodname_modifier = '-' + os.environ['SPECPROD_MOD']
    else:
        prodname_modifier = ''

    return f'processing_table_{specprod}{prodname_modifier}.{extension}'


def get_processing_table_path(specprod=None):
    """
    Defines the default path to save a processing table. If specprod is not given, the environment variable
    'SPECPROD' must exist.

    Args:
        specprod, str or None. The name of the production. If None, it will be taken from the environment variable.

    Returns:
         str. The full path to the directory where the processing table should be written (or is already written). This
              does not including the filename.
    """
    if specprod is None:
        specprod = define_variable_from_environment(env_name='SPECPROD',
                                                    var_descr="Use SPECPROD for unique processing table directories")

    basedir = define_variable_from_environment(env_name='DESI_SPECTRO_REDUX',
                                                  var_descr="The specprod path")
    path = pathjoin(basedir, specprod, 'processing_tables')
    return path


def get_processing_table_pathname(specprod=None, prodmod=None, extension='csv'):  # base_path,specprod
    """
    Defines the default pathname to save a processing table.

    Args:
        specprod, str or None. The name of the production. If None, it will be taken from the environment variable.
        prodmod, str. Additional str that can be added to the production table name to further differentiate it.
            Used in daily workflow to add the night to the name and make it unique from other nightly tables.
        extension, str. The extension (and therefore data format) without a leading period of the saved table.
            Default is 'csv'.

    Returns:
         str. The full pathname where the processing table should be written (or is already written). This
              includes the filename.
    """
    if specprod is None:
        specprod = define_variable_from_environment(env_name='SPECPROD',
                                                    var_descr="Use SPECPROD for unique processing table directories")

    path = get_processing_table_path(specprod)
    table_name = get_processing_table_name(specprod, prodmod, extension)
    return pathjoin(path, table_name)

def instantiate_processing_table(colnames=None, coldtypes=None, rows=None):
    """
    Create an empty processing table with proper column names and datatypes. If rows is given, it inserts the rows
    into the table, otherwise it returns a table with no rows.

    Args:
        colnames, list. List of column names for a procesing table.
        coldtypes, list. List of column datatypes for the names in colnames.
        rows, list or np.array of Table.Rows or dicts. An iterable set of Table.Row's or dicts with keys/colnames and value
            pairs that match the default column names and data types of the
            default exposure table.

    Returns:
        processing_table, Table. An astropy Table with the column names and data types for a DESI workflow processing
            table. If the input rows was not None, it contains those rows, otherwise it has no rows.
    """
    ## Define the column names for the exposure table and their respective datatypes
    if colnames is None or coldtypes is None:
        colnames, coldtypes = get_processing_table_column_defs()

    processing_table = Table(names=colnames, dtype=coldtypes)
    if rows is not None:
        for row in rows:
            processing_table.add_row(row)
    return processing_table

def exptable_to_proctable(input_exptable, obstypes=None):
    """
    Converts an exposure table to a processing table and an unprocessed table. The columns unique to a processing table
    are filled with default values. If comments are made in COMMENTS or HEADERERR, those will be adjusted in the values
    stored in the processing table.

    Args:
        input_exptable, Table. An exposure table. Each row will be converted to a row of an processing table. If
            comments are made in COMMENTS or HEADERERR, those will be adjusted in the values
            stored in the processing table.
        obstypes, list or np.array. Optional. A list of exposure OBSTYPE's that should be processed (and therefore
            added to the processing table).

    Returns:
        processing_table, Table. The output processing table. Each row corresponds with an exposure that should be processed.
        unprocessed_table, Table. The output unprocessed table. Each row is an exposure that should not be processed.

    """
    log = get_logger()
    exptable = input_exptable.copy()

    if obstypes is None:
        obstypes = default_obstypes_for_exptable()

    ## Define the column names for the exposure table and their respective datatypes
    colnames, coldtypes, coldefaults = get_processing_table_column_defs(return_default_values=True)

    # for col in ['COMMENTS']: #'HEADERERR',
    #     if col in exptable.colnames:
    #         for ii, arr in enumerate(exptable[col]):
    #             for item in arr:
    #                 clean_item = item.strip(' \t')
    #                 if len(clean_item) > 6:
    #                     keyval = None
    #                     for symb in [':', '=']:
    #                         if symb in clean_item:
    #                             keyval = [val.strip(' ') for val in clean_item.split(symb)]
    #                             break
    #                     if keyval is not None and len(keyval) == 2 and keyval[0].upper() in exptable.colnames:
    #                         key, newval = keyval[0].upper(), keyval[1]
    #                         expid, oldval = exptable['EXPID'][ii], exptable[key][ii]
    #                         log.info(
    #                             f'Found a requested correction to ExpID {expid}: Changing {key} val from {oldval} to {newval}')
    #                         exptable[key][ii] = newval

    good_exps = (exptable['EXPFLAG'] == 0)
    good_types = np.array([val in obstypes for val in exptable['OBSTYPE']]).astype(bool)
    good = (good_exps & good_types)
    good_table = exptable[good]
    unprocessed_table = exptable[~good]

    ## Remove columns that aren't relevant to processing, they will be added back in the production tables for
    ## end user viewing
    for col in ['REQRA', 'REQDEC', 'TARGTRA', 'TARGTDEC', 'HEADERERR', 'COMMENTS', 'BADEXP']:
        if col in exptable.colnames:
            good_table.remove_column(col)

    if len(good_table) > 0:
        rows = []
        for erow in good_table:
            prow = erow_to_prow(erow)#, colnames, coldtypes, coldefaults)
            rows.append(prow)
        processing_table = Table(names=colnames, dtype=coldtypes, rows=rows)
    else:
        processing_table = Table(names=colnames, dtype=coldtypes)

    return processing_table, unprocessed_table

def erow_to_prow(erow):#, colnames=None, coldtypes=None, coldefaults=None, joinsymb='|'):
    """
    Converts an exposure table row to a processing table row. The columns unique to a processing table
    are filled with default values. If comments are made in COMMENTS or HEADERERR, those are ignored.

    Args:
        erow, Table.Row or dict. An exposure table row. The row will be converted to a row of an processing table.
            If comments are made in COMMENTS or HEADERERR, those are ignored.

    Returns:
        prow, dict. The output processing table row.
    """
    log = get_logger()
    erow = table_row_to_dict(erow)
    row_names = list(erow.keys())

    ## Define the column names for the exposure table and their respective datatypes
    #if colnames is None:
    colnames, coldtypes, coldefaults = get_processing_table_column_defs(return_default_values=True)
    colnames, coldtypes, coldefaults = np.array(colnames,dtype=object), \
                                       np.array(coldtypes,dtype=object), \
                                       np.array(coldefaults,dtype=object)

    prow = dict()
    for nam, typ, defval in zip(colnames, coldtypes, coldefaults):
        if nam == 'PROCCAMWORD':
            if 'BADCAMWORD' in row_names:
                badcamword = erow['BADCAMWORD']
            else:
                badcamword = ''
            prow[nam] = difference_camwords(erow['CAMWORD'],badcamword)
        elif nam == 'OBSDESC':
            if nam in colnames:
                prow[nam] = coldefaults[colnames == nam][0]
            else:
                prow[nam] = ''
            for word in ['dither', 'acquisition', 'focus', 'test']:
                if 'PROGRAM' in row_names and word in erow['PROGRAM'].lower():
                    prow[nam] = word
        elif nam == 'EXPID':
            prow[nam] = np.array([erow[nam]])
        elif nam in row_names:
            prow[nam] = erow[nam]
        else:
            prow[nam] = defval

    ## For obstypes that aren't science, BADAMPS loses it's relevance. For processing,
    ## convert those into bad cameras in BADCAMWORD, so the cameras aren't processed.
    ## Otherwise we'll have nightly calibrations with only half the fibers useful.
    if prow['OBSTYPE'] != 'science' and prow['BADAMPS'] != '':
        badcams = []
        for (camera, petal, amplifier) in parse_badamps(prow['BADAMPS']):
            badcams.append(f'{camera}{petal}')
        newbadcamword = create_camword(badcams)
        log.info("For nonsscience exposure: {}, converting BADAMPS={} to bad cameras={}.".format( erow['EXPID'],
                                                                                                  prow['BADAMPS'],
                                                                                                  newbadcamword    ) )
        prow['PROCCAMWORD'] = difference_camwords(prow['PROCCAMWORD'],newbadcamword)
        prow['BADAMPS'] = ''

    return prow

def default_prow():
    """
    Creates a processing table row. The columns are filled with default values.

    Args:
        None

    Returns:
        prow, dict. The output processing table row.
    """
    ## Define the column names for the exposure table and their respective datatypes
    #if colnames is None:
    colnames, coldtypes, coldefaults \
        = get_processing_table_column_defs(return_default_values=True)
    colnames = np.array(colnames,dtype=object)
    coldefaults = np.array(coldefaults,dtype=object)

    prow = dict()
    for nam, defval in zip(colnames, coldefaults):
        prow[nam] = defval
    return prow

def table_row_to_dict(table_row):
    """
    Helper function to convert a table row to a dictionary, which is much easier to work with for some applications

    Args:
        table_row, Table.Row or dict. The row of an astropy table that you want to convert into a dictionary where
            each key is a column name and the values are the column entry.

    Returns:
        out, dict. Dictionary where each key is a column name and the values are the column entry.
    """
    if type(table_row) is Table.Row:
        out = {coln: table_row[coln] for coln in table_row.colnames}
        return out
    elif type(table_row) in [dict, OrderedDict]:
        return table_row
    else:
        log = get_logger()
        typ = type(table_row)
        log.error(f"Received table_row of type {typ}, can't convert to a dictionary. Exiting.")
        raise TypeError(f"Received table_row of type {typ}, can't convert to a dictionary. Exiting.")
