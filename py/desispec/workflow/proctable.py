#!/usr/bin/env python
# coding: utf-8


import numpy as np
import os

from astropy.table import Table, vstack
from collections import OrderedDict

## Import some helper functions, you can see their definitions by uncomenting the bash shell command
from desispec.workflow.exptable import default_exptypes_for_exptable

from desispec.workflow.utils import pathjoin

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
        coldeflts, list. Optionally returned if return_default_values is True. List of default values for the
                         corresponding colnames.
    """
    ## Define the column names for the internal production table and their respective datatypes, split in two
    ##     only for readability's sake

    colnames1 = ['EXPID'              , 'NIGHT' , 'OBSTYPE', 'CAMWORD'    , 'TILEID', 'CALIBRATOR']
    coltypes1 = [np.ndarray           , int     , 'S10'    , 'S40'        , int     , np.int8     ]
    coldefault1 = [np.ndarray(shape=0), 20000101, 'unknown', 'a0123456789', -99     , 0           ]

    colnames2 = ['INTID', 'OBSDESC', 'SCRIPTNAME', 'JOBDESC', 'LATEST_QID', 'SUBMIT_DATE', 'STATUS']
    coltypes2 = [int    , 'S16'    , 'S40'       , 'S12'    , int         ,  int        , 'S10'   ]
    coldefault2 = [-99  , 'unknown', 'unknown'   , 'unknown', -99         , -99      , 'U'     ]

    colnames3 =   ['INT_DEP_IDS'                  , 'LATEST_DEP_QID'               , 'ALL_QIDS'                     ]
    coltypes3 =   [np.ndarray                     , np.ndarray                     , np.ndarray                     ]
    coldefault3 = [np.ndarray(shape=0).astype(int), np.ndarray(shape=0).astype(int), np.ndarray(shape=0).astype(int)]

    #     colnames1 = ['INTERNAL_ID', 'EXPID'    , 'NIGHT', 'INT_DEP_IDS', 'JOBNAME', 'TYPE']
    #     coltypes1 = [ int         ,  list      ,  int    , list        , 'S30'    , 'S10']

    #     colnames2 = ['LATEST_QID', 'LAST_SUBMIT_DATE', 'STATUS', 'LATEST_DEP_QID', 'ALL_QIDS']
    #     coltypes2 = [ int        , 'S20'             , 'S10'   ,  list           ,  list      ]

    colnames = colnames1 + colnames2 + colnames3
    coldtypes = coltypes1 + coltypes2 + coltypes3
    coldefaults = coldefault1 + coldefault2 + coldefault3

    if return_default_values:
        if overlap_only:
            return colnames1, coltypes1, coldefault1
        elif unique_only:
            return (colnames2 + colnames3), (coltypes2 + coltypes3), (coldefault2 + coldefault3)
        else:
            return colnames, coldtypes, coldefaults
    else:
        if overlap_only:
            return colnames1, coltypes1
        elif unique_only:
            return (colnames2 + colnames3), (coltypes2 + coltypes3)
        else:
            return colnames, coldtypes

def default_exptypes_for_proctable():
    """
    Defines the exposure types to be recognized by the workflow and saved in the processing table by default.

    Returns:
        list. A list of default obstypes to be included in a processing table.
    """
    ## Define the science types to be included in the exposure table (case insensitive)
    return ['arc','flat','twilight','science','sci','dither']

def get_processing_table_name(prodname=None, prodmod=None, extension='csv'):
    """
    Defines the default processing name given the prodname of the production and the optional extension.

    Args:
        prodname, str or None. The name of the production. If None, it will be taken from the environment variable.
        prodmod, str. Additional str that can be added to the production table name to further differentiate it.
                      Used in daily workflow to add the night to the name and make it unique from other nightly tables.
        extension, str. The extension (and therefore data format) without a leading period of the saved table.
                        Default is 'csv'.

    Returns:
        str. The processing table name given the input night and extension.
    """
    if prodname is None and 'SPECPROD' in os.environ:
        prodname = os.environ['SPECPROD']
    else:
        prodname = 'unknown'

    if prodmod is not None:
        prodname_modifier = '-' + str(prodmod)
    elif 'SPECPROD_MOD' in os.environ:
        prodname_modifier = '-' + os.environ['SPECPROD_MOD']
    else:
        prodname_modifier = ''

    return f'processing_table_{prodname}{prodname_modifier}.{extension}'


def get_processing_table_path(prodname=None):
    """
    Defines the default path to save a processing table. If prodname is not given, the environment variable
    'SPECPROD' must exist.

    Args:
        prodname, str or None. The name of the production. If None, it will be taken from the environment variable.

    Returns:
         str. The full path to the directory where the processing table should be written (or is already written). This
              does not including the filename.
    """
    if prodname is None and 'SPECPROD' in os.environ:
        prodname = os.environ['SPECPROD']
    else:
        prodname = 'unknown'

    path = pathjoin(os.environ['DESI_SPECTRO_REDUX'], prodname, 'processing_tables')
    return path


def get_processing_table_pathname(prodname=None, prodmod=None, extension='csv'):  # base_path,prodname
    """
    Defines the default pathname to save a processing table.

    Args:
        prodname, str or None. The name of the production. If None, it will be taken from the environment variable.
        prodmod, str. Additional str that can be added to the production table name to further differentiate it.
                      Used in daily workflow to add the night to the name and make it unique from other nightly tables.
        extension, str. The extension (and therefore data format) without a leading period of the saved table.
                        Default is 'csv'.

    Returns:
         str. The full pathname where the processing table should be written (or is already written). This
              includes the filename.
    """
    if prodname is None and 'SPECPROD' in os.environ:
        prodname = os.environ['SPECPROD']
    else:
        prodname = 'unknown'

    path = get_processing_table_path(prodname)
    table_name = get_processing_table_name(prodname, prodmod, extension)
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
        processing_table, Table. The output processing table. Each row corresponds with an exposure that should be
                                 processed.
        unprocessed_table, Table. The output unprocessed table. Each row is an exposure that should not be processed.
    """
    exptable = input_exptable.copy()

    if obstypes is None:
        obstypes = default_exptypes_for_exptable()

    ## Define the column names for the exposure table and their respective datatypes
    colnames, coldtypes, coldefaults = get_processing_table_column_defs(return_default_values=True)

    for col in ['HEADERERR', 'COMMENTS']:
        if col in exptable.colnames:
            for ii, arr in enumerate(exptable[col]):
                for item in arr:
                    clean_item = item.strip(' \t')
                    if len(clean_item) > 6:
                        keyval = None
                        for symb in [':', '=']:
                            if symb in clean_item:
                                keyval = [val.strip(' ') for val in clean_item.split(symb)]
                                break
                        if keyval is not None and len(keyval) == 2 and keyval[0].upper() in exptable.colnames:
                            key, newval = keyval[0].upper(), keyval[1]
                            expid, oldval = exptable['EXPID'][ii], exptable[key][ii]
                            print(
                                f'Found a requested correction to ExpID {expid}: Changing {key} val from {oldval} to {newval}')
                            exptable[key][ii] = newval

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

    rows = []
    for erow in good_table:
        prow = erow_to_prow(erow, colnames, coldtypes, coldefaults)
        rows.append(prow)
    processing_table = Table(names=colnames, dtype=coldtypes, rows=rows)

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
    if type(erow) in [dict, OrderedDict]:
        ecols = erow.keys()
    else:
        ecols = erow.colnames

    ## Define the column names for the exposure table and their respective datatypes
    #if colnames is None:
    colnames, coldtypes, coldefaults = get_processing_table_column_defs(return_default_values=True)
    colnames, coldtypes, coldefaults = np.array(colnames), np.array(coldtypes), np.array(coldefaults)

    prow = dict()
    for nam, typ, defval in zip(colnames, coldtypes, coldefaults):
        if nam == 'OBSDESC':
            if nam in colnames:
                prow[nam] = coldefaults[colnames == nam][0]
            else:
                prow[nam] = 'unknown'
            for word in ['dither', 'acquisition', 'focus']:
                if 'PROGRAM' in ecols and word in erow['PROGRAM'].lower():
                    prow[nam] = word
        elif nam == 'EXPID':
            prow[nam] = np.array([erow[nam]])
        elif nam in ecols:
            prow[nam] = erow[nam]
        else:
            prow[nam] = defval
    return prow

def erow_to_prow_with_overrides(input_erow):#, colnames=None, coldtypes=None, coldefaults=None):
    """
    Converts an exposure table row to a processing table row. The columns unique to a processing table
    are filled with default values. If comments are made in COMMENTS or HEADERERR, those will be adjusted in the values
    stored in the processing table row.

    Args:
        input_erow, Table.Row or dict. An exposure table row. The row will be converted to a row of an processing table.
                                       If comments are made in COMMENTS or HEADERERR, those will be adjusted in
                                       the values stored in the processing table.

    Returns:
        prow, dict. The output processing table row.
    """
    erow = input_erow.copy()
    if type(erow) in [dict, OrderedDict]:
        ecols = erow.keys()
    else:
        ecols = erow.colnames

    for col in ['HEADERERR', 'COMMENTS']:
        if col in ecols:
            for item in erow[col]:
                clean_item = item.strip(' \t')
                if len(clean_item) > 6:
                    keyval = None
                    for symb in [':', '=']:
                        if symb in clean_item:
                            keyval = [val.strip(' ') for val in clean_item.split(symb)]
                            break
                    if keyval is not None and len(keyval) == 2 and keyval[0].upper() in ecols:
                        key, newval = keyval[0].upper(), keyval[1]
                        expid, oldval = erow['EXPID'], erow[key]
                        print(
                            f'Found a requested correction to ExpID {expid}: Changing {key} val from {oldval} to {newval}')
                        erow[key] = newval

    ## Define the column names for the exposure table and their respective datatypes
    # if colnames is None:
    colnames, coldtypes, coldefaults = get_processing_table_column_defs(return_default_values=True)
    colnames, coldtypes, coldefaults = np.array(colnames), np.array(coldtypes), np.array(coldefaults)

    prow = dict()
    for nam, typ, defval in zip(colnames, coldtypes, coldefaults):
        if nam == 'OBSDESC':
            if nam in colnames:
                prow[nam] = coldefaults[colnames == nam][0]
            else:
                prow[nam] = 'unknown'
            for word in ['dither', 'acquisition', 'focus']:
                if 'PROGRAM' in ecols and word in erow['PROGRAM'].lower():
                    prow[nam] = word
        elif nam == 'EXPID':
            prow[nam] = np.array([erow[nam]])
        elif nam in ecols:
            prow[nam] = erow[nam]
        else:
            prow[nam] = defval
    return prow