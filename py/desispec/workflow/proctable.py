#!/usr/bin/env python
# coding: utf-8


import numpy as np
import os

from astropy.table import Table, vstack
from collections import OrderedDict

## Import some helper functions, you can see their definitions by uncomenting the bash shell command
from desispec.workflow.exptable import default_exptypes_for_exptable

from desispec.workflow.os import pathjoin

def get_processing_table_column_defs(return_default_values=False, overlap_only=False, unique_only=False):
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
    ## Define the science types to be included in the exposure table (case insensitive)
    return ['arc','flat','twilight','science','sci','dither']

def exptable_to_proctable(input_exptable, science_types=None, addtnl_colnames=None, addtnl_coldtypes=None,
                          joinsymb='|'):
    exptable = input_exptable.copy()

    if science_types is None:
        science_types = default_exptypes_for_exptable()

    ## Define the column names for the exposure table and their respective datatypes
    if addtnl_colnames is None:
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
    good_types = np.array([val in science_types for val in exptable['OBSTYPE']]).astype(bool)
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

    # for nam, typ, default in zip(colnames, coldtypes, coldefaults):
    #     if typ is str or type(typ) is str:
    #         data = np.array([joinsymb] * len(processing_table))
    #     else:
    #         data = np.zeros(len(processing_table)).astype(typ)
    #     processing_table.add_column(Table.Column(name=nam, dtype=typ, data=data))

    return processing_table, unprocessed_table


def instantiate_processing_table(colnames=None, coldtypes=None):
    ## Define the column names for the exposure table and their respective datatypes
    if colnames is None:
        colnames, coldtypes = get_processing_table_column_defs()
    processing_table = Table(names=colnames, dtype=coldtypes)
    return processing_table


def erow_to_prow_with_overrides(input_erow, colnames=None, coldtypes=None, coldefaults=None, joinsymb='|'):
    erow = input_erow.copy()
    if type(erow) in [dict, OrderedDict]:
        ecols = erow.keys()
    else:
        ecols = erow.colnames

    for col in ['HEADERERR', 'COMMENTS']:
        if col in ecols:
            for item in erow.split(joinsymb):
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
    if colnames is None:
        colnames, coldtypes, coldefaults = get_processing_table_column_defs(return_default_values=True)

    prow = OrderedDict()
    for nam, typ, defval in zip(colnames, coldtypes, coldefaults):
        if nam == 'OBSDESC':
            prow[nam] = ' '
            for word in ['dither', 'acquisition', 'focus']:
                if 'PROGRAM' in ecols and word in erow['PROGRAM'].lower():
                    prow[nam] = word
        elif nam in ecols:
            prow[nam] = erow[nam]
        else:
            prow[nam] = defval
    return prow


def erow_to_prow(erow, colnames=None, coldtypes=None, coldefaults=None, joinsymb='|'):
    if type(erow) in [dict, OrderedDict]:
        ecols = erow.keys()
    else:
        ecols = erow.colnames

    ## Define the column names for the exposure table and their respective datatypes
    if colnames is None:
        colnames, coldtypes, coldefaults = get_processing_table_column_defs(return_default_values=True)

    colnames, coldtypes, coldefaults = np.array(colnames), np.array(coldtypes), np.array(coldefaults)
    prow = OrderedDict()
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


def get_processing_table_name(prodname=None, prodmod=None, extension='csv'):
    if prodname is None and 'SPECPROD' in os.environ:
        prodname = os.environ['SPECPROD']
    else:
        prodname = 'unknown'

    if prodmod is not None:
        prodname_modifier = '-' + str(prodmod)
    elif 'SPECPROD_MOD' in os.environ:
        prodname_modifier = '-' + os.environ['SPECPROD_MOD']
    # elif prodname == 'daily' and 'PROD_NIGHT' in os.environ:
    #     prodname_modifier = '-' + os.environ['PROD_NIGHT']
    else:
        prodname_modifier = ''

    return f'processing_table_{prodname}{prodname_modifier}.{extension}'


def get_processing_table_path(prodname=None):
    if prodname is None:
        prodname = os.environ['SPECPROD']
    path = pathjoin(os.environ['DESI_SPECTRO_REDUX'], prodname, 'processing_tables')
    return path


def get_processing_table_pathname(prodname=None, prodmod=None, extension='csv'):  # base_path,prodname
    if prodname is None:
        prodname = os.environ['SPECPROD']

    path = get_processing_table_path(prodname)
    table_name = get_processing_table_name(prodname, prodmod, extension)
    return pathjoin(path, table_name)





# def get_proctab_uniq_columns():
#     colnames =  ['LATEST_JOBID', 'LATEST_JOBFILE', 'JOBFILES', 'JOBIDS', 'REDUX_STATUS']
#     coldtypes = [int           , 'S20'           , 'S80'     , 'S30'   , int]
#     return colnames, coldtypes



