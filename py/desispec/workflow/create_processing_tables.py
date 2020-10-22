#!/usr/bin/env python
# coding: utf-8


import numpy as np
import os

from astropy.table import Table, vstack
from collections import OrderedDict

## Import some helper functions, you can see their definitions by uncomenting the bash shell command
from desispec.workflow.helper_funcs import opj, listpath, night_to_month, define_variable_from_environment
from desispec.workflow.create_exposure_tables import get_exposure_table_path, get_exposure_table_name, default_exptypes_for_exptable


def create_processing_tables(nights=None, prodname=None, exp_table_path=None, proc_table_path=None,
                            science_types=None, overwrite_files=False, verbose=False,
                            exp_filetype='csv', prod_filetype='csv', joinsymb='|'):
    from desispec.workflow.helper_funcs import write_table, load_table
    if nights is None:
        print("Need to provide nights to create processing tables for. If you want all nights, use 'all'")
        
    ## Define where to find the data
    if exp_table_path is None:
        exp_table_path = get_exposure_table_path()

    if prodname is None:
        prodname = define_variable_from_environment(env_name='SPECPROD',
                                                    var_descr="The production name")

    ## Define where to save the data
    if proc_table_path is None:
        proc_table_path = get_processing_table_path()

    if type(nights) is str and nights == 'all':
        exptables = []
        for month in listpath(exp_table_path):
            exptables += listpath(exp_table_path, month)

        nights = np.unique(
            [file.split('_')[2].split('.')[0] for file in sorted(exptables) if '.' + exp_filetype in file]).astype(int)

    if verbose:
        print(f'Nights: {nights}')

    ## Make the save directory exists
    os.makedirs(exp_table_path, exist_ok=True)

    ## Make the save directory if it doesn't exist
    if not os.path.isdir(proc_table_path):
        print(f'Creating directory: {proc_table_path}')
        os.makedirs(proc_table_path)

    ## Create an astropy table for each night. Define the columns and datatypes, but leave each with 0 rows
    combined_table = Table()

    ## Loop over nights
    for night in nights:
        if verbose:
            print(f'Processing {night}')
        month = night_to_month(night)
        exptab_name = opj(exp_table_path, month, get_exposure_table_name(night=night, extension=exp_filetype))
        exptable = load_table(exptab_name, process_mixins=False)

        if night == nights[0]:
            combined_table = exptable.copy()
        else:
            # for col in exptable.colnames:
            #     if col in combined_table.colnames:
            #         newtype = exptable[col].dtype
            #         curtype = combined_table[col].dtype
            #         if newtype != curtype:
            #             print(night, col, curtype, newtype)
            #             if newtype == object:
            #                 combined_as_list = np.array([[val] for val in combined_table[col]])
            #                 print(combined_table[col])
            #                 print(combined_as_list)
            #                 combined_table.replace_column(col, Table.Column(name=col,data=combined_as_list,dtype=object))
            #             elif curtype == object:
            #                 exp_as_list = np.array([[val] for val in exptable[col]])
            #                 print(exptable[col])
            #                 print(exp_as_list)
            #                 exptable.replace_column(col, Table.Column(name=col,data=exp_as_list,dtype=object))
            #             newtype = exptable[col].dtype
            #             curtype = combined_table[col].dtype
            #             print(night, col, curtype, newtype)
            #             print(exptable[col])
            #             print(combined_table[col])

            combined_table = vstack([combined_table, exptable])

    processing_table, unprocessed_table = exptable_to_proctable(combined_table, science_types=science_types,
                                                                joinsymb=joinsymb)

    ## Save the tables
    proc_name = get_processing_table_name(extension=prod_filetype)
    unproc_name = proc_name.replace('processing', 'unprocessed')
    for tab, name in zip([processing_table, unprocessed_table], [proc_name, unproc_name]):
        if len(tab) > 0:
            pathname = opj(proc_table_path, name)
            write_table(tab, pathname, overwrite=overwrite_files)
            print(f'Wrote file: {name}')


def exptable_to_proctable(input_exptable, science_types=None, addtnl_colnames=None, addtnl_coldtypes=None,
                          joinsymb='|'):
    exptable = input_exptable.copy()

    if science_types is None:
        science_types = default_exptypes_for_exptable()

    ## Define the column names for the exposure table and their respective datatypes
    if addtnl_colnames is None:
        colnames, coldtypes, coldefaults = get_internal_production_table_column_defs(return_default_values=True)

    for col in ['HEADERERR', 'COMMENTS']:
        if col in exptable.colnames:
            for ii, row in enumerate(exptable[col]):
                for item in row.split(joinsymb):
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
        irow = erow_to_irow(erow, colnames, coldtypes, coldefaults)
        rows.append(irow)
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
        colnames, coldtypes = get_internal_production_table_column_defs()
    processing_table = Table(names=colnames, dtype=coldtypes)
    return processing_table


def erow_to_irow_with_overrides(input_erow, colnames=None, coldtypes=None, coldefaults=None, joinsymb='|'):
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
        colnames, coldtypes, coldefaults = get_internal_production_table_column_defs(return_default_values=True)

    irow = OrderedDict()
    for nam, typ, defval in zip(colnames, coldtypes, coldefaults):
        if nam == 'OBSDESC':
            irow[nam] = ' '
            for word in ['dither', 'acquisition', 'focus']:
                if 'PROGRAM' in ecols and word in erow['PROGRAM'].lower():
                    irow[nam] = word
        elif nam in ecols:
            irow[nam] = erow[nam]
        else:
            irow[nam] = defval
    return irow


def erow_to_irow(erow, colnames=None, coldtypes=None, coldefaults=None, joinsymb='|'):
    if type(erow) in [dict, OrderedDict]:
        ecols = erow.keys()
    else:
        ecols = erow.colnames

    ## Define the column names for the exposure table and their respective datatypes
    if colnames is None:
        colnames, coldtypes, coldefaults = get_internal_production_table_column_defs(return_default_values=True)

    irow = OrderedDict()
    for nam, typ, defval in zip(colnames, coldtypes, coldefaults):
        if nam == 'OBSDESC':
            irow[nam] = ' '
            for word in ['dither', 'acquisition', 'focus']:
                if 'PROGRAM' in ecols and word in erow['PROGRAM'].lower():
                    irow[nam] = word
        elif nam in ecols:
            irow[nam] = erow[nam]
        else:
            irow[nam] = defval
    return irow


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
    path = opj(os.environ['DESI_SPECTRO_REDUX'], prodname, 'processing_tables')
    return path


def get_processing_table_pathname(prodname=None, prodmod=None, extension='csv'):  # base_path,prodname
    if prodname is None:
        prodname = os.environ['SPECPROD']

    path = get_processing_table_path(prodname)
    table_name = get_processing_table_name(prodname, prodmod, extension)
    return opj(path, table_name)


def get_internal_production_table_column_defs(return_default_values=False, overlap_only=False, unique_only=False):
    ## Define the column names for the internal production table and their respective datatypes, split in two
    ##     only for readability's sake

    colnames1 = ['EXPID', 'NIGHT', 'OBSTYPE', 'CAMWORD', 'TILEID']
    coltypes1 = [np.ndarray, int, 'S10', 'S40', int]
    coldefault1 = [np.ndarray(shape=0), 20000101, 'unknown', 'a0123456789', -99]

    colnames2 = ['INTID', 'OBSDESC', 'SCRIPTNAME', 'JOBDESC', 'LATEST_QID', 'SUBMIT_DATE', 'STATUS', 'CALIBRATOR']
    coltypes2 = [int, 'S16', 'S40', 'S10', int, 'S20', 'S10', bool]
    coldefault2 = [-99, ' ', 'unknown', 'unknown', -99, 'never', 'U', False]

    colnames3 = ['INT_DEP_IDS', 'LATEST_DEP_QID', 'ALL_QIDS']
    coltypes3 = [np.ndarray, np.ndarray, np.ndarray]
    coldefault3 = [np.ndarray(shape=0), np.ndarray(shape=0), np.ndarray(shape=0)]

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


def create_internal_production_table(rows=None):
    colnames, coldtypes = get_internal_production_table_column_defs()
    outtab = Table(names=colnames, dtype=coldtypes, rows=rows)
    return outtab


# def get_proctab_uniq_columns():
#     colnames =  ['LATEST_JOBID', 'LATEST_JOBFILE', 'JOBFILES', 'JOBIDS', 'REDUX_STATUS']
#     coldtypes = [int           , 'S20'           , 'S80'     , 'S30'   , int]
#     return colnames, coldtypes


if __name__ == '__main__':
    overwrite_files = False
    verbose = False

    prodname = 'testprod'

    exp_filetype = 'csv'
    prod_filetype = 'csv'
    exp_table_path = os.path.abspath('./exposure_tables')
    proc_table_path = os.path.abspath('./production_tables')

    ## Define the nights of interest
    # nights = list(range(20200218,20200230)) + list(range(20200301,20200316))
    nights = 'all'  # this will be reset to a list of all nights in the exposure_tables subdirectories with valid filetype

    ## Define the science types of interest
    science_types = ['arc', 'flat', 'twilight', 'science', 'sci', 'dither']

    create_processing_table(nights, prodname, exp_table_path=exp_table_path, proc_table_path=proc_table_path,
                            science_types=science_types, overwrite_files=overwrite_files, verbose=verbose,
                            exp_filetype=exp_filetype, prod_filetype=prod_filetype, joinsymb='|')
