

import os
import numpy as np
from astropy.table import Table, vstack
from astropy.io import fits
## Import some helper functions, you can see their definitions by uncomenting the bash shell command
from desispec.workflow.exptable import get_exposure_table_path, get_exposure_table_name, get_exposure_table_pathname,night_to_month

from desispec.workflow.os import define_variable_from_environment, listpath, pathjoin
from desispec.workflow.proctable import default_exptypes_for_proctable, get_processing_table_path, exptable_to_proctable, \
                                        get_processing_table_name
from desispec.workflow.tableio import load_table, write_table


def create_processing_tables(nights=None, prodname=None, exp_table_path=None, proc_table_path=None,
                             science_types=None, overwrite_files=False, verbose=False,
                             exp_filetype='csv', prod_filetype='csv', joinsymb='|'):
    """
    Generates processing tables for the nights requested. Requires exposure tables to exist on disk.

    Args:
        nights: str, int, or comma separated list. The night(s) to generate procesing tables for.
        prodname: str. The name of the current production. If used, this will overwrite the SPECPROD environment variable.
        exp_table_path: str. Full path to where to exposure tables are stored, WITHOUT the monthly directory included.
        proc_table_path: str. Full path to where to processing tables to be written.
        science_types: str or comma separated list of strings. The exposure OBSTYPE's that you want to include in the processing table.
        overwrite_files: boolean. Whether to overwrite processing tables if they exist. True overwrites.
        verbose: boolean. Whether to give verbose output information or not. True prints more information.
        exp_filetype: str. The file extension (without the '.') of the exposure tables.
        prod_filetype: str. The file extension (without the '.') of the processing tables.
        joinsymb: str. Symbol to use to indicate the separation of array values when converting to and from strings for
                       saving to csv. Default is highly advised and is '|'. Using a comma will break many things.

    Returns: Nothing

    Notes:
        Requires exposure tables to exist on disk. Either in the default location or at the location specified
        using the function arguments.
    """
    if nights is None:
        print("Need to provide nights to create processing tables for. If you want all nights, use 'all'")

    if science_types is None:
        science_types = default_exptypes_for_proctable()
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
        exptab_name = get_exposure_table_name(night=night, extension=exp_filetype)
        month = night_to_month(night)
        exptable = load_table(pathjoin(exp_table_path,month,exptab_name), process_mixins=False)

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
            pathname = pathjoin(proc_table_path, name)
            write_table(tab, pathname, overwrite=overwrite_files)
            print(f'Wrote file: {name}')



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

    create_processing_tables(nights, prodname, exp_table_path=exp_table_path, proc_table_path=proc_table_path,
                            science_types=science_types, overwrite_files=overwrite_files, verbose=verbose,
                            exp_filetype=exp_filetype, prod_filetype=prod_filetype, joinsymb='|')