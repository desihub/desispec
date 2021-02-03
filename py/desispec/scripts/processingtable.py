

import os
import sys
import numpy as np
from astropy.table import Table, vstack
from astropy.io import fits
## Import some helper functions, you can see their definitions by uncomenting the bash shell command
from desispec.workflow.exptable import get_exposure_table_path, get_exposure_table_name, night_to_month

from desispec.workflow.utils import define_variable_from_environment, listpath, pathjoin, get_printable_banner
from desispec.workflow.proctable import default_exptypes_for_proctable, get_processing_table_path, exptable_to_proctable, \
                                        get_processing_table_name
from desispec.workflow.tableio import load_table, write_table


def create_processing_tables(nights=None, prodname=None, exp_table_path=None, proc_table_path=None,
                             obstypes=None, overwrite_files=False, verbose=False, no_specprod_exptab=False,
                             exp_filetype='csv', prod_filetype='csv', joinsymb='|'):
    """
    Generates processing tables for the nights requested. Requires exposure tables to exist on disk.

    Args:
        nights: str, int, or comma separated list. The night(s) to generate procesing tables for.
        prodname: str. The name of the current production. If used, this will overwrite the SPECPROD environment variable.
        exp_table_path: str. Full path to where to exposure tables are stored, WITHOUT the monthly directory included.
        proc_table_path: str. Full path to where to processing tables to be written.
        obstypes: str or comma separated list of strings. The exposure OBSTYPE's that you want to include in the processing table.
        overwrite_files: boolean. Whether to overwrite processing tables if they exist. True overwrites.
        verbose: boolean. Whether to give verbose output information or not. True prints more information.
        no_specprod_exptab: boolean. Read exposure table in repository location rather than the SPECPROD location.
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

    if obstypes is None:
        obstypes = default_exptypes_for_proctable()
    ## Define where to find the data
    if exp_table_path is None:
        usespecprod = (not no_specprod_exptab)
        exp_table_path = get_exposure_table_path(night=None,usespecprod=usespecprod)

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
            print(get_printable_banner(input_str=night))
        else:
            print(f'Processing {night}')
        exptab_name = get_exposure_table_name(night=night, extension=exp_filetype)
        month = night_to_month(night)
        exptable = load_table(pathjoin(exp_table_path,month,exptab_name), process_mixins=False)

        if night == nights[0]:
            combined_table = exptable.copy()
        else:
            combined_table = vstack([combined_table, exptable])

    processing_table, unprocessed_table = exptable_to_proctable(combined_table, obstypes=obstypes)#,joinsymb=joinsymb)

    ## Save the tables
    proc_name = get_processing_table_name(extension=prod_filetype)
    unproc_name = proc_name.replace('processing', 'unprocessed')
    for tab, name in zip([processing_table, unprocessed_table], [proc_name, unproc_name]):
        if len(tab) > 0:
            pathname = pathjoin(proc_table_path, name)
            write_table(tab, pathname, overwrite=overwrite_files)
            print(f'Wrote file: {name}')

    print("Processing table generations complete")
    ## Flush the outputs
    sys.stdout.flush()
    sys.stderr.flush()