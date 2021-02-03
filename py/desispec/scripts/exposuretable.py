
import os
import sys
import numpy as np
from astropy.table import Table
from astropy.io import fits
## Import some helper functions, you can see their definitions by uncomenting the bash shell command

from desispec.workflow.exptable import summarize_exposure, default_exptypes_for_exptable, \
                                       instantiate_exposure_table, get_exposure_table_column_defs, \
                                       get_exposure_table_path, get_exposure_table_name, night_to_month
from desispec.workflow.utils import define_variable_from_environment, listpath, pathjoin, get_printable_banner
from desispec.workflow.tableio import write_table



def create_exposure_tables(nights, path_to_data=None, exp_table_path=None, obstypes=None, \
                           exp_filetype='csv', verbose=False, no_specprod=False, overwrite_files=False):
    """
    Generates processing tables for the nights requested. Requires exposure tables to exist on disk.

    Args:
        nights: str, int, or comma separated list. The night(s) to generate procesing tables for.
        path_to_data: str. The path to the raw data and request*.json and manifest* files.
        exp_table_path: str. Full path to where to exposure tables should be saved, WITHOUT the monthly directory included.
        obstypes: str or comma separated list of strings. The exposure OBSTYPE's that you want to include in the exposure table.
        exp_filetype: str. The file extension (without the '.') of the exposure tables.
        verbose: boolean. Whether to give verbose output information or not. True prints more information.
        no_specprod: boolean. Create exposure table in repository location rather than the SPECPROD location
        overwrite_files: boolean. Whether to overwrite processing tables if they exist. True overwrites.

    Returns: Nothing
    """
    ## Define where to find the data
    if path_to_data is None:
        path_to_data = define_variable_from_environment(env_name='DESI_SPECTRO_DATA',
                                                        var_descr="The data path")

    ## Define where to save the data
    usespecprod = (not no_specprod)
    if exp_table_path is None:
        exp_table_path = get_exposure_table_path(night=None,usespecprod=usespecprod)
    if obstypes is None:
        obstypes = default_exptypes_for_exptable()

    ## Make the save directory exists
    os.makedirs(exp_table_path, exist_ok=True)

    ## Create an astropy table for each night. Define the columns and datatypes, but leave each with 0 rows
    # colnames, coldtypes = get_exposure_table_column_defs()
    # nightly_tabs = { night : Table(names=colnames,dtype=coldtypes) for night in nights }
    nightly_tabs = { night : instantiate_exposure_table() for night in nights }

    ## Loop over nights
    colnames, coltypes, coldefaults = get_exposure_table_column_defs(return_default_values=True)
    for night in nights:
        print(get_printable_banner(input_str=night))

        ## Loop through all exposures on disk
        for exp in listpath(path_to_data,str(night)):
            rowdict = summarize_exposure(path_to_data, night=night, exp=exp, obstypes=obstypes, \
                                         colnames=colnames, coldefaults=coldefaults, verbosely=verbose)
            if verbose:
                print("Rowdict:\n",rowdict,"\n\n")
            if rowdict is not None and type(rowdict) is not str:
                ## Add the dictionary of column values as a new row
                nightly_tabs[night].add_row(rowdict)

        if len(nightly_tabs[night]) > 0:
            month = night_to_month(night)
            exptab_path = pathjoin(exp_table_path,month)
            os.makedirs(exptab_path,exist_ok=True)
            exptab_name = get_exposure_table_name(night, extension=exp_filetype)
            exptab_name = pathjoin(exptab_path, exptab_name)
            write_table(nightly_tabs[night], exptab_name, overwrite=overwrite_files)
        else:
            print('No rows to write to a file.')

        print("Exposure table generations complete")
        ## Flush the outputs
        sys.stdout.flush()
        sys.stderr.flush()
    return nightly_tabs
