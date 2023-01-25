"""
desispec.scripts.processingtable
================================

"""
import os
import sys
import numpy as np
import re
from astropy.table import Table, vstack
from astropy.io import fits
from desispec.io.util import parse_cameras, difference_camwords, validate_badamps
## Import some helper functions, you can see their definitions by uncomenting the bash shell command
from desispec.workflow.exptable import get_exposure_table_path, get_exposure_table_name, \
                                        night_to_month

from desispec.workflow.utils import define_variable_from_environment, listpath, pathjoin, get_printable_banner
from desispec.workflow.proctable import default_obstypes_for_proctable, get_processing_table_path, exptable_to_proctable, \
                                        get_processing_table_name
from desispec.workflow.tableio import load_table, write_table


def create_processing_tables(nights=None, night_range=None, exp_table_path=None, proc_table_path=None,
                             obstypes=None, overwrite_files=False, verbose=False, no_specprod_exptab=False,
                             exp_filetype='csv', prod_filetype='csv', joinsymb='|'):
    """
    Generates processing tables for the nights requested. Requires exposure tables to exist on disk.

    Args:
        nights: str, int, or comma separated list. The night(s) to generate procesing tables for.
        night_range: str, comma separated pair of nights in form YYYYMMDD,YYYYMMDD for first_night,last_night
                          specifying the beginning and end of a range of nights to be generated.
                          last_night should be inclusive.
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
    if nights is None and night_range is None:
        raise ValueError("Must specify either nights or night_range")
    elif nights is not None and night_range is not None:
        raise ValueError("Must only specify either nights or night_range, not both")

    if nights is None or nights == 'all':
        nights = list()
        for n in listpath(os.getenv('DESI_SPECTRO_DATA')):
            # - nights are 20YYMMDD
            if re.match('^20\d{6}$', n):
                nights.append(n)
    else:
        nights = [int(val.strip()) for val in nights.split(",")]

    nights = np.array(nights)

    if night_range is not None:
        if ',' not in night_range:
            raise ValueError("night_range must be a comma separated pair of nights in form YYYYMMDD,YYYYMMDD")
        nightpair = night_range.split(',')
        if len(nightpair) != 2 or not nightpair[0].isnumeric() or not nightpair[1].isnumeric():
            raise ValueError("night_range must be a comma separated pair of nights in form YYYYMMDD,YYYYMMDD")
        first_night, last_night = nightpair
        nights = nights[np.where(int(first_night) <= nights.astype(int))[0]]
        nights = nights[np.where(int(last_night) >= nights.astype(int))[0]]

    if obstypes is not None:
        obstypes = [ val.strip('\t ') for val in obstypes.split(",") ]
    else:
        obstypes = default_obstypes_for_proctable()

    ## Define where to find the data
    if exp_table_path is None:
        usespecprod = (not no_specprod_exptab)
        exp_table_path = get_exposure_table_path(night=None,usespecprod=usespecprod)

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