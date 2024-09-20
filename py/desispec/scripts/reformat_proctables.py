"""
desispec.scripts.reformat_proctables
====================================

"""
import argparse
import os
import glob
import sys
import numpy as np
import re
import time
from astropy.table import Table

from desispec.io.meta import findfile
from desispec.workflow.proctable import get_processing_table_column_defs
from desispec.workflow.utils import define_variable_from_environment, listpath, \
                                    pathjoin
from desispec.workflow.tableio import write_table, load_table
from desispec.scripts.exposuretable import create_exposure_tables


def get_parser():
    """
    Creates an arguments parser for the desi_reformat_processing_tables script
    """
    parser = argparse.ArgumentParser(usage = "{prog} [options]")
    parser.add_argument("-n", "--nights", type=str,  default=None, help="nights as comma separated string")
    parser.add_argument("--night-range", type=str, default=None, help="comma separated pair of nights in form YYYYMMDD,YYYYMMDD"+\
                                                                      "for first_night,last_night specifying the beginning"+\
                                                                      "and end of a range of nights to be generated. "+\
                                                                      "last_night should be inclusive.")
    parser.add_argument("--orig-filetype", type=str, default='csv', help="format type for original exposure tables")
    parser.add_argument("--out-filetype", type=str, default='csv', help="format type for output exposure tables")
    parser.add_argument("--dry-run", action="store_true",
                        help="Perform a dry run, printing the changes that would be made and the final output table "+
                             "but not overwriting the actual files on disk.")
    return parser

def reformat_processing_tables(nights=None, night_range=None, orig_filetype='csv',
                           out_filetype='csv', dry_run=False):
    """
    Generates updated processing tables for the nights requested. Requires
    a current processing table to exist on disk.

    Args:
        nights: str, int, or comma separated list. The night(s) to generate
                                                   processing tables for.
        night_range: str. comma separated pair of nights in form
                          YYYYMMDD,YYYYMMDD for first_night,last_night
                          specifying the beginning and end of a range of
                          nights to be generated. first_night and last_night are
                          inclusive.
        orig_filetype: str. The file extension (without the '.') of the processing
                            tables.
        out_filetype: str. The file extension for the outputted processing tables
                           (without the '.').

    Returns:
        Nothing
    """
    # log = get_logger()
    ## Make sure user specified what nights to run on
    if nights is None and night_range is None:
        raise ValueError("Must specify either nights or night_range."
                         +" To process all nights give nights=all")

    ## Get all nights in 2020's with data
    proctab_template = findfile('proctable', night=99999999)
    proctab_template = proctab_template.replace('99999999', '202[0-9][01][0-9][0-3][0-9]')
    proctab_template = proctab_template.replace('.csv', f'.{orig_filetype}')
    nights_with_proctables = list()
    for ptabfn in glob.glob(proctab_template):
        ## nights are 202YMMDD
        matches = re.findall('202\d{5}', os.path.basename(ptabfn))
        if len(matches) == 1:
            n = int(matches[0])
            nights_with_proctables.append(n)
        else:
            print(f"Couldn't parse a night from proctable file: {ptabfn}")

    ## If unpecified or given "all", set nights to all nights with data
    check_night = False
    if nights is None or nights == 'all':
        nights = nights_with_proctables
        ## No need to check nights since derived from disk
    else:
        nights = [int(val.strip()) for val in nights.split(",")]
        ## If nights are specified, make sure we check that there is actually data
        check_night = True
    nights = np.sort(nights)

    ## If user specified a night range, cut nights to that range of dates
    if night_range is not None:
        if ',' not in night_range:
            raise ValueError("night_range must be a comma separated pair of "
                             + "nights in form YYYYMMDD,YYYYMMDD")
        nightpair = night_range.split(',')
        if len(nightpair) != 2 or not nightpair[0].isnumeric() \
                or not nightpair[1].isnumeric():
            raise ValueError("night_range must be a comma separated pair of "
                             + "nights in form YYYYMMDD,YYYYMMDD")
        first_night, last_night = nightpair
        nights = nights[np.where(int(first_night) <= nights.astype(int))[0]]
        nights = nights[np.where(int(last_night) >= nights.astype(int))[0]]

    ## Get current set of expected columns
    ptab_cols, ptab_dtypes, ptab_defs = get_processing_table_column_defs(return_default_values=True)
    ptab_cols, ptab_dtypes = np.array(ptab_cols), np.array(ptab_dtypes)

    ## Tell user the final list of nights and starting looping over them
    print("Nights: ", nights)
    for night in nights:
        if check_night and night not in nights_with_proctables:
            print(f"Night {night} doesn't have a processing table: Skipping.")
            continue

        ## If the processing table doesn't exist, skip, since we are updating
        ## not generating.
        orig_pathname = findfile('proctable', night=night).replace('.csv', f'.{orig_filetype}')
        if not os.path.exists(orig_pathname):
            print(f'Could not find processing table for night={night} at:'
                  + f' {orig_pathname}. Skipping this night.')
            continue

        ## Load the old and new tables to compare
        origtable = load_table(orig_pathname, tabletype='proctab')
        curr_colnames = np.array(list(origtable.colnames))
        expected_cols = np.isin(curr_colnames, ptab_cols)
        found_cols = np.isin(ptab_cols, curr_colnames)

        ## If everything is present, don't try to do anything
        if np.all(expected_cols) and np.all(found_cols):
            print(f"{orig_pathname} has all of the expected columns, not updating this table.")
            continue

        unexpected = list(curr_colnames[~expected_cols])
        missing = list(ptab_cols[~found_cols])
        print(f"Found the following unexpected columns: {unexpected}")
        print(f"Found the following missing columns: {missing}")

        ## Solving the only cases I'm currently aware of
        if 'CAMWORD' in unexpected and 'PROCCAMWORD' in missing:
            print(f"CAMWORD listed instead of PROCCAMWORD. Updating that.")
            origtable.rename_column('CAMWORD', 'PROCCAMWORD')
            unexpected.remove('CAWORD')
            missing.remove('PROCCAMWORD')

        if len(unexpected) > 0:
            print(f"WARNING: Script detected unexpected columns. Only handle "
                  + f"the case where 'CAMWORD' is defined instead of PROCCAMWORD. "
                  + f"The following unexpected columns will be dropped without "
                  + f"using the information they contain: {unexpected}.")
            for colname in unexpected:
                origtable.remove_column(colname)

        ## Add any missing columns
        for colname in missing:
            if colname not in ['BADAMPS', 'LASTSTEP', 'EXPFLAG']:
                print(f"WARNING: Script didn't expect {colname} to be missing. "
                      + f"Replacing with default values, but this may have "
                      + f"downstream consequences.")
            colindex = np.where(ptab_cols==colname)[0][0]
            newdat = [ptab_defs[colindex]] * len(origtable)
            newcol = Table.Column(name=colname, data=newdat, dtype=ptab_dtypes[colindex])
            origtable.add_column(newcol)

        ## Finally, reorder to the current column ordering
        origtable = origtable[list(ptab_cols)]

        ## If just testing, print the table and a cell-by-cell equality test
        ## for the scalar columns
        ## If not testing, move the original table to an archived filename
        ## and save the updated table to the official exptable pathname
        if dry_run:
            print("\n\nOutput file would have been:")
            origtable.pprint_all()
        else:
            ftime = time.strftime("%Y%m%d_%Hh%Mm")
            replaced_pathname = orig_pathname.replace(f".{orig_filetype}",
                                                      f".replaced-{ftime}.{orig_filetype}")
            print(f"Moving original file from {orig_pathname} to {replaced_pathname}")
            os.rename(orig_pathname,replaced_pathname)
            time.sleep(0.1)
            out_pathname = orig_pathname.replace(f".{orig_filetype}", f".{out_filetype}")
            write_table(origtable, out_pathname)
            print(f"Updated file saved to {out_pathname}. Original archived as {replaced_pathname}")

            print("\n\n")

        ## Flush the outputs
        sys.stdout.flush()
        sys.stderr.flush()
    print("Processing table regenerations complete")
