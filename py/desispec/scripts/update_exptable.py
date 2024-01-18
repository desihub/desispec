"""
desispec.scripts.update_exptable
=================================

"""
import numpy as np
import os
import sys
import time
from astropy.table import Table
import glob

## Import some helper functions, you can see their definitions by uncomenting the bash shell command
from desispec.workflow.tableio import load_table, write_table
from desispec.workflow.utils import verify_variable_with_environment, pathjoin, listpath, \
                                    get_printable_banner, sleep_and_report
from desispec.workflow.timing import during_operating_hours, what_night_is_it
from desispec.workflow.exptable import default_obstypes_for_exptable, get_exposure_table_column_defs, \
    get_exposure_table_path, get_exposure_table_name, summarize_exposure
from desispec.io.util import difference_camwords, parse_badamps, validate_badamps
from desiutil.log import get_logger
def update_exposure_table(night=None, specprod=None, exp_table_path=None,
                          path_to_data=None, exp_obstypes=None, camword=None,
                          badcamword=None, badamps=None, tab_filetype='csv',
                          exps_to_ignore=None, exp_cadence_time=2,
                          dry_run_level=0, verbose=False):
    """
    Generates processing tables for the nights requested. Requires exposure tables to exist on disk.

    Args:
        night: str or int. 8 digit night, e.g. 20200314, of data to run on. If None, it runs on the current night.
        specprod: str. The name of the current production. If used, this will overwrite the SPECPROD environment variable.
        exp_table_path: str. Full path to where to exposure tables are stored, WITHOUT the monthly directory included.
        path_to_data: str. Path to the raw data.
        exp_obstypes: str or comma separated list of strings. The exposure OBSTYPE's that you want to include in the exposure table.
        camword: str. Camword that, if set, alters the set of cameras that will be set for processing.
                      Examples: a0123456789, a1, a2b3r3, a2b3r4z3.
        badcamword: str. Camword that, if set, will be removed from the camword defined in camword if given, or the camword
                         inferred from the data if camword is not given.
        badamps: str. Comma seperated list of bad amplifiers that should not be processed. Should be of the
                      form "{camera}{petal}{amp}", i.e. "[brz][0-9][ABCD]". Example: 'b7D,z8A'
        tab_filetype: str. The file extension (without the '.') of the exposure and processing tables.
        exps_to_ignore: list. A list of exposure id's that should not be processed. Each should be an integer.
        exp_cadence_time: int. Wait time in seconds between loops over each science exposure. Default 2.
        dry_run_level, int, If nonzero, this is a simulated run. If dry_run_level=1 the scripts will be written or submitted. If
                      dry_run_level=2, the scripts will not be writter or submitted. Logging will remain the same
                      for testing as though scripts are being submitted. Default is 0 (false).
        verbose: bool. True if you want more verbose output, false otherwise. Current not propagated to lower code,
                       so it is only used in the main daily_processing script itself.

    Returns:
        etable: astropy Table. An exposure table where each row is an expoure for a given night.

    Notes:
        Generates the exposure table and saves it. This can be run repeatedly throughout a night.
    """
    log = get_logger()

    ## What night are we running on?
    true_night = what_night_is_it()
    if night is not None:
        night = int(night)
        log.info(f"True night is {true_night}, but running for {night=}")
    else:
        night = true_night
    log.info(f"Updating the exposure table for {night=}")

    ## Define the obstypes to save information for in the exposure table
    if exp_obstypes is None:
        exp_obstypes = default_obstypes_for_exptable()
    elif isinstance(exp_obstypes, str):
        exp_obstypes = exp_obstypes.split(',')

    ## Warn people if changing camword
    finalcamword = 'a0123456789'
    if camword is not None and badcamword is None:
        badcamword = difference_camwords(finalcamword,camword)
        finalcamword = camword
    elif camword is not None and badcamword is not None:
        finalcamword = difference_camwords(camword, badcamword)
        badcamword = difference_camwords('a0123456789', finalcamword)
    elif badcamword is not None:
        finalcamword = difference_camwords(finalcamword,badcamword)
    else:
        badcamword = ''

    if badcamword != '':
        ## Inform the user what will be done with it.
        log.info(f"Modifying camword of data to be processed with badcamword: {badcamword}. "+\
              f"Camword to be processed: {finalcamword}")

    ## Make sure badamps is formatted properly
    if badamps is None:
        badamps = ''
    else:
        badamps = validate_badamps(badamps)

    ## Define the set of exposures to ignore
    if exps_to_ignore is None:
        exps_to_ignore = set()
    else:
        exps_to_ignore = np.sort(np.array(exps_to_ignore).astype(int))
        log.info(f"\nReceived exposures to ignore: {exps_to_ignore}")
        exps_to_ignore = set(exps_to_ignore)

    ## Get context specific variable values
    colnames, coltypes, coldefaults = get_exposure_table_column_defs(return_default_values=True)

    ## Define where to find the data
    path_to_data = verify_variable_with_environment(var=path_to_data,var_name='path_to_data', env_name='DESI_SPECTRO_DATA')
    specprod = verify_variable_with_environment(var=specprod,var_name='specprod',env_name='SPECPROD')

    ## Define the naming scheme for the raw data
    ## Manifests (describing end of cals, etc.) don't have a data file, so search for those separately
    data_glob = os.path.join(path_to_data, str(night), '*', 'desi-*.fit*')
    manifest_glob = os.path.join(path_to_data, str(night), '*', 'manifest_*.json')

    ## Determine where the exposure table will be written
    if exp_table_path is None:
        exp_table_path = get_exposure_table_path(night=night, usespecprod=True)
    os.makedirs(exp_table_path, exist_ok=True)
    name = get_exposure_table_name(night=night, extension=tab_filetype)
    exp_table_pathname = pathjoin(exp_table_path, name)

    ## Load in the file defined above
    etable = load_table(tablename=exp_table_pathname,  tabletype='exptable')

    ## Get relevant data from the tables
    all_exps = set(etable['EXPID'])

    ## Get a list of new exposures that have been found
    log.info(f"\n\n\nPreviously known exposures: {all_exps}")
    data_exps = set(sorted([int(os.path.basename(os.path.dirname(fil))) for fil in glob.glob(data_glob)]))
    manifest_exps = set(sorted([int(os.path.basename(os.path.dirname(fil))) for fil in glob.glob(manifest_glob)]))
    located_exps = data_exps.union(manifest_exps)

    new_exps = located_exps.difference(all_exps)
    log.info(f"\nNew exposures: {new_exps}\n\n")

    ## Loop over new exposures and add them to the exposure_table
    for exp in sorted(list(new_exps)):
        if verbose:
            log.info(get_printable_banner(str(exp)))
        else:
            log.info(f'\n\n##################### {exp} #########################')

        ## Open relevant raw data files to understand what we're dealing with
        erow = summarize_exposure(path_to_data, night, exp, exp_obstypes, colnames, coldefaults, verbosely=verbose)

        ## If there was an issue, continue. If it's a string summarizing the end of some sequence, use that info.
        ## If the exposure is assosciated with data, process that data.
        if erow is None or isinstance(erow, str):
            continue

        erow['BADCAMWORD'] = badcamword
        erow['BADAMPS'] = badamps
        if exp in exps_to_ignore:
            log.info(f"\n{exp} given as exposure id to ignore. Not processing.")
            erow['LASTSTEP'] = 'ignore'
        elif erow['LASTSTEP'] == 'ignore':
            log.info(f"\n{exp} identified by the pipeline as something to ignore. Not processing.")

        log.info(f"\nFound: {erow}")
        etable.add_row(erow)

        sleep_and_report(exp_cadence_time, message_suffix=f"after exposure",
                         dry_run=(dry_run_level > 0))

        ## Flush the outputs
        sys.stdout.flush()
        sys.stderr.flush()

    ## Only write out the table at the end and only if dry_run_level dictates
    if dry_run_level < 3:
        write_table(etable, tablename=exp_table_pathname)

    log.info(f"Completed exposure_table generation for exposures from night {night}.")
    log.info("Exiting")

    ## Flush the outputs
    sys.stdout.flush()
    sys.stderr.flush()

    return etable