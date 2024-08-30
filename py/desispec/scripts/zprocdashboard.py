"""
desispec.scripts.zprocdashboard
===============================

"""
import argparse
import os, glob
import sys
import re
from astropy.io import fits, ascii
from astropy.table import Table, vstack
import time, datetime
import numpy as np
from os import listdir
import json

from desispec.workflow.queue import update_from_queue
# import desispec.io.util
from desiutil.log import get_logger
from desispec.workflow.exptable import get_exposure_table_pathname, \
    default_obstypes_for_exptable, \
    get_exposure_table_column_types, \
    get_exposure_table_column_defaults, read_minimal_science_exptab_cols
from desispec.workflow.proc_dashboard_funcs import get_skipped_ids, \
    return_color_profile, find_new_exps, _hyperlink, _str_frac, \
    get_output_dir, get_nights_dict, make_html_page, read_json, write_json, \
    get_terminal_steps, get_tables
from desispec.workflow.proctable import get_processing_table_pathname, \
    erow_to_prow, instantiate_processing_table
from desispec.workflow.tableio import load_table
from desispec.io.meta import specprod_root, rawdata_root, findfile
from desispec.io.util import decode_camword, camword_to_spectros, \
    difference_camwords, parse_badamps, create_camword, camword_union, \
    columns_to_goodcamword, spectros_to_camword

def parse(options):
    """
    Initialize the parser to read input
    """
    # Initialize
    parser = argparse.ArgumentParser(
        description="Search the filesystem and summarize the existance of files output from " +
                    "the daily processing pipeline. Can specify specific nights, give a number of past nights," +
                    " or use --all to get all past nights.")

    # File I/O
    parser.add_argument('--redux-dir', type=str,
                        help="Product directory, point to $DESI_SPECTRO_REDUX by default ")
    parser.add_argument('--output-dir', type=str, default=None,
                        help="output portal directory for the html pages, which defaults to your home directory ")
    parser.add_argument('--output-name', type=str, default='zdashboard.html',
                        help="name of the html page (to be placed in --output-dir).")
    parser.add_argument('--specprod', type=str,
                        help="overwrite the environment keyword for $SPECPROD")
    parser.add_argument("-e", "--skip-tileid-file", type=str, required=False,
                        help="Relative pathname for file containing tileid's to skip. " + \
                             "Automatically. They are assumed to be in a column" + \
                             "format, one per row. Stored internally as integers, so zero padding is " + \
                             "accepted but not required.")
    # parser.add_argument("--skip-null", type=str, required=False,
    #                    help="Relative pathname for file containing expid's to skip. "+\
    #                         "Automatically. They are assumed to be in a column"+\
    #                         "format, one per row. Stored internally as integers, so zero padding is "+\
    #                         "accepted but not required.")
    # Specify Nights of Interest
    parser.add_argument('-n', '--nights', type=str, default=None,
                        required=False,
                        help="nights to monitor. Can be 'all', a comma separated list of YYYYMMDD, or a number " +
                             "specifying the previous n nights to show (counting in reverse chronological order).")
    parser.add_argument('--start-night', type=str, default=None, required=False,
                        help="This specifies the first night to include in the dashboard. " +
                             "Default is the earliest night available.")
    parser.add_argument('--end-night', type=str, default=None, required=False,
                        help="This specifies the last night (inclusive) to include in the dashboard. Default is today.")
    parser.add_argument('--check-on-disk', action="store_true",
                        help="Check raw data directory for additional unaccounted for exposures on disk " +
                             "beyond the exposure table.")
    parser.add_argument('--no-emfits', action="store_true",
                        help="Set if you don't want the dashboard to count emlinefit files.")
    parser.add_argument('--no-qsofits', action="store_true",
                        help="Set if you don't want the dashboard to count qn and MgII files.")
    parser.add_argument('--no-tileqa', action="store_true",
                        help="Set if you don't want the dashboard to count tile QA files.")
    parser.add_argument('--no-zmtl', action="store_true",
                        help="Set if you don't want the dashboard to count zmtl files.")
    parser.add_argument('--ignore-json-archive', action="store_true",
                        help="Ignore the existing json archive of good exposure rows, regenerate all rows from " +
                             "information on disk. As always, this will write out a new json archive," +
                             " overwriting the existing one.")
    # Read in command line and return
    args = parser.parse_args(options)

    return args

######################
### Main Functions ###
######################
def main(args=None):
    """ Code to generate a webpage for monitoring of desi_dailyproc production status
    Usage:
    -n can be 'all' or series of nights separated by comma or blank like 20200101,20200102 or 20200101 20200102
    Normal Mode:
    desi_proc_dashboard -n 3  --output-dir /global/cfs/cdirs/desi/www/collab/dailyproc/
    desi_proc_dashboard -n 20200101,20200102 --output-dir /global/cfs/cdirs/desi/www/collab/dailyproc/
    """
    log = get_logger()
    if not isinstance(args, argparse.Namespace):
        args = parse(args)

    args.show_null = True
    doem, doqso = (not args.no_emfits), (not args.no_qsofits)
    dotileqa, dozmtl = (not args.no_tileqa), (not args.no_zmtl)

    output_dir, prod_dir = get_output_dir(args.redux_dir, args.specprod,
                                          args.output_dir, makedir=True)
    os.makedirs(os.path.join(output_dir, 'zjsons'), exist_ok=True)
    ############
    ## Input ###
    ############
    if args.skip_tileid_file is not None:
        skipd_tileids = set(
            get_skipped_ids(args.skip_tileid_file, skip_ids=True))
    else:
        skipd_tileids = None

    nights_dict, nights = get_nights_dict(args.nights, args.start_night,
                                          args.end_night, prod_dir)

    log.info(f'Searching {prod_dir} for: {nights}')
    
    ## Get all the exposure tables for cross-night dependencies
    all_exptabs = read_minimal_science_exptab_cols(nights=None)

    ## We don't want future days mixing in
    all_exptabs = all_exptabs[all_exptabs['NIGHT'] <= np.max(nights)]
    ## Restrict to only the exptabs relevant to the current dashboard
    night_selection = np.isin(all_exptabs['NIGHT'],nights)
    tiles = all_exptabs['TILEID'][night_selection]
    subset_exptabs = all_exptabs[np.isin(all_exptabs['TILEID'], tiles)]
    
    monthly_tables = {}
    for month, nights_in_month in nights_dict.items():
        log.info("Month: {}, nights: {}".format(month, nights_in_month))
        nightly_tables = {}
        for night in nights_in_month:
            ## Load previous info if any
            filename_json = os.path.join(output_dir, 'zjsons',
                                         f'zinfo_{os.environ["SPECPROD"]}'
                                         + f'_{night}.json')
            night_json_zinfo = None
            if not args.ignore_json_archive:
                night_json_zinfo = read_json(filename_json=filename_json)

            ## only send table for tiles on the given night
            tiles = all_exptabs['TILEID'][all_exptabs['NIGHT']==night]
            subset_exptabs = all_exptabs[np.isin(all_exptabs['TILEID'], tiles)]
            
            ## get the per exposure info for a night
            night_zinfo = populate_night_zinfo(night, doem, doqso,
                                               dotileqa, args.check_on_disk,
                                               night_json_zinfo=night_json_zinfo,
                                               skipd_tileids=skipd_tileids,
                                               all_exptabs=subset_exptabs)
            
            if len(night_zinfo) == 0:
                continue

            nightly_tables[night] = night_zinfo.copy()

            ## write out the night_info to json file
            write_json(output_data=night_zinfo, filename_json=filename_json)

        monthly_tables[month] = nightly_tables.copy()

    outfile = os.path.abspath(os.path.join(output_dir, args.output_name))
    make_html_page(monthly_tables, outfile, titlefill='z Processing',
                   show_null=args.show_null)



def populate_night_zinfo(night, doem=True, doqso=True, dotileqa=True, dozmtl=True,
                         check_on_disk=False, night_json_zinfo=None,
                         skipd_tileids=None, all_exptabs=None):
    """
    For a given night, return the file counts and other information
    for each zproc job (either per-exposure, per-night, or cumulative for
    each tile on the requested night. Uses cached values supplied in
    night_json_zinfo and skips tiles listed in skipd_tileids.

    Args:
        night (int): the night to check the status of the processing for.
        doem (bool): true if it should expect emline files. Default is True.
        doqso (bool): true if it should expect qso_qn and qso_mgii files.
            Default is True.
        dotileqa (bool): true if it should expect tileqa files. Default is True.
        dozmtl (bool): true if it should expect zmtl files. Default is True.
        check_on_disk (bool): true if it should check on disk for missing
            exposures and tiles that aren't represented in the exposure tables.
        night_json_zinfo (dict): A dictionary of dicts where each key is a unique
            identifier to the row. Each value is a dictionary container the
            column information in addition to other metadata. Meant to be a
            way of passing cached values from a previous run of this function.
        skipd_tileids (list): List of tileids that should be skipped and not
            listed in the output dashboard.
        all_exptabs (astropy.table.Table): A stacked exposure table with minimal
            columns returned from read_minimal_science_exptab_cols(). Used for
            cumulative redshifts jobs to identify tile data from previous nights.

    Returns dict:
        A dictionary of dicts. Each item is information for a row of the output
            dashboard for a redshift job on the requested night. Each key is a
            unique identifier to the row. Each value is a dictionary container
            the column information in addition to other metadata about the state
            of the processing and file counts.
    """
    log = get_logger()
    
    if skipd_tileids is None:
        skipd_tileids = []
    ## Note that the following list should be in order of processing. I.e. the first filetype given should be the
    ## first file type generated. This is assumed for the automated "terminal step" determination that follows
    expected_by_type = dict()
    expected_by_type['cumulative'] =      {'rr': 1,  'tile-qa': 1, 'zmtl': 1,
                                           'qso': 0, 'em': 0}
    expected_by_type['pernight'] =        {'rr': 1,  'tile-qa': 1, 'zmtl': 0,
                                           'qso': 0, 'em': 0}
    expected_by_type['perexp'] =          {'rr': 1,  'tile-qa': 0, 'zmtl': 0,
                                           'qso': 0, 'em': 0}
    expected_by_type['null'] =            {'rr': 0,  'tile-qa': 0, 'zmtl': 0,
                                           'qso': 0, 'em': 0}

    for ztype in expected_by_type.keys():
        if ztype == 'null':
            continue
        if doem:
            expected_by_type[ztype]['em'] = 1
        if doqso:
            expected_by_type[ztype]['qso'] = 1
        ## These are special to specific redshift types, so we remove
        ## if not asked to do them rather than setting specific redshift
        ## types to 1 as done above
        if not dotileqa:
            expected_by_type[ztype]['tile-qa'] = 0
        if not dozmtl:
            expected_by_type[ztype]['zmtl'] = 0

    ## Determine the last filetype that is expected for each obstype
    terminal_steps = get_terminal_steps(expected_by_type)

    specproddir = specprod_root()
    webpage = os.environ['DESI_DASHBOARD']
    logpath = os.path.join(specproddir, 'run', 'scripts', 'tiles')

    orig_exptab, proctab, \
    unaccounted_for_expids,\
    unaccounted_for_tileids = get_tables(night, check_on_disk=check_on_disk,
                                                exptab_colnames=None)

    exptab = orig_exptab[((orig_exptab['OBSTYPE'] == 'science')
                    & (orig_exptab['LASTSTEP'] == 'all'))]

    if proctab is None:
        if len(exptab) == 0:
            log.warning(f"No redshiftable exposures found on night {night}. Skipping")
            ## There is nothing on this night, return blank and move on
            return {}
        else:
            proctab = instantiate_processing_table()
            if str(os.environ['SPECPROD']).lower() == 'daily':
                ztypes = ['cumulative']
            else:
                ztypes = ['cumulative', 'perexp', 'pernight']
            log.warning(f"No processed data on night {night}. Assuming "
                  + f"{os.environ['SPECPROD']} implies ztypes={ztypes}")
    else:
        ## Update the STATUS of the
        proctab = update_from_queue(proctab)
        proctab = proctab[np.array([job in ['pernight', 'perexp', 'cumulative']
                                    for job in proctab['JOBDESC']])]
        ztypes = np.unique(proctab['JOBDESC'])

    ## Determine what was processed
    uniqs_processing = []
    for prow in proctab:
        ztype = str(prow['JOBDESC'])
        tileid = str(prow['TILEID'])
        zfild_expid = str(prow['EXPID'][0]).zfill(8)
        if ztype == 'perexp':
            uniqs_processing.append(f'{zfild_expid}_{ztype}')
        else:
            uniqs_processing.append(f'{tileid}_{ztype}')

    ## Determine what should have been processed but isn't in the processing table
    if 'pernight' in ztypes or 'cumulative' in ztypes:
        etiles = np.unique(exptab['TILEID'])
        ptiles = np.unique(proctab['TILEID'])
        missing_tiles = list(set(etiles).difference(set(ptiles)))
        if len(missing_tiles) > 0:
            for tileid in missing_tiles:
                tilematches = exptab[exptab['TILEID']==tileid]
                first_exp = tilematches[0]
                prow = erow_to_prow(first_exp)
                prow['EXPID'] = np.array(list(tilematches['EXPID']))
                ## pernight and cumulative proccamword will be determined in the main loop
                proccamword = ''
                prow['PROCCAMWORD'] = proccamword
                ## perexp are dealt with separately
                for ztype in set(ztypes).difference({'perexp'}):
                    prow['JOBDESC'] = ztype
                    proctab.add_row(prow.copy())
    if 'perexp' in ztypes:
        perexp_subset = proctab[proctab['JOBDESC']=='perexp']
        exps = np.array(exparr[0] for exparr in perexp_subset['EXPID'])
        for i, exp in enumerate(exptab['EXPID']):
            if exp not in exps:
                erow = exptab[i]
                prow = erow_to_prow(erow)
                prow['JOBDESC'] = 'perexp'
                proctab.add_row(prow.copy())

    ## Now that proctable has even missing entries, loop over the proctable
    ## and summarize the file existance for each entry
    output = dict()
    for row in proctab:
        int_tileid = int(row['TILEID'])
        if int_tileid in skipd_tileids:
            continue

        ztype = str(row['JOBDESC'])
        tileid = str(row['TILEID'])
        zfild_expid = str(row['EXPID'][0]).zfill(8)

        ## files and dashboard are per night so these are unique without night
        ## in the key
        if ztype == 'perexp':
            unique_key = f'{zfild_expid}_{ztype}'
        else:
            unique_key = f'{tileid}_{ztype}'

        ## For those already marked as GOOD or NULL in cached rows, take that and move on
        if night_json_zinfo is not None and unique_key in night_json_zinfo \
                and night_json_zinfo[unique_key]["COLOR"] in ['GOOD', 'NULL']:
            output[unique_key] = night_json_zinfo[unique_key]
            continue

        tilematches = exptab[exptab['TILEID'] == int(tileid)]
        if len(tilematches) == 0:
            if int(tileid) not in orig_exptab['TILEID']:
                log.error(f"ERROR: Tile {tileid} found in processing table not present "
                      + f"in exposure table.")
                log.info(f"exptab tileids: {np.unique(orig_exptab['TILEID'].data)}!")
                log.info(f"proctab tileids: {np.unique(proctab['TILEID'].data)}!")
            else:
                log.warning(f"Tile {tileid} found in processing table has no valid "
                      + f"exposures in the exposure table. Skipping this tile.")
            continue
        exptab_row = tilematches[0]

        ## Assign or derive proccamword and nspectros
        if ztype == 'cumulative':
            spectros = set()
            sel = ((all_exptabs['TILEID']==int(tileid))
                   & (all_exptabs['NIGHT']<=int(night)))
            tilerows = all_exptabs[sel]
            ## Each night needs to be able to calibrate petal, so treat separate
            ## then combine complete petals across nights
            for nit in np.unique(tilerows['NIGHT']):
                nightrows = tilerows[tilerows['NIGHT']==nit]
                proccamwords = []
                for erow in nightrows:
                    proccamwords.append(columns_to_goodcamword(camword=erow['CAMWORD'],
                                                               badcamword=erow['BADCAMWORD'],
                                                               badamps=None, obstype='science',
                                                               suppress_logging=True))
                night_pcamword = camword_union(proccamwords)
                spectros = spectros.union(set(camword_to_spectros(night_pcamword,
                                                                  full_spectros_only=True)))
            proccamword = spectros_to_camword(spectros)
        elif ztype == 'pernight':
            tilerows = all_exptabs[all_exptabs['TILEID']==int(tileid)]
            nightrows = tilerows[tilerows['NIGHT']==night]
            proccamwords = []
            for erow in nightrows:
                proccamwords.append(columns_to_goodcamword(camword=erow['CAMWORD'],
                                                           badcamword=erow['BADCAMWORD'],
                                                           badamps=None, obstype='science',
                                                           suppress_logging=True))
            proccamword = camword_union(proccamwords)
            spectros = camword_to_spectros(proccamword, full_spectros_only=True)
        else:
            proccamword = row['PROCCAMWORD']
            spectros = camword_to_spectros(proccamword, full_spectros_only=True)

        nspecs = len(spectros)

        ## files and dashboard are per night so these are unique without night
        ## in the key
        if ztype == 'perexp':
            logfiletemplate = os.path.join(logpath, '{ztype}', '{tileid}', '{zexpid}',
                                           'ztile-{tileid}-{zexpid}{jobid}.{ext}')
        elif ztype =='cumulative':
            logfiletemplate = os.path.join(logpath, '{ztype}', '{tileid}', '{night}',
                                           'ztile-{tileid}-thru{night}{jobid}.{ext}')
        else:
            # pernight
            logfiletemplate = os.path.join(logpath, '{ztype}', '{tileid}', '{night}',
                                           'ztile-{tileid}-{night}{jobid}.{ext}')

        succinct_expid = ''
        if len(row['EXPID']) == 1:
            succinct_expid = str(row['EXPID'][0])
        else:
            str_expids = np.sort(row['EXPID']).astype(str)
            for i in range(len(str_expids[0])):
                ith_digit = str_expids[0][i]
                if np.all([ith_digit == expid[i] for expid in str_expids]):
                    succinct_expid += ith_digit
                else:
                    succinct_expid += f'[{str_expids[0][i:]}-{str_expids[-1][i:]}]'
                    break

        obstype = str(row['OBSTYPE']).lower().strip()
        
        zfild_tid = tileid.zfill(6)
        linkloc = f"https://data.desi.lbl.gov/desi/target/fiberassign/tiles/" \
                  + f"trunk/{zfild_tid[0:3]}/fiberassign-{zfild_tid}.png"
        tileid_str = _hyperlink(linkloc, tileid)

        tilematches = exptab[exptab['TILEID'] == int(tileid)]
        if len(tilematches) == 0:
            log.error(f"Tile {tileid} found in processing table not present "
                  + f"in exposure table.")
            log.info(f"exptab tileids: {np.unique(exptab['TILEID'].data)}!")
            log.info(f"proctab tileids: {np.unique(proctab['TILEID'].data)}!")
            continue
        exptab_row = tilematches[0]
        #exptime = np.round(exptab_row['EXPTIME'], decimals=1)
        # laststep = str(row['LASTSTEP'])

        ## temporary hack to remove annoying "aborted exposure" comments that happened on every exposure in SV3
        comments = []
        if ztype == 'perexp' or len(tilematches) == 1:
            for comment in list(exptab_row['COMMENTS']):
                if 'For EXPTIME: req=' not in comment:
                    comments.append(comment)
            comments = ', '.join(comments)
        else:
            for erow in tilematches:
                ecomments = []
                for ecomment in list(erow['COMMENTS']):
                    if 'For EXPTIME: req=' not in ecomment:
                        ecomments.append(ecomment)
                if len(ecomments) > 0:
                    comments.append(f"{erow['EXPID']}: " + ', '.join(ecomments))
            comments = '; '.join(comments)

        if 'FA_SURV' in exptab_row.colnames and exptab_row['FA_SURV'] != 'unknown':
            fasurv = exptab_row['FA_SURV']
        else:
            fasurv = 'unkwn'
        if 'FAPRGRM' in exptab_row.colnames and exptab_row['FAPRGRM'] != 'unknown':
            faprog = exptab_row['FAPRGRM']
        else:
            faprog = 'unkwn'

        if ztype in expected_by_type.keys():
            expected = expected_by_type[ztype].copy()
            terminal_step = terminal_steps[ztype]
        else:
            expected = expected_by_type['null'].copy()
            terminal_step = None

        # if laststep != 'all':
        #     expected = expected_by_type['null'].copy()
        #     terminal_step = None

        nfiles = dict()
        for ftype in ['spectra', 'coadd', 'redrock', 'rrdetails', 'tile-qa',
                      'zmtl', 'qso_qn', 'qso_mgii', 'emline']:
            nfiles[ftype] = count_num_files(ztype, ftype, tileid,
                                            zfild_expid, night)
        ## Commented out: Count regardless, just don't expect them if flags are false
        # if doem:
        #     nfiles['emfit'] = count_num_files(ztype, 'emfit', tileid, expid, night)
        # if doqso:
        #     nfiles['qn'] = count_num_files(ztype, 'qn', tileid, expid, night)
        #     nfiles['mgii'] = count_num_files(ztype, 'mgii', tileid, expid, night)
        # if dotileqa:
        #     nfiles['tileqa'] = count_num_files(ztype, 'tileqa', tileid, expid, night)

        npossible = nspecs
        true_terminal_step = terminal_step
        if terminal_step is not None:
            if terminal_step == 'tile-qa':
                npossible = 2
            elif terminal_step == 'rr':
                true_terminal_step = 'rrdetails'
            elif terminal_step == 'qso':
                true_terminal_step = 'qso_qn'
            elif terminal_step == 'em':
                true_terminal_step = 'emline'

        if true_terminal_step is None:
            row_color = 'NULL'
        elif expected[terminal_step] == 0:
            row_color = 'NULL'
        elif nfiles[true_terminal_step] == 0:
            row_color = 'BAD'
        elif nfiles[true_terminal_step] < npossible:
            row_color = 'INCOMPLETE'
        elif nfiles[true_terminal_step] == npossible:
            row_color = 'GOOD'
        else:
            row_color = 'OVERFULL'

        if unique_key in uniqs_processing:
            status = row['STATUS']
        elif unique_key in unaccounted_for_tileids:
            status = 'unrecorded'
        else:
            status = 'unprocessed'

        slurm_hlink, log_hlink = '----', '----'
        if row_color not in ['GOOD', 'NULL']:
            templatelog = logfiletemplate.format(ztype=ztype, tileid=tileid,
                                                 night=night, zexpid=zfild_expid,
                                                 jobid='*', ext='log')
            lognames = glob.glob(templatelog)
            ## If that template had no results, try to old naming scheme for results
            if len(lognames) == 0:
                templatelog = templatelog.replace(f'{ztype}-', 'coadd-redshifts-')
                lognames = glob.glob(templatelog)

            newest_jobid, logfile = 0, None
            for log in lognames:
                jobid = int(log.split('-')[-1].split('.')[0])
                if jobid > newest_jobid:
                    newest_jobid = jobid
                    logname = log
            if newest_jobid > 0:
                slurmname = logname.replace(f'-{jobid}.log', '.slurm')
                slurm_hlink = _hyperlink(os.path.relpath(slurmname, webpage), 'Slurm')
                log_hlink = _hyperlink(os.path.relpath(logname, webpage), 'Log')

        rd = dict()
        rd["COLOR"] = row_color
        rd["TILEID"] = tileid_str
        rd["ZTYPE"] = ztype
        rd["EXPIDS"] = succinct_expid
        rd["FA SURV"] = fasurv
        rd["FA PRGRM"] = faprog
        # rd["LAST STEP"] = laststep
        # rd["EXP TIME"] = str(exptime)
        rd["PROC CAMWORD"] = proccamword
        for ftype in ['spectra', 'coadd', 'redrock', 'rrdetails']:
            rd[ftype.upper()] = _str_frac(nfiles[ftype], nspecs * expected['rr'])
        rd['TILEQA'] = _str_frac(nfiles['tile-qa'], 2 * expected['tile-qa'])
        rd['ZMTL'] = _str_frac(nfiles['zmtl'], nspecs * expected['zmtl'])
        rd['QN'] = _str_frac(nfiles['qso_qn'], nspecs * expected['qso'])
        rd['MGII'] = _str_frac(nfiles['qso_mgii'], nspecs * expected['qso'])
        rd['EMLINE'] = _str_frac(nfiles['emline'], nspecs * expected['em'])
        rd["SLURM FILE"] = slurm_hlink
        rd["LOG FILE"] = log_hlink
        rd["COMMENTS"] = comments
        rd["STATUS"] = status
        output[unique_key] = rd.copy()
    return output

def count_num_files(ztype, ftype, tileid, expid, night):
    filename = findfile(filetype='spectra', night=int(night), expid=int(expid),
                        camera='b1', tile=int(tileid), groupname=ztype,
                        spectrograph=1)

    if ftype == 'tile-qa':
        fileroot = filename.replace(f'spectra-1-', f'{ftype}-').split('.')[0]
    else:
        fileroot = filename.replace(f'spectra-1-', f'{ftype}-?-').split('.')[0]

    fileglob = f'{fileroot}.*'
    return len(glob.glob(fileglob))

if __name__ == "__main__":
    args = parse(options=None)
    main(args)
