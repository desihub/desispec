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

# import desispec.io.util
from desispec.workflow.exptable import get_exposure_table_pathname, \
    default_obstypes_for_exptable, \
    get_exposure_table_column_types, \
    get_exposure_table_column_defaults
from desispec.workflow.proc_dashboard_funcs import get_skipped_ids, \
    return_color_profile, find_new_exps, _hyperlink, _str_frac, \
    get_output_dir, get_nights_dict, make_html_page, read_json, write_json, \
    get_terminal_steps, get_tables
from desispec.workflow.proctable import get_processing_table_pathname, \
    table_row_to_dict
from desispec.workflow.tableio import load_table
from desispec.io.meta import specprod_root, rawdata_root
from desispec.io.util import decode_camword, camword_to_spectros, \
    difference_camwords, parse_badamps, create_camword, camword_intersection


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
    parser.add_argument('--output-name', type=str, default='dashboard.html',
                        help="name of the html page (to be placed in --output-dir).")
    parser.add_argument('--specprod', type=str,
                        help="overwrite the environment keyword for $SPECPROD")
    parser.add_argument("-e", "--skip-expid-file", type=str, required=False,
                        help="Relative pathname for file containing expid's to skip. " + \
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
    if not isinstance(args, argparse.Namespace):
        args = parse(args)

    args.show_null = True
    output_dir, prod_dir = get_output_dir(args.redux_dir, args.specprod,
                                          args.output_dir, makedir=True)
    os.makedirs(os.path.join(output_dir, 'expjsons'), exist_ok=True)
    
    ############
    ## Input ###
    ############
    if args.skip_expid_file is not None:
        skipd_expids = set(
            get_skipped_ids(args.skip_expid_file, skip_ids=True))
    else:
        skipd_expids = None

    nights_dict, nights = get_nights_dict(args.nights, args.start_night,
                                          args.end_night, prod_dir)

    print(f'Searching {prod_dir} for: {nights}')

    monthly_tables = {}
    for month, nights_in_month in nights_dict.items():
        print("Month: {}, nights: {}".format(month, nights_in_month))
        nightly_tables = {}
        for night in nights_in_month:
            ## Load previous info if any
            filename_json = os.path.join(output_dir, 'expjsons',
                                         f'expinfo_{os.environ["SPECPROD"]}'
                                         + f'_{night}.json')
            night_json_info = None
            if not args.ignore_json_archive:
                night_json_info = read_json(filename_json=filename_json)

            ## get the per exposure info for a night
            night_info = populate_night_info(night, args.check_on_disk,
                                             night_json_info=night_json_info,
                                             skipd_expids=skipd_expids)
            nightly_tables[night] = night_info.copy()

            ## write out the night_info to json file
            write_json(output_data=night_info, filename_json=filename_json)

        monthly_tables[month] = nightly_tables.copy()

    outfile = os.path.abspath(os.path.join(output_dir, args.output_name))
    make_html_page(monthly_tables, outfile, titlefill='Exp. Processing',
                   show_null=args.show_null)



def populate_night_info(night, check_on_disk=False,
                        night_json_info=None, skipd_expids=None):
    """
    For a given night, return the file counts and other other information for each exposure taken on that night
    input: night
    output: a dictionary containing the statistics with expid as key name
    FLAVOR: FLAVOR of this exposure
    OBSTYPE: OBSTYPE of this exposure
    EXPTIME: Exposure time
    SPECTROGRAPHS: a list of spectrographs used
    n_spectrographs: number of spectrographs
    n_psf: number of PSF files
    n_ff:  number of fiberflat files
    n_frame: number of frame files
    n_sframe: number of sframe files
    n_cframe: number of cframe files
    n_sky: number of sky files
    """
    if skipd_expids is None:
        skipd_expids = []
    ## Note that the following list should be in order of processing. I.e. the first filetype given should be the
    ## first file type generated. This is assumed for the automated "terminal step" determination that follows
    expected_by_type = dict()
    expected_by_type['zero'] =     {'psf': 0,    'frame': 0, 'ff': 0,
                                    'sframe': 0, 'std': 0,   'cframe': 0}
    expected_by_type['arc'] =      {'psf': 1,    'frame': 0, 'ff': 0,
                                    'sframe': 0, 'std': 0,   'cframe': 0}
    expected_by_type['cteflat'] =  {'psf': 1,    'frame': 1, 'ff': 0,
                                    'sframe': 0, 'std': 0,   'cframe': 0}
    expected_by_type['flat'] =     {'psf': 1,    'frame': 1, 'ff': 1,
                                    'sframe': 0, 'std': 0,   'cframe': 0}
    expected_by_type['nightlyflat'] = {'psf': 0,    'frame': 0, 'ff': 1,
                                    'sframe': 0, 'std': 0,   'cframe': 0}
    expected_by_type['science'] =  {'psf': 1,    'frame': 1, 'ff': 0,
                                    'sframe': 1, 'std': 1,   'cframe': 1}
    expected_by_type['twilight'] = {'psf': 1,    'frame': 1, 'ff': 0,
                                    'sframe': 0, 'std': 0,   'cframe': 0}
    expected_by_type['dark'] = expected_by_type['zero']
    expected_by_type['ccdcalib'] = expected_by_type['zero']
    expected_by_type['psfnight'] = expected_by_type['arc']
    expected_by_type['sky'] = expected_by_type['science']
    expected_by_type['null'] = expected_by_type['zero']

    ## Determine the last filetype that is expected for each obstype
    terminal_steps = get_terminal_steps(expected_by_type)

    specproddir = specprod_root()
    webpage = os.environ['DESI_DASHBOARD']
    logpath = os.path.join(specproddir, 'run', 'scripts', 'night', night)

    exptab, proctab, \
    unaccounted_for_expids,\
    unaccounted_for_tileids = get_tables(night, check_on_disk=check_on_disk,
                                                exptab_colnames=None)

    preproc_glob = os.path.join(specproddir, 'preproc',
                                str(night), '[0-9]*[0-9]')
    expid_processing = set(
        [int(os.path.basename(fil)) for fil in glob.glob(preproc_glob)])

    ## Add a new indexing column to include calibnight rows in correct location
    exptab.add_column(Table.Column(data=2*np.arange(1,len(exptab)+1),name="ORDER"))
    if proctab is not None and len(proctab) > 0:
        new_proc_expids = set(np.concatenate(proctab['EXPID']).astype(int))
        expid_processing.update(new_proc_expids)
        for jobdesc in ['ccdcalib', 'psfnight', 'nightlyflat']:
            if jobdesc in proctab['JOBDESC']:
                jobrow = proctab[proctab['JOBDESC']==jobdesc][0]
                expids = jobrow['EXPID']
                lastexpid = expids[-1]
                if lastexpid in exptab['EXPID']:
                    joint_erow = table_row_to_dict(exptab[exptab['EXPID']==lastexpid][0])
                    joint_erow['OBSTYPE'] = jobdesc
                    joint_erow['ORDER'] = erow['ORDER']+1

                ## Derive the appropriate PROCCAMWORD from the exposure table
                pcamwords = []
                for expid in expids:
                    if expid in exptab['EXPID']:
                        erow = table_row_to_dict(exptab[exptab['EXPID'] == expid][0])
                        if 'BADCAMWORD' in erow:
                            pcamword = difference_camwords(erow['CAMWORD'], erow['BADCAMWORD'])
                        else:
                            pcamword = erow['CAMWORD']
                        if len(erow['BADAMPS']) > 0:
                            badcams = []
                            for (camera, petal, amplifier) in parse_badamps(erow['BADAMPS']):
                                badcams.append(f'{camera}{petal}')
                            badampcamword = create_camword(list(set(badcams)))
                            pcamword = difference_camwords(pcamword, badampcamword)
                        pcamwords.append(pcamword)

                if len(pcamwords) == 0:
                    print(f"Couldn't find exposures {expids} for joint job {jobdesc}")
                    continue
                ## For flats we want any camera that exists in all 12 exposures
                ## For arcs we want any camera that exists in at least 3 exposures
                if jobdesc == 'nightlyflat':
                    joint_erow['CAMWORD'] = camword_intersection(pcamwords,
                                                       full_spectros_only=False)
                elif jobdesc == 'psfnight':
                    ## Count number of exposures each camera is present for
                    camcheck = {}
                    for camword in pcamwords:
                        for cam in decode_camword(camword):
                            if cam in camcheck:
                                camcheck[cam] += 1
                            else:
                                camcheck[cam] = 1
                    ## if exists in 3 or more exposures, then include it
                    goodcams = []
                    for cam, camcount in camcheck.items():
                        if camcount >= 3:
                            goodcams.append(cam)
                    joint_erow['CAMWORD'] = create_camword(goodcams)

                joint_erow['BADCAMWORD'] = ''
                joint_erow['BADAMPS'] = ''
                exptab.add_row(joint_erow)

    del proctab
    exptab.sort(['ORDER'])

    logfiletemplate = os.path.join(logpath,
                                   '{pre}-{night}-{zexpid}-{specs}{jobid}.{ext}')
    fileglob_template = os.path.join(specproddir, 'exposures', str(night),
                                     '{zexpid}', '{ftype}-{cam}[0-9]-{zexpid}.{ext}')
    fileglob_calib_template = os.path.join(specproddir, 'calibnight', str(night),
                                           '{ftype}-{cam}[0-9]-{night}.{ext}')

    def count_num_files(ftype, expid=None):
        if ftype == 'stdstars':
            cam = ''
        else:
            cam = '[brz]'
        if ftype == 'badcolumns':
            ext = 'csv'
        elif ftype == 'biasnight':
            ext = 'fits.gz'
        else:
            ext = 'fits*'  # - .fits or .fits.gz
        if expid is None:
            fileglob = fileglob_calib_template.format(ftype=ftype, cam=cam,
                                                      night=night, ext=ext)
        else:
            zfild_expid = str(expid).zfill(8)
            fileglob = fileglob_template.format(ftype=ftype, zexpid=zfild_expid,
                                                cam=cam, ext=ext)
        return len(glob.glob(fileglob))

    output = dict()
    lasttile, first_exp_of_tile = None, None
    for row in exptab:
        expid = int(row['EXPID'])
        if expid in skipd_expids:
            continue
        obstype = str(row['OBSTYPE']).lower().strip()
        key = f'{obstype}_{expid}'
        ## For those already marked as GOOD or NULL in cached rows, take that and move on
        if night_json_info is not None and key in night_json_info \
                and night_json_info[key]["COLOR"] in ['GOOD', 'NULL']:
            output[key] = night_json_info[key]
            continue

        zfild_expid = str(expid).zfill(8)
        tileid = str(row['TILEID'])
        if obstype == 'science':
            zfild_tid = tileid.zfill(6)
            linkloc = f"https://data.desi.lbl.gov/desi/target/fiberassign/tiles/" \
                      + f"trunk/{zfild_tid[0:3]}/fiberassign-{zfild_tid}.png"
            tileid_str = _hyperlink(linkloc, tileid)
            if lasttile != tileid:
                first_exp_of_tile = zfild_expid
                lasttile = tileid
        elif obstype == 'zero':  # or obstype == 'other':
            continue
        else:
            tileid_str = '----'

        exptime = np.round(row['EXPTIME'], decimals=1)
        proccamword = row['CAMWORD']
        if 'BADCAMWORD' in exptab.colnames:
            proccamword = difference_camwords(proccamword, row['BADCAMWORD'])
        if obstype != 'science' and 'BADAMPS' in exptab.colnames and row['BADAMPS'] != '':
            badcams = []
            for (camera, petal, amplifier) in parse_badamps(row['BADAMPS']):
                badcams.append(f'{camera}{petal}')
            badampcamword = create_camword(list(set(badcams)))
            proccamword = difference_camwords(proccamword, badampcamword)

        cameras = decode_camword(proccamword)
        nspecs = len(camword_to_spectros(proccamword, full_spectros_only=False))
        ncams = len(cameras)

        laststep = str(row['LASTSTEP'])
        ## temporary hack to remove annoying "aborted exposure" comments that happened on every exposure in SV3
        comments = list(row['COMMENTS'])
        bad_ind = None
        for ii, comment in enumerate(comments):
            if 'For EXPTIME: req=' in comment:
                bad_ind = ii
        if bad_ind is not None:
            comments.pop(bad_ind)
        comments = ', '.join(comments)

        if 'FA_SURV' in row.colnames and row['FA_SURV'] != 'unknown':
            fasurv = row['FA_SURV']
        else:
            fasurv = 'unkwn'
        if 'FAPRGRM' in row.colnames and row['FAPRGRM'] != 'unknown':
            faprog = row['FAPRGRM']
        else:
            faprog = 'unkwn'
        if obstype not in ['science', 'twilight']:
            if fasurv == 'unkwn':
                fasurv = '----'
            if faprog == 'unkwn':
                faprog = '----'

        derived_obstype = obstype
        if obstype == 'flat' and exptime < 2.0:
            derived_obstype = 'cteflat'

        if derived_obstype in expected_by_type.keys():
            expected = expected_by_type[derived_obstype].copy()
            terminal_step = terminal_steps[derived_obstype]
        else:
            expected = expected_by_type['null'].copy()
            terminal_step = None

        if laststep == 'ignore':
            expected = expected_by_type['null'].copy()
            terminal_step = None
        elif laststep != 'all' and obstype == 'science':
            if laststep == 'skysub':
                expected['std'] = 0
                expected['cframe'] = 0
                terminal_step = 'sframe'
            elif laststep == 'fluxcal':
                pass
            else:
                print(
                    f"WARNING: didn't understand science exposure expid={expid} of night {night}: laststep={laststep}")
        elif laststep != 'all' and obstype != 'science':
            print(
                f"WARNING: didn't understand non-science exposure expid={expid} of night {night}: laststep={laststep}")

        nfiles = {step:0 for step in ['psf','frame','ff','sky','sframe','std','cframe']}
        if obstype == 'arc':
            nfiles['psf'] = count_num_files(ftype='fit-psf', expid=expid)
        elif obstype == 'psfnight':
            nfiles['psf'] = count_num_files(ftype='psfnight')
        elif obstype != 'nightlyflat':
            nfiles['psf'] = count_num_files(ftype='psf', expid=expid)

        if obstype in ['ccdcalib', 'psfnight']:
            pass
        elif obstype == 'nightlyflat':
            nfiles['ff'] = count_num_files(ftype='fiberflatnight')
        else:
            nfiles['frame'] = count_num_files(ftype='frame', expid=expid)
            nfiles['ff'] = count_num_files(ftype='fiberflat', expid=expid)
            nfiles['sky'] = count_num_files(ftype='sky', expid=expid)
            nfiles['sframe'] = count_num_files(ftype='sframe', expid=expid)
            nfiles['std'] = count_num_files(ftype='stdstars', expid=expid)
            nfiles['cframe'] = count_num_files(ftype='cframe', expid=expid)

        if terminal_step == 'std':
            nexpected = nspecs
        else:
            nexpected = ncams

        if terminal_step is None:
            row_color = 'NULL'
        elif expected[terminal_step] == 0:
            row_color = 'NULL'
        elif nfiles[terminal_step] == 0:
            row_color = 'BAD'
        elif nfiles[terminal_step] < nexpected:
            row_color = 'INCOMPLETE'
        elif nfiles[terminal_step] == nexpected:
            row_color = 'GOOD'
        else:
            row_color = 'OVERFULL'

        if expid in expid_processing:
            status = 'processing'
        elif expid in unaccounted_for_expids:
            status = 'unaccounted'
        else:
            status = 'unprocessed'

        slurm_hlink, log_hlink = '----', '----'
        if row_color not in ['GOOD', 'NULL'] and obstype.lower() in ['arc',
                                                                     'flat',
                                                                     'science']:
            file_head = obstype.lower()
            lognames = glob.glob(
                logfiletemplate.format(pre=file_head, night=night,
                                       zexpid=zfild_expid, specs='*', jobid='',
                                       ext='log'))
            ## If no unified science script, identify which log to point to
            if obstype.lower() == 'science' and len(lognames) == 0:
                ## First chronologically is the prestdstar
                lognames = glob.glob(logfiletemplate.format(pre='prestdstar',
                                                            night=night,
                                                            zexpid=zfild_expid,
                                                            specs='*', jobid='',
                                                            ext='log'))
                file_head = 'prestdstar'
                lognames_std = glob.glob(
                    logfiletemplate.format(pre='stdstarfit',
                                           night=night,
                                           zexpid=first_exp_of_tile,
                                           specs='*', jobid='', ext='log'))
                ## If stdstar logs exist and we have all files for prestdstar
                ## link to stdstar
                if nfiles['sframe'] == ncams and len(lognames_std) > 0:
                    lognames = lognames_std
                    file_head = 'stdstarfit'
                    lognames_post = glob.glob(
                        logfiletemplate.format(pre='poststdstar',
                                               night=night, zexpid=zfild_expid,
                                               specs='*', jobid='', ext='log'))
                    ## If poststdstar logs exist and we have all files for stdstar
                    ## link to poststdstar
                    if nfiles['std'] == nspecs and len(lognames_post) > 0:
                        lognames = lognames_post
                        file_head = 'poststdstar'

            newest_jobid = '00000000'
            spectrographs = ''

            for log in lognames:
                jobid = log[-12:-4]
                if int(jobid) > int(newest_jobid):
                    newest_jobid = jobid
                    spectrographs = log.split('-')[-2]
            if newest_jobid != '00000000' and len(spectrographs) != 0:
                if file_head == 'stdstarfit':
                    zexp = first_exp_of_tile
                else:
                    zexp = zfild_expid
                logname = logfiletemplate.format(pre=file_head, night=night,
                                                 zexpid=zexp,
                                                 specs=spectrographs,
                                                 jobid='-' + newest_jobid,
                                                 ext='log')
                slurmname = logfiletemplate.format(pre=file_head, night=night,
                                                   zexpid=zexp,
                                                   specs=spectrographs,
                                                   jobid='', ext='slurm')

                slurm_hlink = _hyperlink(os.path.relpath(slurmname, webpage),
                                         'Slurm')
                log_hlink = _hyperlink(os.path.relpath(logname, webpage), 'Log')

        rd = dict()
        rd["COLOR"] = row_color
        rd["EXPID"] = str(expid)
        rd["TILEID"] = tileid_str
        rd["OBSTYPE"] = obstype
        rd["FA SURV"] = fasurv
        rd["FA PRGRM"] = faprog
        rd["LAST STEP"] = laststep
        rd["EXP TIME"] = str(exptime)
        rd["PROC CAMWORD"] = proccamword
        rd["PSF"] = _str_frac(nfiles['psf'], ncams * expected['psf'])
        rd["FRAME"] = _str_frac(nfiles['frame'],
                                        ncams * expected['frame'])
        rd["FFLAT"] = _str_frac(nfiles['ff'], ncams * expected['ff'])
        rd["SFRAME"] = _str_frac(nfiles['sframe'],
                                        ncams * expected['sframe'])
        rd["SKY"] = _str_frac(nfiles['sky'],
                                        ncams * expected['sframe'])
        rd["STD"] = _str_frac(nfiles['std'],
                                        nspecs * expected['std'])
        rd["CFRAME"] = _str_frac(nfiles['cframe'],
                                        ncams * expected['cframe'])
        rd["SLURM FILE"] = slurm_hlink
        rd["LOG FILE"] = log_hlink
        rd["COMMENTS"] = comments
        rd["STATUS"] = status
        output[key] = rd.copy()
    return output


if __name__ == "__main__":
    args = parse(options=None)
    main(args)
