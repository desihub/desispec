"""
desispec.scripts.procdashboard
==============================

"""
import argparse
import os, glob
from astropy.table import Table
import numpy as np

from desispec.workflow.proc_dashboard_funcs import get_skipped_ids, \
    _hyperlink, _str_frac, \
    get_output_dir, make_html_page, read_json, write_json, \
    get_terminal_steps, get_tables, populate_monthly_tables, get_nights
from desispec.workflow.proctable import table_row_to_dict
from desispec.workflow.queue import update_from_queue, get_non_final_states
from desispec.io.meta import specprod_root, get_readonly_filepath
from desispec.io.util import decode_camword, camword_to_spectros, \
    difference_camwords, erow_to_goodcamword
from desiutil.log import get_logger


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
    parser.add_argument('--nproc', type=int, default=1, required=False,
                        help="The number of processors to use with multiprocessing. " +
                             "Default is 1.")
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
    """ Code to generate a webpage for monitoring the spectra processing in a production.

    Args:
        args (argparse.Namespace): The arguments generated from
            desispec.scripts.procdashboard.parse()
    """
    log = get_logger()
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

    nights = get_nights(args.nights, args.start_night, args.end_night, prod_dir)
    log.info(f'Searching {prod_dir} for: {nights}')

    ## Define location of cache files
    archive_fname_template = os.path.join(output_dir, 'expjsons',
                                  f'expinfo_{os.environ["SPECPROD"]}'
                                  + '_{night}.json')

    ## Assign additional function arguments to dictionary to pass in
    func_args = {'check_on_disk': args.check_on_disk,
                 'skipd_expids': skipd_expids}

    ## Bundle the information together so that we can properly run it in parallel
    daily_info_args = {'pernight_info_func': populate_exp_night_info,
                       'pernight_info_func_args': func_args,
                       'archive_fname_template': archive_fname_template,
                       'ignore_json_archive': args.ignore_json_archive}

    monthly_tables = populate_monthly_tables(nights=nights,
                                             daily_info_args=daily_info_args,
                                             nproc=args.nproc)

    outfile = os.path.abspath(os.path.join(output_dir, args.output_name))
    make_html_page(monthly_tables, outfile, titlefill='Exp. Processing',
                   show_null=args.show_null)

def populate_exp_night_info(night, night_json_info=None, check_on_disk=False, skipd_expids=None):
    """
    Use all available information in the SPECPROD to determine whether specific
    jobs and exposures have been successfully processed or not based on the existence
    of files on disk.

    Args:
        night (int): the night to check the status of the processing for.
        night_json_info (dict of dicts): Dictionary of dictionarys. See output
            definition for format.
        check_on_disk (bool, optional): True if you want to submit
            other jobs even the loaded processing table has incomplete jobs in
            it. Use with caution. Default is False.
        skipd_expids (bool, optional): Default is False. If False,
            the code checks for the existence of the expected final data
            products for the script being submitted. If all files exist and
            this is False, then the script will not be submitted. If some
            files exist and this is False, only the subset of the cameras
            without the final data products will be generated and submitted.

    Returns:
        output (dict of dicts): keys are generally JOBDESC_EXPID. Each value
            is a dict with keys of the column names and values as the elements
            of the row in the table for each column. The one exception is COLOR
            which is used to define the coloring of the row in the dashboard.
            Current keys are: "COLOR", "EXPID", "TILEID", "OBSTYPE", "FA SURV",
            "FA PRGRM", "LAST STEP", "EXP TIME" ,"PROC CAMWORD", "PSF", "FFLAT",
            "FRAME", "SFRAME", "SKY", "STD", "CFRAME", "SLURM FILE", "LOG FILE",
            "COMMENTS", and "STATUS".
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

    ## Get non final Slurm states
    non_final_states = get_non_final_states()

    specproddir = specprod_root()
    webpage = os.environ['DESI_DASHBOARD']

    exptab, proctab, \
    unaccounted_for_expids,\
    unaccounted_for_tileids = get_tables(night, check_on_disk=check_on_disk,
                                                exptab_colnames=None)

    preproc_glob = os.path.join(specproddir, 'preproc',
                                str(night), '[0-9]*[0-9]')
    expid_processing = set(
        [int(os.path.basename(fil)) for fil in glob.glob(preproc_glob)])

    ## Add a new indexing column to include calibnight rows in correct location
    exptab.add_column(Table.Column(data=2 * np.arange(1,len(exptab)+1),name="ORDER"))
    exptab.add_column(Table.Column(data=[0] * len(exptab), name="PTAB_INTID"))
    exptab.add_column(Table.Column(data=[0] * len(exptab), name="LATEST_QID"))
    exptab.add_column(Table.Column(data=['unknown'] * len(exptab), name="STATUS", dtype='S20'))
    exptab.add_column(Table.Column(data=['unknown'] * len(exptab), name="JOBDESC", dtype='S20'))
    if proctab is not None and len(proctab) > 0:
        ## Update the STATUS of the
        proctab = update_from_queue(proctab)
        new_proc_expids = set(np.concatenate(proctab['EXPID']).astype(int))
        expid_processing.update(new_proc_expids)
        expjobs_ptab = proctab[np.isin(proctab['JOBDESC'],
                                       [b'arc', b'flat', b'tilenight',
                                        b'prestdstar', b'stdstar', b'poststdstar'])]
        for i,erow in enumerate(exptab):
            ## proctable has an array of expids, so check for them in a loop
            for prow in expjobs_ptab:
                if erow['EXPID'] in prow['EXPID']:
                    exptab['STATUS'][i] = prow['STATUS']
                    exptab['LATEST_QID'][i] = prow['LATEST_QID']
                    exptab['PTAB_INTID'][i] = prow['INTID']
                    exptab['JOBDESC'][i] = prow['JOBDESC']
        caljobs_ptab = proctab[np.isin(proctab['JOBDESC'],
                                       [b'ccdcalib', b'psfnight', b'nightlyflat'])]
        for prow in caljobs_ptab:
            jobdesc = prow['JOBDESC']
            expids = prow['EXPID']
            if jobdesc == 'ccdcalib':
                expid = expids[0]
            else:
                expid = expids[-1]
            if expid in exptab['EXPID']:
                joint_erow = table_row_to_dict(exptab[exptab['EXPID']==expid][0])
                joint_erow['OBSTYPE'] = jobdesc
                joint_erow['ORDER'] = joint_erow['ORDER']+1
                if len(expids) < 5:
                    joint_erow['COMMENTS'] = [f"Exposure(s) {','.join(np.array(expids).astype(str))}"]
                else:
                    joint_erow['COMMENTS'] = [f"Exposures {expids[0]}-{expids[-1]}"]
                # ## Derive the appropriate PROCCAMWORD from the exposure table
                # pcamwords = []
                # for expid in expids:
                #     if expid in exptab['EXPID']:
                #         erow = table_row_to_dict(exptab[exptab['EXPID'] == expid][0])
                #         pcamword = ''
                #         if 'BADCAMWORD' in erow:
                #             if 'BADAMPS' in erow:
                #                 pcamword = erow_to_goodcamword(erow,
                #                                                suppress_logging=True,
                #                                                exclude_badamps=False)
                #             else:
                #                 pcamword = difference_camwords(erow['CAMWORD'], erow['BADCAMWORD'])
                #         else:
                #             pcamword = erow['CAMWORD']
                #         pcamwords.append(pcamword)
                #
                # if len(pcamwords) == 0:
                #     print(f"Couldn't find exposures {expids} for joint job {jobdesc}")
                #     continue
                # ## For flats we want any camera that exists in all 12 exposures
                # ## For arcs we want any camera that exists in at least 3 exposures
                # if jobdesc == 'nightlyflat':
                #     joint_erow['CAMWORD'] = camword_intersection(pcamwords,
                #                                        full_spectros_only=False)
                # elif jobdesc == 'psfnight':
                #     ## Count number of exposures each camera is present for
                #     camcheck = {}
                #     for camword in pcamwords:
                #         for cam in decode_camword(camword):
                #             if cam in camcheck:
                #                 camcheck[cam] += 1
                #             else:
                #                 camcheck[cam] = 1
                #     ## if exists in 3 or more exposures, then include it
                #     goodcams = []
                #     for cam, camcount in camcheck.items():
                #         if camcount >= 3:
                #             goodcams.append(cam)
                #     joint_erow['CAMWORD'] = create_camword(goodcams)

                joint_erow['CAMWORD'] = prow['PROCCAMWORD']
                joint_erow['BADCAMWORD'] = ''
                joint_erow['BADAMPS'] = ''
                joint_erow['STATUS'] = prow['STATUS']
                joint_erow['LATEST_QID'] = prow['LATEST_QID']
                joint_erow['PTAB_INTID'] = prow['INTID']
                joint_erow['JOBDESC'] = prow['JOBDESC']
                exptab.add_row(joint_erow)

    del proctab
    exptab.sort(['ORDER'])

    readonly_specproddir = get_readonly_filepath(specproddir)
    logpath = os.path.join(specproddir, 'run', 'scripts', 'night', str(night))
    logfiletemplate = os.path.join(logpath, '{pre}-{night}-{zexpid}-{specs}{jobid}.{ext}')
    fileglob_template = os.path.join(readonly_specproddir, 'exposures', str(night),
                                     '{zexpid}', '{ftype}-{cam}[0-9]-{zexpid}.{ext}')
    fileglob_calib_template = os.path.join(readonly_specproddir, 'calibnight', str(night),
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

        if 'BADCAMWORD' in row.colnames:
            if 'BADAMPS' in row.colnames:
                proccamword = erow_to_goodcamword(row,
                                                  suppress_logging=True,
                                                  exclude_badamps=False)
            else:
                proccamword = difference_camwords(row['CAMWORD'],
                                                  row['BADCAMWORD'])
        else:
            proccamword = row['CAMWORD']

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
        if obstype == 'flat' and row['EXPTIME'] <= 10.0:
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
                print("WARNING: didn't understand science exposure "
                      + f"expid={expid} of night {night}: laststep={laststep}")
        elif laststep != 'all' and obstype != 'science':
            print("WARNING: didn't understand non-science exposure "
                  + f"expid={expid} of night {night}: laststep={laststep}")

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

        if expid in expid_processing:
            status = row['STATUS']
        elif expid in unaccounted_for_expids:
            status = 'unaccounted'
        else:
            status = 'unprocessed'

        if terminal_step is None:
            row_color = 'NULL'
        elif expected[terminal_step] == 0:
            row_color = 'NULL'
        elif status in non_final_states:
            row_color = status
        elif nfiles[terminal_step] == 0:
            row_color = 'BAD'
        elif nfiles[terminal_step] < nexpected:
            row_color = 'INCOMPLETE'
        elif nfiles[terminal_step] == nexpected:
            if status in ['COMPLETED', 'NULL']:
                row_color = 'GOOD'
            else:
                row_color = 'INCOMPLETE'
        else:
            row_color = 'OVERFULL'

        slurm_hlink, log_hlink = '----', '----'
        if row_color not in ['GOOD', 'NULL', 'PENDING'] \
                and obstype.lower() in ['arc', 'flat', 'science']:
            file_head = obstype.lower()
            lognames = glob.glob(
                logfiletemplate.format(pre=file_head, night=night,
                                       zexpid=zfild_expid, specs='*', jobid='',
                                       ext='log'))
            ## If no unified science script, identify which log to point to
            if row['JOBDESC'] == 'tilenight':
                file_head = 'tilenight'
                lognames = glob.glob(logfiletemplate.format(pre=file_head,
                                                            night=night,
                                                            zexpid=tileid,
                                                            specs='*', jobid='',
                                                            ext='log'))
            elif obstype.lower() == 'science' and len(lognames) == 0:
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


            newest_jobid, logfile = 0, None

            for itlog in lognames:
                jobid = int(itlog.split('-')[-1].split('.')[0])
                if jobid > newest_jobid:
                    newest_jobid = jobid
                    logname = itlog
            if newest_jobid > 0:
                slurmname = logname.replace(f'-{jobid}.log', '.slurm')
                slurm_hlink = _hyperlink(os.path.relpath(slurmname, webpage), 'Slurm')
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
        rd["FRAME"] = _str_frac(nfiles['frame'], ncams * expected['frame'])
        rd["FFLAT"] = _str_frac(nfiles['ff'], ncams * expected['ff'])
        rd["SFRAME"] = _str_frac(nfiles['sframe'], ncams * expected['sframe'])
        rd["SKY"] = _str_frac(nfiles['sky'], ncams * expected['sframe'])
        rd["STD"] = _str_frac(nfiles['std'], nspecs * expected['std'])
        rd["CFRAME"] = _str_frac(nfiles['cframe'], ncams * expected['cframe'])
        rd["SLURM FILE"] = slurm_hlink
        rd["LOG FILE"] = log_hlink
        rd["COMMENTS"] = comments
        rd["STATUS"] = status
        output[key] = rd.copy()
    return output


if __name__ == "__main__":
    args = parse(options=None)
    main(args)
