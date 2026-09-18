"""
desispec.scripts.procdashboard
==============================

"""
import argparse
import os, glob
from astropy.table import Table
import numpy as np

from desispec.workflow.exptable import get_exposure_table_column_defaults
from desispec.workflow.proc_dashboard_funcs import get_skipped_ids, \
    _hyperlink, _str_frac, \
    get_output_dir, make_html_page, read_json, write_json, \
    get_terminal_steps, get_tables, populate_monthly_tables, get_nights
from desispec.workflow.proctable import table_row_to_dict
from desispec.workflow.queue import update_from_queue, get_non_final_states, \
    get_resubmission_states
from desispec.workflow.submission import get_linkcal_refnight
from desispec.io.meta import specprod_root, get_readonly_filepath
from desispec.io.util import decode_camword, camword_to_spectros, \
    difference_camwords, erow_to_goodcamword
from desiutil.log import get_logger


## Jobs that process many exposures at once and therefore get a dashboard row
## of their own in addition to the rows of the exposures they consume. Ordered
## by where they fall in the nightly calibration sequence.
CALIB_JOBDESCS = ('linkcal', 'biasnight', 'biaspdark', 'pdark', 'ccdcalib',
                  'psfnight', 'nightlyflat', 'cteflat')

## The joint fits are named after the last exposure they consume, which is where
## they belong chronologically. The rest are named after the first.
CALIB_JOBDESCS_ANCHORED_ON_LAST_EXPID = ('psfnight', 'nightlyflat', 'cteflat')

## Jobs whose full output set cannot be counted. Use Slurm status, supplemented
## by a bias-link count for linkcal:
##  - a linkcal links whichever calibnight prefixes the night's override file
##    asks for, possibly psfnight and fiberflatnight rather than bias or dark,
##    and which those are isn't recorded in the processing table
##  - which steps a ccdcalib runs (darknight, badcolumn, ctecorr) likewise isn't
##    recorded, so which of their outputs to expect is unknowable
##  - the dark preprocs a pdark writes are large intermediates that productions
##    delete once the darknight is built, so their absence proves nothing
## Their count columns read 0/0, the same as any other job that skips a step.
STATUS_ONLY_JOBDESCS = ('linkcal', 'ccdcalib', 'pdark')

## Slurm states that mean a job is still on its way but that
## get_non_final_states() doesn't list, because the pipeline never has to act on
## them. A status-only row would otherwise read as a failure for the few seconds
## a job spends tearing down, which also turns the whole night's header orange.
TRANSIENT_SLURM_STATES = ('COMPLETING', 'CONFIGURING', 'SIGNALING', 'STAGE_OUT')

## Columns of the per-exposure dashboard table, in the order they are written.
## Used to detect json archive entries written by an older version of this code,
## which would otherwise be pasted into the html with the wrong set of columns.
DASHBOARD_COLNAMES = ['COLOR', 'EXPID', 'TILEID', 'OBSTYPE', 'FA SURV',
                      'FA PRGRM', 'LAST STEP', 'EXP TIME', 'PROC CAMWORD',
                      'BIAS', 'PSF', 'FRAME', 'FFLAT', 'SFRAME', 'SKY', 'STD',
                      'CFRAME', 'SLURM FILE', 'LOG FILE', 'COMMENTS', 'STATUS']


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

def _empty_expid_lists(nrows):
    """
    Create the data for an object column of empty exposure id arrays, one per row.

    Args:
        nrows (int): the number of rows the column should have.

    Returns:
        np.ndarray: object array of empty integer arrays.
    """
    column = np.empty(nrows, dtype=object)
    for i in range(nrows):
        column[i] = np.array([], dtype=int)
    return column


def _standalone_calib_erow(exptab, night):
    """
    Create a dashboard row for a job that has no exposure of its own on a night.

    A linkcal is the main case: its processing table EXPID column is either
    empty or holds exposures of the reference night being linked from, so there
    is no exposure table row to hang it on. Such a row is given an ORDER that
    sorts it ahead of every exposure of the night, which is where these jobs run.

    Args:
        exptab (Table): the exposure table the row will be added to.
        night (int): the night the job ran for.

    Returns:
        dict: an exposure table row with default values.
    """
    defaults = get_exposure_table_column_defaults(asdict=True)
    erow = dict()
    for col in exptab.colnames:
        if col in defaults:
            erow[col] = defaults[col]
        else:
            ## a column added below, or one too new to have a default; an empty
            ## value of the right type is enough for the ones not set here
            erow[col] = np.zeros(1, dtype=exptab[col].dtype)[0]
    erow['NIGHT'] = night
    erow['ORDER'] = 1
    erow['PTAB_INTID'] = 0
    erow['LATEST_QID'] = 0
    erow['STATUS'] = 'unknown'
    erow['JOBDESC'] = 'unknown'
    erow['PTAB_EXPIDS'] = np.array([], dtype=int)
    return erow


def _exposure_comments(exptab, expids):
    """
    Collect what the exposure table says about a job's exposures.

    A calibration that went wrong usually has its reason recorded against the
    exposures rather than the job, so a job row is worth repeating them on.
    Exposures of a set normally share their comments, so each distinct comment
    is listed once, named by exposure only when it doesn't apply to all of them.

    Args:
        exptab (Table): the night's exposure table.
        expids (np.array): the exposures the job processed.

    Returns:
        list of str: the comments, in the order they were first seen.
    """
    owners_by_comment = dict()
    for expid in expids:
        match = exptab[exptab['EXPID'] == expid]
        if len(match) == 0:
            ## a job can span nights, so not every exposure is in this table
            continue
        for comment in match['COMMENTS'][0]:
            comment = str(comment).strip()
            if comment:
                owners_by_comment.setdefault(comment, []).append(int(expid))

    comments = []
    for comment, owners in owners_by_comment.items():
        if len(owners) == len(expids):
            comments.append(comment)
        else:
            comments.append(f"{','.join(str(e) for e in owners)}: {comment}")
    return comments


def _linkcal_link_info(night, proccamword):
    """
    Work out what a night's linkcal job links, from the override that drove it.

    The processing table records neither which calibrations a linkcal links nor
    which cameras its bias link covers, so the override file is the only source.
    get_linkcal_refnight() raises on an override it can't interpret, which is
    right for submission but not here: the dashboard reads whatever a production
    happens to hold and covers every night in one process, so one malformed file
    must not take the whole run down with it.

    Args:
        night (int): the night the linkcal ran for.
        proccamword (str): the linkcal job's processing camera word, which is
            the bias link's camera set unless the override narrows it.

    Returns:
        tuple: (comment, bias_camword) where comment describes the link for the
            dashboard COMMENTS column and bias_camword is the cameras whose
            biases the link supplies: a camword, '' when it links no biases, or
            None when the override doesn't say and it can't be known. None is
            distinct from '' on purpose, since the night's own bias job is only
            excused from the cameras a link is known to cover.
    """
    try:
        refnight, files_to_link, linkcal = get_linkcal_refnight(night)
    except Exception as err:      # noqa: BLE001 - one night shouldn't break the rest
        get_logger().warning(f"Could not interpret the linkcal override of "
                             + f"{night}, so reporting the link as unknown: {err}")
        return "Links unknown calibrations from another night", None

    ## get_linkcal_refnight gives a refnight for every override it could read,
    ## so the absence of one means there was nothing to read
    if refnight is None:
        return "Links unknown calibrations from another night", None

    bias_camword = ''
    if 'biasnight' in files_to_link:
        bias_camword = linkcal.get('biaslink_camword', proccamword)
    described = ', '.join(sorted(files_to_link))
    return f"Links {described} from night {refnight}", bias_camword


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
            The keys are those listed in DASHBOARD_COLNAMES.
    """
    if skipd_expids is None:
        skipd_expids = []
    ## Note that the following list should be in order of processing. I.e. the first filetype given should be the
    ## first file type generated. This is assumed for the automated "terminal step" determination that follows
    ## 'bias' is the nightly bias, the one pre-PSF calibration product the
    ## dashboard can count, and lives in calibnight rather than in exposures
    expected_by_type = dict()
    expected_by_type['zero'] =     {'bias': 0, 'psf': 0,    'frame': 0, 'ff': 0,
                                    'sframe': 0, 'std': 0,   'cframe': 0}
    expected_by_type['arc'] =      {'bias': 0, 'psf': 1,    'frame': 0, 'ff': 0,
                                    'sframe': 0, 'std': 0,   'cframe': 0}
    expected_by_type['cteflat'] =  {'bias': 0, 'psf': 1,    'frame': 1, 'ff': 0,
                                    'sframe': 0, 'std': 0,   'cframe': 0}
    expected_by_type['flat'] =     {'bias': 0, 'psf': 1,    'frame': 1, 'ff': 1,
                                    'sframe': 0, 'std': 0,   'cframe': 0}
    expected_by_type['nightlyflat'] = {'bias': 0, 'psf': 0, 'frame': 0, 'ff': 1,
                                    'sframe': 0, 'std': 0,   'cframe': 0}
    expected_by_type['science'] =  {'bias': 0, 'psf': 1,    'frame': 1, 'ff': 0,
                                    'sframe': 1, 'std': 1,   'cframe': 1}
    expected_by_type['twilight'] = {'bias': 0, 'psf': 1,    'frame': 1, 'ff': 0,
                                    'sframe': 0, 'std': 0,   'cframe': 0}
    expected_by_type['biasnight'] = {'bias': 1, 'psf': 0,   'frame': 0, 'ff': 0,
                                    'sframe': 0, 'std': 0,   'cframe': 0}
    expected_by_type['dark'] = expected_by_type['zero']
    ## a biaspdark writes the nightly bias and then preprocesses the darks; only
    ## the bias half leaves a durable product to check
    expected_by_type['biaspdark'] = expected_by_type['biasnight']
    ## pdark and ccdcalib have nothing countable. linkcal's bias expectation is
    ## determined separately from its override, see STATUS_ONLY_JOBDESCS.
    expected_by_type['pdark'] = expected_by_type['zero']
    expected_by_type['ccdcalib'] = expected_by_type['zero']
    expected_by_type['linkcal'] = expected_by_type['zero']
    expected_by_type['psfnight'] = expected_by_type['arc']
    expected_by_type['sky'] = expected_by_type['science']
    expected_by_type['null'] = expected_by_type['zero']

    ## Determine the last filetype that is expected for each obstype
    terminal_steps = get_terminal_steps(expected_by_type)

    ## Get non final Slurm states
    non_final_states = get_non_final_states()
    ## Include pipeline submission/dependency failures as well as Slurm failures.
    failed_states = get_resubmission_states()

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
    ## Rows of jobs that span many exposures keep the full list here, since the
    ## batch script and log of such a job are named after its first exposure
    ## while the row itself may be anchored on the last
    exptab.add_column(Table.Column(data=_empty_expid_lists(len(exptab)),
                                   name="PTAB_EXPIDS"))
    ## Cameras whose nightly biases the night's linkcal supplies, which the
    ## night's own bias job is therefore not responsible for producing. None
    ## until a linkcal row says otherwise, meaning nothing is known to be linked.
    linkcal_bias_camword = None
    if proctab is not None and len(proctab) > 0:
        ## Update the STATUS of the
        proctab = update_from_queue(proctab)
        new_proc_expids = set(np.concatenate(proctab['EXPID']).astype(int))
        expid_processing.update(new_proc_expids)
        ## Calibration bundles carry one row for many exposures, so the
        ## bundle's status is what each of its exposures should show. Arcs and
        ## flats no longer have rows of their own, so without the bundle
        ## descriptors here their exposures would all read 'unknown'.
        ## biaspdark and pdark are the only jobs that touch an individual dark,
        ## so a dark's status is theirs. ccdcalib is deliberately absent: its
        ## exposure list also includes the CTE flats, whose status belongs to
        ## the flat bundle instead.
        expjobs_ptab = proctab[np.isin(proctab['JOBDESC'],
                                       [b'arc', b'flat', b'tilenight',
                                        b'prestdstar', b'stdstar', b'poststdstar',
                                        b'psfnight', b'nightlyflat', b'cteflat',
                                        b'biaspdark', b'pdark'])]
        for i,erow in enumerate(exptab):
            ## proctable has an array of expids, so check for them in a loop
            for prow in expjobs_ptab:
                if erow['EXPID'] in prow['EXPID']:
                    exptab['STATUS'][i] = prow['STATUS']
                    exptab['LATEST_QID'][i] = prow['LATEST_QID']
                    exptab['PTAB_INTID'][i] = prow['INTID']
                    exptab['JOBDESC'][i] = prow['JOBDESC']
        caljobs_ptab = proctab[np.isin(proctab['JOBDESC'],
                                       [desc.encode() for desc in CALIB_JOBDESCS])]
        ## A night can carry several identical rows for the same job, e.g. one
        ## biasnight row per night that a cross-night bias submission covered,
        ## which would otherwise become duplicate dashboard rows
        seen_caljobs = set()
        for prow in caljobs_ptab:
            jobdesc = str(prow['JOBDESC'])
            expids = np.asarray(prow['EXPID'], dtype=int)
            if len(expids) == 0:
                expid = None
            elif jobdesc in CALIB_JOBDESCS_ANCHORED_ON_LAST_EXPID:
                expid = expids[-1]
            else:
                expid = expids[0]
            if (jobdesc, expid) in seen_caljobs:
                continue
            seen_caljobs.add((jobdesc, expid))
            ## A linkcal has no exposure of its own on this night: its EXPID
            ## column is either empty or holds reference night exposures. The
            ## exposures of a cross-night pdark can likewise be missing here.
            ## Those jobs get a standalone row at the head of the night.
            if expid is not None and expid in exptab['EXPID']:
                joint_erow = table_row_to_dict(exptab[exptab['EXPID']==expid][0])
                joint_erow['ORDER'] = joint_erow['ORDER']+1
            else:
                joint_erow = _standalone_calib_erow(exptab, night)
            joint_erow['OBSTYPE'] = jobdesc
            joint_erow['PTAB_EXPIDS'] = expids
            ## the job ran over every exposure it was given, so an individual
            ## exposure being marked 'ignore' says nothing about the job itself
            joint_erow['LASTSTEP'] = 'all'
            if jobdesc == 'linkcal':
                ## the exposures a linkcal lists are those of the reference
                ## night it depends on, which would be confusing here
                comment, linkcal_bias_camword = _linkcal_link_info(
                        night, str(prow['PROCCAMWORD']))
                joint_erow['COMMENTS'] = [comment]
            elif len(expids) == 0:
                joint_erow['COMMENTS'] = []
            else:
                if len(expids) < 5:
                    described = [f"Exposure(s) {','.join(expids.astype(str))}"]
                else:
                    described = [f"Exposures {expids[0]}-{expids[-1]}"]
                ## For a handful of exposures there is room to say what the
                ## exposure table records about them as well
                if len(expids) <= 5:
                    described += _exposure_comments(exptab, expids)
                joint_erow['COMMENTS'] = described
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

    def count_num_files(ftype, expid=None, links=None):
        """Count matching products, optionally selecting links or non-links."""
        if ftype == 'stdstars':
            cam = ''
        else:
            cam = '[brz]'
        if ftype == 'badcolumns':
            ext = 'csv'
        else:
            ext = 'fits*'  # - .fits or .fits.gz
        if expid is None:
            fileglob = fileglob_calib_template.format(ftype=ftype, cam=cam,
                                                      night=night, ext=ext)
        else:
            zfild_expid = str(expid).zfill(8)
            fileglob = fileglob_template.format(ftype=ftype, zexpid=zfild_expid,
                                                cam=cam, ext=ext)
        filenames = glob.glob(fileglob)
        if ftype == 'biasnight':
            filenames = [filename for filename in filenames if filename.endswith(('.fits', '.fits.gz'))]
        if links is not None:
            ## glob lists a symlink whose target is gone, and islink() is True
            ## for it, so require that a link actually resolves. A linkcal that
            ## leaves dangling links is the failure this is meant to catch.
            filenames = [filename for filename in filenames
                         if os.path.islink(filename) == links
                         and os.path.exists(filename)]
        return len(filenames)

    output = dict()
    lasttile, first_exp_of_tile = None, None
    for row in exptab:
        expid = int(row['EXPID'])
        if expid in skipd_expids:
            continue
        obstype = str(row['OBSTYPE']).lower().strip()
        ## Zeros and darks produce nothing any of the columns track, so a row of
        ## their own is always an empty gray one. What is made from them is
        ## reported by the biasnight, biaspdark, pdark and ccdcalib rows, which
        ## reach here carrying their job description rather than 'zero'/'dark'.
        ## This has to precede the archive lookup below, or rows cached before
        ## darks were dropped would be restored from it.
        if obstype in ('zero', 'dark'):
            continue
        key = f'{obstype}_{expid}'
        ## For those already marked as GOOD or NULL in cached rows, take that and move on.
        ## A row cached by an older version of this code can have a different set
        ## of columns, which would misalign the html table, so regenerate those.
        ## Refresh bias jobs and linkcal because cached counts may predate the
        ## distinction between regular files and links, or the override changed.
        ## Status-based jobs must reflect the current queue state, including
        ## jobs that older dashboard versions incorrectly cached as NULL.
        if night_json_info is not None and key in night_json_info \
                and night_json_info[key]["COLOR"] in ['GOOD', 'NULL'] \
                and list(night_json_info[key].keys()) == DASHBOARD_COLNAMES \
                and obstype not in ('biasnight', 'biaspdark') + STATUS_ONLY_JOBDESCS:
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
        ## semicolons, not commas: a comment can itself hold a comma separated
        ## list of exposures, which a comma here would read as part of
        comments = '; '.join(comments)

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

        nfiles = {step:0 for step in ['bias','psf','frame','ff','sky','sframe',
                                      'std','cframe']}
        nbias_expected = ncams * expected['bias']
        if obstype in ['biasnight', 'biaspdark']:
            ## These write one nightly bias per camera and nothing else that the
            ## other columns track. Count all non-links so extra files are still
            ## flagged, while biases supplied by linkcal are counted separately.
            nfiles['bias'] = count_num_files(ftype='biasnight', links=False)
            ## Cameras the night's linkcal supplies aren't this job's to write.
            ## Usually there is no overlap, since a bias job is only submitted
            ## for the cameras a link doesn't cover, but a night can still carry
            ## a leftover bias row for cameras that ended up linked instead.
            nbias_expected = len(decode_camword(difference_camwords(
                    proccamword, linkcal_bias_camword or '',
                    suppress_logging=True)))
        elif obstype == 'linkcal':
            if linkcal_bias_camword is None:
                ## nothing readable says what was linked, so leave the count
                ## blank rather than guess, and judge the row by its status
                pass
            else:
                nfiles['bias'] = count_num_files(ftype='biasnight', links=True)
                nbias_expected = len(decode_camword(linkcal_bias_camword))
        elif obstype in STATUS_ONLY_JOBDESCS:
            ## Nothing countable, the coloring below falls back on the status
            pass
        else:
            if obstype == 'arc':
                nfiles['psf'] = count_num_files(ftype='fit-psf', expid=expid)
            elif obstype == 'psfnight':
                nfiles['psf'] = count_num_files(ftype='psfnight')
            elif obstype != 'nightlyflat':
                nfiles['psf'] = count_num_files(ftype='psf', expid=expid)

            if obstype == 'psfnight':
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
        elif terminal_step == 'bias':
            ## not simply ncams: a linkcal can supply some of the cameras
            nexpected = nbias_expected
        else:
            nexpected = ncams

        ## A job taken from the processing table knows its own status even when
        ## it has no exposure of this night to be found under preproc
        if expid in expid_processing or row['PTAB_INTID'] > 0:
            status = row['STATUS']
        elif expid in unaccounted_for_expids:
            status = 'unaccounted'
        else:
            status = 'unprocessed'

        ## Use queue status where the full output set cannot be enumerated,
        ## but a completed linkcal must also have the expected bias-link count.
        if obstype in STATUS_ONLY_JOBDESCS:
            if status in non_final_states:
                row_color = status
            elif status in TRANSIENT_SLURM_STATES:
                ## on its way, not a failure, but with no color of its own, so
                ## show it like any other job that hasn't finished
                row_color = 'PENDING'
            elif status in failed_states:
                row_color = 'BAD'
            elif status == 'COMPLETED':
                known_bias_link = (obstype == 'linkcal'
                                   and linkcal_bias_camword is not None)
                if known_bias_link and nfiles['bias'] > nbias_expected:
                    row_color = 'OVERFULL'
                elif known_bias_link and nfiles['bias'] < nbias_expected:
                    row_color = 'BAD' if nfiles['bias'] == 0 else 'INCOMPLETE'
                else:
                    row_color = 'GOOD'
            else:
                ## A job with unresolved status is not an intentional no-op.
                row_color = 'INCOMPLETE'
        elif terminal_step is None:
            row_color = 'NULL'
        elif expected[terminal_step] == 0 or nexpected == 0:
            ## nothing was expected of this job, e.g. a bias job for a night
            ## whose biases all came from a link instead
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
                and obstype.lower() in ('arc', 'flat', 'science') + CALIB_JOBDESCS:
            file_head = obstype.lower()
            if obstype in CALIB_JOBDESCS:
                ## Calibration jobs are named after the first exposure they
                ## processed, which isn't necessarily the one their row sits on.
                ## A linkcal has no exposure in its name at all.
                job_expids = np.asarray(row['PTAB_EXPIDS'], dtype=int)
                if obstype == 'linkcal' or len(job_expids) == 0:
                    lognames = glob.glob(os.path.join(logpath,
                                                      f'{file_head}-{night}-*.log'))
                else:
                    lognames = glob.glob(
                        logfiletemplate.format(pre=file_head, night=night,
                                               zexpid=str(job_expids[0]).zfill(8),
                                               specs='*', jobid='', ext='log'))
            else:
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


            newest_jobid, logname = 0, None

            for itlog in lognames:
                jobid = int(itlog.split('-')[-1].split('.')[0])
                if jobid > newest_jobid:
                    newest_jobid = jobid
                    logname = itlog
            if newest_jobid > 0:
                slurmname = logname.replace(f'-{newest_jobid}.log', '.slurm')
                slurm_hlink = _hyperlink(os.path.relpath(slurmname, webpage), 'Slurm')
                log_hlink = _hyperlink(os.path.relpath(logname, webpage), 'Log')

        rd = dict()
        rd["COLOR"] = row_color
        ## Jobs given a row without an exposure of their own have no expid to show
        rd["EXPID"] = str(expid) if expid >= 0 else '----'
        rd["TILEID"] = tileid_str
        rd["OBSTYPE"] = obstype
        rd["FA SURV"] = fasurv
        rd["FA PRGRM"] = faprog
        rd["LAST STEP"] = laststep
        rd["EXP TIME"] = str(exptime)
        rd["PROC CAMWORD"] = proccamword
        ## A step a job doesn't produce reads 0/0, the same as for every other
        ## job that skips a step. The status-only jobs expect nothing anywhere,
        ## except a linkcal's bias count once its override says what was linked.
        rd["BIAS"] = _str_frac(nfiles['bias'], nbias_expected)
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
