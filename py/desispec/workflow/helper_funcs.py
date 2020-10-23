import os
import glob
import json
from astropy.io import fits
from astropy.table import Table, join
import numpy as np
# import numpy as np

import argparse
import re
import time, datetime
import psutil
from os import listdir
from collections import OrderedDict
import subprocess
import sys
#from subprocess import check_output as subprocess_check_output
#from subprocess import STDOUT as subprocess_STDOUT
#from desispec.io.util import decode_camword

# from desispec.io.util import create_camword, decode_camword
# import desispec.io.util.create_camword as create_camword
# import desispec.io.util.decode_camword as decode_camword
#from create_processing_table import erow_to_irow, get_internal_production_table_column_defs


class get_logger:
    def __init__(self):
        pass

    def info(self, text):
        print("INFO: " + text)

    def debug(self, text):
        print("DEBUG: " + text)

    def error(self, text):
        print("ERROR: " + text)


########################
###  Defined Values ####
########################
# resubmission_states = ['UNSUBMITTED', 'BOOT_FAIL', 'DEADLINE', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED', 'TIMEOUT']
# termination_states = ['COMPLETED','CANCELLED','FAILED']
def get_resubmission_states():
    return ['UNSUBMITTED', 'BOOT_FAIL', 'DEADLINE', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED', 'TIMEOUT']


def get_termination_states():
    return ['COMPLETED', 'CANCELLED', 'FAILED']


###########################
### IO Helper Functions ###
###########################

opj = os.path.join


def listpath(*args):
    # return np.sort(os.listdir(opj(*args))).tolist()
    if os.path.exists(opj(*args)):
        srtlist = sorted(os.listdir(opj(*args)))
        if '.DS_Store' in srtlist:
            srtlist.remove('.DS_Store')
        return srtlist
    else:
        return []


def globpath(*args):
    # return np.sort(glob.glob(opj(*args))).tolist()
    if os.path.exists(opj(*args)):
        srtlist = sorted(glob.glob(opj(*args)))
        if '.DS_Store' in srtlist:
            srtlist.remove('.DS_Store')
        return srtlist
    else:
        return []


def get_json_dict(reqpath):
    req_dict = {}
    if os.path.isfile(reqpath):
        with open(reqpath, 'r') as req:
            req_dict = json.load(req)
    return req_dict


def ensure_scalar(val, joinsymb='|'):
    if val is None or type(val) in [str, np.str, np.str_, np.ma.core.MaskedConstant] or np.isscalar(val):
        return val
    else:
        val = np.atleast_1d(val).astype(str)
        return joinsymb.join(val)


def split_str(val, joinsymb='|'):
    if type(val) in [str, np.str, np.str_]:
        if val.isnumeric():
            if '.' in val:
                return float(val)
            else:
                return int(val)
        elif joinsymb not in val:
            return val
        else:
            split_list = np.array(val.split(joinsymb))
            if '.' in split_list[0] and split_list[0].isnumeric():
                return split_list.astype(float)
            elif split_list[0].isnumeric():
                return split_list.astype(int)
            else:
                split_list = np.array([val.strip('\t ') for val in split_list.astype(str)]).astype(str)
                return split_list
    else:
        return val


def write_table(origtable, tablename=None, table_type=None, joinsymb='|', overwrite=True, verbose=False):
    if tablename is None and table_type is None:
        print("Pathname or type of table is required to save the table")

    if tablename is None:
        tablename = translate_type_to_pathname(table_type)

    print("In write table",tablename,'\n',table_type)
    print(origtable[0:2])
    basename, ext = os.path.splitext(tablename)
    temp_name = f'{basename}.temp{ext}'
    print(ext,temp_name)
    table = origtable.copy()
    if ext in ['.csv', '.ecsv']:
        if verbose:
            print("Given table: ", table.info)
        # replace_cols = {}

        for nam in table.colnames:
            ndim = table[nam].ndim
            if ndim > 1 or type(table[nam][0]) in [list, np.ndarray, np.array]:
                if verbose:
                    print(f'{nam} is {ndim} dimensions, changing to string')
                col = [ensure_scalar(row, joinsymb=joinsymb) for row in table[nam]]
                # replace_cols[nam] = Table.Column(name=nam,data=col)
                if type(table[nam]) is Table.MaskedColumn:
                    col = Table.MaskedColumn(name=nam, data=col)
                else:
                    col = Table.Column(name=nam, data=col)
                table.replace_column(nam, col)

        # for nam, col in replace_cols.items():
        #     t.replace_column(nam,col)

        if np.any([c.ndim > 1 or type(table[nam][0]) in [list, np.ndarray, np.array] for c in
                   table.itercols()]) and verbose:
            print("A column was still more than one dimensional")
            print(table.info())

        table.write(temp_name, format=f'ascii{ext}', overwrite=overwrite)
    else:
        table.write(temp_name, overwrite=True)

    os.rename(temp_name, tablename)
    if verbose:
        print("Written table: ", table.info)


def translate_type_to_pathname(table_type):
    from desispec.workflow.create_exposure_tables import get_exposure_table_path, get_exposure_table_pathname, get_exposure_table_name
    from desispec.workflow.create_processing_table import get_processing_table_path, get_processing_table_pathname, get_processing_table_name
    if table_type.lower() in ['exp', 'exposure', 'etable']:
        tablename = get_exposure_table_pathname()
    elif table_type.lower() in ['proc', 'processing', 'int', 'itable', 'interal']:
        tablename = get_processing_table_pathname()
    elif table_type.lower() in ['unproc', 'unprocessed', 'unprocessing']:
        tablepath = get_processing_table_path()
        tablename = get_processing_table_name().replace("processing", 'unprocessed')
        tablename = opj(tablepath, tablename)
    return tablename


def load_table(tablename=None, table_type=None, joinsymb='|', verbose=False, process_mixins=True):
    if tablename is None:
        tablename = translate_type_to_pathname(table_type)

    basename, ext = os.path.splitext(tablename)
    if ext in ['.csv', '.ecsv']:
        table = Table.read(tablename, format=f'ascii{ext}')
        if verbose:
            print("Raw loaded table: ", table.info)
        # replace_cols = {}
        if process_mixins and len(table)>0:
            for nam in table.colnames:
                if type(table[nam]) is Table.MaskedColumn and np.sum(~table[nam].mask)==0:
                    continue
                elif type(table[nam]) is Table.MaskedColumn:
                    first = table[nam][np.bitwise_not(table[nam].mask)][0]
                else:
                    first = table[nam][0]
                if verbose:
                    print(first, type(first), type(first) in [str, np.str, np.str_])
                if type(first) in [str, np.str, np.str_] and joinsymb in first:
                    if verbose:
                        print(type(first) in [str, np.str, np.str_], joinsymb in first)

                    col = [split_str(row, joinsymb=joinsymb) for row in table[nam]]

                    # replace_cols[nam] = Table.Column(name=nam,data=col)
                    col = Table.Column(name=nam, data=col)
                    table.replace_column(nam, col)

        # for nam, col in replace_cols.items():
        #     t.replace_column(nam,col)

        # print(np.any([c.ndim > 1 for c in t.itercols()]))
    else:
        table = Table.read(tablename)

    if verbose:
        print("Expanded table: ", table.info)
    return table


# def backup_tables(tables, fullpathnames=None, table_types=None):
#     return write_tables(tables, fullpathnames, table_types)

def write_tables(tables, fullpathnames=None, table_types=None):
    if fullpathnames is None and table_types is None:
        print("Need to define either fullpathnames or the table types in write_tables")
    elif fullpathnames is None:
        for tabl, tabltyp in zip(tables, table_types):
            if len(tabl) > 0:
                write_table(tabl, table_type=tabltyp)
    else:
        for tabl, tablname in zip(tables, fullpathnames):
            if len(tabl) > 0:
                write_table(tabl, tablename=tablname)


def load_tables(fullpathnames=None, tabtypes=None):
    tabs = []
    if fullpathnames is None:
        for tabltyp in tabtypes:
            tabs.append(load_table(table_type=tabltyp))
    else:
        for tablname in fullpathnames:
            tabs.append(load_table(tablname))
    return tabs


########################
### Helper Functions ###
########################

def return_color_profile():
    color_profile = {}
    color_profile['NULL'] = {'font': '#34495e', 'background': '#ccd1d1'}  # gray
    color_profile['BAD'] = {'font': '#000000', 'background': '#d98880'}  # red
    color_profile['INCOMPLETE'] = {'font': '#000000', 'background': '#f39c12'}  # orange
    color_profile['GOOD'] = {'font': '#000000', 'background': '#7fb3d5'}  # blue
    color_profile['OVERFUL'] = {'font': '#000000', 'background': '#c39bd3'}  # purple
    return color_profile


def get_file_list(filename, doaction=True):
    if doaction and filename is not None and os.path.exists(filename):
        output = np.atleast_1d(np.loadtxt(filename, dtype=int)).tolist()
    else:
        output = []
    return output


def get_skipped_expids(expid_filename, skip_expids=True):
    return get_file_list(filename=expid_filename, doaction=skip_expids)


def what_night_is_it():
    """
    Return the current night
    """
    d = datetime.datetime.utcnow() - datetime.timedelta(7 / 24 + 0.5)
    tonight = int(d.strftime('%Y%m%d'))
    return tonight


def find_newexp(night, fileglob, known_exposures):
    """
    Check the path given for new exposures
    """
    datafiles = sorted(glob.glob(fileglob))
    newexp = list()
    for filepath in datafiles:
        expid = int(os.path.basename(os.path.dirname(filepath)))
        if (night, expid) not in known_exposures:
            newexp.append((night, expid))

    return set(newexp)


def check_running(proc_name='desi_dailyproc', suppress_outputs=False):
    """
    Check if the desi_dailyproc process is running
    """
    running = False
    mypid = os.getpid()
    for p in psutil.process_iter():
        if p.pid != mypid and proc_name in ' '.join(p.cmdline()):
            if not suppress_outputs:
                print('ERROR: {} already running as PID {}:'.format(proc_name, p.pid))
                print('  ' + ' '.join(p.cmdline()))
            running = True
            break
    return running


###################################################
########## Exposure Table Functions ###############
###################################################

def get_survey_definitions():
    ## Create a rudimentary way of assigning "SURVEY keywords based on what date range a night falls into"
    survey_def = {0: (20200201, 20200315), 1: (
        20201201, 20210401)}  # 0 is CMX, 1 is SV1, 2 is SV2, ..., 99 is any testing not in these timeframes
    return survey_def


def get_surveynum(night, survey_definitions=None):
    if survey_definitions is None:
        survey_definitions = get_survey_definitions()
    for survey, (low, high) in survey_definitions.items():
        if night >= low and night <= high:
            return survey
    return 99


def give_relevant_details(verbose_output, non_verbose_output=None, verbosely=False):
    if verbosely:
        print(verbose_output)
    elif non_verbose_output is not None:
        print(non_verbose_output)
    else:
        pass


def get_night_banner(night):
    return '\n#############################################' + \
           f'\n################ {night} ###################' + \
           '\n#############################################'


# def define_variable_from_environment(env_name, error_message):
#     if env_name in os.environ:
#         return os.environ[env_name]
#     else:
#         print(error_message)
#         exit(1)
def define_variable_from_environment(env_name, var_descr):
    if env_name in os.environ:
        return os.environ[env_name]
    else:
        print(f'{var_descr} needs to be given explicitly or set using environment variable {env_name}')
        exit(1)


##########################
###  All Helper Funcs ####
##########################


#######################################
####### Table Row Functions ###########
#######################################
def get_tile_rows(table, tilename, subselect=None):
    outtab = table[table['LABEL'] == tilename]
    if subselect is not None:
        outtab = outtab[outtab['JOBTYPE'] == subselect]
    return outtab


def get_arbtype_rows(table, arbtype):
    return table[table['TYPE'] == arbtype]


def get_arc_rows(table):
    return get_arbtype_rows(table, 'arc')


def get_flat_rows(table):
    return get_arbtype_rows(table, 'flat')


#######################################
########## Time Functions #############
#######################################
def get_nightly_start_time():
    return 14  # 2PM local Tucson time


def get_nightly_end_time():
    month = time.localtime().tm_mon
    if np.abs(month - 6) < 2:
        end_night = 10
    else:
        end_night = 8
    return end_night  # local Tucson time the following morning


def ensure_tucson_time():
    if 'TZ' not in os.environ.keys() or os.environ['TZ'] != 'US/Arizona':
        os.environ['TZ'] = 'US/Arizona'
    time.tzset()


def nersc_format_datetime(timetup=time.localtime()):
    # YYYY-MM-DD[THH:MM[:SS]]
    return time.strftime('%Y-%m-%dT%H:%M:%S', timetup)


def nersc_start_time(obsnight=what_night_is_it(), starthour=get_nightly_start_time()):
    starthour = int(starthour)
    timetup = time.strptime(f'{obsnight}{starthour:02d}', '%Y%m%d%H')
    return nersc_format_datetime(timetup)


def nersc_end_time(obsnight=what_night_is_it(), endhour=get_nightly_end_time()):
    endhour = int(endhour)
    one_day_in_seconds = 24 * 60 * 60

    yester_timetup = time.strptime(f'{obsnight}{endhour:02d}', '%Y%m%d%H')
    yester_sec = time.mktime(yester_timetup)

    today_sec = yester_sec + one_day_in_seconds
    today_timetup = time.localtime(today_sec)
    return nersc_format_datetime(today_timetup)


def during_operating_hours(dry_run=False, start_hour=get_nightly_start_time(), end_hour=get_nightly_end_time()):
    ensure_tucson_time()
    hour = time.localtime().tm_hour
    return dry_run or (hour < end_hour) or (hour > start_hour)


#################################################
############ Script Functions ###################
#################################################
def get_batch_dir(night):
    reduxdir = os.path.join(os.environ['DESI_SPECTRO_REDUX'], os.environ['SPECPROD'])
    batchdir = os.path.join(reduxdir, 'run', 'scripts', 'night', str(night))
    os.makedirs(batchdir, exist_ok=True)
    return batchdir


def batch_script_name(irow):
    batchdir = get_batch_dir(str(irow['NIGHT']))
    camword = irow['CAMWORD']
    if type(irow['EXPID']) in [list, np.array]:
        expid_str = '-'.join(['{:08d}'.format(int(exp)) for exp in irow['EXPID']])
    else:
        expid_str = '{:08d}'.format(int(irow['EXPID']))
    jobname = '{}-{}-{}-{}'.format(str(irow['OBSTYPE']).lower(), str(irow['NIGHT']), expid_str, camword)
    scriptfile = os.path.join(batchdir, jobname + '.slurm')
    return scriptfile


def joint_batch_script_name(irow):
    batchdir = get_batch_dir(str(irow['NIGHT']))
    camword = irow['CAMWORD']
    if type(irow['EXPID']) in [list, np.array]:
        expid_str = '-'.join(['{:08d}'.format(int(exp)) for exp in irow['EXPID']])
    else:
        expid_str = '{:08d}'.format(int(irow['EXPID']))
    jobname = '{}-{}-{}-{}'.format(str(irow['OBSTYPE']).lower(), str(irow['NIGHT']), expid_str, camword)
    scriptfile = os.path.join(batchdir, jobname + '.slurm')
    return scriptfile


def create_and_submit_joint(irow, queue=None, dry_run=False):
    irow = create_joint_batch_script(irow, queue=queue, dry_run=dry_run)
    irow = submit_batch_script(irow, dry_run=dry_run)
    return irow


def create_and_submit_exposure(irow, queue=None, dry_run=False):
    if irow is None:
        import pdb
        pdb.set_trace()
    irow = create_batch_script(irow, queue=queue, dry_run=dry_run)
    irow = submit_batch_script(irow, dry_run=dry_run)
    return irow


def create_batch_script(irow, queue=None, dry_run=False):
    if irow is None:
        import pdb
        pdb.set_trace()
    cmd = 'desi_proc'
    cmd += ' --batch'
    cmd += ' --nosubmit'
    cmd += ' --traceshift'
    if queue is not None:
        cmd += f' -q {queue}'
    if irow['OBSTYPE'].lower() == 'science':
        if irow['JOBDESC'] == 'prestd':
            cmd += ' --nostdstarfit --nofluxcalib'
        elif irow['JOBDESC'] == 'poststd':
            cmd += ' --noprestdstarfit --nostdstarfit'
    specs = ','.join(str(irow['CAMWORD'])[1:])
    cmd += ' --cameras={} -n {} -e {}'.format(specs, irow['NIGHT'], irow['EXPID'])

    print(cmd)
    scriptpathname = batch_script_name(irow)
    if dry_run:
        print("\tOutput file would have been: {}".format(scriptpathname))
        print("\tCommand to be run: {}".format(cmd.split()))
    else:
        print("\tRunning: {}".format(cmd.split()))
        subprocess.call(cmd.split())
        print("\tOutfile is: ".format(scriptpathname))
    sys.stdout.flush()

    irow['SCRIPTNAME'] = os.path.basename(scriptpathname)
    return irow


def create_joint_batch_script(irow, descriptor=None, queue=None, dry_run=False):
    cmd = 'desi_proc_joint_fit'
    cmd += ' --batch'
    cmd += ' --nosubmit'
    cmd += ' --traceshift'
    if queue is not None:
        cmd += f' -q {queue}'
    if descriptor is None:
        descriptor = irow['OBSTYPE'].lower()

    night = irow['NIGHT']
    specs = ','.join(str(irow['CAMWORD'])[1:])
    expids = irow['EXPID']
    expid_str = ','.join([str(eid) for eid in expids])

    cmd += f' --obstype {descriptor}'
    cmd += ' --cameras={} -n {} -e {}'.format(specs, night, expid_str)

    print(cmd)
    scriptpathname = joint_batch_script_name(irow)
    if dry_run:
        print("\tOutput file would have been: {}".format(scriptpathname))
        print("\tCommand to be run: {}".format(cmd.split()))
    else:
        print("\tRunning: {}".format(cmd.split()))
        subprocess.call(cmd.split())
        print("\tOutfile is: ".format(scriptpathname))
    sys.stdout.flush()

    irow['SCRIPTNAME'] = os.path.basename(scriptpathname)
    return irow


def submit_batch_script(submission, dry_run=False):
    jobname = batch_script_name(submission)
    dependencies = submission['LATEST_DEP_QID']
    print(dependencies,type(dependencies))
    dep_list, dep_str = '', ''
    if dependencies is not None:
        dep_str = '--dependency=afterok:'
        if np.isscalar(dependencies):
            dep_list = str(dependencies).strip(' \t')
            if dep_list == '':
                dep_str = ''
            else:
                dep_str += dep_list
        else:
            if len(dependencies)>0:
                dep_list = ','.join(np.array(dependencies).astype(str))
                dep_str += dep_list
            else:
                dep_str = ''

    ## True function will actually submit to SLURM
    if dry_run:
        current_qid = int(time.time() - 1.6e9)
    else:
        # script = f'{jobname}.slurm'
        # script_path = opj(batchdir, script)
        batchdir = get_batch_dir(submission['NIGHT'])
        script_path = opj(batchdir, jobname)
        if dep_str == '':
            current_qid = subprocess.check_output(['sbatch', '--parsable', f'{script_path}'],
                                                  stderr=subprocess.STDOUT, text=True)
        else:
            current_qid = subprocess.check_output(['sbatch', '--parsable',f'{dep_str}',f'{script_path}'],
                                                  stderr=subprocess.STDOUT, text=True)
        current_qid = int(current_qid.strip(' \t\n'))

    print(f'Submitted {jobname}.slurm\t\t with dependencies {dep_list}. Returned qid: {current_qid}')

    submission['LATEST_QID'] = current_qid
    submission['STATUS'] = 'SU'

    return submission


def night_to_month(night):
    return str(night)[:-2]


def night_to_starting_iid(night):
    night = int(night)
    internal_id = (night - 20180000) * 1000
    return internal_id


###############################
#####   Mock Functions   ######
###############################
def refresh_queue_info_table(start_time=None, end_time=None, user=None, \
                             columns='jobid,state,submit,eligible,start,end,jobname', dry_run=False):
    # global queue_info_table
    if dry_run:
        cmd_as_list = ['cat', 'sacct_example.csv']
    else:
        # -A desi   # account information is redundant with desi user
        # jobs = '-j <list of jobs>'
        # --name=jobname_list
        # -s state_list, --state=state_list   e.g. CA or ca or CANCELLED for cancelled jobs
        #     will only show currently running jobs in queue unless times are explicitly given
        #     BF BOOT_FAIL   Job terminated due to launch failure
        #     CA CANCELLED Job was explicitly cancelled by the user or system administrator. The job may or may not have been initiated.
        #     CD COMPLETED Job has terminated all processes on all nodes with an exit code of zero.
        #     DL DEADLINE Job terminated on deadline.
        #     F FAILED Job terminated with non-zero exit code or other failure condition.
        #     NF NODE_FAIL Job terminated due to failure of one or more allocated nodes.
        #     OOM OUT_OF_MEMORY Job experienced out of memory error.
        #     PD PENDING Job is awaiting resource allocation.
        #     PR PREEMPTED Job terminated due to preemption.
        #     R RUNNING Job currently has an allocation.
        #     RQ REQUEUED Job was requeued.
        #     RS RESIZING Job is about to change size.
        #     RV REVOKED Sibling was removed from cluster due to other cluster starting the job.
        #     S SUSPENDED Job has an allocation, but execution has been suspended and CPUs have been released for other jobs.
        #     TO TIMEOUT Job terminated upon reaching its time limit.
        #
        # other format columns: jobid,state,submit,eligible,start,end,elapsed,suspended,exitcode,derivedexitcode,reason,priority,jobname
        if user is None:
            user = os.environ['USER']
        if start_time is None:
            start_time = '2020-04-26T00:00'
        if end_time is None:
            end_time = '2020-05-01T00:00'
        cmd_as_list = ['sacct', '-X', '--parsable2', '--delimiter=,', \
                       '-S', start_time, \
                       '-E', end_time, \
                       '-u', user, \
                       f'--format={columns}']

    queue_info_table = Table.read(subprocess.check_output(cmd_as_list, stderr=subprocess.STDOUT, text=True),
                                  format='ascii.csv')
    for col in queue_info_table.colnames:
        queue_info_table.rename_column(col, col.upper())
    return queue_info_table


def get_prod_jobs_in_queue(queue_info_table=None, dry_run=False):
    if queue_info_table is None:
        queue_info_table = refresh_queue_info_table(dry_run=dry_run)

    return np.sum(((queue_info_table['State'] == 'PENDING') | (queue_info_table['State'] == 'RUNNING')))


# def create_batchjob_file(entry, typ):
#     ## when implemented this would actually create the file and save it to disk
#     jobname = entry['JOBNAME']
#     month = night_to_month(entry['NIGHT'])
#     if not os.path.exists(opj(prod_path, month)):
#         os.makedirs(opj(prod_path, month))
#     path_to_file = opj(prod_path, month)
#     full_path = opj(path_to_file, f'{jobname}.slurm')


def update_from_queue(table, qtable=None, dry_run=False, start_time=None, end_time=None):
    if qtable is None:
        qtable = refresh_queue_info_table(start_time=start_time, end_time=end_time, dry_run=dry_run)

    for row in qtable:
        match = (int(row['JOBID']) == table['LATEST_QID'])
        # 'jobid,state,submit,eligible,start,end,jobname'
        if np.any(match):
            ind = np.where(match)[0][0]
            table['STATUS'][ind] = row['STATE']

    return table


def recursive_submit_failed(rown, int_table, submits, id_to_row_map, itab_name=None,
                            resubmission_states=None, dry_run=False):
    if resubmission_states is None:
        resubmission_states = get_resubmission_states()
    ideps = split_str(int_table['INT_DEP_IDS'][rown], joinsymb='|')
    if ideps is None:
        int_table['LATEST_DEP_QID'][rown] = None
    else:
        qdeps = []
        for idep in np.sort(np.atleast_1d(ideps)):
            if int_table['STATUS'][id_to_row_map[idep]] in resubmission_states:
                int_table, submits = recursive_submit_failed(id_to_row_map[idep], \
                                                             int_table, submits, id_to_row_map)
            qdeps.append(int_table['LATEST_QID'][id_to_row_map[idep]])

        if len(qdeps) == 1:
            int_table['LATEST_DEP_QID'][rown] = qdeps[0]
        elif len(qdeps) > 1:
            int_table['LATEST_DEP_QID'][rown] = ensure_scalar(qdeps, joinsymb='|')
        else:
            print("Error: number of qdeps should be 1 or more")
            print(f'Rown {rown}, ideps {ideps}')

    int_table[rown] = submit_batch_script(int_table[rown], dry_run=dry_run)
    int_table['ALL_QIDS'][rown] = int_table['ALL_QIDS'][rown] + '|' + str(int_table['LATEST_DEP_QID'][rown])
    int_table['STATUS'][rown] = 'SU'
    submits += 1

    if dry_run:
        pass
    else:
        time.sleep(2)
        if submits % 10 == 0:
            if itab_name is None:
                write_table(int_table, table_type='processing', overwrite=True)
            else:
                write_table(int_table, tablename=itab_name, overwrite=True)
            time.sleep(60)
        if submits % 100 == 0:
            time.sleep(540)
            int_table = update_from_queue(int_table)
            if itab_name is None:
                write_table(int_table, table_type='processing', overwrite=True)
            else:
                write_table(int_table, tablename=itab_name, overwrite=True)

    return int_table, submits


# def map_rows_to_intid(internal_prod_table):
#     id_to_row_map = {row['INTERNAL_ID']: rown for rown, row in enumerate(internal_prod_table)}
#     return id_to_row_map


def update_and_recurvsively_submit(internal_prod_table, submits=0, resub_states=None, start_time=None, end_time=None,
                                   itab_name=None, dry_run=False):
    if resub_states is None:
        resub_states = get_resubmission_states()
    internal_prod_table = update_from_queue(internal_prod_table, start_time=start_time, end_time=end_time)
    id_to_row_map = {row['INTID']: rown for rown, row in enumerate(internal_prod_table)}
    for rown in range(len(internal_prod_table)):
        if internal_prod_table['STATUS'][rown] in resub_states:
            internal_prod_table, submits = recursive_submit_failed(rown, internal_prod_table, \
                                                                   submits, id_to_row_map, itab_name,
                                                                   resub_states, dry_run)
    return internal_prod_table, submits


def continue_looping(statuses, termination_states=None):
    if termination_states is None:
        termination_states = get_termination_states()
    return np.any([status not in termination_states for status in statuses])


def joint_fit(etable, itable, irows, internal_id, descriptor=None):
    if descriptor is None:
        return etable, itable, None
    if descriptor == 'science':
        descriptor = 'stdstarfit'
    elif descriptor == 'arc':
        descriptor = 'psfnight'
    elif descriptor == 'flat':
        descriptor = 'nightlyflat'

    if descriptor not in ['stdstarfit', 'psfnight', 'nightlyflat']:
        return etable, itable, None

    joint_irow = make_joint_irow(irows, descriptor=descriptor, initid=internal_id)
    joint_irow = create_and_submit_joint(joint_irow)
    itable.add_row(joint_irow)

    if descriptor == 'stdstarfit':
        for row in irows:
            row['JOBDESC'] = 'poststd'
            row['INTID'] = internal_id
            internal_id += 1
            row = assign_dependency(row, joint_irow)
            row = create_and_submit_exposure(row)
            itable.add_row(row)
    else:
        etable, itable = set_calibrator_flag(irows, etable, itable)

    return etable, itable, joint_irow


def science_joint_fit(etable, itable, sciences, internal_id):
    return joint_fit(etable=etable, itable=itable, irows=sciences, internal_id=internal_id, descriptor='stdstarfit')


def flat_joint_fit(etable, itable, flats, internal_id):
    return joint_fit(etable=etable, itable=itable, irows=flats, internal_id=internal_id, descriptor='nightlyflat')


def arc_joint_fit(etable, itable, arcs, internal_id):
    return joint_fit(etable=etable, itable=itable, irows=arcs, internal_id=internal_id, descriptor='psfnight')


# def science_joint_fit(etable, itable, sciences, internal_id):
#     tilejob = make_joint_irow(sciences, descriptor='stdstarfit', initid=internal_id)
#     tilejob = create_and_submit_joint(tilejob)
#     itable.add_row(tilejob)
#     for row in sciences:
#         row['JOBDESC'] = 'poststd'
#         row['INTID'] = internal_id
#         internal_id += 1
#         row = assign_dependency(row,tilejob)
#         irow = create_and_submit_exposure(row)
#         itable.add_row(irow)
#
#     return etable, itable, tilejob
#
# def flat_joint_fit(etable, itable, flats, internal_id):
#     flatjob = make_joint_irow(flats, descriptor='nightlyflat', initid=internal_id)
#     flatjob = create_and_submit_joint(flatjob)
#     itable.add_row(flatjob)
#     itable.add_row(flatjob)
#     etable, itable = set_calibrator_flag(flats, etable, itable)
#
#     return etable, itable, flatjob
#
# def arc_joint_fit(etable, itable, arcs, internal_id):
#     arcjob = make_joint_irow(arcs, descriptor='psfnight', initid=internal_id)
#     arcjob = create_and_submit_joint(arcjob)
#     itable.add_row(arcjob)
#     etable,itable = set_calibrator_flag(arcs, etable, itable)
#
#     return etable, itable, arcjob

def make_joint_irow(irows, descriptor, initid):
    irow = irows[0].copy()
    irow['INTID'] = initid
    irow['JOBDESC'] = descriptor

    if type(irows) in [list, np.array]:
        ids, qids, expids = [], [], []
        for currow in irows:
            ids.append(currow['INTID'])
            qids.append(currow['LATEST_QID'])
            expids.append(currow['EXPID'])
        irow['DEP_ID'] = ids
        irow['LATEST_DEP_QID'] = qids
        irow['EXPID'] = expids
    else:
        irow['DEP_ID'] = irows['INTID']
        irow['LATEST_DEP_QID'] = irows['LATEST_QID']
        irow['EXPID'] = irows['EXPID']

    return irow


def set_calibrator_flag(matchrows, etable, itable):
    for irow in matchrows:
        etable['CALIBRATOR'][etable['EXPID'] == irow['EXPID'][0]] = True
        itable['CALIBRATOR'][itable['INTID'] == irow['INTID']] = True
    return etable, itable


def define_and_assign_dependency(irow, arcjob, flatjob):
    irow['JOBDESC'] = irow['OBSTYPE']
    if irow['OBSTYPE'] in ['science', 'twiflat']:
        dependency = flatjob
        irow['JOBDESC'] = 'prestd'
    elif irow['OBSTYPE'] == 'flat':
        dependency = arcjob
    else:
        dependency = None

    irow = assign_dependency(irow, dependency)

    return irow


def assign_dependency(irow, dependency):
    if dependency is not None:
        if type(dependency) in [list, np.array]:
            ids, qids = [], []
            for curdep in dependency:
                ids.append(curdep['INTID'])
                qids.append(curdep['LATEST_QID'])
            irow['DEP_ID'] = ids
            irow['LATEST_DEP_QID'] = qids
        else:
            irow['DEP_ID'] = dependency['INTID']
            irow['LATEST_DEP_QID'] = dependency['LATEST_QID']
    return irow


def get_type_and_tile(erow):
    return str(erow['OBSTYPE']).lower(), erow['TILEID']


def verify_variable_with_environment(var, var_name, env_name, output_mechanism=print):
    if var is not None:
        if env_name in os.environ and var != os.environ[env_name]:
            old = os.environ['SPECPROD']
            output_mechanism(f"Warning, overwriting what the environment variable is for {env_name}")
            output_mechanism(f"\tOld {env_name}: {old}")
            output_mechanism(f"\tNew {env_name}: {var}")

        os.environ[env_name] = var
    else:
        if env_name in os.environ:
            var = os.environ[env_name]
        else:
            output_mechanism(f"Must define either {var_name} or the environment variable {env_name}")

    return var
