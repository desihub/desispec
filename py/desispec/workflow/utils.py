

import os
import numpy as np
import glob
import json
import psutil


pathjoin = os.path.join

def listpath(*args):
    # return np.sort(os.listdir(pathjoin(*args))).tolist()
    if os.path.exists(pathjoin(*args)):
        srtlist = sorted(os.listdir(pathjoin(*args)))
        if '.DS_Store' in srtlist:
            srtlist.remove('.DS_Store')
        return srtlist
    else:
        return []


def globpath(*args):
    # return np.sort(glob.glob(pathjoin(*args))).tolist()
    if os.path.exists(pathjoin(*args)):
        srtlist = sorted(glob.glob(pathjoin(*args)))
        if '.DS_Store' in srtlist:
            srtlist.remove('.DS_Store')
        return srtlist
    else:
        return []


def get_file_list(filename, doaction=True):
    if doaction and filename is not None and os.path.exists(filename):
        output = np.atleast_1d(np.loadtxt(filename, dtype=int)).tolist()
    else:
        output = []
    return output


########################
### Helper Functions ###
########################



def get_skipped_expids(expid_filename, skip_expids=True):
    return get_file_list(filename=expid_filename, doaction=skip_expids)

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

def get_json_dict(reqpath):
    req_dict = {}
    if os.path.isfile(reqpath):
        with open(reqpath, 'r') as req:
            req_dict = json.load(req)
    return req_dict

def give_relevant_details(verbose_output, non_verbose_output=None, verbosely=False):
    if verbosely:
        print(verbose_output)
    elif non_verbose_output is not None:
        print(non_verbose_output)
    else:
        pass

def define_variable_from_environment(env_name, var_descr):
    if env_name in os.environ:
        return os.environ[env_name]
    else:
        print(f'{var_descr} needs to be given explicitly or set using environment variable {env_name}')
        exit(1)

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