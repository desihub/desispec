

import os
import time
import numpy as np
import glob
import json
import psutil

from desiutil.log import get_logger
## Give a shortcut name to os.path.join
pathjoin = os.path.join

def listpath(*args):
    """
    Helper function that takes the same arguments as os.path.join (i.e. a comma separated set of directory names, paths,
    and file names that can be joined to created a single unified file path), and returns the list of files in that
    location.

    Args:
        args, undetermined number of input args that are each a string. The values to be given to os.path.join (i.e. pathjoin)
                                                                        for which you want a list of files in that location.
    Returns:
        list. Sorted list of files in the location defined by the input args. Ignores Mac file .DS_STORE. If location
              doesn't exist, it returns an empty list.
    """
    # return np.sort(os.listdir(pathjoin(*args))).tolist()
    if os.path.exists(pathjoin(*args)):
        srtlist = sorted(os.listdir(pathjoin(*args)))
        if '.DS_Store' in srtlist:
            srtlist.remove('.DS_Store')
        return srtlist
    else:
        return []


def globpath(*args):
    """
    Helper function that takes the same arguments as os.path.join (i.e. a comma separated set of directory names, paths,
    and file names that can be joined to created a single unified file path), and returns the glob'ed list of files in that
    location that matched the specified parameters.

    Args:
        args, undetermined number of input args that are each a string. The values to be given to os.path.join (i.e. pathjoin)
                                                                        for which you want a list of files in that location.
                                                                        Any of the strings can be wildcards used by glob.glob,
                                                                        so long as they don't confuse os.path.join().
    Returns:
        list. Sorted list of files in the location defined by the input args that are consistent with all wildcards (if
              any). Ignores Mac file .DS_STORE. If location doesn't exist, it returns an empty list.
    """
    # return np.sort(glob.glob(pathjoin(*args))).tolist()
    if os.path.exists(pathjoin(*args)):
        srtlist = sorted(glob.glob(pathjoin(*args)))
        if '.DS_Store' in srtlist:
            srtlist.remove('.DS_Store')
        return srtlist
    else:
        return []

# def find_newexp(night, fileglob, known_exposures):
#     """
#     Check the path given for new DESI exposures. Assumes data is in DESI raw data directory format: /{base_dir}/{NIGHT}/{EXPID}/{filename}.{ext}
#     """
#     datafiles = sorted(glob.glob(fileglob))
#     newexp = list()
#     for filepath in datafiles:
#         expid = int(os.path.basename(os.path.dirname(filepath)))
#         if (night, expid) not in known_exposures:
#             newexp.append((night, expid))
#
#     return set(newexp)

def get_json_dict(reqpath):
    """
    Return a dictionary representation of the json file at the specified location. If it doesn't exist, return an
    empty dictionary.

    Args:
        reqpath, str. The full pathname including file name of the json file. A relative path can be given, so long
                      as it is accurate based on the current working directory.

    Retrun:
        req_dict, dict. A dictionary with keys and values defined by the json file.
    """
    req_dict = {}
    if os.path.isfile(reqpath):
        with open(reqpath, 'r') as req:
            req_dict = json.load(req)
    return req_dict


def define_variable_from_environment(env_name, var_descr):
    """
    Returns the environment variable if it exists, otherwise raises an error telling the user that a variable must
    be specified either directly or by defining the environment variable. It exits with exit code 1 if no environment
    variable exists.

    Args:
        env_name, str. The name of the environment variable.
        var_descr, str. A description of the variable you are trying to define with environment variable env_name.

    Returns:
        str or Nothing. If the environment variable exists, it returns the value of that environment variable.
                        Otherwise it raises an exit code with status 1.
    """
    log = get_logger()
    if env_name in os.environ:
        return os.environ[env_name]
    else:
        log.error(f'{var_descr} needs to be given explicitly or set using environment variable {env_name}')
        exit(1)

def verify_variable_with_environment(var, var_name, env_name):
    """
    Helper function that assigns a variable based on the inputs and gives relevant outputs to understand what is being
    done. If the variable is defined, it will make sure that the environment variable (used by some lower-level code)
    is consistent before returning the user specified value. If it is not specified, then the environment variable is used.
    If the environment variable is also undefined, then it gives a useful output and then exits.

    Args:
        var, any type. Can be any python data type that can be assigned from an environment variable.
        var_name, str. The name of the variable (used exclusively for outputting useful messages).
        env_name, str. The name of the environment variable that would hold the value relevant to the var variable.

    Returns:
        var, any type. Either the input var if that is not NoneType. Otherwise the value from the environment variable.
        If neither is defined it exits with status 1 rather than returning anything.
    """
    log = get_logger()
    if var is not None:
        if env_name in os.environ and var != os.environ[env_name]:
            old = os.environ[env_name]
            log.warning(f"Warning, overwriting what the environment variable is for {env_name}")
            log.info(f"\tOld {env_name}: {old}")
            log.info(f"\tNew {env_name}: {var}")

        os.environ[env_name] = var
    else:
        var = define_variable_from_environment(env_name, var_name)

    return var


def check_running(proc_name='desi_daily_proc_manager', suppress_outputs=False):
    """
    Check if the given process name is running. Default is desi_daily_proc_manager.

    Args:
        proc_name, str. The name of the process as it would appear in the os's process tables.
        suppress_outputs, bool. True if you don't want to anything to be printed.

    Returns:
         running, bool. True if the process name was found in the list of processes and has a pid different that
                        the current process (signifying a second instance of that program).
    """
    log = get_logger()
    if not suppress_outputs:
        log.info(f"Looking for other python processes with \'{proc_name}\' in command")

    mypid = int(os.getpid())
    ## getlogin() doesn't work with desi account, so try environment variable first
    if 'USER' in os.environ:
        user = os.getenv('USER')
    else:
        user = os.getlogin()
    if not suppress_outputs:
        log.info(f"pid of this process: {mypid}, under username={user}")

    running = False
    ## Loop through all processes on the machine
    for p in psutil.process_iter():
        ## In testing there was an exception thrown when a process ended during the loop. Protect with try-except
        try:
            cmdline = ' '.join(p.cmdline())
            status = str(p.status()).lower()
            ppid = int(p.pid)
            puser = p.username()
            pname = p.name()
        except:
            if not suppress_outputs:
                log.error("Couldn't get process information from process_iter object.")
            continue
        # ( status in ['running','sleeping'] )
        ## if not this PID, but same user and command, then it the given command is already running
        if (ppid != mypid) and (proc_name in cmdline) and \
           (puser == user) and (pname in ['python', proc_name]):
            if not suppress_outputs:
                log.warning(f"{proc_name} already running as PID {ppid}:")
                log.info(f"\t {cmdline}")
                log.info(f"\t status={status}, user={puser}, name={pname}")
            running = True
            break

    return running

def get_printable_banner(input_str=None):
    """
    Returns a string that when printed shows a banner with the given string in the center.

    Args:
        input_str: str. Any string of moderate length roughly less than 30 that you want placed inside a
                        pound/hashtag (#) banner.

    Returns:
        banner: str. A banner comprised of ascii pound symbols (#) and the given string in the center.
    """
    if type(input_str) is not str:
        input_str = str(input_str)
    ninput = len(input_str)
    nhash = max(4+ninput, 36)
    nshort1 = (nhash - ninput - 2) // 2
    nshort2 = nhash - ninput - 2 - nshort1
    banner = '\n' + '#'*nhash + \
             '\n' + '#'*nshort1 + f' {input_str} ' + '#'*nshort2 + \
             '\n' + '#'*nhash
    return banner

def sleep_and_report(sleep_duration, message_suffix="", logfunc=print, dry_run=False):
    """
    Sleeps for the given number of seconds, printing a message before and afterward

    Args:
        sleep_duration: int or float. Number of seconds for the process to sleep.
        message_suffix: str. The message to be printed after "Sleeping XXs (suffix)" prior to sleeping.
        logfunc: func. Default is print. The function used to serve the message to the user.
        dry_run: bool. Default is False. Whether to perform the action (dry_run=False) or pretend (True).
    """
    message = f"Sleeping {sleep_duration}s {message_suffix}"
    if dry_run:
        logfunc(f"Dry run, sleeping 1s instead of: '{message}'")
        time.sleep(1)
    else:
        logfunc(message)
        time.sleep(sleep_duration)
        logfunc(f"Resuming...")