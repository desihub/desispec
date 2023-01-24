"""
desispec.workflow.timing
========================

Please add module-level documentation.
"""
import os
import numpy as np

import time, datetime


#######################################
########## Time Functions #############
#######################################
def what_night_is_it():
    """
    Return the current night
    """
    d = datetime.datetime.utcnow() - datetime.timedelta(7 / 24 + 0.5)
    tonight = int(d.strftime('%Y%m%d'))
    return tonight

def get_nightly_start_time():
    """
    Defines the time of day that the desi_daily_proc_manager should start (in Tucson local time).
    Before this time, the manager being woken by a cron job will exit immediately. Selected to give plenty of time
    between end of night and start of the next, but early enough to catch and begin running afternoon calibrations.

    Returns:
        14: int. The number of hours after midnight that signifies the start of a new night of observing.
    """
    return 14  # 2PM local Tucson time


def get_nightly_end_time():
    """
    Defines when a night ends for desi_daily_proc_manager in local Tucson time. Once this time is exceeded,
    the manager will enter into queue cleanup mode and then exit when the jobs have finished. End of night is altered slightly
    for summer vs. winter.

    Returns:
        end_night: int. The number of hours after midnight that signifies the end of a night of observing.
    """
    month = time.localtime().tm_mon
    if np.abs(month - 6) > 2:
        end_night = 8
    else:
        end_night = 7
    return end_night  # local Tucson time the following morning


def ensure_tucson_time():
    """
    Define the start and end of a 'night' based on times at the mountain. So ensure that the times are with respect to Arizona.
    """
    if 'TZ' not in os.environ.keys() or os.environ['TZ'] != 'US/Arizona':
        os.environ['TZ'] = 'US/Arizona'
    time.tzset()


def nersc_format_datetime(timetup=None):
    """
    Given a time tuple from the time module, this will return a string in the proper format to be properly interpreted
    by the NERSC Slurm queue scheduler.

    Args:
        timetup: tuple. A time.time() tuple object representing the time you want to trasnform into a Slurm readable string.

    Returns:
        str. String of the form YYYY-mm-ddTHH:MM:SS.
    """
    if timetup is None:
        timetup = time.localtime()
    # YYYY-MM-DD[THH:MM[:SS]]
    return time.strftime('%Y-%m-%dT%H:%M:%S', timetup)


def nersc_start_time(night=None, starthour=None):
    """
    Transforms a night and time into a YYYY-MM-DD[THH:MM[:SS]] time string Slurm can interpret

    Args:
        night: str or int. In the form YYYMMDD, the night the jobs are being run.
        starthour: str or int. The number of hours (between 0 and 24) after midnight where you began submitting jobs to the queue.

    Returns:
        str. String of the form YYYY-mm-ddTHH:MM:SS. Based on the given night and starthour
    """
    if night is None:
        night = what_night_is_it()
    if starthour is None:
        starthour = get_nightly_start_time()
    starthour = int(starthour)
    timetup = time.strptime(f'{night}{starthour:02d}', '%Y%m%d%H')
    return nersc_format_datetime(timetup)


def nersc_end_time(night=None, endhour=None):
    """
    Transforms a night and time into a YYYY-MM-DD[THH:MM[:SS]] time string Slurm can interpret. Correctly accounts for the fact
    that the night is defined starting at Noon on a given day.

    Args:
        night: str or int. In the form YYYMMDD, the night the jobs are being run.
        endhour: str or int. The number (between 0 and 24) of hours after midnight where you stop submitting jobs to the queue.

    Returns:
        str. String of the form YYYY-mm-ddTHH:MM:SS. Based on the given night and endhour
    """
    if night is None:
        night = what_night_is_it()
    if endhour is None:
        endhour = get_nightly_end_time()

    endhour = int(endhour)
    yester_timetup = time.strptime(f'{night}{endhour:02d}', '%Y%m%d%H')
    yester_sec = time.mktime(yester_timetup)

    ## If ending in the PM, then the defined night is the same as the day it took place
    ## If in the AM then it corresponds to the following day, which requires adding 24 hours to the time.
    if endhour > 12:
        today_sec = yester_sec
    else:
        one_day_in_seconds = 24 * 60 * 60
        today_sec = yester_sec + one_day_in_seconds

    today_timetup = time.localtime(today_sec)
    return nersc_format_datetime(today_timetup)


def during_operating_hours(dry_run=False, starthour=None, endhour=None):
    """
    Determines if the desi_daily_proc_manager should be running or not based on the time of day. Can be overwridden
    with dry_run for testing purposes.

    Args:
        dry_run: bool. If true, this is a simulation so return True so that the simulation can proceed.
        starthour: str or int. The number of hours (between 0 and 24) after midnight.
        endhour: str or int. The number (between 0 and 24) of hours after midnight. Assumes an endhour smaller than starthour
                implies the following day.

    Returns:
        bool. True if dry_run is true OR if the current time is between the starthour and endhour.
    """
    if starthour is None:
        starthour = get_nightly_start_time()
    if endhour is None:
        endhour = get_nightly_end_time()
    ensure_tucson_time()
    hour = time.localtime().tm_hour

    if endhour < starthour:
        return dry_run or (hour < endhour) or (hour > starthour)
    else:
        return dry_run or ( (hour < endhour) and (hour > starthour) )
