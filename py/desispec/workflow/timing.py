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

def get_night_banner(night=None):
    if night is None:
        night = what_night_is_it()
    return '\n#############################################' + \
           f'\n################ {night} ###################' + \
           '\n#############################################'