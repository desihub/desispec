import os
import re
import datetime

def format(readwrite, filename, duration):
    """Return standardized I/O timing message string for logging

    Args:
        readwrite (str): either "read" or "write"
        filename (str): filename that was read or written
        duration (float): time in seconds to perform I/O operations

    Returns: I/O timing message to log

    Note: this function does not call log.info() itself so that the logging
    source file and line number can be associated with the I/O function itself
    instead of this utility formatting function.
    """
    basename = os.path.basename(filename)
    timestamp = datetime.datetime.now().isoformat()
    msg = f"iotime {duration:.3f} sec to {readwrite} {basename} at {timestamp}"
    return msg

_iotime_regex = re.compile(r'(.*)iotime ([\d.]+) sec to (read|write) (.*) at (.*)')
_log_regex = re.compile(r'^(DEBUG|INFO|WARNING|ERROR|CRITICAL):(\w+\.py):(\d+):(\w+):(.*)')

def parse(line):
    """Parse a line for an iotime message produced by `format`
    
    Args:
        line (str): the line to parse

    Returns None if no match, or dict with keys function,duration,readwrite,filename,timestamp    
    """
    m = _iotime_regex.match(line)
    if m is not None:
        prefix, duration, readwrite, filename, timestamp = m.groups()
        duration = float(duration)
        logmatch = _log_regex.match(line)
        if prefix == '' or logmatch is None:
            function = 'unknown'
        else:
            function = logmatch.group(4)
            
        return dict(
            function=function, duration=duration, readwrite=readwrite,
            filename=filename, timestamp=timestamp)
    else:
        return None

def parse_logfile(logfile):
    """
    Read iotime log entries from logfile

    Return Table with columns function duration readwrite filename timestamp datetime
    """
    from astropy.table import Table
    from astropy.time import Time

    rows = list()
    with open(logfile) as fx:
        for line in fx:
            row = parse(line)
            if row is not None:
                rows.append(row)

    timing = Table(rows=rows)
    timing['datetime'] = Time(timing['timestamp']).datetime

    return timing

"""
Utilities for parsing and plotting I/O timing from logfiles
"""

import re
import numpy as np
from astropy.table import Table, vstack
from astropy.time import Time

def _ordered_unique_names(names):
    """Return unique list of names, ordered by first appearance in list
   
    Doesn't scale well; intended for inputs <10000 long
    """
    unique = list()
    for n in names:
        if n not in unique:
            unique.append(n)

    return unique

def hist_iotimes(timing, tmax=10, plottitle=None):
    """
    Histogram function timing from Table read with read_iotimes

    Args:
        timing: Table with columns FUNC, DATETIME, IOTIME

    Options:
        tmax (float): upper bound of histogram (overflows included in last bin)o
        plottitle (str): plot title

    Returns matplotlib Figure
    """
    import matplotlib.pyplot as plt
    funcnames = _ordered_unique_names(timing['function'])
    nfunc = len(funcnames)
    fig = plt.figure(figsize=(6,8))
    for i, func in enumerate(funcnames):
        jj = timing['function'] == func
        t = timing['duration'][jj].clip(0, tmax)
        plt.subplot(nfunc, 1, i+1)
        plt.hist(t, 25, (0, tmax+1e-3))
        plt.text(tmax, 1, func, ha='right')
        if i != nfunc-1:
            locs, labels = plt.xticks()
            plt.xticks(locs, ['',]*len(locs))

        plt.xlim(-0.1, tmax+0.1)

        if i == 0 and plottitle is not None:
            plt.title(plottitle)

    plt.xlabel('I/O time')

    return fig


def plot_iotimes(timing, plottitle=None, outfile=None):
    """
    Plot I/O duration vs. time of I/O operation

    Args:
        timing: Table with columns FUNC, DATETIME, IOTIME

    Options:
        plottitle (str): Title to include for plot
        outfile (str): write plot to this file

    Returns matplotlib figure; does *not* call plt.show()
    """
    import matplotlib.pyplot as plt
    funcnames = _ordered_unique_names(timing['function'])
    nfunc = len(funcnames)
    fig = plt.figure()

    for i, func in enumerate(funcnames):
        ii = timing['function'] == func
        marker = ('.', 'x', '+', 's', '^', 'v')[i//10]
        plt.plot(timing['datetime'][ii], timing['duration'][ii], marker,
                 label=func)

    plt.xlabel('datestamp of I/O')
    plt.ylabel('I/O time [sec]')

    #- Allow room for large legend
    plt.legend(ncol=2, fontsize='small')
    tmax = np.max(timing['duration'])
    plt.ylim(-0.5, 2*tmax)

    if plottitle is not None:
        plt.title(plottitle)

    if outfile is not None:
        plt.savefig(outfile)

    return fig

#-----
#- for convenience, optionally use this as a script without putting it
#- into the default PATH
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(usage = "{prog} [options]")
    parser.add_argument("--logfiles", type=str, nargs="*", required=True,
                        help="input log files")

    args = parser.parse_args()

    import matplotlib.pyplot as plt
    timing = vstack([parse_logfile(logfile) for logfile in args.logfiles])
    hist_iotimes(timing)
    plot_iotimes(timing)
    plt.show()

