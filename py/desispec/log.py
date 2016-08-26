"""
desispec.log
============

Utility functions to dump log messages. We can have something specific for
DESI in the future but for now we use the standard Python.
"""
from __future__ import absolute_import, division, print_function
import sys
import logging
import os
import string

desi_logger = None

# just for convenience to avoid importing logging
# we duplicate the logging levels
DEBUG=logging.DEBUG        # Detailed information, typically of interest only when diagnosing problems.
INFO=logging.INFO          # Confirmation that things are working as expected.
WARNING=logging.WARNING    # An indication that something unexpected happened, or indicative of some problem
                           # in the near future (e.g. "disk space low"). The software is still working as expected.
ERROR=logging.ERROR        # Due to a more serious problem, the software has not been able to perform some function.
CRITICAL=logging.CRITICAL  # A serious error, indicating that the program itself may be unable to continue running.

# see example of usage in test/test_log.py


def get_logger(level=None) :
    """Returns a default DESI logger

    Args:
       level: debugging level.

    If environment variable :envvar:`DESI_LOGLEVEL` exists and has value  DEBUG,INFO,WARNING or ERROR (upper or lower case),
    it overules the level argument.
    If :envvar:`DESI_LOGLEVEL` is not set and level=None, the default level is set to INFO.
    """

    global desi_logger

    desi_level=os.getenv("DESI_LOGLEVEL")
    if desi_level is not None and (desi_level != "" ) :
        # forcing the level to the value of DESI_LOGLEVEL, ignoring the requested logging level.
        desi_level=string.upper(desi_level)
        dico={"DEBUG":DEBUG,"INFO":INFO,"WARNING":WARNING,"ERROR":ERROR}
        if desi_level in dico:
            level=dico[desi_level]
        else :
            # amusingly I would need the logger to dump a warning here
            # but this recursion can be problematic
            message="ignore DESI_LOGLEVEL='%s' (only recognize"%desi_level
            for k in dico :
                message+=" %s"%k
            message+=")"
            print(message)

    if level is None :
        level=INFO

    if desi_logger is not None :
        if level is not None :
            desi_logger.setLevel(level)
        return desi_logger

    desi_logger = logging.getLogger("DESI")

    desi_logger.setLevel(level)

    while len(desi_logger.handlers) > 0:
        h = desi_logger.handlers[0]
        desi_logger.removeHandler(h)

    ch = logging.StreamHandler(sys.stdout)

    #formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
    formatter = logging.Formatter('%(levelname)s:%(filename)s:%(lineno)s:%(funcName)s: %(message)s')

    ch.setFormatter(formatter)


    desi_logger.addHandler(ch)

    return desi_logger
