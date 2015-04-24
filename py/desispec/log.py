"""
Utility functions to dump log messages
We can have something specific for DESI in the future but for now we use the standard python
"""

import sys
import logging
import os

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
    """ 
    returns a default desi logger

    Args:
       level: debugging level.
    
    If level=None, will look for environment variable DESI_LOGLEVEL, accepting only values DEBUG,INFO,WARNING,ERROR.
    If DESI_LOGLEVEL is not set, default level is INFO.    
    """

    if level is None :
        desi_level=os.getenv("DESI_LOGLEVEL")
        if desi_level is None : 
            level=INFO
        else :
            dico={"DEBUG":DEBUG,"INFO":INFO,"WARNING":WARNING,"ERROR":ERROR}
            if dico.has_key(desi_level) :
                level=dico[desi_level]
            else :
                # amusingly I need the logger to dump a warning here
                logger=get_logger(level=WARNING)
                message="ignore DESI_LOGLEVEL=%s (only recognize"%desi_level
                for k in dico :
                    message+=" %s"%k
                message+=")"
                logger.warning(message)
                level=INFO
            
                
    global desi_logger
    
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



