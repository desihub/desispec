"""
Utility functions to dump log messages
We can have something specific for DESI in the future but for now we use the standard python
"""

import sys
import logging


desi_logger = None

def get_logger(level=logging.DEBUG) :
    """ 
    returns a default desi logger 
    """

    global desi_logger
    
    if desi_logger is not None :
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



