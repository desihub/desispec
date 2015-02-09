"""
Utility functions to dump log messages
We can have something specific for DESI in the future but for now we use the standard python
"""

import sys
import logging


""" 
provides default desi logger 
"""

def desi_logger() :
    logger = logging.getLogger("DESI")
    
    logger.setLevel(logging.DEBUG)
    
    while len(logger.handlers) > 0:
        h = logger.handlers[0]
        logger.removeHandler(h)
    
    ch = logging.StreamHandler(sys.stdout)
    #formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
    formatter = logging.Formatter('%(levelname)s:%(filename)s:%(lineno)s:%(funcName)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger



