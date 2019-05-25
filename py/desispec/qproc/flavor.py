import numpy as np

from desiutil.log import get_logger

# tool to check the flavor of a qframe

def check_qframe_flavor(qframe,input_flavor=None):
    """
    Tool to check the flavor of a qframe
    
    Args:
      qframe : DESI QFrame object
    
    Optional: 
         input_flavor

    Return:
         flavor string
    """
    log = get_logger()
    
    log.debug("Checking qframe flavor...")
    log.warning("NOT IMPLEMENTED YET, PLACEHOLDER FOR NOW")
    
    if input_flavor is None :
        return "ZERO"
    else :
        return input_flavor
