import unittest, os
from uuid import uuid1

import desispec.log as log
import os

class TestLog(unittest.TestCase):
    
    
    
    def test_log(self):
        print "Testing desispec.log"
        desi_level=os.getenv("DESI_LOGLEVEL")
        for level in [None,log.DEBUG,log.INFO,log.WARNING,log.ERROR] :
            logger=log.get_logger(level)
            print "with the requested debugging level=",level
            if desi_level is not None and (desi_level != "" ) :
                print "(but overuled by env. DESI_LOGLEVEL='%s')"%desi_level
            print "--------------------------------------------------"
            logger.debug("This is a debugging message")
            logger.info("This is an information")
            logger.warning("This is an warning")
            logger.error("This is an error")
            logger.critical("This is a critical error")
            
        

#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()           
