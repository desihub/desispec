import unittest, os
from uuid import uuid1

import desispec.log as log

class TestLog(unittest.TestCase):
    
    
    
    def test_log(self):
        print "Testing desispec.log"
        for level in [None,log.DEBUG,log.INFO,log.WARNING,log.ERROR] :
            logger=log.get_logger(level)
            print "with the debugging level=",level
            print "--------------------------------------------------"
            logger.debug("This is a debugging message")
            logger.info("This is an information")
            logger.warning("This is an warning")
            logger.error("This is an error")
            logger.critical("This is a critical error")
            
        

#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()           
