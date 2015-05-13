"""
test desispec.log
"""
from __future__ import absolute_import, print_function
import unittest, os
import desispec.log as log

class TestLog(unittest.TestCase):

    def test_log(self):
        desi_level=os.getenv("DESI_LOGLEVEL")
        for level in [None,log.DEBUG,log.INFO,log.WARNING,log.ERROR] :
            logger=log.get_logger(level)
            print("with the requested debugging level={0}".format(level))
            if desi_level is not None and (desi_level != "" ) :
                print("(but overuled by env. DESI_LOGLEVEL='{0}')".format(desi_level))
            print("--------------------------------------------------")
            logger.debug("This is a debugging message")
            logger.info("This is an information")
            logger.warning("This is an warning")
            logger.error("This is an error")
            logger.critical("This is a critical error")



#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
