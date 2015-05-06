from __future__ import absolute_import

import os.path
import unittest

def runtests():
    """
    Run all tests in desispec.test.test_*
    """
    #- Load all TestCase classes from desispec/test/test_*.py
    tests = unittest.TestLoader().discover(os.path.dirname(__file__))
    #- Run them
    unittest.TextTestRunner(verbosity=2).run(tests)
    