"""
desispec.test.desispec_test_suite
=================================

Used to initialize the unit test framework via ``python setup.py test``.
"""
#
from __future__ import absolute_import, division, print_function, unicode_literals
#
import unittest
from os.path import dirname
#
#- This is factored out separately from runtests() so that it can be used by
#- python setup.py test
def desispec_test_suite():
    """Returns unittest.TestSuite of desispec tests"""
    desispec_dir = dirname(dirname(__file__))
    return unittest.defaultTestLoader.discover(desispec_dir)

def runtests():
    """
    Run all tests in desispec.test.test_*
    """
    #- Load all TestCase classes from desispec/test/test_*.py
    tests = desispec_test_suite()
    
    #- Run them
    unittest.TextTestRunner(verbosity=2).run(tests)

def orig_runtests():
    """
    Run all tests in desispec.test.test_*
    """
    import os.path
    #- Load all TestCase classes from desispec/test/test_*.py
    tests = unittest.TestLoader().discover(os.path.dirname(__file__))
        
    #- Run them
    unittest.TextTestRunner(verbosity=2).run(tests)
