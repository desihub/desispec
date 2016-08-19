"""
desispec.test.desispec_test_suite
=================================

Used to initialize the unit test framework via ``python setup.py test``.
"""
#
from __future__ import absolute_import, division, print_function
#
import unittest
#
#- This is factored out separately from runtests() so that it can be used by
#- python setup.py test
def desispec_test_suite():
    """Returns unittest.TestSuite of desispec tests"""
    from os.path import dirname
    desispec_dir = dirname(dirname(__file__))
    # print(desispec_dir)
    return unittest.defaultTestLoader.discover(desispec_dir,
        top_level_dir=dirname(desispec_dir))

def runtests():
    """Run all tests in desispec.test.test_*.
    """
    #- Load all TestCase classes from desispec/test/test_*.py
    tests = desispec_test_suite()
    #- Run them
    unittest.TextTestRunner(verbosity=2).run(tests)

if __name__ == "__main__":
    runtests()
