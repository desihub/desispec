"""
Example test that can be used as a starting point for other tests

run these with pytest py/desispec/test/test_example.py
"""

import unittest

class TestExample(unittest.TestCase):

    # setUpClass runs once at the start before any tests
    @classmethod
    def setUpClass(cls):
        pass

    # tearDownClass runs once at the end after every test
    @classmethod
    def tearDownClass(cls):
        pass

    # setUp runs before every test
    def setUp(self):
        pass

    # setUp runs after every test
    def tearDown(self):
        pass

    def test_blat(self):
        self.assertEqual(1, 1)


