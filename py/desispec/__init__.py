# License information goes here
# -*- coding: utf-8 -*-
"""
========
template
========

This package is a template for other DESI_ Python_ packages.

The function desiUtil.install.version() should be used to set the ``__version__``
package variable.  In order for this to work properly, the svn property
svn:keywords must be set to HeadURL on this file.

.. _DESI: http://desi.lbl.gov
.. _Python: http://python.org
"""
#
from __future__ import absolute_import, division, print_function, unicode_literals
# The line above will help with 2to3 support.
from desiUtil.install import version
#
# Set version string.
#
__version__ = version('$HeadURL: https://desi.lbl.gov/svn/code/tools/desiTemplate/trunk/py/desiTemplate/__init__.py $')
#
# Clean up namespace
#
del version
