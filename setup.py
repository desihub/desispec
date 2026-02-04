#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

# NOTE: The configuration for the package, including the name, version, and
# other information are set in the setup.cfg file.

import os
import glob
import sys
from setuptools import setup

# First provide helpful messages if contributors try and run legacy commands
# for tests or docs.

API_HELP = """
Note: Generating api.rst files is no longer done using 'python setup.py api'. Instead
you will need to run:

    desi_api_file

which is part of the desiutil package. If you don't already have desiutil installed, you can install it with:

    pip install desiutil
"""

MODULE_HELP = """
Note: Generating Module files is no longer done using 'python setup.py api'. Instead
you will need to run:

    desiInstall

or

    desi_module_file

depending on your exact situation.  desiInstall is preferred.  Both commands are
part of the desiutil package. If you don't already have desiutil installed, you can install it with:

    pip install desiutil
"""

VERSION_HELP = """
Note: Generating version strings is no longer done using 'python setup.py version'. Instead
you will need to run:

    desi_update_version [-t TAG] desiutil

which is part of the desiutil package. If you don't already have desiutil installed, you can install it with:

    pip install desiutil
"""

TEST_HELP = """
Note: running tests is no longer done using 'python setup.py test'. Instead
you will need to run:

    pytest

If you don't already have pytest installed, you can install it with:

    pip install pytest
"""

DOCS_HELP = """
Note: building the documentation is no longer done using
'python setup.py {0}'. Instead you will need to run:

    sphinx-build -W --keep-going -b html doc doc/_build/html

If you don't already have Sphinx installed, you can install it with:

    pip install Sphinx
"""

message = {'api': API_HELP,
           'module_file': MODULE_HELP,
           'test': TEST_HELP,
           'version': VERSION_HELP,
           'build_docs': DOCS_HELP.format('build_docs'),
           'build_sphinx': DOCS_HELP.format('build_sphinx'), }

for m in message:
    if m in sys.argv:
        print(message[m])
        sys.exit(1)
#
# Begin setup
#
setup_keywords = dict()
#
# Set other keywords for the setup function.  These are automated, & should
# be left alone unless you are an expert.
#
# Treat everything in bin/ except *.rst as a script to be installed.
#
if os.path.isdir('bin'):
    setup_keywords['scripts'] = [fname for fname in glob.glob(os.path.join('bin', '*'))
                                 if os.access(fname, os.X_OK)]
#
# Run setup command.
#
setup(**setup_keywords)
