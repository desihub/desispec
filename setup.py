#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE
from __future__ import absolute_import, print_function
import glob
import os
import re
from subprocess import Popen, PIPE
from setuptools import setup, Command, find_packages
from distutils.log import INFO


def update_version_py(tag=None,debug=False):
    """Update the _version.py file.

    Args:
        tag (str, optional) : Set the version to this string, unconditionally.
        debug (bool, optional) : Print extra debug information.
    Returns:
        None
    """
    if tag is not None:
        ver = tag
    else:
        if not os.path.isdir(".git"):
            print("This is not a git repository.")
            return
        no_git = "Unable to run git, leaving py/desispec/_version.py alone."
        try:
            p = Popen(["git", "describe", "--tags", "--dirty", "--always"], stdout=PIPE, stderr=PIPE)
        except EnvironmentError:
            print("Could not run 'git describe'!")
            print(no_git)
            return
        out, err = p.communicate()
        if p.returncode != 0:
            print("Returncode = {0}".format(p.returncode))
            print(no_git)
            return
        ver = out.rstrip().split('-')[0]+'.dev'
        try:
            p = Popen(["git", "rev-list", "--count", "HEAD"], stdout=PIPE, stderr=PIPE)
        except EnvironmentError:
            print("Could not run 'git rev-list'!")
            print(no_git)
            return
        out, err = p.communicate()
        if p.returncode != 0:
            print("Returncode = {0}".format(p.returncode))
            print(no_git)
            return
        ver += out.rstrip()
    with open("py/desispec/_version.py", "w") as f:
        f.write( "__version__ = '{}'\n".format( ver ) )
    if debug:
        print("Set py/desispec/_version.py to {}".format( ver ))
    return

def get_version(debug=False):
    if not os.path.isfile("py/desispec/_version.py"):
        if debug:
            print('Creating initial version file.')
        update_version_py(debug=debug)
    ver = 'unknown'
    with open("py/desispec/_version.py", "r") as f:
        for line in f.readlines():
            mo = re.match("__version__ = '(.*)'", line)
            if mo:
                ver = mo.group(1)
    return ver


class Version(Command):
    description = "update _version.py from git repo"
    user_options = [ ('tag=', 't', 'Set the version to a name in preparation for tagging.'), ]
    boolean_options = []
    def initialize_options(self):
        self.tag = None
    def finalize_options(self):
        pass
    def run(self):
        update_version_py(tag=self.tag)
        ver = get_version()
        self.announce("Version is now {}.".format( ver ), level=INFO)


current_version = get_version()

setup (
    name='desispec',
    provides='desispec',
    version=current_version,
    description='DESI Spectroscopic Tools',
    author='DESI Collaboration',
    author_email='desi-data@desi.lbl.gov',
    url='https://github.com/desihub/desispec',
    package_dir={'':'py'},
    packages=find_packages('py'),
    scripts=[ fname for fname in glob.glob(os.path.join('bin', '*.py')) ],
    license='BSD',
    requires=['Python (>2.7.0)', ],
    use_2to3=True,
    zip_safe=False,
    cmdclass={'version': Version},
    test_suite='desispec.test.desispec_test_suite.desispec_test_suite'
)
