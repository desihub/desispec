#!/usr/bin/env python

import glob
import os
import re
from subprocess import Popen, PIPE
from setuptools import setup, Command, find_packages


def update_version_py():
    if not os.path.isdir(".git"):
        print "This is not a git repository."
        return
    try:
        p = Popen(["git", "describe", "--tags", "--dirty", "--always"], stdout=PIPE)
    except EnvironmentError:
        print "unable to run git, leaving py/desispec/_version.py alone"
        return
    out = p.communicate()[0]
    ver = out.rstrip()
    if p.returncode != 0:
        print "unable to run git, leaving py/desispec/_version.py alone"
        return
    f = open("py/desispec/_version.py", "w")
    f.write( '__version__ = \'{}\''.format( ver ) )
    f.close()
    print "Set py/desispec/_version.py to {}".format( ver )


def get_version():
    if not os.path.isfile("py/desispec/_version.py"):
        print 'Creating initial version file'
        update_version_py()
    ver = 'unknown'
    f = open("py/desispec/_version.py", "r")
    for line in f.readlines():
        mo = re.match("__version__ = '(.*)'", line)
        if mo:
            ver = mo.group(1)
    f.close()
    return ver


class Version(Command):
    description = "update _version.py from git repo"
    user_options = []
    boolean_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        update_version_py()
        ver = get_version()
        print "Version is now {}".format( ver )


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
