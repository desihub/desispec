#!/usr/bin/env python

"""
This bundles up scripts with pyinstaller.  

Run this from the top level of the desispec source tree.  This same
functionality could be trivially added to a normal setup.py install
target, but DESI does not currently support "python setup.py install".
"""

import sys
import os
import numpy as np
import argparse
import re
import subprocess as sp
import shutil

def main():
    parser = argparse.ArgumentParser(description='Bundle apps with pyinstaller.')
    parser.add_argument('--prefix', required=True, default=None, help='The install prefix directory.  Apps will be installed to <prefix>/bin.')
    args = parser.parse_args()

    apps = ['desi_pipe_run_mpi']

    installer = None

    for dir in os.getenv("PATH").split(':'):                                           
        if (os.path.exists(os.path.join(dir, "pyinstaller"))):
            installer = os.path.exists(os.path.join(dir, "pyinstaller"))

    if installer is None:
        print("pyinstaller executable was not found in PATH")
        sys.exit(0)

    sp.check_call(["pyinstaller", "etc/desi_bundle.spec"])
    for ap in apps:
        shutil.copy2(os.path.join("dist", ap), os.path.join(args.prefix, "bin", ap))


if __name__ == "__main__":
    main()

