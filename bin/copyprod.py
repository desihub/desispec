#!/usr/bin/env python

"""
Copy a DESI production from one directory to another

Currently this only copies the directory tree and the extracted frame files.
This script will be expanded to support copying through any level of
processing, including the metadata of how we got to that processing step.

Stephen Bailey
April 2015
"""

import os
import shutil
import optparse

parser = optparse.OptionParser(usage = "%prog [options] indir outdir")
# parser.add_option("-i", "--input", type="string",  help="input data")
# parser.add_option("-x", "--xxx",   help="some flag", action="store_true")
opts, args = parser.parse_args()

inroot, outroot = args

if not os.path.exists(outroot):
    os.makedirs(outroot)

#- Copy exposures
for indir, subdirs, filenames in os.walk(inroot+'/exposures'):
    outdir = indir.replace(inroot, outroot)
    print outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for name in filenames:
        if name.startswith('frame-'):
            shutil.copy2(indir+'/'+name, outdir+'/'+name)
            

