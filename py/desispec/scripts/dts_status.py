# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desispec.scripts.dts_status
===========================

Entry point for :command:`desi_dts_status`.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import json
import os
import shutil
import sys
from argparse import ArgumentParser
from pkg_resources import resource_filename


def main():
    """Entry point for :command:`desi_dts_status`.

    Returns
    -------
    :class:`int`
        An integer suitable for passing to :func:`sys.exit`.
    """
    prsr = ArgumentParser(description='Update the status of DESI DTS raw data transfers.',
                          prog=os.path.basename(sys.argv[0]))
    prsr.add_argument('-d', '--directory', dest='directory', metavar='DIR',
                      default=os.path.join(os.environ['DESI_ROOT'],
                                           'spectro', 'staging', 'status'),
                      help="Install and update files in DIR.")
    prsr.add_argument('-f', '--failure', action='store_true', dest='failure',
                      help='Indicate that the transfer failed somehow.')
    prsr.add_argument('-i', '--install', action='store_true', dest='install',
                      help='Ensure that HTML and related files are in place.')
    prsr.add_argument('-l', '--last', dest='last', default='',
                      choices=['flats', 'arcs', 'science'],
                      help='Indicate that a certain set of exposures is complete.')
    prsr.add_argument('night', type=int, metavar='YYYYMMDD',
                      help="Night of observation.")
    prsr.add_argument('expid', type=int, metavar='N',
                      help="Exposure number.")
    options = prsr.parse_args()
    json_file = os.path.join(options.directory, 'dts_status.json')
    if options.install:
        for ext in ('html', 'js'):
            src = resource_filename('desispec', 'data/dts/dts_status.' + ext)
            if ext == 'html':
                shutil.copyfile(src, os.path.join(options.directory, 'index.html'))
            else:
                shutil.copy(src, options.directory)
    row = [options.night, options.expid, not options.failure, options.last]
    with open(json_file) as j:
        s = json.load(j)
    s.append(row)
    with open(json_file, 'w') as j:
        json.dump(s, j, indent=None, separators=(',', ':'))
    return 0
