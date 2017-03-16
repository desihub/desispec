# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desispec.scripts.delivery
=========================

Entry point for :command:`desi_dts_delivery`.
"""
from __future__ import absolute_import, division, print_function, unicode_literals


def parse_delivery(*args):
    """Parse command-line options for DTS delivery script.

    Parameters
    ----------
    args : iterable
        Arguments to the function will be parsed for testing purposes.

    Returns
    -------
    :class:`argparse.Namespace`
        The parsed command-line options.
    """
    from os.path import basename
    from sys import argv
    from argparse import ArgumentParser
    desc = "Script called by DTS when files are delivered."
    prsr = ArgumentParser(prog=basename(argv[0]), description=desc)
    prsr.add_argument('filename', metavar='FILE',
                      help='Filename with path of delivered file.')
    prsr.add_argument('exposure', type=int, metavar='EXPID',
                      help='Exposure number.')
    prsr.add_argument('night', metavar='YYYYMMDD', help='Night ID.')
    prsr.add_argument('nightStatus',
                      choices=('start', 'update', 'end'),
                      help='Start/end info.')
    if len(args) > 0:
        options = prsr.parse_args(args)
    else:  # pragma: no cover
        options = prsr.parse_args()
    return options


def main():
    """Entry point for :command:`desi_dts_delivery`.

    Returns
    -------
    :class:`int`
        An integer suitable for passing to :func:`sys.exit`.
    """
    options = parse_delivery()
    return 0
