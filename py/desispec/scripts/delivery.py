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
    prsr.add_argument('-n', '--nersc-host', metavar='NERSC_HOST',
                      dest='nersc_host', default='edison',
                      help='Run night commands on this host (default %(default)s).')
    prsr.add_argument('-p', '--prefix', metavar='PREFIX', action='append',
                      help="Prepend one or more commands to the night command.")
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
    from subprocess import Popen
    from shlex import split, quote
    from desiutil.log import get_logger
    log = get_logger()
    options = parse_delivery()
    remote_command = ['desi_{0.nightStatus}_night {0.night}'.format(options)]
    if options.prefix is not None:
        remote_command = options.prefix + remote_command
    remote_command = ('(' +
                      '; '.join([c + ' &> /dev/null' for c in remote_command]) +
                      ' &)')
    command = ['ssh', '-n', '-q', options.nersc_host, quote(remote_command)]
    log.info("Received file {0.filename} with exposure number {0.exposure:d}.".format(options))
    log.info("Calling: {0}.".format(' '.join(command)))
    proc = Popen(command)
    return 0
