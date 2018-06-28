# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desispec.scripts.delivery
=========================

Entry point for :command:`desi_dts`.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os


def pack_args(options):
    """Parse and format NERSC-specific command-line options.

    Parameters
    ----------
    :class:`argparse.Namespace`
        The parsed command-line options.

    Returns
    -------
    :class:`list`
        Command-line options that can be appended to an existing command.
    """
    optlist = ("nersc",
               "nersc_queue",
               "nersc_queue_redshifts",
               "nersc_maxtime",
               "nersc_maxnodes",
               "nersc_maxnodes_small",
               "nersc_maxnodes_redshifts",
               "nersc_shifter",
               "mpi_procs",
               "mpi_run",
               "procs_per_node")
    varg = vars(options)
    opts = list()
    for k, v in varg.items():
        if k in optlist:
            if v is not None:
                opts.append("--{0}".format(k))
                if not isinstance(v, bool):
                    opts.append(v)
    return opts


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
    from sys import argv
    from argparse import ArgumentParser
    desc = "Script called by DTS when files are delivered."
    prsr = ArgumentParser(prog=os.path.basename(argv[0]), description=desc)
    prsr.add_argument('-n', '--nersc', default='edison', metavar='NERSC_HOST'
                      help="Run a script on this NERSC system (default %(default)s).")
    prsr.add_argument('-p', '--prefix', metavar='PREFIX', action='append',
                      help="Prepend one or more commands to the night command.")
    prsr.add_argument('-s', '--staging', metavar='DIR',
                      default=os.path.join(os.environ['DESI_ROOT'],
                                           'spectro', 'staging', 'raw'),
                      help=("Staging directory containing night directories " +
                            "(default %(default)s)."))
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


def check_exposure(dst, expid):
    """Ensure that all files associated with an exposure have arrived.

    Parameters
    ----------
    dst : :class:`str`
        Delivery directory, typically ``DESI_SPECTRO_DATA/NIGHT``.
    expid : :class:`int`
        Exposure number.

    Returns
    -------
    :class:`bool`
        ``True`` if all files have arrived.
    """
    files = ('fibermap-{0:08d}.fits', 'desi-{0:08d}.fits.fz', 'guider-{0:08d}.fits.fz')
    return all([os.path.exists(os.path.join(dst, f.format(expid))) for f in files])


def move_file(filename, dst):
    """Move delivered file from the DTS spool to the final raw data area.

    This function will ensure that the destination directory exists.

    Parameters
    ----------
    filename : :class:`str`
        The name, including full path, of the file to move.
    dst : :class:`str`
        The destination *directory*.

    Returns
    -------
    :class:`str`
        The value returned by :func:`shutil.move`.
    """
    from os import mkdir
    from os.path import exists, isdir
    from shutil import move
    from desiutil.log import get_logger
    log = get_logger()
    if not exists(dst):
        log.info("mkdir('{0}', 0o2770)".format(dst))
        mkdir(dst, 0o2770)
    log.info("move('{0}', '{1}')".format(filename, dst))
    return move(filename, dst)


def main():
    """Entry point for :command:`desi_dts_delivery`.

    Returns
    -------
    :class:`int`
        An integer suitable for passing to :func:`sys.exit`.
    """
    from os import environ
    from os.path import dirname, join
    from subprocess import Popen
    from desiutil.log import get_logger
    log = get_logger()
    options = parse_delivery()
    remote_command = ['desi_{0.nightStatus}_night {0.night}'.format(options)]
    if options.prefix is not None:
        remote_command = options.prefix + remote_command
    remote_command = ('(' +
                      '; '.join([c + ' &> /dev/null' for c in remote_command]) +
                      ' &)')
    command = ['ssh', '-n', '-q', options.nersc_host, remote_command]
    log.info("Received file {0.filename} with exposure number {0.exposure:d}.".format(options))
    dst = join(environ['DESI_SPECTRO_DATA'], options.night)
    log.info("Using {0} as raw data directory.".format(dst))
    move_file(options.filename, dst)
    exposure_arrived = check_exposure(dst, options.exposure)
    if options.nightStatus in ('start', 'end') or exposure_arrived:
        log.info("Calling: {0}.".format(' '.join(command)))
        proc = Popen(command)
    return 0
