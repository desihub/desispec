# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desispec.database.util
======================

Classes and functions for use by all database code.
"""
from datetime import datetime
from os.path import expanduser


_surveyid = {'cmx': 1, 'special': 2, 'sv1': 3, 'sv2': 4, 'sv3':5, 'main':6}
_programid = {'backup': 1, 'bright': 2, 'dark': 3, 'other': 4}
_spgrpid = {'1x_depth': 1, '4x_depth': 2, 'cumulative': 3, 'lowspeed': 4, 'perexp': 5, 'pernight': 6}


def cameraid(camera):
    """Converts `camera` (*e.g.* 'b0') to an integer in a simple but ultimately
    arbitrary way.

    Parameters
    ----------
    camera : :class:`str`
        Camera name.

    Returns
    -------
    :class:`int`
        An arbitrary integer, though in the range [0, 29].
    """
    return 'brz'.index(camera[0]) * 10 + int(camera[1])


def frameid(expid, camera):
    """Converts the pair `expid`, `camera` into an arbitrary integer
    suitable for use as a primary key.
    """
    return 100*expid + cameraid(camera)


def surveyid(survey):
    """Converts `survey` (*e.g.* 'main') to an integer in a simple but ultimately
    arbitrary way.

    Parameters
    ----------
    survey : :class:`str`
        Survey name.

    Returns
    -------
    :class:`int`
        An arbitrary, small integer.
    """
    return _surveyid[survey]


def programid(program):
    """Converts `program` (*e.g.* 'bright') to an integer in a simple but ultimately
    arbitrary way.

    Parameters
    ----------
    program : :class:`str`
        Program name.

    Returns
    -------
    :class:`int`
        An arbitrary, small integer.
    """
    return _programid[program]


def spgrpid(spgrp):
    """Converts `spgrp` (*e.g.* 'cumulative') to an integer in a simple but ultimately
    arbitrary way.

    Parameters
    ----------
    spgrp : :class:`str`
        SPGRP name.

    Returns
    -------
    :class:`int`
        An arbitrary, small integer.
    """
    return _spgrpid[spgrp]


def targetphotid(targetid, tileid, survey):
    """Convert inputs into an arbitrary large integer.

    Parameters
    ----------
    targetid : :class:`int`
        Standard ``TARGETID``.
    tileid : :class:`int`
        Standard ``TILEID``.
    survey : :class:`str`

    Returns
    -------
    :class:`int`
        An arbitrary integer, which will be greater than :math:`2^64` but
        less than :math:`2^128`.
    """
    return (surveyid(survey) << 96) | (tileid << 64) | targetid


def zpixid(targetid, survey, program):
    """Convert inputs into an arbitrary large integer.

    Parameters
    ----------
    targetid : :class:`int`
        Standard ``TARGETID``.
    survey : :class:`str`
        Survey name.
    program : :class:`str`
        Program name.

    Returns
    -------
    :class:`int`
        An arbitrary integer, which will be greater than :math:`2^64` but
        less than :math:`2^128`.
    """
    return (programid(program) << 96) | (surveyid(survey) << 64) | targetid


def ztileid(targetid, spgrp, spgrpval, tileid):
    """Convert inputs into an arbitrary large integer.

    Parameters
    ----------
    targetid : :class:`int`
        Standard ``TARGETID``.
    spgrp : :class:`str`
        Tile grouping.
    spgrpval : :class:`str`
        Id with in `spgrp`.
    tileid : :class:`int`
        Standard ``TILEID``.

    Returns
    -------
    :class:`int`
        An arbitrary integer, which will be greater than :math:`2^64` but
        less than :math:`2^128`.
    """
    spgrpid = (_spgrpid[spgrp] << 27) | spgrpval  # effective 32-bit integer
    return (spgrpid << 96) | (tileid << 64) | targetid


def fiberassignid(targetid, tileid, location):
    """Convert inputs into an arbitrary large integer.

    Parameters
    ----------
    targetid : :class:`int`
        Standard ``TARGETID``.
    tileid : :class:`int`
        Standard ``TILEID``.
    location : :class:`int`
        Location on the tile.

    Returns
    -------
    :class:`int`
        An arbitrary integer, which will be greater than :math:`2^64` but
        less than :math:`2^128`.
    """
    return (location << 96) | (tileid << 64) | targetid


def convert_dateobs(timestamp, tzinfo=None):
    """Convert a string `timestamp` into a :class:`datetime.datetime` object.

    Parameters
    ----------
    timestamp : :class:`str`
        Timestamp in string format.
    tzinfo : :class:`datetime.tzinfo`, optional
        If set, add time zone to the timestamp.

    Returns
    -------
    :class:`datetime.datetime`
        The converted `timestamp`.
    """
    x = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%f')
    if tzinfo is not None:
        x = x.replace(tzinfo=tzinfo)
    return x


def parse_pgpass(hostname='nerscdb03.nersc.gov', username='desidev_admin'):
    """Read a ``~/.pgpass`` file.

    Parameters
    ----------
    hostname : :class:`str`, optional
        Database hostname.
    username : :class:`str`, optional
        Database username.

    Returns
    -------
    :class:`str`
        A string suitable for creating a SQLAlchemy database engine, or None
        if no matching data was found.
    """
    fmt = "postgresql://{3}:{4}@{0}:{1}/{2}"
    try:
        with open(expanduser('~/.pgpass')) as p:
            lines = p.readlines()
    except FileNotFoundError:
        return None
    data = dict()
    for l in lines:
        d = l.strip().split(':')
        if d[0] in data:
            data[d[0]][d[3]] = fmt.format(*d)
        else:
            data[d[0]] = {d[3]: fmt.format(*d)}
    if hostname not in data:
        return None
    try:
        pgpass = data[hostname][username]
    except KeyError:
        return None
    return pgpass
