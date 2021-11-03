# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desispec.database.util
======================

Classes and functions for use by all database code.
"""
from datetime import datetime
from os.path import expanduser


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
