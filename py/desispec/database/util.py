# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desispec.database.util
======================

Classes and functions for use by all database code.
"""


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
    from datetime import datetime
    x = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%f')
    if tzinfo is not None:
        x = x.replace(tzinfo=tzinfo)
    return x


def parse_pgpass(hostname='scidb2.nersc.gov', username='desidev_admin'):
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
    from os.path import expanduser
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
