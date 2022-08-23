# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desispec.database.duplicates
============================

Find rows in a file that contain the same value in a certain column.
The file in question could potentially be very large.
"""
import os
import sys
import json
import numpy as np
from argparse import ArgumentParser
from astropy.table import Table, Column, MaskedColumn


def get_options(*args):
    """Parse command-line options.

    Parameters
    ----------
    args : iterable
        If arguments are passed, use them instead of ``sys.argv``.

    Returns
    -------
    :class:`argparse.Namespace`
        The parsed options.
    """
    prsr = ArgumentParser(description=("Find rows in a file that contain the same value in a certain column."),
                          prog=os.path.basename(sys.argv[0]))
    prsr.add_argument('-H', '--hdu', action='store', type=int, default=1, dest='hdu', metavar='HDU', help="Read tabular data from HDU (default %(default)s).")
    # prsr.add_argument('-f', '--filename', action='store', dest='dbfile',
    #                   default='redshift.db', metavar='FILE',
    #                   help="Store data in FILE (default %(default)s).")
    # prsr.add_argument('-v', '--verbose', action='store_true', dest='verbose',
    #                   help='Print extra information.')
    prsr.add_argument('column', metavar='COL', help='Search for duplicate values in column COL.')
    prsr.add_argument('filename', metavar='FILE', help='Data are in FILE.')
    options = prsr.parse_args()
    return options


def main():
    """Entry point for command-line script.

    Returns
    -------
    :class:`int`
        An integer suitable for passing to :func:`sys.exit`.
    """
    options = get_options()
    data = Table.read(options.filename, hdu=options.hdu)
    column = data[options.column].data
    unique_values, unique_indexes, column_indexes, column_counts = np.unique(column, return_index=True, return_inverse=True, return_counts=True)
    duplicate_values = np.nonzero(column_counts > 1)[0]
    map_duplicates_to_rows = dict()
    for i in duplicate_values:
        try:
            v = int(unique_values[i])
        except ValueError:
            v = str(unique_values[i])
        rows = np.nonzero(column_indexes == i)[0]
        assert rows.shape[0] > 1
        map_duplicates_to_rows[v] = rows.tolist()

    output = os.path.join(os.environ['SCRATCH'], os.path.splitext(os.path.basename(options.filename))[0] + '.json')
    with open(output, 'w') as fp:
        json.dump(map_duplicates_to_rows, fp, indent=None, separators=(',', ':'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
