#!/usr/bin/env python
#
# See top-level LICENSE file for Copyright information
#
# -*- coding: utf-8 -*-

import argparse

import desispec.io

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fibermap', default = None, metavar = 'FILE',
        help = 'Filename containing fibermap to read.')
    args = parser.parse_args()

    if args.fibermap is None:
        print 'Missing required fibermap argument.'
        return -1

    fibermap,hdr = desispec.io.read_fibermap(args.fibermap)
    print fibermap.dtype

if __name__ == '__main__':
    main()
