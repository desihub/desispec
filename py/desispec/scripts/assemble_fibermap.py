#!/usr/bin/env python

import os, sys, json
import argparse

from desiutil.log import get_logger
from desispec.io.fibermap import assemble_fibermap
from desispec.pixgroup import fibermap2tilepix
    
def parse(options=None):
        
    parser = argparse.ArgumentParser(usage = "{prog} [options]")
    parser.add_argument("-n", "--night", type=int, required=True,help="input night")
    parser.add_argument("-e", "--expid", type=int, required=True,help="spectroscopic exposure ID")
    parser.add_argument("-o", "--outfile", type=str, required=True,help="output filename")
    parser.add_argument("-t", "--tilepix", type=str, required=False,help="output tilepix json filename")
    parser.add_argument("-b","--badamps", type=str,help="comma separated list of {camera}{petal}{amp} i.e. [brz][0-9][ABCD]. Example: 'b7D,z8A'")
    parser.add_argument("--badfibers", type=str,help="filename with table of bad fibers (with at least FIBER and FIBERSTATUS columns)")
    parser.add_argument("--debug", action="store_true",help="enter ipython debug mode at end")
    parser.add_argument("--overwrite", action="store_true",help="overwrite pre-existing output file")
    parser.add_argument("--force", action="store_true",help="make fibermap even if missing input guide or coordinates files")
    parser.add_argument("--no-svn-override", action="store_true",help="Do not allow fiberassign SVN to override raw data")

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args

def main(args=None):

    log = get_logger()

    if args is None:
        args = parse()

    if os.path.exists(args.outfile):
        if args.overwrite:
            log.info(f'Overwriting pre-existing {args.outfile}')
            os.remove(args.outfile)
        else:
            log.critical(f'{args.outfile} already exists; remove or use --overwrite')
            sys.exit(1)

    fibermap = assemble_fibermap(args.night, args.expid, badamps=args.badamps,
            badfibers_filename=args.badfibers, force=args.force,
            allow_svn_override=(not args.no_svn_override) )

    tmpfile = args.outfile+'.tmp'
    fibermap.writeto(tmpfile, output_verify='fix+warn', overwrite=args.overwrite, checksum=True)
    os.rename(tmpfile, args.outfile)
    log.info(f'Wrote {args.outfile}')

    if args.tilepix:
        tileid = fibermap['FIBERMAP'].header['TILEID']
        tilepix = {str(tileid): fibermap2tilepix(fibermap['FIBERMAP'].data)}
        tmpfile = args.tilepix+'.tmp'
        with open(tmpfile, 'w') as fp:
            json.dump(tilepix, fp)
        os.rename(tmpfile, args.tilepix)
        log.info(f'Wrote {args.tilepix}')

    if args.debug:
        import IPython; IPython.embed()

    return 0
