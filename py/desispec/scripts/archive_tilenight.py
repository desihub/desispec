#!/usr/bin/env python

"""
Utility script to remove archive tiles for the merged target list (MTL).
"""

import os, sys, glob
import argparse
import subprocess
import shutil
import datetime

import numpy as np
from astropy.table import Table

from desiutil.log import get_logger
from desispec.io import specprod_root, findfile

def archivetile(tiledir, archivedir, dryrun=False):
    """
    Archive tiledir to archivedir, leaving symlink behind

    Args:
        tiledir: full path to tiles/cumulative/TILEID/LASTNIGHT
        archivedir: full path to tiles/archive/TILEID/ARCHIVEDATE

    Options:
        dryrun: if True, print messages but don't move directories.

    Returns int error code, non-zero if there was a problem
    """
    log = get_logger()
    if os.path.isdir(archivedir):
        log.info(f'{archivedir} already exists')
        return 0

    if not os.path.isdir(tiledir):
        log.error(f'{tiledir} missing ... skipping')
        return 1

    err = 0
    if dryrun:
        log.info(f'Dry run: archive {tiledir} -> {archivedir}')
    else:
        #- Move tiledir -> archivedir
        outdir = os.path.dirname(archivedir)
        os.makedirs(outdir, exist_ok=True)
        shutil.move(tiledir, archivedir)

        #- Create relative link from original tiledir -> new archivedir
        src = os.path.relpath(archivedir, os.path.dirname(tiledir))
        dst = tiledir
        os.symlink(src, dst)

        #- Remove write access
        err = freezedir(archivedir, dryrun=dryrun)
        if err != 0:
            log.error(f'problem removing write access from {archivedir}')

    return err


def freezedir(path, dryrun=False):
    """
    Remove write permission from path unless dryrun

    Args:
        path (str): path to directory to remove

    Options:
        dryrun: if True, print info but don't actually remove write access

    Returns non-zero error code upon failure (not an exception)
    """
    log = get_logger()
    err = 0
    if not os.path.isdir(path):
        log.error(f'Not a directory; skipping {path}')
        err = 1
    elif not os.access(path, os.W_OK):
        log.info(f'{path} already frozen')
    else:
        if dryrun:
            log.info(f'Dry run: freeze {path}')
        else:
            log.info(f'Freezing {path}')
            cmd = f'chmod -R a-w {path}'
            err = subprocess.call(cmd.split())
            if err != 0:
                log.error(f'Freezing {path} failed')

    return err

#-------------------------------------------------------------------------

def parse(options=None):
    parser = argparse.ArgumentParser(
        description='Archive tiles to read-only tiles/archive/TILEID/ARCHIVEDATE')
    parser.add_argument('-t', '--tileids', type=str,
            help='archive only these TILEIDs (comma separated)')
    parser.add_argument('--thru', type=int,
            help='archive tiles observed this LASTNIGHT or before')
    parser.add_argument('--since', type=int,
            help='archive tiles observed this LASTNIGHT or after')
    parser.add_argument('-n', '--night', type=int,
            help='archive only tiles observed on this LASTNIGHT')
    parser.add_argument('-p', '--prod', type=str,
            help = 'Path to input reduction, e.g. '
                   '/global/cfs/cdirs/desi/spectro/redux/daily, '
                   'or simply prod version, like daily. '
                   'Default is $DESI_SPECTRO_REDUX/$SPECPROD.')
    parser.add_argument('--specstatus', type=str,
            help='tiles-specstatus.ecsv file to use; archive tiles with '
            'ZDONE=false and QA=true; update to ZDONE=true and set ARCHIVEDATE')
    parser.add_argument('--dry-run', action='store_true',
            help='print what directories to archive, without archiving them')
    parser.add_argument('--only-tiles', action='store_true',
            help="archive only tiles dirs; don't freeze exposures, preproc")

    args = parser.parse_args(options)
    return args

def main(options=None):
    args = parse(options)
    log = get_logger()

    if args.prod is None:
        reduxdir = specprod_root()
    elif args.prod.count("/") == 0:
        # convert prod as a SPECPROD name to a full path
        reduxdir = specprod_root(args.prod)
    else:
        reduxdir = args.prod

    if args.specstatus is None:
        if os.path.exists('./tiles-specstatus.ecsv'):
            log.info('Using tiles-specstatus.ecsv in current directory')
            args.specstatus = os.path.abspath('./tiles-specstatus.ecsv')
        else:
            log.info('Using default surveyops/ops/tiles-specstatus.ecsv')
            args.specstatus = os.path.expandvars(
                '$DESI_ROOT/survey/ops/surveyops/trunk/ops/tiles-specstatus.ecsv')

    if not os.path.exists(args.specstatus):
        log.critical(f'Missing {args.specstatus}')
        sys.exit(1)

    log.info(f'Reading tiles from {args.specstatus}')
    if not args.dry_run and not os.access(args.specstatus, os.W_OK):
        filename = os.path.basename(tiles.specstatus)
        log.critical('Need write access to {filename} to update ZDONE and ARCHIVEDATE')
        sys.exit(1)

    log.info(f'Archiving tiles in {reduxdir}')
    os.chdir(reduxdir)

    tiles = Table.read(args.specstatus)
    log.info('Archiving tiles with SURVEY=main|sv3, ZDONE=false, QA=good')
    keep = (tiles['SURVEY'] == 'main') | (tiles['SURVEY'] == 'sv3')
    keep &= (tiles['ZDONE'] == 'false') & (tiles['QA'] == 'good')
    if args.thru is not None:
        log.info(f'Filtering to LASTNIGHT<={args.thru}')
        keep &= tiles['LASTNIGHT'] <= args.thru
    if args.since is not None:
        log.info(f'Filtering to LASTNIGHT>={args.since}')
        keep &= tiles['LASTNIGHT'] >= args.since
    if args.night is not None:
        log.info(f'Filtering to just LASTNIGHT={args.night}')
        keep &= tiles['LASTNIGHT'] == args.night
    if args.tileids is not None:
        tileids = [int(t) for t in args.tileids.split(',')]
        log.info(f'Filtering to just TILEIDs {tileids}')
        keep &= np.isin(tiles['TILEID'], tileids)

    archivetiles = tiles[keep]

    archivedate = datetime.datetime.now().strftime('%Y%m%d')
    ntiles = len(archivetiles)
    if ntiles == 0:
        log.warning('No tiles to archive; exiting')
        sys.exit(0)

    log.info(f'Archiving {ntiles} tiles using ARCHIVEDATE={archivedate}')

    #- find and read exposures table for that specprod
    specprod = os.path.basename(reduxdir)
    for expfile in [
        f'{reduxdir}/exposures-{specprod}.fits',
        f'{reduxdir}/tsnr-exposures.fits',
        ]:
        if os.path.exists(expfile):
            log.info(f'Reading exposures from {expfile}')
            #- HDU name changed mid-2021, so read HDU 1 for backwards compatibility
            #- with other productions (though this may only be used with daily)
            exposures = Table.read(expfile, 1)
            break
    else:
        #- for-loop else only runs if loop finishes without break
        log.error(f'Unable to find an exposures files in {reduxdir}; not freezing exposures')
        exposures = None

    failed_archive_tiles = list()
    for tileid, lastnight in archivetiles['TILEID','LASTNIGHT']:
        log.info(f'-- Archiving {tileid} {lastnight}')

        #- Archive tile
        tiledir = f'tiles/cumulative/{tileid}/{lastnight}'
        archivedir = f'tiles/archive/{tileid}/{archivedate}'
        tile_err = archivetile(tiledir, archivedir, args.dry_run)

        #- Remove write access from any input preproc and exposures
        if (exposures is not None) and (not args.only_tiles):
            ii = (exposures['TILEID'] == tileid)
            for prefix in ['preproc', 'frame']:
                for expnight, expid, efftime in exposures['NIGHT', 'EXPID', 'EFFTIME_SPEC'][ii]:
                    tmpfile = findfile(prefix, expnight, expid, 'b0',
                                       specprod_dir=reduxdir)
                    path = os.path.dirname(tmpfile).replace(reduxdir+'/', '')

                    #- missing dir ok if efftime==0, otherwise error
                    if efftime == 0.0 and not os.path.exists(path):
                        continue

                    exp_err = freezedir(path, args.dry_run)
                    tile_err |= exp_err
                    if exp_err != 0:
                        log.error(f'problem removing write access from {path}')

        if tile_err == 0:
            i = np.where(tiles['TILEID'] == tileid)[0][0]
            tiles['ARCHIVEDATE'][i] = archivedate
            tiles['ZDONE'][i] = 'true'
        else:
            log.error(f'Tile {tileid} had errors while archiving; not setting ARCHIVEDATE')
            failed_archive_tiles.append(tileid)

    if not args.dry_run:
        tiles.write(args.specstatus, overwrite=True)
        log.info(f'Updated {args.specstatus} with new ZDONE and ARCHIVEDATE')
        log.info('Remember to svn commit that file now')

    if len(failed_archive_tiles) > 0:
        log.critical(f'Some tiles failed archiving: {failed_archive_tiles}')
        sys.exit(1)

    return 0

