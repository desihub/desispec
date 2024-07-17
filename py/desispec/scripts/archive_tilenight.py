#!/usr/bin/env python

"""
Utility script to remove archive tiles for the merged target list (MTL).
"""

import os, sys, glob
import argparse
import subprocess
import shutil
import stat
import datetime

import numpy as np
from astropy.table import Table
from astropy.io import fits

from desiutil.log import get_logger
from desispec.io import specprod_root, findfile
from desitarget.targetmask import zwarn_mask

def check_missing_zmtl(tiledir, archivetileroot):
    """
    Check for zmtl file that exist in previous archives but not latest prod

    Args:
        tiledir: full path to tiles/cumulative/TILEID/LASTNIGHT
        archivetileroot: full path to tiles/archive/TILEID

    Returns: dict missing[petal] /path/to/zmtl/in/archive/not/in/prod
    """

    def petal_from_zmtlfilename(zmtlfile):
        return int(os.path.basename(zmtlfile).split('-')[1])


    #- Loop over zmtl files in archive; using sorted will give us most recent
    latest_archived_zmtl = dict()
    for zmtlfile in sorted(glob.glob(f'{archivetileroot}/*/zmtl-*.fits')):
        petal = petal_from_zmtlfilename(zmtlfile)
        latest_archived_zmtl[petal] = zmtlfile

    #- Loop over zmtl files in production tiledir
    latest_proc_zmtl = dict()
    for zmtlfile in sorted(glob.glob(f'{tiledir}/zmtl-*.fits')):
        petal = petal_from_zmtlfilename(zmtlfile)
        latest_proc_zmtl[petal] = zmtlfile

    #- Compare archive to production to look for missing
    missing = dict()
    for petal in latest_archived_zmtl:
        if petal not in latest_proc_zmtl:
            missing[petal] = latest_archived_zmtl[petal]

    return missing

def create_badpetal_zmtl(inzmtl, outzmtl):
    """
    Using inzmtl as template, create outzmtl with ZWARN BAD_PETALQA set

    Args:
        inzmtl: full path to input zmtl file
        outzmtl: full path to output zmtl file
    """
    out_hdus = list()
    with fits.open(inzmtl, mode='readonly') as fin:
        for hdu in fin:
            if ('EXTNAME' in hdu.header) and (hdu.header['EXTNAME'] == 'ZMTL'):
                hdu.data['ZWARN'] |= zwarn_mask.BAD_PETALQA

            out_hdus.append(hdu)

        hx = fits.HDUList(out_hdus)

        #- Add write permission to output dir if needed
        outdir = os.path.dirname(outzmtl)
        outmode = os.stat(outdir).st_mode
        user_rwx = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
        os.chmod(outdir, outmode | user_rwx)

        try:
            hx.writeto(outzmtl, overwrite=True)
        except Exception as err:
            os.chmod(outdir, outmode)  #- reset permissions before crashing
            raise err

        #- Reset output permissions to whatever they were before
        os.chmod(outdir, outmode)

def move_and_link_directory(src, dst):
    """
    Move src to dst, and create link from src -> dst.
    If src is already a link, recursively copy instead.

    Args:
        src: full path to a source directory
        dst: full path to a destination directory that doesn't yet exist

    After this is run:

      * dst will have the contents originally in src
      * src will be a link to dst
    """

    #- remove trailing slashes, etc., needed for correct interpretation
    #- of links vs. directories downstream
    src = os.path.normpath(src)
    dst = os.path.normpath(dst)

    #- Create destination root if needed
    outdir = os.path.dirname(dst)
    os.makedirs(outdir, exist_ok=True)

    #- if link, copy contents and remove link; if regular directory, move it
    if os.path.islink(os.path.normpath(src)):
        shutil.copytree(src, dst)
        os.remove(src)
    else:
        shutil.move(src, dst)

    #- Create relative link from original src -> new dst
    linksrc = os.path.relpath(dst, os.path.dirname(src))
    linkdst = src
    os.symlink(linksrc, linkdst)

def archivetile(tiledir, archivedir, badpetals=list(), dryrun=False):
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

    #- check for petals missing from this prod but in an earlier archive
    missing = check_missing_zmtl(tiledir, os.path.dirname(archivedir))
    for petal in missing:
        if petal not in badpetals:
            log.error(f'Production missing zmtl for tiles previously processed: {missing[petal]}; either fix in prod or use --badpetals {petal} to override')
            return 1

    #- also check that all badpetals are actually missing
    for petal in badpetals:
        if petal not in missing:
            log.error(f"'Bad' petal {petal} actually exists; either remove from prod or don't include in --badpetals")
            return 1

    err = 0
    if dryrun:
        log.info(f'Dry run: archive {tiledir} -> {archivedir}')
    else:
        move_and_link_directory(tiledir, archivedir)

        #- Create BAD_PETALQA zmtl files if needed
        #- previous checks ensured these are valid
        for petal in missing:
            log.info(f'Creating BAD_PETALQA zmtl for bad petal {petal}')
            inzmtl = missing[petal]
            outzmtl = os.path.join(archivedir, os.path.basename(inzmtl))
            create_badpetal_zmtl(inzmtl, outzmtl)

        #- Remove write access; force=True needed for re-re-archiving
        err = freezedir(archivedir, dryrun=dryrun, force=True)
        if err != 0:
            log.error(f'problem removing write access from {archivedir}')

    return err


def freezedir(path, dryrun=False, force=False):
    """
    Remove write permission from path unless dryrun

    Args:
        path (str): path to directory to remove

    Options:
        dryrun: if True, print info but don't actually remove write access
        force: if True, rerun chmod even if directory already appears frozen

    Returns non-zero error code upon failure (not an exception)

    Note: dryrun overrides force
    """
    log = get_logger()
    err = 0
    if not os.path.isdir(path):
        log.error(f'Not a directory; skipping {path}')
        err = 1
    elif not os.access(path, os.W_OK) and not force:
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
    parser.add_argument('--badpetals', type=str,
            help = 'Comma separated list of known bad petals not in production but archived from earlier prod')
    parser.add_argument('--specstatus', type=str,
            help='tiles-specstatus.ecsv file to use; archive tiles with '
            'ZDONE=false and QA=true; update to ZDONE=true and set ARCHIVEDATE')
    parser.add_argument('--dry-run', action='store_true',
            help='print what directories to archive, without archiving them')
    parser.add_argument('--only-tiles', action='store_true',
            help="archive only tiles dirs; don't freeze exposures, preproc")
    parser.add_argument('--rearchive', action='store_true',
            help="Rearchive requested tiles, even if ZDONE=true")

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

    if args.badpetals is None:
        args.badpetals = list()
    else:
        args.badpetals = [int(petal) for petal in args.badpetals.split(',')]

    if args.specstatus is None:
        if os.path.exists('./tiles-specstatus.ecsv'):
            log.info('Using tiles-specstatus.ecsv in current directory')
            args.specstatus = os.path.abspath('./tiles-specstatus.ecsv')
        else:
            log.info('Using default surveyops/ops/tiles-specstatus.ecsv')
            args.specstatus = os.path.expandvars(
                '$DESI_SURVEYOPS/ops/tiles-specstatus.ecsv')

    if not os.path.exists(args.specstatus):
        log.critical(f'Missing {args.specstatus}')
        sys.exit(1)

    log.info(f'Reading tiles from {args.specstatus}')
    if not args.dry_run and not os.access(args.specstatus, os.W_OK):
        filename = os.path.basename(args.specstatus)
        log.critical(f'Need write access to {filename} to update ZDONE and ARCHIVEDATE')
        sys.exit(1)

    #- before changing directories, convert paths to absolute
    args.specstatus = os.path.abspath(args.specstatus)

    log.info(f'Archiving tiles in {reduxdir}')
    os.chdir(reduxdir)

    tiles = Table.read(args.specstatus)
    log.info('Archiving tiles with SURVEY=main|sv3, QA=good')
    keep = (tiles['SURVEY'] == 'main') | (tiles['SURVEY'] == 'sv3')
    keep &= (tiles['QA'] == 'good')
    if not args.rearchive:
        log.info('Requiring ZDONE=false (use --rearchive to override)')
        keep &= (tiles['ZDONE'] == 'false')

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
        tile_err = archivetile(tiledir, archivedir, badpetals=args.badpetals, dryrun=args.dry_run)

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
        if len(failed_archive_tiles) < len(archivetiles):
            tiles.write(args.specstatus, overwrite=True)
            log.info(f'Updated {args.specstatus} with new ZDONE and ARCHIVEDATE')
            log.info('Remember to svn commit that file now')
        else:
            log.error(f'All tiles failed archiving; not updating {args.specstatus}')

    if len(failed_archive_tiles) > 0:
        log.critical(f'Some tiles failed archiving: {failed_archive_tiles}')
        sys.exit(1)

    return 0

