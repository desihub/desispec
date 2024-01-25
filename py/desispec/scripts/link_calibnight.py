
import os, sys
import argparse

from desiutil.log import get_logger

from desispec.io import findfile
from desispec.io.util import decode_camword, relsymlink

def parse(options=None):
    """parse options from sys.argv or input list of options"""
    p = argparse.ArgumentParser()
    p.add_argument('--refnight', type=int, required=True,
                   help='Reference night with calibs')
    p.add_argument('--newnight', type=int, required=True,
                   help='New night without calibs, to link to refnight')
    p.add_argument('-c', '--cameras', type=str, default='a0123456789',
                   help='Camword of cameras to link, e.g. a0123b6r7z8')
    p.add_argument('--include', type=str, 
                   default='badcolumns,biasnight,fiberflatnight,psfnight,ctecorr',
                   help='prefixes of types of calibnight files to create links')
    p.add_argument('--exclude', type=str, 
                   help='prefixes of types of calibnight files to exclude from links')
    p.add_argument('--dryrun', action='store_true',
                   help="dry run; don't actually create links")

    args = p.parse_args(options)

    if args.exclude is None:
        args.exclude = set()
    else:
        args.exclude = set(args.exclude.split(','))

    args.include = set(args.include.split(',')) - args.exclude

    args.cameras = decode_camword(args.cameras)

    return args

def check_link(newfile, reffile):
    """Check if link from newfile -> reffile is ok

    Args:
        newfile (str): path to new link location
        reffile (str): apth to reference file to link to

    Returns True if everything is ok.

    Returns False if refile is missing, or newfile already exists
    and isn't a link.

    Logs a warning if newfile exists as a link, but different than reffile.
    """
    log = get_logger()
    if not os.path.exists(reffile):
        log.error(f'Missing {reffile}')
        return False

    #- notes on islink vs. exists:
    #-   a bad link passes islink but not exists;
    #-   a good link passes both;
    #-   a non-link file passes exists but not islink
    if os.path.islink(newfile):
        #- if link, check if it is a different destination
        orig_link = os.readlink(newfile)
        new_link = os.path.relpath(reffile, os.path.dirname(newfile))
        if orig_link != new_link:
            log.error(f'Pre-existing link {newfile} -> {orig_link} != {new_link}')
            return False

    elif os.path.exists(newfile):
        #- pre-existing but not a link is bad
        log.error(f"Pre-existing non-link {newfile}; won't override")
        return False

    return True


def main(args=None):

    log = get_logger()

    if not isinstance(args, argparse.Namespace):
        args = parse(args)

    if len(args.include) == 0:
        log.critical(f'No prefixes to include? check --include and --exclude options')
        sys.exit(1)

    #- check for any problems before creating any links
    num_errors = 0
    reffiles = list()
    newfiles = list()
    for prefix in args.include:
        if prefix == 'ctecorr':
            log.warning('ctecorr not yet supported; see PR #2163')
            continue

        for camera in args.cameras:
            reffile = findfile(prefix, night=args.refnight, camera=camera)
            newfile = findfile(prefix, night=args.newnight, camera=camera)

            if check_link(newfile, reffile):
                reffiles.append(reffile)
                newfiles.append(newfile)
            else:
                num_errors += 1

            #- ctecorr is per-night, not per-camera per-night so we only
            #- have to check/link it once. Break out of camera loop and
            #- continuewith other prefixes.
            if prefix == 'ctecorr':
                break

    if num_errors > 0:
        log.critical(f'{num_errors} errors; not proceeding with making links')
        sys.exit(1)

    log.info(f'Linking {args.include} files from {args.newnight} -> {args.refnight}')
    if args.dryrun:
        log.info('Dry run: not actually making links')
    else:
        newdir = os.path.dirname(newfiles[0])
        os.makedirs(newdir, exist_ok=True)

    for reffile, newfile in zip(reffiles, newfiles):
        relpath = os.path.relpath(reffile, os.path.dirname(newfile))
        if args.dryrun:
            log.info(f'Dry run: would create link {newfile} -> {relpath}')
        else:
            #- pre-flight checks confirmed this is ok
            if os.path.islink(newfile) or os.path.exists(newfile):
                os.remove(newfile)

            log.debug(f'Linking {newfile} -> {relpath}')
            relsymlink(reffile, newfile)


