
import os, sys
import argparse

from desiutil.log import get_logger

from desispec.io import findfile
from desispec.io.util import parse_cameras, decode_camword, relsymlink

calibnight_prefixes = ('badcolumns','biasnight','fiberflatnight','psfnight','ctecorrnight')
_prefixstr = ','.join(calibnight_prefixes)

def derive_include_exclude(input_include, input_exclude):
    """
    Take the defined include or exclude list and produces the

    Args:
        input_include, str or None. Comma separated prefixes of calibnight
            files to create links.
        input_exclude, str or None. Comma separated prefixes of calibnight
            files to exclude from links'
    Returns:
        include, set. Complete set of strings of calibration files that
            should be linked.
        exclude, set. Complete set of strings of calibration files that
            should not be linked.
    """
    if input_include is not None and input_exclude is not None:
        raise ValueError('include and exclude cannot both be defined: '
                         + f'{input_include=}, {input_exclude=}')

    include, exclude = None, None
    if input_include is not None:
        include = set([x.strip() for x in input_include.split(',')])
    else:
        include = set(calibnight_prefixes)

    # --include and --exclude are mutually exclusive, so if --exclude is set,
    # then --include is default value; remove --exclude options from that
    if input_exclude is not None:
        exclude = set([x.strip() for x in input_exclude.split(',')])
        extras = exclude - include
        if len(extras) > 0:
            raise ValueError(f'--exclude has values not found in default --include: {extras}')

        include -= exclude

    ## Now include is completely consistent with inputs. Next let's make exclude
    ## also consistent
    exclude = set(calibnight_prefixes) - include
    return include, exclude

def parse(options=None):
    """parse options from sys.argv or input list of options"""
    p = argparse.ArgumentParser()
    p.add_argument('--refnight', type=int, required=True,
                   help='Reference night with calibs')
    p.add_argument('--newnight', type=int, required=True,
                   help='New night without calibs, to link to refnight')
    p.add_argument('-c', '--cameras', type=str, default='a0123456789',
                   help='Camword of cameras to link, e.g. a01 or b0,r1,z2 [default a0123456789]')

    inout = p.add_mutually_exclusive_group()
    inout.add_argument('--include', type=str,
                   help=f'comma separated prefixes of calibnight files to create links [default "{_prefixstr}"]')
    inout.add_argument('--exclude', type=str,
                   help='comma separated prefixes of calibnight files to exclude from links')

    p.add_argument('--dryrun', action='store_true',
                   help="dry run; don't actually create links")

    args = p.parse_args(options)

    args.include, args.exclude = derive_include_exclude(args.include, args.exclude)
    args.cameras = decode_camword(parse_cameras(args.cameras, loglevel='WARNING'))

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

        #- special case: ctecorr is per-night, not per-camera,
        #- and allow it to be missing (revisit after PR #2163 is merged)
        if prefix == 'ctecorrnight':
            reffile = findfile(prefix, night=args.refnight)
            newfile = findfile(prefix, night=args.newnight)

            if not os.path.exists(reffile):
                #- warn, but not fatal, i.e. proceed with other links
                log.warning(f'Skipping missing reference {reffile}')
            else:
                if check_link(newfile, reffile):
                    reffiles.append(reffile)
                    newfiles.append(newfile)
                else:
                    num_errors += 1

            #- done with ctecorr, move on to next prefix
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
            if os.path.islink(newfile):
                os.remove(newfile)

            log.info(f'Linking {newfile} -> {relpath}')
            relsymlink(reffile, newfile)


