#!/usr/bin/env python

"""
Combine individual redrock files into individual zcatalogs for all of a --group type

Anthony Kremin
LBNL

updated Fall 2024
"""

from __future__ import absolute_import, division, print_function

import sys, os, glob, time
import argparse
from astropy.table import Table
import numpy as np

from desispec.parallel import stdouterr_redirected
from desispec.util import runcmd
from desiutil.log import get_logger, DEBUG
from desispec.io.meta import findfile
from desispec.io import specprod_root
from desispec.scripts import zcatalog as zcatalog_script
from desispec.zcatalog import create_summary_catalog_wrapper

def parse(options=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-g", "--group", type=str,
            help="Add columns specific to this spectral grouping "
                 "e.g. pernight, perexp, cumulative, healpix")

    parser.add_argument("--cat-version",type=str, default=None,
            help="The version number of the output catalogs")
    parser.add_argument("-i", "--indir",  type=str, default=None,
            help="input directory")
    parser.add_argument("-o", "--outdir", type=str, default=None,
            help="output directory")
    parser.add_argument("--header", type=str, default=None,
            help="KEYWORD=VALUE entries to add to the output header")
    # parser.add_argument("--survey", type=str, nargs="*", default=None,
    #         help="DESI survey, e.g. sv1, sv3, main")

    parser.add_argument('--nproc', type=int, default=1,
            help="Number of multiprocessing processes to use")

    parser.add_argument("--minimal", action='store_true',
            help="only include minimal output columns")
    parser.add_argument('--patch-missing-ivar-w12', action='store_true',
            help="Use target files to patch missing FLUX_IVAR_W1/W2 values")
    parser.add_argument('--recoadd-fibermap', action='store_true',
            help="Re-coadd FIBERMAP from spectra files")
    parser.add_argument('--add-units', action='store_true',
            help="Add units to output catalog from desidatamodel "
                 "column descriptions")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Set log level to DEBUG.")
    args = parser.parse_args(options)

    return args


def main(args=None):
    if not isinstance(args, argparse.Namespace):
        args = parse(options=args)

    if args.verbose:
        log=get_logger(DEBUG)
    else:
        log=get_logger()

    ## Ensure we know where to save the files
    if args.outdir is None and args.cat_version is None:
        log.critical(f"Either --outdir or --cat-version must be specifiied. Exiting")
        return 1

    ## If adding units, check dependencies before doing a lot of work
    if args.add_units:
        try:
            import desidatamodel
        except ImportError:
            log.critical('Unable to import desidatamodel, required to add units (try "module load desidatamodel" first)')
            return 1

    ## Define filetype based on healpix vs not healpix
    if args.group == 'healpix':
        ftype = 'zcat_hp'
    else:
        ftype = 'zcat_tile'

    ## Ensure input directory exists
    if args.indir is not None and not os.path.exists(args.indir):
        log.critical(f"Input directory {args.indir} does not exist.")
        return 1

    ## If outdir is None, and cat_version is None, raise an error
    if args.outdir is None and args.cat_version is None:
        log.critical(f"Either --outdir or --cat-version must be specified.")
        return 1

    if args.outdir is None:
        args.outdir = os.path.dirname(findfile(ftype,
                                                  version=args.cat_version,
                                                  groupname=args.group))
    if not os.path.exists(args.outdir):
        log.info(f"Output directory {args.outdir} does not exist, creating now.")
        os.makedirs(args.outdir)

    ## Load the tiles file to know what to run
    tilesfile = findfile('tiles')
    if os.path.exists(tilesfile):
        tiles_tab = Table.read(tilesfile, format='ascii.csv')
    else:
        log.warning(f'Tiles file {tilesfile} does not exist. Trying CSV instead.')
        tilesfile = findfile('tiles_csv')
        if os.path.exists(tilesfile):
            tiles_tab = Table.read(tilesfile, format='ascii.csv')
        else:
            log.critical(f"Could not find a valid tiles file!")
            return 1

    # ## if user didn't specify survey or surveys, run over all surveys in tiles file
    # if args.survey is None:
    #     args.survey = np.unique(tiles_tab['SURVEY'])

    ## Define the generic command to be run each time
    cmd = f'desi_zcatalog -g {args.group} --nproc={args.nproc}'
    if args.indir is not None:
        cmd += f" --indir='{args.indir}'"
    if args.header is not None:
        cmd += f" --header='{args.header}'"
    for argument, argval in [('-v', args.verbose),
                             ("--minimal", args.minimal),
                             ('--patch-missing-ivar-w12', args.patch_missing_ivar_w12),
                             ('--recoadd-fibermap', args.recoadd_fibermap),
                             ('--add-units', args.add_units)]:
        if argval:
            cmd += f" {argument}"

    error_count = 0
    survey_program_outfiles = []
    #for survey in args.survey:
    for survey in np.unique(tiles_tab['SURVEY']):
        for program in np.unique(tiles_tab['PROGRAM'][tiles_tab['SURVEY']==survey]):
            out_fname = os.path.basename(findfile(ftype, groupname=args.group,
                                             survey=survey,
                                             faprogram=program))
            outfile = os.path.join(args.outdir, out_fname)
            current_cmd = cmd + f" --survey={survey} --program={program} --outfile={outfile}"
            cmdargs = current_cmd.split()[1:]
            log_fname = os.path.splitext(out_fname)[0] + '.log'
            outlog = os.path.join(args.outdir, 'logs', log_fname)
            with stdouterr_redirected(outlog):
                result, success = runcmd(zcatalog_script.main, args=cmdargs, inputs=[], outputs=[outfile,])

            survey_program_outfiles.append(outfile)
            if not success:
                error_count += 1
                log.warning(
                    f"Failed to produce output: {outfile}, see {outlog}")
            else:
                log.info(f"Success in producing output: {outfile}")

    ## If all runs above were successful and running cumulative or healpix,
    ## then run zall as well
    if error_count == 0 and len(survey_program_outfiles) > 0 and args.group in ['healpix', 'cumulative']:
        """
        create_summary_catalog(specgroup, indir=None, specprod=None,
                               all_columns=True, columns_list=None,
                               output_filename=None)
        """
        out_fname = os.path.basename(findfile(ftype.replace('zcat', 'zall')))
        outfile = os.path.join(args.outdir, out_fname)
        if args.group == 'healpix':
            specgroup = 'zpix'
        else:
            specgroup = 'ztile'
        kwargs = {'specgroup': specgroup, 'indir': args.outdir, 'output_filename': outfile}
        log_fname = os.path.splitext(out_fname)[0] + '.log'
        outlog = os.path.join(args.outdir, 'logs', log_fname)
        with stdouterr_redirected(outlog):
            result, success = runcmd(create_summary_catalog_wrapper, args=[kwargs],
                                     inputs=survey_program_outfiles, outputs=[outfile])
        if not success:
            error_count += 1
            log.warning(f"Failed to produce output: {outfile}, see {outlog}")
        else:
            log.info(f"Success for job producing output: {outfile}")
    if error_count == 0:
        log.info(f"SUCCESS: All done at {time.asctime()}")
    else:
        log.info(f"{error_count} FAILURES: All done at {time.asctime()}")

