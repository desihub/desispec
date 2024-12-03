#!/usr/bin/env python

"""
Combine individual redrock files into individual zcatalogs for all of a --group type

Anthony Kremin
LBNL

updated Fall 2024
"""

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
from desispec.zcatalog import create_summary_catalog


def parse(options=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-g", "--group", type=str, required=True,
            help="Add columns specific to this spectral grouping "
                 "e.g. pernight, perexp, cumulative, healpix")
    parser.add_argument('-V', "--cat-version",type=str, required=True,
            help="The version number of the output catalogs")

    parser.add_argument("-i", "--indir",  type=str, default=None,
            help="Input directory")
    parser.add_argument("-o", "--outdir", type=str, default=None,
            help="Output directory without version number included.")
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
    parser.add_argument('--do-not-add-units', action='store_true',
            help="Set if you do not want to add units to output catalog from desidatamodel "
                 "column descriptions")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Set log level to DEBUG.")
    args = parser.parse_args(options)

    return args

def _create_summary_catalog_wrapper(kwargs):
    """
    Trivial wrapper around create_summary_catalog that takes a dict
    and passes the key-value pairs to create_summary_catalog as keyword arguments
    """
    return create_summary_catalog(**kwargs)

def main(args=None):
    if not isinstance(args, argparse.Namespace):
        args = parse(options=args)

    if args.verbose:
        log=get_logger(DEBUG)
    else:
        log=get_logger()

    ## If adding units, check dependencies before doing a lot of work
    if not args.do_not_add_units:
        try:
            import desidatamodel
        except ImportError:
            log.critical('Unable to import desidatamodel, required to add units.'
                         + ' Try "module load desidatamodel" first or use '
                         + '--do-not-add-units')
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

    if args.outdir is None:
        ## since we only care about the directory path, we can
        ## use dummy values for survey and faprogram
        args.outdir = os.path.dirname(findfile(ftype,
                                               version=args.cat_version,
                                               groupname=args.group,
                                               survey='dummy', faprogram='dummy'))
    else:
        args.outdir = os.path.join(args.outdir, args.cat_version)
    log.info(f"Writing outputs to the following directory: {args.outdir}")

    logdir = os.path.join(args.outdir, 'logs')
    if not os.path.exists(logdir):
        log.info(f"Output log directory {logdir} does not exist, creating now.")
        os.makedirs(logdir)

    ## Load the tiles file to know what to run
    tilesfile = findfile('tiles')
    if os.path.exists(tilesfile):
        tiles_tab = Table.read(tilesfile, format='fits')
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
    if args.header is not None:
        cmd += f" --header='{args.header}'"
    for argument, argval in [('-v', args.verbose),
                             ("--minimal", args.minimal),
                             ('--patch-missing-ivar-w12', args.patch_missing_ivar_w12),
                             ('--recoadd-fibermap', args.recoadd_fibermap),
                             ('--do-not-add-units', args.do_not_add_units)]:
        if argval:
            cmd += f" {argument}"

    error_count = 0
    survey_program_outfiles = []
    #for survey in args.survey:
    for survey in np.unique(tiles_tab['SURVEY']):
        for program in np.unique(tiles_tab['PROGRAM'][tiles_tab['SURVEY']==survey]):
            ## note that the version here isn't actually used because we only
            ## take basename of findfile output
            out_fname = os.path.basename(findfile(ftype, groupname=args.group,
                                                  survey=survey, faprogram=program,
                                                  version=args.cat_version))
            outfile = os.path.join(args.outdir, out_fname)

            ## update the base command with the program and survey information
            current_cmd = cmd + f" --survey={survey} --program={program} --outfile={outfile}"
            if args.indir is not None:
                ## the healpix path includes survey and program and the
                ## zcatalog assumes healpix indir has them included
                if args.group == 'healpix':
                    current_cmd += f" --indir={args.indir}/{survey}/{program}"
                else:
                    current_cmd += f" --indir={args.indir}"
            cmdargs = current_cmd.split()[1:]

            log.info(f"Running {survey=}, {program=}, to produce {outfile}")
            ## create a log file with the same name as the output except *.log
            log_fname = os.path.splitext(out_fname)[0] + '.log'
            outlog = os.path.join(args.outdir, 'logs', log_fname)

            ## redirect stdout and stderr to the log file and only run if
            ## outfile doesn't exist
            with stdouterr_redirected(outlog):
                result, success = runcmd(zcatalog_script.main, args=cmdargs, inputs=[], outputs=[outfile,])

            ## Save the outfile so we know what infiles to expect for zall
            survey_program_outfiles.append(outfile)

            ## Track the number of failures and report result
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
        ## note that the version here isn't actually used because we only
        ## take basename of findfile output
        out_fname = os.path.basename(findfile(ftype.replace('zcat', 'zall'),
                                              groupname=args.group,
                                              version=args.cat_version))
        outfile = os.path.join(args.outdir, out_fname)
        ## summary catalog code calls these zpix and ztile instead
        if args.group == 'healpix':
            specgroup = 'zpix'
        else:
            specgroup = 'ztile'
        ## input here is the output file of the previous loop
        kwargs = {'specgroup': specgroup, 'indir': args.outdir, 'output_filename': outfile}

        log.info(f"Running zall generation to produce {outfile}")
        log_fname = os.path.splitext(out_fname)[0] + '.log'
        outlog = os.path.join(args.outdir, 'logs', log_fname)
        ## Redirect logging to seperate file and only run of all files output in the last
        ## step exist
        with stdouterr_redirected(outlog):
            result, success = runcmd(_create_summary_catalog_wrapper, args=kwargs,
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

