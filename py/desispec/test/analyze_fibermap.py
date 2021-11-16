# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
desispec.test.analyze_fibermap
==============================

Test desispec.io.fibermap.assemble_fibermap() on many nights.

Guidelines for picking exposures to test:

1. Test nights/exposures that are already specifically special-cased in
   assemble_fibermap()
2. Otherwise, just test one exposure per night that appears in ``everest``.

Invocation::

    python -m desispec.test.analyze_fibermap --help

"""
import glob
import os
import sys
import random
from argparse import ArgumentParser
import numpy as np
from astropy.table import Table
from astropy.io import fits
from desispec.io.fibermap import assemble_fibermap, _set_fibermap_columns
from desispec.io.meta import faflavor2program
from desiutil.log import get_logger, DEBUG
# from desiutil.iers import freeze_iers


def main():
    """Entry point for command-line scripts.

    Returns
    -------
    :class:`int`
        An integer suitable for passing to :func:`sys.exit`.
    """
    #
    # Get options
    #
    parser = ArgumentParser(description=__doc__.split('\n')[4])
    parser.add_argument('-a', '--analyze', action='store_true', help='Analyze the new fibermap files.')
    parser.add_argument('-g', '--generate', action='store_true', help='Generate the new fibermap files.')
    parser.add_argument('-s', '--seed', type=int, action='store', default=20201220, metavar='SEED', help='Set random seed (default = %(default)s).')
    # parser.add_argument('-v', '--verbose', action='store_true', help='Set log level to DEBUG.')
    options = parser.parse_args()
    log = get_logger()
    # if options.verbose:
    #     log = get_logger(DEBUG)
    # else:
    #     log = get_logger()
    if not options.analyze and not options.generate:
        log.error("You must select --generate or --analyze in order for this program to do anything!")
        return 1
    #
    # Setup and print environment.
    #
    log.debug("random.seed(%d)", options.seed)
    random.seed(options.seed)
    log.debug('DESISPEC=%s', os.environ['DESISPEC'])
    log.debug('SPECPROD=%s', os.environ['SPECPROD'])
    preproc = os.path.join(os.environ['DESI_SPECTRO_REDUX'], os.environ['SPECPROD'], 'preproc')
    analyze_columns = _set_fibermap_columns()
    #
    # Load exposures catalog.
    #
    exposures_file = os.path.join(os.environ['DESI_SPECTRO_REDUX'],
                                  os.environ['SPECPROD'],
                                  "exposures-{SPECPROD}.fits".format(**os.environ))
    exposures = Table.read(exposures_file, 'EXPOSURES')
    program = faflavor2program(exposures['FAFLAVOR'])
    #
    # Find nights.
    #
    log.debug("nights = list(map(int, os.listdir('%s')))", preproc)
    nights = list(map(int, os.listdir(preproc)))
    expids = {
              20201214: 67678,
              20201220: 69029,
              20210110: 71721,
              20210224: 77902,
              20210402: 83144,
             }
    #
    # Run assemble_fibermap on one exposure per night.
    #
    for night in nights:
        try:
            expid = expids[night]
        except KeyError:
            try:
                expid = int(os.path.basename(os.path.dirname(random.choice(glob.glob(os.path.join(preproc, '{0:08d}'.format(night), '????????', 'fibermap-????????.fits'))))))
            except IndexError:
                log.error("Night %08d contains no fibermap files!", night)
                continue
            expids[night] = expid
        try:
            exposures_row = np.nonzero((exposures['NIGHT'] == night) & (exposures['EXPID'] == expid))[0][0]
            survey_program = "{0}-{1}".format(exposures['SURVEY'][exposures_row], program[exposures_row])
        except IndexError:
            log.error("Could not find %08d/%08d in exposures catalog!", night, expid)
            exposures_row = None
            survey_program = "UNKNOWN"
        outfile = os.path.join(os.environ['DESI_SPECTRO_REDUX'],
                               os.environ['USER'],
                               'preproc',
                               '{0:08d}'.format(night),
                               '{0:08d}'.format(expid),
                               'fibermap-{0:08d}.fits'.format(expid))
        if options.generate:
            log.info("Generating %s, survey-program = %s.", outfile, survey_program)
            log.debug("os.makedirs('%s', exist_ok=True)", os.path.dirname(outfile))
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            log.debug("fibermap = assemble_fibermap(%d, %d)", night, expid)
            fibermap = assemble_fibermap(night, expid)
            tmpfile = outfile+'.tmp'
            log.debug("fibermap.writeto('%s', output_verify='fix+warn, overwrite=True, checksum=True')", tmpfile)
            fibermap.writeto(tmpfile, output_verify='fix+warn', overwrite=True, checksum=True)
            log.debug("os.rename('%s', '%s')", tmpfile, outfile)
            os.rename(tmpfile, outfile)
        if options.analyze:
            log.info("Analyzing %s, survey-program = %s.", outfile, survey_program)
            fibermap = Table.read(outfile, "FIBERMAP")
            # fibermap_header = fits.getheader(outfile, 'FIBERMAP')
            survey = 'main'
            for sv in (1,2,3):
                if f"SV{sv:d}_DESI_TARGET" in fibermap.colnames:
                    survey = f"sv{sv:d}"
                    break
            if 'CMX_TARGET' in fibermap.colnames:
                survey = 'cmx'
            if fibermap.colnames == analyze_columns[survey]:
                log.debug("Column names match %s standard for %s.", survey, outfile)
            else:
                for i in range(len(fibermap.colnames)):
                    if fibermap.colnames[i] != analyze_columns[survey][i]:
                        log.error('FIBERMAP table column mismatch at index %d ("%s" != "%s")!', i, fibermap.colnames[i], analyze_columns[survey][i])
                        break
    return 0


if __name__ == '__main__':
    # freeze_iers()
    sys.exit(main())
