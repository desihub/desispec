#!/usr/bin/env python
#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
Read fibermaps and zbest files to generate QA related to redshifts
 using the 'true' values
"""

import argparse
import os.path

import yaml
import pdb
from matplotlib.backends.backend_pdf import PdfPages

import desispec.io
from desispec.log import get_logger, DEBUG

from desisim.spec_qa import redshifts as dsqa_z

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', action = 'store_true',
        help = 'Provide verbose reporting of progress.')
    parser.add_argument('--night', type=str, default = None, metavar = 'YYYYMMDD',
        help = 'Night to process in the format YYYYMMDD')
    parser.add_argument('--specprod', type = str, default = None, metavar = 'PATH',
                        help = 'Override default path ($DESI_SPECTRO_REDUX/$PRODNAME) to processed data.')
    parser.add_argument('--qafile', type = str, default = None, required=False,
                        help = 'path of QA file.')
    parser.add_argument('--qafig', type=str, default=None, help = 'path of QA figure file')
    args = parser.parse_args()
    if args.verbose:
        log = get_logger(DEBUG)
    else:
        log = get_logger()

    if args.night is None:
        log.critical('Missing required night argument.')
        return -1

    # Grab list of fibermap files
    fibermap_files = []
    zbest_files = []
    for exposure in desispec.io.get_exposures(args.night, specprod_dir = args.specprod):
        # Ignore exposures with no fibermap, assuming they are calibration data.
        fibermap_path = desispec.io.findfile(filetype = 'fibermap',night = args.night,
                                             expid = exposure, specprod_dir = args.specprod)
        if not os.path.exists(fibermap_path):
            log.debug('Skipping exposure %08d with no fibermap.' % exposure)
            continue
        else:
            fibermap_files.append(fibermap_path)
        # Search for zbest files
        fibermap_data = desispec.io.read_fibermap(fibermap_path)
        brick_names = set(fibermap_data['BRICKNAME'])
        for brick in brick_names:
            zbest_path=desispec.io.findfile('zbest',brickname=brick)
            if os.path.exists(zbest_path):
                zbest_files.append(zbest_path)
    zbest_files = list(set(zbest_files))

    # Load Table
    simz_tab = dsqa_z.load_z(fibermap_files, zbest_files)

    # Meta data
    meta = dict(SIMSPECV='9.999', PRODNAME=os.getenv('PRODNAME'))

    # Run stats
    if args.qafile is not None:
        summ_dict = dsqa_z.summ_stats(simz_tab)
        # Write yaml
        with open(args.qafile, 'w') as outfile:
            outfile.write(yaml.dump(meta))#, default_flow_style=True) )
            outfile.write(yaml.dump(summ_dict, default_flow_style=False) )

    if args.qafig is not None:
        pp = PdfPages(args.qafig)
        # Summ
        dsqa_z.summ_fig(simz_tab, summ_dict, meta, pp=pp)
        for objtype in ['ELG','LRG', 'QSO_T', 'QSO_L']:
            dsqa_z.obj_fig(simz_tab, objtype, summ_dict, pp=pp)
        # All done
        pp.close()

if __name__ == '__main__':
    main()
