"""
desispec.scripts.makezmtl
=========================

"""
import os
import sys
import traceback
from desispec.zmtl import get_qn_model_fname, load_qn_model
from desispec.zmtl import get_sq_model_fname, load_sq_model
from desispec.zmtl import create_zmtl, tmark

from desispec.io import specprod_root

# ADM set up the DESI default logger.
from desiutil.log import get_logger



# EBL Handy tileid/nightid combos for testing
#    -  1, 20210406, Note: missing petal 7, good test for skipping bad petal.
#    - 84, 20210410
#    - 85, 20210412
# For the VI'd tiles using the processed r_depth_ebvair of ~1000
#    TILEIDs: 80605, 80607, 80609, 80620, 80622
#    NIGHTID: All use 20210302 (when I made those files). Date is
#             meaningless in this case, just there due to filename
#             format requirements.

import argparse

def parse(options=None):
    # SB default output to $DESI_SPECTRO_REDUX/$SPECPROD
    reduxdir = specprod_root()

    # EBL retrieve the file names for the QuasarNP and SQUEzE models
    #     from environment variables.
    qnmodel_fname = get_qn_model_fname()
    sqmodel_fname = get_sq_model_fname()

    # ADM by default we're working with tiles/cumulative.
    sub_dir = os.path.join('tiles', 'cumulative')

    description = 'Create a zmtl file for LyA decisions. Pass (at least) both of '
    description += '--input_file and --output_file OR both of --tile and --night'

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-in', '--input_file',
                     default=None,
                     help='Full path to an input redrock file. Overrides --tile,     \
                     --night, --petal, --input_dir, --output_dir, --sub_dir')
    parser.add_argument('-out', '--output_file',
                     default=None,
                     help='Full path to an output zmtl file. Overrides --tile,     \
                     --night, --petal, --input_dir, --output_dir, --sub_dir')
    parser.add_argument('-t', '--tile',
                     default=None,
                     help='TILEID(s) of tiles to process. Pass a comma-separated   \
                     list (e.g. 1 for TILE 1 or "1,84,85" for TILEs 1, 84, 85)')
    parser.add_argument('-n', '--night',
                     default=None,
                     help='NIGHTID(s) of tiles to process in YYYYMMDD. Pass a comma \
                     -separated list (e.g. 20210406 or "20210406,20210410,20210412")\
                     which must correspond to the TILEIDs')
    parser.add_argument('-p', '--petal',
                     default="0,1,2,3,4,5,6,7,8,9",
                     help='Petals to run. Pass comma-separated integers            \
                     (e.g. "1" for petal 1, "3,5,8" for petals 3, 5 and 8)         \
                     Defaults to 0,1,2,3,4,5,6,7,8,9 (all petals)')
    parser.add_argument('-i', '--input_dir',
                     metavar='REDUX_DIR', default=reduxdir,
                     help='The root input directory to use for DESI spectro files. \
                     Defaults to {}'.format(reduxdir))
    parser.add_argument('-o', '--output_dir',
                     metavar='REDUX_DIR', default=reduxdir,
                     help='The root output directory to use for zmtl files.        \
                     Defaults to {}'.format(reduxdir))
    parser.add_argument('-sd', '--sub_dir',
                     default=sub_dir,
                     help='The sub-directories that house redrock files.              \
                     Defaults to {}'.format(sub_dir))
    parser.add_argument('-noq', '--no_quasarnp',
                     action='store_true',
                     help='QuasarNP is added by default. Send this to NOT add it.')
    parser.add_argument('-s', '--add_squeze',
                     action='store_true',
                     help='Add SQUEzE data to zmtl file.')
    parser.add_argument('-m', '--add_mgii',
                     action='store_true',
                     help='Add MgII absorption data to zmtl file.')
    parser.add_argument('-zc', '--add_zcomb',
                     action='store_true',
                     help='Add combined redshift information.')
    parser.add_argument('-qn', '--qn_model_file',
                     metavar='QN_MODEL_FILE',
                     default=qnmodel_fname,
                     help='The full path and filename for the QuasarNP model        \
                     file. Defaults to {}'.format(qnmodel_fname))
    parser.add_argument('-sq', '--sq_model_file',
                     metavar='SQ_MODEL_FILE',
                     default=sqmodel_fname,
                     help='The full path and filename for the SQUEzE model          \
                     file. Defaults to {}'.format(sqmodel_fname))

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args


def main(args=None):
    if not isinstance(args, argparse.Namespace):
        args = parse(args)

    log = get_logger()

    # ADM by default we're working with tiles/cumulative.
    sub_dir = os.path.join('tiles', 'cumulative')

    add_quasarnp = not(args.no_quasarnp)

    # EBL Load the QuasarNP model file if QuasarNP is activated.
    if add_quasarnp:
        tmark('    Loading QuasarNP Model file and lines of interest')
        qnp_model, qnp_lines, qnp_lines_bal, qnp_grid = load_qn_model(args.qn_model_file)
        tmark('      QNP model file loaded')
    else:
        qnp_model, qnp_lines, qnp_lines_bal, qnp_grid = None, None, None, None

    if args.add_squeze:
        tmark('    Loading SQUEzE Model file')
        sq_model = load_sq_model(args.sq_model_file)
        tmark('      Model file loaded')
    else:
        sq_model = None

    # ADM if input_file and output_file were added, override TILE/NIGHT.
    if args.input_file is not None or args.output_file is not None:
        if args.input_file is None or args.output_file is None:
            msg = "if one of --input_file or --output_file is passed then"
            msg += " the other must be too!!!"
            log.critical(msg)
            raise IOError(msg)
        tile=None
        create_zmtl(args.input_file, args.output_file, tile=tile,
                    qn_flag=add_quasarnp, qnp_model=qnp_model,
                    qnp_model_file=args.qn_model_file, qnp_lines=qnp_lines,
                    qnp_grid=qnp_grid,
                    qnp_lines_bal=qnp_lines_bal, sq_flag=args.add_squeze,
                    squeze_model=sq_model, squeze_model_file=args.sq_model_file,
                    abs_flag=args.add_mgii, zcomb_flag=args.add_zcomb)
    else:
        # ADM add, e.g., tiles/cumulative to the directory structure.
        input_dir = os.path.join(args.input_dir, args.sub_dir)
        output_dir = os.path.join(args.output_dir, args.sub_dir)

        tiles = [int(t) for t in args.tile.split(',')]
        nights = [int(n) for n in args.night.split(',')]
        petals = [int(p) for p in args.petal.split(',')]

        numerr = 0
        for tile, night in zip(tiles, nights):
            for pnum in petals:
                log.info("processing TILEID={}, NIGHTID={}, petal={}".format(
                    tile, night, pnum))
                try:
                    create_zmtl(input_dir, output_dir, tile=tile,
                            night=night, petal_num=pnum, qn_flag=add_quasarnp,
                            qnp_model=qnp_model, qnp_model_file=args.qn_model_file,
                            qnp_lines=qnp_lines, qnp_lines_bal=qnp_lines_bal,
                            sq_flag=args.add_squeze, squeze_model=sq_model,
                            squeze_model_file=args.sq_model_file,
                            abs_flag=args.add_mgii, zcomb_flag=args.add_zcomb)
                except Exception as err:
                    numerr += 1
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
                    print(''.join(lines))
                    log.error(f'Tile {tile} night {night} petal {pnum} failed; continuing')

        return numerr

    return 0
