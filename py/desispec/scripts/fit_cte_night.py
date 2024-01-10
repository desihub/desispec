"""
desispec.scripts.fit_cte_night
==============================

"""

from __future__ import absolute_import, division

import argparse
from desiutil.log import get_logger
from desispec.correct_cte import fit_cte_night




def parse(options=None):
    parser = argparse.ArgumentParser(description="Compute the sky model.")

    parser.add_argument('-n','--night', type = int, default = None, required=True,
                        help = 'night')
    parser.add_argument('-c','--camera', type = str, default = None, required=True,
                        help = 'camera')
    parser.add_argument('-o','--outfile', type = str, default = None, required=True,
                        help = 'path of output cvs table')
    args = parser.parse_args(options)
    return args


def main(args=None) :

    if not isinstance(args, argparse.Namespace):
        args = parse(args)

    log=get_logger()
    table=fit_cte_night(night=args.night,camera=args.camera)
    table.write(args.outfile,overwrite=True)
    log.info("successfully wrote %s"%args.outfile)
