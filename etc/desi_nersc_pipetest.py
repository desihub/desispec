#!/usr/bin/env python
#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-

"""Test pipeline processing.
"""

from __future__ import absolute_import, division, print_function

import sys
import os
import shutil
import argparse

from desispec.util import sprun


def create_args(args):
    optlist = [
        "root",
        "basis",
        "calib",
    ]
    varg = vars(args)
    opts = list()
    for k, v in varg.items():
        if k in optlist:
            if v is not None:
                opts.append("--{}".format(k))
                if not isinstance(v, bool):
                    opts.append(v)
    return opts


def chain_args(args):
    optlist = [
        "nersc",
        "nersc_full_queue",
    ]
    varg = vars(args)
    opts = list()
    for k, v in varg.items():
        if k in optlist:
            if k == "nersc_full_queue":
                opts.append("--nersc_queue")
                opts.append(v)
            else:
                if v is not None:
                    opts.append("--{}".format(k))
                    if not isinstance(v, bool):
                        opts.append(v)
    return opts


def dts_args(args):
    optlist = [
        "nersc",
        "nersc_nightly_queue",
        "nersc_nightly_maxnodes"
    ]
    varg = vars(args)
    opts = list()
    for k, v in varg.items():
        if k in optlist:
            if k == "nersc_nightly_maxnodes":
                opts.append("--nersc_maxnodes")
                opts.append(v)
            elif k == "nersc_nightly_queue":
                opts.append("--nersc_queue")
                opts.append(v)
            else:
                if v is not None:
                    opts.append("--{}".format(k))
                    if not isinstance(v, bool):
                        opts.append(v)
    return opts


def load_setup(path):
    vnames = [
        "DESI_ROOT",
        "DESI_BASIS_TEMPLATES",
        "DESI_CCD_CALIBRATION_DATA",
        "DESI_SPECTRO_DATA",
        "DESI_SPECTRO_REDUX",
        "SPECPROD",
        "DESI_SPECTRO_DB",
        "DESI_LOGLEVEL"
    ]
    ret, lines = sprun(["bash", "-c", "source {} && env".format(path)],
        capture=True)
    for line in lines:
        (key, _, value) = line.partition("=")
        if key in vnames:
            os.environ[key] = value
    return


def main():
    parser = argparse.ArgumentParser(description="Test DESI pipeline on a day"
        " of data")

    parser.add_argument("--input", required=True, default=None,
        help="Location of original input raw data")

    parser.add_argument("--output", required=True, default=None,
        help="Output working directory")

    parser.add_argument("--root", required=False, default=None,
        help="Override value for DESI_ROOT")

    parser.add_argument("--basis", required=False, default=None,
        help="Override value for DESI_BASIS_TEMPLATES")

    parser.add_argument("--calib", required=False, default=None,
        help="Override value for DESI_CCD_CALIBRATION_DATA")

    parser.add_argument("--nersc", required=False, default=None,
        help="Run on this NERSC system (edison | cori-haswell "
        "| cori-knl).  Default uses $NERSC_HOST")

    parser.add_argument("--nersc_full_queue", required=False, default="regular",
        help="Run the full test in this queue")

    parser.add_argument("--nersc_nightly_queue", required=False,
        default="regular", help="Run the nightly test in this queue")

    parser.add_argument("--nersc_nightly_maxnodes", required=False,
        default=None, help="The maximum number of nodes to use for nightly "
        "processing.  Default is the maximum nodes for the specified queue.")

    args = parser.parse_args()

    if args.nersc is None:
        if "NERSC_HOST" in os.environ:
            if os.environ["NERSC_HOST"] == "cori":
                args.nersc = "cori-haswell"
            else:
                args.nersc = os.environ["NERSC_HOST"]

    createopt = create_args(args)
    chainopt = chain_args(args)
    dtsopt = dts_args(args)

    # Data locations

    inputdir = os.path.abspath(args.input)
    if not os.path.isdir(inputdir):
        print("Input directory {} does not exist".format(inputdir))

    outputdir = os.path.abspath(args.output)
    if not os.path.isdir(outputdir):
        print("Creating output dir {}".format(outputdir))
        os.makedirs(outputdir)
    else:
        print("Using output dir {}".format(outputdir))

    stagedir = os.path.join(outputdir, "raw_nightly")
    if os.path.isdir(stagedir):
        print("Wiping {}".format(stagedir))
        os.shutil.rmtree(stagedir)
    os.makedirs(stagedir)

    # Create production for nightly per-exposure processing

    prodnight = os.path.join(outputdir, "nightly")
    if os.path.isdir(prodnight):
        os.shutil.rmtree(prodnight)

    com = ["desi_pipe", "create", "--db-sqlite"]
    com.extend(["--data", stagedir])
    com.extend(["--redux", outputdir])
    com.extend(["--prod", "nightly"])
    com.extend(createopt)

    print("Running {}".format(" ".join(com)))
    ret = sprun(com)
    if ret != 0:
        sys.exit(ret)

    # Create production for full chain processing

    prodfull = os.path.join(outputdir, "full")
    if os.path.isdir(prodfull):
        os.shutil.rmtree(prodfull)

    com = ["desi_pipe", "create", "--db-sqlite"]
    com.extend(["--data", inputdir])
    com.extend(["--redux", outputdir])
    com.extend(["--prod", "full"])
    com.extend(createopt)

    print("Running {}".format(" ".join(com)))
    sys.stdout.flush()
    ret = sprun(com)
    if ret != 0:
        sys.exit(ret)

    # Submit jobs for the Full case.  Pack jobs to minimize the number that
    # are queued.

    blocks = [
        "preproc,psf,psfnight",
        "traceshift,extract",
        "fiberflat,fiberflatnight,sky,starfit,fluxcalib,cframe",
        "spectra,redshift"
    ]

    setupfile = os.path.join(prodfull, "setup.sh")
    load_setup(setupfile)

    previous = None
    for blk in blocks:
        com = ["desi_pipe", "chain", "--tasktypes", blk, "--pack"]
        com.extend(chainopt)
        if previous is not None:
            com.extend(["--depjobs", previous])
        print("Running {}".format(" ".join(com)))
        sys.stdout.flush()
        ret, out = sprun(com, capture=True)
        if ret != 0:
            sys.exit(ret)
        jid = None
        for line in out:
            jid = line
        previous = jid

    # Run fake DTS to test per-exposure processing

    setupfile = os.path.join(prodnight, "setup.sh")
    load_setup(setupfile)

    com = ["desi_fake_dts", "--staging", inputdir, "--exptime_arc", "0", "--exptime_flat", "0", "--exptime_science", "0"]
    com.extend(dtsopt)

    print("Running {}".format(" ".join(com)))
    sys.stdout.flush()
    ret = sprun(com)
    if ret != 0:
        sys.exit(ret)

    return


if __name__ == '__main__':
    main()
