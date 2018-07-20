#!/usr/bin/env python
#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-

"""Test pipeline processing.

This is a high-level integration test which can process a specified data set
in 2 modes: "all at once", with all exposures run through the sequence of
pipeline steps together and "nightly", with a fake DTS script submitting jobs
one exposure at a time.

In order to run this integration test, you will need simulated raw data that is
in the current format with the fibermap and raw data files in per-exposure
directories.  A small, 3-exposure dataset for use with this test can be found
at NERSC here:

/global/projecta/projectdirs/desi/spectro/sim/pipetest

To run the this script, choose an output directory (./output in this example)
and run the integration test script from a desispec checkout:

$>  python <path_to_desispec>/desispec/etc/desi_nersc_pipetest.py \
--input /global/projecta/projectdirs/desi/spectro/sim/pipetest \
--output ./output \
--root /project/projectdirs/desi \
--basis /project/projectdirs/desi/spectro/templates/basis_templates/v2.5 \
--calib /project/projectdirs/desi/spectro/ccd_calibration_data/trunk \
--nersc edison \
--nersc_queue debug

The above command will run in "full" mode where pipeline steps are run in
sequence for all exposures.  This will take about 45 minutes on edison not
including queue time.

You can also run the test in "nightly" mode.  This will submit too many jobs
for the debug queue, so you should run in the regular or realtime queues.  For
example:

$>  python <path_to_desispec>/desispec/etc/desi_nersc_pipetest.py \
--input /global/projecta/projectdirs/desi/spectro/sim/pipetest \
--output ./output \
--nightly \
--root /project/projectdirs/desi \
--basis /project/projectdirs/desi/spectro/templates/basis_templates/v2.5 \
--calib /project/projectdirs/desi/spectro/ccd_calibration_data/trunk \
--nersc edison \
--nersc_queue realtime

At the end of the test run, the production directory (output/full or
output/nightly) should contain zbest redshift results in the "spectra-64"
subdirectories.

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

def run_args(args):
    optlist = [
        "nersc",
        "nersc_queue",
        "nersc_maxnodes"
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

    parser.add_argument("--nightly", required=False, default=False,
        action="store_true", help="Run in nightly mode rather than full mode")

    parser.add_argument("--root", required=False, default=None,
        help="Override value for DESI_ROOT")

    parser.add_argument("--basis", required=False, default=None,
        help="Override value for DESI_BASIS_TEMPLATES")

    parser.add_argument("--calib", required=False, default=None,
        help="Override value for DESI_CCD_CALIBRATION_DATA")

    parser.add_argument("--nersc", required=False, default=None,
        help="Run on this NERSC system (edison | cori-haswell "
        "| cori-knl).  Default uses $NERSC_HOST")

    parser.add_argument("--nersc_queue", required=False, default="regular",
        help="Run the test in this queue")

    parser.add_argument("--nersc_maxnodes", required=False,
        default=None, help="The maximum number of nodes to use for "
        "processing.  Default is the maximum nodes for the specified queue.")

    args = parser.parse_args()

    if args.nersc is None:
        if "NERSC_HOST" in os.environ:
            if os.environ["NERSC_HOST"] == "cori":
                args.nersc = "cori-haswell"
            else:
                args.nersc = os.environ["NERSC_HOST"]

    createopt = create_args(args)
    runopt = run_args(args)

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

    if args.nightly:
        # We are running in "nightly" mode.
        # Set up data staging location.
        stagedir = os.path.join(outputdir, "raw_nightly")
        if os.path.isdir(stagedir):
            print("Wiping {}".format(stagedir))
            shutil.rmtree(stagedir)
        os.makedirs(stagedir)

        # Pre-create prod.
        prodnight = os.path.join(outputdir, "nightly")
        if os.path.isdir(prodnight):
            shutil.rmtree(prodnight)

        com = ["desi_pipe", "create", "--db-sqlite"]
        com.extend(["--data", stagedir])
        com.extend(["--redux", outputdir])
        com.extend(["--prod", "nightly"])
        com.extend(createopt)

        print("Running {}".format(" ".join(com)))
        sys.stdout.flush()
        ret = sprun(com)
        if ret != 0:
            sys.exit(ret)

        # Run fake DTS to test per-exposure processing

        setupfile = os.path.join(prodnight, "setup.sh")
        load_setup(setupfile)

        com = ["desi_fake_dts", "--staging", inputdir, "--exptime_arc", "0", "--exptime_flat", "0", "--exptime_science", "0"]
        com.extend(runopt)

        print("Running {}".format(" ".join(com)))
        sys.stdout.flush()
        ret = sprun(com)
        if ret != 0:
            sys.exit(ret)

    else:
        # We are running all exposures at once.
        # Create production for full chain processing
        prodfull = os.path.join(outputdir, "full")
        if os.path.isdir(prodfull):
            shutil.rmtree(prodfull)

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
            com.extend(runopt)
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

    return


if __name__ == '__main__':
    main()
