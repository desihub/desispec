#
# See top-level LICENSE file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desispec.pipeline.core
======================

Core functions.
"""

from __future__ import absolute_import, division

import os
import time

import desispec.log

def runcmd(cmd, inputs=[], outputs=[], clobber=False):
    """
    Runs a command, checking for inputs and outputs

    Args:
        cmd : command string to run with os.system()
        inputs : list of filename inputs that must exist before running
        outputs : list of output filenames that should be created
        clobber : if True, run even if outputs already exist

    Returns:
        error code from command or input/output checking; 0 is good

    TODO:
        Should it raise an exception instead?

    Notes:
        If any inputs are missing, don't run cmd.
        If outputs exist and have timestamps after all inputs, don't run cmd.

    """
    log = desispec.log.get_logger()
    #- Check that inputs exist
    err = 0
    input_time = 0  #- timestamp of latest input file
    for x in inputs:
        if not os.path.exists(x):
            log.error("missing input "+x)
            err = 1
        else:
            input_time = max(input_time, os.stat(x).st_mtime)

    if err > 0:
        return err

    #- Check if outputs already exist and that their timestamp is after
    #- the last input timestamp
    already_done = (not clobber) and (len(outputs) > 0)
    if not clobber:
        for x in outputs:
            if not os.path.exists(x):
                already_done = False
                break
            if len(inputs)>0 and os.stat(x).st_mtime < input_time:
                already_done = False
                break

    if already_done:
        log.info("SKIPPING: "+ cmd)
        return 0

    #- Green light to go; print input/output info
    #- Use log.level to decide verbosity, but avoid long prefixes
    log.info(time.asctime())
    log.info("RUNNING: " + cmd)
    if log.level <= desispec.log.INFO:
        if len(inputs) > 0:
            print "  Inputs"
            for x in inputs:
                print "   ", x
        if len(outputs) > 0:
            print "  Outputs"
            for x in outputs:
                print "   ", x

    #- run command
    err = os.system(cmd)
    log.info(time.asctime())
    if err > 0:
        log.critical("FAILED: "+cmd)
        return err

    #- Check for outputs
    err = 0
    for x in outputs:
        if not os.path.exists(x):
            log.error("missing output "+x)
            err = 2
    if err > 0:
        return err

    log.info("SUCCESS: " + cmd)
    return 0
