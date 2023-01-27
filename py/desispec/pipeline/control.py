#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
desispec.pipeline.control
=========================

Tools for controling pipeline production.
"""

from __future__ import absolute_import, division, print_function

import os
import sys
import re
import time

from collections import OrderedDict

import numpy as np

from desiutil.log import get_logger

from .. import io

from ..parallel import (dist_uniform, dist_discrete, dist_discrete_all,
    stdouterr_redirected)

from .defs import (task_states, prod_options_name,
    task_state_to_int, task_int_to_state)

from . import prod as pipeprod
from . import db as pipedb
from . import run as piperun
from . import tasks as pipetasks
from . import scriptgen as scriptgen


class clr:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    def disable(self):
        self.HEADER = ""
        self.OKBLUE = ""
        self.OKGREEN = ""
        self.WARNING = ""
        self.FAIL = ""
        self.ENDC = ""


def create(root=None, data=None, redux=None, prod=None, force=False,
    basis=None, calib=None, db_sqlite=False, db_sqlite_path=None,
    db_postgres=False, db_postgres_host="nerscdb03.nersc.gov",
    db_postgres_port=5432, db_postgres_name="desidev",
    db_postgres_user="desidev_admin", db_postgres_authorized="desidev_ro",
    nside=64 ):
    """Create (or re-create) a production.

    Args:
        root (str): value to use for DESI_ROOT.
        data (str): value to use for DESI_SPECTRO_DATA.
        redux (str): value to use for DESI_SPECTRO_REDUX.
        prod (str): value to use for SPECPROD.
        force (bool): if True, overwrite existing production DB.
        basis (str): value to use for DESI_BASIS_TEMPLATES.
        calib (str): value to use for DESI_SPECTRO_CALIB.
        db_sqlite (bool): if True, use SQLite for the DB.
        db_sqlite_path (str): override path to SQLite DB.
        db_postgres (bool): if True, use PostgreSQL for the DB.
        db_postgres_host (str): PostgreSQL hostname.
        db_postgres_port (int): PostgreSQL connection port number.
        db_postgres_name (str): PostgreSQL DB name.
        db_postgres_user (str): PostgreSQL user name.
        db_postgres_authorized (str): Additional PostgreSQL users to
            authorize.
        nside (int): HEALPix nside value used for spectral grouping.

    """
    log = get_logger()

    # Check desi root location

    desiroot = None
    if root is not None:
        desiroot = os.path.abspath(root)
        os.environ["DESI_ROOT"] = desiroot
    elif "DESI_ROOT" in os.environ:
        desiroot = os.environ["DESI_ROOT"]
    else:
        log.error("You must set DESI_ROOT in your environment or "
            "set the root keyword argument")
        raise RuntimeError("Invalid DESI_ROOT")

    # Check raw data location

    rawdir = None
    if data is not None:
        rawdir = os.path.abspath(data)
        os.environ["DESI_SPECTRO_DATA"] = rawdir
    elif "DESI_SPECTRO_DATA" in os.environ:
        rawdir = os.environ["DESI_SPECTRO_DATA"]
    else:
        log.error("You must set DESI_SPECTRO_DATA in your environment or "
            "set the data keyword argument")
        raise RuntimeError("Invalid DESI_SPECTRO_DATA")

    # Check production name

    prodname = None
    if prod is not None:
        prodname = prod
        os.environ["SPECPROD"] = prodname
    elif "SPECPROD" in os.environ:
        prodname = os.environ["SPECPROD"]
    else:
        log.error("You must set SPECPROD in your environment or "
            "set the prod keyword argument")
        raise RuntimeError("Invalid SPECPROD")

    # Check spectro redux location

    specdir = None
    if redux is not None:
        specdir = os.path.abspath(redux)
        os.environ["DESI_SPECTRO_REDUX"] = specdir
    elif "DESI_SPECTRO_REDUX" in os.environ:
        specdir = os.environ["DESI_SPECTRO_REDUX"]
    else:
        log.error("You must set DESI_SPECTRO_REDUX in your environment or "
            "set the redux keyword argument")
        raise RuntimeError("Invalid DESI_SPECTRO_REDUX")

    proddir = os.path.join(specdir, prodname)
    if os.path.exists(proddir) and not force :
        log.error("Production {} exists.\n"
            "Either remove this directory if you want to start fresh\n"
            "or use 'desi_pipe update' to update a production\n"
            "or rerun with --force option.".format(proddir))
        raise RuntimeError("production already exists")

    # Check basis template location

    if basis is not None:
        basis = os.path.abspath(basis)
        os.environ["DESI_BASIS_TEMPLATES"] = basis
    elif "DESI_BASIS_TEMPLATES" in os.environ:
        basis = os.environ["DESI_BASIS_TEMPLATES"]
    else:
        log.error("You must set DESI_BASIS_TEMPLATES in your environment or "
            "set the basis keyword argument")
        raise RuntimeError("Invalid DESI_BASIS_TEMPLATES")

    # Check calibration location

    if calib is not None:
        calib = os.path.abspath(calib)
        os.environ["DESI_SPECTRO_CALIB"] = calib
    elif "DESI_SPECTRO_CALIB" in os.environ:
        calib = os.environ["DESI_SPECTRO_CALIB"]
    else:
        log.error("You must set DESI_SPECTRO_CALIB in your environment "
            " or set the calib keyword argument")
        raise RuntimeError("Invalid DESI_SPECTRO_CALIB")

    # Construct our DB connection string

    dbpath = None
    if db_postgres:
        # We are creating a new postgres backend. Explicitly create the
        # database, so that we can get the schema key.
        db = pipedb.DataBasePostgres(host=db_postgres_host,
            port=db_postgres_port, dbname=db_postgres_name,
            user=db_postgres_user, schema=None,
            authorize=db_postgres_authorized)

        dbprops = [
            "postgresql",
            db_postgres_host,
            "{}".format(db_postgres_port),
            db_postgres_name,
            db_postgres_user,
            db.schema
        ]
        dbpath = ":".join(dbprops)
        os.environ["DESI_SPECTRO_DB"] = dbpath

    elif db_sqlite:
        # We are creating a new sqlite backend
        if db_sqlite_path is not None:
            # We are using a non-default path
            dbpath = os.path.abspath(db_sqlite_path)
        else:
            # We are using sqlite with the default location
            dbpath = os.path.join(proddir, "desi.db")
            if not os.path.isdir(proddir):
                os.makedirs(proddir)

        # Create the database
        db = pipedb.DataBaseSqlite(dbpath, "w")

        os.environ["DESI_SPECTRO_DB"] = dbpath

    elif "DESI_SPECTRO_DB" in os.environ:
        # We are using an existing prod
        dbpath = os.environ["DESI_SPECTRO_DB"]

    else:
        # Error- we have to get the DB info from somewhere
        log.error("You must set DESI_SPECTRO_DB in your environment or "
            "use the db_sqlite or db_postgres arguments")
        raise RuntimeError("Invalid DESI_SPECTRO_DB")

    pipeprod.update_prod(nightstr=None, hpxnside=nside)

    # create setup shell snippet

    setupfile = os.path.abspath(os.path.join(proddir, "setup.sh"))
    with open(setupfile, "w") as s:
        s.write("# Generated by desi_pipe\n")
        s.write("export DESI_ROOT={}\n\n".format(desiroot))
        s.write("export DESI_BASIS_TEMPLATES={}\n".format(basis))
        s.write("export DESI_SPECTRO_CALIB={}\n\n".format(calib))
        s.write("export DESI_SPECTRO_DATA={}\n\n".format(rawdir))
        s.write("# Production originally created at\n")
        s.write("#   $DESI_SPECTRO_REDUX={}\n".format(specdir))
        s.write("#   $SPECPROD={}\n".format(prodname))
        s.write("#\n")
        s.write("# Support the ability to move the production\n")
        s.write("#   - get abspath to directory where this script is located\n")
        s.write("#   - unpack proddir=$DESI_SPECTRO_REDUX/$SPECPROD\n\n")
        s.write('proddir=$(cd $(dirname "$BASH_SOURCE"); pwd)\n')
        s.write("export DESI_SPECTRO_REDUX=$(dirname $proddir)\n")
        s.write("export SPECPROD=$(basename $proddir)\n\n")
        # s.write("export DESI_SPECTRO_REDUX={}\n".format(specdir))
        # s.write("export SPECPROD={}\n".format(specprod))
        s.write("export DESI_SPECTRO_DB={}\n".format(dbpath))
        s.write("\n")
        if "DESI_LOGLEVEL" in os.environ:
            s.write("export DESI_LOGLEVEL=\"{}\"\n\n"\
                .format(os.environ["DESI_LOGLEVEL"]))
        else:
            s.write("#export DESI_LOGLEVEL=\"DEBUG\"\n\n")

    log.info("\n\nTo use this production, you should do:\n%> source {}\n\n"\
        .format(setupfile))

    return


def update(nightstr=None, nside=64, expid=None):
    """Update a production.

    Args:
        nightstr (str):  Comma separated (YYYYMMDD) or regex pattern.  Only
            nights matching these patterns will be considered.
        nside (int): HEALPix nside value used for spectral grouping.
        expid (int): Only update the production for a single exposure ID.

    """
    pipeprod.update_prod(nightstr=nightstr, hpxnside=nside, expid=expid)

    return


def get_tasks_type(db, tasktype, states, nights, expid=None, spec=None):
    """Get tasks of one type that match certain criteria.

    Args:
        db (DataBase): the production DB.
        tasktype (str): a valid task type.
        states (list): list of task states to select.
        nights (list): list of nights to select.
        expid (int): exposure ID to select.
        spec (int): spectrograph to select.

    Returns:
        (list): list of tasks meeting the criteria.

    """
    ntlist = ",".join(nights)
    if (expid is not None) and (len(nights) > 1):
        raise RuntimeError("Only one night should be specified when "
                           "getting tasks for a single exposure.")

    tasks = list()
    with db.cursor() as cur:

        if tasktype == "spectra" or tasktype == "redshift":

            cmd = "select pixel from healpix_frame where night in ({})".format(ntlist)
            cur.execute(cmd)
            pixels = np.unique([ x for (x,) in cur.fetchall() ]).tolist()
            pixlist = ",".join([ str(p) for p in pixels])
            cmd = "select name,state from {} where pixel in ({})".format(tasktype, pixlist)
            cur.execute(cmd)
            tasks = [ x for (x, y) in cur.fetchall() if \
                      task_int_to_state[y] in states ]

        else :
            cmd = "select name, state from {} where night in ({})"\
                .format(tasktype, ntlist)
            if expid is not None:
                cmd = "{} and expid = {}".format(cmd, expid)
            if spec is not None:
                cmd = "{} and spec = {}".format(cmd, spec)
            cur.execute(cmd)
            tasks = [ x for (x, y) in cur.fetchall() if \
                      task_int_to_state[y] in states ]
    return tasks


def get_tasks(db, tasktypes, nights, states=None, expid=None, spec=None,
    nosubmitted=False, taskfile=None):
    """Get tasks of multiple types that match certain criteria.

    Args:
        db (DataBase): the production DB.
        tasktypes (list): list of valid task types.
        states (list): list of task states to select.
        nights (list): list of nights to select.
        expid (int): exposure ID to select.
        spec (int): spectrograph to select.
        nosubmitted (bool): if True, ignore tasks that were already
            submitted.

    Returns:
        list: all tasks of all types.

    """
    all_tasks = list()
    for tt in tasktypes:
        tasks = get_tasks_type(db, tt, states, nights, expid=expid, spec=spec)
        if nosubmitted:
            if (tt != "spectra") and (tt != "redshift"):
                sb = db.get_submitted(tasks)
                tasks = [ x for x in tasks if not sb[x] ]
        all_tasks.extend(tasks)

    return all_tasks


def tasks(tasktypes, nightstr=None, states=None, expid=None, spec=None,
    nosubmitted=False, db_postgres_user="desidev_ro", taskfile=None):
    """Get tasks of multiple types that match certain criteria.

    Args:
        tasktypes (list): list of valid task types.
        nightstr (list): comma separated (YYYYMMDD) or regex pattern.
        states (list): list of task states to select.
        expid (int): exposure ID to select.
        spec (int): spectrograph to select.
        nosubmitted (bool): if True, ignore tasks that were already
            submitted.
        db_postgres_user (str): If using postgres, connect as this
            user for read-only access"
        taskfile (str): if set write to this file, else write to STDOUT.

    """
    if states is None:
        states = task_states
    else:
        for s in states:
            if s not in task_states:
                raise RuntimeError("Task state '{}' is not valid".format(s))

    dbpath = io.get_pipe_database()
    db = pipedb.load_db(dbpath, mode="r", user=db_postgres_user)

    allnights = io.get_nights(strip_path=True)
    nights = pipeprod.select_nights(allnights, nightstr)

    ttypes = list()
    for tt in pipedb.all_task_types():
        if tt in tasktypes:
            ttypes.append(tt)

    all_tasks = get_tasks(db, ttypes, nights, states=states, expid=expid,
        spec=spec, nosubmitted=nosubmitted)

    pipeprod.task_write(taskfile, all_tasks)

    return


def getready(db, nightstr=None):
    """Update forward dependencies in the database.

    Update database for one or more nights to ensure that forward
    dependencies know that they are ready to run.

    Args:
        db (DataBase): the production DB.
        nightstr (list): comma separated (YYYYMMDD) or regex pattern.

    """
    allnights = io.get_nights(strip_path=True)
    nights = pipeprod.select_nights(allnights, nightstr)
    for nt in nights:
        db.getready(night=nt)
    return


def check_tasks(tasks, db=None):
    """Check the state of pipeline tasks.

    If the database handle is given, use the DB for checking.  Otherwise
    use the filesystem.

    Args:
        tasks (list): list of tasks to check.
        db (DataBase): the database to use.

    Returns:
        OrderedDict: dictionary of the state of each task.

    """
    states = pipedb.check_tasks(tasks, db=db)

    tskstate = OrderedDict()
    for tsk in tasks:
        tskstate[tsk] = states[tsk]

    return tskstate


def sync(db, nightstr=None, specdone=False):
    """Synchronize DB state based on the filesystem.

    This scans the filesystem for all tasks for the specified nights,
    and updates the states accordingly.

    Args:
        db (DataBase): the production DB.
        nightstr (list): comma separated (YYYYMMDD) or regex pattern.
        specdone: If true, set spectra to done if files exist.
    """
    allnights = io.get_nights(strip_path=True)
    nights = pipeprod.select_nights(allnights, nightstr)

    for nt in nights:
        db.sync(nt,specdone=specdone)
    return


def cleanup(db, tasktypes, failed=False, submitted=False, expid=None):
    """Clean up stale tasks in the DB.

    Args:
        db (DataBase): the production DB.
        tasktypes (list): list of valid task types.
        failed (bool): also clear failed states.
        submitted (bool): also clear submitted flag.
        expid (int): only clean this exposure ID.

    """
    exid = None
    if expid is not None and expid >= 0:
        exid = expid

    db.cleanup(tasktypes=tasktypes, expid=exid, cleanfailed=failed,
        cleansubmitted=submitted)
    return


def dryrun(tasks, nersc=None, nersc_queue="regular", nersc_maxtime=0,
    nersc_maxnodes=0, nersc_shifter=None, mpi_procs=1, mpi_run="",
    procs_per_node=0, nodb=False, db_postgres_user="desidev_ro", force=False):
    """Print equivalent command line jobs.

    For the specified tasks, print the equivalent stand-alone commands
    that would be run on each task.  A pipeline job calls the internal
    desispec.scripts entrypoints directly.

    Args:
        tasks (list): list of tasks to run.
        nersc (str): if not None, the name of the nersc machine to use
            (cori-haswell | cori-knl).
        nersc_queue (str): the name of the queue to use
            (regular | debug | realtime).
        nersc_maxtime (int): if specified, restrict the runtime to this
            number of minutes.
        nersc_maxnodes (int): if specified, restrict the job to use this
            number of nodes.
        nersc_shifter (str): the name of the shifter image to use.
        mpi_run (str): if specified, and if not using NERSC, use this
            command to launch MPI executables in the shell scripts.  Default
            is to not use MPI.
        mpi_procs (int): if not using NERSC, the number of MPI processes
            to use in shell scripts.
        procs_per_node (int): if specified, use only this number of
            processes per node.  Default runs one process per core.
        nodb (bool): if True, do not use the production DB.
        db_postgres_user (str): If using postgres, connect as this
            user for read-only access"
        force (bool): if True, print commands for all tasks, not just the ones
            in a ready state.

    """
    tasks_by_type = pipedb.task_sort(tasks)

    (db, opts) = pipeprod.load_prod("r", user=db_postgres_user)
    if nodb:
        db = None

    ppn = None
    if procs_per_node > 0:
        ppn = procs_per_node

    if nersc is None:
        # Not running at NERSC
        if ppn is None:
            ppn = mpi_procs
        for tt, tlist in tasks_by_type.items():
            piperun.dry_run(tt, tlist, opts, mpi_procs,
                ppn, db=db, launch="mpirun -n", force=force)
    else:
        # Running at NERSC
        hostprops = scriptgen.nersc_machine(nersc,
            nersc_queue)

        for tt, tlist in tasks_by_type.items():
            joblist = scriptgen.nersc_job_size(tt, tlist,
                nersc, nersc_queue, nersc_maxtime,
                nersc_maxnodes, nodeprocs=ppn, db=db)

            launch="srun -n"
            for (jobnodes, jobppn, jobtime, jobworkers, jobtasks) in joblist:
                jobprocs = jobnodes * jobppn
                piperun.dry_run(tt, jobtasks, opts, jobprocs,
                    jobppn, db=db, launch=launch, force=force)
    return


def gen_scripts(tasks_by_type, nersc=None, nersc_queue="regular",
    nersc_maxtime=0, nersc_maxnodes=0, nersc_shifter=None, mpi_procs=1,
    mpi_run="", procs_per_node=0, nodb=False, out=None, debug=False,
    db_postgres_user="desidev_ro"):
    """Generate scripts to run tasks of one or more types.

    If multiple task type keys are contained in the dictionary, they will
    be packed into a single batch job.

    Args:
        tasks_by_type (dict): each key is the task type and the value is
            a list of tasks.
        nersc (str): if not None, the name of the nersc machine to use
            (cori-haswell | cori-knl).
        nersc_queue (str): the name of the queue to use
            (regular | debug | realtime).
        nersc_maxtime (int): if specified, restrict the runtime to this
            number of minutes.
        nersc_maxnodes (int): if specified, restrict the job to use this
            number of nodes.
        nersc_shifter (str): the name of the shifter image to use.
        mpi_run (str): if specified, and if not using NERSC, use this
            command to launch MPI executables in the shell scripts.  Default
            is to not use MPI.
        mpi_procs (int): if not using NERSC, the number of MPI processes
            to use in shell scripts.
        procs_per_node (int): if specified, use only this number of
            processes per node.  Default runs one process per core.
        nodb (bool): if True, do not use the production DB.
        out (str): Put task scripts and logs in this directory relative to
            the production 'scripts' directory.  Default puts task directory
            in the main scripts directory.
        debug (bool): if True, enable DEBUG log level in generated scripts.
        db_postgres_user (str): If using postgres, connect as this
            user for read-only access"

    Returns:
        list: the generated script files

    """
    ttypes = list(tasks_by_type.keys())

    if len(ttypes)==0 :
        return None

    jobname = ttypes[0]
    if len(ttypes) > 1:
        jobname = "{}-{}".format(ttypes[0], ttypes[-1])

    proddir = os.path.abspath(io.specprod_root())

    import datetime
    now = datetime.datetime.now()
    outtaskdir = "{}_{:%Y%m%d-%H%M%S-%f}".format(jobname, now)

    if out is None:
        outdir = os.path.join(proddir, io.get_pipe_rundir(),
            io.get_pipe_scriptdir(), outtaskdir)
    else:
        outdir = os.path.join(proddir, io.get_pipe_rundir(),
            io.get_pipe_scriptdir(), out, outtaskdir)

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    mstr = "run"
    if nersc is not None:
        mstr = nersc

    outscript = os.path.join(outdir, mstr)
    outlog = os.path.join(outdir, mstr)

    (db, opts) = pipeprod.load_prod("r", user=db_postgres_user)
    if nodb:
        db = None

    ppn = None
    if procs_per_node > 0:
        ppn = procs_per_node

    # FIXME: Add openmp / multiproc function to task classes and
    # call them here.

    scripts = None

    if nersc is None:
        # Not running at NERSC
        scripts = scriptgen.batch_shell(tasks_by_type,
            outscript, outlog, mpirun=mpi_run,
            mpiprocs=mpi_procs, openmp=1, db=db)
    else:
        # Running at NERSC
        scripts = scriptgen.batch_nersc(tasks_by_type,
            outscript, outlog, jobname, nersc, nersc_queue,
            nersc_maxtime, nersc_maxnodes, nodeprocs=ppn,
            openmp=False, multiproc=False, db=db,
            shifterimg=nersc_shifter, debug=debug)

    return scripts


def script(taskfile, nersc=None, nersc_queue="regular",
    nersc_maxtime=0, nersc_maxnodes=0, nersc_shifter=None, mpi_procs=1,
    mpi_run="", procs_per_node=0, nodb=False, out=None, debug=False,
    db_postgres_user="desidev_ro"):
    """Generate pipeline scripts for a taskfile.

    This gets tasks from the taskfile and sorts them by type.  Then it
    generates the scripts.

    Args:
        taskfile (str): read tasks from this file (if not specified,
            read from STDIN).
        nersc (str): if not None, the name of the nersc machine to use
            (cori-haswell | cori-knl).
        nersc_queue (str): the name of the queue to use
            (regular | debug | realtime).
        nersc_maxtime (int): if specified, restrict the runtime to this
            number of minutes.
        nersc_maxnodes (int): if specified, restrict the job to use this
            number of nodes.
        nersc_shifter (str): the name of the shifter image to use.
        mpi_run (str): if specified, and if not using NERSC, use this
            command to launch MPI executables in the shell scripts.  Default
            is to not use MPI.
        mpi_procs (int): if not using NERSC, the number of MPI processes
            to use in shell scripts.
        procs_per_node (int): if specified, use only this number of
            processes per node.  Default runs one process per core.
        nodb (bool): if True, do not use the production DB.
        out (str): Put task scripts and logs in this directory relative to
            the production 'scripts' directory.  Default puts task directory
            in the main scripts directory.
        debug (bool): if True, enable DEBUG log level in generated scripts.
        db_postgres_user (str): If using postgres, connect as this
            user for read-only access"

    Returns:
        list: the generated script files

    """
    tasks = pipeprod.task_read(taskfile)

    scripts = list()
    if len(tasks) > 0:
        tasks_by_type = pipedb.task_sort(tasks)
        scripts = gen_scripts(
            tasks_by_type,
            nersc=nersc,
            nersc_queue=nersc_queue,
            nersc_maxtime=nersc_maxtime,
            nersc_maxnodes=nersc_maxnodes,
            nersc_shifter=nersc_shifter,
            mpi_procs=mpi_procs,
            mpi_run=mpi_run,
            procs_per_node=procs_per_node,
            nodb=nodb,
            out=out,
            debug=debug,
            db_postgres_user=db_postgres_user)
    else:
        import warnings
        warnings.warn("Input task list is empty", RuntimeWarning)

    return scripts


def run_scripts(scripts, deps=None, slurm=False):
    """Run job scripts with optional dependecies.

    This either submits the jobs to the scheduler or simply runs them
    in order with subprocess.

    Args:
        scripts (list): list of pathnames of the scripts to run.
        deps (list): optional list of job IDs which are dependencies for
            these scripts.
        slurm (bool): if True use slurm to submit the jobs.

    Returns:
        list: the job IDs returned by the scheduler.

    """
    import subprocess as sp

    log = get_logger()

    depstr = ""
    if deps is not None and len(deps)>0 :
        depstr = "-d afterok"
        for d in deps:
            depstr = "{}:{}".format(depstr, d)

    jobids = list()
    if slurm:
        # submit each job and collect the job IDs
        for scr in scripts:
            scom = "sbatch {} {}".format(depstr, scr)
            #print("RUN SCRIPTS: {}".format(scom))
            log.debug(time.asctime())
            log.info(scom)
            sout = sp.check_output(scom, shell=True, universal_newlines=True)
            log.info(sout)
            p = sout.split()
            jid = re.sub(r'[^\d]', '', p[3])
            jobids.append(jid)
    else:
        # run the scripts one at a time
        for scr in scripts:
            rcode = sp.call(scr, shell=True)
            if rcode != 0:
                log.warning("script {} had return code = {}".format(scr,
                    rcode))
    return jobids


def run(taskfile, nosubmitted=False, depjobs=None, nersc=None,
    nersc_queue="regular", nersc_maxtime=0, nersc_maxnodes=0,
    nersc_shifter=None, mpi_procs=1, mpi_run="", procs_per_node=0, nodb=False,
    out=None, debug=False):
    """Create job scripts and run them.

    This gets tasks from the taskfile and sorts them by type.  Then it
    generates the scripts.  Finally, it runs or submits those scripts
    to the scheduler.

    Args:
        taskfile (str): read tasks from this file (if not specified,
            read from STDIN).
        nosubmitted (bool): if True, do not run jobs that have already
            been submitted.
        depjobs (list): list of job ID dependencies.
        nersc (str): if not None, the name of the nersc machine to use
            (cori-haswell | cori-knl).
        nersc_queue (str): the name of the queue to use
            (regular | debug | realtime).
        nersc_maxtime (int): if specified, restrict the runtime to this
            number of minutes.
        nersc_maxnodes (int): if specified, restrict the job to use this
            number of nodes.
        nersc_shifter (str): the name of the shifter image to use.
        mpi_run (str): if specified, and if not using NERSC, use this
            command to launch MPI executables in the shell scripts.  Default
            is to not use MPI.
        mpi_procs (int): if not using NERSC, the number of MPI processes
            to use in shell scripts.
        procs_per_node (int): if specified, use only this number of
            processes per node.  Default runs one process per core.
        nodb (bool): if True, do not use the production DB.
        out (str): Put task scripts and logs in this directory relative to
            the production 'scripts' directory.  Default puts task directory
            in the main scripts directory.
        debug (bool): if True, enable DEBUG log level in generated scripts.

    Returns:
        list: the job IDs returned by the scheduler.

    """
    log = get_logger()
    tasks = pipeprod.task_read(taskfile)

    jobids = list()

    if len(tasks) > 0:
        tasks_by_type = pipedb.task_sort(tasks)
        tasktypes = list(tasks_by_type.keys())
        # We are packing everything into one job
        scripts = gen_scripts(
            tasks_by_type,
            nersc=nersc,
            nersc_queue=nersc_queue,
            nersc_maxtime=nersc_maxtime,
            nersc_maxnodes=nersc_maxnodes,
            nersc_shifter=nersc_shifter,
            mpi_procs=mpi_procs,
            mpi_run=mpi_run,
            procs_per_node=procs_per_node,
            nodb=nodb,
            out=out,
            debug=debug)

        log.info("wrote scripts {}".format(scripts))

        deps = None
        slurm = False
        if nersc is not None:
            slurm = True
        if depjobs is not None:
            deps = depjobs

        # Run the jobs
        if not nodb:
            # We can use the DB, mark tasks as submitted.
            if slurm:
                dbpath = io.get_pipe_database()
                db = pipedb.load_db(dbpath, mode="w")
                for tt in tasktypes:
                    if (tt != "spectra") and (tt != "redshift"):
                        db.set_submitted_type(tt, tasks_by_type[tt])

        jobids = run_scripts(scripts, deps=deps, slurm=slurm)
    else:
        import warnings
        warnings.warn("Input task list is empty", RuntimeWarning)

    return jobids


def chain(tasktypes, nightstr=None, states=None, expid=None, spec=None,
    pack=False, nosubmitted=False, depjobs=None, nersc=None,
    nersc_queue="regular", nersc_maxtime=0, nersc_maxnodes=0,
    nersc_shifter=None, mpi_procs=1, mpi_run="", procs_per_node=0, nodb=False,
    out=None, debug=False, dryrun=False):
    """Run a chain of jobs for multiple pipeline steps.

    For the list of task types, get all ready tasks meeting the selection
    criteria.  Then either pack all tasks into one job or submit
    each task type as its own job.  Input job dependencies can be
    specified, and dependencies are tracked between jobs in the chain.

    Args:
        tasktypes (list): list of valid task types.
        nightstr (str):  Comma separated (YYYYMMDD) or regex pattern.  Only
            nights matching these patterns will be considered.
        states (list): list of task states to select.
        nights (list): list of nights to select.
        expid (int): exposure ID to select.
        pack (bool): if True, pack all tasks into a single job.
        nosubmitted (bool): if True, do not run jobs that have already
            been submitted.
        depjobs (list): list of job ID dependencies.
        nersc (str): if not None, the name of the nersc machine to use
            (cori-haswell | cori-knl).
        nersc_queue (str): the name of the queue to use
            (regular | debug | realtime).
        nersc_maxtime (int): if specified, restrict the runtime to this
            number of minutes.
        nersc_maxnodes (int): if specified, restrict the job to use this
            number of nodes.
        nersc_shifter (str): the name of the shifter image to use.
        mpi_run (str): if specified, and if not using NERSC, use this
            command to launch MPI executables in the shell scripts.  Default
            is to not use MPI.
        mpi_procs (int): if not using NERSC, the number of MPI processes
            to use in shell scripts.
        procs_per_node (int): if specified, use only this number of
            processes per node.  Default runs one process per core.
        nodb (bool): if True, do not use the production DB.
        out (str): Put task scripts and logs in this directory relative to
            the production 'scripts' directory.  Default puts task directory
            in the main scripts directory.
        debug (bool): if True, enable DEBUG log level in generated scripts.
        dryrun (bool): if True, do not submit the jobs.

    Returns:
        list: the job IDs from the final step in the chain.

    """

    log = get_logger()

    machprops = None
    if nersc is not None:
        machprops = scriptgen.nersc_machine(nersc, nersc_queue)

    if states is None:
        states = task_states
    else:
        for s in states:
            if s not in task_states:
                raise RuntimeError("Task state '{}' is not valid".format(s))

    ttypes = list()
    for tt in pipetasks.base.default_task_chain:
        if tt in tasktypes:
            ttypes.append(tt)

    if (machprops is not None) and (not pack):
        if len(ttypes) > machprops["submitlimit"]:
            log.error("Queue {} on machine {} limited to {} jobs."\
                .format(nersc_queue, nersc,
                machprops["submitlimit"]))
            log.error("Use a different queue or shorter chains of tasks.")
            raise RuntimeError("Too many jobs")

    slurm = False
    if nersc is not None:
        slurm = True

    dbpath = io.get_pipe_database()
    db = pipedb.load_db(dbpath, mode="w")

    allnights = io.get_nights(strip_path=True)
    nights = pipeprod.select_nights(allnights, nightstr)

    outdeps = None
    indeps = None
    if depjobs is not None:
        indeps = depjobs

    tasks_by_type = OrderedDict()

    for tt in ttypes:
        # Get the tasks.  We select by state and submitted status.
        tasks = get_tasks_type(db, tt, states, nights, expid=expid, spec=spec)
        #print("CHAIN:  ", tt, tasks)
        if nosubmitted:
            if (tt != "spectra") and (tt != "redshift"):
                sb = db.get_submitted(tasks)
                tasks = [ x for x in tasks if not sb[x] ]
        #print("CHAIN:  nosubmitted:  ", tt, tasks)

        if len(tasks) == 0:
            import warnings
            warnings.warn("Input task list for '{}' is empty".format(tt),
                          RuntimeWarning)
            continue # might be tasks to do in other ttype
        tasks_by_type[tt] = tasks

    scripts = None
    tscripts = None
    if pack:
        # We are packing everything into one job
        scripts = gen_scripts(
            tasks_by_type,
            nersc=nersc,
            nersc_queue=nersc_queue,
            nersc_maxtime=nersc_maxtime,
            nersc_maxnodes=nersc_maxnodes,
            nersc_shifter=nersc_shifter,
            mpi_procs=mpi_procs,
            mpi_run=mpi_run,
            procs_per_node=procs_per_node,
            nodb=nodb,
            out=out,
            debug=debug)
        if scripts is not None and len(scripts)>0 :
            log.info("wrote scripts {}".format(scripts))
    else:
        # Generate individual scripts
        tscripts = dict()
        for tt in ttypes:
            onetype = OrderedDict()
            onetype[tt] = tasks_by_type[tt]
            tscripts[tt] = gen_scripts(
                onetype,
                nersc=nersc,
                nersc_queue=nersc_queue,
                nersc_maxtime=nersc_maxtime,
                nersc_maxnodes=nersc_maxnodes,
                nersc_shifter=nersc_shifter,
                mpi_procs=mpi_procs,
                mpi_run=mpi_run,
                procs_per_node=procs_per_node,
                nodb=nodb,
                out=out,
                debug=debug)
            if tscripts[tt] is not None :
                log.info("wrote script {}".format(tscripts[tt]))

    if dryrun :
        log.warning("dry run: do not submit the jobs")
        return None

    # Run the jobs
    if slurm:
        for tt in ttypes:
            if (tt != "spectra") and (tt != "redshift"):
                if tt in tasks_by_type.keys() :
                    db.set_submitted_type(tt, tasks_by_type[tt])

    outdeps = None
    if pack:
        # Submit one job
        if scripts is not None and len(scripts)>0 :
            outdeps = run_scripts(scripts, deps=indeps, slurm=slurm)
    else:
        # Loop over task types submitting jobs and tracking dependencies.
        for tt in ttypes:
            if tscripts[tt] is not None :
                outdeps = run_scripts(tscripts[tt], deps=indeps,
                                      slurm=slurm)
            if outdeps is not None and len(outdeps) > 0:
                indeps = outdeps
            else:
                indeps = None

    return outdeps


def status_color(state):
    col = clr.ENDC
    if state == "done":
        col = clr.OKGREEN
    elif state == "running":
        col = clr.WARNING
    elif state == "failed":
        col = clr.FAIL
    elif state == "ready":
        col = clr.OKBLUE
    return col


def status_task(task, ttype, state, logdir):
    fields = pipetasks.base.task_classes[ttype].name_split(task)
    tasklog = None
    if "night" in fields:
        tasklogdir = os.path.join(
            logdir, io.get_pipe_nightdir(),
            "{:08d}".format(fields["night"])
        )
        tasklog = os.path.join(
            tasklogdir,
            "{}.log".format(task)
        )
    elif "pixel" in fields:
        tasklogdir = os.path.join(
            logdir, "healpix",
            io.healpix_subdirectory(fields["nside"],fields["pixel"])
        )
        tasklog = os.path.join(
            tasklogdir,
            "{}.log".format(task)
        )
    col = status_color(state)
    print("Task {}".format(task))
    print(
        "State = {}{}{}".format(
            col,
            state,
            clr.ENDC
        )
    )
    if os.path.isfile(tasklog):
        print("Dumping task log {}".format(tasklog))
        print("=========== Begin Log =============")
        print("")
        with open(tasklog, "r") as f:
            logdata = f.read()
            print(logdata)
        print("")
        print("============ End Log ==============")
        print("", flush=True)
    else:
        print("Task log {} does not exist".format(tasklog), flush=True)
    return


def status_taskname(tsklist):
    for tsk in tsklist:
        st = tsk[1]
        col = status_color(st)
        print(
            "  {:20s}:  {}{}{}".format(tsk[0], col, st, clr.ENDC),
            flush=True
        )


def status_night_totals(tasktypes, nights, tasks, tskstates):
    # Accumulate totals for each night and type
    sep = "------------------+---------+---------+---------+---------+---------+"
    ntlist = list()
    nighttot = OrderedDict()
    for tt in tasktypes:
        if tt == "spectra" or tt == "redshift":
            # This function only prints nightly tasks
            continue
        for tsk in tasks[tt]:
            fields = pipetasks.base.task_classes[tt].name_split(tsk)
            nt = fields["night"]
            if nt not in nighttot:
                nighttot[nt] = OrderedDict()
            if tt not in nighttot[nt]:
                nighttot[nt][tt] = OrderedDict()
                for s in task_states:
                    nighttot[nt][tt][s] = 0
            st = tskstates[tt][tsk]
            nighttot[nt][tt][st] += 1
    for nt, ttstates in nighttot.items():
        ntstr = "{:08d}".format(nt)
        if ntstr in nights:
            ntlist.append(nt)
    ntlist = list(sorted(ntlist))
    for nt in ntlist:
        ttstates = nighttot[nt]
        ntstr = "{:08d}".format(nt)
        if ntstr in nights:
            header = "{:18s}|".format(ntstr)
            for s in task_states:
                col = status_color(s)
                header = "{} {}{:8s}{}|".format(
                    header, col, s, clr.ENDC
                )
            print(sep)
            print(header)
            print(sep)
            for tt, totst in ttstates.items():
                line = "  {:16s}|".format(tt)
                for s in task_states:
                    line = "{}{:9d}|".format(line, totst[s])
                print(line)
            print("", flush=True)


def status_pixel_totals(tasktypes, tasks, tskstates):
    # Accumulate totals for each type
    sep = "------------------+---------+---------+---------+---------+---------+"
    pixtot = OrderedDict()
    for tt in tasktypes:
        if (tt != "spectra") and (tt != "redshift"):
            # This function only prints pixel tasks
            continue
        for tsk in tasks[tt]:
            if tt not in pixtot:
                pixtot[tt] = OrderedDict()
                for s in task_states:
                    pixtot[tt][s] = 0
            st = tskstates[tt][tsk]
            pixtot[tt][st] += 1
    header = "{:18s}|".format("Pixel Tasks")
    for s in task_states:
        col = status_color(s)
        header = "{} {}{:8s}{}|".format(
            header, col, s, clr.ENDC
        )
    print(sep)
    print(header)
    print(sep)
    for tt, totst in pixtot.items():
        line = "  {:16s}|".format(tt)
        for s in task_states:
            line = "{}{:9d}|".format(line, totst[s])
        print(line)
    print("", flush=True)


def status_night_tasks(tasktypes, nights, tasks, tskstates):
    # Sort the tasks into nights
    nighttasks = OrderedDict()
    ntlist = list()
    for tt in tasktypes:
        if tt == "spectra" or tt == "redshift":
            # This function only prints nightly tasks
            continue
        for tsk in tasks[tt]:
            fields = pipetasks.base.task_classes[tt].name_split(tsk)
            nt = fields["night"]
            if nt not in nighttasks:
                nighttasks[nt] = list()
            nighttasks[nt].append((tsk, tskstates[tt][tsk]))
    for nt, tsklist in nighttasks.items():
        ntstr = "{:08d}".format(nt)
        if ntstr in nights:
            ntlist.append(nt)
    ntlist = list(sorted(ntlist))
    for nt in ntlist:
        tsklist = nighttasks[nt]
        ntstr = "{:08d}".format(nt)
        if ntstr in nights:
            print(nt)
            status_taskname(tsklist)


def status_pixel_tasks(tasktypes, tasks, tskstates):
    for tt in tasktypes:
        tsklist = list()
        if (tt != "spectra") and (tt != "redshift"):
            # This function only prints pixel tasks
            continue
        for tsk in tasks[tt]:
            tsklist.append((tsk, tskstates[tt][tsk]))
        print(tt)
        status_taskname(tsklist)


def status_summary(tasktypes, nights, tasks, tskstates):
    sep = "----------------+---------+---------+---------+---------+---------+"
    hline = "-----------------------------------------------"
    print(sep)
    header_state = "{:16s}|".format("   Task Type")
    for s in task_states:
        col = status_color(s)
        header_state = "{} {}{:8s}{}|".format(
            header_state, col, s, clr.ENDC
        )
    print(header_state)
    print(sep)
    for tt in tasktypes:
        line = "{:16s}|".format(tt)
        for s in task_states:
            tsum = np.sum(
                np.array(
                    [1 for x, y in tskstates[tt].items() if y == s],
                    dtype=np.int32
                )
            )
            line = "{}{:9d}|".format(line, tsum)
        print(line, flush=True)


def status(task=None, tasktypes=None, nightstr=None, states=None,
           expid=None, spec=None, db_postgres_user="desidev_ro"):
    """Check the status of pipeline tasks.


    Args:

    Returns:
        None

    """
    dbpath = io.get_pipe_database()
    db = pipedb.load_db(dbpath, mode="r", user=db_postgres_user)

    rundir = io.get_pipe_rundir()
    logdir = os.path.join(rundir, io.get_pipe_logdir())

    tasks = OrderedDict()

    summary = False
    if (tasktypes is None) and (nightstr is None):
        summary = True

    if task is None:
        ttypes = None
        if tasktypes is not None:
            ttypes = list()
            for tt in pipetasks.base.default_task_chain:
                if tt in tasktypes:
                    ttypes.append(tt)
        else:
            ttypes = list(pipetasks.base.default_task_chain)

        if states is None:
            states = task_states
        else:
            for s in states:
                if s not in task_states:
                    raise RuntimeError("Task state '{}' is not valid".format(s))

        allnights = io.get_nights(strip_path=True)
        nights = pipeprod.select_nights(allnights, nightstr)

        for tt in ttypes:
            tasks[tt] = get_tasks(
                db, [tt], nights, states=states, expid=expid, spec=spec
            )
    else:
        ttypes = [pipetasks.base.task_type(task)]
        tasks[ttypes[0]] = [task]

    tstates = OrderedDict()
    for typ, tsks in tasks.items():
        tstates[typ] = pipedb.check_tasks(tsks, db=db)

    if len(ttypes) == 1 and len(tasks[ttypes[0]]) == 1:
        # Print status of this specific task
        thistype = ttypes[0]
        thistask = tasks[thistype][0]
        status_task(thistask, thistype, tstates[thistype][thistask], logdir)
    else:
        if len(ttypes) > 1 and len(nights) > 1:
            # We have multiple nights and multiple task types.
            # Just print totals.
            if summary:
                status_summary(ttypes, nights, tasks, tstates)
            else:
                status_night_totals(ttypes, nights, tasks, tstates)
                status_pixel_totals(ttypes, tasks, tstates)
        elif len(ttypes) > 1:
            # Multiple task types for one night. Print the totals for each
            # task type.
            thisnight = nights[0]
            status_night_totals(ttypes, nights, tasks, tstates)
        elif len(nights) > 1:
            # We have just one task type, print the state totals for each night
            # OR the full task list for redshift or spectra tasks.
            thistype = ttypes[0]
            print("Task type {}".format(thistype))
            if thistype == "spectra" or thistype == "redshift":
                status_pixel_tasks(ttypes, tasks, tstates)
            else:
                status_night_totals(ttypes, nights, tasks, tstates)
        else:
            # We have one type and one night, print the full state of every
            # task.
            thistype = ttypes[0]
            thisnight = nights[0]
            print("Task type {}".format(thistype))
            status_night_tasks(ttypes, nights, tasks, tstates)
            status_pixel_tasks(ttypes, tasks, tstates)

    return
