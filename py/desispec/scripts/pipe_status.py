#!/usr/bin/env python
#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-

"""
Inspect the current state of a pipeline production and retry failed steps.
"""

from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
import re
import glob
import subprocess

from .. import io
from .. import pipeline as pipe


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


class pipe_status(object):

    def __init__(self):
        #self.pref = "DESI"
        self.pref = ""

        parser = argparse.ArgumentParser(
            description="Explore DESI pipeline status",
            usage="""desi_pipe_status <command> [options]

Where supported commands are:
    all   Overview of the whole production.
   step   Details about a particular pipeline step.
   task   Explore a particular task.
""")
        parser.add_argument("command", help="Subcommand to run")
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Unrecognized command")
            parser.print_help()
            sys.exit(0)

        # load plan and common metadata

        self.rawdir = io.rawdata_root()
        self.proddir = io.specprod_root()
        self.plandir = io.get_pipe_plandir()
        self.rundir = io.get_pipe_rundir()

        self.expdir = os.path.join(self.proddir, "exposures")
        self.logdir = os.path.join(self.rundir, io.get_pipe_logdir())
        self.faildir = os.path.join(self.rundir, io.get_pipe_faildir())
        self.cal2d = os.path.join(self.proddir, "calib2d")
        self.calpsf = os.path.join(self.cal2d, "psf")

        print("{}{:<22} = {}{}{}".format(self.pref, "Raw data directory", clr.OKBLUE, self.rawdir, clr.ENDC))
        print("{}{:<22} = {}{}{}".format(self.pref, "Production directory", clr.OKBLUE, self.proddir, clr.ENDC))
        print("")

        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()


    def load_state(self):
        (self.state_file, self.state_ftime, self.state_jobid, self.state_running) = pipe.graph_db_info()
        self.grph = None
        if self.state_file == "":
            # no state files exist- manually check all files
            self.grph = pipe.load_prod()
            pipe.graph_db_check(self.grph)
        else:
            # load the latest state
            self.grph = pipe.graph_db_read(self.state_file)
        return


    def all(self):
        self.load_state()
        # go through the current state and accumulate success / failure
        status = {}
        for st in pipe.step_types:
            status[st] = {}
            status[st]["total"] = 0
            status[st]["none"] = 0
            status[st]["running"] = 0
            status[st]["fail"] = 0
            status[st]["done"] = 0

        fts = pipe.file_types_step
        for name, nd in self.grph.items():
            tp = nd["type"]
            if tp in fts.keys():
                status[fts[tp]]["total"] += 1
                status[fts[tp]][nd["state"]] += 1
        
        for st in pipe.step_types:
            beg = ""
            if status[st]["done"] == status[st]["total"]:
                beg = clr.OKGREEN
            elif status[st]["fail"] > 0:
                beg = clr.FAIL
            elif status[st]["running"] > 0:
                beg = clr.WARNING
            print("{}    {}{:<12}{} {:>5} tasks".format(self.pref, beg, st, clr.ENDC, status[st]["total"]))
        print("")
        return


    def step(self):
        parser = argparse.ArgumentParser(description="Details about a particular pipeline step")
        parser.add_argument("step", help="Step name (allowed values are: bootcalib, specex, psfcombine, extract, fiberflat, sky, stdstars, fluxcal, procexp, and zfind).")
        parser.add_argument("--state", required=False, default=None, help="Only list tasks in this state (allowed values are: done, fail, running, none)")
        # now that we"re inside a subcommand, ignore the first
        # TWO argvs
        args = parser.parse_args(sys.argv[2:])

        if args.step not in pipe.step_types:
            print("Unrecognized step name")
            parser.print_help()
            sys.exit(0)

        self.load_state()

        tasks_done = []
        tasks_none = []
        tasks_fail = []
        tasks_running = []

        fts = pipe.step_file_types[args.step]
        for name, nd in self.grph.items():
            tp = nd["type"]
            if tp == fts:
                stat = nd["state"]
                if stat == "done":
                    tasks_done.append(name)
                elif stat == "fail":
                    tasks_fail.append(name)
                elif stat == "running":
                    tasks_running.append(name)
                else:
                    tasks_none.append(name)

        if (args.state is None) or (args.state == "done"):
            for tsk in sorted(tasks_done):
                print("{}    {}{:<20}{}".format(self.pref, clr.OKGREEN, tsk, clr.ENDC))
        if (args.state is None) or (args.state == "fail"):
            for tsk in sorted(tasks_fail):
                print("{}    {}{:<20}{}".format(self.pref, clr.FAIL, tsk, clr.ENDC))
        if (args.state is None) or (args.state == "running"):
            for tsk in sorted(tasks_running):
                print("{}    {}{:<20}{}".format(self.pref, clr.WARNING, tsk, clr.ENDC))
        if (args.state is None) or (args.state == "none"):
            for tsk in sorted(tasks_none):
                print("{}    {:<20}".format(self.pref, tsk))


    def task(self):
        parser = argparse.ArgumentParser(description="Details about a specific pipeline task")
        parser.add_argument("task", help="Task name (as displayed by the \"step\" command).")
        parser.add_argument("--log", required=False, default=False, action="store_true", help="Print the log and traceback, if applicable")
        parser.add_argument("--retry", required=False, default=False, action="store_true", help="Retry the specified task")
        parser.add_argument("--opts", required=False, default=None, help="Retry using this options file")
        # now that we're inside a subcommand, ignore the first
        # TWO argvs
        args = parser.parse_args(sys.argv[2:])

        self.load_state()

        if args.task not in self.grph.keys():
            print("Task {} not found in graph.".format(args.task))
            sys.exit(0)

        nd = self.grph[args.task]
        stat = nd["state"]

        beg = ""
        if stat == "done":
            beg = clr.OKGREEN
        elif stat == "fail":
            beg = clr.FAIL
        elif stat == "running":
            beg = clr.WARNING

        filepath = pipe.graph_path(args.task)

        (night, gname) = pipe.graph_night_split(args.task)
        nfaildir = os.path.join(self.faildir, night)
        nlogdir = os.path.join(self.logdir, night)

        logpath = os.path.join(nlogdir, "{}.log".format(gname))

        ymlpath = os.path.join(nfaildir, "{}_{}.yaml".format(pipe.file_types_step[nd["type"]], args.task))

        if args.retry:
            if stat != "fail":
                print("Task {} has not failed, cannot retry".format(args.task))
            else:
                if os.path.isfile(ymlpath):
                    newopts = None
                    if args.opts is not None:
                        newopts = pipe.yaml_read(args.opts)
                    try:
                        pipe.retry_task(ymlpath, newopts=newopts)
                    finally:
                        self.grph[args.task]["state"] = "done"
                        pipe.graph_db_write(self.grph)
                else:
                    print("Failure yaml dump does not exist!")
        else:
            print("{}{}:".format(self.pref, args.task))
            print("{}    state = {}{}{}".format(self.pref, beg, stat, clr.ENDC))
            print("{}    path = {}".format(self.pref, filepath))
            print("{}    logfile = {}".format(self.pref, logpath))
            print("{}    inputs required:".format(self.pref))
            for d in sorted(nd["in"]):
                print("{}      {}".format(self.pref, d))
            print("{}    output dependents:".format(self.pref))
            for d in sorted(nd["out"]):
                print("{}      {}".format(self.pref, d))
            print("")

            if args.log:
                print("=========== Begin Log =============")
                print("")
                with open(logpath, "r") as f:
                    logdata = f.read()
                    print(logdata)
                print("")
                print("============ End Log ==============")
                print("")

        return


def main():
    pipe_status()

