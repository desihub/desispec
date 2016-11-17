.. _overview:


Overview
==============

The DESI spectroscopic pipeline is a collection of software designed to 
efficiently take DESI raw data and produce redshift estimates.  This software
consists of low-level functions that perform individual processing tasks as 
well as higher-level tools to collectively run whole steps of the pipeline.


Hardware Constraints
------------------------------

To the extent possible, the DESI instrument has been designed to make analysis straightforward.  The spectrographs are isolated from the telescope to improve stability, fiber traces are arranged in bundles which should be independent of each other on the CCD, etc.  Because of this, we are able to break up the most expensive analysis steps into small independent pieces, and run those small pieces in parallel.


Computing Infrastructure
------------------------------

The workflow of the spectroscopic pipeline reduces to a set of many tasks that depend on each other.  Tasks of different types often have different computational needs and run using a varying number of processes.  Although this type of workflow could be run on commercial cloud computing platforms (at significant cost), the primary systems available to the DESI project for analysis are the HPC systems at NERSC.  These machines are traditional supercomputers with a high speed interconnect between the nodes.  The nodes are lightweight and have no local disk.  Instead, all nodes share a common Lustre filesystem.  SLURM is used for job scheduling, and the machine is used by a variety of workflows and apps across many projects and science domains.


Software Constraints
------------------------------

Given our instrument hardware and computing infrastructure, the spectroscopic pipeline must:

#. Be able to process DESI data at high concurrency on machines at NERSC.  HPC 
   software has further requirements / guidelines:
      
   - Large jobs must not log excessively to stdout/stderr.
   - Many processes should not read/write to the same files.
   - Python startup time (due to filesystem contention loading shared libraries 
     and modules) must be overcome by some method.

#. Be robust against failures of individual small tasks.  A failure of one task 
   should only cause failures of future tasks that depend on those specific 
   outputs.  This fault tolerance is handled automatically with something like 
   Spark.  On an HPC system we need to track such failures.

#. If an individual step/task fails, it must be possible to determine how that 
   step was run and retry it for debugging.

#. It should be possible to determine the current status of a pipeline and 
   which things have failed.

#. (Eventually) the pipeline should interface with a database for tracking the 
   state of individual tasks, rather than querying the filesystem.
