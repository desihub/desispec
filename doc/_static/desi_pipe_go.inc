.. code-block:: console

    $> usage: desi_pipe go [options] (use --help for details)

    Run a full production from start to finish. This will pack steps into 3 jobs
    per night and then run redshift fitting after all nights are done. Note that
    if you are running multiple nights you should use the regular queue.

    optional arguments:
      -h, --help            show this help message and exit
      --nights NIGHTS       comma separated (YYYYMMDD) or regex pattern- only
                            nights matching these patterns will be generated.
      --states STATES       comma separated list of states. This argument is
                            passed to chain (see desi_pipe chain --help for more
                            info).
      --resume              same as --states waiting,ready
      --dryrun              do not submit the jobs.
      --nersc NERSC         write a script for this NERSC system (edison | cori-
                            haswell | cori-knl). Default uses $NERSC_HOST
      --shell               generate normal bash scripts, even if run on a NERSC
                            system
      --nersc_queue NERSC_QUEUE
                            write a script for this NERSC queue (debug | regular)
      --nersc_maxtime NERSC_MAXTIME
                            Then maximum run time (in minutes) for a single job.
                            If the list of tasks cannot be run in this time,
                            multiple job scripts will be written. Default is the
                            maximum time for the specified queue.
      --nersc_maxnodes NERSC_MAXNODES
                            The maximum number of nodes to use. Default is the
                            maximum nodes for the specified queue.
      --nersc_shifter NERSC_SHIFTER
                            The shifter image to use for NERSC jobs
      --mpi_procs MPI_PROCS
                            The number of MPI processes to use for non-NERSC shell
                            scripts (default 1)
      --mpi_run MPI_RUN     The command to launch MPI programs for non-NERSC shell
                            scripts (default do not use MPI)
      --procs_per_node PROCS_PER_NODE
                            The number of processes to use per node. If not
                            specified it uses a default value for each machine.
      --outdir OUTDIR       Put task scripts and logs in this directory relative
                            to the production 'scripts' directory. Default puts
                            task directory in the main scripts directory.
      --debug               debugging messages in job logs
