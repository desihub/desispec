.. code-block:: console

    $> desi_pipe --help

    usage: desi_pipe <command> [options]

    Where supported commands are (use desi_pipe <command> --help for details):
       (------- High-Level -------)
       create   Create a new production.
       go       Run a full production.
       update   Update an existing production.
       top      Live display of production database.
       status   Overview of production.
       (------- Mid-Level --------)
       chain    Run all ready tasks for multiple pipeline steps.
       cleanup  Reset "running" (or optionally "failed") tasks back to "ready".
       (------- Low-Level --------)
       tasks    Get all possible tasks for a given type and states.
       check    Check the status of tasks.
       dryrun   Return the equivalent command line entrypoint for tasks.
       script   Generate a shell or slurm script.
       run      Generate a script and run it.
       getready Auto-Update of prod DB.
       sync     Synchronize DB state based on the filesystem.
       env      Print current production location.

    DESI pipeline control

    positional arguments:
      command     Subcommand to run

    optional arguments:
      -h, --help  show this help message and exit
