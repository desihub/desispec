.. _pipeline:

Pipeline Use
=========================

The DESI spectroscopic pipeline is used to run real or simulated data through one or more stages of a standard sequence of processing operations.  The pipeline is designed to function on a supercomputer (e.g. NERSC) or cluster, but can also run locally for small data tests.


Overview
------------------------

The starting point of the pipeline is real or simulated raw exposures.  These exposures are either arcs, flats, or science exposures.  The exposures are grouped by night.  Each exposure consists of images from up to 10 spectrographs with 3 cameras (r, b, z) each.  The processing "steps" that are defined in the pipeline are:

* **preproc**: (all exposure types)  Apply image pre-processing.
* **psf**: (only for arcs)  Estimate the PSF.
* **psfnight**: (one per night, only for arcs)  Combine all PSF estimates for the night.
* **traceshift**: (only for flats and science)  Compute the trace locations in preparation for extractions.
* **extract**: (only for flats and science)  Extract the maximum likelihood spectra from the pixel data.
* **fiberflat**: (only for flats)  Compute a fiber flat from an extracted continuum lamp exposure.
* **fiberflatnight**: (one per night, only for flats)  Build the nightly fiberflat.
* **sky**: (only for science)  Apply the fiberflat to sky fibers to compute the sky model.
* **starfit**: (only for science)  For each spectrograph, apply fiberflat and sky subtraction to standards and fit the result to stellar models.
* **fluxcalib**: (only for science)  Apply the fiberflat and sky subtraction to science fibers and then calibrate against the stellar fits.
* **cframe**: (only for science)  Apply the calibration to the extracted frames.
* **spectra**:  The calibrated output science spectra are re-grouped into files based on their sky location (healpix pixels).
* **redshift**:  The redshifts are estimated from re-grouped spectral files.

For a given pipeline "step", there are frequently many independent processing "tasks" that can be batched together.  Each processing task usually has some input dependencies (data files) and generates some outputs.  In general, a single task has exactly one output file.  This allows reconstruction of the state of the processing from examining the filesystem.  The pipeline infrastructure is designed to track the dependencies between tasks as well as the current state of each task.  When the pipeline actually does the processing, it generates scripts (either slurm scripts for submission to a queueing system or plain bash scripts) that batch together many tasks.

**Example:**  Imagine you had 5 arc exposures you wanted to estimate the PSF on in one job.  Estimating the PSF for one exposure consists of 30 individual tasks (one per spectrograph and camera), so there are 150 tasks in this example.  Additionally, each of those tasks can run in parallel using one MPI process per fiber bundle and several threads per process.

For a single set of raw data, we might want to have multiple "data reductions" that use different versions of the pipeline software or use different options for the processing.  Each independent reduction of some raw data is called a "production".  A "production" on disk consists of a directory hierarchy where the data outputs, logs, and scripts are stored.  A database is used to track the dependencies and states of all tasks in a production.


User Interface
--------------------

As discussed above, a single data processing "production" essentially consists of a database and a directory structure of job scripts, logs, and output data files.  The primary user interface for running the pipeline on a specific production is the `desi_pipe` command line tool.  This takes a command followed by command-specific options.  If you want to write a custom script which controls the pipeline in a particular way, then you can also call the same high-level interface used by `desi_pipe`.  This interface is found in the `desispec.pipeline.control` module.

Command Help
~~~~~~~~~~~~~~~~~~

An overview of available commands can be displayed with:

.. include:: _static/desi_pipe_help.inc

Creating a Production
~~~~~~~~~~~~~~~~~~~~~~~~

The first step to using the pipeline is to create a "production" directory for the data processing outputs:

.. include:: _static/desi_pipe_create.inc

Before creating a production you should have on hand some information about the data and tools you want to use:

    1.  The location of the raw data.
    2.  The location of the "spectro" directory containing various auxiliary
        files (this is location you want to become $DESI_ROOT for the
        production).
    3.  The location of the "top-level" directory where you want to put your
        productions.
    4.  The name of your production (which will become a subdirectory).
    5.  The spectro calibration data from an svn checkout.
    6.  The basis templates data from an svn checkout.

Here is an example, using some simulated data from a survey validation data challenge:

.. include:: _static/desi_pipe_create_ex.inc

This creates the production directory and subdirectories for all output data products considering the raw data that exists at the time you run the command.  If you add new raw data to your data directory, see the "update" command below.

Just creating a production does not change anything in your environment and the pipeline has no idea how many productions you have created.  In order to "activate" your production and use it for future desi_pipe commands, you must source the setup.sh file.  In the example above, you would now do:

.. code-block:: console

    source ./desi_test/redux/svdc/setup.sh

And now all future commands will use this production.


Monitoring a Production
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a quick snapshot of the production you can use the "top" command to display updates on the number of tasks in different states.  This is refreshed every 10 seconds.  For a single snapshot we can use the "--once" option.  Building on our example above:

.. include:: _static/desi_pipe_top_ex.inc

Here we see that no tasks have been run yet.  The "preproc" tasks are in the "ready" state (their dependencies are met).  The remaining tasks are in the "waiting" state, since their dependencies are not yet complete.

Whenever a single task runs, it will write a log specific to that task.  This file can always be found in the same place within the production directory (run/logs/night/[night]/).  If you re-run a specific task (either because it failed or you simpled wanted to run it again), then the per-task log is overwritten in the same location.  The pipeline only tracks the current state of a task from its most recent execution, and the per-task log is the output from that most recent run.

The logs from a "job" (the simultaneous batched execution of many tasks) is stored in a per-job directory located in run/scripts/ and named according to the range of processing steps run in the job, the date and job ID.  These logs will contain output about the overall number of tasks that were run, how many tasks succeeded and failed, and any errors due to the scheduling system or runtime environment.  A new log directory is created for every job that is submitted.


Processing Data with Minimal Interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When doing large-scale processing (or re-processing) of many nights of exposures, it is convenient to have a high-level wrapper that submits many jobs to the queueing system with dependencies between jobs to ensure that the processing happens in the correct sequence.  This can be done using the "go" command:

.. include:: _static/desi_pipe_go.inc

There are many options to this command that control things like the NERSC sytem to use, the job submission queue, the maximum runtime and number of nodes to use, etc.  By default, jobs are submitted to the regular queue with maximum job sizes and run times given by the limits for that queue.  Before using non-default values for these at NERSC, you should read and familiarize yourself with the different queues and their limits in the NERSC online documentation.

If the "--nersc" option is not specified, then bash scripts will be generated.  You can use other options to enable the use of MPI in these bash scripts and specify the node sizes and properties.

Continuing our example, we could submit several jobs to process all tasks on the cori-knl nodes with:

.. include:: _static/desi_pipe_go_ex.inc

This will submit 3 jobs per night and a final job to do the spectral regrouping and redshift fitting.  If some of these jobs fail for some reason, you can cleanup the production (see the cleanup command below with the "--submitted" option) and then re-run the "go" command with the "--resume" option:

.. code-block:: console

    $> desi_pipe go --nersc cori-knl --resume


Updating a Production
~~~~~~~~~~~~~~~~~~~~~~~~~

When new raw data arrives in the input data directory, we must add the processing tasks for this new data to our database.  This is done using the "update" command:

.. include:: _static/desi_pipe_update.inc

By default, the update command looks across all nights in the raw data.  This can be time consuming if you have only added a new night of data or a single exposure.  Use the options above to restrict the update to only certain nights or exposures.


Cleaning Up When Jobs Fail
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There will always be cases where jobs submitted to a queuing system on a large supercomputer will fail.  This could be due to a problem with the scheduler, a problem with the filesystem that makes jobs take longer and run out of time, etc.  During the running of a job, the state of individual **tasks** are updated as they complete.  Even when a job dies or is killed, any completed tasks are marked as done.  However, tasks that were in a "running" state when the job ended need to be reset into the "ready" state.  This is done using the "cleanup" command:

.. include:: _static/desi_pipe_cleanup.inc

You should only run this command if there are no pipeline jobs from the current production running.  Additionally, if you are using the "desi_pipe go" command, then tasks already submitted are ignored in future runs.  In that case you must use the "--submitted" option to the cleanup command.


Manually Running Processing Steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Manually running pipeline steps involves first selecting tasks and then running some set of processing steps on these using all the various NERSC queue options.

**TO-DO:** Document the commands for all this, including:

    * tasks
    * dryrun
    * check
    * script
    * run
    * chain

When Something Goes Wrong
---------------------------------

If a job dies, even if due to an external system issue, it is always good to look at the job logs and verify that everything went well up to the point that it failed.  The job logs are organized in the run/scripts directory and named after the steps being run, the date and the job ID.  For NERSC jobs, you can browse https://my.nersc.gov to get a list of all jobs you have submitted.  After verifying that the job ended due to external factors, you can cleanup (see above) and retry.

A pipeline job usually runs many individual tasks.  Each task can succeed or fail independently.  A pipeline job might complete successfully (from the viewpoint of the queueing system) even if some individual tasks fail.  If all tasks fail, the job will exit with a non-zero exit code so that future jobs with a dependency hold are not launched.

If you have a job where one or more tasks failed, you should examine the logs for that task.  As discussed before, the per-task logs are in run/logs.

In an extreme case where you believe the production database is invalid or corrupted, you can force the re-creation of the database using only the files that exist on disk.  Ensure that all jobs are killed and then do:

.. code-block:: console

    $> desi_pipe sync

This scans the outputs of the production and generates a new DB from scratch.


Example 1: Large (Re)Processing of Many Exposures
---------------------------------------------------------

Our in-line example in the usage section shows how "desi_pipe go" can be used to submit sets of jobs (3 per night) in a dependency chain and then a final job to do the spectral regrouping and redshift fitting.


Example 2: Process One Exposure
--------------------------------------------




Example 3: Nightly Processing
---------------------------------------

TO-DO:  Document what happens when the "desi_night" command is triggered by the data transfer.


Example 4: Skipping Steps Using External Tools
--------------------------------------------------

If you use some of the DESI "quick" simulation tools to produce uncalibrated frame data (or calibrated frame data) directly, then there is a special step that must be taken.  In this scenario, the production database has no idea that you have injected data into the processing chain.  The only option is to use a recovery step ("desi_pipe sync") which will scan the production output directories and rebuild the database with your injected files included in the dependencies and marked as "done".
