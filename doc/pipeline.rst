.. _pipeline:


Pipeline Tools
=================

A pipeline "production" is defined as a processing of DESI data using a consistent software stack and set of options.  After setting up a production, the pipeline steps are run in order and the status of individual tasks are tracked.  If a task fails, all later processing steps requiring the outputs of that task will also be marked as failing.


Creating a Production
-------------------------

To create a production for some raw data, we use the desi_pipe commandline tool::

    %> desi_pipe --help

.. include:: _static/desi_pipe_help.inc


