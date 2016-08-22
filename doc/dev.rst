.. _dev:


Algorithm Development and Testing
===================================

When developing new algorithms and debugging problems, it can be use to run a pipeline task on a manually specified set of input files.  The desispec repo includes commandline tools which are mostly just hooks into the underlying main function for each processing task.  This allows for convenient serial testing of the underlying functions.  If a pipeline production is run with the log level set to "DEBUG", then the logs for each task will include the equivalent serial command that can be run to produce the output files.

PUT API DOCS HERE...


