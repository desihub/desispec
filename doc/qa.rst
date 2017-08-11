.. _qa:

*****************
Quality Assurance
*****************

Overview
=================

The DESI spectroscopic pipeline includes a series of
routines that monitor the quality of the pipeline products
and may be used to inspect outputs across exposures, nights,
or a full production.


Scripts
=======

desi_qa_frame
+++++++++++++

desi_qa_prod
++++++++++++

This script is used to both generate and analyze the
QA outputs for a complete production.

usage
-----

Here is the usage::

    usage: desi_qa_prod [-h] --specprod_dir SPECPROD_DIR
                        [--make_frameqa MAKE_FRAMEQA] [--slurp] [--remove]
                        [--clobber] [--channel_hist CHANNEL_HIST]

    Generate/Analyze Production Level QA [v1.2]

    optional arguments:
      -h, --help            show this help message and exit
      --specprod_dir SPECPROD_DIR
                            Path containing the exposures/directory to use
      --make_frameqa MAKE_FRAMEQA
                            Bitwise flag to control remaking the QA files (1) and
                            figures (2) for each frame in the production
      --slurp               slurp production QA files into one?
      --remove              remove frame QA files?
      --clobber             clobber existing QA files?
      --channel_hist CHANNEL_HIST
                            Generate channel histogram(s)



frameqa
-------

One generates the frame QA, the YAML and/or figure files
with the --make_frameqa flag::

    desi_qa_prod --make_frameqa=1  # Generate all the QA YAML files
    desi_qa_prod --make_frameqa=2  # Generate all the QA figure files
    desi_qa_prod --make_frameqa=3  # Generate YAML and figures

The optional --remove and --clobber flags can be used to remove/clobber
the QA files.

slurp
-----

By using the --slurp flag, one generates a full
YAML file of all the QA outputs::

    desi_qa_prod --slurp   # Collate all the QA YAML files
    desi_qa_prod --slurp --remove  # Collate and remove the individual files
