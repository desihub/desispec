.. _qa:

*****************
Quality Assurance
*****************

Overview
========

The DESI spectroscopic pipeline includes a series of
routines that monitor the quality of the pipeline products
and may be used to inspect outputs across exposures, nights,
or a full production.


Scripts
=======

desi_qa_frame
+++++++++++++

Generate the QA for an input frame file.
The code can be written anywhere and the
output is written to its "proper" location.

usage
-----

Here is the usage::

    usage: desi_qa_frame [-h] --frame_file FRAME_FILE [--reduxdir PATH]
                         [--make_plots]

    Generate Frame Level QA [v1.0]

    optional arguments:
      -h, --help            show this help message and exit
      --frame_file FRAME_FILE
                            Frame filename. Full path is not required nor desired.
      --reduxdir PATH       Override default path ($DESI_SPECTRO_REDUX/$SPECPROD)
                            to processed data.
      --make_plots          Generate QA figs too?


examples
--------

Generate the QA YAML file::

    desi_qa_frame --frame_file=frame-r7-00000077.fits

Generate the QA YAML file and figures::

    desi_qa_frame --frame_file=frame-r7-00000077.fits --make_plots

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
