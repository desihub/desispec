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

desi_qa_skyresid
++++++++++++++++

This script examines sky subtraction resdiuals
for an exposure, night or production.

usage
-----

Here is the usage::

    usage: desi_qa_skyresid [-h] [--reduxdir PATH] [--expid EXPID] [--night NIGHT]
                        [--channels CHANNELS] [--prod] [--gauss]
                        [--nights NIGHTS]

    Generate QA on Sky Subtraction residuals [v1.2]

    optional arguments:
      -h, --help           show this help message and exit
      --reduxdir PATH      Override default path ($DESI_SPECTRO_REDUX/$SPECPROD)
                           to processed data.
      --expid EXPID        Generate exposure plot on given exposure
      --night NIGHT        Generate night plot on given night
      --channels CHANNELS  List of channels to include
      --prod               Results for full production run
      --gauss              Expore Gaussianity for full production run
      --nights NIGHTS      List of nights to limit prod plots


Exposure
--------

Generate a plot of the sky subtraction residuals for an
input Exposure ID. e.g. ::

    desi_qa_sky --expid=123

Production
----------

Generate a plot of the sky subtraction residuals for the
Production.  If reduxdir is not provided, then the script
will use the $SPECPROD and $DESI_SPECTRO_REDUX environemental
variables.  Simply called::

    desi_qa_sky --prod

Gaussianity
-----------

Examine whether the residuals are distributed
as Gaussian statistics.  Here is an example::


    desi_qa_sky --gauss


desi_qa_prod
++++++++++++

This script is used to both generate and analyze the
QA outputs for a complete production.

usage
-----

Here is the usage::

    usage: desi_qa_prod [-h] [--reduxdir REDUXDIR] [--make_frameqa MAKE_FRAMEQA]
                        [--slurp] [--remove] [--clobber]
                        [--channel_hist CHANNEL_HIST] [--time_series TIME_SERIES]

    Generate/Analyze Production Level QA [v1.3]

    optional arguments:
      -h, --help            show this help message and exit
      --reduxdir REDUXDIR   Override default path ($DESI_SPECTRO_REDUX/$SPECPROD)
                            to processed data.
      --make_frameqa MAKE_FRAMEQA
                            Bitwise flag to control remaking the QA files (1) and
                            figures (2) for each frame in the production
      --slurp               slurp production QA files into one?
      --remove              remove frame QA files?
      --clobber             clobber existing QA files?
      --channel_hist CHANNEL_HIST
                            Generate channel histogram(s)
      --time_series TIME_SERIES
                            Generate time series plot. Input is QATYPE-METRIC,
                            e.g. SKYSUB-MED_RESID

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

Channel Histograms
------------------

Using the --channel_hist flag, the script will generate a series
of histogram plots on default metrics: FIBERFLAT: MAX_RMS,
SKYSUB: MED_RESID, FLUXCALIB: MAX_ZP_OFF::

    desi_qa_prod --channel_hist

Time Series Plot
----------------

Using the --time_series input with a *qatype* and *metric* produces
a Time Series plot of that metric for all nights/exposures/frames
in the production, by channel, e.g.::

    desi_qa_prod --time_series=SKYSUB-MED_RESID
    desi_qa_prod --time_series=FLUXCALIB-ZP
