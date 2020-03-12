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

Expose QA
=========

Here is the magic to expose a set of QA products
made at NERSC to the world:

1. cp -rp QA into www area :: /project/projectdirs/desi/www
2. fix_permissions.sh -a QA  [This may no longer be necessary]

These are then exposed at https://portal.nersc.gov/cfs/desi/rest_of_path

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

    Generate Frame Level QA [v0.4.2]

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

desi_qa_exposure
++++++++++++++++

Generates Exposure level QA.   The current
implementation is only for the flat flux.

usage
-----

Here is the usage::

    usage: desi_qa_exposure [-h] --expid EXPID [--qatype QATYPE]
                            [--channels CHANNELS] [--reduxdir PATH] [--rebuild]
                            [--qamulti_root QAMULTI_ROOT] [--slurp SLURP]

    Generate Exposure Level QA [v0.5.0]

    optional arguments:
      -h, --help            show this help message and exit
      --expid EXPID         Exposure ID
      --qatype QATYPE       Type of QA to generate [fiberflat, s2n]
      --channels CHANNELS   List of channels to include. Default = b,r,z]
      --reduxdir PATH       Override default path ($DESI_SPECTRO_REDUX/$SPECPROD)
                            to processed data.
      --rebuild             Regenerate the QA files for this exposure?
      --qamulti_root QAMULTI_ROOT
                            Root name for a set of slurped QA files (e.g.
                            mini_qa). Uses $SPECPROD/QA for path
      --slurp SLURP         Root name for slurp QA file to add to (e.g. mini_qa).
                            Uses $SPECPROD/QA for path

fiberflat
---------

Generate QA on the fiber flat across the exposure for one or more channels.::

     desi_qa_exposure --expid=96 --qatype=fiberflat



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

    Generate QA on Sky Subtraction residuals [v0.4.2]

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


desi_qa_night
+++++++++++++

This script is used to analyze the QA outputs
from a given night.  Note that we use desi_qa_prod (below)
to generate the QA YAML files.

usage
-----

Here is the usage::

    usage: desi_qa_night [-h] [--expid_series] [--bright_dark BRIGHT_DARK]
                         [--qaprod_dir QAPROD_DIR] [--specprod_dir SPECPROD_DIR]
                         [--night NIGHT]

    Generate/Analyze Production Level QA [v0.5.0]

    optional arguments:
      -h, --help            show this help message and exit
      --expid_series        Generate exposure series plots.
      --bright_dark BRIGHT_DARK
                            Restrict to bright/dark (flag: 0=all; 1=bright;
                            2=dark; only used in time_series)
      --qaprod_dir QAPROD_DIR
                            Path to where QA is generated. Default is qaprod_dir
      --specprod_dir SPECPROD_DIR
                            Path to spectro production folder. Default is
                            specprod_dir
      --night NIGHT         Night; required


Current recommendation
----------------------

First generate the QA for the given night with desi_qa_prod, e.g.::

    desi_qa_prod --make_frameqa 1 --specprod_dir /global/projecta/projectdirs/desi/spectro/redux/daily --night 20200224 --qaprod_dir /global/projecta/projectdirs/desi/spectro/redux/xavier/daily/QA --slurp


Then generate the Night plots::

    desi_qa_night --specprod_dir /global/projecta/projectdirs/desi/spectro/redux/daily --qaprod_dir /global/projecta/projectdirs/desi/spectro/redux/xavier/daily/QA --night 20200224 --expid_series


desi_qa_prod
++++++++++++

This script is used to both generate and analyze the
QA outputs for a complete production.

usage
-----

Here is the usage::

    usage: desi_qa_prod [-h] [--make_frameqa MAKE_FRAMEQA] [--slurp] [--remove]
                        [--clobber] [--channel_hist CHANNEL_HIST]
                        [--time_series TIME_SERIES] [--bright_dark BRIGHT_DARK]
                        [--html] [--qaprod_dir QAPROD_DIR] [--S2N_plot]
                        [--ZP_plot] [--xaxis XAXIS]

    Generate/Analyze Production Level QA [v0.5.0]

    optional arguments:
      -h, --help            show this help message and exit
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
      --bright_dark BRIGHT_DARK
                            Restrict to bright/dark (flag: 0=all; 1=bright;
                            2=dark; only used in time_series)
      --html                Generate HTML files?
      --qaprod_dir QAPROD_DIR
                            Path to where QA is generated. Default is qaprod_dir
      --S2N_plot            Generate a S/N plot for the production (vs. xaxis)
      --ZP_plot             Generate a ZP plot for the production (vs. xaxis)
      --xaxis XAXIS         Specify x-axis for S/N and ZP plots


frameqa
-------

One generates the frame QA, the YAML and/or figure files
with the --make_frameqa flag.  These files are created
in a folder tree QA/ that is parallel to the exposures and
calib2d folders.::

    desi_qa_prod --make_frameqa=1  # Generate all the QA YAML files
    desi_qa_prod --make_frameqa=2  # Generate all the QA figure files
    desi_qa_prod --make_frameqa=3  # Generate YAML and figures

The optional --remove and --clobber flags can be used to remove/clobber
the QA files.

slurp
-----

By using the --slurp flag, one generates a full
YAML file of all the QA outputs::

    desi_qa_prod --slurp   # Collate all the QA YAML files into a series of JSON files, one per night
    desi_qa_prod --slurp --remove  # Collate and remove the individual files

html
----

A set of static HTML files that provide simple links
to the QA figures may be generated::

    desi_qa_prod --html  # Generate HTML files

The top-level QA file (in the QA/ folder) includes any PNG
files located at the top-level of that folder.

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

By default, these files are placed in the QA/ folder in
the $DESI_SPECTRO_REDUX/$SPECPROD folder.

<S/N> Plot
----------

Generate a plot of <S/N> for a standard set of fiducials --
object type at a given magnitude in a given channel
(e.g. ELG, 23 mag in channel r).  The x-axis is controlled
by the `--xaxis` option and may be MJD, texp (exposure time),
or expid.  Here is a sample call::

    desi_qa_prod --S2N_plot --xaxis texp

ZP Plot
-------

Similar to the <S/N> plot above but for the Zero Point
calculated in the three channels.  Again, `--xaxis`
controls the abscissa axis.  An example::

    desi_qa_prod --ZP_plot --xaxis texp

