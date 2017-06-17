===================
desispec Change Log
===================

0.15.1 (unreleased)
-------------------

* Update desispec.io.findfile spectra path to match dc17a

0.15.0 (2017-06-15)
-------------------

* Refactor database subpackage and enable loading of both quicksurvey and
  pipeline outputs (PR `#400`_).
* Clean up pipeline script naming to be grouped by night.
* Modify pipeline to use Spectra objects grouped by HEALPix pixels instead
  of bricks.  Add entry point to regroup cframe data by pixel (PR `#394`_).
* Add a new class, Spectra, which encapsulates a grouping of 1D spectra
  in one or more bands.  Includes selection, updating, and I/O.
* Removed ``desispec.brick`` as it's now in :mod:`desiutil.brick` (PR `#392`_).
* Added function to calculate brick vertices at a given location (PR `#388`_).
* Added function to calculate brick areas at a given location (PR `#384`_).
* Add scripts for submitting nightly job chains.
* Production creation now correctly handles slicing by spectrograph.
* Pipeline job concurrency now computed based on task run time and
  efficient packing.
* Set default brick size to 0.25 sq. deg. in desispec.brick (PR `#378`_).
* Added function to calculate BRICKID at a given location (PR `#378`_).
* Additional LOCATION, DEVICE_LOC, and PETAL_LOC columns for fibermap (PR `#379`_).
* Create util.py in tests/ which is intended to contain methods to facilitate test runs
* Add vette() method for Frame class (PR `#386`_)
* Began a desispec parameter file:  data/params/desispec_param.yml
* Flux calibration improvements (PR `#390`_).

.. _`#386`: https://github.com/desihub/desispec/pull/386
.. _`#388`: https://github.com/desihub/desispec/pull/388
.. _`#384`: https://github.com/desihub/desispec/pull/384
.. _`#378`: https://github.com/desihub/desispec/pull/378
.. _`#379`: https://github.com/desihub/desispec/pull/379
.. _`#390`: https://github.com/desihub/desispec/pull/390
.. _`#392`: https://github.com/desihub/desispec/pull/392
.. _`#394`: https://github.com/desihub/desispec/pull/394
.. _`#400`: https://github.com/desihub/desispec/pull/400

0.14.0 (2017-04-13)
-------------------

* Replace all instances of :mod:`desispec.log` with ``desiutil.log``;
  :func:`~desispec.log.get_logger` now prints a warning that users need
  to switch.
* Working DTS delivery script and DTS simulator (PR `#367`_).
* Preproc updates for crosstalk and teststand data (PR `#370`_).
* Flux calibration algorithm updates (PR `#371`_).
* Adds quicklook integration test (PR `#361`_).
* Fixes brickname calculation (PR `#373`_).

.. _`#367`: https://github.com/desihub/desispec/pull/367
.. _`#370`: https://github.com/desihub/desispec/pull/370
.. _`#371`: https://github.com/desihub/desispec/pull/371
.. _`#361`: https://github.com/desihub/desispec/pull/361
.. _`#373`: https://github.com/desihub/desispec/pull/361

0.13.2 (2017-03-27)
-------------------

* Add framework for DTS delivery and nightly processing scripts (PR `#365`_).
* Force documentation errors to cause Travis errors (PR `#364`_).

.. _`#364`: https://github.com/desihub/desispec/pull/364
.. _`#365`: https://github.com/desihub/desispec/pull/365

0.13.1 (2017-03-03)
-------------------

* Fix installation of ``data/ccd/ccd_calibration.yaml``.

0.13.0 (2017-03-03)
-------------------

* Fix brick update corruption (PR `#314`_).
* Close PSF file after initializing PSF object.
* Refactor :mod:`desispec.io.database` to use SQLAlchemy_.
* Fix :func:`~desispec.pipeline.graph.graph_path` usage in workers.
* Update :func:`desispec.io.raw.write_raw` to enable writing simulated raw
  data with new headers.
* Allow ``test_bootcalib`` to run even if NERSC portal is returning 403 errors.
* Add ``bricksize`` property to :class:`desispec.brick.Bricks`; allow
  :meth:`~desispec.brick.Bricks.brickname` to specify bricksize.
* Do SVD inverses when cholesky decompositions fail in fiberflat, sky
  subtraction, and flux calibration.
* Algorithm updates for teststand and BOSS data
* pipeline updates for docker/shifter
* quicklook updates

.. _`#314`: https://github.com/desihub/desispec/pull/314
.. _SQLAlchemy: http://www.sqlalchemy.org

0.12.0 (2016-11-09)
-------------------

* Update integration test to use stdstar_templates_v1.1.fits.
* Support asymmetric resolution matrices (PR `#288`_).
* Quicklook updates (PR `#294`_, `#293`_, `#285`_).
* Fix BUNIT and wavelength f4 *versus* f8.
* Significant pipeline code refactor (PR `#300`_ and `#290`_).
* fix docstrings for sphinx build (PR `#308`_).

.. _`#288`: https://github.com/desihub/desispec/pull/288
.. _`#294`: https://github.com/desihub/desispec/pull/294
.. _`#293`: https://github.com/desihub/desispec/pull/293
.. _`#285`: https://github.com/desihub/desispec/pull/285
.. _`#300`: https://github.com/desihub/desispec/pull/300
.. _`#290`: https://github.com/desihub/desispec/pull/290
.. _`#308`: https://github.com/desihub/desispec/pull/308


0.11.0 (2016-10-14)
-------------------

* Update template Module file to reflect DESI+Anaconda infrastructure.
* Update redmonster wrapper for reproducibility.
* :meth:`desispec.io.brick.BrickBase.get_target_ids` returns target IDs in the order they appear in input file.
* Set BUNIT header keywords (PR `#284`_).
* Improved pipeline logging robustness.
* MPI updates for robustness and non-NERSC operation.
* More py3 fixes.

.. _`#284`: https://github.com/desihub/desispec/pull/284

0.10.0 (2016-09-10)
-------------------

PR `#266`_ update for Python 3.5:

* Many little updates to work for both python 2.7 and 3.5.
* Internally fibermap is now a :class:`~astropy.table.Table` instead of :class:`~astropy.io.fits.FITS_rec` table.
* Bug fix for flux calibration QA.
* Requires desiutil_ >= 1.8.0.

.. _`#266`: https://github.com/desihub/desispec/pull/266
.. _desiutil: https://github.com/desihub/desiutil

0.9.0 (2016-08-18)
------------------

PR `#258`_ (requires specter_ >= 0.6.0)

* Propagate pixel model goodness of fit to flag outliers from unmasked cosmics.
* desi_extract_spectra --model option to output 2D pixel model
* fix pipeline bug in call to desi_bootcalib (no --qafig option)
* adds extraction tests

Misc:

* desi_qa_skysub -- plots residuals (PR #259)
* More quicklook QA (PR #260 and #262)
* Added support for template groups in redmonster (PR #255)
* Lots more pipeline docs (PR #261)

.. _specter: https://github.com/desihub/specter
.. _`#258`: https://github.com/desihub/desispec/pull/258

0.8.1 (2016-07-18)
------------------

* added QA_Prod
* refactor of fluxcalib QA
* fixed pipeline QA figure output (pdf vs. yaml)

0.8.0 (2016-07-14)
------------------

* bootcalib robustness improvements
* improved fibermap propagation
* PRODNAME -> SPECPROD, TYPE -> SPECTYPE
* meaningful batch job names for each step
* better test coverage; more robust to test data download failures
* more quicklook metrics
* used for "oak1" production

0.7.0 and prior
----------------

* No changes.rst yet
