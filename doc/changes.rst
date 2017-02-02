
Desispec Change Log
========================

0.12.1 (unreleased)
-------------------

* Fix brick update corruption (PR #314)
* close PSF file after initializing PSF object
* fix graph_path usage in workers
* update io.write_raw to enable writing simulated raw data with new headers

0.12.0 (2016-11-09)
-------------------

* Update integration test to use stdstar_templates_v1.1.fits
* Support asymmetric resolution matrices (PR #288)
* Quicklook updates (PR #294, #293, #285)
* Fix BUNIT and wavelength f4 vs. f8 
* Significant pipeline code refactor (PR #300 and #290)
* fix docstrings for sphinx build (PR #308)

0.11.0 (2016-10-14)
-------------------

* Update template Module file to reflect DESI+Anaconda infrastructure.
* update redmonster wrapper for reproducibility
* Brick.get_target_ids() returns them in the order they appear in input file
* set BUNIT header keywords (#284)
* Improved pipeline logging robustness
* MPI updates for robustness and non-NERSC operation
* more py3 fixes

0.10.0 (2016-09-10)
-------------------

PR #266 update for python 3.5:

* Many little updates to work for both python 2.7 and 3.5
* internally fibermap is now an astropy Table instead of FITS_rec table
* Bug fix for flux calibration QA
* requires desiutil >= 1.8.0

0.9.0 (2016-08-18)
------------------

PR #258 (requires specter >=0.6.0)

* propagate pixel model goodness of fit to flag outliers from unmasked cosmics
* desi_extract_spectra --model option to output 2D pixel model
* fix pipeline bug in call to desi_bootcalib (no --qafig option)
* adds extraction tests

Misc:

* desi_qa_skysub -- plots residuals (PR #259)
* More quicklook QA (PR #260 and #262)
* Added support for template groups in redmonster (PR #255)
* Lots more pipeline docs (PR #261)

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
