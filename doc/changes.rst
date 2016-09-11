===================
desispec change log
===================

0.10.1 (unreleased)
-------------------

* no changes yet

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
