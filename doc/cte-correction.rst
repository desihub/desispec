.. _cte-correction:

***************************************
Charge Transfer Inefficiency Correction
***************************************

Overview
========

Charge traps are found in the serial registers of
several LBNL CCD quadrants. In the DESI pipeline, two
algorithms have been developed to mitigate the effect.

CTE correction during preprocessing
===================================

A model for the effect of charge traps has been developed. For
amplifiers with keywords CCDCOLSX=BEGIN:END with X being an amplifier
id (A,B,C,D) in their calibration (as found with the CalibFinder with
looks into yaml file in the $DESI_SPECTRO_REDUX repository), a model
is fit on flatfield LED exposures and saved in tables
$DESI_SPECTRO_CALIB/$SPECPROD/calibnight/NIGHT/ctecorr-CAMERA-NIGHT.csv

For the same cameras with the keyword CCDCOLSX, the model is applied
during preprocessing to correct the CCD image. It relies on an
iterative fit where the spectra are extracted from the CCD image, then
projected back to obtain a CCD image model that is used to compute the
effect. The effect is then subtracted from the true image.

This is performed ONLY for CCD amps with the keyword CCDCOLSX

CTE correction during sky subtraction
=====================================

For CCDs with the keywords OFFCOLSX=BEGIN:END OR CCDCOLSX=BEGIN:END,
a offset and a slope as a function of wavelength are fit at the same
time as the sky model for fiber traces overlapping the affected region
(defined by the CCD columns BEGIN to END).  This method is sufficient
for charge traps that hold just a few electrons but is insufficient
for traps with larger effects.  For those using the keyword CCDCOLSX
also triggers a correction at the preprocessing level.
