# THIS MAY MOVE ELSEWHERE IF WE MERGE IMAGING, TARGETING, SPECTRO MASK
# BITS INTO ONE PLACE.

"""
desispec.maskbits
=================

Mask bits for the spectro pipeline.

Stephen Bailey, LBNL, January 2015

Example::

    from desispec.maskbits import ccdmask

    #- bit operations
    mask |= ccdmask.COSMIC     #- set ccdmask.COSMIC in integer/array `mask`
    mask & ccdmask.COSMIC      #- get ccdmask.COSMIC from integer/array `mask`
    (mask & ccdmask.COSMIC) != 0  #- test boolean status of ccdmask.COSMIC in integer/array `mask`
    ccdmask.COSMIC | specmask.SATURATED  #- Combine two bitmasks.

    #- bit attributes
    ccdmask.mask('COSMIC')     #- 2**0, same as ccdmask.COSMIC
    ccdmask.mask(0)            #- 2**0, same as ccdmask.COSMIC
    ccdmask.COSMIC             #- 2**0, same as ccdmask.mask('COSMIC')
    ccdmask.bitnum('COSMIC')   #- 0
    ccdmask.bitname(0)         #- 'COSMIC'
    ccdmask.names()            #- ['COSMIC', 'HOT', 'DEAD', 'SATURATED', ...]
    ccdmask.names(3)           #- ['COSMIC', 'HOT']
    ccdmask.comment(0)         #- "Cosmic ray"
    ccdmask.comment('BADPIX')  #- "Cosmic ray"


"""

#- Move these definitions into a separate yaml file
import yaml
from desiutil.bitmask import BitMask

_bitdefs = yaml.safe_load("""
#- CCD pixel mask
ccdmask:
    - [BAD,             0, "Pre-determined bad pixel (any reason)"]
    - [HOT,             1, "Hot pixel"]
    - [DEAD,            2, "Dead pixel"]
    - [SATURATED,       3, "Saturated pixel from object"]
    - [COSMIC,          4, "Cosmic ray"]
    - [PIXFLATZERO,     5, "pixflat is 0"]
    - [PIXFLATLOW,      6, "pixflat < 0.1"]
    - [HIGHVAR,         7, "High variability in pixel value"]
    - [BADREADNOISE,    8, "Very high CCD amplifier read noise"]

#- Mask bits that apply to an entire fiber
#- Note: some of these are informational and aren't necessarily bad
fibermask:
    - [UNASSIGNED,      0, "Fiber is not assigned to a known target or sky location"]
    - [STUCKPOSITIONER, 1, "INFO: Stuck positioner (but could still be on a good sky location)"]
    - [BROKENFIBER,     2, "Broken fiber"]
    - [RESTRICTED,      3, "INFO: Positioner has restricted reach (but might still be on valid target)"]
    - [MISSINGPOSITION, 8, "Fiber location information is missing"]
    - [BADPOSITION,     9, "Fiber >100 microns from target location"]
    - [POORPOSITION,   10, "Fiber >30 microns from target location"]
    - [LOWEFFTIME,     15, "Effective time for this fiber is too low"]
    - [BADFIBER,       16, "Unusable fiber"]
    - [BADTRACE,       17, "Bad trace solution"]
    - [BADFLAT,        18, "Bad fiber flat"]
    - [BADARC,         19, "Bad arc solution"]
    - [MANYBADCOL,     20, ">10% of pixels are bad columns"]
    - [MANYREJECTED,   21, ">10% of pixels rejected in extraction"]
    - [BADAMPB,        22, "Issues in the amplifier readouts of camera B make this unusable"]
    - [BADAMPR,        23, "Issues in the amplifier readouts of camera R make this unusable"]
    - [BADAMPZ,        24, "Issues in the amplifier readouts of camera Z make this unusable"]
#- Following bits are reserved for spectro QA (see more precise definition in data/qa/qa-params.yaml)
    - [BADPETALPOS,    25, "Too many fibers with bad positioning in petal"]
    - [BADPETALSKY,    26, "Bad sky model across petal"]
    - [BADPETALSTDSTAR,27, "To few standard stars or rms between stars too large in the petal"]
    - [BADPETALFLUXCAL,28, "Unphysical flux calibration for the petal (calib vector too high or too low)"]
    - [BADPETALSNR,    29, "TSNR is too low for this petal compared to the others"]
    - [BADREADNOISE,   30, "Bad read noise in one of the 3 cameras"]
    - [RESERVED31,     31, "Reserved sign bit; do not use"]

#- Spectral pixel mask: bits that apply to individual spectral bins
specmask:
    - [SOMEBADPIX,   0, "Some input pixels were masked or ivar=0"]
    - [ALLBADPIX,    1, "All input pixels were masked or ivar=0"]
    - [COSMIC,       2, "Input pixels included a masked cosmic"]
    - [LOWFLAT,      3, "Fiber flat < 0.5"]
    - [BADFIBERFLAT, 4, "Bad fiber flat solution"]
    - [BRIGHTSKY,    5, "Bright sky level (details TBD)"]
    - [BADSKY,       6, "Bad sky model"]
    - [BAD2DFIT,     7, "Bad fit of extraction 2D model to pixel data"]
    - [NODATA,       8, "No data exists"]
    - [BADFIBER,     9, "fibermask has a non-zero bit"]

#- zmask: reasons why redshift fitting failed
""")

#-------------------------------------------------------------------------
#- The actual masks
try:
    specmask = BitMask('specmask', _bitdefs)
    ccdmask = BitMask('ccdmask', _bitdefs)
    fibermask = BitMask('fibermask', _bitdefs)
except TypeError:
    #
    # This is needed to allow documentation to build even if desiutil is not
    # installed.
    #
    specmask = object()
    ccdmask = object()
    fibermask = object()
