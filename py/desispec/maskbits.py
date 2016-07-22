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

_bitdefs = yaml.load("""
#- CCD pixel mask
ccdmask:
    - [BAD,         0, "Pre-determined bad pixel (any reason)"]
    - [HOT,         1, "Hot pixel"]
    - [DEAD,        2, "Dead pixel"]
    - [SATURATED,   3, "Saturated pixel from object"]
    - [COSMIC,      4, "Cosmic ray"]
    - [PIXFLATZERO, 5, "pixflat is 0"]
    - [PIXFLATLOW,  6, "pixflat < 0.1"]

#- Mask bits that apply to an entire fiber
fibermask:
    - [BADFIBER,     0, "Broken or otherwise unusable fiber"]
    - [BADTRACE,     1, "Bad trace solution"]
    - [BADFLAT,      2, "Bad fiber flat"]
    - [BADARC,       3, "Bad arc solution"]
    - [MANYBADCOL,   4, ">10% of pixels are bad columns"]
    - [MANYREJECTED, 5, ">10% of pixels rejected in extraction"]

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
