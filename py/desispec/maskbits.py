# THIS MAY MOVE ELSEWHERE IF WE MERGE IMAGING, TARGETING, SPECTRO MASK
# BITS INTO ONE PLACE.

"""
desispec.maskbits
=================

Mask bits for the spectro pipeline.

Stephen Bailey, LBNL, January 2015

Example::

    from desispec.maskbits import ccdmask

    ccdmask.COSMIC | specmask.SATURATED
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
_bitdefs = yaml.load("""
#- CCD pixel mask
ccdmask:
    - [BAD,       0, "Pre-determined bad pixel (any reason)"]
    - [HOT,       1, "Hot pixel"]
    - [DEAD,      2, "Dead pixel"]
    - [SATURATED, 3, "Saturated pixel from object"]
    - [COSMIC,    4, "Cosmic ray"]

#- Mask bits that apply to an entire fiber
fibermask:
    - [BADFIBER,     0, "Broken or otherwise unusable fiber"]
    - [BADTRACE,     1, "Bad trace solution"]
    - [BADFLAT,      2, "Bad fiber flat"]
    - [BADARC,       3, "Bad arc solution"]
    - [MANYBADCOL,   4, ">10% of pixels are bad columns"]
    - [MANYREJECTED, 5, ">10% of pixels rejected in extraction"]

#- Spectral pixel mask: bits that apply to individual spectral bins
spmask:
    - [SOMEBADPIX,   0, "Some input pixels were masked or ivar=0"]
    - [ALLBADPIX,    1, "All input pixels were masked or ivar=0"]
    - [COSMIC,       2, "Input pixels included a masked cosmic"]
    - [LOWFLAT,      3, "Fiber flat < 0.5"]
    - [BADFIBERFLAT, 4, "Bad fiber flat solution"]
    - [BRIGHTSKY,    5, "Bright sky level (details TBD)"]
    - [BADSKY,       6, "Bad sky model"]

#- zmask: reasons why redshift fitting failed
""")

#- Class to provide mask bit utility functions
class BitMask(object):
    """BitMask object.
    """
    def __init__(self, name, bitdefs):
        """
        Args:
            name : name of this mask, must be key in bitdefs
            bitdefs : dictionary of different mask bit definitions each value is a list of [bitname, bitnum, comment]

        Users are not expected to create BitMask objects directly.

        See maskbits.ccdmask, maskbits.spmask, maskbits.fibermask, ...
        """
        self._name = name
        self._bitname = dict()  #- key num -> value name
        self._bitnum = dict()   #- key name -> value num
        self._comment = dict()  #- key name or num -> comment
        for bitname, bitnum, comment in bitdefs[name]:
            assert bitname not in self._bitnum
            assert bitnum not in self._bitname
            self._bitnum[bitname] = bitnum
            self._bitname[bitnum] = bitname
            self._comment[bitname] = comment
            self._comment[bitnum] = comment

    def bitnum(self, bitname):
        """Return bit number (int) for bitname (string)"""
        return self._bitnum[bitname]

    def bitname(self, bitnum):
        """Return bit name (string) for this bitnum (integer)"""
        return self._bitname[bitnum]

    def comment(self, bitname_or_num):
        """Return comment for this bit name or bit number"""
        return self._comment[bitname_or_num]

    def mask(self, name_or_num):
        """Return mask value, i.e. 2**bitnum for this name or number"""
        if isinstance(name_or_num, int):
            return 2**name_or_num
        else:
            return 2**self._bitnum[name_or_num]

    def names(self, mask=None):
        """Return list of names of masked bits.
        If mask=None, return names of all known bits.
        """
        names = list()
        if mask is None:
            for bitnum in sorted(self._bitname.keys()):
                names.append(self._bitname[bitnum])
        else:
            bitnum = 0
            while bitnum < mask:
                if (2**bitnum & mask):
                    if bitnum in self._bitname.keys():
                        names.append(self._bitname[bitnum])
                    else:
                        names.append('UNKNOWN'+str(bitnum))
                bitnum += 1

        return names

    #- Allow access via mask.BITNAME
    def __getattr__(self, name):
        if name in self._bitnum:
            return 2**self._bitnum[name]
        else:
            raise AttributeError('Unknown mask bit name '+name)

    #- What to print
    def __repr__(self):
        result = list()
        result.append( self._name+':' )
        for i in sorted(self._bitname.keys()):
            result.append('    - [{:16s} {:2d}, "{}"]'.format(self._bitname[i]+',', i, self._comment[i]))

        return "\n".join(result)

#-------------------------------------------------------------------------
#- The actual masks
spmask = BitMask('spmask', _bitdefs)
ccdmask = BitMask('ccdmask', _bitdefs)
fibermask = BitMask('fibermask', _bitdefs)
