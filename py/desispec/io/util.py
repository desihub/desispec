"""
desispec.io.util
================

Utility functions for desispec IO.
"""
import os
import glob
import astropy.io
import numpy as np
from astropy.table import Table
from desiutil.log import get_logger

from ..util import healpix_degrade_fixed


def iterfiles(root, prefix):
    '''
    Returns iterator over files starting with `prefix` found under `root` dir
    '''
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        for filename in filenames:
            if filename.startswith(prefix):
                yield os.path.join(dirpath, filename)

def header2wave(header):
    """Converts header keywords into a wavelength grid.

    returns wave = CRVAL1 + range(NAXIS1)*CDELT1

    if LOGLAM keyword is present and true/non-zero, returns 10**wave
    """
    w = header['CRVAL1'] + np.arange(header['NAXIS1'])*header['CDELT1']
    if 'LOGLAM' in header and header['LOGLAM']:
        w = 10**w

    return w

def fitsheader(header):
    """Convert header into astropy.io.fits.Header object.

    header can be:
      - None: return a blank Header
      - list of (key, value) or (key, (value,comment)) entries
      - dict d[key] -> value or (value, comment)
      - Header: just return it unchanged

    Returns fits.Header object
    """
    if header is None:
        return astropy.io.fits.Header()

    if isinstance(header, list):
        hdr = astropy.io.fits.Header()
        for key, value in header:
            hdr[key] = value

        return hdr

    if isinstance(header, dict):
        hdr = astropy.io.fits.Header()
        for key, value in header.items():
            hdr[key] = value
        return hdr

    if isinstance(header, astropy.io.fits.Header):
        return header

    raise ValueError("Can't convert {} into fits.Header".format(type(header)))

def native_endian(data):
    """Convert numpy array data to native endianness if needed.

    Returns new array if endianness is swapped, otherwise returns input data

    Context:
    By default, FITS data from astropy.io.fits.getdata() are not Intel
    native endianness and scipy 0.14 sparse matrices have a bug with
    non-native endian data.
    """
    if data.dtype.isnative:
        return data
    else:
        return data.byteswap().newbyteorder()

def add_columns(data, colnames, colvals):
    '''
    Adds extra columns to a data table

    Args:
        data: astropy Table or numpy structured array
        colnames: list of new column names to add
        colvals: list of column values to add;
            each element can be scalar or vector

    Returns:
        new table with extra columns added

    Example:
        fibermap = add_columns(fibermap,
                    ['NIGHT', 'EXPID'], [20102020, np.arange(len(fibermap))])

    Notes:
        This is similar to `numpy.lib.recfunctions.append_fields`, but
        it also accepts astropy Tables as the `data` input, and accepts
        scalar values to expand as entries in `colvals`.
    '''
    if isinstance(data, Table):
        data = data.copy()
        for key, value in zip(colnames, colvals):
            data[key] = value
    else:
        nrows = len(data)
        colvals = list(colvals)  #- copy so that we can modify
        for i, value in enumerate(colvals):
            if np.isscalar(value):
                colvals[i] = np.full(nrows, value, dtype=type(value))
            else:
                assert len(value) == nrows

        data = np.lib.recfunctions.append_fields(data, colnames, colvals, usemask=False)

    return data

def makepath(outfile, filetype=None):
    """Create path to outfile if needed.

    If outfile isn't a string and filetype is set, interpret outfile as
    a tuple of parameters to locate the file via findfile().

    Returns /path/to/outfile

    TODO: maybe a different name?
    """
    from .meta import findfile
    #- if this doesn't look like a filename, interpret outfile as a tuple of
    #- (night, expid, ...) via findfile.  Only works if filetype is set.
    ### if (filetype is not None) and (not isinstance(outfile, (str, unicode))):
    if (filetype is not None) and (not isinstance(outfile, str)):
        outfile = findfile(filetype, *outfile)

    #- Create the path to that file if needed
    path = os.path.normpath(os.path.dirname(outfile))
    if not os.path.exists(path):
        os.makedirs(path)

    return outfile

def write_bintable(filename, data, header=None, comments=None, units=None,
                   extname=None, clobber=False, primary_extname='PRIMARY'):
    """Utility function to write a fits binary table complete with
    comments and units in the FITS header too.  DATA can either be
    dictionary, an Astropy Table, a numpy.recarray or a numpy.ndarray.
    """
    from astropy.table import Table
    from desiutil.io import encode_table

    log = get_logger()

    #- Convert data as needed
    if isinstance(data, (np.recarray, np.ndarray, Table)):
        outdata = encode_table(data, encoding='ascii')
    else:
        outdata = encode_table(_dict2ndarray(data), encoding='ascii')

    # hdu = astropy.io.fits.BinTableHDU(outdata, header=header, name=extname)
    hdu = astropy.io.fits.convenience.table_to_hdu(outdata)
    if extname is not None:
        hdu.header['EXTNAME'] = extname
    else:
        log.warning("Table does not have EXTNAME set!")

    if header is not None:
        if isinstance(header, astropy.io.fits.header.Header):
            for key, value in header.items():
                comment = header.comments[key]
                hdu.header[key] = (value, comment)
        else:
            hdu.header.update(header)

    #- Allow comments and units to be None
    if comments is None:
        comments = dict()
    if units is None:
        units = dict()
    #
    # Add comments and units to the *columns* of the table.
    #
    for i in range(1, 999):
        key = 'TTYPE'+str(i)
        if key not in hdu.header:
            break
        else:
            value = hdu.header[key]
            if value in comments:
                hdu.header[key] = (value, comments[value])
            if value in units:
                hdu.header['TUNIT'+str(i)] = (units[value], value+' units')
    #
    # Add checksum cards.
    #
    hdu.add_checksum()

    #- Write the data and header

    if os.path.isfile(filename):
        if not(extname is None and clobber):
            #
            # Always open update mode with memmap=False, but keep the
            # formal check commented out in case we need it in the future.
            #
            memmap = False
            #
            # Check to see if filesystem supports memory-mapping on update.
            #
            # memmap = _supports_memmap(filename)
            # if not memmap:
            #     log.warning("Filesystem does not support memory-mapping!")
            with astropy.io.fits.open(filename, mode='update', memmap=memmap) as hdulist:
                if extname is None:
                    #
                    # In DESI, we should *always* be setting the extname, so this
                    # might never be called.
                    #
                    log.debug("Adding new HDU to %s.", filename)
                    hdulist.append(hdu)
                else:
                    if extname in hdulist:
                        if clobber:
                            log.debug("Replacing HDU with EXTNAME = '%s' in %s.", extname, filename)
                            hdulist[extname] = hdu
                        else:
                            log.warning("Do not modify %s because EXTNAME = '%s' exists.", filename, extname)
                    else:
                        log.debug("Adding new HDU with EXTNAME = '%s' to %s.", extname, filename)
                        hdulist.append(hdu)
            return
    #
    # If we reach this point, we're writing a new file.
    #
    if os.path.isfile(filename):
        log.debug("Overwriting %s.", filename)
    else:
        log.debug("Writing new file %s.", filename)
    hdu0 = astropy.io.fits.PrimaryHDU()
    hdu0.header['EXTNAME'] = primary_extname
    hdulist = astropy.io.fits.HDUList([hdu0, hdu])
    hdulist.writeto(filename, overwrite=clobber, checksum=True)
    return


def _supports_memmap(filename):
    """Returns ``True`` if the filesystem containing `filename` supports
    opening memory-mapped files in update mode.
    """
    m = True
    testfile = os.path.join(os.path.dirname(os.path.realpath(filename)),
                            'foo.dat')
    try:
        f = np.memmap(testfile, dtype='f4', mode='w+', shape=(3, 4))
    except OSError:
        m = False
    finally:
        os.remove(testfile)
    return m


def _dict2ndarray(data, columns=None):
    """
    Convert a dictionary of ndarrays into a structured ndarray

    Also works if DATA is an AstroPy Table.

    Args:
        data: input dictionary, each value is an ndarray
        columns: optional list of column names

    Returns:
        structured numpy.ndarray with named columns from input data dictionary

    Notes:
        data[key].shape[0] must be the same for every key
        every entry in columns must be a key of data

    Example
        d = dict(x=np.arange(10), y=np.arange(10)/2)
        nddata = _dict2ndarray(d, columns=['x', 'y'])
    """
    if columns is None:
        columns = list(data.keys())

    dtype = list()
    for key in columns:
        ### dtype.append( (key, data[key].dtype, data[key].shape) )
        if data[key].ndim == 1:
            dtype.append( (key, data[key].dtype) )
        else:
            dtype.append( (key, data[key].dtype, data[key].shape[1:]) )

    nrows = len(data[key])  #- use last column to get length
    xdata = np.empty(nrows, dtype=dtype)

    for key in columns:
        xdata[key] = data[key]

    return xdata


def healpix_subdirectory(nside, pixel):
    """
    Return a fixed directory path for healpix grouped files.

    Given an NSIDE and NESTED pixel index, return a directory
    named after a degraded NSIDE and pixel followed by the
    original nside and pixel.  This function is just to ensure
    that this subdirectory path is always created by the same
    code.

    Args:
        nside (int): a valid NSIDE value.
        pixel (int): the NESTED pixel index.

    Returns (str):
        a path containing the low and high resolution
        directories.

    """
    subdir = str(pixel//100)
    pixdir = str(pixel)
    return os.path.join(subdir, pixdir)

    #- Note: currently nside isn't used, but if we did want to do a strict
    #- superpix grouping, we would need to know nside and do something like:

    # subnside, subpixel = healpix_degrade_fixed(nside, pixel)
    # return os.path.join("{}-{}".format(subnside, subpixel),
    #     "{}-{}".format(nside, pixel))

    
def create_camword(cameras):
    """
    Function that takes in a list of cameras and creates a succinct listing
    of all spectrographs in the list with cameras. It uses "a" followed by
    numbers to mean that "all" (b,r,z) cameras are accounted for for those numbers.
    b, r, and z represent the camera of the same name. All trailing numbers
    represent the spectrographs for which that camera exists in the list.

    Args:
       cameras (1-d array or list): iterable containing strings of
                                     cameras, e.g. 'b0','r1',...
    Returns (str):
       A string representing all information about the spectrographs/cameras
       given in the input iterable, e.g. a01234678b59z9
    """
    camdict = {'r':[],'b':[],'z':[]}

    for c in cameras:
        camdict[c[0]].append(c[1])

    allcam = np.sort(list((set(camdict['r']).intersection(set(camdict['b'])).intersection(set(camdict['z'])))))

    outstr = 'a'+''.join(allcam)

    for key in np.sort(list(camdict.keys())):
        val = camdict[key]
        if len(val) == 0:
            continue
        uniq = np.sort(list(set(val).difference(allcam)))
        if len(uniq) > 0:
            outstr += (key + ''.join(uniq))
    return outstr

def decode_camword(camword):
    """                                                                                                                             Function that takes in a succinct listing                                                     
    of all spectrographs and outputs a 1-d numpy array with a list of all
    spectrograph/camera pairs. It uses "a" followed by                                                      
    numbers to mean that "all" (b,r,z) cameras are accounted for for those numbers.                                             
    b, r, and z represent the camera of the same name. All trailing numbers                                                     
    represent the spectrographs for which that camera exists in the list.                                                       
                                                                                                                                
    Args:                      
       camword (str): A string representing all information about the spectrographs/cameras                                    
                        e.g. a01234678b59z9                                                                                                  
    Returns (np.ndarray, 1d):  an array containing strings of                                                              
                                cameras, e.g. 'b0','r1',...                                                                
    """
    searchstr = camword
    camlist = []
    while len(searchstr) > 1:
        key = searchstr[0]
        searchstr = searchstr[1:]

        while len(searchstr) > 0 and searchstr[0].isnumeric():
            if key == 'a':
                camlist.append('b'+searchstr[0])
                camlist.append('r'+searchstr[0])
                camlist.append('z'+searchstr[0])
            else:
                camlist.append(key+searchstr[0])
            searchstr = searchstr[1:]
    return np.sort(camlist)

def get_speclog(nights, rawdir=None):
    """
    Scans raw data headers to return speclog of observations. Slow.

    Args:
        nights: list of YEARMMDD nights to scan

    Options:
        rawdir (str): overrides $DESI_SPECTRO_DATA

    Returns:
        Table with columns NIGHT,EXPID,MJD,FLAVOR,OBSTYPE,EXPTIME

    Scans headers of rawdir/NIGHT/EXPID/desi-EXPID.fits.fz
    """
    #- only import fitsio if this function is called
    import fitsio

    if rawdir is None:
        rawdir = os.environ['DESI_SPECTRO_DATA']

    rows = list()
    for night in nights:
        for filename in sorted(glob.glob(f'{rawdir}/{night}/*/desi-*.fits.fz')):
            hdr = fitsio.read_header(filename, 1)
            rows.append([night, hdr['EXPID'], hdr['MJD-OBS'],
                hdr['FLAVOR'], hdr['OBSTYPE'], hdr['EXPTIME']])

    speclog = Table(
        names = ['NIGHT', 'EXPID', 'MJD', 'FLAVOR', 'OBSTYPE', 'EXPTIME'],
        rows = rows,
    )

    return speclog

