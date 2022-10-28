"""
desispec.io.util
================

Utility functions for desispec IO.
"""
import os
import glob
import time
import datetime
import subprocess
import fitsio
import astropy.io
import numpy as np
from astropy.table import Table
from desiutil.log import get_logger

from ..util import healpix_degrade_fixed

def checkgzip(filename):
    """
    Check for existence of filename, with or without .gz extension

    Args:
        filename (str): filename to check for

    Returns path of existing file without or without .gz,
    or raises FileNotFoundError if neither exists
    """
    if os.path.exists(filename):
        return filename

    if filename.endswith('.gz'):
        altfilename = filename[0:-3]
    else:
        altfilename = filename + '.gz'

    if os.path.exists(altfilename):
        return altfilename
    else:
        raise FileNotFoundError(f'Neither {filename} nor {altfilename}')

def iterfiles(root, prefix, suffix=None):
    '''
    Returns iterator over files starting with `prefix` found under `root` dir
    Optionally also check if filename ends with `suffix`
    '''
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        for filename in filenames:
            if filename.startswith(prefix):
                if suffix is None or filename.endswith(suffix):
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

    if isinstance(header, fitsio.FITSHDR):
        hdr = astropy.io.fits.Header()
        for key in header.keys():
            value = header.get(key)
            comment = header.get_comment(key)
            hdr[key] = (value, comment)
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

    log = get_logger()

    #- Convert data as needed
    if isinstance(data, (np.recarray, np.ndarray, Table)):
        outdata = Table(data)
    else:
        outdata = Table(_dict2ndarray(data))

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
    log = get_logger()
    camdict = {'r':[],'b':[],'z':[]}

    for c in cameras:
        if len(c) != 2:
            log.error(f"Couldn't understand camera {c}.")
            raise ValueError(f"Couldn't understand camera {c}.")
        elif c[0] in ['r','b','z'] and c[1].isnumeric():
            camdict[c[0]].append(c[1])
        else:
            camname,camnum = c[0],c[1]
            log.error(f"Couldn't understand key {camname}{camnum}.")
            raise ValueError(f"Couldn't understand key {camname}{camnum}.")

    allcam = np.sort(list((set(camdict['r']).intersection(set(camdict['b'])).intersection(set(camdict['z'])))))

    outstr = ''
    if len(allcam) > 0:
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
    """
    Function that takes in a succinct listing
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
    log = get_logger()
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
            elif key in ['b','r','z']:
                camlist.append(key+searchstr[0])
            else:
                log.error(f"Couldn't understand key={key} in camword={camword}.")
                raise ValueError(f"Couldn't understand key={key} in camword={camword}.")
            searchstr = searchstr[1:]
    return sorted(camlist)

def parse_cameras(cameras, loglevel='INFO'):
    """
    Function that takes in a representation
    of all spectrographs and outputs a string that succinctly lists all
    spectrograph/camera pairs. It uses "a" followed by
    numbers to mean that "all" (b,r,z) cameras are accounted for for those numbers.
    b, r, and z represent the camera of the same name. All trailing numbers
    represent the spectrographs for which that camera exists in the list.

    Args:
       cameras, str. 1-d array, list: Either a str that is a comma separated list or a series of spectrographs.
                                      Also accepts a list or iterable that is processed with create_camword().
    Options:
        loglevel, str: use e.g. "WARNING" to avoid INFO-level log messages for just this call

    Returns (str):
       camword, str. A string representing all information about the spectrographs/cameras
                     given in the input iterable, e.g. a01234678b59z9
    """
    log = get_logger(loglevel)
    if cameras is None:
        camword = None
    elif type(cameras) is str:
        ## Clean the string
        cameras = cameras.strip(' \t').lower()
        ## Check to see if it already has cameras names specified
        if 'a' in cameras or 'b' in cameras or 'r' in cameras or 'z' in cameras:
            ## If there is a comma, treat each substring
            ## else decode and re-encode the camword to get the simplest camword represention
            if ',' in cameras:
                camlist = []
                ## Treat each substring as it's own substring
                for substr in cameras.split(','):
                    ## If len 1 and numeric, its a spectrograph name
                    if len(substr) == 1 and substr.isnumeric():
                        camlist.append('b' + substr)
                        camlist.append('r' + substr)
                        camlist.append('z' + substr)
                    ## If larger than 2 chars and has letters, it's a camword. Decode it
                    elif len(substr) > 2 and substr[0] in ['a','b','r','z']:
                        camlist.extend(decode_camword(substr))
                    ## If larger than 2 chars and no letters, it's a list of spectrographs
                    elif len(substr) > 2 and substr[0].isnumeric():
                        for char in substr:
                            if char.isnumeric():
                                camlist.append('b' + char)
                                camlist.append('r' + char)
                                camlist.append('z' + char)
                    ## If len 2 and starts with a, it's the full spectrograph
                    elif 'a' == substr[0]:
                        camlist.append('b'+substr[1])
                        camlist.append('r'+substr[1])
                        camlist.append('z'+substr[1])
                    ## else add the one camera if it is a known camera
                    elif substr[0] in ['b','r','z']:
                        camlist.append(substr)
                    ## otherwise throw a message
                    else:
                        log.error(f"Couldn't understand substring={substr}.")
                        raise ValueError(f"Couldn't understand substring={substr}.")

                ## Encode the given list of spectrographs to the simplest camword form
                camword = create_camword(camlist)
            else:
                ## decode and re-encode the camword to get the simplest camword represention
                camword = create_camword(decode_camword(cameras))
        ## if no letters present, then assume the comma separated list is of spectrographs
        elif ',' in cameras:
            camword = 'a'+cameras.replace(',','')
        ## if no letters or commas present, then assume the string is of spectrographs without commas
        else:
            camword = 'a'+cameras
    ## If it is a list or array, treat it as a camlist to encode
    elif not np.isscalar(cameras):
        camword = create_camword(cameras)
    ## Otherwise give an error with the cameras given
    else:
        log.error(f"Couldn't understand cameras={cameras}.")
        raise ValueError(f"Couldn't understand cameras={cameras}.")
    if camword == '':
        log.error(f"The returned camword was empty for input: {cameras}. Please check the supplied string for errors. ")
        raise ValueError(f"The returned camword was empty for input: {cameras}.")

    log.info(f"Converted input cameras={cameras} to camword={camword}")
    return camword

def difference_camwords(fullcamword,badcamword,suppress_logging=False):
    """
    Returns the difference of two camwords. The second argument cameras are removed from the first argument and the
    remainer is returned. Smart enough to ignore bad cameras if they don't exist in full camword list.

    Args:
        fullcamword, str. The camword of all cameras (including the bad ones to be removed).
        badcamword, str. A camword defining the bad cameras you don't want to include in the final camword that is output

    Returns:
        str. A camword of cameras in fullcamword that are not in badcamword.
    """
    log = get_logger()
    full_cameras = decode_camword(fullcamword)
    bad_cameras = decode_camword(badcamword)
    for cam in bad_cameras:
        if cam in full_cameras:
            full_cameras.remove(cam)
        else:
            log.info(f"Can't remove {cam}: not in the fullcamword. fullcamword={fullcamword}, badcamword={badcamword}")
    return create_camword(full_cameras)

def camword_union(camwords, full_spectros_only=False):
    """
    Returns the union of a list of camwords. Optionally can return only
    those spectros with complete b, r, and z cameras. Note this intentionally
    does the union before truncating spectrographs, so two partial camwords
    can lead to an entire spectrograph,

       e.g. [a0b1z1, a3r1z2] -> [a013z2] if full_spectros_only=False
            [a0b1z1, a3r1z2] -> [a013] if full_spectros_only=True

    even through no camword has a complete set of camera 1, a complete set is
    represented in the union.

    Args:
        camwords, list or array of strings. List of camwords.
        full_spectros_only, bool. True if only complete spectrographs with
                  b, r, and z cameras in the funal union should be returned.

    Returns:
        final_camword, str. The final union of all input camwords, where
             truncation of incomplete spectrographs may or may not be performed
             based on full_spectros_only.
    """
    camword = ''
    if np.isscalar(camwords):
        if not isinstance(camwords, str):
            ValueError(f"camwords must be array-like or str. Received type: {type(camwords)}")
        else:
            camword = camwords
    else:
        cams = set(decode_camword(camwords[0]))
        for camword in camwords[1:]:
            cams = cams.union(set(decode_camword(camword)))
        camword = create_camword(list(cams))

    if full_spectros_only:
        full_sps = np.sort(camword_to_spectros(camword,
                                               full_spectros_only=True)).astype(str)
        final_camword = 'a' + ''.join(full_sps)
    else:
        final_camword = camword
    return final_camword

def camword_to_spectros(camword, full_spectros_only=False):
    """
    Takes a camword as input and returns any spectrograph represented within that camword. By default this includes partial
    spectrographs (with one or two cameras represented). But if full_spectros_only is set to True, only spectrographs
    with all cameras represented are given.

    Args:
        camword, str. The camword of all cameras.
        full_spectros_only, bool. Default is False. Flag to specify if you want all spectrographs with any cameras existing
                                  in the camword (the default) or if you only want fully populated spectrographs.

    Returns:
        spectros, list. A list of integer spectrograph numbers represented in the camword input.
    """
    spectros = set()
    for char in camword:
        if char.isnumeric():
            spectros.add(int(char))
        elif full_spectros_only and char in ['b','r','z']:
            break
    return list(spectros)

def parse_badamps(badamps,joinsymb=','):
    """
    Parses badamps string from an exposure or processing table into the (camera,petal,amplifier) sets,
    with appropriate checking of those values to make sure they're valid.
    Returns empty list if badamps is None.

    Args:
        badamps, str. A string of {camera}{petal}{amp} entries separated by symbol given with joinsymb (comma
                      by default). I.e. [brz][0-9][ABCD]. Example: 'b7D,z8A'.
        joinsymb, str. The symbol separating entries in the str list given by badamps.

    Returns:
        cam_petal_amps, list. A list where each entry is a length 3 tuple of (camera,petal,amplifier).
                              Camera is a lowercase string in [b, r, z]. Petal is an int from 0 to 9.
                              Amplifier is an upper case string in [A, B, C, D].

    """
    cam_petal_amps = []
    if badamps is None:
        return cam_petal_amps

    for cpa in badamps.split(joinsymb):
        cpa = cpa.strip()
        if len(cpa) != 3:
            raise ValueError("Each BADAMPS entry must be a comma separated list of {camera}{petal}{amp} " +
                             f"(e.g. r7A,b8B). Given: {cpa}")
        camera, petal, amplifier = cpa[0].lower(), cpa[1], cpa[2].upper()
        if camera not in ['b', 'r', 'z']:
            raise ValueError(f"For badamps, camera must be b, r, or z. Received: {camera}")
        if not petal.isnumeric():
            raise ValueError(f"For badamps, petal must be between 0 and 9. Received: {petal}")
        if amplifier not in ['A', 'B', 'C', 'D']:
            raise ValueError(f"For badamps, amplifier must be A, B, C, and D. Received: {amplifier}")
        cam_petal_amps.append((camera, int(petal), amplifier))
    return cam_petal_amps

def validate_badamps(badamps,joinsymb=','):
    """
    Checks (and transforms) badamps string for consistency with the for need in an exposure or processing table
    for use in the Pipeline. Specifically ensure they come in (camera,petal,amplifier) sets,
    with appropriate checking of those values to make sure they're valid. Returns the input string
    except removing whitespace and replacing potential character separaters with joinsymb (default ',').
    Returns None if None is given.

    Args:
        badamps, str. A string of {camera}{petal}{amp} entries separated by symbol given with joinsymb (comma
                      by default). I.e. [brz][0-9][ABCD]. Example: 'b7D,z8A'.
        joinsymb, str. The symbol separating entries in the str list given by badamps.

    Returns:
        newbadamps, str. Input badamps string of {camera}{petal}{amp} entries separated by symbol given with
                      joinsymb (comma by default). I.e. [brz][0-9][ABCD]. Example: 'b7D,z8A'.
                      Differs from input in that other symbols used to separate terms are replaced by joinsymb
                      and whitespace is removed.

    """
    if badamps is None:
        return badamps

    log = get_logger()
    ## Possible other joining symbols to automatically replace
    symbs = [';', ':', '|', '.', ',','-','_']

    ## Not necessary, as joinsymb would just be replaced with itself, but this is good better form
    if joinsymb in symbs:
        symbs.remove(joinsymb)

    ## Remove whitespace and replace possible joining symbols with the designated one.
    newbadamps = badamps.replace(' ', '').strip()
    for symb in symbs:
        newbadamps = newbadamps.replace(symb, joinsymb)

    ## test that the string can be parsed. Raises exception if it fails to parse
    throw = parse_badamps(newbadamps, joinsymb=joinsymb)

    ## Inform user of the result
    if badamps == newbadamps:
        log.info(f'Badamps given as: {badamps} verified to work')
    else:
        log.info(f'Badamps given as: {badamps} verified to work with modifications to: {newbadamps}')
    return newbadamps

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

def replace_prefix(filepath, oldprefix, newprefix):
    """
    Replace filename prefix even if prefix is elsewhere in path or filename

    Args:
        filepath : filename, optionally including path
        oldprefix: original prefix to filename part of filepath
        newprefix: new prefix to use for filename

    Returns:
        new filepath with replaced prefix

    e.g. replace_prefix('/blat/foo/blat-bar-blat.fits', 'blat', 'quat')
    returns '/blat/foo/quat-bar-blat.fits'
    """
    path, filename = os.path.split(filepath)
    if not filename.startswith(oldprefix):
        raise ValueError(f'{filename} does not start with {oldprefix}')

    filename = filename.replace(oldprefix, newprefix, 1)
    return os.path.join(path, filename)

def get_tempfilename(filename):
    """
    Returns unique tempfile in same directory as filename with same extension

    Args:
        filename (str): input filename

    Returns unique filename in same directory

    Example intended usage::

       tmpfile = get_tempfile(filename)
       table.write(tmpfile)
       os.rename(tmpfile, filename)

    By keeping the same extension as the input file, this preserves the
    ability of table.write to derive the format to use, and if something
    goes wrong with the I/O it doesn't leave a corrupted partially written
    file with the final name.  The tempfile includes the PID to provide some
    race condition protection (the last one do do os.rename wins, but at least
    different processes won't corrupt each other's files).
    """
    pid = os.getpid()
    if filename.endswith(('.gz', '.fz')):
        filename, second_ext = os.path.splitext(filename)
    else:
        second_ext = ''
    base, extension = os.path.splitext(filename)
    tempfile = f'{base}_tmp{pid}{extension}{second_ext}'
    return tempfile

def addkeys(hdr1, hdr2, skipkeys=None):
    """
    Add new header keys from hdr2 to hdr1, skipping skipkeys

    Arguments:
        hdr1 (dict-like): destination header for keywords
        hdr2 (dict-like): source header for keywords

    Modifies hdr1 in place
    """
    log = get_logger()
    #- standard keywords that should be skipped
    stdkeys = ['EXTNAME', 'COMMENT', 'CHECKSUM', 'DATASUM',
                'PCOUNT', 'GCOUNT', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2',
                'XTENSION', 'TFIELDS', 'SIMPLE']

    for key in hdr2.keys():
        if key not in stdkeys and \
               ((skipkeys is None) or (key not in skipkeys)) \
               and not key.startswith('TTYPE') \
               and not key.startswith('TFORM') \
               and not key.startswith('TUNIT') \
               and key not in hdr1:
            log.debug('Adding %s', key)
            hdr1[key] = hdr2[key]
        else:
            log.debug('Skipping %s', key)

def is_svn_current(dirname):
    """
    Return True/False for if svn checkout dirname is up-to-date with server

    Raises ValueError if unable to determine (e.g. dirname isn't svn checkout)
    """
    cmd = f"svn diff -r BASE:HEAD {dirname}"
    args = cmd.split()
    try:
        results = subprocess.run(args, check=True, stdout=subprocess.PIPE).stdout
        #- no stdout = no diffs = up-to-date
        return len(results) == 0
    except subprocess.CalledProcessError:
        log = get_logger()
        msg = f'FAILED {cmd}'
        log.error(msg)
        raise ValueError(msg)

def relsymlink(src, dst, pathonly=False):
    """
    Create a relative symlink from dst -> src, while also handling
    $DESI_ROOT vs $DESI_ROOT_READONLY

    Args:
        src (str): the pre-existing file to point to
        dst (str): the symlink file to create

    Options:
        pathonly (bool): return the path, but don't make the link

    Returns:
        the releative path from dst -> src
    """
    #- Standardize path
    src = os.path.normpath(os.path.abspath(src))
    dst = os.path.normpath(os.path.abspath(dst))

    #- handle DESI_ROOT (required) vs. DESI_ROOT_READONLY (optional)
    if 'DESI_ROOT_READONLY' in os.environ:
        ro_root = os.path.normpath(os.environ['DESI_ROOT_READONLY'])
        rw_root = os.path.normpath(os.environ['DESI_ROOT'])

        if (ro_root != rw_root) and src.startswith(ro_root):
            src = src.replace(ro_root, rw_root, 1)

    relpath = os.path.relpath(src, os.path.dirname(dst))

    if not pathonly:
        os.symlink(relpath, dst)

    return relpath

