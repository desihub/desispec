"""
io routines for frame

"""
import os.path
from astropy.io import fits
import scipy,scipy.sparse
from desispec.io import findfile
from desispec.io.util import fitsheader, native_endian, makepath

def write_frame(outfile, flux,ivar,wave,resolution_data, header=None) :
    """
    Write a frame fits file and returns path to file written
    
    Args:
        TODO
    """
    outfile = makepath(outfile, 'frame')

    hdr = fitsheader(header)
    hdr['EXTNAME'] = ('FLUX', 'no dimension')
    fits.writeto(outfile,flux,header=hdr, clobber=True)
    
    hdr['EXTNAME'] = ('IVAR', 'no dimension')
    hdu = fits.ImageHDU(ivar, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)
    
    hdr['EXTNAME'] = ('WAVELENGTH', '[Angstroms]')
    hdu = fits.ImageHDU(wave, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)
    
    hdr['EXTNAME'] = ('RESOLUTION', 'no dimension')
    hdu = fits.ImageHDU(resolution_data, header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)
    
    return outfile
    
def read_frame(filename):
    """
    reads a frame fits file and returns its data
    
    Args:
        filename: path to a file, or (night, expid, camera) tuple where
            night = string YEARMMDD
            expid = integer exposure ID
            camera = b0, r1, .. z9
        
    returns tuple of:
        phot[nspec, nwave] : uncalibrated photons per bin
        ivar[nspec, nwave] : inverse variance of phot
        wave[nwave] : vacuum wavelengths [Angstrom]
        resolution[nspec, ndiag, nwave] : TODO DOCUMENT THIS FORMAT
    """
    #- check if filename is (night, expid, camera) tuple instead
    if not isinstance(filename, (str, unicode)):
        night, expid, camera = filename
        filename = findfile('frame', night, expid, camera)
    
    if not os.path.isfile(filename) :
        raise IOError("cannot open"+filename)
    
    hdr = fits.getheader(filename)
    flux = native_endian(fits.getdata(filename, 0))
    ivar = native_endian(fits.getdata(filename, "IVAR")) 
    wave = native_endian(fits.getdata(filename, "WAVELENGTH"))
    resolution_data = native_endian(fits.getdata(filename, "RESOLUTION"))
    
    return flux,ivar,wave,resolution_data, hdr

def resolution_data_to_sparse_matrix(resolution_data,fiber) :
    """
    convert the resolution data for a given fiber into a sparse matrix
    use function M.todense() or M.toarray() to convert output sparse matrix M to a dense matrix or numpy array
    """
    
    if len(resolution_data.shape)==3 :
        nfibers=resolution_data.shape[0]
        d=resolution_data.shape[1]/2
        nwave=resolution_data.shape[2]
        offsets = range(d,-d-1,-1)
        return scipy.sparse.dia_matrix((resolution_data[fiber],offsets),(nwave,nwave))
    elif len(resolution_data.shape)==2 :
        if fiber>0 :
            print "error in resolution_data_to_sparse_matrix, shape=",resolution_data.shape," and requested fiber=",fiber
            sys.exit(12)
        nfibers=1
        d=resolution_data.shape[0]/2
        nwave=resolution_data.shape[1]
        offsets = range(d,-d-1,-1)
        return scipy.sparse.dia_matrix((resolution_data,offsets),(nwave,nwave))
    else :
        print "error in resolution_data_to_sparse_matrix, shape=",resolution_data.shape
        sys.exit(12)
    
