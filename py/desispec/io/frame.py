"""
io routines for frame

"""
import os
from astropy.io import fits
import scipy,scipy.sparse

def read_frame(filename) :
    """
    reads a frame fits file and returns its data
    """
    
    if not os.path.isfile(filename) :
        raise IOError("cannot open"+filename)
    
     #hdr = fits.getheader(filename)
    flux = fits.getdata(filename, 0).astype('float64') # import on edison.nersc.edu
    ivar = fits.getdata(filename, "IVAR").astype('float64') 
    wave = fits.getdata(filename, "WAVELENGTH").astype('float64')
    resolution_data = fits.getdata(filename, "RESOLUTION").astype('float64')
    
    return flux,ivar,wave,resolution_data


def resolution_data_to_sparse_matrix(resolution_data,fiber) :
    """
    convert the resolution data for a given fiber into a sparse matrix
    use function M.todense() or M.toarray() to convert output sparse matrix M to a dense matrix or numpy array
    """
    nfibers=resolution_data.shape[0]
    d=resolution_data.shape[1]/2
    nwave=resolution_data.shape[2]
    offsets = range(d,-d-1,-1)
    return scipy.sparse.dia_matrix((resolution_data[fiber],offsets),(nwave,nwave))

