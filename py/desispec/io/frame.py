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

def write_frame(filename,head,flux,ivar,wave,resolution_data) :
    """
    write a frame fits file
    """
    hdr = head
    hdr['EXTNAME'] = ('FLUX', 'no dimension')
    fits.writeto(filename,flux,header=hdr, clobber=True)
    
    hdr['EXTNAME'] = ('IVAR', 'no dimension')
    hdu = fits.ImageHDU(ivar, header=hdr)
    fits.append(filename, hdu.data, header=hdu.header)
    
    hdr['EXTNAME'] = ('WAVELENGTH', '[Angstroms]')
    hdu = fits.ImageHDU(wave, header=hdr)
    fits.append(filename, hdu.data, header=hdu.header)
    
    hdr['EXTNAME'] = ('RESOLUTION', 'no dimension')
    hdu = fits.ImageHDU(resolution_data, header=hdr)
    fits.append(filename, hdu.data, header=hdu.header)
    

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
    
