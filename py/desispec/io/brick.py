"""
desispec.io.brick
=================

I/O routines for working with per-brick files.

See ``doc/DESI_SPECTRO_REDUX/PRODNAME/bricks/BRICKID/*-BRICKID.rst`` in desiDataModel
for a description of the relevant data models.

See :doc:`coadd` and `DESI-doc-1056 <https://desi.lbl.gov/DocDB/cgi-bin/private/ShowDocument?docid=1056>`_
for general information about the coaddition dataflow and algorithms.
"""

import os
import os.path

import numpy as np
import astropy.io.fits

import desispec.io.util

class BrickBase(object):
    """Represents objects in a single brick and possibly also a single band b,r,z.

    The constructor will open an existing file and create a new file and parent
    directory if necessary.  The :meth:`close` method must be called for any updates
    or new data to be recorded. Successful completion of the constructor does not
    guarantee that :meth:`close` will succeed.

    Args:
        path(str): Path to the brick file to open.
        mode(str): File access mode to use. Should normally be 'readonly' or 'update'. Use 'update' to create a new file and its parent directory if necessary.
        header: An optional header specification used to create a new file. See :func:`desispec.io.util.fitsheader` for details on allowed values.

    Raises:
        RuntimeError: Invalid mode requested.
        IOError: Unable to open existing file in 'readonly' mode.
        OSError: Unable to create a new parent directory in 'update' mode.
    """
    def __init__(self,path,mode = 'readonly',header = None):
        if mode not in ('readonly','update'):
            raise RuntimeError('Invalid mode %r' % mode)
        self.path = path
        self.mode = mode
        # Create a new file if necessary.
        if self.mode == 'update' and not os.path.exists(self.path):
            # Create the parent directory, if necessary.
            head,tail = os.path.split(self.path)
            if not os.path.exists(head):
                os.makedirs(head)
            # Create empty HDUs. It would be good to refactor io.frame to avoid any duplication here.
            hdr = desispec.io.util.fitsheader(header)
            hdr['EXTNAME'] = ('FLUX', 'no dimension')
            hdu0 = astropy.io.fits.PrimaryHDU(header = hdr)
            hdr['EXTNAME'] = ('IVAR', 'no dimension')
            hdu1 = astropy.io.fits.ImageHDU(header = hdr)
            hdr['EXTNAME'] = ('WAVELENGTH', '[Angstroms]')
            hdu2 = astropy.io.fits.ImageHDU(header = hdr)
            hdr['EXTNAME'] = ('RESOLUTION', 'no dimension')
            hdu3 = astropy.io.fits.ImageHDU(header = hdr)
            # Create an HDU4 using the columns from fibermap with a few extras added.
            columns = desispec.io.fibermap.fibermap_columns[:]
            columns.extend([
                ('NIGHT','i4'),
                ('EXPID','i4'),
                ('INDEX','i4'),
                ])
            data = np.empty(shape = (0,),dtype = columns)
            hdr = desispec.io.util.fitsheader(header)
            hdu4 = astropy.io.fits.BinTableHDU(data=data, header=hdr, name='FIBERMAP')
            # Add comments for fibermap columns.
            num_fibermap_columns = len(desispec.io.fibermap.fibermap_comments)
            for i in range(1,1+num_fibermap_columns):
                key = 'TTYPE%d' % i
                name = hdu4.header[key]
                comment = desispec.io.fibermap.fibermap_comments[name]
                hdu4.header[key] = (name,comment)
            # Add comments for our additional columns.
            hdu4.header['TTYPE%d' % (1+num_fibermap_columns)] = ('NIGHT','Night of exposure YYYYMMDD')
            hdu4.header['TTYPE%d' % (2+num_fibermap_columns)] = ('EXPID','Exposure ID')
            hdu4.header['TTYPE%d' % (3+num_fibermap_columns)] = ('INDEX','Index of this object in other HDUs')
            self.hdu_list = astropy.io.fits.HDUList([hdu0,hdu1,hdu2,hdu3,hdu4])
        else:
            self.hdu_list = astropy.io.fits.open(path,mode = self.mode)

    def add_objects(self,flux,ivar,wave,resolution):
        """Add a list of objects to this brick file from the same night and exposure.

        Args:
            flux(numpy.ndarray): Array of (nobj,nwave) flux values for nobj objects tabulated at nwave wavelengths.
            ivar(numpy.ndarray): Array of (nobj,nwave) inverse-variance values.
            wave(numpy.ndarray): Array of (nwave,) wavelength values in Angstroms. All objects are assumed to use the same wavelength grid.
            resolution(numpy.ndarray): Array of (nobj,nres,nwave) resolution matrix elements.

        Raises:
            RuntimeError: Can only add objects in update mode.
        """
        if self.mode != 'update':
            raise RuntimeError('Can only add objects in update mode.')
        # Concatenate the new per-object image HDU data or use it to initialize the HDU.
        # HDU2 contains the wavelength grid shared by all objects so we only add it once.
        if self.hdu_list[0].data is not None:
            self.hdu_list[0].data = np.concatenate((self.hdu_list[0].data,flux,))
            self.hdu_list[1].data = np.concatenate((self.hdu_list[1].data,ivar,))
            assert np.array_equal(self.hdu_list[2].data,wave),'Wavelength arrays do not match.'
            self.hdu_list[3].data = np.concatenate((self.hdu_list[3].data,resolution,))
        else:
            self.hdu_list[0].data = flux
            self.hdu_list[1].data = ivar
            self.hdu_list[2].data = wave
            self.hdu_list[3].data = resolution

    def get_wavelength_grid(self):
        """Return the wavelength grid used in this brick file.
        """
        return self.hdu_list[2].data

    def get_target(self,target_id):
        """Get the spectra and info for one target ID.

        Args:
            target_id(int): Target ID number to lookup.

        Returns:
            tuple: Tuple of numpy arrays (flux,ivar,resolution,info) of data associated
                with this target ID. The flux,ivar,resolution arrays will have one entry
                for each spectrum and the info array will have one entry per exposure.
                The returned arrays are slices into the FITS file HDU data arrays, so this
                call is relatively cheap (and any changes will be saved to the file if it
                was opened in update mode.)
        """
        exposures = (self.hdu_list[4].data['TARGETID'] == target_id)
        index_list = np.unique(self.hdu_list[4].data['INDEX'][exposures])
        return (self.hdu_list[0].data[index_list],self.hdu_list[1].data[index_list],
            self.hdu_list[3].data[index_list],self.hdu_list[4].data[exposures])

    def get_target_ids(self):
        """Return set of unique target IDs in this brick.
        """
        return list(set(self.hdu_list[4].data['TARGETID']))

    def get_num_spectra(self):
        """Get the number of spectra contained in this brick file.

        Returns:
            int: Number of objects contained in this brick file.
        """
        return len(self.hdu_list[0].data)

    def get_num_targets(self):
        """Get the number of distinct targets with at least one spectrum in this brick file.

        Returns:
            int: Number of unique targets represented with spectra in this brick file.
        """
        return len(np.unique(self.hdu_list[4].data['TARGETID']))

    def close(self):
        """Write any updates and close the brick file.
        """
        if self.mode == 'update':
            self.hdu_list.writeto(self.path,clobber = True)
        self.hdu_list.close()

class Brick(BrickBase):
    """Represents the combined cframe exposures in a single brick and band.

    See :class:`BrickBase` for constructor info.
    """
    def __init__(self,path,mode = 'readonly',header = None):
        BrickBase.__init__(self,path,mode,header)

    def add_objects(self,flux,ivar,wave,resolution,object_data,night,expid):
        """Add a list of objects to this brick file from the same night and exposure.

        Args:
            flux(numpy.ndarray): Array of (nobj,nwave) flux values for nobj objects tabulated at nwave wavelengths.
            ivar(numpy.ndarray): Array of (nobj,nwave) inverse-variance values.
            wave(numpy.ndarray): Array of (nwave,) wavelength values in Angstroms. All objects are assumed to use the same wavelength grid.
            resolution(numpy.ndarray): Array of (nobj,nres,nwave) resolution matrix elements.
            object_data(numpy.ndarray): Record array of fibermap rows for the objects to add.
            night(str): Date string for the night these objects were observed in the format YYYYMMDD.
            expid(int): Exposure number for these objects.

        Raises:
            RuntimeError: Can only add objects in update mode.
        """
        BrickBase.add_objects(self,flux,ivar,wave,resolution)
        # Augment object_data with constant NIGHT and EXPID columns.
        augmented_data = np.empty(shape = object_data.shape,dtype = self.hdu_list[4].data.dtype)
        for column_def in desispec.io.fibermap.fibermap_columns:
            name = column_def[0]
            # Special handling for the fibermap FILTER array, which is not output correctly
            # by astropy.io.fits so we convert it to a comma-separated list.
            if name == 'FILTER' and augmented_data[name].shape != object_data[name].shape:
                for i,filters in enumerate(object_data[name]):
                    augmented_data[name][i] = ','.join(filters)
            else:
                augmented_data[name] = object_data[name]
        augmented_data['NIGHT'] = int(night)
        augmented_data['EXPID'] = expid
        begin_index = len(self.hdu_list[4].data)
        end_index = begin_index + len(flux)
        augmented_data['INDEX'] = np.arange(begin_index,end_index,dtype=int)
        # Always concatenate to our table since a new file will be created with a zero-length table.
        self.hdu_list[4].data = np.concatenate((self.hdu_list[4].data,augmented_data,))

class CoAddedBrick(BrickBase):
    """Represents the co-added exposures in a single brick and, possibly, a single band.

    See :class:`BrickBase` for constructor info.
    """
    def __init__(self,path,mode = 'readonly',header = None):
        BrickBase.__init__(self,path,mode,header)
