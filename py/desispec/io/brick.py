"""
desispec.io.brick
=================

I/O routines for working with per-brick files.

See ``doc/DESI_SPECTRO_REDUX/SPECPROD/bricks/BRICKID/*-BRICKID.rst`` in desidatamodel
for a description of the relevant data models.

See :doc:`coadd` and `DESI-doc-1056 <https://desi.lbl.gov/DocDB/cgi-bin/private/ShowDocument?docid=1056>`_
for general information about the coaddition dataflow and algorithms.
"""

import os
import os.path
import re
import warnings

import numpy as np
import astropy.io.fits
from astropy import table

from desiutil.depend import add_dependencies
import desispec.io.util
import desiutil.io

#- For backwards compatibility, derive brickname from filename
def _parse_brick_filename(filepath):
    """return (channel, brickname) from /path/to/brick-[brz]-{brickname}.fits
    """
    filename = os.path.basename(filepath)
    warnings.warn('Deriving channel and brickname from filename {} instead of contents'.format(filename))
    m = re.match('brick-([brz])-(\w+).fits', filename)
    if m is None:
        raise ValueError('Unable to derive channel and brickname from '+filename)
    else:
        return m.groups()  #- (channel, brickname)

class BrickBase(object):
    """Represents objects in a single brick and possibly also a single band b,r,z.

    The constructor will open an existing file and create a new file and parent
    directory if necessary.  The :meth:`close` method must be called for any updates
    or new data to be recorded. Successful completion of the constructor does not
    guarantee that :meth:`close` will succeed.

    Args:
        path(str): Path to the brick file to open.
        mode(str): File access mode to use. Should normally be 'readonly' or 'update'. Use 'update' to create a new file and its parent directory if necessary.
        header: header used to create a new file. See :func:`desispec.io.util.fitsheader` for details on allowed types.
            required for new files and must have BRICKNAM keyword; ignored when opening existing files

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
            # BRICKNAM must be in header if creating the file for the first time
            if header is None or 'BRICKNAM' not in header:
                raise ValueError('header must have BRICKNAM when creating new brick file')

            self.brickname = header['BRICKNAM']
            if 'CHANNEL' in header:
                self.channel = header['CHANNEL']
            else:
                self.channel = 'brz'  #- could be any spectrograph channel
                
            # Create the parent directory, if necessary.
            head,tail = os.path.split(self.path)
            if not os.path.exists(head):
                os.makedirs(head)
            # Create empty HDUs. It would be good to refactor io.frame to avoid any duplication here.
            hdr = desispec.io.util.fitsheader(header)
            add_dependencies(hdr)
            hdr['EXTNAME'] = ('FLUX', '1e-17 erg/(s cm2 Angstrom)')
            hdr['BUNIT'] = '1e-17 erg/(s cm2 Angstrom)'
            hdu0 = astropy.io.fits.PrimaryHDU(header=hdr)
            hdu1 = astropy.io.fits.ImageHDU(name='IVAR')
            hdu2 = astropy.io.fits.ImageHDU(name='WAVELENGTH')
            hdu2.header['BUNIT'] = 'Angstrom'
            hdu3 = astropy.io.fits.ImageHDU(name='RESOLUTION')
            # Create an HDU4 using the columns from fibermap with a few extras added.
            columns = desispec.io.fibermap.fibermap_columns[:]
            columns.extend([
                ('NIGHT','i4'),
                ('EXPID','i4'),
                ('INDEX','i4'),
                ])
            data = np.empty(shape = (0,),dtype = columns)
            data = desiutil.io.encode_table(data)   #- unicode -> bytes
            data.meta['EXTNAME'] = 'FIBERMAP'
            for key, value in header.items():
                data.meta[key] = value
            hdu4 = astropy.io.fits.convenience.table_to_hdu(data)

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
            try:
                self.brickname = self.hdu_list[0].header['BRICKNAM']
                self.channel = self.hdu_list[0].header['CHANNEL']
            except KeyError:
                self.channel, self.brickname = _parse_brick_filename(path)

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
        return (self.hdu_list[0].data[exposures],self.hdu_list[1].data[exposures],
            self.hdu_list[3].data[exposures],self.hdu_list[4].data[exposures])

    def get_target_ids(self):
        """Return list of unique target IDs in this brick
        in the order that they first appear in the file input file.
        """
        uniq, indices = np.unique(self.hdu_list[4].data['TARGETID'], return_index=True)
        return uniq[indices.argsort()]

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
            object_data(astropy.table.Table): fibermap rows for the objects to add.
            night(str): Date string for the night these objects were observed in the format YYYYMMDD.
            expid(int): Exposure number for these objects.

        Raises:
            RuntimeError: Can only add objects in update mode.
        """
        BrickBase.add_objects(self,flux,ivar,wave,resolution)

        augmented_data = table.Table(object_data)
        augmented_data['NIGHT'] = int(night)
        augmented_data['EXPID'] = expid

        fibermap_hdu = self.hdu_list['FIBERMAP']
        if len(fibermap_hdu.data) > 0:
            orig_data = table.Table(fibermap_hdu.data)
            augmented_data = table.vstack([orig_data, augmented_data])

        #- unicode -> ascii columns
        augmented_data = desiutil.io.encode_table(augmented_data)

        updated_hdu = astropy.io.fits.convenience.table_to_hdu(augmented_data)
        updated_hdu.header = fibermap_hdu.header
        self.hdu_list['FIBERMAP'] = updated_hdu

class CoAddedBrick(BrickBase):
    """Represents the co-added exposures in a single brick and, possibly, a single band.

    See :class:`BrickBase` for constructor info.
    """
    def __init__(self,path,mode = 'readonly',header = None):
        BrickBase.__init__(self,path,mode,header)
