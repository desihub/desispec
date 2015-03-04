"""
I/O routines for working with brick files.

See doc/DESI_SPECTRO_REDUX/PRODNAME/bricks/BRICKID/brick-BRICKID.rst in desiDataModel
for a description of the brick file data model.
"""

import os
import os.path

import numpy as np
import astropy.io.fits

import desispec.io.util

class Brick(object):
	"""
	Represents objects in a single band (b,r,z) and brick.

	The constructor will open an existing file and create a new file and parent
	directory if necessary.  The :meth:`close` method must be called for any updates
	or new data to be recorded. Successful completion of the constructor does not
	guarantee that :meth:`close` will suceed.

	Args:
		path(str): Path to the brick file to open.
		mode(str): File access mode to use. Should normally be 'readonly' or 'update'.
			Use 'update' to create a new file and its parent directory if necessary.
		header: An optional header specification used to create a new file. See
			:func:`desispec.io.util.fitsheader` for details on allowed values.

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
			self.hdu_list = astropy.io.fits.HDUList([hdu0,hdu1,hdu2,hdu3])
		else:
			self.hdu_list = astropy.io.fits.open(path,mode = self.mode)
			if len(self.hdu_list) != 4:
				raise RuntimeError('Unexpected number of HDUs (%d) in %s' % (
					len(self.hdu_list),self.path))

	def add_objects(self,flux,ivar,wave,resolution,object_data):
		"""
		Add a list of objects.
		"""
		if self.mode != 'update':
			raise RuntimeError('Can only add objects in update mode.')
		if len(self.hdu_list[0].data) > 0:
			print self.hdu_list[0].data.shape,flux.shape
			self.hdu_list[0].data = np.concatenate((self.hdu_list[0].data,flux,))
		else:
			self.hdu_list[0].data = flux

	def close(self):
		"""
		Write any updates and close the brick file.
		"""
		if self.mode == 'update':
			self.hdu_list.writeto(self.path,clobber = True)
		self.hdu_list.close()
