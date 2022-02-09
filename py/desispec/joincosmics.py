"""
desispec.cosmics-joiner
================

Utility functions to join detected cosmic rays
"""


from desiutil.log import get_logger
import numpy as np
import os
import scipy.ndimage
import time

from desispec.maskbits import ccdmask
from desispec.maskbits import specmask

try:
    # Note: scikit-image is not part of desiconda.
    from skimage.morphology import binary_closing
    from skimage import __version__ as _skimage_version
except ImportError as e:
    _skimage_version = '0.0.0'
    # If scikit-image is not available, redefine the interface.
    def binary_dilation(image, selem=None, out=None):
        if out is None:
            out = np.empty(image.shape, dtype=bool)
        scipy.ndimage.binary_dilation(image, structure=selem, output=out)
        return out

    def binary_erosion(image, selem=None, out=None):
        if out is None:
            out = np.empty(image.shape, dtype=bool)
        scipy.ndimage.binary_erosion(image, structure=selem, output=out, border_value=True)
        return out

    def binary_closing(image, selem=None, out=None):
        dilated = binary_dilation(image, selem)
        out = binary_erosion(dilated, selem, out=out)
        return out



class LinearSelectionElement:
    """Define a selection element for morphological binary image processing."""

    def __init__(self, n, m, angle):
        """This will produce an n x m selection element with a line going
        through the center according to some angle.

        Parameters
        ----------
        n : int
            Number of rows in selection element.
        m : int
            Number of columns in selection element.
        angle : float
            Angle of line through center, in deg [0,180].
        """
        self.se = None
        self.angle = angle

        se = np.zeros((m,n), dtype=int)
        xc, yc = n//2, m//2 # row, col

        if angle >= 0 and angle < 45:
            b = np.tan(np.deg2rad(angle))
        elif angle >= 45 and angle < 90:
            b = np.tan(np.deg2rad(90 - angle))
        elif angle >= 90 and angle < 135:
            b = np.tan(np.deg2rad(angle-90))
        elif angle >= 135 and angle < 180:
            b = np.tan(np.deg2rad(180-angle))
        else:
            raise ValueError('Angle ({}) must be in [0,180]'.format(angle))

        for x in range(0, n):
            y = int(yc + b*(x-xc))
            if y >= 0 and y < m:
                se[y,x] = 1

        if angle < 45:
            self.se = se
        elif angle >= 45 and angle < 90:
            self.se = se.T
        elif angle >= 90 and angle < 135:
            self.se = se.T[:,::-1]
        else:
            self.se = se[:,::-1]

    def plot(self):
        """Return a plot of the selection element (a bitmap).

        Returns
        -------
        fig : matplotlib.Figure
            Figure object for plotting/saving.
        """
        #- Isolated mpl imports to work in batch with no $DISPLAY
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        n, m = self.se.shape
        fig, ax = plt.subplots(1,1, figsize=(0.2*n, 0.2*m), tight_layout=True)
        ax.imshow(self.se, cmap='gray', origin='lower',
                  interpolation='nearest', vmin=0, vmax=1)
        ax.xaxis.set_major_locator(mpl.ticker.LinearLocator(n+1))
        ax.yaxis.set_major_locator(mpl.ticker.LinearLocator(m+1))
        ax.set(xticklabels=[], yticklabels=[])
        ax.grid(color='gray')
        ax.tick_params(axis='both', length=0)
        return fig


class RepairMask:

    def __init__(self, n=11, m=11):
        """Initialize filter to clean binary 2D pixel mask using binary closure.

        Parameters
        ----------
        n : int
            Number of rows in binary selection element.
        m : int
            Number of columns in binary selection element.
        """
        # Set up linear selection elements for binary image processing.
        self.selems = []
        for ang in [0, 20, 45, 70, 110, 135, 160]:
            lse = LinearSelectionElement(n, m, ang)
            self.selems.append(lse)

    def repair(self, mask):
        """Apply binary closure using selection elements specified in the class
        constructor. OR the results together.

        Parameters
        ----------
        mask : ndarray
            2D rejection mask for spectrograph images.

        Returns
        -------
        repaired_mask : ndarray
            2D spectrograph cosmic ray mask with binary closure applied.
        """
        # Convert mask to binary.
        bmask = np.zeros(mask.shape, dtype=mask.dtype)
        bmask[mask > 0] = 1

        # Apply binary closure using each selection element. OR results.
        bc = np.zeros(mask.shape, dtype=mask.dtype)

        for se in self.selems:
            if _skimage_version < '0.19.0':
                bc = bc | binary_closing(bmask, selem=se.se)
            else:
                bc = bc | binary_closing(bmask, footprint=se.se)

        return bc

    def plot(self, img, mask, repaired_mask=None, prefix='test', downsample=1):
        """Plot the input and masks for testing.

        Parameters
        ----------
        img : ndarray
            2D spectrograph image.
        mask : ndarray
            2D rejection mask for spectrograph images.
        repaired_mask : ndarray or None
            Repaired 2D rejection mask.
        prefix : str
            Prefix path for output file names.
        downsample : int
            Downsample factor for saving large images.

        Returns
        -------
        fig : matplotlib.Figure
            Figure object for saving/writing.
        """
        #- Isolated mpl imports to work in batch with no $DISPLAY
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        # Plot the input.
        dpi = 256
        m = downsample_image(img, downsample) if downsample>1 else np.copy(img)
        fig = plt.figure(figsize=(m.shape[1]/dpi, m.shape[0]/dpi), dpi=dpi)
        ax = plt.Axes(fig, [0,0,1,1])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(m, cmap='gray', vmin=0, vmax=100., origin='lower')
        fig.canvas.draw()
        fig.savefig('{}_IMG.png'.format(prefix), dpi=dpi)
        plt.close()

        # Plot the mask.
        m = downsample_image(mask, downsample) if downsample>1 else np.copy(mask)
        fig = plt.figure(figsize=(m.shape[1]/dpi, m.shape[0]/dpi), dpi=dpi)
        ax = plt.Axes(fig, [0,0,1,1])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(m, cmap='gray', vmin=0, vmax=1., origin='lower')
        fig.canvas.draw()
        fig.savefig('{}_MASK.png'.format(prefix), dpi=dpi)
        plt.close()

        # Plot the repaired mask.
        if repaired_mask is not None:
            m = downsample_image(repaired_mask, downsample) if downsample>1 else np.copy(repaired_mask)
            fig = plt.figure(figsize=(m.shape[1]/dpi, m.shape[0]/dpi), dpi=dpi)
            ax = plt.Axes(fig, [0,0,1,1])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(m, cmap='gray', vmin=0, vmax=1., origin='lower')
            fig.canvas.draw()
            fig.savefig('{}_NEWMASK.png'.format(prefix), dpi=dpi)
            plt.close()

    def plot_chunks(self, img, mask, repaired_mask=None, prefix='test'):
        """Split the input and masks into an 8x8 grid and plot the results.

        Parameters
        ----------
        img : ndarray
            2D spectrograph image.
        mask : ndarray
            2D rejection mask for spectrograph images.
        repaired_mask : ndarray or None
            Repaired 2D rejection mask.
        prefix : str
            Prefix path for output file names.

        Returns
        -------
        fig : matplotlib.Figure
            Figure object for saving/writing.
        """
        #- Isolated mpl imports to work in batch with no $DISPLAY
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        nrow, ncol = img.shape
        erow = np.linspace(0, nrow, 9, dtype=int)
        ecol = np.linspace(0, ncol, 9, dtype=int)

        for i, (r0, r1) in enumerate(zip(erow[:-1], erow[1:])):
            for j, (c0, c1) in enumerate(zip(ecol[:-1], ecol[1:])):
                output = '{}_{:02d}_{:02d}.png'.format(prefix, i,j)
                subimg = img[r0:r1, c0:c1]
                submask = mask[r0:r1, c0:c1]

                if repaired_mask is None:
                    fig, axes = plt.subplots(1,2, figsize=(8,4))
                else:
                    fig, axes = plt.subplots(1,3, figsize=(12,4))
                    subproc = repaired_mask[r0:r1, c0:c1]

                ax = axes[0]
                im = ax.imshow(subimg, cmap='gray', origin='lower', interpolation='nearest', vmin=0, vmax=100)
                ax.set(xticks=[], yticks=[], title='{}: IMAGE'.format(prefix))
                ax.text(0.02,0.02, '{}:{}, {}:{}'.format(r0,r1,c0,c1), color='yellow', fontsize=8,
                        transform=ax.transAxes)
                ax.text(0.02,0.96, '{}, {}'.format(i,j), color='yellow', fontsize=8,
                        transform=ax.transAxes)
                
                ax = axes[1]
                im = ax.imshow(submask, cmap='gray', origin='lower', interpolation='nearest')
                ax.set(xticks=[], yticks=[], title='{}: MASK'.format(prefix))

                if repaired_mask is not None:
                    ax = axes[2]
                    im = ax.imshow(subproc, cmap='gray', origin='lower', interpolation='nearest')
                    ax.set(xticks=[], yticks=[], title='{}: REPAIRED'.format(prefix));

                fig.tight_layout()
                fig.savefig(output, dpi=150)
                plt.close()


def downsample_image(image, n):
    """Downsample 2D input image n x n.

    Parameters
    ----------
    image : ndarray
        2D input image.
    n : int
        Downsampling factor, applied to both image dimensions.
    Returns
    -------
    result : ndarray
        Resampled image with shape = image.shape//n.
    """
    ny, nx = image.shape
    ny = (ny//n) * n
    nx = (nx//n) * n
    result = image[0:ny, 0:nx].reshape(ny//n,n,nx//n,n).mean(axis=-1).mean(axis=-2)
    return result

