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

from skimage.transform import probabilistic_hough_line
from skimage.morphology import binary_dilation, binary_closing

import matplotlib as mpl
import matplotlib.pyplot as plt


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

    def __init__(self, img, mask, n=11, m=11):
        """Initialize filter to clean binary 2D pixel mask.

        Parameters
        ----------
        img : ndarray or desispec.Image.pix
            Input image from spectograph (counts in pixels, 2D image).
        mask : ndarray
            Rejection mask the same size as the input image.
        n : int
            Number of rows in binary selection element.
        m : int
            Number of columns in binary selection element.
        """
        # Do a little bounds checking.
        if img.shape != mask.shape:
            raise ValueError('2D image and mask size must be identical.')

        self.img = img
        self.mask = np.zeros_like(mask, dtype=int)  # convert mask to binary
        self.mask[mask > 0] = 1
        self.repaired_mask = None

        # Set up linear selection elements for binary image processing.
        self.selems = []
        for ang in [0, 20, 45, 70, 110, 135, 160]:
            lse = LinearSelectionElement(n, m, ang)
            self.selems.append(lse)

    def repair(self):
        """Apply binary closure using selection elements specified in the class
        constructor. OR the results together.

        Returns
        -------
        repaired_mask : ndarray
            2D spectrograph cosmic ray mask with binary closure applied.
        """
        if self.repaired_mask is None:
            bc = np.zeros_like(self.mask, dtype=int)

            # Apply binary closure using each selection element. OR results.
            for se in self.selems:
                bc = bc | binary_closing(self.mask, selem=se.se)
            self.repaired_mask = bc

        return self.repaired_mask

    def plot(self, prefix='test', downsample=1):
        """Plot the input and masks.
        """
        # Plot the input.
        dpi = 256
        m = downsample_image(self.img, downsample) if downsample>1 else np.copy(self.img)
        fig = plt.figure(figsize=(m.shape[1]/dpi, m.shape[0]/dpi), dpi=dpi)
        ax = plt.Axes(fig, [0,0,1,1])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(m, cmap='gray', vmin=0, vmax=100., origin='lower')
        fig.canvas.draw()
        fig.savefig('{}_IMG.png'.format(prefix), dpi=dpi)
        plt.close()

        # Plot the mask.
        m = downsample_image(self.mask, downsample) if downsample>1 else np.copy(self.mask)
        fig = plt.figure(figsize=(m.shape[1]/dpi, m.shape[0]/dpi), dpi=dpi)
        ax = plt.Axes(fig, [0,0,1,1])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(m, cmap='gray', vmin=0, vmax=1., origin='lower')
        fig.canvas.draw()
        fig.savefig('{}_MASK.png'.format(prefix), dpi=dpi)
        plt.close()

        # Plot the repaired mask.
        if self.repaired_mask is not None:
            m = downsample_image(self.repaired_mask, downsample) if downsample>1 else np.copy(self.repaired_mask)
            fig = plt.figure(figsize=(m.shape[1]/dpi, m.shape[0]/dpi), dpi=dpi)
            ax = plt.Axes(fig, [0,0,1,1])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(m, cmap='gray', vmin=0, vmax=1., origin='lower')
            fig.canvas.draw()
            fig.savefig('{}_NEWMASK.png'.format(prefix), dpi=dpi)
            plt.close()

    def plot_chunks(self, prefix='test'):
        """Split the input and masks into an 8x8 grid and plot the results.

        Parameters
        ----------
        prefix : str
            Output file prefix.
        """
        nrow, ncol = self.img.shape
        erow = np.linspace(0, nrow, 9, dtype=int)
        ecol = np.linspace(0, ncol, 9, dtype=int)

        for i, (r0, r1) in enumerate(zip(erow[:-1], erow[1:])):
            for j, (c0, c1) in enumerate(zip(ecol[:-1], ecol[1:])):
                output = '{}_{:02d}_{:02d}.png'.format(prefix, i,j)
                subimg = self.img[r0:r1, c0:c1]
                submask = self.mask[r0:r1, c0:c1]

                if self.repaired_mask is None:
                    fig, axes = plt.subplots(1,2, figsize=(8,4))
                else:
                    fig, axes = plt.subplots(1,3, figsize=(12,4))
                    subproc = self.repaired_mask[r0:r1, c0:c1]

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

                if self.repaired_mask is not None:
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


def categorize(lines, mask):
    '''Groups lines according to their label in the mask.
    Returns the grouped lines.
    '''
    # Empty list of lists with number of label categories
    groups = [[] for i in range(np.max(mask))]

    for line in lines:
        p0, p1 = line

        # Coords are flipped to the points in the line
        cat = mask[p1[1], p1[0]]

        # Triggers if the number of groups doesn't match up for some reason.
        try:
            groups[cat].append(line)
        except:
            continue

    return groups

def cosmic_lines(mask):
    '''Finds a line representing each cosmic track.
    Returns three points representing each line.
    '''
    m = mask > 0

    # Dilation to expand disconnected cosmics
    k = np.ones((10, 10))
    m_dilate = binary_dilation(m, k)

    # Label each discrete
    m_labels, num_labels = scipy.ndimage.label(m_dilate)

    # Perform the hough transform to find straight lines
    lines = probabilistic_hough_line(m, threshold=10, line_length=5, line_gap=6, seed=1)

    # Reduces the number of lines by categorizing.
    # Checks end points of lines, sees which category they're in and returns.
    groups = categorize(lines, m_labels)

    # Gets all the start/end points of the lines contained in each group
    group_points = [[] for i in range(len(groups))]
    for i, g in enumerate(groups):
        for line in g:
            p0, p1 = line
            group_points[i].append(np.asarray(p0))
            group_points[i].append(np.asarray(p1))

    # Condenses each group down to three key points: leftmost, rightmost
    # and median x position.
    # Could easily justify using upper, lower and median y positions as well.
    new_groups = []
    for i, g in enumerate(group_points):
        t = sorted(np.asarray(g), key = lambda x: x[0])
        if len(t) > 0:
            new_groups.append([t[0], np.median(t, axis=0), t[-1]])

    return new_groups

def cosmics(mask, downsample=1):
    '''Joins the cosmics in a given mask.
    Returns an updated mask with each cosmic joined across gaps.
    '''
    # Downsample for speed.
    if downsample > 1:
        m = downsample_image(mask, downsample)
    else:
        m = np.copy(mask)

    # We need to carefully control the figure size in order
    # to get the masking image to come out exactly the same size as before.
    dpi = 256
    fig = plt.figure(figsize=(m.shape[1] / dpi, m.shape[0] / dpi), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])

    # Turn off the actual visual axes for visual niceness.
    # Then add axes to figure
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(m, cmap="gray", vmin=0, vmax=1, origin="lower")

    # Gets the points corresponding to a line representing each cosmic
    groups = cosmic_lines(m)
    for line in groups:
        p0, p1, p2 = line
        ax.plot((p0[0], p1[0], p2[0]), (p0[1], p1[1], p2[1]), linewidth=1, color="white")

    # You can only extract from the figure canvas in RGB, which is then shoved
    # into a single array of length x*y*3.
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype="uint8").reshape((m.shape[0], m.shape[1], -1))
    data = np.dot(data[...,:3], [0.299, 0.587, 0.114]) > 0 # Converts to greyscale then binary.
    plt.close()

    return data

