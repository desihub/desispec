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
from skimage.morphology import binary_dilation

import matplotlib.pyplot as plt


def downsample_image(image, n):
    '''Downsample input image n x n
    Returns resampled images with shape = image.shape//n
    '''
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
    # Downsample the image by 2 for speed
    m2 = downsample_image(mask, 2) > 0

    # Dilation to expand disconnected cosmics
    k = np.ones((10, 10))
    m_dilate = binary_dilation(m2, k)

    # Label each discrete
    m_labels, num_labels = scipy.ndimage.label(m_dilate)

    # Perform the hough transform to find straight lines
    lines = probabilistic_hough_line(m2, threshold=10, line_length=5, line_gap=6, seed=1)

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

def cosmics(mask):
    '''Joins the cosmics in a given mask.
    Returns an updated mask with each cosmic joined across gaps.
    '''
    # Downsample for speed.
    m = downsample_image(mask, 2)

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
    groups = cosmic_lines(mask)
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

