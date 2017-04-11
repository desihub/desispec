"""
desispec.brick
==============

Code for calculating bricks, which are a tiling of the sky with the following
properties:

- bricks form rows in dec like a brick wall; edges are constant RA or dec
- they are rectangular with longest edge shorter or equal to bricksize
- circles at the poles with diameter=bricksize
- there are an even number of bricks per row

Use this with caution!  In most cases you should be propagating brick
info from input targeting, not recalculating brick locations and names.
"""

from __future__ import absolute_import, division, print_function

import numpy as np

class Bricks(object):
    """Bricks Object
    """
    def __init__(self, bricksize=0.5):
        """Create Bricks object such that all bricks have longest size < bricksize
        """
        #- Brick row centers and edges
        center_dec = np.arange(-90.0, +90.0+bricksize/2, bricksize)
        edges_dec = np.arange(-90.0-bricksize/2, +90.0+bricksize, bricksize)
        nrow = len(center_dec)

        #- How many columns per row: even number, no bigger than bricksize
        ncol_per_row = np.zeros(nrow, dtype=int)
        for i in range(nrow):
            declo = np.abs(center_dec[i])-bricksize/2
            n = (360/bricksize * np.cos(declo*np.pi/180))
            ncol_per_row[i] = int(np.ceil(n/2)*2)

        #- special cases at the poles
        ncol_per_row[0] = 1
        ncol_per_row[-1] = 1

        #- ra
        center_ra = list()
        edges_ra = list()
        for i in range(nrow):
            edges = np.linspace(0, 360, ncol_per_row[i]+1)
            edges_ra.append( edges )
            center_ra.append( 0.5*(edges[0:-1]+edges[1:]) )
            ### dra = edges[1]-edges[0]
            ### center_ra.append(dra/2 + np.arange(ncol_per_row[i])*dra)

        #- More special cases at the poles
        edges_ra[0] = edges_ra[-1] = np.array([0, 360])
        center_ra[0] = center_ra[-1] = np.array([180,])

        #- Brick names [row, col]
        brickname = list()
        for i in range(nrow):
            pm = 'p' if center_dec[i] >= 0 else 'm'
            dec = center_dec[i]
            names = list()
            for j in range(ncol_per_row[i]):
                ra = center_ra[i][j]
                names.append('{:04d}{}{:03d}'.format(int(ra*10), pm, int(abs(dec)*10)))
            brickname.append(names)

        self._bricksize = bricksize
        self._ncol_per_row = ncol_per_row
        self._brickname = brickname
        self._center_dec = center_dec
        self._edges_dec = edges_dec
        self._center_ra = center_ra
        self._edges_ra = edges_ra

    @property
    def bricksize(self):
        return self._bricksize

    def brickname(self, ra, dec):
        """Return string name of brick that contains (ra, dec) [degrees]

        Args:
            ra (array or float) : Right Ascension in degrees
            dec (array or float) : Declination in degrees

        Returns:
            brick name string or array of strings
        """
        inra, indec = ra, dec
        dec = np.atleast_1d(dec)
        ra = np.atleast_1d(ra) % 360
        irow = ((dec+90.0+self._bricksize/2)/self._bricksize).astype(int)

        ncol = self._ncol_per_row[irow]
        jj = (ra/360.0 * ncol).astype(int)
        names = np.empty(len(ra), dtype='U8')
        for thisrow in set(irow):
            these = np.where(thisrow == irow)[0]
            names[these] = np.array(self._brickname[thisrow])[jj[these]]

        if np.isscalar(inra):
            return names[0]
        else:
            return names

    def brick_radec(self, ra, dec):
        """Return center (ra,dec) of brick that contains input (ra, dec) [deg]
        """
        inra, indec = ra, dec
        dec = np.asarray(dec)
        ra = np.asarray(ra)
        irow = ((dec+90.0+self._bricksize/2)/self._bricksize).astype(int)
        jcol = (ra/360 * self._ncol_per_row[irow]).astype(int)

        if np.isscalar(inra):
            xra = self._center_ra[irow][jcol]
            xdec = self._center_dec[irow]
        else:
            xra = np.array([self._center_ra[i][j] for i,j in zip(irow, jcol)])
            xdec = self._center_dec[irow]

        return xra, xdec

_bricks = None
def brickname(ra, dec, bricksize=0.5):
    """Return brick name of brick covering (ra, dec) [degrees]
    """
    global _bricks
    if _bricks is None or _bricks.bricksize != bricksize:
        _bricks = Bricks(bricksize=bricksize)

    return _bricks.brickname(ra, dec)
