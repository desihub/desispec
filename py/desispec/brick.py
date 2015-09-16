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

    def brickname(self, ra, dec):
        """Return string name of brick that contains (ra, dec) [degrees]
        
        Args:
            ra (float) : Right Ascension in degrees
            dec (float) : Declination in degrees
            
        Returns:
            brick name string
        """
        inra, indec = ra, dec
        dec = np.atleast_1d(dec)
        ra = np.atleast_1d(ra)
        irow = ((dec+90.0+self._bricksize/2)/self._bricksize).astype(int)
        names = list()
        for i in range(len(ra)):
            ncol = self._ncol_per_row[irow[i]]
            j = int(ra[i]/360 * ncol)
            names.append(self._brickname[irow[i]][j])

        if np.isscalar(inra):
            return names[0]
        else:
            return np.array(names)

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
def brickname(ra, dec):
    """Return brick name of brick covering (ra, dec) [degrees]
    """
    global _bricks
    if _bricks is None:
        _bricks = Bricks()

    return _bricks.brickname(ra, dec)
#
# THIS CODE SHOULD BE MOVED TO A TEST.
#
if __name__ == '__main__':
    import os
    from astropy.io import fits
    d = fits.getdata(os.getenv('HOME')+'/temp/bricks-0.50.fits')
    b = Bricks(0.5)
    ntest = 10000

    ra = np.random.uniform(0, 360, size=ntest)
    dec = np.random.uniform(-90, 90, size=ntest)
    bricknames = b.brickname(ra, dec)

    for row in range(len(b._center_dec)):
        n = len(d.BRICKROW[d.BRICKROW==row])
        if n != b._ncol_per_row[row]:
            print(row, n, len(b._center_ra[row]))

    for i in range(ntest):
        ii = np.where( (d.DEC1 <= dec[i]) & (dec[i] < d.DEC2) & (d.RA1 <= ra[i]) & (ra[i] < d.RA2) )[0][0]
        if bricknames[i] != d.BRICKNAME[ii]:
            print(bricknames[i], d.BRICKNAME[ii], ra[i], dec[i], b.brick_radec(ra[i], dec[i]), (d.RA[ii], d.DEC[ii]))
