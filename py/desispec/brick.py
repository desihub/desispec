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
    def __init__(self, bricksize=0.25):
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

        #ADM to calculate brick areas, we can't exceed the poles
        edges_dec_pole_limit = edges_dec
        edges_dec_pole_limit[0] = -90.
        edges_dec_pole_limit[-1] = 90.

        #- Brick names [row, col]        
        brickname = list()
        #ADM brick areas [row, col]
        brickarea = list()

        for i in range(nrow):
            pm = 'p' if center_dec[i] >= 0 else 'm'
            dec = center_dec[i]
            names = list()
            for j in range(ncol_per_row[i]):
                ra = center_ra[i][j]
                names.append('{:04d}{}{:03d}'.format(int(ra*10), pm, int(abs(dec)*10)))
            brickname.append(names)
            #ADM integrate area factors between Dec edges and RA edges in degrees
            decfac = np.diff(np.degrees(np.sin(np.radians(edges_dec_pole_limit[i:i+2]))))
            rafac = np.diff(edges_ra[i])
            brickarea.append(list(rafac*decfac))

        self._bricksize = bricksize
        self._ncol_per_row = ncol_per_row
        self._brickname = brickname
        self._brickarea = brickarea
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

    def brickid(self, ra, dec):
        """Return the BRICKID for a given location

        Parameters
        ----------
        ra : array_like.
            The Right Ascensions of the locations of interest
        dec : array_like.
            The Declinations of the locations of interest
        
        Returns
        -------
        brickid : array_like.
            The legacysurvey BRICKID at the locations of interest
        """
        #ADM record whether the user wanted non-array behavior
        inscalar = np.isscalar(ra)

        #ADM enforce array behavior and correct for wraparound
        ra = np.atleast_1d(ra) % 360
        dec = np.atleast_1d(dec)

        #ADM the brickrow based on the declination
        brickrow = ((dec+90.0+self._bricksize/2)/self._bricksize).astype(int)

        #ADM the brickcolumn based on the RA
        ncol = self._ncol_per_row[brickrow]
        brickcol = (ra/360.0 * ncol).astype(int)

        #ADM the total number of BRICKIDs at the START of a given row
        ncolsum = np.cumsum(np.append(0,self._ncol_per_row))

        #ADM the BRICKID is just the sum of the number of columns up until
        #ADM the row of interest, and the number of columns along that row
        #ADM accounting for the indexes of the columns starting at 0
        brickid = ncolsum[brickrow] + brickcol + 1

        #ADM returns the brickid as a scalar or array (depending on what was passed)
        if inscalar:
            return brickid[0]
        return brickid

    def brickarea(self, ra, dec):
        """Return the area of the brick that given locations lie in

        Parameters
        ----------
        ra : array_like.
            The Right Ascensions of the locations of interest
        dec : array_like.
            The Declinations of the locations of interest
        
        Returns
        -------
        brickarea : array_like.
            The areas of the brick in which the locations of interest lie
        """
        #ADM record whether the user wanted non-array behavior
        inscalar = np.isscalar(ra)

        #ADM enforce array behavior and correct for wraparound
        ra = np.atleast_1d(ra) % 360
        dec = np.atleast_1d(dec)

        #ADM the brickrow based on the declination
        brickrow = ((dec+90.0+self._bricksize/2)/self._bricksize).astype(int)

        #ADM the brickcolumn based on the RA
        ncol = self._ncol_per_row[brickrow]
        brickcol = (ra/360.0 * ncol).astype(int)

        #ADM the list of areas to return
        areas = np.empty(len(ra), dtype='<f4')

        #ADM grab the areas from the class
        for row in set(brickrow):
            cols = np.where(row == brickrow)
            areas[cols] = np.array(self._brickarea[row])[brickcol[cols]]

        #ADM returns the area as a scalar or array (depending on what was passed)
        if inscalar:
            return areas[0]
        return areas

    def brickvertices(self, ra, dec):
        """Return the vertices in RA/Dec of the brick that given locations lie in

        Parameters
        ----------
        ra : array_like.
            The Right Ascensions of the locations of interest
        dec : array_like.
            The Declinations of the locations of interest
        
        Returns
        -------
        vertices : array_like.
            The 4 vertices of the brick in which the locations of interest lie in (an
            array with 4 columns of (RA, Dec) and len(ra) rows)

        Notes
        -----
        The vertices are ordered counter-clockwise from the minimum (RA, Dec)            
        """
        #ADM record whether the user wanted non-array behavior
        inscalar = np.isscalar(ra)

        #ADM enforce array behavior and correct for wraparound
        ra = np.atleast_1d(ra) % 360
        dec = np.atleast_1d(dec)

        #ADM the brickrow based on the declination
        brickrow = ((dec+90.0+self._bricksize/2)/self._bricksize).astype(int)

        #ADM the brickcolumn based on the RA
        ncol = self._ncol_per_row[brickrow]
        brickcol = (ra/360.0 * ncol).astype(int)

        #ADM grab the edges from the class
        ramin, ramax = np.array([self._edges_ra[row][col:col+2] for row,col in zip(brickrow, brickcol)]).T
        decmin, decmax = self._edges_dec[brickrow], self._edges_dec[brickrow+1]
        vertices = np.reshape(
            np.vstack([ramin,decmin,ramax,decmin,ramax,decmax,ramin,decmax]).T
            ,(len(ra),4,2))

        #ADM return the vertex array with one less dimension if a scalar was passed
        if inscalar:
            return vertices[0]
        return vertices

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
def brickname(ra, dec, bricksize=0.25):
    """Return brick name of brick covering (ra, dec) [degrees]
    """
    global _bricks
    if _bricks is None or _bricks.bricksize != bricksize:
        _bricks = Bricks(bricksize=bricksize)

    return _bricks.brickname(ra, dec)
