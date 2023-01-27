"""
desispec.heliocentric
=====================

heliocentric correction routine
"""

import os
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.constants

# In restricted environments, such as ReadTheDocs, this throws
# an exception.
try:
    kpno = EarthLocation.from_geodetic(lat=31.96403 * u.deg,
                                       lon=-111.59989 * u.deg,
                                       height =  2097 * u.m)
except TypeError:
    kpno = None


def heliocentric_velocity_corr_kms(ra, dec, mjd) :
    """Heliocentric velocity correction routine.

    See http://docs.astropy.org/en/stable/coordinates/velocities.html for more details.
    The computed correction can be added to any observed radial velocity to determine
    the final heliocentric radial velocity. In other words, wavelength calibrated with
    lamps have to be multiplied by (1+vcorr/cspeed) to bring them to the heliocentric frame.

    Args:
        ra - Right ascension [degrees] in ICRS system
        dec - Declination [degrees]  in ICRS system
        mjd - Decimal Modified Julian date.  Note this should probably be type DOUBLE.

    Returns:
        vcorr - Velocity correction term, in km/s, to add to measured
            radial velocity to convert it to the heliocentric frame.
    """


    # Note:
    #
    # This gives the opposite sign from the IDL routine idlutils/pro/coord/heliocentric.pro (v5_5_17)
    # Accounting for this difference in definition, the maximum difference is about ~ 0.2 km/s


    sc = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    obstime = Time(mjd,format="mjd")
    v_kms   = sc.radial_velocity_correction('heliocentric', obstime=obstime, location=kpno).to(u.km/u.s).value
    return v_kms

def heliocentric_velocity_multiplicative_corr(ra, dec, mjd) :
    """Heliocentric velocity correction routine.

    See http://docs.astropy.org/en/stable/coordinates/velocities.html for more details.
    The computed correction can be added to any observed radial velocity to determine
    the final heliocentric radial velocity. In other words, wavelength calibrated with
    lamps have to be multiplied by (1+vcorr/cspeed) to bring them to the heliocentric frame.

    Args:
        ra  - Right ascension [degrees] in ICRS system
        dec - Declination [degrees]  in ICRS system
        mjd - Decimal Modified Julian date.  Note this should probably be type DOUBLE.

    Returns:
        (1+vcorr/c) - multiplicative term to correct the wavelength
    """

    return 1.+heliocentric_velocity_corr_kms(ra, dec, mjd)/astropy.constants.c.to(u.km/u.s).value

def barycentric_velocity_corr_kms(ra, dec, mjd) :
    """Barycentric velocity correction routine.

    See http://docs.astropy.org/en/stable/coordinates/velocities.html for more details.
    The computed correction can be added to any observed radial velocity to determine
    the final barycentric radial velocity. In other words, wavelength calibrated with
    lamps have to be multiplied by (1+vcorr/cspeed) to bring them to the heliocentric frame.

    Args:
        ra - Right ascension [degrees] in ICRS system
        dec - Declination [degrees]  in ICRS system
        mjd - Decimal Modified Julian date.  Note this should probably be type DOUBLE.

    Returns:
        vcorr - Velocity correction term, in km/s, to add to measured
            radial velocity to convert it to the heliocentric frame.
    """

    sc = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    obstime = Time(mjd,format="mjd")
    v_kms   = sc.radial_velocity_correction(obstime=obstime, location=kpno).to(u.km/u.s).value
    return v_kms

def barycentric_velocity_multiplicative_corr(ra, dec, mjd) :
    """Barycentric velocity correction routine.

    See http://docs.astropy.org/en/stable/coordinates/velocities.html for more details.
    The computed correction can be added to any observed radial velocity to determine
    the final barycentric radial velocity. In other words, wavelength calibrated with
    lamps have to be multiplied by (1+vcorr/cspeed) to bring them to the barycentric frame.

    Args:
        ra  - Right ascension [degrees] in ICRS system
        dec  - Declination [degrees]  in ICRS system
        mjd  - Decimal Modified Julian date.  Note this should probably be type DOUBLE.

    Returns:
        (1+vcorr/c)  - multiplicative term to correct the wavelength
    """

    return 1.+barycentric_velocity_corr_kms(ra, dec, mjd)/astropy.constants.c.to(u.km/u.s).value


def main() :
    """Entry-point for command-line scripts.

    Comparison test with IDL routine::

        pro helio
        dec=20
        epoch=2000
        longitude=-111.59989
        latitude=31.96403
        altitude=2097
        for j=0,2 do begin
        mjd=58600+100*j
        jd=mjd+2400000.5
        for i=0,20 do begin
            ra=(360.*i)/20
            vkms = heliocentric(ra, dec, epoch, jd=jd, longitude=longitude, latitude=latitude, altitude=altitude)
            print,"ra,dec,mjd,vkms=",ra,dec,mjd,vkms
        endfor
        endfor
        end


        ra,dec,mjd,vkms=      0.00000      20       58600      -12.407597
        ra,dec,mjd,vkms=      18.0000      20       58600      -5.1777692
        ra,dec,mjd,vkms=      36.0000      20       58600       2.8805023
        ra,dec,mjd,vkms=      54.0000      20       58600       10.978417
        ra,dec,mjd,vkms=      72.0000      20       58600       18.323295
        ra,dec,mjd,vkms=      90.0000      20       58600       24.196169
        ra,dec,mjd,vkms=      108.000      20       58600       28.022160
        ra,dec,mjd,vkms=      126.000      20       58600       29.426754
        ra,dec,mjd,vkms=      144.000      20       58600       28.272459
        ra,dec,mjd,vkms=      162.000      20       58600       24.672266
        ra,dec,mjd,vkms=      180.000      20       58600       18.978587
        ra,dec,mjd,vkms=      198.000      20       58600       11.748759
        ra,dec,mjd,vkms=      216.000      20       58600       3.6904873
        ra,dec,mjd,vkms=      234.000      20       58600      -4.4074277
        ra,dec,mjd,vkms=      252.000      20       58600      -11.752306
        ra,dec,mjd,vkms=      270.000      20       58600      -17.625179
        ra,dec,mjd,vkms=      288.000      20       58600      -21.451170
        ra,dec,mjd,vkms=      306.000      20       58600      -22.855764
        ra,dec,mjd,vkms=      324.000      20       58600      -21.701469
        ra,dec,mjd,vkms=      342.000      20       58600      -18.101277
        ra,dec,mjd,vkms=      360.000      20       58600      -12.407597
        ra,dec,mjd,vkms=      0.00000      20       58700      -23.152438
        ra,dec,mjd,vkms=      18.0000      20       58700      -27.333872
        ra,dec,mjd,vkms=      36.0000      20       58700      -29.104026
        ra,dec,mjd,vkms=      54.0000      20       58700      -28.289624
        ra,dec,mjd,vkms=      72.0000      20       58700      -24.970386
        ra,dec,mjd,vkms=      90.0000      20       58700      -19.471222
        ra,dec,mjd,vkms=      108.000      20       58700      -12.330428
        ra,dec,mjd,vkms=      126.000      20       58700      -4.2469952
        ra,dec,mjd,vkms=      144.000      20       58700       3.9878138
        ra,dec,mjd,vkms=      162.000      20       58700       11.567919
        ra,dec,mjd,vkms=      180.000      20       58700       17.751326
        ra,dec,mjd,vkms=      198.000      20       58700       21.932760
        ra,dec,mjd,vkms=      216.000      20       58700       23.702914
        ra,dec,mjd,vkms=      234.000      20       58700       22.888512
        ra,dec,mjd,vkms=      252.000      20       58700       19.569274
        ra,dec,mjd,vkms=      270.000      20       58700       14.070110
        ra,dec,mjd,vkms=      288.000      20       58700       6.9293160
        ra,dec,mjd,vkms=      306.000      20       58700      -1.1541168
        ra,dec,mjd,vkms=      324.000      20       58700      -9.3889258
        ra,dec,mjd,vkms=      342.000      20       58700      -16.969031
        ra,dec,mjd,vkms=      360.000      20       58700      -23.152438
        ra,dec,mjd,vkms=      0.00000      20       58800       19.006564
        ra,dec,mjd,vkms=      18.0000      20       58800       12.826196
        ra,dec,mjd,vkms=      36.0000      20       58800       5.1371364
        ra,dec,mjd,vkms=      54.0000      20       58800      -3.3079572
        ra,dec,mjd,vkms=      72.0000      20       58800      -11.682420
        ra,dec,mjd,vkms=      90.0000      20       58800      -19.166501
        ra,dec,mjd,vkms=      108.000      20       58800      -25.027606
        ra,dec,mjd,vkms=      126.000      20       58800      -28.692009
        ra,dec,mjd,vkms=      144.000      20       58800      -29.801014
        ra,dec,mjd,vkms=      162.000      20       58800      -28.246063
        ra,dec,mjd,vkms=      180.000      20       58800      -24.179365
        ra,dec,mjd,vkms=      198.000      20       58800      -17.998997
        ra,dec,mjd,vkms=      216.000      20       58800      -10.309937
        ra,dec,mjd,vkms=      234.000      20       58800      -1.8648438
        ra,dec,mjd,vkms=      252.000      20       58800       6.5096188
        ra,dec,mjd,vkms=      270.000      20       58800       13.993700
        ra,dec,mjd,vkms=      288.000      20       58800       19.854805
        ra,dec,mjd,vkms=      306.000      20       58800       23.519208
        ra,dec,mjd,vkms=      324.000      20       58800       24.628213
        ra,dec,mjd,vkms=      342.000      20       58800       23.073262
        ra,dec,mjd,vkms=      360.000      20       58800       19.006564

"""
    maxdiff=0.
    #read the above
    file=open(os.path.abspath(__file__))
    for line in file.readlines() :
        if line.find("ra,dec,mjd,vkms=")==0 :
            vals=line.split()
            ra=float(vals[1])
            dec=float(vals[2])
            mjd=float(vals[3])
            vkms_idl=float(vals[4])
            vkms_py = heliocentric_velocity_corr_kms(ra, dec, mjd)
            print("RA={:d} Dec={:d} MJD={:d} vcorr -IDL={:4.3f} this={:4.3f} diff={:4.3f} km/s".format(int(ra),int(dec),int(mjd),-vkms_idl,vkms_py,vkms_idl+vkms_py))
            maxdiff=max(maxdiff,np.abs(vkms_idl+vkms_py))
    file.close()
    print("maximum difference = ",maxdiff,"km/s")

if __name__ == "__main__" :
    main()
