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

def heliocentric_shift_res_data(fibermap, resolution_data, wave, heliocor=None):
    """
    Shift resolution matrix data based on heliocentric correction mismatch.

    Args:
        fibermap: Table-like object with columns TARGET_RA, TARGET_DEC, and (MJD or MJD-OBS)
        resolution_data: (nspec, ndiag, nwave) array of resolution matrices
        wave: (nwave,) array of wavelengths
        heliocor: (float, optional) Adopted multiplicative barycentric factor.

    Returns:
        shifted_res_data: (nspec, ndiag, nwave) array of shifted resolution matrices
        (fibermap is modified in place to add/update HELIOCOR_OFFSET column)
    """
    from .resolution import resolution_mat_torows, resolution_mat_tocolumns, shift_resolution_matrix_by_pixel

    nspec, ndiag, nwave = resolution_data.shape
    shifted_res_data = np.zeros_like(resolution_data)

    c_kms = astropy.constants.c.to(u.km/u.s).value

    dwave = np.zeros_like(wave)
    dwave[1:-1] = (wave[2:] - wave[:-2]) / 2.
    dwave[0] = wave[1] - wave[0]
    dwave[-1] = wave[-1] - wave[-2]

    mjd_col = None
    if "MJD" in fibermap.colnames:
        mjd_col = "MJD"
    elif "MJD-OBS" in fibermap.colnames:
        mjd_col = "MJD-OBS"

    if 'HELIOCOR_OFFSET' not in fibermap.colnames:
        fibermap['HELIOCOR_OFFSET'] = np.zeros(len(fibermap), dtype='f4')

    if mjd_col is None:
        return resolution_data.copy()

    if heliocor is None:
        return resolution_data.copy()

    if not np.isscalar(heliocor):
        raise ValueError("heliocor must be a scalar float")

    v_field = (heliocor - 1.0) * c_kms

    for j in range(nspec):
        mjd = fibermap[mjd_col][j]

        if (not np.isnan(fibermap["TARGET_RA"][j]) and
            not np.isnan(fibermap["TARGET_DEC"][j])):

            v_fiber = barycentric_velocity_corr_kms(
                fibermap["TARGET_RA"][j],
                fibermap["TARGET_DEC"][j],
                mjd
            )
            vshift = v_fiber - v_field
            fibermap['HELIOCOR_OFFSET'][j] = vshift / c_kms

            # only apply if vshift is significant (more than 10 m/s)
            if np.abs(vshift) < 0.01:
                shifted_res_data[j] = resolution_data[j]
                continue

            # this is the velocity correction that needs to be added to the object
            # velocity that means that the resolution matrix shift needs to be of
            # opposite sign
            deltas = (-1 * vshift / c_kms) * (wave / dwave)

            kernels = resolution_mat_torows(resolution_data[j])
            shifted_kernels = shift_resolution_matrix_by_pixel(kernels, deltas)
            shifted_res_data[j] = resolution_mat_tocolumns(shifted_kernels)
        else:
            shifted_res_data[j] = resolution_data[j]

    return shifted_res_data


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
