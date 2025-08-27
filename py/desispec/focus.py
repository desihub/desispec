"""
desispec.focus
==============

Utility functions for spectrographs focus.
"""

import numpy as np
from desiutil.log import get_logger


RPIXSCALE=2048.


def piston_and_tilt_to_gauge_offsets(camera_spectro,focus_plane_coefficients) :
    """
    Computes gauge offsets to apply for a given best focus plane as a function of xccd and yccd

    Args:
       camera_spectro: str, camera identifier starting with b, r or z for NIR
       focus_plane_coefficients: array with 3 values (c0,c1,c2), defining the best focus plane offsets: c0+c1*(xccd/2048-1)+c2*(yccd/2048-1) in microns

    Returns:
       dictionnary of offsets to apply, in mm, for the "TOP","LEFT" and "RIGHT" gauge
    """


    c=camera_spectro[0].upper()
    if c=="B" :
        camera="BLUE"
    elif c=="R" :
        camera="RED"
    elif c=="Z" :
        camera="NIR"
    else :
        raise ValueError("unexpected camera = '{}' (expect r4,b2,z7 ...,)".format(camera_spectro))

    log=get_logger()



    # from outil_tilt.ppt , email from Sandrine on 2020/11/12
    x_plate_gauge={}
    y_plate_gauge={}
    x_plate_gauge["TOP"]=328/2*np.sin(9*np.pi/180)
    y_plate_gauge["TOP"]=328/2*np.cos(9*np.pi/180)
    x_plate_gauge["LEFT"]=-328/2*np.cos(39*np.pi/180)
    y_plate_gauge["LEFT"]=-328/2*np.sin(39*np.pi/180)
    x_plate_gauge["RIGHT"]=328/2*np.cos(39*np.pi/180)
    y_plate_gauge["RIGHT"]=-328/2*np.sin(39*np.pi/180)
    log.debug("X_PLATE_GAUGE={}".format(x_plate_gauge))
    log.debug("Y_PLATE_GAUGE={}".format(y_plate_gauge))

    x_pix_gauge={}
    y_pix_gauge={}
    for k in ["TOP","LEFT","RIGHT"] :
        x_pix_gauge[k] = -y_plate_gauge[k]/0.015+2048
        if camera.upper() == "BLUE" or  camera.upper() == "NIR" :
            y_pix_gauge[k] = -x_plate_gauge[k]/0.015+2048
        elif camera.upper() == "RED" :
            y_pix_gauge[k] = x_plate_gauge[k]/0.015+2048
        else :
            raise ValueError("don't know camera "+camera)
    log.debug("X_PIX_GAUGE={}".format(x_pix_gauge))
    log.debug("Y_PIX_GAUGE={}".format(y_pix_gauge))

    best_focus_gauge_offset = {}
    for k in ["TOP","LEFT","RIGHT"] :
        best_focus_gauge_offset[k] = focus_plane_coefficients[0]/1000. + focus_plane_coefficients[1]*(x_pix_gauge[k]/RPIXSCALE-1)/1000.+ focus_plane_coefficients[2]*(y_pix_gauge[k]/RPIXSCALE-1)/1000.
    log.debug("Best focus gauge offsets = {} mm".format(best_focus_gauge_offset))
    return best_focus_gauge_offset

def test_gauge_offsets() :
    """
    Test function
    """
    # copied from DESI-5084 https://desi.lbl.gov/DocDB/cgi-bin/private/RetrieveFile?docid=5084;filename=DESI_doc_5084_Spectrographs_Tests_Summary_Mayall.xlsx;version=1

    # BLUE     RED	NIR
    # piston (mm)	tilt/X(deg)	tilt/Y(deg)	piston (mm)	tilt/X(deg)	tilt/Y(deg)	piston (mm)	tilt/X(deg)	tilt/Y(deg)
    log=get_logger()
    plane={}
    plane["SM1"]=[0.076,-0.00722,-0.00722,0.055,-0.01690,0.0252,0.050,-0.01765,0.01377]
    plane["SM2"]=[0.022,0.0214,0.02661,0.043,0.00854,0.02465,0.042,0.02101,0.00988]
    plane["SM3"]=[-0.012,0.01487,-0.00906,0.031,0.02189,0.0084,0.034,0.00855,0.0172]
    plane["SM4"]=[0.082,0.00588,0.0442,0.101,0.01326,0.02309,0.099,0.02772,0.03108]
    plane["SM5"]=[0.043,0.01552,-0.00571,0.048,0.00313,-0.00067,0.049,0.01882,0.01281]
    plane["SM6"]=[0.078,-0.0076,0.0023,0.053,0.01989,-0.00802,0.051,0.02632,0.01648]

    #final @WL									final @Mayall
    #Nir			Red			Blue			Nir			Red			Blue
    #Top	Left	Right	Top	Left	Right	Top	Left	Right	Top	Left	Right	Top	Left	Right	Top	Left	Right
    gauges={}
    gauges["SM1"]=[62.988,61.988,62.688,61.190,61.354,61.815,61.650,63.358,61.814,63.037,62.037,62.737,61.302,61.337,61.881,61.722,63.430,61.886]
    gauges["SM2"]=[62.380,62.214,62.231,60.307,60.707,61.032,62.050,61.967,62.205,62.369,62.313,62.270,60.334,60.708,61.142,61.998,62.085,62.205]
    gauges["SM3"]=[62.413,62.159,62.476,61.355,60.867,60.420,62.574,62.847,61.843,62.418,62.244,62.487,61.319,60.927,60.508,62.557,62.830,61.827]
    gauges["SM4"]=[61.681,62.534,62.490,60.633,61.312,61.062,62.696,62.667,62.506,61.694,62.759,62.576,60.707,61.385,61.238,62.747,62.864,62.506]
    gauges["SM5"]=[62.368,62.647,62.363,61.014,60.984,60.956,62.066,62.911,62.263,62.358,62.758,62.417,61.062,61.032,61.004,62.062,62.963,62.341]
    gauges["SM6"]=[61.971,62.483,62.674,60.959,61.504,60.886,62.200,62.959,62.082,61.942,62.620,62.738,60.954,61.613,60.959,62.274,63.033,62.156]

    for cam in ["r"] :
        for spec in ["SM1","SM2","SM3","SM4","SM5","SM6"] :
            if cam == "b" :
                i=0
            elif cam == "r" :
                i=3
            elif cam == "z" :
                i=6
            else :
                raise ValueError("cam={} ???".format(cam))

            pixelsize_um = 15.
            tilt_deg_to_pixel_coeff  = np.pi/180. * pixelsize_um * RPIXSCALE
            focus_plane_coefficients = [ 1000 * plane[spec][i] ,  tilt_deg_to_pixel_coeff * plane[spec][i+1],  tilt_deg_to_pixel_coeff * plane[spec][i+2] ] # um,um/pix

            log.debug("focus_plane_coefficients = {}".format(focus_plane_coefficients))

            if cam == "b" :
                i=6
            elif cam == "r" :
                i=3
            elif cam == "z" :
                i=0
            else :
                raise ValueError("cam={} ???".format(cam))

            # top left right
            winlight_gauges = np.array([gauges[spec][i+0],gauges[spec][i+1],gauges[spec][i+2]])
            mayall_gauges   = np.array([gauges[spec][i+0+9],gauges[spec][i+1+9],gauges[spec][i+2+9]])
            log.debug("winlight_gauges = {}".format(winlight_gauges))
            log.debug("mayall_gauges   = {}".format(mayall_gauges))

            reported_gauge_offsets  = mayall_gauges - winlight_gauges
            res = piston_and_tilt_to_gauge_offsets(cam,focus_plane_coefficients)
            my_gauge_offsets = np.array([res[key] for key in ["TOP","LEFT","RIGHT"]])

            delta_gauge = my_gauge_offsets - reported_gauge_offsets
            log.info("cam={} spec={} delta gauge (this code - DESI-5084) = {} mm".format(cam,spec,delta_gauge))
