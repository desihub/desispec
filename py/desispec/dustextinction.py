"""
desispec.dustextinction
========================

Milky Way dust extinction curve routines.
"""
from __future__ import absolute_import
import numpy as np

def ext_odonnell(wave,Rv=3.1) :
    """  Return extinction curve from Odonnell (1994), defined in the wavelength
    range [3030,9091] Angstroms.  Outside this range, use CCM (1989).
    
    Args:
        wave : 1D array of vacuum wavelength [Angstroms] 
        Rv   : Value of R_V (scalar); default is 3.1
    
     Returns:
        1D array of A(lambda)/A(V)
    """
    
    # direct python translation of idlutils/pro/dust/ext_odonnell.pro
    
    A  = np.zeros(wave.shape)
    xx = 10000. / wave
    
    optical_waves = (xx>=1.1)&(xx<=3.3)
    other_waves   = (xx<1.1)|(xx>3.3)
    
    if np.sum(optical_waves)>0 :
        yy = xx[optical_waves]-1.82
        afac = 1.0 + 0.104*yy - 0.609*yy**2 + 0.701*yy**3 + 1.137*yy**4 - 1.718*yy**5 - 0.827*yy**6 + 1.647*yy**7 - 0.505*yy**8
        bfac = 1.952*yy + 2.908*yy**2 - 3.989*yy**3 - 7.985*yy**4 + 11.102*yy**5 + 5.491*yy**6 - 10.805*yy**7 + 3.347*yy**8
        A[optical_waves] = afac + bfac / Rv
    if np.sum(other_waves)>0 :
        A[other_waves]   =  ext_ccm(wave[other_waves],Rv=3.1)

    return A


def ext_ccm(wave,Rv=3.1) :
    """  Return extinction curve from CCM (1989), defined in the wavelength
    range [1250,33333] Angstroms.
    
    Args:
        wave : 1D array of vacuum wavelength [Angstroms] 
        Rv   : Value of R_V (scalar); default is 3.1
    
     Returns:
        1D array of A(lambda)/A(V)
    """

    # direct python translation of idlutils/pro/dust/ext_ccm.pro
    # numeric values checked with other implementation
    
    A  = np.zeros(wave.shape)
    xx = 10000. / wave
   
    
    # Limits for CCM fitting function
    qLO  = (xx > 8.0)                # No data, lambda < 1250 Ang
    qUV  = (xx > 3.3)&(xx <= 8.0)    # UV + FUV
    qOPT = (xx > 1.1)&(xx <= 3.3)    #  Optical/NIR
    qIR  = (xx > 0.3)&(xx <= 1.1)    # IR
    qHI  = (xx <= 0.3)               # No data, lambda > 33,333 Ang
    
    # For lambda < 1250 Ang, arbitrarily return Alam=5
    if np.sum(qLO) > 0 :
        A[qLO] = 5.0

    if np.sum(qUV) > 0 :
        xt = xx[qUV]
        afac = 1.752 - 0.316*xt - 0.104 / ( (xt-4.67)**2 + 0.341 )
        bfac = -3.090 + 1.825*xt + 1.206 / ( (xt-4.62)**2 + 0.263 )
        
        qq = (xt >= 5.9)&(xt <= 8.0)
        if np.sum(qq)> 0 :
            Fa = -0.04473*(xt[qq]-5.9)**2 - 0.009779*(xt[qq]-5.9)**3
            Fb = 0.2130*(xt[qq]-5.9)**2 + 0.1207*(xt[qq]-5.9)**3
            afac[qq] += Fa
            bfac[qq] += Fb
        
        A[qUV] = afac + bfac / Rv
      
    if np.sum(qOPT) > 0 :
        yy = xx[qOPT] - 1.82
        afac = 1.0 + 0.17699*yy - 0.50447*yy**2 - 0.02427*yy**3 + 0.72085*yy**4 + 0.01979*yy**5 - 0.77530*yy**6 + 0.32999*yy**7
        bfac = 1.41338*yy + 2.28305*yy**2 + 1.07233*yy**3 - 5.38434*yy**4 - 0.62251*yy**5 + 5.30260*yy**6 - 2.09002*yy**7
        A[qOPT] = afac + bfac / Rv
   
    if np.sum(qIR) > 0 :
        yy = xx[qIR]**1.61
        afac = 0.574*yy
        bfac = -0.527*yy
        A[qIR] = afac + bfac / Rv
      
    # For lambda > 33,333 Ang, arbitrarily extrapolate the IR curve
    if np.sum(qHI) > 0 :
        yy = xx[qHI]**1.61
        afac = 0.574*yy
        bfac = -0.527*yy
        A[qHI] = afac + bfac / Rv

    return A
