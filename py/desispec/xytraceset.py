"""
desispec.xytraceset
==============

Lightweight wrapper class for trace coordinates and wavelength solution, to be returned by io.read_xytraceset
"""

from specter.traceset import TraceSet 

class XYTraceSet(object):
    def __init__(self, xcoef, ycoef, wavemin,wavemax) :
        """
        Lightweight wrapper for trace coordinates and wavelength solution
        

        Args:
            xcoef: 2D[ntrace, ncoef] Legendre coefficient of x as a function of wavelength
            ycoef: 2D[ntrace, ncoef] Legendre coefficient of y as a function of wavelength
            wavemin : float 
            wavemax : float. wavemin and wavemax are used to define a reduced variable legx(wave,wavemin,wavemax)=2*(wave-wavemin)/(wavemax-wavemin)-1
        used to compute the traces, xccd=legval(legx(wave,wavemin,wavemax),xtrace[fiber])
        """
        assert(xcoef.shape[0] == ycoef.shape[0]) 
        self.nspec   = xcoef.shape[0]
        self.wavemin = wavemin
        self.wavemax = wavemax
        
        self.x_vs_wave_traceset = TraceSet(xcoef,wavemin,wavemax)
        self.x_vs_wave_traceset = TraceSet(ycoef,wavemin,wavemax)
    
    def x_vs_wave(self,fiber,wavelength) :
        return self.x_vs_wave_traceset.eval(fiber,wavelength)
    
    def y_vs_wave(self,fiber,wavelength) :
        return self.y_vs_wave_traceset.eval(fiber,wavelength)
    
        
