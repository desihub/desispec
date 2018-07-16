"""
desispec.xytraceset
==============

Lightweight wrapper class for trace coordinates and wavelength solution, to be returned by io.read_xytraceset
"""

from specter.util.traceset import TraceSet 

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
        
        self.x_vs_wave_traceset = TraceSet(xcoef,[wavemin,wavemax])
        self.y_vs_wave_traceset = TraceSet(ycoef,[wavemin,wavemax])
        self.wave_vs_y_traceset = None
        self.x_vs_y_traceset    = None
        
    
    def x_vs_wave(self,fiber,wavelength) :
        return self.x_vs_wave_traceset.eval(fiber,wavelength)
    
    def y_vs_wave(self,fiber,wavelength) :
        return self.y_vs_wave_traceset.eval(fiber,wavelength)
    
    def wave_vs_y(self,fiber,y) :
        if self.wave_vs_y_traceset is None :
            self.wave_vs_y_traceset = self.y_vs_wave_traceset.invert()
        return self.wave_vs_y_traceset.eval(fiber,y)
    
    def x_vs_y(self,fiber,y) :
        FINISH THIS WORK
        twave = self.wave_vs_y(fiber,y)
        FINISH THIS WORK
