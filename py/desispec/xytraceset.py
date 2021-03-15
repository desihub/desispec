"""
desispec.xytraceset
===================

Lightweight wrapper class for trace coordinates and wavelength solution, to be returned by :func:`~desispec.io.xytraceset.read_xytraceset`.
"""

class XYTraceSet(object):
    def __init__(self, xcoef, ycoef, wavemin, wavemax, npix_y, xsigcoef = None, ysigcoef = None, meta = None) :
        """
        Lightweight wrapper for trace coordinates and wavelength solution
        

        Args:
            xcoef: 2D[ntrace, ncoef] Legendre coefficient of x as a function of wavelength
            ycoef: 2D[ntrace, ncoef] Legendre coefficient of y as a function of wavelength
            wavemin : float 
            wavemax : float. wavemin and wavemax are used to define a reduced variable legx(wave,wavemin,wavemax)=2*(wave-wavemin)/(wavemax-wavemin)-1
        used to compute the traces, xccd=legval(legx(wave,wavemin,wavemax),xtrace[fiber])
        """
        from specter.util.traceset import TraceSet 

        assert(xcoef.shape[0] == ycoef.shape[0])
        if xsigcoef is not None :
            assert(xcoef.shape[0] == xsigcoef.shape[0]) 
        if ysigcoef is not None :
            assert(xcoef.shape[0] == ysigcoef.shape[0]) 
            
        self.nspec   = xcoef.shape[0]
        self.wavemin = wavemin
        self.wavemax = wavemax
        self.npix_y = npix_y
        
        self.x_vs_wave_traceset = TraceSet(xcoef,[wavemin,wavemax])
        self.y_vs_wave_traceset = TraceSet(ycoef,[wavemin,wavemax])
        
        self.xsig_vs_wave_traceset = None
        self.ysig_vs_wave_traceset = None
        
        if xsigcoef is not None :
            self.xsig_vs_wave_traceset = TraceSet(xsigcoef,[wavemin,wavemax])
        if ysigcoef is not None :
            self.ysig_vs_wave_traceset = TraceSet(ysigcoef,[wavemin,wavemax])
        
        self.wave_vs_y_traceset = None
        self.meta = meta

    def x_vs_wave(self,fiber,wavelength) :
        return self.x_vs_wave_traceset.eval(fiber,wavelength)
    
    def y_vs_wave(self,fiber,wavelength) :
        return self.y_vs_wave_traceset.eval(fiber,wavelength)
    
    def xsig_vs_wave(self,fiber,wavelength) :
        if self.xsig_vs_wave_traceset is None :
            raise RuntimeError("no xsig coefficents were read in the PSF")
        
        return self.xsig_vs_wave_traceset.eval(fiber,wavelength)
    
    def ysig_vs_wave(self,fiber,wavelength) :
        if self.ysig_vs_wave_traceset is None :
            raise RuntimeError("no ysig coefficents were read in the PSF")
            
        return self.ysig_vs_wave_traceset.eval(fiber,wavelength)
    
    def wave_vs_y(self,fiber,y) :
        if self.wave_vs_y_traceset is None :
            self.wave_vs_y_traceset = self.y_vs_wave_traceset.invert()
        return self.wave_vs_y_traceset.eval(fiber,y)
    
    def x_vs_y(self,fiber,y) :
        return self.x_vs_wave(fiber,self.wave_vs_y(fiber,y))

    def xsig_vs_y(self,fiber,y) :
        return self.xsig_vs_wave(fiber,self.wave_vs_y(fiber,y))
    
    def ysig_vs_y(self,fiber,y) :
        return self.ysig_vs_wave(fiber,self.wave_vs_y(fiber,y))
    
    """
        if self.x_vs_y_traceset is None :
            if self.wave_vs_y_traceset is None :
                self.wave_vs_y_traceset = self.y_vs_wave_traceset.invert()
            ymin  = self.wave_vs_y_traceset._xmin
            ymax  = self.wave_vs_y_traceset._xmax
            ncoef = np.max(self.x_vs_wave_traceset._coeff.shape[1],self.y_vs_wave_traceset._coeff.shape[1]) + 1. # one more deg for inversion
            ty    = np.linspace(ymin,ymax,ncoef+1)
            rty   = 2*(ty-ymin)/(ymax-ymin) - 1.0 # [-1,+1] range
            coef  = np.zeros((self.nspec,ncoef))
            for i in range(self.nspec) :
                twave = self.wave_vs_y(i,ty)
                tx    = self.x_vs_wave(i,twave)
                coef[fiber] = legfit(rty,tx,ncoef-1)
            self.x_vs_y_traceset =  TraceSet(coef, domain=[ymin, ymax])
        
        return self.x_vs_y_traceset.eval(fiber,y)
     """   
