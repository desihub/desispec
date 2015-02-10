import numpy as np
import scipy,scipy.linalg,scipy.interpolate
from desispec.log import get_logger

def cholesky_solve(A,B,overwrite=False,lower=False) :
    posv, = scipy.linalg.get_lapack_funcs(('posv',), (A,))
    L,X,status=posv(A,B,lower=lower,overwrite_a=overwrite)
    
    if status  != 0 :
        get_logger().error("dposv status=%d"%status)
        raise Exception("cholesky_solve_and_invert error dposv status=%d"%status)
    
    return X

def cholesky_solve_and_invert(A,B,overwrite=False,lower=False) :
    posv, = scipy.linalg.get_lapack_funcs(('posv',), (A,))
    L,X,status=posv(A,B,lower=lower,overwrite_a=overwrite)
    
    if status != 0  :
        get_logger().error("dposv status=%d"%status)
        raise Exception("cholesky_solve_and_invert error dposv status=%d"%status)
    
    potri, = scipy.linalg.get_lapack_funcs(('potri',), (L,))
    inv,status=potri(L,lower=(not lower)) # 'not lower' is not a mistake, there is a BUG in scipy!!!!   
        
    if status  != 0 :
        get_logger().error("dpotri status=%d"%status)
        raise Exception("cholesky_solve_and_invert error dpotri status=%d"%status)


    #this routine can lead to nan without warning !!!
    tmp=np.diagonal(inv)
    if np.isnan(np.sum(tmp)) :
        get_logger().error("covariance has NaN")
        raise Exception("covariance has NaN")

    
    #symmetrize Ai
    if True :
        for i in range(inv.shape[0]) :
            for j in range(i) :
                if  not lower :
                    inv[i,j]=inv[j,i]
                else :
                    inv[j,i]=inv[i,j]
    return X,inv


def spline_fit(output_wave,input_wave,input_flux,required_resolution,input_ivar=None,order=3) :
    """
    performs a spline fit 
    """
    if input_ivar is not None :
        selection=np.where(input_ivar>0)[0]
        if selection.size < 2 :
            log=get_logger()
            log.error("cannot do spline fit because only %d values with ivar>0"%selection.size)
            raise Error 
        w1=input_wave[selection[0]]
        w2=input_wave[selection[-1]]
    else :
        w1=input_wave[0]
        w2=input_wave[-1]

    res=required_resolution
    n=int((w2-w1)/res)
    res=(w2-w1)/(n+1)
    knots=w1+res*(0.5+np.arange(n))
    toto=scipy.interpolate.splrep(input_wave,input_flux,w=input_ivar,k=order,task=-1,t=knots)
    output_flux = scipy.interpolate.splev(output_wave,toto)
    return output_flux
