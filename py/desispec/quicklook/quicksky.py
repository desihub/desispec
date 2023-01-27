"""
desispec.quicklook.quicksky
===========================

Here will be the sky computing and sky subtraction routines for QL.
"""
import sys
import numpy as np
from desispec.sky import SkyModel
from desispec import util
from desispec import frame as fr
import scipy
from desispec.resolution import Resolution
from desispec.linalg import cholesky_solve

def compute_sky(fframe,fibermap=None,nsig_clipping=4., apply_resolution=False):
    """
    Adding in the offline algorithm here to be able to apply resolution for sky compute.
    We will update this here as needed for quicklook.
    The original weighted sky compute still is the default.

    Args: fframe: fiberflat fielded frame object
          fibermap: fibermap object
          apply_resolution: if True, uses the resolution in the frame object to evaluate
          sky allowing fiber to fiber variation of resolution.
    """
    nspec=fframe.nspec
    nwave=fframe.nwave

    #- Check with fibermap. exit if None
    #- use fibermap from frame itself if exists

    if fframe.fibermap is not None:
        fibermap=fframe.fibermap

    if fibermap is None:
        print("Must have fibermap for Sky compute")
        sys.exit(0)

    #- get the sky
    skyfibers = np.where(fibermap['OBJTYPE'] == 'SKY')[0]
    skyfluxes=fframe.flux[skyfibers]
    skyivars=fframe.ivar[skyfibers]


    nfibers=len(skyfibers)

    if apply_resolution:
        max_iterations=100
        current_ivar=skyivars.copy()
        Rsky = fframe.R[skyfibers]
        sqrtw=np.sqrt(skyivars)
        sqrtwflux=sqrtw*skyfluxes

        chi2=np.zeros(skyfluxes.shape)

        nout_tot=0
        for iteration in range(max_iterations) :

            A=scipy.sparse.lil_matrix((nwave,nwave)).tocsr()
            B=np.zeros((nwave))
            # diagonal sparse matrix with content = sqrt(ivar)*flat of a given fiber
            SD=scipy.sparse.lil_matrix((nwave,nwave))
            # loop on fiber to handle resolution
            for fiber in range(nfibers) :
                if fiber%10==0 :
                    print("iter %d fiber %d"%(iteration,fiber))
                R = Rsky[fiber]

                # diagonal sparse matrix with content = sqrt(ivar)
                SD.setdiag(sqrtw[fiber])

                sqrtwR = SD*R # each row r of R is multiplied by sqrtw[r]

                A = A+(sqrtwR.T*sqrtwR).tocsr()
                B += sqrtwR.T*sqrtwflux[fiber]

            print("iter %d solving"%iteration)

            w = A.diagonal()>0
            A_pos_def = A.todense()[w,:]
            A_pos_def = A_pos_def[:,w]
            skyflux = B*0
            try:
                skyflux[w]=cholesky_solve(A_pos_def,B[w],rcond=None)
            except:
                print("cholesky failed, trying svd in iteration {}".format(iteration))
                skyflux[w]=np.linalg.lstsq(A_pos_def,B[w],rcond=None)[0]

            print("iter %d compute chi2"%iteration)

            for fiber in range(nfibers) :

                S = Rsky[fiber].dot(skyflux)
                chi2[fiber]=current_ivar[fiber]*(skyfluxes[fiber]-S)**2

            print("rejecting")

            nout_iter=0
            if iteration<1 :
                # only remove worst outlier per wave
                # apply rejection iteratively, only one entry per wave among fibers
                # find waves with outlier (fastest way)
                nout_per_wave=np.sum(chi2>nsig_clipping**2,axis=0)
                selection=np.where(nout_per_wave>0)[0]
                for i in selection :
                    worst_entry=np.argmax(chi2[:,i])
                    current_ivar[worst_entry,i]=0
                    sqrtw[worst_entry,i]=0
                    sqrtwflux[worst_entry,i]=0
                    nout_iter += 1
            else :
                # remove all of them at once
                bad=(chi2>nsig_clipping**2)
                current_ivar *= (bad==0)
                sqrtw *= (bad==0)
                sqrtwflux *= (bad==0)
                nout_iter += np.sum(bad)

            nout_tot += nout_iter

            sum_chi2=float(np.sum(chi2))
            ndf=int(np.sum(chi2>0)-nwave)
            chi2pdf=0.
            if ndf>0 :
                chi2pdf=sum_chi2/ndf
            print("iter #%d chi2=%f ndf=%d chi2pdf=%f nout=%d"%(iteration,sum_chi2,ndf,chi2pdf,nout_iter))

            if nout_iter == 0 :
                break

        print("nout tot=%d"%nout_tot)
        # solve once again to get deconvolved sky variance
        #skyflux,skycovar=cholesky_solve_and_invert(A.todense(),B)
        skyflux = np.linalg.lstsq(A.todense(),B,rcond=None)[0]
        skycovar = np.linalg.pinv(A.todense())
        #- sky inverse variance, but incomplete and not needed anyway
        # skyvar=np.diagonal(skycovar)
        # skyivar=(skyvar>0)/(skyvar+(skyvar==0))

        # Use diagonal of skycovar convolved with mean resolution of all fibers
        # first compute average resolution
        #- computing mean from matrix itself
        R= (fframe.R.sum()/fframe.nspec).todia()
        #mean_res_data=np.mean(fframe.resolution_data,axis=0)
        #R = Resolution(mean_res_data)
        # compute convolved sky and ivar
        cskycovar=R.dot(skycovar).dot(R.T.todense())
        cskyvar=np.diagonal(cskycovar)
        cskyivar=(cskyvar>0)/(cskyvar+(cskyvar==0))

        # convert cskyivar to 2D; today it is the same for all spectra,
        # but that may not be the case in the future
        finalskyivar = np.tile(cskyivar, nspec).reshape(nspec, nwave)

        # Convolved sky
        finalskyflux = np.zeros(fframe.flux.shape)
        for i in range(nspec):
            finalskyflux[i] = fframe.R[i].dot(skyflux)

        # need to do better here
        mask = (finalskyivar==0).astype(np.uint32)

    else: #- compute weighted average sky ignoring the fiber/wavelength resolution
        if skyfibers.shape[0] > 1:

            weights=skyivars
            #- now get weighted meansky and ivar
            meanskyflux=np.average(skyfluxes,axis=0,weights=weights)
            wtot=weights.sum(axis=0)
            werr2=(weights**2*(skyfluxes-meanskyflux)**2).sum(axis=0)
            werr=np.sqrt(werr2)/wtot
            meanskyivar=1./werr**2
        else:
            meanskyflux=skyfluxes
            meanskyivar=skyivars

        #- Create a 2d- sky model replicating this
        finalskyflux=np.tile(meanskyflux,nspec).reshape(nspec,nwave)
        finalskyivar=np.tile(meanskyivar,nspec).reshape(nspec,nwave)
        mask=fframe.mask

    skymodel=SkyModel(fframe.wave,finalskyflux,finalskyivar,mask)
    return skymodel


def subtract_sky(fframe,skymodel):
    """
    skymodel: skymodel object.
    fframe: frame object to do the sky subtraction, should be already fiber flat fielded
    need same number of fibers and same wavelength grid
    """
    #- Check number of specs
    assert fframe.nspec == skymodel.nspec
    assert fframe.nwave == skymodel.nwave

    #- check same wavelength grid, die if not
    if not np.allclose(fframe.wave, skymodel.wave):
        message = "frame and sky not on same wavelength grid"
        raise ValueError(message)

    #SK. This wouldn't work since not all properties of the input
    #frame is modified. Just modify input frame directly instead!

    fframe.flux= fframe.flux-skymodel.flux
    fframe.ivar = util.combine_ivar(fframe.ivar.clip(1e-8), skymodel.ivar.clip(1e-8))
    fframe.mask = fframe.mask | skymodel.mask
    #- create a frame object now
    #sframe=fr.Frame(fframe.wave,sflux,sivar,smask,fframe.resolution_data,meta=fframe.meta,fibermap=fframe.fibermap)
    return fframe

