# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
from astropy.modeling import models, fitting
from desispec.hartmann.centroid import centroid
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
#import pdb
#import photutils as phot

def tiedXmean(GaussTrapModel):
	return GaussTrapModel.x_mean_0

def tiedYmean(GaussTrapModel):
	return GaussTrapModel.y_mean_0

def PSF_fit(im, gaussian=True, trapezoidal=True, moffat=True, maxiter=200,acc=1e-7, estimates=None, doSkySub=True):
	"""
	INPUTS:
	------
	
	im: a small image with a reasonnably well centered source
	
	estimates: a dictionnary giving estimates of the best fitting gaussian:
		estimates={'amplitude':a0,'x_mean':x0,'y_mean':y0,'x_stddev':sigmaX0,'y_stddev':sigmaY0}
	
	doSkySub: if set, computes the sky level on a 3-pixel wide outer ring, and subtract it. Defaults to True
	NOTE: The model has no background level. So if doSkySub is set to False, make sure the provided image is properly sky-subtracted.

	"""
	
	(ny,nx)=im.shape
	YY, XX = np.mgrid[0:ny, 0:nx]

	# sky subtraction
	if doSkySub:
		mask=np.zeros_like(im,dtype='bool')
		mask[3:ny-3,3:nx-3]=True
		sky_mean,sky_med,sky_std = sigma_clipped_stats(im,mask=mask)
		im -= sky_mean
	
	if estimates is None:
		x0,y0=centroid(im)
		M = im.max()
		
		# première estimation de la FWHM (sur profil intégré azimuthalement)
		Dist = np.sqrt((XX-x0)**2+(YY-y0)**2)

		# affichage du profile sur tous les azimuth
		Dist_vect = Dist.flatten()
		im_vect = im.flatten()
		Dist_vect = Dist[Dist < 10.0]
		im_vect = im[Dist < 10.0]
			
		inds = Dist_vect.argsort()
		# profile artificiellement rendu symmetrique autour du 0, pour le calcul du sigma
		#Dist_vect = np.hstack((-Dist_vect[inds][-1:0:-1],Dist_vect[inds]))
		#im_vect =  np.hstack((im_vect[inds][-1:0:-1],im_vect[inds]))
		
		Dist_vect = np.hstack((-Dist_vect[inds][999:0:-1], Dist_vect[inds][0:1000]))
		im_vect =  np.hstack((im_vect[inds][999:0:-1],im_vect[inds][0:1000]))
		
		dist_mean = np.sum(Dist_vect*im_vect)/np.sum(im_vect) # moyenne pondérée du vecteur Dist_vect
		sigma_estim =  np.sqrt(np.sum((Dist_vect - dist_mean)**2*im_vect)/np.sum(im_vect)) # écart type en X

		estimates={'amplitude':M,'x_mean':x0,'y_mean':y0,'x_stddev':sigma_estim,'y_stddev':sigma_estim}

	
	Gauss = models.Gaussian2D(theta=0, **estimates)
	Trap = models.TrapezoidDisk2D(amplitude=0, x_0=estimates['x_mean'], y_0=estimates['y_mean'], R_0=estimates['x_stddev'], slope=50.)
	Moff = models.Moffat2D(amplitude = 0, x_0=estimates['x_mean'], y_0=estimates['y_mean'], gamma=estimates['x_stddev'], alpha=2.0)

	# contraindre le trapèze sur le centre de la gaussienne + limites inférieures à 0
	Gauss.x_stddev.min=0.001
	Gauss.y_stddev.min=0.001
	Gauss.theta.fixed=True
	Trap.slope.min=0
	Trap.R_0.min=0
	Trap.amplitude.min=0

	if gaussian:
		ZeModel = Gauss
		if trapezoidal:
			ZeModel += Trap
		if moffat:
			ZeModel += Moff
	elif trapezoidal:
		ZeModel = Trap
		if moffat:
			ZeModel += Moff
	else:
		ZeModel = Moff

	fitter=fitting.LevMarLSQFitter()
	
	# Weighting function allows to discard potential cosmic ray falling near the PSF,
	# or pollution by a second spectral line very nearby. Arbitrary set to gaussian of 4*sigma 
	W = np.exp( -0.5*((XX-estimates['x_mean'])**2+(YY-estimates['y_mean'])**2) / (estimates['x_stddev'] * 4.0)**2 )

	#pdb.set_trace()
	fit = fitter(ZeModel, XX, YY, im, weights=W,maxiter=maxiter,acc=acc)
	#fit = fitter(ZeModel, XX, YY, im, weights=abs(im),maxiter=maxiter,acc=acc)
	
	if fitter.fit_info['ierr']==5:
		print(fitter.fit_info['message'])

	chi2 = np.sqrt(np.sum((im-fit(XX,YY))**2))
	
	return fit, chi2

def FWHM(x,y):
	"""
	Return full width at half-maximum of the 1D curve y, of coordinate x.
	Expect a curve with a unique maximum.
	x and y are numpy vectors of the same length
	"""
	
	# linear interpolation between two points surrounding the half-max, on both sides
	m = y.max()
	i = y.argmax()

	try:
		x2 = np.interp(m/2.0,y[::-1][:len(x)-i-1],x[::-1][:len(x)-i-1])  # y array must be in ascending order
		x1 = np.interp(m/2.0,y[:i-1],x[:i-1])
		return x2 - x1
	except:
		return -999

def PSF_Params(im, sampling_factor=50.0, display=False, estimates=None, doSkySub=True):
	"""
	a wrapper function returning parameter of a PSF (amplitude, width, centroid)
	from a fit (gaussian works best).
	
	sampling_factor: multiply the sampling of the original image by this sampling_factor
					to compute the 1D model (model from which FWHM is derived)
	
	estimates is None by default, or a dictionnary giving estimates of the best fitting gaussian:
		estimates={'amplitude':a0,'x_mean':x0,'y_mean':y0,'x_stddev':sigmaX0,'y_stddev':sigmaY0}
	"""
	
	(ny,nx)=im.shape
	fit, chi2 = PSF_fit(im, gaussian=True, trapezoidal=False, moffat=False, maxiter=2000,acc=1e-6, estimates=estimates, doSkySub=doSkySub)
	#print(fit)
	try:
		xmean = fit.x_mean.value
		ymean = fit.y_mean.value
	except AttributeError:
		try:
			xmean = fit.x_mean_0.value
			ymean = fit.y_mean_0.value
		except AttributeError:
			try:
				xmean = fit.x_0.value
				ymean = fit.y_0.value
			except AttributeError:
				try:
					xmean = fit.x_0_0.value
					ymean = fit.y_0_0.value
				except AttributeError:
					pass
				
	if display:
		fig=plt.figure(1)
		ax1=fig.add_subplot(1,2,1)
		ax2=fig.add_subplot(1,2,2)
		ax1.clear()
		ax2.clear()

	x = np.linspace(0,nx-1,(nx-1)*sampling_factor+1)
	y = np.linspace(ymean,ymean,(nx-1)*sampling_factor+1)
	profX=fit(x,y)
	FWHMx = FWHM(x,profX)
	
	if display:
		ax1.plot(x,profX)
		ax1.plot(im[int(np.around(ymean)),:],'--o')

	x = np.linspace(xmean,xmean,(ny-1)*sampling_factor+1)
	y = np.linspace(0,ny-1,(ny-1)*sampling_factor+1)
	profY=fit(x,y)
	FWHMy = FWHM(y,profY)

	if display:
		ax2.plot(y,profY)
		ax2.plot(im[:,int(np.around(xmean))],'--o')

	return (profX.max(), xmean, ymean, FWHMx, FWHMy,chi2)

#def EE(im, Rad=None, GaussFitParam=None, doSkySub=True):
#	"""
#	Compute Encircled energy on a PSF, in a circle of radius Rad (in pixels).
#	
#	If Rad is None, defaults to 3 sigma
#	
#	INPUTS:
#	------
#	im: the image containing the PSF. 
#	
#	Rad: the radius on which to compute the encircled energy. If None, defaults to 3 sigma
#	
#	GaussFitParam: dictionnary giving estimates of the best fitting gaussian, with at least the following keys:
#				{'x_mean':x0,'y_mean':y0,'x_stddev':sigmaX0,'y_stddev':sigmaY0}
#				If None (defaults), the best fitting gaussian is computed from PSF_Params()
#	
#	doSkySub: if set, computes the sky level on a 3-pixel wide outer ring, and subtract it. Defaults to True
#	NOTE: phot.aperture_photometry() expects a sky-subtracted image, so if doSkySub is set to False,
#		  make sure the provided image is properly sky-subtracted.
#
#	OUTPUTS:
#	-------
#		The encircled energy, in fraction of the total energy.
#		
#		"""
#	if GaussFitParam is None:
#		(a0, x0, y0, FWHMx, FWHMy,chi2) = PSF_Params(im)
#	else:
#		x0 = GaussFitParam['x_mean']
#		y0 = GaussFitParam['y_mean']
#		FWHMx = GaussFitParam['x_stddev']*2.35
#		FWHMy = GaussFitParam['y_stddev']*2.35
#
#	if doSkySub:
#		(ny,nx)=im.shape
#		mask=np.zeros_like(im,dtype='bool')
#		mask[3:ny-3,3:nx-3]=True
#		sky_mean,sky_med,sky_std = sigma_clipped_stats(im,mask=mask)
#		im -= sky_mean
#		
#	if Rad is None:
#		Rad = 3.0*(FWHMx+FWHMy)*0.424661/2.0
#
#	aper = phot.CircularAperture((x0,y0),Rad)
#
#	table = phot.aperture_photometry(im,aper)
#	EE = table['aperture_sum'].data[0]
#	
#	return EE
#
