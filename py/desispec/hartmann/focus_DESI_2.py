#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

# usual stuff
from sys import version_info
import numpy as np
import photutils as phot
import matplotlib.pylab as plt
import pdb
if version_info.major==2:
	import cPickle as pk
else:
	import pickle as pk
import sys
import argparse
import os
from mpl_toolkits.mplot3d import Axes3D
from sys import stdout

# astropy libraries
from astropy.stats import mad_std, sigma_clipped_stats
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.table import Table, Column
import astropy.visualization as visu
from astropy.visualization.mpl_normalize import ImageNormalize

from scipy.interpolate import RectBivariateSpline

# home-made stuff
#from rebin import rebin
 #from centroid import centroid
import filesTools as ft
from copy import deepcopy
import DESI_tools as desi
import PSFstuff as psf

# default parameters for matplotlib
plt.rcParams['image.aspect']='auto'
plt.rcParams['image.interpolation']='none'
plt.rcParams['image.cmap']='gist_heat'
plt.rcParams['savefig.dpi']=300

def tiedSlopeFunc(model):
	"""
	Function used to tie the Slope of the trapezoidal model (either 1D or 2D)
	to the amplitude, so they are of the same sign
	S. Ronayette, 2016, April 13th
	"""
	s = np.copysign(model.slope, model.amplitude)
	return s

def GetAllSourcesMetrics(SrcList, ee=0.90, display=False):
	"""
		Get metrics (FWHM, Ree) of all sources		
		
		Inputs:
		-------
		SrcList: a list of dictionnaries containing the sources, as returned by getSources()
		ee: the fraction of encircled energy for which the radius is computed
			
		Outputs:
		--------
		The reduced data in an astropy Table. Contents depend on the fit type.
	"""
	
	if SrcList is None:
	   print("Sources variable must be provided. Consider using getSources()")
	   return 0
	
	if display:
		fig = plt.figure('Data and fit profiles', figsize=(14, 11))
	
	# PARAMETERS
	# ----------
	pix_sz = 0.015  # pixel size in mm
	im0 = SrcList[0]['image']
	n = im0.shape[0]  # size of subimages
	x = np.linspace(0, n - 1, n)  # abcissa values for plotting x profile
	y = x  # abcissa values for plotting y profile

	table0 = Table(names=('defocus','xcentroid','ycentroid','Ree','FWHMx','FWHMy','Amp'),dtype=['float32']*7)
	
	SrcData = []
	for i, s in enumerate(SrcList):
		print("Processing source {:d}/{:d}".format(i + 1, len(SrcList)),end="\r")
		print(s)
		stdout.flush()
		defoc = s['defocus']
		table = table0[:]
		xcent0 = s['xcent']
		ycent0 = s['ycent']
		
		for k, dz in enumerate(defoc):
			subim = s['image'][:, :, k]
			FWHM_estim = s['fwhm_mean'][k]
			
			if not np.isnan(FWHM_estim):
				
				(A, xcentroid, ycentroid, FWHMx, FWHMy,chi2) = psf.PSF_Params(subim, sampling_factor=20.0, display=False, \
						estimates={'amplitude':subim.max(),'x_mean':xcent0,'y_mean':ycent0,'x_stddev':FWHM_estim/2.35,'y_stddev':FWHM_estim/2.35}, \
						doSkySub=False)
					
				GFitParam = {'amplitude':A, \
							 'x_mean':xcentroid, \
							 'y_mean':ycentroid, \
							 'x_stddev':FWHMx/2.0/np.sqrt(2.0*np.log(2)), \
							 'y_stddev':FWHMy/2.0/np.sqrt(2.0*np.log(2))}
				
				radii = np.linspace(0.1,n/2-2,50)
				EEvect = np.array([psf.EE(subim, r, GFitParam, doSkySub=False) for r in radii])
				maxEE = np.mean(EEvect[-5:])
				Ree = np.interp(ee*maxEE, EEvect, radii)
				table.add_row([dz, xcentroid, ycentroid, Ree, FWHMx, FWHMy, A])

				#DISPLAY THE FITTED PROFILES AND ENCIRCLED ENERGY
				#------------------------------------------------
				if display:
					fig.clear()
					fit = models.Gaussian2D(**GFitParam)
					# higher resolution grid for plotting
					col = int(round(xcentroid))
					row = int(round(ycentroid))
					HRfactor = 8.0
					XX_highR, YY_highR = np.meshgrid(np.linspace(0, n - 1, (n - 1) * HRfactor + 1),
													np.linspace(0, n - 1, (n - 1) * HRfactor + 1))
					x_highR = np.linspace(0,n-1,(n - 1) * HRfactor + 1)
					y_highR = np.linspace(0,n-1,(n - 1) * HRfactor + 1)

					z = fit(XX_highR, YY_highR)  # the fit at higher resolution, for plotting
					col_highR = int(round(xcentroid * HRfactor))
					row_highR = int(round(ycentroid * HRfactor))

					ax1 = fig.add_axes([0.05,0.5,0.4,0.42], title = 'X profile', xlabel='pixels',ylabel='ADU')
					ax2 = fig.add_axes([0.55,0.5,0.4,0.42], title = 'Y profile', xlabel='pixels')
					ax3 = fig.add_axes([0.41,0.62,0.18,0.18], xticks=[], yticks=[], aspect='equal')
					ax4 = fig.add_axes([0.15,0.05,0.7,0.37], title='encircled energy', xlabel='pixels',ylabel='fraction')
					fig.suptitle('Defocus {:.3f}, source {:d}\nAnalysis type: Gaussian fit'.format(dz, i + 1))

					ax4.plot(radii, EEvect)
					ax4.plot([0,Ree],[ee,ee], '--', color='black')
					ax4.plot([Ree,Ree],[0,ee], '--', color='black')
					ax4.annotate('R{:02d}: {:.1f} pixels'.format(int(ee*100),Ree), (0.5,0.05), xycoords="axes fraction")
					
					profile1 = subim[row, :]
					profile2 = z[row_highR, :]

					ax1.plot(x, profile1, '--', linewidth=2, label='data')
					ax1.plot(XX_highR[0, :], profile2, linewidth=2, label='fit')
					m = A
					xstart = xcentroid - FWHMx / 2.0  # exact abscissa position of the left point at half max (analytical function)

					ax1.plot([xstart, xstart + FWHMx], [m / 2, m / 2], '-x', color='black', \
							label='FWHM')  # plot line at FWHM
					ax1.annotate('FWHMx: {:.1f} pixels\nCHI2: {:.1f} ADU'.format(FWHMx, np.nan), \
								(2, subim.max()), va='top', color='green')
					ax1.set_ylim(-50, 1.05*subim.max())
					ax1.set_xlim(0, n)
					ax1.set_title('X profile')
					ax1.set_xlabel('pixels')
					ax1.set_ylabel('ADU')

					profile1 = subim[:, col]
					profile2 = z[:, col_highR]

					ax2.plot(y, profile1, '--', linewidth=2)
					ax2.plot(YY_highR[:, 0], profile2, linewidth=2, label='fit')
					m = A
					xstart = ycentroid - FWHMy / 2.0  # exact abscissa position of the left point at half max (analytical function)
					ax2.plot([xstart, xstart + FWHMy], [m / 2, m / 2], '-x', color='black')
					ax2.annotate('FWHMy: {:.1f} pixels\nCHI2: {:.1f} ADU'.format(FWHMy, np.nan), \
								(1, subim.max()), va='top', color='green')
					ax2.set_ylim(-50, 1.05*subim.max())
					ax2.set_xlim(0, n)
					ax2.set_title('Y profile')
					ax2.set_xlabel('pixels')

					ax3.imshow(np.arcsinh(subim - subim.min() + 0.1), aspect='equal')
					ax3.annotate('+', (xcentroid, ycentroid), color='blue', ha='center', va='center', fontsize='xx-large',
								fontweight='light')

					ax1.legend(fancybox=True, shadow=True, ncol=1, fontsize=14.0, loc='best')
					plt.pause(0.00001)  # necessary to update the figure. There is a delay already...


		if len(table) !=0:
			# sinon, c'est qu'il n'y avait que des nan dan fwhm_mean, ce qui correspond à 
			# une sources entièrement mise de côté (pour toutes les positions de focus)
			currentSourceRes = {'table':table, 'source':s['source'], 'fiber_num':s['fiber_num'], 'wave_num':s['wave_num'], 'x0':s['x0'],'y0':s['y0']}
			
			SrcData.append(currentSourceRes)
	print()
	return SrcData

def analyse(SrcData):
	"""
	Analyse data from the focus test.
	
	Inputs:
	-------
	SrcData: An astropy Table containing the reduced data of all sources as returned by GetAllSourcesMetrics()
		
	Outputs:
	--------
	The reduced data in a list of dictionaries (one list element per source), with the following keys:
		bestX: focus location where FWHMx is minimum
		bestY: focus location where FWHMy is minimum
		best: the best overall focus for the given source (average of bestX and bestY)
		polyX: the coefficients of the 4th degres polynomial fitting the FWHMx VS defocus data
		polyY: the coefficients of the 4th degres polynomial fitting the FWHMy VS defocus data
		source: the source numberç
		table: a an astropy Table, with the source fit results for each focus position. Contents depend on the fit type.
		
	Print the overall best focus, according to various criteria.
	"""
	k=0
	for s in SrcData:
		table=s['table']
		defoc = table['defocus']
		
		valid = np.logical_not(np.isnan(table['FWHMx']))
		if len(valid[valid == True]) > 0:
			(s['FWHMx_poly'], s['FWHMx_best']) = getMinimum2(defoc[valid],table['FWHMx'][valid],4)
			if s['FWHMx_poly'][0] is np.nan:
			# rerun analysis with polynomial of degree 2
				(s['FWHMx_poly'], s['FWHMx_best']) = getMinimum2(defoc[valid],table['FWHMx'][valid],2)
		else:
			s['FWHMx_best'] = np.nan
			s['FWHMx_poly'] = [np.nan]*3

		valid = np.logical_not(np.isnan(table['FWHMy']))
		if len(valid[valid == True]) > 0:
			(s['FWHMy_poly'], s['FWHMy_best'])= getMinimum2(defoc[valid],table['FWHMy'][valid],4)
			if s['FWHMy_poly'][0] is np.nan:
			# rerun analysis with polynomial of degree 2
				(s['FWHMy_poly'], s['FWHMy_best']) = getMinimum2(defoc[valid],table['FWHMy'][valid],2)
		else:
			s['FWHMy_best'] = np.nan
			s['FWHMy_poly'] = [np.nan]*3
		
		s['FWHM_best'] = (s['FWHMx_best'] + s['FWHMy_best']) / 2.0

		valid = np.logical_not(np.isnan(table['Ree']))
		if len(valid[valid == True]) > 0:
			(s['Ree_poly'], s['Ree_best']) = getMinimum2(defoc,s['table']['Ree'],4)
			if s['Ree_poly'][0] is np.nan:
				# rerun analysis with polynomial of degree 2
				(s['Ree_poly'], s['Ree_best']) = getMinimum2(defoc,s['table']['Ree'],2)
		else:
			s['Ree_best'] = np.nan
			s['Ree_poly'] = [np.nan]*3
	 
		k+=1
		#s['bestX']=bestX
		#s['bestY']=bestY
		#s['best']=(bestX+bestY)/2.0
		#s['polyX'] = px
		#s['polyY'] = py

	# -----------------------------------
	#  Work on the FWHMs polynomial fits 
	# -----------------------------------
	
	# for each defocus value, the FWHMs in x and y are computed for all sources in field
	# and the maximum value in field is retained.
	# the defocus value retained is the one where this max value is minimum
	alldefoc=[d for s in SrcData for d in list(s['table']['defocus'])]
	defoc_highR = np.linspace(min(alldefoc),max(alldefoc),500)
	import pdb;pdb.set_trace()
	maxes = np.asarray([np.max([s['FWHMx_poly'](z) for s in SrcData if np.isnan(s['FWHMx_poly'][0])==False] + \
		[s['FWHMy_poly'](z) for s in SrcData if np.isnan(s['FWHMx_poly'][0])==False]) \
			for z in defoc_highR])
	
	meanFWHM = np.asarray([(s['FWHMx_poly'](defoc_highR)+s['FWHMy_poly'](defoc_highR))/2 for s in SrcData if np.isnan(s['FWHMx_poly'][0])==False]) # 2D array with average of X and Y for all sources
	meanFWHM = np.mean(meanFWHM, axis=0)

	bests = [x['FWHM_best'] for x in SrcData]
	FWHM_best_average_z = np.mean(bests)
	
	FWHM_best_average_FWHM = defoc_highR[np.where(meanFWHM == meanFWHM.min())][0]
	FWHM_best_mindesmax = defoc_highR[np.where(maxes == maxes.min())][0]
	
#	print("\nbest average Z: {:.3f} mm\nbest average FWHM: {:.3f} mm\nbest min of maxes: {:.3f} mm".format(best_average_z,best_average_FWHM,best_mindesmax))
	
	
	# fit a plane to the data
	# ----------------------
	
	# The X's and Y's coordinates and Best focus for each source:
	#ndefoc=len(defoc)
	Xs = [np.median(s['table']['xcentroid']+s['x0']) for s in SrcData if np.isnan(s['FWHM_best'])==False]
	Ys = [np.median(s['table']['ycentroid']+s['y0']) for s in SrcData if np.isnan(s['FWHM_best'])==False]
	Bs = [s['FWHM_best'] for s in SrcData if np.isnan(s['FWHM_best'])==False]
	model = models.Polynomial2D(1) # a 2D polynomial of degree 1 = a plane
	Fitter=fitting.LinearLSQFitter()
	FWHMPlanefit= Fitter(model, Xs, Ys, Bs)
	
	# -----------------------------------
	#  Work on the Ree polynomial fits 
	# -----------------------------------
	# for each defocus value, the Ree is computed for all sources in field
	# and the maximum value in field is retained.
	# the defocus value retained is the one where this max value is minimum
	maxes = np.asarray([np.max([s['Ree_poly'](z) for s in SrcData if np.isnan(s['Ree_poly'][0])==False]) for z in defoc_highR])
	
	meanRee = np.asarray([s['Ree_poly'](defoc_highR) for s in SrcData if np.isnan(s['Ree_poly'][0])==False]) # 2D array with average of X and Y for all sources
	meanRee = np.mean(meanRee, axis=0)

	bests = [x['Ree_best'] for x in SrcData]
	Ree_best_average_z = np.mean(bests)
	Ree_best_average_Ree = defoc_highR[np.where(meanRee == meanRee.min())][0]
	Ree_best_mindesmax = defoc_highR[np.where(maxes == maxes.min())][0]
	#print("max Ree at best focus: {}".format(maxes.min()))
	
#	print("\nbest average Z: {:.3f} mm\nbest average FWHM: {:.3f} mm\nbest min of maxes: {:.3f} mm".format(best_average_z,best_average_FWHM,best_mindesmax))
	
	
	# fit a plane to the data
	# ----------------------
	
	# The X's and Y's coordinates and Best focus for each source:
	#Xs = [x['table']['xcentroid'][ndefoc/2] for x in SrcData]
	#Ys = [x['table']['ycentroid'][ndefoc/2] for x in SrcData]
	Xs = [np.median(s['table']['xcentroid']+s['x0']) for s in SrcData if np.isnan(s['Ree_best'])==False]
	Ys = [np.median(s['table']['ycentroid']+s['y0']) for s in SrcData if np.isnan(s['Ree_best'])==False]
	Bs = [s['Ree_best'] for s in SrcData if np.isnan(s['Ree_best'])==False]
	model = models.Polynomial2D(1) # a 2D polynomial of degree 1 = a plane
	Fitter=fitting.LinearLSQFitter()
	ReePlanefit= Fitter(model, Xs, Ys, Bs)
	
	OverallResult = {'FWHM_best_average_z':FWHM_best_average_z, 'FWHM_best_average_FWHM':FWHM_best_average_FWHM,
					 'FWHM_best_mindesmax':FWHM_best_mindesmax, 'FWHMPlanefit':FWHMPlanefit,
					 'Ree_best_average_z':Ree_best_average_z, 'Ree_best_average_Ree':Ree_best_average_Ree,
					 'Ree_best_mindesmax':Ree_best_mindesmax, 'ReePlanefit':ReePlanefit}
	
	return SrcData, OverallResult

def getSources(files, autoSelect=True, files2=None, channel='R1'):
	"""
	Return a list in which each element is a dictionnary containing the subimages centered on the sources for all defocus
	(hence, a cube image), the coordinates of subimage[0,0] in the original image,
	and the centroid of the source on each subimage
	if files2 is provided, the sources at the same location are extracted in the "files2" images
	(useful when processing hartmann doors test. Typically, files is the images with one door closed, and files2 is
	the image with the other door closed)
	"""

	if type(files) is not list: files=[files]
	
	# sort files by defocus
	#try:
		#dzArr = [fits.open(f)[channel].header['Z'] for f in files]
	#except KeyError:
		#if len(files)==1:
			#dzArr = np.array([0])  # dz = 0, just on file here, we are not doing a focus test
		#else:
			#dzArr  = np.linspace(-1.0,1.0,len(files)) # a fake defocus array

	color=channel[-2] # = the letter R, B or Z, corresponding to the channel
	dzArr=[]
	for f in files:
		try:
			PLCHead = fits.open(f)['PLC'].header
		except Exception:
			PLCHead = fits.open(f)['IMAGE'].header # données traitées avec le pipeline de J. Guy. Les données PLC sont dans l'extension "IMAGE"
		top = PLCHead['GAUGE'+color+'T']
		left = PLCHead['GAUGE'+color+'L']
		right = PLCHead['GAUGE'+color+'R']
		dzArr.append(np.mean([top,left,right]))
		
	L = zip(files, dzArr)
	#L.sort(key=lambda x: x[1])  # sort according to dz. python2 method only.
	L=sorted(L, key=lambda x: x[1]) # method for python3. Work in python 2.7.12 as well
	(files, dzArr) = zip(*L)  # unzip

	if files2 is not None:
		dzArr=[]
		for f in files2:
			try:
				PLCHead = fits.open(f)['PLC'].header
			except Exception:
				PLCHead = fits.open(f)['IMAGE'].header # données traitées avec le pipeline de J. Guy. Les données PLC sont dans l'extension "IMAGE"
			top = PLCHead['GAUGE'+color+'T']
			left = PLCHead['GAUGE'+color+'L']
			right = PLCHead['GAUGE'+color+'R']
			dzArr.append(np.mean([top,left,right]))
			
		L = zip(files2, dzArr)
		#L.sort(key=lambda x: x[1])  # sort according to dz. python2 method only.
		L=sorted(L, key=lambda x: x[1]) # method for python3. Work in python 2.7.12 as well
		(files2, dzArr) = zip(*L)  # unzip
	
	
	# PARAMETERS
	# ----------
	pix_sz = 0.015  # pixel size in mm
	n = 100  # size of subimage for analysis

	# Quick estimate for the FWHM as a function on defocus
	# quadratic sum of image of fiber size (3.33 pix) and defocus size (dz/2/F)
	# FWHM_estim = [np.sqrt((z/2.0/1.7/pix_sz)**2+3.333**2) for z in dz]
	# direct sum of fiber + defocus
	# FWHM_estim = [abs(z) / 2.0 / 1.7 / pix_sz + 3.333 for z in dz]
	FWHM_estim = lambda z: abs(z) / 2.0 / 1.7 / pix_sz + 3.

	# look at position of the sources on the central image
	# These positions are kept for extracting all the sources on all the other images

	f = files[int(len(files) / 2)]
	HDUs = fits.open(f)
	names = [h.name for h in HDUs]
	if channel in names:
		image_ext=channel # pour les données processées "maison" (avec DESI_tools.DESIImage)
	if 'IMAGE' in names:
		image_ext='IMAGE' # pour les données processées avec le pipeline de J. Guy

	hdu=HDUs[image_ext]
	#hdu = desi.DESIImage(f, silent=True)[channel]
	dz0 = dzArr[int(len(files) / 2)]
	im = np.float64(hdu.data)

	# work on rebinned image
	# may avoid some double source due to local max in very defocused image + quicker computation
	sz = im.shape
	#scale_fact = 2
	#zim = rebin(im, sz[0]/scale_fact, sz[1]/scale_fact)
	bkg_sigma = mad_std(im)

	if autoSelect:
		# fetch sources automatically, one by one in small subimages centered on
		# reference positions.
		refdatadir='./' #'/desi/sronayet/PERFORMANCETESTS/CODE/FOCUS/'
		#RefSources=pk.load(open(refdatadir+'sourcesEM1_{}_HgAr_Kr_Ne_Cd.dat'.format(channel),'rb'),encoding='latin1')
		RefSources=Table.read(refdatadir+'sourcesSM03_{}_HgAr_Kr_Ne_Cd.dat'.format(channel),format='ascii')
		k=1
		indind=1
		for s in RefSources:
			print('Identifying Source ',indind)
			indind+=1
			x0=s['xcentroid']
			y0=s['ycentroid']
			xmin = int(max(x0 - n / 2, 0.0))
			xmax = xmin + n
			if xmax > sz[1]:
				xmax = sz[1]
				xmin = xmax - n
			ymin = int(max(y0 - n / 2, 0.0))
			ymax = ymin + n
			if ymax > sz[0]:
				ymax = sz[0]
				ymin = ymax - n
			subim = im[ymin:ymax, xmin:xmax]
			mask=np.zeros_like(subim,dtype='bool')
			mask[3:n-3,3:n-3]=True
			sky_level,sky_med,sky_std = sigma_clipped_stats(subim,mask=mask)
			subim -= sky_level

			#OneSource = phot.daofind(subim, fwhm=FWHM_estim(0), threshold=8.*sky_std, \
			#	sharplo=0.1, sharphi=0.85, roundlo=-0.6, roundhi=0.6, exclude_border=True)
			daofind = phot.DAOStarFinder(8.0*sky_std,FWHM_estim(0), sharplo=0.1, sharphi=0.85, roundlo=-0.6, roundhi=0.6, exclude_border=True)
			OneSource = daofind(subim)
			if len(OneSource) >= 1:
				# if more than one source in the subimage, we keep only the
				# one closest to the expected position
				#from pylab import imread,subplot,imshow,show
				#import matplotlib.pyplot as plt
				#plt.imshow(subim)
				#plt.show()
				print('Find one')
				Dist=[(ss['xcentroid']-(x0-xmin))**2+(ss['ycentroid']-(y0-ymin))**2 for ss in OneSource]
				OneSource=OneSource[Dist.index(min(Dist)):Dist.index(min(Dist))+1]
				OneSource['id']=k
				OneSource['xcentroid']+=xmin
				OneSource['ycentroid']+=ymin
				if k==1:
					sources=OneSource
				else:
					sources.add_row(OneSource[0])
				k+=1
			else:
				#Case "no source found"
				pass
	else:
		# Manual select of sources, by clicking on the image
		daofind = phot.DAOStarFinder(200.*bkg_sigma,FWHM_estim(0),sharplo=0.1, sharphi=0.85, roundlo=-0.5, roundhi=0.5, exclude_border=True)
		sources = daofind(im)
		#sources = phot.daofind(im, fwhm=FWHM_estim(0), threshold=200.*bkg_sigma, \
		#	sharplo=0.1, sharphi=0.85, roundlo=-0.5, roundhi=0.5, exclude_border=True)
		import matplotlib.pyplot as plt
		plt.ioff()
		sourceSel = SourceSelector(im, sources, channel)
		sources = sourceSel.sources
                #print(len(sources[0]))
                #print(sources.colnames)
		plt.ion()



                ##Permet de creer un tableau contenant les 450 sources vertes de SM02 (fichier sourceSelcroixvert.dat)
                #data_row = [0,0,0,0,0,0,0,0,0,0,0]
                #final = Table(rows=[data_row],names=('id','xcentroid','ycentroid', 'sharpness', 'roundness1', 'roundness2', 'npix', 'sky', 'peak', 'flux', 'mag'),dtype=('f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4'))


                #for i in range(0,len(sources)):

                        #final.add_row( [i, sources[i][1] , sources[i][2], sources[i][3] , sources[i][4], sources[i][5] , sources[i][6], sources[i][7] , sources[i][8], sources[i][9] , sources[i][10]  ] )

                #final.remove_row(0)
                #final.write('desi/daoud/results/sourceSelcroixvert.dat',format='ascii')        #SM01verttableau.dat                  

	
	print("Found {} sources".format(len(sources)))
	if files2 is not None:
		# for Hartmann test only
		allfiles=[files,files2]
	else:
		allfiles=[files]
	
	# Now, fetch sources in all images (for all focus positions)
	SrcList = []
	for fichiers in allfiles:
		# allfiles is a list of length=2 if analysing Hartmann Test, with:
		#	allfiles[0] = files for door 1 for all defocus positions
		#	allfiles[1] = files for door 2 for all defocus positions
		# otherwise, allfiles is a list of length=1, with allfiles[0] = files for all defocus positions
		AllSources = []
		YY, XX = np.mgrid[:n, :n]
		DiscardFlag=False
		noerrorlist=[1,2,3,4] #https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html#scipy.optimize.leastsq
		for i, s in enumerate(sources):
			print("Retrieving source {:2d} / {:2d}\r".format(i + 1, len(sources)),end="")
			stdout.flush()
			cube = np.zeros((n, n, len(fichiers)), dtype='float32')
			xcent = np.zeros(len(fichiers), dtype='float32')
			ycent = np.zeros(len(fichiers), dtype='float32')
			defocus = np.zeros(len(fichiers), dtype='float32')
			fwhm_mean = np.zeros(len(fichiers), dtype='float32')
			
			x0 = s['xcentroid']
			y0 = s['ycentroid']

			xmin = int(max(x0 - n / 2, 0.0))
			xmax = xmin + n
			if xmax > sz[1]:
				xmax = sz[1]
				xmin = xmax - n
			ymin = int(max(y0 - n / 2, 0.0))
			ymax = ymin + n
			if ymax > sz[0]:
				ymax = sz[0]
				ymin = ymax - n

			# centroid of the source in the middle image. Must be recomputed for other defocus images
			xcent0 = x0 - xmin
			ycent0 = y0 - ymin
			
			for k, f in enumerate(fichiers):
				hdu = fits.open(f)[image_ext]
				#hdu = desi.DESIImage(f, silent=True)[channel]
				im = hdu.data
				dz = dzArr[k]-dz0
				subim = np.float32(im[ymin:ymax, xmin:xmax])

				if subim.max() < 25000:
					# (newY0,newX0)=cntrd(subim,yestim=ycent0,xestim=xcent0,box=15)
					
					mask=np.zeros_like(subim,dtype='bool')
					mask[3:n-3,3:n-3]=True
					sky_level,sky_med,sky_std = sigma_clipped_stats(subim,mask=mask)
					subim -= sky_level
					
					model = models.Gaussian2D(amplitude=subim.max(), x_mean=xcent0, y_mean=ycent0,
											x_stddev=FWHM_estim(dz) * 0.42, y_stddev=FWHM_estim(dz)* 0.42, theta=0.0)
					Fitter = fitting.LevMarLSQFitter()
					fit = Fitter(model, XX, YY, subim)
					err = Fitter.fit_info['ierr']
					
					if fit.x_stddev+fit.y_stddev <= 0:
						# seen sometimes: the fit appears to be successful, but fwhm negative...
						# force it to "unsuccessful".
						err=5
					
					cube[:, :, k] = subim
					defocus[k] = dz
				else:
					print("Source {} for focus {} is near or at saturation. Discarding".format(i+1,dz))
					err = 5
					
				try:
					noerrorlist.index(err)
					xcent[k] = fit.x_mean.value
					ycent[k] = fit.y_mean.value
					fwhm_mean[k] = (fit.x_stddev+fit.y_stddev)*np.sqrt(2*np.log(2))
				except ValueError:
					print("Unsuccessful Fit. Discarding Source {} for focus {}".format(i+1,dz))
					xcent[k] = np.nan
					ycent[k] = np.nan
					fwhm_mean[k] = np.nan

			xmed = np.median(xcent[np.logical_not(np.isnan(xcent))])
			ymed = np.median(ycent[np.logical_not(np.isnan(ycent))])
			Dict = {'image': cube, 'x0': xmin, 'y0': ymin, 'xcent': xmed, 'ycent': ymed, \
					'defocus': defocus, 'fwhm_mean':fwhm_mean, 'source':i+1}
			
			AllSources.append(Dict)

		print()
		SrcList.append(deepcopy(renumber(AllSources,channel)))
	
	if len(SrcList)==1:
		# only one list of files (with both door open. Not doing Hartmann test)
		SrcList=SrcList[0]
		
	return SrcList

def getMinimum(x,y):
	"""
	Determine the minimum of a curve y=f(x) by fitting a polynomial of degree 4.
	
	Inputs:
	------
		x : the independant variables
		y : the data, as a function on x. Same length as x
		
	Outputs:
	--------
	
		Returns the fitting polynomial, and the abscissa position where the polynomial is minimum
	"""
	if len(x)>4:
		# 4th degrees polynomial to fit the data (or less if not enough data points)
		coeffs = np.polyfit(x,y,4)
		p = np.poly1d(coeffs)
		
		# roots of the derivative = extremums of the fitting polynomial
		roots = np.roots(coeffs[0:-1]*[4,3,2,1])
		## A PRIORI, there are three roots, and the one of interest is the middle one
		## or the only real one
		#if len(roots) > 3:
			#print "WARNING:found {:d} roots in X. Manual check required".format(len(roots))
		#if all(roots.imag == 0):
			#r = np.median(roots)
		#else:
			#r = roots[np.where(roots.imag == 0)][0].real
		
		# A PRIORI, there are three roots, and the one of interest is the one
		# inside the range (x.min, xmax), and real one. If there are several, keep one
		# at which the polynomial is minimum. If there are no root (len(r)=0), pass.
		roots = roots[np.where(roots.imag == 0)].real
		r = roots[(roots < x.max()) & (roots > x.min())]
		if len(r)==0:
			p=[np.nan]*3
			r=np.nan
		else:
			r = r[p(r) == min(p(r))][0]

	else:
		p = np.poly1d(np.nan)
		r = x[0]

	return (p,r)

def getMinimum2(x,y,deg):
	"""
	Determine the minimum of a curve y=f(x) by fitting a polynomial of degree "deg".
	
	Inputs:
	------
		x : the independant variables
		y : the data, as a function on x. Same length as x
		
	Outputs:
	--------
	
		Returns the fitting polynomial, and the abscissa position where the polynomial is minimum
	"""
	if len(x)>deg:
		# 4th degrees polynomial to fit the data (or less if not enough data points)
		coeffs = np.polyfit(x,y,deg)
		p = np.poly1d(coeffs)
		
		# roots of the derivative = extremums of the fitting polynomial
		roots = np.roots(coeffs[0:-1]*[deg-i for i in range(deg)])
		## A PRIORI, there are three roots, and the one of interest is the middle one
		## or the only real one
		#if len(roots) > 3:
			#print "WARNING:found {:d} roots in X. Manual check required".format(len(roots))
		#if all(roots.imag == 0):
			#r = np.median(roots)
		#else:
			#r = roots[np.where(roots.imag == 0)][0].real
		
		# A PRIORI, there are three roots, and the one of interest is the one
		# inside the range (x.min, xmax), and real one. If there are several, keep one
		# at which the polynomial is minimum. If there are no root (len(r)=0), pass.
		roots = roots[np.where(roots.imag == 0)].real
		r = roots[(roots < x.max()) & (roots > x.min())]
		if len(r)==0:
			p=[np.nan]*3
			r=np.nan
		else:
			r = r[p(r) == min(p(r))][0]

	else:
		p = np.poly1d(np.nan)
		r = x[0]

	return (p,r)

def HartmannAnalysis(SrcDataL,SrcDataR, display=False):
	"""
	Inputs are "sources tables", as returned by the 
	GetAllSourcesMetrics() and analyse() methods,
	for the Left and Right Hartmann doors scans.
	The sources in Right and Left have to be the same.
	The "defocus" values should be the same for all sources.
	"""
	
	defoc = np.array(SrcDataL[0]['table']['defocus'])
	ax=plt.gca()
	ax.clear()
	bestHartmann = []
	slope=[]
	x=np.array([-0.2,0.2])
	for sL,sR in zip(SrcDataL,SrcDataR):
		yL=np.array(sL['table']['ycentroid'])
		yR=np.array(sR['table']['ycentroid'])
		pL = np.polyfit(defoc,yL,1)
		pR = np.polyfit(defoc,yR,1)
		p = pL-pR
		slope.append(p[0])
		bestHartmann.append(-p[1]/p[0])
		sL['Hart_best']=bestHartmann[-1]
		sR['Hart_best']=bestHartmann[-1]
		sL['Hart_poly']=np.poly1d(pL)
		sR['Hart_poly']=np.poly1d(pR)
		if display:
			ax.plot(defoc,yL,'-o',label='right door closed',lw=3,ms=8)
			ax.plot(defoc,yR,'-o',label='left door closed',lw=3,ms=8)
			ax.plot(x,pL[0]*x+pL[1],'--', label='linear fit (right door closed)',lw=2, color='b')
			ax.plot(x,pR[0]*x+pR[1],'--', label='linear fit (left door closed)',lw=2, color='g')
			ax.set_title("Hartmann test for source {}".format(sL['source']))
			ax.set_xlabel('defocus (mm)')
			ax.set_ylabel('y centroid (pixels)')
			ax.set_xlim(-0.25,0.25)
			y0,y1=ax.get_ylim()
			ax.annotate("Best focus: {:.3f}".format(bestHartmann[-1]), (-0.2, y0+0.08*(y1-y0)), va='top',ha='left')
			ax.plot([bestHartmann[-1]]*2,[y0,pL[0]*bestHartmann[-1]+pL[1]],'--',color='black')
			plt.draw()
			plt.legend(fontsize=14)
			raw_input('')
			ax.clear()

	return bestHartmann, slope

def viewResults(SrcData, OverallResult, suptitle='', datatype='FWHM', savefigures=False, SrcData2=None):
	"""
	Plot the results as a function of defocus. Data in a astropy Tables, as returned by analyse()
	"""
	fontsize0=plt.rcParams['font.size']
	plt.rcParams['font.size']=18
	
	fig = plt.figure(figsize=(19, 11))

	fibs = list(set([s['fiber_num'] for s in SrcData]))
	waves = list(set([s['wave_num'] for s in SrcData]))
	
	# consider only a reduced set of waves and fiber for the first plot. (plot too crowded otherwise)
	fibs_=fibs[::2]
	waves_=waves[::2]

	indexes=[k for k in range(len(SrcData)) if (fibs_.count(SrcData[k]['fiber_num'])!=0) & (waves_.count(SrcData[k]['wave_num']) != 0)]
	SrcData_=[SrcData[k] for k in indexes]
	if SrcData2 is not None:
		SrcData2_=[SrcData2[k] for k in indexes]

	#fibs.sort()
	#waves.sort()
	
	nx = len(set(fibs_))
	ny = len(set(waves_))
	
	if datatype == 'FWHM':
		# --------------------------------------
		# PLOT FWHMx/y VS DEFOCUS individually
		# --------------------------------------
		for source in SrcData_:
			k=source['source'] # source number
			ax = plt.subplot2grid((ny,nx),(waves_.index(source['wave_num']), fibs_.index(source['fiber_num'])))
			#ax = fig.add_subplot(ny, nx, k)
			for item in ax.get_yticklabels()+ax.get_xticklabels(): item.set_fontsize(10)

			x = source['table']['defocus']
			px = source['FWHMx_poly'] # polynomial fitting the FWHM
			py = source['FWHMy_poly']
			ax.plot(x, source['table']['FWHMx'], '-', label='FWHMx', linewidth=1)
			ax.plot(x, source['table']['FWHMy'], '-', label='FWHMy', linewidth=1)
			ax.set_ylim(2,6)
			ax.set_xlim(-0.15,0.15)

			ax.plot([source['FWHMx_best']]*2,[1,5],'--', linewidth=1)
			ax.plot([source['FWHMy_best']]*2,[1,5],'--', linewidth=1)
			ax.plot([source['FWHM_best']]*2,[1,5], linewidth=1)
			if np.isnan(px[0])==False:
				ax.plot(x,px(x),'--', linewidth=1)
				ax.plot(x,py(x),'--', linewidth=1)
			ax.annotate("Best: {:.3f} mm".format(source['FWHM_best']),(0.0,5.5), fontsize=11.0, ha='center')
			ax.set_title("source {}".format(k),fontsize=11.0)

		fig.suptitle(suptitle+' FWHMx/y VS Defocus',x=0.005,y=0.995, ha='left',va='top',fontsize=14)
		ax0 = fig.get_axes()[0]
		ax0.legend(fancybox=True, shadow=True, ncol=1, fontsize=12.0, loc='lower left')
		ax0.set_ylabel('FWHM (pixels)', fontsize=12.0)
		fig.tight_layout()

		#fig.tight_layout()
		if savefigures:
			fig.savefig('FWHMs_VS_Focus.png',dpi=200)
		
		# -----------------------------------
		# PLOT FWHMx VS DEFOCUS all at once
		# -----------------------------------
		fig=plt.figure()
		ax = fig.add_subplot(111)
		for source in SrcData:
			ax.plot(source['table']['defocus'],source['table']['FWHMx'])

		ma=max([s['table']['FWHMx'].max() for s in SrcData])
		mi=max([s['table']['FWHMx'].min() for s in SrcData])


		ax.plot([OverallResult['FWHM_best_average_FWHM']]*2,[mi*0.7,mi*1.5],'--',lw=2, color='grey')
		ax.set_xlabel('defocus (mm)')
		ax.set_ylabel('FWHM (pixel)')
		ax.set_ylim((mi*0.7,ma*1.05))
		
		fig.suptitle(suptitle,x=0.005,y=0.995, ha='left',va='top',fontsize=14)
		ax.set_title('All FWHMx VS Defocus') 
		fig.tight_layout()

		if savefigures:
			fig.savefig('All_FWHMx_VS_Focus.png',dpi=200)
		
		# -----------------------------------
		# PLOT FWHMy VS DEFOCUS all at once
		# -----------------------------------
		fig=plt.figure()
		ax = fig.add_subplot(111)
		for source in SrcData:
			ax.plot(source['table']['defocus'],source['table']['FWHMy'])
		
		ma=max([s['table']['FWHMy'].max() for s in SrcData])
		mi=max([s['table']['FWHMy'].min() for s in SrcData])
		
		ax.plot([OverallResult['FWHM_best_average_FWHM']]*2,[mi*0.7,mi*1.5],'--',lw=2,color='grey')
		ax.set_xlabel('defocus (mm)')
		ax.set_ylabel('FWHM (pixel)')
		ax.set_ylim((mi*0.7,ma*1.05))
		
		fig.suptitle(suptitle,x=0.005,y=0.995, ha='left',va='top',fontsize=14)
		ax.set_title('All FWHMy VS Defocus') 
		fig.tight_layout()
		if savefigures:
			fig.savefig('All_FWHMy_VS_Focus.png',dpi=200)
		
		# ----------------------------
		# PLOT BEST FOCUS VS X/Y (3D)
		# ----------------------------
		
		# The X's and Y's coordinates and Best focus for each source:
		Xs = [np.median(s['table']['xcentroid']+s['x0']) for s in SrcData]
		Ys = [np.median(s['table']['ycentroid']+s['y0']) for s in SrcData]
		Bs = [x['FWHM_best'] for x in SrcData]

		# four points, to display the plane surface
		XX,YY=np.meshgrid([0,4500],[0,4500])
		P = OverallResult['FWHMPlanefit'](XX,YY)
		
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d') 
		ax.plot(Xs,Ys,Bs, 'o', color='red')
		ax.plot_surface(XX,YY,P)
		ax.set_title('BEST FOCUS vs X/Y')

		#-------------------------
		# PLOT BEST FOCUS VS FIBER
		# ------------------------
		fig=plt.figure(figsize=(11,7))
		fig.suptitle(suptitle,x=0.005,y=0.995, ha='left',va='top',fontsize=14)
		ax=fig.add_subplot(111)

		ax.set_xticks(range(21))
		ax.grid(which='both')

		allXs=[s['fiber_num'] for s in SrcData]
		allYs=[s['FWHM_best'] for s in SrcData]
		pol=np.poly1d(np.polyfit(allXs,allYs,1))
		ax.plot([0.0,20],pol([0.0,20]),'--',color='blue',label='Best plane',lw=2)

		for w in range(30):
			sample=[s for s in SrcData if s['wave_num']==w]
			x=[s['fiber_num'] for s in sample]
			y=[s['FWHM_best'] for s in sample]
			if len(x)!=0: ax.plot(x,y,'-o',label="wave num={}".format(w))

		#ax.plot([-1,21],[OverallResult['FWHM_best_average_FWHM']]*2,'--',lw=2,color='grey',label='best focus')
		ax.set_xlim(-1,21)
		ax.set_ylim(-np.max(np.abs(allYs))*1.05,np.max(np.abs(allYs))*1.05)
		ax.set_xlabel("Fiber number")
		ax.set_ylabel("Best focus (mm)")
		ax.set_title("Best focus VS fiber number")

		ax.legend(loc="best",ncol=2,fontsize='small')
		fig.tight_layout()
		if savefigures:
			fig.savefig('BestFocusVSfiber.png',dpi=200)
		
		#-------------------------
		# PLOT BEST FOCUS VS WAVE
		# ------------------------
		fig=plt.figure(figsize=(11,7))
		fig.suptitle(suptitle,x=0.005,y=0.995, ha='left',va='top',fontsize=14)
		ax = fig.add_subplot(111)
		
		ax.grid(which='both')

		allXs=[np.mean(s['table']['ycentroid']+s['y0']) for s in SrcData]
		allYs=[s['FWHM_best'] for s in SrcData]    
		pol=np.poly1d(np.polyfit(allXs,allYs,1))
		ax.plot([50,3950],pol([50,3950]),'--',color='blue',label='Best plane',lw=2)

		for fib in range(0,21,2):                   
			sample=[s for s in SrcData if s['fiber_num']==fib]
			x=[np.mean(s['table']['ycentroid'])+s['y0'] for s in sample]
			y=[s['FWHM_best'] for s in sample]
			if len(x)!=0: ax.plot(x,y,'-o',label="fiber={}".format(fib))
		ax.set_xlabel("Y - axis pixel (wavelength)")
		ax.set_ylabel("Best FWHM focus (mm)")
		ax.set_title("Best focus VS wavelength")

		#ax.plot([0,4000],[OverallResult['FWHM_best_average_FWHM']]*2,'--',lw=2,color='grey',label='best focus')
		ax.set_xlim(0,4000)
		ax.legend(loc='best',fontsize='small',ncol=4)
		fig.tight_layout()
		if savefigures:
			fig.savefig('BestFocusVSwave.png',dpi=200)

		plt.pause(0.01) # this line makes the plots actually appear.

		print("\nBEST FOCUS IN TERMS OF FWHM:")
		print("----------------------------")
		for tup in [(k,OverallResult[k]) for k in OverallResult.keys() if k[0:5]=='FWHM_']:
			print("{}: {:.3f} mm".format(tup[0],tup[1]))
		print("Best plane tilt X: {:.5f} deg".format(np.arctan(OverallResult['FWHMPlanefit'].parameters[1])*180.0/np.pi))
		print("Best plane tilt Y: {:.5f} deg".format(np.arctan(OverallResult['FWHMPlanefit'].parameters[2])*180.0/np.pi))

	if datatype == 'Hartmann':
		# --------------------------------------
		# PLOT Ycentroid VS DEFOCUS individually
		# --------------------------------------
		if SrcData2 is None:
			print('SrcData2 is required for hartmann data type. Returning')
			sys.exit(0)
			
		for source, source2 in zip(SrcData_, SrcData2_):
			k=source['source'] # source number
			ax = plt.subplot2grid((ny,nx),(waves_.index(source['wave_num']), fibs_.index(source['fiber_num'])))
			#ax = fig.add_subplot(ny, nx, k)
			for item in ax.get_yticklabels()+ax.get_xticklabels(): item.set_fontsize(10)

			x = source['table']['defocus']
			pH = source['Hart_poly'] # polynomial fitting the FWHM
			pH2 = source2['Hart_poly'] # polynomial fitting the FWHM
			ax.plot(x, source['table']['ycentroid'], '-o', label='Ycentroid 1', linewidth=2)
			ax.plot(x, source2['table']['ycentroid'], '-o', label='Ycentroid 2', linewidth=2)
			ax.set_xlim(-0.2,0.2)

			ax.plot([source['Hart_best']]*2,[pH([x[0]]),pH([x[-1]])],'--', linewidth=1)
			if np.isnan(pH[0])==False:
				ax.plot(x,pH(x),'--', linewidth=1)
				ax.plot(x,pH2(x),'--', linewidth=1)
			ax.annotate("Best: {:.3f} mm".format(source['Hart_best']),(0.0,pH([x[0]])+0.1), fontsize=11.0, ha='center')
			ax.set_title("source {}".format(k),fontsize=11.0)

		fig.suptitle(suptitle+' Ycentroids VS Defocus',x=0.005,y=0.995, ha='left',va='top',fontsize=14)
		ax0 = fig.get_axes()[0]
		ax0.legend(fancybox=True, shadow=True, ncol=1, fontsize=12.0, loc='lower left')
		ax0.set_ylabel('Y centroid (pixels)', fontsize=12.0)
		fig.tight_layout()

		#fig.tight_layout()
		if savefigures:
			fig.savefig('Ycentroid_VS_Focus.png',dpi=200)
		
		# --------------------------------------
		# PLOT Y centroid VS DEFOCUS all at once
		# --------------------------------------
		fig=plt.figure()
		ax = fig.add_subplot(111)
		for source in SrcData:
			ax.plot(source['table']['defocus'],source['table']['ycentroid'],color='black')
		for source in SrcData2:
			ax.plot(source['table']['defocus'],source['table']['ycentroid'], color='Red')

		ma=max([s['table']['ycentroid'].max() for s in SrcData])
		mi=max([s['table']['ycentroid'].min() for s in SrcData])
		BestHartmanns=[s['Hart_best'] for s in SrcData] ## list of all Best Hartmann focus for each source

		ax.plot([np.mean(BestHartmanns)]*2,[mi*0.7,mi*1.5],'--',lw=2, color='grey')
		ax.set_xlabel('defocus (mm)')
		ax.set_ylabel('Y centroid (pixel)')
		ax.set_ylim((mi*0.7,ma*1.05))
		
		fig.suptitle(suptitle,x=0.005,y=0.995, ha='left',va='top',fontsize=14)
		ax.set_title('All Y centroids VS Defocus') 
		fig.tight_layout()

		if savefigures:
			fig.savefig('All_Ycentroid_VS_Focus.png',dpi=200)
		
		
		# ----------------------------
		# PLOT BEST FOCUS VS X/Y (3D)
		# ----------------------------
		
		# The X's and Y's coordinates and Best focus for each source:
		Xs = [np.median(s['table']['xcentroid']+s['x0']) for s in SrcData]
		Ys = [np.median(s['table']['ycentroid']+s['y0']) for s in SrcData]
		Bs = [x['Hart_best'] for x in SrcData]
		
		model = models.Polynomial2D(1) # a 2D polynomial of degree 1 = a plane
		Fitter=fitting.LinearLSQFitter()
		HartPlanefit= Fitter(model, Xs, Ys, Bs)
		
		# four points, to display the plane surface
		XX,YY=np.meshgrid([0,4500],[0,4500])
		P = HartPlanefit(XX,YY)
		
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d') 
		ax.plot(Xs,Ys,Bs, 'o', color='red')
		ax.plot_surface(XX,YY,P)
		ax.set_title('BEST Hartmann FOCUS vs X/Y')

		#-------------------------
		# PLOT BEST FOCUS VS FIBER
		# ------------------------
		fig=plt.figure(figsize=(11,7))
		fig.suptitle(suptitle,x=0.005,y=0.995, ha='left',va='top',fontsize=14)
		ax=fig.add_subplot(111)

		ax.set_xticks(range(21))
		ax.grid(which='both')

		allXs=[s['fiber_num'] for s in SrcData]
		allYs=[s['Hart_best'] for s in SrcData]
		pol=np.poly1d(np.polyfit(allXs,allYs,1))
		ax.plot([0.0,20],pol([0.0,20]),'--',color='blue',label='Best plane',lw=2)

		for w in range(30):
			sample=[s for s in SrcData if s['wave_num']==w]
			x=[s['fiber_num'] for s in sample]
			y=[s['Hart_best'] for s in sample]
			if len(x)!=0: ax.plot(x,y,'-o',label="wave num={}".format(w))

		ax.set_xlim(-1,21)
		ax.set_ylim(-np.max(np.abs(allYs))*1.05,np.max(np.abs(allYs))*1.05)
		ax.set_xlabel("Fiber number")
		ax.set_ylabel("Best Hartmann focus (mm)")
		ax.set_title("Best Hartmann focus VS fiber number")

		ax.legend(loc="best",ncol=2,fontsize='small')
		fig.tight_layout()
		if savefigures:
			fig.savefig('BestHartmannFocusVSfiber.png',dpi=200)
		
		#-------------------------
		# PLOT BEST FOCUS VS WAVE
		# ------------------------
		fig=plt.figure(figsize=(11,7))
		fig.suptitle(suptitle,x=0.005,y=0.995, ha='left',va='top',fontsize=14)
		ax = fig.add_subplot(111)
		
		ax.grid(which='both')

		allXs=[np.mean(s['table']['ycentroid']+s['y0']) for s in SrcData]
		allYs=[s['Hart_best'] for s in SrcData]    
		pol=np.poly1d(np.polyfit(allXs,allYs,1))
		ax.plot([50,3950],pol([50,3950]),'--',color='blue',label='Best plane',lw=2)

		for fib in range(0,21,2):                   
			sample=[s for s in SrcData if s['fiber_num']==fib]
			x=[np.mean(s['table']['ycentroid'])+s['y0'] for s in sample]
			y=[s['Hart_best'] for s in sample]
			if len(x)!=0: ax.plot(x,y,'-o',label="fiber={}".format(fib))
		ax.set_xlabel("Y - axis pixel (wavelength)")
		ax.set_ylabel("Best Hartmann focus (mm)")
		ax.set_title("Best Hartmann focus VS wavelength")

		#ax.plot([0,4000],[OverallResult['FWHM_best_average_FWHM']]*2,'--',lw=2,color='grey',label='best focus')
		ax.set_xlim(0,4000)
		ax.legend(loc='best',fontsize='small',ncol=4)
		fig.tight_layout()
		if savefigures:
			fig.savefig('BestHartmannFocusVSwave.png',dpi=200)

		plt.pause(0.01) # this line makes the plots actually appear.

		print("\nBEST HARTMANN FOCUS:")
		print("----------------------------")
		print("Best Hartmann focus: {:.3f} mm".format(np.mean(BestHartmanns)))
		#print("Best plane tilt X: {:.3f} deg".format(np.arctan(HartPlanefit.parameters[1])*180.0/np.pi))
		#print("Best plane tilt Y: {:.3f} deg".format(np.arctan(HartPlanefit.parameters[2])*180.0/np.pi))

	if datatype == 'Ree':
		for source in SrcData_:
			k=source['source'] # source number
			ax = plt.subplot2grid((ny,nx),(waves_.index(source['wave_num']), fibs_.index(source['fiber_num'])))
			for item in ax.get_yticklabels()+ax.get_xticklabels(): item.set_fontsize(9)
			
			x = source['table']['defocus']
			px = source['Ree_poly'] # polynomial fitting the FWHM
			
			ax.plot(x, source['table']['Ree'], '-', label='Ree', linewidth=1)
			ax.set_ylim(2,5)
			ax.set_xlim(-0.23,0.23)

			ax.plot([source['Ree_best']]*2,[1,5],'--', linewidth=1)
			if np.isnan(px[0])==False:
				ax.plot(x,px(x),'--', linewidth=1)
			ax.annotate("Best: {:.3f} mm".format(source['Ree_best']),(0.0,4.5), fontsize=8.0, ha='center')
			ax.set_title("source {}".format(k),fontsize=7.0)

		fig.suptitle(suptitle,y=1.0,fontsize=12)
		ax0 = fig.get_axes()[0]
		ax0.legend(fancybox=True, shadow=True, ncol=1, fontsize=10.0, loc='lower left')
		ax0.set_xlabel(xlabel, fontsize=11.0)
		ax0.set_ylabel(ylabel, fontsize=11.0)
			
		fig.tight_layout()
		plt.draw()
		
		# The X's and Y's coordinates and Best focus for each source:
		Xs = [np.median(s['table']['xcentroid']+s['x0']) for s in SrcData]
		Ys = [np.median(s['table']['ycentroid']+s['y0']) for s in SrcData]
		Bs = [x['Ree_best'] for x in SrcData]

		# four points, to display the plane surface
		XX,YY=np.meshgrid([0,4500],[0,4500])
		P = OverallResult['ReePlanefit'](XX,YY)
		
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d') 
		ax.plot(Xs,Ys,Bs, 'o', color='red')
		ax.plot_surface(XX,YY,P)
		ax.set_title('Ree')
		
		print("\nBEST FOCUS IN TERMS OF Ree:")
		print("---------------------------")
		for tup in [(k,OverallResult[k]) for k in OverallResult.keys() if k[0:4]=='Ree_']:
			print("{}: {:.3f} mm".format(tup[0],tup[1]))
		#print("Best plane tilt X: {:.3f} deg".format(np.arctan(OverallResult['ReePlanefit'].parameters[1])*180.0/np.pi))
		#print("Best plane tilt Y: {:.3f} deg".format(np.arctan(OverallResult['ReePlanefit'].parameters[2])*180.0/np.pi))
		
	plt.rcParams['font.size']=fontsize0

class SourceSelector():
	"""
	Tool created to click a source on a image, and return
	the source found in "sources"
	(astropy Table returned by phot.daofind())	
	"""
	
	def __init__(self, im, sources, channel):
		self.sources = sources
		self.channel = channel
		self.im = im
		self.fig = plt.figure(figsize=(12,10))
		self.ax = self.fig.add_subplot(111,aspect='equal')
		self.fig.tight_layout()
		self.state = '' # état du selecteur (selecting, adding or deleting)
		
		# données utiles pour daofind (si ajout de sources à la main)
		pix_sz = 0.015  # pixel size in mm
		self.bkg_sigma = mad_std(im) 
		self.FWHM_estim = lambda z: abs(z) / 2.0 / 1.7 / pix_sz + 3.333
		
		#intervalInstance = visu.AsymmetricPercentileInterval(0.1,99.9)
		#lim = intervalInstance.get_limits(im)
		#norm = ImageNormalize(vmin=lim[0], vmax=lim[1], stretch=visu.SqrtStretch())
		
		#other way of computing norme for vizualisation
		H,bin_edges=np.histogram(im,bins=20000)
		x=(bin_edges[:-1]+bin_edges[1:])/2.0
		Fitter=fitting.LevMarLSQFitter()
		model = models.Gaussian1D(amplitude=H.max(),mean=np.mean(im),stddev=self.bkg_sigma)
		fit = Fitter(model, x, H)
		mm = fit.mean.value
		self.bkg_sigma = np.abs(fit.stddev.value)
		# high vmax better with R and Z that have bad amps (with high values)
		norm = ImageNormalize(vmin=mm-4*self.bkg_sigma, vmax=mm+40*self.bkg_sigma, stretch=visu.SqrtStretch())

		self.ax.imshow(im, aspect='equal', norm=norm)

		self.crosses, = self.ax.plot(self.sources['xcentroid'],self.sources['ycentroid'],'x',color='green', mew=2,ms=8)
		self.sh=im.shape
		self.ax.set_xlim(0,self.sh[0])
		self.ax.set_ylim(0,self.sh[1])
		
		self.ax.format_coord = self.format_coord # reassign format coord function to my function (just added the "status" display)
		self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
		self.cidkey1 = self.fig.canvas.mpl_connect('key_press_event', self.onkeypress)
		self.cidkey2 = self.fig.canvas.mpl_connect('key_release_event', self.onkeyrelease)

		self.columns = []
		self.rows = []
		
		print("Welcome to the Source Selector tool")
		print("------------------------------------")
		print("Press and hold 'Shift' and click left or right to add or suppress sources")
		print("Press and hold 'Ctrl' and click left or right to build a grid")
		print("Press 'Enter' to validate")
		print("Press 'q' to quit")

		plt.show()
		
	def onclick(self, event):
		if self.fig.canvas.toolbar._active is not None:
			return

		distances = np.asarray(np.sqrt((self.sources['xcentroid']-event.xdata)**2+(self.sources['ycentroid']-event.ydata)**2))
		ind, = np.where(distances <= 20.0)
		if len(ind) > 1:
			# s'il y a plusieurs sources, on garde celle au plus près du clique
			ind = ind[np.where(distances[ind] == distances[ind].min())]

		if all([ind.shape[0] == 0,any([self.state!='add or suppress',event.button!=1])]):
		# s'il n'y pas de source à moins de 20 pix du clique et qu'on n'est pas
		# à vouloir ajouter une sources, alors, on ne fait rien
			return
		
		if len(ind) >= 1:
			ind = int(ind[0])	# ind est un numpy array. On prend la valeur scalaire, et on la convertit en entier

		if self.state=='':
			print(self.sources[ind], "\n")
		
		if all([event.button == 3, self.state == 'add or suppress']):   # DELETING SOURCE 
			print('deleting source {}'.format(self.sources['id'][ind]))
			self.sources.remove_row(ind)
			self.crosses.set_data(self.sources['xcentroid'],self.sources['ycentroid'])
			plt.pause(0.01)
		
		if all([event.button == 1, self.state == 'add or suppress']): # ADDING SOURCE 
			hw = 7
			x0 = event.xdata - hw
			x1 = event.xdata + hw
			y0 = event.ydata - hw
			y1 = event.ydata + hw
			daofind = phot.DAOStarFinder(3.*self.bkg_sigma,self.FWHM_estim(0),sharplo=0.02, sharphi=0.95, roundlo=-0.9, roundhi=0.9)
			newsource = daofind(self.im[int(y0):int(y1),int(x0):int(x1)])
			#newsource = phot.daofind(self.im[int(y0):int(y1),int(x0):int(x1)], fwhm=self.FWHM_estim(0), threshold=3.*self.bkg_sigma, sharplo=0.02, sharphi=0.95, roundlo=-0.9, roundhi=0.9)
			if len(newsource) != 0:
				j = newsource['peak'].argmax()
				newsource = newsource[j]
				newsource['xcentroid'] = newsource['xcentroid'] + x0
				newsource['ycentroid'] = newsource['ycentroid'] + y0
				print('adding source \n{}'.format(newsource))
				if len(self.sources) != 0:
					self.sources.add_row(newsource)
				else:
					self.sources = Table(newsource) # if self.sources was empty (no source found automatically), create it.
				self.crosses.set_data(self.sources['xcentroid'],self.sources['ycentroid'])
				plt.pause(0.01)
		
		if self.state == 'building grid':
			x0 = self.sources['xcentroid'][ind]
			y0 = self.sources['ycentroid'][ind]
			
			if event.button == 1:
				self.columns.append(x0)
				plt.plot([x0,x0], [0,self.sh[0]], color="blue")
				plt.pause(0.01)
				
			if event.button == 3:
				#equation de parabole y = ax^2 + bx + c = a(x-alpha)^2+beta (forme canonique)
				# voir https://fr.wikipedia.org/wiki/%C3%89quation_du_second_degr%C3%A9#Forme_canonique

				"""Explanation:
				Parabola passing by 3 points of a given spectral line in computed
				Done for 2 different lines (different y coordinate)
				Linear dependance of parameters "a" and "alpha" with coordinate y is derived
				Works for spectrograph EM1.
				Alpha is the position of the minimum of the parabola (should be constant with y in no tilt of focal plane)
				For a more general equation, set alpha to center of the image (self.sh[1]/2)
				"""
				if self.channel == 'B1':
					a = 2.76e-9*y0 + 1.399e-5
					alpha = 1897.2+1.385e-2*y0 # minimum de la parabole
				if self.channel == 'R1':
					a = 3.38e-9*y0 + 2.38e-5
					alpha = 1995.8+5.875e-3*y0
				if self.channel == 'Z1':
					a = 2.7e-9*y0+2.94e-5
					alpha = 2012.68+1.595e-3*y0
				
				#alpha = self.sh[1]/2 # minimum de la parabole, au centre de l'image
				b = -2*a*alpha
				beta = y0-a*(x0-alpha)**2
				c = beta+b**2/4.0/a
				p=models.Polynomial1D(2,c0=c,c1=b,c2=a)
				self.rows.append(p)
				x = np.linspace(0,self.sh[1],100)
				plt.plot(x,p(x), color="blue")
				plt.pause(0.01)

		return
	  
	def onkeypress(self, event):
		if (event.key.lower() == 'q'):
			print("Quitting the Source Selector...")
			self.fig.canvas.mpl_disconnect(self.cid)
			self.fig.canvas.mpl_disconnect(self.cidkey1)
			self.fig.canvas.mpl_disconnect(self.cidkey2)
			indexes=np.linspace(1,len(self.sources),len(self.sources))
			self.sources['id']=indexes
			plt.close(self.fig)
					
		if (event.key.lower() == 'shift'):
			self.state = 'add or suppress'

		if (event.key.lower() == 'control'):
			self.state = 'building grid'
		
		if (event.key.lower() == 'enter'):
			if any([len(self.columns)==0, len(self.rows)==0]):
				return
			
			(Xs,Ys) = self.intersects(self.rows, self.columns)
			
			finalsources = self.sources[0:0] # an astropy Table of the same format, with no row.
			for i,x0 in enumerate(Xs):
				y0 = Ys[i]
				distances = np.asarray(np.sqrt((self.sources['xcentroid']-x0)**2+(self.sources['ycentroid']-y0)**2))
				ind, = np.where(distances <= 20.0)

				if len(ind) > 1:
					# s'il y a plusieurs sources, on garde celle au plus près du clique
					ind = ind[np.where(distances[ind] == distances[ind].min())]

				if ind.shape[0] != 0:
					ind = int(ind[0])		  
					finalsources.add_row(self.sources[ind])

			self.ax.plot(finalsources['xcentroid'],finalsources['ycentroid'],'+',color='yellow', mew=2,ms=8)
			plt.pause(0.01)
			
			self.sources = finalsources # overwrite the "sources" Table with the few sources kept.

	def intersects(self, polynomes, droites):
		"""
		fonction très spécifique, qui renvoie les coordonnées des intersections
		entre tous les polynomes du 2nd degré définis dans "polynome" et 
		toutes les droites verticales d'abscisse contenues dans "droites"
		"""
		X=[]
		Y=[]
		for p in polynomes:
			P = p.parameters
			for x0 in droites:
				X.append(x0)
				Y.append(P[2]*x0**2+P[1]*x0+P[0])

		return (X,Y)
		
	def onkeyrelease(self, event):
			self.state = ''

	def format_coord(self, x, y):
		"""Return a format string formatting the *x*, *y* coord"""
		if x is None:
			xs = '???'
		else:
			xs = self.ax.format_xdata(x)
		if y is None:
			ys = '???'
		else:
			ys = self.ax.format_ydata(y)
		if all([self.fig.canvas.toolbar._active is None, self.state != '']):
			return '%s SOURCES...   x=%s y=%s' % (self.state.upper(), xs, ys)
		else:
			return 'x=%s y=%s' % (xs, ys)

def cleanup(srclist):
	fi=open('cleanup.txt','r')
	lines=fi.readlines()
	Srcindex = [int(l.split()[0]) for l in lines]
	mini = [float(l.split()[1]) for l in lines]
	maxi = [float(l.split()[2]) for l in lines]
	fi.close()
	
	for i,ind in enumerate(Srcindex):
		BoolArray = (srclist[ind-1]['defocus'] < mini[i])  | (srclist[ind-1]['defocus'] > maxi[i])
		srclist[ind-1]['fwhm_mean'][BoolArray] = np.nan
		srclist[ind-1]['xcent'][BoolArray] = np.nan
		srclist[ind-1]['ycent'][BoolArray] = np.nan
		
	return srclist

def renumber(srclist, channel='R1'):
	"""
	renumber source list (increasing fiber num, and wavelength)
	Sources are numbered on a regular grid. Even empty grid places are accounted for.
	srclist is a list as returned by getSources()
	"""
	if channel == 'R1':
		#good values for EM1 RED
		xlims = np.hstack((np.arange(20,4000,192),4000)) # limits of the 20 fibers, hard-coded
		transform = lambda x,y: y - (2.60e-5+2e-9*y)*(x-2057.0)**2
	
	if channel == 'B1':
		#good values for EM1 BLUE
		xlims = np.hstack((np.arange(20,4000,190),4000)) # limits of the 20 fibers, hard-coded
		transform = lambda x,y: y - (1.90e-5+2e-9*y)*(x-2057.0)**2
	if channel == 'B2':
                #good values for SM03 BLUE?
                xlims = np.hstack((np.arange(20,4000,190),4000)) # limits of the 20 fibers, hard-coded
                transform = lambda x,y: y - (1.90e-5+2e-9*y)*(x-2057.0)**2
	
	if channel == 'Z1':
		#good values for EM1 NIR
		xlims = np.hstack((np.arange(80,4000,190),4000)) # limits of the 20 fibers, hard-coded
		transform = lambda x,y: y - (2.8e-5+2e-9*y)*(x-2057.0)**2
	if channel == 'Z2':
                #good values for SM03 NIR?
                xlims = np.hstack((np.arange(80,4000,190),4000)) # limits of the 20 fibers, hard-coded
                transform = lambda x,y: y - (2.8e-5+2e-9*y)*(x-2057.0)**2	
	allxs=np.array([s['x0']+s['xcent'] for s in srclist])
	allys=np.array([s['y0']+s['ycent'] for s in srclist])
	allYs = np.array([transform(allxs[i],allys[i]) for i in range(len(allxs))])
	
	#Y= np.array([ys[i] - (2.65e-5+2e-9*ys[i])*(xs[i]-2057.0)**2 for i in range(len(xs))]) # spectre "mis à plat" (correction distortion)

	# FIRST COLUMN OF SOURCES (leftmost.. fiber n°20 on test bench November 2016 with firts test slit)
	# must have all the wavelengths!
	#subset1 = [srclist[i] for i in range(len(srclist)) if (allxs[i] > xlims[0]) & (allxs[i] < xlims[1])]
	
	# loop to find the column where the number of waves (spectral lines) is maximum:
	subset1=[]
	for j in range(len(xlims)-1):
		subset_temp=[srclist[i] for i in range(len(srclist)) if (allxs[i] > xlims[j]) & (allxs[i] < xlims[j+1])]
		if len(subset_temp)>len(subset1):
			subset1 = subset_temp
	xs = np.array([s['x0']+s['xcent'] for s in subset1])
	ys = np.array([s['y0']+s['ycent'] for s in subset1])
	Ys = np.array([transform(xs[i],ys[i]) for i in range(len(xs))])
	Ys.sort()
	ylims = np.hstack((0,(Ys[0:-1]+Ys[1:])/2.0,4000))
	#xlims = xlims[::-1] # reverse order if fiber 0 rightmost, fiber 20 left most
	#ylims = ylims[::-1] # reverse order if long waves at the bottom and short wave at the top
	k=1
	for j in range(len(ylims)-1):
		ymin = min(ylims[j],ylims[j+1]) # proceed like this if order of ylims is changed. Would still work
		ymax = max(ylims[j],ylims[j+1])
		for i in range(len(xlims)-1):
			xmin = min(xlims[i],xlims[i+1]) # proceed like this if order of xlims is changed. Would still work
			xmax = max(xlims[i],xlims[i+1])
			u=np.where((allxs>xmin) & (allxs<xmax) & (allYs>ymin) & (allYs<ymax))[0]
			if len(u)==1:
				s=srclist[u[0]]
				s['source']=k
				s['fiber_num'] = i
				s['wave_num'] = j
			k = k+1
	
	srclist.sort(key=lambda s:s['source'])

	return srclist

def main(files, channel='R1'):	
	"""
		use this function if working from python interpreter
		import focus.src.focus_DESI_2 as foc
		SrcData, OverallResult = foc.main(files, channel='B1')
	"""
	
	sources = getSources(files, channel=channel,autoSelect=False)
	SrcData = GetAllSourcesMetrics(sources, ee=0.80, display=False)
	SrcData, OverallResult = analyse(SrcData)
	viewResults(SrcData, OverallResult, xlabel='defocus (mm)', ylabel='FWHM (pixels)',title='FOCUS TEST / FWHM (gaussian fit)',datatype='FWHM')
	#viewResults(SrcData, OverallResult, xlabel='defocus (mm)', ylabel='Ree (pixels)',title='FOCUS TEST / R80',datatype='Ree')

	return SrcData, OverallResult

if __name__ == '__main__':   
	##cols=[fits.Column(name=table.columns[k].name, format='E', array=table.columns[k].data) for k in range(len(table.columns))]
	#cols = fits.ColDefs(cols)
	#tbhdu = fits.BinTableHDU.from_columns(cols)
	#tbhdu.writeto(outdir+'focusSimul_Gaussian.fits')
	
	plt.ion()
	parser = argparse.ArgumentParser()
	
	parser.add_argument("-get","--getsources",
						help="Read the fits files and retrieve the sources.",
						action="store_true")
	parser.add_argument("-a","--analyse",
						help="Analyse the data in the \"sources\" data file",
						action="store_true")
	parser.add_argument("-view","--viewresults",
						help="View the results from the analysis",
						action="store_true")
		
	parser.add_argument("-i", "--inputfile",
						help="String giving the generic filename of the files to read if --getsources is set. Ex: '/full_path/FOCUS_16062016_\*.fits'. \
						Otherwise, the path to the sources.dat file if --analyse is set \
						or the results.dat file is --viewresults is set.",
						type=str)
	parser.add_argument("-o","--outputfile",
						help="The output filename (full path) for the sources data if --getsources is set \
						or the results if --analyse is set.",
						type=str)
	parser.add_argument("--display", help="If set, display the profiles fitted for each source\
						Only valid in --analyse is set",
						action="store_true",
						default=False)
	args = parser.parse_args()
	
	A = args.getsources
	B = args.analyse
	C = args.viewresults
	if (A & B) | (A & C) | (B & C):
		print("It is a mistake to specify more than one argument amongst: --getsources, --analyse and --viewresults.\nReturning")
		sys.exit(1)
	
	if args.getsources:
		if args.inputfile is None:
			print("input files pattern is missing")
			sys.exit(1)
		print("\nGetting sources...\n")
		files = ft.fsearch(ft.file_basename(args.inputfile), ft.file_dirname(args.inputfile))
		if len(files) == 0:
			sys.exit(1)
		sources = getSources(files, autoSelect=False, channel='B1')
		if args.outputfile is None:
			args.outputfile='./outputs/sources.dat'
		try:
			os.makedirs(ft.file_dirname(args.outputfile))
		except OSError:
			None	   # output directory already exists. Nothing to do.
		pk.dump(sources, open(args.outputfile,'wb'))

		print("\nSources saved to {}".format(args.outputfile))
	
	if args.analyse:
		print("\nAnalysing data...\n")
		if args.inputfile is None:
			args.inputfile = './outputs/sources.dat'
		sources = pk.load(open(args.inputfile,'rb'))
		SrcData = GetAllSourcesMetrics(sources, ee=0.80, display=args.display)
		SrcData, OverallResult = analyse(SrcData)
		if args.outputfile is None:
			args.outputfile='./outputs/results.dat'
		try:
			os.makedirs(ft.file_dirname(args.outputfile))
		except OSError:
			None	   # output directory already exists. Nothing to do.
		pk.dump((SrcData, OverallResult), open(args.outputfile,'wb'))
		print("\nResults saved to {}".format(args.outputfile))
	
	if args.viewresults:
		if args.inputfile is None:
			args.inputfile='./outputs/results.dat'
		SrcData, OverallResult = pk.load(open(args.inputfile,'rb'))
		viewResults(SrcData, OverallResult, xlabel='defocus (mm)', ylabel='FWHM (pixels)',title='FOCUS TEST / FWHM (Gaussian fit)',datatype='FWHM')
		viewResults(SrcData, OverallResult, xlabel='defocus (mm)', ylabel='Ree (pixels)',title="FOCUS TEST / R80",datatype='Ree')
		raw_input("\nPress enter to finish")

	print("\nDone")
	print("--------------------------------\n")
	
