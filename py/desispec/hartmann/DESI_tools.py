# -*- coding: utf-8 -*-

from __future__ import print_function
from astropy.io import fits
from astropy.io.fits.hdu import HDUList
#from qla import qla
# usual stuff
import numpy as np
import re
from filesTools import fsearch
import sys
import os
from datetime import datetime
from numpy.polynomial.legendre import legval

#import pdb

# default parameters for matplotlib
#plt.rcParams['image.aspect']='auto'
#plt.rcParams['image.interpolation']='none'
#plt.rcParams['image.cmap']='gist_heat'
#plt.rcParams['savefig.dpi']=300

def _parse_sec_keyword(value):
	'''
	parse keywords like BIASSECB='[7:56,51:4146]' into python slices

	python and FITS have almost opposite conventions,
		* FITS 1-indexed vs. python 0-indexed
		* FITS upperlimit-inclusive vs. python upperlimit-exclusive
		* FITS[x,y] vs. python[y,x]

	i.e. BIASSEC2='[7:56,51:4146]' -> (slice(50,4146), slice(6,56))

	taken from desi pipeline (10/01/2017)
	'''
	m = re.search('\[(\d+):(\d+)\,(\d+):(\d+)\]', value)
	if m is None:
		m = re.search('\[(\d+):(\d+)\, (\d+):(\d+)\]', value)
		if m is None :
			raise ValueError('unable to parse {} as [a:b, c:d]'.format(value))

	xmin, xmax, ymin, ymax = tuple(map(int, m.groups()))

	return np.s_[ymin-1:ymax, xmin-1:xmax]

def add_sector_info(header):
	# tuned to first teststand image
	# where keywords don't appear

	header["DATASEC1"]='[10:2064,4:2065]'
	d1xmin,d1xmax,d1ymin,d1ymax = parse_sec_keyword(header["DATASEC1"])
	header["PRESEC1"]='[1:%d,%d:%d]'%(d1xmin-1,d1ymin,d1ymax)
	header["CCDSEC1"]='[1:%d,1:%d]'%(d1xmax-d1xmin+1,d1ymax-d1ymin+1)
	header["BIASSEC1"]='[%d:2100,%d:%d]'%(d1xmax+1,d1ymin,d1ymax)
	c1xmin,c1xmax,c1ymin,c1ymax = parse_sec_keyword(header["CCDSEC1"])
	p1xmin,p1xmax,p1ymin,p1ymax = parse_sec_keyword(header["PRESEC1"])
	b1xmin,b1xmax,b1ymin,b1ymax = parse_sec_keyword(header["BIASSEC1"])

	header["DATASEC2"]='[2137:4187,%d:%d]'%(d1ymin,d1ymax)
	d2xmin,d2xmax,d2ymin,d2ymax = parse_sec_keyword(header["DATASEC2"])
	header["PRESEC2"]='[%d:%d,%d:%d]'%(d2xmax,header["NAXIS1"],d2ymin,d2ymax)    
	header["BIASSEC2"]='[%d:%d,%d:%d]'%(b1xmax+1,d2xmin-1,d2ymin,d2ymax)    
	header["CCDSEC2"]='[%d:%d,%d:%d]'%(c1xmax+1,c1xmax+1+d2xmax-d2xmin,1,d2ymax-d2ymin+1)    
	c2xmin,c2xmax,c2ymin,c2ymax = parse_sec_keyword(header["CCDSEC2"])
	p2xmin,p2xmax,p2ymin,p2ymax = parse_sec_keyword(header["PRESEC2"])
	b2xmin,b2xmax,b2ymin,b2ymax = parse_sec_keyword(header["BIASSEC2"])

	header["DATASEC3"]='[%d:%d,2136:4197]'%(d1xmin,d1xmax)
	d3xmin,d3xmax,d3ymin,d3ymax = parse_sec_keyword(header["DATASEC3"])    
	header["PRESEC3"]='[%d:%d,%d:%d]'%(p1xmin,p1xmax,d3ymin,d3ymax)
	header["BIASSEC3"]='[%d:%d,%d:%d]'%(b1xmin,b1xmax,d3ymin,d3ymax)
	header["CCDSEC3"]='[%d:%d,%d:%d]'%(c1xmin,c1xmax,c1ymax+1,c1ymax+1+d3ymax-d3ymin)
	c3xmin,c3xmax,c3ymin,c3ymax = parse_sec_keyword(header["CCDSEC3"])

	header["DATASEC4"]='[%d:%d,%d:%d]'%(d2xmin,d2xmax,d3ymin,d3ymax)
	header["PRESEC4"]='[%d:%d,%d:%d]'%(p2xmin,p2xmax,d3ymin,d3ymax)
	header["BIASSEC4"]='[%d:%d,%d:%d]'%(b2xmin,b2xmax,d3ymin,d3ymax)
	header["CCDSEC4"]='[%d:%d,%d:%d]'%(c2xmin,c2xmax,c3ymin,c3ymax)

	return header

#def change_sector_info(header, channel):
	##header['DATASEC1'] = '[9:2064,1:2065]'
	##header['DATASEC2'] = '[2193:4249,1:2065]'
	##header['DATASEC3'] = '[9:2064,2128:4192]'
	##header['DATASEC4'] = '[2193:4249,2128:4192]'
	
	## required modification for images taken between Dec 1st and ....
	#sec1 = header['DATASEC1']
	#sec2 = header['DATASEC2']
	#sec3 = header['DATASEC3']
	#sec4 = header['DATASEC4']

	#if channel = 'R1':
		#pass
	#if channel = 'B1':
		#header['DATASEC1'] = 
		#header['DATASEC2'] = 
		#header['DATASEC3'] = 
		#header['DATASEC4'] = 

	#if channel = 'Z1':
		#header['DATASEC1'] =
		#header['DATASEC2'] =
		#header['DATASEC3'] =
		#header['DATASEC4'] =


	#return header

def fixHeaderBeforeRun1341(hdu):
	datasec1 = _parse_sec_keyword(hdu.header["DATASEC1"])
	datasec2 = _parse_sec_keyword(hdu.header["DATASEC2"])         
	datasec3 = _parse_sec_keyword(hdu.header["DATASEC3"]) 
	datasec4 = _parse_sec_keyword(hdu.header["DATASEC4"])      
	
	y1min,y1max,x1min,x1max = datasec1[0].start, datasec1[0].stop, datasec1[1].start, datasec1[1].stop
	y2min,y2max,x2min,x2max = datasec2[0].start, datasec2[0].stop, datasec2[1].start, datasec2[1].stop
	y3min,y3max,x3min,x3max = datasec3[0].start, datasec3[0].stop, datasec3[1].start, datasec3[1].stop
	y4min,y4max,x4min,x4max = datasec4[0].start, datasec4[0].stop, datasec4[1].start, datasec4[1].stop

	if hdu.name=='B1':
		# fix for Blue channel before image 1341
		y1max += 1
		y2max += 1
		y3min -= 1
		y4min -= 1
		
	hdu.header["DATASEC1"] = '[{}:{}, {}:{}]'.format(x1min+1,x1max,y1min+1,y1max)
	hdu.header["DATASEC2"] = '[{}:{}, {}:{}]'.format(x2min+1,x2max,y2min+1,y2max)
	hdu.header["DATASEC3"] = '[{}:{}, {}:{}]'.format(x3min+1,x3max,y3min+1,y3max)
	hdu.header["DATASEC4"] = '[{}:{}, {}:{}]'.format(x4min+1,x4max,y4min+1,y4max)

	for old,new in zip(('1','2','3','4'),('A','B','C','D')):
		hdu.header.rename_keyword('DATASEC{}'.format(old),'DATASEC{}'.format(new))
		hdu.header.rename_keyword('BIASSEC{}'.format(old),'BIASSEC{}'.format(new))
		hdu.header.rename_keyword('CCDSEC{}'.format(old),'CCDSEC{}'.format(new))
		

	return hdu.header

def _clipped_std_bias(nsigma):
	'''
	Returns the bias on the standard deviation of a sigma-clipped dataset

	Divide by the returned bias to get a corrected value::

		a = nsigma
		bias = sqrt((integrate x^2 exp(-x^2/2), x=-a..a) / (integrate exp(-x^2/2), x=-a..a))
				= sqrt(1 - 2a exp(-a^2/2) / (sqrt(2pi) erf(a/sqrt(2))))

	See http://www.wolframalpha.com/input/?i=(integrate+x%5E2+exp(-x%5E2%2F2),+x+%3D+-a+to+a)+%2F+(integrate+exp(-x%5E2%2F2),+x%3D-a+to+a)

	taken from desi pipeline (10/01/2017)
	'''
	from scipy.special import erf
	a = float(nsigma)
	stdbias = np.sqrt(1 - 2*a*np.exp(-a**2/2.) / (np.sqrt(2*np.pi) * erf(a/np.sqrt(2))))
	return stdbias

def _overscan(pix, nsigma=5, niter=3):
	'''
	returns overscan, readnoise from overscan image pixels

	Args:
		pix (ndarray) : overscan pixels from CCD image

	Optional:
		nsigma (float) : number of standard deviations for sigma clipping
		niter (int) : number of iterative refits
		
	taken from desi pipeline (10/01/2017)

	'''
	#- normalized median absolute deviation as robust version of RMS
	#- see https://en.wikipedia.org/wiki/Median_absolute_deviation


	overscan = np.median(pix)
	absdiff = np.abs(pix - overscan)
	readnoise = 1.4826*np.median(absdiff)

	#- input pixels are integers, so iteratively refit
	for i in range(niter):
		absdiff = np.abs(pix - overscan)
		good = absdiff < nsigma*readnoise
		if len(np.where(good == True)[0]) >= 1:
			overscan = np.mean(pix[good])
			readnoise = np.std(pix[good])

	#- correct for bias from sigma clipping
	readnoise /= _clipped_std_bias(nsigma)

	return overscan, readnoise

def extract(hdu, silent=False):    
	"""
	Extracts the data area in the image, and return the array
	TODO: Prescan and bias pixels can be recovered here to
	"""
	try:
		hdu.header['DATASEC1']
	except KeyError:
		try:
			hdu.header['DATASECA']
		except KeyError:
			return hdu.data # nothing to do, the array as already been extracted (typically, fits files already read with DESIImage, and written to disk)

	if not silent: print("Extracting {} DATA array...".format(hdu.name))

	if hdu.header['EXPNUM'] < 1341:
		hdu.header = fixHeaderBeforeRun1341(hdu)
	
	allQ = []
	Vx = [] # vecteurs utilisés pour la détermination des positions des quadrants (upper, lower, left, right)
	Vy = []
	
	for amp in ['A','B','C','D']:
		datapix = _parse_sec_keyword(hdu.header['DATASEC'+amp])
		Vx.append(datapix[1].start+1)
		Vy.append(datapix[0].start+1)
		
		overscanpix = _parse_sec_keyword(hdu.header['BIASSEC'+amp])
		overscan, rdnoise = _overscan(hdu.data[overscanpix])
		
		if 'GAIN'+amp in hdu.header:
			gain = hdu.header['GAIN'+amp]          #- gain = electrons / ADU
		else:
			if not silent: print('Missing keyword GAIN{}; using 1.0'.format(amp))
			gain = 1.0
		
		rdnoise *= gain
		hdu.header['OVERSCN'+amp] = overscan
		hdu.header['OBSRDN'+amp] = rdnoise
        
		# store data for this quadrant in the list allQ
		allQ.append((hdu.data[datapix]-overscan)*gain)

	Vx = np.array(Vx,dtype='float32')
	Vy = np.array(Vy,dtype='float32')
	S = list(np.sqrt(Vx**2+Vy**2)) # distance
	A = list(np.arctan(Vy/Vx)) # angle

	LL = allQ[S.index(min(S))]
	UR = allQ[S.index(max(S))]
	LR = allQ[A.index(min(A))]
	UL = allQ[A.index(max(A))]
	
	Low = np.concatenate((LL,LR),axis=1)
	Up = np.concatenate((UL,UR),axis=1)
	newarray = np.asarray(np.concatenate((Low,Up),axis=0), dtype='float32')
			
	for amp in ['A','B','C','D']:
		#print(hdu.header['DATASEC{:s}'.format(amp)])
		hdu.header.remove('DATASEC{:s}'.format(amp))
	
	return newarray

def get_files_names(directory,numlist):
	from filesTools import fsearch
	
	#return ["{:s}/WINLIGHT_{:08d}.fits".format(directory,num) for num in numlist]
	return [fsearch("*{:08d}.fits".format(num),directory,silent=True)[0] for num in numlist]

class DESIImage(HDUList):
	"""
	A class derived from HDUList class, with added methods to
	manipulate the DESI images.
	The input parameter is a fits file name or a run number, or a list of fits
	file name and/or run number. If a list of fits files is given, the mean (or median)
	of the images is computed.
	
	ex:
		A=DESIImage([1298,'WINLIGHT_00001299.fits','WINLIGHT_00001300.fits',1301,1302])
		A=DESIImage(range(1298,1303))
	"""

	def __init__(self, fitsFiles, masterbias = None, average_function='median', silent=False):
		#masterbias = '/home/samuel/DESI/PERFORMANCETESTS/DATA/MasterBias_Images1351to1389.fits'
		if type(fitsFiles) is not list:
			fitsFiles = [fitsFiles]
		
		if average_function not in ['median','mean']:
			sys.exit("Error in 'average_function' parameters. Accepted values are: 'median' (default) or 'mean'")
			
		try:
			for i,f in enumerate(fitsFiles):
				if type(f) is int:
					fitsFiles[i]=fsearch("*{:08d}.fits".format(f),silent=True)[0]
		except:
			sys.exit("Error opening fits file. You may check the path or file number.")
		
		if not silent: print("Reading image {}".format(fitsFiles[0]))
		HDUs = fits.open(fitsFiles[0])
		super(self.__class__, self).__init__(hdus=HDUs)
		self.writeto = HDUs.writeto # for some obscure reason, writeto doesn't work in I don't do that.
		for h in self:
			if h.name in ['CCDS1R','CCDS1B','CCDS1Z']:
				# change old naming convention to new one (CCDS1R to R1, CCDS1B to B1, CCDS1Z to Z1)
				# (concerns images before run 655)
				h.name = h.name[-1:-3:-1] 
		
		self.names=[h.name for h in self if h.name in ['R1','B1','Z1']] # get the names of fits extension containing CCD data

		if len(fitsFiles) > 1:
			self.coadd(fitsFiles, average_function, silent=silent)
			
		if masterbias is not None:
			if not silent: print("Subtracting Master bias")
			B=fits.open(masterbias)
			for name in self.names:
				self[name].data = np.float32(self[name].data)
				self[name].data -= B[name].data
			
		for name in self.names:
			#self[name].header =  change_sector_info(self[name].header) # made up DATASEC values (sept 2016)
			try:
				self[name].header.remove('BSCALE')
				self[name].header.remove('BZERO')
			except Exception:
				pass
			self[name].header['ORIGIN']='OHP / S. Ronayette', 'Custom made for visualization purpose'
			self[name].data = extract(self[name], silent=silent)


	def subtract(self, theInput):
		"""
		subtract an array, to self['CCD*'].data (typically, a dark)
		
		theInput is a DESIImage object
		"""
		
		for name in self.names:
			self[name].data -= theInput[name].data
         
	#def show(self, x0=None, y0=None, w=None, h=None, name='R1'):
		#viewer = qla(self[name].data, x0, y0, w, h)
        
	def coadd(self, files_list, function='median', silent=False):
		"""
		Compute the median or mean of all the data array contained in the files given
		by files_list, and put the result in self.data
		"""

		dataDict={}
		for name in self.names:
			dataDict[name]=np.zeros(self[name].shape+(len(files_list),),dtype=self[name].data.dtype)
			dataDict[name][:,:,0] = self[name].data
			
		for k,f in enumerate(files_list[1:]):
			if not silent: print("Reading image {}".format(f))
			HDUs = fits.open(f)
			for name in self.names:
				dataDict[name][:,:,k+1] = HDUs[name].data

		if function.lower()=='median':
			if not silent: print("computing median...")
			for name in self.names:
				self[name].data = np.median(dataDict[name],axis=2)
				
		if function.lower()=='mean':
			if not silent: print("computing mean...")
			for name in self.names:
				self[name].data = np.mean(dataDict[name],axis=2)

class ImagesInfo():
	"""
	Object containing a few informations from the headers
	+ method to print it in a nice readable form.
	
	S. Ronayette, 06/01/2017
	"""
	def __init__(self, fitsFiles, quiet=False):
		self.listinfo = self.getImageInfos(fitsFiles)
		if not quiet: self.nicePrint()

	def getImageInfos(self, files):
		"""
		Retrieve quickly useful information in files
		Put it in a list of dictonnaries
		Example of use, to get only the file names of the "winlight" exposures of 400 sec on fiber 10:
		
			ff = [i['fname'] for i in info if i['exptime']==400 and i['exptype']=='winlight' and i['fiber']==10]

		#Retrieve info for all files on mardesi, to build a DB (see DBQuery.py):
		import os
		import cPickle as pk
		import TOOLS.src.DESI_tools as desi 
		rootdir='/desidata/EMSpectrograph/PerformanceTestWL/RawData/'
		directories=[rootdir+'2017/'+d for d in os.listdir(rootdir+'2017/')]+[rootdir+'2016/'+d for d in os.listdir(rootdir+'2016/')]
		files_basename = ['WINLIGHT_{:08d}.fits'.format(i) for i in range(1,9000)]
		files = [d+'/'+f for d in directories for f in os.listdir(d) if f in files_basename]
		files.sort()
		info=desi.ImagesInfo(files,quiet=True)
		pk.dump(info.listinfo,open('EM1_db.dat','wb'))

		"""
		
		LEDS = {'LED1':'LED370','LED2':'LED465','LED3':'LED591','LED4':'LED631','LED5':'LED870','LED6':'LED940'}
		LAMPS = {'LAMP1':'W','LAMP2':'HgAr','LAMP3':'Ne','LAMP4':'Kr','LAMP5':'Cd'}
		timefmt = "%Y-%m-%dT%H:%M:%S.%f"
		
		# workaround for exposure between 432 and 575, for "ND" field (missing in PLC header)
		NDvsEXPNUM=[(432,'OPEN'),(433,'OPEN'),(434,'DARK'),(435,'DARK'),(436,'DARK'),(437,'DARK'),(438,'NE30B'),(439,'NE20B'),\
			(440,'DARK'),(441,'DARK'),(442,'DARK'),(443,'DARK'),(444,'DARK'),(445,'DARK'),(446,'DARK'),(447,'DARK'),(448,'DARK'),\
			(449,'DARK'),(450,'DARK'),(451,'DARK'),(452,'DARK'),(453,'DARK'),(454,'DARK'),(455,'DARK'),(456,'DARK'),(457,'DARK'),\
			(458,'DARK'),(459,'DARK'),(460,'DARK'),(461,'DARK'),(462,'DARK'),(463,'DARK'),(464,'DARK'),(465,'DARK'),(466,'DARK'),\
			(467,'UNKNOWN'),(470,'OPEN'),(474,'UNKNOWN'),(475,'UNKNOWN'),(476,'UNKNOWN'),(477,'UNKNOWN'),(478,'UNKNOWN'),(479,'UNKNOWN'),(481,'UNKNOWN'),\
			(499,'NE10B'),(500,'OPEN'),(501,'OPEN'),(502,'OPEN'),(503,'OPEN'),(504,'OPEN'),(505,'OPEN'),(506,'OPEN'),(509,'OPEN'),\
			(510,'OPEN'),(511,'NE05B'),(512,'OPEN'),(513,'OPEN'),(514,'OPEN'),(515,'OPEN'),(516,'OPEN'),(517,'OPEN'),(518,'NE30B'),\
			(519,'NE30B'),(520,'NE30B'),(521,'NE30B'),(522,'NE30B'),(523,'NE30B'),(555,'OPEN'),(556,'OPEN'),(557,'OPEN'),(558,'OPEN'),\
			(559,'OPEN'),(560,'OPEN'),(561,'OPEN'),(562,'OPEN'),(563,'OPEN'),(571,'OPEN'),(572,'NE05B'),(573,'OPEN'),(574,'OPEN'),(575,'OPEN')]
		expnum_fix = [a[0] for a in NDvsEXPNUM]
		nd_fix=[a[1] for a in NDvsEXPNUM]
		
		if len(files) !=0: files.sort()
		
		info = []
		for f in files:
			A = fits.open(f)
			H0 = A['PRIMARY'].header
			exptype = H0['OBSTYPE']
			exptime = H0['EXPREQ']
			expnum = H0['EXPNUM']
			#date = H0['DATE-OBS']
			date= datetime.strptime(H0['DATE-OBS'][0:-6],timefmt)

			try:
				H1= A['PLC'].header
				if expnum < 431:
					#before expnum 431, PLC header not mature, or just not there.
					A['IWantToRaiseTheError'] # only to throw the KeyError and jump to the "except" statement
					
				ledsON = [h for h in H1['LED*'] if H1['LED*'][h] == 'on']
				if len(ledsON) >= 1:
					leds = ",".join([LEDS[i] for i in ledsON])
				else:
					leds = 'n/a'
				
				lampsON = [h for h in H1['LAMP*'] if H1['LAMP*'][h] == 'on']
				if len(lampsON) >= 1:
					lamps = ",".join([LAMPS[i] for i in lampsON])
				else:
					lamps = 'n/a'
				
				filt = H1['FILTER']
				if expnum < 576:
					nd=nd_fix[expnum_fix.index(expnum)]
				else:
					nd = H1['ND']
					
				fiber = H1['FIBER']
				pd_ = H1['PD'].split(',')
				pd1 = pd_[0]
				pd2 = pd_[1]
				
				if fiber==999: fiber='all'
			except KeyError:
				filt = 'n/a'
				nd = 'n/a'
				leds = 'n/a'
				lamps = 'n/a'
				fiber = 'n/a'
				pd1 = 'n/a'
				pd2 = 'n/a'
				
			
			try:
				sp = A['SPECTCON1'].data
				sp['HARTL']
			except KeyError:
				# HARTL and HARTR not in SPECTCON data at beginning.
				# --> set the values manually for some images
				sp={}
				if expnum in range(0,606)+range(606,628,3)+range(634,656,3):
					sp['HARTL']=["OPEN"]
					sp['HARTR']=["OPEN"]
				elif expnum in range(607,628,3)+range(629,655,3):
					sp['HARTL']=["CLOSED"]
					sp['HARTR']=["OPEN"]
				elif expnum in range(608,628,3)+range(630,655,3):
					sp['HARTL']=["OPEN"]
					sp['HARTR']=["CLOSED"]
				elif expnum==631:
					sp['HARTL']=["CLOSED"]
					sp['HARTR']=["CLOSED"]
				else:
				# if we are here, it means that the SPECTCON1 extension in not present.
					sp['HARTL']=["n/a"]
					sp['HARTR']=["n/a"]
			info.append({'exptype':exptype,'exptime':exptime,'expnum':expnum,'date':date,'fname':f,\
						'filt':filt,'nd':nd,'lamps':lamps,'leds':leds,'fiber':fiber,'HARTL':sp['HARTL'][0],'HARTR':sp['HARTR'][0], 'pd1':pd1,'pd2':pd2})
				
		return info
	
	def nicePrint(self,*params):
		if len(params) == 0:
			# a few usefull parameters
			params = ['expnum','exptype','exptime','fiber','lamps','leds','filt','nd']
			
		col_width = [max([len(p)]+[len(str(i[p])) for i in self.listinfo])+3 for p in params]
		if 'date' in params: col_width[params.index('date')]=13

		print()
		#print("\n"+''.join(['-']*(sum(col_width)+3)))
		#for p,w in zip(params,col_width):
			#fmt = "{{:{:d}s}}".format(w)
			#print(fmt.format(p),end="")
		#print("\n"+''.join(['-']*(sum(col_width)+3)))

		for k,i in enumerate(self.listinfo):
			if k%70 == 0 :
				print("\n"+''.join(['-']*(sum(col_width)+3)))
				for p,w in zip(params,col_width):
					fmt = "{{:{:d}s}}".format(w)
					print(fmt.format(p),end="")
				print("\n"+''.join(['-']*(sum(col_width)+3)))

			for p,w in zip(params,col_width):
				fmt = "{{:{:d}s}}".format(w)
				if p=='date':
					print(fmt.format(str(i[p]).split()[0]),end="")
				else:
					print(fmt.format(str(i[p])),end="")
			print()

def Coord2Wave(fiber,y, channel,CALDATADIR='/home/samuel/DESI/PERFORMANCETESTS/CODE/TOOLS/database/calibration_data/20170118/'):
	"""
	Returns the wavelength of a spectral line at a given position in the field
	
	Inputs:
		- the fiber number (0 to 20)
		- the y coordinate in pixels (on preprocessed image - overscan pixels removed)
		- channel: a string giving the channel: R1, B1 or Z1.
	
	Outputs:
		- the wavelength en nm	
	"""

	files = [f for f in os.listdir(CALDATADIR) if f[-5:]=='.fits']
	calfile = [f for f in files if f.find(channel.lower())!=-1][0]

#	CAL = fits.open(CALDATADIR+'psf-{:s}.fits'.format(channel.lower()))
	CAL = fits.open(calfile)
	
	
	wavemin=CAL["YTRACE"].header["WAVEMIN"]
	wavemax=CAL["YTRACE"].header["WAVEMAX"] 
	
	ycoef=CAL["YTRACE"].data
	
	#EM1 test stand. No fiber n°2
	if fiber==2: return None
	if fiber>2: fiber = fiber-1
	
	waves = np.linspace(wavemin,wavemax,5000)
	Ys=legval(2.*(waves-wavemin)/(wavemax-wavemin)-1.,ycoef[fiber]) 
	
	return np.interp(y,Ys,waves)/10.0

def Wave2Coord(fiber, wave, channel, CALDATADIR='/home/samuel/DESI/PERFORMANCETESTS/CODE/TOOLS/database/calibration_data/20170118/'):
	"""
	Returns the position in field of a spectral line
	
	Inputs:
		- the fiber number (0 to 20)
		- the wavelength of the spectral line, in nm
		- channel: a string giving the channel: R1, B1 or Z1.
	
	Outputs:
		- the x,y coordinate in pixels (on preprocessed image - overscan pixels removed)
	"""
	
	files = [f for f in os.listdir(CALDATADIR) if f[-5:]=='.fits']
	calfile = [CALDATADIR+f for f in files if f.find(channel.lower())!=-1][0]

#	CAL = fits.open(CALDATADIR+'psf-{:s}.fits'.format(channel.lower()))
	CAL = fits.open(calfile)
	
	wavemin=CAL["YTRACE"].header["WAVEMIN"]
	wavemax=CAL["YTRACE"].header["WAVEMAX"] 
	
	xcoef=CAL["XTRACE"].data
	ycoef=CAL["YTRACE"].data
	
	#EM1 test stand. No fiber n°2
	if fiber==2: return None
	if fiber>2: fiber = fiber-1
	
	x=legval(2.*(wave*10.0-wavemin)/(wavemax-wavemin)-1.,xcoef[fiber]) 
	y=legval(2.*(wave*10.0-wavemin)/(wavemax-wavemin)-1.,ycoef[fiber]) 
	
	return x,y
	
