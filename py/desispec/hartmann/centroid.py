# -*- coding: utf-8 -*-
"""
@author: sronayette

"""

import numpy as np

def centroid(image, hw=5, threshold=0.05, quiet=False, XY0_estim=None):
	"""
	Floating centroid based on A Gibberd's implemetation of the STSci floating window algorithm.

	inputs:
	-------
		image : a 2D numpy array of floats
		hw: half-width of a box in which the centroid is computed. Default: 5 pixels
		threshold = value under which the computation stops. Default: 0.05 pixels.
		XY0_estim= (x,y) tuple giving the estimate of the location of the centroide. If none, the max of the image is used
		
	outputs:
	--------
		the (x,y) coordinates of the centroid

	"""
	sz = image.shape
	
	if XY0_estim is None:
		imax = image.argmax() % sz[1]
		jmax = image.argmax() / sz[1]
	else:
		pass #TBW
    
	if any([imax < hw, jmax<hw, imax>=sz[1]-hw, jmax>=sz[0]-hw]):
		hw = np.min([imax, jmax, sz[1]-imax-1, sz[0]-jmax-1])
		if hw < 2:
			if not quiet: print("ERROR in centroid.py: source too close to the edge. Cannot compute centroid")
			return -1,-1
		else:
			if not quiet: print("WARNING in centroid.py: box half-width too large, reducing to {} pixels".format(hw))

    
	# Weight the pixels in square aperture with half width hw
	box = image[jmax-hw:jmax+hw+1,imax-hw:imax+hw+1]
	total = box.sum()
	YY,XX = np.mgrid[jmax-hw:jmax+hw+1,imax-hw:imax+hw+1]
	Xcen = (XX*box).sum() / total
	Ycen = (YY*box).sum() / total

	# Adjusts Xcen/Ycen for box Coordinates (with zero in corner of pixel, instead of middle)
	Xcen=Xcen-imax+hw+0.5
	Ycen=Ycen-jmax+hw+0.5
		
	box = image[jmax-hw:jmax+hw+1,imax-hw:imax+hw+1]
	# Corrected half width adds 0.5 a pixel because hw is a measure of pixels either side of centre
	hwc = hw+0.5 

	YY,XX = np.mgrid[0:2*hwc,0:2*hwc]

	r = threshold + 1

	while(r > threshold):

		Xcen_old, Ycen_old = Xcen, Ycen

		arr1 = 1-(Xcen-hwc-XX)
		arr2 = Xcen+hwc-XX
		xweight=np.where(XX<Xcen,arr1,arr2)
		xweight=np.clip(xweight,0,1)

		arr1 = 1-(Ycen-hwc-YY)
		arr2 = Ycen+hwc-YY
		yweight=np.where(YY<Ycen,arr1,arr2)
		yweight=np.clip(yweight,0,1)

		# Combine weighting and Calculates new centroid in X/Y directions
		weight = xweight * yweight
		Tot = (box*weight).sum()
		Xcen = ((XX+1)*box*weight).sum() / Tot - 0.5
		Ycen = ((YY+1)*box*weight).sum() / Tot - 0.5

		# Converts error to radial
		r = np.sqrt((Xcen-Xcen_old)**2+(Ycen-Ycen_old)**2)

	# back to coordinates in original image
	Xcen = Xcen+imax-hw-0.5
	Ycen = Ycen+jmax-hw-0.5

	return (Xcen, Ycen)
