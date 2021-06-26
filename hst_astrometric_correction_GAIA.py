import sys
import os
import glob
import shutil
import subprocess

import numpy as np

import scipy.optimize

import matplotlib
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ImageNormalize, LogStretch

import photutils
from photutils import DAOStarFinder


#**********************************************************************************************#	

def rotate(degs):
    """Return a rotation matrix for counterclockwise rotation by ``deg`` degrees."""
    rads = np.radians(degs)
    s = np.sin(rads)
    c = np.cos(rads)
    return np.array([[c, -s],
                     [s,  c]])

class WcsModel(object):
	
	""" Modified from code found at http://www.astropython.org/snippets/fix-the-wcs-for-a-fits-image-file62/
	
	"""
	
	def __init__(self, wcs, sky, pix0):
		self.wcs = wcs.deepcopy()   # Image WCS transformation object
		self.sky = sky   # Reference (correct) source positions in RA, Dec
		self.pix0 = pix0.flatten()  # Source pixel positions
		# Copy the original WCS CRVAL and CD values
		self.crval = self.wcs.wcs.crval.copy()
		if hasattr(self.wcs.wcs, 'cd'):
			self.cd = self.wcs.wcs.cd.copy()
		else:
			print('I can only work with WCS information in the CD format')

	def calc_pix(self, pars):
		"""For the given d_ra, d_dec, and d_theta pars, update the WCS
		transformation and calculate the new pixel coordinates for each
		reference source position.
		
		"""
		d_ra, d_dec, d_theta = pars
		self.wcs.wcs.crval = self.crval + np.array([d_ra, d_dec]) / 3600.0
		self.wcs.wcs.cd = np.dot(rotate(d_theta), self.cd)
		pix = self.wcs.wcs_world2pix(self.sky, 1)
		return pix.flatten()

	def calc_resid2(self, pars):
		"""Return the squared sum of the residual difference between the
		original pixel coordinates and the new pixel coords (given offset
		specified in ``pars``)

		This gets called by the scipy.optimize.fmin function.
		"""
		pix = self.calc_pix(pars)
		return np.sum((self.pix0 - pix)**2)  # assumes uniform errors


def match_wcs(wcs_img, sky_img, sky_ref):
	"""Adjust ``wcs_img`` (CRVAL{1,2} and CD{1,2}_{1,2}) using a rotation and linear
	   offset so that ``coords_img`` matches ``coords_ref``.
		
	   param sky_img: list of (world_x, world_y) [aka RA, Dec] coords in input image
	   param sky_ref: list of reference (world_x, world_y) coords to match
	   param wcs_img: pywcs WCS object for input image
		
		:returns: d_ra, d_dec, d_theta
	"""
	pix_img = wcs_img.wcs_world2pix(sky_img, 1)
	wcsmodel = WcsModel(wcs_img, sky_ref, pix_img)
	y = np.array(pix_img).flatten()
	
	x0 = np.array([0.0, 0.0, 0.0])
	d_ra, d_dec, d_theta = scipy.optimize.fmin(wcsmodel.calc_resid2, x0)
	print('Best fit values (deltaRA (arcsec), deltaDEC (arcsec), rotation (degrees)): {0:6.4f}, {1:6.4f}, {2:7.3f}'.format( \
		   d_ra*np.cos(np.radians(wcs_img.wcs.crval[1])), d_dec, d_theta))
	
	return wcsmodel.wcs
    
#**********************************************************************************************#	

class HSTAstrometricCorrectionGAIA(object):

	""" A class that reads in an HST image, processes it (WCS information, masking), creates a GAIA catalog,
		finds stars in the image, matches stars between GAIA and image including visual inspection,
		solves for an updated WCS including shift and rotation, and updates the image or creates a copy of the
		image with updated photometry
	"""

	#-------------------------------------------	
	
	def __init__(self,imagefilename,source_position=np.array([-99.0,-99.0]),filterband=None,instrument=None):		
		""" 
			imagefilename: full path to the file containing the HST image
			source_position: a two-element array containing the RA, Dec (J2000) of the primary source in decimal degrees
			filterband: the name of the HST filter for the image, as a string
		"""

		self.imfilename = imagefilename
		hdulist = fits.open(imagefilename)
		# HST images are assumed to have the data in the first HDU, a weightmap in the second HDU, 
		# and astrometric information in the header of the first HDU.
		self.image  = hdulist[1].data
		self.imshape = self.image.shape
		self.whtmap = hdulist[2].data
		self.header = hdulist[1].header
		hdulist.close()
		self.imwcs = WCS(self.header)
		if filterband != None:
			self.filterband = filterband
			if ((filterband == 'F160W') | (filterband == 'F190N') | (filterband == 'F200N')): self.waveband = 'NIR'
			if ((filterband == 'F606W') | (filterband == 'F475W') | (filterband == 'F438W') | 
				(filterband == 'F547M') | (filterband == 'F550M') | (filterband == 'F555W') |
				(filterband == 'F814W')): self.waveband = 'optical'
		else:
			self.filterband = 'Generic'
			self.waveband   = 'Generic'
		if instrument != None:
			self.instrument = instrument
		else:
			self.instrument = 'Generic'


		# Image pixel scale in arcsec/pixel
		pixscale_matrix = self.imwcs.pixel_scale_matrix
		self.pixscale = np.sqrt(pixscale_matrix[0,0]**2 + pixscale_matrix[0,1]**2)*3600.0  

		# Create an basic image mask that will be used in source detection.
		# Mask out bad data based on pixel values or weightmap values.
		self.immask = np.zeros(self.image.shape,dtype=bool)
		index = (self.image == 0) | (np.logical_not(np.isfinite(self.image))) | (self.whtmap == 0) | (np.logical_not(np.isfinite(self.whtmap)))
		self.immask[index] = True

		# Some basic stats of the image to get the approximate sky level for point source extraction
		mean, self.median, self.std = sigma_clipped_stats(self.image, mask=self.immask, sigma=3.0, maxiters=5)
		
		# If a specific source position is provided, determine its coordinates in the image pixels
		if (len(source_position) == 2):
			
			if (source_position[0] >= 0.0) & (source_position[1] >= -90.0):
				self.source_coord = SkyCoord(source_position[0],source_position[1],unit=u.deg)
				self.sourcecoord_pix = self.imwcs.all_world2pix(self.source_coord.ra,self.source_coord.dec,0)
								
				try:
					cutout = Cutout2D(self.image,self.source_coord,wcs=self.imwcs,size=10.0*u.arcsec)
					self.immax = np.max(cutout.data)
				except:
					self.immax = np.max(self.image)
		else:
			self.immax = np.max(self.image)	
					
	#-------------------------------------------	

	def get_gaia_catalog(self,gaiacatfilename=None, magcut=-99.0):		
		""" Generate or read in a GAIA catalog file and store internally
			
			gaiacatfilename: full path to an existing GAIA catalog for this image, in FITS table format
		"""
		
		if gaiacatfilename != None:

			self.gaiacat = Table.read(gaiacatfilename,format='fits')
	
			if magcut > 0:
				index, = np.where(self.gaiacat['phot_g_mean_mag'] < magcut)
				self.gaiacat = self.gaiacat[index]

			self.gaia_coords = SkyCoord(self.gaiacat['ra'],self.gaiacat['dec'])

		else:

			raise ValueError('Currently no support for programmatic construction of the GAIA catalog. This is coming!')

	#-------------------------------------------	

	def find_stars_in_image(self,mask_source=True,mask_radius=20.0,relax_datamask=False,edge_maskfraction=0.1,\
							default_threshold=True, daofwhm=3.5, daothreshold=50.0,\
							adaptive_threshold=False, adaptive_Ncut=80):
		""" Use Photutils DAOFIND to identify possible stars in the unmasked part of the image
				daofwhm is the FWHM of a point source in the image, in pixels
				daothreshold is the minimum number of sigma above which to consider the source as a detection.

			mask_source: If true, a region around the source position with the mask_radius is excluded from the
						 star search
			relax_datamask: If true, masking of bad/saturated pixels is disabled.
			edge_maskfraction: The size of the masking strip around the image edge, as a fraction of the image size. 
			if default_threshold is False, the user-supplied daothreshold and daofwhm is used.
			if adaptive_threshold is True, the threshold is raised in steps of a factor of 2 till the number of stars found
						drops below adaptive_Ncut. This is useful if the data is noisy or has large number of residual CRs.
		"""

		# Find point sources brighter than  the sigma-clipped standard deviation of the image.
		# The galaxies are large and have strong gradients. I experimented with subtracting off a smooth version
		# of the galaxy based on elliptical aperture photometry, but the residuals are too strong. Instead, I just
		# mask out any sources found near the centre of the galaxy.
			
		
		if default_threshold:
			if (self.instrument == 'Generic') | (self.waveband == 'Generic'):
				daofwhm = 3.5
				daothreshold = 50.0
			else:
				# Use DAOfind parameters optimised for the camera, based on trial/error
				if (self.instrument == 'WFC3') & (self.waveband == 'optical'):
					daofwhm = 6.0
					daothreshold = 50.0
				elif (self.instrument == 'WFC3') & (self.waveband == 'NIR'):
					daofwhm = 2.5
					daothreshold = 5.0
				elif (self.instrument == 'WFPC2') & (self.waveband == 'optical'):
					daofwhm = 3.5
					daothreshold = 50.0
				elif (self.instrument == 'ACS') & (self.waveband == 'optical'):
					daofwhm = 5.0
					daothreshold = 50.0
				elif (self.instrument == 'NICMOS') & (self.waveband == 'NIR'):
					daofwhm = 2.5
					daothreshold = 10.0

		# Run DAO star finder.
		DAOcomplete_flag = False  # When the DAO source finding is completed, this will be set to true.
		thresholdscale = 1.0  #  changed if using Adaptive threshold mode
	
		while DAOcomplete_flag == False:

			daofind = DAOStarFinder(fwhm=daofwhm, threshold=thresholdscale*daothreshold*self.std)
			sources = daofind(self.image)		

			if len(sources) > 0:

				source_selectflag = np.zeros(len(sources),dtype=bool)

				# Remove stars that lie near artifacts or are saturated, 
				#   or which lie within mask_radius of the galaxy if mask_source is set.
				for i in range(len(sources)):
					xstar = sources['xcentroid'][i]
					ystar = sources['ycentroid'][i]
					# Exclude stars near the image edges
					if (xstar > edge_maskfraction*self.imshape[1]) & (xstar < (1-edge_maskfraction)*self.imshape[1]) & \
					   (ystar > edge_maskfraction*self.imshape[0]) & (ystar < (1-edge_maskfraction)*self.imshape[0]):

						# Exclude stars which lie within 1 pixels of masked pixels 
						substamp = self.immask[int(np.floor(ystar-1)):int(np.ceil(ystar+1)),\
										       int(np.floor(xstar-1)):int(np.ceil(xstar+1))]
						if (np.count_nonzero(substamp) == 0): 
							if mask_source:
								# Exclude stars within mask_radius of the galaxy nucleus
								dist = np.sqrt((self.sourcecoord_pix[0]-sources['xcentroid'][i])**2 + \
											   (self.sourcecoord_pix[1]-sources['ycentroid'][i])**2)*self.pixscale
								if dist > mask_radius:
									source_selectflag[i] = True
							else:
								source_selectflag[i] = True 

				sources = sources[source_selectflag]
				
				if adaptive_threshold & (len(sources) > adaptive_Ncut):
					thresholdscale *= 2.0
					print('Adapting the threshold to '+str(np.int(thresholdscale*daothreshold)))
				else:
					DAOcomplete_flag = True	

			else:
				DAOcomplete_flag = True

	
		if len(sources) == 0:
				print('No stars identified in image')
				self.found_stars = 0
				return False
		else:
			self.found_stars = sources

		if len(sources) > adaptive_Ncut:
			print('Large number of point sources identified. Consider rerunning with a more conservative threshold.')
		
		return True

	#-------------------------------------------	

	def examine_stars_in_image(self):
		""" Plot the stars on the image for visual examination. Allow manual removal of stars.
		"""

		plt.close('all')

		imfig = plt.figure(figsize=(7,7))
		imax = imfig.add_subplot(111)
		imnorm = ImageNormalize(self.image, vmin=-2.0*self.std, vmax=self.immax,stretch=LogStretch())
		imax.imshow(self.image,origin='lower',norm=imnorm,cmap='gray',interpolation='nearest')
		imax.xaxis.set_visible(False)
		imax.yaxis.set_visible(False)

		stars_image_xy     = list(zip(self.found_stars['xcentroid'],self.found_stars['ycentroid']))
		for istar in range(len(stars_image_xy)):
			imax.text(stars_image_xy[istar][0],stars_image_xy[istar][1],str(istar),
					  ha='center',va='center',size='small',color='orange')
		if not matplotlib.is_interactive():
			plt.show()

		reject_string = input('Comma-separated list of stars to remove (leave empty to keep all): ').strip()
		if len(reject_string) != 0:
			reject_array = np.array([np.int(irej) for irej in reject_string.split(',')])
			accepted_indices = np.setdiff1d(np.arange(len(stars_image_xy)),reject_array,assume_unique=True)
			self.found_stars = self.found_stars[accepted_indices]

	#-------------------------------------------	

	def pick_reference_stars(self, gaia_match_tolerance=1.0, interactive=True, no_rotation=False):		
		""" Cross-match GAIA and found stars to identify a subset of astrometric reference stars.
			
			Matches are made within a gaia_match_tolerance in arcsec, default is 5 arcsec.
			If interactive is set, the user can exclude stars that have issues after a visual inspection.
			if no_rotation is set, only a shift correction is calculated.
		"""
		
		star_pos = self.imwcs.all_pix2world(self.found_stars['xcentroid'],self.found_stars['ycentroid'],0)
		star_coords = SkyCoord(star_pos[0],star_pos[1],unit=u.degree)
		indexstars, indexgaia, dist2d, dist3d = self.gaia_coords.search_around_sky(star_coords,gaia_match_tolerance*u.arcsec)
#		match,dist2d,dist3d = star_coords.match_to_catalog_sky(self.gaia_coords)
#		gindex, = np.where(dist2d.arcsec < gaia_match_tolerance)

		# Run through gaia stars that are matched. Remove all duplicates, multiple stars matched to a single GAIA star
		gaiamatch = np.full(len(star_coords),-1,dtype='i4')
		for ig in np.unique(indexgaia):
			index, = np.where(indexgaia == ig)
			if len(index) == 1:
				# Store the associations only for single matches
				gaiamatch[indexstars[index[0]]] = ig
		gindex, = np.where(gaiamatch >= 0)

		if len(gindex) > 0:
			print('{0:>3d} initial matches between GAIA sources and stars in image'.format(len(gindex)))

			# Preliminary astrometric correction
			matched_image_xy     = list(zip(self.found_stars['xcentroid'][gindex],self.found_stars['ycentroid'][gindex]))
			matched_image_coords = [(coord.ra.value,coord.dec.value) for coord in star_coords[gindex]]
			matched_ref_coords   = [(coord.ra.value,coord.dec.value) for coord in self.gaia_coords[gaiamatch[gindex]]]

			if (no_rotation == False):
				new_wcs = match_wcs(self.imwcs, matched_image_coords, matched_ref_coords)
			else:
				ra_shift = np.median(np.array([coord[0] for coord in matched_ref_coords]) - np.array([coord[0] for coord in matched_image_coords]))
				dec_shift = np.median(np.array([coord[1] for coord in matched_ref_coords]) - np.array([coord[1] for coord in matched_image_coords]))
				print('Calculated values (deltaRA (arcsec), deltaDEC (arcsec)): {0:6.4f}, {1:6.4f}'.format( \
					   ra_shift*np.cos(np.radians(self.imwcs.wcs.crval[1]))*3600.0, dec_shift*3600.0))
				new_wcs = self.imwcs.deepcopy()
				new_wcs.wcs.crval[0] += ra_shift
				new_wcs.wcs.crval[1] += dec_shift


			if interactive:

				plt.close('all')

				imfig = plt.figure(figsize=(7,7))
				imax = imfig.add_subplot(111)
				imnorm = ImageNormalize(self.image, vmin=-2.0*self.std, vmax=self.immax,stretch=LogStretch())
				imax.imshow(self.image,origin='lower',norm=imnorm,cmap='gray',interpolation='nearest')
				imax.xaxis.set_visible(False)
				imax.yaxis.set_visible(False)
				for istar in range(len(gindex)):
					imax.text(matched_image_xy[istar][0],matched_image_xy[istar][1],str(istar),
							  ha='center',va='center',size='small',color='orange')
				if not matplotlib.is_interactive():
					plt.show()

				ncols = 10
				nrows = np.int(np.ceil(len(gindex)/ncols))
				dx = (0.98-0.02)/ncols	
				dy = (0.98-0.02)/nrows	
			
				stampfig = plt.figure(figsize=(ncols*0.7,nrows*0.7))
			
				for istar in range(len(gindex)):
					temp_coord = SkyCoord(matched_ref_coords[istar][0],matched_ref_coords[istar][1],unit=u.deg)
					cutout = Cutout2D(self.image,temp_coord,wcs=new_wcs,size=1.0*u.arcsec)
				
					ix = istar % ncols
					iy = np.int(istar/ncols)
					stampax = stampfig.add_axes([0.02+ix*dx,0.98-(iy+1)*dy,dx,dy])
					stampax.imshow(cutout.data,origin='lower',interpolation='nearest',vmin=-3.0*self.std,vmax=np.max(cutout.data),cmap='coolwarm')
					stampax.tick_params(axis='both',top=False,bottom=False,left=False,right=False,\
										labeltop=False,labelbottom=False,labelleft=False,labelright=False)
					stampfig.text(0.02+(ix+0.15)*dx,0.98-(iy+0.95)*dy,str(istar),ha='center',va='bottom',size='small',color='yellow')
				if not matplotlib.is_interactive():
					plt.show()
			
				ch = input('Continue? (y/n): ').strip()
				if ch == 'n':
					print('Skipping the rest of the alignment process. No outputs will be written.')
					return False	
							
				reject_string = input('Comma-separated list of stars to remove (leave empty to keep all): ').strip()
				if len(reject_string) != 0:
					reject_array = np.array([np.int(irej) for irej in reject_string.split(',')])
					accepted_indices = np.setdiff1d(np.arange(len(gindex)),reject_array,assume_unique=True)
					gindex = gindex[accepted_indices]
		
			self.matched_image_coords = [(coord.ra.value,coord.dec.value) for coord in star_coords[gindex]]
			self.matched_ref_coords   = [(coord.ra.value,coord.dec.value) for coord in self.gaia_coords[gaiamatch[gindex]]]
			return True
		else:	

			print('No matches between stars on image and GAIA catalog entries.')
			return False

	#-------------------------------------------	

	def transform_wcs(self,outfile=None,no_rotation=False):		
		""" Perform the final update on the WCS and write out the final image
			
			outfile: full path name of a file to contain the updated image. If None, the original image is overwritten.
			if no_rotation is set, only a shift correction is calculated.
		"""
		if (no_rotation == False):
			new_wcs = match_wcs(self.imwcs, self.matched_image_coords, self.matched_ref_coords)
		else:
			ra_shift = np.median(np.array([coord[0] for coord in self.matched_ref_coords]) -\
							     np.array([coord[0] for coord in self.matched_image_coords]))
			dec_shift = np.median(np.array([coord[1] for coord in self.matched_ref_coords]) -\
								  np.array([coord[1] for coord in self.matched_image_coords]))
			print('Calculated values (deltaRA (arcsec), deltaDEC (arcsec)): {0:6.4f}, {1:6.4f}'.format(\
					   ra_shift*np.cos(np.radians(self.imwcs.wcs.crval[1]))*3600.0, dec_shift*3600.0))
			new_wcs = self.imwcs.deepcopy()
			new_wcs.wcs.crval[0] += ra_shift
			new_wcs.wcs.crval[1] += dec_shift

		self.imwcs = new_wcs
		self.header['CRVAL1'] = self.imwcs.wcs.crval[0]
		self.header['CRVAL2'] = self.imwcs.wcs.crval[1]
		self.header['CD1_1'] = self.imwcs.wcs.cd[0,0]
		self.header['CD1_2'] = self.imwcs.wcs.cd[0,1]
		self.header['CD2_1'] = self.imwcs.wcs.cd[1,0]
		self.header['CD2_2'] = self.imwcs.wcs.cd[1,1]
		
		if outfile != None:
			fits.writeto(outfile,self.image,header=self.header,overwrite=True) # Primary HDU is the image with the header and WCS
			fits.append(outfile,self.whtmap) # HDU=1 is the weight map with basic header
		else:
			fits.update(self.imfilename,self.image,self.header,1)
			
