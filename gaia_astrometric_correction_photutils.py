import sys
import os
import glob
import shutil
import subprocess

import numpy as np

import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.stats import sigma_clipped_stats

import photutils
from photutils import DAOStarFinder

""" A script that aligns an HST image of LLAMA galaxies to GAIA sources using DAO source-finding tools from PHOTUTILS
Options:
	- Fit all images or only those for which a solution doesn't currently exist.
	- Apply a cut to the visual magnitude of the GAIA catalog sources.
	- Use default DAO parameters, or choose a constant SNR threshold for all images
	- Adaptively change the SNR threshold to limit the number of image point sources. Useful if the field is crowded.
	- Change the crossmatch tolerance between GAIA and image point sources
	- Examine the matches visually and discard any that look suspicious
"""

OnlyIncomplete = False  # Only run on the datasets for which an astrometric solution has not been found yet
# The local table 'astrometric_correction.logtable' lists the object names and waveband which have been successfully corrected

# DAO source-finding configuration
DefaultThreshold = True  # If True, use defaults for each camera/filter as defined in the code below
UserThreshold    = 5.0   # If DefaultThreshold is False, use this as the DAO source-finding threshold SNR
# Parameters to adaptively change the DAO threshold
AdaptThreshold = True  # If set, then increase the threshold in multiplicative steps of 2 till the number of sources < AdaptiveNCut  
AdaptiveNCut   = 120    # The max number of sources that should be found by DAO. Used in Adaptive mode, and to provide source-finding advice

# GAIA catalog parameters
PhotCut = -99.0 # Apply this gmag cut to the GAIA catalog. If <= 0, no cut is applied.  
gaia_match_tolerance = 5.0 # Crossmatch tolerance between identified stars and GAIA stars, in arcseconds

# Set true to examine matched stars visually in cutouts
ExamineMatches = True

imagesroot = '/home/rosario/data/LLAMA/Imaging/HST/HST-LLAMA/image_collection/'

llama_table = Table.read('/home/rosario/Projects/LLAMA/llama_main_properties.fits',format='fits')

listroot='/home/rosario/Projects/LLAMA/HST_LLAMA/file_management/' # dir which has the table that lists HST-LLAMA image files
file_listing = Table.read(listroot+'final_working_images.tab',format='ascii',data_start=1) 

if os.path.isfile('astrometric_correction.logtable'):
	complete_table = Table.read('astrometric_correction.logtable',format='ascii.commented_header')

complete_objid    = []
complete_waveband = []

for ifile in range(len(file_listing)):
	
	plt.close('all')
	
#	Only used for testing purposes
#	if ifile != 5:
#		continue

	objname = file_listing['ObjID'][ifile].strip()   # Galaxy ID matching LLAMA naming scheme 
	lindex, = np.where(llama_table['id'] == objname)
	objcoord_sky = (llama_table['RA (deg)'][lindex[0]],llama_table['DEC (deg)'][lindex[0]]) # LLAMA coordinates for galaxy.
	# The galaxy coordinates are based on RC3 or similar surveys, and won't be accurate at the scale of the nucleus in HST data.
	# Used here to choose a set of stars for astrometric association within a few arcmin of the galaxy.
	
	#	Read the appropriate Gaia source catalog.
	#	Restrict to sources with gmag < 20.0 which have better astrometry (except NICMOS, which is source limited)
	gaiacatfile = '/home/rosario/data/LLAMA/Imaging/HST/HST-LLAMA/gaia_catalogs/'+objname+'_gaiacat.fits'	
	gaiacat = Table.read(gaiacatfile)
	if PhotCut > 0:
		index, = np.where(gaiacat['phot_g_mean_mag'] < PhotCut)
		gaiacat_sub = gaiacat[index]
	gaia_coords = SkyCoord(gaiacat['ra'],gaiacat['dec'])

	# Identify and read in the image, and initialise an output file in the correct directory
	srdir = imagesroot+objname+'/'
	srfilename = srdir+file_listing['Filename'][ifile].strip()+'.fits'
	srinst = file_listing['Instrument'][ifile].strip()
	exptime = file_listing['ExpTime'][ifile] 
	
	# 	Identify the waveband.
	filtname = file_listing['Filter'][ifile].strip()
	if ((filtname == 'F160W') | (filtname == 'F190N') | (filtname == 'F200N')): waveband = 'NIR'
	if ((filtname == 'F606W') | (filtname == 'F475W') | (filtname == 'F438W') | 
		(filtname == 'F547M') | (filtname == 'F550M') | (filtname == 'F555W') |
		(filtname == 'F814W')): waveband = 'optical'
	destfilename = srdir+waveband+'_registered.fits'

	if OnlyIncomplete:
		index, = np.where(np.core.defchararray.strip(complete_table['ObjID']) == objname)
		if (len(index) > 0) & (waveband in complete_table['WaveBand'][index]):
			continue   	

	# Copy file to this directory and rename.
	shutil.copy2(srfilename,'working_image.fits')
	
	print('Processing '+srinst+' '+waveband+' image for '+objname)

	# Open the file and get the header with WCS information, as well as the image
	hdulist = fits.open('working_image.fits')
	image  = hdulist[1].data
	whtmap = hdulist[2].data
	header = hdulist[1].header
	hdulist.close()
	imwcs = WCS(header)

	# Create an image mask that will be used in source detection.
	# The masking cuts depends on the instrument and processing level.
	immask = np.zeros(image.shape,dtype=bool)
	index = (image == 0) | (np.logical_not(np.isfinite(image))) | (whtmap == 0) | (np.logical_not(np.isfinite(whtmap)))
	immask[index] = True

	# Image scales and galaxy position in pixel space
	pix_coord = imwcs.all_world2pix(objcoord_sky[0],objcoord_sky[1],0)
	pixscale_matrix = imwcs.pixel_scale_matrix
	pixscale = np.sqrt(pixscale_matrix[0,0]**2 + pixscale_matrix[0,1]**2)*3600.0 # pixel scale in arcsec/pixel

	# Some basic stats of the image to get the approximate sky level for point source extraction
	mean, median, std = sigma_clipped_stats(image, mask=immask, sigma=3.0, iters=5)
	# subtract the median 'sky' from the image. This is not necessarily the sky, it can be determined by the galaxy light
	#    in a small field image
	image = image-median 


	# Find point sources brighter than  the sigma-clipped standard deviation of the image.
	# The galaxies are large and have strong gradients. I experimented with subtracting off a smooth version
	# of the galaxy based on elliptical aperture photometry, but the residuals are too strong. Instead, I just
	# mask out any sources found near the centre of the galaxy.

	# Use DAOfind parameters optimised for the camera:
	#  daofwhm is the FWHM of a point source in the image, in pixels
	#  daothreshold is the minimum number of sigma above which to consider the source as a detection.
	if (srinst == 'WFC3') & (waveband == 'optical'):
		daofwhm = 6.0
		daothreshold = 50.0
	elif (srinst == 'WFC3') & (waveband == 'NIR'):
		daofwhm = 2.5
		daothreshold = 5.0
	elif (srinst == 'WFPC2') & (waveband == 'optical'):
		daofwhm = 3.5
		daothreshold = 50.0
	elif (srinst == 'ACS') & (waveband == 'optical'):
		daofwhm = 5.0
		daothreshold = 50.0
	elif (srinst == 'NICMOS') & (waveband == 'NIR'):
		daofwhm = 2.5
		daothreshold = 10.0
	else:
		daofwhm = 3.5
		daothreshold = 50.0

	# Update the SNR threshold if not using the defaults
	if not DefaultThreshold:
		daothreshold = UserThreshold

	# Run DAO. If Adaptive mode and the number of sources is too high, change the threshold till the number of sources is below AdaptiveNCut
	
	DAOcomplete_flag = False  # When the DAO source finding is completed, this will be set to true.
	thresholdscale = 1.0  #  changed if using Adaptive threshold mode
	
	while DAOcomplete_flag == False:


		daofind = DAOStarFinder(fwhm=daofwhm, threshold=thresholdscale*daothreshold*std)
		sources = daofind(image)		

		if len(sources) > 0:

			source_selectflag = np.zeros(len(sources),dtype=bool)

			# Remove stars that lie near artifacts or are saturated, or which lie within 20" of the galaxy.
			# Except for NICMOS, where sources are limited for matching
			for i in range(len(sources)):
				xstar = sources['xcentroid'][i]
				ystar = sources['ycentroid'][i]
				# Exclude stars near the image edges
				if (xstar > 0.1*image.shape[1]) & (xstar < 0.9*image.shape[1]) & \
	   			   (ystar > 0.1*image.shape[0]) & (ystar < 0.9*image.shape[0]):
					# Exclude stars which lie within 1 pixels of masked pixels 
					substamp = immask[int(np.floor(ystar-1)):int(np.ceil(ystar+1)),\
		    			              int(np.floor(xstar-1)):int(np.ceil(xstar+1))]
					if np.count_nonzero(substamp) == 0:
						# Exclude stars within 20" of the galaxy nucleus
						dist = np.sqrt((pix_coord[0]-sources['xcentroid'][i])**2 + (pix_coord[1]-sources['ycentroid'][i])**2)*pixscale
						if (dist > 20.0) | (srinst == 'NICMOS'):
							source_selectflag[i] = True

			sources = sources[source_selectflag]
				
			if AdaptThreshold & (len(sources) > AdaptiveNCut):
				thresholdscale *= 2.0
				print('Adapting the threshold to '+str(np.int(thresholdscale*daothreshold)))
			else:
				DAOcomplete_flag = True	

		else:
			print('No stars identified in image')
			DAOcomplete_flag = True
			continue

	
	if len(sources) == 0:
		continue
	if len(sources) > AdaptiveNCut:
		print('Large number of point sources identified. Consider rerunning with a more conservative threshold.')

	# Match GAIA stars to point sources in image
	temp_xy = imwcs.all_pix2world(sources['xcentroid'],sources['ycentroid'],0)
	star_coords = SkyCoord(temp_xy[0],temp_xy[1],unit=u.degree)
	match,dist2d,dist3d = star_coords.match_to_catalog_sky(gaia_coords)
	gindex, = np.where(dist2d.arcsec < gaia_match_tolerance)
	

	if len(gindex) > 0:
		print('{0:>3d} matches between GAIA sources and stars in image'.format(len(gindex)))

		# Preliminary matches
		matched_star_coords = star_coords[gindex]		
		matched_gaia_coords = gaia_coords[match[gindex]]		

		# Preliminary astrometric offset - currently only XY shift
		raoffset  = np.median(matched_gaia_coords.ra  - matched_star_coords.ra)
		decoffset = np.median(matched_gaia_coords.dec - matched_star_coords.dec)

		if ExamineMatches:
			ncols = 10
			nrows = np.int(np.ceil(len(gindex)/ncols))
			dx = (0.98-0.02)/ncols	
			dy = (0.98-0.02)/nrows	
			
			stampfig = plt.figure(figsize=(ncols*1.0,nrows*1.0))
			
			for istar in range(len(gindex)):
				temp_coord = SkyCoord(matched_gaia_coords.ra[istar]-raoffset,matched_gaia_coords.dec[istar]-decoffset)
				cutout = Cutout2D(image,temp_coord,wcs=imwcs,size=1.0*u.arcsec)
				
				ix = istar % ncols
				iy = np.int(istar/ncols)
				stampax = stampfig.add_axes([0.02+ix*dx,0.98-(iy+1)*dy,dx,dy])
				stampax.imshow(cutout.data,origin='lower',interpolation='nearest',vmin=-3.0*std,vmax=np.max(cutout.data),cmap='jet')
				stampax.tick_params(axis='both',top=False,bottom=False,left=False,right=False,\
									labeltop=False,labelbottom=False,labelleft=False,labelright=False)
				stampfig.text(0.02+(ix+0.15)*dx,0.98-(iy+0.95)*dy,str(istar),ha='center',va='bottom',size='small',color='yellow')
			plt.show()
			
			ch = input('Continue? (y/n): ').strip()
			if ch == 'n':
				print('Skipping the rest of the alignment process. No outputs will be written.')
				sys.exit()
			
			reject_string = input('Comma-separated list of stars to remove (leave empty to keep all): ').strip()
			if len(reject_string) != 0:
				reject_array = np.array([np.int(irej) for irej in reject_string.split(',')])
				accepted_indices = np.setdiff1d(np.arange(len(gindex)),reject_array,assume_unique=True)
				gindex = gindex[accepted_indices]

			# Reapply star selection, after manual cuts
			matched_star_coords = star_coords[gindex]		
			matched_gaia_coords = gaia_coords[match[gindex]]		

		outcoo = sources['xcentroid','ycentroid'][gindex]
		outcoo['xcentroid'] += 1 # Convert to DS9 standard
		outcoo['ycentroid'] += 1 # Convert to DS9 standard
		outcoo.write(objname+'_'+waveband+'_dao.coo',format='ascii',overwrite=True)

		raoffset  = np.median(matched_gaia_coords.ra  - matched_star_coords.ra)
		decoffset = np.median(matched_gaia_coords.dec - matched_star_coords.dec)
		print('RAoff, DECoff = {0:10.7f}, {1:10.7f}'.format(raoffset.arcsec*np.cos((objcoord_sky[1]/180.0)*np.pi),decoffset.arcsec))
		
		header['CRVAL1'] = header['CRVAL1'] + raoffset.value
		header['CRVAL2'] = header['CRVAL2'] + decoffset.value
		fits.update('working_image.fits',image,header,1)
		
		complete_objid.append(objname)
		complete_waveband.append(waveband)
		
	else:
		print('Not enough matches to update the WCS')
		continue
	
	shutil.copy2('working_image.fits',destfilename)

Table([complete_objid,complete_waveband],names=('ObjID','WaveBand')).write(\
	  'completed_astrometric_correction.txt',format='ascii.commented_header',overwrite=True)
	
