import sys
import os
import glob
import shutil
import subprocess

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from astropy.table import Table

import hst_astrometric_correction_GAIA


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
# The local table 'completed_astrometric_correction.txt' lists the object names and waveband which have been successfully corrected

# DAO source-finding configuration
DefaultThreshold = True  # If True, use defaults for each camera/filter as defined in the code below
UserThreshold    = 5.0   # If DefaultThreshold is False, use this as the DAO source-finding threshold SNR
# Parameters to adaptively change the DAO threshold
AdaptThreshold = True  # If set, then increase the threshold in multiplicative steps of 2 till the number of sources < AdaptiveNCut  
AdaptiveNCut   = 80    # The max number of sources that should be found by DAO. Used in Adaptive mode, and to provide source-finding advice

# GAIA catalog parameters
PhotCut = -99.0 # Apply this gmag cut to the GAIA catalog. If <= 0, no cut is applied.  
gaia_match_tolerance = 5.0 # Crossmatch tolerance between identified stars and GAIA stars, in arcseconds

# Set true to examine matched stars visually in cutouts
ExamineMatches = True

# Directories which contain various necessary files and images. Change as required.
imagesroot = '/Volumes/Outlander/LLAMA/HST_LLAMA/student_project_dir/' # root of the LLAMA images directory tree
maintableroot = '/Users/davidrosario/Projects/LLAMA/' # dir with the main LLAMA data table
listroot='/Users/davidrosario/Projects/LLAMA/HST-LLAMA/' # dir with the table that lists HST-LLAMA image files

llama_table = Table.read(maintableroot+'llama_main_properties.fits',format='fits')

file_listing = Table.read(listroot+'final_working_images.tab',format='ascii',data_start=1) 


objname    = 'NGC7582'
filterband = 'F606W'
instrument = 'WFPC2'
waveband='optical'

lindex, = np.where(llama_table['id'] == objname)
objcoord_sky = (llama_table['RA (deg)'][lindex[0]],llama_table['DEC (deg)'][lindex[0]]) # LLAMA coordinates for galaxy.

findex, = np.where((file_listing['ObjID'] == objname) & (file_listing['Filter'] == filterband) & (file_listing['Instrument'] == instrument))
# srfilename = imagesroot+objname+'/'+file_listing['Filename'][findex[0]].strip()+'.fits'
srfilename = imagesroot+objname+'/'+waveband+'.fits'
destfilename = imagesroot+objname+'/'+waveband+'_registered.fits'
catfilename = imagesroot+'gaia_catalogs/'+objname+'_gaiacat.fits'

test = hst_astrometric_correction_GAIA.HSTAstrometricCorrectionGAIA(srfilename,source_position=objcoord_sky,
		filterband=filterband,instrument=instrument)
test.get_gaia_catalog(gaiacatfilename=catfilename)
f = open('temp.coo','w')                                                                                       
for icoo in range(len(test.gaia_coords)): 
	writestring = '{0:<13.6f}   {1:<13.6f}'.format(test.gaia_coords[icoo].ra.value,test.gaia_coords[icoo].dec.value) 
	f.write(writestring+' \n') 
f.close()          

#test.find_stars_in_image(mask_source=False,default_threshold=False,daofwhm=2.5,daothreshold=5.0)
star_status = test.find_stars_in_image(mask_source=True,
	edge_maskfraction=0.01,
	default_threshold=False,
	daofwhm=3,daothreshold=20.0)

if not star_status:
	sys.exit()
match_status = test.pick_reference_stars(no_rotation=False, 
			   gaia_match_tolerance=2.0)
if not match_status:
	sys.exit()

test.transform_wcs(outfile=destfilename,no_rotation=True)
