# LLAMA-astrometric-alignment

Code and documentation for a script used to create astrometrically aligned multi-band images of LLAMA galaxies.
Dependencies: Python 3, AstroPy, NumPy, Photutils, Matplotlib
Repository contains :
  the main code as a script, 
  the LLAMA sample data table, 
  a text table listing the LLAMA image filenames that we will use for the HST project, 
  a Jupyter notebook showing examples of use of the script with different settings.
  
To use:

 - Ensure that you have a Python 3 installation with the necessary dependencies installed.
 
 - Download the script into a working directory on your machine.
 
 - Download the llama_table into a location on your machine.
 
 - Download the file listing into a location on your machine.
 
 - Download and untar the directory tree containing the LLAMA images.
 
 - Edit the script variables imagesroot, maintableroot, listroot to point to the directories with the various tables
   and image files.
 
 - If running a test, uncomment the second and third lines of code that lie under the comment "TEST MODE".
   Change the second line of code ("if ifile != 0:") accordingly to allow only one of the objects to be fit.
   If running over multiple images or a final run, keep these lines commented.
 
 - Examine the examples.ipynb Jupyter notebook for some chosen test cases to see what we expect to get after running the code.
   The script spits out some RunTimeWarnings because of remaining bad data values. This won't affect the performance of the code
   but I'll be updating things later after I figure out how to get rid of these.
   
 - When I'm running over multiple images, I tend to use the default thresholds, set AdaptThreshold = True and AdaptiveNCut = 80, 
   and set a largish gaia_match_tolerance = 5.0. I typically remove and stars that have a companion, clear non-circular structure
   due to saturation or blending, or cases where the same star appears multiple times (if I can identify the duplicates easily).
   
 - If you only want to run the code on images that have not been examined before, you can set OnlyIncomplete = True. This uses
   a logtable (completed_astrometric_correction.txt) in the working directory to determine which images have already been
   processed, and only runs over new ones. The logtable file can be edited to remove images which you may want to
   process again.

