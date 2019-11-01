Use hatmann_shift.py to derive Hartmann shift across the CCD for one channel. 
Returns median values for correction reference. 
Also outputs plot of hartmann shift across CCD and histogram. 
Code to calcuate the Hartmann shift given a pair of left/right hartmann exposures. 

Example
python3 hartmann_shift.py -r_d /project/projectdirs/desi/spectro/data/20191030/ -c B3 -p /project/projectdirs/desi/users/jguy/kpno/sm4-bootcalib/psfboot-b3.fits -l 22754 -r 22757 -d 0. -rd False -dp False 

hartmann_correction.py combines 3 channels. 
