import numpy as np
import pandas as pd

##  http://www.sdss3.org/svn//repo/idlspec2d/trunk/etc/emlines.par
lines          = pd.read_csv('../data/emlines.par', sep='\s+', skiprows=16, names=['LINEID', 'WAVELENGTH', 'NAME', 'REDSHIFT GROUP', 'WIDTH GROUP', 'FLUX GROUP', 'SCALE FACTOR'])
lines['INDEX'] = np.arange(len(lines))
lines['GROUP'] = lines.groupby(['REDSHIFT GROUP', 'WIDTH GROUP']).ngroup()

ugroups        = np.unique(lines['GROUP'] )

##  ----  OII wavelengths  ----
##  lines.loc[6,'WAVELENGTH'] 
##  lines.loc[7,'WAVELENGTH']

if __name__ == '__main__':
    print(lines)


    print(ugroups)
