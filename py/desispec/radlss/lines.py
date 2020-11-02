import numpy as np
import pandas as pd

from astropy.table import Table


##  http://www.sdss3.org/svn//repo/idlspec2d/trunk/etc/emlines.par
lines                = pd.read_csv('../data/emlines.par', sep='\s+', skiprows=16, names=['LINEID', 'WAVELENGTH', 'NAME', 'REDSHIFT GROUP', 'WIDTH GROUP', 'FLUX GROUP', 'SCALE FACTOR'])
lines['INDEX']       = np.arange(len(lines))
lines['GROUP']       = lines.groupby(['REDSHIFT GROUP', 'WIDTH GROUP']).ngroup()
lines['DOUBLET']     = np.zeros(len(lines), dtype=np.int) - 99

lines                = Table(lines.to_numpy(), names=lines.columns)

for i, x in enumerate([[6, 7], [16, 17], [25, 27]]):
    for y in x:
        lines['DOUBLET'][y] = i

ugroups              = np.array(np.unique(lines['GROUP']))

##  ----  OII wavelengths  ----
##  lines.loc[6,'WAVELENGTH'] 
##  lines.loc[7,'WAVELENGTH']

if __name__ == '__main__':
    print(lines)


    print(ugroups)
