import numpy as np
import pandas as pd

from   astropy.table import Table
from   pkg_resources import resource_exists, resource_filename



fname                = resource_filename('desispec', 'data/emlines.par')

##  http://www.sdss3.org/svn//repo/idlspec2d/trunk/etc/emlines.par
lines                = pd.read_csv(fname, sep='\s+', skiprows=16, names=['LINEID', 'WAVELENGTH', 'NAME', 'REDSHIFT GROUP', 'WIDTH GROUP', 'FLUX GROUP', 'SCALE FACTOR'], comment='#')
lines['INDEX']       = np.arange(len(lines))
lines['GROUP']       = lines.groupby(['REDSHIFT GROUP', 'WIDTH GROUP']).ngroup()
lines['DOUBLET']     = np.zeros(len(lines), dtype=np.int) - 99

lines                = Table(lines.to_numpy(), names=lines.columns)

for i, x in enumerate([[6, 7], [16, 17], [25, 27]]):
    for y in x:
        lines['DOUBLET'][y] = i

lines['MASKED']        = np.zeros(len(lines), dtype=np.int)

# Ignored in chi sq. and not plotted; 4, 5, 8, 13, 14.
for x in [8, 13, 14]:
    lines['MASKED'][x] = 1

# Balmer.
# for x in [11, 12, 15]:
#    lines['MASKED'][x] = 1
    
ugroups                = np.array(np.unique(lines['GROUP']))

##  ----  OII wavelengths  ----
##  lines.loc[6,'WAVELENGTH'] 
##  lines.loc[7,'WAVELENGTH']

if __name__ == '__main__':
    print(lines)


    print(ugroups)
