import os
import json
import numpy as np

from astropy.table import Table, join
from desispec.io   import findfile
from desiutil.log import get_logger
from pkg_resources import resource_filename


log=get_logger()

fname='/project/projectdirs/desi/spectro/redux/daily/tsnr-exposures.fits'
opath=resource_filename('desispec','data/tsnr/tsnr_refset_etc.csv')

log.info('Writing to {}.'.format(opath))

daily_tsnrs=Table.read(fname, 'TSNR2_EXPID')
daily_tsnrs.pprint()

tokeep = []

for night, expid in zip(daily_tsnrs['NIGHT'], daily_tsnrs['EXPID']):
    etcpath=findfile('etc', night=night, expid=expid)
    etcdata=None
    
    if os.path.exists(etcpath):
        with open(etcpath) as f:
                etcdata = json.load(f)

    else:
        continue
                
    etc_fiberfracs = {}
                
    try:
        for tracer in ['psf', 'elg', 'bgs']:
            etc_fiberfracs[tracer]=etcdata['expinfo']['ffrac_{}'.format(tracer)] 

        log.info('Found etc ffracs for {} on {} ({})'.format(expid, night, etc_fiberfracs))

        tokeep.append([expid, etc_fiberfracs['psf'], etc_fiberfracs['elg'], etc_fiberfracs['bgs']])
        
    except:
        pass
        
tokeep = np.array(tokeep)    
tokeep = Table(tokeep, names=['EXPID', 'ETCFFRAC_PSF', 'ETCFFRAC_ELG', 'ETCFFRAC_BGS'])
tokeep['EXPID'] = tokeep['EXPID'].data.astype(np.int)
# tokeep.pprint()

tokeep = join(daily_tsnrs[np.isin(daily_tsnrs['EXPID'], tokeep['EXPID'])], tokeep, join_type='left', keys='EXPID')

for x in ['PSF', 'ELG', 'BGS']:
    print(np.sort(tokeep['ETCFFRAC_{}'.format(x)].data))

# print(tokeep.dtype.names)

fnight = tokeep['NIGHT'].min()
lnight = tokeep['NIGHT'].max()

tokeep.meta['comments'] = ['----  TSNR reference catalog 20210528 ----',\
                           '  MJW',\
                           '  EFFTIME_SPEC normalization based on SV1 (commit a056732 on Mar 19 2021, e.g. data/tsnr/tsnr-efftime.yaml); This cat. propagates this normalization to sv3 & later, where etc ffracs and updated tsnrs are available.',\
                           '  {}'.format(fname),\
                           '  NUMEXP: {}'.format(len(tokeep)),\
                           '  NIGHTS: {} to {}'.format(fnight, lnight)]

tokeep.write(opath, format='csv', overwrite=True, comment='#')

tokeepcsv = Table.read(opath, comment='#')
print(tokeepcsv.meta)
tokeepcsv.pprint()


##  Check.
for i, (night, expid, ffrac_psf, ffrac_elg, ffrac_bgs) in enumerate(zip(tokeep['NIGHT'], tokeep['EXPID'], tokeep['ETCFFRAC_PSF'], tokeep['ETCFFRAC_ELG'], tokeep['ETCFFRAC_BGS'])):
    etcpath=findfile('etc', night=night, expid=expid)
    etcdata=None

    with open(etcpath) as f:
        etcdata = json.load(f)

    etc_fiberfracs = {}
    
    for tracer in ['psf', 'elg', 'bgs']:
        etc_fiberfracs[tracer]=etcdata['expinfo']['ffrac_{}'.format(tracer)]

    assert  ffrac_psf == etc_fiberfracs['psf']
    assert  ffrac_elg == etc_fiberfracs['elg']
    assert  ffrac_bgs == etc_fiberfracs['bgs']

    print('Row {}: expid {} on night {} passes etc check'.format(i, expid, night))

print('\n\nDone.\n\n')
    
