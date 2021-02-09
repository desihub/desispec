'''
Generate Master TSNR ensemble DFLUX files.  See doc. 4723.  Note: in this instance, ensemble avg. of flux 
is written, in order to efficiently generate tile depths.  

Currently assumes redshift and mag. ranges derived from FDR, but uniform in both.
'''
import sys
import copy
import yaml
import pickle
import desisim
import argparse
import os.path                       as     path
import numpy                         as     np
import astropy.io.fits               as     fits
import desisim.templates

from   astropy.convolution           import convolve, Box1DKernel
from   pathlib                       import Path
from   desiutil.dust                 import mwdust_transmission
from   desiutil.log                  import get_logger
from   pkg_resources                 import resource_filename

np.random.seed(seed=314)

# AR/DK DESI spectra wavelengths
# TODO:  where are brz extraction wavelengths defined?  https://github.com/desihub/desispec/issues/1006.                                                                                                                              
wmin, wmax, wdelta = 3600, 9824, 0.8
wave               = np.round(np.arange(wmin, wmax + wdelta, wdelta), 1)
cslice             = {"b": slice(0, 2751), "r": slice(2700, 5026), "z": slice(4900, 7781)}

def parse(options=None):
    parser = argparse.ArgumentParser(description="Generate a sim. template ensemble stack of given type and write it to disk at --outdir.")
    parser.add_argument('--nmodel', type = int, default = 2000, required=True,
                        help='Number of galaxies in the ensemble.')
    parser.add_argument('--tracer', type = str, default = 'bgs', required=True,
                        help='Tracer to generate of [bgs, lrg, elg, qso].')
    parser.add_argument('--configdir', type = str, default = None, required=False,
                        help='Directory to config files if not desispec repo.')
    parser.add_argument('--outdir', type = str, default = 'bgs', required=True,
			help='Directory to write to.')
    
    args = None

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args

class Config(object):
    def __init__(self, cpath):
        with open(cpath) as f:
            d = yaml.load(f, Loader=yaml.FullLoader)
        
        for key in d:
            setattr(self, key, d[key])
        
class template_ensemble(object):
    '''                                                                                                                                                                                                                                   
    Generate an ensemble of templates to sample tSNR for a range of points in                                                                                                                                                             
    (z, m, OII, etc.) space.                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    If conditioned, uses deepfield redshifts and (currently r) magnitudes to condition simulated templates.                                                                                                                               
    '''
    def __init__(self, outdir, tracer='elg', nmodel=5, log=None):
        if log is None:
            log = get_logger()
        
        def tracer_maker(wave, tracer=tracer, nmodel=nmodel, redshifts=None, mags=None):
            '''
            Dedicated wrapeper for desisim.templates.GALAXY.make_templates call, stipulating templates in a
            redshift range suggested by the FDR.  Further, assume fluxes close to the expected (within ~0.5 mags.)
            in the appropriate band.   

            Class init will write ensemble stack to disk at outdir, for a given tracer [bgs, lrg, elg, qso], having 
            generated nmodel templates.  Optionally, provide redshifts and mags. to condition appropriately at cose 
            of runtime.    
            '''
            # https://arxiv.org/pdf/1611.00036.pdf
            #

            if args.configdir == None:
                cpath = resource_filename('desispec', 'data/tsnr/tsnr-config-{}.yaml'.format(tracer))
            else:
                cpath = args.configdir + '/tsnr-config-{}.yaml'.format(tracer)
                
            config   = Config(cpath) 

            normfilter_south=config.filter

            zrange   = (config.zlo, config.zhi)
            magrange = (config.med_mag, config.limit_mag)

            log.info('nmodel: {:d}'.format(nmodel))
            
            log.info('{} filter: {}'.format(tracer, config.filter))
            log.info('{} zrange: {} - {}'.format(tracer,   zrange[0],   zrange[1]))
            log.info('{} magrange: {} - {}'.format(tracer, magrange[0], magrange[1]))
            
            if tracer == 'bgs':
                maker    = desisim.templates.BGS(wave=wave, normfilter_south=normfilter_south)
                flux, wave, meta, objmeta = maker.make_templates(nmodel=nmodel, redshift=redshifts, mag=mags, south=True, zrange=zrange, magrange=magrange)

            elif tracer == 'lrg':
                # https://github.com/desihub/desitarget/blob/dd353c6c8dd8b8737e45771ab903ac30584db6db/py/desitarget/cuts.py#L447
                maker    = desisim.templates.LRG(wave=wave, normfilter_south=normfilter_south)
                flux, wave, meta, objmeta = maker.make_templates(nmodel=nmodel, redshift=redshifts, mag=mags, south=True, zrange=zrange, magrange=magrange)
            
            if tracer == 'elg':
                # https://github.com/desihub/desitarget/blob/dd353c6c8dd8b8737e45771ab903ac30584db6db/py/desitarget/cuts.py#L517
                maker    = desisim.templates.ELG(wave=wave, normfilter_south=normfilter_south)
                flux, wave, meta, objmeta = maker.make_templates(nmodel=nmodel, redshift=redshifts, mag=mags, south=True, zrange=zrange, magrange=magrange)
                
            elif tracer == 'qso':
                # https://github.com/desihub/desitarget/blob/dd353c6c8dd8b8737e45771ab903ac30584db6db/py/desitarget/cuts.py#L1422
                maker    = desisim.templates.QSO(wave=wave, normfilter_south=normfilter_south)
                flux, wave, meta, objmeta = maker.make_templates(nmodel=nmodel, redshift=redshifts, mag=mags, south=True, zrange=zrange, magrange=magrange)
                                
            else:
                raise  ValueError('{} is not an available tracer.'.format(tracer))

            return  wave, flux, meta, objmeta
        
        _, flux, meta, objmeta         = tracer_maker(wave, tracer=tracer, nmodel=nmodel)
                
        self.ensemble_flux             = {}
        self.ensemble_dflux            = {}
        self.ensemble_meta             = meta
        self.ensemble_objmeta          = objmeta
        self.ensemble_dflux_stack      = {}
        
        # Generate template (d)fluxes for brz bands.                                                                                                                                                                                          
        for band in ['b', 'r', 'z']:
            band_wave                     = wave[cslice[band]]

            in_band                       = np.isin(wave, band_wave)

            self.ensemble_flux[band]      = flux[:, in_band]

            dflux                         = np.zeros_like(self.ensemble_flux[band])
        
            # Retain only spectral features < 100. Angstroms.                                                                                                                                                                                 
            # dlambda per pixel = 0.8; 100A / dlambda per pixel = 125.                                                                                                                                                                        
            for i, ff in enumerate(self.ensemble_flux[band]):
                sflux                     = convolve(ff, Box1DKernel(125), boundary='extend')
                dflux[i,:]                = ff - sflux

            self.ensemble_dflux[band]     = dflux

        # Stack ensemble.
        for band in ['b', 'r', 'z']:
            self.ensemble_dflux_stack[band] = np.sqrt(np.mean(self.ensemble_dflux[band]**2., axis=0).reshape(1, len(self.ensemble_dflux[band].T)))

        hdr = fits.Header()
        hdr['NMODEL'] = nmodel
        hdr['TRACER'] = tracer

        hdu_list = [fits.PrimaryHDU(header=hdr)]

        for band in ['b', 'r', 'z']:
            hdu_list.append(fits.ImageHDU(wave[cslice[band]], name='WAVE_{}'.format(band.upper())))
            hdu_list.append(fits.ImageHDU(self.ensemble_dflux_stack[band], name='DFLUX_{}'.format(band.upper())))

        hdu_list = fits.HDUList(hdu_list)
            
        hdu_list.writeto('{}/tsnr-ensemble-{}.fits'.format(outdir, tracer), overwrite=True)

        log.info('Successfully written to {}.'.format('{}/tsnr-ensemble-{}.fits'.format(outdir, tracer)))
        
def main():
    log = get_logger()

    args = parse()
    
    rads = template_ensemble(args.outdir, tracer=args.tracer, nmodel=args.nmodel, log=log)

if __name__ == '__main__':
    main()
