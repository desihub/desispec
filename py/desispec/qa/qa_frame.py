"""
Classes to organize and execute QA for a DESI exposure
"""

from __future__ import print_function, absolute_import, division

import warnings

from desiutil.log import get_logger

from desispec.io import read_params
desi_params = read_params()

# log=get_logger()

class QA_Frame(object):
    def __init__(self, inp):
        """
        Class to organize and execute QA for a DESI frame

        x.flavor, x.qa_data, x.camera

        Args:
            inp : Frame object or dict
              * Frame -- Must contain meta data
              * dict -- Usually read from hard-drive

        Notes:

        """
        if isinstance(inp,dict):
            assert len(inp) == 1
            self.night = list(inp.keys())[0]  # Requires night in first key
            assert len(inp[self.night]) == 1
            self.expid = list(inp[self.night].keys())[0]
            assert len(inp[self.night][self.expid]) == 2
            self.flavor = inp[self.night][self.expid].pop('flavor')
            self.camera = list(inp[self.night][self.expid].keys())[0]
            assert self.camera[0] in ['b','r','z']
            self.qa_data = inp[self.night][self.expid][self.camera]
        else:
            # Generate from Frame and init QA data
            qkeys = ['flavor', 'camera', 'expid', 'night']
            for key in qkeys:
                setattr(self, key, inp.meta[key.upper()])  # FITS header
            self.qa_data = {}

        # Final test
        assert self.flavor in desi_params['frame_types']

    def init_qatype(self, qatype, param, re_init=False):
        """Initialize parameters for a given qatype
        qatype: str
          Type of QA to be performed (e.g. SKYSUB)
        param: dict
          Dict of parameters to guide QA
        re_init: bool, (optional)
          Re-initialize parameter dict
          Code will always add new parameters if any exist
        """
        # Fill and return if not set previously or if re_init=True
        if (qatype not in self.qa_data) or re_init:
            self.qa_data[qatype] = {}
            self.qa_data[qatype]['PARAMS'] = param
            return

        # Update the new parameters only
        for key in param:
            if key not in self.qa_data[qatype]['PARAMS']:
                self.qa_data[qatype]['PARAMS'][key] = param[key]

    def init_fiberflat(self, re_init=False):
        """Initialize parameters for FIBERFLAT QA
        QA method is desispec.fiberflat.qa_fiberflat

        Parameters:
        ------------
        re_init: bool, (optional)
          Re-initialize FIBERFLAT parameter dict
        """
        #
        assert self.flavor in ['flat']

        # Standard FIBERFLAT input parameters
        fflat_dict = dict(MAX_N_MASK=20000,  # Maximum number of pixels to mask
                          MAX_SCALE_OFF=0.05,  # Maximum offset in counts (fraction)
                          MAX_OFF=0.15,       # Maximum offset from unity
                          MAX_MEAN_OFF=0.05,  # Maximum offset in mean of fiberflat
                          MAX_RMS=0.02,      # Maximum RMS in fiberflat
                          )
        # Init
        self.init_qatype('FIBERFLAT', fflat_dict, re_init=re_init)

    def init_fluxcalib(self, re_init=False):
        """ Initialize parameters for FLUXCALIB QA
        Args:
            re_init: bool, (optional)
              Re-initialize  parameter dict

        Returns:

        """
        log=get_logger()
        assert self.flavor == 'science'

        # Standard FLUXCALIB input parameters
        flux_dict = dict(ZP_WAVE=0.,        # Wavelength for ZP evaluation (camera dependent)
                         MAX_ZP_OFF=0.2,    # Max offset in ZP for individual star
                         )

        if self.camera[0] == 'b':
            flux_dict['ZP_WAVE'] = 4800.  # Ang
        elif self.camera[0] == 'r':
            flux_dict['ZP_WAVE'] = 6500.  # Ang
        elif self.camera[0] == 'z':
            flux_dict['ZP_WAVE'] = 8250.  # Ang
        else:
            log.error("Not ready for camera {}!".format(self.camera))

        # Init
        self.init_qatype('FLUXCALIB', flux_dict, re_init=re_init)

    def init_skysub(self, re_init=False):
        """Initialize parameters for SkySub QA
        QA method is desispec.sky.qa_skysub

        Parameters:
        ------------
        re_init: bool, (optional)
          Re-initialize SKYSUB parameter dict
        """
        assert self.flavor == 'science'

        sky_dict = desi_params['qa']['skysub']['PARAMS']
        # Standard SKYSUB input parameters
        #sky_dict = dict(
        #    PCHI_RESID=0.05, # P(Chi^2) limit for bad skyfiber model residuals
        #    PER_RESID=95.,   # Percentile for residual distribution
        #    BIN_SZ=0.1, #- Bin size for residual/sigma histogram
        #    )
        # Init
        self.init_qatype('SKYSUB', sky_dict, re_init=re_init)

    def run_qa(self, qatype, inputs, clobber=True):
        """Run QA tests of a given type
        Over-writes previous QA of this type, unless otherwise specified

        qatype: str
          Type of QA to be performed (e.g. SKYSUB)
        inputs: tuple
          Set of inputs for the tests
        clobber: bool, optional [True]
          Over-write previous QA
        """
        from desispec.sky import qa_skysub
        from desispec.fiberflat import qa_fiberflat
        from desispec.fluxcalibration import qa_fluxcalib

        # Check for previous QA if clobber==False
        if (not clobber) and (qatype in self.qa_data.keys()):
            # QA previously performed?
            if 'METRICS' in self.qa_data[qatype]:
                return
        # Run
        if qatype == 'SKYSUB':
            # Expecting: frame, skymodel
            assert len(inputs) == 2
            # Init parameters (as necessary)
            self.init_skysub()
            # Run
            qadict = qa_skysub(self.qa_data[qatype]['PARAMS'],
                inputs[0], inputs[1])
        elif qatype == 'FIBERFLAT':
            # Expecting: frame, fiberflat
            assert len(inputs) == 2
            # Init parameters (as necessary)
            self.init_fiberflat()
            # Run
            qadict = qa_fiberflat(self.qa_data[qatype]['PARAMS'], inputs[0], inputs[1])
        elif qatype == 'FLUXCALIB':
            # Expecting: frame, fluxcalib
            assert len(inputs) == 2
            # Init parameters (as necessary)
            self.init_fluxcalib()
            # Run
            qadict = qa_fluxcalib(self.qa_data[qatype]['PARAMS'], inputs[0], inputs[1])
        else:
            raise ValueError('Not ready to perform {:s} QA'.format(qatype))
        # Update
        self.qa_data[qatype]['METRICS'] = qadict

    def __repr__(self):
        """ Print formatting
        """
        return ('{:s}: night={:s}, expid={:d}, camera={:s}, flavor={:s}'.format(
                self.__class__.__name__, self.night, self.expid, self.camera, self.flavor))

def qaframe_from_frame(frame_file, specprod_dir=None, make_plots=False, qaprod_dir=None,
                       output_dir=None, clobber=True):
    """  Generate a qaframe object from an input frame_file name (and night)

    Write QA to disk
    Will also make plots if directed
    Args:
        frame_file: str
        specprod_dir: str, optional
        qa_dir: str, optional -- Location of QA
        make_plots: bool, optional
        output_dir: str, optional

    Returns:

    """
    import glob
    import os

    from desispec.io import read_frame
    from desispec.io import meta
    from desispec.io.qa import load_qa_frame, write_qa_frame
    from desispec.io.qa import qafile_from_framefile
    from desispec.io.frame import search_for_framefile
    from desispec.io.fiberflat import read_fiberflat
    from desispec.fiberflat import apply_fiberflat
    from desispec.qa import qa_plots
    from desispec.io.sky import read_sky
    from desispec.io.fluxcalibration import read_flux_calibration

    if '/' in frame_file:  # If present, assume full path is used here
        pass
    else: # Find the frame file in the desispec hierarchy?
        frame_file = search_for_framefile(frame_file)

    # Load frame
    frame = read_frame(frame_file)
    frame_meta = frame.meta
    night = frame_meta['NIGHT'].strip()
    camera = frame_meta['CAMERA'].strip()
    expid = frame_meta['EXPID']
    spectro = int(frame_meta['CAMERA'][-1])

    # Filename
    qafile, qatype = qafile_from_framefile(frame_file, qaprod_dir=qaprod_dir, output_dir=output_dir)
    qaframe = load_qa_frame(qafile, frame, flavor=frame.meta['FLAVOR'])
    # Flat QA
    if frame_meta['FLAVOR'] in ['flat']:
        fiberflat_fil = meta.findfile('fiberflat', night=night, camera=camera, expid=expid,
                                      specprod_dir=specprod_dir)
        try: # Backwards compatibility
            fiberflat = read_fiberflat(fiberflat_fil)
        except FileNotFoundError:
            fiberflat_fil = fiberflat_fil.replace('exposures', 'calib2d')
            path, basen = os.path.split(fiberflat_fil)
            path,_ = os.path.split(path)
            fiberflat_fil = os.path.join(path, basen)
            fiberflat = read_fiberflat(fiberflat_fil)
        qaframe.run_qa('FIBERFLAT', (frame, fiberflat), clobber=clobber)
        if make_plots:
            # Do it
            qafig = meta.findfile('qa_flat_fig', night=night, camera=camera, expid=expid,
                                  specprod_dir=specprod_dir, outdir=output_dir)
            qa_plots.frame_fiberflat(qafig, qaframe, frame, fiberflat)
    # SkySub QA
    if qatype == 'qa_data':
        sky_fil = meta.findfile('sky', night=night, camera=camera, expid=expid, specprod_dir=specprod_dir)
        # Flat field first
        dummy_fiberflat_fil = meta.findfile('fiberflat', night=night, camera=camera, expid=expid,
                                            specprod_dir=specprod_dir) # This is dummy
        path,_ = os.path.split(dummy_fiberflat_fil)
        fiberflat_files = glob.glob(os.path.join(path,'fiberflat-'+camera+'*.fits'))
        # Backwards compatibility (for now)
        if len(fiberflat_files) == 0:
            path = path.replace('exposures', 'calib2d')
            path,_ = os.path.split(path) # Remove night
            fiberflat_files = glob.glob(os.path.join(path,'fiberflat-'+camera+'*.fits'))
        # Sort and take the first (same as current pipeline)
        fiberflat_files.sort()
        fiberflat = read_fiberflat(fiberflat_files[0])
        apply_fiberflat(frame, fiberflat)
        # Load sky model and run
        try:
            skymodel = read_sky(sky_fil)
        except FileNotFoundError:
            warnings.warn("Sky file {:s} not found.  Skipping..".format(sky_fil))
        else:
            qaframe.run_qa('SKYSUB', (frame, skymodel), clobber=clobber)
            if make_plots:
                qafig = meta.findfile('qa_sky_fig', night=night, camera=camera, expid=expid,
                                      specprod_dir=specprod_dir, outdir=output_dir)
                qafig2 = meta.findfile('qa_skychi_fig', night=night, camera=camera, expid=expid,
                                      specprod_dir=specprod_dir, outdir=output_dir)
                qa_plots.frame_skyres(qafig, frame, skymodel, qaframe)
                #qa_plots.frame_skychi(qafig2, frame, skymodel, qaframe)
    # FluxCalib QA
    if qatype == 'qa_data':
        # Standard stars
        stdstar_fil = meta.findfile('stdstars', night=night, camera=camera, expid=expid, specprod_dir=specprod_dir,
                                    spectrograph=spectro)
        # try:
        #    model_tuple=read_stdstar_models(stdstar_fil)
        # except FileNotFoundError:
        #    warnings.warn("Standard star file {:s} not found.  Skipping..".format(stdstar_fil))
        # else:
        flux_fil = meta.findfile('calib', night=night, camera=camera, expid=expid, specprod_dir=specprod_dir)
        try:
            fluxcalib = read_flux_calibration(flux_fil)
        except FileNotFoundError:
            warnings.warn("Flux file {:s} not found.  Skipping..".format(flux_fil))
        else:
            qaframe.run_qa('FLUXCALIB', (frame, fluxcalib))  # , model_tuple))#, indiv_stars))
            if make_plots:
                qafig = meta.findfile('qa_flux_fig', night=night, camera=camera, expid=expid,
                                      specprod_dir=specprod_dir, outdir=output_dir)
                qa_plots.frame_fluxcalib(qafig, qaframe, frame, fluxcalib)  # , model_tuple)
    # Write
    write_qa_frame(qafile, qaframe, verbose=True)
    return qaframe
