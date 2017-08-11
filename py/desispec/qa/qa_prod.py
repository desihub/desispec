""" Class to organize QA for a full DESI production run
"""

from __future__ import print_function, absolute_import, division

import numpy as np
import glob, os
import warnings

from desispec.io import get_exposures
from desispec.io import get_files
from desispec.io import read_frame
from desispec.io import read_meta_frame
from desispec.io import specprod_root

from desiutil.log import get_logger

# log = get_logger()


class QA_Prod(object):
    def __init__(self, specprod_dir=None):
        """ Class to organize and execute QA for a DESI production

        Args:
            specprod_dir(str): Path containing the exposures/ directory to use. If the value
                is None, then the value of :func:`specprod_root` is used instead.
        Notes:

        Attributes:
            qa_exps : list
              List of QA_Exposure classes, one per exposure in production
            data : dict
        """
        if specprod_dir is None:
            specprod_dir = specprod_root()
        self.specprod_dir = specprod_dir
        tmp = specprod_dir.split('/')
        self.prod_name = tmp[-1] if (len(tmp[-1]) > 0) else tmp[-2]
        self.qa_exps = []
        #
        self.data = {}

    def get_qa_table(self, qatype, metric, nights='all', channels='all'):
        """ Generate a table of QA values from .data
        Args:
            qatype: str
              FIBERFLAT, SKYSUB
            metric: str
            nights: str or list of str, optional
            channels: str or list of str, optional
              'b', 'r', 'z'

        Returns:
            qa_tbl: Table
        """
        from astropy.table import Table
        out_list = []
        out_nights = []
        out_expid = []
        out_expmeta = []
        # Nights
        for night in self.data:
            if (night not in nights) and (nights != 'all'):
                continue
            # Exposures
            for expid in self.data[night]:
                # Cameras
                exp_meta = self.data[night][expid]['meta']
                for camera in self.data[night][expid]:
                    if camera in ['flavor', 'meta']:
                        continue
                    if (camera[0] not in channels) and (channels != 'all'):
                        continue
                    # Grab
                    try:
                        val = self.data[night][expid][camera][qatype]['METRICS'][metric]
                    except KeyError:  # Each exposure has limited qatype
                        pass
                    except TypeError:
                        import pdb; pdb.set_trace()
                    else:
                        if isinstance(val, (list,tuple)):
                            out_list.append(val[0])
                        else:
                            out_list.append(val)
                        # Meta data
                        out_expid.append(expid)
                        out_expmeta.append(exp_meta)
        # Return Table
        qa_tbl = Table()
        qa_tbl[metric] = out_list
        qa_tbl['EXPID'] = out_expid
        # Add expmeta
        for key in out_expmeta[0].keys():
            tmp_list = []
            for exp_meta in out_expmeta:
                tmp_list.append(exp_meta[key])
            qa_tbl[key] = tmp_list
        return qa_tbl

    def load_data(self):
        """ Load QA data from disk
        """
        from desispec.io.qa import load_qa_prod
        #
        inroot = self.specprod_dir+'/'+self.prod_name+'_qa'
        self.data = load_qa_prod(inroot)

    def make_frameqa(self, make_plots=False, clobber=True):
        """ Work through the Production and make QA for all frames

        Parameters:
            make_plots: bool, optional
              Remake the plots too?
            clobber: bool, optional
        Returns:

        """
        # imports
        from desispec.io import meta
        from desispec.io.qa import load_qa_frame, write_qa_frame
        from desispec.io.fiberflat import read_fiberflat
        from desispec.io.sky import read_sky
        from desispec.io.fluxcalibration import read_flux_calibration
        from desispec.qa import qa_plots
        from desispec.io.fluxcalibration import read_stdstar_models
        log = get_logger()

        # Loop on nights
        path_nights = glob.glob(self.specprod_dir+'/exposures/*')
        nights = [ipathn[ipathn.rfind('/')+1:] for ipathn in path_nights]
        for night in nights:
            for exposure in get_exposures(night, specprod_dir = self.specprod_dir):
                # Object only??
                frames_dict = get_files(filetype = str('frame'), night = night,
                        expid = exposure, specprod_dir = self.specprod_dir)
                for camera,frame_fil in frames_dict.items():
                    # Load frame
                    frame_meta = read_meta_frame(frame_fil)  # Only meta to speed it up
                    spectro = int(frame_meta['CAMERA'][-1])
                    if frame_meta['FLAVOR'] in ['flat','arc']:
                        qatype = 'qa_calib'
                    else:
                        qatype = 'qa_data'
                    qafile = meta.findfile(qatype, night=night, camera=camera, expid=exposure, specprod_dir=self.specprod_dir)
                    if (not clobber) & os.path.isfile(qafile):
                        log.info("qafile={:s} exists.  Not over-writing.  Consider clobber=True".format(qafile))
                        continue
                    else:  # Now the full read
                        frame = read_frame(frame_fil)
                    # Load
                    try:
                        qaframe = load_qa_frame(qafile, frame, flavor=frame.meta['FLAVOR'])
                    except AttributeError:
                        import pdb; pdb.set_trace
                    # Flat QA
                    if frame.meta['FLAVOR'] in ['flat']:
                        fiberflat_fil = meta.findfile('fiberflat', night=night, camera=camera, expid=exposure, specprod_dir=self.specprod_dir)
                        fiberflat = read_fiberflat(fiberflat_fil)
                        qaframe.run_qa('FIBERFLAT', (frame, fiberflat), clobber=clobber)
                        if make_plots:
                            # Do it
                            qafig = meta.findfile('qa_flat_fig', night=night, camera=camera, expid=exposure, specprod_dir=self.specprod_dir)
                            qa_plots.frame_fiberflat(qafig, qaframe, frame, fiberflat)
                    # SkySub QA
                    if qatype == 'qa_data':
                        sky_fil = meta.findfile('sky', night=night, camera=camera, expid=exposure, specprod_dir=self.specprod_dir)
                        try:
                            skymodel = read_sky(sky_fil)
                        except FileNotFoundError:
                            warnings.warn("Sky file {:s} not found.  Skipping..".format(sky_fil))
                        else:
                            qaframe.run_qa('SKYSUB', (frame, skymodel))
                            if make_plots:
                                qafig = meta.findfile('qa_sky_fig', night=night, camera=camera, expid=exposure, specprod_dir=self.specprod_dir)
                                qa_plots.frame_skyres(qafig, frame, skymodel, qaframe)
                    # FluxCalib QA
                    if qatype == 'qa_data':
                        # Standard stars
                        stdstar_fil = meta.findfile('stdstars', night=night, camera=camera, expid=exposure, specprod_dir=self.specprod_dir,
                                                    spectrograph=spectro)
                        #try:
                        #    model_tuple=read_stdstar_models(stdstar_fil)
                        #except FileNotFoundError:
                        #    warnings.warn("Standard star file {:s} not found.  Skipping..".format(stdstar_fil))
                        #else:
                        flux_fil = meta.findfile('calib', night=night, camera=camera, expid=exposure, specprod_dir=self.specprod_dir)
                        try:
                            fluxcalib = read_flux_calibration(flux_fil)
                        except FileNotFoundError:
                            warnings.warn("Flux file {:s} not found.  Skipping..".format(flux_fil))
                        else:
                            qaframe.run_qa('FLUXCALIB', (frame, fluxcalib)) #, model_tuple))#, indiv_stars))
                            if make_plots:
                                qafig = meta.findfile('qa_flux_fig', night=night, camera=camera, expid=exposure, specprod_dir=self.specprod_dir)
                                qa_plots.frame_fluxcalib(qafig, qaframe, frame, fluxcalib)#, model_tuple)
                    # Write
                    write_qa_frame(qafile, qaframe)

    def slurp(self, make_frameqa=False, remove=True, **kwargs):
        """ Slurp all the individual QA files into one master QA file
        Args:
            make_frameqa: bool, optional
              Regenerate the individual QA files (at the frame level first)
            remove: bool, optional
              Remove

        Returns:

        """
        from desispec.qa import QA_Exposure
        from desispec.io import write_qa_prod
        log = get_logger()
        # Remake?
        if make_frameqa:
            self.make_frameqa(**kwargs)
        # Loop on nights
        path_nights = glob.glob(self.specprod_dir+'/exposures/*')
        nights = [ipathn[ipathn.rfind('/')+1:] for ipathn in path_nights]
        # Reset
        log.info("Resetting qa_exps in qa_prod")
        self.qa_exps = []
        # Loop
        for night in nights:
            # Loop on exposures
            for exposure in get_exposures(night, specprod_dir = self.specprod_dir):
                frames_dict = get_files(filetype = str('frame'), night = night,
                                        expid = exposure, specprod_dir = self.specprod_dir)
                if len(frames_dict) == 0:
                    continue
                # Load any frame (for the type and meta info)
                key = list(frames_dict.keys())[0]
                frame_fil = frames_dict[key]
                frame_meta = read_meta_frame(frame_fil)
                qa_exp = QA_Exposure(exposure, night, frame_meta['FLAVOR'],
                                     specprod_dir=self.specprod_dir, remove=remove)
                qa_exp.load_meta(frame_meta)
                # Append
                self.qa_exps.append(qa_exp)
        # Write
        outroot = self.specprod_dir+'/'+self.prod_name+'_qa'
        write_qa_prod(outroot, self)

    def __repr__(self):
        """ Print formatting
        """
        return ('{:s}: specprod_dir={:s}'.format(self.__class__.__name__, self.specprod_dir))
