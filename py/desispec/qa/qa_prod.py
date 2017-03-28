""" Class to organize QA for a full DESI production run
"""

from __future__ import print_function, absolute_import, division

import numpy as np
import glob, os

from desispec.io import get_exposures
from desispec.io import get_files
from desispec.io import read_frame

from desiutil.log import get_logger

# log = get_logger()


class QA_Prod(object):
    def __init__(self, specprod_dir):
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
        self.specprod_dir = specprod_dir
        tmp = specprod_dir.split('/')
        self.prod_name = tmp[-1] if (len(tmp[-1]) > 0) else tmp[-2]
        self.qa_exps = []
        #
        self.data = {}

    def get_qa_array(self, qatype, metric, nights='all', channels='all'):
        """ Generate an array of QA values from .data
        Args:
            qatype: str
              FIBERFLAT, SKYSUB
            metric: str
            nights: str or list of str, optional
            channels: str or list of str, optional
              'b', 'r', 'z'

        Returns:
            array: ndarray
            ne_dict: dict
              dict of nights and exposures contributing to the array
        """
        import pdb
        out_list = []
        ne_dict = {}
        # Nights
        for night in self.data:
            if (night not in nights) and (nights != 'all'):
                continue
            # Exposures
            for expid in self.data[night]:
                # Cameras
                for camera in self.data[night][expid]:
                    if camera == 'flavor':
                        continue
                    if (camera[0] not in channels) and (channels != 'all'):
                        continue
                    # Grab
                    try:
                        val = self.data[night][expid][camera][qatype]['METRICS'][metric]
                    except KeyError:  # Each exposure has limited qatype
                        pass
                    except TypeError:
                        pdb.set_trace()
                    else:
                        if isinstance(val, (list,tuple)):
                            out_list.append(val[0])
                        else:
                            out_list.append(val)
                        # dict
                        if night not in ne_dict:
                            ne_dict[night] = []
                        if expid not in ne_dict[night]:
                            ne_dict[night].append(expid)
        # Return
        return np.array(out_list), ne_dict

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
                    frame = read_frame(frame_fil)
                    spectro = int(frame.meta['CAMERA'][-1])
                    if frame.meta['FLAVOR'] in ['flat','arc']:
                        qatype = 'qa_calib'
                    else:
                        qatype = 'qa_data'
                    qafile = meta.findfile(qatype, night=night, camera=camera, expid=exposure, specprod_dir=self.specprod_dir)
                    if (not clobber) & os.path.isfile(qafile):
                        log.info("qafile={:s} exists.  Not over-writing.  Consider clobber=True".format(qafile))
                        continue
                    # Load
                    qaframe = load_qa_frame(qafile, frame, flavor=frame.meta['FLAVOR'])
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
                        skymodel = read_sky(sky_fil)
                        qaframe.run_qa('SKYSUB', (frame, skymodel))
                        if make_plots:
                            qafig = meta.findfile('qa_sky_fig', night=night, camera=camera, expid=exposure, specprod_dir=self.specprod_dir)
                            qa_plots.frame_skyres(qafig, frame, skymodel, qaframe)
                    # FluxCalib QA
                    if qatype == 'qa_data':
                        # Standard stars
                        stdstar_fil = meta.findfile('stdstars', night=night, camera=camera, expid=exposure, specprod_dir=self.specprod_dir,
                                                    spectrograph=spectro)
                        model_tuple=read_stdstar_models(stdstar_fil)
                        flux_fil = meta.findfile('calib', night=night, camera=camera, expid=exposure, specprod_dir=self.specprod_dir)
                        fluxcalib = read_flux_calibration(flux_fil)
                        qaframe.run_qa('FLUXCALIB', (frame, fluxcalib, model_tuple))#, indiv_stars))
                        if make_plots:
                            qafig = meta.findfile('qa_flux_fig', night=night, camera=camera, expid=exposure, specprod_dir=self.specprod_dir)
                            qa_plots.frame_fluxcalib(qafig, qaframe, frame, fluxcalib, model_tuple)
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
        from desispec.io import meta
        from desispec.qa import QA_Exposure
        from desispec.io import write_qa_prod
        import pdb
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
                # Load any frame (for the type)
                key = list(frames_dict.keys())[0]
                frame_fil = frames_dict[key]
                frame = read_frame(frame_fil)
                qa_exp = QA_Exposure(exposure, night, frame.meta['FLAVOR'],
                                         specprod_dir=self.specprod_dir, remove=remove)
                # Append
                self.qa_exps.append(qa_exp)
        # Write
        outroot = self.specprod_dir+'/'+self.prod_name+'_qa'
        write_qa_prod(outroot, self)

    def __repr__(self):
        """ Print formatting
        """
        return ('{:s}: specprod_dir={:s}'.format(self.__class__.__name__, self.specprod_dir))
