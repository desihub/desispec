import numpy as np
import yaml
from desispec.io import findfile
import os,sys
from desispec.quicklook import qlexceptions,qllogger
qlog=qllogger.QLLogger("QuickLook",20)
log=qlog.getlog()


class Config(object):
    """ 
    A class to generate Quicklook configurations for a given desi exposure. 
    expand_config will expand out to full format as needed by quicklook.setup
    """

    def __init__(self, configfile, night, camera, expid, amps=True,rawdata_dir=None,specprod_dir=None, outdir=None,qlf=False):
        """
        configfile: a configuration file for QL eg: desispec/data/quicklook/qlconfig_dark.yaml
        night: night for the data to process, eg.'20191015'
        camera: which camera to process eg 'r0'
        expid: exposure id for the image to be processed 
        amps: for outputing amps level QA
        Note:
        rawdata_dir and specprod_dir: if not None, overrides the standard DESI convention       
        """  
  
        #- load the config file and extract
        self.conf = yaml.load(open(configfile,"r"))
        self.night = night
        self.expid = expid
        self.camera = camera
        self.amps = amps
        self.rawdata_dir = rawdata_dir 
        self.specprod_dir = specprod_dir
        self.outdir = outdir
        self.dumpintermediates = self.conf["WriteIntermediatefiles"]
        self.writepixfile = self.conf["WritePixfile"]
        self.writeskymodelfile = self.conf["WriteSkyModelfile"]
        self.writestaticplots = self.conf["WriteStaticPlots"]
        self.usesigma = self.conf["UseResolution"]
        self.pipeline = self.conf["Pipeline"]
        self.algorithms = self.conf["Algorithms"]
        self._palist = Palist(self.pipeline,self.algorithms)
        self.pamodule = self._palist.pamodule
        self.qamodule = self._palist.qamodule
        if "BoxcarExtract" in self.algorithms.keys():
            if "wavelength" in self.algorithms["BoxcarExtract"].keys():
                self.wavelength = self.algorithms["BoxcarExtract"]["wavelength"][self.camera[0]]
            else: self.wavelength = None
        self._qlf=qlf


    @property
    def mode(self):
        """ what mode of QL, online? offline?
        """
        return self._palist.mode

    @property
    def qlf(self):
        return self._qlf

    @property
    def palist(self): 
        """ palist for this config
            see :class: `Palist` for details.
        """
        return self._palist.palist

    @property
    def qalist(self):
        """ qalist for the given palist
        """
        return self._palist.qalist

    @property
    def paargs(self,psfspfile=None):
        """
        Many arguments for the PAs are taken default. Some of these may need to be variable
        psfspfile is for offline extraction case
        """
        wavelength=self.wavelength
        if self.wavelength is None:
            #- setting default wavelength for extraction for different cam
            if self.camera[0] == 'b':
                self.wavelength='3570,5730,0.8'
            elif self.camera[0] == 'r':
                self.wavelength='5630,7740,0.8'
            elif self.camera[0] == 'z':
                self.wavelength='7420,9830,0.8'

        #- Make kwargs less verbose using '%%' marker for global variables. Pipeline will map them back
        paopt_initialize={'camera': self.camera}

        if self.writepixfile:
            pixfile=self.dump_pa("Preproc")
        else: 
            pixfile = None
        paopt_preproc={'camera': self.camera,'dumpfile': pixfile}

        if self.dumpintermediates:
            if self.conf["Flavor"] == 'arcs':
                calibdir=os.path.join(os.environ['QL_SPEC_REDUX'],'calib2d',self.night)
                framefile=findfile('frame',night=self.night,expid=self.expid,camera=self.camera,outdir=calibdir)
            else:
                framefile=self.dump_pa("BoxcarExtract")
            fframefile=self.dump_pa("ApplyFiberFlat_QL")
            sframefile=self.dump_pa("SkySub_QL")
        else:
            framefile=None
            fframefile=None
            sframefile=None

        if self.conf["Flavor"] == 'arcs':
            arcimg=findfile('pix',night=self.night,expid=self.expid,camera=self.camera,rawdata_dir=self.rawdata_dir)
            flatimg=findfile('pix',night=self.night,expid=self.conf["FiberflatExpid"],camera=self.camera,rawdata_dir=self.rawdata_dir)
            bootfile=findfile('psfboot',night=self.night,camera=self.camera,specprod_dir=self.specprod_dir)
            psfnightfile=findfile('psfnight',night=self.night,camera=self.camera,specprod_dir=self.specprod_dir)
        else:
            arcimg=None
            flatimg=None
            bootfile=None
            psfnightfile=None

        paopt_bootcalib={'ArcLampImage':arcimg, 'FlatImage':flatimg, 'outputFile':bootfile}

        paopt_extract={'BoxWidth': 2.5, 'FiberMap': self.fibermap, 'Wavelength': self.wavelength, 'Nspec': 500, 'PSFFile': self.psf,'usesigma': self.usesigma, 'dumpfile': framefile}

        paopt_resfit={'PSFbootfile':bootfile, 'PSFoutfile': psfnightfile, 'usesigma': self.usesigma}

        paopt_apfflat={'FiberFlatFile': self.fiberflat, 'dumpfile': fframefile}

        if self.writeskymodelfile:
            outskyfile = findfile('sky',night=self.night,expid=self.expid, camera=self.camera, rawdata_dir=self.rawdata_dir,specprod_dir=self.specprod_dir,outdir=self.outdir)
        else:
            outskyfile=None       
        paopt_skysub={'Outskyfile': outskyfile, 'dumpfile': sframefile,'Apply_resolution': self.usesigma}

        paopts={}
        for PA in self.palist:
            if PA=='Initialize':
                paopts[PA]=paopt_initialize
            elif PA=='Preproc':
                paopts[PA]=paopt_preproc
            elif PA=='BootCalibration':
                paopts[PA]=paopt_bootcalib
            elif PA=='BoxcarExtract':
                paopts[PA]=paopt_extract
            elif PA=='ResolutionFit':
                paopts[PA]=paopt_resfit
            elif PA=='ApplyFiberFlat_QL':
                paopts[PA]=paopt_apfflat
            elif PA=='SkySub_QL':
                paopts[PA]=paopt_skysub
            else:
                paopts[PA]={}
        #- Ignore intermediate dumping and write explicitly the outputfile for 
        self.outputfile=self.dump_pa(self.palist[-1]) 

        return paopts 
        
    def dump_pa(self,paname):
        """
        dump the PA outputs to respective files. This has to be updated for fframe and sframe files as QL anticipates for dumpintermediate case.
        """
        pafilemap={'Preproc': 'pix', 'BootCalibration': 'psfboot', 'BoxcarExtract': 'frame', 'ResolutionFit': 'psfnight', 'ApplyFiberFlat_QL': 'fframe', 'SkySub_QL': 'sframe'}
        
        if paname in pafilemap:
            filetype=pafilemap[paname]
        else:
            raise IOError("PA name does not match any file type. Check PA name in config") 
           
        pafile=findfile(filetype,night=self.night,expid=self.expid,camera=self.camera,rawdata_dir=self.rawdata_dir,specprod_dir=self.specprod_dir,outdir=self.outdir)

        return pafile

    def dump_qa(self): 
        """ 
        yaml outputfile for the set of qas for a given pa
        Name and default locations of files are handled by desispec.io.meta.findfile
        """

        #- both PA level and QA level outputs
        qa_pa_outfile = {}
        qa_pa_outfig = {}
        qa_outfile = {}
        qa_outfig = {}
        for PA in self.palist:
            #- pa level outputs
            if self.dumpintermediates:
                qa_pa_outfile[PA] = self.io_qa_pa(PA)[0]
                qa_pa_outfig[PA] = self.io_qa_pa(PA)[1]
            else:
                qa_pa_outfile[PA] = None
                qa_pa_outfig[PA] = None
            #- qa_level output
            for QA in self.qalist[PA]:
                qa_outfile[QA] = self.io_qa(QA)[0]
                qa_outfig[QA] = self.io_qa(QA)[1]
                
                #- make path if needed
                path = os.path.normpath(os.path.dirname(qa_outfile[QA]))
                if not os.path.exists(path):
                    os.makedirs(path)

        return ((qa_outfile,qa_outfig),(qa_pa_outfile,qa_pa_outfig))

    @property
    def qaargs(self):

        qaopts = {}
        
        for PA in self.palist:
            for qa in self.qalist[PA]: #- individual QA for that PA
                if self.writestaticplots:
                    qaplot = self.dump_qa()[0][1][qa]
                else:
                    qaplot = None

                params=self._qaparams(qa)
                qaopts[qa]={'camera': self.camera, 'paname': PA, 'PSFFile': self.psf, 'amps': self.amps, 'qafile': self.dump_qa()[0][0][qa],'qafig': qaplot, 'FiberMap': self.fibermap, 'param': params, 'qlf': self.qlf}
                
        return qaopts 
   
    def _qaparams(self,qa):
            
        params={}
        if self.algorithms is not None:
            for PA in self.palist:
                if qa in self.qalist[PA]:
                    params[qa]=self.algorithms[PA]['QA'][qa]['PARAMS']

        else:
            if qa == 'Count_Pixels':
                params[qa]= dict(
                                CUTLO = 100,
                                 CUTHI = 500
                                )
            elif qa == 'CountSpectralBins':
                params[qa]= dict(
                                 CUTLO = 100,   # low threshold for number of counts
                                 CUTMED = 250,
                                 CUTHI = 500
                                )
            elif qa == 'Sky_Residual':
                params[qa]= dict(
                                 PCHI_RESID=0.05, # P(Chi^2) limit for bad skyfiber model residuals
                                 PER_RESID=95.,   # Percentile for residual distribution
                                 BIN_SZ=0.1,) # Bin size for residual/sigma histogram
            else:
                params[qa]= dict()
        
        return params[qa]

    def io_qa_pa(self,paname):
        """
        Specify the filenames: yaml and png of the pa level qa files"
        """
        if self.conf["Flavor"] == 'arcs':
            filemap={'Initialize': 'ql_initial_arc',
                     'Preproc': 'ql_preproc_arc',
                     'BootCalibration': 'ql_bootcalib',
                     'BoxcarExtract': 'ql_boxextract_arc',
                     'ResolutionFit': 'ql_resfit_arc'
                     }
        else:
            filemap={'Initialize': 'ql_initial',
                     'Preproc': 'ql_preproc',
                     'BootCalibration': 'ql_bootcalib',
                     'BoxcarExtract': 'ql_boxextract',
                     'ResolutionFit': 'ql_resfit',
                     'ApplyFiberFlat_QL': 'ql_fiberflat',
                     'SkySub_QL': 'ql_skysub'
                     }

        if paname in filemap:
            filetype=filemap[paname]+'_file'
            figtype=filemap[paname]+'_fig'
        else:
            raise IOError("PA name does not match any file type. Check PA name in config for {}".format(paname))

        outfile=findfile(filetype,night=self.night,expid=self.expid, camera=self.camera, rawdata_dir=self.rawdata_dir,specprod_dir=self.specprod_dir,outdir=self.outdir)

        outfig=findfile(figtype,night=self.night,expid=self.expid, camera=self.camera, rawdata_dir=self.rawdata_dir,specprod_dir=self.specprod_dir,outdir=self.outdir)

        return (outfile,outfig)


    def io_qa(self,qaname):
        """
        Specify the filenames: yaml and png for the given qa output
        """
        if self.conf["Flavor"] == 'arcs':
            filemap={'Bias_From_Overscan': 'ql_getbias_arc',
                     'Get_RMS' : 'ql_getrms_arc',
                     'Count_Pixels': 'ql_countpix_arc',
                     'Calc_XWSigma': 'ql_xwsigma_arc',
                     'CountSpectralBins': 'ql_countbins_arc'
                     }
        else:
            filemap={'Bias_From_Overscan': 'ql_getbias',
                     'Get_RMS' : 'ql_getrms',
                     'Count_Pixels': 'ql_countpix',
                     'Calc_XWSigma': 'ql_xwsigma',
                     'CountSpectralBins': 'ql_countbins',
                     'Sky_Continuum': 'ql_skycont',
                     'Sky_Peaks': 'ql_skypeak',
                     'Sky_Residual': 'ql_skyresid',
                     'Integrate_Spec': 'ql_integ',
                     'Calculate_SNR': 'ql_snr'
                     }

        if qaname in filemap:
            filetype=filemap[qaname]+'_file'
            figtype=filemap[qaname]+'_fig'
        else:
            raise IOError("QA name does not match any file type. Check QA name in config for {}".format(qaname))

        outfile=findfile(filetype,night=self.night,expid=self.expid, camera=self.camera, rawdata_dir=self.rawdata_dir,specprod_dir=self.specprod_dir,outdir=self.outdir)

        outfig=findfile(figtype,night=self.night,expid=self.expid, camera=self.camera, rawdata_dir=self.rawdata_dir,specprod_dir=self.specprod_dir,outdir=self.outdir)

        return (outfile,outfig)

    def expand_config(self):
        """
        config: desispec.quicklook.qlconfig.Config object
        """
        log.info("Building Full Configuration")

        self.program = self.conf["Program"]
        self.flavor = self.conf["Flavor"]
        self.debuglevel = self.conf["Debuglevel"]
        self.period = self.conf["Period"]
        self.timeout = self.conf["Timeout"]
        self.fiberflatexpid = self.conf["FiberflatExpid"]
        self.psftype = self.conf["PSFType"]

        #- some global variables:
        self.rawfile=findfile("raw",night=self.night,expid=self.expid, camera=self.camera, rawdata_dir=self.rawdata_dir,specprod_dir=self.specprod_dir)

        self.fibermap=findfile("fibermap", night=self.night,expid=self.expid,camera=self.camera, rawdata_dir=self.rawdata_dir,specprod_dir=self.specprod_dir)
 
        self.fiberflat=findfile("fiberflat",night=self.night,expid=self.fiberflatexpid,camera=self.camera, rawdata_dir=self.rawdata_dir,specprod_dir=self.specprod_dir) #- TODO: Assuming same night for calibration files (here and psf)
        
        self.psf=findfile(self.psftype,night=self.night,expid=self.expid,camera=self.camera, rawdata_dir=self.rawdata_dir,specprod_dir=self.specprod_dir)  
        
        outconfig={}

        outconfig['Night'] = self.night
        outconfig['Program'] = self.program
        outconfig['Flavor'] = self.flavor
        outconfig['Camera'] = self.camera
        outconfig['Expid'] = self.expid
        outconfig['DumpIntermediates'] = self.dumpintermediates
        outconfig['FiberMap']=self.fibermap
        outconfig['FiberFlatFile'] = self.fiberflat
        outconfig['PSFFile'] = self.psf
        outconfig['Period'] = self.period

        pipeline = []
        for ii,PA in enumerate(self.palist):
            pipe={'OutputFile': self.dump_qa()[1][0][PA]} #- integrated QAs for that PA. 
            pipe['PA'] = {'ClassName': PA, 'ModuleName': self.pamodule, 'kwargs': self.paargs[PA]}
            pipe['QAs']=[]
            for jj, QA in enumerate(self.qalist[PA]):
                pipe_qa={'ClassName': QA, 'ModuleName': self.qamodule, 'kwargs': self.qaargs[QA]}
                pipe['QAs'].append(pipe_qa)
            pipe['StepName']=PA
            pipeline.append(pipe)

        outconfig['PipeLine'] = pipeline
        outconfig['RawImage'] = self.rawfile
        outconfig["OutputFile"] = self.outputfile
        outconfig['Timeout'] = self.timeout

        #- Check if all the files exist for this QL configuraion
        check_config(outconfig)

        return outconfig

def check_config(outconfig):
    """
    Given the expanded config, check for all possible file existence etc....
    """

    if outconfig["Flavor"]=="science":
        files = [outconfig["RawImage"], outconfig["FiberMap"], outconfig["FiberFlatFile"], outconfig["PSFFile"]]
        log.info("Checking if all the necessary files exist.")
        for thisfile in files:
            if not os.path.exists(thisfile):
                sys.exit("File does not exist: {}".format(thisfile))
            else:
                log.info("File check: Okay: {}".format(thisfile))
        log.info("All necessary files exist for science configuration.")
    elif outconfig["Flavor"]=="arcs":
        files = [outconfig["RawImage"], outconfig["FiberMap"]]
        log.info("Checking if all the necessary files exist.")
        for thisfile in files:
            if not os.path.exists(thisfile):
                sys.exit("File does not exist: {}".format(thisfile))
            else:
                log.info("File check: Okay: {}".format(thisfile))
        log.info("All necessary files exist for arc configuration.")
    return 


class Palist(object):
    
    """
    Generate PA list and QA list for the Quicklook Pipeline for the given exposure
    """
    def __init__(self,thislist=None,algorithms=None,flavor=None,mode=None):
        """
        thislist: given list of PAs
        algorithms: Algorithm list coming from config file: e.g desispec/data/quicklook/qlconfig_dark.yaml
        flavor: only needed if new list is to be built.
        mode: online offline?
        """
        self.flavor=flavor
        self.mode=mode
        self.thislist=thislist
        self.algorithms=algorithms
        self.palist=self._palist()
        self.qalist=self._qalist()
        
    def _palist(self):
        
        if self.thislist is not None:
            pa_list=self.thislist
        else: #- construct palist
            if self.flavor == "arcs":
                pa_list=['Initialize','Preproc','BootCalibration','BoxcarExtract','ResolutionFit'] #- class names for respective PAs (see desispec.quicklook.procalgs)
            elif self.flavor == "flat":
                pa_list=['Initialize','Preproc','BoxcarExtract', 'ComputeFiberFlat_QL']
            elif self.flavor == "science":
                pa_list=['Initialize','Preproc','BoxcarExtract', 'ApplyFiberFlat_QL','SkySub_QL']
            else:
                log.warning("Not a valid flavor. Use a valid flavor type to build a palist. Exiting.")
                sys.exit(0)
        self.pamodule='desispec.quicklook.procalgs'
        return pa_list       
    

    def _qalist(self):

        if self.thislist is not None:
            qalist={}
            for PA in self.thislist:
                qalist[PA]=self.algorithms[PA]['QA'].keys()
        else:
            if self.flavor =="arcs":
                QAs_initial=['Bias_From_Overscan']
                QAs_preproc=['Get_RMS','Count_Pixels']
                QAs_bootcalib=['Calc_XWSigma']
                QAs_extract=['CountSpectralBins']
            else:
                QAs_initial=['Bias_From_Overscan']
                QAs_preproc=['Get_RMS','Count_Pixels','Calc_XWSigma']
                QAs_extract=['CountSpectralBins']
                QAs_apfiberflat=['Sky_Continuum','Sky_Peaks']
                QAs_SkySub=['Sky_Residual','Integrate_Spec','Calculate_SNR']
        
            qalist={}
            for PA in self.palist:
                if PA == 'Initialize':
                    qalist[PA] = QAs_initial
                elif PA == 'Preproc':
                    qalist[PA] = QAs_preproc
                elif PA == 'BootCalibration':
                    qalist[PA] = QAs_bootcalib
                elif PA == 'BoxcarExtract':
                    qalist[PA] = QAs_extract
                elif PA == 'ResolutionFit':
                    qalist[PA] = QAs_resfit
                elif PA == 'ApplyFiberFlat_QL':
                    qalist[PA] = QAs_apfiberflat
                elif PA == 'SkySub_QL':
                    qalist[PA] = QAs_SkySub
                else:
                    qalist[PA] = None #- No QA for this PA
        self.qamodule='desispec.qa.qa_quicklook'
        return qalist


