"""
desispec.quicklook.qlconfig
===========================

"""
import numpy as np
import json
import yaml
import astropy.io.fits as pyfits
from desiutil.log import get_logger
from desispec.io import findfile
from desispec.calibfinder import CalibFinder
import os,sys
from desispec.quicklook import qlexceptions,qllogger

class Config(object):
    """
    A class to generate Quicklook configurations for a given desi exposure.
    expand_config will expand out to full format as needed by quicklook.setup
    """
    def __init__(self, configfile, night, camera, expid, singqa, amps=True,rawdata_dir=None,specprod_dir=None, outdir=None,qlf=False,psfid=None,flatid=None,templateid=None,templatenight=None,qlplots=False,store_res=None):
        """
        configfile: a configuration file for QL eg: desispec/data/quicklook/qlconfig_dark.yaml
        night: night for the data to process, eg.'20191015'
        camera: which camera to process eg 'r0'
        expid: exposure id for the image to be processed
        amps: for outputing amps level QA
        Note:
        rawdata_dir and specprod_dir: if not None, overrides the standard DESI convention
        """
        with open(configfile, 'r') as f:
            self.conf = yaml.safe_load(f)
            f.close()
        self.night = night
        self.expid = expid
        self.psfid = psfid
        self.flatid = flatid
        self.templateid = templateid
        self.templatenight = templatenight
        self.camera = camera
        self.singqa = singqa
        self.amps = amps
        self.rawdata_dir = rawdata_dir
        self.specprod_dir = specprod_dir
        self.outdir = outdir
        self.flavor = self.conf["Flavor"]

        #- Options to write out frame, fframe, preproc, and sky model files
        self.dumpintermediates = False
        self.writepreprocfile = self.conf["WritePreprocfile"]
        self.writeskymodelfile = False

        self.plotconf = None
        self.hardplots = False
        #- Load plotting configuration file
        if qlplots != 'noplots' and qlplots is not None:
            with open(qlplots, 'r') as pf:
                self.plotconf = yaml.safe_load(pf)
                pf.close()
        #- Use hard coded plotting algorithms
        elif qlplots is None:
            self.hardplots = True

        # Use --resolution to store full resolution informtion
        if store_res:
            self.usesigma = True
        else:
            self.usesigma = False

        self.pipeline = self.conf["Pipeline"]
        self.algorithms = self.conf["Algorithms"]
        self._palist = Palist(self.pipeline,self.algorithms)
        self.pamodule = self._palist.pamodule
        self.qamodule = self._palist.qamodule

        algokeys = self.algorithms.keys()

        # Extract mapping of scalar/refence key names for each QA
        qaRefKeys = {}
        for i in algokeys:
            for k in self.algorithms[i]["QA"].keys():
                if k == "Check_HDUs":
                    qaRefKeys[k] = "CHECKHDUS"
                qaparams=self.algorithms[i]["QA"][k]["PARAMS"]
                for par in qaparams.keys():
                    if "NORMAL_RANGE" in par:
                        scalar = par.replace("_NORMAL_RANGE","")
                        qaRefKeys[k] = scalar

        # Special additional parameters to read in.
        self.wavelength = None
        for key in ["BoxcarExtract","Extract_QP"] :
            if key in self.algorithms.keys():
                if "wavelength" in self.algorithms[key].keys():
                    self.wavelength = self.algorithms[key]["wavelength"][self.camera[0]]

        self._qlf=qlf
        qlog=qllogger.QLLogger(name="QLConfig")
        self.log=qlog.getlog()
        self._qaRefKeys = qaRefKeys

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

        peaks=None
        if 'Initialize' in self.algorithms.keys():
            if 'PEAKS' in self.algorithms['Initialize'].keys():
                peaks=self.algorithms['Initialize']['PEAKS']
        if self.flavor == 'bias' or self.flavor == 'dark':
            paopt_initialize={'Flavor':self.flavor,'Camera':self.camera}
        else:
            paopt_initialize={'Flavor':self.flavor,'FiberMap':self.fibermap,'Camera':self.camera,'Peaks':peaks}

        if self.writepreprocfile:
            preprocfile=self.dump_pa("Preproc")
        else:
            preprocfile = None
        paopt_preproc={'camera': self.camera,'dumpfile': preprocfile}

        if self.dumpintermediates:
            framefile=self.dump_pa("BoxcarExtract")
            fframefile=self.dump_pa("ApplyFiberFlat_QL")
            qlsframefile=self.dump_pa("SkySub_QL")
            qframefile=self.dump_pa("Extract_QP")
            fframefile=self.dump_pa("ApplyFiberFlat_QP")
            sframefile=self.dump_pa("SkySub_QP")

        else:
            qframefile=None
            framefile=None
            fframefile=None
            qlsframefile=None
            sframefile=None

        if self.flavor == 'arcs':
            arcimg=findfile('preproc',night=self.night,expid=self.expid,camera=self.camera,specprod_dir=self.specprod_dir)
            flatimg=self.fiberflat
            psffile=findfile('psf',expid=self.expid,night=self.night,camera=self.camera,specprod_dir=self.specprod_dir)
        else:
            arcimg=None
            flatimg=None
            psffile=None

        preproc_file=findfile('preproc',self.night,self.expid,self.camera,specprod_dir=self.specprod_dir)
        paopt_flexure={'preprocFile':preproc_file, 'inputPSFFile': self.calibpsf, 'outputPSFFile': self.psf_filename}

        paopt_extract={'Flavor': self.flavor, 'BoxWidth': 2.5, 'FiberMap': self.fibermap, 'Wavelength': self.wavelength, 'Nspec': 500, 'PSFFile': self.calibpsf,'usesigma': self.usesigma, 'dumpfile': framefile}

        paopt_extract_qp={'Flavor': self.flavor, 'FullWidth': 7, 'FiberMap': self.fibermap, 'Wavelength': self.wavelength, 'Nspec': 500, 'PSFFile': self.psf_filename,'usesigma': self.usesigma, 'dumpfile': qframefile}

        paopt_resfit={'PSFinputfile': self.psf_filename, 'PSFoutfile': psffile, 'usesigma': self.usesigma}

        paopt_comflat={'outputFile': self.fiberflat}

        paopt_apfflat={'FiberFlatFile': self.fiberflat, 'dumpfile': fframefile}

        cframefile=self.dump_pa("ApplyFluxCalibration")
        paopt_fluxcal={'outputfile': cframefile}

        if self.writeskymodelfile:
            outskyfile = findfile('sky',night=self.night,expid=self.expid, camera=self.camera, rawdata_dir=self.rawdata_dir,specprod_dir=self.specprod_dir,outdir=self.outdir)
        else:
            outskyfile=None
        paopt_skysub={'Outskyfile': outskyfile, 'dumpfile': qlsframefile, 'Apply_resolution': self.usesigma}
        paopt_skysub_qp={'dumpfile': sframefile, 'Apply_resolution': False}

        paopts={}
        defList={
            'Initialize':paopt_initialize,
            'Preproc':paopt_preproc,
            'Flexure':paopt_flexure,
            'BoxcarExtract':paopt_extract,
            'ResolutionFit':paopt_resfit,
            'Extract_QP':paopt_extract_qp,
            'ComputeFiberflat_QL':paopt_comflat,
            'ComputeFiberflat_QP':paopt_comflat,
            'ApplyFiberFlat_QL':paopt_apfflat,
            'ApplyFiberFlat_QP':paopt_apfflat,
            'SkySub_QL':paopt_skysub,
            'SkySub_QP':paopt_skysub_qp,
            'ApplyFluxCalibration':paopt_fluxcal
        }

        def getPAConfigFromFile(PA,algs):
            def mergeDicts(source,dest):
                for k in source:
                    if k not in dest:
                        dest[k]=source[k]
            userconfig={}
            if PA in algs:
                fc=algs[PA]
                for k in fc: #do a deep copy leave QA config out
                    if k != "QA":
                        userconfig[k]=fc[k]
            defconfig={}
            if PA in defList:
                defconfig=defList[PA]
            mergeDicts(defconfig,userconfig)
            return userconfig

        for PA in self.palist:
            paopts[PA]=getPAConfigFromFile(PA,self.algorithms)
        #- Ignore intermediate dumping and write explicitly the outputfile for
        self.outputfile=self.dump_pa(self.palist[-1])

        return paopts

    def dump_pa(self,paname):
        """
        dump the PA outputs to respective files. This has to be updated for fframe and sframe files as QL anticipates for dumpintermediate case.
        """
        pafilemap={'Preproc': 'preproc', 'Flexure': None, 'BoxcarExtract': 'frame','ResolutionFit': None, 'Extract_QP': 'qframe', 'ComputeFiberflat_QL': 'fiberflat', 'ComputeFiberflat_QP': 'fiberflat', 'ApplyFiberFlat_QL': 'fframe', 'ApplyFiberFlat_QP': 'fframe', 'SkySub_QL': 'sframe', 'SkySub_QP': 'sframe', 'ApplyFluxCalibration': 'cframe'}

        if paname in pafilemap:
            filetype=pafilemap[paname]
        else:
            raise IOError("PA name does not match any file type. Check PA name in config")

        pafile=None
        if filetype is not None:
            pafile=findfile(filetype,night=self.night,expid=self.expid,camera=self.camera,rawdata_dir=self.rawdata_dir,specprod_dir=self.specprod_dir,outdir=self.outdir)

        return pafile

    def dump_qa(self):
        """
        yaml outputfile for the set of qas for a given pa
        Name and default locations of files are handled by desispec.io.meta.findfile
        """
        #- QA level outputs
        #qa_outfile = {}
        qa_outfig = {}
        for PA in self.palist:
            for QA in self.qalist[PA]:
                #qa_outfile[QA] = self.io_qa(QA)[0]
                qa_outfig[QA] = self.io_qa(QA)[1]

                #- make path if needed
                path = os.path.normpath(os.path.dirname(qa_outfig[QA]))
                if not os.path.exists(path):
                    os.makedirs(path)

        return (qa_outfig)
#        return ((qa_outfile,qa_outfig),(qa_pa_outfile,qa_pa_outfig))

    @property
    def qaargs(self):
        qaopts = {}
        referencemetrics=[]
        for PA in self.palist:
            for qa in self.qalist[PA]: #- individual QA for that PA
                pa_yaml = PA.upper()
                params=self._qaparams(qa)
                qaopts[qa]={'night' : self.night, 'expid' : self.expid,
                            'camera': self.camera, 'paname': PA, 'PSFFile': self.psf_filename,
                            'amps': self.amps, #'qafile': self.dump_qa()[0][qa],
                            'qafig': self.dump_qa()[qa], 'FiberMap': self.fibermap,
                            'param': params, 'refKey':self._qaRefKeys[qa],
                            'singleqa' : self.singqa,
                            'plotconf':self.plotconf, 'hardplots': self.hardplots
                            }
                if qa == 'Calc_XWSigma':
                    qaopts[qa]['Peaks']=self.algorithms['Initialize']['PEAKS']
                    qaopts[qa]['Flavor']=self.flavor
                    qaopts[qa]['PSFFile']=self.calibpsf
                if qa == 'Sky_Peaks':
                    qaopts[qa]['Peaks']=self.algorithms['Initialize']['PEAKS']
                if self.singqa is not None:
                    qaopts[qa]['rawdir']=self.rawdata_dir
                    qaopts[qa]['specdir']=self.specprod_dir
                    if qa == 'Sky_Residual':
                        skyfile = findfile('sky',night=self.night,expid=self.expid, camera=self.camera, rawdata_dir=self.rawdata_dir,specprod_dir=self.specprod_dir,outdir=self.outdir)
                        qaopts[qa]['SkyFile']=skyfile

                if self.reference != None:
                    refkey=qaopts[qa]['refKey']
                    for padict in range(len(self.reference)):
                        pa_metrics=self.reference[padict].keys()
                        if refkey in pa_metrics:
                            qaopts[qa]['ReferenceMetrics']={'{}'.format(refkey): self.reference[padict][refkey]}
        return qaopts

    def _qaparams(self,qa):
        params={}
        if self.algorithms is not None:
            for PA in self.palist:
                if qa in self.qalist[PA]:
                    params[qa]=self.algorithms[PA]['QA'][qa]['PARAMS']
        else:
            # RK:  Need to settle optimal error handling in cases like this.
            raise qlexceptions.ParameterException("Run time PARAMs not provided for QA")

        return params[qa]

    def io_qa_pa(self,paname):
        """
        Specify the filenames: json and png of the pa level qa files"
        """
        filemap={'Initialize': 'initial',
                 'Preproc': 'preproc',
                 'Flexure': 'flexure',
                 'BoxcarExtract': 'boxextract',
                 'Extract_QP': 'extractqp',
                 'ComputeFiberflat_QL': 'computeflat',
                 'ComputeFiberflat_QP': 'computeflatqp',
                 'ApplyFiberFlat_QL': 'fiberflat',
                 'ApplyFiberFlat_QP': 'fiberflatqp',
                 'SkySub_QL': 'skysub',
                 'SkySub_QP': 'skysubqp',
                 'ResolutionFit': 'resfit',
                 'ApplyFluxCalibration': 'fluxcalib'
                 }

        if paname in filemap:
            outfile=findfile('ql_file',night=self.night,expid=self.expid, camera=self.camera, rawdata_dir=self.rawdata_dir,specprod_dir=self.specprod_dir,outdir=self.outdir)
            outfile=outfile.replace('qlfile',filemap[paname])
            outfig=findfile('ql_fig',night=self.night,expid=self.expid, camera=self.camera, rawdata_dir=self.rawdata_dir,specprod_dir=self.specprod_dir,outdir=self.outdir)
            outfig=outfig.replace('qlfig',filemap[paname])
        else:
            raise IOError("PA name does not match any file type. Check PA name in config for {}".format(paname))

        return (outfile,outfig)


    def io_qa(self,qaname):
        """
        Specify the filenames: json and png for the given qa output
        """
        filemap={'Check_HDUs':'checkHDUs',
                 'Trace_Shifts':'trace',
                 'Bias_From_Overscan': 'getbias',
                 'Get_RMS' : 'getrms',
                 'Count_Pixels': 'countpix',
                 'Calc_XWSigma': 'xwsigma',
                 'CountSpectralBins': 'countbins',
                 'Sky_Continuum': 'skycont',
                 'Sky_Rband': 'skyRband',
                 'Sky_Peaks': 'skypeak',
                 'Sky_Residual': 'skyresid',
                 'Integrate_Spec': 'integ',
                 'Calculate_SNR': 'snr',
                 'Check_Resolution': 'checkres',
                 'Check_FiberFlat': 'checkfibflat'
                 }

        if qaname in filemap:
            outfile=findfile('ql_file',night=self.night,expid=self.expid, camera=self.camera, rawdata_dir=self.rawdata_dir,specprod_dir=self.specprod_dir,outdir=self.outdir)
            outfile=outfile.replace('qlfile',filemap[qaname])
            outfig=findfile('ql_fig',night=self.night,expid=self.expid, camera=self.camera, rawdata_dir=self.rawdata_dir,specprod_dir=self.specprod_dir,outdir=self.outdir)
            outfig=outfig.replace('qlfig',filemap[qaname])
        else:
            raise IOError("QA name does not match any file type. Check QA name in config for {}".format(qaname))

        return (outfile,outfig)

    def expand_config(self):
        """
        config: desispec.quicklook.qlconfig.Config object
        """
        self.log.debug("Building Full Configuration")
        self.debuglevel = self.conf["Debuglevel"]
        self.period = self.conf["Period"]
        self.timeout = self.conf["Timeout"]

        #- some global variables:
        self.rawfile=findfile("raw",night=self.night,expid=self.expid,camera=self.camera,rawdata_dir=self.rawdata_dir,specprod_dir=self.specprod_dir)

        self.fibermap=None
        if self.flavor != 'bias' and self.flavor != 'dark':
            self.fibermap=findfile("fibermap", night=self.night,expid=self.expid,camera=self.camera,rawdata_dir=self.rawdata_dir,specprod_dir=self.specprod_dir)

        hdulist=pyfits.open(self.rawfile)
        primary_header=hdulist[0].header
        camera_header =hdulist[self.camera].header

        self.program=primary_header['PROGRAM']

        hdulist.close()

        cfinder = CalibFinder([camera_header,primary_header])
        if self.flavor == 'dark' or self.flavor == 'bias' or self.flavor == 'zero':
            self.calibpsf=None
        else:
            self.calibpsf=cfinder.findfile("PSF")

        if self.psfid is None:
            self.psf_filename=findfile('psf',night=self.night,expid=self.expid,camera=self.camera,rawdata_dir=self.rawdata_dir,specprod_dir=self.specprod_dir)
        else:
            self.psf_filename=findfile('psf',night=self.night,expid=self.psfid,camera=self.camera,rawdata_dir=self.rawdata_dir,specprod_dir=self.specprod_dir)

        if self.flavor == 'dark' or self.flavor == 'bias' or self.flavor == 'zero':
            self.fiberflat=None
        elif self.flatid is None and self.flavor != 'flat':
            self.fiberflat=cfinder.findfile("FIBERFLAT")
        elif self.flavor == 'flat':
            self.fiberflat=findfile('fiberflat',night=self.night,expid=self.expid,camera=self.camera,rawdata_dir=self.rawdata_dir,specprod_dir=self.specprod_dir)
        else:
            self.fiberflat=findfile('fiberflat',night=self.night,expid=self.flatid,camera=self.camera,rawdata_dir=self.rawdata_dir,specprod_dir=self.specprod_dir)

        #SE: QL no longer get references from a template or merged json
        #- Get reference metrics from template json file
        self.reference=None

        outconfig={}
        outconfig['Night'] = self.night
        outconfig['Program'] = self.program
        outconfig['Flavor'] = self.flavor
        outconfig['Camera'] = self.camera
        outconfig['Expid'] = self.expid
        outconfig['DumpIntermediates'] = self.dumpintermediates
        outconfig['FiberMap'] = self.fibermap
        outconfig['Period'] = self.period

        pipeline = []
        for ii,PA in enumerate(self.palist):
            pipe={}
            pipe['PA'] = {'ClassName': PA, 'ModuleName': self.pamodule, 'kwargs': self.paargs[PA]}
            pipe['QAs']=[]
            for jj, QA in enumerate(self.qalist[PA]):
                pipe_qa={'ClassName': QA, 'ModuleName': self.qamodule, 'kwargs': self.qaargs[QA]}
                pipe['QAs'].append(pipe_qa)
            pipe['StepName']=PA
            pipeline.append(pipe)

        outconfig['PipeLine'] = pipeline
        outconfig['RawImage'] = self.rawfile
        outconfig['singleqa'] = self.singqa
        outconfig['Timeout'] = self.timeout
        outconfig['FiberFlatFile'] = self.fiberflat
        outconfig['PlotConfig'] = self.plotconf

        #- Check if all the files exist for this QL configuraion
        check_config(outconfig,self.singqa)
        return outconfig

def check_config(outconfig,singqa):
    """
    Given the expanded config, check for all possible file existence etc....
    """
    if singqa is None:
        qlog=qllogger.QLLogger(name="QLConfig")
        log=qlog.getlog()
        log.info("Checking if all the necessary files exist.")

        if outconfig["Flavor"]=='science':
            files = [outconfig["RawImage"], outconfig["FiberMap"], outconfig["FiberFlatFile"]]
            for thisfile in files:
                if not os.path.exists(thisfile):
                    sys.exit("File does not exist: {}".format(thisfile))
                else:
                    log.info("File check: Okay: {}".format(thisfile))
        elif outconfig["Flavor"]=="flat":
            files = [outconfig["RawImage"], outconfig["FiberMap"]]
            for thisfile in files:
                if not os.path.exists(thisfile):
                    sys.exit("File does not exist: {}".format(thisfile))
                else:
                    log.info("File check: Okay: {}".format(thisfile))
        log.info("All necessary files exist for {} configuration.".format(outconfig["Flavor"]))

    return

class Palist(object):
    """
    Generate PA list and QA list for the Quicklook Pipeline for the given exposure
    """
    def __init__(self,thislist=None,algorithms=None):
        """
        thislist: given list of PAs
        algorithms: Algorithm list coming from config file: e.g desispec/data/quicklook/qlconfig_dark.yaml
        flavor: only needed if new list is to be built.
        mode: online offline?
        """
        self.thislist=thislist
        self.algorithms=algorithms
        self.palist=self._palist()
        self.qalist=self._qalist()

    def _palist(self):
        palist=self.thislist
        self.pamodule='desispec.quicklook.procalgs'
        return palist

    def _qalist(self):
        qalist={}
        for PA in self.thislist:
            qalist[PA]=self.algorithms[PA]['QA'].keys()
        self.qamodule='desispec.qa.qa_quicklook'
        return qalist


