import numpy as np
import yaml
from desispec.io import findfile
import os,sys
from desispec.quicklook import qlexceptions,qllogger

qlog=qllogger.QLLogger("QuickLook",20)
log=qlog.getlog()


class Config(object):
    """ 
    A class to generate Quicklook configurations for a given desi exposure. build_config will call this object to generate a configuration needed by quicklook
    """

    def __init__(self, night,flavor,expid,camera,palist,debuglevel=20,period=5.,psfboot=None,wavelength=None, dumpintermediates=True,amps=True,rawdata_dir=None,specprod_dir=None, outdir=None,timeout=120.,fiberflat=None,outputfile=None,qlf=False):
        """
        psfboot- does not seem to have a desispec.io.findfile entry, so passing this in argument. 
                 May be this will be useful even so.
        palist: Palist object. See class Palist below
        Note: fiberflat will have a different expid. Passing the file directly in the path
        """  
  
        self.night=night
        self.expid=expid
        self.flavor=flavor
        self.camera=camera
        self.psfboot=psfboot
        self.fiberflat=fiberflat
        self.outputfile=outputfile #- final outputfile.
        self.wavelength=wavelength
        self.debuglevel=debuglevel
        self.period=period
        self.dumpintermediates=dumpintermediates
        self.amps=amps

        if rawdata_dir is None:
            rawdata_dir=os.getenv('DESI_SPECTRO_DATA')
        self.rawdata_dir=rawdata_dir 

        if specprod_dir is None:
            specprod_dir=os.path.join(os.getenv('DESI_SPECTRO_REDUX'), os.getenv('SPECPROD'))
        self.specprod_dir=specprod_dir

        self.outdir=outdir
        self.timeout=timeout
        self._palist=palist
        self.pamodule=palist.pamodule
        self.qamodule=palist.qamodule
        self._qlf=qlf

        #- some global variables
        self.rawfile=findfile("raw",night=self.night,expid=self.expid, camera=self.camera, rawdata_dir=self.rawdata_dir,specprod_dir=self.specprod_dir)

        self.fibermap=findfile("fibermap", night=self.night,expid=self.expid,camera=self.camera, rawdata_dir=self.rawdata_dir,specprod_dir=self.specprod_dir)


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
            if self.camera[0] == 'r':
                self.wavelength='5630,7740,0.8'
            elif self.camera[0] == 'b':
                self.wavelength='3550,5730,0.8'
            elif self.camera[0] == 'z':
                self.wavelength='7650,9840,0.8'

        paopt_initialize={'camera': self.camera}

        paopt_preproc={'camera': self.camera, 'DumpIntermediates': self.dumpintermediates, 'dumpfile': self.dump_pa("pix")} 

        paopt_extract={'BoxWidth': 2.5, 'FiberMap': self.fibermap, 'Wavelength': self.wavelength, 'Nspec': 500, 'PSFFile': self.psfboot, 'DumpIntermediates': self.dumpintermediates, 'dumpfile': self.dump_pa("frame")}

        paopt_apfflat={'FiberFlatFile': self.fiberflat, 'DumpIntermediates': self.dumpintermediates, 'dumpfile': self.dump_pa("fframe")}
       
        paopt_skysub={'DumpIntermediates': self.dumpintermediates,'dumpfile': self.dump_pa("sframe")}

        paopts={}
        for PA in self.palist:
            if PA=='Initialize':
                paopts[PA]=paopt_initialize
            elif PA=='Preproc':
                paopts[PA]=paopt_preproc
            elif PA=='BoxcarExtraction':
                paopts[PA]=paopt_extract
            elif PA=='ApplyFiberFlat_QL':
                paopts[PA]=paopt_apfflat
            elif PA=='SubtractSky_QL':
                paopts[PA]=paopt_skysub
            else:
                paopts[PA]={}
        return paopts 
        
    def dump_pa(self,filetype):
        """
        dump the PA outputs to respective files. This has to be updated for fframe and sframe files as QL anticipates for dumpintermediate case.
        """
        if filetype in ["fframe","sframe"]: #- fiberflat fielded or sky subtracted intermediate files       
            pafile=os.path.join(self.specprod_dir,'exposures',self.night,"%08d"%self.expid,"%s-%s-%08d.fits"%(filetype,self.camera,self.expid))
        else:
            pafile=findfile(filetype,night=self.night,expid=self.expid, camera=self.camera, rawdata_dir=self.rawdata_dir,specprod_dir=self.specprod_dir,outdir=self.outdir)
            
        return pafile

    def dump_qa(self): 
        """ 
        yaml outputfile for the set of qas for a given pa

        File naming set by pa names, we don't have a io for these. Where to write them? desispec.io.findfile?
        """

        #- both PA level and QA level outputs

        qa_pa_outfile={}
        qa_pa_outfig={}
        qa_outfile={}
        qa_outfig={}
        for PA in self.palist:
            #- pa level outputs
            qa_pa_outfile[PA]=os.path.join(self.specprod_dir,'exposures',self.night,"%08d"%self.expid,"ql-%s-%s-%08d.yaml"%(PA.lower(),self.camera,self.expid))
            qa_pa_outfig[PA]=os.path.join(self.specprod_dir,'exposures',self.night,"%08d"%self.expid,"ql-%s-%s-%08d.png"%(PA.lower(),self.camera,self.expid))

            #- qa_level output
            for QA in self.qalist[PA]:
                qa_outfile[QA]=os.path.join(self.specprod_dir,'exposures',self.night,"%08d"%self.expid,"qa","ql-%s-%s-%08d.yaml"%(QA.lower(),self.camera,self.expid))
                qa_outfig[QA]=os.path.join(self.specprod_dir,'exposures',self.night,"%08d"%self.expid,"qa","ql-%s-%s-%08d.png"%(QA.lower(),self.camera,self.expid))
                
                #- make path if needed
                path = os.path.normpath(os.path.dirname(qa_outfile[QA]))
                if not os.path.exists(path):
                    os.makedirs(path)

        return ((qa_outfile,qa_outfig),(qa_pa_outfile,qa_pa_outfig))

    @property
    def qaargs(self):

        qaopts={}
        for PA in self.palist:
            for qa in self.qalist[PA]: #- individual QA for that PA
                qaopts[qa]={'camera': self.camera, 'expid': "%08d"%self.expid, 'paname': PA, 'PSFFile': self.psfboot, 'amps': self.amps, 'qafile': self.dump_qa()[0][0][qa],'qafig': self.dump_qa()[0][1][qa], 'FiberMap': self.fibermap, 'qlf': self.qlf}
                
        return qaopts    
        

class Palist(object):
    
    """
    Generate PA list and QA list for the Quicklook Pipeline for the given exposure
    """
    def __init__(self,flavor,mode="online"):
        """
        flavor: flavor can be arc,flat, dark, lrg, elg etc....
        mode: mode can be one in ["online", "offline"]
        """
        self.flavor=flavor
        self.mode=mode
        self.palist=self._palist()
        self.qalist=self._qalist()

    def _palist(self):

        if self.flavor == "arcs":
            pa_list=['Initialize','Preproc','BoxcarExtraction'] #- class names for respective PAs (see desispec.quicklook.procalgs)
        elif self.flavor == "flat":
            pa_list=['Initialize','Preproc','BoxcarExtraction', 'ComputeFiberFlat_QL']
        elif self.flavor in ['dark','elg','lrg','qso','bright','grey','gray','bgs','mws']:
            pa_list=['Initialize','Preproc','BoxcarExtraction', 'ApplyFiberFlat_QL','SubtractSky_QL']
        else:
            log.warning("Not a valid flavor type. Use a valid flavor type. Exiting.")
            sys.exit(0)
        self.pamodule='desispec.quicklook.procalgs'
        return pa_list       
    

    def _qalist(self):
        QAs_initial=['Bias_From_Overscan']
        QAs_preproc=['Get_RMS','Count_Pixels','Calc_XWSigma']
        QAs_extract=['CountSpectralBins']
        QAs_apfiberflat=['Sky_Continuum','Sky_Peaks']
        QAs_subtractsky=['Sky_Residual','Integrate_Spec','Calculate_SNR']
        
        qalist={}
        for PA in self.palist:
            if PA == 'Initialize':
                qalist[PA] = QAs_initial
            elif PA == 'Preproc':
                qalist[PA] = QAs_preproc
            elif PA == 'BoxcarExtraction':
                qalist[PA] = QAs_extract
            elif PA == 'ApplyFiberFlat_QL':
                qalist[PA] = QAs_apfiberflat
            elif PA == 'SubtractSky_QL':
                qalist[PA] = QAs_subtractsky
            else:
                qalist[PA] = None #- No QA for this PA
        self.qamodule='desispec.qa.qa_quicklook'
        return qalist
                
                         
def build_config(config):
    """
    config: desispec.quicklook.qlconfig.Config object
    """
    log.info("Building Configuration")

    outconfig={}

    outconfig['Camera'] = config.camera
    #DataType=config.datatype
    outconfig['DumpIntermediates'] = config.dumpintermediates
    outconfig['FiberMap']=config.fibermap
    outconfig['FiberFlatFile'] = config.fiberflat
    outconfig['PSFFile'] = config.psfboot
    outconfig['Period'] = config.period
    if config.outputfile is not None: #- Global final output file
        outconfig["OutputFile"] = config.outputfile
    else: outconfig["OutputFile"]="lastframe-%s-%08d.fits"%(config.camera,config.expid)

    pipeline = []
    for ii,PA in enumerate(config.palist):
        pipe={'OutputFile': config.dump_qa()[1][0][PA]}
        pipe['PA'] = {'ClassName': PA, 'ModuleName': config.pamodule, 'Name': PA, 'kwargs': config.paargs[PA]}
        pipe['QAs']=[]
        for jj, QA in enumerate(config.qalist[PA]):
            pipe_qa={'ClassName': QA, 'ModuleName': config.qamodule, 'Name': QA, 'kwargs': config.qaargs[QA]}
            pipe['QAs'].append(pipe_qa)
        pipe['StepName']=PA
        pipeline.append(pipe)

    outconfig['PipeLine']=pipeline
    outconfig['RawImage']=config.rawfile
    outconfig['Timeout']=config.timeout
    return outconfig        
     
            
        
        


    
                

        
        
