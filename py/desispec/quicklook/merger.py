#
# A class to merge quicklook qa outputs.
# 
from __future__ import absolute_import, division, print_function
from desiutil.io import yamlify
import yaml
import json
import numpy as np
import datetime
import pytz
###################################
# SE: added this to facilitate the GENERAL_INFO section
def delKey(d, k, val=None, remove=True):
    
    if isinstance(d, dict):
        key_list = [] 
        for key, value in  d.items(): 
           if key==k:
            
              val = value
              key_list.append(key)
           val = delKey(value, k, val=val, remove=remove)
        if remove:
            for key in key_list:
                del d[key]
   
    elif isinstance(d, list): 
 
        try: 
          for i in range(len(d)): 
             val = delKey(d[i], k, val=val, remove=remove)
        except:
            return val
    
    else: return val
    
    return val

###################################
# SE: added this to facilitate the GENERAL_INFO section

def reOrderDict(mergeDict):
    
  for Night in mergeDict["NIGHTS"]:
      for Exposure in Night["EXPOSURES"]:
          for Camera in Exposure["CAMERAS"]:

             ra  = delKey(Camera, "RA")
             dec = delKey(Camera, "DEC")
             sky_fiberid = delKey(Camera, "SKY_FIBERID")
             skyfiberid = delKey(Camera, "SKYFIBERID")
             airmass = delKey(Camera, "AIRMASS")
             seeing = delKey(Camera, "SEEING")
             exptime = delKey(Camera, "EXPTIME")
             desispec_run_ver = delKey(Camera, "PROC_DESISPEC_VERSION") # desispec version in the raw FITS header 
             desispec_fits_ver = delKey(Camera, "FITS_DESISPEC_VERSION") # desispec version of the software release
             quicklook_run_ver = delKey(Camera, "PROC_QuickLook_VERSION") # version of the quivklook development state
             
             if sky_fiberid is None:
                 sky_fiberid = skyfiberid
             
             elg_fiberid = delKey(Camera, "ELG_FIBERID")
             lrg_fiberid = delKey(Camera, "LRG_FIBERID") 
             qso_fiberid = delKey(Camera, "QSO_FIBERID") 
             star_fiberid = delKey(Camera, "STAR_FIBERID", remove=False)
             
             std_fiberid = delKey(Camera, "STD_FIBERID", remove=False)
             
             if star_fiberid is None:
                 star_fiberid = std_fiberid
             
             b_peaks = delKey(Camera, "B_PEAKS") 
             r_peaks = delKey(Camera, "R_PEAKS")
             z_peaks = delKey(Camera, "Z_PEAKS")
            
             try: ra = [float("%.5f" % m) for m in ra]
             except: ra=None
             
             try: dec = [float("%.5f" % m) for m in dec]
             except: dec=None
             
             #placeholder for mags
             imaging_mag=[22.]*500
             
             # Date/time of the merger i.e., QL run - time is in UTC = Mayall local time + 7h
             def utcnow():
               return datetime.datetime.now(tz=pytz.utc)
             
             QLrun_datime = utcnow().isoformat()

             datetime.datetime.now(datetime.timezone.utc)
             datetime.datetime.now(tz=pytz.utc)
             Camera["GENERAL_INFO"]={"QLrun_datime_UTC":QLrun_datime ,"SEEING":seeing,"AIRMASS":airmass,"EXPTIME":exptime,"FITS_DESISPEC_VERSION":desispec_fits_ver,"PROC_DESISPEC_VERSION":desispec_run_ver,"PROC_QuickLook_VERSION":quicklook_run_ver, "RA":ra, "DEC":dec, "SKY_FIBERID":sky_fiberid, "ELG_FIBERID":elg_fiberid ,"LRG_FIBERID":lrg_fiberid, "QSO_FIBERID":qso_fiberid ,"STAR_FIBERID":star_fiberid ,"B_PEAKS":b_peaks ,"R_PEAKS":r_peaks ,"Z_PEAKS":z_peaks,"IMAGING_MAG": imaging_mag}   


class QL_QAMerger:
    def __init__(self,night,expid,flavor,camera,program):
        self.__night=night
        self.__expid=expid
        self.__flavor=flavor
        self.__camera=camera
        self.__program=program
        self.__stepsArr=[]
        self.__schema={'NIGHTS':[{'NIGHT':night,'EXPOSURES':[{'EXPID':expid,'FLAVOR':flavor,'PROGRAM':program, 'CAMERAS':[{'CAMERA':camera, 'PIPELINE_STEPS':self.__stepsArr}]}]}]}
        
        
    class QL_Step:
        def __init__(self,paName,paramsDict,metricsDict):
            self.__paName=paName
            self.__pDict=paramsDict
            self.__mDict=metricsDict
        def getStepName(self):
            return self.__paName
        def addParams(self,pdict):
            self.__pDict.update(pdict)
        def addMetrics(self,mdict):
            self.__mDict.update(mdict)
    def addPipelineStep(self,stepName):
        metricsDict={}
        paramsDict={}
        stepDict={"PIPELINE_STEP":stepName.upper(),'METRICS':metricsDict,'PARAMS':paramsDict}
        self.__stepsArr.append(stepDict)
        return self.QL_Step(stepName,paramsDict,metricsDict)

    #def getYaml(self):
        #yres=yamlify(self.__schema)
        #reOrderDict(yres)
        #return yaml.dump(yres)
    #def getJson(self):
        #import json
        #return json.dumps(yamlify(self.__schema))
    #def writeToFile(self,fileName):
        #with open(fileName,'w') as f:
            #f.write(self.getYaml())
    def writeTojsonFile(self,fileName):
        g=open(fileName.split('.yaml')[0]+'.json',"w")
        myDict = yamlify(self.__schema)
        reOrderDict(myDict)
        json.dump(myDict, g, sort_keys=True, indent=4)
        g.close()   
