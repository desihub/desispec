"""
desispec.quicklook.merger
=========================

A class to merge quicklook qa outputs.
"""
from __future__ import absolute_import, division, print_function
from desiutil.io import yamlify
import yaml
import json
import numpy as np
import datetime
import pytz

###############################################################
def remove_task(myDict, Key):
    if Key in myDict:
        del myDict[Key]
    return myDict
###############################################################
def rename_task(myDict, oldKey, newKey):

    if oldKey in myDict:

        task_data = myDict[oldKey]
        del myDict[oldKey]
        myDict[newKey] = task_data

    return myDict
###############################################################
## KeyHead = "KeyHead" or "PARAMS"

def transferKEY(myDict, KeyHead, old_task, new_task, keyList):

    if old_task in myDict and new_task in myDict:
        for key in keyList:
            if key in myDict[old_task][KeyHead]:
                data = myDict[old_task][KeyHead][key]
                del myDict[old_task][KeyHead][key]
                myDict[new_task][KeyHead][key] = data

    return myDict

###############################################################
### Please Give the correct Re-arrangmenet recipe here ...

def modify_tasks(myDict):

    ################
    ### Moving all keys in keyList under Metrics (from PREPROC to BOXCAREXTRACT)
    keyList = ["XWSIGMA", "XWSIGMA_AMP", "XWSIGMA_STATUS"]
    if "EXTRACT_QP" in myDict:
        myDict = transferKEY(myDict, "METRICS", "EXTRACT_QP", "PREPROC", keyList)
    elif "BOXCAREXTRACT" in myDict:
        myDict = transferKEY(myDict, "METRICS", "BOXCAREXTRACT", "PREPROC", keyList)

    ################
    keyList = ["XWSIGMA_NORMAL_RANGE", "XWSIGMA_REF", "XWSIGMA_WARN_RANGE"]
    if "EXTRACT_QP" in myDict:
        myDict = transferKEY(myDict, "PARAMS", "EXTRACT_QP", "PREPROC",keyList)
    elif "BOXCAREXTRACT" in myDict:
        myDict = transferKEY(myDict, "PARAMS", "BOXCAREXTRACT", "PREPROC",keyList)

    ################
    keyList = ["CHECKHDUS","EXPNUM","CHECKHDUS_STATUS","EXPNUM_STATUS"]
    myDict = transferKEY(myDict, "METRICS", "INITIALIZE", "PREPROC", keyList)

    ################

    keyList = ["XYSHIFTS","XYSHIFTS_STATUS"]
    if "EXTRACT_QP" in myDict:
        myDict = transferKEY(myDict, "METRICS", "FLEXURE", "EXTRACT_QP", keyList)
    elif "BOXCAREXTRACT" in myDict:
        myDict = transferKEY(myDict, "METRICS", "FLEXURE", "BOXCAREXTRACT", keyList)

    ################
    keyList = ["XYSHIFTS_NORMAL_RANGE", "XYSHIFTS_WARN_RANGE", "XYSHIFTS_DARK_REF", "XYSHIFTS_GRAY_REF","XYSHIFTS_BRIGHT_REF"]
    if "EXTRACT_QP" in myDict:
        myDict = transferKEY(myDict, "PARAMS", "FLEXURE", "EXTRACT_QP", keyList)
    elif "BOXCAREXTRACT" in myDict:
        myDict = transferKEY(myDict, "PARAMS", "FLEXURE", "BOXCAREXTRACT", keyList)

    ################
    keyList = ["PEAKCOUNT","PEAKCOUNT_FIB","PEAKCOUNT_NOISE","PEAKCOUNT_STATUS","SKYCONT","SKYCONT_FIBER","SKYCONT_STATUS","SKYRBAND","SKY_RFLUX_DIFF","SKY_FIB_RBAND","FIDSNR_TGT","FIDSNR_TGT_STATUS","FITCOEFF_TGT","MEDIAN_SNR","NUM_NEGATIVE_SNR","SNR_MAG_TGT","SNR_RESID","OBJLIST"]
    if "APPLYFIBERFLAT_QP" in myDict:
        myDict = transferKEY(myDict, "METRICS", "APPLYFIBERFLAT_QP", "SKYSUB_QP", keyList)
        myDict = transferKEY(myDict, "METRICS", "SKYSUB_QP", "APPLYFLUXCALIBRATION", keyList)
    elif "APPLYFIBERFLAT_QL" in myDict:
        myDict = transferKEY(myDict, "METRICS", "APPLYFIBERFLAT_QL", "SKYSUB_QL", keyList)
        myDict = transferKEY(myDict, "METRICS", "SKYSUB_QL", "APPLYFLUXCALIBRATION", keyList)

    ################
    keyList = ["B_CONT","R_CONT","Z_CONT","PEAKCOUNT_NORMAL_RANGE","PEAKCOUNT_BRIGHT_REF","PEAKCOUNT_DARK_REF","PEAKCOUNT_GRAY_REF","PEAKCOUNT_WARN_RANGE","SKYCONT_NORMAL_RANGE","SKYCONT_REF","SKYCONT_WARN_RANGE","SKYCONT_BRIGHT_REF","SKYCONT_DARK_REF","SKYCONT_GRAY_REF","RESIDUAL_CUT","SIGMA_CUT","FIDSNR_TGT_NORMAL_RANGE","FIDSNR_TGT_WARN_RANGE","FIDSNR_TGT_BRIGHT_REF","FIDSNR_TGT_DARK_REF","FIDSNR_TGT_GRAY_REF","FIDMAG"]
    if "APPLYFIBERFLAT_QP" in myDict:
        myDict = transferKEY(myDict, "PARAMS", "APPLYFIBERFLAT_QP", "SKYSUB_QP", keyList)
        myDict = transferKEY(myDict, "PARAMS", "SKYSUB_QP", "APPLYFLUXCALIBRATION", keyList)
    elif "APPLYFIBERFLAT_QL" in myDict:
        myDict = transferKEY(myDict, "PARAMS", "APPLYFIBERFLAT_QL", "SKYSUB_QL", keyList)
        myDict = transferKEY(myDict, "PARAMS", "SKYSUB_QL", "APPLYFLUXCALIBRATION", keyList)

    ### Changing Task Names
    myDict = rename_task(myDict, "PREPROC", "CHECK_CCDs")
    myDict = rename_task(myDict, "BOXCAREXTRACT", "CHECK_FIBERS")
    myDict = rename_task(myDict, "EXTRACT_QP", "CHECK_FIBERS")
    myDict = rename_task(myDict, "APPLYFLUXCALIBRATION", "CHECK_SPECTRA")
    myDict = rename_task(myDict, "RESOLUTIONFIT", "CHECK_ARC")
    myDict = rename_task(myDict, "COMPUTEFIBERFLAT_QL", "CHECK_FIBERFLAT")
    myDict = rename_task(myDict, "COMPUTEFIBERFLAT_QP", "CHECK_FIBERFLAT")
    ### Removing empty (or unused Pipeline steps
    myDict = remove_task(myDict, "FLEXURE")
    myDict = remove_task(myDict, "APPLYFIBERFLAT_QL")
    myDict = remove_task(myDict, "APPLYFIBERFLAT_QP")
    myDict = remove_task(myDict, "SKYSUB_QL")
    myDict = remove_task(myDict, "SKYSUB_QP")
    myDict = remove_task(myDict, "INITIALIZE")

    return myDict


###############################################################
### Replacing "PIPELINE_STEPS" with "TASKS"
### Re-ordering Task metrics and Params

def taskMaker(myDict):

    if "PIPELINE_STEPS" in myDict:

        tasks = {}
        task_data = myDict["PIPELINE_STEPS"]

        task_data = modify_tasks(task_data)

        del myDict["PIPELINE_STEPS"]
        myDict["TASKS"] = task_data

    return myDict
###############################################################


###################################
# GENERAL_INFO section
#def delKey(d, k, val=None, remove=True):

    #if isinstance(d, dict):
        #key_list = []
        #for key, value in  d.items():
           #if key==k:

              #val = value
              #key_list.append(key)
           #val = delKey(value, k, val=val, remove=remove)
        #if remove:
            #for key in key_list:
                #del d[key]

    #elif isinstance(d, list):

        #try:
          #for i in range(len(d)):
             #val = delKey(d[i], k, val=val, remove=remove)
        #except:
            #return val

    #else: return val

    #return val



def delKey(d, k, val=None, remove=True, include=False):

    if isinstance(d, dict):
        key_list = []
        for key, value in  d.items():
           if (key==k and not include) or (k in key and include):

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
# facilitate the GENERAL_INFO section

def reOrderDict(mergeDict):

  for Night in mergeDict["NIGHTS"]:
      for Exposure in Night["EXPOSURES"]:
          for Camera in Exposure["CAMERAS"]:

             ra  = delKey(Camera, "RA")
             dec = delKey(Camera, "DEC")
             program = delKey(Camera, "PROGRAM")

             airmass = delKey(Camera, "AIRMASS")
             seeing = delKey(Camera, "SEEING")
             exptime = delKey(Camera, "EXPTIME")
             desispec_run_ver = delKey(Camera, "PROC_DESISPEC_VERSION") # desispec version in the raw FITS header
             desispec_fits_ver = delKey(Camera, "FITS_DESISPEC_VERSION") # desispec version of the software release
             quicklook_run_ver = delKey(Camera, "PROC_QuickLook_VERSION") # version of the quicklook development state
             fibermags = delKey(Camera,"FIBER_MAGS")
             skyfib_id = delKey(Camera,"SKYFIBERID")
             nskyfib = delKey(Camera,"NSKY_FIB")

             delKey(Camera, "SKYSUB_QL")
             delKey(Camera, "MED_RESID")
             delKey(Camera, "MED_RESID_FIBER")
             delKey(Camera, "MED_RESID_WAVE")
             delKey(Camera, "MED_RESID")
             delKey(Camera, "MED_RESID_FIBER")
             delKey(Camera, "RESID_PER")
             delKey(Camera, "RESID_STATUS")
             delKey(Camera, "BIAS")
             delKey(Camera, "NOISE")

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


             # Date/time of the merger i.e., QL run - time is in UTC = Mayall local time + 7h
             def utcnow():
               return datetime.datetime.now(tz=pytz.utc)

             QLrun_datime = utcnow().isoformat()

             datetime.datetime.now(datetime.timezone.utc)
             datetime.datetime.now(tz=pytz.utc)


             Camera["GENERAL_INFO"]={"QLrun_datime_UTC":QLrun_datime,"PROGRAM":format(program).upper(),"SEEING":seeing,"AIRMASS":airmass,"EXPTIME":exptime,"FITS_DESISPEC_VERSION":desispec_fits_ver,"PROC_DESISPEC_VERSION":desispec_run_ver,"PROC_QuickLook_VERSION":quicklook_run_ver,"RA":ra,"DEC":dec,"SKY_FIBERID":skyfib_id,"ELG_FIBERID":elg_fiberid,"LRG_FIBERID":lrg_fiberid,"QSO_FIBERID":qso_fiberid,"STAR_FIBERID":star_fiberid,"B_PEAKS":b_peaks,"R_PEAKS":r_peaks,"Z_PEAKS":z_peaks,"FIBER_MAGS":fibermags,"NSKY_FIB":nskyfib}

###################################

def EditDic(Camera):
    desispec_run_ver = delKey(Camera, "PROC_DESISPEC_VERSION") # desispec version in the raw FITS header
    desispec_fits_ver = delKey(Camera, "FITS_DESISPEC_VERSION") # desispec version of the software release
    quicklook_run_ver = delKey(Camera, "PROC_QuickLook_VERSION") # version of the quivklook development state

    delKey(Camera, "SKYSUB_QL")
    delKey(Camera, "MED_RESID")
    delKey(Camera, "MED_RESID_FIBER")
    delKey(Camera, "MED_RESID_WAVE")
    delKey(Camera, "MED_RESID")
    delKey(Camera, "MED_RESID_FIBER")
    delKey(Camera, "RESID_PER")
    delKey(Camera, "RESID_STATUS")
    delKey(Camera, "BIAS")
    delKey(Camera, "NOISE")
    delKey(Camera, "XWSHIFT_AMP")
    delKey(Camera, "XWSIGMA_SHIFT")
    delKey(Camera, "NREJ")
    delKey(Camera, "MED_SKY")
    delKey(Camera, "NBAD_PCHI")

    all_Steps=delKey(Camera,"PIPELINE_STEPS")   # returns a list of dictionaries, each holding one step
    step_dict={}
    for step in all_Steps:
        if step['PIPELINE_STEP'] == 'INITIALIZE':
            Camera['GENERAL_INFO']=delKey(step,"METRICS",remove=False,include=True)
        else:
            step_Name=delKey(step,"PIPELINE_STEP")
            step_dict[step_Name]=step
    Camera["PIPELINE_STEPS"]=step_dict

    program=Camera['GENERAL_INFO']['PROGRAM']
    sciprog = ["DARK","GRAY","BRIGHT"]
    QAlist=["BIAS_AMP","LITFRAC_AMP","NOISE_AMP","XWSIGMA","XYSHIFTS","NGOODFIB","DELTAMAG_TGT","FIDSNR_TGT","SKYRBAND","PEAKCOUNT", "SKYCONT"]

    if program in sciprog:
        sciprog.remove(program)
        for prog in sciprog:
            for qa in QAlist:
                delKey(Camera,qa+'_'+prog+"_REF",include=True)

    Camera["GENERAL_INFO"]["FITS_DESISPEC_VERSION"]=desispec_fits_ver
    Camera["GENERAL_INFO"]["PROC_DESISPEC_VERSION"]=desispec_run_ver
    Camera["GENERAL_INFO"]["PROC_QuickLook_VERSION"]=quicklook_run_ver

###################################


class QL_QAMerger:
    def __init__(self,night,expid,flavor,camera,program,convdict):
        self.__night=night
        self.__expid=expid
        self.__flavor=flavor
        self.__camera=camera
        self.__program=program
        self.__stepsArr=[]
        #self.__schema={'NIGHTS':[{'NIGHT':night,'EXPOSURES':[{'EXPID':expid,'FLAVOR':flavor,'PROGRAM':program, 'CAMERAS':[{'CAMERA':camera, 'PIPELINE_STEPS':self.__stepsArr}]}]}]}

        #general_Info = esnEditDic(self.__stepsArr)

        # Get flux information from fibermap and convert to fiber magnitudes
        if flavor == 'science':
            if camera[0].lower()=='b':decamfilter='G'
            elif camera[0].lower()=='r': decamfilter='R'
            elif camera[0].lower()=='z': decamfilter='Z'
            self.__schema={'NIGHT':night,'EXPID':expid,'CAMERA':camera,'FLAVOR':flavor,'PIPELINE_STEPS':self.__stepsArr}
        else:
            self.__schema={'NIGHT':night,'EXPID':expid,'CAMERA':camera,'FLAVOR':flavor,'PIPELINE_STEPS':self.__stepsArr}

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



    def writeTojsonFile(self,fileName):
        g=open(fileName,'w')


        myDict = yamlify(self.__schema)
        #reOrderDict(myDict)

        # remove lists ... after this step there is no list of dictionaries
        EditDic(myDict)

        # this step modifies Takse, renames them, and re-arrange Metrics and corresponding Paramas
        myDict = taskMaker(myDict)

        json.dump(myDict, g, sort_keys=True, indent=4)
        g.close()
