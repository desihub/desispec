#
# A class to merge quicklook qa outputs.
#  
# Author: Sami Kama
#
from __future__ import absolute_import, division, print_function
from desiutil.io import yamlify
import yaml
import json

class QL_QAMerger:
    def __init__(self,night,expid,flavor,camera):
        self.__night=night
        self.__expid=expid
        self.__flavor=flavor
        self.__camera=camera
        self.__stepsArr=[]
        self.__schema={'NIGHTS':[{'NIGHT':night,'EXPOSURES':[{'EXPID':expid,'FLAVOR':flavor,'CAMERAS':[{'CAMERA':camera,'PIPELINE_STEPS':self.__stepsArr}]}]}]}
        
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

    def getYaml(self):
        yres=yamlify(self.__schema)
        return yaml.dump(yres)
    def getJson(self):
        import json
        return json.dumps(yamlify(self.__schema))
    def writeToFile(self,fileName):
        with open(fileName,'w') as f:
            f.write(self.getYaml())
    def writeTojsonFile(self,fileName):
        g=open(fileName.split('.yaml')[0]+'.json',"w")
        json.dump(yamlify(self.__schema), g, sort_keys=True, indent=4)
        g.close()   