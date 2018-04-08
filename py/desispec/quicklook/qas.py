from desispec.quicklook import qllogger 
from desispec.quicklook import qlexceptions
import collections
import numpy as np
from enum import Enum

class QASeverity(Enum):
    ALARM=30
    WARNING=20
    NORMAL=0

class MonitoringAlg:
    """ Simple base class for monitoring algorithms """
    def __init__(self,name,inptype,config,logger=None):
        if logger is None:
            self.m_log=qllogger.QLLogger().getlog(name)
        else:
            self.m_log=logger
        self.__inpType__=type(inptype)
        self.name=name
        self.config=config
        self.__deviation=None
        self.m_log.debug("initializing Monitoring alg {}".format(name))
    def __call__(self,*args,**kwargs):
        res=self.run(*args,**kwargs)
        res["QA_STATUS"]="UNKNOWN"
        cargs=self.config['kwargs']
        params=cargs['param']
        metrics=res["METRICS"] if 'METRICS' in res else None
        if metrics is None:
            metrics={}
            res["METRICS"]=metrics
        deviation=None
        reskey="RESULT"
        QARESULTKEY="QA_STATUS"
        if "SAMI_QASTATUSKEY" in cargs:
            QARESULTKEY=cargs["SAMI_QASTATUSKEY"]
        if "SAMI_RESULTKEY" in cargs:
            reskey=cargs["SAMI_RESULTKEY"]
        if reskey in metrics and "REFERENCE" in params:
            current=metrics[reskey]
            old=params["REFERENCE"]
            currlist=isinstance(current,(np.ndarray,collections.Sequence))
            oldlist=isinstance(old,(np.ndarray,collections.Sequence))
            if currlist != oldlist: # different types
                self.m_log.critical("QL {} : REFERENCE({}) and RESULT({}) are of different types!".format(self.name,type(old),type(current)))
            elif currlist: #both are lists
                if len(old)==len(current):
                    self.__deviation=[c-o for c,o in zip(current,old)]
                else:
                    self.m_log.critical("QL {} : REFERENCE({}) and RESULT({}) are of different length!".format(self.name,len(old),len(current)))
            else: # both are scalars
                self.__deviation=current-old

        # check RANGES given in config and set QA_STATUS keyword
        # it should be a sorted overlapping list of range tuples in the form [ ((interval),QASeverity),((-1.0,1.0),QASeverity.NORMAL),(-2.0,2.0),QAStatus.WARNING)]
        # for multiple results, thresholds should be a list of lists as given above (one range list per result)
        # intervals should be non overlapping.
        # lower bound is inclusive upper bound is exclusive
        # first matching interval will be used
        # if no interval contains the deviation, it will be set to QASeverity.ALARM
        # if RANGES or REFERENCE are not given in config, QA_STATUS will be set to UNKNOWN 
        def findThr(d,t):
            val=QASeverity.ALARM
            for l in t:
                if d>=l[0][0] and d<l[0][1]:
                    val=l[1]
            return val
        if self.__deviation is not None and "RANGES" in cargs:
            thr=cargs["RANGES"]
            metrics[QARESULTKEY]="ERROR"
            thrlist=isinstance(thr[0][0][0],(np.ndarray,collections.Sequence))  #multiple threshols for multiple results
            devlist=isinstance(self.__deviation,(np.ndarray,collections.Sequence))
            if devlist!=thrlist and len(thr)!=1:  #different types and thresholds are a list
                self.m_log.critical("QL {} : dimension of RANGES({}) and RESULTS({}) are incompatible! Check configuration RANGES={}, RESULTS={}".format(self.name,len(thr),len(self.__deviation), thr,current))
                return res
            else: #they are of the same type
                if devlist: # if results are a list
                    if len(thr)==1: # check all results against same thresholds
                        metrics[QARESULTKEY]=[findThr(d,thr) for d in self.__deviation] 
                    else: # each result has its own thresholds
                        metrics[QARESULTKEY]=[str(findThr(d,t)) for d,t in zip(self.__deviation,thr)]
                else: #result is a scalar
                    metrics[QARESULTKEY]=str(findThr(self.__deviation,thr))
        return res
    def run(self,*argv,**kwargs):
        pass
    def is_compatible(self,Type):
        return isinstance(Type,self.__inpType__)
    def check_reference():
        return self.__deviation
    def get_default_config(self):
        """ return a dictionary of 3-tuples,
        field 0 is the name of the parameter
        field 1 is the default value of the parameter
        field 2 is the comment for human readable format.
        Field 2 can be used for QLF to dynamically setup the display"""
        return None
