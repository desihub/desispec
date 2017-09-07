from desispec.quicklook import qllogger 
from desispec.quicklook import qlexceptions
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
        deviation=None
        if "RESULT" in res and "REFERENCE" in self.config:
            current=resDict["RESULT"]
            old=self.config["REFERENCE"]
            currlist=isinstance(current,list)
            oldlist=isinstance(old,list)
            if currlist != oldlist: # different types
                self.m_log.critical("QL {} : REFERENCE({}) and RESULT({}) are of different types!".format(self.name,type(old),type(current)))
            elif currlist: #both are lists
                if len(old)==len(current):
                    self.__deviation=[abs(c-o) for c,o in zip(current,old)]
                else:
                    self.m_log.critical("QL {} : REFERENCE({}) and RESULT({}) are of different length!".format(self.name,len(old),len(current)))
            else: # both are scalars
                self.__deviation=abs(current-old)

        # check THRESHOLDS given in config and set QA_STATUS keyword
        # it should be a sorted list of tuples in the form [ ((interval),"Keyword"),((0.,5.)"OK"),((5.10),"Acceptable"),((10.,15),"broken")]
        # for multiple results, thresholds should be a list of list as given above (one threshold list per result)
        # intervals should be non overlapping.
        # lower bound is inclusive upper bound is exclusive
        # first matching interval will be used
        # if no interval contains the deviation, it will be set to "OUTOFBOUNDS"
        # if THRESHOLDS or REFERENCE are not given in config, QA_STATUS will be set to UNKNOWN 
        def findThr(d,t):
            for l in t:
                if d>=l[0][0] and d<l[0][1]:
                    return l[1]
            return "OUTOFBOUNDS"
        if self.__deviation and "THRESHOLDS" in self.config:
            thr=self.config["THRESHOLDS"]
            res["QA_STATUS"]="ERROR"
            thrlist=isinstance(thr[0][0][0],list) #multiple threshols for multiple results
            devlist=isinstance(self.__deviation,list)
            if devlist!=thrlist:
                self.m_log.critical("QL {} : dimension of THRESHOLD({}) and RESULTS({}) are incompatible!".format(self.name,len(thr),len(self.__deviation)))
                return res
            else:
                if devlist:
                    res["QA_STATUS"]=[findThr(d,t) for d,t in zip(self.__deviation,thr)]
                else:
                    res["QA_STATUS"]=findThr(self.__deviation,thr)
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
        field 2 is the comment for human readeable format.
        Field 2 can be used for QLF to dynamically setup the display"""
        return None
