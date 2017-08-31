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
        if "RESULT" in res and "REFERENCE" in self.config:
            self.__deviation=res["RESULT"] - self.config["REFERENCE"]
        #check THRESHOLDS given in config and set QA_STATUS keyword
        # it should be a sorted list of tuples in the form [ ((interval),"Keyword"),((0.,5.)"OK"),((5.10),"Acceptable"),((10.,15),"broken")]
        # intervals should be non overlapping.
        # lower bound is inclusive upper bound is exclusive
        # first matching interval will be used
        # if no interval contains the deviation, it will be set to "OUTOFBOUNDS"
        # if THRESHOLDS or REFERENCE are not given in config, QA_STATUS will be set to UNKNOWN 
        if "THRESHOLDS" in self.config and self.__deviation is not None:
            res["QA_STATUS"]="OUTOFBOUNDS"
            d=self.__deviation
            for l in self.config["THRESHOLDS"]:
                if d>=l[0][0] and d<l[0][1]:
                    res["QA_STATUS"]=l[1]
                    break
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
