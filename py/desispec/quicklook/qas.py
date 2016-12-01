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
        self.m_log.debug("initializing Monitoring alg {}".format(name))
    def __call__(self,*args,**kwargs):
        return self.run(*args,**kwargs)
    def run(self,*argv,**kwargs):
        pass
    def is_compatible(self,Type):
        return isinstance(Type,self.__inpType__)
    def get_default_config(self):
        """ return a dictionary of 3-tuples,
        field 0 is the name of the parameter
        field 1 is the default value of the parameter
        field 2 is the comment for human readeable format.
        Field 2 can be used for QLF to dynamically setup the display"""
        return None
