import logging
#from datetime import datetime

class QLLogger:
    """ Simple logger class using logging """
    __logLvl__=None
    __loggerName__="QuickLook"
    def __init__(self,name=None,logLevel=logging.INFO):
        if name is not None:
            self.__loggerName__=name
        if self.__logLvl__ is None:
            self.__logLvl__=logLevel
        FORMAT = '%(asctime)-15s %(name)s %(levelname)s : %(message)s'
        logging.basicConfig(format=FORMAT,level=self.__logLvl__)
    def getLog(self,name=None):
        if name is None:
            loggerName=self.__loggerName__
        else:
            loggerName=name
        return logging.getLogger(loggerName)
