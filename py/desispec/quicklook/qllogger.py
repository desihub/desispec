import logging
#from datetime import datetime

class QLLogger:
    """ Simple logger class using logging """
    __loglvl__=None
    __loggername__="QuickLook"
    def __init__(self,name=None,loglevel=logging.INFO):
        if name is not None:
            self.__loggername__=name
        if self.__loglvl__ is None:
            self.__loglvl__=loglevel
        format = '%(asctime)-15s %(name)s %(levelname)s : %(message)s'
        logging.basicConfig(format=format,level=self.__loglvl__)
    def getlog(self,name=None):
        if name is None:
            loggername=self.__loggername__
        else:
            loggername=name
        return logging.getLogger(loggername)
