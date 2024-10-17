"""
desispec.quicklook.qlheartbeat
==============================

"""
from threading import Thread
import time

class QLHeartbeat:
    def __init__(self,logger,beatinterval,timeout,precision=0.1,level=20):
        self.__logger__=logger
        self.__bint__=beatinterval
        self.__timeout__=timeout
        self.__message__="Heartbeat"
        self.__thread__=None
        self.__keep_running__=False
        self.__precision__=precision
        self.__running__=False
        self.__level=level # set the message level for the heart beat
    def __del__(self):
        if self.__running__:
            self.stop()

    def start(self,message,bint=None,timeout=None):
        self.__message__=message
        tnow=time.time()
        self.__tstart__=tnow
        if timeout is None:
            ttimeout=self.__tstart__+self.__timeout__
        else:
            ttimeout=self.__tstart__+timeout
            self.__timeout__=timeout
        if bint is None:
            tstep=self.__tstart__+self.__bint__
        else:
            tstep=self.__tstart__+bint
            self.__bint__=bint
        if self.__running__:
            self.stop()
        self.__logger__.log(self.__level,self.__message__)
        self.__keep_running__=True
        loop=lambda self: self.doloop()
        self.__thread__=Thread(None,target=loop,args=[self])
        self.__thread__.daemon=True
        self.__thread__.start()
        self.__running__=True

    def doloop(self):
        tnow=self.__tstart__
        ttimeout=self.__tstart__+self.__timeout__
        beattime=self.__tstart__+self.__bint__
        while(self.__keep_running__ and tnow<ttimeout):
            time.sleep(self.__precision__)
            tn=time.time()
            tcheck=tn-self.__tstart__
            if tcheck<0 or tcheck>3000 : #time change >+1hrs -beatinterval
                self.__logger__.log(self.__level+10,"Clock skew detected")
            tnow=tn
            if tnow>=beattime:
                beattime+=self.__bint__
                self.__logger__.log(self.__level,self.__message__)
    def stop(self,msg=None):
        self.__keep_running__=False
        self.__thread__.join()
        self.__running__=False
        if msg is not None:
            self.__logger__.log(self.__level,msg)
