"""
desispec.quicklook.qas
======================

"""
from desispec.quicklook import qllogger
from desispec.quicklook import qlexceptions
import collections
import numpy as np
from enum import Enum
from astropy.io import fits


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
        self.__deviation = None
        self.m_log.debug("initializing Monitoring alg {}".format(name))

    def __call__(self,*args,**kwargs):
        res=self.run(*args,**kwargs)
        cargs=self.config['kwargs']
        params=cargs['param']

        metrics=res["METRICS"] if 'METRICS' in res else None
        if metrics is None:
            metrics={}
            res["METRICS"]=metrics

        reskey="RESULT"
        QARESULTKEY="QA_STATUS"
        if res['FLAVOR'] == 'science':
           REFNAME = cargs["RESULTKEY"]+'_'+format(res['PROGRAM']).upper()+'_REF' # SE: get the REF name from cargs
        else:
           REFNAME = cargs["RESULTKEY"]+'_REF'

        NORM_range = cargs["RESULTKEY"]+'_NORMAL_RANGE'
        WARN_range = cargs["RESULTKEY"]+'_WARN_RANGE'
        norm_range_val = [0,0]
        warn_range_val = [0,0]

        if "QASTATUSKEY" in cargs:
            QARESULTKEY=cargs["QASTATUSKEY"]
        if "RESULTKEY" in cargs:
            reskey=cargs["RESULTKEY"]

        if cargs["RESULTKEY"] == 'CHECKHDUS':
             stats=[]
             stats.append(metrics['CHECKHDUS_STATUS'])
             stats.append(metrics['EXPNUM_STATUS'])
             if  np.isin(stats,'NORMAL').all():
                    metrics[QARESULTKEY]='NORMAL'
             elif np.isin(stats,'ALARM').any():
                    metrics[QARESULTKEY] = 'ALARM'

             self.m_log.info("{}: {}".format(QARESULTKEY,metrics[QARESULTKEY]))

        if reskey in metrics:
            current = metrics[reskey]

 #SE: Replacing this chunk (between the dashed lines) with an alternative that accomodates receiving the REF keys from the configuration  -----------------------------------------------------------------------------------------------------------------
            #if "REFERENCE" in cargs:

                #refval=cargs["REFERENCE"]

##                print(refval,"MA inside if")

            #else: #- For absolute value checks
                #self.m_log.warning("No reference given. STATUS will be assigned for the Absolute Value. Confirm your ranges.")
                ##- check the data type
                #if isinstance(current,float) or isinstance(current,np.float32) or isinstance(current,int):
                    #refval=0
                #else:
                    #refval=np.zeros(len(current)) #- 1D list or array
            ##- Update PARAMS ref key
            #res["PARAMS"][reskey+'_REF']=refval

            #currlist=isinstance(current,(np.ndarray,collections.Sequence))
            #reflist=isinstance(refval,(np.ndarray,collections.Sequence))
            #if currlist != reflist: # different types
                #self.m_log.critical("QL {} : REFERENCE({}) and RESULT({}) are of different types!".format(self.name,type(refval),type(current)))
            #elif currlist: #both are lists
                #if len(refval)==len(current):
                    #self.__deviation=[c-r for c,r in zip(current,refval)]
                #else:
                    #self.m_log.critical("QL {} : REFERENCE({}) and RESULT({}) are of different length!".format(self.name,len(refval),len(current)))
            #else: # both are scalars
                #self.__deviation=sorted(current)-sorted(refval)

        ## check RANGES given in config and set QA_STATUS keyword
        ## it should be a sorted overlapping list of range tuples in the form [ ((interval),QASeverity),((-1.0,1.0),QASeverity.NORMAL),(-2.0,2.0),QAStatus.WARNING)]
        ## for multiple results, thresholds should be a list of lists as given above (one range list per result)
        ## intervals should be non overlapping.
        ## lower bound is inclusive upper bound is exclusive
        ## first matching interval will be used
        ## if no interval contains the deviation, it will be set to QASeverity.ALARM
        ## if RANGES or REFERENCE are not given in config, QA_STATUS will be set to UNKNOWN
        #def findThr(d,t):
            #val=QASeverity.ALARM
            #for l in list(t):
                 #if d>=l[0][0] and d<l[0][1]:
                    #val=l[1]
            #return val

        #metrics[QARESULTKEY]='NORMAL'

        #if self.__deviation is not None and "RANGES" in cargs:
            #self.m_log.info("QL Reference checking for QA {}".format(self.name))
            #thr=cargs["RANGES"]
            #print(thr)
            #metrics[QARESULTKEY]="ERROR"

            #thrlist=isinstance(thr[0][0][0],(np.ndarray,collections.Sequence))  #multiple threshols for multiple results
            #devlist=isinstance(self.__deviation,(np.ndarray,collections.Sequence))
            ##if devlist!=thrlist and len(thr)!=1:  #different types and thresholds are a list
            ##    self.m_log.critical("QL {} : dimension of RANGES({}) and RESULTS({}) are incompatible! Check configuration RANGES={}, RESULTS={}".format(self.name,len(thr),len(self.__deviation), thr,current))
            ##    return res
            ##else: #they are of the same type



            #if devlist:  # if results are a list
                #if len(thr)==2: # check all results against same thresholds
                    ##- maximum deviation
                    #kk=np.argmax(np.abs(self.__deviation).flatten()) #- flatten for > 1D array
                    #metrics[QARESULTKEY]=findThr(np.array(self.__deviation).flatten()[kk],thr)
                    ##metrics[QARESULTKEY]=[findThr(d,thr) for d in self.__deviation]
                ##else: # each result has its own thresholds
                ##    metrics[QARESULTKEY]=[str(findThr(d,t)) for d,t in zip(self.__deviation,thr)]

            #else: #result is a scalar
                #metrics[QARESULTKEY]=findThr(self.__deviation,thr)
            #if metrics[QARESULTKEY]==QASeverity.NORMAL:
                #metrics[QARESULTKEY]='NORMAL'
            #elif metrics[QARESULTKEY]==QASeverity.WARNING:
                #metrics[QARESULTKEY]='WARNING'
            #else:
                #metrics[QARESULTKEY]='ALARM'
        #else:
            #self.m_log.warning("No Reference checking for QA {}".format(self.name))

        #self.m_log.info("{}: {}".format(QARESULTKEY,metrics[QARESULTKEY]))
        #return res
    #def run(self,*argv,**kwargs):
        #pass
    #def is_compatible(self,Type):
        #return isinstance(Type,self.__inpType__)
    #def check_reference():
        #return self.__deviation
    #def get_default_config(self):
        #""" return a dictionary of 3-tuples,
        #field 0 is the name of the parameter
        #field 1 is the default value of the parameter
        #field 2 is the comment for human readable format.
        #Field 2 can be used for QLF to dynamically setup the display"""
        #return None
    #----------------------------------------------------------------------------------------------------------

            if REFNAME in params:  #SE: get the REF value/ranges from params

                refval=params[REFNAME]

                if len(refval) ==1:
                    refval = refval[0]

                refval = np.asarray(refval)
                current = np.asarray(current)
                norm_range_val=params[NORM_range]
                warn_range_val=params[WARN_range]

                #SE: just in case any nan value sneaks in the array of the scalar metrics
                ind = np.argwhere(np.isnan(current))

                if (ind.shape[0] > 0 and refval.shape[0] == current.shape[0]):
                   self.m_log.critical("QL {} : elements({}) of the result are returned as NaN! STATUS is determined for the real values".format(self.name,str(ind)))

                   ind = list(np.hstack(ind))
                   for index in sorted(ind, reverse=True):
                       del current[index]
                       del refval[index]

            else:
                self.m_log.warning("No reference given. Update the configuration file to include reference value for QA: {}".format(self.name))

            currlist=isinstance(current,(np.ndarray,collections.Sequence))
            reflist=isinstance(refval,(np.ndarray,collections.Sequence))

            if currlist != reflist:
                self.m_log.critical("QL {} : REFERENCE({}) and RESULT({}) are of different types!".format(self.name,type(refval),type(current)))
            elif currlist:

                if refval.size == current.size and current.size >1:

                    self.__deviation=[c-r for c,r in zip(np.sort(current),np.sort(refval))]
                elif refval.size == current.size and current.size and current.size == 1:
                    self.__deviation =  current - refval
                elif np.size(current) == 0 or np.size(refval) == 0:
                    self.m_log.warning("No measurement is done or no reference is available for this QA!- check the configuration file for references!")
                    metrics[QARESULTKEY]='UNKNOWN'
                    self.m_log.info("{}: {}".format(QARESULTKEY,metrics[QARESULTKEY]))
                elif refval.size != current.size:
                    self.m_log.critical("QL {} : REFERENCE({}) and RESULT({}) are of different length!".format(self.name,refval.size,current.size))
                    metrics[QARESULTKEY]='UNKNOWN'
                    self.m_log.info("{}: {}".format(QARESULTKEY,metrics[QARESULTKEY]))

            else:
                #SE "sorting" eliminate the chance of randomly shuffling items in the list that we observed in the past
                self.__deviation=(np.sort(current)-np.sort(refval))/np.sort(current)

        def findThr(d,t):
            if d != None and len(list(t)) >1:
               val=QASeverity.ALARM
               for l in list(t):

                 if d>=l[0][0] and d<l[0][1]:
                    val=l[1]
            else:
                 if d>=l and d<l:
                    val=l
            return val

        devlist = self.__deviation
        thr = norm_range_val
        wthr = warn_range_val

        if devlist is None:
            pass
        #SE: temporarily here until we know OBJLIST is ['SCIENCE', 'STD'] or anything else----------- line below should only be "elif len(thr)==2 and len(wthr)==2:"

        # RS: if one fit fails SNR but the rest pass, return normal
        elif (cargs["RESULTKEY"] == 'FIDSNR_TGT'):
            devlist = current
            stats = []
            nofit = np.where(devlist==0.0)[0]
            if len(nofit) >= 2:
                stats.append('ALARM')
            else:
                for i,val in enumerate(devlist):
                    if len(nofit) != 0 and i == nofit[0]:
                        stats.append('NORMAL')
                    else:
                        diff = refval[i] - val
                        if thr[0]<= diff <= thr[1]:
                            stats.append('NORMAL')
                        elif wthr[0] <= diff <= wthr[1]:
                            stats.append('WARNING')
                        else:
                            stats.append('ALARM')

            if  np.isin(stats,'NORMAL').all():
                metrics[QARESULTKEY]='NORMAL'
            elif np.isin(stats,'WARNING').any() and np.isin(stats,'ALARM').any():
                metrics[QARESULTKEY] = 'ALARM'
            elif np.isin(stats,'ALARM').any():
                metrics[QARESULTKEY] = 'ALARM'
            elif np.isin(stats,'WARNING').any():
                metrics[QARESULTKEY] = 'WARNING'

            self.m_log.info("{}: {}".format(QARESULTKEY,metrics[QARESULTKEY]))

        elif  (len(thr)==2 and len(wthr)==2):

                    if np.size(devlist)== 1:
                        d=[]
                        d.append(devlist)
                        devlist = d
                    stats = []
                    for val in devlist:
                      if thr[0] <= val <= thr[1]:
                        stats.append('NORMAL')
                      elif wthr[0] <= val <= wthr[1]:
                          stats.append('WARNING')
                      else:
                          stats.append('ALARM')

                    if  np.isin(stats,'NORMAL').all():
                        metrics[QARESULTKEY]='NORMAL'
                    elif np.isin(stats,'WARNING').any() and np.isin(stats,'ALARM').any():
                        metrics[QARESULTKEY] = 'ALARM'
                    elif np.isin(stats,'ALARM').any():
                        metrics[QARESULTKEY] = 'ALARM'
                    elif np.isin(stats,'WARNING').any():
                        metrics[QARESULTKEY] = 'WARNING'
                    self.m_log.info("{}: {}".format(QARESULTKEY,metrics[QARESULTKEY]))

        return res

    def run(self,*argv,**kwargs):
        pass
    def is_compatible(self,Type):
        return isinstance(Type,self.__inpType__)
    def check_reference():
        return self.__deviation
    def get_default_config(self):
        return None
