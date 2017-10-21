"""
desispec.quicklook.util
=======================

put pipeline related utility functions here.
"""

import os
import numpy as np
import yaml
from desispec.quicklook import qllogger

def merge_QAs(qaresult,config):

    """
    Per QL pipeline level merging of QA results
    qaresult: list of [pa,qa]; where pa is pa name; qa is the list of qa results. 
    This list is created inside the QL pipeline.
    
    """
    qlog=qllogger.QLLogger("QuickLook",20)
    log=qlog.getlog()

    night=config['Night']
    flavor=config['Flavor']
    camera=config['Camera']
    expid=config['Expid']

    mergedQA={'NIGHT':night,
              'FLAVOR':flavor,
              'CAMERA':camera,
              'EXPID':expid
              }

    for s,result in enumerate(qaresult):
        pa=result[0].upper() 
        mergedQA[pa]={}
        mergedQA[pa]['PARAMS']={}
        mergedQA[pa]['METRICS']={}

        #- now merge PARAM and QA metrics for all QAs
        for qa in result[1]:
            if 'PARAMS' in result[1][qa]:
                mergedQA[pa]['PARAMS'].update(result[1][qa]['PARAMS'])
            if 'METRICS' in result[1][qa]:
                mergedQA[pa]['METRICS'].update(result[1][qa]['METRICS'])

    from desiutil.io import yamlify
    from desispec.io import findfile
    qadict=yamlify(mergedQA)
    specprod=os.environ['QL_SPEC_REDUX']
    if flavor == 'arcs':
        merged_file=findfile('ql_mergedQAarc_file',night=night,expid=expid,camera=camera,specprod_dir=specprod)
    else:
        merged_file=findfile('ql_mergedQA_file',night=night,expid=expid,camera=camera,specprod_dir=specprod)
    f=open(merged_file,'w')
    f.write(yaml.dump(qadict))
    f.close()

    log.info("Wrote merged QA file {}".format(merged_file))

    return


