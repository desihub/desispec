"""
desispec.quicklook.util
=======================

put pipeline related utility functions here.
"""

import numpy as np
import yaml

def merge_QAs(qaresult):

    """
    Per QL pipeline level merging of QA results
    qaresult: list of [pa,qa]; where pa is pa name; qa is the list of qa results. 
    This list is created inside the QL pipeline.
    
    """
    mergedQA={}

    for s,result in enumerate(qaresult):
        pa=result[0]
        mergedQA[pa]={}
        mergedQA[pa]['QA']={}
        for qa in result[1]:
            if 'EXPID' not in mergedQA:
                mergedQA['EXPID']=result[1][qa]['EXPID']
            if 'CAMERA' not in mergedQA:
                mergedQA['CAMERA']=result[1][qa]['CAMERA']
            if 'FLAVOR' not in mergedQA:
                mergedQA['FLAVOR']=result[1][qa]['FLAVOR']

            mergedQA[pa]['QA'][qa]={}
            if 'PARAMS' in result[1][qa]:
                mergedQA[pa]['QA'][qa]['PARAMS']=result[1][qa]['PARAMS']
            if 'METRICS' in result[1][qa]:
                mergedQA[pa]['QA'][qa]['METRICS']=result[1][qa]['METRICS']
    from desiutil.io import yamlify
    qadict=yamlify(mergedQA)
    f=open('mergedQA-{}-{}.yaml'.format(mergedQA['CAMERA'],mergedQA['EXPID']),'w') #- IO/file naming should move from here. 
    f.write(yaml.dump(qadict))
    f.close()
    return

    
    
        
