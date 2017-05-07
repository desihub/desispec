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
        night=result[1].values()[0]['NIGHT']
        expid=int(result[1].values()[0]['EXPID'])
        camera=result[1].values()[0]['CAMERA']
        flavor=result[1].values()[0]['FLAVOR']
            
        if night not in mergedQA:
            mergedQA[night]={} #- top level key
        if expid not in mergedQA[night]:
            mergedQA[night][expid]={}
        if camera not in mergedQA[night][expid]:
            mergedQA[night][expid][camera]={}
        if 'flavor' not in mergedQA[night][expid]:
            mergedQA[night][expid]['flavor']=flavor
        mergedQA[night][expid][camera][pa]={}
        mergedQA[night][expid][camera][pa]['PARAM']={}
        mergedQA[night][expid][camera][pa]['QA']={}

        #- now merge PARAM and QA metrics for all QAs
        for qa in result[1]:
            if 'PARAMS' in result[1][qa]:
                mergedQA[night][expid][camera][pa]['PARAM'].update(result[1][qa]['PARAMS'])
            if 'METRICS' in result[1][qa]:
                mergedQA[night][expid][camera][pa]['QA'].update(result[1][qa]['METRICS'])
    from desiutil.io import yamlify
    qadict=yamlify(mergedQA)
    f=open('mergedQA-{}-{:08d}.yaml'.format(camera,expid),'w') #- IO/file naming should move from here. 
    f.write(yaml.dump(qadict))
    f.close()
    return

    
    
        
