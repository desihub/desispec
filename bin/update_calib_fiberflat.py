#!/usr/bin/env python


import os
import sys
import fitsio

caldir=os.environ["DESI_SPECTRO_CALIB"]

for ifilename in sys.argv[1:] :
    
    head=fitsio.read_header(ifilename)
    sm="sm{}".format(head["SPECID"])
    cam=head["CAMERA"][0]
    ofilename="{}/spec/{}/{}".format(caldir,sm,os.path.basename(ifilename))
    ofilename2="spec/{}/{}".format(sm,os.path.basename(ifilename))
    cmd="cp {} {}".format(ifilename,ofilename)
    yamlfile="{}/spec/{}/{}-{}.yaml".format(caldir,sm,sm,cam)
    print(cmd)
    replaceline=""
    with open(yamlfile) as file :
        for line in file.readlines() :
            if line.find("FIBERFLAT")>=0 :
                replaceline=line.strip()
                break
    if replaceline == "" :
        with open(yamlfile) as file :
            for line in file.readlines() :
                if line.find("PSF")>=0 :
                    addafterline=line.strip()
                    break
        print("echo need to edit manually the fiberflat in {}".format(yamlfile))
    else :
        cmd2="cat {} | sed 's# {}# FIBERFLAT: {}#' > {}.mod".format(yamlfile,replaceline,ofilename2,yamlfile)
        print(cmd2)
        cmd3="mv {}.mod {}".format(yamlfile,yamlfile)
        print(cmd3)
