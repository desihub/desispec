import numpy as np

def parse_fibers(fiber_string) :
    
    if fiber_string is None :
        return None
    
    fibers=[]
    for sub in fiber_string.split(',') :
        if sub.isdigit() :
            fibers.append(int(sub))
            continue
        
        tmp = sub.split(':')
        if ((len(tmp) is 2) and tmp[0].isdigit() == True and tmp[1].isdigit() == True) :
            for f in range(int(tmp[0]),int(tmp[1])) :
                fibers.append(f)
        else :
            print("--fibers parsing error.\nCorrect format is either  : --fibers=begin,end (excluded)\nand/or  : --fibers=begin:end (excluded)\nYou can use : --fibers=2,5,6:8,3,10")
            sys.exit(1)
    return np.array(fibers)

