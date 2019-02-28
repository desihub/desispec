import csv 
import os
import glob
import numpy as np
pos_id='00893'
data_dir=os.environ('POSITIONER_LOGS_PATH')+'/xytest_data/'
filenames = glob.glob(data_dir+'/unit_M'+pos_id+'*.csv')
for i in np.shape(filenames)-1:
    with open(filenames[i], 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            print ', '.join(row)
