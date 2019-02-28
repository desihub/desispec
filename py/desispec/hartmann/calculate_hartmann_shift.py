"""
Fit arclines for images after preprocessing.
compute the spot shift of left and right hartmann images

"""

import focus_DESI_2 as foc
import fit_arc as fit_arc
import pdb
import matplotlib.pyplot as plt
import astropy
from astropy.table import Table
import pickle as pk
from astropy.modeling import models, fitting
import numpy as np
# '/project/projectdirs/desi/spectro/teststand/rawdata/SM03/2018/20180906/WINLIGHT_00009777.fits'
from matplotlib.backends.backend_pdf import PdfPages
from desispec.calibfinder import CalibFinder

def calculate_hartmann_shift(serial_left=9538,spectrograph='SM03',channel='B2',psf_file='/project/projectdirs/desi/users/jguy/teststand/20181023/psf-b2.fits',rawdata_dir='/project/projectdirs/desi/spectro/teststand/rawdata/SM03/2018/20181010/',fit_display=False):

    if not isinstance(serial_left,list):
        serial_left=[serial_left]
    serial_arr_all_left=np.array(serial_left)#+np.arange(2)*5  # Left hartmann serials 17
    serial_arr_all_right=serial_arr_all_left-1
    serial_arr_left=[str(i).zfill(8) for i in serial_arr_all_left]
    serial_arr_right=[str(i).zfill(8) for i in serial_arr_all_right]

    """
    spectrograph='SM03'
    channel='R2' # Not sure if the transform data is correct
    psf_file='/project/projectdirs/desi/users/jguy/teststand/20181023/psf-r2.fits'
    date='/2018/20181003/' # First focus loop for B2
    serial_arr_all_left=9287+np.arange(17)*5  # Left hartmann serials 17
    serial_arr_all_right=serial_arr_all_left-1
    serial_arr_left=[str(i).zfill(8) for i in serial_arr_all_left]
    serial_arr_right=[str(i).zfill(8) for i in serial_arr_all_right]

    spectrograph='SM03'
    channel='Z2' # Not sure if the transform data is correct
    psf_file='/project/projectdirs/desi/users/jguy/teststand/20181023/psf-z2.fits'
    date='/2018/20181018/'
    serial_arr_all_left=10168+np.arange(9)*5  # Left hartmann serials
    serial_arr_all_right=serial_arr_all_left-1
    serial_arr_left=[str(i).zfill(8) for i in serial_arr_all_left]
    serial_arr_right=[str(i).zfill(8) for i in serial_arr_all_right]

    """

    if channel[0] == 'Z':
        thres=200
    elif channel[0] == 'R':
        thres=200
    else:
        thres=500

    Data_all_left,Data_all_right=[],[]

    n_exposure=len(serial_arr_left)


    for i in range(n_exposure):
        serial_left=serial_arr_left[i]
        serial_right=serial_arr_right[i]
        print(serial_left,',',serial_right)
        file_dir=rawdata_dir
        file_in_left=file_dir+'WINLIGHT_'+serial_left+'.fits'
        file_in_right=file_dir+'WINLIGHT_'+serial_right+'.fits'
        Data_left=fit_arc.fit_arc(file_in_left,psf_file,channel,0.,line_file='lines_for_hartmann.ascii',thres=thres,ee=0.90,display=fit_display) 
        Data_right=fit_arc.fit_arc(file_in_right,psf_file,channel,0.,line_file='lines_for_hartmann.ascii',thres=thres,ee=0.90,display=fit_display)
        Data_all_left.append(Data_left)
        Data_all_right.append(Data_right)

    hdulist = astropy.io.fits.open("file_temp.fits")
    cfinder = CalibFinder([hdulist[0].header])
    hartmanncoef=cfinder.value("HARTMANNCOEF")
    hartmannwave=cfinder.value("HARTMANNWAVE")


    fiber_all=list(set([s['fiber'] for Data in Data_all_left for s in Data]))
    lineid_all=list(set([s['lineid'] for Data in Data_all_left for s in Data]))
    n_fiber=len(fiber_all)
    n_line=len(lineid_all)


    # Group by sources
    source_table_all=[]

    for i in range(n_fiber):
        for j in range(n_line):
            fiber_this=fiber_all[i]
            lineid_this=lineid_all[j]

            source_table=Table([[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]],names=['fiber','lineid','wave','x_left','y_left','Ree_left','FWHMx_left','FWHMy_left','Amp_left','x_right','y_right','Ree_right','FWHMx_right','FWHMy_right','Amp_right'],dtype=['i4','i4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4'])

            for k in range(n_exposure):
                print(k+1,'th exposure')
                Data_left=Data_all_left[k]
                Data_right=Data_all_right[k]
                Data_left.add_index('fiber')
                Data_left.add_index('lineid')
                Data_right.add_index('fiber')
                Data_right.add_index('lineid')
                try:
                    if n_line == 1:
                        data_left_this=Data_left.loc['fiber',fiber_this]
                        data_right_this=Data_right.loc['fiber',fiber_this]
                    else:
                        data_left_this=Data_left.loc['fiber',fiber_this].loc['lineid',lineid_this]
                        data_right_this=Data_right.loc['fiber',fiber_this].loc['lineid',lineid_this]

                    data_this=[[fiber_this],[lineid_this],[data_left_this['wave']],
                               [data_left_this['xcentroid']],[data_left_this['ycentroid']],[data_left_this['Ree']],[data_left_this['FWHMx']],[data_left_this['FWHMy']],[data_left_this['Amp']],
                               [data_right_this['xcentroid']],[data_right_this['ycentroid']],[data_right_this['Ree']],[data_right_this['FWHMx']],[data_right_this['FWHMy']],[data_right_this['Amp']]]
                except:
                    print('Could not find the source')
                    data_this=[[fiber_this],[lineid_this],[-999],[-999],[-999],[-999],[-999],[-999],[-999],[-999],[-999],[-999],[-999],[-999],[-999]]
                source_table.add_row(data_this)
            source_table_all.append(source_table)


    y_shift_arr=[]
    for source_table in source_table_all:
        x_arr_left=np.array(source_table['x_left'].tolist())
        y_arr_left=np.array(source_table['y_left'].tolist())
        x_arr_right=np.array(source_table['x_right'].tolist())
        y_arr_right=np.array(source_table['y_right'].tolist())
        FWHMx_arr_left=np.array(source_table['FWHMx_left'].tolist())
        Amp_arr_left=np.array(source_table['Amp_left'].tolist())
        FWHMx_arr_right=np.array(source_table['FWHMx_right'].tolist())
        Amp_arr_right=np.array(source_table['Amp_right'].tolist())
        y_shift=y_arr_left-y_arr_right
        y_shift_arr.append(y_shift)
    y_shift_arr=np.array(y_shift_arr)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('We use line '+str(hartmannwave)+' for derivng Hartmann shift')
    print('The calculated y shifts are:\n',y_shift_arr)
    output=np.median(y_shift_arr,axis=0)
    print('The median value of y shift(left-right) is:',output)
    print('Use Hartmann coeff='+str(hartmanncoef))
    print('Please shift the focus by ',-output/hartmanncoef)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    return output 



if __name__ == '__main__':
    f = calculate_hartmann_shift()
