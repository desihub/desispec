"""
Fit arclines for images after preprocessing.
"""

import focus_DESI_2 as foc
import fit_arc as fit_arc
import pdb
import matplotlib.pyplot as plt
from astropy.table import Table
import pickle as pk
from astropy.modeling import models, fitting
import numpy as np
# '/project/projectdirs/desi/spectro/teststand/rawdata/SM03/2018/20180906/WINLIGHT_00009777.fits'
from matplotlib.backends.backend_pdf import PdfPages

rawdata_dir='/project/projectdirs/desi/spectro/teststand/rawdata/'
spectrograph='SM02'
channel='B1'
date='/2018/20180906/'
serial='00008880' # 8872 best focus
autoSelect=True

spectrograph='SM02'
channel='B1' # Not sure if the transform data is correct
psf_file='/project/projectdirs/desi/users/jguy/teststand/20181130/psf-b1.fits'
date='/2018/20181119/' # First focus loop for B1
serial_arr_all_left=11966+np.arange(17)*5  # Left hartmann serials 17
serial_arr_all_right=serial_arr_all_left-1
serial_arr_left=[str(i).zfill(8) for i in serial_arr_all_left]
serial_arr_right=[str(i).zfill(8) for i in serial_arr_all_right]
defocus_arr=-0.2+np.arange(17)*0.025
defocus_arr=defocus_arr.tolist()
type='focus_loop'
read=False
fit_display=False
show_poly_fit=False



"""
spectrograph='SM03'
channel='B2' # Not sure if the transform data is correct
psf_file='/project/projectdirs/desi/users/jguy/teststand/20181023/psf-b2.fits'
date='/2018/20181010/' # First focus loop for B2
#serial_arr_all=9495+np.arange(17)*5
serial_arr_all_left=9498+np.arange(17)*5  # Left hartmann serials 17
serial_arr_all_right=serial_arr_all_left-1
serial_arr_left=[str(i).zfill(8) for i in serial_arr_all_left]
serial_arr_right=[str(i).zfill(8) for i in serial_arr_all_right]
defocus_arr=-0.2+np.arange(17)*0.025
defocus_arr=defocus_arr.tolist()
type='focus_loop'
read=True
fit_display=False
show_poly_fit=False


spectrograph='SM03'
channel='R2' # Not sure if the transform data is correct
psf_file='/project/projectdirs/desi/users/jguy/teststand/20181023/psf-r2.fits'
date='/2018/20181003/' # First focus loop for B2
#serial_arr_all=9495+np.arange(17)*5
serial_arr_all_left=9287+np.arange(17)*5  # Left hartmann serials 17
serial_arr_all_right=serial_arr_all_left-1
serial_arr_left=[str(i).zfill(8) for i in serial_arr_all_left]
serial_arr_right=[str(i).zfill(8) for i in serial_arr_all_right]
defocus_arr=-0.2+np.arange(17)*0.025
defocus_arr=defocus_arr.tolist()
type='focus_loop'
read=True
fit_display=False
show_poly_fit=False


spectrograph='SM03'
channel='Z2' # Not sure if the transform data is correct
psf_file='/project/projectdirs/desi/users/jguy/teststand/20181023/psf-z2.fits'
date='/2018/20181018/'
serial_arr_all_left=10168+np.arange(9)*5  # Left hartmann serials
serial_arr_all_right=serial_arr_all_left-1
serial_arr_left=[str(i).zfill(8) for i in serial_arr_all_left]
serial_arr_right=[str(i).zfill(8) for i in serial_arr_all_right]
defocus_arr=-0.2+np.arange(9)*0.05
defocus_arr=defocus_arr.tolist()
type='focus_loop'
read=False
fit_display=False
show_poly_fit=False


serial_arr=['00009747','00009752','00009757','00009762','00009767','00009772','00009777','00009782','00009787'] # Focus loop after tilt correction
defocus_arr=[-0.1,-0.075,-0.05,-0.025,0.0,0.025,0.05,0.075,0.1]
type='focus_loop'
read=True

spectrograph='SM03'
channel='B2' # Not sure if the transform data is correct
psf_file='/project/projectdirs/desi/users/jguy/teststand/20181023/psf-b2.fits'
date='/2018/20181016/'
show_poly_fit=True
serial_arr=['00010006','00010011','00010016','00010021']
defocus_arr=[0.125,0.15,0.175,0.2]
type='focus_loop'
read=False


"""

if channel[0] == 'Z':
    thres=200
elif channel[0] == 'R':
    thres=200
else:
    thres=500

Data_all_left,Data_all_right=[],[]

n_exposure=len(serial_arr_left)


file_pk_left='fit_hartmann_preproc_left_'+spectrograph+'_'+channel+'_'+date.replace('/','_')+type+'.dat'
file_pk_right='fit_hartmann_preproc_right_'+spectrograph+'_'+channel+'_'+date.replace('/','_')+type+'.dat'
file_pk2='fit_hartmann_preproc_'+spectrograph+'_'+channel+'_'+date.replace('/','_')+type+'.dat2'

if read:
    Data_all_left=pk.load(open(file_pk_left,'rb'))
    Data_all_right=pk.load(open(file_pk_right,'rb'))
    source_table_all=pk.load(open(file_pk2,'rb'))

    fiber_all=list(set([s['fiber'] for Data in Data_all_left for s in Data]))
    lineid_all=list(set([s['lineid'] for Data in Data_all_left for s in Data]))
    n_fiber=len(fiber_all)
    n_line=len(lineid_all)

else:   # Fitting
    for i in range(n_exposure):
        serial_left=serial_arr_left[i]
        serial_right=serial_arr_right[i]
        print(serial_left,',',serial_right)
        file_dir=rawdata_dir+spectrograph+date
        file_in_left=file_dir+'WINLIGHT_'+serial_left+'.fits'
        file_in_right=file_dir+'WINLIGHT_'+serial_right+'.fits'
        Data_left=fit_arc.fit_arc(file_in_left,psf_file,channel,defocus_arr[i],thres=thres,ee=0.90,display=fit_display) 
        Data_right=fit_arc.fit_arc(file_in_right,psf_file,channel,defocus_arr[i],thres=thres,ee=0.90,display=fit_display)
        Data_all_left.append(Data_left)
        Data_all_right.append(Data_right)



    fiber_all=list(set([s['fiber'] for Data in Data_all_left for s in Data]))
    lineid_all=list(set([s['lineid'] for Data in Data_all_left for s in Data]))
    n_fiber=len(fiber_all)
    n_line=len(lineid_all)


    if type == 'focus_loop':  # Make a match
        # Group by sources
        source_table_all=[]

        for i in range(n_fiber):
            for j in range(n_line):
                fiber_this=fiber_all[i]
                lineid_this=lineid_all[j]

                source_table=Table([[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]],names=['fiber','lineid','wave','defocus','x_left','y_left','Ree_left','FWHMx_left','FWHMy_left','Amp_left','x_right','y_right','Ree_right','FWHMx_right','FWHMy_right','Amp_right'],dtype=['i4','i4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4'])

                for k in range(n_exposure):
                    print(k+1,'th exposure')
                    Data_left=Data_all_left[k]
                    Data_right=Data_all_right[k]
                    Data_left.add_index('fiber')
                    Data_left.add_index('lineid')
                    Data_right.add_index('fiber')
                    Data_right.add_index('lineid')

                    try:
                        data_left_this=Data_left.loc['fiber',fiber_this].loc['lineid',lineid_this]
                        data_right_this=Data_right.loc['fiber',fiber_this].loc['lineid',lineid_this]
                        data_this=[[fiber_this],[lineid_this],[data_left_this['wave']],[defocus_arr[k]],
                                   [data_left_this['xcentroid']],[data_left_this['ycentroid']],[data_left_this['Ree']],[data_left_this['FWHMx']],[data_left_this['FWHMy']],[data_left_this['Amp']],
                                   [data_right_this['xcentroid']],[data_right_this['ycentroid']],[data_right_this['Ree']],[data_right_this['FWHMx']],[data_right_this['FWHMy']],[data_right_this['Amp']]]
                    except:
                        print('Could not find the source')
                        data_this=[[fiber_this],[lineid_this],[-999],[defocus_arr[k]],[-999],[-999],[-999],[-999],[-999],[-999],[-999],[-999],[-999],[-999],[-999],[-999]]
                    source_table.add_row(data_this)
                source_table_all.append(source_table)

    pk.dump(Data_all_left, open(file_pk_left,'wb'))
    pk.dump(Data_all_right, open(file_pk_right,'wb'))
    pk.dump(source_table_all, open(file_pk2,'wb'))

pp=PdfPages(channel+'_hartmann_plot.pdf')
plt.figure(0,figsize=(9.5,14))
font = {'family':'sans-serif',
        'weight':'normal',
        'size'  :12}
plt.rc('font', **font)
plt.subplot(321)
plt.xlabel('Defocus')
plt.ylabel('Hartmann Shift')

if type == 'focus_loop':
    i=0
    slope_table=Table([[],[],[],[],[],[]],names=['fiber','lineid','wave','x','y','slope'],dtype=['i4','i4','f4','f4','f4','f4'])
    slope_arr=[]
    for source_table in source_table_all:
            x_arr_left=np.array(source_table['x_left'].tolist())
            y_arr_left=np.array(source_table['y_left'].tolist())
            x_arr_right=np.array(source_table['x_right'].tolist())
            y_arr_right=np.array(source_table['y_right'].tolist())
            FWHMx_arr_left=np.array(source_table['FWHMx_left'].tolist())
            Amp_arr_left=np.array(source_table['Amp_left'].tolist())
            FWHMx_arr_right=np.array(source_table['FWHMx_right'].tolist())
            Amp_arr_right=np.array(source_table['Amp_right'].tolist())
            defocus_arr_this=np.array(source_table['defocus'].tolist())
            y_shift_abs=np.abs(y_arr_left-y_arr_right)
            source_table.add_index('defocus')
            ind1=np.where(FWHMx_arr_left >2) 
            ind2=np.where(FWHMx_arr_right >2) 
            ind3=np.where(Amp_arr_left >50) 
            ind4=np.where(y_shift_abs <6)
            ind5=np.where(defocus_arr_this>-0.19)

            ind=set(ind1[0].tolist()) & set(ind2[0].tolist()) & set(ind3[0].tolist()) & set(ind4[0].tolist()) & set(ind5[0].tolist())

            ind=np.array(list(ind)) 
            if len(ind)>3:
                x=np.array(defocus_arr)[ind]
                y2=x_arr_left[ind]-x_arr_right[ind]
                y=y_arr_left[ind]-y_arr_right[ind]
                p = models.Polynomial1D(1, n_models=1)
                pfit = fitting.LinearLSQFitter()
                new_model = pfit(p, x, y)
                table_infocus=source_table.loc[0]
                if table_infocus['x_left']>0:
                    x_output=table_infocus['x_left']
                    y_output=table_infocus['y_left']
                else:
                    x_output=source_table[ind[0]]['x_left']
                    y_output=source_table[ind[0]]['y_left']    
                slope_table.add_row([[table_infocus['fiber']],[table_infocus['lineid']],[table_infocus['wave']],[x_output],[y_output],[new_model.c1.value]])
                y_model=new_model(x)
                plt.plot(x,y)
                plt.plot(x,y_model,'b+')
                if show_poly_fit:
                    if i ==0:
                        plt.figure(1,figsize=(15,15))
                        plt.subplot(111)
                        plt.xlabel('defocus')
                        plt.ylabel('xshift left-right')
                    plt.plot(x,y)
                    plt.plot(x,y_model,'b+')
                    i+=1
    plt.subplot(323)

    x_arr=slope_table['x']
    y_arr=slope_table['y']
    z_arr=slope_table['slope']

    plt.scatter(x_arr,z_arr,c=z_arr,alpha=0.5,label='')
    plt.xlabel('X')
    plt.ylabel('Hartmann Slope')
    plt.legend(loc='upper left')

    plt.subplot(324)

    plt.scatter(y_arr,z_arr,c=z_arr,alpha=0.5,label='')
    plt.xlabel('Y')
    plt.ylabel('Hartmann Slope')
    plt.legend(loc='upper left')

if type =='single':
    x=[SrcData_all[0][i]['table']['FWHMx'] for i in range(len(SrcData_all[0]))]
    y=[SrcData_all[0][i]['table']['FWHMy'] for i in range(len(SrcData_all[0]))]
    plt.plot(x,y,'b+')
    plt.plot([min(x),min(x)],[max(y),max(y)])
    plt.xlabel('FWHMx')
    plt.ylabel('FWHMy')
    plt.show()

    x_arr=[]
    y_arr=[]
    z_arr=[]
    for src in SrcData:
        x_arr.append(src['x0']+src['table']['xcentroid'][0])
        y_arr.append(src['y0']+src['table']['ycentroid'][0])
        z_arr.append(src['table']['FWHMx'][0])
    ax = plt.axes(projection='3d')
    ax.scatter3D(x_arr, y_arr, z_arr, c=z_arr, cmap='jet')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('FWHMx')
    plt.show()


elif type =='focus_loop':
    x_arr=slope_table['x']
    y_arr=slope_table['y']
    z_arr=slope_table['slope']

    plt.subplot(322)
    plt.gca().set_aspect('equal')
    plt.tricontour(x_arr,y_arr,z_arr)
    plt.colorbar()


    """
    ax = plt.axes(projection='3d')
    ax.scatter3D(x_arr, y_arr, z_arr, c=z_arr, cmap='jet')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('slope')
    plt.show()
    """
pp.savefig()
plt.close()
pp.close()

pdb.set_trace()

#SrcData, OverallResult = foc.main('/project/projectdirs/desi/spectro/teststand/rawdata/SM02/2018/20180906/WINLIGHT_00008872.fits', channel='B1')
#SrcData, OverallResult = foc.main('/project/projectdirs/desi/spectro/teststand/rawdata/SM01/2018/20180326/WINLIGHT_00007435.fits', channel='B1')




