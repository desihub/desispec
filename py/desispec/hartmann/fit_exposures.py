""" Code to fit arclines using unprocessed raw image
"""

import focus_DESI_2 as foc
import pdb
import matplotlib.pyplot as plt
from astropy.table import Table
import pickle as pk
from astropy.modeling import models, fitting
import numpy as np
# '/project/projectdirs/desi/spectro/teststand/rawdata/SM03/2018/20180906/WINLIGHT_00009777.fits'

rawdata_dir='/project/projectdirs/desi/spectro/teststand/rawdata/'
spectrograph='SM02'
channel='B1'
date='/2018/20180906/'
serial='00008880' # 8872 best focus
autoSelect=True

spectrograph='SM03'
channel='B2' # Not sure if the transform data is correct
date='/2018/20181013/'
autoSelect=True
show_poly_fit=True

serial_arr=['00009767'] # 9767 best focus
defocus_arr=[0.]
type='single'
read=False


spectrograph='SM03'
channel='B2' # Not sure if the transform data is correct
date='/2018/20181010/' # First focus loop for B2
autoSelect=True
show_poly_fit=True
serial_arr_all=9495+np.arange(17)*5
serial_arr=[str(i).zfill(8) for i in serial_arr_all]
defocus_arr=-0.2+np.arange(17)*0.025
defocus_arr=defocus_arr.tolist()
type='focus_loop'
read=False

"""
serial_arr=['00009747','00009752','00009757','00009762','00009767','00009772','00009777','00009782','00009787'] # Focus loop after tilt correction
defocus_arr=[-0.1,-0.075,-0.05,-0.025,0.0,0.025,0.05,0.075,0.1]
type='focus_loop'
read=True

spectrograph='SM03'
channel='B2' # Not sure if the transform data is correct
date='/2018/20181016/'
autoSelect=True
show_poly_fit=True

serial_arr=['00010006','00010011','00010016','00010021']
defocus_arr=[0.125,0.15,0.175,0.2]
type='focus_loop'
read=False

spectrograph='SM03'
channel='Z2' # Not sure if the transform data is correct
date='/2018/20181018/'
serial='00010184' # 9767 best focus
autoSelect=False
"""


SrcData_all=[]
n_exposure=len(serial_arr)
file_pk=spectrograph+'_'+channel+'_'+date.replace('/','_')+type+'.dat'
if read:
    SrcData_all=pk.load(open(file_pk,'rb'))
else:   # Fitting
    for i in range(n_exposure):
        serial=serial_arr[i]
        print(serial)
        file_dir=rawdata_dir+spectrograph+date
        file_in=file_dir+'WINLIGHT_'+serial+'.fits'
        file_out='sources'+spectrograph+'_'+channel+'_HgAr_Kr_Ne_Cd.dat'
        sources = foc.getSources(file_in, channel=channel,autoSelect=autoSelect)
        SrcData = foc.GetAllSourcesMetrics(sources, ee=0.80, display=False)
        SrcData_all.append(SrcData)
    pk.dump(SrcData_all, open(file_pk,'wb'))

fiber_num_all=list(set([s['fiber_num'] for SrcData in SrcData_all for s in SrcData]))
wave_num_all=list(set([s['wave_num'] for SrcData in SrcData_all for s in SrcData]))
source_all=list(set([s['source'] for SrcData in SrcData_all for s in SrcData]))


SrcTable_all=[]
for i in range(n_exposure):
    SrcData=SrcData_all[i]
    fibs = list(set([s['fiber_num'] for s in SrcData]))
    waves = list(set([s['wave_num'] for s in SrcData]))
    # Organize the data structure first
    SrcTable=Table([[],[],[],[],[],[],[],[],[],[],[],[]],names=['source','fiber_num', 'wave_num', 'x0', 'y0','defocus','xcentroid','ycentroid','Ree','FWHMx','FWHMy','Amp'],dtype=['i4','i4','i4','i4','i4','f4','f4','f4','f4','f4','f4','f4'])
    for j in range(len(SrcData)):
        t=SrcData[j]
        SrcTable.add_row([t['source'],t['fiber_num'],t['wave_num'],t['x0'],t['y0'],t['table']['defocus'],t['table']['xcentroid'],t['table']['ycentroid'],t['table']['Ree'],t['table']['FWHMx'],t['table']['FWHMy'],t['table']['Amp']])
    SrcTable_all.append(SrcTable)

if type == 'focus_loop':
    # Group by sources
    source_table_all=[]
    best_focus_table=Table([[],[],[],[],[],[],[]],names=['source','fiber_num', 'wave_num', 'x', 'y','best_focus','best_focus_fwhm'],dtype=['i4','i4','i4','f4','f4','f4','f4'])

    for i in range(len(source_all)):
        source_this=source_all[i]
        print('Source:',source_this)
        source_table=Table([[],[],[],[],[],[],[],[],[],[],[],[]],names=['source','fiber_num', 'wave_num', 'x0', 'y0','defocus','xcentroid','ycentroid','Ree','FWHMx','FWHMy','Amp'],dtype=['i4','i4','i4','i4','i4','f4','f4','f4','f4','f4','f4','f4'])

        for j in range(len(SrcTable_all)):
            print(j+1,'th exposure')
            SrcTable=SrcTable_all[j]
            SrcTable.add_index('source')
            try:
                data_this=SrcTable.loc[source_this]
                data_this['defocus']=defocus_arr[j]
                fiber_num_this=data_this['fiber_num']
                wave_num_this=data_this['wave_num']
                x_this=data_this['x0']+data_this['xcentroid']
                y_this=data_this['y0']+data_this['ycentroid']
            except:
                print('Could not find the source')
                data_this=[[source_this],[-999],[-999],[-999],[-999],[defocus_arr[j]],[-999],[-999],[-999],[-999],[-999],[-999]]                
            source_table.add_row(data_this)
    
        FWHMx_arr=np.array(source_table['FWHMx'].tolist())
        FWHMy_arr=np.array(source_table['FWHMy'].tolist())
        ind=np.where(FWHMx_arr>0)
        x=np.array(defocus_arr)[ind]
        y=FWHMx_arr[ind]
        y2=FWHMy_arr[ind]
        p = models.Polynomial1D(2, n_models=1)
        pfit = fitting.LinearLSQFitter()
        new_model = pfit(p, x, y)
        y_model=new_model(x)
        plt.plot(x,y)
        plt.plot(x,y_model,'b+')
        x_dense=np.arange(400)/1000.-0.2
        y_dense=new_model(x_dense)
        ind_min=np.where(y_dense == min(y_dense))
        best_focus_this=x_dense[ind_min[0][0]]
        best_focus_fwhm_this=y_dense[ind_min[0][0]]
        best_focus_table.add_row([[source_this],[fiber_num_this],[wave_num_this],[x_this],[y_this],[best_focus_this],[best_focus_fwhm_this]])
        if show_poly_fit:
            if i ==0:
                plt.figure(1,figsize=(15,15))
                plt.subplot(111)
                plt.xlabel('defocus')
                plt.ylabel('FWHMx')
            plt.plot(x,y)
            plt.plot(x,y_model,'b+')
        source_table_all.append(source_table)
    plt.show()
pdb.set_trace()

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

    idx=[i+1 for i in range(len(x_arr))]
    table_out=Table([idx,x_arr,y_arr],names=('id','xcentroid','ycentroid'),dtype=('i8', 'f8', 'f8'))

    if not autoSelect:
        table_out.write(file_out,format='ascii',overwrite=True)
    pdb.set_trace()

elif type =='focus_loop':
    x_arr=best_focus_table['x']
    y_arr=best_focus_table['y']
    z_arr=best_focus_table['best_focus']

    ax = plt.axes(projection='3d')
    ax.scatter3D(x_arr, y_arr, z_arr, c=z_arr, cmap='jet')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Best Focus')
    plt.show()


    z_arr=best_focus_table['best_focus_fwhm']

    ax = plt.axes(projection='3d')
    ax.scatter3D(x_arr, y_arr, z_arr, c=z_arr, cmap='jet')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Best FWHMx')
    plt.show()



#SrcData, OverallResult = foc.main('/project/projectdirs/desi/spectro/teststand/rawdata/SM02/2018/20180906/WINLIGHT_00008872.fits', channel='B1')
#SrcData, OverallResult = foc.main('/project/projectdirs/desi/spectro/teststand/rawdata/SM01/2018/20180326/WINLIGHT_00007435.fits', channel='B1')




