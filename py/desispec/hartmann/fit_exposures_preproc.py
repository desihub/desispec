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
from matplotlib.backends.backend_pdf import PdfPages
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
psf_file='/project/projectdirs/desi/users/jguy/teststand/20181023/psf-b2.fits'
date='/2018/20181010/' # First focus loop for B2
serial_arr_all=9495+np.arange(17)*5
serial_arr=[str(i).zfill(8) for i in serial_arr_all]
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
serial_arr_all=9283+np.arange(17)*5  # Left hartmann serials 17
serial_arr=[str(i).zfill(8) for i in serial_arr_all]
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
serial_arr_all=10164+np.arange(9)*5
serial_arr=[str(i).zfill(8) for i in serial_arr_all]
defocus_arr=-0.2+np.arange(9)*0.05
defocus_arr=defocus_arr.tolist()
type='focus_loop'
read=True
fit_display=False

"""
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

Data_all=[]
n_exposure=len(serial_arr)

fiber_all=list(set([s['fiber'] for Data in Data_all for s in Data]))
lineid_all=list(set([s['lineid'] for Data in Data_all for s in Data]))
n_fiber=len(fiber_all)
n_line=len(lineid_all)



file_pk='fit_exposures_preproc_'+spectrograph+'_'+channel+'_'+date.replace('/','_')+type+'.dat'
file_pk2='fit_exposures_preproc_'+spectrograph+'_'+channel+'_'+date.replace('/','_')+type+'.dat2'

if read:
    best_focus_table=Table([[],[],[],[],[],[]],names=['fiber', 'lineid', 'x', 'y','best_focus','best_focus_fwhm'],dtype=['i4','i4','f4','f4','f4','f4'])
    Data_all=pk.load(open(file_pk,'rb'))
    source_table_all=pk.load(open(file_pk2,'rb'))
else:   # Fitting
    for i in range(n_exposure):
        serial=serial_arr[i]
        print(serial)
        file_dir=rawdata_dir+spectrograph+date
        file_in=file_dir+'WINLIGHT_'+serial+'.fits'
        file_out='sources'+spectrograph+'_'+channel+'_HgAr_Kr_Ne_Cd.dat'
        Data=fit_arc.fit_arc(file_in,psf_file,channel,defocus_arr[i],thres=thres,ee=0.90,display=fit_display,file_temp='file_temp_exp.fits') 
        Data_all.append(Data)

    fiber_all=list(set([s['fiber'] for Data in Data_all for s in Data]))
    lineid_all=list(set([s['lineid'] for Data in Data_all for s in Data]))
    n_fiber=len(fiber_all)
    n_line=len(lineid_all)


    if type == 'focus_loop':
        # Group by sources
        source_table_all=[]
        best_focus_table=Table([[],[],[],[],[],[]],names=['fiber', 'lineid', 'x', 'y','best_focus','best_focus_fwhm'],dtype=['i4','i4','f4','f4','f4','f4'])

        for i in range(n_fiber):
            for j in range(n_line):
                fiber_this=fiber_all[i]
                lineid_this=lineid_all[j]

                source_table=Table([[],[],[],[],[],[],[],[],[],[]],names=['defocus','xcentroid','ycentroid','fiber', 'lineid', 'wave','Ree','FWHMx','FWHMy','Amp'],dtype=['f4','f4','f4','i4','i4','f4','f4','f4','f4','f4'])

                for k in range(n_exposure):
                    print(k+1,'th exposure')
                    Data=Data_all[k]
                    Data.add_index('fiber')
                    Data.add_index('lineid')
                    try:
                        data_this=Data.loc['fiber',fiber_this].loc['lineid',lineid_this]
                        data_this['defocus']=defocus_arr[k]
                        x_this=data_this['xcentroid']
                        y_this=data_this['ycentroid']
                    except:
                        print('Could not find the source')
                        data_this=[[defocus_arr[k]],[-999],[-999],[fiber_this],[lineid_this],[-999],[-999],[-999],[-999],[-999]]
                    source_table.add_row(data_this)
                source_table_all.append(source_table)

    pk.dump(Data_all, open(file_pk,'wb'))
    pk.dump(source_table_all, open(file_pk2,'wb'))

#pdb.set_trace()

pp=PdfPages(channel+'_exposures_plot.pdf')
plt.figure(0,figsize=(9.5,14))
font = {'family':'sans-serif',
        'weight':'normal',
        'size'  :12}
plt.rc('font', **font)
plt.subplot(321)
plt.xlabel('Defocus')
plt.ylabel('FWHMx')

if type == 'focus_loop':
    i=0
    for source_table in source_table_all:
            FWHMx_arr=np.array(source_table['FWHMx'].tolist())
            FWHMy_arr=np.array(source_table['FWHMy'].tolist())
            Amp_arr=np.array(source_table['Amp'].tolist())
            ind1=np.where(Amp_arr>500)
            ind2=np.where(FWHMx_arr>1.5)
            ind=set(ind1[0]) & set(ind2[0])
            ind=np.array(list(ind))
            if len(ind)>3:
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
                for temp in range(len(source_table)):
                    if source_table[temp]['xcentroid']>0:
                        ind_good=temp
                best_focus_table.add_row([[source_table[0]['fiber']],[source_table[0]['lineid']],[source_table[ind_good]['xcentroid']],[source_table[ind_good]['ycentroid']],[best_focus_this],[best_focus_fwhm_this]])
                if show_poly_fit:
                    if i ==0:
                        plt.xlabel('defocus')
                        plt.ylabel('FWHMx')
                    plt.plot(x,y,'b+')
                    plt.plot(x,y_model)
                    i+=1
    plt.subplot(323)
    x_arr=best_focus_table['x']
    y_arr=best_focus_table['y']
    z_arr=best_focus_table['best_focus']

    plt.scatter(x_arr,z_arr,c=z_arr,alpha=0.5,label='')
    plt.xlabel('X')
    plt.ylabel('Best Focus')
    plt.legend(loc='upper left')

    plt.subplot(324)
    plt.scatter(y_arr,z_arr,c=z_arr,alpha=0.5,label='')
    plt.xlabel('Y')
    plt.ylabel('Best Focus')
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

    idx=[i+1 for i in range(len(x_arr))]
    table_out=Table([idx,x_arr,y_arr],names=('id','xcentroid','ycentroid'),dtype=('i8', 'f8', 'f8'))

    if not autoSelect:
        table_out.write(file_out,format='ascii',overwrite=True)
    pdb.set_trace()

elif type =='focus_loop':
    x_arr=best_focus_table['x']
    y_arr=best_focus_table['y']
    z_arr=best_focus_table['best_focus']
    plt.subplot(322)
    plt.gca().set_aspect('equal')
    plt.tricontour(x_arr,y_arr,z_arr)
    plt.colorbar()
    """
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
    """
    z_arr=best_focus_table['best_focus_fwhm']
    plt.subplot(325)
    plt.scatter(x_arr,z_arr,c=z_arr,alpha=0.5,label='')
    plt.xlabel('X')
    plt.ylabel('Best Focus FWHMx')
    plt.legend(loc='upper left')

    plt.subplot(326)
    plt.scatter(y_arr,z_arr,c=z_arr,alpha=0.5,label='')
    plt.xlabel('Y')
    plt.ylabel('Best Focus FWHMx')
    plt.legend(loc='upper left')

pp.savefig()
plt.close()
pp.close()

pdb.set_trace()

#SrcData, OverallResult = foc.main('/project/projectdirs/desi/spectro/teststand/rawdata/SM02/2018/20180906/WINLIGHT_00008872.fits', channel='B1')
#SrcData, OverallResult = foc.main('/project/projectdirs/desi/spectro/teststand/rawdata/SM01/2018/20180326/WINLIGHT_00007435.fits', channel='B1')




