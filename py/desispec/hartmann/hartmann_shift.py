import desispec.hartmann.focus_DESI_2 as foc
import desispec.hartmann.fit_arc as fit_arc
import pdb
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import ascii
import pickle as pk
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import argparse
from pkg_resources import resource_exists, resource_filename

"""
Code to calcuate the Hartmann shift given a pair of left/right hartmann exposures. 

Example
desi_hartmann -r_d /project/projectdirs/desi/spectro/data/20191030/ -c B3 -p /project/projectdirs/desi/users/jguy/kpno/sm4-bootcalib/psfboot-b3.fits -l 22754 -r 22757 -d 0. -rd False -dp False 
"""

def parse(options=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r_d','--rawdata_dir', type=str, default = None, required = True, help="Raw data directory, including the date.")
    parser.add_argument('-c','--channel', type=str, default = None, required = True, help="Which channel(camera) to process.")
    parser.add_argument('-p','--psf', type=str, default = None, required = True, help="psf file.")
    parser.add_argument('-l','--left', type=int, default = None, required = True, help="expid of left hartmann")
    parser.add_argument('-r','--right', type=int, default = None, required = True, help="expid of right hartmann")
    parser.add_argument('-d','--defocus', type=float, default = None, required = True, help="defocus")
    parser.add_argument('-rd','--read', type=str, default = 'False', required = False, help="False=fit raw data. True=read archive data")
    parser.add_argument('-dp','--fit_display', type=str, default = 'False', required = False, help="True=show 2D fitting result.")
    parser.add_argument('-o','--output_dir', type=str, default = '', required = False, help="Output directory for all files generated. Current directory by default.")
 
    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args

def main(args):

    ########################################
    ########### Setup Input ################
    ########################################
    
    rawdata_dir=args.rawdata_dir
    channel=args.channel
    psf_file=args.psf
    serial_arr_all_left=np.array([args.left])  # Left hartmann serials 17
    serial_arr_all_right=np.array([args.right])
    serial_arr_left=[str(i).zfill(8) for i in serial_arr_all_left]
    serial_arr_right=[str(i).zfill(8) for i in serial_arr_all_right]
    defocus_arr=np.array([args.defocus])
    defocus_arr=defocus_arr.tolist()
    read = True if (args.read.upper()=='TRUE' or args.read.upper()=='T') else False
    fit_display= True if (args.fit_display.upper()=='TRUE' or args.fit_display.upper()=='T') else False
    output_dir=args.output_dir

    Data_all_left,Data_all_right=[],[]

    n_exposure=len(serial_arr_left)


    if read:
        pass
    else:   # Fitting
        for i in range(n_exposure):  # Fitting, pretty fast
            serial_left=serial_arr_left[i]
            serial_right=serial_arr_right[i]
            print(serial_left,',',serial_right)
            file_dir=rawdata_dir
            file_in_left=file_dir+'/'+serial_left+'/desi-'+serial_left+'.fits.fz'
            file_in_right=file_dir+'/'+serial_right+'/desi-'+serial_right+'.fits.fz'
            line_file = resource_filename('desispec','/data/arc_lines/goodlines_vacuum_hartmann.ascii' )
            Data_left=fit_arc.fit_arc(file_in_left,psf_file,channel,defocus_arr[i],ee=0.90,display=fit_display,line_file=line_file,file_temp=output_dir+'/file_temp_'+channel+'_l.fits') 
            Data_right=fit_arc.fit_arc(file_in_right,psf_file,channel,defocus_arr[i],ee=0.90,display=fit_display,line_file=line_file,file_temp=output_dir+'/file_temp_'+channel+'_r.fits')
            Data_all_left.append(Data_left)
            Data_all_right.append(Data_right)

        #fiber_all=list(set([s['fiber'] for Data in Data_all_left for s in Data]))
        #lineid_all=list(set([s['lineid'] for Data in Data_all_left for s in Data]))
        #n_fiber=len(fiber_all)
        #n_line=len(lineid_all)


        # Group by sources
        source_table_all=[]

        n_dot=len(Data_all_left[0]['fiber'])
        for i in range(n_dot):
            source_table=Table([[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]],names=['fiber','lineid','wave','defocus','x_left','y_left','Ree_left','FWHMx_left','FWHMy_left','Amp_left','x_right','y_right','Ree_right','FWHMx_right','FWHMy_right','Amp_right'],dtype=['i4','i4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4'])
            for k in range(n_exposure):
                print(k+1,'th exposure')
                Data_left=Data_all_left[k]
                Data_right=Data_all_right[k]
                try:
                    data_left_this=Data_left[i]
                    data_right_this=Data_right[i]
                    fiber_this=data_left_this['fiber']
                    lineid_this=data_left_this['lineid']
                    data_this=[[fiber_this],[lineid_this],[data_left_this['wave']],[defocus_arr[k]],
                                   [data_left_this['xcentroid']],[data_left_this['ycentroid']],[data_left_this['Ree']],[data_left_this['FWHMx']],[data_left_this['FWHMy']],[data_left_this['Amp']],
                                   [data_right_this['xcentroid']],[data_right_this['ycentroid']],[data_right_this['Ree']],[data_right_this['FWHMx']],[data_right_this['FWHMy']],[data_right_this['Amp']]]
                except:
                    fiber_this=data_left_this['fiber']
                    lineid_this=data_left_this['lineid']
                    print('Could not find the source')
                    data_this=[[fiber_this],[lineid_this],[-999],[defocus_arr[k]],[-999],[-999],[-999],[-999],[-999],[-999],[-999],[-999],[-999],[-999],[-999],[-999]]
                source_table.add_row(data_this)
            source_table_all.append(source_table)

    #################################
    # Make a shift map and histogram#
    #################################

    if read:
        df_shift_table=pd.read_csv(output_dir+'/shift_table_'+channel+'.csv')    
    else:
        source_table=source_table_all[0]
        shift_table=Table([[],[],[],[],[],[],[]],names=['fiber','lineid','wave','x','y','shift','flux'],dtype=['i4','i4','f4','f4','f4','f4','f4'])
        shift_arr=[]
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
            y_shift=y_arr_left-y_arr_right
            table_use=source_table[0]
            if table_use['lineid'] != 30 and table_use['FWHMx_left']>1 and table_use['FWHMx_left']<4:
                shift_table.add_row([[table_use['fiber']],[table_use['lineid']],[table_use['wave']],[table_use['x_left']],[table_use['y_left']],[y_shift[0]],[table_use['Amp_left']]])


        ##  write shift table to csv file ##
        df_shift_table=shift_table.to_pandas()
        df_shift_table.to_csv(output_dir+'/shift_table_'+channel+'.csv',index=False)

    pp=PdfPages(output_dir+'/'+channel+'_hartmann_plot.pdf')
    plt.figure(0,figsize=(9.5,4))
    font = {'family':'sans-serif',
            'weight':'normal',
            'size'  :7}
    plt.rc('font', **font)
    plt.subplot(121)
    plt.xlabel('Defocus')
    plt.ylabel('Hartmann Shift')

    x_arr=df_shift_table['x']
    y_arr=df_shift_table['y']
    z_arr=df_shift_table['shift']

    ax=plt.gca()
    ax.set_aspect('equal')
    plt.scatter(x_arr,y_arr,c=z_arr,s=10,alpha=0.5,label='',cmap='rainbow',vmin=-1,vmax=1)
    plt.axis([0,4000, 0, 4000])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(channel+' Hartmann Shift')
    plt.legend(loc='upper left')
    plt.colorbar(fraction=0.046,pad=0.04)


    plt.subplot(122)
    plt.hist(z_arr,20,facecolor='blue',alpha=0.5,range=(-1.5,1.5),label='Median='+str(np.median(z_arr)).strip()[0:5])
    plt.xlabel(channel+' Hartmann Shift')
    plt.ylabel('N')
    plt.legend(loc='upper left')

    pp.savefig()
    plt.close()
    pp.close()

    print('Median Hartmann shift for '+channel+' = '+str(np.median(z_arr)).strip()[0:5])
    print('Check the plot in '+output_dir+'/'+channel+'_hartmann_plot.pdf. xdg-open should work.')




