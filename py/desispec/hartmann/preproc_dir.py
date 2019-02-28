import glob
import os,pdb
which_spec_date='SM02/2018/20180906/'
in_dir='/project/projectdirs/desi/spectro/teststand/rawdata/'+which_spec_date
out_dir='/scratch2/scratchdirs/zhangkai/preproc/'+which_spec_date
#os.chdir(in_dir)
files=glob.glob(in_dir+'*.fits')
n_file=len(files)
camera_arr=['b1']
n_camera=len(camera_arr)
for i in range(n_file):
    file_this=files[i]
    ind_w=file_this.find('WINLIGHT')
    expnum=file_this[ind_w+9:ind_w+17]
    
    for j in range(n_camera):
        camera_this=camera_arr[j]
        command='desi_preproc -i '+file_this+' --cameras '+camera_this+' --outdir '+out_dir+' -o '+out_dir+'/WINLIGHT_'+expnum+'_preproc.fits'
        os.system(command)
