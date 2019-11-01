import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
b=pd.read_csv('shift_table_B3.csv')
r=pd.read_csv('shift_table_R3.csv')
z=pd.read_csv('shift_table_Z3.csv')

pp=PdfPages('hartmann_correction.pdf')
plt.figure(0,figsize=(9.5,6))
font = {'family':'sans-serif',
        'weight':'normal',
        'size'  :7}
plt.rc('font', **font)
plt.subplot(231)
x_arr=b['x']
y_arr=b['y']
z_arr=b['shift']
s1=5

ax=plt.gca()
ax.set_aspect('equal')
plt.scatter(x_arr,y_arr,c=z_arr,s=s1,alpha=0.5,label='',cmap='rainbow',vmin=-1,vmax=1)
plt.axis([0,4000, 0, 4000])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('B3 hartmann shift')
plt.legend(loc='upper left')
plt.colorbar(fraction=0.046,pad=0.04)

plt.subplot(232)
x_arr=r['x']
y_arr=r['y']
z_arr=r['shift']

ax=plt.gca()
ax.set_aspect('equal')
plt.scatter(x_arr,y_arr,c=z_arr,s=s1,alpha=0.5,label='',cmap='rainbow',vmin=-1,vmax=1)
plt.axis([0,4000, 0, 4000])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('R3 hartmann shift')
plt.legend(loc='upper left')
plt.colorbar(fraction=0.046,pad=0.04)

plt.subplot(233)
x_arr=z['x']
y_arr=z['y']
z_arr=z['shift']

ax=plt.gca()
ax.set_aspect('equal')
plt.scatter(x_arr,y_arr,c=z_arr,s=s1,alpha=0.5,label='',cmap='rainbow',vmin=-1,vmax=1)
plt.axis([0,4000, 0, 4000])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Z3 hartmann shift')
plt.legend(loc='upper left')
plt.colorbar(fraction=0.046,pad=0.04)

###########################################
########## Histgram #######################
###########################################

plt.subplot(234)
plt.hist(b['shift'],20,facecolor='blue',alpha=0.5,range=(-0.5,1.5),label='Median='+str(np.median(b['shift'])).strip()[0:5])
plt.xlabel('B3 Hartmann Shift')
plt.ylabel('N')
plt.legend(loc='upper left')


plt.subplot(235)
plt.hist(r['shift'],20,facecolor='blue',alpha=0.5,range=(-0.5,1.5),label='Median='+str(np.median(r['shift'])).strip()[0:5])
plt.xlabel('R3 Hartmann Shift')
plt.ylabel('N')
plt.legend(loc='upper left')


plt.subplot(236)
plt.hist(z['shift'],20,facecolor='blue',alpha=0.5,range=(-0.5,1.5),label='Median='+str(np.median(z['shift'])).strip()[0:5])
plt.xlabel('Z3 Hartmann Shift')
plt.ylabel('N')
plt.legend(loc='upper left')

plt.tight_layout()

pp.savefig()
plt.close()
pp.close()






