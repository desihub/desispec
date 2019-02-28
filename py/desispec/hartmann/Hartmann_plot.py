#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.backends.backend_pdf import PdfPages
# scp jguy@mardesi.in2p3.fr:/desi/sperruch/SM02/focus/Blue/HD/resultsHD.dat .

f=open("resultsHD.dat","rb")
d=pickle.load(f, encoding='latin1')
f.close()
pp = PdfPages('Hartmann_plot.pdf')

first=True
plt.figure(0,figsize=(9.5,14))
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)
head_width=0.1
head_length=0.15

plt.subplot(321)
axes = plt.gca()
plt.title('')
plt.xlabel("defocus")
plt.ylabel("ycentroid-xcentroid")

for dd in d[0] :
    
    if first :
        print(dd)
        first=False
        
    table=dd["table"]
    plt.plot(table["defocus"],table["ycentroid"]-table["xcentroid"],"o-")

#plt.show()
plt.subplot(322)
axes = plt.gca()
plt.title('')
plt.xlabel("xcentroid")
plt.ylabel("ycentroid")
#plt.xlim([0,1])
#plt.ylim([0,25])

for dd in d[0] :

    if first :
        print(dd)
        first=False

    table=dd["table"]
    plt.plot(table["xcentroid"],table["ycentroid"],"o-")


plt.subplot(323)
axes = plt.gca()
plt.title('')
plt.xlabel("defocus")
plt.ylabel("FWHM_X")
plt.ylim([2.5,6.0])
for dd in d[0] :
    table=dd["table"]
    plt.plot(table["defocus"],table["FWHMx"],"o-")

plt.subplot(324)
axes = plt.gca()
plt.title('')
plt.xlabel("defocus")
plt.ylabel("FWHM_Y")
plt.ylim([2.5,6.0])
for dd in d[0] :
    table=dd["table"]
    plt.plot(table["defocus"],table["FWHMy"],"o-")

pp.savefig()
plt.close()
pp.close()
