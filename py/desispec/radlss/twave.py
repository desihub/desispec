import numpy as np

def twave(tmin, tmax, dtwave=0.4, shift=5.):
    return  np.arange(-shift + tmin, shift + dtwave + tmax, dtwave)

if __name__ == '__main__':
    twave = twave(5., 15., dtwave=0.4, shift=5.)

    print(twave)
