import numpy as np
import sys
import desispec.io
import argparse
import astropy.io.fits as fits
from scipy import interpolate

basedir='/Users/christophermanser/Storage/PhD_files/DESI/code/DESI_fittingcode/data/WD_calib'
c = 299792.458 # Speed of light in km/s
model_wave = np.arange(3100, 10902, 2)
stdwave = np.arange(3000,11000.1, 0.1) # The models have fine structure which we want to preserve.
fits_models = fits.open(basedir + '/DESI_WD_2.7_interp.fits')

def parse(options=None):
    parser = argparse.ArgumentParser(description="Fit of white dwarf standard star spectra in frames.")
    parser.add_argument('--frames', type = str, default = None, required=True,
                        help = 'list of path to DESI frame fits files (needs to be same exposure, spectro)')
    #parser.add_argument('--starmodels', type = str, help = 'path of spectro-photometric white dwarf spectra fits')
    parser.add_argument('-o','--outfile', type = str, help = 'output file for normalized white dwarf stdstar model flux')

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args


def air_to_vac(wavin):
    """Converts spectra from air wavelength to vacuum to compare to models"""
    return wavin/(1.0 + 2.735182e-4 + 131.4182/wavin**2 + 2.76249e8/wavin**4)


def fit_DA(spectra, plot = False):
    import matplotlib.pyplot as plt
    from scipy import optimize
    # Loads the input spectrum as inp (read in spectrum), first input, and normalises
    #load lines to fit and crops them
    spec_w = spectra[:,0]
    spec_n, cont_flux = norm_spectra(spectra)
    line_crop = np.loadtxt(basedir+'/line_crop.dat')
    l_crop = line_crop[(line_crop[:,0]>spec_w.min()) & (line_crop[:,1]<spec_w.max())]
    #fit entire grid to find good starting point
    best=fit_line(spec_n, l_crop)
    first_T, first_g = best[2][0], best[2][1]
    all_chi, all_TL  = best[4], best[3]
    
    #-----finds which fitting regions to use based on temperature 
    if first_T>=16000 and first_T<=40000:
        line_crop = np.loadtxt(basedir+'/line_crop.dat')
    elif first_T<16000:
        line_crop = np.loadtxt(basedir+'/line_crop_cool.dat')
    elif first_T>40000:
        line_crop = np.loadtxt(basedir+'/line_crop_hot.dat')
    l_crop = line_crop[(line_crop[:,0]>spec_w.min()) & (line_crop[:,1]<spec_w.max())]
    #------find starting point for secondary solution
    if first_T <= 13000.:
        other_TL, other_chi = all_TL[all_TL[:,0]>=13000.], all_chi[all_TL[:,0]>=13000.]
        other_sol = other_TL[other_chi==np.min(other_chi)]
    elif first_T > 13000.:
        other_TL, other_chi = all_TL[all_TL[:,0]<13000.], all_chi[all_TL[:,0]<13000.]
        other_sol = other_TL[other_chi==np.min(other_chi)]

    # Find best fitting model and then calculate error
    new_best= optimize.fmin(fit_func,(first_T,first_g,10.),args=(spec_n,l_crop,0),
                            disp=0,xtol=1.,ftol=1.,full_output=1)
    other_T = optimize.fmin(err_func,(first_T,first_g),args=(new_best[0][2],
                            new_best[1],spec_n,l_crop),disp=0,xtol=1.,
                            ftol=1.,full_output=0)
    best_T, best_g, best_rv = new_best[0][0], new_best[0][1], new_best[0][2]
    print("\nFirst solution")
    print("T = ", best_T, abs(best_T-other_T[0]))
    print("logg = ", best_g/100, abs(best_g-other_T[1])/100)
    print("rv =",best_rv)
    
    #new_best= optimize.minimize(fit_func,(first_T,first_g,10.), bounds=((6005,99000),(401,945),(None,None)),
    #                            args=(spec_n,l_crop,0), method="L-BFGS-B")#xtol=1. ftol=1.
    #other_T = optimize.minimize(err_func,(first_T,first_g), bounds=((6005,99900),(401,945)),
    #                            args=(new_best.x[2],new_best.fun,spec_n,l_crop),method="L-BFGS-B")
    #best_T, best_g, best_rv = new_best.x[0], new_best.x[1], new_best.x[2]
    #print("\nFirst solution")
    #print("T = ", best_T, abs(best_T-other_T.x[0]))
    #print("logg = ", best_g/100, abs(best_g-other_T.x[1])/100)
    #print("rv =",best_rv)
    
    sec_T, sec_g = other_sol[0][0], other_sol[0][1]
    #-----as above, for the second solution. 
    if sec_T>=16000 and sec_T<=40000:
        line_crop = np.loadtxt(basedir+'/line_crop.dat')
    elif sec_T<16000:
        line_crop = np.loadtxt(basedir+'/line_crop_cool.dat')
    elif sec_T>40000:
        line_crop = np.loadtxt(basedir+'/line_crop_hot.dat')
    l_crop = line_crop[(line_crop[:,0]>spec_w.min()) & (line_crop[:,1]<spec_w.max())]
    
    #repeat fit for secondary solution
    sec_best = optimize.fmin(fit_func,(sec_T,sec_g,best_rv), args=(spec_n,l_crop,0),
                             disp=0,xtol=1.,ftol=1.,full_output=1)
    other_T2 = optimize.fmin(err_func,(sec_T,sec_g), args=(best_rv,sec_best[1],spec_n,
                             l_crop,),disp=0, xtol=1.,ftol=1.,full_output=0)
    s_best_T, s_best_g, s_best_rv = sec_best[0][0], sec_best[0][1], sec_best[0][2]
    
    s_best_T, s_best_g, s_best_rv = sec_best[0][0], sec_best[0][1], sec_best[0][2]
    print("\nSecond solution")
    print("T = ", s_best_T, abs(s_best_T-other_T2[0]))
    print("logg = ", s_best_g/100, abs(s_best_g-other_T2[1])/100)
  
    #sec_best= optimize.minimize(fit_func,(other_sol[0][0],other_sol[0][1],best_rv),#
    #                            bounds=((6005,99900),(401,945),(None,None)),args=(spec_n,l_crop,0),
    #                            method="L-BFGS-B")#xtol=1. ftol=1.
    #other_T2 = optimize.minimize(err_func,(other_sol[0][0],other_sol[0][1]),
    #                            bounds=((6005,99900),(401,945)),
    #                            args=(new_best.x[2],sec_best.fun,spec_n,l_crop),method="L-BFGS-B")
    #s_best_T, s_best_g, s_best_rv = sec_best.x[0], sec_best.x[1], sec_best.x[2]
    
    #print("\nSecond solution")
    #print("T = ", s_best_T, abs(s_best_T-other_T2.x[0]))
    #print("logg = ", s_best_g/100, abs(s_best_g-other_T2.x[1])/100)

    StN = (np.sum(spectra[:,1][(spec_w >= 4500.0) & (spec_w <= 4750.0)] / 
          spectra[:,2][(spec_w >= 4500.0) & (spec_w <= 4750.0)] ) / 
          spectra[:,1][(spec_w >= 4500.0) & (spec_w <= 4750.0)].size)
    print('StN = %.1f'%(StN))    
    return best_T, best_g, best_rv


def err_func(x,rv,valore,specn,lcrop):
    """Script finds errors by minimising function at chi+1 rather than chi
       Requires: x; rv - initial guess of T, g; rv
       valore - the chi value of the best fit
       specn/lcrop - normalised spectrum / list of cropped lines to fit"""
    tmp = tmp_func(x[0], x[1], rv, specn, lcrop)
    if tmp != 1: return abs(tmp[3]-(valore+1.)) #this is quantity that gets minimized 
    else: return 1E30


def fit_func(x,specn,lcrop,mode=0):
    """Requires: x - initial guess of T, g, and rv
       specn/lcrop - normalised spectrum / list of cropped lines to fit
       mode=0 is for finding bestfit, mode=1 for fitting & retriving specific model """
    tmp = tmp_func(x[0], x[1], x[2], specn, lcrop)
    if tmp == 1: return 1E30
    elif mode==0: return tmp[3] #this is the quantity that gets minimized
    elif mode==1: return tmp[0], tmp[1], tmp[2]


def fit_line(_sn, l_crop):
    """Input _sn, l_crop <- Normalised spectrum & a cropped line list
    Calc and return chi2, list of arrays of spectra, and scaled models at lines """
    #load normalised models and linearly interp models onto spectrum wave
    m_wave = np.arange(3700,7002,2)
    m_flux_n,m_param = norm_models()
    sn_w = _sn[:,0]
    m_flux_n_i = interpolate.interp1d(m_wave,m_flux_n,kind='linear', bounds_error=False,
                                      fill_value=0)(sn_w)
    #Crops models and spectra in a line region, renorms models, calculates chi2
    tmp_lines_m, lines_s, l_chi2 = [],[],[]
    for i in range(len(l_crop)):
        l_c0,l_c1 = l_crop[i,0],l_crop[i,1] 
        l_m = m_flux_n_i.transpose()[(sn_w>=l_c0)&(sn_w<=l_c1)].transpose()
        l_s = _sn[(sn_w>=l_c0)&(sn_w<=l_c1)]
        l_m = l_m*np.sum(l_s[:,1])/np.sum(l_m,axis=1).reshape([len(l_m),1])
        l_chi2.append( np.sum(((l_s[:,1]-l_m)/l_s[:,2])**2,axis=1) )
        tmp_lines_m.append(l_m)
        lines_s.append(l_s)
    #mean chi2 over lines and stores best model lines for output
    lines_chi2, lines_m = np.sum(np.array(l_chi2),axis=0), []
    is_best = lines_chi2==lines_chi2.min()
    for i in range(len(l_crop)): lines_m.append(tmp_lines_m[i][is_best][0])
    best_TL = m_param[is_best][0]
    return  lines_s,lines_m,best_TL,m_param,lines_chi2

def interpolating_model_DA(temp,grav, fine_models = False):
    """Interpolate model atmospheres given an input Teff and logg
    models are saved as a numpy array to increase speed"""
    # PARAMETERS # 
    if (temp<6000. or temp>100000. or grav<4.0 or grav>9.5): return [],[]
    teff=np.array([6000.,6250.,6500.,6750.,7000.,7250.,7500.,7750.,8000.,
                   8250.,8500.,8750.,9000.,9250.,9500.,9750.,10000.,10100.,
                   10200.,10250.,10300.,10400.,10500.,10600.,10700.,10750.,
                   10800.,10900.,11000.,11100.,11200.,11250.,11300.,11400.,
                   11500.,11600.,11700.,11750.,11800.,11900.,12000.,12100.,
                   12200.,12250.,12300.,12400.,12500.,12600.,12700.,12750.,
                   12800.,12900.,13000.,13500.,14000.,14250.,14500.,14750.,
                   15000.,15250.,15500.,15750.,16000.,16250.,16500.,16750.,
                   17000.,17250.,17500.,17750.,18000.,18250.,18500.,18750.,
                   19000.,19250.,19500.,19750.,20000.,21000.,22000.,23000.,
                   24000.,25000.,26000.,27000.,28000.,29000.,30000.,35000.,
                   40000.,45000.,50000.,55000.,60000.,65000.,70000.,75000.,
                   80000.,90000.,100000.])
    logg=np.arange(4.00, 9.75, 0.25)
    # INTERPOLATION #
    g1,g2 = np.max(logg[logg<=grav]),np.min(logg[logg>=grav])
    if g1!=g2: g = (grav-g1)/(g2-g1)
    else: g=0
    t1,t2 = np.max(teff[teff<=temp]),np.min(teff[teff>=temp])
    if t1!=t2: t = (temp-t1)/(t2-t1)          
    else: t=0	
    models = ['%06d_%d'%(i, j*100) for i in [t1,t2] for j in [g1,g2]]
    try:
        if fine_models == True:
            fits_model_path = basedir + '/fine_model_grid/'
            m11 = np.genfromtxt('%sda%s_fine.dat'%(fits_model_path, models[0]))
            m12 = np.genfromtxt('%sda%s_fine.dat'%(fits_model_path, models[1]))
            m21 = np.genfromtxt('%sda%s_fine.dat'%(fits_model_path, models[2]))
            m22 = np.genfromtxt('%sda%s_fine.dat'%(fits_model_path, models[3]))
            flux_i = (1-t)*(1-g)*m11 + t*(1-g)*m21 +t*g*m22 + (1-t)*g*m12
            return flux_i
        else:
            m11 = fits_models[models[0]].data['FLUX']
            m12 = fits_models[models[1]].data['FLUX']
            m21 = fits_models[models[2]].data['FLUX']
            m22 = fits_models[models[3]].data['FLUX']
            flux_i = (1-t)*(1-g)*m11 + t*(1-g)*m21 +t*g*m22 + (1-t)*g*m12
            return np.dstack((model_wave, flux_i))[0]
    except: return [],[]


def norm_models():
    """ Import Normalised WD Models
    Return [out_m_wave,norm_m_flux,model_param] """
    model_param = np.loadtxt(basedir+ '/da.lst', usecols=[1,2])
    out_m_wave = np.arange(3700, 7002, 2)
    norm_m_flux = np.loadtxt(basedir + '/DESI_WD_norm_interp.grid')
    if out_m_wave.shape[0] != norm_m_flux.shape[1]:
        raise wdfitError('l and f arrays not correct shape in norm_models')
    return [norm_m_flux,model_param]
    
    
def models():
    """Loads interpolated WD models that are convolved for quick fitting"""
    return fits.open(basedir + '/DESI_WD_2.7_interp.fits')


def norm_spectra(spectra):
    """ Normalised spectra by DA WD continuum regions. Spec of form 
    array([wav,flux,err]) (err not necessary)
    Optional:
        Edit n_range_s to change whether region[j] is fitted for a peak or mean'd
        add_infinity=False : add a spline point at [inf,0]
    returns spectra, cont_flux """
    start_n=np.array([3770.,3796.,3835.,3895.,3995.,4130.,4490.,4620.,5070.,5200.,
                      6000.,7000.,7550.,8400.])
    end_n=np.array([3795.,3830.,3885.,3960.,4075.,4290.,4570.,4670.,5100.,5300.,
                    6100.,7050.,7600.,8450.])
    n_range_s=np.array(['P','P','P','P','P','P','M','M','M','M','M','M','M','M'])
    if len(spectra[0])>2:
        snr = np.zeros([len(start_n),3])
        spectra[:,2][spectra[:,2]==0.] = spectra[:,2].max()
    else: 
        snr = np.zeros([len(start_n),2])
    wav = spectra[:,0]
    for j in range(len(start_n)):
        if (start_n[j] < wav.max()) & (end_n[j] > wav.min()):
            _s = spectra[(wav>=start_n[j])&(wav<=end_n[j])]
            _w = _s[:,0]
            #Avoids gappy spectra
            k=3 # Check if there are more points than 3
            if len(_s)>k:
                #interpolate onto 10* resolution
                l = np.linspace(_w.min(),_w.max(),(len(_s)-1)*10+1)
                if len(spectra[0])>2:
                    tck = interpolate.splrep(_w,_s[:,1],w=1/_s[:,2], s=2000)
                    #median errors for max/mid point
                    snr[j,2] = np.median(_s[:,2]) / np.sqrt(len(_w))
                else: tck = interpolate.splrep(_w,_s[:,1],s=0.0)
                f = interpolate.splev(l,tck)
                #find maxima and save
                if n_range_s[j]=='P': snr[j,0], snr[j,1] = l[f==f.max()][0], f.max()
                #find mean and save
                elif n_range_s[j]=='M': snr[j,0:2] = np.mean(l), np.mean(f)
                else: print('Unknown n_range_s, ignoring')
    snr = snr[ snr[:,0] != 0 ]
    #t parameter chosen by eye. Position of knots.
    if snr[:,0].max() < 6460: knots = [3000,4900,4100,4340,4860,int(snr[:,0].max()-5)]
    else: knots = [3885,4340,4900,6460]
    if snr[:,0].min() > 3885:
        print('Warning: knots used for spline norm unsuitable for high order fitting')
        knots=knots[1:]
    if (snr[:,0].min() > 4340) or (snr[:,0].max() < 4901): 
        knots=None # 'Warning: knots used probably bad'
    try: #weight by errors
        if len(spectra[0])>2: 
            tck = interpolate.splrep(snr[:,0],snr[:,1], w=1/snr[:,2], t=knots, k=3)
        else: tck = interpolate.splrep(snr[:,0],snr[:,1], t=knots, k=3)
    except ValueError:
        knots=None
        if len(spectra[0])>2: 
            tck = interpolate.splrep(snr[:,0],snr[:,1], w=1/snr[:,2], t=knots, k=3)
        else: tck = interpolate.splrep(snr[:,0],snr[:,1], t=knots, k=3)
    cont_flux = interpolate.splev(wav,tck).reshape(wav.size, 1)
    spectra_ret = np.copy(spectra)
    spectra_ret[:,1:] = spectra_ret[:,1:]/cont_flux
    return spectra_ret, cont_flux    

def tmp_func(_T, _g, _rv, _sn, _l):
    model=interpolating_model_DA(_T,(_g/100))
    try: norm_model, m_cont_flux=norm_spectra(model)
    except:
        print("Could not load the model")
        return 1
    else:
        #interpolate normalised model and spectra onto same wavelength scale
        m_wave_n, m_flux_n, sn_w = norm_model[:,0]*(_rv+c)/c, norm_model[:,1], _sn[:,0]
        m_flux_n_i = interpolate.interp1d(m_wave_n,m_flux_n,kind='linear')(sn_w)
        #Initialise: normalised models and spectra in line region, and chi2
        lines_m, lines_s, sum_l_chi2 = [],[],0
        flux_s,err_s=[],[]
        for i in range(len(_l)):
            # Crop model and spec to line
            l_c0,l_c1 = _l[i,0],_l[i,1]
            l_m = m_flux_n_i.transpose()[(sn_w>=l_c0)&(sn_w<=l_c1)].transpose()
            l_s = _sn[(sn_w>=l_c0)&(sn_w<=l_c1)]
            #renormalise models to spectra in line region & calculate chi2+sum
            l_m = l_m*np.sum(l_s[:,1])/np.sum(l_m)
            lines_m.append(l_m), lines_s.append(l_s)
            flux_s.append(l_s[:,1]),err_s.append(l_s[:,2])
        all_lines_m=np.concatenate((lines_m),axis=0)
        all_lines_s=np.concatenate((flux_s),axis=0)
        all_err_s=np.concatenate((err_s),axis=0)
        sum_l_chi2=np.sum(((all_lines_s-all_lines_m)/all_err_s)**2)
        return lines_s, lines_m, model, sum_l_chi2


def line_info(spec_blue, spec_red):
    import matplotlib.pyplot as plt
    # Define the normalising regions
    wb, wr = spec_blue[:,0], spec_red[:,0]
    fb, fr = spec_blue[:,1], spec_red[:,1]
    w_bool_b = (((wb > 3650) & (wb < 3700)) | ((wb > 3850) & (wb < 3870)) | 
               ((wb > 4220) & (wb < 4245)) | ((wb > 4525) & (wb < 4575)) | 
               ((wb > 5250) & (wb < 5400)))# | ((wb > 5750) & (wb < 5825)) | 
               #((wb > 6000) & (wb < 6550)) | ((wb > 6920) & (wb < 7000)) | 
               #((wb > 7140) & (wb < 7220))]

    w_bool_r = (((wr > 3650) & (wr < 3700)) | ((wr > 3850) & (wr < 3870)) | 
               ((wr > 4220) & (wr < 4245)) | ((wr > 4525) & (wr < 4575)) | 
               ((wr > 5250) & (wr < 5400)) | ((wr > 5750) & (wr < 5825)) | 
               ((wr > 6000) & (wr < 6550)) | ((wr > 6920) & (wr < 7000)) | 
               ((wr > 7140) & (wr < 7220)))
                
    tmp_wb, tmp_fb = wb[w_bool_b], fb[w_bool_b]
    func_poly = np.polyfit(tmp_wb, tmp_fb,5)
    p = np.poly1d(func_poly)
    ybest_b=p(wb)
    
    tmp_wr, tmp_fr = wr[w_bool_r], fr[w_bool_r]
    func_poly = np.polyfit(tmp_wr, tmp_fr,7)
    p = np.poly1d(func_poly)
    ybest_r=p(wr)
    
    # PART OF TESTING NORMALISATION
    #plt.axvline(3700, ls= '--')
    #plt.axvline(5300, ls= '--')
    #plt.axvline(5800, ls= '--')
    #plt.axvline(7200, ls= '--')
    #start,end=np.loadtxt(basedir + '/features.lst',skiprows=1, comments='#',
    #                     delimiter='-', usecols=(0,1),unpack=True)
    #print(start, end)
    #for j in range(start.size):
    #  plt.scatter(start[j], 3)
    #  plt.scatter(end[j],   3)
    #plt.plot(wb, fb)
    #plt.plot(wb, ybest_b)
    #plt.plot(wb, fb/ybest_b)
    #plt.plot(wr, fr)
    #plt.plot(wr, ybest_r)
    #plt.plot(wr, fr/ybest_r)
    #plt.show()
    
    # Calculate *all* line strengths and then returns them.
    start,end=np.loadtxt(basedir + '/features.lst', comments='#',
                         delimiter='-', usecols=(0,1),unpack=True)
    tmp_rtn = np.zeros(start.size)
    for i in range(start.size):
        if start[i] < 5500: wave, flux, ybest = wb, fb, ybest_b
        else: wave, flux, ybest = wr, fr, ybest_r
        line_bool = (wave>start[i]) & (wave<end[i])
        f_s, f_m = flux[line_bool], ybest[line_bool]
        ratio_line=np.average(f_s)/np.average(f_m)
        tmp_rtn[i] = ratio_line
    return tmp_rtn


def load_training_set():
    import pickle
    with open(basedir + '/training_test', 'rb') as f: return pickle.load(f) 


def WD_classify(spec_blue, spec_red, training_set = None):
    import pickle
    lines=line_info(spec_blue, spec_red)
    if training_set == None:
        print("Please feed me a training set")
        sys.exit()
    predictions = training_set.predict(lines.reshape(1, -1)) 
    return predictions


def WD_stdstar_models():
    """ Fits the balmer lines for each WD STD star on a petal and generates a best 
    fitting model, which is then saved to a fits file to be used in the 
    flux_calibration.py. code.
    Returns: - A list of temp, logg, and the wave & flux for the best fitting models
             - The fibers which contain white dwarfs if it is needed."""
    training_set = load_training_set()
    args = parse()
    frame = args.frames
    # Reading the frames
    frame_b = desispec.io.read_frame(frame)
    frame_r = desispec.io.read_frame(frame.replace('-b', '-r'))
    wave_b, wave_r = frame_b.wave, frame_r.wave
    # NEEDS TO BE UPDATED TO READ ONLY WD STD STARS!!!!
    wd_fibers = np.where(frame_b.fibermap['OBJTYPE'] == 'STD')[0]
    noss = wd_fibers.size
    model_lst = [0]*noss
    temp_lst, logg_lst, rv_lst = np.zeros(noss), np.zeros(noss), np.zeros(noss)
    for i, fiber in enumerate(wd_fibers):
        flux_b, err_b = frame_b.flux[fiber], frame_b.ivar[fiber] ** -0.5 
        flux_r, err_r = frame_r.flux[fiber], frame_r.ivar[fiber] ** -0.5
        b_spec = np.vstack((wave_b, flux_b, err_b)).transpose()
        r_spec = np.vstack((wave_r, flux_r, err_r)).transpose()
        b_spec = b_spec[np.isnan(b_spec[:,1])==False]
        r_spec = r_spec[np.isnan(r_spec[:,1])==False]
        
        # Determines whether the WD is a DA or not.
        WD_type = WD_classify(b_spec, r_spec, training_set = training_set)
        print(WD_type[0])
        
        # Finding the best fitting model on the blue arm (only arm needed for DA WDs)
        temp, logg, rv = fit_DA(b_spec)
        flux_i = interpolating_model_DA(temp,logg/100, fine_models = True)
        # NEED TO READ IN GAIA MAG FROM FRAME
        gaia_mag = np.random.randint(16, 19)
        # Rescales the model to the Gaia G magnitude of the object
        model_lst[i] = scale_model_gaiaG(stdwave, flux_i, gaia_mag)
        temp_lst[i], logg_lst[i], rv_lst[i] = temp, logg, rv*10E3
    
    # Now write the normalized flux for all best models to a file
    normflux= np.array(model_lst) # THIS IS NOT NORMALISED YET!!!
    data={}
    data['LOGG'] = logg_lst
    data['TEFF'] = temp_lst
    data['REDSHIFT'] = rv_lst
    desispec.io.write_stdstar_models(args.outfile, normflux, stdwave, wd_fibers, data)
    return model_lst, temp, logg, rv, wd_fibers
    

def scale_model_gaiaG(model_wave, model_flux,ob_mag):
    """
    Takes a white dwarf model and returns the model normalised
    to a given Gaia G magnitude.
    """
    from scipy.integrate import simps
    # Load Gaia filter 
    filter_w, filter_r = np.loadtxt(basedir + '/GAIA_GAIA2r.G.dat',usecols=(0,1),unpack=True)
    norm_filter = simps(filter_r,filter_w)                 #Integrate over wavelength
    bool_check1 = (filter_w>np.min(model_wave)) & (filter_w<np.max(model_wave))
    filter_r, filter_w = filter_r[bool_check1]/norm_filter, filter_w[bool_check1]
    min, max = np.min(filter_w)-100., np.max(filter_w)+100.
    bool_check2 = (model_wave >= min) & (model_wave <= max)
    filt_spec2, filt_spec1 = model_flux[bool_check2], model_wave[bool_check2]
    
    tck_filt = interpolate.interp1d(filt_spec1,filt_spec2,kind='linear',assume_sorted=True)
    spectral_filt = tck_filt(filter_w)
    to_integrate = spectral_filt*filter_r
    flux_filt = simps(to_integrate,filter_w)
    # Convert Flux from Flam to Fnu (Jy), and then to magnitude
    # Constants for effective wavelength and zeropoint for Gaia G
    ew = 5836. # for Gaia G
    c = 3E10 # cgs speed of light
    zp = 2.495e-9 * ew**2 *1.e15 / c # Gaia G then convert to Jy  
    fluxval = flux_filt * (ew**2 * 1.e-8 / c)  
    band_mag = -2.5 * np.log10(fluxval / (zp * 1.e-23))
    # Scale the model to the correct magnitude
    return model_flux*10**(0.4*(band_mag-ob_mag))