"""
Coadd spectra
"""

from __future__ import absolute_import, division, print_function

from desiutil.log import get_logger
from desispec.io import read_spectra,write_spectra
from desispec.coaddition import coadd,resample_spectra_lin_or_log

def parse(options=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--infile", type=str,  help="input spectra file")
    parser.add_argument("-o","--outfile", type=str,  help="output spectra file")
    parser.add_argument("--nsig", type=float, default=None, help="nsigma rejection threshold for cosmic rays")
    parser.add_argument("--lin-step", type=float, default=None, help="resampling to single linear wave array of given step in A")
    parser.add_argument("--log10-step", type=float, default=None, help="resampling to single log10 wave array of given step in units of log10")
    parser.add_argument("--wave-min", type=float, default=None, help="specify the min wavelength in A (default is the min wavelength in the input spectra)")
    parser.add_argument("--spectro-perf", action="store_true", help="spectro-perf in 1D, i.e. output is uncorrelated and resolution matrix is meaningful, but it's horribly slow (need to implement divide and conquer...)")
    
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args
    
def main(args=None):

    log = get_logger()

    if args is None:
        args = parse()

    if args.lin_step is not None and args.log10_step is not None :
        print("cannot have both linear and logarthmic bins :-), choose either --lin-step or --log10-step")
        return 12
    
    spectra = read_spectra(args.infile)

    coadd(spectra,cosmics_nsig=args.nsig)

    if args.lin_step is not None :
        spectra = resample_spectra_lin_or_log(spectra, linear_step=args.lin_step, wave_min =args.wave_min, spectro_perf = args.spectro_perf)
    if args.log10_step is not None :
        spectra = resample_spectra_lin_or_log(spectra, log10_step=args.log10_step, wave_min =args.wave_min, spectro_perf = args.spectro_perf)
    
    log.debug("writing {} ...".format(args.outfile))
    write_spectra(args.outfile,spectra)
