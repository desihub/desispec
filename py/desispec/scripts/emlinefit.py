"""
desispec.scripts.emlinefit
==========================

Please add module-level documentation.
"""
from time import time
from astropy.time import Time
from desiutil.log import get_logger
import argparse
from desispec.io.emlinefit import (
    get_targetids,
    read_emlines_inputs,
    write_emlines,
    plot_emlines,
)
from desispec.emlinefit import (
    allowed_emnames,
    get_rf_em_waves,
    get_emlines,
)


def parse(options=None, log=None):
    if log is None:
        log = get_logger()
    # AR some defaults
    default_emnames = "OII,HDELTA,HGAMMA,HBETA,OIII,HALPHA"
    default_rr_keys = "TARGETID,Z,ZWARN,SPECTYPE,DELTACHI2"
    default_fm_keys = "TARGET_RA,TARGET_DEC,OBJTYPE"
    #
    parser = argparse.ArgumentParser(description="Simple emission line fitter; primarily designed for ELGs.")
    parser.add_argument(
        "--redrock",
        help="full path to a redrock/zbest file (default=None)",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--coadd",
        help="full path to a coadd file (everest-formatted) (default=None)",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--bitnames",
        help="comma-separated list of target bits to fit from the CMX_TARGET, SV{1,2,3}_DESI_TARGET, or DESI_TARGET mask; if 'ALL', fits all fibers (default=ALL)",
        type=str,
        default="ALL",
        required=False
    )
    parser.add_argument(
        "--output",
        help="full path to output fits file (default=None)",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--emnames",
        help="comma-separated list of emission lines to fit; allowed: {} (default={})".format(",".join(allowed_emnames), default_emnames),
        type=str,
        default=",".join(allowed_emnames),
    )
    parser.add_argument(
        "--rf_fit_hw",
        help="*rest-frame* wavelength width (in A) used for fitting on each side of the line (default=40)",
        type=float,
        default=40,
    )
    parser.add_argument(
        "--min_rf_fit_hw",
        help="minimum requested *rest-frame* width (in A) on each side of the line to consider the fitting (default=20)",
        type=float,
        default=20,
    )
    parser.add_argument(
        "--rf_cont_w",
        help="*rest-frame* wavelength extent (in A) to fit the continuum (default=200)",
        type=float,
        default=200,
    )
    parser.add_argument(
        "--rv",
        help="value of R_V to convert EBV to magnitudes (default=3.1)",
        type=float,
        default=3.1,
    )
    parser.add_argument(
        "--outpdf",
        help="PDF filename for plotting the fitted lines (data + fit) (default=None)",
        type=str,
        default=None,
        required=False
    )
    parser.add_argument(
        "--rr_keys",
        help="comma-separated list of columns from the REDSHIFTS extension to propagate (default={})".format(default_rr_keys),
        type=str,
        default=default_rr_keys,
        required=False,
    )
    parser.add_argument(
        "--fm_keys",
        help="comma-separated list of columns from the FIBERMAP extension to propagate (default={})".format(default_fm_keys),
        type=str,
        default=default_fm_keys,
        required=False,
    )

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    # AR sanity check
    for emname in args.emnames.split(","):
        if emname not in allowed_emnames:
            msg = "{} not in allowed args.emnames ({})".format(emname, ",".join(allowed_emnames))
            log.error(msg)
            raise ValueError(msg)
    #
    for kwargs in args._get_kwargs():
        log.info(kwargs)
    return args


def main(args=None):
    start = time()
    log = get_logger()
    log.info("{:.1f}s\tstart\tTIMESTAMP={}".format(time() - start, Time.now().isot))

    # AR read arguments
    if not isinstance(args, argparse.Namespace):
        args = parse(options=args, log=log)

    # AR read columns + spectra
    rr, fm, waves, fluxes, ivars = read_emlines_inputs(
        args.redrock,
        args.coadd,
        mwext_corr=True,
        rv=args.rv,
        bitnames=args.bitnames,
        rr_keys=args.rr_keys,
        fm_keys=args.fm_keys,
        log=log,
    )
    log.info("{:.1f}s\tread_done\tTIMESTAMP={}".format(time() - start, Time.now().isot))

    # AR fit the emission lines
    emdict = get_emlines(
        rr["Z"],
        waves,
        fluxes,
        ivars,
        emnames=args.emnames.split(","),
        rf_fit_hw=args.rf_fit_hw,
        min_rf_fit_hw=args.min_rf_fit_hw,
        rf_cont_w=args.rf_cont_w,
        log=log,
    )
    log.info("{:.1f}s\tfit_done\tTIMESTAMP={}".format(time() - start, Time.now().isot))

    # AR write output fits
    write_emlines(
        args.output,
        emdict,
        rr=rr,
        fm=fm,
        redrock=args.redrock,
        coadd=args.coadd,
        rf_fit_hw=args.rf_fit_hw,
        min_rf_fit_hw=args.min_rf_fit_hw,
        rf_cont_w=args.rf_cont_w,
        rv=args.rv,
        log=log,
    )
    log.info("{:.1f}s\twrite_done\tTIMESTAMP={}".format(time() - start, Time.now().isot))

    # AR plot?
    if args.outpdf is not None:
        # AR in read_emlines_inputs(), we force Z and TARGETID to be there
        objtypes, spectypes, deltachi2s = None, None, None
        if "OBJTYPE" in fm.dtype.names:
            objtypes = fm["OBJTYPE"]
        if "SPECTYPE" in rr.dtype.names:
            spectypes = rr["SPECTYPE"]
        if "DELTACHI2" in rr.dtype.names:
            deltachi2s = rr["DELTACHI2"]
        plot_emlines(
            args.outpdf,
            rr["Z"],
            emdict,
            emnames=args.emnames.split(","),
            targetids=rr["TARGETID"],
            objtypes=objtypes,
            spectypes=spectypes,
            deltachi2s=deltachi2s,
        )
        log.info("{:.1f}s\tplot_done\tTIMESTAMP={}".format(time() - start, Time.now().isot))

    log.info("{:.1f}s\tdone\tTIMESTAMP={}".format(time() - start, Time.now().isot))
    return 0


if __name__ == "__main__":
    sys.exit(main())

