"""
desispec.scripts.createoverride
=================================

"""
import yaml
import os
import time
import numpy as np

from desispec.io.meta import findfile


def valid_night(night):
    """
    Test whether or not the input night is valid for a YEARMMDD DESI night or not.

    Args:
        night, str, 8 char YEARMMDD.

    Returns:
        bool, True if date is consistent.

    NOTE: Theres a Y2K-esque bug here for dates before 2013 and after 2033
    """
    return night.isnumeric() \
        and np.abs(int(night[:4]) - 2023) < 10 \
        and np.abs(int(night[4:6]) - 6.5) < 6 \
        and np.abs(int(night[6:]) - 16) < 15

def valid_yes_no(string):
    """
    Test whether or not the input is valid for a yes/no answer

    Args:
        string, str, string to test.

    Returns:
        bool, True if input string starts with 'y' or 'n'.
    """
    string = string.lower()
    return len(string) > 0 and string[0] in ['y', 'n']


def get_response(input_question, validation_func):
    """
    Keep requesting inputs until a valid response is given

    Args:
        input_question, str, question to prompt to the user.
        validation_func, function, function that takes a string as input and
            returns a boolean representing whether the string is a valid response
            or not.

    Returns:
        response, str, the string provided by the user that passes validation.
    """
    good_format = False
    while not good_format:
        response = input(input_question)
        if validation_func(response):
            print(f"--> Received valid response: {response}")
            good_format = True
        else:
            print(
                f"--> Format of response {response} was incorrect, please try again")
    return response

def get_night(prompt):
    """
    Prompt the user for a night and return the night as an integer
    """
    night = get_response(prompt, valid_night)
    return int(night)

def is_yes(prompt):
    """
    Prompt the user with a yes/no question and return a boolean yes<->True
    """
    response = get_response(prompt, valid_yes_no)
    return str(response)[0] == 'y'

def create_override_file(args):
    """
    Create an override file with the given inputs. If no inputs given, prompt
    the user for all necessary information.
    """
    if args.night is None:
        night = get_night("What night is it? ")
    else:
        night = int(args.night)

    outdict = {'calibration': dict()}
    if args.linkcal is None:
        linkcal = is_yes("Does this night require cal linking? ")
    else:
        linkcal = args.linkcal.lower() == 'true'
    if linkcal:
        refnight = get_night("What is the reference night? ")
        good_bias = is_yes("Are there valid zeros for biases? ")
        good_cte = is_yes("Are there valid falts for cte corections? ")
        good_badcol = is_yes("Is there a valid dark for badcolumn detection? ")
        good_psf = is_yes("Are there valid arcs for psf generation? ")
        good_flats = is_yes("Are there valid flats for "
                            + "fiberflatnight generation? ")
        if not good_psf and good_flats:
            print("Since good_psf is False, we cannot use the flats. "
                  + "Setting good_flats to False")
            good_flats = False

        if good_bias and good_cte and good_badcol and good_psf and good_flats:
            print("WARNING: All calibrations given as good for the current "
                  + "night, so *NOT* linking calibrations in the override file.")
            linkcal = False
        else:
            outdict['calibration'] = {'linkcal': dict()}
            linkcal_include_list, goods, bads = list(), list(), list()
            if not good_bias:
                linkcal_include_list.append('biasnight')
                bads.append('bias')
            else:
                goods.append('bias')
            if not good_cte:
                linkcal_include_list.append('ctecorrnight')
                bads.append('cte flats')
            else:
                goods.append('cte flats')
            if not good_badcol:
                linkcal_include_list.append('badcolumns')
                bads.append('badcols')
            else:
                goods.append('badcols')
            if not good_psf:
                linkcal_include_list.append('psfnight')
                bads.append('psfs')
            else:
                goods.append('psfs')
            if not good_flats:
                linkcal_include_list.append('fiberflatnight')
                bads.append('flats')
            else:
                goods.append('flats')
            linkcal_include = ','.join(linkcal_include_list)
            outdict['calibration']['linkcal'] = {'refnight': refnight,
                                                 'include': linkcal_include}

    if args.ff_solve_grad is None:
        ff_solve_grad = is_yes("Do we need to set solve_gradient in autocalib_fiberflat? ")
    else:
        ff_solve_grad = args.ff_solve_grad.lower() == 'true'

    if ff_solve_grad:
        print("\nNOTE: Remember to set the reference night in "
              + "desi_spectro_calib\n")
        outdict['calibration']['nightlyflat'] = {'extra_cmd_args':
                                                      ['--autocal-ff-solve-grad']}

    if not linkcal and not ff_solve_grad:
        print("There is nothing to be set in an override "
              + "file, so not creating one.")
    else:
        pathname = findfile('override', night=night)
        dirname = os.path.dirname(pathname)
        if not os.path.exists(dirname):
            print(f"{dirname} doesn't exist. Creating directory.")
            os.makedirs(dirname)
        if os.path.exists(pathname):
            timestamp = time.strftime('%Y%m%d_%Hh%Mm')
            archivepathname = pathname.replace('.yaml', f".{timestamp}.yaml")
            os.rename(pathname, archivepathname)
            print(f"\nWARNING: {pathname} exists. Moving that to "
                  + f"{archivepathname}.\n")
        with open(pathname, 'w') as fil:
            fil.write(f"# DESI override file for {night}\n")
            if linkcal:
                goodstr = ','.join(goods)
                badstr = ','.join(bads)
                fil.write(f"# linkcal notes: current night's {goodstr} were ok; {badstr} were bad\n")
            if ff_solve_grad:
                fil.write(f"# nightlyflat notes: using '--autocal-ff-solve-grad'\n")

            fil.write("\n")
            ## Write out the yaml portion in proper format
            ## default_flow_style is set to None to get bracketed list ([]) but that also does
            ## curly brackets in linkcal, so avoid that if doing linkcal
            if linkcal:
                yaml.safe_dump(outdict, fil)
            else:
                yaml.safe_dump(outdict, fil, default_flow_style=None)
