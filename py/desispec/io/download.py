"""
desispec.io.download
====================

Download files from DESI repository.
"""

from requests import get
from requests.auth import HTTPDigestAuth
from netrc import netrc


def _auth(machine='portal.nersc.gov'):
    """Get authentication credentials.
    """
    n = netrc()
    u,foo,p = n.authenticators(machine)
    return HTTPDigestAuth(u,p)

def download(filenames,baseurl='https://portal.nersc.gov/project/desi'):
    """Download files from the DESI repository.

    Args:
        filenames: string or list-like object containing filenames.
        baseurl: (optional) URL to look for files.

    Returns:
        Full, local path to the file(s) downloaded.
    """
    if isinstance(filenames,str):
        file_list = [filenames]
    else:
        file_list = filenames
    machine = baseurl.split('/')[2]
