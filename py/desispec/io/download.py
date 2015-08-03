"""
desispec.io.download
====================

Download files from DESI repository.
"""
from __future__ import absolute_import, division, print_function
from os import environ, makedirs, stat, utime
from os.path import dirname, exists, join
from calendar import timegm
from datetime import datetime
from requests import get
from requests.auth import HTTPDigestAuth
from netrc import netrc


def _auth(machine='portal.nersc.gov'):
    """Get authentication credentials.
    """
    n = netrc()
    u,foo,p = n.authenticators(machine)
    return HTTPDigestAuth(u,p)

def download(filenames,baseurl='https://portal.nersc.gov/project/desi/collab'):
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
    local_cache = join(environ['HOME'],'Desktop','desi')
    a = _auth()
    downloaded_list = list()
    for f in file_list:
        dst = join(local_cache,f)
        if not exists(dst):
            src = join(baseurl,f)
            r = get(src,auth=a)
            if not exists(dirname(dst)):
                makedirs(dirname(dst))
            with open(dst,'w') as d:
                d.write(r.content)
            atime = stat(dst).st_atime
            mtime = timegm(datetime.strptime(r.headers['last-modified'],'%a, %d %b %Y %H:%M:%S %Z').utctimetuple())
            utime(dst,(atime,mtime))
            downloaded_list.append(dst)
    return downloaded_list
