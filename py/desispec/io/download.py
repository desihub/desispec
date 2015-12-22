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
from multiprocessing import Pool, cpu_count
from netrc import netrc
from .meta import specprod_root


def _auth(machine='portal.nersc.gov'):
    """Get authentication credentials.
    """
    from netrc import netrc
    from requests.auth import HTTPDigestAuth
    n = netrc()
    u,foo,p = n.authenticators(machine)
    return HTTPDigestAuth(u,p)

def filepath2url(path,baseurl='https://portal.nersc.gov/project/desi',release='collab',specprod=None):
    """Convert a fully-qualified file path to a URL.

    Args:
        path: string containing full path to a filename
        baseurl: (optional) string containing the URL of the top-level DESI directory.
        release: (optional) Release version.
        specprod: (optional) String that can be used to override the output of specprod_root().
    """
    if specprod is None:
        specprod = specprod_root()
    if release != 'collab':
        if not release.startswith('release'):
            release = join('release',release)
    return path.replace(specprod,join(baseurl,release,'spectro','redux',environ['PRODNAME']))

def download(filenames,single_thread=False,workers=None):
    """Download files from the DESI repository.

    Args:
        filenames: string or list-like object containing filenames.
        single_thread: (optional) if ``True``, do not use multiprocessing to
            download files.
        workers: (optional) integer indicating the number of worker
            processes to create.

    Returns:
        Full, local path to the file(s) downloaded.
    """
    if isinstance(filenames,str):
        file_list = [ filenames ]
        single_thread = True
    else:
        file_list = filenames
    http_list = [ filepath2url(f) for f in file_list ]
    machine = http_list[0].split('/')[2]
    # local_cache = specprod_root()
    try:
        a = _auth(machine)
    except IOError:
        return [None for f in file_list]
    if single_thread:
        downloaded_list = list()
        for k,f in enumerate(file_list):
            foo = _map_download((file_list[k],http_list[k],a))
            downloaded_list.append(foo)
    else:
        if workers is None:
            workers = cpu_count()
        p = Pool(workers)
        downloaded_list = p.map(_map_download,zip(file_list,http_list,[a]*len(file_list)))
    return downloaded_list

def _map_download(map_tuple):
    """Wrapper function to pass to multiprocess.Pool.map().
    """
    from requests import get
    filename, httpname, auth = map_tuple
    download_success = False
    if exists(filename):
        return filename
    else:
        if auth is None:
            r = get(httpname)
        else:
            r = get(httpname,auth=auth)
        if r.status_code != 200:
            return None
        if not exists(dirname(filename)):
            makedirs(dirname(filename))
        with open(filename,'w') as d:
            d.write(r.content)
        atime = stat(filename).st_atime
        mtime = timegm(datetime.strptime(r.headers['last-modified'],'%a, %d %b %Y %H:%M:%S %Z').utctimetuple())
        utime(filename,(atime,mtime))
        download_success = True
    if download_success:
        return filename
    return None
