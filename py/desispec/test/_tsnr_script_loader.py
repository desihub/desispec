"""
Shared helper for loading bin/desi_tsnr_afterburner in test environments.

The bin script imports the full desispec stack which requires packages
(pytz, numba, speclite, sqlalchemy) that may not be installed in the
test environment.  This module stubs those packages before loading so
that the import succeeds without the full DESI software stack.
"""

import importlib.util
import os
import sys
import types
from importlib.machinery import SourceFileLoader


_SCRIPT_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'bin', 'desi_tsnr_afterburner'
))


def _make_stub(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


def _ensure_stubs():
    """Insert minimal stubs for packages that may not be installed."""
    if 'pytz' not in sys.modules:
        try:
            import pytz  # noqa: F401
        except ImportError:
            m = _make_stub('pytz')
            m.UTC = None
            m.utc = None
            m.timezone = lambda tz: None

    if 'numba' not in sys.modules:
        try:
            import numba  # noqa: F401
        except ImportError:
            m = _make_stub('numba')

            def _passthrough(func=None, **kwargs):
                if func is not None:
                    return func
                return lambda f: f

            m.jit = _passthrough
            m.njit = _passthrough
            m.float64 = None
            m.int64 = None
            m.boolean = None
            m.prange = range
            _make_stub('numba.core')
            _make_stub('numba.core.types')

    if 'speclite' not in sys.modules:
        try:
            import speclite  # noqa: F401
        except ImportError:
            spec_m = _make_stub('speclite')
            filters_m = _make_stub('speclite.filters')
            spec_m.filters = filters_m

    if 'sqlalchemy' not in sys.modules:
        try:
            import sqlalchemy  # noqa: F401
        except ImportError:
            sa = _make_stub('sqlalchemy')
            _make_stub('sqlalchemy.orm')
            _make_stub('sqlalchemy.exc')
            sa.create_engine = lambda *a, **kw: None
            sa.Column = lambda *a, **kw: None
            sa.Integer = None
            sa.String = None


def load_script():
    """Load bin/desi_tsnr_afterburner as a module and return it.

    Stubs unavailable optional packages so the import succeeds in test
    environments that do not have the full DESI software stack installed.

    Returns:
        The loaded module object.
    """
    _ensure_stubs()
    loader = SourceFileLoader('desi_tsnr_afterburner', _SCRIPT_PATH)
    spec = importlib.util.spec_from_file_location(
        'desi_tsnr_afterburner', _SCRIPT_PATH, loader=loader
    )
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod
