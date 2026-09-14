import ast
import os
import tempfile
import unittest
from pathlib import Path


class _FakeNumpy:
    @staticmethod
    def array(values):
        return list(values)

    @staticmethod
    def median(values):
        values = sorted(values)
        n = len(values)
        if n % 2 == 1:
            return values[n // 2]
        return 0.5 * (values[n // 2 - 1] + values[n // 2])


class _FakeFitsIO:
    def __init__(self, headers, skies):
        self.headers = headers
        self.skies = skies

    def read_header(self, filename, _extname):
        return self.headers[filename]

    def read(self, filename, _extname):
        return self.skies[filename]


class _FakeLog:
    def info(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None


def _load_get_ctedet_night_expid(glob_func, fitsio_obj):
    nightqa_path = Path(__file__).resolve().parents[1] / "night_qa.py"
    module_ast = ast.parse(nightqa_path.read_text())
    function_node = next(
        node for node in module_ast.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_ctedet_night_expid"
    )
    function_mod = ast.Module(body=[function_node], type_ignores=[])
    namespace = {
        "os": os,
        "glob": glob_func,
        "fitsio": fitsio_obj,
        "np": _FakeNumpy(),
        "log": _FakeLog(),
    }
    exec(compile(function_mod, str(nightqa_path), "exec"), namespace)
    return namespace["get_ctedet_night_expid"]


class TestCTEDetNightExpid(unittest.TestCase):
    night = 20250101

    def _make_case(self, preproc_by_expid, sky_by_expid=None):
        sky_by_expid = dict() if sky_by_expid is None else sky_by_expid
        with tempfile.TemporaryDirectory() as tmpdir:
            prod = os.path.join(tmpdir, "prod")
            preproc_root = os.path.join(prod, "preproc", f"{self.night}")
            os.makedirs(preproc_root)

            expids = sorted(set(preproc_by_expid) | set(sky_by_expid))
            headers, skies, glob_map = {}, {}, {}

            preproc_dirs = []
            for expid in expids:
                preproc_dir = os.path.join(preproc_root, f"{expid:08d}")
                preproc_dirs.append(preproc_dir)
                preproc_pattern = os.path.join(
                    preproc_dir, f"preproc-??-{expid:08d}.fits*"
                )
                if expid in preproc_by_expid:
                    filename = os.path.join(preproc_dir, f"preproc-r0-{expid:08d}.fits")
                    headers[filename] = preproc_by_expid[expid]
                    glob_map[preproc_pattern] = [filename]
                else:
                    glob_map[preproc_pattern] = []

            glob_map[os.path.join(prod, "preproc", f"{self.night}", "*")] = preproc_dirs

            for expid, skyinfo in sky_by_expid.items():
                sky_pattern = os.path.join(
                    prod, "exposures", f"{self.night}", f"{expid:08d}", f"sky-r?-{expid:08d}.fits*"
                )
                skyfile = os.path.join(
                    prod, "exposures", f"{self.night}", f"{expid:08d}", f"sky-r0-{expid:08d}.fits"
                )
                headers[skyfile] = {"OBSTYPE": skyinfo["OBSTYPE"], "EXPID": expid}
                skies[skyfile] = [skyinfo["SKY"]]
                glob_map[sky_pattern] = [skyfile]

            def fake_glob(pattern):
                return list(glob_map.get(pattern, []))

            get_ctedet_night_expid = _load_get_ctedet_night_expid(
                fake_glob,
                _FakeFitsIO(headers, skies),
            )
            return get_ctedet_night_expid(self.night, prod)

    def test_prefers_1s_flat(self):
        expid = self._make_case(
            preproc_by_expid={
                11: {"OBSTYPE": "FLAT", "REQTIME": 1, "EXPID": 11},
                12: {"OBSTYPE": "FLAT", "REQTIME": 3, "EXPID": 12},
                13: {"OBSTYPE": "FLAT", "REQTIME": 10, "EXPID": 13},
            }
        )
        self.assertEqual(expid, 11)

    def test_falls_back_to_3s_then_10s(self):
        expid = self._make_case(
            preproc_by_expid={
                12: {"OBSTYPE": "FLAT", "REQTIME": 3, "EXPID": 12},
                13: {"OBSTYPE": "FLAT", "REQTIME": 10, "EXPID": 13},
            }
        )
        self.assertEqual(expid, 12)

        expid = self._make_case(
            preproc_by_expid={
                12: {"OBSTYPE": "DARK", "REQTIME": 3, "EXPID": 12},
                13: {"OBSTYPE": "FLAT", "REQTIME": 10, "EXPID": 13},
            }
        )
        self.assertEqual(expid, 13)

    def test_falls_back_to_science_if_no_valid_cte_flat(self):
        expid = self._make_case(
            preproc_by_expid={
                12: {"OBSTYPE": "DARK", "REQTIME": 3, "EXPID": 12},
                13: {"OBSTYPE": "FLAT", "REQTIME": 30, "EXPID": 13},
            },
            sky_by_expid={
                20: {"OBSTYPE": "SCIENCE", "SKY": 8},
                21: {"OBSTYPE": "SCIENCE", "SKY": 3},
            },
        )
        self.assertEqual(expid, 21)


if __name__ == "__main__":
    unittest.main()
