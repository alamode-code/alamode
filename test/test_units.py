#!/usr/bin/env python
"""Tests for the LENGTH_UNIT / FORCE_UNIT / FCS_UNIT_OUTPUT options.

1. An alm run whose inputs (&cell, &cutoff, DFSET) are given in angstrom and
   eV/angstrom must reproduce the force constants of the canonical bohr /
   Ry/bohr run.
2. FCS_UNIT_OUTPUT = eV/angstrom must write the .h5 with converted values and
   matching "unit" attributes, while the .fcs stays in Ry/bohr.
3. anphon must give identical phonon bands from (a) the Ry/bohr .h5,
   (b) the eV/angstrom .h5, and (c) a copy of (a) with the unit attributes
   stripped (backward compatibility with older files).
4. FC2FIX with the eV/angstrom .h5 must reproduce FC2FIX with the bohr .h5.
"""

import os
import re
import shutil
import subprocess
import sys

import h5py
import numpy as np

from test_si import gen_alminput_si

BOHR_IN_ANGSTROM = 0.52917721092
RYD_IN_EV = 13.605693122994


def convert_dfset_to_ev_angstrom(src, dst):
    """Rewrite a DFSET (bohr, Ry/bohr) in angstrom and eV/angstrom."""
    with open(src) as f_in, open(dst, "w") as f_out:
        for line in f_in:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                f_out.write(line)
                continue
            vals = [float(x) for x in stripped.split()]
            u = [v * BOHR_IN_ANGSTROM for v in vals[:3]]
            f = [v * RYD_IN_EV / BOHR_IN_ANGSTROM for v in vals[3:]]
            f_out.write(" ".join("%23.16e" % v for v in u + f) + "\n")


def gen_alminput_units(fname, norder, prefix, dfset, opt_extra=None):
    """Same model as gen_alminput_si, but with all inputs in angstrom / eV/angstrom."""
    gen_alminput_si(fname, norder, prefix=prefix, dfset=dfset, opt_extra=opt_extra)
    with open(fname) as f:
        text = f.read()
    text = text.replace(
        "&general\n",
        "&general\n LENGTH_UNIT = angstrom; FORCE_UNIT = eV/angstrom\n"
        " FCS_UNIT_OUTPUT = eV/angstrom\n",
        1,
    )
    text = text.replace(
        "&cell\n 20.406\n", "&cell\n %.16f\n" % (20.406 * BOHR_IN_ANGSTROM)
    )
    text = text.replace(
        "Si-Si None 7.6", "Si-Si None %.16f" % (7.6 * BOHR_IN_ANGSTROM)
    )
    with open(fname, "w") as f:
        f.write(text)


def run_alm(almbin, input_file, log_file):
    with open(log_file, "w") as f:
        ret = subprocess.run([almbin, input_file], stdout=f)
    return ret.returncode


def read_irreducible_fc2(fname):
    """Read the irreducible harmonic FC values from a .fcs file."""
    return np.loadtxt(fname, comments=["#", "*"], skiprows=15, max_rows=26, usecols=[2])


def check_fcs_equivalence():
    ref = read_irreducible_fc2("si222u_a.fcs")
    now = read_irreducible_fc2("si222u_b.fcs")
    if not np.allclose(now, ref, rtol=1e-8, atol=1e-12):
        print("Failed: si222u_b.fcs (angstrom/eV input) != si222u_a.fcs (bohr/Ry input)")
        print("max abs diff:", np.max(np.abs(now - ref)))
        return 1
    return 0


def check_h5_units():
    with h5py.File("si222u_a.h5") as h5a, h5py.File("si222u_b.h5") as h5b:
        checks = [
            ("SuperCell/lattice_vector", "bohr", "angstrom", BOHR_IN_ANGSTROM),
            ("PrimitiveCell/lattice_vector", "bohr", "angstrom", BOHR_IN_ANGSTROM),
            ("ForceConstants/Order2/shift_vectors", "bohr", "angstrom", BOHR_IN_ANGSTROM),
            (
                "ForceConstants/Order2/force_constant_values",
                "Ry/bohr^2",
                "eV/angstrom^2",
                RYD_IN_EV / BOHR_IN_ANGSTROM**2,
            ),
        ]
        for path, unit_a, unit_b, factor in checks:
            attr_a = h5a[path].attrs["unit"]
            attr_b = h5b[path].attrs["unit"]
            if isinstance(attr_a, bytes):
                attr_a = attr_a.decode()
            if isinstance(attr_b, bytes):
                attr_b = attr_b.decode()
            if attr_a != unit_a or attr_b != unit_b:
                print(
                    "Failed: unit attribute of %s: expected (%s, %s), got (%s, %s)"
                    % (path, unit_a, unit_b, attr_a, attr_b)
                )
                return 1
            if not np.allclose(h5b[path][:], h5a[path][:] * factor, rtol=1e-8, atol=1e-12):
                print("Failed: values of %s do not match the analytic unit factor" % path)
                return 1
    return 0


def strip_unit_attributes(src, dst):
    shutil.copy(src, dst)
    with h5py.File(dst, "r+") as h5:
        def strip(name, obj):
            if "unit" in obj.attrs:
                del obj.attrs["unit"]
        h5.visititems(strip)


def gen_anphoninput_units(prefix, fcsfile, fname):
    with open(fname, "w") as f:
        f.write(
            "&general\n PREFIX = %s; MODE = phonons; FCSFILE = %s; KD = Si\n/\n"
            % (prefix, fcsfile)
        )
        f.write("&cell\n  10.203\n  0.0 0.5 0.5\n  0.5 0.0 0.5\n  0.5 0.5 0.0\n/\n")
        f.write(
            "&kpoint\n  1\n  G 0.0 0.0 0.0 X 0.5 0.5 0.0 51\n"
            "  X 0.5 0.5 1.0 G 0.0 0.0 0.0 51\n  G 0.0 0.0 0.0 L 0.5 0.5 0.5 51\n/\n"
        )


def run_anphon(anphonbin, input_file, log_file):
    with open(log_file, "w") as f:
        ret = subprocess.run([anphonbin, input_file], stdout=f)
    return ret.returncode


def check_bands_equivalence():
    bands_a = np.loadtxt("bands_a.bands")
    bands_b = np.loadtxt("bands_b.bands")
    bands_c = np.loadtxt("bands_c.bands")
    if not np.allclose(bands_b, bands_a, atol=1e-4):
        print("Failed: bands from the eV/angstrom .h5 differ from the Ry/bohr .h5")
        print("max abs diff:", np.max(np.abs(bands_b - bands_a)))
        return 1
    if not np.allclose(bands_c, bands_a, atol=1e-8):
        print("Failed: bands from the attribute-stripped .h5 differ (backward compat)")
        print("max abs diff:", np.max(np.abs(bands_c - bands_a)))
        return 1
    return 0


def read_irreducible_fc3(fname):
    """Read the irreducible cubic FC values from a NORDER=2 .fcs file."""
    return np.loadtxt(fname, comments=["#", "*"], skiprows=43, max_rows=36, usecols=[2])


def check_fc2fix_equivalence():
    ref = read_irreducible_fc3("si222u_fix_a.fcs")
    now = read_irreducible_fc3("si222u_fix_b.fcs")
    if not np.allclose(now, ref, rtol=1e-8, atol=1e-12):
        print("Failed: FC2FIX with the eV/angstrom .h5 != FC2FIX with the bohr .h5")
        print("max abs diff:", np.max(np.abs(now - ref)))
        return 1
    return 0


def runtest_units_python():
    """Python API: unit kwargs must reproduce the canonical fit; save_fc units."""
    try:
        from alm import ALM
    except ImportError:
        print("Units Python API             --> skipped (alm package not installed)")
        return 0

    a = 10.263
    lavec = a * np.eye(3)
    xcoord = np.array(
        [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0],
         [0.25, 0.25, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25]]
    )
    numbers = [14] * 8
    rng = np.random.default_rng(1)
    u = 0.02 * rng.standard_normal((12, 8, 3))
    f = -0.1 * u + 0.001 * rng.standard_normal((12, 8, 3))

    def fit(lavec_in, u_in, f_in, length_unit, force_unit):
        obj = ALM(lavec_in, xcoord, numbers,
                  length_unit=length_unit, force_unit=force_unit)
        obj.define(1)
        obj.set_training_data(u_in, f_in)
        obj.optimize()
        values, indices = obj.get_fc(1, mode="origin")
        # The ordering of get_fc entries is not deterministic; key by index.
        return obj, {tuple(idx): val for idx, val in zip(indices, values)}

    alm_ry, fc_ry = fit(lavec, u, f, "bohr", "Ry/bohr")
    _, fc_ev = fit(lavec * BOHR_IN_ANGSTROM, u * BOHR_IN_ANGSTROM,
                   f * RYD_IN_EV / BOHR_IN_ANGSTROM, "angstrom", "eV/angstrom")
    _, fc_ha = fit(lavec, u, 0.5 * f, "bohr", "Ha/bohr")

    scale = max(abs(v) for v in fc_ry.values())
    for name, fc in [("angstrom/eV", fc_ev), ("Ha/bohr", fc_ha)]:
        if set(fc) != set(fc_ry) or max(
                abs(fc_ry[k] - fc[k]) for k in fc_ry) > 1e-10 * scale:
            print("Failed: Python fit with %s input != canonical fit" % name)
            return 1

    # save_fc unit handling (values, attributes, non-stickiness)
    alm_ry.save_fc("py_ry.h5", format="alamode_h5")
    alm_ry.save_fc("py_ev.h5", format="alamode_h5", fc_unit="eV/angstrom")
    alm_ry.save_fc("py_again.h5", format="alamode_h5")
    with h5py.File("py_ry.h5") as h5r, h5py.File("py_ev.h5") as h5e, \
            h5py.File("py_again.h5") as h5a:
        path = "ForceConstants/Order2/force_constant_values"
        unit_e = h5e[path].attrs["unit"]
        if isinstance(unit_e, bytes):
            unit_e = unit_e.decode()
        if unit_e != "eV/angstrom^2":
            print("Failed: save_fc(fc_unit='eV/angstrom') attribute is %r" % unit_e)
            return 1
        factor = RYD_IN_EV / BOHR_IN_ANGSTROM**2
        if not np.allclose(h5e[path][:], h5r[path][:] * factor, rtol=1e-10):
            print("Failed: save_fc(fc_unit='eV/angstrom') values not converted")
            return 1
        if not np.array_equal(h5a[path][:], h5r[path][:]):
            print("Failed: fc_unit leaked into a subsequent default save_fc call")
            return 1

    # invalid-unit error paths
    for kwargs in [dict(length_unit="nm"), dict(force_unit="N")]:
        try:
            ALM(lavec, xcoord, numbers, **kwargs)
        except ValueError:
            pass
        else:
            print("Failed: invalid unit %r accepted" % kwargs)
            return 1
    try:
        alm_ry.save_fc("py.xml", format="alamode", fc_unit="eV/angstrom")
    except ValueError:
        pass
    else:
        print("Failed: fc_unit accepted for the XML format")
        return 1

    print("Units Python API             --> pass")
    return 0


def runtest_units(almbin, anphonbin, project_root):
    shutil.copy(
        "%s/example/Si/reference/DFSET_harmonic" % project_root, "DFSET_harmonic"
    )
    shutil.copy("%s/example/Si/reference/DFSET_cubic" % project_root, "DFSET_cubic")
    with open("DFSET_merged", "w") as f:
        subprocess.run(["cat", "DFSET_harmonic", "DFSET_cubic"], stdout=f)
    convert_dfset_to_ev_angstrom("DFSET_harmonic", "DFSET_harmonic_ang")

    # --- alm: canonical run (A) vs angstrom / eV/angstrom run (B) ---
    gen_alminput_si("UNITS_A.in", 1, prefix="si222u_a", dfset="DFSET_harmonic")
    gen_alminput_units("UNITS_B.in", 1, prefix="si222u_b", dfset="DFSET_harmonic_ang")
    for input_file, log_file in [("UNITS_A.in", "UNITS_A.log"), ("UNITS_B.in", "UNITS_B.log")]:
        if run_alm(almbin, input_file, log_file) != 0:
            print("ALM failed on %s" % input_file)
            return 1

    info = check_fcs_equivalence()
    print("Units ALM input equivalence --> %s" % ("pass" if info == 0 else "failed"))
    if info:
        return 1

    info = check_h5_units()
    print("Units FCS_UNIT_OUTPUT h5     --> %s" % ("pass" if info == 0 else "failed"))
    if info:
        return 1

    # --- anphon: Ry/bohr h5, eV/angstrom h5, attribute-stripped h5 ---
    strip_unit_attributes("si222u_a.h5", "si222u_c.h5")
    for prefix, fcsfile in [
        ("bands_a", "si222u_a.h5"),
        ("bands_b", "si222u_b.h5"),
        ("bands_c", "si222u_c.h5"),
    ]:
        gen_anphoninput_units(prefix, fcsfile, "%s.in" % prefix)
        if run_anphon(anphonbin, "%s.in" % prefix, "%s.log" % prefix) != 0:
            print("ANPHON failed on %s.in" % prefix)
            return 1

    info = check_bands_equivalence()
    print("Units ANPHON h5 roundtrip    --> %s" % ("pass" if info == 0 else "failed"))
    if info:
        return 1

    # --- FC2FIX with mixed-unit h5 files ---
    gen_alminput_si(
        "UNITS_FIX_A.in", 2, prefix="si222u_fix_a", dfset="DFSET_merged",
        opt_extra="FC2FIX = si222u_a.h5\n",
    )
    gen_alminput_si(
        "UNITS_FIX_B.in", 2, prefix="si222u_fix_b", dfset="DFSET_merged",
        opt_extra="FC2FIX = si222u_b.h5\n",
    )
    for input_file, log_file in [
        ("UNITS_FIX_A.in", "UNITS_FIX_A.log"),
        ("UNITS_FIX_B.in", "UNITS_FIX_B.log"),
    ]:
        if run_alm(almbin, input_file, log_file) != 0:
            print("ALM failed on %s" % input_file)
            return 1

    info = check_fc2fix_equivalence()
    print("Units FC2FIX h5 units        --> %s" % ("pass" if info == 0 else "failed"))
    if info:
        return 1

    return runtest_units_python()


if __name__ == "__main__":
    build_dir = os.getcwd()
    project_root = os.path.dirname(build_dir)

    dirname = "%s/test/si_units" % project_root
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    sys.path.insert(0, "%s/test" % project_root)
    os.chdir(dirname)

    almbin = "%s/_build/alm/alm" % project_root
    anphonbin = "%s/_build/anphon/anphon" % project_root

    sys.exit(runtest_units(almbin, anphonbin, project_root))
