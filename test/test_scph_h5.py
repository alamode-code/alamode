#!/usr/bin/env python
"""Tests of the unified SCPH state file (PREFIX.scph.h5).

Covers: fresh-run schema and physical-output consistency (BaTiO3 SCPH with
cell relaxation), pure-h5 restart, and the FC2_TEMPERATURE direct read of
the temperature-dependent FC2, validated against the tools/dfc2.py route.
"""

import os
import shutil
import subprocess
import sys

import h5py
import numpy as np

from test_batio3 import check_consistency_anphon, copy_input_files

PREFIX = "cBTO222_scph"


def run_anphon(anphonbin, input_file, logfile):
    with open(logfile, "w") as f:
        ret = subprocess.run([anphonbin, input_file], stdout=f, stderr=subprocess.STDOUT)
    return ret.returncode


def check_fresh_run(anphonbin, reference_dir):
    if run_anphon(anphonbin, "BTO_scph_thermo.in", "fresh.log") != 0:
        print("fresh SCPH run failed")
        return 1

    # Physical text outputs must match the references (same checks as test_batio3).
    if check_consistency_anphon(reference_dir, abs_tol=1.0e-8, rel_tol=1.0e-9) != 0:
        return 1

    # Legacy restart files must not be written in h5 mode (.V0 is kept as a
    # human-readable output).
    for fname in (PREFIX + ".scph_dymat", PREFIX + ".renorm_harm_dymat"):
        if os.path.exists(fname):
            print("legacy restart file %s should not be written in h5 mode" % fname)
            return 1

    with h5py.File(PREFIX + ".scph.h5", "r") as f:
        if f.attrs["schema"] != "alamode:scph_state" or f.attrs["format_version"] != 1:
            print("schema attributes are wrong:", dict(f.attrs))
            return 1
        if f.attrs["mode"] != "SCPH" or f.attrs["complete"] != 1:
            print("mode/complete attributes are wrong")
            return 1

        nt = f["settings/temperatures"].shape[0]
        delta = f["dymat/delta"][...]
        if delta.dtype != np.complex128 or delta.shape[0] != nt:
            print("dymat/delta has wrong dtype or shape")
            return 1

        # /V0 must equal the text .V0 output.
        v0_txt = np.loadtxt(PREFIX + ".V0")
        if not np.allclose(f["V0"][...], v0_txt[:, 1], rtol=1e-12):
            print("/V0 disagrees with the text .V0 output")
            return 1

        # The temperature-dependent FC2 minus the base FC2 must reproduce the
        # text .scph_dfc2 corrections (first temperature block).
        base = f["ForceConstants/Order2/force_constant_values"][...]
        total = f["ForceConstants/Order2_temperature_dependent/force_constant_values"][...]
        rows = []
        with open(PREFIX + ".scph_dfc2") as fh:
            lines = fh.read().splitlines()
        natmin = f["PrimitiveCell/number_of_atoms"][()]
        idx = 5 + int(natmin)  # 3 lattice + 1 counts + 1 elements + natmin positions
        if not lines[idx].startswith("# Temp"):
            print("unexpected .scph_dfc2 layout")
            return 1
        idx += 1
        while idx < len(lines) and lines[idx].strip() and not lines[idx].startswith("# Temp"):
            rows.append(float(lines[idx].split()[-1]))
            idx += 1
        if not np.allclose(total[0] - base, np.array(rows), rtol=1e-8, atol=1e-14):
            print("temperature-dependent FC2 disagrees with .scph_dfc2")
            return 1
    return 0


def check_h5_restart(anphonbin):
    thermo_ref = np.loadtxt(PREFIX + ".scph_thermo")
    if run_anphon(anphonbin, "BTO_scph_thermo.in", "restart.log") != 0:
        print("h5 restart run failed")
        return 1
    with open("restart.log") as f:
        log = f.read()
    if "RESTART_SCPH is true" not in log:
        print("restart was not triggered")
        return 1
    if not np.allclose(thermo_ref, np.loadtxt(PREFIX + ".scph_thermo"), rtol=1e-10):
        print(".scph_thermo differs after h5 restart")
        return 1
    return 0


def check_fc2_temperature(anphonbin, project_root):
    # Reference route: bare FC2 + .scph_dfc2 via tools/dfc2.py.
    ret = subprocess.run(
        [
            sys.executable,
            os.path.join(project_root, "tools", "dfc2.py"),
            "-i", "cBTO222.h5",
            "-o", "cBTO222_300K.h5",
            "--dfc2", PREFIX + ".scph_dfc2",
            "--temp", "300",
        ],
        capture_output=True,
    )
    if ret.returncode != 0:
        print("dfc2.py failed:", ret.stderr.decode())
        return 1

    kpath = "&kpoint\n 1\n G 0.0 0.0 0.0 X 0.5 0.0 0.5 51\n/\n"
    with open("band_dfc2.in", "w") as f:
        f.write("&general\n PREFIX = bto_dfc2; MODE = phonons; FCSFILE = cBTO222_300K.h5\n/\n")
        f.write(kpath)
    with open("band_fc2t.in", "w") as f:
        f.write("&general\n PREFIX = bto_fc2t; MODE = phonons; FCSFILE = %s.scph.h5;" % PREFIX)
        f.write(" FC2_TEMPERATURE = 300\n/\n")
        f.write(kpath)

    if run_anphon(anphonbin, "band_dfc2.in", "band_dfc2.log") != 0:
        print("dfc2-route band run failed")
        return 1
    if run_anphon(anphonbin, "band_fc2t.in", "band_fc2t.log") != 0:
        print("FC2_TEMPERATURE band run failed")
        return 1

    a = np.loadtxt("bto_dfc2.bands")
    b = np.loadtxt("bto_fc2t.bands")
    # The two routes are exact-identical when KMESH_INTERPOLATE matches the
    # supercell dimensions (the case here); the tolerance covers the .bands
    # print precision.
    if a.shape != b.shape or not np.allclose(a[:, 1:], b[:, 1:], atol=2e-4):
        print("FC2_TEMPERATURE bands disagree with the dfc2.py route")
        diff = np.abs(a[:, 1:] - b[:, 1:]).max()
        print("max difference (cm^-1):", diff)
        return 1
    return 0


def check_kappa_on_scph(anphonbin):
    # Kappa on top of SCPH: one RTA run per basis temperature, all
    # accumulating into a single temperature-resolved kappa.h5 (v2 layout).
    def gen_rta(temp, fname):
        with open(fname, "w") as f:
            f.write("&general\n PREFIX = kbto; MODE = RTA; FCSFILE = cBTO222.h5;\n")
            f.write(" FC2FILE = %s.scph.h5; FC2_TEMPERATURE = %d; TMIN = %d; TMAX = %d\n/\n"
                    % (PREFIX, temp, temp, temp))
            f.write("&kpoint\n 2\n 2 2 2\n/\n")

    for temp in (280, 300):
        gen_rta(temp, "rta_%d.in" % temp)
        if run_anphon(anphonbin, "rta_%d.in" % temp, "rta_%d.log" % temp) != 0:
            print("kappa run at %d K failed" % temp)
            return 1

    with h5py.File("kbto.kappa.h5", "r") as f:
        if f.attrs["format_version"] != 2 or f.attrs["temperature_resolved"] != 1:
            print("temperature-resolved layout attributes are wrong")
            return 1
        if not np.allclose(f["metadata/temperatures"][...], [280.0, 300.0]):
            print("merged temperature grid is wrong")
            return 1
        if not np.allclose(f["metadata/fc2_temperatures"][...], [280.0, 300.0]):
            print("fc2_temperatures record is wrong")
            return 1
        g = f["scattering/3ph"]
        freq = g["frequencies"][...]
        if freq.ndim != 3 or freq.shape[0] != 2 or g["velocities"].shape[0] != 2:
            print("per-temperature basis datasets have wrong shapes")
            return 1
        if np.allclose(freq[0], freq[1]):
            print("SCPH bases at 280 K and 300 K should differ")
            return 1
        if not g["gamma_computed"][...].all():
            print("per-(mode, T) flags are not all set")
            return 1
        if not (f["kappa/valid"][...] == 1).all():
            print("per-temperature kappa validity flags are not set")
            return 1

    # Rerunning one temperature must be a per-column no-op restart.
    if run_anphon(anphonbin, "rta_280.in", "rta_280_restart.log") != 0:
        print("kappa restart run failed")
        return 1
    with open("rta_280_restart.log") as f:
        if "Total Number of phonon modes to be calculated : 0" not in f.read():
            print("per-column restart recomputed modes")
            return 1
    with h5py.File("kbto.kappa.h5", "r") as f:
        if not (f["kappa/valid"][...] == 1).all():
            print("restart clobbered another temperature's validity")
            return 1
    return 0


def runtest_scph_h5(anphonbin, project_root):
    scph_example_dir = os.path.join(project_root, "example", "BaTiO3", "scph_relax")
    reference_dir = os.path.join(scph_example_dir, "reference_for_test")
    fc_reference_dir = os.path.join(project_root, "example", "BaTiO3", "anharm_IFCs", "4_optimize",
                                    "reference")

    if copy_input_files(os.getcwd(), scph_example_dir, fc_reference_dir) != 0:
        return 1

    if check_fresh_run(anphonbin, reference_dir):
        return 1
    print("Fresh SCPH run + h5 schema --> pass")

    if check_h5_restart(anphonbin):
        return 1
    print("Pure h5 restart --> pass")

    if check_fc2_temperature(anphonbin, project_root):
        return 1
    print("FC2_TEMPERATURE vs dfc2.py --> pass")

    if check_kappa_on_scph(anphonbin):
        return 1
    print("Kappa on SCPH (temperature-resolved kappa.h5) --> pass")

    return 0


if __name__ == "__main__":
    build_dir = os.getcwd()
    project_root = os.path.dirname(build_dir)

    dirname = "%s/test/scph_h5" % project_root
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)
    sys.path.insert(0, "%s/test" % project_root)
    os.chdir(dirname)

    anphonbin = "%s/_build/anphon/anphon" % project_root

    info = runtest_scph_h5(anphonbin, project_root)

    sys.exit(0 if info == 0 else 1)
