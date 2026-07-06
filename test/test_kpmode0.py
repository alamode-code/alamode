#!/usr/bin/env python
"""Regression test for SCPH postprocess on a general k-point list (KPMODE = 0)
with the non-analytic correction enabled.

This is the only coverage of the kpoint_general branch of
ScphQhaCommon::postprocess (the mesh and band-path branches are exercised by
the other SCPH fixtures): SrTiO3, NONANALYTIC = 3, a 2x2x2 SCPH mesh, and a
small list of arbitrary k points including Gamma and low-symmetry points.
The SCPH eigenvalues at those k points are compared against a committed
reference (example/SrTiO3/reference_for_test).

Run from the build directory: python3 ../test/test_kpmode0.py
"""

import bz2
import os
import shutil
import subprocess
import sys

import numpy as np

INPUT_TEXT = """&general
 PREFIX = STO_kp0
 MODE = SCPH
 KD = Sr Ti O
 FCSFILE = STO_anharm.xml
 NONANALYTIC = 3; BORNINFO = BORN
 TMIN = 0; TMAX = 200; DT = 100
/

&cell
7.363
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
/

&kpoint
0
0.00 0.00 0.00
0.10 0.00 0.00
0.25 0.15 0.05
0.50 0.50 0.50
0.33 0.21 0.11
/

&scph
 SELF_OFFDIAG = 0
 MAXITER = 500
 MIXALPHA = 0.2
 KMESH_INTERPOLATE = 2 2 2
 KMESH_SCPH = 2 2 2
/
"""


def isclose(a, b, rel_tol, abs_tol):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def compare_eval(file_now, file_ref, rel_tol=1.0e-5, abs_tol=1.0e-3):
    data_ref = np.loadtxt(file_ref, ndmin=2)
    data_now = np.loadtxt(file_now, ndmin=2)

    if data_ref.shape != data_now.shape:
        print(
            "shape mismatch in %s: ref %s vs now %s"
            % (file_now, data_ref.shape, data_now.shape)
        )
        return 1

    ok = True
    for i in range(data_ref.shape[0]):
        for j in range(data_ref.shape[1]):
            ok = ok & isclose(data_ref[i, j], data_now[i, j], rel_tol, abs_tol)
    if not ok:
        diff = np.abs(data_now - data_ref)
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(
            "mismatch in %s at %s: ref %.16e now %.16e"
            % (file_now, str(idx), data_ref[idx], data_now[idx])
        )
        return 1
    return 0


def setup_workdir(workdir, sto_dir):
    for file in ["STO_anharm.xml.bz2", "BORN"]:
        src = os.path.join(sto_dir, file)
        if not os.path.exists(src):
            print("File %s not found in %s" % (file, sto_dir))
            return 1
        shutil.copy(src, workdir)

    with bz2.open(os.path.join(workdir, "STO_anharm.xml.bz2"), "rb") as f_in:
        with open(os.path.join(workdir, "STO_anharm.xml"), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    with open(os.path.join(workdir, "sto_kpmode0.in"), "w") as f:
        f.write(INPUT_TEXT)
    return 0


if __name__ == "__main__":
    build_dir = os.getcwd()
    project_root = os.path.dirname(build_dir)
    sto_dir = os.path.join(project_root, "example/SrTiO3/reference")
    refdir = os.path.join(project_root, "example/SrTiO3/reference_for_test")

    workdir = os.path.join(project_root, "test/sto")
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)

    anphonbin = os.path.join(project_root, "_build/anphon/anphon")

    if setup_workdir(workdir, sto_dir):
        sys.exit(1)

    with open("sto_kpmode0.log", "w") as f:
        proc = subprocess.run([anphonbin, "sto_kpmode0.in"], stdout=f)
    if proc.returncode != 0:
        print("anphon exited with code %d" % proc.returncode)
        print("STO KPMODE=0 --> failed")
        sys.exit(1)

    info = compare_eval("STO_kp0.scph_eval", os.path.join(refdir, "STO_kp0.scph_eval"))

    if info == 0:
        print("STO KPMODE=0 --> pass")
        sys.exit(0)
    print("STO KPMODE=0 --> failed")
    sys.exit(1)
