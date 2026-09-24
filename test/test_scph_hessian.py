#!/usr/bin/env python
"""Regression test: BUBBLE = 4, the curvature of the SCP free energy.

A 1x1x2 cell of cubic BaTiO3 (KMESH_SCPH = KMESH_INTERPOLATE = 1 1 1, so the
Z point of the primitive cell folds onto Gamma and the cubic couplings
between Gamma and Z do not vanish) is relaxed with RELAX_STR = 2 from a small
random P1 displacement, so that the SCP loop applies no symmetry projection.
At the converged structure:

- the analytic force Jacobian (static bubble + quartic ladder, solved by
  GMRES) must match central finite differences of the SCP force
  (BUBBLE_FD_CHECK = 1) to the accuracy of the SCP solves;
- it must be symmetric;
- without the ladder (BUBBLE_LADDER = 0) the finite differences must NOT be
  matched, or the test could not see the ladder;
- a 2-rank run (batched V4 contraction over distributed rows) must reproduce
  the serial one, and its ordinary SCPH outputs (no FD check) must equal those
  of the FD-checked run, whose SCP state is restored after the displacements.
"""

import os
import random
import re
import shutil
import subprocess
import sys

import numpy as np

WORKDIR = "scph_hessian"
A = 7.53159676409  # Bohr, cubic BaTiO3 fixture


def run_anphon(cmd, logfile):
    with open(logfile, "w") as f:
        return subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT, timeout=1800
        ).returncode


def write_input(prefix, extra):
    with open("BTO_scph_thermo.in") as f:
        src = f.read()
    src = re.sub(r"PREFIX\s*=\s*\S+", "PREFIX = %s" % prefix, src, count=1)
    src = re.sub(r"TMIN\s*=\s*\S+", "TMIN = 300", src, count=1)
    src = re.sub(r"TMAX\s*=\s*\S+", "TMAX = 300", src, count=1)
    src = re.sub(r"KMESH_INTERPOLATE\s*=.*", "KMESH_INTERPOLATE = 1 1 1", src, count=1)
    src = re.sub(r"KMESH_SCPH\s*=.*", "KMESH_SCPH = 1 1 1", src, count=1)
    src = re.sub(
        r"MAXITER\s*=\s*\S+", "MAXITER = 5000\n  TOL_SCPH = 1.0e-12", src, count=1
    )
    src = re.sub(
        r"RELAX_STR\s*=\s*\S+", "RELAX_STR = 2\n  BUBBLE = 4\n" + extra, src, count=1
    )
    head = src.split("&displace")[0]
    head = head.replace(
        "&scph",
        "&cell\n 1.0\n %.10f 0.0 0.0\n 0.0 %.10f 0.0\n 0.0 0.0 %.10f\n/\n&scph"
        % (A, A, 2 * A),
        1,
    )
    rng = random.Random(11)
    lines = [
        "&displace",
        "0",
        "  %.11f" % A,
        "  1.0 0.0 0.0",
        "  0.0 1.0 0.0",
        "  0.0 0.0 2.0",
    ]
    for _ in range(10):
        lines.append(" ".join("%.6e" % (rng.uniform(-1, 1) * 2e-3) for _ in range(3)))
    lines.append("/\n\n&kpoint\n  2\n  8 8 4\n/\n")
    with open(prefix + ".in", "w") as f:
        f.write(head + "\n".join(lines))


def fd_mismatch(logfile):
    with open(logfile) as f:
        m = re.search(
            r"BUBBLE_FD_CHECK: max \|J - J_fd\| / max \|J\| = (\S+)", f.read()
        )
    return float(m.group(1)) if m else None


def curvature(prefix):
    """(asymmetry, free-energy curvature frequencies) of the last temperature block."""
    with open(prefix + ".scph_hessian") as f:
        text = f.read()
    block = text.split("# T = ")[-1]
    asym = float(re.search(r"asymmetry = (\S+)", block).group(1))
    rows = [line.split() for line in block.splitlines()[1:] if line.strip()]
    return asym, np.array([float(r[2]) for r in rows])


def main():
    test_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(test_root)
    anphonbin = os.path.join(project_root, "_build/anphon/anphon")
    workdir = os.path.join(test_root, WORKDIR)
    if os.path.exists(workdir):
        shutil.rmtree(workdir)
    os.makedirs(workdir)
    sys.path.insert(0, test_root)
    from test_batio3 import copy_input_files

    scph_example_dir = os.path.join(project_root, "example", "BaTiO3", "scph_relax")
    fc_reference_dir = os.path.join(
        project_root, "example", "BaTiO3", "anharm_IFCs", "4_optimize", "reference"
    )
    if copy_input_files(workdir, scph_example_dir, fc_reference_dir) != 0:
        return 1
    os.chdir(workdir)

    ok = True
    write_input("full", "  BUBBLE_FD_CHECK = 1\n")
    write_input("noladder", "  BUBBLE_FD_CHECK = 1\n  BUBBLE_LADDER = 0\n")
    for prefix in ("full", "noladder"):
        if run_anphon([anphonbin, prefix + ".in"], prefix + ".log"):
            print("%s run failed, see %s/%s.log" % (prefix, WORKDIR, prefix))
            return 1

    full = fd_mismatch("full.log")
    if full is None or not full < 1.0e-7:
        print("analytic Jacobian vs finite differences: %s" % full)
        ok = False
    asym, w_full = curvature("full")
    if not asym < 1.0e-8:
        print("the Jacobian is not symmetric: %g" % asym)
        ok = False
    noladder = fd_mismatch("noladder.log")
    if noladder is None or not noladder > 1.0e-3:
        print(
            "without the ladder the finite differences are still matched (%s)"
            % noladder
        )
        ok = False
    print(
        "BUBBLE = 4 vs finite differences, ladder on/off --> %s"
        % ("pass" if ok else "fail")
    )
    if not ok:
        return 1

    if shutil.which("mpirun") is not None:
        write_input("np2", "")
        if run_anphon(["mpirun", "-np", "2", anphonbin, "np2.in"], "np2.log"):
            print("2-rank run failed, see %s/np2.log" % WORKDIR)
            return 1
        _, w_np2 = curvature("np2")
        if w_np2.shape != w_full.shape or not np.allclose(w_np2, w_full, rtol=1e-6):
            print("2-rank curvature differs from the serial one")
            ok = False
        # The finite-difference check restores the SCP state it displaces: the
        # ordinary outputs of the FD run must equal those of this run (FD off).
        for ext in (".scph_thermo", ".atom_disp"):
            a = np.loadtxt("full" + ext)
            b = np.loadtxt("np2" + ext)
            if a.shape != b.shape or not np.allclose(a, b, rtol=1e-6, atol=1e-10):
                print("%s differs between the FD-checked and the plain run" % ext)
                ok = False
        print("BUBBLE = 4 on 2 MPI ranks --> %s" % ("pass" if ok else "fail"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
