#!/usr/bin/env python
"""Regression test: V4_REAL_SPACE, the SCP loop without the V4 tensor.

On the 5-atom BaTiO3 cell:

- V4_REAL_SPACE = 2 (both, compared): the on-site V4, the first SCP-matrix
  contraction and the q0 sweep from the real-space FC4 must equal the V4
  tensor, with unequal meshes (KMESH_INTERPOLATE = 1 1 2, KMESH_SCPH = 1 1 4,
  RELAX_STR = 2 from a P1 start) and in the diagonal-only mode
  (SELF_OFFDIAG = 0, no relaxation, 2 2 2 / 4 4 4).
- V4_REAL_SPACE = 1 (no tensor) must reproduce the outputs of V4_REAL_SPACE = 0,
  also on 2 MPI ranks, and the BUBBLE = 4 curvature (Gamma and finite Q) on a
  1 1 3 mesh.
"""

import os
import re
import shutil
import subprocess
import sys

import numpy as np

WORKDIR = "v4_real_space"


def run_anphon(cmd, logfile):
    with open(logfile, "w") as f:
        return subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT, timeout=1800
        ).returncode


def variant(src, prefix, v4rs, **tags):
    src = re.sub(r"PREFIX\s*=\s*\S+", "PREFIX = %s" % prefix, src, count=1)
    src = src.replace(
        "  BUBBLE = 4\n", "  BUBBLE = 4\n  V4_REAL_SPACE = %d\n" % v4rs, 1
    )
    for tag, value in tags.items():
        src = re.sub(r"%s\s*=.*" % tag, "%s = %s" % (tag, value), src, count=1)
    with open(prefix + ".in", "w") as f:
        f.write(src)


def checks(logfile):
    with open(logfile) as f:
        text = f.read()
    return [
        float(x)
        for line in text.splitlines()
        if "real space vs tensor" in line
        for x in re.findall(r"\d\.\d+e[+-]\d+", line)
    ]


def same_outputs(a, b, exts, atol):
    for ext in exts:
        x, y = np.loadtxt(a + ext), np.loadtxt(b + ext)
        if x.shape != y.shape or not np.allclose(x, y, rtol=0.0, atol=atol):
            print("%s%s differs from %s%s" % (b, ext, a, ext))
            return False
    return True


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
    from test_scph_hessian import curvature, curvature_q, write_input

    scph_example_dir = os.path.join(project_root, "example", "BaTiO3", "scph_relax")
    fc_reference_dir = os.path.join(
        project_root, "example", "BaTiO3", "anharm_IFCs", "4_optimize", "reference"
    )
    if copy_input_files(workdir, scph_example_dir, fc_reference_dir) != 0:
        return 1
    os.chdir(workdir)

    ok = True
    write_input("base", "", mesh="1 1 2", ncell=1)
    with open("base.in") as f:
        base = f.read()

    runs = []
    # unequal meshes with a P1 relaxation (no curvature)
    for m in (2, 0, 1):
        variant(base, "u%d" % m, m, KMESH_SCPH="1 1 4", BUBBLE="0")
        runs.append("u%d" % m)
    # diagonal-only SCPH at the cubic structure
    for m in (2, 0, 1):
        variant(
            base,
            "d%d" % m,
            m,
            KMESH_INTERPOLATE="2 2 2",
            KMESH_SCPH="4 4 4",
            BUBBLE="0",
            RELAX_STR="0",
            SELF_OFFDIAG="0",
        )
        runs.append("d%d" % m)
    # the curvature on a mesh
    for m in (0, 1):
        variant(base, "h%d" % m, m, KMESH_INTERPOLATE="1 1 3", KMESH_SCPH="1 1 3")
        runs.append("h%d" % m)
    for prefix in runs:
        if run_anphon([anphonbin, prefix + ".in"], prefix + ".log"):
            print("%s run failed, see %s/%s.log" % (prefix, WORKDIR, prefix))
            return 1

    for prefix, n in (("u2", 4), ("d2", 2)):
        c = checks(prefix + ".log")
        if len(c) != n or not max(c) < 1.0e-12:
            print("%s: real space vs V4 tensor %s" % (prefix, c))
            ok = False
    exts = (".scph_thermo", ".atom_disp", ".umn_tensor")
    ok = same_outputs("u0", "u1", exts, 1e-10) and ok
    ok = same_outputs("d0", "d1", (".scph_thermo",), 1e-10) and ok
    ok = same_outputs("h0", "h1", exts, 1e-10) and ok
    # Gamma (12 optical modes) and the one other irreducible Q of 1 1 3 (15 modes)
    g0, g1 = curvature("h0")[1], curvature("h1")[1]
    q0, q1 = curvature_q("h0"), curvature_q("h1")
    if (
        len(g0) != 12
        or len(g1) != 12
        or not np.allclose(g0, g1, rtol=0.0, atol=1e-4)
        or len(q0) != 1
        or sorted(q0) != sorted(q1)
        or any(len(q0[q]) != 15 or len(q1[q]) != 15 for q in q0)
        or any(not np.allclose(q0[q], q1[q], rtol=0.0, atol=1e-4) for q in q0)
    ):
        print("the curvature differs (or is missing) without the V4 tensor")
        ok = False
    print("V4_REAL_SPACE vs the V4 tensor --> %s" % ("pass" if ok else "fail"))

    if shutil.which("mpirun") is not None:
        variant(base, "u1np2", 1, KMESH_SCPH="1 1 4", BUBBLE="0")
        if run_anphon(["mpirun", "-np", "2", anphonbin, "u1np2.in"], "u1np2.log"):
            print("2-rank run failed, see %s/u1np2.log" % WORKDIR)
            return 1
        mpi_ok = same_outputs("u0", "u1np2", exts, 1e-10)
        print(
            "V4_REAL_SPACE = 1 on 2 MPI ranks --> %s" % ("pass" if mpi_ok else "fail")
        )
        ok = ok and mpi_ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
