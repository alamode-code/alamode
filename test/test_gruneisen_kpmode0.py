#!/usr/bin/env python
"""GRUNEISEN on a KPMODE = 0 list writes PREFIX.gru_kpoints, with values equal to
PREFIX.gru_all (KPMODE = 2) at the same k points, for GRUNEISEN = 1, 2 and 3.

Si harmonic + cubic IFCs (example/Si/reference/si222_cubic.xml.bz2).

Run from the build directory: python3 ../test/test_gruneisen_kpmode0.py
"""

import bz2
import os
import shutil
import subprocess
import sys

import numpy as np

HEAD = """&general
 PREFIX = %s; MODE = phonons; FCSFILE = si222_cubic.xml; KD = Si
/
&cell
 10.203
 0.0 0.5 0.5
 0.5 0.0 0.5
 0.5 0.5 0.0
/
"""
KLIST = [[0.25, 0.0, 0.0], [0.25, 0.5, -0.25], [0.5, 0.5, 0.5]]


def read_gru(fname):
    """{tuple(xk): array(ns, ncol)} from a gru_all / gru_kpoints file."""
    out, xk, rows = {}, None, []
    for line in open(fname):
        if line.startswith("# knum ="):
            if xk is not None:
                out[xk] = np.array(rows)
            xk = tuple(np.round(np.array(line.split()[4:7], dtype=float) % 1.0, 6))
            rows = []
        elif not line.startswith("#") and line.strip():
            rows.append([float(x) for x in line.split()[2:]])
    out[xk] = np.array(rows)
    return out


def block_sums(rows, tol=1e-3):
    """Sum the rows (omega, gammas) over degenerate modes. Off-diagonal generalized
    parameters of degenerate modes depend on the eigenvector gauge, which differs
    between k and k + G (e.g. -0.25 in the list vs 0.75 on the mesh); the block sum
    (a trace over the degenerate subspace) does not."""
    out, start = [], 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or abs(rows[i, 0] - rows[start, 0]) > tol:
            out.append(rows[start:i].sum(axis=0))
            start = i
    return np.array(out)


if __name__ == "__main__":
    project_root = os.path.dirname(os.getcwd())
    anphon = os.path.join(project_root, "_build/anphon/anphon")
    workdir = os.path.join(project_root, "test/si/gruneisen_kpmode0")
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir)
    os.chdir(workdir)
    src = os.path.join(project_root, "example/Si/reference/si222_cubic.xml.bz2")
    with bz2.open(src, "rb") as fi, open("si222_cubic.xml", "wb") as fo:
        shutil.copyfileobj(fi, fo)

    klist = "&kpoint\n 0\n" + "".join(" %g %g %g\n" % tuple(k) for k in KLIST) + "/\n"
    nfail = 0
    for mode in (1, 2, 3):
        ana = "&analysis\n GRUNEISEN = %d\n/\n" % mode
        for tag, kp in (("mesh", "&kpoint\n 2\n 4 4 4\n/\n"), ("list", klist)):
            prefix = "si_%s_%d" % (tag, mode)
            with open(prefix + ".in", "w") as f:
                f.write(HEAD % prefix + kp + ana)
            with open(prefix + ".log", "w") as f:
                subprocess.run([anphon, prefix + ".in"], stdout=f, check=True)
        mesh = read_gru("si_mesh_%d.gru_all" % mode)
        lst = read_gru("si_list_%d.gru_kpoints" % mode)
        if "si_list_%d.gru_kpoints" % mode not in open("si_list_%d.log" % mode).read():
            print(
                "GRUNEISEN = %d: gru_kpoints missing from the files-created list" % mode
            )
            nfail += 1
        for k in KLIST:
            key = tuple(np.round(np.array(k) % 1.0, 6))
            a, b = block_sums(lst[key]), block_sums(mesh[key])
            if not np.allclose(a, b, rtol=1e-5, atol=1e-5):
                print("GRUNEISEN = %d, k = %s: gru_kpoints != gru_all" % (mode, k))
                print(a - b)
                nfail += 1
    print("Gruneisen KPMODE = 0 --> " + ("pass" if nfail == 0 else "failed"))
    sys.exit(1 if nfail else 0)
