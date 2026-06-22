#!/usr/bin/env python
"""Equivalence test for the L1 solvers (coordinate descent vs FISTA).

Both ``L1_SOLVER = cd`` and ``L1_SOLVER = fista`` minimize the *same* elastic-net
objective, so at a fixed ``L1_ALPHA`` they must converge to the same force
constants.  This test runs the Si cubic fit four times -- {cd, fista} x
{L1_RATIO = 1.0 (LASSO), L1_RATIO = 0.5 (elastic net)} -- and checks that:

  1. CD and FISTA agree for L1_RATIO = 1.0 (LASSO),
  2. CD and FISTA agree for L1_RATIO = 0.5 (exercises the corrected
     elastic-net coordinate update / the L2 denominator),
  3. the fits are non-trivial (some, but not all, force constants are nonzero).

Run directly with the alm binary path (defaults to ../build-eigenblas/alm or
../_build/alm/alm, overridable via the ALM_BIN env var or argv[1]):

    python test_fista_cd_equiv.py [/path/to/alm]
"""

import os
import shutil
import subprocess
import sys

import numpy as np

from test_si import gen_alminput_si


def parse_fcs(fname):
    """Return {(order, atom-descriptor): fc_value} for every irreducible FC.

    Keyed by the atom-index columns (invariant to any renumbering or to the
    solver dropping exactly-zero coefficients), so the two solvers' outputs can
    be matched even if their supports differ.
    """
    fcs = {}
    order = None
    with open(fname) as f:
        for line in f:
            s = line.strip()
            if s.startswith("*FC"):
                order = s.split()[0]  # e.g. "*FC2", "*FC3"
                continue
            if order is None:
                continue
            tok = s.split()
            # data row: <global> <local> <value> <mult> <atom1> <atom2> ... <distance>
            if len(tok) < 6:
                continue
            try:
                int(tok[0])
                int(tok[1])
                value = float(tok[2])
            except ValueError:
                continue
            descriptor = (order,) + tuple(tok[4:-1])  # atom columns, drop distance
            fcs[descriptor] = value
    return fcs


def aligned_vectors(fcs_a, fcs_b):
    """Two value arrays over the union of keys (missing coefficient -> 0)."""
    keys = sorted(set(fcs_a) | set(fcs_b))
    va = np.array([fcs_a.get(k, 0.0) for k in keys])
    vb = np.array([fcs_b.get(k, 0.0) for k in keys])
    return va, vb


def gen_enet_input(fname, prefix, dfset, l1_ratio, l1_solver):
    opt_extra = (
        "LMODEL = enet\n"
        "CV = 0\n"
        "L1_ALPHA = 1.0e-5\n"
        "L1_RATIO = %s\n"
        "L1_SOLVER = %s\n"
        "STANDARDIZE = 1\n"
        "CONV_TOL = 1.0e-8\n"
        "MAXITER = 500000\n"
    ) % (l1_ratio, l1_solver)
    gen_alminput_si(
        fname, norder=2, prefix=prefix, dfset=dfset, opt_extra=opt_extra
    )


def run_alm(almbin, infile, logfile):
    with open(logfile, "w") as f:
        ret = subprocess.run([almbin, infile], stdout=f, stderr=subprocess.STDOUT)
    return ret.returncode


def runtest_fista_cd(almbin, project_root, rtol=5.0e-3):
    shutil.copy(
        "%s/example/Si/reference/DFSET_harmonic" % project_root, "DFSET_harmonic"
    )
    shutil.copy("%s/example/Si/reference/DFSET_cubic" % project_root, "DFSET_cubic")
    with open("DFSET_merged", "w") as f:
        subprocess.run(["cat", "DFSET_harmonic", "DFSET_cubic"], stdout=f)

    overall_ok = True
    for ratio_tag, ratio in (("r10", "1.0"), ("r05", "0.5")):
        coefs = {}
        for solver in ("cd", "fista", "admm"):
            prefix = "si_enet_%s_%s" % (ratio_tag, solver)
            infile = prefix + ".in"
            gen_enet_input(infile, prefix, "DFSET_merged", ratio, solver)
            if run_alm(almbin, infile, prefix + ".log") != 0:
                print("ALM failed for %s" % prefix)
                return 1
            coefs[solver] = parse_fcs(prefix + ".fcs")

        # FISTA and ADMM solve the same objective as CD, so all three must reach the same minimizer.
        for other in ("fista", "admm"):
            va, vb = aligned_vectors(coefs["cd"], coefs[other])
            denom = np.linalg.norm(va)
            rel = np.linalg.norm(va - vb) / denom if denom > 0 else np.linalg.norm(va - vb)
            nnz_cd = int(np.count_nonzero(np.abs(va) > 1.0e-12))
            nnz_other = int(np.count_nonzero(np.abs(vb) > 1.0e-12))

            ok = rel < rtol and 0 < nnz_cd < len(va)
            overall_ok &= ok
            print(
                "L1_RATIO=%s  CD vs %-5s: rel.L2 diff=%.3e  nnz(cd)=%d nnz(%s)=%d / %d  --> %s"
                % (ratio, other.upper(), rel, nnz_cd, other, nnz_other, len(va), "pass" if ok else "FAILED")
            )
            if not ok:
                imax = int(np.argmax(np.abs(va - vb)))
                print(
                    "  Max abs diff = %.6e (cd=%.8e %s=%.8e)"
                    % (abs(va[imax] - vb[imax]), va[imax], other, vb[imax])
                )

    return 0 if overall_ok else 1


def resolve_almbin(project_root):
    if os.environ.get("ALM_BIN"):
        return os.environ["ALM_BIN"]
    if len(sys.argv) > 1:
        return sys.argv[1]
    for cand in ("build-eigenblas/alm", "_build/alm/alm"):
        path = os.path.join(project_root, cand)
        if os.path.exists(path):
            return path
    return os.path.join(project_root, "_build/alm/alm")


if __name__ == "__main__":
    test_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(test_dir)

    almbin = resolve_almbin(project_root)

    workdir = "%s/test/si_fista_cd" % project_root
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)

    info = runtest_fista_cd(almbin, project_root)
    if info == 0:
        print("Si FISTA/CD equivalence --> pass")
        sys.exit(0)
    else:
        print("Si FISTA/CD equivalence --> failed")
        sys.exit(1)
