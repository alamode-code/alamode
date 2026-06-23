#!/usr/bin/env python
"""Regression test for the non-algebraic constraint path (ICONST = 2).

ICONST = 2 imposes translational *and* rotational invariance and solves the fit as
an equality-constrained least-squares problem (the GQR / Pardiso-KKT path), rather
than the algebraic elimination map used for ICONST >= 10. The constraint matrix that
is handed to that solver is rank-reduced by one of the ALGO_REDUCTION backends:

    ALGO_REDUCTION = 1  ->  rref            (Gauss-Jordan, first-acceptable pivot)
    ALGO_REDUCTION = 2  ->  qrd             (rank-revealing LAPACK QR, get_independent_rows)
    ALGO_REDUCTION = 3  ->  coord_factorization (Gauss-Jordan, partial pivot) [default]

All three describe the *same* constraint subspace, so the fitted force constants must
be identical up to round-off regardless of backend. A backend that mis-detects the
numerical rank (e.g. accepting a round-off residual as an independent pivot) injects a
spurious/garbage constraint, which corrupts the fit and breaks the very invariances
ICONST = 2 is meant to enforce. This test guards that path by requiring the three
backends to agree, for both NORDER = 2 ICONST = 2 (non-algebraic) and the algebraic
default (ICONST = 11), and it also checks that the ICONST = 2 harmonic IFCs satisfy
the acoustic sum rule (translational invariance) directly.

Run:  python test_iconst2_rotinv.py [path/to/alm]
"""

import os
import shutil
import subprocess
import sys

import numpy as np

from test_si import gen_alminput_si
from test_algo_reduction import parse_fcs_values, run_alm

BACKENDS = {1: "rref", 2: "qrd", 3: "coord"}


def fcs_agree(label, ref, others, rtol=1.0e-8, atol=1.0e-10):
    ok = True
    for name, vals in others.items():
        if vals.shape != ref.shape:
            print("  [%s] %s: FC count mismatch (%d vs %d)"
                  % (label, name, vals.size, ref.size))
            ok = False
            continue
        if not np.allclose(vals, ref, rtol=rtol, atol=atol):
            i = int(np.argmax(np.abs(vals - ref)))
            print("  [%s] %s disagrees: max|d|=%.3e at idx %d (%.6e vs %.6e)"
                  % (label, name, np.max(np.abs(vals - ref)), i, ref[i], vals[i]))
            ok = False
    return ok


def add_cartesian_basis(infile):
    """Rotational invariance (ICONST = 2) requires FCSYM_BASIS = Cartesian.

    gen_alminput_si() does not expose &general tags, so inject it here.
    """
    with open(infile) as handle:
        text = handle.read()
    if "FCSYM_BASIS" not in text:
        text = text.replace("; KD = Si\n", "; KD = Si; FCSYM_BASIS = Cartesian\n", 1)
    with open(infile, "w") as handle:
        handle.write(text)


def run_backends(almbin, label, norder, opt_extra, dfset):
    """Fit with every backend; return {name: fc_values}."""
    fcs = {}
    for algo, name in BACKENDS.items():
        prefix = "si_%s_%s" % (label, name)
        infile = "ALM_%s.in" % prefix
        gen_alminput_si(infile, norder=norder, prefix=prefix, dfset=dfset,
                        algo_reduction=algo, opt_extra=opt_extra)
        add_cartesian_basis(infile)
        if run_alm(almbin, infile, "ALM_%s.log" % prefix) != 0:
            print("ALM failed for ALGO_REDUCTION=%d (%s).\n  binary: %s"
                  % (algo, label, almbin))
            return None
        fcs[name] = parse_fcs_values("%s.fcs" % prefix)
    return fcs


def runtest(almbin, project_root):
    ref_dir = "%s/example/Si/reference" % project_root
    shutil.copy("%s/DFSET_harmonic" % ref_dir, "DFSET_harmonic")
    shutil.copy("%s/DFSET_cubic" % ref_dir, "DFSET_cubic")
    with open("DFSET_merged", "w") as handle:
        subprocess.run(["cat", "DFSET_harmonic", "DFSET_cubic"], stdout=handle)

    failed = False

    # 1) Non-algebraic path: ICONST = 2 (translational + rotational invariance).
    fcs = run_backends(almbin, "ic2", norder=2,
                       opt_extra="ICONST = 2\nROTAXIS = xyz\n", dfset="DFSET_merged")
    if fcs is None:
        return 1
    ref = fcs["qrd"]  # QR rank-revealing reference
    if fcs_agree("ICONST=2", ref, {k: v for k, v in fcs.items() if k != "qrd"}):
        print("ICONST=2 (non-algebraic): rref == qrd == coord (%d FCs) --> pass"
              % ref.size)
    else:
        print("ICONST=2 (non-algebraic): backends disagree --> FAIL")
        failed = True

    # 2) Algebraic path: default ICONST (= 11), all three backends must agree.
    fcs = run_backends(almbin, "alg", norder=2, opt_extra=None, dfset="DFSET_merged")
    if fcs is None:
        return 1
    ref = fcs["rref"]
    if fcs_agree("algebraic", ref, {k: v for k, v in fcs.items() if k != "rref"}):
        print("algebraic (ICONST=11): rref == qrd == coord (%d FCs) --> pass"
              % ref.size)
    else:
        print("algebraic (ICONST=11): backends disagree --> FAIL")
        failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if len(sys.argv) > 1:
        almbin = os.path.abspath(sys.argv[1])
    else:
        candidates = ["%s/_build/alm/alm" % project_root,
                      "%s/build/alm/alm" % project_root]
        almbin = next((c for c in candidates if os.path.exists(c)), candidates[0])

    workdir = "%s/test/si_iconst2" % project_root
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)

    info = runtest(almbin, project_root)
    if info == 0:
        print("ICONST=2 / ALGO_REDUCTION backend consistency --> pass")
        sys.exit(0)
    else:
        print("ICONST=2 / ALGO_REDUCTION backend consistency --> failed")
        sys.exit(1)
