#!/usr/bin/env python
"""Regression test for the constraint-reduction backends (ALGO_REDUCTION).

ALAMODE eliminates the redundant degrees of freedom of the force constants with
a constraint map (const_fix / const_relate / index_bimap) that is derived from a
reduced row echelon form of the per-order constraint matrix. Two backends produce
that echelon form:

    ALGO_REDUCTION = 1  ->  rref (Gauss-Jordan, first-acceptable pivot)        [legacy]
    ALGO_REDUCTION = 3  ->  coord_factorization (Gauss-Jordan, partial pivot)  [default]

coord_factorization selects the same pivot columns as rref (the leftmost linearly
independent columns) but with maximum-magnitude row pivoting, so the resulting map
-- and therefore the fitted force constants -- must be numerically identical to
rref up to round-off. This test fits the Si example (NORDER = 2, default ICONST = 11
so the algebraic mapping is exercised) with both backends and checks that:

  1. the two .fcs files agree to tight tolerance (rref == coord_factorization), and
  2. coord_factorization reproduces the committed reference si222_cubic.fcs.

Run:  python test_algo_reduction.py [path/to/alm]
"""

import os
import shutil
import subprocess
import sys

import numpy as np

from test_si import gen_alminput_si, print_max_errors


def parse_fcs_values(fname):
    """Return the force-constant value column of an ALAMODE .fcs file.

    Data lines have the form '<global> <local> <value> <mult> <pairs...> <dist>';
    header / comment / '*FCn' lines are skipped.
    """
    values = []
    with open(fname) as handle:
        for line in handle:
            tokens = line.split()
            if len(tokens) < 3:
                continue
            try:
                int(tokens[0])
                int(tokens[1])
                value = float(tokens[2])
            except ValueError:
                continue
            values.append(value)
    return np.asarray(values)


def run_alm(almbin, infile, logfile):
    with open(logfile, "w") as handle:
        ret = subprocess.run([almbin, infile], stdout=handle)
    return ret.returncode


def runtest_algo_reduction(almbin, project_root, rtol=1.0e-8, atol=1.0e-10):
    ref_dir = "%s/example/Si/reference" % project_root
    shutil.copy("%s/DFSET_harmonic" % ref_dir, "DFSET_harmonic")
    shutil.copy("%s/DFSET_cubic" % ref_dir, "DFSET_cubic")
    with open("DFSET_merged", "w") as handle:
        subprocess.run(["cat", "DFSET_harmonic", "DFSET_cubic"], stdout=handle)

    # Fit the same cubic problem with each reduction backend.
    backends = {1: "si_rref", 3: "si_coord"}
    for algo, prefix in backends.items():
        gen_alminput_si(
            "ALM_%s.in" % prefix,
            norder=2,
            prefix=prefix,
            dfset="DFSET_merged",
            algo_reduction=algo,
        )
        if run_alm(almbin, "ALM_%s.in" % prefix, "ALM_%s.log" % prefix) != 0:
            print(
                "ALM failed to execute for ALGO_REDUCTION=%d.\n"
                "Please check the alm binary at %s" % (algo, almbin)
            )
            return 1

    fc_rref = parse_fcs_values("si_rref.fcs")
    fc_coord = parse_fcs_values("si_coord.fcs")

    if fc_rref.shape != fc_coord.shape or fc_rref.size == 0:
        print(
            "Force-constant count mismatch: rref=%d, coord_factorization=%d"
            % (fc_rref.size, fc_coord.size)
        )
        return 1

    # 1) coord_factorization must match rref (legacy compatibility / Policy A).
    if not np.allclose(fc_coord, fc_rref, rtol=rtol, atol=atol):
        print(
            "ALGO_REDUCTION=3 (coord_factorization) disagrees with ALGO_REDUCTION=1 (rref)"
        )
        print_max_errors("si_coord.fcs", fc_rref, fc_coord)
        return 1
    print("coord_factorization == rref (%d force constants) --> pass" % fc_coord.size)

    # 2) coord_factorization must reproduce the committed reference.
    fc_ref = parse_fcs_values("%s/si222_cubic.fcs" % ref_dir)
    if fc_coord.shape == fc_ref.shape and np.allclose(
        fc_coord, fc_ref, rtol=1.0e-6, atol=1.0e-2
    ):
        print("coord_factorization == reference si222_cubic.fcs --> pass")
    else:
        print("coord_factorization disagrees with reference si222_cubic.fcs")
        if fc_coord.shape == fc_ref.shape:
            print_max_errors("si222_cubic.fcs", fc_ref, fc_coord)
        else:
            print(
                "  count mismatch: coord=%d, reference=%d"
                % (fc_coord.size, fc_ref.size)
            )
        return 1

    return 0


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if len(sys.argv) > 1:
        # Resolve to an absolute path: the test chdir's into its work directory below,
        # which would otherwise break a relative binary path.
        almbin = os.path.abspath(sys.argv[1])
    else:
        candidates = [
            "%s/_build/alm/alm" % project_root,
            "%s/build/alm/alm" % project_root,
        ]
        almbin = next((c for c in candidates if os.path.exists(c)), candidates[0])

    workdir = "%s/test/si_algo_reduction" % project_root
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    os.chdir(workdir)

    info = runtest_algo_reduction(almbin, project_root)
    if info == 0:
        print("ALGO_REDUCTION consistency --> pass")
        sys.exit(0)
    else:
        print("ALGO_REDUCTION consistency --> failed")
        sys.exit(1)
