#!/usr/bin/env python
"""BaTiO3 SCPH regression with IALGO = 1 (band-parallel V4 kernel).

Runs the same input as test_batio3.py with IALGO = 1 added, and compares
against the same reference outputs: compute_V4_elements_mpi_over_band must
reproduce compute_V4_elements_mpi_over_kpoint exactly, so no separate
reference data is needed.
"""

import argparse
import os
import sys

from test_batio3 import (
    check_consistency_anphon,
    copy_input_files,
    run_anphon_batio3,
)


def add_ialgo_tag(input_file, ialgo=1):
    with open(input_file) as f:
        lines = f.readlines()
    with open(input_file, "w") as f:
        for line in lines:
            f.write(line)
            if line.strip().startswith("SELF_OFFDIAG"):
                f.write("  IALGO = %d\n" % ialgo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs", type=str, default="all", help="Job types (all, copy, run, compare)"
    )
    args = parser.parse_args()

    build_dir = os.getcwd()
    project_root = os.path.dirname(build_dir)

    scph_example_dir = os.path.join(project_root, "example/BaTiO3/scph_relax")
    reference_dir = os.path.join(scph_example_dir, "reference_for_test")
    fc_reference_dir = os.path.join(
        project_root, "example/BaTiO3/anharm_IFCs/4_optimize/reference"
    )

    workdir = f"{project_root}/test/batio3_ialgo"
    if not os.path.exists(workdir):
        os.mkdir(workdir)
    os.chdir(workdir)

    anphonbin = "%s/_build/anphon/anphon" % project_root
    info = 0

    if args.jobs in ["all", "copy"]:
        info = copy_input_files(workdir, scph_example_dir, fc_reference_dir)
        if info > 0:
            sys.exit(1)
        add_ialgo_tag("BTO_scph_thermo.in")

    if args.jobs in ["all", "run"]:
        info = run_anphon_batio3(anphonbin)
        if info > 0:
            print(
                "ANPHON code failed to execute.\nPlease check if the anphon binary exists at %s"
                % anphonbin
            )
            sys.exit(1)

    if args.jobs in ["all", "compare"]:
        info = check_consistency_anphon(reference_dir, abs_tol=1.0e-8, rel_tol=1.0e-9)

    if info == 0:
        print("BaTiO3 ANPHON (IALGO=1) --> pass")
    else:
        print("BaTiO3 ANPHON (IALGO=1) --> failed")

    if info == 0:
        sys.exit(0)
    else:
        sys.exit(1)
