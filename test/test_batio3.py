#!/usr/bin/env python

import argparse
import os
import shutil
import subprocess
import sys

import numpy as np


def isclose(a, b, rel_tol=1e-5, abs_tol=1.0e-12):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def format_index(index):
    if index == ():
        return "scalar"
    return str(tuple(int(i) for i in index))


def print_max_errors(file, data_ref, data_now, rel_min_scale=1.0e-15):
    data_ref = np.asarray(data_ref)
    data_now = np.asarray(data_now)
    abs_errors = np.abs(data_now - data_ref)
    scale = np.maximum(np.abs(data_ref), np.abs(data_now))
    rel_mask = scale >= rel_min_scale
    rel_errors = np.divide(
        abs_errors,
        scale,
        out=np.zeros_like(abs_errors, dtype=float),
        where=rel_mask,
    )

    abs_index = np.unravel_index(np.argmax(abs_errors), abs_errors.shape)
    print(
        "Max absolute error in %s at index %s: %.6e"
        % (file, format_index(abs_index), abs_errors[abs_index])
    )
    print(
        "  reference = %.16e, current = %.16e"
        % (data_ref[abs_index], data_now[abs_index])
    )
    if np.any(rel_mask):
        rel_index = np.unravel_index(
            np.argmax(np.where(rel_mask, rel_errors, -np.inf)), rel_errors.shape
        )
        print(
            "Max relative error in %s at index %s (scale >= %.1e): %.6e"
            % (file, format_index(rel_index), rel_min_scale, rel_errors[rel_index])
        )
        print(
            "  reference = %.16e, current = %.16e"
            % (data_ref[rel_index], data_now[rel_index])
        )
    else:
        print(
            "Max relative error in %s: skipped (all scales < %.1e)"
            % (file, rel_min_scale)
        )


def run_anphon_batio3(anphonbin):
    try:
        with open("BTO_scph_thermo.log", "w") as f:
            proc = subprocess.run([anphonbin, "BTO_scph_thermo.in"], stdout=f)
        if proc.returncode != 0:
            return 1
    except Exception:
        return 1

    return 0


def check_consistency_anphon(reference_dir, abs_tol=0.01, rel_tol=1.0e-9):
    files_to_compare = [
        ("cBTO222_scph.atom_disp", 0),
        ("cBTO222_scph.V0", 0),
        ("cBTO222_scph.scph_thermo", 0),
        ("cBTO222_scph.umn_tensor", 0),
    ]

    for file, skiprows in files_to_compare:
        path_ref = os.path.join(reference_dir, file)
        path_now = file

        data_ref = np.loadtxt(path_ref, skiprows=skiprows)
        data_now = np.loadtxt(path_now, skiprows=skiprows)

        if np.shape(data_ref) != np.shape(data_now):
            print("Failed to match shape of %s" % file)
            return 1

        isclose_all = True
        data_ref_flat = np.ravel(data_ref)
        data_now_flat = np.ravel(data_now)
        for val_ref, val_now in zip(data_ref_flat, data_now_flat):
            isclose_all = isclose_all & isclose(
                val_ref, val_now, rel_tol=rel_tol, abs_tol=abs_tol
            )

        if not isclose_all:
            print("Failed to match %s" % file)
            print_max_errors(file, data_ref, data_now)
            return 1

    return 0


def copy_input_files(workdir, scph_example_dir, fc_reference_dir):
    reference_dir = os.path.join(scph_example_dir, "reference_for_test")
    source_and_dest = [
        (
            os.path.join(reference_dir, "BTO_scph_thermo.in"),
            "BTO_scph_thermo.in",
        ),
        (os.path.join(fc_reference_dir, "cBTO222.h5"), "cBTO222.h5"),
    ]

    for src, dest in source_and_dest:
        if os.path.exists(src):
            shutil.copy(src, os.path.join(workdir, dest))
        else:
            print(f"File {src} not found")
            return 1

    source_strain_dir = os.path.join(reference_dir, "strain_phonon")
    if os.path.exists(source_strain_dir):
        shutil.copytree(
            source_strain_dir,
            os.path.join(workdir, "strain_phonon"),
            dirs_exist_ok=True,
        )
    else:
        print(f"Directory {source_strain_dir} not found")
        return 1

    return 0


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

    workdir = f"{project_root}/test/batio3"
    if not os.path.exists(workdir):
        os.mkdir(workdir)
    os.chdir(workdir)

    anphonbin = "%s/_build/anphon/anphon" % project_root
    info = 0

    if args.jobs in ["all", "copy"]:
        info = copy_input_files(workdir, scph_example_dir, fc_reference_dir)
        if info > 0:
            sys.exit(1)

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
        print("BaTiO3 ANPHON --> pass")
    else:
        print("BaTiO3 ANPHON --> failed")

    if info == 0:
        sys.exit(0)
    else:
        sys.exit(1)
