#!/usr/bin/env python

import argparse
import glob
import os
import shutil
import subprocess
import sys
import zipfile

import numpy as np


def isclose(a, b, rel_tol=1e-5, abs_tol=1.0e-12):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def run_anphon_zno(anphonbin):
    try:
        with open("ZnO_qha_thermo.log", "w") as f:
            subprocess.run([anphonbin, "ZnO_qha_thermo.in"], stdout=f)
    except Exception:
        return 1

    return 0


def check_consistency_anphon(example_dir, abs_tol=0.01, rel_tol=1.0e-9):
    files_to_compare = [
        "ZnO_qha.atom_disp",
        "ZnO_qha.umn_tensor",
        "ZnO_qha.V0",
        "ZnO_qha.qha_thermo",
    ]

    for file in files_to_compare:
        data_ref = np.loadtxt("%s/reference/%s" % (example_dir, file))
        data_now = np.loadtxt(file)

        m1, n1 = np.shape(data_ref)
        m2, n2 = np.shape(data_now)
        if not (m1 == m2 and n1 == n2):
            return 1

        isclose_all = True
        for i in range(m1):
            for j in range(n1):
                isclose_all = isclose_all & isclose(
                    data_ref[i, j], data_now[i, j], abs_tol, rel_tol
                )
        if not isclose_all:
            print("Failed to match %s" % file)
            return 1

    if not isclose_all:
        return 1

    return 0


def runtest_zno(anphonbin, example_dir):
    info = run_anphon_zno(anphonbin)

    if info > 0:
        print(
            "ANPHON code failed to execute.\nPlease check if the alm binary exist at %s"
            % anphonbin
        )
        return 1

    info = check_consistency_anphon(example_dir, abs_tol=1.0e-12, rel_tol=1.0e-12)

    if info == 0:
        print("ZnO ANPHON --> pass")
    else:
        print("ZnO ANPHON --> failed")

    return info


def copy_input_files(workdir, example_dir):
    file_list = [
        "ZnO332_500K_cutoff1208_nbody233.xml.zip",
        "ZnO442_harmonic.xml.zip",
        "ZnO_qha_thermo.in",
    ]
    for file in file_list:
        if os.path.exists(os.path.join(example_dir, file)):
            shutil.copy(os.path.join(example_dir, file), workdir)
        else:
            print(f"File {file} not found in {example_dir}")
            return 1

    if os.path.exists(os.path.join(example_dir, "strain_IFC")):
        shutil.copytree(
            os.path.join(example_dir, "strain_IFC"),
            os.path.join(workdir, "strain_IFC"),
            dirs_exist_ok=True,
        )
    else:
        print(f"Directory strain_IFC not found in {example_dir}")
        return 1

    return 0


def uncompress_files(workdir):
    file_list = [
        "ZnO332_500K_cutoff1208_nbody233.xml.zip",
        "ZnO442_harmonic.xml.zip",
    ]
    try:
        for file in file_list:
            with zipfile.ZipFile(file, "r") as zip_ref:
                zip_ref.extractall(workdir)
    except Exception:
        return 1

    files_list = glob.glob("strain_IFC/*.zip")
    try:
        for file in files_list:
            with zipfile.ZipFile(file, "r") as zip_ref:
                zip_ref.extractall(os.path.join(workdir, "strain_IFC"))
    except Exception:
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
    example_dir = os.path.join(project_root, "example/ZnO/qha_relax/")

    workdir = f"{project_root}/test/zno"
    if not os.path.exists(workdir):
        os.mkdir(workdir)
    os.chdir(workdir)

    # almbin = "%s/_build/alm/alm" % project_root
    anphonbin = "%s/_build/anphon/anphon" % project_root

    if args.jobs in ["all", "copy"]:
        info = copy_input_files(workdir, example_dir)
        if info > 0:
            sys.exit(1)

        info = uncompress_files(workdir)
        if info > 0:
            print("Failed to uncompress files")
            sys.exit(1)

    if args.jobs in ["all", "run"]:
        info = run_anphon_zno(anphonbin)

        if info > 0:
            print(
                "ANPHON code failed to execute.\nPlease check if the alm binary exist at %s"
                % anphonbin
            )
            sys.exit(1)

    if args.jobs in ["all", "compare"]:
        info = check_consistency_anphon(example_dir, abs_tol=1.0e-8, rel_tol=1.0e-10)

    if info == 0:
        print("ZnO ANPHON --> pass")
    else:
        print("ZnO ANPHON --> failed")

    if info == 0:
        sys.exit(0)
    else:
        sys.exit(1)
