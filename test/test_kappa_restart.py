#!/usr/bin/env python
"""Tests of the HDF5 kappa restart file (PREFIX.kappa.h5).

Covers: fresh-run schema, no-op restart, recovery from partially computed
files (completion flags), RESTART = 0 reset, the FILE_FORMAT = text legacy
path, one-way import of legacy text .result files, and (optionally, set
ALAMODE_TEST_KILL=1) physical crash recovery via SIGKILL.
"""

import hashlib
import os
import shutil
import signal
import subprocess
import sys
import time

import h5py
import numpy as np

from test_si import run_alm_si

KMESH = "6 6 6"
PREFIX = "si222"


def file_sha256(fname):
    with open(fname, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def gen_rta_input(fname, extra_general=""):
    with open(fname, "w") as f:
        f.write(
            "&general\n PREFIX = %s; MODE = RTA; FCSFILE = si222_cubic.xml; KD = Si\n"
            % PREFIX
        )
        if extra_general:
            f.write(" %s\n" % extra_general)
        f.write("/\n")
        f.write("&cell\n  10.203\n  0.0 0.5 0.5\n  0.5 0.0 0.5\n  0.5 0.5 0.0\n/\n")
        f.write("&kpoint\n  2\n  %s\n/\n" % KMESH)


def run_anphon(anphonbin, input_file, logfile):
    with open(logfile, "w") as f:
        ret = subprocess.run([anphonbin, input_file], stdout=f, stderr=subprocess.STDOUT)
    return ret.returncode


def log_contains(logfile, text):
    with open(logfile) as f:
        return text in f.read()


def check_fresh_run(anphonbin):
    for fname in (PREFIX + ".kappa.h5", PREFIX + ".result", PREFIX + ".kl"):
        if os.path.exists(fname):
            os.remove(fname)
    gen_rta_input("RTA.in")
    if run_anphon(anphonbin, "RTA.in", "fresh.log") != 0:
        print("fresh run failed")
        return 1

    if os.path.exists(PREFIX + ".result"):
        print("legacy text .result should not be written in h5 mode")
        return 1

    with h5py.File(PREFIX + ".kappa.h5", "r") as f:
        if f.attrs["schema"] != "alamode:kappa_result" or f.attrs["format_version"] != 1:
            print("schema attributes are wrong:", dict(f.attrs))
            return 1
        g = f["scattering/3ph"]
        nmodes = g.attrs["nk_irred"] * g.attrs["nbranches"]
        flags = g["gamma_computed"][...]
        if flags.shape[0] != nmodes or not flags.all():
            print("gamma_computed flags are not all set")
            return 1
        if g["gamma"].shape != (nmodes, f["metadata/temperatures"].shape[0]):
            print("gamma dataset shape mismatch")
            return 1
        if f["kappa/valid"][0] != 1:
            print("kappa/valid is not set")
            return 1
        # kappa_total exists only when the coherent term is computed.
        if "kappa_total" in f["kappa"]:
            print("kappa_total should be absent without KAPPA_COHERENT")
            return 1
        # Isotope scattering is on by default; its data must be stored.
        if f["metadata/isotope"][()] < 1:
            print("metadata/isotope should default to 1")
            return 1
        gi = f["scattering/isotope/gamma"][...]
        if gi.shape != (nmodes // f["scattering/3ph"].attrs["nbranches"],
                        f["scattering/3ph"].attrs["nbranches"]) or not np.isfinite(gi).all() \
                or gi.max() <= 0.0:
            print("isotope gamma dataset is missing or empty")
            return 1
        kappa_h5 = f["kappa/kappa_peierls"][...]

    kl = np.loadtxt(PREFIX + ".kl")
    if not np.allclose(kappa_h5[:, [0, 1, 2], [0, 1, 2]], kl[:, [1, 5, 9]], atol=1e-3):
        print("/kappa/kappa_peierls disagrees with the text .kl file")
        return 1
    np.save("kl_fresh.npy", kl)
    return 0


def check_noop_restart(anphonbin):
    kl_before = np.loadtxt(PREFIX + ".kl")
    if run_anphon(anphonbin, "RTA.in", "restart_noop.log") != 0:
        print("no-op restart run failed")
        return 1
    if not log_contains("restart_noop.log", "Total Number of phonon modes to be calculated : 0"):
        print("no-op restart recomputed modes")
        return 1
    if not np.array_equal(kl_before, np.loadtxt(PREFIX + ".kl")):
        print(".kl changed after a no-op restart")
        return 1
    return 0


def check_partial_restart(anphonbin):
    # Simulate a run interrupted mid-way: clear the completion flags of the
    # last nmodes-keep rows and scribble their gamma values. Only those
    # modes must be recomputed, and the final result must be unchanged.
    with h5py.File(PREFIX + ".kappa.h5", "r+") as f:
        g = f["scattering/3ph"]
        gamma_ref = g["gamma"][...]
        nmodes = g["gamma_computed"].shape[0]
        keep = nmodes // 3
        g["gamma_computed"][keep:] = 0
        g["gamma"][keep:, :] = -12345.0

    kl_before = np.loadtxt(PREFIX + ".kl")
    if run_anphon(anphonbin, "RTA.in", "restart_partial.log") != 0:
        print("partial restart run failed")
        return 1
    if not log_contains(
        "restart_partial.log", "Total Number of phonon modes to be calculated : %d" % (nmodes - keep)
    ):
        print("partial restart did not recompute exactly the missing modes")
        return 1
    with h5py.File(PREFIX + ".kappa.h5", "r") as f:
        g = f["scattering/3ph"]
        if not g["gamma_computed"][...].all():
            print("flags not fully set after partial restart")
            return 1
        if not np.allclose(g["gamma"][...], gamma_ref, rtol=1e-10, atol=1e-12):
            print("recomputed gamma rows differ from the original run")
            return 1
    if not np.allclose(kl_before, np.loadtxt(PREFIX + ".kl"), rtol=1e-8):
        print(".kl differs after partial restart")
        return 1

    # An interior gap (flags cleared in the middle) must trigger the
    # prefix rule: everything from the gap on is recomputed.
    with h5py.File(PREFIX + ".kappa.h5", "r+") as f:
        g = f["scattering/3ph"]
        g["gamma_computed"][10:20] = 0
    if run_anphon(anphonbin, "RTA.in", "restart_gap.log") != 0:
        print("gap restart run failed")
        return 1
    if not log_contains("restart_gap.log", "recorded after an incomplete batch"):
        print("gap restart did not report the non-prefix flags")
        return 1
    if not log_contains(
        "restart_gap.log", "Total Number of phonon modes to be calculated : %d" % (nmodes - 10)
    ):
        print("gap restart did not recompute from the gap onwards")
        return 1
    if not np.allclose(kl_before, np.loadtxt(PREFIX + ".kl"), rtol=1e-8):
        print(".kl differs after gap restart")
        return 1
    return 0


def check_forced_recompute(anphonbin):
    kl_before = np.loadtxt(PREFIX + ".kl")
    with h5py.File(PREFIX + ".kappa.h5", "r") as f:
        nmodes = f["scattering/3ph/gamma_computed"].shape[0]
    gen_rta_input("RTA0.in", extra_general="RESTART = 0")
    if run_anphon(anphonbin, "RTA0.in", "restart_off.log") != 0:
        print("RESTART=0 run failed")
        return 1
    if not log_contains(
        "restart_off.log", "Total Number of phonon modes to be calculated : %d" % nmodes
    ):
        print("RESTART=0 did not recompute all modes")
        return 1
    if not np.allclose(kl_before, np.loadtxt(PREFIX + ".kl"), rtol=1e-8):
        print(".kl differs after forced recompute")
        return 1
    return 0


def check_legacy_text_roundtrip(anphonbin):
    # 1) FILE_FORMAT = text writes the legacy .result file (escape hatch).
    for fname in (PREFIX + ".kappa.h5", PREFIX + ".result"):
        if os.path.exists(fname):
            os.remove(fname)
    kl_ref = np.loadtxt(PREFIX + ".kl")
    gen_rta_input("RTA_text.in", extra_general="FILE_FORMAT = text")
    if run_anphon(anphonbin, "RTA_text.in", "text_mode.log") != 0:
        print("FILE_FORMAT=text run failed")
        return 1
    if not os.path.exists(PREFIX + ".result") or os.path.exists(PREFIX + ".kappa.h5"):
        print("FILE_FORMAT=text produced the wrong files")
        return 1
    if not np.allclose(kl_ref, np.loadtxt(PREFIX + ".kl"), rtol=1e-8):
        print(".kl differs in text mode")
        return 1

    # 2) The next default-format run imports the text file, leaves it
    #    byte-identical, and reproduces the result without recomputing.
    digest_before = file_sha256(PREFIX + ".result")
    if run_anphon(anphonbin, "RTA.in", "import.log") != 0:
        print("import run failed")
        return 1
    if not log_contains("import.log", "Imported"):
        print("legacy import did not run")
        return 1
    if not log_contains("import.log", "Total Number of phonon modes to be calculated : 0"):
        print("import run recomputed modes")
        return 1
    if file_sha256(PREFIX + ".result") != digest_before:
        print("legacy .result file was modified by the import")
        return 1
    if not os.path.exists(PREFIX + ".kappa.h5"):
        print("import did not create the kappa.h5 file")
        return 1
    # The legacy text format stores gamma with ~6 significant digits, so a
    # kappa recomputed from imported values only matches to that precision.
    if not np.allclose(kl_ref, np.loadtxt(PREFIX + ".kl"), rtol=1e-4):
        print(".kl differs after import")
        return 1
    return 0


def check_kill_recovery(anphonbin, fresh_runtime):
    # Physical crash: SIGKILL mid-run, then restart. The file must open,
    # some modes may be lost (uncommitted), none may be corrupt.
    for fname in (PREFIX + ".kappa.h5", PREFIX + ".result"):
        if os.path.exists(fname):
            os.remove(fname)
    # Compare against the fresh full-precision run, not the last .kl on disk
    # (the import test leaves a text-precision variant behind).
    kl_ref = np.load("kl_fresh.npy")
    with open("killed.log", "w") as f:
        proc = subprocess.Popen([anphonbin, "RTA.in"], stdout=f, stderr=subprocess.STDOUT)
        time.sleep(max(0.5, 0.5 * fresh_runtime))
        proc.send_signal(signal.SIGKILL)
        proc.wait()

    if run_anphon(anphonbin, "RTA.in", "recover.log") != 0:
        print("recovery run after SIGKILL failed")
        return 1
    with h5py.File(PREFIX + ".kappa.h5", "r") as f:
        if not f["scattering/3ph/gamma_computed"][...].all():
            print("flags not fully set after recovery")
            return 1
    if not np.allclose(kl_ref, np.loadtxt(PREFIX + ".kl"), rtol=1e-8):
        print(".kl differs after crash recovery")
        return 1
    return 0


def runtest_kappa_restart(almbin, anphonbin, project_root):
    if run_alm_si(almbin, project_root) != 0:
        print("alm setup failed")
        return 1

    t0 = time.time()
    if check_fresh_run(anphonbin):
        return 1
    fresh_runtime = time.time() - t0
    print("Fresh run + schema --> pass")

    if check_noop_restart(anphonbin):
        return 1
    print("No-op restart --> pass")

    if check_partial_restart(anphonbin):
        return 1
    print("Partial + gap restart --> pass")

    if check_forced_recompute(anphonbin):
        return 1
    print("RESTART = 0 --> pass")

    if check_legacy_text_roundtrip(anphonbin):
        return 1
    print("Text mode + legacy import --> pass")

    if os.environ.get("ALAMODE_TEST_KILL", "1") != "0":
        if check_kill_recovery(anphonbin, fresh_runtime):
            return 1
        print("SIGKILL recovery --> pass")

    return 0


if __name__ == "__main__":
    build_dir = os.getcwd()
    project_root = os.path.dirname(build_dir)

    dirname = "%s/test/kappa_restart" % project_root
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    sys.path.insert(0, "%s/test" % project_root)
    os.chdir(dirname)

    almbin = "%s/_build/alm/alm" % project_root
    anphonbin = "%s/_build/anphon/anphon" % project_root

    info = runtest_kappa_restart(almbin, anphonbin, project_root)

    sys.exit(0 if info == 0 else 1)
