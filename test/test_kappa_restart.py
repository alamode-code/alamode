#!/usr/bin/env python
"""Tests of the HDF5 kappa restart file (PREFIX.kappa.h5).

Covers: fresh-run schema, no-op restart, recovery from partially computed
files (completion flags), RESTART = 0 reset, the FILE_FORMAT = text legacy
path, one-way import of legacy text .result files, the /iterativebte group
written by SOLVER = IBTE (per-temperature restart), the SOLVER = VBTE
conjugate-gradient solver against the IBTE solution, and (optionally, set
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


def gen_rta_input(fname, extra_general="", extra_kappa="", mode="kappa"):
    with open(fname, "w") as f:
        f.write(
            "&general\n PREFIX = %s; MODE = %s; FCSFILE = si222_cubic.xml; KD = Si\n"
            % (PREFIX, mode)
        )
        if extra_general:
            f.write(" %s\n" % extra_general)
        f.write("/\n")
        f.write("&cell\n  10.203\n  0.0 0.5 0.5\n  0.5 0.0 0.5\n  0.5 0.5 0.0\n/\n")
        f.write("&kpoint\n  2\n  %s\n/\n" % KMESH)
        if extra_kappa:
            f.write("&kappa\n %s\n/\n" % extra_kappa)


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
        # The kappa tensors must state which scattering processes they include.
        if f["kappa"].attrs["includes_isotope_scattering"] != 1:
            print("includes_isotope_scattering flag is wrong")
            return 1
        if f["kappa/kappa_peierls"].attrs["scattering_processes"] != "3ph+isotope":
            print("scattering_processes label is wrong:",
                  f["kappa/kappa_peierls"].attrs["scattering_processes"])
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
    gen_rta_input("RTA0.in", extra_kappa="RESTART = 0")
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

    # The deprecated &general location must still work, with a warning.
    gen_rta_input("RTA0g.in", extra_general="RESTART = 0", mode="RTA")
    if run_anphon(anphonbin, "RTA0g.in", "restart_off_general.log") != 0:
        print("deprecated &general RESTART=0 run failed")
        return 1
    if not log_contains("restart_off_general.log", "RESTART and RESTART_4PH in the &general field are deprecated"):
        print("deprecated &general RESTART did not warn")
        return 1
    if not log_contains("restart_off_general.log", "MODE = RTA is deprecated"):
        print("deprecated MODE = RTA did not warn")
        return 1
    if not log_contains(
        "restart_off_general.log", "Total Number of phonon modes to be calculated : %d" % nmodes
    ):
        print("deprecated &general RESTART=0 was not honored")
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


IBTE_PREFIX = "si222_ibte"


def gen_ibte_input(fname, extra_kappa=""):
    with open(fname, "w") as f:
        f.write(
            "&general\n PREFIX = %s; MODE = kappa; FCSFILE = si222_cubic.xml; KD = Si\n"
            % IBTE_PREFIX
        )
        f.write(" TMIN = 200; TMAX = 300; DT = 100\n/\n")
        f.write("&cell\n  10.203\n  0.0 0.5 0.5\n  0.5 0.0 0.5\n  0.5 0.5 0.0\n/\n")
        f.write("&kpoint\n  2\n  4 4 4\n/\n")
        f.write("&kappa\n SOLVER = IBTE\n %s\n/\n" % extra_kappa)


def kl_iter_matches(reference):
    now = np.loadtxt(IBTE_PREFIX + ".kl_iter")
    return np.allclose(now, reference, rtol=1e-8, atol=1e-12)


def check_ibte_h5(anphonbin):
    """SOLVER = IBTE: /iterativebte group, per-temperature restart, reset."""
    for fname in (IBTE_PREFIX + ".kappa.h5", IBTE_PREFIX + ".kl_iter", IBTE_PREFIX + ".result"):
        if os.path.exists(fname):
            os.remove(fname)
    gen_ibte_input("ibte.in")
    if run_anphon(anphonbin, "ibte.in", "ibte_fresh.log"):
        print("IBTE fresh run failed")
        return 1
    if os.path.exists(IBTE_PREFIX + ".result"):
        print("IBTE run wrote the legacy .result file in h5 mode")
        return 1
    with h5py.File(IBTE_PREFIX + ".kappa.h5", "r") as f:
        if "iterativebte" not in f:
            print("/iterativebte group missing")
            return 1
        for name in ("Q", "dF", "kappa", "converged", "computed"):
            if name not in f["iterativebte"]:
                print("/iterativebte/%s missing" % name)
                return 1
        if not np.all(f["iterativebte/computed"][...] == 1):
            print("computed flags not fully set after the fresh IBTE run")
            return 1
    kl_fresh = np.loadtxt(IBTE_PREFIX + ".kl_iter")

    # No-op restart: every temperature restored, L not rebuilt.
    if run_anphon(anphonbin, "ibte.in", "ibte_noop.log"):
        print("IBTE no-op restart failed")
        return 1
    if not log_contains("ibte_noop.log", "skipping the calculation of the transition probabilities"):
        print("IBTE no-op restart rebuilt the transition probabilities")
        return 1
    if not kl_iter_matches(kl_fresh):
        print(".kl_iter differs after the IBTE no-op restart")
        return 1

    # Partial restart: discard one temperature and scribble its kappa row;
    # only that temperature is recomputed and the result is unchanged.
    with h5py.File(IBTE_PREFIX + ".kappa.h5", "r+") as f:
        flags = f["iterativebte/computed"][...]
        flags[0] = 0
        f["iterativebte/computed"][...] = flags
        f["iterativebte/kappa"][0, :, :] = -1.0
    if run_anphon(anphonbin, "ibte.in", "ibte_partial.log"):
        print("IBTE partial restart failed")
        return 1
    if not log_contains("ibte_partial.log", "restored from the kappa.h5 file"):
        print("IBTE partial restart did not restore the intact temperature")
        return 1
    if not kl_iter_matches(kl_fresh):
        print(".kl_iter differs after the IBTE partial restart")
        return 1
    with h5py.File(IBTE_PREFIX + ".kappa.h5", "r") as f:
        if not np.all(f["iterativebte/computed"][...] == 1):
            print("computed flags not restored after the partial restart")
            return 1

    # RESTART = 0 discards the stored temperatures.
    gen_ibte_input("ibte_r0.in", extra_kappa="RESTART = 0")
    if run_anphon(anphonbin, "ibte_r0.in", "ibte_r0.log"):
        print("IBTE RESTART=0 run failed")
        return 1
    if log_contains("ibte_r0.log", "restored from the kappa.h5 file"):
        print("IBTE RESTART=0 did not discard the previous results")
        return 1
    if not kl_iter_matches(kl_fresh):
        print(".kl_iter differs after IBTE RESTART=0")
        return 1

    # A truncated run (MAX_CYCLE below MIN_CYCLE, so convergence is never
    # reached) must be flagged unconverged; the next run warm-starts those
    # temperatures from the stored deviation function and converges.
    gen_ibte_input("ibte_short.in", extra_kappa="RESTART = 0; MAX_CYCLE = 3")
    if run_anphon(anphonbin, "ibte_short.in", "ibte_short.log"):
        print("IBTE truncated run failed")
        return 1
    with h5py.File(IBTE_PREFIX + ".kappa.h5", "r") as f:
        if np.any(f["iterativebte/converged"][...] != 0):
            print("truncated run was not flagged unconverged")
            return 1
    with open(IBTE_PREFIX + ".kl_iter") as f:
        if "WARNING" not in f.read():
            print(".kl_iter is missing the unconverged warning")
            return 1
    if run_anphon(anphonbin, "ibte.in", "ibte_continue.log"):
        print("IBTE warm restart failed")
        return 1
    if not log_contains("ibte_continue.log", "continuing from the stored deviation function"):
        print("warm restart did not continue from the stored dF")
        return 1
    with h5py.File(IBTE_PREFIX + ".kappa.h5", "r") as f:
        if not np.all(f["iterativebte/converged"][...] == 1):
            print("warm restart did not converge")
            return 1
    now = np.loadtxt(IBTE_PREFIX + ".kl_iter")
    if not np.allclose(now[:, [1, 5, 9]], kl_fresh[:, [1, 5, 9]], rtol=1e-2):
        # both runs stop within ITER_THRESHOLD of the fixed point, along
        # different iteration paths
        print("kappa after warm restart deviates from the fresh result")
        return 1
    return 0


ISO_PREFIX = "si222_iso"


def check_isotope_inscattering(anphonbin):
    """Isotope in-scattering raises kappa vs the diagonal-only treatment."""
    kappa = {}
    for state, tag in (("on", ""), ("off", " ISOTOPE_INSCATTERING = 0\n")):
        with open("iso_%s.in" % state, "w") as f:
            f.write(
                "&general\n PREFIX = %s_%s; MODE = kappa; FCSFILE = si222_cubic.xml; KD = Si\n"
                % (ISO_PREFIX, state)
            )
            f.write(" TMIN = 300; TMAX = 300; DT = 100\n/\n")
            f.write("&cell\n  10.203\n  0.0 0.5 0.5\n  0.5 0.0 0.5\n  0.5 0.5 0.0\n/\n")
            f.write("&kpoint\n  2\n  4 4 4\n/\n")
            f.write("&kappa\n SOLVER = IBTE\n ISOFACT = 0.02\n%s/\n" % tag)
        if run_anphon(anphonbin, "iso_%s.in" % state, "iso_%s.log" % state):
            print("isotope in-scattering run (%s) failed" % state)
            return 1
        kappa[state] = np.loadtxt("%s_%s.kl_iter" % (ISO_PREFIX, state)).reshape(-1)[1]
    if not kappa["on"] > kappa["off"]:
        print(
            "in-scattering did not raise kappa (on %.5f vs off %.5f)"
            % (kappa["on"], kappa["off"])
        )
        return 1
    if (kappa["on"] - kappa["off"]) / kappa["off"] > 0.1:
        print("in-scattering effect is implausibly large")
        return 1
    return 0


VBTE_PREFIX = "si222_vbte"


def check_vbte(anphonbin):
    """SOLVER = VBTE: CG solves the same system as IBTE; restart shared."""
    for fname in (VBTE_PREFIX + ".kappa.h5", VBTE_PREFIX + ".kl_iter"):
        if os.path.exists(fname):
            os.remove(fname)
    with open("ibte.in") as f:
        content = f.read().replace(IBTE_PREFIX, VBTE_PREFIX).replace("SOLVER = IBTE", "SOLVER = VBTE")
    with open("vbte.in", "w") as f:
        f.write(content)
    if run_anphon(anphonbin, "vbte.in", "vbte_fresh.log"):
        print("VBTE fresh run failed")
        return 1
    with h5py.File(VBTE_PREFIX + ".kappa.h5", "r") as f:
        if not np.all(f["iterativebte/computed"][...] == 1):
            print("VBTE computed flags not fully set")
            return 1
        if not np.all(f["iterativebte/converged"][...] == 1):
            print("VBTE did not converge")
            return 1
    # Same linear system as IBTE: the two solutions agree within the
    # solver tolerances (ITER_THRESHOLD on each side).
    kv = np.loadtxt(VBTE_PREFIX + ".kl_iter")
    ki = np.loadtxt(IBTE_PREFIX + ".kl_iter")
    if not np.allclose(kv[:, [1, 5, 9]], ki[:, [1, 5, 9]], rtol=5e-2):
        print("VBTE kappa deviates from the IBTE solution")
        return 1
    # No-op restart skips everything.
    if run_anphon(anphonbin, "vbte.in", "vbte_noop.log"):
        print("VBTE no-op restart failed")
        return 1
    if not log_contains("vbte_noop.log", "skipping the calculation of the transition probabilities"):
        print("VBTE no-op restart rebuilt L")
        return 1
    return 0


def check_dbte(anphonbin):
    """SOLVER = DBTE: dense diagnostic solver on a small mesh."""
    prefix = "si222_dbte"
    for fname in (prefix + ".kappa.h5", prefix + ".kl_iter"):
        if os.path.exists(fname):
            os.remove(fname)
    with open("ibte.in") as f:
        content = f.read().replace(IBTE_PREFIX, prefix).replace("SOLVER = IBTE", "SOLVER = DBTE")
    with open("dbte.in", "w") as f:
        f.write(content)
    if run_anphon(anphonbin, "dbte.in", "dbte_fresh.log"):
        print("DBTE run failed")
        return 1
    for text in ("assembly cross-check", "eigenvalues (Omega normalization", "dense collision kernel"):
        if not log_contains("dbte_fresh.log", text):
            print("DBTE diagnostics missing: %s" % text)
            return 1
    with h5py.File(prefix + ".kappa.h5", "r") as f:
        if not np.all(f["iterativebte/computed"][...] == 1):
            print("DBTE computed flags not fully set")
            return 1
    # Same physics as IBTE up to the explicit symmetrization of the kernel.
    kd = np.loadtxt(prefix + ".kl_iter")
    ki = np.loadtxt(IBTE_PREFIX + ".kl_iter")
    if not np.allclose(kd[:, [1, 5, 9]], ki[:, [1, 5, 9]], rtol=1e-1):
        print("DBTE kappa deviates too far from the IBTE solution")
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

    if check_ibte_h5(anphonbin):
        return 1
    print("IBTE /iterativebte restart --> pass")

    if check_vbte(anphonbin):
        return 1
    print("VBTE (CG) solver --> pass")

    if check_isotope_inscattering(anphonbin):
        return 1
    print("Isotope in-scattering --> pass")

    if check_dbte(anphonbin):
        return 1
    print("DBTE diagnostic solver --> pass")

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
