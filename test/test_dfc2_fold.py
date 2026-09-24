#!/usr/bin/env python
"""Regression test: DFC2FILE from an SCPH run whose &cell is a supercell of the
primitive cell.

An SCPH run on a 1x1x2 cell of cubic BaTiO3 (KMESH_INTERPOLATE = KMESH_SCPH =
2 2 1, which match the 2x2x2 DFT supercell) samples exactly the same q points
as an SCPH run on the 5-atom primitive cell with 2 2 2 / 2 2 2, so both must
converge to the same anharmonic FC2 correction. A MODE = phonons run on the
primitive cell must therefore give the same frequencies whether it reads the
primitive-cell state file (same-cell path) or folds the doubled-cell one.
"""

import os
import shutil
import subprocess
import sys

import numpy as np

WORKDIR = "dfc2_fold"
FIXTURE = os.path.join("scph_h5", "cBTO222.h5")
A = 7.53159676409  # Bohr, cubic BaTiO3 fixture


def run_anphon(anphonbin, input_file, logfile):
    with open(logfile, "w") as f:
        return subprocess.run(
            [anphonbin, input_file], stdout=f, stderr=subprocess.STDOUT
        ).returncode


def read_eval(prefix):
    """Eigenvalues per k point from the text PRINTEVAL output (branch : value lines)."""
    blocks, cur = [], []
    with open(prefix + ".eval") as f:
        for line in f:
            p = line.split()
            if len(p) == 3 and p[1] == ":":
                if p[0] == "1" and cur:
                    blocks.append(cur)
                    cur = []
                cur.append(float(p[2]))
    if cur:
        blocks.append(cur)
    return [np.sort(np.array(b)) for b in blocks]


def write_inputs():
    cell_doubled = (
        "&cell\n 1.0\n %.10f 0.0 0.0\n 0.0 %.10f 0.0\n 0.0 0.0 %.10f\n/\n"
        % (A, A, 2 * A)
    )
    with open("scph_x2.in", "w") as f:
        f.write(
            "&general\n PREFIX = bto_x2; MODE = SCPH; FCSFILE = cBTO222.h5; TMIN = 300; TMAX = 300; DT = 100\n/\n"
        )
        f.write(cell_doubled)
        f.write(
            "&scph\n KMESH_INTERPOLATE = 2 2 1; KMESH_SCPH = 2 2 1; SELF_OFFDIAG = 1; MAXITER = 500; MIXALPHA = 0.2\n/\n"
        )
        f.write("&kpoint\n 2\n 2 2 1\n/\n&analysis\n QUARTIC = 1\n/\n")
    with open("scph_p.in", "w") as f:
        f.write(
            "&general\n PREFIX = bto_p; MODE = SCPH; FCSFILE = cBTO222.h5; TMIN = 300; TMAX = 300; DT = 100\n/\n"
        )
        f.write(
            "&scph\n KMESH_INTERPOLATE = 2 2 2; KMESH_SCPH = 2 2 2; SELF_OFFDIAG = 1; MAXITER = 500; MIXALPHA = 0.2\n/\n"
        )
        f.write("&kpoint\n 2\n 2 2 2\n/\n&analysis\n QUARTIC = 1\n/\n")
    kpts = "&kpoint\n 0\n 0.0 0.0 0.0\n 0.0 0.0 0.5\n 0.5 0.0 0.0\n 0.5 0.0 0.5\n 0.25 0.0 0.5\n/\n&analysis\n PRINTEVAL = 1\n/\n"
    for prefix, state in (("ph_fold", "bto_x2"), ("ph_same", "bto_p")):
        with open(prefix + ".in", "w") as f:
            f.write(
                "&general\n PREFIX = %s; MODE = phonons; FCSFILE = cBTO222.h5; DFC2FILE = %s.scph.h5;"
                % (prefix, state)
            )
            f.write(" FC2_TEMPERATURE = 300; FILE_FORMAT = text\n/\n")
            f.write(kpts)


def main():
    test_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(test_root)
    anphonbin = os.path.join(project_root, "_build/anphon/anphon")
    fixture = os.path.join(test_root, FIXTURE)
    if not os.path.exists(fixture):
        print("fixture %s not found (run test_scph_h5.py first)" % fixture)
        return 1

    workdir = os.path.join(test_root, WORKDIR)
    os.makedirs(workdir, exist_ok=True)
    shutil.copy(fixture, os.path.join(workdir, "cBTO222.h5"))
    os.chdir(workdir)
    write_inputs()

    for inp in ("scph_x2.in", "scph_p.in", "ph_fold.in", "ph_same.in"):
        if run_anphon(anphonbin, inp, inp.replace(".in", ".log")) != 0:
            print("%s failed, see %s" % (inp, inp.replace(".in", ".log")))
            return 1
    with open("ph_fold.log") as f:
        log = f.read()
    if "corrections are folded" not in log:
        print("the primitive-cell run did not fold the doubled-cell correction")
        return 1

    fold = read_eval("ph_fold")
    same = read_eval("ph_same")
    if len(fold) != 5 or len(same) != 5:
        print("unexpected number of k points in the eval files:", len(fold), len(same))
        return 1
    ok = True
    for ik, (a, b) in enumerate(zip(fold, same)):
        scale = max(np.abs(b).max(), 1e-12)
        # both SCPH runs are iterated to TOL_SCPH = 1e-10 on the same q set
        if not np.allclose(a, b, atol=1e-6 * scale):
            print(
                "mismatch at k point %d: max |diff| = %.3e (scale %.3e)"
                % (ik, np.abs(a - b).max(), scale)
            )
            ok = False
    print("BaTiO3 DFC2FILE fold --> %s" % ("pass" if ok else "fail"))
    if not ok:
        return 1

    ok = check_translation_breaking(anphonbin)
    print(
        "BaTiO3 DFC2FILE, correction breaking the primitive translations --> %s"
        % ("pass" if ok else "fail")
    )
    return 0 if ok else 1


def check_translation_breaking(anphonbin):
    """DFC2FILE must keep a correction that breaks the primitive translations.

    A relaxed SCPH cell (a cell-doubling distortion, say) carries different
    corrections on atoms the FCS-file primitive cell treats as equivalent.
    Here the self terms of one atom of the second copy are shifted in a copy
    of the doubled-cell state file. On that doubled cell, the harmonic FC2 +
    DFC2FILE route must then agree with the total FC2 read directly
    (FC2_TEMPERATURE) at q points of the SCPH mesh; replicating the first
    copy's correction onto the second, as a translation-assuming build would,
    breaks the agreement.
    """
    import h5py

    shutil.copy("bto_x2.scph.h5", "bto_x2_broken.scph.h5")
    with h5py.File("bto_x2_broken.scph.h5", "r+") as f:
        g = f["ForceConstants/Order2"]
        ai = g["atom_indices"][...]
        ci = g["coord_indices"][...]
        sv = g["shift_vectors"][...]
        xf = f["PrimitiveCell/fractional_coordinate"][...]
        second_copy = [i for i in range(len(xf)) if xf[i][2] % 1.0 >= 0.5 - 1e-6]
        atom = second_copy[0]
        rows = (ai[:, 0] == atom) & (ai[:, 1] == atom) & (ci[:, 0] == ci[:, 1])
        rows &= np.abs(sv).sum(axis=1) < 1e-8
        if rows.sum() != 3:
            print("could not find the self terms of atom %d" % atom)
            return False
        vals = f["ForceConstants/Order2_temperature_dependent/force_constant_values"]
        data = vals[...]
        data[0, rows] += 0.02 * np.abs(data[0]).max()
        vals[...] = data

    cell_doubled = (
        "&cell\n 1.0\n %.10f 0.0 0.0\n 0.0 %.10f 0.0\n 0.0 0.0 %.10f\n/\n"
        % (A, A, 2 * A)
    )
    kpts = "&kpoint\n 0\n 0.0 0.0 0.0\n 0.5 0.0 0.0\n/\n&analysis\n PRINTEVAL = 1\n/\n"
    runs = {
        "tb_dfc2": "FCSFILE = cBTO222.h5; DFC2FILE = bto_x2_broken.scph.h5",
        "tb_total": "FCSFILE = bto_x2_broken.scph.h5",
        "tb_orig": "FCSFILE = cBTO222.h5; DFC2FILE = bto_x2.scph.h5",
    }
    for prefix, src in runs.items():
        with open(prefix + ".in", "w") as f:
            f.write(
                "&general\n PREFIX = %s; MODE = phonons; %s; FC2_TEMPERATURE = 300;"
                " FILE_FORMAT = text\n/\n" % (prefix, src)
            )
            f.write(cell_doubled)
            f.write(kpts)
        if run_anphon(anphonbin, prefix + ".in", prefix + ".log") != 0:
            print("%s failed, see %s.log" % (prefix, prefix))
            return False

    dfc2 = read_eval("tb_dfc2")
    total = read_eval("tb_total")
    orig = read_eval("tb_orig")
    if not (len(dfc2) == len(total) == len(orig) == 2):
        print("unexpected number of k points:", len(dfc2), len(total), len(orig))
        return False
    ok = True
    for ik in range(2):
        scale = max(np.abs(total[ik]).max(), 1e-12)
        if not np.allclose(dfc2[ik], total[ik], atol=1e-6 * scale):
            print(
                "k point %d: DFC2FILE and total FC2 differ by %.3e"
                % (ik, np.abs(dfc2[ik] - total[ik]).max())
            )
            ok = False
    # The correction is read on rank 0 and appended on every rank after
    # replication (a collective): a 2-rank run must reproduce the serial one.
    if shutil.which("mpirun") is not None:
        with open("tb_dfc2_np2.in", "w") as f:
            f.write(
                open("tb_dfc2.in")
                .read()
                .replace("PREFIX = tb_dfc2", "PREFIX = tb_dfc2_np2")
            )
        with open("tb_dfc2_np2.log", "w") as f:
            ret = subprocess.run(
                ["mpirun", "-np", "2", anphonbin, "tb_dfc2_np2.in"],
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=600,
            ).returncode
        if ret != 0:
            print("DFC2FILE run on 2 MPI ranks failed, see tb_dfc2_np2.log")
            return False
        np2 = read_eval("tb_dfc2_np2")
        if len(np2) != 2 or any(
            not np.allclose(np2[ik], dfc2[ik], rtol=1e-10, atol=0.0) for ik in range(2)
        ):
            print("DFC2FILE on 2 MPI ranks differs from the serial run")
            return False

    # eval files hold omega^2 in atomic units (~1e-6): compare on that scale
    if all(
        np.allclose(dfc2[ik], orig[ik], atol=1e-3 * max(np.abs(orig[ik]).max(), 1e-12))
        for ik in range(2)
    ):
        print("the modified correction did not reach the frequencies")
        ok = False
    return ok


if __name__ == "__main__":
    sys.exit(main())
