#!/usr/bin/env python
"""Regression test: DFC2FILE from an SCPH run whose &cell is a supercell of the
primitive cell.

An SCPH run on a 1x1x2 cell of cubic BaTiO3 (KMESH_INTERPOLATE = KMESH_SCPH =
2 2 1, which match the 2x2x2 DFT supercell) samples exactly the same q points
as an SCPH run on the 5-atom primitive cell with 2 2 2 / 2 2 2, so both must
converge to the same anharmonic FC2 correction. A MODE = phonons run on the
primitive cell must therefore give the same frequencies whether it reads the
primitive-cell state file (same-cell path) or folds the doubled-cell one.

check_skewed_cell_na3: SCPH with NONANALYTIC = 3 on a 10-atom cell with a skewed
basis, whose supercell images of the primitive atoms are wrapped by a lattice
vector, on a mesh equivalent to the cubic 4x4x4 one (not commensurate with the
2x2x2 IFC supercell), must reproduce the 5-atom SCPH frequencies at the folded q.
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
    if not ok:
        return 1

    ok = check_skewed_cell_na3(anphonbin, project_root)
    print(
        "BaTiO3 SCPH, NONANALYTIC = 3, skewed 10-atom cell == primitive cell --> %s"
        % ("pass" if ok else "fail")
    )
    if not ok:
        return 1

    ok = check_fc2_export_round_trip(anphonbin)
    print(
        "BaTiO3 SCPH state file with NONANALYTIC = 1/2/3 read back as FC2 --> %s"
        % ("pass" if ok else "fail")
    )
    return 0 if ok else 1


def check_fc2_export_round_trip(anphonbin):
    """The FC2 of an SCPH state file, read with the same NONANALYTIC, must give back the
    SCP frequencies at the q of the coarse mesh: through DFC2FILE, through the
    temperature-dependent FC2 (FCSFILE or FC2FILE + FC2_TEMPERATURE), and the harmonic
    ones through FC2FILE alone. 4/4 meshes, not commensurate with the 2x2x2 IFCs, so the
    nonanalytic term of the run differs from that of the IFCs alone on that mesh;
    NA_SIGMA = 1 for NONANALYTIC = 1 keeps its (complex) zone-boundary term.
    """
    qs = [[0, 0, 0.25], [0.25, 0.25, 0], [0.5, 0.25, 0], [0.5, 0.5, 0.5]]
    kpts = "&kpoint\n 0\n%s\n/\n" % "\n".join(" %g %g %g" % tuple(q) for q in qs)

    def run(prefix, general, na, extra=""):
        if os.path.exists(prefix + ".scph.h5"):
            os.remove(prefix + ".scph.h5")
        with open(prefix + ".in", "w") as f:
            f.write(
                "&general\n PREFIX = %s; %s\n NONANALYTIC = %d; BORNINFO = BORNINFO%s\n/\n%s%s"
                % (
                    prefix,
                    general,
                    na,
                    "; NA_SIGMA = 1.0" if na == 1 else "",
                    extra,
                    kpts,
                )
            )
        if run_anphon(anphonbin, prefix + ".in", prefix + ".log") != 0:
            print("%s failed, see %s.log" % (prefix, prefix))
            return None
        if general.startswith("MODE = SCPH"):
            return np.loadtxt(prefix + ".scph_eval")[:, 3].reshape(len(qs), -1)
        with open(prefix + ".log") as f:
            freq = [
                float(line.split()[1])
                for line in f
                if "cm^-1" in line and line.split()[0].isdigit()
            ]
        return np.array(freq).reshape(len(qs), -1)

    ok = True
    for na in (1, 2, 3):
        p = "rt%d" % na
        scph = run(
            p,
            "MODE = SCPH; FCSFILE = cBTO222.h5; TMIN = 300; TMAX = 300",
            na,
            "&scph\n KMESH_INTERPOLATE = 4 4 4; KMESH_SCPH = 4 4 4; SELF_OFFDIAG = 1;"
            " MAXITER = 500; MIXALPHA = 0.2\n/\n",
        )
        harm = run(p + "_h", "MODE = phonons; FCSFILE = cBTO222.h5", na)
        routes = {
            "DFC2FILE": "FCSFILE = cBTO222.h5; DFC2FILE = %s.scph.h5; FC2_TEMPERATURE = 300",
            "FCSFILE + FC2_TEMPERATURE": "FCSFILE = %s.scph.h5; FC2_TEMPERATURE = 300",
            "FC2FILE + FC2_TEMPERATURE": "FCSFILE = cBTO222.h5; FC2FILE = %s.scph.h5; FC2_TEMPERATURE = 300",
            "FC2FILE (harmonic)": "FCSFILE = cBTO222.h5; FC2FILE = %s.scph.h5",
        }
        for name, src in routes.items():
            f = run(p + "_r", "MODE = phonons; " + src % p, na)
            ref = harm if "harmonic" in name else scph
            if scph is None or harm is None or f is None or f.shape != ref.shape:
                return False
            diff = np.abs(f - ref).max()
            # the frequencies are printed with 4 (log) and 6 (scph_eval) digits
            if not np.isfinite(diff) or diff > 2.0e-3:
                print(
                    "NONANALYTIC = %d, %s: max |diff| = %.3e cm^-1" % (na, name, diff)
                )
                ok = False

    # The FC2 of a state file are read only with a compatible NONANALYTIC (2 and 3
    # interchangeable), and a restart only with the same one.
    def refused(prefix, text, message):
        with open(prefix + ".in", "w") as f:
            f.write(text)
        if run_anphon(anphonbin, prefix + ".in", prefix + ".log") == 0:
            return False
        with open(prefix + ".log") as f:
            return message in f.read()

    for na_file, na_reader in ((1, 0), (3, 1), (2, 0)):
        text = (
            "&general\n PREFIX = rt_x; MODE = phonons; FCSFILE = rt%d.scph.h5;"
            " FC2_TEMPERATURE = 300\n NONANALYTIC = %d; BORNINFO = BORNINFO\n/\n%s"
            % (na_file, na_reader, kpts)
        )
        if not refused("rt_x", text, "this run uses NONANALYTIC"):
            print(
                "FC2 of a NONANALYTIC = %d file read with %d: not refused"
                % (na_file, na_reader)
            )
            ok = False
    # a NONANALYTIC = 0 state file read with a nonanalytic term: allowed, with a warning
    if (
        run(
            "rt0",
            "MODE = SCPH; FCSFILE = cBTO222.h5; TMIN = 300; TMAX = 300",
            0,
            "&scph\n KMESH_INTERPOLATE = 4 4 4; KMESH_SCPH = 4 4 4; SELF_OFFDIAG = 1;"
            " MAXITER = 500; MIXALPHA = 0.2\n/\n",
        )
        is None
    ):
        return False
    for na_reader in (3, 1):
        f = run(
            "rt0_r%d" % na_reader,
            "MODE = phonons; FCSFILE = rt0.scph.h5; FC2_TEMPERATURE = 300",
            na_reader,
        )
        with open("rt0_r%d.log" % na_reader) as fh:
            warned = "computed without the long-range" in fh.read()
        if f is None or not warned:
            print(
                "NONANALYTIC = 0 file read with %d: not accepted with a warning"
                % na_reader
            )
            ok = False
    cross = run(
        "rt_c", "MODE = phonons; FCSFILE = rt3.scph.h5; FC2_TEMPERATURE = 300", 2
    )
    ref3 = np.loadtxt("rt3.scph_eval")[:, 3].reshape(len(qs), -1)
    if cross is None or np.abs(cross - ref3).max() > 2.0e-3:
        print(
            "a NONANALYTIC = 3 state file read with NONANALYTIC = 2 does not reproduce it"
        )
        ok = False
    with open("rt3.in") as f:
        restart = (
            f.read()
            .replace("NONANALYTIC = 3", "NONANALYTIC = 2")
            .replace("SELF_OFFDIAG = 1;", "SELF_OFFDIAG = 1; RESTART_SCPH = 1;")
        )
    if not refused("rt3", restart, "NONANALYTIC tag is not consistent"):
        print("a restart with a different NONANALYTIC: not refused")
        ok = False
    return ok


def check_skewed_cell_na3(anphonbin, project_root):
    """SCPH, NONANALYTIC = 3, 10-atom cell a(2,0,0), a(1,1,0), a(0,0,1) on 2 4 4 meshes
    (= cubic 4x4x4) against the 5-atom cell on 4 4 4 meshes, at 300 K.

    The Ewald terms and the IFCs must use one Bloch gauge (lattice vectors measured
    from the primitive-cell positions), and the SCP correction must be taken against
    the harmonic matrix the interpolation adds back. Before, the two gauges differed
    at q off the 2x2x2 set (images wrapped by a(2,0,0)) and the folded frequencies were
    off by up to 4e2 cm^-1, with NONANALYTIC = 0 exact.
    """
    import re

    lat10 = np.array([[2, 0, 0], [1, 1, 0], [0, 0, 1]], float)  # rows, units of A
    b10 = np.linalg.inv(lat10).T  # reciprocal vectors / (2 pi / A)
    sites = np.array(
        [[0, 0, 0], [0.5, 0.5, 0.5], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    )
    shutil.copy(
        os.path.join(project_root, "example/BaTiO3/scph_relax/BORNINFO"), "BORNINFO"
    )
    cell10 = "&cell\n %.11f\n%s\n/\n" % (
        A,
        "\n".join(" %g %g %g" % tuple(r) for r in lat10),
    )

    def general(prefix, mode, born):
        na = "NONANALYTIC = 3; BORNINFO = %s" % born if born else "NONANALYTIC = 0"
        return (
            "&general\n PREFIX = %s; MODE = %s; FCSFILE = cBTO222.h5\n %s\n"
            " TMIN = 300; TMAX = 300\n/\n" % (prefix, mode, na)
        )

    def kpoints(ks):
        return "&kpoint\n 0\n%s\n/\n" % "\n".join(
            " %.12f %.12f %.12f" % tuple(k) for k in ks
        )

    def run(prefix, text):
        if os.path.exists(prefix + ".scph.h5"):  # would turn the run into a restart
            os.remove(prefix + ".scph.h5")
        with open(prefix + ".in", "w") as f:
            f.write(text)
        if run_anphon(anphonbin, prefix + ".in", prefix + ".log") != 0:
            print("%s failed, see %s.log" % (prefix, prefix))
            return False
        return True

    # Z* in the atom order of the 10-atom cell (BORN holds one tensor per atom); the
    # order is read from a run without the nonanalytic term
    if not run(
        "na3_pos10",
        general("na3_pos10", "phonons", None) + cell10 + kpoints([[0, 0, 0]]),
    ):
        return False
    with open("na3_pos10.log") as f:
        txt = (
            f.read()
            .split("Atomic positions in the primitive cell (fractional):")[1]
            .split("\n\n")[0]
        )
    born = np.loadtxt("BORNINFO")
    rows = [born[:3]]
    for *xf, _ in re.findall(r"^\s*\d+:\s+(\S+)\s+(\S+)\s+(\S+)\s+(\w+)", txt, re.M):
        d = (np.array(xf, float) @ lat10 - sites + 0.5) % 1.0 - 0.5
        k = int(np.argmin(np.linalg.norm(d, axis=1)))
        if np.linalg.norm(d[k]) > 1.0e-6:
            print("atom at %s of the 10-atom cell is not a cubic site" % xf)
            return False
        rows.append(born[3 + 3 * k : 6 + 3 * k])
    if len(rows) != 11:
        print("expected 10 atoms in the 10-atom cell, found %d" % (len(rows) - 1))
        return False
    np.savetxt("BORN10", np.vstack(rows), fmt="%16.8f")

    # 10-atom k (on the 2 4 4 mesh); each holds the cubic q = k and k + b1
    k10 = np.array(
        [[0, 0, 0], [0.5, 0, 0], [0, 0, 0.5], [0.5, 0.25, 0.25], [0, 0.25, 0.5]]
    )
    q5 = [q for k in k10 for q in (k @ b10, k @ b10 + b10[0])]
    scph = "&scph\n KMESH_INTERPOLATE = %s; KMESH_SCPH = %s\n SELF_OFFDIAG = 1; MAXITER = 500; MIXALPHA = 0.2\n/\n"
    if not run(
        "na3_p",
        general("na3_p", "SCPH", "BORNINFO") + scph % ("4 4 4", "4 4 4") + kpoints(q5),
    ):
        return False
    if not run(
        "na3_s10",
        general("na3_s10", "SCPH", "BORN10")
        + cell10
        + scph % ("2 4 4", "2 4 4")
        + kpoints(k10),
    ):
        return False

    def read_scph_eval(prefix, ns):
        return np.loadtxt(prefix + ".scph_eval")[:, 3].reshape(-1, ns)

    f5, f10 = read_scph_eval("na3_p", 15), read_scph_eval("na3_s10", 30)
    ok = len(f5) == 2 * len(k10) and len(f10) == len(k10)
    for ik in range(len(k10)) if ok else []:
        diff = np.abs(
            np.sort(f10[ik]) - np.sort(np.concatenate(f5[2 * ik : 2 * ik + 2]))
        ).max()
        # both runs converge to SCPH_TOL on the same q set; the eval file has 6 digits
        if not np.isfinite(diff) or diff > 1.0e-3:
            print(
                "k = %s of the 10-atom cell: max |diff| = %.3e cm^-1" % (k10[ik], diff)
            )
            ok = False
    return ok


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
