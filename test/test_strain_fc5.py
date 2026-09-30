#!/usr/bin/env python
"""Regression test: STRAIN_FC5, dPhi4/du from the quintic IFCs (&relax).

On the 5-atom BaTiO3 cell with cBTO222.h5 plus a synthetic, translationally
invariant Order5 (a fifth-power Ti-O bond-stretch potential, cubic symmetric):

- STRAIN_FC5 = 0 with the Order5 group present must reproduce the run on the
  original file exactly.
- STRAIN_FC5 = 1 must change the result, and V4_REAL_SPACE = 1 must reproduce
  V4_REAL_SPACE = 0.
- The strain gradient dF/du_mn printed at a prescribed, general strain
  (RELAX_STR = 4, VERBOSITY = 2) must be the derivative of the free energy
  F_total: central differences in u_xx and in the symmetric u_yz (which tests
  the three explicit stress terms of the correction and the off-diagonal
  channels).
"""

import itertools
import os
import re
import shutil
import subprocess
import sys

import h5py
import numpy as np

WORKDIR = "strain_fc5"
US = np.array(
    [[0.0, 0.002, 0.003], [0.002, 0.0, 0.004], [0.003, 0.004, 0.0]]
)  # shear only
U0 = np.array([[0.004, 0.001, 0.0015], [0.001, 0.003, 0.002], [0.0015, 0.002, 0.012]])
H = 2.0e-3  # finite-difference steps H and 2H, Richardson-extrapolated
C5 = 5.0  # Ry/bohr^5


def run_anphon(cmd, logfile):
    with open(logfile, "w") as f:
        return subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT, timeout=1800
        ).returncode


def add_order5(src, dest):
    """Copy src and add the Order5 of a cubic-symmetric Ti-O bond potential,
    V = sum_bonds s [C5/120 d_a^5 + C5/12 d_a^3 (d_b^2 + d_c^2)], d = u_O - u_Ti,
    a the bond axis, b, c the other two, s the sign of the bond along a:
    Phi5 = C5 s (-1)^n_Ti for the legs a^5 and a^3 b^2, translationally
    invariant by construction. The a^3 b^2 legs give the shear channels
    dPhi4/du_ba."""
    shutil.copy(src, dest)
    with h5py.File(dest, "r+") as h:
        lat = h["SuperCell/lattice_vector"][()]
        assert np.allclose(lat, np.diag(np.diag(lat)))  # the cubic 2x2x2 cell
        xf = h["SuperCell/fractional_coordinate"][()]
        kinds = h["SuperCell/atomic_kinds"][()]
        maps = h["SuperCell/mapping_table"][()]
        prim_of = {int(s): p for p, row in enumerate(maps) for s in row}
        ti = int(maps[1][0])
        assert kinds[ti] == 1

        def rel(a, b):  # Cartesian minimum-image vector from atom a to atom b
            return ((xf[b] - xf[a] + 0.5) % 1.0 - 0.5) * np.diag(lat)

        def atom_at(f):
            return int(np.argmin(np.abs((xf - f % 1.0 + 0.5) % 1.0 - 0.5).max(axis=1)))

        oxygens = np.nonzero(kinds == 2)[0]
        dist = np.array([np.linalg.norm(rel(ti, o)) for o in oxygens])
        rows = {}  # (leg 0, sorted legs 1-4) -> [value, shifts]; a leg is (3 * atom + xyz)
        for o in oxygens[dist < dist.min() + 1e-6]:
            o = int(o)
            vec = rel(ti, o)
            a = int(np.argmax(np.abs(vec)))
            sign = np.sign(vec[a])
            patterns = [[a] * 5] + [[a, a, a, b, b] for b in range(3) if b != a]
            elements = set()
            for coords in patterns:
                for atoms in itertools.product((ti, o), repeat=5):
                    elements.add(
                        tuple(sorted(3 * t + c for t, c in zip(atoms, coords)))
                    )
            # the bond seen from Ti and from the primitive image of O
            o0 = int(maps[prim_of[o]][0])
            image = {ti: atom_at(xf[ti] + xf[o0] - xf[o]), o: o0}
            for elem in elements:
                value = C5 * sign * (-1.0) ** sum(leg // 3 == ti for leg in elem)
                for center, legs, r in (
                    (ti, list(elem), vec),
                    (o0, [3 * image[x // 3] + x % 3 for x in elem], -vec),
                ):
                    for head in sorted({x for x in legs if x // 3 == center}):
                        tail = sorted(legs)
                        tail.remove(head)
                        shifts = np.concatenate(
                            [np.zeros(3) if x // 3 == center else r for x in tail]
                        )
                        rows.setdefault((head, tuple(tail)), [0.0, shifts])[0] += value
        keys = [k for k in sorted(rows) if abs(rows[k][0]) > 1e-12]
        legs = np.array([[k[0], *k[1]] for k in keys])
        g = h.create_group("ForceConstants/Order5")
        g["atom_indices"] = np.vectorize(lambda x: prim_of[int(x) // 3])(legs).astype(
            np.int32
        )
        g["atom_indices_supercell"] = (legs // 3).astype(np.int32)
        g["coord_indices"] = (legs % 3).astype(np.int32)
        g["shift_vectors"] = np.array([rows[k][1] for k in keys])
        g["shift_vectors"].attrs["basis"] = "Cartesian"
        g["shift_vectors"].attrs["unit"] = "bohr"
        g["force_constant_values"] = np.array([rows[k][0] for k in keys])
        g["force_constant_values"].attrs["unit"] = "Ry/bohr^5"
        return len(keys)


def write_input(prefix, fcsfile, tags="", relax_tags="", U=U0):
    with open("BTO_scph_thermo.in") as f:
        src = f.read()
    src = re.sub(r"PREFIX\s*=\s*\S+", "PREFIX = %s" % prefix, src, count=1)
    src = re.sub(
        r"FCSFILE\s*=\s*\S+", "FCSFILE = %s\n  VERBOSITY = 2" % fcsfile, src, count=1
    )
    src = re.sub(r"TMIN\s*=\s*\S+", "TMIN = 300", src, count=1)
    src = re.sub(r"TMAX\s*=\s*\S+", "TMAX = 300", src, count=1)
    src = re.sub(r"KMESH_SCPH\s*=.*", "KMESH_SCPH = 2 2 2", src, count=1)
    src = re.sub(
        r"MAXITER\s*=\s*\S+", "MAXITER = 5000\n  TOL_SCPH = 1.0e-12", src, count=1
    )
    src = re.sub(r"RELAX_STR\s*=\s*\S+", "RELAX_STR = 4" + tags, src, count=1)
    src = re.sub(r"COORD_CONV_TOL\s*=\s*\S+", "COORD_CONV_TOL = 1.0e-9", src, count=1)
    src = re.sub(r"SET_INIT_STR\s*=\s*\S+", "SET_INIT_STR = 1", src, count=1)
    src = re.sub(r"\n\s*COOLING_U0_\w+\s*=\s*\S+", "", src)
    src = src.replace(
        "STRAIN_IFC_DIR = strain_phonon",
        "STRAIN_IFC_DIR = strain_phonon" + relax_tags,
        1,
    )
    # F_total on the SCPH mesh itself, so that it is the functional dF/du_mn differentiates
    src = re.sub(r"&kpoint\n.*?/", "&kpoint\n  2\n  2 2 2\n/", src, count=1, flags=re.S)
    strain = "\n".join(" ".join("%.10f" % x for x in row) for row in U)
    src = re.sub(r"&strain\n.*?/", "&strain\n%s\n/" % strain, src, count=1, flags=re.S)
    with open(prefix + ".in", "w") as f:
        f.write(src)


def free_energy(prefix):
    return np.loadtxt(prefix + ".scph_thermo", ndmin=2)[-1, 5]


def strain_gradient(prefix):
    with open(prefix + ".log") as f:
        found = re.findall(r"strain gradient dF/du_mn \[Ry\] :(.*)", f.read())
    return np.array(found[-1].split(), float)


def main():
    test_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(test_root)
    anphonbin = os.path.join(project_root, "_build/anphon/anphon")
    workdir = os.path.join(test_root, WORKDIR)
    if os.path.exists(workdir):
        shutil.rmtree(workdir)
    os.makedirs(workdir)
    sys.path.insert(0, test_root)
    from test_batio3 import copy_input_files

    scph_example_dir = os.path.join(project_root, "example", "BaTiO3", "scph_relax")
    fc_reference_dir = os.path.join(
        project_root, "example", "BaTiO3", "anharm_IFCs", "4_optimize", "reference"
    )
    if copy_input_files(workdir, scph_example_dir, fc_reference_dir) != 0:
        return 1
    os.chdir(workdir)
    nrows = add_order5("cBTO222.h5", "cBTO222_fc5.h5")
    print("synthetic Order5: %d rows" % nrows)
    shutil.copy("cBTO222_fc5.h5", "cBTO222_fc5zero.h5")
    with h5py.File("cBTO222_fc5zero.h5", "r+") as h:
        h["ForceConstants/Order5/force_constant_values"][...] = 0.0

    on = "\n  STRAIN_FC5 = 1"
    write_input("ref", "cBTO222.h5")
    write_input("off", "cBTO222_fc5.h5", relax_tags="\n  STRAIN_FC5 = 0")
    write_input("on0", "cBTO222_fc5.h5", relax_tags=on)
    write_input("on1", "cBTO222_fc5.h5", tags="\n  V4_REAL_SPACE = 1", relax_tags=on)
    fd = {"xx": [(0, 0)], "yz": [(1, 2), (2, 1)]}
    for c, idx in fd.items():
        for s, h in (("p1", H), ("m1", -H), ("p2", 2 * H), ("m2", -2 * H)):
            U = U0.copy()
            for i, j in idx:
                U[i, j] += h
            write_input("%s_%s" % (c, s), "cBTO222_fc5.h5", relax_tags=on, U=U)
    # all-zero Order5; a shear-only strain with STRAIN_FC5_CHANNELS = DIAG (no active channel)
    write_input("zero", "cBTO222_fc5zero.h5", relax_tags=on)
    for prefix, extra in (
        ("s_off", ""),
        ("s_all", on),
        ("s_diag", on + "\n  STRAIN_FC5_CHANNELS = DIAG"),
    ):
        write_input(prefix, "cBTO222_fc5.h5", relax_tags=extra, U=US)
    runs = ["ref", "off", "on0", "on1", "zero", "s_off", "s_all", "s_diag"] + [
        "%s_%s" % (c, s) for c in fd for s in ("p1", "m1", "p2", "m2")
    ]
    for prefix in runs:
        if run_anphon([anphonbin, prefix + ".in"], prefix + ".log"):
            print("%s run failed, see %s/%s.log" % (prefix, WORKDIR, prefix))
            return 1

    ok = True
    exts = (".scph_thermo", ".atom_disp", ".umn_tensor")
    for ext in exts:
        a, b = np.loadtxt("ref" + ext), np.loadtxt("off" + ext)
        if a.shape != b.shape or not np.array_equal(a, b):
            print("STRAIN_FC5 = 0: %s differs with the Order5 group present" % ext)
            ok = False
    for ext in exts:
        a, b = np.loadtxt("on0" + ext), np.loadtxt("on1" + ext)
        if a.shape != b.shape or not np.allclose(a, b, rtol=0.0, atol=1e-10):
            print("STRAIN_FC5 = 1: V4_REAL_SPACE = 1 differs in %s" % ext)
            ok = False
    for ext in exts:
        a, b = np.loadtxt("ref" + ext), np.loadtxt("zero" + ext)
        if a.shape != b.shape or not np.allclose(a, b, rtol=0.0, atol=1e-12):
            print(
                "STRAIN_FC5 = 1 with an all-zero Order5 differs from the run without it in %s"
                % ext
            )
            ok = False

    # DIAG at a shear-only strain: the correction has zero weights, so the structure and F
    # are those without it and so are the shear gradients; the normal-strain gradients get
    # the dPhi4/du_mm terms. ALL changes F.
    for ext in exts:
        a, b = np.loadtxt("s_off" + ext), np.loadtxt("s_diag" + ext)
        if a.shape != b.shape or not np.allclose(a, b, rtol=0.0, atol=1e-10):
            print("DIAG at a shear strain differs from STRAIN_FC5 = 0 in %s" % ext)
            ok = False
    gs_off, gs_diag = strain_gradient("s_off"), strain_gradient("s_diag")
    shear = [1, 2, 3, 5, 6, 7]
    if not np.allclose(gs_off[shear], gs_diag[shear], rtol=0.0, atol=1e-10):
        print("DIAG changes the shear gradients: %s" % (gs_diag - gs_off))
        ok = False
    if np.abs(gs_diag - gs_off)[[0, 4, 8]].max() < 1e-5:
        print("DIAG: no normal-strain gradient from dPhi4/du")
        ok = False
    if abs(free_energy("s_all") - free_energy("s_off")) < 1e-7:
        print("ALL at a shear strain has no effect")
        ok = False

    g_off, g = strain_gradient("off"), strain_gradient("on0")
    df = abs(free_energy("on0") - free_energy("ref"))
    if df < 1e-6 or np.abs(g - g_off).max() < 1e-5:
        print(
            "STRAIN_FC5 = 1 has no effect: dF = %.2e, |dg| = %.2e"
            % (df, np.abs(g - g_off).max())
        )
        ok = False
    print(
        "STRAIN_FC5 changes F by %.3e Ry, dF/du by up to %.3e Ry"
        % (df, np.abs(g - g_off).max())
    )

    for c, idx in fd.items():
        d1 = (free_energy(c + "_p1") - free_energy(c + "_m1")) / (2 * H)
        d2 = (free_energy(c + "_p2") - free_energy(c + "_m2")) / (4 * H)
        g_fd = (4 * d1 - d2) / 3
        g_an = sum(g[3 * i + j] for i, j in idx)
        g_an_off = sum(g_off[3 * i + j] for i, j in idx)
        err = abs(g_fd - g_an)
        print(
            "dF/du_%s: finite difference %.6e, analytic %.6e (without the FC5 terms %.6e)"
            % (c, g_fd, g_an, g_an_off)
        )
        # F_total has 7 digits: ~1e-8 Ry / H of noise
        if err > 2e-5 or err > 0.05 * abs(g_an - g_an_off):
            print("dF/du_%s does not match the finite difference" % c)
            ok = False
    # restart: a state file of another STRAIN_FC5 / STRAIN_FC5_CHANNELS must be refused
    for prefix, extra, want in (
        ("off", on, "STRAIN_FC5 tag"),
        ("on0", on + "\n  STRAIN_FC5_CHANNELS = DIAG", "STRAIN_FC5_CHANNELS tag"),
    ):
        write_input(prefix, "cBTO222_fc5.h5", relax_tags=extra)
        failed = run_anphon([anphonbin, prefix + ".in"], prefix + "_restart.log") != 0
        with open(prefix + "_restart.log") as f:
            refused = want in f.read()
        if not (failed and refused):
            print(
                "restart of %s with%s was not refused"
                % (prefix, extra.replace("\n", ""))
            )
            ok = False

    print("STRAIN_FC5 --> %s" % ("pass" if ok else "fail"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
