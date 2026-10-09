#!/usr/bin/env python
"""Regression test for the little-group symmetrization of the group velocities.

With NONANALYTIC = 2 the dynamical matrix is not periodic in reciprocal space,
so on the zone boundary the velocities depend on which of the equally short
images of k the mesh keeps, and break the crystal symmetry unless they are
averaged over the little group of k (velocity_symmetry.h). Cubic SrTiO3
(SCPH-renormalized FC2 at 300 K, so that no mode is unstable) on a 4x4x4 mesh,
which contains the X, M and R points, must then give an isotropic kappa: without
the symmetrization kxy/kxx is about -1e-2 here.

Checks (one SCPH run and three short kappa runs):
  - NONANALYTIC = 2 with KAPPA_COHERENT = 1: Peierls and coherent tensors finite,
    kxx > 0, every off-diagonal element and every diagonal difference below 1e-10
    of kxx, and kxx against the reference value;
  - the finite-difference velocities (scattering/3ph/velocities) are invariant
    under the little group of every mesh point;
  - NONANALYTIC = 3, where the velocities are already symmetric: kappa equal to the
    value of the unsymmetrized code (NONANALYTIC = 0 is covered by test_si.py and
    test_kappa_restart.py, whose Si references did not move);
  - NONANALYTIC = 2 on 2 MPI ranks (when mpirun is available) equals the serial run;
  - polar P4mm (Ti shifted along z) with NONANALYTIC = 3, KAPPA_COHERENT = 1 and a small
    acoustic-sum-rule violation in the FC2 (Gamma acoustic modes at ~0.002-0.005 cm^-1,
    above the frequency cutoff): the velocity matrix rows and columns of the uniform
    translations at Gamma must be zeroed. Before, their 0/0 velocities gave a coherent
    kxx of 3e8 W/mK and kxy/kxx = 2.6e-6; now kxy/kxx = 1e-9, the floor of the NA = 3 runs.
The algebra of the average itself (idempotence, invariance, gauge invariance of the
block traces) is covered by the C++ unit test test_velocity_symmetry.

Run from the build directory: python3 ../test/test_velsym.py
"""

import bz2
import itertools
import os
import re
import shutil
import subprocess
import sys

import h5py
import numpy as np

CELL = """&cell
7.363
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
/
"""

SCPH_INPUT = (
    """&general
 PREFIX = sto_scph
 MODE = SCPH
 KD = Sr Ti O
 FCSFILE = STO_anharm.xml
 NONANALYTIC = 3; BORNINFO = BORN
 TMIN = 300; TMAX = 300; DT = 100
/
"""
    + CELL
    + """&kpoint
0
0.0 0.0 0.0
/
&scph
 SELF_OFFDIAG = 0
 MAXITER = 500
 MIXALPHA = 0.2
 KMESH_INTERPOLATE = 2 2 2
 KMESH_SCPH = 2 2 2
/
"""
)

NMESH = 4

# kxx [W/mK] of the 4x4x4 runs. NONANALYTIC = 3 is the value of the code before the
# symmetrization (cedeeea9: 6.0401775101, 1.2e-9 away) and must not move; NONANALYTIC = 2
# moved from 6.0627378683 (kxy/kxx = -1.2e-2) to the value below. The tolerance allows
# for platform differences of the SCPH step.
REF_KXX = {2: 5.9942129, 3: 6.0401775}
REF_TOL = 1.0e-6


def kappa_input(prefix, na, coherent=False):
    text = """&general
 PREFIX = %s
 MODE = kappa
 KD = Sr Ti O
 FCSFILE = STO_anharm.xml
 DFC2FILE = sto_scph.scph.h5
 FC2_TEMPERATURE = 300
 NONANALYTIC = %d; BORNINFO = BORN
 TMIN = 300; TMAX = 300; DT = 50
/
""" % (prefix, na)
    text += CELL + "&kpoint\n2\n%d %d %d\n/\n" % (NMESH, NMESH, NMESH)
    if coherent:
        text += "&kappa\n KAPPA_COHERENT = 1\n/\n"
    return text


def run(cmd, prefix, text, env):
    with open(prefix + ".in", "w") as f:
        f.write(text)
    with open(prefix + ".log", "w") as f:
        proc = subprocess.run(
            cmd + [prefix + ".in"], stdout=f, stderr=subprocess.STDOUT, env=env
        )
    if proc.returncode != 0:
        print("anphon exited with code %d for %s" % (proc.returncode, prefix))
    return proc.returncode


def read_tensor(prefix, name):
    with h5py.File(prefix + ".kappa.h5", "r") as f:
        return np.array(f["kappa/" + name]).reshape(-1, 3, 3)[0]


def check_cubic(label, k, kxx_ref=None, tol=1.0e-10):
    """Finite, kxx > 0, all off-diagonal elements and diagonal differences < tol * kxx."""
    ok = bool(np.all(np.isfinite(k))) and k[0, 0] > 0.0
    if not ok:
        print("  %s: kappa not finite or kxx <= 0:\n%s" % (label, k))
        return False
    kxx = k[0, 0]
    off = max(abs(k[i, j]) for i in range(3) for j in range(3) if i != j) / kxx
    dia = max(abs(k[1, 1] - kxx), abs(k[2, 2] - kxx)) / kxx
    msg = "  %s: kxx = %.10f, max|k_ij|/kxx = %.2e, max|k_ii - kxx|/kxx = %.2e" % (
        label,
        kxx,
        off,
        dia,
    )
    ok = off < tol and dia < tol
    if kxx_ref is not None:
        rel = abs(kxx - kxx_ref) / kxx_ref
        msg += ", |kxx - ref|/ref = %.1e" % rel
        ok = ok and rel < REF_TOL
    print(msg + ("" if ok else "  <-- FAILED"))
    return ok


def make_p4mm(xml_in, h5_in, xml_out, h5_out, shift=0.005):
    """P4mm SrTiO3: Ti moved by shift (supercell fraction) along z in the FCS and DFC2
    files, the FC2 values kept (cubic, hence also P4mm symmetric), and an on-site term
    on Sr that violates the acoustic sum rule slightly but keeps the P4mm symmetry."""
    with open(xml_in) as f_in, open(xml_out, "w") as f_out:
        for line in f_in:
            m = re.match(
                r'(\s*<pos index="\d+" element="Ti">)\s*(\S+)\s+(\S+)\s+(\S+)(</pos>.*)',
                line,
            )
            if m:
                x, y, z = (float(m.group(i)) for i in (2, 3, 4))
                line = "%s %.15e %.15e %.15e%s\n" % (
                    m.group(1),
                    x,
                    y,
                    z + shift,
                    m.group(5),
                )
            f_out.write(line)
    shutil.copy(h5_in, h5_out)
    with h5py.File(h5_out, "r+") as f:
        for grp, d in (("PrimitiveCell", 2 * shift), ("SuperCell", shift)):
            x = f[grp + "/fractional_coordinate"][...]
            x[f[grp + "/atomic_kinds"][...] == 1, 2] += d  # kind 1 = Ti
            f[grp + "/fractional_coordinate"][...] = x
        g = f["ForceConstants/Order2"]
        ai, ci, sv = (
            g[n][...] for n in ("atom_indices", "coord_indices", "shift_vectors")
        )
        ti = f["SuperCell/atomic_kinds"][...][g["atom_indices_supercell"][...]] == 1
        sv[:, 2] += (
            shift
            * f["SuperCell/lattice_vector"][2, 2]
            * (ti[:, 1].astype(float) - ti[:, 0])
        )
        g["shift_vectors"][...] = sv
        v = f["ForceConstants/Order2_temperature_dependent/force_constant_values"]
        fc = v[...]
        onsite = (
            (ai[:, 0] == 0)
            & (ai[:, 1] == 0)
            & (ci[:, 0] == ci[:, 1])
            & (np.abs(sv).sum(1) < 1e-8)
        )
        for row in np.where(onsite)[0]:
            fc[0, row] += 0.5e-10 if ci[row, 0] == 2 else 3.5e-10  # Ry/bohr^2
        v[...] = fc


def check_p4mm(prefix):
    """Tetragonal kappa (kxx = kyy, no off-diagonal) and a finite coherent term."""
    ok = True
    for name, tol in (("kappa_peierls", 1.0e-8), ("kappa_coherent", 1.0e-7)):
        k = read_tensor(prefix, name)
        kxx = k[0, 0]
        good = bool(np.all(np.isfinite(k))) and 0.0 < kxx < 100.0
        off = max(abs(k[i, j]) for i in range(3) for j in range(3) if i != j) / kxx
        dia = abs(k[1, 1] - kxx) / kxx
        good = good and off < tol and dia < tol
        print(
            "  %s: kxx = %.10f, kzz = %.10f, max|k_ij|/kxx = %.2e, |kyy - kxx|/kxx = %.2e%s"
            % (name, kxx, k[2, 2], off, dia, "" if good else "  <-- FAILED")
        )
        ok &= good
    return ok


def check_gamma_translations(prefix):
    """At Gamma the three translational modes (the lowest three here) have zero
    velocity-matrix rows and columns: their coherent elements (KAPPA_COHERENT = 2
    record) and Peierls diad vanish, while optical elements survive."""
    kc = np.loadtxt(prefix + ".kc_elem")
    g = kc[kc[:, 5] == 1]  # ik_irred = 1 is Gamma
    trans = np.minimum(g[:, 3], g[:, 4]) <= 3
    worst = np.abs(g[trans][:, 8:10]).max()
    optical = np.abs(g[~trans][:, 8]).max()
    with h5py.File(prefix + ".kappa.h5", "r") as f:
        knum = np.array(f["scattering/3ph/equiv_knum"]).ravel()
        diad = np.array(f["scattering/3ph/velocity_diad"])
    diad = diad.reshape(len(knum), -1, 9)[list(knum).index(0)]
    ok = (
        worst == 0.0
        and optical > 0.0
        and np.abs(diad[:3]).max() == 0.0
        and np.abs(diad[3:]).max() > 0.0
    )
    print(
        "  Gamma: max |coherent element| with a translational mode = %.1e (optical-optical max %.1e), "
        "translational diad max %.1e, optical diad max %.1e%s"
        % (
            worst,
            optical,
            np.abs(diad[:3]).max(),
            np.abs(diad[3:]).max(),
            "" if ok else "  <-- FAILED",
        )
    )
    return ok


def cubic_ops():
    ops = []
    for p in itertools.permutations(range(3)):
        for s in itertools.product((1, -1), repeat=3):
            r = np.zeros((3, 3))
            for i in range(3):
                r[i, p[i]] = s[i]
            ops.append(r)
    return ops


def check_fd_velocities(prefix):
    """FD velocities v(k) must satisfy (+-R) v = v for every R of the little group of k."""
    with h5py.File(prefix + ".kappa.h5", "r") as f:
        g = f["scattering/3ph"]
        knum = np.array(g["equiv_knum"])
        vel = np.array(g["velocities"])
    vel = vel.reshape(len(knum), -1, 3)
    n = NMESH
    worst = 0.0
    for p, kn in enumerate(knum):
        kv = np.array([kn // (n * n), (kn // n) % n, kn % n], dtype=float) / n
        for r in cubic_ops():
            for sign in (1, -1):
                d = sign * (r @ kv) - kv
                if np.allclose(d - np.round(d), 0.0, atol=1e-8):
                    worst = max(worst, np.abs(vel[p] @ (sign * r).T - vel[p]).max())
    scale = np.abs(vel).max()
    ok = worst <= 1.0e-10 * scale
    print(
        "  finite-difference velocities: max |(+-R) v - v| = %.2e m/s (max |v| = %.1f)%s"
        % (worst, scale, "" if ok else "  <-- FAILED")
    )
    return ok


if __name__ == "__main__":
    build_dir = os.getcwd()
    project_root = os.path.dirname(build_dir)
    sto_dir = os.path.join(project_root, "example/SrTiO3/reference")
    anphonbin = os.path.join(project_root, "_build/anphon/anphon")

    workdir = os.path.join(project_root, "test/velsym")
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    for name in os.listdir("."):
        if name.endswith(".h5"):
            os.remove(
                name
            )  # stale state or result files would turn the runs into restarts

    shutil.copy(os.path.join(sto_dir, "BORN"), workdir)
    with bz2.open(os.path.join(sto_dir, "STO_anharm.xml.bz2"), "rb") as f_in:
        with open("STO_anharm.xml", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    env = dict(os.environ)
    serial = [anphonbin]
    failed = run(serial, "sto_scph", SCPH_INPUT, env)
    for prefix, na, coh in [("na2", 2, True), ("na3", 3, False)]:
        failed = failed or run(serial, prefix, kappa_input(prefix, na, coh), env)
    if failed:
        print("VELSYM --> failed")
        sys.exit(1)

    ok = True
    print("NONANALYTIC = 2:")
    ok &= check_cubic("Peierls", read_tensor("na2", "kappa_peierls"), REF_KXX[2])
    ok &= check_cubic("coherent", read_tensor("na2", "kappa_coherent"))
    ok &= check_fd_velocities("na2")
    for na in (3,):
        print("NONANALYTIC = %d (already symmetric):" % na)
        # 1e-9 off-diagonal here before and after: the finite-difference derivative of the
        # Ewald term near Gamma, at points whose little group is trivial.
        ok &= check_cubic(
            "Peierls",
            read_tensor("na%d" % na, "kappa_peierls"),
            REF_KXX[na],
            tol=1.0e-8,
        )

    print("P4mm, NONANALYTIC = 3, acoustic-sum-rule residual at Gamma:")
    make_p4mm("STO_anharm.xml", "sto_scph.scph.h5", "STO_p4mm.xml", "sto_p4mm.scph.h5")
    p4mm_input = (
        kappa_input("p4mm", 3, True)
        .replace("KAPPA_COHERENT = 1", "KAPPA_COHERENT = 2")
        .replace("STO_anharm.xml", "STO_p4mm.xml")
        .replace("sto_scph.scph.h5", "sto_p4mm.scph.h5")
    )
    if run(serial, "p4mm", p4mm_input, env):
        ok = False
    else:
        ok &= check_p4mm("p4mm")
        ok &= check_gamma_translations("p4mm")

    mpirun = shutil.which("mpirun")
    if mpirun is None:
        print("mpirun not found; skipping the 2-rank run")
    else:
        env_mpi = dict(env, OMP_NUM_THREADS="1")
        if run(
            [mpirun, "-np", "2", anphonbin],
            "na2_np2",
            kappa_input("na2_np2", 2, True),
            env_mpi,
        ):
            ok = False
        else:
            dk = 0.0
            for name in ("kappa_peierls", "kappa_coherent"):
                a, b = read_tensor("na2", name), read_tensor("na2_np2", name)
                dk = max(dk, np.abs(a - b).max() / abs(a[0, 0]))
            good = dk < 1.0e-10
            print(
                "2 MPI ranks vs 1: max |dkappa|/kxx = %.2e%s"
                % (dk, "" if good else "  <-- FAILED")
            )
            ok &= good

    if ok:
        print("VELSYM --> pass")
        sys.exit(0)
    print("VELSYM --> failed")
    sys.exit(1)
