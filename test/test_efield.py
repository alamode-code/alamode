#!/usr/bin/env python
"""Tests of the EFIELD tag (&relax): static electric field in the SCPH/QHA
structural optimization through the energy -E . sum_k Z*_k u0_k.

Stages:
  T2 harmonic : ZnO QHA relaxation at a fixed cell with FC2 only (the cubic and
                quartic IFCs scaled by 1e-8; anphon does not accept all-zero
                ones). Exact result q = zE / omega^2, i.e.
                Delta P = (1/Omega) Z^T Phi^+ Z E from the FC2 file, at 100 and
                300 K (T-independent).
  T3 energy   : BaTiO3 SCPH, RELAX_STR = 4 (sheared &strain), four prescribed
                structures (3 independent q0 differences), first step with and
                without a generic field. Checks V0_E - V0_0 = -E . sum Z* u0
                (Python) at each structure, g_E - g_0 identical at all of them,
                and (g_E - g_0) . (q_a - q_0) = Delta(V0_E - V0_0).
  T4 symmetry : field-preserving operation counts (E || z: 8, E || [111] from
                the cubic start: 6, z-displaced start with E || x: 2) and
                Delta P(-E) = -Delta P(E), with the same field energy and
                F_total, from the cubic start; the +z run on 4 MPI ranks
                reproduces the serial .polarization.
  T5 units    : PREFIX.polarization against (1/Omega) sum Z* u0 in uC/cm^2 and
                -E . sum Z* u0 from PREFIX.atom_disp / .umn_tensor.
  T6 errors   : missing BORNINFO, RELAX_STR = 3, malformed
                EFIELD, a restart with a different (or absent) EFIELD or Born charges, and a
                legacy-text restart, a BORNINFO with one Z* too few or too
                many or a nan entry, a left-handed &cell and a left-handed
                FCSFILE supercell under a right-handed &cell (needs a python
                with h5py) are rejected; a restart with the same EFIELD is
                accepted.
  seed        : tools/efield_seed.py output restarts the sweep from the
                relaxed structure (&displace / &strain).

Phase 1b (fixed-voltage model; needs a python with h5py and ase for
tools/strainfile.py: STRAINKIT_PYTHON, ~/miniforge3/envs/qmpy2, or the running
one; skipped otherwise). The BaTiO3 runs read the container bto.h5 packed from
strain_phonon (STRAINFILE, STRAIN_COUPLING = 25); bto_piezo.h5 adds a fake
generic clamped-ion e0 with strainfile.py piezo --voigt.
  V1 averaged : RELAX_STR = 4, sheared strain, one step, generic E: the generic
                e0 is averaged over Pm-3m (zero) with a warning; V0, the
                gradients and the field energy equal those without e0.
  V2 E = 0    : e0 + POL_REF leave the steps and V0 bit-identical; P equals
                F (d_ref + A + B) / (Omega_ref det F).
  V3 Maxwell  : d(dt_z)/du_mn |_E = -dG_mn/dE_z |_u for zz and xy (G_xy + G_yx)
                with tightly relaxed internal coordinates, two step sizes.
  1b errors   : POL_REF / e0 restart mismatches both ways, legacy text restarts,
                POL_REF without BORNINFO; the missing-e0 warning for ZnO.

Phase 2 (nonlinear clamped-ion terms; same python requirement): fake generic
B (second_order) and Lambda (born_charge_strain_derivative, ASR-consistent)
written with h5py into copies of the containers: bto_nl.h5 (e0 + B + Lambda),
bto_pB.h5 (e0 + B), bto_pL.h5 (e0 + Lambda), bto_lam.h5 (Lambda only, no
clamped_ion) and bto_nl_nest.h5 (bto_nl.h5 with /ReferenceCell doubled along
a1 and its atoms reversed).
  W1 identity : RELAX_STR = 4, sheared strain, one step, generic E, bto_nl vs
                bto_piezo: V0, the 9 strain gradients and the force against
                Python (|dg|^2 = sum |E.Lambda:u|^2/M and dg.q0), the symmetric
                shear energy finite difference; the nested/permuted reference
                cell is bit-identical; the Lambda-only container against bto.h5;
                mpirun -np 4 against serial.
  S symmetry  : anphon averages e0, B and Lambda over the reference space group
                (production path everywhere): an unsymmetrized Lambda reproduces
                W1's symmetric Lambda-only run, e0 and B vanish (Pm-3m) with a
                warning. The fake Lambda is built Pm-3m symmetric; the BaTiO3
                references use the averaged e0 and B (zero) under a field and
                the raw ones at E = 0.
  W2 Maxwell  : V3 with bto_nl.h5 (matched meshes, 5e-5).
  W3 BUBBLE=4 : BUBBLE_FD_CHECK = 2 on matched meshes at the first structure: the
                stress-displacement block equals the transposed force-strain
                block (< 1e-8), and the explicit blocks differ from the bto_piezo
                run by -Omega E.B (strain-strain) and -L (force-strain); the
                BUBBLE_HESS optimizer path accepts the curvature.
  W4 E = 0    : B and Lambda leave the steps, V0 and the gradients bit-identical;
                dt changes by Omega B:u:u/2 + sum (Lambda:u) u0.
  2 restarts  : B / Lambda restart mismatches both ways, legacy text restarts
                (SCPH and QHA), the same data accepted.
  ZSISA       : QHA, QHA_SCHEME = 2, one step: the renormalized elastic constants
                gain -Omega E.B, dv1/du gains -L (invariants), and dv1/du is the
                central difference of the static force over u_zz and u_yz.

ZnO (Z; P6_3mc, symmetry-allowed nonzero e0, B and Lambda, production path):
  RELAX_STR = 4 (STRAIN_COUPLING = 0), one step, sheared strain, generic E: dt,
  V0 and the 9 strain gradients against Python, the symmetric shear finite
  difference of the field energy, raw data == their space-group average, a
  nested/permuted /ReferenceCell bit-identical, mpirun -np 4 == serial, and a
  restart without B rejected / with the raw container accepted.

Run from the build directory: python ../test/test_efield.py (numpy required)
"""

import os
import re
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np

BOHR = 0.52917721092  # Angstrom
RYD_EV = 13.605693122994
E_CHARGE = 1.6021766208e-19
AMU_RY = 1.660538782e-27 / 9.10938215e-31 / 2.0
EBOHR2_TO_UC_CM2 = E_CHARGE / (BOHR * 1.0e-8) ** 2 * 1.0e6
EV_A_TO_RY = BOHR / RYD_EV  # eV/Angstrom -> Ry/(e Bohr)

BTO_TEMPLATE = """&general
  PREFIX = {prefix}
  MODE = {mode}
  FCSFILE = cBTO222.h5
  {born}
  TMIN = 300; TMAX = 300
  VERBOSITY = {verbosity}
  ALLOW_UNCONVERGED = 1
/
&{block}
  SELF_OFFDIAG = 1
  {scf}
  KMESH_INTERPOLATE = 2 2 2
  {kmesh} = 4 4 4
  RELAX_STR = {relax_str}
  {restart}
/
&relax
  RELAX_ALGO = 3
  MAX_STR_ITER = {max_iter}
  COORD_CONV_TOL = {coord_tol}
  CELL_CONV_TOL = 5.0e-7
  SET_INIT_STR = 1
  ADD_HESS_DIAG = 50.0
  STRAIN_COUPLING = 25
  {strain_source}
  {efield}
  {pol_ref}
/
&strain
{strain}
/
{displace}
&kpoint
  2
  8 8 8
/
"""

STRAIN_ZERO = "0.0 0.0 0.0\n0.0 0.0 0.0\n0.0 0.0 0.0"
DISPLACE_Z = """&displace
1
0.0 0.0 1.0e-4
0.0 0.0 2.0e-2
0.0 0.0 -1.5e-2
0.0 0.0 -1.5e-2
0.0 0.0 -3.0e-2
/"""


# STRAINFILE = bto.h5 once the container is built (main), else the text files
STRAIN_SOURCE = ["STRAIN_IFC_DIR = strain_phonon"]


def bto_input(
    prefix,
    efield=None,
    relax_str=2,
    max_iter=1,
    displace="",
    strain=STRAIN_ZERO,
    verbosity=1,
    born=True,
    mode="SCPH",
    restart=False,
    strainfile=None,
    pol_ref=None,
    extra="",
    coord_tol="1.0e-5",
):
    scph = mode == "SCPH"
    return BTO_TEMPLATE.format(
        prefix=prefix,
        mode=mode,
        born="BORNINFO = BORNINFO" if born else "",
        verbosity=verbosity,
        block="scph" if scph else "qha",
        scf="MAXITER = 500; MIXALPHA = 0.2" if scph else "",
        kmesh="KMESH_SCPH" if scph else "KMESH_QHA",
        relax_str=relax_str,
        restart=("RESTART_SCPH = 1" if scph else "RESTART_QHA = 1") if restart else "",
        max_iter=max_iter,
        coord_tol=coord_tol,
        efield=("" if efield is None else "EFIELD = " + efield) + extra,
        pol_ref="" if pol_ref is None else "POL_REF = " + pol_ref,
        strain_source=STRAIN_SOURCE[0]
        if strainfile is None
        else "STRAINFILE = " + strainfile,
        strain=strain,
        displace=displace,
    )


def run_anphon(anphonbin, prefix, text, nprocs=1):
    """Write prefix.in, run it, return (returncode, log text)."""
    with open(prefix + ".in", "w") as f:
        f.write(text)
    cmd = [anphonbin, prefix + ".in"]
    env = None
    if nprocs > 1:
        cmd = ["mpirun", "-np", str(nprocs)] + cmd
        env = dict(os.environ, OMP_NUM_THREADS="1")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    with open(prefix + ".log", "w") as f:
        f.write(proc.stdout + proc.stderr)
    return proc.returncode, proc.stdout


def read_born(filename):
    z = np.loadtxt(filename)[3:].reshape(-1, 3, 3)
    return z - z.mean(axis=0)  # the ASR correction applied by anphon


def n_field_ops(log):
    m = re.search(r"EFIELD: (\d+) operations", log)
    return int(m.group(1)) if m else -1


def volume_ref(log):
    return float(re.search(r"Volume of the primitive cell : (\S+)", log).group(1))


def masses_ry(log, kinds):
    block = log.split("Mass of atomic species (u):")[1]
    mass = dict(re.findall(r"^\s+(\w+):\s+(\S+)", block, re.M)[: len(set(kinds))])
    return np.array([float(mass[k]) for k in kinds]) * AMU_RY


def step_row(filename, row):
    return np.loadtxt(filename, skiprows=2, ndmin=2)[row, 1:]


def close(a, b, rel=1.0e-5, absol=1.0e-12):
    return abs(a - b) <= max(rel * max(abs(a), abs(b)), absol)


def check(cond, msg):
    print("  %-6s %s" % ("ok" if cond else "FAILED", msg))
    return 0 if cond else 1


# ---------------------------------------------------------------------------


def test_t2(anphonbin, zno_dir):
    for f in ("ZnO442_harmonic.xml.zip", "ZnO332_500K_cutoff1208_nbody233.xml.zip"):
        with zipfile.ZipFile(os.path.join(zno_dir, f)) as z:
            z.extractall(".")
    shutil.copy(os.path.join(zno_dir, "BORNINFO"), "BORNINFO_zno")
    s = open("ZnO332_500K_cutoff1208_nbody233.xml").read()
    s = re.sub(
        r"(<FC[34] [^>]*>)([^<]*)(</FC[34]>)",
        lambda m: m.group(1) + "%.15e" % (float(m.group(2)) * 1.0e-8) + m.group(3),
        s,
    )
    with open("zno_tiny_anharm.xml", "w") as f:
        f.write(s)

    efield = np.array([0.01, 0.0, 0.01])
    alat = 1.88972612462577
    cell = np.array(
        [
            [3.235859326375770, 0.0, 0.0],
            [-1.617929663187880, 2.802336379714220, 0.0],
            [0.0, 0.0, 5.224712025937350],
        ]
    )
    text = """&general
  PREFIX = zno_t2
  MODE = QHA
  FCSFILE = zno_tiny_anharm.xml
  FC2FILE = ZnO442_harmonic.xml
  BORNINFO = BORNINFO_zno
  TMIN = 100; TMAX = 300; DT = 200
/
&qha
  KMESH_INTERPOLATE = 4 4 2
  KMESH_QHA = 4 4 2
  RELAX_STR = 1
/
&relax
  RELAX_ALGO = 2
  MAX_STR_ITER = 20
  COORD_CONV_TOL = 1.0e-10
  MIXBETA_COORD = 1.0
  ADD_HESS_DIAG = 0.0
  EFIELD = %s
/
&cell
  %.14f
%s
/
&kpoint
  2
  8 8 8
/
""" % (
        " ".join(str(x) for x in efield),
        alat,
        "\n".join("  %.15f %.15f %.15f" % tuple(r) for r in cell),
    )
    rc, _ = run_anphon(anphonbin, "zno_t2", text)
    if rc != 0:
        return check(False, "T2: anphon failed")

    # Gamma force-constant matrix from the FC2 file (entries of all images summed)
    x = open("ZnO442_harmonic.xml").read()
    s2p = {
        int(b): int(a)
        for _, a, b in re.findall(r'<map tran="(\d+)" atom="(\d+)">(\d+)</map>', x)
    }
    phi = np.zeros((12, 12))
    for a, i, b, j, _, v in re.findall(
        r'<FC2 pair1="(\d+) (\d+)" pair2="(\d+) (\d+) (\d+)">([^<]+)</FC2>', x
    ):
        phi[3 * (int(a) - 1) + int(i) - 1, 3 * (s2p[int(b)] - 1) + int(j) - 1] += float(
            v
        )
    phi = 0.5 * (phi + phi.T)
    zstar = read_born("BORNINFO_zno")
    e_ry = efield * EV_A_TO_RY
    force = np.einsum("kab,a->kb", zstar, e_ry).ravel()
    w, v = np.linalg.eigh(phi)
    keep = np.abs(w) > 1.0e-8  # drop the acoustic (translation) modes
    u = (v[:, keep] / w[keep]) @ (v[:, keep].T @ force)
    dipole = np.einsum("kab,kb->a", zstar, u.reshape(-1, 3))
    omega = abs(np.linalg.det(alat * cell))
    p_ref = dipole / omega * EBOHR2_TO_UC_CM2
    e_ref = -e_ry @ dipole

    rows = np.loadtxt("zno_t2.polarization", ndmin=2)
    info = check(
        len(rows) == 2 and np.all(rows[:, 8] == 1),
        "T2: relaxation converged at 100 and 300 K",
    )
    info += check(
        np.allclose(rows[0, 1:8], rows[1, 1:8], rtol=1.0e-6, atol=1.0e-8),
        "T2: Delta P and field energy are T-independent: %s / %s"
        % (rows[0, 1:4], rows[1, 1:4]),
    )
    row = rows[-1]
    info += check(
        all(close(a, b, 1.0e-5, 1.0e-8) for a, b in zip(row[1:4], p_ref)),
        "T2: Delta P %s == (1/Omega) Z^T Phi^+ Z E %s" % (row[1:4], p_ref),
    )
    info += check(
        close(row[7], e_ref, 1.0e-5), "T2: field energy %.9e == %.9e" % (row[7], e_ref)
    )
    return info


def parse_v0_grad(log):
    v0 = float(re.search(r"V0 at this structure \[Ry\] =\s*(\S+)", log).group(1))
    grad = np.array(
        re.search(r"gradient dF/dq0 \(all Gamma modes\) :(.*)", log).group(1).split(),
        float,
    )
    return v0, grad


def test_t3(anphonbin):
    # Four prescribed structures (base + 3 independent displacement patterns) at
    # a fixed sheared strain, each evaluated once (MAX_STR_ITER = 1) with and
    # without a generic field. ponytail: 3 directions of the 12-dim optical
    # space, not a full span (26 runs); extend STRUCTURES if needed.
    base = np.array(
        [
            [0.003, -0.002, 0.004],
            [0.010, 0.006, 0.020],
            [-0.004, 0.003, -0.015],
            [0.002, -0.005, -0.012],
            [-0.006, 0.001, -0.030],
        ]
    )
    deltas = [np.zeros((5, 3))] + [np.zeros((5, 3)) for _ in range(3)]
    deltas[1][1] = [0.008, 0.0, -0.004]
    deltas[2][2] = [0.0, 0.006, 0.005]
    deltas[3][4] = [-0.005, 0.004, 0.0]
    deltas[3][0] = [0.0, 0.0, 0.003]
    strain = "0.002 0.001 0.0\n0.001 0.002 0.0\n0.0 0.0 0.004"
    efield = np.array([0.004, -0.003, 0.01])
    e_ry = efield * EV_A_TO_RY
    zstar = read_born("BORNINFO")
    force = np.einsum("kab,a->kb", zstar, e_ry).ravel()  # F = Z^T E [Ry/Bohr]

    info = 0
    q, u, dv, dg = [], [], [], []
    for ia, delta in enumerate(deltas):
        disp = "&displace\n1\n%s\n/" % "\n".join(
            " %.6f %.6f %.6f" % tuple(r) for r in base + delta
        )
        common = dict(
            relax_str=4, max_iter=1, displace=disp, strain=strain, verbosity=2
        )
        rc0, log0 = run_anphon(
            anphonbin, "t3_e0_%d" % ia, bto_input("t3_e0_%d" % ia, **common)
        )
        q_a = step_row("step_q0.txt", 0)
        rc1, log1 = run_anphon(
            anphonbin,
            "t3_e_%d" % ia,
            bto_input("t3_e_%d" % ia, efield=" ".join(map(str, efield)), **common),
        )
        if rc0 or rc1 or not np.allclose(step_row("step_q0.txt", 0), q_a):
            return check(
                False, "T3: anphon failed or the runs start from different structures"
            )
        v0_0, g_0 = parse_v0_grad(log0)
        v0_e, g_e = parse_v0_grad(log1)
        q.append(q_a)
        u.append(step_row("step_u0.txt", 0))
        dv.append(v0_e - v0_0)
        dg.append(g_e - g_0)
        de_ref = -force @ u[-1]
        info += check(
            close(dv[-1], de_ref),
            "T3[%d]: V0_E - V0_0 = %.9e == -E.sum Z*u0 = %.9e" % (ia, dv[-1], de_ref),
        )

    # Zero field with BORNINFO: .polarization is still written (spontaneous P),
    # at the evaluated (initial) structure: P = F B / (Omega_ref det F)
    log0 = open("t3_e0_0.log").read()
    row = np.loadtxt("t3_e0_0.polarization", ndmin=2)[-1]
    umn = np.array([float(x) for x in strain.split()]).reshape(3, 3)
    fmat = np.eye(3) + umn
    omega = volume_ref(log0) * np.linalg.det(fmat)
    p_ref = (
        fmat
        @ np.einsum("kab,kb->a", zstar, u[0].reshape(-1, 3))
        / omega
        * EBOHR2_TO_UC_CM2
    )
    info += check(
        np.allclose(row[1:4], p_ref, rtol=1.0e-6, atol=1.0e-10)
        and np.array_equal(row[1:4], row[4:7])
        and row[7] == 0.0,
        "T3: E = 0 .polarization Delta P %s == (1/Omega) Z* u0 %s, field energy 0"
        % (row[1:4], p_ref),
    )

    dg = np.array(dg)
    q = np.array(q)
    scale = np.abs(dg).max()
    info += check(
        np.abs(dg - dg[0]).max() <= 1.0e-9 * scale,
        "T3: g_E - g_0 is the same vector at all structures (max dev %.1e of %.1e)"
        % (np.abs(dg - dg[0]).max(), scale),
    )
    mass = masses_ry(log1, ["Ba", "Ti", "O", "O", "O"])
    norm_ref = np.sum(force.reshape(-1, 3) ** 2 / mass[:, None])
    info += check(
        close(dg[0] @ dg[0], norm_ref, 1.0e-8),
        "T3: |g_E - g_0|^2 = %.9e == sum |Z^T E|^2/M = %.9e"
        % (dg[0] @ dg[0], norm_ref),
    )
    dq = q[1:] - q[0]
    info += check(
        np.linalg.matrix_rank(dq, tol=1.0e-4 * np.abs(dq).max()) == 3,
        "T3: the three q0 differences are independent",
    )
    for ia in range(1, len(q)):
        lhs = dg[0] @ (q[ia] - q[0])
        rhs = dv[ia] - dv[0]
        info += check(
            close(lhs, rhs, 1.0e-4),
            "T3: (g_E - g_0).(q_%d - q_0) = %.9e == Delta(V0_E - V0_0) = %.9e"
            % (ia, lhs, rhs),
        )
    return info


def test_t4_t5(anphonbin, seed_script):
    info = 0
    rc, log = run_anphon(
        anphonbin, "t4_111", bto_input("t4_111", efield="0.01 0.01 0.01")
    )
    info += check(
        rc == 0 and n_field_ops(log) == 6,
        "T4: E || [111], cubic start: %d ops (6)" % n_field_ops(log),
    )
    rc, log = run_anphon(
        anphonbin, "t4_x", bto_input("t4_x", efield="0.01 0 0", displace=DISPLACE_Z)
    )
    info += check(
        rc == 0 and n_field_ops(log) == 2,
        "T4: E || x, z-displaced: %d ops (2)" % n_field_ops(log),
    )

    pol, ftot, evaluated = {}, {}, {}
    for name, ez in (("t4_mz", "-0.01"), ("t4_pz", "0.01")):
        rc, log = run_anphon(
            anphonbin, name, bto_input(name, efield="0 0 " + ez, max_iter=40)
        )
        info += check(
            rc == 0 and n_field_ops(log) == 8,
            "T4: E || %sz, cubic start: %d ops (8)"
            % ("-" if ez[0] == "-" else "+", n_field_ops(log)),
        )
        pol[name] = np.loadtxt(name + ".polarization", ndmin=2)[-1]
        ftot[name] = np.loadtxt(name + ".scph_thermo", ndmin=2)[-1, 5]
        # the .polarization row is the last evaluated structure: the step
        # before the last optimizer update
        evaluated[name] = (
            step_row("step_u0.txt", -2),
            step_row("step_u_tensor.txt", -2),
        )
    pp, pm = pol["t4_pz"], pol["t4_mz"]
    info += check(pp[8] == 1 and pm[8] == 1, "T4: both +E and -E runs converged")
    info += check(
        pp[3] > 1.0e-3
        and np.allclose(pp[1:4], -pm[1:4], rtol=1.0e-6, atol=1.0e-8)
        and close(pp[7], pm[7], 1.0e-6)
        and close(ftot["t4_pz"], ftot["t4_mz"], 1.0e-6),
        "T4: Delta P(+E) = %s, Delta P(-E) = %s, same field energy and F_total %.7e / %.7e"
        % (pp[1:4], pm[1:4], ftot["t4_pz"], ftot["t4_mz"]),
    )

    # MPI: the +z run on 4 ranks must reproduce the serial .polarization
    if shutil.which("mpirun") is None:
        print("  skip   T4: mpirun not found, MPI comparison skipped")
    else:
        rc, _ = run_anphon(
            anphonbin,
            "t4_pz_mpi",
            bto_input("t4_pz_mpi", efield="0 0 0.01", max_iter=40),
            nprocs=4,
        )
        row = np.loadtxt("t4_pz_mpi.polarization", ndmin=2)[-1] if rc == 0 else None
        info += check(
            row is not None and np.allclose(row, pp, rtol=1.0e-8, atol=1.0e-8),
            "T4: mpirun -np 4 .polarization == serial",
        )

    # T5 on the +z run, at its last evaluated structure
    log = open("t4_pz.log").read()
    zstar = read_born("BORNINFO")
    u0_eval, umn_eval = evaluated["t4_pz"]
    fmat = np.eye(3) + umn_eval.reshape(3, 3)
    omega = volume_ref(log) * np.linalg.det(fmat)
    dipole = np.einsum("kab,kb->a", zstar, u0_eval.reshape(-1, 3))
    p_ref = fmat @ dipole / omega * EBOHR2_TO_UC_CM2
    e_ref = -np.array([0.0, 0.0, 0.01]) * EV_A_TO_RY @ dipole
    info += check(
        np.allclose(pp[1:4], p_ref, rtol=1.0e-6, atol=1.0e-10),
        "T5: P %s == F Z* u0 / (Omega_ref det F) %s" % (pp[1:4], p_ref),
    )
    info += check(
        close(pp[7], e_ref, 1.0e-6),
        "T5: field energy %.9e == -E.sum Z*u0 %.9e" % (pp[7], e_ref),
    )
    u0 = np.loadtxt("t4_pz.atom_disp", ndmin=2)[-1, 1:]
    umn = np.loadtxt("t4_pz.umn_tensor", ndmin=2)[-1, 1:].reshape(3, 3)

    # tools/efield_seed.py: the next run of a sweep starts from this structure
    seed = subprocess.run(
        [sys.executable, seed_script, "t4_pz"], capture_output=True, text=True
    ).stdout
    disp = seed[seed.index("&displace") : seed.index("&strain")]
    strain = seed[seed.index("&strain") + len("&strain") : seed.rindex("/")].strip()
    rc, _ = run_anphon(
        anphonbin,
        "seed",
        bto_input("seed", efield="0 0 0.02", displace=disp, strain=strain),
    )
    info += check(
        rc == 0
        and np.allclose(step_row("step_u0.txt", 0), u0, atol=1.0e-9)
        and np.allclose(step_row("step_u_tensor.txt", 0), umn.ravel(), atol=1.0e-12),
        "seed: efield_seed.py reproduces the relaxed structure as the next start",
    )
    return info


def test_t6(anphonbin):
    cases = [
        (
            "t6_noborn",
            bto_input("t6_noborn", efield="0 0 0.01", born=False),
            "BORNINFO must be set",
        ),
        (
            "t6_pqha",
            bto_input("t6_pqha", efield="0 0 0.01", relax_str=3, mode="QHA"),
            "perturbative QHA",
        ),
        ("t6_twoval", bto_input("t6_twoval", efield="0 0.01"), "exactly three entries"),
        ("t6_nan", bto_input("t6_nan", efield="0 nan 0"), "finite numbers"),
        # restarts from the state file of the +z run of T4
        (
            "t4_pz (restart, EFIELD = 0 0 0.02)",
            bto_input("t4_pz", efield="0 0 0.02", restart=True),
            "EFIELD tag is not consistent",
        ),
        (
            "t4_pz (restart, no EFIELD)",
            bto_input("t4_pz", restart=True),
            "EFIELD tag is not consistent",
        ),
        (
            "t4_pz (restart, other Born charges)",
            bto_input("t4_pz", efield="0 0 0.01", restart=True).replace(
                "= BORNINFO", "= BORNINFO_mod"
            ),
            "Born effective charges (BORNINFO) are not consistent",
        ),
        (
            "t6_text (restart, FILE_FORMAT = text)",
            bto_input("t6_text", efield="0 0 0.01", restart=True).replace(
                "  VERBOSITY", "  FILE_FORMAT = text\n  VERBOSITY"
            ),
            "legacy text restart files",
        ),
        # BORNINFO with one Z* too few / too many for the 5-atom cell
        (
            "t6_born_short",
            bto_input("t6_born_short", efield="0 0 0.01").replace(
                "= BORNINFO", "= BORNINFO_short"
            ),
            "4 Born effective charge tensors; too few",
        ),
        (
            "t6_born_long",
            bto_input("t6_born_long", efield="0 0 0.01").replace(
                "= BORNINFO", "= BORNINFO_long"
            ),
            "6 Born effective charge tensors; too many",
        ),
        # left-handed &cell basis (a2 and a3 swapped)
        (
            "t6_lefthanded",
            bto_input("t6_lefthanded", efield="0 0 0.01").replace(
                "&kpoint", "&cell\n  7.6\n  1 0 0\n  0 0 1\n  0 1 0\n/\n&kpoint"
            ),
            "left-handed",
        ),
        # BORNINFO with a non-finite entry
        (
            "t6_born_nan",
            bto_input("t6_born_nan", efield="0 0 0.01").replace(
                "= BORNINFO", "= BORNINFO_nan"
            ),
            "is not a finite number",
        ),
    ]
    born_nan = np.loadtxt("BORNINFO")
    born_nan[3, 0] = np.nan
    np.savetxt("BORNINFO_nan", born_nan, fmt="%16.8f")
    # FCSFILE supercell with a3 reversed (left-handed) under a right-handed &cell
    py = find_h5py_python()
    if py is None:
        print("  skip   T6: left-handed FCSFILE supercell (no python with h5py)")
    else:
        shutil.copy("cBTO222.h5", "cBTO222_lh.h5")
        subprocess.run(
            [
                py,
                "-c",
                "import h5py\n"
                "with h5py.File('cBTO222_lh.h5', 'r+') as f:\n"
                "    lat = f['SuperCell/lattice_vector']; v = lat[()]; v[2, 2] = -v[2, 2]; lat[...] = v\n"
                "    xf = f['SuperCell/fractional_coordinate']; x = xf[()]\n"
                "    x[:, 2] = (1.0 - x[:, 2]) % 1.0; xf[...] = x\n",
            ],
            check=True,
        )
        cases.append(
            (
                "t6_lh_fcs",
                bto_input("t6_lh_fcs", efield="0 0 0.01")
                .replace("= cBTO222.h5", "= cBTO222_lh.h5")
                .replace(
                    "&kpoint",
                    "&cell\n  7.53159676409\n  1 0 0\n  0 1 0\n  0 0 1\n/\n&kpoint",
                ),
                "supercell lattice of FCSFILE is left-handed",
            )
        )
    # diag(-d, 0, +d) added to Ti and subtracted from Ba: the acoustic sum rule
    # and any linear checksum weighted by the flat index are unchanged
    born = np.loadtxt("BORNINFO")
    d = 0.01
    born[6, 0] -= d
    born[8, 2] += d
    born[3, 0] += d
    born[5, 2] -= d
    np.savetxt("BORNINFO_mod", born, fmt="%16.8f")
    np.savetxt("BORNINFO_short", born[:-3], fmt="%16.8f")
    np.savetxt("BORNINFO_long", np.vstack([born, born[-3:]]), fmt="%16.8f")
    info = 0
    for name, text, message in cases:
        rc, log = run_anphon(anphonbin, name.split()[0], text)
        info += check(
            rc != 0 and message in log, "T6: %s rejected (%s)" % (name, message)
        )
    rc, log = run_anphon(
        anphonbin, "t4_pz", bto_input("t4_pz", efield="0 0 0.01", restart=True)
    )
    info += check(
        rc == 0 and "RESTART_SCPH is true" in log,
        "T6: restart with the same EFIELD accepted",
    )
    return info


# ------------------------------------------------------------ phase 1b
# Fake generic clamped-ion e0 [C/m^2], Voigt xx yy zz yz xz xy, all nonzero
# (symmetry is irrelevant for the identity checks).
PIEZO_FAKE = np.array(
    [
        [0.31, -0.12, 0.08, 0.21, -0.17, 0.05],
        [-0.07, 0.26, -0.14, 0.09, 0.33, -0.11],
        [0.19, 0.04, -0.41, -0.06, 0.13, 0.28],
    ]
)
VOIGT_PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
C_M2_PER_E_BOHR2 = E_CHARGE / (BOHR * 1.0e-10) ** 2
POL_REF = "0.05 -0.02 0.26"  # C/m^2
OMEGA_REF = [None]  # bohr^3, from cBTO222.h5 (build_containers)
STRAIN_SHEAR = "0.002 0.0015 -0.001\n0.0015 -0.001 0.0012\n-0.001 0.0012 0.003"
DISPLACE_1B = """&displace
1
 0.003 -0.002 0.004
 0.010 0.006 0.020
 -0.004 0.003 -0.015
 0.002 -0.005 -0.012
 -0.006 0.001 -0.030
/"""


def piezo_e_bohr2(field=False):
    """The fake e0 [e/Bohr^2]; with ``field`` as anphon uses it under EFIELD (its
    Pm-3m average, zero)."""
    e = np.zeros((3, 3, 3))
    for c, (j, k) in enumerate(VOIGT_PAIRS):
        e[:, j, k] = e[:, k, j] = PIEZO_FAKE[:, c]
    e /= C_M2_PER_E_BOHR2
    return symmetrize_bto(e, per_atom=False) if field else e


def find_h5py_python():
    """A python with h5py for the strainkit tools, or None."""
    for py in (
        os.environ.get("STRAINKIT_PYTHON"),
        os.path.expanduser("~/miniforge3/envs/qmpy2/bin/python"),
        sys.executable,
    ):
        if py and os.path.exists(py):
            rc = subprocess.run(
                [py, "-c", "import h5py, ase, spglib"], capture_output=True
            )
            if rc.returncode == 0:
                return py
    return None


def build_containers(py, tools_dir):
    """bto.h5 packed from strain_phonon, and bto_piezo.h5 = bto.h5 + PIEZO_FAKE.
    Raises RuntimeError with the stderr of a failing step."""
    sfile = os.path.join(tools_dir, "strainfile.py")

    def run(cmd):
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError("%s failed:\n%s" % (" ".join(cmd[:3]), res.stderr))
        return res.stdout

    run(
        [py, sfile, "pack", "--strain-ifc-dir", "strain_phonon"]
        + ["--fcs", "cBTO222.h5", "-o", "bto.h5", "--force"]
    )
    run(
        [
            py,
            "-c",
            "import sys, h5py; sys.path.insert(0, %r)\n"
            "from strainkit import strainfile as sf\n"
            "c = sf.read_cell_group(h5py.File('bto.h5', 'r')['ReferenceCell'])\n"
            "import ase, ase.io\n"
            "ase.io.write('POSCAR_bto', ase.Atoms(c.elements, scaled_positions=c.xf,"
            " cell=c.lavec, pbc=True), format='vasp', direct=True)" % tools_dir,
        ]
    )
    # the reference volume at full precision (the log prints 7 digits)
    code = "import h5py, numpy as np; f = h5py.File('cBTO222.h5', 'r'); "
    code += (
        "print(repr(float(abs(np.linalg.det(f['PrimitiveCell/lattice_vector'][()])))))"
    )
    OMEGA_REF[0] = float(run([py, "-c", code]))
    shutil.copy("bto.h5", "bto_piezo.h5")
    np.savetxt("piezo_fake.txt", PIEZO_FAKE)
    run(
        [py, sfile, "piezo", "bto_piezo.h5", "--voigt", "piezo_fake.txt"]
        + ["--structure", "POSCAR_bto"]
    )


def last_values(log, label):
    return np.array(re.findall(re.escape(label) + r"(.*)", log)[-1].split(), float)


LBL_G = "strain gradient dF/du_mn [Ry] :"
LBL_DT = "dipole dt = Omega_ref (e0:u + B2:u:u/2) + sum_k (Z*_k + Lambda_k:u) u0_k [e Bohr] ="
LBL_V0 = "V0 at this structure [Ry] ="


def test_v1(anphonbin):
    # Fixed sheared strain, one step, generic E and generic e0: under the field
    # anphon averages e0 over Pm-3m of the reference, which makes it zero (odd
    # rank, centrosymmetric), with a warning; V0, the gradients and the field
    # energy equal those without e0.
    efield = np.array([0.004, -0.003, 0.01])
    common = dict(
        relax_str=4,
        max_iter=1,
        displace=DISPLACE_1B,
        strain=STRAIN_SHEAR,
        verbosity=2,
        efield=" ".join(map(str, efield)),
    )
    rc0, log0 = run_anphon(
        anphonbin, "v1_0", bto_input("v1_0", strainfile="bto.h5", **common)
    )
    q_0 = step_row("step_q0.txt", 0)
    rc1, log1 = run_anphon(
        anphonbin, "v1_p", bto_input("v1_p", strainfile="bto_piezo.h5", **common)
    )
    if rc0 or rc1 or not np.array_equal(step_row("step_q0.txt", 0), q_0):
        return check(False, "V1: anphon failed or the runs start differently")
    info = check(
        "may be missing" not in log0,
        "V1: no missing-e0 warning for the centrosymmetric BaTiO3 reference",
    )
    info += check(
        np.abs(piezo_e_bohr2(field=True)).max() < 1.0e-15
        and re.search(r"^ e0 changes by 100.0% of its maximum", log1, re.M),
        "V1: the generic e0 is averaged to zero over Pm-3m, with the warning",
    )
    dv = last_values(log1, LBL_V0)[0] - last_values(log0, LBL_V0)[0]
    g0 = last_values(log0, LBL_G)
    dg = last_values(log1, LBL_G) - g0
    r0 = np.loadtxt("v1_0.polarization", ndmin=2)[-1]
    r1 = np.loadtxt("v1_p.polarization", ndmin=2)[-1]
    info += check(
        abs(dv) <= 1.0e-12 * abs(last_values(log0, LBL_V0)[0])
        and np.abs(dg).max() <= 1.0e-12 * np.abs(g0).max()
        and abs(r1[7] - r0[7]) <= 1.0e-12 * abs(r0[7]),
        "V1: V0, G and the field energy with the averaged e0 == without e0 "
        "(dV0 %.1e, max dG %.1e)" % (dv, np.abs(dg).max()),
    )
    return info


def run_nl_script(py, *args):
    """Copy a container and add the fake B / Lambda (see NL_SCRIPT)."""
    res = subprocess.run(
        [py, "-c", NL_SCRIPT] + list(args), capture_output=True, text=True
    )
    if res.returncode != 0:
        raise RuntimeError("adding B/Lambda failed:\n%s" % res.stderr)


# ------------------------------------------------------------ phase 2
def fake_nl(natom=5):
    """Generic B [C/m^2] (symmetric in jk, lm, jk <-> lm) and Lambda [e]
    (symmetric in mn, sum_k Lambda_k = 0)."""
    rng = np.random.default_rng(20260930)
    b = rng.uniform(-0.3, 0.3, (3, 3, 3, 3, 3))
    for axes in ((0, 2, 1, 3, 4), (0, 1, 2, 4, 3), (0, 3, 4, 1, 2)):
        b = 0.5 * (b + b.transpose(axes))
    lam = rng.uniform(-2.0, 2.0, (natom, 3, 3, 3, 3))
    lam = 0.5 * (lam + lam.transpose(0, 1, 2, 4, 3))
    return b, lam - lam.mean(axis=0)


# Atoms of the BaTiO3 reference (cBTO222.h5 PrimitiveCell, cubic, Ba at the origin)
BTO_XF = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ]
)
BTO_KINDS = [0, 1, 2, 2, 2]


def cubic_ops():
    """Pm-3m of the BaTiO3 reference: the 48 signed permutation matrices about Ba
    (symmorphic) with their atom permutations."""
    import itertools

    ops = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            r = np.zeros((3, 3))
            r[range(3), perm] = signs
            d = (BTO_XF @ r.T)[:, None, :] - BTO_XF[None, :, :]
            d -= np.round(d)
            hit = np.abs(d).max(axis=2) < 1.0e-8
            p = [
                next(j for j in range(5) if hit[k, j] and BTO_KINDS[j] == BTO_KINDS[k])
                for k in range(5)
            ]
            ops.append((r, p))
    return ops


def symmetrize_bto(t, per_atom=True):
    """Average over Pm-3m, the convention of anphon (Relaxation::load_piezo): per
    atom Lambda'_P(k) = R R R R Lambda_k, or a global tensor R...R t."""
    out = np.zeros_like(t)
    for r, p in cubic_ops():
        rt = t
        for ax in range(1 if per_atom else 0, t.ndim):
            rt = np.moveaxis(np.tensordot(r, rt, axes=(1, ax)), 0, ax)
        if per_atom:
            out[p] += rt
        else:
            out += rt
    return out / 48.0


# The fake Lambda is built Pm-3m symmetric, so anphon's space-group average leaves it
# unchanged.  The fake e0 and B are generic: under a field anphon averages them to
# E0_EFF and B_EFF (zero: odd rank, centrosymmetric reference); at E = 0 the raw
# data are used.  Their nonzero field terms are tested on ZnO (stage Z).
B_FAKE, LAMBDA_RAW = fake_nl()
LAMBDA_FAKE = symmetrize_bto(LAMBDA_RAW)
B_EFF = symmetrize_bto(B_FAKE, per_atom=False)


PAIRS = [
    [(0, 0)],
    [(1, 1)],
    [(2, 2)],
    [(1, 2), (2, 1)],
    [(2, 0), (0, 2)],
    [(0, 1), (1, 0)],
]

# argv: src dst what; what contains B (second_order), L (Lambda), N (nested
# reference cell: doubled along a1, atoms reversed, Lambda blocks tiled), R
# (the unsymmetrized Lambda)
NL_SCRIPT = r"""
import shutil, sys
import h5py, numpy as np
src, dst, what = sys.argv[1:4]
shutil.copy(src, dst)
b, lam = np.load("fake_b.npy"), np.load("fake_lambda.npy")
if "R" in what:
    lam = np.load("fake_lambda_raw.npy")
with h5py.File(dst, "r+") as f:
    g = f.require_group("Piezoelectric")
    if "B" in what:
        d = g.create_dataset("second_order", data=b)
        d.attrs["unit"] = "C/m^2"
        d.attrs["convention"] = "proper, clamped-ion, per linear strain u"
    if "N" in what:
        rc = f["ReferenceCell"]
        xf = rc["fractional_coordinate"][()] * [0.5, 1.0, 1.0]
        kinds = rc["atomic_kinds"][()]
        lat = rc["lattice_vector"][()]
        lat[0] *= 2.0
        for name, v in (
            ("fractional_coordinate", np.vstack([xf, xf + [0.5, 0.0, 0.0]])[::-1]),
            ("atomic_kinds", np.concatenate([kinds, kinds])[::-1]),
        ):
            del rc[name]
            rc[name] = v
        rc["lattice_vector"][...] = lat
        rc["number_of_atoms"][()] = 2 * len(kinds)
        lam = np.concatenate([lam, lam])[::-1]
    if "L" in what:
        d = g.create_dataset("born_charge_strain_derivative", data=lam)
        d.attrs["unit"] = "e"
        d.attrs["convention"] = "reduced (F^-1 Z*), per linear strain u"
"""


def build_nl_containers(py, tools_dir):
    np.save("fake_b.npy", B_FAKE)
    np.save("fake_lambda.npy", LAMBDA_FAKE)
    np.save("fake_lambda_raw.npy", LAMBDA_RAW)
    # the same data through the strainkit writer (end-to-end writer/reader check)
    code = (
        "import shutil, sys\n"
        "import numpy as np\n"
        "sys.path.insert(0, %r)\n"
        "from strainkit import strainfile as sf\n"
        "b, lam = np.load('fake_b.npy'), np.load('fake_lambda.npy')\n"
        "shutil.copy('bto_piezo.h5', 'bto_nl_py.h5')\n"
        "sf.add_piezo_datasets('bto_nl_py.h5', {sf.SECOND_ORDER: b, sf.BORN_DERIV: lam},"
        " log=lambda *a: None)\n"
        "shutil.copy('bto.h5', 'bto_lam_py.h5')\n"
        "sf.add_piezo_datasets('bto_lam_py.h5', {sf.BORN_DERIV: lam}, log=lambda *a: None)\n"
        % tools_dir
    )
    res = subprocess.run([py, "-c", code], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError("strainfile.add_piezo_datasets failed:\n%s" % res.stderr)
    run_nl_script(py, "bto_piezo.h5", "bto_nl.h5", "BL")
    run_nl_script(py, "bto_piezo.h5", "bto_pB.h5", "B")
    run_nl_script(py, "bto_piezo.h5", "bto_pL.h5", "L")
    run_nl_script(py, "bto.h5", "bto_lam.h5", "L")
    run_nl_script(py, "bto_piezo.h5", "bto_nl_nest.h5", "BLN")
    run_nl_script(py, "bto.h5", "bto_lam_raw.h5", "LR")


def nl_terms(efield, umn, u0):
    """Python reference of the phase-2 field terms at strain umn and Cartesian u0
    [Bohr, (natom, 3)]: dt_B + dt_L [e Bohr], the change of the 9 strain gradients
    [Ry], the field-force pattern f_kb = E.(Lambda_k:u) [Ry/Bohr]."""
    omega = OMEGA_REF[0]
    e_ry = efield * EV_A_TO_RY
    bb = (B_EFF if np.any(efield) else B_FAKE) / C_M2_PER_E_BOHR2
    lam_u = np.einsum("kibmn,mn->kib", LAMBDA_FAKE, umn)
    dt = 0.5 * omega * np.einsum("ijklm,jk,lm->i", bb, umn, umn)
    dt += np.einsum("kib,kb->i", lam_u, u0)
    dg = -omega * np.einsum("i,imnpq,pq->mn", e_ry, bb, umn)
    dg -= np.einsum("i,kibmn,kb->mn", e_ry, LAMBDA_FAKE, u0)
    return dt, dg.ravel(), np.einsum("i,kib->kb", e_ry, lam_u)


LBL_Q = "gradient dF/dq0 (all Gamma modes) :"


def mode_lambda(log, efield):
    """L[mn, s] = sum_{k,i,b} E_i Lambda_k,ib,mn e_s(kb) / sqrt(M_k) in the harmonic
    Gamma mode basis printed by anphon (VERBOSITY >= 2 under EFIELD)."""
    ev = np.array(
        [x.split()[2:] for x in log.splitlines() if x.startswith("  EV ")], float
    )
    mass = masses_ry(log, ["Ba", "Ti", "O", "O", "O"])
    f = np.einsum("i,kibmn->mnkb", efield * EV_A_TO_RY, LAMBDA_FAKE).reshape(9, -1)
    return (f / np.repeat(np.sqrt(mass), 3)) @ ev.T


def same_scaled(a, ref, rel):
    """|a - ref| <= rel * max |ref| componentwise (scale-aware, no absolute floor)."""
    return np.abs(np.asarray(a) - ref).max() <= rel * np.abs(ref).max()


def test_w1(anphonbin):
    # bto_nl.h5 (e0 + B + Lambda) against v1_p (bto_piezo.h5, e0 only) of V1:
    # same field, strain and q0; one step.
    efield = np.array([0.004, -0.003, 0.01])
    umn = np.array(STRAIN_SHEAR.split(), float).reshape(3, 3)
    common = dict(
        relax_str=4,
        max_iter=1,
        displace=DISPLACE_1B,
        strain=STRAIN_SHEAR,
        verbosity=2,
        efield=" ".join(map(str, efield)),
    )
    rc, log = run_anphon(
        anphonbin, "w1_nl", bto_input("w1_nl", strainfile="bto_nl.h5", **common)
    )
    if rc:
        return check(False, "W1: anphon failed with bto_nl.h5")
    q0 = step_row("step_q0.txt", 0)
    u0 = step_row("step_u0.txt", 0).reshape(-1, 3)
    log_p = open("v1_p.log").read()
    dt_ref, dg_ref, force = nl_terms(efield, umn, u0)
    e_ry = efield * EV_A_TO_RY
    dv = last_values(log, LBL_V0)[0] - last_values(log_p, LBL_V0)[0]
    ddt = last_values(log, LBL_DT) - last_values(log_p, LBL_DT)
    info = check(
        close(dv, -e_ry @ ddt, 1.0e-9),
        "W1: V0(nl) - V0(e0) = %.10e == -E0.(dt(nl) - dt(e0)) = %.10e"
        % (dv, -e_ry @ ddt),
    )
    info += check(
        np.allclose(ddt, dt_ref, rtol=1.0e-5, atol=1.0e-6 * np.abs(dt_ref).max()),
        "W1: dt(nl) - dt(e0) %s == Omega B:u:u/2 + sum (Lambda:u) u0 %s"
        % (ddt, dt_ref),
    )
    dg = last_values(log, LBL_G) - last_values(log_p, LBL_G)
    info += check(
        np.allclose(dg, dg_ref, rtol=1.0e-5, atol=1.0e-5 * np.abs(dg_ref).max()),
        "W1: G(nl) - G(e0) == -Omega E0.B:u - E0.Lambda u0 (max dev %.1e of %.1e)"
        % (np.abs(dg - dg_ref).max(), np.abs(dg_ref).max()),
    )
    # the force difference, mode by mode: -sum_mn L_mn,s u_mn (u exact from the input)
    dq = last_values(log, LBL_Q) - last_values(log_p, LBL_Q)
    dq_ref = -umn.ravel() @ mode_lambda(log, efield)
    info += check(
        same_scaled(dq, dq_ref, 1.0e-9),
        "W1: g(nl) - g(e0) == -L:u componentwise (max dev %.1e of %.1e)"
        % (np.abs(dq - dq_ref).max(), np.abs(dq_ref).max()),
    )
    info += check(
        close(dq @ q0, -np.sum(force * u0), 1.0e-5, 0.0),
        "W1: (g(nl) - g(e0)).q0 = %.9e == -E0.sum (Lambda:u) u0 = %.9e (step-file u0)"
        % (dq @ q0, -np.sum(force * u0)),
    )

    # the same data on a doubled, reordered /ReferenceCell: bit-identical
    rc, log_n = run_anphon(
        anphonbin,
        "w1_nest",
        bto_input("w1_nest", strainfile="bto_nl_nest.h5", **common),
    )
    same = rc == 0 and all(
        np.array_equal(last_values(log, lbl), last_values(log_n, lbl))
        for lbl in (LBL_V0, LBL_G, LBL_Q, LBL_DT)
    )
    info += check(
        same and "translation images averaged" in log_n,
        "W1: nested + permuted /ReferenceCell gives bit-identical V0, G, forces and dt",
    )

    # the containers written by strainfile.add_piezo_datasets: the same numbers
    py_logs = {}
    for name, sfile in (("w1_nl_py", "bto_nl_py.h5"), ("w1_lam_py", "bto_lam_py.h5")):
        rc, py_logs[name] = run_anphon(
            anphonbin, name, bto_input(name, strainfile=sfile, **common)
        )
        if rc:
            return info + check(False, "W1: anphon failed with %s" % sfile)

    # Lambda only (no clamped_ion, no B) against v1_0 (bto.h5)
    rc, log_l = run_anphon(
        anphonbin, "w1_lam", bto_input("w1_lam", strainfile="bto_lam.h5", **common)
    )
    if rc:
        return info + check(False, "W1: anphon failed with the Lambda-only container")
    log_0 = open("v1_0.log").read()
    dv = last_values(log_l, LBL_V0)[0] - last_values(log_0, LBL_V0)[0]
    ref = -np.sum(force * u0)
    info += check(
        close(dv, ref, 1.0e-5) and "no e0, no B, Lambda" in log_l,
        "W1: Lambda-only container: V0 - V0(bto.h5) = %.10e == -E0.sum (Lambda:u) u0 = %.10e"
        % (dv, ref),
    )
    info += check(
        all(
            np.array_equal(last_values(a, lbl), last_values(b, lbl))
            for a, b in ((log, py_logs["w1_nl_py"]), (log_l, py_logs["w1_lam_py"]))
            for lbl in (LBL_V0, LBL_G, LBL_Q, LBL_DT)
        ),
        "W1: containers written by strainfile.add_piezo_datasets (B + Lambda, Lambda only)"
        " give bit-identical V0, G, forces and dt",
    )
    header = open("w1_nl.polarization").read()
    info += check(
        "bto_nl.h5:/Piezoelectric/second_order" in header
        and "bto_nl.h5:/Piezoelectric/born_charge_strain_derivative" in header
        and "none (B = 0)" in open("w1_lam.polarization").read(),
        "W1: the .polarization headers name the B and Lambda sources",
    )

    # symmetric shear u_xy = u_yx += +-h at fixed q0: the field energy against
    # G_xy + G_yx of all field terms (e0, B, Lambda)
    h = 1.0e-4
    ef = {}
    for sign in (1, -1):
        umn_s = umn.copy()
        umn_s[0, 1] += sign * h
        umn_s[1, 0] += sign * h
        name = "w1_s%+d" % sign
        text = bto_input(
            name,
            strainfile="bto_nl.h5",
            **dict(
                common, strain="\n".join(" %.12e %.12e %.12e" % tuple(r) for r in umn_s)
            ),
        )
        if run_anphon(anphonbin, name, text)[0]:
            return info + check(False, "W1: shear finite-difference run failed")
        ef[sign] = np.loadtxt(name + ".polarization", ndmin=2)[-1, 7]
    fd = (ef[1] - ef[-1]) / (2.0 * h)
    g_field = (
        -OMEGA_REF[0] * np.einsum("i,imn->mn", e_ry, piezo_e_bohr2(True)).ravel()
        + dg_ref
    )
    ref = g_field[1] + g_field[3]
    info += check(
        close(fd, ref, 1.0e-5),
        "W1: d(E_field)/d(u_xy = u_yx) = %.10e == G_xy + G_yx of the field terms = %.10e"
        % (fd, ref),
    )

    # MPI against serial
    if shutil.which("mpirun") is None:
        print("  skip   W1: mpirun not found, MPI comparison skipped")
        return info
    rc, log_m = run_anphon(
        anphonbin,
        "w1_nl_mpi",
        bto_input("w1_nl_mpi", strainfile="bto_nl.h5", **common),
        nprocs=4,
    )
    info += check(
        rc == 0
        and all(
            np.allclose(
                last_values(log, lbl),
                last_values(log_m, lbl),
                rtol=1.0e-9,
                atol=1.0e-14,
            )
            for lbl in (LBL_V0, LBL_G, LBL_Q, LBL_DT)
        ),
        "W1: mpirun -np 4 V0, G, forces and dt == serial (bto_nl.h5)",
    )
    return info


def test_symmetrize(anphonbin):
    # anphon averages e0, B and Lambda over the space group of the reference
    # (Pm-3m): the unsymmetrized Lambda gives the run of the symmetric one (W1
    # w1_lam, raw tensors kept), and the odd-rank e0 and B vanish (with a warning).
    efield = np.array([0.004, -0.003, 0.01])
    common = dict(
        relax_str=4,
        max_iter=1,
        displace=DISPLACE_1B,
        strain=STRAIN_SHEAR,
        verbosity=2,
        efield=" ".join(map(str, efield)),
    )
    log_l = open("w1_lam.log").read()
    info = 0
    for name, sfile, what in (
        ("s_lam_raw", "bto_lam_raw.h5", "Lambda"),
        ("s_nl", "bto_nl.h5", "e0, B"),
    ):
        rc, log = run_anphon(
            anphonbin,
            name,
            bto_input(name, strainfile=sfile, **common),
        )
        if rc:
            return info + check(False, "S: anphon failed with %s" % sfile)
        dev = max(
            np.abs(last_values(log, lbl) - last_values(log_l, lbl)).max()
            / np.abs(last_values(log_l, lbl)).max()
            for lbl in (LBL_V0, LBL_G, LBL_Q, LBL_DT)
        )
        info += check(
            dev < 1.0e-10 and "averaged over the 48 symmetry operations" in log,
            "S: %s symmetrized: V0, G, forces and dt == symmetric Lambda only (rel. dev %.1e)"
            % (sfile, dev),
        )
        warned = re.findall(r"^ (\S+) changes by", log, re.M)
        info += check(
            ", ".join(warned) == what,
            "S: %s: the large-correction warning names %s (got %s)"
            % (sfile, what, warned),
        )
    return info


def test_v2(anphonbin):
    # E = 0: e0 and POL_REF are output-only. Same steps and V0 bit for bit.
    common = dict(
        relax_str=4, max_iter=2, displace=DISPLACE_1B, strain=STRAIN_SHEAR, verbosity=2
    )
    out = {}
    for name, sfile, pol in (
        ("v2_0", "bto.h5", None),
        ("v2_p", "bto_piezo.h5", POL_REF),
    ):
        rc, log = run_anphon(
            anphonbin, name, bto_input(name, strainfile=sfile, pol_ref=pol, **common)
        )
        if rc:
            return check(False, "V2: anphon failed (%s)" % name)
        out[name] = dict(
            log=log,
            steps=[open(f).read() for f in ("step_u0.txt", "step_u_tensor.txt")],
            v0=re.findall(re.escape(LBL_V0) + r"(.*)", log),
            u0=step_row("step_u0.txt", -2),
            dt=last_values(log, LBL_DT),
            header=open(name + ".polarization").read(),
            row=np.loadtxt(name + ".polarization", ndmin=2)[-1],
        )
    V2_OUT.update(out)
    a, b = out["v2_0"], out["v2_p"]
    info = check(
        a["steps"] == b["steps"] and a["v0"] == b["v0"] and len(a["v0"]) == 2,
        "V2: step_u0/step_u_tensor/V0 bit-identical with and without e0 + POL_REF",
    )
    omega = OMEGA_REF[0]
    umn = np.array(STRAIN_SHEAR.split(), float).reshape(3, 3)
    fmat = np.eye(3) + umn
    zstar = read_born("BORNINFO")
    # B from the step file (7 digits); dt = A + B from the verbose print (full precision)
    b_file = np.einsum("kab,kb->a", zstar, a["u0"].reshape(-1, 3))
    a_pz = omega * np.einsum("ikl,kl->i", piezo_e_bohr2(), umn)
    b_ion = b["dt"] - a_pz
    info += check(
        np.allclose(b_ion, b_file, rtol=1.0e-6, atol=1.0e-12)
        and np.allclose(a["dt"], b_ion, rtol=1.0e-12, atol=1.0e-15),
        "V2: printed dt - Omega e0:u == sum Z* u0 (step file) == dt without e0",
    )
    d_ref = omega * np.array(POL_REF.split(), float) / C_M2_PER_E_BOHR2
    scale = EBOHR2_TO_UC_CM2 / (omega * np.linalg.det(fmat))
    p_ref = scale * fmat @ (d_ref + b["dt"])
    pion_ref = scale * fmat @ b_ion
    info += check(
        np.allclose(b["row"][1:4], p_ref, rtol=1.0e-9, atol=1.0e-12)
        and np.allclose(b["row"][4:7], pion_ref, rtol=1.0e-9, atol=1.0e-12)
        and b["row"][7] == 0.0,
        "V2: P %s == F (d_ref + A + B) / (Omega det F) %s" % (b["row"][1:4], p_ref),
    )
    info += check(
        np.array_equal(a["row"][1:4], a["row"][4:7])
        and np.array_equal(a["row"][4:7], b["row"][4:7]),
        "V2: without e0 and POL_REF, P == P_ion, and P_ion is the same with them",
    )
    info += check(
        "POL_REF not given" in a["header"]
        and "none (e0 = 0)" in a["header"]
        and "POL_REF = 0.05 -0.02 0.26" in b["header"]
        and "bto_piezo.h5:/Piezoelectric/clamped_ion" in b["header"],
        "V2: the .polarization headers echo POL_REF (or its absence) and the e0 source",
    )
    return info


V2_OUT = {}


def test_w4(anphonbin):
    # E = 0 with B and Lambda: the energies and gradients are those of V2's
    # bto.h5 run bit for bit; only dt (and P) change.
    common = dict(
        relax_str=4, max_iter=2, displace=DISPLACE_1B, strain=STRAIN_SHEAR, verbosity=2
    )
    rc, log = run_anphon(
        anphonbin, "w4_nl", bto_input("w4_nl", strainfile="bto_nl.h5", **common)
    )
    if rc or "v2_0" not in V2_OUT:
        return check(False, "W4: anphon failed (or V2 did not run)")
    a = V2_OUT["v2_0"]
    steps = [open(f).read() for f in ("step_u0.txt", "step_u_tensor.txt")]

    def grads(lg):
        return [re.findall(re.escape(lbl) + r"(.*)", lg) for lbl in (LBL_G, LBL_Q)]

    info = check(
        steps == a["steps"]
        and re.findall(re.escape(LBL_V0) + r"(.*)", log) == a["v0"]
        and grads(log) == grads(a["log"]),
        "W4: E = 0: steps, V0 and all gradients bit-identical with and without B + Lambda",
    )
    umn = np.array(STRAIN_SHEAR.split(), float).reshape(3, 3)
    dt_ref = nl_terms(np.zeros(3), umn, a["u0"].reshape(-1, 3))[0]
    ddt = last_values(log, LBL_DT) - V2_OUT["v2_p"]["dt"]  # both carry e0
    info += check(
        np.allclose(ddt, dt_ref, rtol=1.0e-5, atol=1.0e-6 * np.abs(dt_ref).max()),
        "W4: dt(nl) - dt(e0) %s == Omega B:u:u/2 + sum (Lambda:u) u0 %s (P changes)"
        % (ddt, dt_ref),
    )
    return info


def test_2_restarts(anphonbin):
    # state files: w1_nl (e0 + B + Lambda) and v1_p (e0) of W1 / V1
    efield = "0.004 -0.003 0.01"
    common = dict(
        relax_str=4,
        max_iter=1,
        displace=DISPLACE_1B,
        strain=STRAIN_SHEAR,
        efield=efield,
        restart=True,
    )
    msg_l = "strain derivative of the Born charges"
    text_scph = bto_input("t6_text_lam", strainfile="bto_lam.h5", restart=True)
    text_qha = bto_input(
        "t6_text_qha", strainfile="bto_pB.h5", restart=True, mode="QHA"
    )
    cases = [
        (
            "w1_nl (restart, no Lambda)",
            bto_input("w1_nl", strainfile="bto_pB.h5", **common),
            msg_l,
        ),
        (
            "v1_p (restart, Lambda added)",
            bto_input("v1_p", strainfile="bto_pL.h5", **common),
            msg_l,
        ),
        (
            "t6_text_lam (SCPH restart, FILE_FORMAT = text, Lambda)",
            text_scph.replace("  VERBOSITY", "  FILE_FORMAT = text\n  VERBOSITY"),
            "legacy text restart files",
        ),
        (
            "t6_text_qha (QHA restart, FILE_FORMAT = text, B)",
            text_qha.replace("  VERBOSITY", "  FILE_FORMAT = text\n  VERBOSITY"),
            "legacy text restart files",
        ),
    ]
    info = 0
    for name, text, message in cases:
        rc, log = run_anphon(anphonbin, name.split()[0], text)
        info += check(
            rc != 0 and message in log, "2: %s rejected (%s)" % (name, message)
        )
    rc, log = run_anphon(
        anphonbin, "w1_nl", bto_input("w1_nl", strainfile="bto_nl.h5", **common)
    )
    info += check(
        rc == 0 and "RESTART_SCPH is true" in log,
        "2: restart with the same B and Lambda accepted",
    )
    return info


def test_v3(anphonbin):
    # Maxwell identity of the implemented free energy at fixed strain, internal
    # coordinates relaxed: d(dt_i)/du_mn |_E = -dG_mn/dE_i |_u (i = z).
    # Matched meshes (KMESH_SCPH = KMESH_INTERPOLATE): with Fourier
    # interpolation the SCP functional is stationary on the coarse mesh only, so
    # force and stress are not exact derivatives of one free energy and the
    # identity breaks at the interpolation-error level (~2e-4 with 2/4 meshes).
    efield = np.array([0.004, -0.003, 0.01])
    umn0 = np.array(STRAIN_SHEAR.split(), float).reshape(3, 3)
    tight = dict(
        relax_str=4,
        max_iter=80,
        verbosity=2,
        strainfile="bto_nl.h5",  # e0 + B + Lambda (W2)
        coord_tol="1.0e-10",
        extra="\n  GRADIENT_CONV_TOL = 1.0e-10",
    )

    def relax(name, umn, e, displace):
        # a PREFIX.scph.h5 left by an earlier run would turn this into a restart
        for f in (name + ".scph.h5", name + ".polarization"):
            if os.path.exists(f):
                os.remove(f)
        strain = "\n".join(" %.12e %.12e %.12e" % tuple(r) for r in umn)
        text = bto_input(
            name,
            efield=" ".join("%.12e" % x for x in e),
            strain=strain,
            displace=displace,
            **tight,
        ).replace("KMESH_SCPH = 4 4 4", "KMESH_SCPH = 2 2 2")
        rc, log = run_anphon(anphonbin, name, text)
        row = np.loadtxt(name + ".polarization", ndmin=2)[-1] if rc == 0 else None
        ok = row is not None and row[8] == 1 and "SCP NOT converged" not in log[-3000:]
        return ok, log

    ok, log = relax("v3_base", umn0, efield, DISPLACE_1B)
    if not ok:
        return check(False, "V3: base relaxation failed or did not converge")
    u0 = np.loadtxt("v3_base.atom_disp", ndmin=2)[-1, 1:].reshape(-1, 3)
    seed = "&displace\n1\n%s\n/" % "\n".join(
        " %.12e %.12e %.12e" % tuple(r) for r in u0
    )

    # The E side (soft polar mode) has an O(h_E^2) central-difference error:
    # Richardson-extrapolate two field steps. The strain side is linear enough.
    ez = np.array([0.0, 0.0, 1.0])
    h_u = 1.0e-4
    dg_de = {}
    for h_e in (5.0e-4, 2.5e-4):
        g = {}
        for sign in (1, -1):
            ok, log = relax(
                "v3_e%+d_%.0e" % (sign, h_e), umn0, efield + sign * h_e * ez, seed
            )
            if not ok:
                return check(False, "V3: E run did not converge")
            g[sign] = last_values(log, LBL_G).reshape(3, 3)
        dg_de[h_e] = (g[1] - g[-1]) / (2.0 * h_e * EV_A_TO_RY)
    dg_de = (4.0 * dg_de[2.5e-4] - dg_de[5.0e-4]) / 3.0

    info = 0
    scale_zz = 0.0
    for label, (m, n) in (("zz", (2, 2)), ("xy", (0, 1))):
        dts = {}
        for sign in (1, -1):
            umn = umn0.copy()
            umn[m, n] += sign * h_u
            if m != n:
                umn[n, m] += sign * h_u
            ok, log = relax("v3_%s%+d" % (label, sign), umn, efield, seed)
            if not ok:
                return check(False, "V3: strain run did not converge")
            dts[sign] = last_values(log, LBL_DT)
        lhs = (dts[1][2] - dts[-1][2]) / (2.0 * h_u)
        rhs = -dg_de[m, n] - (dg_de[n, m] if m != n else 0.0)
        # The finite differences carry an absolute noise of ~1e-7 of the large zz
        # response; the xy response of the cubic reference is ~1e3 times smaller
        # (with the averaged e0 = B = 0), so it is measured against 1% of zz.
        scale = max(abs(lhs), abs(rhs))
        scale_zz = max(scale_zz, scale)
        rel = abs(lhs - rhs) / max(scale, 1.0e-2 * scale_zz)
        info += check(
            rel < 5.0e-5,
            "V3 %s: d(dt_z)/du = %.10e == -dG/dE_z = %.10e (extrapolated), rel. diff %.2e"
            % (label, lhs, rhs, rel),
        )
    return info


FE_DISP = """&displace
1
0.0 0.0 0.004
0.0 0.0 0.11
0.0 0.0 -0.088
0.0 0.0 -0.088
0.0 0.0 -0.154
/"""


def test_fe(anphonbin):
    # The reported F_total (PREFIX.scph_thermo) is the free energy whose
    # derivatives are the printed stress, force and -S, also with Fourier interpolation
    # (KMESH_INTERPOLATE < KMESH_SCPH; F_total is evaluated on the &kpoint mesh, set =
    # KMESH_SCPH). Polar BaTiO3 at 300 K, bto.h5, RELAX_STR = 4. The q0- and
    # strain-renormalized harmonic matrix of the solve is interpolated from the coarse
    # mesh, so its vertices must be too (ScphQhaCommon::coarse_occupation_matrices); the
    # vertices taken at the dense k gave a stress 1-3e-4 Ry off dF/du here, the force
    # 1e-2 off, and dF/dE off -d by 1.6e-3. What is left is the residual of the
    # interpolated self-energy, which is not self-adjoint (~1e-6 Ry in the stress, 1e-3
    # of the small force here; not a general bound, e.g. near soft modes).
    umn0 = np.diag([0.004, 0.004, 0.01])
    shear = np.array(
        [[0.0, 0.002, -0.001], [0.002, 0.0, 0.0015], [-0.001, 0.0015, 0.0]]
    )

    def run(
        name,
        umn,
        temp=300.0,
        displace=FE_DISP,
        relax=False,
        efield=None,
        nonanalytic=0,
    ):
        if os.path.exists(name + ".scph.h5"):
            os.remove(name + ".scph.h5")
        text = bto_input(
            name,
            efield=efield,
            relax_str=4,
            max_iter=80 if relax else 1,
            displace=displace,
            strain="\n".join(" %.12e %.12e %.12e" % tuple(r) for r in umn),
            verbosity=2,
            strainfile="bto.h5",
            coord_tol="1.0e-9" if relax else "10.0",
            extra="\n  GRADIENT_CONV_TOL = 1.0e-9" if relax else "",
        )
        text = text.replace("  8 8 8", "  4 4 4")
        text = text.replace(
            "TMIN = 300; TMAX = 300", "TMIN = %g; TMAX = %g" % (temp, temp)
        )
        if nonanalytic:
            text = text.replace(
                "BORNINFO = BORNINFO",
                "BORNINFO = BORNINFO\n  NONANALYTIC = %d" % nonanalytic,
            )
        rc, log = run_anphon(anphonbin, name, text)
        if rc or "Structural optimization converged" not in log:
            return None
        th = np.loadtxt(name + ".scph_thermo", ndmin=2)[-1]
        return dict(
            F=th[-2], S=th[-1], G=last_values(log, LBL_G).reshape(3, 3), log=log
        )

    def fd_strain(prefix, direction, hs, umn=umn0, **kw):
        # central differences of F_total along umn + h direction; with two steps,
        # Richardson-extrapolated
        d = []
        for i, h in enumerate(hs):
            f = {}
            for sign in (1, -1):
                r = run("%s%d%+d" % (prefix, i, sign), umn + sign * h * direction, **kw)
                if r is None:
                    return None
                f[sign] = r["F"]
            d.append((f[1] - f[-1]) / (2.0 * h))
        return d[0] if len(d) == 1 else (4.0 * d[1] - d[0]) / 3.0

    ezz = np.zeros((3, 3))
    ezz[2, 2] = 1.0
    two = (1.0e-3, 5.0e-4)  # the O(h^2) error at 1e-3 is ~2e-5
    info = 0

    def strain_check(label, prefix, base, direction, hs=two, tol=5.0e-6, **kw):
        d = fd_strain(prefix, direction, hs, **kw)
        g = np.sum(base["G"] * direction)
        return check(
            d is not None and abs(d - g) < tol,
            "FE %s: dF_total/du = %.6e == G = %.6e" % (label, d or 0.0, g),
        )

    # 1. fixed structure, 2/4 meshes: zz, the force, -S
    base = run("fe_0", umn0)
    if base is None:
        return check(False, "FE: base run failed")
    info += strain_check("zz (fixed structure)", "fe_zz", base, ezz)
    # the force: a displacement that keeps the centre of mass (Ti z against O3 z),
    # q0 of the evaluated structures from step_q0.txt (row 0)
    g_q = last_values(base["log"], LBL_Q)
    mass = masses_ry(base["log"], ["Ba", "Ti", "O", "O", "O"])
    disp0 = np.array([r.split() for r in FE_DISP.splitlines()[2:7]], float)
    dvec = np.zeros((5, 3))
    dvec[1, 2], dvec[4, 2] = 1.0, -mass[1] / mass[4]
    # ratio Delta F / (g . Delta q0), Richardson over two steps (O(h^2): 5e-3 at 2e-3 Bohr)
    ratio = []
    for i, h in enumerate((2.0e-3, 1.0e-3)):
        f, q = {}, {}
        for sign in (1, -1):
            d = disp0 + sign * h * dvec
            disp = "&displace\n1\n%s\n/" % "\n".join(
                " %.12e %.12e %.12e" % tuple(r) for r in d
            )
            r = run("fe_q%d%+d" % (i, sign), umn0, displace=disp)
            if r is None:
                return info + check(False, "FE: displacement runs failed")
            f[sign], q[sign] = r["F"], step_row("step_q0.txt", 0)
        ratio.append((f[1] - f[-1]) / (g_q @ (q[1] - q[-1])))
    ratio = (4.0 * ratio[1] - ratio[0]) / 3.0
    info += check(
        abs(ratio - 1.0) < 3.0e-3,  # 1e-3 left (self-energy residual), 1e-2 before
        "FE force (fixed structure): Delta F_total / (g . Delta q0) = %.6f (extrapolated)"
        % ratio,
    )
    t = [run("fe_t%+d" % s, umn0, temp=300.0 + s) for s in (1, -1)]
    if None in t:
        return info + check(False, "FE: T runs failed")
    dfdt = (t[0]["F"] - t[1]["F"]) / 2.0
    kb_ry = 1.380649e-23 / 4.3597447222071e-18 * 2.0
    info += check(
        abs(dfdt + base["S"] * kb_ry) < 1.0e-4 * base["S"] * kb_ry,
        "FE T (fixed structure): dF_total/dT = %.8e == -S = %.8e Ry/K"
        % (dfdt, -base["S"] * kb_ry),
    )
    # 2. a sheared cell: the shear stress (G_xy + G_yx)
    exy = np.zeros((3, 3))
    exy[0, 1] = exy[1, 0] = 1.0
    sh = run("fe_s0", umn0 + shear)
    if sh is None:
        return info + check(False, "FE: shear run failed")
    info += strain_check("xy (sheared cell)", "fe_xy", sh, exy, umn=umn0 + shear)
    # 3. NONANALYTIC = 3 on 2/4 meshes, commensurate with the 2x2x2 harmonic supercell
    #    (with a coarse mesh that is not, F_total has a separate known inconsistency).
    #    3/6 meshes pass with and without the coarse-mesh adjoint (a 3x3x3 coarse cell
    #    holds the range of the 2x2x2 IFCs) and are not a test of it.
    b = run("fe_na3_0", umn0, nonanalytic=3)
    if b is None:
        return info + check(False, "FE: NONANALYTIC = 3 base run failed")
    info += strain_check(
        "zz (NONANALYTIC = 3)", "fe_na3", b, ezz, tol=1.0e-5, nonanalytic=3
    )
    # 4. internal coordinates relaxed (the force enters through dq0/du): the envelope
    #    dF_total/du_zz == G_zz, and dF_total/dE_z == -d_z
    rel = run("fe_r0", umn0, relax=True)
    if rel is None:
        return info + check(False, "FE: relaxation failed")
    u0 = np.loadtxt("fe_r0.atom_disp", ndmin=2)[-1, 1:].reshape(-1, 3)
    seed = "&displace\n1\n%s\n/" % "\n".join(
        " %.12e %.12e %.12e" % tuple(r) for r in u0
    )
    info += strain_check(
        "zz (relaxed q0)",
        "fe_rzz",
        rel,
        ezz,
        hs=(1.0e-3,),
        tol=2.0e-5,
        displace=seed,
        relax=True,
    )
    # the polar soft mode makes F(E) strongly curved: Richardson over two field steps
    dfde, dz = [], []
    for h_e in (1.0e-3, 5.0e-4):  # V/A
        fe = {}
        for sign in (1, -1):
            efield = "0 0 %.6e" % (sign * h_e)
            r = run("fe_e%+d" % sign, umn0, displace=seed, relax=True, efield=efield)
            if r is None:
                return info + check(False, "FE: field runs failed")
            fe[sign] = (r["F"], last_values(r["log"], LBL_DT)[2])
        dfde.append((fe[1][0] - fe[-1][0]) / (2.0 * h_e * EV_A_TO_RY))
        dz.append(0.5 * (fe[1][1] + fe[-1][1]))
    dfde = (4.0 * dfde[1] - dfde[0]) / 3.0
    dz = (4.0 * dz[1] - dz[0]) / 3.0
    info += check(
        abs(dfde + dz) < 5.0e-4 * abs(dz),
        "FE E (relaxed q0): dF_total/dE_z = %.6e == -d_z = %.6e e Bohr (extrapolated)"
        % (dfde, -dz),
    )
    return info


def test_w3(anphonbin):
    # BUBBLE = 4 at the first structure (COORD_CONV_TOL = 10: converged at once),
    # matched meshes, bto_nl.h5 and bto_piezo.h5 at the same field.
    efield = np.array([0.004, -0.003, 0.01])
    blocks, logs = {}, {}
    for name, sfile, extra in (
        ("w3_nl", "bto_nl.h5", ""),
        ("w3_p", "bto_piezo.h5", ""),
    ):
        text = bto_input(
            name,
            strainfile=sfile,
            relax_str=4,
            max_iter=1,
            displace=DISPLACE_1B,
            strain=STRAIN_SHEAR,
            verbosity=2,
            coord_tol="10.0",
            efield=" ".join(map(str, efield)),
            extra=extra,
        )
        text = text.replace("KMESH_SCPH = 4 4 4", "KMESH_SCPH = 2 2 2")
        text = text.replace(
            "SELF_OFFDIAG = 1", "SELF_OFFDIAG = 1\n  BUBBLE = 4\n  BUBBLE_FD_CHECK = 2"
        )
        rc, log = run_anphon(anphonbin, name, text)
        lines = log.splitlines()
        ss = np.array([x.split()[1:] for x in lines if x.startswith("   ss")], float)
        qs = [x.split() for x in lines if x.startswith("   qs")]
        if rc or ss.shape != (6, 6) or not qs:
            return check(False, "W3: %s failed or printed no explicit blocks" % name)
        blocks[name] = (
            ss,
            [int(x[1]) for x in qs],
            np.array([x[2:] for x in qs], float),
        )
        logs[name] = log
    log = logs["w3_nl"]
    m = re.search(r"transpose of force-strain block: max diff / max = (\S+)", log)
    info = check(
        m is not None and float(m.group(1)) < 1.0e-8,
        "W3: explicit stress-displacement block == force-strain block^T (%s)"
        % (m.group(1) if m else "-"),
    )
    m = re.search(r"asymmetry \|J - J\^T\| / \|J\| \(diagonally scaled\) = (\S+)", log)
    info += check(
        m is not None and float(m.group(1)) < 1.0e-6,
        "W3: Jacobian symmetric (%s)" % (m.group(1) if m else "-"),
    )
    # the optimizer path (BUBBLE_HESS; its projected response needs 1 1 1 meshes)
    text = bto_input(
        "w3_opt",
        strainfile="bto_nl.h5",
        relax_str=4,
        max_iter=1,
        displace=DISPLACE_1B,
        strain=STRAIN_SHEAR,
        efield=" ".join(map(str, efield)),
        extra="\n  BUBBLE_HESS = 1",
    )
    text = text.replace("KMESH_SCPH = 4 4 4", "KMESH_SCPH = 1 1 1")
    text = text.replace("KMESH_INTERPOLATE = 2 2 2", "KMESH_INTERPOLATE = 1 1 1")
    text = text.replace("SELF_OFFDIAG = 1", "SELF_OFFDIAG = 1\n  BUBBLE = 4")
    rc, log_o = run_anphon(anphonbin, "w3_opt", text)
    msg = "BUBBLE_HESS: optimizer Hessian from the free-energy curvature (with strain)"
    info += check(
        rc == 0 and msg in log_o,
        "W3: BUBBLE_HESS optimizer path accepts the curvature with B and Lambda",
    )
    omega = OMEGA_REF[0]
    e_ry = efield * EV_A_TO_RY
    c = -omega * np.einsum("a,amnpq->mnpq", e_ry, B_EFF / C_M2_PER_E_BOHR2)
    s_ref = np.array(
        [
            [
                sum(c[i, j, p, q] for i, j in PAIRS[a] for p, q in PAIRS[b])
                for b in range(6)
            ]
            for a in range(6)
        ]
    )
    d_ss = blocks["w3_nl"][0] - blocks["w3_p"][0]
    scale = np.abs(blocks["w3_p"][0]).max()
    info += check(
        np.abs(d_ss - s_ref).max() <= 1.0e-10 * scale,
        "W3: strain-strain block increment == -Omega E0.B of the averaged B (zero; "
        "max dev %.1e of the block %.1e)" % (np.abs(d_ss - s_ref).max(), scale),
    )
    idx = blocks["w3_nl"][1]
    d_qs = blocks["w3_nl"][2] - blocks["w3_p"][2]
    lmode = mode_lambda(log, efield)
    qs_ref = -np.array(
        [[sum(lmode[3 * p + q, s] for p, q in PAIRS[n]) for n in range(6)] for s in idx]
    )
    info += check(
        blocks["w3_nl"][1] == blocks["w3_p"][1] and same_scaled(d_qs, qs_ref, 1.0e-7),
        "W3: force-strain block increment == -L componentwise (max dev %.1e of %.1e)"
        % (np.abs(d_qs - qs_ref).max(), np.abs(qs_ref).max()),
    )
    return info


def test_cell_relax(anphonbin, seed_script):
    # RELAX_STR = 2 under the field with e0 + B + Lambda (1 1 1 meshes, so that the
    # BUBBLE_HESS optimizer Hessian with its strain blocks is used), converged tightly;
    # then the stress and force at the final structure are re-evaluated independently
    # (RELAX_STR = 4, one step): zero with bto_nl.h5, and the B/Lambda part with
    # bto_piezo.h5 (so the check is sensitive to them).
    efield = "0.004 -0.003 0.01"

    def text_for(name, sfile, relax_str, max_iter, displace, strain, verbosity):
        t = bto_input(
            name,
            strainfile=sfile,
            relax_str=relax_str,
            max_iter=max_iter,
            displace=displace,
            strain=strain,
            verbosity=verbosity,
            coord_tol="1.0e-7",
            efield=efield,
            extra="\n  GRADIENT_CONV_TOL = 1.0e-8\n  CELL_GRADIENT_CONV_TOL = 1.0e-7"
            + ("\n  BUBBLE_HESS = 1" if relax_str == 2 else ""),
        )
        t = t.replace("KMESH_SCPH = 4 4 4", "KMESH_SCPH = 1 1 1")
        t = t.replace("KMESH_INTERPOLATE = 2 2 2", "KMESH_INTERPOLATE = 1 1 1")
        t = t.replace("CELL_CONV_TOL = 5.0e-7", "CELL_CONV_TOL = 1.0e-7")
        if relax_str == 2:
            t = t.replace("SELF_OFFDIAG = 1", "SELF_OFFDIAG = 1\n  BUBBLE = 4")
        return t

    rc, log = run_anphon(
        anphonbin,
        "c2_nl",
        text_for("c2_nl", "bto_nl.h5", 2, 60, DISPLACE_1B, STRAIN_SHEAR, 1),
    )
    info = check(
        rc == 0
        and "Structural optimization converged" in log
        and "BUBBLE_HESS: optimizer Hessian from the free-energy curvature (with strain)"
        in log,
        "cell: RELAX_STR = 2 with B + Lambda converged, BUBBLE_HESS strain Hessian used",
    )
    if info:
        return info
    seed = subprocess.run(
        [sys.executable, seed_script, "c2_nl"], capture_output=True, text=True
    ).stdout
    disp = seed[seed.index("&displace") : seed.index("&strain")]
    strain = seed[seed.index("&strain") + len("&strain") : seed.rindex("/")].strip()
    g = {}
    for tag, sfile in (("nl", "bto_nl.h5"), ("p", "bto_piezo.h5")):
        name = "c2_chk_" + tag
        rc, lg = run_anphon(
            anphonbin, name, text_for(name, sfile, 4, 1, disp, strain, 2)
        )
        if rc:
            return check(False, "cell: re-evaluation failed (%s)" % sfile)
        gm = last_values(lg, LBL_G).reshape(3, 3)
        g[tag] = gm + gm.T - np.diag(np.diag(gm))  # the optimizer's variables
    scale = np.abs(g["p"]).max()
    info += check(
        np.abs(g["nl"]).max() < 1.0e-3 * scale,
        "cell: stress at the relaxed structure %.1e << its B/Lambda part %.1e"
        % (np.abs(g["nl"]).max(), scale),
    )
    return info


def scheme_stress(log, c2, dv1, umn, vzsisa):
    """Qha::compute_ZSISA_stress / compute_vZSISA_stress in Python."""
    opt = [
        int(x)
        for x in re.search(r"ZSISA inputs: optical modes :(.*)", log).group(1).split()
    ]
    v1 = last_values(log, "ZSISA inputs: QHA force (all Gamma modes) :")
    g_qha = last_values(log, "ZSISA inputs: QHA stress dF/du_mn [Ry] :")
    g_static = last_values(log, "ZSISA inputs: static stress dV/du_mn [Ry] :")
    v2 = np.array(
        [x.split()[2:] for x in log.splitlines() if x.startswith("  V2 ")], float
    )
    dq = np.zeros((len(v1), 9))  # dq*/du_mn = -V2^-1 dv1/du_mn on the optical modes
    dq[opt] = -np.linalg.solve(v2[np.ix_(opt, opt)], dv1[:, opt].T)
    g_z = g_qha + v1 @ dq
    if not vzsisa:
        return g_z
    c2z = c2 + dv1 @ dq
    f = np.eye(3) + umn
    cof = np.linalg.det(f) * np.linalg.inv(f).T  # d det F / du_mn
    ddet = cof.ravel()
    ut = ddet / np.linalg.norm(ddet)
    pairs = [(a, a) for a in range(3)] + [((a + 1) % 3, (a + 2) % 3) for a in range(3)]
    vec = np.array([cof[p] for p in pairs])
    cm = np.zeros((6, 6))
    for a, (i, j) in enumerate(pairs):
        for b, (k, l_) in enumerate(pairs):
            cm[a, b] = (2.0 if b >= 3 else 1.0) * c2z[3 * i + j, 3 * k + l_]
    d6 = np.linalg.solve(cm, vec)
    d9 = np.zeros((3, 3))
    for a, (i, j) in enumerate(pairs):
        d9[i, j] = d9[j, i] = d6[a]
    d9 = d9.ravel() / (ut @ d9.ravel())
    return g_static + ut * (d9 @ g_z - ut @ g_static)


def test_zsisa(anphonbin):
    # QHA with v-ZSISA, one step at the initial structure; the static force is
    # what the step prints (v1 = v1_renorm under ZSISA / v-ZSISA).
    efield = np.array([0.004, -0.003, 0.01])
    umn0 = np.array(STRAIN_SHEAR.split(), float).reshape(3, 3)

    def run(name, sfile, umn, scheme=2):
        text = bto_input(
            name,
            mode="QHA",
            strainfile=sfile,
            relax_str=2,
            max_iter=1,
            displace=DISPLACE_1B,
            strain="\n".join(" %.12e %.12e %.12e" % tuple(r) for r in umn),
            verbosity=2,
            efield=" ".join(map(str, efield)),
        ).replace("RELAX_STR = 2", "RELAX_STR = 2\n  QHA_SCHEME = %d" % scheme)
        rc, log = run_anphon(anphonbin, name, text)
        lines = log.splitlines()
        c2 = np.array([x.split()[2:] for x in lines if x.startswith("  C2 ")], float)
        dv1 = np.array([x.split()[2:] for x in lines if x.startswith("  DV1 ")], float)
        if rc or c2.shape != (9, 9) or dv1.shape[0] != 9:
            return None
        return c2, dv1, last_values(log, LBL_Q), log

    base_nl = run("wq_nl", "bto_nl.h5", umn0)
    u0 = step_row("step_u0.txt", 0).reshape(-1, 3)
    base_p = run("wq_p", "bto_piezo.h5", umn0)
    if base_nl is None or base_p is None:
        return check(False, "ZSISA: QHA runs failed or printed no ZSISA inputs")
    # The unstable QHA Hessian of cubic BaTiO3 proposes a ~7 bohr step here:
    # the trust region must scale it to exactly 1 bohr and say so.
    m = re.search(r"optimizer step \(du0 = (\S+) bohr", base_p[3])
    info0 = check(
        m is not None
        and float(m.group(1)) > 1.0
        and "optimizer history is reset" in base_p[3]
        and " du0 =   1.000000e+00 [Bohr]" in base_p[3],
        "QHA: an oversized optimizer step is scaled to the 1 bohr trust region",
    )
    omega = OMEGA_REF[0]
    e_ry = efield * EV_A_TO_RY
    c_ref = -omega * np.einsum("a,amnpq->mnpq", e_ry, B_EFF / C_M2_PER_E_BOHR2).reshape(
        9, 9
    )
    d_c2 = base_nl[0] - base_p[0]
    scale = np.abs(base_p[0]).max()
    info = info0 + check(
        np.abs(d_c2 - c_ref).max() <= 1.0e-10 * scale,
        "ZSISA: C2_renorm increment == -Omega E0.B of the averaged B (zero; max dev "
        "%.1e of C2 %.1e)" % (np.abs(d_c2 - c_ref).max(), scale),
    )
    d_dv1 = base_nl[1] - base_p[1]
    l_ref = -mode_lambda(base_nl[3], efield)
    info += check(
        same_scaled(d_dv1, l_ref, 1.0e-9),
        "ZSISA: dv1/du increment == -L componentwise (max dev %.1e of %.1e)"
        % (np.abs(d_dv1 - l_ref).max(), np.abs(l_ref).max()),
    )
    # the QHA stress before the scheme overwrite: the field-term increment
    dg_ref = nl_terms(efield, umn0, u0)[1]
    lbl_qha = "ZSISA inputs: QHA stress dF/du_mn [Ry] :"
    dg = last_values(base_nl[3], lbl_qha) - last_values(base_p[3], lbl_qha)
    info += check(
        same_scaled(dg, dg_ref, 1.0e-5),
        "ZSISA: QHA stress increment == -Omega E0.B:u - E0.Lambda u0 (max dev %.1e of %.1e)"
        % (np.abs(dg - dg_ref).max(), np.abs(dg_ref).max()),
    )
    # the final corrected stress of both schemes, recomputed in Python from the
    # printed ingredients (checked above or field-independent)
    base_z = run("wq1_nl", "bto_nl.h5", umn0, scheme=1)
    if base_z is None:
        return info + check(False, "ZSISA: QHA_SCHEME = 1 run failed")
    for label, res in (
        ("ZSISA (QHA_SCHEME = 1)", base_z),
        ("v-ZSISA (QHA_SCHEME = 2)", base_nl),
    ):
        ref = scheme_stress(res[3], res[0], res[1], umn0, label.startswith("v"))
        got = last_values(res[3], LBL_G)
        info += check(
            same_scaled(got, ref, 1.0e-9),
            "ZSISA: final %s stress == Python from the printed inputs (max dev %.1e of %.1e)"
            % (label, np.abs(got - ref).max(), np.abs(ref).max()),
        )
    # derivative regression: dv1/du against central differences of the static force
    h = 1.0e-4
    for label, (m, n) in (("zz", (2, 2)), ("yz", (1, 2))):
        v1 = {}
        for sign in (1, -1):
            umn = umn0.copy()
            umn[m, n] += sign * h
            if m != n:
                umn[n, m] += sign * h
            res = run("wq_%s%+d" % (label, sign), "bto_nl.h5", umn)
            if res is None:
                return info + check(False, "ZSISA: finite-difference run failed")
            v1[sign] = res[2]
        fd = (v1[1] - v1[-1]) / (2.0 * h)
        ref = base_nl[1][3 * m + n] + (base_nl[1][3 * n + m] if m != n else 0.0)
        dev = np.abs(fd - ref).max() / np.abs(ref).max()
        info += check(
            dev < 1.0e-5,
            "ZSISA: dv1/du_%s == central difference of the static force (rel. dev %.1e)"
            % (label, dev),
        )
    return info


def test_1b_errors(anphonbin):
    # restarts from the V2 state files: v2_0 (no e0, no POL_REF), v2_p (both)
    common = dict(
        relax_str=4, max_iter=2, displace=DISPLACE_1B, strain=STRAIN_SHEAR, restart=True
    )
    cases = [
        (
            "v2_p (restart, no POL_REF)",
            bto_input("v2_p", strainfile="bto_piezo.h5", **common),
            "POL_REF tag is not consistent",
        ),
        (
            "v2_p (restart, no e0)",
            bto_input("v2_p", strainfile="bto.h5", pol_ref=POL_REF, **common),
            "piezoelectric tensor (STRAINFILE /Piezoelectric) is not consistent",
        ),
        (
            "v2_0 (restart, POL_REF added)",
            bto_input("v2_0", strainfile="bto.h5", pol_ref=POL_REF, **common),
            "POL_REF tag is not consistent",
        ),
        (
            "v2_0 (restart, e0 added)",
            bto_input("v2_0", strainfile="bto_piezo.h5", **common),
            "piezoelectric tensor (STRAINFILE /Piezoelectric) is not consistent",
        ),
        (
            "t6_text_pol (restart, FILE_FORMAT = text, POL_REF)",
            bto_input("t6_text_pol", pol_ref=POL_REF, restart=True).replace(
                "  VERBOSITY", "  FILE_FORMAT = text\n  VERBOSITY"
            ),
            "legacy text restart files",
        ),
        (
            "t6_text_e0 (restart, FILE_FORMAT = text, e0)",
            bto_input("t6_text_e0", strainfile="bto_piezo.h5", restart=True).replace(
                "  VERBOSITY", "  FILE_FORMAT = text\n  VERBOSITY"
            ),
            "legacy text restart files",
        ),
        (
            "t6_pol_noborn (POL_REF without BORNINFO, E = 0)",
            bto_input("t6_pol_noborn", pol_ref=POL_REF, born=False),
            "POL_REF is given",
        ),
        ("t6_pol_two", bto_input("t6_pol_two", pol_ref="0.1 0.2"), "exactly three"),
    ]
    info = 0
    for name, text, message in cases:
        rc, log = run_anphon(anphonbin, name.split()[0], text)
        info += check(
            rc != 0 and message in log, "1b: %s rejected (%s)" % (name, message)
        )
    rc, log = run_anphon(
        anphonbin,
        "v2_p",
        bto_input("v2_p", strainfile="bto_piezo.h5", pol_ref=POL_REF, **common),
    )
    info += check(
        rc == 0 and "RESTART_SCPH is true" in log,
        "1b: restart with the same POL_REF and e0 accepted",
    )
    return info


# ------------------------------------------------- ZnO (6mm): production path
# Generic e0, B and Lambda (raw) and their P6_3mc averages (sym), built by
# ZNO_SCRIPT with strainkit; the containers hold only /ReferenceCell and
# /Piezoelectric (STRAIN_COUPLING = 0: couplings from the IFCs).
ZNO_SCRIPT = r"""
import re, shutil, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import ase
from strainkit import strainfile as sf
from strainkit.symmetry import space_group_operations, symmetrize_atomic
from strainkit.writers import ReferenceCell
lat = np.array([float(x) for x in sys.argv[2].split()]).reshape(3, 3)  # rows, Angstrom
x = open("ZnO442_harmonic.xml").read()
pos = {int(i): (e, [float(v) for v in p.split()]) for i, e, p in
       re.findall(r'<pos index="(\d+)" element="(\w+)">([^<]+)</pos>', x)}
prim = sorted((int(a), int(s)) for t, a, s in
              re.findall(r'<map tran="(\d+)" atom="(\d+)">(\d+)</map>', x) if t == "1")
el = [pos[s][0] for _, s in prim]
xf = np.array([pos[s][1] for _, s in prim]) * [4.0, 4.0, 2.0] % 1.0
cell = ReferenceCell(lat, el, xf)
rots, perms, images = ops = space_group_operations(
    ase.Atoms(el, cell=lat, scaled_positions=xf, pbc=True))


def glob(t):
    out = np.zeros_like(t)
    for r in rots:
        rt = t
        for ax in range(t.ndim):
            rt = np.moveaxis(np.tensordot(r, rt, axes=(1, ax)), 0, ax)
        out += rt
    return out / len(rots)


rng = np.random.default_rng(20261007)
e0 = rng.uniform(-0.6, 0.6, (3, 3, 3))
e0 = 0.5 * (e0 + e0.transpose(0, 2, 1))
b = rng.uniform(-2.0, 2.0, (3, 3, 3, 3, 3))
for axes in ((0, 2, 1, 3, 4), (0, 1, 2, 4, 3), (0, 3, 4, 1, 2)):
    b = 0.5 * (b + b.transpose(axes))
lam = rng.uniform(-1.0, 1.0, (4, 3, 3, 3, 3))
lam = 0.5 * (lam + lam.transpose(0, 1, 2, 4, 3))
lam -= lam.mean(axis=0)
raw = {sf.CLAMPED_ION: e0, sf.SECOND_ORDER: b, sf.BORN_DERIV: lam}
sym = {sf.CLAMPED_ION: glob(e0), sf.SECOND_ORDER: glob(b),
       sf.BORN_DERIV: symmetrize_atomic(lam, *ops)}
np.save("zno_sym_e0.npy", sym[sf.CLAMPED_ION])
np.save("zno_sym_b.npy", sym[sf.SECOND_ORDER])
np.save("zno_sym_lam.npy", sym[sf.BORN_DERIV])
quiet = lambda *a: None
for name, data in (("zno_ref.h5", {}), ("zno_nl.h5", sym), ("zno_raw.h5", raw),
                   ("zno_eL.h5", {k: sym[k] for k in (sf.CLAMPED_ION, sf.BORN_DERIV)})):
    with sf.update(name, cell) as f:
        if data:
            sf.write_piezo_datasets(f, data)
# nested and permuted: /ReferenceCell doubled along a3, atoms reversed
big = ReferenceCell(lat * [[1], [1], [2]], (el * 2)[::-1],
                    np.vstack([xf * [1, 1, 0.5], xf * [1, 1, 0.5] + [0, 0, 0.5]])[::-1])
lam2 = np.concatenate([sym[sf.BORN_DERIV]] * 2)[::-1]
with sf.update("zno_nest.h5", big) as f:
    sf.write_piezo_datasets(f, dict(sym, **{sf.BORN_DERIV: lam2}))
"""

ZNO_ALAT = 1.88972612462577
ZNO_CELL = np.array(
    [
        [3.235859326375770, 0.0, 0.0],
        [-1.617929663187880, 2.802336379714220, 0.0],
        [0.0, 0.0, 5.224712025937350],
    ]
)
DISPLACE_ZNO = """&displace
1
 0.004 -0.002 0.006
 -0.003 0.005 -0.004
 0.002 0.003 0.010
 -0.005 -0.001 -0.008
/"""


def zno_input(name, sfile, umn, efield, restart=False):
    text = open("zno_t2.in").read()
    text = text.replace("PREFIX = zno_t2", "PREFIX = %s\n  VERBOSITY = 2" % name)
    text = text.replace(
        "RELAX_STR = 1", "RELAX_STR = 4" + ("\n  RESTART_QHA = 1" if restart else "")
    )
    text = text.replace(
        "MAX_STR_ITER = 20",
        "MAX_STR_ITER = 1\n  STRAIN_COUPLING = 0\n  STRAINFILE = %s" % sfile,
    )
    text = text.replace(
        "TMIN = 100; TMAX = 300; DT = 200",
        "TMIN = 300; TMAX = 300" + ("\n  ALLOW_UNCONVERGED = 1" if restart else ""),
    )
    if not restart and os.path.exists(name + ".qha.h5"):
        os.remove(name + ".qha.h5")  # else RESTART_QHA would read it
    text = re.sub(
        r"EFIELD = .*", "EFIELD = " + " ".join("%.10f" % x for x in efield), text
    )
    strain = "\n".join(" %.12e %.12e %.12e" % tuple(r) for r in umn)
    return text + "&strain\n%s\n/\n%s\n" % (strain, DISPLACE_ZNO)


def test_fe_qha(anphonbin):
    # FE for the iterative QHA with Fourier interpolation: ZnO with its full anharmonic
    # IFCs (3x3x2 cubic/quartic supercell) and zno_ref.h5, KMESH_INTERPOLATE = 2 2 1 <
    # KMESH_QHA = 4 4 2 (= the &kpoint mesh), RELAX_STR = 4, one step at a displaced and
    # strained structure: dF_total/du == the QHA stress for zz and xy, and the force.
    # (QHA of cubic BaTiO3 has imaginary modes, which F_vib skips.)
    umn0 = np.array(
        [[0.003, 0.001, 0.0], [0.001, -0.002, 0.0005], [0.0, 0.0005, 0.004]]
    )
    lbl_g = "ZSISA inputs: QHA stress dF/du_mn [Ry] :"
    lbl_q = "ZSISA inputs: QHA force (all Gamma modes) :"

    def run(name, umn, displace=DISPLACE_ZNO):
        text = zno_input(name, "zno_ref.h5", umn, np.zeros(3))
        text = text.replace(DISPLACE_ZNO, displace)
        text = text.replace(
            "FCSFILE = zno_tiny_anharm.xml",
            "FCSFILE = ZnO332_500K_cutoff1208_nbody233.xml",
        )
        text = text.replace("KMESH_INTERPOLATE = 4 4 2", "KMESH_INTERPOLATE = 2 2 1")
        text = text.replace("  8 8 8", "  4 4 2").replace(
            "COORD_CONV_TOL = 1.0e-10", "COORD_CONV_TOL = 10.0"
        )
        rc, log = run_anphon(anphonbin, name, text)
        if rc or "Structural optimization converged" not in log or "negative" in log:
            return None
        th = np.loadtxt(name + ".qha_thermo", ndmin=2)[-1]
        return dict(
            F=th[-2], G=last_values(log, lbl_g).reshape(3, 3), g=last_values(log, lbl_q)
        )

    base = run("feq_0", umn0)
    if base is None:
        return check(False, "FE QHA: base run failed (or imaginary modes)")
    info = 0
    for label, (m, n) in (("zz", (2, 2)), ("xy", (0, 1))):
        e = np.zeros((3, 3))
        e[m, n] = e[n, m] = 1.0
        d = []
        for i, h in enumerate((1.0e-3, 5.0e-4)):
            f = [run("feq_%s%d%+d" % (label, i, s), umn0 + s * h * e) for s in (1, -1)]
            if None in f:
                return info + check(False, "FE QHA: strain runs failed")
            d.append((f[0]["F"] - f[1]["F"]) / (2.0 * h))
        d = (4.0 * d[1] - d[0]) / 3.0
        g = np.sum(base["G"] * e)
        info += check(
            abs(d - g) < 1.0e-6 * max(1.0, abs(base["G"]).max() / 1.0e-2),
            "FE QHA %s: dF_total/du = %.8e == G = %.8e (extrapolated)" % (label, d, g),
        )
    # the force: Zn1 z against O1 z (equal displacement of the centre of mass kept)
    disp0 = np.array([r.split() for r in DISPLACE_ZNO.splitlines()[2:6]], float)
    mass = np.array([65.38, 65.38, 15.999, 15.999])
    dvec = np.zeros((4, 3))
    dvec[0, 2], dvec[2, 2] = 1.0, -mass[0] / mass[2]
    ratio = []
    for i, h in enumerate((2.0e-3, 1.0e-3)):
        f, q = {}, {}
        for sign in (1, -1):
            d = disp0 + sign * h * dvec
            disp = "&displace\n1\n%s\n/" % "\n".join(
                " %.12e %.12e %.12e" % tuple(x) for x in d
            )
            r = run("feq_q%d%+d" % (i, sign), umn0, disp)
            if r is None:
                return info + check(False, "FE QHA: displacement runs failed")
            f[sign], q[sign] = r["F"], step_row("step_q0.txt", 0)
        ratio.append((f[1] - f[-1]) / (base["g"] @ (q[1] - q[-1])))
    ratio = (4.0 * ratio[1] - ratio[0]) / 3.0
    info += check(
        abs(ratio - 1.0) < 5.0e-4,
        "FE QHA force: Delta F_total / (g . Delta q0) = %.6f (extrapolated)" % ratio,
    )
    return info


def test_zno_piezo(anphonbin, py, tools_dir):
    # ZnO (P6_3mc): e0, B and Lambda all have symmetry-allowed nonzero components,
    # so their field terms are tested through the production path (anphon's
    # space-group average on): RELAX_STR = 4, one step, sheared strain, generic E.
    res = subprocess.run(
        [
            py,
            "-c",
            ZNO_SCRIPT,
            tools_dir,
            " ".join(map(str, (ZNO_ALAT * BOHR * ZNO_CELL).ravel())),
        ],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        print(res.stderr)
        return check(False, "Z: building the ZnO containers")
    e0 = np.load("zno_sym_e0.npy") / C_M2_PER_E_BOHR2
    bb = np.load("zno_sym_b.npy") / C_M2_PER_E_BOHR2
    lam = np.load("zno_sym_lam.npy")
    efield = np.array([0.006, -0.004, 0.01])
    umn = np.array(STRAIN_SHEAR.split(), float).reshape(3, 3)
    omega = abs(np.linalg.det(ZNO_ALAT * ZNO_CELL))
    e_ry = efield * EV_A_TO_RY

    def run(name, sfile, u=umn, nprocs=1, restart=False):
        rc, log = run_anphon(
            anphonbin, name, zno_input(name, sfile, u, efield, restart), nprocs
        )
        return (log if rc == 0 else None), step_row("step_u0.txt", 0).reshape(
            -1, 3
        ) if rc == 0 else None

    log0, _ = run("z_0", "zno_ref.h5")
    log, u0 = run("z_nl", "zno_nl.h5")
    if log0 is None or log is None:
        return check(False, "Z: anphon failed")
    info = check(
        "averaged over the 12 symmetry operations" in log and "changes by" not in log,
        "Z: 12 operations; the symmetric data are left unchanged (no warning)",
    )
    dt_ref = omega * (
        np.einsum("ikl,kl->i", e0, umn)
        + 0.5 * np.einsum("ijklm,jk,lm->i", bb, umn, umn)
    ) + np.einsum("kibmn,mn,kb->i", lam, umn, u0)
    dg_ref = (
        -omega * np.einsum("i,imn->mn", e_ry, e0 + np.einsum("imnpq,pq->imn", bb, umn))
        - np.einsum("i,kibmn,kb->mn", e_ry, lam, u0)
    ).ravel()
    ddt = last_values(log, LBL_DT) - last_values(log0, LBL_DT)
    dv = last_values(log, LBL_V0)[0] - last_values(log0, LBL_V0)[0]
    dg = last_values(log, LBL_G) - last_values(log0, LBL_G)
    info += check(
        np.allclose(ddt, dt_ref, rtol=1.0e-6, atol=1.0e-6 * np.abs(dt_ref).max()),
        "Z: dt(nl) - dt(0) %s == Omega (e0:u + B:u:u/2) + sum (Lambda:u) u0 %s"
        % (ddt, dt_ref),
    )
    info += check(
        close(dv, -e_ry @ ddt, 1.0e-8),
        "Z: V0(nl) - V0(0) = %.10e == -E0.(dt(nl) - dt(0)) = %.10e" % (dv, -e_ry @ ddt),
    )
    info += check(
        np.allclose(dg, dg_ref, rtol=1.0e-6, atol=1.0e-6 * np.abs(dg_ref).max()),
        "Z: G(nl) - G(0) == -Omega E0.(e0 + B:u) - E0.Lambda u0 (max dev %.1e of %.1e)"
        % (np.abs(dg - dg_ref).max(), np.abs(dg_ref).max()),
    )
    # symmetric shear u_xy = u_yx += +-h: the field energy against G_xy + G_yx
    h = 1.0e-4
    ef = {}
    for sign in (1, -1):
        us = umn.copy()
        us[0, 1] += sign * h
        us[1, 0] += sign * h
        if run("z_s%+d" % sign, "zno_nl.h5", us)[0] is None:
            return info + check(False, "Z: shear finite-difference run failed")
        ef[sign] = np.loadtxt("z_s%+d.polarization" % sign, ndmin=2)[-1, 7]
    fd = (ef[1] - ef[-1]) / (2.0 * h)
    ref = dg_ref[1] + dg_ref[3]
    info += check(
        close(fd, ref, 1.0e-5),
        "Z: d(E_field)/d(u_xy = u_yx) = %.10e == G_xy + G_yx of the field terms = %.10e"
        % (fd, ref),
    )

    labels = (LBL_V0, LBL_G, LBL_Q, LBL_DT)
    # the raw (unsymmetrized) data give the run of their average, with the warning
    log_r, _ = run("z_raw", "zno_raw.h5")
    dev = (
        max(
            np.abs(last_values(log_r, lbl) - last_values(log, lbl)).max()
            / np.abs(last_values(log, lbl)).max()
            for lbl in labels
        )
        if log_r
        else 1.0
    )
    warned = re.findall(r"^ (\S+) changes by", log_r or "", re.M)
    info += check(
        dev < 1.0e-9 and warned == ["e0", "B", "Lambda"],
        "Z: raw e0, B, Lambda == their space-group average (rel. dev %.1e), warned: %s"
        % (dev, warned),
    )
    # nested, permuted /ReferenceCell: bit-identical
    log_n, _ = run("z_nest", "zno_nest.h5")
    info += check(
        log_n is not None
        and "translation images averaged" in log_n
        and all(
            np.array_equal(last_values(log, lbl), last_values(log_n, lbl))
            for lbl in labels
        ),
        "Z: nested + permuted /ReferenceCell gives bit-identical V0, G, forces and dt",
    )
    if shutil.which("mpirun") is None:
        print("  skip   Z: mpirun not found, MPI comparison skipped")
    else:
        log_m, _ = run("z_mpi", "zno_nl.h5", nprocs=4)
        info += check(
            log_m is not None
            and all(
                np.allclose(
                    last_values(log, lbl),
                    last_values(log_m, lbl),
                    rtol=1.0e-9,
                    atol=1.0e-14,
                )
                for lbl in labels
            ),
            "Z: mpirun -np 4 V0, G, forces and dt == serial (zno_nl.h5)",
        )
    # restarts from z_nl: without B rejected; the raw container (same average) accepted
    rc, log_x = run_anphon(
        anphonbin, "z_nl", zno_input("z_nl", "zno_eL.h5", umn, efield, True)
    )
    info += check(
        rc != 0
        and "second-order clamped-ion piezoelectric tensor" in log_x
        and "not compatible" in log_x,
        "Z: restart without B rejected (with the note on unsymmetrized restart files)",
    )
    rc, log_x = run_anphon(
        anphonbin, "z_nl", zno_input("z_nl", "zno_raw.h5", umn, efield, True)
    )
    info += check(
        rc == 0 and "RESTART_QHA is true" in log_x,
        "Z: restart with the raw container (same space-group average) accepted",
    )
    return info


def test_missing_e0_warning(anphonbin):
    # ZnO (no inversion), strained cell (RELAX_STR = 4, couplings from the IFCs),
    # EFIELD != 0 and no e0: the warning fires. Reuses the T2 files.
    text = open("zno_t2.in").read()
    text = text.replace("PREFIX = zno_t2", "PREFIX = zno_warn")
    text = text.replace("RELAX_STR = 1", "RELAX_STR = 4")
    text = text.replace("MAX_STR_ITER = 20", "MAX_STR_ITER = 1\n  STRAIN_COUPLING = 0")
    text = text.replace("TMIN = 100; TMAX = 300; DT = 200", "TMIN = 300; TMAX = 300")
    text += "&strain\n 0 0 0\n 0 0 0\n 0 0 0\n/\n"
    rc, log = run_anphon(anphonbin, "zno_warn", text)
    return check(
        "may be missing" in log,
        "missing-e0 warning for ZnO (no inversion) under EFIELD with a strained cell"
        + ("" if rc == 0 else " (run exit %d)" % rc),
    )


if __name__ == "__main__":
    build_dir = os.getcwd()
    project_root = os.path.dirname(build_dir)
    anphonbin = "%s/_build/anphon/anphon" % project_root
    bto_dir = os.path.join(project_root, "example/BaTiO3/scph_relax")
    zno_dir = os.path.join(project_root, "example/ZnO/qha_relax")

    workdir = os.path.join(project_root, "test/efield")
    shutil.rmtree(workdir, ignore_errors=True)
    os.mkdir(workdir)
    os.chdir(workdir)
    shutil.copy(
        os.path.join(
            project_root, "example/BaTiO3/anharm_IFCs/4_optimize/reference/cBTO222.h5"
        ),
        ".",
    )
    shutil.copy(os.path.join(bto_dir, "BORNINFO"), ".")
    shutil.copytree(
        os.path.join(bto_dir, "reference_for_test/strain_phonon"), "strain_phonon"
    )

    # Phase 1b is skipped only without the optional dependency (a python with
    # h5py and ase); a failing strainfile.py step fails the suite.
    info = 0
    py = find_h5py_python()
    have_containers = False
    if py is None:
        print(
            "  skip   phase-1b tests (V1-V3, POL_REF/e0 restarts): no python with h5py"
            " and ase (set STRAINKIT_PYTHON)"
        )
    else:
        try:
            build_containers(py, os.path.join(project_root, "tools"))
            build_nl_containers(py, os.path.join(project_root, "tools"))
            have_containers = True
            STRAIN_SOURCE[0] = "STRAINFILE = bto.h5"
        except RuntimeError as exc:
            print(exc)
            info += check(False, "phase 1b: building the strain-coupling containers")

    stages = [
        ("T3", test_t3, (anphonbin,)),
        (
            "T4/T5",
            test_t4_t5,
            (anphonbin, os.path.join(project_root, "tools/efield_seed.py")),
        ),
        ("T6", test_t6, (anphonbin,)),
    ]
    if have_containers:
        stages += [
            ("V1", test_v1, (anphonbin,)),
            ("W1", test_w1, (anphonbin,)),
            ("S", test_symmetrize, (anphonbin,)),
            ("V2", test_v2, (anphonbin,)),
            ("W4", test_w4, (anphonbin,)),
            ("1b errors", test_1b_errors, (anphonbin,)),
            ("phase-2 restarts", test_2_restarts, (anphonbin,)),
            ("V3/W2", test_v3, (anphonbin,)),
            ("FE", test_fe, (anphonbin,)),
            ("W3", test_w3, (anphonbin,)),
            (
                "cell",
                test_cell_relax,
                (anphonbin, os.path.join(project_root, "tools/efield_seed.py")),
            ),
            ("ZSISA", test_zsisa, (anphonbin,)),
        ]
    stages += [
        ("T2", test_t2, (anphonbin, zno_dir)),
        ("missing e0", test_missing_e0_warning, (anphonbin,)),
    ]
    if have_containers:
        stages.append(
            ("Z", test_zno_piezo, (anphonbin, py, os.path.join(project_root, "tools")))
        )
        stages.append(("FE QHA", test_fe_qha, (anphonbin,)))
    for label, func, args in stages:
        t0 = time.time()
        print("%s:" % label)
        info += func(*args)
        print("  (%.1f s)" % (time.time() - t0))

    print("EFIELD --> %s" % ("pass" if info == 0 else "failed"))
    sys.exit(0 if info == 0 else 1)
