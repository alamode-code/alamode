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
                legacy-text restart are rejected; a restart with the same
                EFIELD is accepted.
  seed        : tools/efield_seed.py output restarts the sweep from the
                relaxed structure (&displace / &strain).

Phase 1b (fixed-voltage model; needs a python with h5py and ase for
tools/strainfile.py: STRAINKIT_PYTHON, ~/miniforge3/envs/qmpy2, or the running
one; skipped otherwise). The BaTiO3 runs read the container bto.h5 packed from
strain_phonon (STRAINFILE, STRAIN_COUPLING = 25); bto_piezo.h5 adds a fake
generic clamped-ion e0 with strainfile.py piezo --voigt.
  V1 identity : RELAX_STR = 4, sheared strain, one step, generic E: the V0 and
                9-gradient differences between the containers with and without
                e0 are -Omega_ref E0 . e0:u and -Omega_ref E0_i e0_imn.
  V2 E = 0    : e0 + POL_REF leave the steps and V0 bit-identical; P equals
                F (d_ref + A + B) / (Omega_ref det F).
  V3 Maxwell  : d(dt_z)/du_mn |_E = -dG_mn/dE_z |_u for zz and xy (G_xy + G_yx)
                with tightly relaxed internal coordinates, two step sizes.
  1b errors   : POL_REF / e0 restart mismatches both ways, legacy text restarts,
                POL_REF without BORNINFO; the missing-e0 warning for ZnO.

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
    ]
    # diag(-d, 0, +d) added to Ti and subtracted from Ba: the acoustic sum rule
    # and any linear checksum weighted by the flat index are unchanged
    born = np.loadtxt("BORNINFO")
    d = 0.01
    born[6, 0] -= d
    born[8, 2] += d
    born[3, 0] += d
    born[5, 2] -= d
    np.savetxt("BORNINFO_mod", born, fmt="%16.8f")
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


def piezo_e_bohr2():
    e = np.zeros((3, 3, 3))
    for c, (j, k) in enumerate(VOIGT_PAIRS):
        e[:, j, k] = e[:, k, j] = PIEZO_FAKE[:, c]
    return e / C_M2_PER_E_BOHR2


def find_h5py_python():
    """A python with h5py for the strainkit tools, or None."""
    for py in (
        os.environ.get("STRAINKIT_PYTHON"),
        os.path.expanduser("~/miniforge3/envs/qmpy2/bin/python"),
        sys.executable,
    ):
        if py and os.path.exists(py):
            rc = subprocess.run([py, "-c", "import h5py, ase"], capture_output=True)
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
LBL_DT = "dipole dt = Omega_ref e0:u + sum_k Z*_k u0_k [e Bohr] ="
LBL_V0 = "V0 at this structure [Ry] ="


def test_v1(anphonbin):
    # Fixed sheared strain, one step, generic E: the containers with and without
    # e0 differ by -Omega_ref E0 . e0:u in V0 and -Omega_ref E0_i e0_imn in G.
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
    omega = OMEGA_REF[0]
    umn = np.array(STRAIN_SHEAR.split(), float).reshape(3, 3)
    e_ry = efield * EV_A_TO_RY
    e0 = piezo_e_bohr2()
    dv_ref = -omega * np.einsum("i,ikl,kl->", e_ry, e0, umn)
    dg_ref = -omega * np.einsum("i,imn->mn", e_ry, e0).ravel()
    dv = last_values(log1, LBL_V0)[0] - last_values(log0, LBL_V0)[0]
    dg = last_values(log1, LBL_G) - last_values(log0, LBL_G)
    info += check(
        close(dv, dv_ref, 1.0e-9),
        "V1: V0(e0) - V0(0) = %.10e == -Omega E0.e0:u = %.10e" % (dv, dv_ref),
    )
    info += check(
        np.allclose(dg, dg_ref, rtol=1.0e-9, atol=1.0e-12 * np.abs(dg_ref).max()),
        "V1: G(e0) - G(0) == -Omega E0_i e0_imn (max dev %.1e of %.1e)"
        % (np.abs(dg - dg_ref).max(), np.abs(dg_ref).max()),
    )
    r0 = np.loadtxt("v1_0.polarization", ndmin=2)[-1]
    r1 = np.loadtxt("v1_p.polarization", ndmin=2)[-1]
    info += check(
        close(r1[7] - r0[7], dv_ref, 1.0e-8),
        "V1: E_field column difference %.10e == -E0.A" % (r1[7] - r0[7]),
    )
    # symmetric shear u_xy = u_yx += +-h: the finite difference of the field
    # energy (same q0, so only -E0.A changes) against G_xy + G_yx of the e0 term
    h = 1.0e-4
    ef = {}
    for sign in (1, -1):
        umn_s = umn.copy()
        umn_s[0, 1] += sign * h
        umn_s[1, 0] += sign * h
        name = "v1_s%+d" % sign
        common["strain"] = "\n".join(" %.12e %.12e %.12e" % tuple(r) for r in umn_s)
        rc, _ = run_anphon(
            anphonbin, name, bto_input(name, strainfile="bto_piezo.h5", **common)
        )
        if rc:
            return info + check(False, "V1: shear finite-difference run failed")
        ef[sign] = np.loadtxt(name + ".polarization", ndmin=2)[-1, 7]
    fd = (ef[1] - ef[-1]) / (2.0 * h)
    ref = dg_ref[1] + dg_ref[3]
    info += check(
        close(fd, ref, 1.0e-6),
        "V1: d(E_field)/d(u_xy = u_yx) = %.10e == G_xy + G_yx of the e0 term = %.10e"
        % (fd, ref),
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


def test_v3(anphonbin):
    # Maxwell identity of the implemented free energy at fixed strain, internal
    # coordinates relaxed: d(dt_i)/du_mn |_E = -dG_mn/dE_i |_u (i = z).
    efield = np.array([0.004, -0.003, 0.01])
    umn0 = np.array(STRAIN_SHEAR.split(), float).reshape(3, 3)
    tight = dict(
        relax_str=4,
        max_iter=60,
        verbosity=2,
        strainfile="bto_piezo.h5",
        coord_tol="1.0e-9",
        extra="\n  GRADIENT_CONV_TOL = 1.0e-9",
    )

    def relax(name, umn, e, displace):
        # a PREFIX.scph.h5 left by an earlier run would turn this into a restart
        for f in (name + ".scph.h5", name + ".polarization"):
            if os.path.exists(f):
                os.remove(f)
        strain = "\n".join(" %.12e %.12e %.12e" % tuple(r) for r in umn)
        rc, log = run_anphon(
            anphonbin,
            name,
            bto_input(
                name,
                efield=" ".join("%.12e" % x for x in e),
                strain=strain,
                displace=displace,
                **tight,
            ),
        )
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

    info = 0
    ez = np.array([0.0, 0.0, 1.0])
    results = []
    # The E side is the less linear one (soft polar mode): its central
    # difference error is O(h_E^2) (ratio ~4 per halving from h_E = 2e-3 down to
    # 2.5e-4 eV/A). Extrapolated to h_E -> 0 a residual of ~2e-4 (zz) and ~1e-4
    # (xy) remains; it does not change with a 10x tighter TOL_SCPH, so it is not
    # SCP convergence noise but the consistency of G with F at this level.
    for h_u, h_e in ((2.0e-4, 5.0e-4), (1.0e-4, 2.5e-4)):
        g = {}
        for sign in (1, -1):
            ok, log = relax(
                "v3_e%+d_%.0e" % (sign, h_e), umn0, efield + sign * h_e * ez, seed
            )
            if not ok:
                return check(False, "V3: E run did not converge")
            g[sign] = last_values(log, LBL_G).reshape(3, 3)
        dg_de = (g[1] - g[-1]) / (2.0 * h_e * EV_A_TO_RY)
        for label, (m, n) in (("zz", (2, 2)), ("xy", (0, 1))):
            dts = {}
            for sign in (1, -1):
                umn = umn0.copy()
                umn[m, n] += sign * h_u
                if m != n:
                    umn[n, m] += sign * h_u
                ok, log = relax("v3_%s%+d_%.0e" % (label, sign, h_u), umn, efield, seed)
                if not ok:
                    return check(False, "V3: strain run did not converge")
                dts[sign] = last_values(log, LBL_DT)
            lhs = (dts[1][2] - dts[-1][2]) / (2.0 * h_u)
            rhs = -dg_de[m, n] - (dg_de[n, m] if m != n else 0.0)
            results.append((label, h_u, h_e, lhs, rhs))
    for label, h_u, h_e, lhs, rhs in results:
        rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs))
        print(
            "         V3 %s: h_u = %.1e, h_E = %.1e eV/A: d(dt_z)/du = %.10e, "
            "-dG/dE_z = %.10e, rel. diff %.2e" % (label, h_u, h_e, lhs, rhs, rel)
        )
        info += check(
            rel < 5.0e-4, "V3 %s (h_u = %.1e): Maxwell identity" % (label, h_u)
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
            ("V2", test_v2, (anphonbin,)),
            ("1b errors", test_1b_errors, (anphonbin,)),
            ("V3", test_v3, (anphonbin,)),
        ]
    stages += [
        ("T2", test_t2, (anphonbin, zno_dir)),
        ("missing e0", test_missing_e0_warning, (anphonbin,)),
    ]
    for label, func, args in stages:
        t0 = time.time()
        print("%s:" % label)
        info += func(*args)
        print("  (%.1f s)" % (time.time() - t0))

    print("EFIELD --> %s" % ("pass" if info == 0 else "failed"))
    sys.exit(0 if info == 0 else 1)
