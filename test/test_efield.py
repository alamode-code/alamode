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

Run from the build directory: python3 ../test/test_efield.py
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
  COORD_CONV_TOL = 1.0e-5
  CELL_CONV_TOL = 5.0e-7
  SET_INIT_STR = 1
  ADD_HESS_DIAG = 50.0
  STRAIN_COUPLING = 25
  STRAIN_IFC_DIR = strain_phonon
  {efield}
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
        efield="" if efield is None else "EFIELD = " + efield,
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
        len(rows) == 2 and np.all(rows[:, 5] == 1),
        "T2: relaxation converged at 100 and 300 K",
    )
    info += check(
        np.allclose(rows[0, 1:5], rows[1, 1:5], rtol=1.0e-6, atol=1.0e-8),
        "T2: Delta P and field energy are T-independent: %s / %s"
        % (rows[0, 1:4], rows[1, 1:4]),
    )
    row = rows[-1]
    info += check(
        all(close(a, b, 1.0e-5, 1.0e-8) for a, b in zip(row[1:4], p_ref)),
        "T2: Delta P %s == (1/Omega) Z^T Phi^+ Z E %s" % (row[1:4], p_ref),
    )
    info += check(
        close(row[4], e_ref, 1.0e-5), "T2: field energy %.9e == %.9e" % (row[4], e_ref)
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

    pol, ftot = {}, {}
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
    pp, pm = pol["t4_pz"], pol["t4_mz"]
    info += check(pp[5] == 1 and pm[5] == 1, "T4: both +E and -E runs converged")
    info += check(
        pp[3] > 1.0e-3
        and np.allclose(pp[1:4], -pm[1:4], rtol=1.0e-6, atol=1.0e-8)
        and close(pp[4], pm[4], 1.0e-6)
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

    # T5 on the +z run
    log = open("t4_pz.log").read()
    zstar = read_born("BORNINFO")
    u0 = np.loadtxt("t4_pz.atom_disp", ndmin=2)[-1, 1:]
    umn = np.loadtxt("t4_pz.umn_tensor", ndmin=2)[-1, 1:].reshape(3, 3)
    omega = volume_ref(log) * np.linalg.det(np.eye(3) + umn)
    dipole = np.einsum("kab,kb->a", zstar, u0.reshape(-1, 3))
    p_ref = dipole / omega * EBOHR2_TO_UC_CM2
    e_ref = -np.array([0.0, 0.0, 0.01]) * EV_A_TO_RY @ dipole
    info += check(
        np.allclose(pp[1:4], p_ref, rtol=1.0e-5, atol=1.0e-8),
        "T5: Delta P %s == (1/Omega) Z* u0 %s" % (pp[1:4], p_ref),
    )
    info += check(
        close(pp[4], e_ref),
        "T5: field energy %.9e == -E.sum Z*u0 %.9e" % (pp[4], e_ref),
    )

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

    info = 0
    for label, func, args in (
        ("T3", test_t3, (anphonbin,)),
        (
            "T4/T5",
            test_t4_t5,
            (anphonbin, os.path.join(project_root, "tools/efield_seed.py")),
        ),
        ("T6", test_t6, (anphonbin,)),
        ("T2", test_t2, (anphonbin, zno_dir)),
    ):
        t0 = time.time()
        print("%s:" % label)
        info += func(*args)
        print("  (%.1f s)" % (time.time() - t0))

    print("EFIELD --> %s" % ("pass" if info == 0 else "failed"))
    sys.exit(0 if info == 0 else 1)
