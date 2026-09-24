#!/usr/bin/env python
"""Regression test: BUBBLE = 4, the curvature of the SCP free energy.

A 1x1x2 cell of cubic BaTiO3 (KMESH_SCPH = KMESH_INTERPOLATE = 1 1 1, so the
Z point of the primitive cell folds onto Gamma and the cubic couplings
between Gamma and Z do not vanish).

- Relaxed with RELAX_STR = 2 from a small random P1 displacement (no symmetry
  projection in the SCP loop), the analytic Jacobian of the SCP force and
  stress (static bubble + quartic ladder, solved by GMRES; displacements and
  the six strain components) must match central finite differences of them
  (BUBBLE_FD_CHECK = 1), block by block, and be symmetric; without the ladder
  (BUBBLE_LADDER = 0) the finite differences must NOT be matched.
- At the cubic structure in its 30 K, -4 GPa cell (fixed, RELAX_STR = 4,
  reached from 300 K; a saddle the SCP loop keeps, all SCPH frequencies
  real) the curvature must be negative along one direction, exported to
  PREFIX.scph_hessian_displace as a P4mm distortion that a new run reads back
  exactly.
- BUBBLE_HESS = 1 at 300 K: Newton with the full Hessian (strain block) from
  the P1 start reaches the default minimum (atoms and cell) in no more steps,
  also on 2 ranks, and BFGS with the projected Hessian (the SCP loop keeps
  P4mm) returns a small P4mm displacement to the cubic structure.
- At a polar minimum (9 % tetragonal strain, 100 K) the explicit
  stress-displacement block, differentiated independently
  (BUBBLE_FD_CHECK = 2), must equal the transposed force-strain block that J
  uses.
- A 2-rank run (batched V4 contraction over distributed rows) must reproduce
  the serial curvature, and its ordinary SCPH outputs (no FD check) must equal
  those of the FD-checked run, whose SCP state is restored.
"""

import os
import random
import re
import shutil
import subprocess
import sys

import numpy as np

WORKDIR = "scph_hessian"
A = 7.53159676409  # Bohr, cubic BaTiO3 fixture


def run_anphon(cmd, logfile):
    with open(logfile, "w") as f:
        return subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT, timeout=1800
        ).returncode


def write_input(prefix, extra, temp=300, pressure=None, rattle=True):
    with open("BTO_scph_thermo.in") as f:
        src = f.read()
    src = re.sub(r"PREFIX\s*=\s*\S+", "PREFIX = %s" % prefix, src, count=1)
    src = re.sub(r"TMIN\s*=\s*\S+", "TMIN = %g" % temp, src, count=1)
    src = re.sub(r"TMAX\s*=\s*\S+", "TMAX = %g" % temp, src, count=1)
    if pressure is not None:
        src = src.replace("&relax", "&relax\n  STAT_PRESSURE = %g" % pressure, 1)
    src = re.sub(r"KMESH_INTERPOLATE\s*=.*", "KMESH_INTERPOLATE = 1 1 1", src, count=1)
    src = re.sub(r"KMESH_SCPH\s*=.*", "KMESH_SCPH = 1 1 1", src, count=1)
    src = re.sub(
        r"MAXITER\s*=\s*\S+", "MAXITER = 5000\n  TOL_SCPH = 1.0e-12", src, count=1
    )
    src = re.sub(
        r"RELAX_STR\s*=\s*\S+", "RELAX_STR = 2\n  BUBBLE = 4\n" + extra, src, count=1
    )
    head = src.split("&displace")[0]
    head = head.replace(
        "&scph",
        "&cell\n 1.0\n %.10f 0.0 0.0\n 0.0 %.10f 0.0\n 0.0 0.0 %.10f\n/\n&scph"
        % (A, A, 2 * A),
        1,
    )
    rng = random.Random(11)
    lines = [
        "&displace",
        "0",
        "  %.11f" % A,
        "  1.0 0.0 0.0",
        "  0.0 1.0 0.0",
        "  0.0 0.0 2.0",
    ]
    for _ in range(10):
        amp = 2e-3 if rattle else 0.0
        lines.append(" ".join("%.6e" % (rng.uniform(-1, 1) * amp) for _ in range(3)))
    lines.append("/\n\n&kpoint\n  2\n  8 8 4\n/\n")
    with open(prefix + ".in", "w") as f:
        f.write(head + "\n".join(lines))


def fd_mismatch(logfile, block=None, relative=False):
    """Scaled max |J - J_fd|: overall, or of one block (qq, qu, uq, uu) relative
    to the whole matrix or (relative=True) to that block."""
    with open(logfile) as f:
        text = f.read()
    m = re.search(r"BUBBLE_FD_CHECK: max \|J - J_fd\| / max \|J\| = (\S+)", text)
    if block is not None:
        head = r"relative to the block \(" if relative else r"max \|J - J_fd\|.*?\("
        m = re.search(head + r"blocks .*?\b%s (\S+?)[,;]" % block, text)
    return float(m.group(1)) if m else None


def reciprocity(logfile):
    """(difference, size) of the last explicit stress-displacement check."""
    with open(logfile) as f:
        found = re.findall(
            r"explicit stress-displacement block .*? = (\S+) \(max (\S+)\)", f.read()
        )
    return tuple(float(x) for x in found[-1]) if found else (None, None)


def curvature_rows(prefix):
    """Mode rows (index, SCPH, free energy) of the last temperature block."""
    with open(prefix + ".scph_hessian") as f:
        block = f.read().split("# T = ")[-1]
    rows = [
        line.split()
        for line in block.splitlines()[1:]
        if line.strip() and not line.startswith("#")
    ]
    return block, np.array(rows, dtype=float)


def curvature(prefix):
    """(asymmetry, free-energy curvature frequencies) of the last temperature block."""
    block, rows = curvature_rows(prefix)
    asym = float(re.search(r"asymmetry = (\S+)", block).group(1))
    return asym, rows[:, 2]


def relaxed_elastic(prefix):
    """Relaxed-ion elastic curvature (Voigt, GPa) of the last temperature block."""
    block, _ = curvature_rows(prefix)
    lines = block.split("# relaxed ions")[1].splitlines()[1:7]
    return np.array([line.lstrip("#").split() for line in lines], dtype=float)


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

    ok = True
    write_input("full", "  BUBBLE_FD_CHECK = 1\n")
    write_input("noladder", "  BUBBLE_FD_CHECK = 1\n  BUBBLE_LADDER = 0\n")
    for prefix in ("full", "noladder"):
        if run_anphon([anphonbin, prefix + ".in"], prefix + ".log"):
            print("%s run failed, see %s/%s.log" % (prefix, WORKDIR, prefix))
            return 1

    # displacements to the accuracy of the SCP solves; the strain columns also
    # carry the O(h^2) truncation of the strain differences (h = 1e-4)
    # (qu and uq are ~1e-5 of J at this nearly centrosymmetric structure, too
    # small to be checked relative to themselves; the polar case below checks
    # the explicit coupling independently)
    full_qq = fd_mismatch("full.log", "qq", relative=True)
    full_uu = fd_mismatch("full.log", "uu", relative=True)
    full = fd_mismatch("full.log")
    if (
        None in (full_qq, full_uu, full)
        or not full_qq < 1.0e-8
        or not full_uu < 1.0e-6
        or not full < 1.0e-6
    ):
        print(
            "analytic Jacobian vs finite differences: %s (qq %s, uu %s, each relative to its block)"
            % (full, full_qq, full_uu)
        )
        ok = False
    asym, w_full = curvature("full")
    if not asym < 1.0e-8:
        print("the Jacobian is not symmetric: %g" % asym)
        ok = False
    noladder = fd_mismatch("noladder.log")
    if noladder is None or not noladder > 1.0e-3:
        print(
            "without the ladder the finite differences are still matched (%s)"
            % noladder
        )
        ok = False
    print(
        "BUBBLE = 4 vs finite differences, ladder on/off --> %s"
        % ("pass" if ok else "fail")
    )
    if not ok:
        return 1

    # The cubic structure at the cell it takes at 30 K under -4 GPa, held fixed
    # (RELAX_STR = 4; the SCP solve at 30 K is reached from 300 K, LOWER_TEMP):
    # every SCPH frequency is real, but the free energy curves down along the
    # polar mode, and the exported direction must break the symmetry to P4mm.
    write_input("saddle", "", temp=30, pressure=-4, rattle=False)
    with open("saddle.in") as f:
        src = f.read()
    src = src.replace("RELAX_STR = 2", "RELAX_STR = 4\n  LOWER_TEMP = 1", 1)
    src = re.sub(r"TMAX\s*=\s*\S+", "TMAX = 300", src, count=1)
    src = re.sub(r"DT\s*=\s*\S+", "DT = 270", src, count=1)
    src = re.sub(
        r"&strain.*?\n/\n",
        "&strain\n0.027071539788 0.0 0.0\n0.0 0.027071539788 0.0\n"
        "0.0 0.0 0.034980863080\n/\n",
        src,
        count=1,
        flags=re.S,
    )
    with open("saddle.in", "w") as f:
        f.write(src)
    if run_anphon([anphonbin, "saddle.in"], "saddle.log"):
        print("saddle run failed, see %s/saddle.log" % WORKDIR)
        return 1
    _, rows = curvature_rows("saddle")
    w_scph, w_free = rows[:, 1], rows[:, 2]
    if not (w_scph.min() > 0.0 and w_free[0] < 0.0 and w_free[1] > 0.0):
        print("saddle: SCPH %s, free energy %s" % (w_scph[:3], w_free[:3]))
        ok = False
    if not os.path.exists("saddle.scph_hessian_displace"):
        print("saddle: no unstable direction exported")
        ok = False
    else:
        with open("saddle.scph_hessian_displace") as f:
            text = f.read()
        blocks = re.findall(r"&displace\n 1\n(.*?)\n/", text, re.S)
        u = np.array([[float(x) for x in b.split()] for b in blocks])
        if "P4mm (#99)" not in text or u.shape != (1, 30):
            print("saddle: unexpected displacement export\n%s" % text)
            ok = False
        elif not np.isclose(np.abs(u).max(), 0.05):
            print("saddle: displacement not scaled to 0.05 bohr")
            ok = False
    if not ok:
        print("BUBBLE = 4 unstable direction at a saddle --> fail")
        return 1

    # Restart from the exported structure (one step: the SCP solve near this
    # saddle at 30 K is too stiff for a regression-test relaxation). The run
    # must start from exactly the exported displacements and strain, in P4mm,
    # and remove an export left over from an earlier run with its PREFIX.
    with open("saddle.in") as f:
        src = f.read()
    exported = text.split("\n&displace")[1].split("\n\n")[0]
    src = src.replace("PREFIX = saddle", "PREFIX = follow", 1)
    src = src.replace("RELAX_STR = 4\n  LOWER_TEMP = 1", "RELAX_STR = 2", 1)
    src = re.sub(r"TMAX\s*=\s*\S+", "TMAX = 30", src, count=1)
    src = src.replace("MAX_STR_ITER = 1000", "MAX_STR_ITER = 1", 1)
    src = re.sub(r"&strain.*?\n/\n", "", src, count=1, flags=re.S)
    src = src.split("&displace")[0] + "&displace" + exported + "\n\n"
    src += "&kpoint\n  2\n  8 8 4\n/\n"
    with open("follow.in", "w") as f:
        f.write(src)
    shutil.copy("saddle.scph_hessian_displace", "follow.scph_hessian_displace")
    if run_anphon([anphonbin, "follow.in"], "follow.log"):
        print("run from the exported direction failed, see %s/follow.log" % WORKDIR)
        return 1
    with open("follow.log") as f:
        log = f.read()
    u_echo = np.array(
        [
            [float(x) for x in line.split(":")[1].split()]
            for line in log.split("Initial atomic displacements [Bohr] :")[1]
            .split("\n\n")[0]
            .strip()
            .splitlines()
        ]
    ).ravel()
    strain_echo = np.array(
        log.split("Initial strain (displacement gradient tensor u_{mu nu}) :")[1]
        .split("\n\n")[0]
        .split(),
        dtype=float,
    )
    strain_exported = np.array(
        exported.split("&strain")[1].split("/")[0].split(), dtype=float
    )
    if not (
        np.allclose(u_echo, u[0], atol=1e-8)
        and np.allclose(strain_echo, strain_exported, atol=1e-8)
        and "Space group :  P4mm (#99)" in log
    ):
        print("the exported structure is not what the new run starts from")
        ok = False
    if os.path.exists("follow.scph_hessian_displace"):
        print("a stale PREFIX.scph_hessian_displace survived a new run")
        ok = False
    print(
        "BUBBLE = 4 unstable direction at a saddle --> %s" % ("pass" if ok else "fail")
    )
    if not ok:
        return 1

    # BUBBLE_HESS = 1 at 300 K, where the SCP solves are robust (near the 30 K
    # instability, relaxation outcomes changed with the MPI rank count).
    # (a) Newton with the full Hessian (strain block, coupled solve) from the
    #     rattled P1 start: the minimum of the default run, in no more steps.
    write_input("newton", "")
    with open("newton.in") as f:
        src_n = f.read()
    src_n = src_n.replace("RELAX_ALGO = 3", "RELAX_ALGO = 2\n  BUBBLE_HESS = 1", 1)
    src_n = re.sub(r"MIXBETA_COORD\s*=\s*\S+", "MIXBETA_COORD = 1.0", src_n, count=1)
    src_n = re.sub(r"MIXBETA_CELL\s*=\s*\S+", "MIXBETA_CELL = 1.0", src_n, count=1)
    with open("newton.in", "w") as f:
        f.write(src_n)
    if run_anphon([anphonbin, "newton.in"], "newton.log"):
        print("BUBBLE_HESS Newton run failed, see %s/newton.log" % WORKDIR)
        return 1
    with open("newton.log") as f:
        log = f.read()
    with open("full.log") as f:
        steps_default = f.read().count("Structure opt. step")
    u_newton = np.loadtxt("newton.atom_disp", ndmin=2)[-1, 1:]
    u_default = np.loadtxt("full.atom_disp", ndmin=2)[-1, 1:]
    if not (
        "Structural optimization converged" in log
        and "BUBBLE_HESS: optimizer Hessian from the free-energy curvature (with strain)"
        in log
        and log.count("Structure opt. step") <= steps_default
        and np.allclose(u_newton, u_default, rtol=0.0, atol=1.0e-4)
        and np.allclose(
            np.loadtxt("newton.umn_tensor", ndmin=2)[-1, 1:],
            np.loadtxt("full.umn_tensor", ndmin=2)[-1, 1:],
            rtol=0.0,
            atol=1.0e-5,
        )
        and np.linalg.eigvalsh(relaxed_elastic("newton")).min() > 0.0
    ):
        print(
            "BUBBLE_HESS Newton: not converged to the default minimum, see %s/newton.log"
            % WORKDIR
        )
        ok = False
    # (b) BFGS at the fixed saddle cell from a small P4mm displacement (the
    #     exported direction), so that the SCP loop keeps P4mm and the
    #     optimizer Hessian is projected: back to the cubic structure.
    with open("saddle.in") as f:
        src_f = f.read()
    src_f = src_f.replace("PREFIX = saddle", "PREFIX = fixedhess", 1)
    src_f = re.sub(r"TMIN\s*=\s*\S+", "TMIN = 300", src_f, count=1)
    src_f = src_f.replace("&relax", "&relax\n  BUBBLE_HESS = 1", 1)
    small = "\n".join(
        " ".join("%.10e" % (0.2 * float(x)) for x in line.split())
        for line in exported.split("&strain")[0].strip().splitlines()[1:]
        if line.strip() != "/"
    )
    src_f = src_f.split("&displace")[0] + "&displace\n 1\n" + small + "\n/\n\n"
    src_f += "&kpoint\n  2\n  8 8 4\n/\n"
    with open("fixedhess.in", "w") as f:
        f.write(src_f)
    if run_anphon([anphonbin, "fixedhess.in"], "fixedhess.log"):
        print("BUBBLE_HESS fixed-cell run failed, see %s/fixedhess.log" % WORKDIR)
        return 1
    with open("fixedhess.log") as f:
        log = f.read()
    u_fixed = np.loadtxt("fixedhess.atom_disp", ndmin=2)[-1, 1:]
    if not (
        "Structural optimization converged" in log
        and "BUBBLE_HESS: optimizer Hessian from the free-energy curvature" in log
        and np.abs(u_fixed).max() < 1.0e-3
    ):
        print(
            "BUBBLE_HESS fixed cell: not back to cubic, see %s/fixedhess.log" % WORKDIR
        )
        ok = False
    # (c) the Newton run on 2 ranks (the curvature through the V4 service)
    if shutil.which("mpirun") is not None:
        with open("newton_np2.in", "w") as f:
            f.write(src_n.replace("PREFIX = newton", "PREFIX = newton_np2", 1))
        if run_anphon(
            ["mpirun", "-np", "2", anphonbin, "newton_np2.in"], "newton_np2.log"
        ):
            print("2-rank BUBBLE_HESS run failed, see %s/newton_np2.log" % WORKDIR)
            return 1
        with open("newton_np2.log") as f:
            log_np2 = f.read()
        if not (
            "Structural optimization converged" in log_np2
            and "BUBBLE_HESS: optimizer Hessian from the free-energy curvature (with strain)"
            in log_np2
        ):
            print("BUBBLE_HESS: the 2-rank run did not converge with the curvature")
            ok = False
        for ext, tol in ((".atom_disp", 1.0e-6), (".umn_tensor", 1.0e-8)):
            a = np.loadtxt("newton" + ext, ndmin=2)[-1, 1:]
            b = np.loadtxt("newton_np2" + ext, ndmin=2)[-1, 1:]
            if not np.allclose(a, b, rtol=0.0, atol=tol):
                print("BUBBLE_HESS: 2-rank %s differs from the serial run" % ext)
                ok = False
    print("BUBBLE_HESS relaxations --> %s" % ("pass" if ok else "fail"))
    if not ok:
        return 1

    # A polar minimum: the saddle cell stretched to 9 % along z, 100 K reached
    # from 300 K. The explicit stress-displacement block, differentiated
    # independently (BUBBLE_FD_CHECK = 2), must equal the transposed
    # force-strain block that J uses, where that coupling is substantial.
    with open("fixedhess.in") as f:
        src_p = f.read()
    src_p = src_p.replace("PREFIX = fixedhess", "PREFIX = polar", 1)
    src_p = src_p.replace("  BUBBLE_HESS = 1\n", "", 1)
    src_p = src_p.replace("  BUBBLE = 4", "  BUBBLE = 4\n  BUBBLE_FD_CHECK = 2", 1)
    src_p = re.sub(r"TMIN\s*=\s*\S+", "TMIN = 100", src_p, count=1)
    src_p = re.sub(r"DT\s*=\s*\S+", "DT = 200", src_p, count=1)
    src_p = re.sub(
        r"&strain.*?\n/\n",
        "&strain\n-0.01 0.0 0.0\n0.0 -0.01 0.0\n0.0 0.0 0.09\n/\n",
        src_p,
        count=1,
        flags=re.S,
    )
    with open("polar.in", "w") as f:
        f.write(src_p)
    if run_anphon([anphonbin, "polar.in"], "polar.log"):
        print("polar run failed, see %s/polar.log" % WORKDIR)
        return 1
    diff, size = reciprocity("polar.log")
    u_polar = np.abs(np.loadtxt("polar.atom_disp", ndmin=2)[-1, 1:]).max()
    if diff is None or not (u_polar > 0.1 and size > 1.0e-4 and diff < 1.0e-8):
        print(
            "polar: max |u| %g, explicit coupling %s, transpose mismatch %s"
            % (u_polar, size, diff)
        )
        ok = False
    print(
        "explicit coupling blocks at a polar minimum --> %s"
        % ("pass" if ok else "fail")
    )
    if not ok:
        return 1

    if shutil.which("mpirun") is not None:
        write_input("np2", "")
        if run_anphon(["mpirun", "-np", "2", anphonbin, "np2.in"], "np2.log"):
            print("2-rank run failed, see %s/np2.log" % WORKDIR)
            return 1
        _, w_np2 = curvature("np2")
        if w_np2.shape != w_full.shape or not np.allclose(w_np2, w_full, rtol=1e-6):
            print("2-rank curvature differs from the serial one")
            ok = False
        # The finite-difference check restores the SCP state it displaces: the
        # ordinary outputs of the FD run must equal those of this run (FD off).
        for ext in (".scph_thermo", ".atom_disp"):
            a = np.loadtxt("full" + ext)
            b = np.loadtxt("np2" + ext)
            if a.shape != b.shape or not np.allclose(a, b, rtol=1e-6, atol=1e-10):
                print("%s differs between the FD-checked and the plain run" % ext)
                ok = False
        print("BUBBLE = 4 on 2 MPI ranks --> %s" % ("pass" if ok else "fail"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
