#!/usr/bin/env python
"""Regression test of the KS_INPUT mode analysis (linewidth, shift, Sigma(omega),
V3/Phi3/V4/Phi4 listings) against test/mode_analysis/reference.

Run from the build directory: python ../test/test_mode_analysis.py

Checks
  * Gamma/Shift/Self files and the listings of representative targets: identical.
  * Every listing: sum(mult * value) equals the reference (gauge-invariant total),
    and rows/multiplicities agree as sets.
  * Si V3 listings: the linewidth rebuilt from the listing with the Lorentzian
    smearing formula reproduces PREFIX.Gamma (2*Gamma/S = pi). This is the check
    that catches a wrong momentum partner (q - k vs -q - k), which symmetric
    points such as X and L cannot reveal.
"""

import bz2
import gzip
import os
import shutil
import subprocess
import sys

import numpy as np

CASES = {
    "si": {
        "inputs": [
            "si_v3_1.in",
            "si_v3_2.in",
            "si_tetra.in",
            "si_off.in",
            "si_offt.in",
            "si_se_list.in",
            "si_se_path.in",
            "si_se_h5.in",
            "si_se_int.in",
        ],
        "files": [
            "../../../example/Si/reference/si222_cubic.xml.bz2"
        ],  # relative to test/mode_analysis/si
        "nk": 1000,
    },
    "bto": {
        "inputs": ["bto_v4_1.in", "bto_v4_2.in", "bto_off_1.in", "bto_off_2.in"],
        "files": [
            "../../../example/BaTiO3/anharm_IFCs/4_optimize/reference/cBTO222.h5"
        ],
        "nk": 8,
    },
}


def read_rows(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        lines = f.readlines()
    header = [l for l in lines if l.startswith("#")]
    rows = [
        [float(x) for x in l.split()]
        for l in lines
        if l.strip() and not l.startswith("#")
    ]
    return header, np.array(rows)


def value_total(name, a):
    if ".Phi" in name:  # ... re im mult
        return ((a[:, -3] ** 2 + a[:, -2] ** 2) * a[:, -1]).sum()
    if ".V" in name:  # ... value mult
        return (a[:, -2] * a[:, -1]).sum()
    return None


def gamma_from_listing(header, a, nk, eps=2.0, temp=300.0):
    omega0 = float([h for h in header if h.startswith("# Frequency")][0].split()[-1])
    w1, w2, v, m = a[:, 2], a[:, 5], a[:, 6], a[:, 7]
    ok = (w1 > 1e-3) & (w2 > 1e-3)
    w1, w2, v, m = w1[ok], w2[ok], v[ok], m[ok]
    lor = lambda w: eps / np.pi / (w * w + eps * eps)
    bose = lambda w: 1.0 / np.expm1(1.4387769 * w / temp)
    f1, f2 = bose(w1), bose(w2)
    d0 = lor(omega0 - w1 - w2) - lor(omega0 + w1 + w2)
    d1 = lor(omega0 - w1 + w2) - lor(omega0 + w1 - w2)
    return np.pi * (m * v * ((f1 + f2 + 1.0) * d0 - (f1 - f2) * d1)).sum() / nk


def compare_case(case, workdir, refdir, nk):
    nfail = 0
    for ref in sorted(os.listdir(refdir)):
        name = ref[:-3] if ref.endswith(".gz") else ref
        now = os.path.join(workdir, name)
        if not os.path.exists(now):
            print(f"  {case}/{name}: missing output")
            nfail += 1
            continue
        hr, ar = read_rows(os.path.join(refdir, ref))
        hn, an = read_rows(now)
        if ar.shape != an.shape:
            print(f"  {case}/{name}: shape {ar.shape} vs {an.shape}")
            nfail += 1
            continue
        tot = value_total(name, ar)
        if tot is None:
            ok = np.allclose(ar, an, rtol=1e-5, atol=1e-10)
        else:
            tot_now = value_total(name, an)
            ok = abs(tot - tot_now) <= 1e-6 * max(abs(tot), 1e-300)
            # the rows (all columns except the values) must agree as a set: sort rows lexicographically
            nval = 2 if ".Phi" in name else 1
            key_cols = [c for c in range(ar.shape[1] - 1 - nval)] + [ar.shape[1] - 1]
            order = lambda a: a[np.lexsort(a[:, key_cols].T[::-1])][:, key_cols]
            ok &= np.allclose(order(ar), order(an), rtol=1e-6, atol=1e-8)
            if ".V3." in name and case == "si":
                gfile = os.path.join(workdir, name.replace(".V3.", ".Gamma."))
                two_gamma = float(open(gfile).read().split()[-1])
                ratio = two_gamma / (gamma_from_listing(hn, an, nk) / np.pi)
                if abs(ratio - np.pi) > 1e-3:
                    print(f"  {case}/{name}: 2Gamma/S = {ratio:.6f}, expected pi")
                    ok = False
        if not ok:
            print(f"  {case}/{name}: mismatch")
            nfail += 1
    return nfail


def check_offmesh(workdir):
    """Consistency of the off-mesh kernels with the on-mesh ones.

    si_off.in (smearing): on-mesh targets are numbered first (1: (0.1,0.2,0.3), 2: X),
    then the off-mesh ones (3: 1e-7 off (0.1,0.2,0.3), 4: the same plus a reciprocal
    lattice vector, 5: (0.15,0.2,0.3), 6: 1e-7 off X). Smearing is continuous in q, so
    the agreement is tight.

    si_offt.in (tetrahedron): 1: X representative (0,0.5,0.5), 2: (0.1,0.2,0.3) on mesh;
    3: 1e-7 off the X representative, 4: 1e-7 off (0.1,0.2,0.3), 5: (0.15,0.2,0.3).
    The tetrahedron weights are discontinuous where corner energies are exactly
    degenerate at mesh nodes, so a 1e-7 offset changes the result by O(1e-3) at a
    generic point and O(1e-1) at X; only the generic point is checked, loosely. At an
    exact mesh point the two kernels agree to all digits (verified by forcing the
    off-mesh path); the reference files guard that."""
    nfail = 0
    g = lambda n: np.loadtxt(os.path.join(workdir, f"si_off.Gamma.{n}"))[:, 1]
    for a, b, tol, what in (
        (1, 3, 1e-4, "mesh vs 1e-7 off mesh"),
        (3, 4, 1e-8, "q vs q+G"),
        (2, 6, 1e-4, "X on vs off mesh"),
    ):
        rel = np.abs(g(a) - g(b)).max() / np.abs(g(a)).max()
        if rel > tol:
            print(f"  si/si_off: {what}: relative difference {rel:.2e} > {tol:.0e}")
            nfail += 1
    gt = lambda n: np.loadtxt(os.path.join(workdir, f"si_offt.Gamma.{n}"))[:, 1]
    rel = np.abs(gt(2) - gt(4)).max() / np.abs(gt(2)).max()
    if rel > 1e-2:
        print(
            f"  si/si_offt: tetrahedron mesh vs 1e-7 off mesh: relative difference {rel:.2e} > 1e-2"
        )
        nfail += 1
    # REALPART (smearing): tadpole and bubble shifts, generic point (1 vs 3) and X (2 vs 6)
    sh = lambda n: np.loadtxt(os.path.join(workdir, f"si_off.Shift.{n}"))[:, 1:3]
    for a, b, what in ((1, 3, "generic point"), (2, 6, "X")):
        rel = np.abs(sh(a) - sh(b)).max() / np.abs(sh(a)).max()
        if rel > 1e-4:
            print(
                f"  si/si_off: shift {what} mesh vs 1e-7 off mesh: relative difference {rel:.2e} > 1e-4"
            )
            nfail += 1
    # FSTATE_W (smearing): generic point on mesh (1) vs 1e-7 off mesh (3), both channels
    fw = lambda n: np.loadtxt(os.path.join(workdir, f"si_off.fw.{n}"))[:, 1:]
    rel = np.abs(fw(1) - fw(3)).max() / np.abs(fw(1)).max()
    if rel > 1e-4:
        print(
            f"  si/si_off: FSTATE_W mesh vs 1e-7 off mesh: relative difference {rel:.2e} > 1e-4"
        )
        nfail += 1
    fwt = lambda n: np.loadtxt(os.path.join(workdir, f"si_offt.fw.{n}"))[:, 1:]
    rel = np.abs(fwt(2) - fwt(4)).max() / np.abs(fwt(2)).max()
    if (
        rel > 1e-1
    ):  # omega-resolved, hence more sensitive to the tetrahedron discontinuity than Gamma(T)
        print(
            f"  si/si_offt: FSTATE_W (tetrahedron) mesh vs 1e-7 off mesh: relative difference {rel:.2e} > 1e-1"
        )
        nfail += 1
    # SELF_W: Im Sigma(omega) at the generic point, on mesh (2) vs 1e-7 off mesh (4)
    st = lambda n: np.loadtxt(os.path.join(workdir, f"si_offt.Self.{n}"))[:, 4]
    rel = np.abs(st(2) - st(4)).max() / np.abs(st(2)).max()
    if rel > 1e-2:
        print(
            f"  si/si_offt: Im Sigma(omega) mesh vs 1e-7 off mesh: relative difference {rel:.2e} > 1e-2"
        )
        nfail += 1
    return nfail


def check_offmesh_bto(workdir):
    """bto_off_*.in list the vertices of X + 1e-7 (off mesh, every partner once, rows keyed
    by coordinates). Their gauge-invariant totals sum(mult * |V|^2) must equal those of the
    on-mesh listings of X (entry 2 of bto_v4_*): same physics, different row layout."""
    nfail = 0
    for off, on, cplx in (
        ("bto_off_1.V3.1", "bto_v4_1.V3.2", False),
        ("bto_off_1.V4.1", "bto_v4_1.V4.2", False),
        ("bto_off_2.Phi3.1", "bto_v4_2.Phi3.2", True),
        ("bto_off_2.Phi4.1", "bto_v4_2.Phi4.2", True),
    ):
        _, a = read_rows(os.path.join(workdir, off))
        _, b = read_rows(os.path.join(workdir, on))
        ta = value_total("." + ("Phi" if cplx else "V") + ".", a)
        tb = value_total("." + ("Phi" if cplx else "V") + ".", b)
        if abs(ta - tb) > 1e-5 * abs(tb):
            print(f"  bto/{off}: total {ta:.8e} differs from on-mesh {tb:.8e}")
            nfail += 1
    return nfail


def check_selfenergy_h5(workdir):
    """si_se_h5.in is si_se_path.in with the default FILE_FORMAT = h5: no per-target text
    files, one PREFIX.selfenergy.h5 with the targets in path order. Its linewidths must
    equal the text files of si_se_path (read back with h5dump when available)."""
    h5 = os.path.join(workdir, "si_se_h5.selfenergy.h5")
    if not os.path.exists(h5):
        print("  si/si_se_h5: PREFIX.selfenergy.h5 not written")
        return 1
    if any(f.startswith("si_se_h5.Gamma.") for f in os.listdir(workdir)):
        print("  si/si_se_h5: text files written although FILE_FORMAT = h5")
        return 1
    if shutil.which("h5dump") is None:
        print("  si/si_se_h5: h5dump not found, contents not checked")
        return 0
    nfail = 0
    # Path order (Gamma, midpoint, X) x branches 4-6: target 00004 is the midpoint, branch 4,
    # which the text run numbers as Gamma.7 (the only off-mesh point, listed after the six
    # on-mesh targets); target 00007 is X, branch 4 = Gamma.4 of the text run.
    for target, text in (("00004", 7), ("00007", 4)):
        out = subprocess.run(
            ["h5dump", "-d", f"/targets/{target}/linewidth", "-y", "-w", "400", h5],
            capture_output=True,
            text=True,
        ).stdout
        data = out.split("DATA {")[1].split("}")[0]
        vals = np.array([float(x) for x in data.replace(",", " ").split()])
        ref = np.loadtxt(os.path.join(workdir, f"si_se_path.Gamma.{text}"))[:, 1]
        if not np.allclose(vals, ref, rtol=1e-5):
            print(
                f"  si/si_se_h5: target {target} linewidth {vals} differs from si_se_path.Gamma.{text} {ref}"
            )
            nfail += 1
    return nfail


def check_interpolate(workdir):
    """si_se_int.in: INTERPOLATE on the Gamma-X path (3 points, branches 4-6). Each
    branch-projected spectrum must carry unit weight and peak within 1.5 cm^-1 of
    omega_j + Delta_j from the direct shift of the same target (on-shell values and the
    self-consistent peak differ by the frequency dependence of Sigma on a coarse mesh).
    Text numbering: on-mesh Gamma (1-3) and X (4-6) first, then the off-mesh midpoint (7-9);
    spectrum q index in path order: Gamma 1, midpoint 2, X 3."""
    nfail = 0
    sp = np.loadtxt(os.path.join(workdir, "si_se_int.spectrum"))
    T = sp[:, 0].max()
    mapping = {
        (1, 4): 1,
        (1, 5): 2,
        (1, 6): 3,
        (3, 4): 4,
        (3, 5): 5,
        (3, 6): 6,
        (2, 4): 7,
        (2, 5): 8,
        (2, 6): 9,
    }
    for (iq, b), n in mapping.items():
        gfile = os.path.join(workdir, f"si_se_int.Gamma.{n}")
        omega_j = float(
            [l for l in open(gfile) if l.startswith("# Frequency")][0].split()[-1]
        )
        shift = np.loadtxt(os.path.join(workdir, f"si_se_int.Shift.{n}"))[-1]
        delta = shift[1] + shift[2]
        rows = sp[(sp[:, 0] == T) & (sp[:, 1] == iq)]
        w, a = rows[:, 3], rows[:, 4 + b]
        peak = w[np.argmax(a)]
        weight = np.trapezoid(a, w) if hasattr(np, "trapezoid") else np.trapz(a, w)
        if abs(peak - (omega_j + delta)) > 1.5 or abs(weight - 1.0) > 0.05:
            print(
                f"  si/si_se_int: q{iq} branch {b}: peak {peak:.2f} vs omega+Delta {omega_j + delta:.2f}, weight {weight:.3f}"
            )
            nfail += 1
    return nfail


def main():
    build_dir = os.getcwd()
    project_root = os.path.dirname(build_dir)
    anphon = os.path.join(project_root, "_build", "anphon", "anphon")
    base = os.path.join(project_root, "test", "mode_analysis")
    nfail = 0
    for case, spec in CASES.items():
        workdir = os.path.join(base, case)
        os.chdir(workdir)
        for f in spec["files"]:
            dest = os.path.basename(f)
            if f.endswith(".bz2"):
                with bz2.open(f, "rb") as src, open(dest[:-4], "wb") as dst:
                    shutil.copyfileobj(src, dst)
            else:
                shutil.copy(f, dest)
        for inp in spec["inputs"]:
            with open(inp.replace(".in", ".log"), "w") as log:
                ret = subprocess.run([anphon, inp], stdout=log)
            if ret.returncode != 0:
                print(f"  {case}/{inp}: anphon failed")
                nfail += 1
        nfail += compare_case(
            case, workdir, os.path.join(base, "reference", case), spec["nk"]
        )
        if case == "si":
            nfail += check_offmesh(workdir)
            nfail += check_selfenergy_h5(workdir)
            nfail += check_interpolate(workdir)
        if case == "bto":
            nfail += check_offmesh_bto(workdir)
    print("mode analysis --> " + ("pass" if nfail == 0 else f"failed ({nfail})"))
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
