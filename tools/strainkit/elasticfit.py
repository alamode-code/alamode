"""Finite-strain fit of first-, second- and third-order elastic constants.

Model (per reference cell, full 9-index sums, exactly anphon's
``Relaxation::renormalize_v0_from_umn``)::

    E(eta) = E0 + V0 [ sum s0_ij eta_ij + 1/2 sum C_ijkl eta_ij eta_kl
                       + 1/6 sum C_ijklmn eta_ij eta_kl eta_mn ]

with ``eta`` the Green-Lagrange strain.  The thermodynamic conjugate of eta is
the second Piola-Kirchhoff stress ``S = det(F) F^-1 sigma F^-T`` (sigma =
Cauchy stress of the deformed cell, ase sign convention: positive = tensile)::

    S(eta) = s0 + C2 eta + 1/2 C3 eta eta      (all in eV/Angstrom^3)

The unknowns are the independent Voigt components: s0 (6), C2 (21), C3 (56)
= 83, plus a small energy offset dE0 for energy-based fits.
"""

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from .strain import (MODE_NAMES, VOIGT_PAIRS, VOIGT_WEIGHT, deformation_gradient,
                     green_lagrange, u_from_voigt, voigt_from_sym)
from .units import EV_PER_ANG3_TO_GPA

N_VOIGT = 6


def independent_indices(order):
    """Sorted Voigt index tuples with repetition (6, 21, 56 for order 1, 2, 3)."""
    return list(itertools.combinations_with_replacement(range(N_VOIGT), order))


IDX1 = independent_indices(1)
IDX2 = independent_indices(2)
IDX3 = independent_indices(3)
POS2 = {t: k for k, t in enumerate(IDX2)}
POS3 = {t: k for k, t in enumerate(IDX3)}
N_PARAM = len(IDX1) + len(IDX2) + len(IDX3)  # 83


def n_distinct_permutations(t):
    counts = {}
    for a in t:
        counts[a] = counts.get(a, 0) + 1
    n = math.factorial(len(t))
    for c in counts.values():
        n //= math.factorial(c)
    return n


def design_row_energy(eta6):
    """Coefficients of [s0(6), C2(21), C3(56)] for E/V0 - E0/V0 (no dE0 column)."""
    eta6 = np.asarray(eta6, dtype=float)
    w = VOIGT_WEIGHT
    row = np.zeros(N_PARAM)
    pos = 0
    for order, idx in ((1, IDX1), (2, IDX2), (3, IDX3)):
        fac = 1.0 / math.factorial(order)
        for t in idx:
            val = fac * n_distinct_permutations(t)
            for a in t:
                val *= w[a] * eta6[a]
            row[pos] = val
            pos += 1
    return row


def design_rows_stress(eta6):
    """(6, 83) coefficient matrix for the six 2nd-PK stress components."""
    eta6 = np.asarray(eta6, dtype=float)
    w = VOIGT_WEIGHT
    rows = np.zeros((N_VOIGT, N_PARAM))
    off2 = len(IDX1)
    off3 = off2 + len(IDX2)
    for alpha in range(N_VOIGT):
        rows[alpha, alpha] = 1.0
        for k, t in enumerate(IDX2):
            if alpha in t:
                rest = list(t)
                rest.remove(alpha)
                rows[alpha, off2 + k] = w[rest[0]] * eta6[rest[0]]
        for k, t in enumerate(IDX3):
            if alpha in t:
                rest = list(t)
                rest.remove(alpha)
                val = 0.5 * n_distinct_permutations(tuple(rest))
                for a in rest:
                    val *= w[a] * eta6[a]
                rows[alpha, off3 + k] = val
    return rows


def second_pk_from_cauchy(sigma, F):
    """Second Piola-Kirchhoff stress from the Cauchy stress of the deformed cell."""
    F = np.asarray(F, dtype=float)
    Finv = np.linalg.inv(F)
    return np.linalg.det(F) * Finv @ np.asarray(sigma, dtype=float) @ Finv.T


def cauchy_from_second_pk(S, F):
    F = np.asarray(F, dtype=float)
    return F @ np.asarray(S, dtype=float) @ F.T / np.linalg.det(F)


# ---------------------------------------------------------------- Voigt <-> full
def voigt_to_full2(c21):
    c = np.zeros((3, 3, 3, 3))
    for k, (a, b) in enumerate(IDX2):
        for (i, j) in ((VOIGT_PAIRS[a][0], VOIGT_PAIRS[a][1]), (VOIGT_PAIRS[a][1], VOIGT_PAIRS[a][0])):
            for (kk, ll) in ((VOIGT_PAIRS[b][0], VOIGT_PAIRS[b][1]), (VOIGT_PAIRS[b][1], VOIGT_PAIRS[b][0])):
                c[i, j, kk, ll] = c21[k]
                c[kk, ll, i, j] = c21[k]
    return c


def full2_to_voigt(c4):
    out = np.zeros(len(IDX2))
    for k, (a, b) in enumerate(IDX2):
        i, j = VOIGT_PAIRS[a]
        kk, ll = VOIGT_PAIRS[b]
        out[k] = c4[i, j, kk, ll]
    return out


def voigt_to_full3(c56):
    c = np.zeros((3,) * 6)
    for k, t in enumerate(IDX3):
        for perm in set(itertools.permutations(t)):
            slots = []
            for a in perm:
                i, j = VOIGT_PAIRS[a]
                slots.append(((i, j), (j, i)))
            for (p0, p1, p2) in itertools.product(*slots):
                c[p0[0], p0[1], p1[0], p1[1], p2[0], p2[1]] = c56[k]
    return c


def full3_to_voigt(c6):
    out = np.zeros(len(IDX3))
    for k, (a, b, d) in enumerate(IDX3):
        i, j = VOIGT_PAIRS[a]
        kk, ll = VOIGT_PAIRS[b]
        m, n = VOIGT_PAIRS[d]
        out[k] = c6[i, j, kk, ll, m, n]
    return out


def voigt66(c4):
    """6x6 Voigt matrix of a rank-4 tensor (tensor components, no factors)."""
    m = np.zeros((6, 6))
    for a, (i, j) in enumerate(VOIGT_PAIRS):
        for b, (k, l) in enumerate(VOIGT_PAIRS):
            m[a, b] = c4[i, j, k, l]
    return m


def full2_to_9x9(c4):
    return c4.reshape(9, 9)


def full3_to_9x9x9(c6):
    return c6.reshape(9, 9, 9)


def from_9x9(c99):
    return np.asarray(c99, dtype=float).reshape(3, 3, 3, 3)


def from_9x9x9(c999):
    return np.asarray(c999, dtype=float).reshape(3, 3, 3, 3, 3, 3)


# ------------------------------------------------------------- direction sets
def direction_set(kind="minimal"):
    """Strain directions in the six-dimensional space of the mode magnitudes.

    ``minimal``: 6 singles e_a + 15 pairs e_a + e_b (21 directions) -- enough to
    determine all 83 parameters from stresses.
    ``full``: adds 15 anti-pairs e_a - e_b and 20 triples e_a + e_b + e_c
    (56 directions) -- needed for energy-only fits of the third-order constants.
    """
    dirs = []
    eye = np.eye(N_VOIGT)
    for a in range(N_VOIGT):
        dirs.append((MODE_NAMES[a], eye[a].copy()))
    for a, b in itertools.combinations(range(N_VOIGT), 2):
        dirs.append((f"{MODE_NAMES[a]}+{MODE_NAMES[b]}", eye[a] + eye[b]))
    if kind == "full":
        for a, b in itertools.combinations(range(N_VOIGT), 2):
            dirs.append((f"{MODE_NAMES[a]}-{MODE_NAMES[b]}", eye[a] - eye[b]))
        for a, b, c in itertools.combinations(range(N_VOIGT), 3):
            dirs.append((f"{MODE_NAMES[a]}+{MODE_NAMES[b]}+{MODE_NAMES[c]}",
                         eye[a] + eye[b] + eye[c]))
    elif kind != "minimal":
        raise ValueError(f"unknown direction set {kind!r} (minimal|full)")
    return dirs


@dataclass
class ElasticPoint:
    index: int
    label: str
    k: int  # magnitude index (0 = reference)
    d6: np.ndarray = field(repr=False)
    u: np.ndarray = field(repr=False)

    @property
    def F(self):
        return deformation_gradient(self.u)

    @property
    def eta(self):
        return green_lagrange(self.F)

    @property
    def eta6(self):
        return voigt_from_sym(self.eta)


def strain_points(dirset, smag, nmag=2):
    """Reference point (index 0) followed by ``u = k smag d`` for k = +-1..+-nmag."""
    if smag <= 0.0:
        raise ValueError("smag must be positive")
    if nmag < 1:
        raise ValueError("nmag must be >= 1")
    points = [ElasticPoint(0, "reference", 0, np.zeros(6), np.zeros((3, 3)))]
    idx = 1
    for label, d6 in dirset:
        for k in range(1, nmag + 1):
            for sign in (1, -1):
                kk = sign * k
                u = u_from_voigt(kk * smag * d6)
                points.append(ElasticPoint(idx, label, kk, kk * smag * d6, u))
                idx += 1
    return points


# ---------------------------------------------------------------------- fit
@dataclass
class FitData:
    """Data of one strained configuration used in the fit."""

    label: str
    eta6: np.ndarray
    energy: float = None  # eV, absolute
    stress6: np.ndarray = None  # 2nd-PK stress, eV/A^3, Voigt (tensor comps)


@dataclass
class FitResult:
    mode: str
    volume: float  # V0 in A^3
    sigma0: np.ndarray  # (6,) eV/A^3
    c2: np.ndarray  # (21,) eV/A^3
    c3: np.ndarray  # (56,) eV/A^3
    de0: float  # eV (energy fits) or 0
    e_ref: float
    rank: int
    expected_rank: int
    cond: float
    singular_values: np.ndarray
    rms_energy: float  # eV/A^3 residual RMS (energy rows) or nan
    rms_stress: float  # eV/A^3 residual RMS (stress rows) or nan
    n_energy: int
    n_stress: int
    residuals: list  # (label, kind, max abs residual) per configuration
    block_rms: dict = None  # 'both' mode: residual RMS of the single-block fits used as weights

    @property
    def c2_full(self):
        return voigt_to_full2(self.c2)

    @property
    def c3_full(self):
        return voigt_to_full3(self.c3)

    @property
    def sigma0_full(self):
        from .strain import sym_from_voigt
        return sym_from_voigt(self.sigma0)


def _assemble(data, volume, mode, e_ref):
    rows, rhs, tags = [], [], []
    use_e = mode in ("energy", "both")
    use_s = mode in ("stress", "both")
    ncol = N_PARAM + (1 if use_e else 0)
    for d in data:
        if use_e and d.energy is not None:
            r = np.zeros(ncol)
            r[:N_PARAM] = design_row_energy(d.eta6)
            r[N_PARAM] = 1.0 / volume  # dE0 / V0 column
            rows.append(r)
            rhs.append((d.energy - e_ref) / volume)
            tags.append((d.label, "energy"))
        if use_s and d.stress6 is not None:
            r = np.zeros((N_VOIGT, ncol))
            r[:, :N_PARAM] = design_rows_stress(d.eta6)
            for a in range(N_VOIGT):
                rows.append(r[a])
                rhs.append(d.stress6[a])
                tags.append((d.label, "stress"))
    if not rows:
        raise ValueError(f"no data available for fit mode {mode!r}")
    return np.array(rows), np.array(rhs), tags, ncol


def _lstsq(A, b):
    x, _, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    return x, rank, sv, cond


def fit_elastic(data, volume, mode="stress", e_ref=None, weight_floor=1.0e-8):
    """Least-squares fit of s0, C2, C3 (and dE0).

    Parameters
    ----------
    data : list of FitData
    volume : float
        Reference-cell volume in Angstrom^3.
    mode : {"stress", "energy", "both"}
    e_ref : float or None
        Reference energy subtracted from all energies (eV).  Defaults to the
        energy of the configuration whose eta vanishes, else the minimum.
    """
    if mode not in ("stress", "energy", "both"):
        raise ValueError("mode must be stress, energy or both")
    energies = [d.energy for d in data if d.energy is not None]
    if e_ref is None:
        ref = [d.energy for d in data
               if d.energy is not None and np.abs(d.eta6).max() < 1.0e-14]
        e_ref = ref[0] if ref else (min(energies) if energies else 0.0)
    for d in data:
        if not np.all(np.isfinite(d.eta6)):
            raise ValueError(f"{d.label}: non-finite strain")
        if d.energy is not None and not np.isfinite(d.energy):
            raise ValueError(f"{d.label}: non-finite energy")
        if d.stress6 is not None and not np.all(np.isfinite(d.stress6)):
            raise ValueError(f"{d.label}: non-finite stress")

    A, b, tags, ncol = _assemble(data, volume, mode, e_ref)
    kinds = np.array([t[1] for t in tags])
    wrow = np.ones(len(b))
    block_rms = {}
    if mode == "both":
        # One reweighting step: fit all rows unweighted first, take the residual
        # RMS of every block from that common solution, and weight the rows by
        # its inverse (floored).  This is symmetric in the two blocks and does
        # not depend on the single-block fits being full rank (an energy-only
        # block is rank deficient on the minimal direction set).
        x0, _, _, _ = _lstsq(A, b)
        res0 = A @ x0 - b
        for kind in ("energy", "stress"):
            sel = kinds == kind
            if sel.sum() == 0:
                continue
            rms = float(np.sqrt(np.mean(res0[sel] ** 2)))
            block_rms[kind] = rms
            wrow[sel] = 1.0 / max(rms, weight_floor)
    x, rank, sv, cond = _lstsq(A * wrow[:, None], b * wrow)
    expected = ncol
    res = A @ x - b
    per_conf = {}
    for (label, kind), r in zip(tags, res):
        key = (label, kind)
        per_conf[key] = max(per_conf.get(key, 0.0), abs(float(r)))
    residuals = [(label, kind, v) for (label, kind), v in per_conf.items()]
    sel_e = kinds == "energy"
    sel_s = kinds == "stress"
    rms_e = float(np.sqrt(np.mean(res[sel_e] ** 2))) if sel_e.any() else float("nan")
    rms_s = float(np.sqrt(np.mean(res[sel_s] ** 2))) if sel_s.any() else float("nan")
    de0 = float(x[N_PARAM]) if ncol > N_PARAM else 0.0
    return FitResult(
        mode=mode, volume=float(volume),
        sigma0=x[:6].copy(), c2=x[6:27].copy(), c3=x[27:83].copy(),
        de0=de0, e_ref=float(e_ref), rank=int(rank), expected_rank=int(expected),
        cond=cond, singular_values=sv, rms_energy=rms_e, rms_stress=rms_s,
        n_energy=int(sel_e.sum()), n_stress=int(sel_s.sum()), residuals=residuals,
        block_rms=block_rms or None,
    )


def evaluate_model(sigma0_6, c2_21, c3_56, eta6, volume, e0=0.0):
    """Energy (eV) and 2nd-PK stress (eV/A^3, Voigt) of the model at eta6."""
    x = np.concatenate([sigma0_6, c2_21, c3_56])
    e = e0 + volume * float(design_row_energy(eta6) @ x)
    s = design_rows_stress(eta6) @ x
    return e, s


# ------------------------------------------------------------------- report
def voigt_table_gpa(c4_ev, title=None):
    """6x6 Voigt table in GPa in the layout of anphon (relaxation.cpp)."""
    m = voigt66(c4_ev) * EV_PER_ANG3_TO_GPA
    lines = []
    if title:
        lines.append(title)
    lines.append("        " + "".join(f"{n:>11s}" for n in ("xx", "yy", "zz", "yz", "zx", "xy")))
    for a, name in enumerate(("xx", "yy", "zz", "yz", "zx", "xy")):
        lines.append(f"  {name:>4s}  " + "".join(f"{m[a, b]:11.3f}" for b in range(6)))
    return "\n".join(lines)


def voigt_names():
    return ["xx", "yy", "zz", "yz", "zx", "xy"]


def format_c3_gpa(c56_ev, min_gpa=0.5):
    names = voigt_names()
    lines = []
    for k, t in enumerate(IDX3):
        v = c56_ev[k] * EV_PER_ANG3_TO_GPA
        if abs(v) >= min_gpa:
            lines.append(f"  C_{names[t[0]]},{names[t[1]]},{names[t[2]]} = {v:12.3f}")
    if not lines:
        lines.append(f"  (no component with |C| >= {min_gpa} GPa)")
    return "\n".join(lines)


def format_report(fit, sigma0_full, c2_full, c3_full, sym_change=None, min_c3=0.5):
    """Human-readable summary.  Tensors are the (symmetrized) ones to report."""
    g = EV_PER_ANG3_TO_GPA
    out = []
    out.append(f"Fit mode: {fit.mode}   configurations: {fit.n_energy} energies, "
               f"{fit.n_stress // 6} stress tensors")
    out.append(f"Rank: {fit.rank} / {fit.expected_rank}   condition number: {fit.cond:.3e}")
    if fit.n_energy:
        out.append(f"Residual RMS (energy rows): {fit.rms_energy * g:.4e} GPa-equivalent"
                   f"  (E0 - E_ref = {fit.de0:.3e} eV)")
    if fit.n_stress:
        out.append(f"Residual RMS (stress rows): {fit.rms_stress * g:.4e} GPa")
    if sym_change is not None:
        out.append("Point-group symmetrization changed the tensors by at most "
                   + ", ".join(f"{k}: {v * g:.3e} GPa" for k, v in sym_change.items()))
    out.append("")
    out.append("Reference stress sigma0 (GPa, tensile positive):")
    for i in range(3):
        out.append("   " + "".join(f"{sigma0_full[i, j] * g:11.4f}" for j in range(3)))
    out.append("")
    out.append(voigt_table_gpa(c2_full, "Second-order elastic constants (GPa, Voigt notation):"))
    out.append("")
    out.append(f"Third-order elastic constants (GPa, |C| >= {min_c3}):")
    out.append(format_c3_gpa(full3_to_voigt(c3_full), min_c3))
    worst = sorted(fit.residuals, key=lambda r: -r[2])[:5]
    if worst:
        out.append("")
        out.append("Largest per-configuration residuals (GPa):")
        for label, kind, v in worst:
            out.append(f"  {label:>16s} {kind:>7s} {v * g:12.4e}")
    return "\n".join(out)
