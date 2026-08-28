# Copyright (c) 2023 Ryota Masuki (strainIFCcoupling,
#                    https://github.com/r-masuki/strainIFCcoupling)
# Copyright (c) 2026 Terumasa Tadano
# MIT license.  See LICENCE.txt of the ALAMODE package.
"""Strain tensors, deformations and the strain-mode sets of anphon.

Conventions (identical to strainIFCcoupling and to anphon):

* ``u`` is the symmetric displacement-gradient tensor; the deformation gradient
  is ``F = I + u``; lattice vectors (rows) transform as ``a' = F a`` and
  Cartesian positions as ``x' = F x`` (fractional coordinates are unchanged).
* The six strain modes ``xx, yy, zz, yz, zx, xy`` carry a magnitude ``smag``.
  The off-diagonal mode tensors have 0.5 on both off-diagonal slots, i.e. the
  deformed cell for mode ``yz`` with magnitude ``s`` has ``u_yz = u_zy = s/2``.
  anphon divides the finite difference by ``smag`` and assigns the result to
  both (yz) and (zy) components (anphon/ifc_derivative.cpp).
* The Green-Lagrange strain is ``eta = 1/2 (F^T F - I) = u + 1/2 u^2`` for
  symmetric ``u`` (the corrected ``Relaxation::calculate_eta_tensor``).
"""

import json
from dataclasses import dataclass, field

import numpy as np

# Order of the mode names used for the default strain set (same as the
# original strainIFCcoupling ids 1..6).  anphon accepts exactly these strings.
MODE_NAMES = ("xx", "yy", "zz", "yz", "zx", "xy")

# Voigt ordering used by anphon's printout (relaxation.cpp) and by ase.
VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1))
VOIGT_INDEX = {}
for _a, (_i, _j) in enumerate(VOIGT_PAIRS):
    VOIGT_INDEX[(_i, _j)] = _a
    VOIGT_INDEX[(_j, _i)] = _a
# Multiplicity of a Voigt component in a full 3x3 contraction.
VOIGT_WEIGHT = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])

_MODE_PAIR = {
    "xx": (0, 0),
    "yy": (1, 1),
    "zz": (2, 2),
    "yz": (1, 2),
    "zx": (2, 0),
    "xy": (0, 1),
}

DEFAULT_SMAG = 0.005


def mode_tensor(mode):
    """Unit strain tensor of a mode name (3x3 numpy array).

    Raises ``ValueError`` for unknown names (the original script silently
    returned ``None``).
    """
    try:
        i, j = _MODE_PAIR[mode]
    except KeyError:
        raise ValueError(
            f"unknown strain mode {mode!r}; valid modes are {', '.join(MODE_NAMES)}"
        ) from None
    t = np.zeros((3, 3))
    if i == j:
        t[i, i] = 1.0
    else:
        t[i, j] = 0.5
        t[j, i] = 0.5
    return t


def mode_pair(mode):
    """Cartesian index pair (i, j) of a mode name."""
    if mode not in _MODE_PAIR:
        raise ValueError(f"unknown strain mode {mode!r}")
    return _MODE_PAIR[mode]


def u_from_voigt(d6):
    """Symmetric 3x3 tensor from six mode magnitudes in the order of MODE_NAMES.

    ``d6 = (t1..t6)`` gives ``u = sum_a t_a * mode_tensor(a)``; therefore a
    shear entry ``t`` yields ``u_ij = u_ji = t/2``.
    """
    d6 = np.asarray(d6, dtype=float)
    if d6.shape != (6,):
        raise ValueError("d6 must have six entries")
    u = np.zeros((3, 3))
    for a, name in enumerate(MODE_NAMES):
        u += d6[a] * mode_tensor(name)
    return u


def voigt_from_sym(t):
    """Six Voigt components (xx, yy, zz, yz, zx, xy) of a symmetric 3x3 tensor.

    No engineering factor is applied: the returned shear entries are the tensor
    components ``t_yz`` etc.
    """
    t = np.asarray(t, dtype=float)
    if np.abs(t - t.T).max() > 1.0e-8 * max(1.0, np.abs(t).max()):
        raise ValueError("tensor is not symmetric")
    return np.array([t[i, j] for (i, j) in VOIGT_PAIRS])


def sym_from_voigt(v6):
    """Symmetric 3x3 tensor from six Voigt components (tensor components)."""
    v6 = np.asarray(v6, dtype=float)
    t = np.zeros((3, 3))
    for a, (i, j) in enumerate(VOIGT_PAIRS):
        t[i, j] = v6[a]
        t[j, i] = v6[a]
    return t


def deformation_gradient(u):
    return np.eye(3) + np.asarray(u, dtype=float)


def green_lagrange(F):
    """eta = 1/2 (F^T F - I)."""
    F = np.asarray(F, dtype=float)
    return 0.5 * (F.T @ F - np.eye(3))


def deform(atoms, F):
    """Return a copy of ``atoms`` deformed by ``F`` (fractional coordinates fixed).

    rows of the new cell: ``a'_i = F a_i``  (``cell' = cell @ F.T``).
    """
    F = np.asarray(F, dtype=float)
    new = atoms.copy()
    new.set_cell(atoms.cell[:] @ F.T, scale_atoms=True)
    return new


@dataclass
class StrainPoint:
    """One strained calculation of the strain-IFC coupling workflow."""

    label: str  # e.g. "xx", "xx-" (central difference, negative member)
    mode: str  # one of MODE_NAMES
    smag: float  # signed magnitude written to the anphon input file
    weight: float  # weight written to the anphon input file
    u: np.ndarray = field(repr=False)  # 3x3 symmetric displacement gradient

    @property
    def F(self):
        return deformation_gradient(self.u)

    @property
    def eta(self):
        return green_lagrange(self.F)


def default_modes(smag=DEFAULT_SMAG):
    """The six default strain modes, weight 1 (one-sided finite difference)."""
    return [
        {"id": i + 1, "mode": m, "weight": 1.0, "strain_mag": smag}
        for i, m in enumerate(MODE_NAMES)
    ]


def load_modes_json(path):
    """Read a ``strain_modes.json`` file (strainIFCcoupling format).

    Each entry must contain ``mode`` and may contain ``weight`` (default 1.0)
    and ``strain_mag`` (default DEFAULT_SMAG).
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("modes", data.get("strain_modes"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a non-empty list of strain-mode entries")
    modes = []
    for k, item in enumerate(data):
        if "mode" not in item:
            raise ValueError(f"{path}: entry {k} has no 'mode' key")
        mode_pair(item["mode"])  # validates the name
        modes.append(
            {
                "id": int(item.get("id", k + 1)),
                "mode": str(item["mode"]),
                "weight": float(item.get("weight", 1.0)),
                "strain_mag": float(item.get("strain_mag", DEFAULT_SMAG)),
            }
        )
    return modes


def modes_from_names(names, smag=DEFAULT_SMAG):
    """Strain-mode entries from a list of mode names (weight 1 each)."""
    out = []
    for k, m in enumerate(names):
        mode_pair(m)
        out.append({"id": k + 1, "mode": m, "weight": 1.0, "strain_mag": smag})
    return out


def ifc_strain_set(modes, smag=None, central=False):
    """Build the list of StrainPoint for the strain-IFC coupling workflow.

    Parameters
    ----------
    modes : list of dict
        Entries with ``mode``, ``weight``, ``strain_mag`` (see load_modes_json).
    smag : float or None
        If not None, overrides ``strain_mag`` of every entry.  Note that 0.0 is a
        valid override (the original script treated 0.0 as "not given").
    central : bool
        Central finite differences: every entry produces two points
        ``(+s, w/2)`` and ``(-s, w/2)``.
    """
    points = []
    for item in modes:
        mode = item["mode"]
        s = float(item["strain_mag"]) if smag is None else float(smag)
        if s == 0.0:
            raise ValueError(f"strain magnitude of mode {mode!r} is zero")
        w = float(item.get("weight", 1.0))
        if central:
            points.append(
                StrainPoint(mode + "+", mode, s, 0.5 * w, s * mode_tensor(mode))
            )
            points.append(
                StrainPoint(mode + "-", mode, -s, 0.5 * w, -s * mode_tensor(mode))
            )
        else:
            points.append(StrainPoint(mode, mode, s, w, s * mode_tensor(mode)))
    return points


def weight_sum_matrix(points):
    """3x3 matrix of the accumulated weights, exactly as anphon computes it.

    (anphon/ifc_derivative.cpp: diagonal modes add to (i,i); off-diagonal
    modes add to both (i,j) and (j,i).)
    """
    wsum = np.zeros((3, 3))
    for p in points:
        i, j = mode_pair(p.mode)
        wsum[i, j] += p.weight
        if i != j:
            wsum[j, i] += p.weight
    return wsum


def check_weight_sums(points, require_all=True, tol=1.0e-6):
    """Raise ValueError unless the weight sums satisfy anphon's requirement.

    ``require_all=True`` (strain_force.in, RENORM_3TO2ND = 2): every one of the
    nine components must sum to 1.  ``require_all=False`` (RENORM_3TO2ND = 3):
    each component must sum to 1 or 0.
    Returns the 3x3 weight-sum matrix.
    """
    wsum = weight_sum_matrix(points)
    bad = []
    for i in range(3):
        for j in range(3):
            ok = abs(wsum[i, j] - 1.0) < tol
            if not require_all:
                ok = ok or abs(wsum[i, j]) < tol
            if not ok:
                bad.append((i, j, wsum[i, j]))
    if bad:
        xyz = "xyz"
        msg = ", ".join(f"({xyz[i]}{xyz[j]}): {w:g}" for i, j, w in bad)
        raise ValueError(
            "sum of the strain-mode weights must be 1"
            + ("" if require_all else " or 0")
            + f" for every component; offending components: {msg}"
        )
    return wsum
