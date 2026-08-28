# Copyright (c) 2023 Ryota Masuki (strainIFCcoupling,
#                    https://github.com/r-masuki/strainIFCcoupling)
# Copyright (c) 2026 Terumasa Tadano
# MIT license.  See LICENCE.txt of the ALAMODE package.
"""Writers (and token-stream readers) for the strain-related anphon input files.

All readers mimic the whitespace-token parsing of the C++ code (``operator>>``)
so that a file accepted here is parsed identically by anphon.

* strain_harmonic.in  -- anphon/ifc_derivative.cpp (calculate_delv2_delumn_finite_difference)
* strain_force.in     -- anphon/ifc_derivative.cpp (calculate_delv1_delumn_finite_difference);
                         forces in eV/Angstrom.
* elastic_constants.in-- anphon/elastic_tensor.cpp (read_elastic_constants); V*C in Ry,
                         81 + 729 values in the full 3x3-index layout (i = 3*mu + nu).
* C1_array.in         -- anphon/elastic_tensor.cpp (read_C1_array); V*sigma in Ry, 9 values.
"""

import numpy as np

from .strain import mode_pair

_VALID_MODES = ("xx", "yy", "zz", "xy", "yz", "zx")


def _tokens(path):
    with open(path) as f:
        return f.read().split()


def _check_finite(arr, what):
    arr = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{what}: non-finite value encountered")
    return arr


# ------------------------------------------------------------ strain_harmonic.in
def write_strain_harmonic_in(path, rows):
    """rows: iterable of (mode, smag, weight, filename)."""
    lines = []
    for mode, smag, weight, filename in rows:
        mode_pair(mode)
        if not np.isfinite(smag) or not np.isfinite(weight) or smag == 0.0:
            raise ValueError(f"strain_harmonic.in: invalid smag/weight for mode {mode}")
        if any(c.isspace() for c in str(filename)):
            raise ValueError("strain_harmonic.in: filenames must not contain whitespace")
        lines.append("{0:4s} {1:25.15f} {2:25.15f} {3:25s}\n".format(
            mode, float(smag), float(weight), str(filename)))
    with open(path, "w") as f:
        f.writelines(lines)


def read_strain_harmonic_in(path):
    tok = _tokens(path)
    if len(tok) % 4 != 0:
        raise ValueError(f"{path}: number of tokens is not a multiple of 4")
    rows = []
    for k in range(0, len(tok), 4):
        mode = tok[k]
        if mode not in _VALID_MODES:
            raise ValueError(f"{path}: invalid mode name {mode!r}")
        rows.append((mode, float(tok[k + 1]), float(tok[k + 2]), tok[k + 3]))
    return rows


# --------------------------------------------------------------- strain_force.in
def write_strain_force_in(path, blocks):
    """blocks: iterable of (mode, smag, weight, forces) with forces (natmin, 3) in eV/A."""
    lines = []
    natmin = None
    for mode, smag, weight, forces in blocks:
        mode_pair(mode)
        forces = _check_finite(forces, f"strain_force.in ({mode})")
        if forces.ndim != 2 or forces.shape[1] != 3:
            raise ValueError("strain_force.in: forces must have shape (natmin, 3)")
        if natmin is None:
            natmin = forces.shape[0]
        elif forces.shape[0] != natmin:
            raise ValueError("strain_force.in: inconsistent number of atoms between blocks")
        if not np.isfinite(smag) or smag == 0.0 or not np.isfinite(weight):
            raise ValueError(f"strain_force.in: invalid smag/weight for mode {mode}")
        lines.append("{0:4s} {1:25.15f} {2:25.15f}\n".format(mode, float(smag), float(weight)))
        for fx, fy, fz in forces:
            lines.append("{0:25.15f} {1:25.15f} {2:25.15f}\n".format(fx, fy, fz))
    with open(path, "w") as f:
        f.writelines(lines)


def read_strain_force_in(path, natmin):
    """Token-stream reader; returns list of (mode, smag, weight, forces(natmin,3))."""
    tok = _tokens(path)
    blocks = []
    k = 0
    per = 3 + 3 * natmin
    if len(tok) % per != 0:
        raise ValueError(f"{path}: token count {len(tok)} is not a multiple of {per} (natmin={natmin})")
    while k < len(tok):
        mode = tok[k]
        if mode not in _VALID_MODES:
            raise ValueError(f"{path}: invalid mode name {mode!r}")
        smag, weight = float(tok[k + 1]), float(tok[k + 2])
        vals = np.array([float(t) for t in tok[k + 3:k + per]]).reshape(natmin, 3)
        blocks.append((mode, smag, weight, vals))
        k += per
    return blocks


# ------------------------------------------------------- elastic_constants.in
def write_elastic_constants_in(path, c2_99_ry, c3_999_ry):
    """c2: (9,9) and c3: (9,9,9) arrays of V*C in Ry (row-major i = 3*mu+nu)."""
    c2 = _check_finite(c2_99_ry, "elastic_constants.in (SOEC)")
    c3 = _check_finite(c3_999_ry, "elastic_constants.in (TOEC)")
    if c2.shape != (9, 9) or c3.shape != (9, 9, 9):
        raise ValueError("elastic_constants.in: expected shapes (9,9) and (9,9,9)")
    with open(path, "w") as f:
        f.write("SOEC\n")
        for v in c2.reshape(-1):
            f.write(f"{v:.12e}\n")
        f.write("TOEC\n")
        for v in c3.reshape(-1):
            f.write(f"{v:.12e}\n")


def read_elastic_constants_in(path):
    tok = _tokens(path)
    if len(tok) != 1 + 81 + 1 + 729:
        raise ValueError(f"{path}: expected {1 + 81 + 1 + 729} tokens, found {len(tok)}")
    c2 = np.array([float(t) for t in tok[1:82]]).reshape(9, 9)
    c3 = np.array([float(t) for t in tok[83:]]).reshape(9, 9, 9)
    return c2, c3


# ------------------------------------------------------------------ C1_array.in
def write_C1_array_in(path, vsigma_ry_3x3):
    s = _check_finite(vsigma_ry_3x3, "C1_array.in")
    if s.shape != (3, 3):
        raise ValueError("C1_array.in: expected a 3x3 array (V * sigma in Ry)")
    with open(path, "w") as f:
        f.write("C1\n")
        for v in s.reshape(-1):
            f.write(f"{v:.12e}\n")


def read_C1_array_in(path):
    tok = _tokens(path)
    if len(tok) != 10:
        raise ValueError(f"{path}: expected 10 tokens, found {len(tok)}")
    return np.array([float(t) for t in tok[1:]]).reshape(3, 3)


# ------------------------------------------------------- anphon-style checks
def asymmetry_rank2(s):
    """max |s_ij - s_ji| (Relaxation::set_elastic_constants warns above 1e-6)."""
    s = np.asarray(s, dtype=float).reshape(3, 3)
    return float(np.abs(s - s.T).max())


def asymmetry_rank4(c99):
    """max deviation from the intrinsic symmetries of C2 (9x9 layout)."""
    c = np.asarray(c99, dtype=float).reshape(3, 3, 3, 3)
    dev = 0.0
    for perm in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
        dev = max(dev, float(np.abs(c - np.transpose(c, perm)).max()))
    return dev


def asymmetry_rank6(c999):
    c = np.asarray(c999, dtype=float).reshape((3,) * 6)
    dev = 0.0
    for perm in ((1, 0, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5), (0, 1, 2, 3, 5, 4),
                 (2, 3, 0, 1, 4, 5), (0, 1, 4, 5, 2, 3)):
        dev = max(dev, float(np.abs(c - np.transpose(c, perm)).max()))
    return dev


def anphon_file_locations():
    return (
        "Where anphon expects these files:\n"
        "  STRAIN_IFC_DIR/  <- elastic_constants.in, strain_harmonic.in (+ the FC files it lists),\n"
        "                      strain_force.in\n"
        "  working dir      <- C1_array.in (read from the directory anphon runs in)\n"
    )
