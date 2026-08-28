# Copyright (c) 2023 Ryota Masuki (strainIFCcoupling,
#                    https://github.com/r-masuki/strainIFCcoupling)
# Copyright (c) 2026 Terumasa Tadano
# MIT license.  See LICENCE.txt of the ALAMODE package.
"""Displacement-force data sets built directly from DFT outputs."""

import warnings

import numpy as np

from .units import ang_to_bohr, ev_per_ang_to_ry_per_bohr


def displacements_from_positions(xf, xf0, cell):
    """Cartesian displacements (A) from fractional coordinates, refolded into
    the nearest image (same convention as tools/extract.py)."""
    d = np.asarray(xf, dtype=float) - np.asarray(xf0, dtype=float)
    d -= np.round(d)
    return d @ np.asarray(cell, dtype=float)


def build_dfset(disp_results, ref_atoms, offset=None, dmag=None):
    """Displacements (A) and forces (eV/A) of shape (nsnap, nat, 3).

    ``ref_atoms``: the undisplaced (strained) reference structure.
    ``offset``: DftResult of the undisplaced structure; its residual
    displacements/forces are subtracted (``extract.py --offset``).
    """
    x0 = ref_atoms.get_scaled_positions(wrap=False)
    cell = np.asarray(ref_atoms.cell[:], dtype=float)
    nat = len(ref_atoms)
    if offset is not None:
        d_off = displacements_from_positions(offset.scaled_positions, x0, cell)
        f_off = offset.forces
        if f_off is None:
            raise ValueError(
                f"{offset.path}: no forces found in the offset calculation"
            )
        if np.abs(d_off).max() > 1.0e-4:
            warnings.warn(
                f"{offset.path}: the undisplaced structure moved by up to "
                f"{np.abs(d_off).max():.2e} A relative to the reference"
            )
    else:
        d_off = np.zeros((nat, 3))
        f_off = np.zeros((nat, 3))
    u, f = [], []
    for r in disp_results:
        if r.forces is None:
            raise ValueError(f"{r.path}: no forces found")
        u.append(displacements_from_positions(r.scaled_positions, x0, cell) - d_off)
        f.append(np.asarray(r.forces, dtype=float) - f_off)
    u = np.array(u)
    f = np.array(f)
    if dmag is not None and len(u):
        umax = np.abs(u).max()
        if not (0.5 * dmag <= umax <= 2.0 * dmag):
            warnings.warn(
                f"largest displacement {umax:.3e} A is far from the expected "
                f"magnitude {dmag:.3e} A; check the parsed outputs"
            )
    return u, f


def write_dfset(path, u_ang, f_ev_ang, labels=None):
    """Classic ALAMODE DFSET (Bohr, Ry/Bohr) for use with the alm CLI."""
    with open(path, "w") as fp:
        for k, (uu, ff) in enumerate(zip(u_ang, f_ev_ang)):
            tag = labels[k] if labels else f"snapshot {k + 1}"
            fp.write(f"# {tag}\n")
            for a in range(len(uu)):
                ub = ang_to_bohr(uu[a])
                fb = ev_per_ang_to_ry_per_bohr(ff[a])
                fp.write(
                    "{:15.7F} {:15.7F} {:15.7F} {:20.8E} {:15.8E} {:15.8E}\n".format(
                        ub[0], ub[1], ub[2], fb[0], fb[1], fb[2]
                    )
                )
