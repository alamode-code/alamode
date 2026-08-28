"""Point-group symmetrization of Cartesian tensors (via spglib)."""

import itertools

import numpy as np


def cartesian_rotations(atoms, symprec=1.0e-5):
    """Unique Cartesian rotation matrices of the space group of ``atoms``.

    spglib returns rotations acting on fractional coordinates (x' = R x).  With
    ``A`` the matrix whose rows are the lattice vectors, r = A^T x, hence
    ``R_cart = A^T R (A^T)^-1``.
    """
    import spglib

    cell = (np.asarray(atoms.cell[:], dtype=float),
            np.asarray(atoms.get_scaled_positions(), dtype=float),
            np.asarray(atoms.numbers, dtype=int))
    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    if dataset is None:
        raise RuntimeError("spglib could not determine the symmetry of the structure")
    rots_frac = np.asarray(dataset.rotations if hasattr(dataset, "rotations")
                           else dataset["rotations"], dtype=float)
    at = np.asarray(atoms.cell[:], dtype=float).T
    at_inv = np.linalg.inv(at)
    seen = []
    for r in rots_frac:
        rc = at @ r @ at_inv
        if np.abs(rc @ rc.T - np.eye(3)).max() > 1.0e-6:
            raise RuntimeError("non-orthogonal Cartesian rotation obtained from spglib")
        if not any(np.abs(rc - s).max() < 1.0e-8 for s in seen):
            seen.append(rc)
    return np.array(seen)


def symmetrize_rank2(t, rots):
    out = np.zeros_like(t, dtype=float)
    for r in rots:
        out += r @ t @ r.T
    return out / len(rots)


def symmetrize_rank4(c, rots):
    out = np.zeros_like(c, dtype=float)
    for r in rots:
        out += np.einsum("ia,jb,kc,ld,abcd->ijkl", r, r, r, r, c)
    return out / len(rots)


def symmetrize_rank6(c, rots):
    out = np.zeros_like(c, dtype=float)
    for r in rots:
        out += np.einsum("ia,jb,kc,ld,me,nf,abcdef->ijklmn", r, r, r, r, r, r, c)
    return out / len(rots)


def enforce_intrinsic_symmetry2(c):
    """Average a rank-4 tensor over its 8 intrinsic index symmetries.

    (ij) <-> (ji), (kl) <-> (lk) and (ij) <-> (kl); same operations as
    ElasticTensor::symmetrize_elastic_tensor2 in anphon.
    """
    out = np.zeros_like(c, dtype=float)
    for perm in ((0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2),
                 (2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0)):
        out += np.transpose(c, perm)
    return out / 8.0


def enforce_intrinsic_symmetry3(c):
    """Average a rank-6 tensor over the 48 intrinsic symmetries (pair permutations
    x intra-pair swaps), as ElasticTensor::symmetrize_elastic_tensor3."""
    out = np.zeros_like(c, dtype=float)
    pairs = ((0, 1), (2, 3), (4, 5))
    n = 0
    for pperm in itertools.permutations(range(3)):
        for flips in itertools.product((False, True), repeat=3):
            perm = []
            for p, fl in zip(pperm, flips):
                a, b = pairs[p]
                perm.extend((b, a) if fl else (a, b))
            out += np.transpose(c, perm)
            n += 1
    return out / n


def max_change(before, after):
    return float(np.abs(np.asarray(after) - np.asarray(before)).max())
