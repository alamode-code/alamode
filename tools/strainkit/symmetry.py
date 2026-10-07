"""Space-group symmetrization of Cartesian tensors (via spglib)."""

import itertools

import numpy as np


def cartesian_rotations(atoms, symprec=1.0e-5):
    """Unique Cartesian rotation matrices of the space group of the crystal of
    ``atoms`` (taken from its primitive cell, so a supercell whose lattice has a
    lower symmetry than the crystal still gives the full point group).

    spglib returns rotations acting on fractional coordinates (x' = R x).  With
    ``A`` the matrix whose rows are the lattice vectors, r = A^T x, hence
    ``R_cart = A^T R (A^T)^-1``.
    """
    seen = []
    for rc in space_group_operations(atoms, symprec)[0]:
        if not any(np.abs(rc - s).max() < 1.0e-8 for s in seen):
            seen.append(rc)
    return np.array(seen)


def space_group_operations(atoms, symprec=1.0e-5):
    """(Cartesian rotations (nop, 3, 3), atom permutations (nop, nprim), images
    (natom,)) of the space group of the crystal of ``atoms``.

    The operations are those of the primitive cell (spglib, same frame and origin
    as ``atoms``), acting on its atoms: R r_p + t = r_perm[p] modulo the primitive
    lattice.  images[k] is the primitive atom of which atom k of ``atoms`` is a
    translation image (the identity for a primitive ``atoms``).  A supercell whose
    lattice has a lower symmetry than the crystal (e.g. 2x1x1 cubic) thus still
    gets the full group: :func:`symmetrize_atomic` averages the images onto the
    primitive atoms, symmetrizes there and copies the result back.
    """
    import spglib

    lat = np.asarray(atoms.cell[:], dtype=float)
    xf = np.asarray(atoms.get_scaled_positions(), dtype=float)
    num = np.asarray(atoms.numbers, dtype=int)
    cart = xf @ lat
    prim = spglib.standardize_cell(
        (lat, xf, num), to_primitive=True, no_idealize=True, symprec=symprec
    )
    if prim is None:
        raise RuntimeError("spglib could not determine the primitive cell")
    plat = np.asarray(prim[0], dtype=float)
    m = lat @ np.linalg.inv(plat)
    if np.abs(m - np.round(m)).max() > 1.0e-6:
        raise RuntimeError("spglib returned a primitive cell in a different frame")
    inv = np.linalg.inv(plat)
    tol = 10.0 * symprec

    def same(d):
        d = d @ inv
        return np.linalg.norm((d - np.round(d)) @ plat, axis=-1) < tol

    rep, images = [], []
    for k in range(len(num)):
        hit = [
            i for i, j in enumerate(rep) if num[j] == num[k] and same(cart[k] - cart[j])
        ]
        if not hit:
            rep.append(k)
            hit = [len(rep) - 1]
        images.append(hit[0])
    pnum = num[rep]
    pcart = cart[rep]
    dataset = spglib.get_symmetry_dataset((plat, pcart @ inv, pnum), symprec=symprec)
    if dataset is None:
        raise RuntimeError("spglib could not determine the symmetry of the structure")
    get = lambda k: getattr(dataset, k) if hasattr(dataset, k) else dataset[k]  # noqa: E731
    at, at_inv = plat.T, inv.T
    rots, perms = [], []
    for r, t in zip(get("rotations"), get("translations")):
        rc = at @ r @ at_inv
        if np.abs(rc @ rc.T - np.eye(3)).max() > 1.0e-6:
            raise RuntimeError("non-orthogonal Cartesian rotation obtained from spglib")
        img = pcart @ rc.T + np.asarray(t) @ plat
        hit = same(img[:, None, :] - pcart[None, :, :]) & (
            pnum[:, None] == pnum[None, :]
        )
        perm = hit.argmax(axis=1)
        if not hit.any(axis=1).all() or sorted(perm) != list(range(len(pnum))):
            raise RuntimeError("could not map the atoms under a space-group operation")
        rots.append(rc)
        perms.append(perm)
    return np.array(rots), np.array(perms), np.array(images)


def symmetrize_atomic(t, rots, perms, images=None):
    """Average per-atom Cartesian tensors t (natom, 3, ..., 3) over the space group:
    t'_perm[p] = R...R t_p on every Cartesian index.  With ``images`` (from
    :func:`space_group_operations`) the translation images are first averaged onto
    the primitive atoms and the result is copied back to all of them."""
    t = np.asarray(t, dtype=float)
    if images is not None:
        images = np.asarray(images)
        nprim = perms.shape[1]
        tp = np.zeros((nprim,) + t.shape[1:])
        np.add.at(tp, images, t)
        tp /= np.bincount(images, minlength=nprim).reshape((-1,) + (1,) * (t.ndim - 1))
        return symmetrize_atomic(tp, rots, perms)[images]
    out = np.zeros_like(t)
    for r, p in zip(rots, perms):
        rt = t
        for ax in range(1, t.ndim):
            rt = np.moveaxis(np.tensordot(r, rt, axes=(1, ax)), 0, ax)
        out[p] += rt
    return out / len(rots)


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
    for perm in (
        (0, 1, 2, 3),
        (1, 0, 2, 3),
        (0, 1, 3, 2),
        (1, 0, 3, 2),
        (2, 3, 0, 1),
        (3, 2, 0, 1),
        (2, 3, 1, 0),
        (3, 2, 1, 0),
    ):
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
