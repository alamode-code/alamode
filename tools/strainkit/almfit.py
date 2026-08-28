# Copyright (c) 2023 Ryota Masuki (strainIFCcoupling,
#                    https://github.com/r-masuki/strainIFCcoupling)
# Copyright (c) 2026 Terumasa Tadano
# MIT license.  See LICENCE.txt of the ALAMODE package.
"""Harmonic force-constant models on (strained) supercells with the in-repo
nanobind ``alm`` package (python/ of the ALAMODE distribution)."""

import os

import numpy as np

_INSTALL_HINT = (
    "The 'alm' Python package (nanobind interface of ALM shipped in the python/ "
    "directory of ALAMODE) is required for this step.  Install it with\n"
    "    pip install /path/to/alamode/python\n"
    "(see the 'ALM Python interface' page of the documentation)."
)


def require_alm():
    try:
        from alm import ALM  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"{_INSTALL_HINT}\nOriginal error: {exc}") from None
    return ALM


def model_kwargs(nbody, cutoff, nkd):
    """Normalize the scalar CLI options to the shapes ALM.define expects."""
    nbody = [int(nbody)]
    if cutoff is None or float(cutoff) < 0:
        cut = np.full((1, nkd, nkd), -1.0)
    else:
        cut = np.full((1, nkd, nkd), float(cutoff))
    return {"nbody": nbody, "cutoff_radii": cut}


def make_alm(atoms, verbosity=0):
    ALM = require_alm()
    # ALM stores the lattice vectors as COLUMNS (as the CLI &cell / POSCAR
    # parsers do), while ase stores them as rows -> transpose.
    return ALM(np.asarray(atoms.cell[:], dtype=float).T.copy(),
               np.asarray(atoms.get_scaled_positions(wrap=False), dtype=float),
               np.asarray(atoms.numbers, dtype=int),
               verbosity=verbosity, length_unit="angstrom", force_unit="eV/angstrom")


def _define_harmonic(alm, atoms, nbody, cutoff):
    nkd = len(np.unique(np.asarray(atoms.numbers)))
    kw = model_kwargs(nbody, cutoff, nkd)
    alm.define(1, cutoff_radii=kw["cutoff_radii"], nbody=kw["nbody"],
               symmetrization_basis="Lattice")


# ---------------------------------------------------------------------------
# Every ALM job runs in a fresh interpreter: creating several ALM objects in
# one process crashes the current nanobind module (segmentation fault in
# define() around the 3rd-4th instance).  Set STRAINKIT_ALM_INPROCESS=1 to
# run in-process (debugging only).  The worker is a plain subprocess (not
# multiprocessing.spawn) so that it does not depend on how the parent was
# started; ALM's console output is captured and shown only on failure or
# when verbosity > 0.
import pickle
import subprocess
import sys
import tempfile

_TOOLS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _subprocess_main(fin, fout):
    with open(fin, "rb") as f:
        name, args, kwargs = pickle.load(f)
    try:
        result = ("ok", globals()[name](*args, **kwargs))
    except BaseException as exc:  # noqa: BLE001
        result = ("err", f"{type(exc).__name__}: {exc}")
    with open(fout, "wb") as f:
        pickle.dump(result, f)


def _isolated(name, *args, **kwargs):
    if os.environ.get("STRAINKIT_ALM_INPROCESS") == "1":
        return globals()[name](*args, **kwargs)
    verbosity = kwargs.get("verbosity", 0)
    with tempfile.TemporaryDirectory(prefix="strainkit_alm_") as td:
        fin = os.path.join(td, "in.pkl")
        fout = os.path.join(td, "out.pkl")
        with open(fin, "wb") as f:
            pickle.dump((name, args, kwargs), f)
        env = dict(os.environ)
        env["PYTHONPATH"] = _TOOLS_DIR + os.pathsep + env.get("PYTHONPATH", "")
        code = "import sys; from strainkit import almfit; almfit._subprocess_main(sys.argv[1], sys.argv[2])"
        proc = subprocess.run([sys.executable, "-c", code, fin, fout], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if verbosity > 0 and proc.stdout:
            print(proc.stdout)
        tail = "\n".join(proc.stdout.strip().splitlines()[-15:]) if proc.stdout else ""
        if not os.path.exists(fout):
            raise RuntimeError(
                f"ALM worker process for {name} died without a result (exit code "
                f"{proc.returncode}; a negative value is a signal, e.g. -11 = segmentation fault)."
                + (f"\nLast output:\n{tail}" if tail else ""))
        with open(fout, "rb") as f:
            status, payload = pickle.load(f)
    if status != "ok":
        raise RuntimeError(f"ALM step {name} failed: {payload}" + (f"\nLast output:\n{tail}" if tail else ""))
    return payload


def suggest_harmonic_patterns(atoms, nbody=2, cutoff=None, verbosity=0):
    """ALM displacement patterns of the harmonic model of ``atoms``.

    Returns a list of patterns; each pattern is a list of (atom_index,
    direction (3,), "Cartesian").
    """
    require_alm()
    return _isolated("_suggest_impl", atoms, nbody, cutoff, verbosity)


def _suggest_impl(atoms, nbody, cutoff, verbosity):
    with make_alm(atoms, verbosity) as alm:
        _define_harmonic(alm, atoms, nbody, cutoff)
        alm.suggest()
        patterns = alm.get_displacement_patterns(1)
    for pat in patterns:
        for _, _, basis in pat:
            if basis != "Cartesian":
                raise RuntimeError(f"unexpected displacement basis {basis!r} from ALM")
    return patterns


def displaced_structures(atoms, patterns, dmag):
    """Structures with the ALM patterns applied (Cartesian, magnitude dmag in A)."""
    out = []
    for pat in patterns:
        a = atoms.copy()
        pos = a.get_positions()
        for iat, direction, _ in pat:
            pos[iat] += float(dmag) * np.asarray(direction, dtype=float)
        a.set_positions(pos)
        out.append(a)
    return out


def fit_harmonic(atoms, u_ang, f_ev_ang, out_file, fmt="xml", nbody=2, cutoff=None,
                 solver="dense", verbosity=0):
    """Fit harmonic force constants and write them (Ry/bohr^2) to ``out_file``.

    Returns a dict with fit information.
    """
    fmt = fmt.lower()
    if fmt not in ("xml", "h5"):
        raise ValueError("fmt must be 'xml' or 'h5'")
    u = np.asarray(u_ang, dtype=float)
    f = np.asarray(f_ev_ang, dtype=float)
    if u.ndim != 3 or u.shape != f.shape or u.shape[1] != len(atoms) or u.shape[2] != 3:
        raise ValueError(f"training data must have shape (nsnap, {len(atoms)}, 3)")
    require_alm()
    return _isolated("_fit_impl", atoms, u, f, out_file, fmt, nbody, cutoff, solver, verbosity)


def _fit_impl(atoms, u, f, out_file, fmt, nbody, cutoff, solver, verbosity):
    with make_alm(atoms, verbosity) as alm:
        _define_harmonic(alm, atoms, nbody, cutoff)
        alm.set_constraint(translation=True)
        alm.set_training_data(u, f)
        alm.set_optimizer_control({"linear_model": 1})
        info = alm.optimize(solver=solver)
        if info != 0:
            raise RuntimeError(f"ALM optimize() failed with status {info}")
        nparam = alm.get_number_of_free_parameters()
        os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
        alm.save_fc(out_file, format="alamode" if fmt == "xml" else "alamode_h5",
                    fc_unit="Ry/bohr")
    return {"nsnap": int(u.shape[0]), "n_free_parameters": int(nparam), "info": int(info)}
