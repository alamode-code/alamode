"""Reading DFT outputs (energy, forces, stress, geometry) through ase."""

from dataclasses import dataclass

import numpy as np

_FORMAT = {"VASP": "vasp-xml", "QE": "espresso-out", "ase": None}


@dataclass
class DftResult:
    path: str
    code: str
    nimages: int
    energy: float  # eV or None
    forces: np.ndarray  # (nat, 3) eV/A or None
    stress: np.ndarray  # (3, 3) eV/A^3, ase sign (tensile positive) or None
    cell: np.ndarray  # (3, 3) A, rows = lattice vectors
    scaled_positions: np.ndarray
    numbers: np.ndarray
    missing: dict = (
        None  # property -> reason, for energy/forces/stress that could not be read
    )

    def require(self, *names):
        """Raise ValueError if any of the named observables is unavailable."""
        for name in names:
            if getattr(self, name) is None:
                reason = (self.missing or {}).get(name, "not present in the file")
                raise ValueError(f"{self.path}: no {name} could be read ({reason})")


def read_dft_output(path, code, image=-1):
    import ase.io
    from .structure import normalize_code

    code = normalize_code(code)
    images = ase.io.read(path, index=":", format=_FORMAT[code])
    if not isinstance(images, list):
        images = [images]
    if not images:
        raise ValueError(f"{path}: no structure found")
    atoms = images[image]

    missing = {}

    def _try(name, getter):
        try:
            return getter()
        except Exception as exc:  # noqa: BLE001  (ase raises PropertyNotImplementedError etc.)
            missing[name] = f"{type(exc).__name__}: {exc}"
            return None

    energy = _try("energy", atoms.get_potential_energy)
    forces = _try("forces", atoms.get_forces)
    stress = _try("stress", lambda: atoms.get_stress(voigt=False))
    return DftResult(
        path=path,
        code=code,
        nimages=len(images),
        energy=None if energy is None else float(energy),
        forces=None if forces is None else np.asarray(forces, dtype=float),
        stress=None if stress is None else np.asarray(stress, dtype=float),
        cell=np.asarray(atoms.cell[:], dtype=float),
        scaled_positions=np.asarray(
            atoms.get_scaled_positions(wrap=False), dtype=float
        ),
        numbers=np.asarray(atoms.numbers, dtype=int),
        missing=missing,
    )


def check_same_species(result, ref_atoms):
    if len(result.numbers) != len(ref_atoms):
        raise ValueError(
            f"{result.path}: {len(result.numbers)} atoms, expected {len(ref_atoms)}"
        )
    if not np.array_equal(result.numbers, np.asarray(ref_atoms.numbers)):
        bad = np.nonzero(result.numbers != np.asarray(ref_atoms.numbers))[0]
        raise ValueError(
            f"{result.path}: atomic species differ from the generated structure at atom index "
            f"{bad[:10].tolist()} (0-based); the atom order must be preserved"
        )


def check_geometry(
    result, expected_atoms, tol_cell=1.0e-4, tol_frac=1.0e-5, strict=True
):
    """Verify that the output geometry equals the generated structure.

    Returns (max cell deviation [A], max fractional deviation).  Raises (or
    warns when ``strict`` is False) if the cell or the fractional coordinates
    changed -- i.e. the DFT run relaxed something the workflow assumed fixed.
    """
    import warnings

    check_same_species(result, expected_atoms)
    dcell = float(np.abs(result.cell - np.asarray(expected_atoms.cell[:])).max())
    d = result.scaled_positions - expected_atoms.get_scaled_positions(wrap=False)
    d -= np.round(d)
    dfrac = float(np.abs(d).max())
    problems = []
    if dcell > tol_cell:
        problems.append(f"cell differs by up to {dcell:.3e} A")
    if dfrac > tol_frac:
        problems.append(f"fractional coordinates differ by up to {dfrac:.3e}")
    if problems:
        msg = (
            f"{result.path}: geometry does not match the generated structure "
            f"({'; '.join(problems)}). The DFT run must not relax the cell or the ions "
            "(clamped-ion, fixed-cell calculation expected)."
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg)
    return dcell, dfrac
