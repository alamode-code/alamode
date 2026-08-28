"""Helpers shared by the EMT end-to-end tests (ase EMT as a fake DFT engine)."""

import glob
import os

import numpy as np


def run_emt(root):
    import ase.io
    from ase.calculators.emt import EMT

    n = 0
    for f in sorted(
        glob.glob(os.path.join(root, "strain_*", "**", "input.extxyz"), recursive=True)
    ):
        a = ase.io.read(f)
        a.calc = EMT()
        a.get_potential_energy()
        a.get_forces()
        a.get_stress()
        ase.io.write(f.replace("input.extxyz", "output.extxyz"), a)
        n += 1
    return n


def relaxed_cu_fcc():
    """EMT-relaxed fcc Cu primitive cell (so that sigma0 ~ 0)."""
    from ase.build import bulk
    from ase.calculators.emt import EMT
    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGS

    a = bulk("Cu", "fcc", a=3.6)
    a.calc = EMT()
    BFGS(FrechetCellFilter(a), logfile=None).run(fmax=1e-5)
    a.calc = None
    return a


def fit_reference_fc2(cell, out_file, fmt="xml"):
    from strainkit import almfit
    from ase.calculators.emt import EMT

    pats = almfit.suggest_harmonic_patterns(cell)
    u, f = [], []
    for s in almfit.displaced_structures(cell, pats, 0.01):
        s.calc = EMT()
        f.append(s.get_forces())
        u.append(s.get_positions() - cell.get_positions())
    almfit.fit_harmonic(cell, np.array(u), np.array(f), out_file, fmt)
    return out_file
