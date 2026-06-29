"""High-level Python interface to the ALAMODE 2.0dev ALM library (nanobind backend).

Drop-in compatible replacement for the legacy ``alm`` package (which used a C/cython wrapper):
the same public API is provided on top of the compiled ``alm._alm.ALMCore`` object.

    with ALM(lavec, xcoord, numbers, verbosity=0) as alm:
        alm.define(maxorder=2, cutoff_radii=rc)
        alm.set_training_data(u, f)             # u, f shape (nsnap, nat, 3)
        A, b = alm.get_matrix_elements()        # ALM's exact sensing matrix (for external solvers)
        alm.set_fc(coefs)                        # inject an external solution ...
        v, idx = alm.get_fc(2, mode="origin")    # ... read it back, fully symmetry-expanded
        alm.save_fc("out.h5", format="alamode_h5")   # write what anphon reads (.h5)

Suggest mode:
    with ALM(lavec, xcoord, numbers) as alm:
        alm.define(2, cutoff_radii=rc)
        alm.suggest()
        patterns = alm.get_displacement_patterns(1)
"""

import warnings
from collections import OrderedDict

import numpy as np

from . import _alm

# Atomic-number -> symbol (Z = 1..118; index 0 is a placeholder). Anphon needs the real
# element symbols in the saved FCs, so keep the full periodic table.
_ELEMENTS = [
    "X",
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]


def _sym(z):
    z = int(z)
    return _ELEMENTS[z] if 1 <= z < len(_ELEMENTS) else f"E{z}"


class ALM:
    """Pythonic wrapper around the pybind11 ALMCore object (legacy-compatible API)."""

    def __init__(self, lavec, xcoord, numbers, verbosity=0):
        self._core = None
        self._maxorder = 1
        self._defined = False
        self._need_transfer = True
        self._kind_names = OrderedDict()
        self._output_filename_prefix = None
        self._verbosity = verbosity
        self._lavec = np.array(lavec, dtype="double", order="C")
        self._xcoord = np.array(xcoord, dtype="double", order="C")
        self._numbers = np.array(numbers, dtype="intc")
        self._u = None
        self._f = None

    # ---- lifecycle / context manager ------------------------------------
    def __enter__(self):
        self.alm_new()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.alm_delete()

    def alm_new(self):
        """Create the underlying C++ ALM instance (also called by ``with``)."""
        if self._core is not None:
            raise RuntimeError("This ALM object is already initialized.")
        self._core = _alm.ALMCore()
        self._core.set_verbosity(self._verbosity)
        if self._output_filename_prefix is not None:
            self._core.set_output_filename_prefix(self._output_filename_prefix)
        self._transfer_parameters()

    def alm_delete(self):
        """Drop the underlying C++ ALM instance (pybind11 frees it)."""
        self._check()
        self._core = None
        self._defined = False

    def _check(self):
        if self._core is None:
            raise RuntimeError(
                "ALM is not initialized. Use it inside a 'with' block or call alm_new().")

    def _check_defined(self):
        if not self._defined:
            raise RuntimeError("define() must be called first.")

    def _check_fc_order(self, fc_order):
        if not (1 <= fc_order <= self._maxorder):
            raise ValueError(
                f"fc_order must be in [1, maxorder={self._maxorder}], got {fc_order}.")

    # ---- structure properties -------------------------------------------
    @property
    def lavec(self):
        return np.array(self._lavec, dtype="double", order="C")

    @lavec.setter
    def lavec(self, lavec):
        self._need_transfer = True
        self._lavec = np.array(lavec, dtype="double", order="C")

    @property
    def xcoord(self):
        return np.array(self._xcoord, dtype="double", order="C")

    @xcoord.setter
    def xcoord(self, xcoord):
        self._need_transfer = True
        self._xcoord = np.array(xcoord, dtype="double", order="C")

    @property
    def numbers(self):
        return np.array(self._numbers, dtype="intc")

    @numbers.setter
    def numbers(self, numbers):
        self._need_transfer = True
        self._numbers = np.array(numbers, dtype="intc")

    @property
    def kind_names(self):
        return self._kind_names

    def _transfer_parameters(self):
        if self._need_transfer and self._core is not None:
            self._set_cell()
            self._need_transfer = False

    def _set_cell(self):
        nat = len(self._xcoord)
        if len(self._numbers) != nat:
            raise RuntimeError("len(numbers) != number of atoms")
        self._kind_names = OrderedDict()
        for z in self._numbers:
            self._kind_names.setdefault(int(z), _sym(z))
        z_list = list(self._kind_names.keys())
        kind = np.array([z_list.index(int(z)) + 1 for z in self._numbers], dtype="intc")
        self._core.set_cell(self._lavec, self._xcoord, kind)
        self._core.set_element_names(list(self._kind_names.values()))

    # ---- verbosity / output ---------------------------------------------
    @property
    def verbosity(self):
        return self._verbosity

    @verbosity.setter
    def verbosity(self, verbosity):
        self._verbosity = int(verbosity)
        if self._core is not None:
            self._core.set_verbosity(self._verbosity)

    def set_verbosity(self, verbosity):
        self.verbosity = verbosity

    @property
    def output_filename_prefix(self):
        return self._output_filename_prefix

    @output_filename_prefix.setter
    def output_filename_prefix(self, prefix):
        if isinstance(prefix, str):
            self._output_filename_prefix = prefix
            if self._core is not None:
                self._core.set_output_filename_prefix(prefix)

    @property
    def cv_l1_alpha(self):
        self._check()
        return self._core.get_cv_l1_alpha()

    def get_cv_l1_alpha(self):
        return self.cv_l1_alpha

    # ---- model definition ------------------------------------------------
    def define(self, maxorder, cutoff_radii=None, nbody=None, symmetrization_basis="Lattice"):
        self._check()
        self._transfer_parameters()
        nkd = len(self._kind_names)
        if nbody is None:
            nbody = [i + 2 for i in range(maxorder)]
        elif len(nbody) != maxorder:
            raise RuntimeError("len(nbody) must equal maxorder")
        if cutoff_radii is None:
            cut = np.zeros(0, dtype="double")
        else:
            cut = np.array(cutoff_radii, dtype="double", order="C").ravel()
            if cut.size != maxorder * nkd * nkd:
                raise RuntimeError(f"cutoff_radii must have {maxorder}*{nkd}*{nkd} elements")
        basis = symmetrization_basis.capitalize()
        if basis not in ("Lattice", "Cartesian"):
            basis = "Lattice"
        self._core.set_forceconstant_basis(basis)
        self._core.define(maxorder, nkd, np.array(nbody, dtype="intc"), cut)
        self._core.init_fc_table()
        self._maxorder = maxorder
        self._defined = True

    def fix_force_constants_from_file(self, order, fc_file):
        """Fix the `order`-th FCs (order=2 harmonic, 3 cubic, ...) to those in `fc_file`
        (ALM .xml/.h5). Mirrors the CLI FC2FIX/FC3FIX tags. Call before define()."""
        self._check()
        self._core.set_fc_file(order, str(fc_file))
        self._core.set_fc_fix(order, True)

    def set_constraint(self, translation=True, rotation=False):
        """ICONST: 11 = algebraic translational invariance (acoustic sum rule)."""
        self._check()
        if rotation:
            raise NotImplementedError(
                "rotation=True is not supported by this wrapper yet "
                "(set ROTAXIS via the input-var interface if you need it).")
        self._core.set_constraint_mode(11 if translation else 0)

    # ---- fixing / freezing FCs ------------------------------------------
    def set_forceconstants_to_fix(self, intpair_fix, values_fix):
        """Fix a subset of FCs (e.g. FC2 from a reference fit)."""
        self._check()
        self._core.set_forceconstants_to_fix(
            [list(map(int, p)) for p in intpair_fix],
            np.asarray(values_fix, dtype="double").tolist())

    def freeze_fc(self, fc_values, fc_indices):
        """Legacy alias: freeze FCs to given values. fc_indices shape (n, order+1)."""
        fc_indices = np.array(fc_indices, dtype="intc", order="C")
        fc_values = np.array(fc_values, dtype="double")
        if fc_indices.ndim != 2:
            raise RuntimeError("fc_indices must be 2-dimensional.")
        if len(fc_indices) != len(fc_values):
            raise RuntimeError("fc_indices and fc_values must have the same length.")
        self.set_forceconstants_to_fix(fc_indices, fc_values)

    # ---- training / validation data -------------------------------------
    @property
    def displacements(self):
        if self._u is None:
            raise RuntimeError("displacements have not been set.")
        return np.array(self._u, dtype="double", order="C")

    @displacements.setter
    def displacements(self, u):
        self._u = np.array(u, dtype="double", order="C")

    @property
    def forces(self):
        if self._f is None:
            raise RuntimeError("forces have not been set.")
        return np.array(self._f, dtype="double", order="C")

    @forces.setter
    def forces(self, f):
        self._f = np.array(f, dtype="double", order="C")

    def set_training_data(self, u, f):
        self._check()
        u = np.array(u, dtype="double", order="C")
        f = np.array(f, dtype="double", order="C")
        if u.shape != f.shape:
            raise RuntimeError("u and f must have the same shape.")
        self._u, self._f = u, f
        nsnap = u.shape[0]
        self._core.set_u_train(u.reshape(nsnap, -1))
        self._core.set_f_train(f.reshape(nsnap, -1))

    def set_displacement_and_force(self, u, f):
        warnings.warn("set_displacement_and_force is deprecated; use set_training_data.",
                      DeprecationWarning)
        self.set_training_data(u, f)

    def set_validation_data(self, u, f):
        self._check()
        u = np.array(u, dtype="double", order="C")
        f = np.array(f, dtype="double", order="C")
        n = u.shape[0]
        self._core.set_validation_data(u.reshape(n, -1), f.reshape(n, -1))

    # ---- optimizer control ----------------------------------------------
    # Legacy key -> 2.0dev field (kept for backward compatibility).
    _OC_ALIASES = {"mirror_image_conv": "periodic_image_conv"}

    @property
    def optimizer_control(self):
        self._check()
        oc = self._core.get_optimizer_control()
        d = {k: getattr(oc, k) for k in dir(oc)
             if not k.startswith("_") and not callable(getattr(oc, k))}
        # expose legacy aliases too
        for old, new in self._OC_ALIASES.items():
            if new in d:
                d[old] = d[new]
        return d

    @optimizer_control.setter
    def optimizer_control(self, params):
        self.set_optimizer_control(params)

    def set_optimizer_control(self, params):
        self._check()
        oc = self._core.get_optimizer_control()
        for k, v in params.items():
            key = self._OC_ALIASES.get(k, k)
            if not hasattr(oc, key):
                raise KeyError(f"unknown optimizer_control key: {k}")
            setattr(oc, key, v)
        self._core.set_optimizer_control(oc)

    # ---- run -------------------------------------------------------------
    def suggest(self):
        """Compute the displacement patterns needed to determine the FCs."""
        self._check()
        self._check_defined()
        self._core.run_suggest()

    def optimize(self, solver="dense"):
        """Fit the FCs. solver: 'dense' (LAPACK SVD) or 'SimplicialLDLT' (Eigen sparse)."""
        self._check()
        self._check_defined()
        solvers = {"dense": "dense", "simplicialldlt": "SimplicialLDLT"}
        if solver.lower() not in solvers:
            raise ValueError("solver must be 'dense' or 'SimplicialLDLT'.")
        oc = self._core.get_optimizer_control()
        if solver.lower() == "dense":
            oc.use_sparse_solver = 0
        else:
            oc.use_sparse_solver = 1
            oc.sparsesolver = solvers[solver.lower()]
        self._core.set_optimizer_control(oc)
        return self._core.run_optimize()

    # ---- sensing matrix & force constants -------------------------------
    def get_matrix_elements(self):
        """Return (A, b): A shape (nrows, n_irred_fc) Fortran-order, b shape (nrows,)."""
        self._check()
        self._check_defined()
        if self._u is None or self._f is None:
            raise RuntimeError("set training data via set_training_data() before get_matrix_elements().")
        amat, bvec, nrows, ncols = self._core.get_matrix_elements()
        return np.reshape(amat, (nrows, ncols), order="F"), bvec

    def get_fc(self, fc_order, mode="origin", permutation=True):
        """Return (values, elem_indices[n, fc_order+1]). mode: origin | irreducible | all."""
        self._check()
        self._check_fc_order(fc_order)
        perm = 1 if permutation else 0
        if mode == "origin":
            v, idx = self._core.get_fc_origin(fc_order, perm)
        elif mode in ("irreducible", "irred"):
            v, idx = self._core.get_fc_irreducible(fc_order)
        elif mode == "all":
            v, idx = self._core.get_fc_all(fc_order, perm)
        else:
            raise ValueError("mode must be origin, irreducible, or all.")
        return v, idx.reshape(-1, fc_order + 1)

    def set_fc(self, fc_in):
        """Inject an irreducible FC vector (e.g. from an external regressor)."""
        self._check()
        fc_in = np.array(fc_in, dtype="double", order="C")
        n = sum(self._core.get_number_of_irred_fc_elements(o + 1)
                for o in range(self._maxorder))
        if fc_in.size != n:
            raise RuntimeError(f"expected {n} irreducible FCs, got {fc_in.size}.")
        self._core.set_fc(fc_in)

    def get_number_of_irred_fc_elements(self, fc_order):
        self._check()
        self._check_fc_order(fc_order)
        return self._core.get_number_of_irred_fc_elements(fc_order)

    # ---- mappings & displacement patterns -------------------------------
    def getmap_primitive_to_supercell(self):
        """Return map_p2s, shape (num_trans, num_atoms_primitive) (legacy orientation)."""
        self._check()
        self._check_defined()
        # C++ returns (nat_primitive, num_trans); transpose to match the legacy API.
        m = np.array(self._core.get_atom_mapping_by_pure_translations(), dtype="intc")
        return m.T.copy()

    def get_displacement_patterns(self, fc_order):
        """Return a list (per pattern) of (atom_index, direction(3,), basis) tuples."""
        self._check()
        self._check_fc_order(fc_order)
        numbers = self._core.get_number_of_displaced_atoms(fc_order)
        atom_indices, disp_flat, nbasis = self._core.get_displacement_patterns(fc_order)
        disp = np.reshape(disp_flat, (-1, 3))
        if nbasis not in (0, 1):
            raise RuntimeError(f"ALM returned invalid displacement basis index {nbasis}")
        basis = ["Cartesian", "Fractional"][nbasis]
        all_disps, pos = [], 0
        for num in numbers:
            disp_one = []
            for _ in range(int(num)):
                disp_one.append((int(atom_indices[pos]), disp[pos], basis))
                pos += 1
            all_disps.append(disp_one)
        return all_disps

    # ---- save ------------------------------------------------------------
    def save_fc(self, filename, format="alamode", maxorder_to_save=None):
        """Write FCs. format: 'alamode' (xml), 'alamode_h5' (.h5 for anphon),
        'shengbte', 'shengbte4', 'qefc', 'hessian'."""
        self._check()
        if maxorder_to_save is None:
            maxorder_to_save = self._maxorder
        self._core.save_fc(filename, format, maxorder_to_save)
