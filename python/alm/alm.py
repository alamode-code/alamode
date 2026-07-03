"""High-level Python interface to the ALAMODE 2.0dev ALM library (nanobind backend).

Drop-in compatible replacement for the legacy ``alm`` package (which used a
C/cython wrapper): the same public API is provided on top of the compiled
``alm._alm.ALMCore`` object, plus cleaner primary names (``close()``,
``fix_fc()``, ``fix_fc_from_file()``); the legacy names remain as aliases.

    a = ALM(lavec, xcoord, numbers, verbosity=0)  # ready to use immediately
    a.define(maxorder=2, cutoff_radii=rc)    # up to cubic (fc_order 1=harmonic, 2=cubic)
    a.set_training_data(u, f)                # u, f shape (nsnap, nat, 3)
    A, b = a.get_matrix_elements()           # ALM's exact sensing matrix (for external solvers)
    a.set_fc(coefs)                          # inject an external solution ...
    v, idx = a.get_fc(1, mode="origin")      # ... read the harmonic FCs back, symmetry-expanded
    a.save_fc("out.h5", format="alamode_h5") # write what anphon reads (.h5)
    # 'a' (and its C++ instance) is freed automatically when it goes out of scope.

Units -- Rydberg atomic units throughout, as in the ALM CLI:

    lattice vectors, displacements, cutoff radii   Bohr
    forces                                         Ry/Bohr
    force constants of order ``fc_order``          Ry/Bohr^(fc_order+1)

Force-constant order convention, used consistently by every method here:

    fc_order = 1 -> harmonic (FC2), 2 -> cubic (FC3), 3 -> quartic (FC4), ...

The C++ instance is created in ``__init__``, so a plain ``a = ALM(...)`` is
enough.  A ``with`` block (or an explicit :meth:`ALM.close`) is optional and
only gives a prompt, deterministic release:

    with ALM(lavec, xcoord, numbers) as a:   # freed at block exit
        a.define(1, cutoff_radii=rc)
        a.suggest()
        patterns = a.get_displacement_patterns(1)
"""

from __future__ import annotations

import warnings
from collections import OrderedDict

import numpy as np

from . import _alm
from ._elements import element_symbol


class ALM:
    """Pythonic wrapper around the nanobind ALMCore object (legacy-compatible API).

    Parameters
    ----------
    lavec : array_like, shape (3, 3)
        Lattice vectors of the (super)cell in Bohr; row i is the i-th vector.
    xcoord : array_like, shape (nat, 3)
        Fractional coordinates of the atoms.
    numbers : array_like, shape (nat,)
        Atomic numbers.
    verbosity : int
        0 (silent, default) or 1 (print progress to stdout).

    Notes
    -----
    Assigning to :attr:`lavec`, :attr:`xcoord` or :attr:`numbers` re-sends the
    cell to the C++ core on the next :meth:`define` call; call :meth:`define`
    again after changing the structure.
    """

    # Sparse solvers accepted by the 2.0dev core (see alm/least_squares.cpp).
    _SPARSE_SOLVERS = ("SimplicialLDLT", "SparseQR", "ConjugateGradient",
                       "LeastSquaresConjugateGradient", "BiCGSTAB",
                       "SuiteSparseQR", "CHOLMOD")
    # Output formats accepted by ALMCore.save_fc (see alm/writer.cpp).
    _SAVE_FORMATS = ("alamode", "alamode_h5", "shengbte", "shengbte4",
                     "qefc", "hessian")
    # Legacy optimizer_control key -> 2.0dev field (kept for backward compatibility).
    _OC_ALIASES = {"mirror_image_conv": "periodic_image_conv"}

    def __init__(self, lavec, xcoord, numbers, verbosity: int = 0):
        self._core = None
        self._maxorder = 1
        self._defined = False
        self._patterns_ready = False
        self._supercell_active = False
        self._need_transfer = True
        self._need_data_transfer = False
        self._kind_names = OrderedDict()
        self._output_filename_prefix = None
        self._verbosity = verbosity
        self._lavec = np.array(lavec, dtype="double", order="C")
        self._xcoord = np.array(xcoord, dtype="double", order="C")
        self._numbers = np.array(numbers, dtype="intc")
        self._u = None
        self._f = None
        # Create the underlying C++ instance right away so the object is usable
        # directly:  a = ALM(lavec, xcoord, numbers).  It is freed automatically
        # when ``a`` is garbage-collected, so 'with'/close() are optional
        # (they only make the release prompt/deterministic).
        self.alm_new()

    # ---- lifecycle / context manager ------------------------------------
    def __enter__(self):
        # The instance already exists from __init__; recreate it only if the
        # user explicitly called close() before (re-)entering a 'with'.
        if self._core is None:
            self.alm_new()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __repr__(self):
        nat = len(self._numbers)
        nkd = np.unique(self._numbers).size
        if self._core is None:
            state = "closed"
        elif self._defined:
            state = f"defined, maxorder={self._maxorder}"
        else:
            state = "not defined"
        return f"<alm.ALM: {nat} atoms, {nkd} kinds ({state})>"

    def alm_new(self):
        """Create the underlying C++ ALM instance.

        Called automatically by ``__init__``, so you normally never call this.
        It is idempotent (a no-op if the instance already exists) and may be
        used to recreate the instance after :meth:`close`.
        """
        if self._core is not None:
            return
        self._core = _alm.ALMCore()
        self._core.set_verbosity(self._verbosity)
        if self._output_filename_prefix is not None:
            self._core.set_output_filename_prefix(self._output_filename_prefix)
        # A freshly created native core has no cell/data yet, so always
        # (re-)send them.  This matters when alm_new() recreates the instance
        # after close(): the transfer flags were cleared by the first transfer
        # and must be re-armed.
        self._need_transfer = True
        if self._u is not None and self._f is not None:
            self._need_data_transfer = True
        self._transfer_parameters()

    def close(self):
        """Free the underlying C++ ALM instance immediately.

        Optional: the instance is also freed automatically when this object is
        garbage-collected.  Call this (or use a ``with`` block) only when you
        want a prompt, deterministic release, e.g. inside a tight loop that
        builds many large sensing matrices.  Idempotent; :meth:`alm_new`
        re-creates the instance.
        """
        if self._core is None:
            return
        self._core = None
        self._defined = False
        self._patterns_ready = False
        self._supercell_active = False

    def alm_delete(self):
        """Legacy alias for :meth:`close`."""
        self.close()

    def _check(self):
        if self._core is None:
            raise RuntimeError(
                "The C++ ALM instance has been released by close()/alm_delete(). "
                "Call alm_new() to re-create it, or construct a new ALM object.")

    def _check_defined(self):
        if not self._defined:
            raise RuntimeError("define() must be called first.")

    def _check_fc_order(self, fc_order: int):
        if not (1 <= fc_order <= self._maxorder):
            raise ValueError(
                f"fc_order must be in [1 (harmonic), maxorder={self._maxorder}], "
                f"got {fc_order}.")

    # ---- structure properties -------------------------------------------
    @property
    def lavec(self) -> np.ndarray:
        """Lattice vectors of the (super)cell in Bohr, shape (3, 3); row i is
        the i-th vector.  Assigning a new value re-sends the cell to the C++
        core on the next :meth:`define` call."""
        return np.array(self._lavec, dtype="double", order="C")

    @lavec.setter
    def lavec(self, lavec):
        self._need_transfer = True
        self._lavec = np.array(lavec, dtype="double", order="C")

    @property
    def xcoord(self) -> np.ndarray:
        """Fractional coordinates of the atoms, shape (nat, 3).  Assigning a
        new value re-sends the cell to the C++ core on the next :meth:`define`
        call."""
        return np.array(self._xcoord, dtype="double", order="C")

    @xcoord.setter
    def xcoord(self, xcoord):
        self._need_transfer = True
        self._xcoord = np.array(xcoord, dtype="double", order="C")

    @property
    def numbers(self) -> np.ndarray:
        """Atomic numbers of the atoms, shape (nat,).  Assigning a new value
        re-sends the cell to the C++ core on the next :meth:`define` call."""
        return np.array(self._numbers, dtype="intc")

    @numbers.setter
    def numbers(self, numbers):
        self._need_transfer = True
        self._numbers = np.array(numbers, dtype="intc")

    @property
    def kind_names(self) -> OrderedDict:
        """Atomic number -> element symbol for the kinds in the cell (a copy)."""
        return OrderedDict(self._kind_names)

    def _transfer_parameters(self):
        if self._need_transfer and self._core is not None:
            self._set_cell()
            self._need_transfer = False

    def _set_cell(self):
        nat = len(self._xcoord)
        if len(self._numbers) != nat:
            raise ValueError(
                f"len(numbers) = {len(self._numbers)} does not match the "
                f"number of atoms in xcoord ({nat}).")
        self._kind_names = OrderedDict()
        for z in self._numbers:
            self._kind_names.setdefault(int(z), element_symbol(z))
        z_list = list(self._kind_names.keys())
        kind = np.array([z_list.index(int(z)) + 1 for z in self._numbers], dtype="intc")
        self._core.set_cell(self._lavec, self._xcoord, kind)
        self._core.set_element_names(list(self._kind_names.values()))
        # Non-magnetic defaults (the CLI always sets these; required for correct symmetry).
        self._core.set_magnetic_params(
            np.zeros((nat, 3), dtype="double"), False, 0, 1, "")

    # ---- verbosity / output ---------------------------------------------
    @property
    def verbosity(self) -> int:
        """Verbosity level (0 = silent, 1 = print progress to stdout)."""
        return self._verbosity

    @verbosity.setter
    def verbosity(self, verbosity):
        self._verbosity = int(verbosity)
        if self._core is not None:
            self._core.set_verbosity(self._verbosity)

    def set_verbosity(self, verbosity: int):
        """Set the verbosity level (same as assigning to :attr:`verbosity`)."""
        self.verbosity = verbosity

    @property
    def output_filename_prefix(self):
        """Prefix of the files written by the C++ core (e.g. the ``.cvscore``
        file of cross-validation); mirrors the CLI ``PREFIX`` tag."""
        return self._output_filename_prefix

    @output_filename_prefix.setter
    def output_filename_prefix(self, prefix):
        if isinstance(prefix, str):
            self._output_filename_prefix = prefix
            if self._core is not None:
                self._core.set_output_filename_prefix(prefix)

    @property
    def cv_l1_alpha(self) -> float:
        """Optimal L1 alpha found by the last cross-validation run."""
        self._check()
        return self._core.get_cv_l1_alpha()

    def get_cv_l1_alpha(self) -> float:
        """Legacy alias for :attr:`cv_l1_alpha`."""
        return self.cv_l1_alpha

    # ---- model definition ------------------------------------------------
    def define(self, maxorder: int, cutoff_radii=None, nbody=None,
               symmetrization_basis: str = "Lattice"):
        """Define the force-constant model (orders, cutoffs, interaction bodies).

        Parameters
        ----------
        maxorder : int
            Highest fc_order to include: 1 = harmonic only, 2 = up to cubic, ...
        cutoff_radii : array_like, shape (maxorder, nkd, nkd), optional
            Interaction cutoff radius in Bohr per order and kind pair.
            A negative value means no cutoff (all pairs included); the default
            ``None`` applies no cutoff at any order.
        nbody : sequence of int, length maxorder, optional
            Maximum number of distinct atoms in a cluster per order.
            Default: 2, 3, 4, ... (i.e. no restriction).
        symmetrization_basis : {'Lattice', 'Cartesian'}
            Basis used to symmetrize the force constants.
        """
        self._check()
        self._transfer_parameters()
        if maxorder < 1:
            raise ValueError(f"maxorder must be a positive integer, got {maxorder}.")
        nkd = len(self._kind_names)
        if nbody is None:
            nbody = [i + 2 for i in range(maxorder)]
        elif len(nbody) != maxorder:
            raise ValueError(
                f"len(nbody) = {len(nbody)} must equal maxorder = {maxorder}.")
        if cutoff_radii is None:
            # Pass explicit -1 (= no cutoff) values instead of letting the C++
            # side fall back to a nullptr: Cluster::define() re-allocates its
            # cutoff array without initializing it when given a nullptr, which
            # silently yields an (almost) empty interaction model.
            cut = np.full(maxorder * nkd * nkd, -1.0, dtype="double")
        else:
            cut = np.array(cutoff_radii, dtype="double", order="C").ravel()
            if cut.size != maxorder * nkd * nkd:
                raise ValueError(
                    f"cutoff_radii must have shape (maxorder={maxorder}, "
                    f"nkd={nkd}, nkd={nkd}), i.e. {maxorder * nkd * nkd} "
                    f"elements in total; got {cut.size}.")
        basis = str(symmetrization_basis).capitalize()
        if basis not in ("Lattice", "Cartesian"):
            raise ValueError(
                "symmetrization_basis must be 'Lattice' or 'Cartesian', "
                f"got {symmetrization_basis!r}.")
        self._core.set_forceconstant_basis(basis)
        self._core.define(maxorder, nkd, np.array(nbody, dtype="intc"), cut)
        self._core.init_fc_table()
        self._maxorder = maxorder
        self._defined = True
        self._patterns_ready = False

    def set_supercell(self, transmat_to_super, transmat_to_prim=None,
                      autoset_primcell: bool = False):
        """Declare the cell given to the constructor as the PRIMITIVE cell.

        The model is then built on the supercell obtained from
        ``transmat_to_super`` -- a (3, 3) integer matrix M with a_super = M
        a_prim.  Mirrors the CLI STRUCTURE_FILE(primitive)+SUPERCELL setup.
        Call before :meth:`define`.
        """
        self._check()
        ts = np.array(transmat_to_super, dtype="double", order="C").reshape(3, 3)
        tp = (np.eye(3) if transmat_to_prim is None
              else np.array(transmat_to_prim, dtype="double", order="C").reshape(3, 3))
        self._core.set_periodicity(np.array([1, 1, 1], dtype="intc"))
        self._core.set_transformation_matrices(ts, tp, int(autoset_primcell))
        # The supercell is generated inside the C++ core, so training-data
        # arrays are sized by the supercell, not by the constructor cell.
        self._supercell_active = True

    def set_constraint(self, translation: bool = True, rotation: bool = False):
        """ICONST=11: algebraic translational invariance (acoustic sum rule).

        Sets both the constraint mode and the algebraic flag (= ICONST//10),
        as the CLI does; required for the elastic-net optimizer.
        """
        self._check()
        if rotation:
            raise NotImplementedError(
                "rotation=True is not supported by this wrapper yet. "
                "The C++ core supports rotational invariance through the CLI "
                "(ICONST=2/3 with ROTAXIS and FCSYM_BASIS=Cartesian).")
        iconst = 11 if translation else 0
        self._core.set_constraint_mode(iconst)
        self._core.set_algebraic_constraint(iconst // 10)

    # ---- fixing / freezing FCs ------------------------------------------
    def fix_fc_from_file(self, fc_order: int, filename: str):
        """Fix the ``fc_order``-th FCs (1 = harmonic, 2 = cubic) to the values
        read from an ALM .xml/.h5 file.

        Mirrors the CLI FC2FIX/FC3FIX tags.  Call before :meth:`define`.
        """
        self._check()
        if fc_order not in (1, 2):
            raise ValueError(
                "fc_order must be 1 (harmonic) or 2 (cubic); fixing higher "
                f"orders from a file is not supported, got {fc_order}.")
        self._core.set_fc_file(fc_order + 1, str(filename))
        self._core.set_fc_fix(fc_order + 1, True)

    def fix_force_constants_from_file(self, order: int, fc_file: str):
        """Legacy alias for :meth:`fix_fc_from_file` using the derivative-order
        convention (order = 2 harmonic, 3 cubic)."""
        self.fix_fc_from_file(int(order) - 1, fc_file)

    def fix_fc(self, fc_values, fc_indices):
        """Fix a subset of FCs to given values during the fit (e.g. FC2 from a
        reference fit).

        Parameters
        ----------
        fc_values : array_like, shape (n,)
            Force-constant values in Ry/Bohr^(fc_order+1).
        fc_indices : array_like, shape (n, fc_order+1)
            Flattened indices 3*atom + xyz of each FC element.
        """
        self._check()
        fc_indices = np.array(fc_indices, dtype="intc", order="C")
        fc_values = np.array(fc_values, dtype="double")
        if fc_indices.ndim != 2:
            raise ValueError("fc_indices must be 2-dimensional.")
        if len(fc_indices) != len(fc_values):
            raise ValueError("fc_indices and fc_values must have the same length.")
        self._core.set_forceconstants_to_fix(
            [list(map(int, p)) for p in fc_indices],
            fc_values.tolist())

    def set_forceconstants_to_fix(self, intpair_fix, values_fix):
        """Legacy alias for :meth:`fix_fc` (note the swapped argument order)."""
        self.fix_fc(values_fix, intpair_fix)

    def freeze_fc(self, fc_values, fc_indices):
        """Legacy alias for :meth:`fix_fc`."""
        self.fix_fc(fc_values, fc_indices)

    # ---- training / validation data -------------------------------------
    def _as_data_array(self, data, name: str) -> np.ndarray:
        """Validate a displacement/force array: (nsnap, nat, 3) or (nsnap, 3*nat)."""
        arr = np.array(data, dtype="double", order="C")
        if self._supercell_active:
            # The supercell is built inside the C++ core, so its atom count is
            # not known here; check only the layout.
            if arr.ndim == 3 and arr.shape[2] == 3:
                return arr
            if arr.ndim == 2 and arr.shape[1] % 3 == 0:
                return arr
            raise ValueError(
                f"{name} must have shape (nsnap, nat_supercell, 3) or "
                f"(nsnap, 3*nat_supercell); got {arr.shape}.")
        nat = len(self._numbers)
        if arr.ndim == 3:
            if arr.shape[1:] != (nat, 3):
                raise ValueError(
                    f"{name} must have shape (nsnap, {nat}, 3) to match the "
                    f"{nat}-atom cell; got {arr.shape}.")
        elif arr.ndim == 2:
            if arr.shape[1] != 3 * nat:
                raise ValueError(
                    f"{name} must have shape (nsnap, {3 * nat}) when 2-D; "
                    f"got {arr.shape}.")
        else:
            raise ValueError(f"{name} must be 2-D or 3-D, got ndim={arr.ndim}.")
        return arr

    def _transfer_training_data(self):
        """Push staged displacements/forces to the C++ core (no-op when up to date)."""
        if not self._need_data_transfer or self._core is None:
            return
        if self._u is None or self._f is None:
            raise RuntimeError(
                "both displacements and forces must be set before fitting; "
                "use set_training_data(u, f) or assign both properties.")
        if self._u.shape != self._f.shape:
            raise ValueError(
                f"displacements {self._u.shape} and forces {self._f.shape} "
                "must have the same shape.")
        nsnap = self._u.shape[0]
        self._core.set_u_train(self._u.reshape(nsnap, -1))
        self._core.set_f_train(self._f.reshape(nsnap, -1))
        self._need_data_transfer = False

    @property
    def displacements(self) -> np.ndarray:
        """Training displacements in Bohr, shape (nsnap, nat, 3)."""
        if self._u is None:
            raise RuntimeError("displacements have not been set.")
        return np.array(self._u, dtype="double", order="C")

    @displacements.setter
    def displacements(self, u):
        self._u = self._as_data_array(u, "displacements")
        self._need_data_transfer = True

    @property
    def forces(self) -> np.ndarray:
        """Training forces in Ry/Bohr, shape (nsnap, nat, 3)."""
        if self._f is None:
            raise RuntimeError("forces have not been set.")
        return np.array(self._f, dtype="double", order="C")

    @forces.setter
    def forces(self, f):
        self._f = self._as_data_array(f, "forces")
        self._need_data_transfer = True

    def set_training_data(self, u, f):
        """Set the displacement-force training data.

        Parameters
        ----------
        u : array_like, shape (nsnap, nat, 3)
            Atomic displacements in Bohr.
        f : array_like, shape (nsnap, nat, 3)
            Forces in Ry/Bohr.
        """
        self._check()
        u = self._as_data_array(u, "u (displacements)")
        f = self._as_data_array(f, "f (forces)")
        if u.shape != f.shape:
            raise ValueError(
                f"u {u.shape} and f {f.shape} must have the same shape.")
        self._u, self._f = u, f
        self._need_data_transfer = True
        self._transfer_training_data()

    def set_displacement_and_force(self, u, f):
        """Deprecated alias for :meth:`set_training_data`."""
        warnings.warn("set_displacement_and_force is deprecated; use set_training_data.",
                      DeprecationWarning)
        self.set_training_data(u, f)

    def set_validation_data(self, u, f):
        """Set the validation set used by cross-validation (same shapes/units
        as :meth:`set_training_data`)."""
        self._check()
        u = self._as_data_array(u, "u (displacements)")
        f = self._as_data_array(f, "f (forces)")
        if u.shape != f.shape:
            raise ValueError(
                f"u {u.shape} and f {f.shape} must have the same shape.")
        n = u.shape[0]
        self._core.set_validation_data(u.reshape(n, -1), f.reshape(n, -1))

    # ---- optimizer control ----------------------------------------------
    @property
    def optimizer_control(self) -> dict:
        """ALM's optimizer settings as a dict, mirroring the CLI ``&optimize``
        tags (e.g. ``linear_model``, ``cross_validation``, ``l1_alpha``,
        ``l1_ratio``).  Assign a dict with only the keys to change; unknown
        keys raise ``KeyError``.  The legacy key ``mirror_image_conv`` is
        accepted as an alias of ``periodic_image_conv``."""
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

    def set_optimizer_control(self, params: dict):
        """Update a subset of the optimizer settings (see
        :attr:`optimizer_control`)."""
        self._check()
        oc = self._core.get_optimizer_control()
        valid = sorted(k for k in dir(oc)
                       if not k.startswith("_") and not callable(getattr(oc, k)))
        for k, v in params.items():
            key = self._OC_ALIASES.get(k, k)
            if key not in valid:
                raise KeyError(
                    f"unknown optimizer_control key: {k!r}; valid keys: {valid}")
            setattr(oc, key, v)
        self._core.set_optimizer_control(oc)

    # ---- run -------------------------------------------------------------
    def suggest(self):
        """Compute the displacement patterns needed to determine the FCs
        (retrieve them with :meth:`get_displacement_patterns`)."""
        self._check()
        self._check_defined()
        self._core.run_suggest()
        self._patterns_ready = True

    def optimize(self, solver: str = "dense") -> int:
        """Fit the force constants to the training data.

        Parameters
        ----------
        solver : str
            'dense' (LAPACK) or one of the sparse solvers (case-insensitive):
            SimplicialLDLT, SparseQR, ConjugateGradient,
            LeastSquaresConjugateGradient, BiCGSTAB, SuiteSparseQR, CHOLMOD.

        Returns
        -------
        int
            Solver info code (0 on success).

        Notes
        -----
        With ``optimizer_control['cross_validation']`` set, this runs
        cross-validation instead of a single fit; the optimal alpha is then
        available as :attr:`cv_l1_alpha`.
        """
        self._check()
        self._check_defined()
        self._transfer_training_data()
        if self._core.get_number_of_data() == 0:
            raise RuntimeError(
                "no training data; call set_training_data(u, f) (or assign the "
                "displacements/forces properties) before optimize().")
        if self._core.get_number_of_free_parameters() == 0:
            warnings.warn(
                "the model has no free FC parameters left after symmetry and "
                "constraint reduction; the fit will be empty. Check the "
                "structure and cutoff_radii.", stacklevel=2)
        canonical = {s.lower(): s for s in self._SPARSE_SOLVERS}
        oc = self._core.get_optimizer_control()
        if solver.lower() == "dense":
            oc.use_sparse_solver = 0
        elif solver.lower() in canonical:
            oc.use_sparse_solver = 1
            oc.sparsesolver = canonical[solver.lower()]
        else:
            raise ValueError(
                f"solver must be 'dense' or one of {self._SPARSE_SOLVERS}, "
                f"got {solver!r}.")
        self._core.set_optimizer_control(oc)
        return self._core.run_optimize()

    # ---- sensing matrix & force constants -------------------------------
    def get_matrix_elements(self) -> tuple:
        """Return the sensing matrix and force vector (A, b) for external solvers.

        A has shape (nrows, n_free_parameters) in Fortran order, b shape
        (nrows,); the fitted parameters solve ``A @ x = b`` and can be injected
        back with :meth:`set_fc`.
        """
        self._check()
        self._check_defined()
        self._transfer_training_data()
        if self._u is None or self._f is None:
            raise RuntimeError(
                "set training data via set_training_data() before get_matrix_elements().")
        amat, bvec, nrows, ncols = self._core.get_matrix_elements()
        return np.reshape(amat, (nrows, ncols), order="F"), bvec

    def get_fc(self, fc_order: int, mode: str = "origin",
               permutation: bool = True) -> tuple:
        """Return force constants as ``(values, elem_indices)``.

        Parameters
        ----------
        fc_order : int
            1 = harmonic (FC2), 2 = cubic (FC3), ...
        mode : {'origin', 'irreducible', 'all'}
            'origin': FCs whose first atom sits in the primitive cell at the
            origin; 'irreducible': the symmetry-irreducible set; 'all': FCs
            expanded over every pure translation.
        permutation : bool
            Include index-permuted copies of each element.

        Returns
        -------
        values : ndarray, shape (n,)
            Force constants in Ry/Bohr^(fc_order+1).
        elem_indices : ndarray, shape (n, fc_order+1)
            Flattened indices 3*atom + xyz of each element.

        Notes
        -----
        The element ordering is not guaranteed to be stable across calls
        (it differs e.g. between optimize() and set_fc()); always pair
        ``values`` with ``elem_indices`` instead of relying on position.
        """
        self._check()
        self._check_defined()
        self._check_fc_order(fc_order)
        perm = 1 if permutation else 0
        if mode == "origin":
            v, idx = self._core.get_fc_origin(fc_order, perm)
        elif mode in ("irreducible", "irred"):
            v, idx = self._core.get_fc_irreducible(fc_order)
        elif mode == "all":
            v, idx = self._core.get_fc_all(fc_order, perm)
        else:
            raise ValueError(
                f"mode must be 'origin', 'irreducible', or 'all', got {mode!r}.")
        return v, idx.reshape(-1, fc_order + 1)

    def set_fc(self, fc_in):
        """Inject a free-parameter FC vector (e.g. from an external regressor
        fit to the matrix returned by :meth:`get_matrix_elements`)."""
        self._check()
        self._check_defined()
        fc_in = np.array(fc_in, dtype="double", order="C")
        n = self._core.get_number_of_free_parameters()
        if fc_in.size != n:
            raise ValueError(f"expected {n} free FCs, got {fc_in.size}.")
        self._core.set_fc(fc_in)

    def get_number_of_free_parameters(self) -> int:
        """Number of free FC parameters after symmetry and constraint
        reduction; equals the column count of the matrix returned by
        :meth:`get_matrix_elements`."""
        self._check()
        self._check_defined()
        return self._core.get_number_of_free_parameters()

    def get_number_of_irred_fc_elements(self, fc_order: int) -> int:
        """Number of symmetry-irreducible FC elements of the given
        ``fc_order`` (1 = harmonic)."""
        self._check()
        self._check_defined()
        self._check_fc_order(fc_order)
        return self._core.get_number_of_irred_fc_elements(fc_order)

    # ---- mappings & displacement patterns -------------------------------
    def getmap_primitive_to_supercell(self) -> np.ndarray:
        """Return map_p2s, shape (num_trans, num_atoms_primitive) (legacy orientation)."""
        self._check()
        self._check_defined()
        # C++ returns (nat_primitive, num_trans); transpose to match the legacy API.
        m = np.array(self._core.get_atom_mapping_by_pure_translations(), dtype="intc")
        return m.T.copy()

    def get_displacement_patterns(self, fc_order: int) -> list:
        """Return a list (per pattern) of ``(atom_index, direction(3,), basis)``
        tuples; requires :meth:`suggest` to have run."""
        self._check()
        self._check_defined()
        self._check_fc_order(fc_order)
        if not self._patterns_ready:
            raise RuntimeError("call suggest() before get_displacement_patterns().")
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
    def save_fc(self, filename: str, format: str = "alamode",
                maxorder_to_save: int = None):
        """Write the force constants to a file.

        Parameters
        ----------
        filename : str
            Output path.
        format : str
            'alamode' (.xml), 'alamode_h5' (.h5, what anphon reads),
            'shengbte', 'shengbte4', 'qefc', or 'hessian'.
        maxorder_to_save : int, optional
            Highest fc_order to write (default: all defined orders).
        """
        self._check()
        self._check_defined()
        fmt = str(format).lower()
        if fmt not in self._SAVE_FORMATS:
            raise ValueError(
                f"format must be one of {self._SAVE_FORMATS}, got {format!r}.")
        if maxorder_to_save is None:
            maxorder_to_save = self._maxorder
        elif not (1 <= maxorder_to_save <= self._maxorder):
            raise ValueError(
                f"maxorder_to_save must be in [1, maxorder={self._maxorder}], "
                f"got {maxorder_to_save}.")
        self._core.save_fc(str(filename), fmt, maxorder_to_save)
