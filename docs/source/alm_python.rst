.. _alm_python:

ALM Python interface (external regressors)
==========================================

Overview
--------

ALAMODE 2.0dev ships a `nanobind <https://nanobind.readthedocs.io>`_ Python
interface to the ALM force-constant library (package name ``alm``).  Besides the
built-in solvers (ordinary least squares, elastic-net, adaptive LASSO), the
interface exposes ALM's **exact sensing matrix** :math:`(A, b)` so that *any*
external regressor can be used to determine the anharmonic interatomic force
constants (IFCs) and the result written back into an anphon-readable file.

The intended workflow is::

   build model  ->  A, b = get_matrix_elements()  ->  fit external regressor
                ->  set_fc(coefs)  ->  save_fc("out.h5", "alamode_h5")

This page documents the installation, the conventions you must respect, and a
complete worked example using the scikit-learn regressors **ARD**,
**BayesianRidge**, **RFECV**, and a small **Bayesian LASSO** (Gibbs) sampler.

.. note::

   This interface targets the ``2.0dev`` branch.  It replaces the legacy
   C/cython wrapper and keeps the same high-level API
   (``define`` / ``set_training_data`` / ``get_matrix_elements`` /
   ``get_fc`` / ``set_fc`` / ``save_fc`` / ``optimizer_control``).

.. note::

   **Object lifetime.**  The C++ instance is created in ``ALM.__init__``, so
   ``a = ALM(lavec, xcoord, numbers)`` is ready to use immediately and is freed
   automatically when ``a`` goes out of scope (the underlying C++ class is fully
   RAII).  A ``with`` block — or an explicit ``alm_delete()`` — is **optional**:
   use it only for a prompt, deterministic release, e.g. inside a loop that
   builds many large sensing matrices.


.. _alm_python_install:

Installation
------------

The Python interface compiles the ALM C++ sources together with the binding, so
it needs the same dependencies as ALM itself (spglib, Boost, Eigen, HDF5, a
BLAS/LAPACK backend, OpenMP) plus the build front-end (``nanobind``,
``scikit-build-core``) and, for the external regressors, ``scikit-learn``.

.. tip::

   The :ref:`pixi route <install_pixi>` sets up an equivalent project-local
   environment from the same environment file and builds the wrapper with a
   single command, ``pixi run 'cd python && pip install .'``.

Step 1. Create an environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest route is conda-forge, reusing the project environment file and
adding the wrapper-specific packages::

   % conda create -n almwrap -c conda-forge python=3.12
   % conda activate almwrap
   % conda env update -f etc/alamode-environment.yml      # spglib, boost, eigen, hdf5, compilers, cmake, numpy, scipy, h5py
   % conda install -c conda-forge scikit-learn threadpoolctl libopenblas
   % pip install "nanobind>=2.0" "scikit-build-core>=0.8"

Step 2. Build and install the wrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From the top of the source tree::

   % cd python
   % pip install .

Inside an activated conda environment (or a pixi environment, see
:ref:`install_pixi`), the wrapper's CMake setup automatically searches the
environment prefix (``$CONDA_PREFIX``) for spglib, Boost, Eigen, and HDF5, so
no further options are needed.  When the dependencies live in custom locations
instead, pass them explicitly::

   % pip install . \
       -C cmake.define.SPGLIB_ROOT=/path/to/spglib \
       -C cmake.define.EIGEN3_INCLUDE=/path/to/eigen3/include \
       -C cmake.define.BOOST_INCLUDE=/path/to/boost/include \
       -C cmake.define.HDF5_ROOT=/path/to/hdf5

If CMake selects the wrong BLAS, add ``-C cmake.define.BLA_VENDOR=OpenBLAS``.

Step 3. Verify
~~~~~~~~~~~~~~

.. code-block:: python

   import alm, numpy as np
   print(alm.__file__)
   print(hasattr(alm.ALM, "get_matrix_elements"))   # -> True

.. _alm_python_backend:

A note on the BLAS backend and performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* On **Linux**, conda's NumPy links a threaded **OpenBLAS**.
* On **macOS**, the pip/conda NumPy may link **Apple Accelerate**, whose
  eigensolver runs single-threaded and ignores ``OMP_NUM_THREADS`` /
  ``OPENBLAS_NUM_THREADS``.

Several external regressors (notably ``ARDRegression``) spend most of their time
in a symmetric eigendecomposition that **parallelizes poorly** regardless of the
backend (measured speed-up at :math:`p\approx8000` was only ~1.16x on 12 threads).
The effective way to use many cores is therefore **process-level parallelism
across independent fits** (different ``ndata`` / functionals / orders), not BLAS
threads — see :ref:`alm_python_performance`.


.. _alm_python_conventions:

Conventions you must respect
----------------------------

These follow the ALM command-line interface (CLI) and are easy to get wrong:

.. warning::

   **Units.**

   * The lattice vectors passed to ``ALM(...)`` must be in **Bohr**
     (multiply Å by ``1.8897259886``).
   * The displacements ``u`` and forces ``f`` in a ``DFSET`` are read **as-is in
     atomic units** (Bohr and Ry/Bohr) — do **not** convert them.
   * The force constants returned by ``get_fc`` / written by ``save_fc`` are
     converted to eV/Å\ :sup:`n` by ALM.

* ``ALM(lavec, xcoord, numbers)`` takes the **primitive** cell: ``lavec`` (3x3,
  Bohr), ``xcoord`` (fractional, ``(nat, 3)``), ``numbers`` (atomic numbers
  :math:`Z`).  Map it to the supercell with ``set_supercell``.
* ``cutoff_radii`` has shape ``(maxorder, nkd, nkd)`` where ``nkd`` is the number
  of atomic species.  Values are **in bohr** (same convention as the CLI
  ``&cutoff`` field).  Use a **negative** value for "no cutoff".  Index 0 is the
  harmonic term, index 1 the cubic, etc.
* ``set_constraint(translation=True)`` applies the algebraic translational
  invariance (acoustic sum rule, ``ICONST=11``).  ``get_matrix_elements`` then
  returns the **constraint-reduced** matrix whose number of columns equals
  ``get_number_of_free_parameters()``.


.. _alm_python_inputs:

Example inputs
--------------

We use the cubic **BaTiO3** example shipped with ALAMODE
(``example/BaTiO3/anharm_IFCs/``).  It determines cubic and quartic IFCs of a
:math:`2\times2\times2` (40-atom) supercell with the harmonic FCs fixed —
exactly the setup of the CLI input ``4_optimize/BTO_alm_opt.in``:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Input
     - Role
   * - ``cBTO222_harmonic.xml``
     - harmonic (FC2) force constants, fixed during the fit (``FC2FIX``)
   * - ``3_cv/reference/DFSET_AIMD_random``
     - 80 displacement-force snapshots of the 40-atom supercell (atomic units;
       produced by the example's MD -> DFSET workflow in steps 1-2, with a
       reference copy shipped under ``3_cv/reference/``)
   * - 5-atom cubic primitive cell
     - Ba/Ti/O, lattice constant :math:`a_0 = 7.5316` bohr (defined inline below)

Model (from the example input): ``NORDER = 3``, ``NBODY = 2 3 3``, cutoff radii
**15.0 bohr (cubic)** and **9.0 bohr (quartic)**, harmonic term fixed.

.. note::

   The CLI input supplies the **supercell** plus ``PRIMCELL`` (super→prim).  The
   Python wrapper instead takes the **primitive** cell and maps it to the
   supercell with ``set_supercell`` (``transmat_to_super = diag(2, 2, 2)``).
   Both describe the same 40-atom system.

A ``DFSET`` lists, for each of the ``nat_super`` atoms, one line
``ux uy uz  fx fy fz`` (Bohr, Ry/Bohr), concatenated over all snapshots.  It is
read with this dependency-free helper (or use ``ase`` / your own reader):

.. code-block:: python

   import numpy as np

   def read_dfset(path, nat):
       """Return (u, f) each of shape (nsnap, nat, 3) in atomic units (Bohr, Ry/Bohr)."""
       rows = [ln.split() for ln in open(path) if ln.strip() and not ln.lstrip().startswith("#")]
       a = np.array([[float(x) for x in r[:6]] for r in rows])
       n = a.shape[0] // nat
       a = a[:n * nat].reshape(n, nat, 6)
       return a[:, :, :3].copy(), a[:, :, 3:].copy()


.. _alm_python_example:

Complete example: fitting with an external regressor
----------------------------------------------------

The function below defines the BaTiO3 primitive cell, extracts ALM's exact
:math:`(A, b)`, **column-normalizes** it (so the regularization treats every
column on an equal footing), fits the chosen regressor, writes the solution
back, and saves an anphon-readable ``.h5``.  It also reports ALM's **relative
error** — the residual normalized by the **total forces** (not by the
anharmonic residual), which is what ALM's ``.cvscore`` files report.

.. code-block:: python

   import numpy as np
   import alm

   # --- cubic BaTiO3 primitive cell (5 atoms); lattice in BOHR (atomic units) ---
   A0 = 7.53159676409                                # bohr
   LAVEC = A0 * np.eye(3)
   XCOORD = np.array([[0.0, 0.0, 0.0],               # Ba
                      [0.5, 0.5, 0.5],               # Ti
                      [0.5, 0.5, 0.0],               # O
                      [0.5, 0.0, 0.5],               # O
                      [0.0, 0.5, 0.5]])              # O
   NUMBERS = [56, 22, 8, 8, 8]                       # atomic numbers: Ba, Ti, O, O, O
   TRANSMAT = np.diag([2, 2, 2]).astype(float)       # 2x2x2 supercell (40 atoms)

   def regressor_coef(est, p):
       """Length-p coefficient vector of a fitted regressor (handles RFECV, which
       exposes the fit only on its selected support)."""
       if hasattr(est, "support_"):                   # RFECV / RFE -> expand to full length
           coef = np.zeros(p); coef[est.support_] = est.estimator_.coef_
           return coef
       return est.coef_                               # ARD, BayesianRidge, LASSO, ...

   def fit_bto(fc2_xml, dfset, regressor, save_to, ndata=80):
       nat_super = len(NUMBERS) * int(round(np.linalg.det(TRANSMAT)))   # 40
       u, f = read_dfset(dfset, nat_super)
       nkd = len(set(NUMBERS))                        # 3 species (Ba, Ti, O)

       cut = np.full((3, nkd, nkd), -1.0)             # (maxorder, nkd, nkd) in BOHR; -1 = no cutoff
       cut[1] = 15.0                                  # cubic   cutoff (bohr)
       cut[2] = 9.0                                   # quartic cutoff (bohr)

       a = alm.ALM(LAVEC, XCOORD, NUMBERS, verbosity=0)   # ready to use immediately
       a.set_supercell(TRANSMAT)                      # primitive -> 2x2x2 supercell
       a.fix_force_constants_from_file(2, fc2_xml)    # FC2FIX (.xml or .h5)
       a.define(maxorder=3, cutoff_radii=cut, nbody=[2, 3, 3])
       a.set_constraint(translation=True)             # acoustic sum rule (ICONST=11)
       a.set_training_data(u[:ndata], f[:ndata])

       A, b = a.get_matrix_elements()                 # ALM's exact sensing matrix
       p = a.get_number_of_free_parameters()          # == A.shape[1]
       col = np.linalg.norm(A, axis=0); col[col == 0] = 1.0
       est = regressor.fit(A / col, b)                 # fit the normalized matrix
       coef_s = regressor_coef(est, p)                 # length-p (handles RFECV)
       coef = coef_s / col                             # back to physical units
       a.set_fc(coef)                                  # inject the solution

       # ALM-convention relative error (residual / total forces):
       relerr = np.linalg.norm((A / col) @ coef_s - b) / np.linalg.norm(f[:ndata].ravel())
       nnz = int(np.sum(np.abs(coef) > 1e-10))
       print(f"p={p}  nonzero={nnz}  relerr={relerr:.4f}")

       a.save_fc(save_to, format="alamode_h5")         # anphon-readable .h5
       return relerr                                   # 'a' is freed when it goes out of scope

Run it from ``example/BaTiO3/anharm_IFCs/`` with ARD:

.. code-block:: python

   from sklearn.linear_model import ARDRegression
   fit_bto(
       fc2_xml="cBTO222_harmonic.xml",
       dfset="3_cv/reference/DFSET_AIMD_random",      # 80 snapshots of the 40-atom cell
       regressor=ARDRegression(copy_X=False),
       save_to="cBTO222_ard.h5")

The resulting ``cBTO222_ard.h5`` can be used directly by anphon (``FCSXML``) for
the thermal-conductivity and Grüneisen calculations in ``example/BaTiO3``.

.. note::

   For a structure stored as a VASP POSCAR (Å), read the cell and convert to
   bohr (``lavec_bohr = lavec_angstrom * 1.8897259886``, e.g. via
   ``ase.io.read``).  The displacements and forces in a ``DFSET`` are already in
   atomic units and must **not** be converted.


.. _alm_python_regressors:

The external regressors
-----------------------

Any estimator exposing a ``.coef_`` after ``.fit(X, y)`` works.  The four below
cover the common compressive-sensing alternatives to elastic-net / adaptive
LASSO.

ARD (Automatic Relevance Determination)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sparse Bayesian linear regression (the regression form of the relevance vector
machine).  Each coefficient gets its own precision (inverse variance) that is
optimized by evidence maximization; irrelevant coefficients are driven to zero.
It is **self-tuning** (no cross-validation needed) and typically the **sparsest**
solution.

.. code-block:: python

   from sklearn.linear_model import ARDRegression
   reg = ARDRegression(copy_X=False)        # copy_X=False saves memory at large p

BayesianRidge
~~~~~~~~~~~~~~

Bayesian linear regression with a single shared Gaussian prior precision (ridge-
like).  Self-tuning and fast, but the solution is **dense** (no exact zeros).

.. code-block:: python

   from sklearn.linear_model import BayesianRidge
   reg = BayesianRidge()

RFECV (recursive feature elimination with CV)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wraps a base estimator and recursively prunes the least important features,
choosing the number of features by cross-validation.  It is the most expensive
option and can over-prune in the data-scarce regime, so it is best for small
``p``.

.. code-block:: python

   from sklearn.feature_selection import RFECV
   from sklearn.linear_model import LinearRegression
   reg = RFECV(LinearRegression(), step=0.15, cv=3, min_features_to_select=5, n_jobs=-1)
   # RFECV exposes .estimator_.coef_ on the selected support; expand back to p:
   #   coef = np.zeros(p); coef[reg.support_] = reg.estimator_.coef_

Bayesian LASSO (Park & Casella 2008)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A Laplace prior implemented as a scale-mixture of normals, sampled with a Gibbs
sampler.  scikit-learn has no Bayesian LASSO, so a compact implementation is
given below.  The **posterior mean** is used as the point estimate (note that it
shrinks but does not produce exact zeros, so it behaves like a "soft LASSO").

.. code-block:: python

   import numpy as np
   import scipy.linalg as sla

   def bayesian_lasso(X, y, n_iter=1200, burn=400, seed=0):
       """Posterior-mean coefficients of the Park-Casella Bayesian LASSO.
       X must already be column-normalized; returns coef in the X-scaling."""
       rng = np.random.default_rng(seed)
       n, p = X.shape
       XtX = X.T @ X; Xty = X.T @ y; yty = float(y @ y)
       beta = np.linalg.lstsq(X, y, rcond=None)[0]
       sigma2 = max(float(np.var(y - X @ beta)), 1e-12)
       invtau2 = np.ones(p); lam2 = 1.0; r, delta = 1.0, 0.1
       acc = np.zeros(p); cnt = 0
       for it in range(n_iter):
           A = XtX + np.diag(invtau2)
           L = np.linalg.cholesky(A)
           mu = sla.cho_solve((L, True), Xty)
           z = rng.standard_normal(p)
           beta = mu + np.sqrt(sigma2) * sla.solve_triangular(L, z, lower=True, trans="T")
           resid = yty - 2.0 * beta @ Xty + beta @ (XtX @ beta)
           shape = (n - 1) / 2.0 + p / 2.0
           scale = 0.5 * max(resid, 1e-12) + 0.5 * float(np.sum(invtau2 * beta ** 2))
           sigma2 = 1.0 / rng.gamma(shape, 1.0 / scale)
           mu_p = np.sqrt(lam2 * sigma2) / np.abs(beta).clip(1e-12)
           invtau2 = rng.wald(mu_p, lam2).clip(1e-12, 1e12)
           lam2 = rng.gamma(p + r, 1.0 / (0.5 * float(np.sum(1.0 / invtau2)) + delta))
           if it >= burn:
               acc += beta; cnt += 1
       return acc / cnt

   # use it like a regressor:
   #   coef_s = bayesian_lasso(A / col, b); coef = coef_s / col; a.set_fc(coef)

At a glance:

.. list-table::
   :header-rows: 1
   :widths: 22 30 18 30

   * - Regressor
     - Prior / mechanism
     - Sparse?
     - Notes
   * - ARD
     - per-coefficient Gaussian precision (evidence max.)
     - yes (strong)
     - self-tuning; sparsest; eigendecomposition cost ~ :math:`O(p^3)`
   * - BayesianRidge
     - single shared Gaussian precision
     - no (dense)
     - self-tuning; fast; good when :math:`n \gg p`
   * - Bayesian LASSO
     - Laplace (scale-mixture), Gibbs
     - no (soft)
     - posterior mean shrinks small terms; MCMC cost
   * - RFECV
     - recursive elimination + CV
     - yes (hard)
     - expensive; best for small :math:`p`


.. _alm_python_performance:

Performance: scaling and parallelism
-------------------------------------

* ``get_matrix_elements`` returns the **constraint-reduced** matrix, so its
  column count is ``get_number_of_free_parameters()`` (much smaller than the raw
  irreducible count).  Reuse the same ``(A, b)`` for several regressors.
* ARD / RFECV scale roughly as :math:`O(p^3)` per iteration; for large cutoffs
  (:math:`p \gtrsim 10^4`) a single fit can take tens of minutes.  LASSO and
  BayesianRidge scale better to large :math:`p`.
* Threads do **not** help much (the bottleneck eigendecomposition is essentially
  serial).  To use many cores, run independent fits as **separate processes**:

  .. code-block:: python

     from concurrent.futures import ProcessPoolExecutor

     def run_one(case):
         # build, fit, save one (functional, order, ndata) case; returns a summary dict
         ...

     cases = [...]  # list of independent fits
     with ProcessPoolExecutor(max_workers=6) as ex:
         results = list(ex.map(run_one, cases))

  Budget memory accordingly (a large fit at :math:`p\approx1.7\times10^4` needs
  ~20 GB); reduce ``max_workers`` if a worker is killed (``BrokenProcessPool``).

* To speed up a single large fit, pre-screen the columns with a cheap LASSO and
  run the expensive regressor on the surviving support, or lower
  ``ARDRegression(max_iter=...)``.


.. _alm_python_api:

API reference
-------------

The public interface is the high-level :class:`alm.ALM` class, a context manager
wrapping the compiled core.  The members below are generated from the in-source
docstrings.

.. currentmodule:: alm

.. autoclass:: ALM
   :members:
   :member-order: bysource

The ``optimizer_control`` property is a ``dict`` mirroring ALM's ``&optimize``
tags; the keys most relevant for comparison with the external regressors are
``linear_model`` (1 = ordinary least squares, 2 = elastic-net,
3 = adaptive LASSO), ``cross_validation``, ``l1_alpha``, and ``l1_ratio``.


See also
--------

* :doc:`alm_root` — the ALM command-line program and its input tags.
* The ``optimizer_control`` property exposes ALM's native solvers
  (``linear_model`` = 1 OLS, 2 elastic-net, 3 adaptive LASSO) with
  cross-validation, for comparison with the external regressors.
