.. |umulaut_u|    unicode:: U+00FC


ANPHON: Input files
-------------------

Format of input files
~~~~~~~~~~~~~~~~~~~~~

Each input file should consist of entry fields.
Available entry fields are 

**&general**, **&cell**, **&scph**, **&qha**, **&relax**, **&kpoint**, **&strain**, **&displace**, **&analysis**, and **&kappa**.

The format of the input file is the same as that of *alm* which can be found :ref:`here <reference_input_alm>`.


.. _label_inputvar_anphon:

List of supported input variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
   :widths: 25, 25, 25, 25

   **&general**
   :ref:`ALLOW_UNCONVERGED <anphon_allow_unconverged>`, :ref:`BCONNECT <anphon_bconnect>`, :ref:`BORNINFO <anphon_borninfo>`, :ref:`BORNSYM <anphon_bornsym>`
   :ref:`CLASSICAL <anphon_classical>`, :ref:`DFC2FILE <anphon_dfc2file>`, :ref:`EMIN <anphon_emin>`, :ref:`EPSILON <anphon_epsilon>`
   :ref:`FC2FILE <anphon_fc2file>`, :ref:`FC2_TEMPERATURE <anphon_fc2_temperature>`, :ref:`FCSFILE <anphon_fcsfile>`, :ref:`FILE_FORMAT <anphon_file_format>`
   :ref:`ISMEAR <anphon_ismear>`, :ref:`KD <anphon_kd>`, :ref:`MASS <anphon_mass>`, :ref:`MODE <anphon_mode>`
   :ref:`NA_SIGMA <anphon_na_sigma>`, :ref:`NONANALYTIC <anphon_nonanalytic>`, :ref:`PREFIX <anphon_prefix>`, :ref:`PRINTSYM <anphon_printsym>`
   :ref:`TMIN <anphon_tmin>`, :ref:`TOLERANCE <anphon_tolerance>`, :ref:`TRISYM <anphon_trisym>`
   **&scph**
   :ref:`BUBBLE <anphon_bubble>`, :ref:`IALGO <anphon_ialgo>`, :ref:`KMESH_INTERPOLATE <anphon_kmesh_interpolate>`, :ref:`KMESH_SCPH <anphon_kmesh_scph>`
   :ref:`LOWER_TEMP <anphon_lower_temp>`, :ref:`MAXITER <anphon_maxiter>`, :ref:`MIXALPHA <anphon_mixalpha>`, :ref:`RELAX_STR <anphon_relax_str>`
   :ref:`RESTART_SCPH <anphon_restart_scph>`, :ref:`SELF_OFFDIAG <anphon_self_offdiag>`, :ref:`TOL_SCPH <anphon_tol_scph>`, :ref:`WARMSTART <anphon_warmstart>`
   **&qha**
   :ref:`KMESH_INTERPOLATE <anphon_qha_kmesh_interpolate>`, :ref:`KMESH_QHA <anphon_qha_kmesh_qha>`, :ref:`LOWER_TEMP <anphon_qha_lower_temp>`, :ref:`QHA_SCHEME <anphon_qha_scheme>`
   :ref:`RELAX_STR <anphon_qha_relax_str>`
   **&relax**
   :ref:`ADD_HESS_DIAG <anphon_add_hess_diag>`, :ref:`ALPHA_STEEPEST_DECENT <anphon_alpha_steepest_decent>`, :ref:`CELL_CONV_TOL <anphon_cell_conv_tol>`, :ref:`COOLING_U0_INDEX <anphon_cooling_u0_index>`
   :ref:`COOLING_U0_THR <anphon_cooling_u0_thr>`, :ref:`COORD_CONV_TOL <anphon_coord_conv_tol>`, :ref:`MAX_STR_ITER <anphon_max_str_iter>`, :ref:`MIXBETA_CELL <anphon_mixbeta_cell>`
   :ref:`MIXBETA_COORD <anphon_mixbeta_coord>`,  :ref:`RELAX_ALGO <anphon_relax_algo>`, :ref:`RENORM_2TO1ST <anphon_renorm_2to1st>`, :ref:`RENORM_34TO1ST <anphon_renorm_34to1st>`
   :ref:`RENORM_3TO2ND <anphon_renorm_3to2nd>`, :ref:`SET_INIT_STR <anphon_set_init_str>`, :ref:`STAT_PRESSURE <anphon_stat_pressure>`, :ref:`STRAIN_IFC_DIR <anphon_strain_ifc_dir>`
   **&analysis**
   :ref:`ANIME <anphon_anime>`, :ref:`ANIME_CELLSIZE <anphon_anime_cellsize>`, :ref:`ANIME_FORMAT <anphon_anime_format>`, :ref:`ANIME_FRAMES <anphon_anime_frames>`
   :ref:`GRUNEISEN <anphon_gruneisen>`, :ref:`PDOS <anphon_pdos>`, :ref:`PRINTEVEC <anphon_printevec>`, :ref:`PRINTMSD <anphon_printmsd>`
   :ref:`PRINTPR <anphon_printpr>`, :ref:`PRINTVEL <anphon_printvel>`, :ref:`PRINTXSF <anphon_printxsf>`, :ref:`SHIFT_UCORR <anphon_shift_ucorr>`
   :ref:`SPS <anphon_sps>`, :ref:`TDOS <anphon_tdos>`, :ref:`UCORR <anphon_ucorr>`, :ref:`ZMODE <anphon_zmode>`
   **&kappa**
   :ref:`INCLUDE_4PH <anphon_include_4ph>`, :ref:`ISOFACT <anphon_isofact>`, :ref:`ISOTOPE <anphon_isotope>`, :ref:`KAPPA_COHERENT <anphon_kappa_coherent>`
   :ref:`KAPPA_SPEC <anphon_kappa_spec>`, :ref:`RESTART <anphon_restart>`, :ref:`RESTART_4PH <anphon_restart_4ph>`




Description of input variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"&general"-field
++++++++++++++++

.. _anphon_prefix:

* **PREFIX**-tag : Job prefix to be used for names of output files

 :Default:  None
 :Type: String

````

.. _anphon_mode:

* **MODE**-tag = phonons | RTA

 ========= ==============================================================
  phonons  | Calculate phonon dispersion relation, phonon DOS, 
           | Gr\ |umulaut_u|\ neisen parameters etc.

    RTA    | Calculate phonon lifetimes and lattice thermal conductivity 
           | based on the Boltzmann transport equation (BTE) 
           | with the relaxation time approximation (RTA).

   SCPH    | Calculate temperature dependent phonon dispersion curves
           | by the self-consistent phonon method.
 ========= ==============================================================

 :Default: None
 :Type: String

````

.. _anphon_kd:

* **KD**-tag : List of the atomic species

 :Default: None
 :Type: Array of strings
 :Example: In the case of GaAs, it should be ``KD = Ga As``.

````

.. _anphon_mass:

* MASS-tag : List of atomic masses, one value per atomic species (in the order of the ``KD``-tag)

 :Default: Standard atomic weight of elements given by the ``KD``-tag
 :Type: Array of double
 :Example: In the case of Bi\ :sub:`2`\ Te\ :sub:`3`, ``MASS`` should be ``MASS = 208.98 127.60``.

````

.. _anphon_fcsfile:

* **FCSFILE**-tag : File (XML or HDF5) containing force constants generated by the program *alm*

 :Default: None
 :Type: String
 :Description: Either ``FCSFILE`` or ``FC2FILE`` must be given to start a phonon calculation.

 .. note::

     ``FCSFILE`` and ``FC2FILE`` replace the former ``FCSXML`` and ``FC2XML`` tags, which are no longer accepted.

 .. note::

     For HDF5 files, the ``unit`` attributes stored in the file (e.g. by *alm* with ``FCS_UNIT_OUTPUT = eV/angstrom``) are honored: the lattice vectors, shift vectors, and force constant values are converted to the internal Rydberg atomic units automatically. Files without unit attributes are assumed to be in bohr and Ry/bohr\ :sup:`n`. XML files are always in Rydberg atomic units.

````

.. _anphon_fc2file:

* FC2FILE-tag : File containing harmonic force constants for a different supercell size

 :Default: None
 :Type: String
 :Description: When ``FC2FILE`` is given, the harmonic force constants in this file are used for calculating dynamical matrices. It is possible to use supercells of different sizes for harmonic and anharmonic terms, which are specified by ``FC2FILE`` and ``FCSFILE`` respectively. Analogously, ``FC3FILE`` and ``FC4FILE`` can be used to supply the cubic and quartic force constants from separate files; when they are not given, the corresponding terms are read from ``FCSFILE``.

````

.. _anphon_tolerance:

* TOLERANCE-tag : Tolerance for finding symmetry operations

 :Default: 1.0e-3
 :Type: Double

````

.. _anphon_printsym:

* PRINTSYM-tag = 0 | 1

 === =======================================================
  0   Symmetry operations won’t be saved in “SYMM_INFO_PRIM”
  1   Symmetry operations will be saved in “SYMM_INFO_PRIM”
 === =======================================================

 :Default: 0
 :type: Integer

````

.. _anphon_nonanalytic:

* NONANALYTIC-tag = 0 | 1 | 2 | 3

 === ===================================================================================
  0  | Non-analytic correction is not considered.

  1  | Include the non-analytic correction by the damping method proposed by Parlinski.

  2  | Include the non-analytic correction by the mixed-space approach 

  3  | Include the non-analytic correction by the Ewald method
 === ===================================================================================

 :Default: 0
 :Type: Integer
 :Description: When ``NONANALYTIC > 0``, appropriate ``BORNINFO`` needs to be given. If ``NONANALYTIC = 1``, one may need to adjust the ``NA_SIGMA`` value to obtain reasonably smooth dispersion curves.

````

.. _anphon_na_sigma:

* NA_SIGMA-tag : Damping factor for the non-analytic term

 :Default: 0.1
 :Type: Double
 :Description: Used when ``NONANALYTIC = 1``. The definition of ``NA_SIGMA`` is described in the formalism section.

````

.. _anphon_borninfo:

* BORNINFO-tag : File containing the macroscopic dielectric tensor and Born effective charges for the non-analytic correction
 
 :Default: None
 :Type: String
 :Description: The details of the file format can be found :ref:`here <label_format_BORNINFO>`.

````

.. _anphon_bornsym:

* BORNSYM-tag = 0 | 1
 
 === =================================================================
  0   Do not symmetrize Born effective charges
  1   Symmetrize Born effective charges by using point group symmetry
 === =================================================================

 :Default: 0
 :Type: Integer

````

.. _anphon_tmin:

* TMIN, TMAX, DT-tags : Temperature range and its stride in units of Kelvin

 :Default: ``TMIN = 0``, ``TMAX = 1000``, ``DT = 10``
 :Type: Double

````

.. _anphon_emin:

* EMIN, EMAX, DELTA_E-tags : Energy range and its stride in units of kayser (cm\ :sup:`-1`)

 :Default: ``EMIN`` and ``EMAX`` are set automatically from the eigenfrequencies as of ver. 1.5.0. The default value for ``DELTA_E`` is 10.0.
 :Type: Double

````

.. _anphon_ismear:

* ISMEAR-tag = -1 | 0 | 1 | 2

 === =======================================================
  -1  Tetrahedron method
  0   Lorentzian smearing with width of ``EPSILON``
  1   Gaussian smearing with width of ``EPSILON``
  2   Adaptive Gaussian smearing
 === =======================================================

 :Default: -1
 :Type: Integer
 :Description: ``ISMEAR`` specifies the method for Brillouin zone integration

````

.. _anphon_epsilon:

* EPSILON-tag : Smearing width in units of Kayser (cm\ :sup:`-1`)

 :Default: 10.0
 :Type: Double
 :Description: This variable is neglected when ``ISMEAR = -1``

````

.. _anphon_bconnect:

* BCONNECT-tag = 0 | 1 | 2 

 === ===================================================================================
  0   | Phonon band is saved without change (sorted in order of energy)

  1   | Phonon band is connected by using the similarity of eigenvectors.

  2   | Same as ``BCONNECT=1``. In addition, information about the connectivity is 
      | saved as ``PREFIX.connection``.
 === ===================================================================================

 :Default: 0
 :Type: Integer
 :Description: The algorithm for connecting a band structure is described here_.

 .. _here : https://www.slideshare.net/TakeshiNishimatsu/two-efficient-algorithms-for-drawing-accurate-and-beautiful-phonon-dispersion

````

.. _anphon_classical:

* CLASSICAL-tag = 0 | 1

 === =======================================================
  0   Use quantum statistics (default)
  1   Use classical statistics
 === =======================================================

 :Default: 0
 :Type: Integer
 :Description: When ``CLASSICAL = 1``, all thermodynamic functions including the occupation function, heat capacity, and mean square displacements are calculated using the classical formulae. This option may be useful when comparing the lattice dynamics and molecular dynamics results.


 .. list-table:: Comparison of quantum and classical values
    :header-rows: 1

    * - Function
      - Quantum (``CLASSICAL = 0``)
      - Classical (``CLASSICAL = 1``)
    * - Occupation number
      - :math:`\displaystyle n_\mathrm{B}=\frac{1}{\exp(\beta\hbar\omega) - 1}`
      - :math:`\displaystyle n_\mathrm{C}=\frac{1}{\beta\hbar\omega}`
    * - Mode specific heat
      - :math:`\displaystyle c_{q} = k_{\mathrm{B}}\left[\frac{\beta\hbar\omega_q}{2}\mathrm{csch}\bigg({\frac{\beta\hbar\omega_q}{2}}\bigg)\right]^2`
      - :math:`\displaystyle c_{q} = k_{\mathrm{B}}`
    * - MSD of normal mode :math:`\braket{Q^{*}_qQ_q}`
      - :math:`\displaystyle \frac{\hbar (1 + n_{\mathrm{B}})}{2\omega_q}`
      - :math:`\displaystyle \frac{1}{\beta\omega_{q}^{2}}`



````

.. _anphon_trisym:

* TRISYM-tag : Flag to use symmetry operations to reduce the number of triples of :math:`k` points for self-energy calculations

 === =======================================================
  0   Symmetry will not be used
  1   Use symmetry to reduce triples of :math:`k` points
 === =======================================================
 
 :Default: 1
 :Type: Integer
 :Description: This variable is used only when ``MODE = RTA``.

 .. Note::

  ``TRISYM = 1`` can reduce the computational cost, but phonon linewidth stored to the file
  ``PREFIX``.result needs to be averaged at points of degeneracy. 
  For that purpose, a subsidiary program ``analyze_phonons.py`` should be used.

````

.. _anphon_file_format:

* FILE_FORMAT-tag : Format of the restart/state files

 ====== =========================================================================
  h5     Unified crash-safe HDF5 files (``PREFIX``.kappa.h5 for ``MODE = RTA``;
         ``PREFIX``.scph.h5 / ``PREFIX``.qha.h5 for ``MODE = SCPH`` / ``QHA``)
  text   Legacy text files (``PREFIX``.result, ``PREFIX``.scph_dymat, etc.)
 ====== =========================================================================

 :Default: h5
 :Type: String
 :Description: The text format is kept for one transition release. Human-readable
               outputs (``PREFIX``.kl, ``PREFIX``.scph_thermo, ``PREFIX``.scph_dfc2,
               ``PREFIX``.V0, ...) are written in both modes.

````

.. _anphon_fc2_temperature:

* FC2_TEMPERATURE-tag : Temperature (K) at which the renormalized FC2 is loaded

 :Default: None
 :Type: Double
 :Description: When ``FCSFILE`` or ``FC2FILE`` points at a ``PREFIX``.scph.h5 /
               ``PREFIX``.qha.h5 file produced by an SCPH/QHA run, the effective
               (temperature-renormalized) harmonic force constants at this
               temperature are loaded directly — no ``dfc2.py`` round-trip is
               needed. Note that this self-contained route folds the harmonic
               FC2 onto the ``KMESH_INTERPOLATE`` cell; to combine the
               correction with a harmonic FC2 of a larger supercell, use
               :ref:`DFC2FILE <anphon_dfc2file>` instead. The temperature must
               be one of the values on the file's temperature grid. The stored FC2 is the short-range part; the
               non-analytic long-range term is added at runtime from ``BORNINFO``
               as usual. In ``MODE = RTA``, this tag also switches
               ``PREFIX``.kappa.h5 to its temperature-resolved layout so that
               runs at different basis temperatures accumulate into one file
               (set ``TMIN = TMAX = FC2_TEMPERATURE`` per run). Temperatures whose
               SCPH iteration or structural optimization did not converge are
               refused unless :ref:`ALLOW_UNCONVERGED <anphon_allow_unconverged>`
               is set.

````

.. _anphon_dfc2file:

* DFC2FILE-tag : SCPH/QHA state file supplying the anharmonic FC2 correction

 :Default: None
 :Type: String
 :Description: Adds the anharmonic FC2 correction :math:`\Delta\Phi_{ij}` stored in a
               ``PREFIX``.scph.h5 / ``PREFIX``.qha.h5 file (at
               :ref:`FC2_TEMPERATURE <anphon_fc2_temperature>`, which must be given)
               onto the harmonic FC2 provided by ``FCSFILE`` or ``FC2FILE``. This is
               the native form of the legacy ``dfc2.py`` workflow: the harmonic FC2
               may come from a **larger supercell** than the SCPH cell, which is
               justified because the anharmonic correction is usually shorter ranged
               than the harmonic force constants. The two supercells must be
               commensurate tilings of the same primitive cell. The convergence
               flags of the state file are enforced as for a direct
               ``FC2_TEMPERATURE`` read.

               ``DFC2FILE`` never replaces the harmonic FC2 — it is purely additive.
               The base FC2 follows the usual precedence (``FC2FILE`` if given,
               otherwise ``FCSFILE``), and ``FC2_TEMPERATURE`` refers to ``DFC2FILE``
               when it is present. The resulting combinations are:

               .. list-table::
                  :header-rows: 1
                  :widths: 55 45

                  * - Input combination
                    - Resulting FC2
                  * - ``FCSFILE`` only
                    - base from ``FCSFILE``
                  * - ``FCSFILE`` + ``FC2FILE``
                    - base from ``FC2FILE`` (FC3/FC4 from ``FCSFILE``)
                  * - either of the above + ``DFC2FILE`` + ``FC2_TEMPERATURE``
                    - same base **+** :math:`\Delta\Phi(T)` from ``DFC2FILE``
                  * - ``FC2FILE`` = ``PREFIX``.scph.h5 + ``FC2_TEMPERATURE``, no ``DFC2FILE``
                    - total (base + :math:`\Delta\Phi(T)`) read directly from the state file

               Note that the crystal structure is still taken from
               ``FCSFILE``/``FC2FILE``; ``DFC2FILE`` contributes force-constant
               corrections only. Giving a state file as the harmonic FC2 source
               *together with* ``DFC2FILE`` uses only its coarse-mesh-folded base
               FC2 and prints a warning, since that is rarely intended.

````

.. _anphon_allow_unconverged:

* ALLOW_UNCONVERGED-tag = 0 | 1

 :Default: 0
 :Type: Integer
 :Description: SCPH/QHA state files record, per temperature, whether the SCPH
               iteration and the structural optimization converged
               (``/convergence`` in ``PREFIX``.scph.h5 / ``PREFIX``.qha.h5).
               By default, later calculations that consume the renormalized
               IFCs or structure — ``FC2_TEMPERATURE`` reads for kappa/DOS/band
               runs, and ``RESTART_SCPH`` / ``RESTART_QHA`` (whose postprocess
               produces DOS, bands, and thermodynamic functions) — refuse
               unconverged temperatures with an error. Set
               ``ALLOW_UNCONVERGED = 1`` to use such data anyway (a warning is
               printed).

````

"&scph"-field (Read only when ``MODE = SCPH``)
++++++++++++++++++++++++++++++++++++++++++++++

.. _anphon_kmesh_interpolate:

* KMESH_INTERPOLATE-tag = k1, k2, k3

 :Default: None
 :Type: Array of integers
 :Description: In the iteration process of the SCPH equation, the interpolation is done using the 
               :math:`k` mesh defined by ``KMESH_INTERPOLATE``. 

````

.. _anphon_kmesh_scph:

* KMESH_SCPH-tag = k1, k2, k3

 :Default: None
 :Type: Array of integers
 :Description: This :math:`k` mesh is used for the inner loop of the SCPH equation. 
               Each value of ``KMESH_SCPH`` must be equal to or a multiple of the number of ``KMESH_INTERPOLATE`` in the same direction.

````

.. _anphon_self_offdiag:

* SELF_OFFDIAG-tag = 0 | 1

 === ================================================================================
  0   Neglect the off-diagonal elements of the loop diagram in the SCPH calculation
  1   Consider the off-diagonal elements of the loop diagram in the SCPH calculation
 === ================================================================================

 :Default: 1
 :Type: Integer
 :Description: ``SELF_OFFDIAG = 1`` is more accurate, but expensive.

````

.. _anphon_tol_scph:

* TOL_SCPH-tag: Stopping criterion of the SCPH iteration

 :Default: 1.0e-10
 :Type: Double
 :Description: The SCPH iteration stops when both :math:`[\frac{1}{N_{q}}\sum_{q} (\Omega_{q}^{(i)}-\Omega_{q}^{(i-1)})^{2}]^{1/2}` < ``TOL_SCPH`` and :math:`(\Omega_{q}^{(i)})^{2} \geq 0 \; (\forall q)` are satisfied. Here, :math:`\Omega_{q}^{(i)}` is the anharmonic phonon frequency in the :math:`i`\ th iteration and :math:`q` is the phonon modes at the irreducible momentum grid of ``KMESH_INTERPOLATE``.

````

.. _anphon_mixalpha:

* MIXALPHA-tag: Mixing parameter used in the SCPH iteration

 :Default: 0.1
 :Type: Double

````

.. _anphon_maxiter:

* MAXITER-tag: Maximum number of the SCPH iteration

 :Default: 1000
 :Type: Integer

````

.. _anphon_lower_temp:

* LOWER_TEMP-tag = 0 | 1

 === ===============================================================================
  0   The SCPH iteration starts from ``TMIN`` and proceeds to ``TMAX``. (Raise the temperature)
  1   The SCPH iteration starts from ``TMAX`` and proceeds to ``TMIN``. (Lower the temperature)
 === ===============================================================================

 :Default: 1
 :Type: Integer

````

.. _anphon_warmstart:

* WARMSTART-tag = 0 | 1

 === ===============================================================================
  0   SCPH iteration is initialized by harmonic frequencies and eigenvectors
  1   SCPH iteration is initialized by the solution of the previous temperature
 === ===============================================================================

 :Default: 1
 :Type: Integer
 :Description: ``WARMSTART = 1`` usually improves the convergence.

````

.. _anphon_ialgo:

* IALGO-tag = 0 | 1

 === ===============================================================================
  0   MPI parallelization for the :math:`k` point
  1   MPI parallelization for the phonon branch
 === ===============================================================================

 :Default: 0
 :Type: Integer
 :Description: Use ``IALGO = 1`` when the primitive cell contains many atoms and the number of :math:`k` points is small.

````

.. _anphon_restart_scph:

* RESTART_SCPH-tag = 0 | 1

 === ==============================================================
  0   Perform a SCPH calculation from scratch
  1   Skip a SCPH iteration by loading a precalculated result
 === ==============================================================

 :Default: 1 if the file ``PREFIX.scph_dymat`` exists in the working directory; 0 otherwise
 :Type: Integer


````

.. _anphon_bubble:

* BUBBLE-tag = 0 | 1

 === ==============================================================
  0   No bubble correction to the dynamical matrix
  1   Calculate bubble correction on top of the SCPH dynamical matrix
 === ==============================================================

 :Default: 0
 :Type: Integer


````

.. _anphon_relax_str:

* RELAX_STR-tag = 0 | 1 | 2 | 3

 === ==============================================================
  0   Don't relax the crystal structure (not supported when ``MODE = QHA``).
  1   Relax atomic positions.
  2   Relax both atomic positions and the shape of the unit cell.
  3   Lowest-order perturbation theory (not supported when ``MODE = SCPH``).
 === ==============================================================

 :Default: 0
 :Type: Integer

````

"&qha"-field (Read only when ``MODE = QHA``)
++++++++++++++++++++++++++++++++++++++++++++++

.. _anphon_qha_kmesh_interpolate:

* KMESH_INTERPOLATE-tag = k1, k2, k3

 :Default: None
 :Type: Array of integers
 :Description: In the structural optimization based on quasiharmonic approximation (QHA), 
               the interpolation is done using the 
               :math:`k` mesh defined by ``KMESH_INTERPOLATE``. 

````

.. _anphon_qha_kmesh_qha:

* KMESH_QHA-tag = k1, k2, k3

 :Default: None
 :Type: Array of integers
 :Description: This :math:`k` mesh is used for the QHA-based structural optimization. 
               Each value of ``KMESH_QHA`` must be equal to or a multiple of the number of ``KMESH_INTERPOLATE`` in the same direction.

````

.. _anphon_qha_relax_str:

* RELAX_STR-tag = 1 | 2 | 3

 === ==============================================================
  1   Relax atomic positions.
  2   Relax both atomic positions and the shape of the unit cell.
  3   Lowest-order perturbation theory (not supported when ``MODE = SCPH``).
 === ==============================================================

 :Default: 1
 :Type: Integer
 :Description: ``RELAX_STR = 0`` is not supported when ``MODE = QHA``.

````

.. _anphon_qha_lower_temp:

* LOWER_TEMP-tag = 0 | 1

 === ===============================================================================
  0   The structural optimization starts from ``TMIN`` and proceeds to ``TMAX``. (Raise the temperature)
  1   The structural optimization starts from ``TMAX`` and proceeds to ``TMIN``. (Lower the temperature)
 === ===============================================================================

 :Default: 1
 :Type: Integer

````

.. _anphon_qha_scheme:

* QHA_SCHEME-tag = 0 | 1 | 2

 === ==============================================================
  0   Full optimization within QHA.
  1   zero-static internal stress approximation (ZSISA).
  2   volumetric ZSISA (v-ZSISA).
 === ==============================================================

 :Default: 0
 :Type: Integer

 :Description: This option is used only when ``mode = QHA`` and ``RELAX_STR = 2``.

````



"&relax"-field (Read only when ``RELAX_STR != 0``)
++++++++++++++++++++++++++++++++++++++++++++++++++

.. _anphon_relax_algo:

* RELAX_ALGO-tag = 1 | 2 | 3

 === ==============================================================
  1   Steepest descent (not recommended)
  2   Newton-like method
  3   BFGS + GDIIS
 === ==============================================================

 :Default: 2
 :Type: Integer

 :Description: Algorithm to update the crystal structure in structural optimization. 
               This option is used only when ``RELAX_STR = 1, 2``.
               ``RELAX_ALGO = 1`` works properly only when the unit cell is fixed (``RELAX_STR = 1``).

````

.. _anphon_alpha_steepest_decent:

* ALPHA_STEEPEST_DECENT-tag: Coefficient of steepest descent in structural optimization

 :Default: 1.0e4
 :Type: Double

 :Description: :math:`\alpha` coefficient in structural optimization with the steepest-descent algorithm.
               The unit is [:math:`m_e a_B^2/(2\text{Ry})`]. 
               This option is used only when ``RELAX_ALGO = 1``.

````

.. _anphon_max_str_iter:

* MAX_STR_ITER-tag: Maximum number of structure updates.

 :Default: 100
 :Type: Integer

 :Description: This option is used only when ``RELAX_STR = 1, 2``.

````

.. _anphon_add_hess_diag:

* ADD_HESS_DIAG-tag: Correction to the estimated Hessian of free energy in units of kayser (cm\ :sup:`-1`)

 :Default: 100.0
 :Type: Double

 :Description: The squared ``ADD_HESS_DIAG`` is added to the diagonal components of estimated Hessians, 
               which is used to update crystal structures in structural optimization.
               ``ADD_HESS_DIAG`` makes the calculation more robust in the presence of soft modes near the structural phase transition, but setting large values will make the convergence slower.
               This option is used only when ``RELAX_ALGO = 2``.

````

.. _anphon_coord_conv_tol:

* COORD_CONV_TOL-tag: Threshold of convergence for atomic positions in structural optimization.

 :Default: 1.0e-5
 :Type: Double

 :Description: The value is interpreted in units of Bohr.
               This option is used only when ``RELAX_STR = 1, 2``.

````

.. _anphon_mixbeta_coord:

* MIXBETA_COORD-tag: Mixing coefficient for atomic positions in structure updates.

 :Default: 0.5
 :Type: Double

 :Description: This option is used only when ``RELAX_STR = 1, 2``.

````

.. _anphon_cell_conv_tol:

* CELL_CONV_TOL-tag: Threshold of convergence for displacement gradient tensor :math:`u_{\mu \nu}` in structural optimization.

 :Default: 1.0e-5
 :Type: Double

 :Description: This option is used only when ``RELAX_STR = 2``.

````

.. _anphon_mixbeta_cell:

* MIXBETA_CELL-tag: Mixing coefficient for displacement gradient tensor :math:`u_{\mu \nu}` in structure updates.

 :Default: 0.5
 :Type: Double

 :Description: This option is used only when ``RELAX_STR = 2``.

````

.. _anphon_set_init_str:

* SET_INIT_STR-tag = 1 | 2 | 3

 === ==============================================================
  1   Set initial structure from the input file at each temperature.
  2   Start from the crystal structure of the previous temperature.
  3   Start from the crystal structure of the previous temperature in low-symmetry phase.
 === ==============================================================

 :Default: 1
 :Type: Integer

 :Description: This option specifies how to set the initial structure of structural optimization at different temperatures.
               This option is used when ``RELAX_STR = 1, 2``.
               In all options, the initial structure at the initial temperature is set from the input file.
               The initial structure of the input file is read from the ``&strain`` and ``&displace`` field.
               When ``SET_INIT_STR = 3``, the initial displacement from the input file is used if the crystal structure converges to the high-symmetry phase at the previous temperature. The criterion to distinguish low-symmetry and high-symmetry phases is explained in :ref:`COOLING_U0_THR <anphon_cooling_u0_thr>`.

````

.. _anphon_cooling_u0_index:

* COOLING_U0_INDEX-tag = 0 | 1 | ... | 3N-1 (N : the number of atoms in the unit cell)

 :Default: 0
 :Type: Integer

 :Description: Specify as :math:`3\times\alpha + \mu`. Here, :math:`\alpha` denotes the atom index in the primitive cell and :math:`\mu` is the xyz index, where both indices are zero-indexed.
  See the description of :ref:`COOLING_U0_THR <anphon_cooling_u0_thr>` for details.
  This option is used only when ``SET_INIT_STR = 3``.

````

.. _anphon_cooling_u0_thr:

* COOLING_U0_THR-tag: Threshold to judge high-symmetry phase in structural optimization [Bohr].

 :Default: 0.001
 :Type: Double

 :Description: The crystal structure is judged to be back to the high-symmetry phase if 
               :math:`u^{(0)}` [``COOLING_U0_INDEX``] < ``COOLING_U0_THR``. 
               This option is useful in cooling calculations because small displacements from the high-symmetry structure are required to induce spontaneous symmetry breaking.
               This option is used only when ``SET_INIT_STR = 3``.
 
````

.. _anphon_stat_pressure:

* STAT_PRESSURE-tag: Hydrostatic pressure in GPa.

 :Default: 0.0
 :Type: Double

````

.. _anphon_renorm_2to1st:

* RENORM_2TO1ST-tag = 0 | 1 | 2

 === ==============================================================
  0   Set zero.
  1   Real-space IFC renormalization. (not recommended)
  2   Finite difference method with respect to strain.
 === ==============================================================

 :Default: 2
 :Type: Integer

 :Description: This option specifies the method to calculate first-order derivatives of first-order IFCs with respect to strain
 
  :math:`\frac{\partial \Phi_{\mu}(0\alpha)}{\partial u_{\mu_1 \nu_1} }`.

  This option is used only when ``RELAX_STR = 2, 3``.
  Note that ``RENORM_2TO1ST = 1`` requires rotational invariance on IFCs, which is not checked in the program ANPHON.
  ``RENORM_2TO1ST = 0`` can be used for high-symmetry materials in which strain-force coupling is zero, which the user needs to confirm.

````

.. _anphon_renorm_34to1st:

* RENORM_34TO1ST-tag = 0 | 1 

 === ==============================================================
  0   Set zero.
  1   Real-space IFC renormalization.
 === ==============================================================

 :Default: 0
 :Type: Integer

 :Description: This option specifies the method to calculate second and higher-order derivatives of first-order IFCs with respect to strain. 

  :math:`\frac{\partial^2 \Phi_{\mu}(0\alpha)}{\partial u_{\mu_1 \nu_1} \partial u_{\mu_2 \nu_2}}`,
  :math:`\frac{\partial^3 \Phi_{\mu}(0\alpha)}{\partial u_{\mu_1 \nu_1} \partial u_{\mu_2 \nu_2} \partial u_{\mu_3 \nu_3}}`  

  This option is used only when ``RELAX_STR = 2, 3``.
  Note that ``RENORM_34TO1ST = 1`` requires rotational invariance on IFCs, which the user needs to confirm.

````

.. _anphon_renorm_3to2nd:

* RENORM_3TO2ND-tag = 1 | 2 | 3

 === ==============================================================
  1   Real-space IFC renormalization.
  2   Finite difference method (Read input from all six strain patterns).
  3   Finite difference method (Read input from specified strain patterns).
 === ==============================================================

 :Default: 2
 :Type: Integer

 :Description: This option specifies the method to calculate first-order derivatives of harmonic IFCs with respect to strain.
 
  :math:`\frac{\partial \Phi_{\mu_1 \mu_2}(0\alpha_1, R \alpha_2)}{\partial u_{\mu \nu}}`

  This option is used only when ``RELAX_STR = 2, 3``.
  To use ``RENORM_3TO2ND = 3``, the entries of the rotation matrices of ALL the space-group operations must be either 0 or :math:`\pm` 1 in Cartesian representation.

````

.. _anphon_strain_ifc_dir:

* STRAIN_IFC_DIR-tag: Directory name of the inputs of strain-IFC couplings.

 :Default: None
 :Type: String

 :Description: When ``RENORM_2TO1ST = 2`` or ``RENORM_3TO2ND = 3``,
   the input files of the strain-IFC couplings must be given properly in this directory.


````

"&cell"-field
+++++++++++++

Please specify the cell parameters of the *primitive cell* as::

 &cell
  a
  a11 a12 a13
  a21 a22 a23
  a31 a32 a33
 /

The cell parameters are then given by :math:`\vec{a}_{1} = a \times (a_{11}, a_{12}, a_{13})`,
:math:`\vec{a}_{2} = a \times (a_{21}, a_{22}, a_{23})`, and :math:`\vec{a}_{3} = a \times (a_{31}, a_{32}, a_{33})`.

.. Note::

 The lattice constant :math:`a` must be consistent with the value used for the program *alm*.
 For example, if one used :math:`a = 20.4 a_{0}` for a 2x2x2 supercell of Si, one should use :math:`a = 10.2 a_{0}`
 here for the primitive cell.

````

"&kpoint"-field
+++++++++++++++

This entry field is used to specify the list of :math:`k` points to be calculated. 
The first entry **KPMODE** specifies the types of calculation which is followed by detailed entries.

* **KPMODE = 0** : Calculate phonon frequencies at given :math:`k` points

 For example, if one wants to calculate phonon frequencies at Gamma (0, 0, 0) and X (0, 1/2, 1/2) of an FCC crystal, 
 the ``&kpoint`` entry should be written as
 ::

  &kpoint
   0
   0.000 0.000 0.000
   0.000 0.500 0.500
  /

* **KPMODE = 1** : Band dispersion calculation

 For example, if one wants to calculate phonon dispersion relations along G\-K\-X\-G\-L of a FCC crystal, 
 the ``&kpoint`` entry should be written as follows::

  &kpoint
   1
   G 0.000 0.000 0.000  K 0.375 0.375 0.750 51
   K 0.375 0.375 0.750  X 0.500 0.500 1.000 51
   X 0.000 0.500 0.500  G 0.000 0.000 0.000 51
   G 0.000 0.000 0.000  L 0.500 0.500 0.500 51
  /

 The 1st and 5th columns specify the character of Brillouin zone edges, 
 which are followed by fractional coordinates of each point. 
 The last column indicates the number of sampling points. 

* **KPMODE = 2** : Uniform :math:`k` grid for phonon DOS and thermal conductivity

 In order to perform a calculation with 20x20x20 :math:`k` grid, the entry should be 
 ::

  &kpoint
   2
   20 20 20
  /

````

"&strain"-field (Read only when ``RELAX_STR = 2``)
++++++++++++++++++++++++++++++++++++++++++++++++++

Please specify the initial displacement gradient tensor :math:`u_{\mu \nu}` for structural optimization as ::

 &strain
 u_xx u_xy u_xz
 u_yx u_yy u_yz
 u_zx u_zy u_zz
 /

Note that the user needs to give a symmetric matrix.

"&displace"-field (Read only when ``RELAX_STR = 1, 2``)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Please specify the initial atomic displacements :math:`u^{(0)}_{\alpha \mu}` [Bohr].

* **DISPMODE = 0** : Fractional coordinate representation

 The ``&displace`` entry should be written as follows.
 The first four lines after DISPMODE (= 0) specifies the unit cell, whose format is the same as the ``&cell`` field.
 Note that the unit cell in the ``&displace`` field is used only for transforming the input to the real space representation. Thus, the unit cell here does not need to be commensurate with the primitive cell or some supercells.
 
 u_ij is the j-th component of the displacement of i-th atom in the primitive cell in fractional coordinate representation.
 ::

  &displace
   0
   a
   a11 a12 a13
   a21 a22 a23
   a31 a32 a33
   u_01, u_02, u_03
   ...
  /

* **DISPMODE = 1** : Cartesian coordinate representation

 Each line after DISPMODE (= 1) specifies the initial atomic displacement in Cartesian representation. 
 u_ij is the j component of the displacement of i-th atom in the primitive cell.
 ::

  &displace
   1
   u_0x, u_0y, u_0z
   ...
  /


"&analysis"-field
+++++++++++++++++

.. _anphon_gruneisen:

* GRUNEISEN-tag = 0 | 1

 === ===================================================================
  0   Gr\ |umulaut_u|\ neisen parameters will not be calculated
  1   Gr\ |umulaut_u|\ neisen parameters will be stored
 === ===================================================================

 :Default: 0
 :Type: Integer
 :Description:  When ``MODE = phonons`` and ``GRUNEISEN = 1``, Gr\ |umulaut_u|\ neisen parameters will be stored in ``PREFIX``.gruneisen (*KPMODE* = 1) or ``PREFIX``.gru_all (*KPMODE* = 2).

.. Note::

 To compute Gr\ |umulaut_u|\ neisen parameters, cubic force constants must be contained in the ``FCSFILE`` file.


````

.. _anphon_printevec:

* PRINTEVEC-tag = 0 | 1

 === ===================================================================
  0   Do not print phonon eigenvectors
  1   Print phonon eigenvectors in the ``PREFIX``.evec file
 === ===================================================================

 :Default: 0
 :Type: Integer

````

.. _anphon_printxsf:

* PRINTXSF-tag = 0 | 1

 === ===================================================================
  0   Do not save an AXSF file
  1   Create an AXSF file ``PREFIX``.axsf
 === ===================================================================

 :Default: 0
 :Type: Integer
 :Description: This is to visualize the direction of vibrational modes at gamma (0, 0, 0) by XCrySDen. 
               This option is valid only when ``MODE = phonons``.

````

.. _anphon_printvel:

* PRINTVEL-tag = 0 | 1

 === ===================================================================
  0   Do not print group velocity
  1   Store phonon velocities to a file
 === ===================================================================

 :Default: 0
 :Type: Integer
 :Description: When ``MODE = phonons`` and ``PRINTVEL = 1``, group velocities of phonons will be stored in ``PREFIX``.phvel (*KPMODE* = 1) or ``PREFIX``.phvel_all (*KPMODE* = 2).

````

.. _anphon_printmsd:

* PRINTMSD-tag = 0 | 1

 === ===================================================================
  0   Do not print mean-square-displacement (MSD) of atoms
  1   Save MSD of atoms to the file ``PREFIX``.msd
 === ===================================================================
 
 :Default: 0
 :Type: Integer
 :Description: This flag is available only when ``MODE = phonons`` and *KPMODE* = 2.

````

.. _anphon_pdos:

* PDOS-tag = 0 | 1

 === ===================================================================
  0   Only the total DOS will be printed in ``PREFIX``.dos
  1   Atom-projected phonon DOS will be stored in ``PREFIX``.dos
 === ===================================================================

 :Default: 0
 :Type: Integer
 :Description: This flag is available only when ``MODE = phonons`` and *KPMODE* = 2.

````

.. _anphon_tdos:

* TDOS-tag = 0 | 1

 === ===================================================================
  0   Do not compute two-phonon DOS
  1   Two-phonon DOSs will be stored in ``PREFIX``.tdos
 === ===================================================================
 
 :Default: 0
 :Type: Integer
 :Description: This flag is available only when ``MODE = phonons`` and *KPMODE* = 2.

 .. Note::

  Calculation of two-phonon DOS is computationally expensive.

````

.. _anphon_sps:

* SPS-tag = 0 | 1 | 2

 === ====================================================================================
  0   Do not compute scattering phase space
  1   | Total and mode-decomposed scattering phase space involving 
      | the three-phonon processes will be stored in ``PREFIX``.sps
  2   Three-phonon scattering phase space with the Bose factor will be stored 
      in ``PREFIX``.sps_Bose
 === ====================================================================================
 
 :Default: 0
 :Type: Integer
 :Description: This flag is available only when ``MODE = phonons`` and *KPMODE* = 2.


````

.. _anphon_printpr:

* PRINTPR-tag = 0 | 1

 === ====================================================================================
  0   Do not compute the (atomic) participation ratio
  1   | Compute participation ratio and atomic participation ratio, which will be 
      | stored in  ``PREFIX``.pr and ``PREFIX``.apr respectively.
 === ====================================================================================
 
 :Default: 0
 :Type: Integer
 :Description: This flag is available when ``MODE = phonons``.


````

.. _anphon_ucorr:

* UCORR-tag = 0 | 1

 === =========================================================================
  0   Do nothing
  1   | Compute the displacement-displacement correlation function.
      | The result is stored in ``PREFIX``.ucorr
 === =========================================================================
 
 :Default: 0
 :Type: Integer
 :Description: The displacement-displacement correlation function involves two atoms. The first atom is located in the primitive cell at the center (shift1=[0,0,0]) and the second atom is located in the :math:`\ell'`\  th cell. The translation vector to the :math:`\ell'`\  th cell can be specified by the ``SHIFT_UCORR`` tag. This tag is effective only when ``MODE = phonons`` and *KPMODE* = 2


````

.. _anphon_shift_ucorr:

* SHIFT_UCORR-tag = l1, l2, l3

 :Default: [0, 0, 0]
 :Type: Array of integers
 :Description: This tag specifies the translation vector used for computing the displacement-displacement (uu) correlation function. For example, if one wants to compute the uu correlation function between an atom 1 in the cell at the center and atom 2 in the neighboring cell at :math:`\boldsymbol{r}(\ell')=(1,0,0)`, ``SHIFT_UCORR`` should be set as ``SHIFT_UCORR = 1 0 0``.

````

.. _anphon_zmode:

* ZMODE-tag = 0 | 1

 === =========================================================================
  0   Do nothing
  1   | Compute the mode effective charges of the zone-center phonons. 
      | The result is stored in ``PREFIX``.zmode
 === =========================================================================
 
 :Default: 0
 :Type: Integer
 :Description: When ``MODE = phonons`` and ``ZMODE = 1``, the mode effective charges are computed for the phonon modes at the Gamma point and saved in ``PREFIX``.zmode. The unit of the mode effective charge is :math:`e \; \text{amu}^{-1/2}`.


````

.. .. _anphon_fe_bubble:

.. * FE_BUBBLE-tag = 0 | 1

..  === ====================================================================================
..   0   Do not compute the vibrational free-energy associated with the bubble diagram
..   1   | Compute the vibrational free-energy associated with the bubble diagram and 
..       | save it in ``PREFIX``.thermo (when ``MODE = phonons``) or ``PREFIX``.scph_thermo (when ``MODE = SCPH``).
..  === ====================================================================================
 
..  :Default: 0
..  :Type: Integer
..  :Description: This tag is used when *KPMODE* = 2.


.. ````

.. _anphon_anime:

* ANIME-tag = k1, k2, k3

 :Default: None
 :Type: Array of doubles
 :Description: This tag is to animate vibrational mode. k1, k2, and k3 specify the momentum of phonon modes to animate,
               which should be given in units of the reciprocal lattice vector. For example, ``ANIME = 0.0 0.0 0.5`` will 
               animate phonon modes at (0, 0, 1/2). When ``ANIME`` is given, ``ANIME_CELLSIZE`` is also necessary.
               You can choose the format of animation files, either AXSF or XYZ, by ``ANIME_FORMAT`` tag.


````

.. _anphon_anime_frames:

* ANIME_FRAMES-tag: The number of frames saved in animation files

 :Default: 20
 :Type: Integer

````

.. _anphon_anime_cellsize:

* ANIME_CELLSIZE-tag = L1, L2, L3

 :Default: None
 :Type: Array of integers
 :Description: This tag specifies the cell size for animation. L1, L2, and L3 should be large enough to be 
               commensurate with the reciprocal point given by the ``ANIME`` tag.

````

.. _anphon_anime_format:

* ANIME_FORMAT = xsf | xyz

 :Default: xyz
 :Type: String
 :Description: When ``ANIME_FORMAT = xsf``, ``PREFIX``.anime???.axsf files are created for XcrySDen.
               When ``ANIME_FORMAT = xyz``, ``PREFIX``.anime???.xyz files are created for VMD (and any other supporting software such as Jmol).


````

"&kappa"-field (Read only when ``MODE = RTA``)
++++++++++++++++++++++++++++++++++++++++++++++

.. _anphon_include_4ph:

* INCLUDE_4PH-tag = 0 | 1

 === ====================================================================================
  0   Compute three-phonon scattering rates only
  1   Additionally compute four-phonon scattering rates
 === ====================================================================================

 :Default: 0
 :Type: Integer
 :Description: The explicit switch for the four-phonon channel. It requires quartic
               force constants (``FCSFILE`` containing Order4, or ``FC4FILE``) and
               activates the FC4 machinery automatically; the related settings are
               ``KMESH_COARSE``, ``ISMEAR_4PH``, ``EPSILON_4PH``, and
               ``INTERPOLATOR``. For backward compatibility, ``QUARTIC > 0`` (an
               ``&analysis`` tag) still implies ``INCLUDE_4PH = 1`` when the tag is
               absent, with a deprecation warning.

````

.. _anphon_restart:

* RESTART-tag : Flag to restart the calculation when ``MODE = RTA``

 === =======================================================
  0   Calculate from scratch
  1   Restart from the existing file
 === =======================================================

 :Default: 1 if there is a file named ``PREFIX``.kappa.h5 or ``PREFIX``.result; 0 otherwise
 :Type: Integer
 :Description: With the default ``FILE_FORMAT = h5``, the restart state lives in
               ``PREFIX``.kappa.h5. A legacy text ``PREFIX``.result file (from an
               older run or ``FILE_FORMAT = text``) is imported into the HDF5 file
               once, read-only, and left untouched. This tag is also accepted in the
               ``&general`` field for backward compatibility (deprecated); the
               ``&kappa`` value wins when both are given.

````

.. _anphon_restart_4ph:

* RESTART_4PH-tag : Flag to restart the four-phonon part of the calculation when ``INCLUDE_4PH = 1``

 === =======================================================
  0   Calculate the four-phonon scattering rates from scratch
  1   Restart from the existing file
 === =======================================================

 :Default: 1 if there is a file named ``PREFIX``.kappa.h5 or ``PREFIX``.4ph.result; 0 otherwise
 :Type: Integer
 :Description: Controls the four-phonon channel independently of :ref:`RESTART <anphon_restart>`
               (the two channels are stored side by side in ``PREFIX``.kappa.h5 but restart
               separately). ``RESTART_4PH = 0`` discards only the previously computed
               four-phonon scattering rates; the three-phonon data are kept. Also accepted
               in ``&general`` for backward compatibility (deprecated).

````

.. _anphon_kappa_coherent:

* KAPPA_COHERENT-tag = 0 | 1 | 2

 === ====================================================================================
  0    Do not compute the coherent component of thermal conductivity
  1    Compute the coherent component of thermal conductivity and save it in ``PREFIX``.kl_coherent.
  2  | In addition to above (``KAPPA_COHERENT = 1``), all elements of the coherent term
     | are saved in ``PREFIX``.kc_elem.
 === ====================================================================================
 
 :Default: 0
 :Type: Integer
 :Description: This flag is available when ``MODE = RTA``. For the theoretical details, please see :ref:`this page <kappa_coherent>`.

 .. caution::

     Still experimental. Please check the validity of results carefully.


````

.. _anphon_kappa_spec:

* KAPPA_SPEC-tag = 0 | 1

 === ====================================================================================
  0   Do not compute the thermal conductivity spectra
  1   Compute the thermal conductivity spectra, which will be 
      stored in  ``PREFIX``.kl_spec
 === ====================================================================================
 
 :Default: 0
 :Type: Integer
 :Description: This flag is available when ``MODE = RTA``.


````

.. _anphon_isotope:

* ISOTOPE-tag = 0 | 1 | 2

 === =========================================================================
  0   Do not consider phonon-isotope scatterings
  1   Consider phonon-isotope scatterings
  2   | Consider phonon-isotope scatterings as in ``ISOTOPE = 1`` and
      | the calculated selfenergy is stored in ``PREFIX``.self_isotope
 === =========================================================================

 :Default: 1
 :Type: Integer
 :Description: When ``MODE = RTA`` and ``ISOTOPE = 1 or 2``, phonon scatterings due to isotopes will be considered perturbatively.
               The computation is negligibly cheap, so it is enabled by default with ``ISOFACT``
               taken from the internal natural-abundance database; set ``ISOTOPE = 0`` for
               isotopically pure crystals, or give ``ISOFACT`` explicitly for enriched samples
               (required for elements without a database entry). The per-mode isotope
               linewidths and the factors used are stored in ``PREFIX``.kappa.h5
               (``/scattering/isotope/gamma``, ``/metadata/isotope_factors``).

````

.. _anphon_isofact:

* ISOFACT-tag : Isotope factor for each atomic species (in the order of the ``KD``-tag)

 :Default: Automatically calculated from the ``KD`` tag
 :Type: Array of doubles
 :Description: Isotope factor is a dimensionless value defined by :math:`\sum_{i} f_{i} (1 - m_{i}/\bar{m})^{2}`. 
               Here, :math:`f_{i}` is the fraction of the :math:`i`\ th isotope of an element having mass :math:`m_{i}`, 
               and :math:`\bar{m}=\sum_{i}f_{i}m_{i}` is the average mass, respectively. 
               This quantity is equivalent to :math:`g_{2}` appearing in the original paper by S. Tamura [Phys. Rev. B, 27, 858.].


````

.. _label_format_BORNINFO:

Format of BORNINFO
~~~~~~~~~~~~~~~~~~

When one wants to consider the LO-TO splitting near the :math:`\Gamma` point, it is necessary to set ``NONANALYTIC > 0`` and
provide ``BORNINFO`` file containing dielectric tensor :math:`\epsilon^{\infty}` and Born effective charge :math:`Z^{*}`.
In ``BORNINFO`` file, the dielectric tensor should be written in first 3 lines which are followed by Born effective charge tensors
for each atom as the following.

.. math::
   :nowrap:

   \begin{eqnarray*}
    \epsilon_{xx}^{\infty} & \epsilon_{xy}^{\infty} & \epsilon_{xz}^{\infty} \\
    \epsilon_{yx}^{\infty} & \epsilon_{yy}^{\infty} & \epsilon_{yz}^{\infty} \\
    \epsilon_{zx}^{\infty} & \epsilon_{zy}^{\infty} & \epsilon_{zz}^{\infty} \\
    Z_{1,xx}^{*} & Z_{1,xy}^{*} & Z_{1,xz}^{*} \\
    Z_{1,yx}^{*} & Z_{1,yy}^{*} & Z_{1,zz}^{*} \\
    Z_{1,zx}^{*} & Z_{1,zy}^{*} & Z_{1,zz}^{*} \\
    & \vdots & \\
    Z_{N_p,xx}^{*} & Z_{N_p,xy}^{*} & Z_{N_p,xz}^{*} \\
    Z_{N_p,yx}^{*} & Z_{N_p,yy}^{*} & Z_{N_p,zz}^{*} \\
    Z_{N_p,zx}^{*} & Z_{N_p,zy}^{*} & Z_{N_p,zz}^{*} \\
   \end{eqnarray*} 

Here, :math:`N_p` is the number of atoms contained in the *primitive cell*.

.. Attention::

 Please pay attention to the order of Born effective charges.	
