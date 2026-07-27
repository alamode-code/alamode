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
   :ref:`CLASSICAL <anphon_classical>`, :ref:`DELTA_E <anphon_emin>`, :ref:`DFC2FILE <anphon_dfc2file>`, :ref:`DT <anphon_tmin>`
   :ref:`EMAX <anphon_emin>`, :ref:`EMIN <anphon_emin>`, :ref:`EPSILON <anphon_epsilon>`, :ref:`FC2FILE <anphon_fc2file>`
   :ref:`FC2_TEMPERATURE <anphon_fc2_temperature>`, :ref:`FC3FILE <anphon_fc2file>`, :ref:`FC4FILE <anphon_fc2file>`, :ref:`FCSFILE <anphon_fcsfile>`
   :ref:`FILE_FORMAT <anphon_file_format>`, :ref:`ISMEAR <anphon_ismear>`, :ref:`KD <anphon_kd>`, :ref:`MASS <anphon_mass>`
   :ref:`MODE <anphon_mode>`, :ref:`NA_SIGMA <anphon_na_sigma>`, :ref:`NBANDS <anphon_nbands>`, :ref:`NONANALYTIC <anphon_nonanalytic>`
   :ref:`PREC_EWALD <anphon_prec_ewald>`, :ref:`PREFIX <anphon_prefix>`, :ref:`PRINTSYM <anphon_printsym>`, :ref:`TMAX <anphon_tmin>`
   :ref:`TMIN <anphon_tmin>`, :ref:`TOLERANCE <anphon_tolerance>`, :ref:`TRISYM <anphon_trisym>`, :ref:`VERBOSITY <anphon_verbosity>`
   **&scph**
   :ref:`BUBBLE <anphon_bubble>`, :ref:`IALGO <anphon_ialgo>`, :ref:`IMIX <anphon_imix>`, :ref:`KMESH_INTERPOLATE <anphon_kmesh_interpolate>`
   :ref:`KMESH_SCPH <anphon_kmesh_scph>`, :ref:`LOWER_TEMP <anphon_lower_temp>`, :ref:`MAXITER <anphon_maxiter>`, :ref:`MIXALPHA <anphon_mixalpha>`
   :ref:`RELAX_STR <anphon_relax_str>`, :ref:`RESTART_SCPH <anphon_restart_scph>`, :ref:`SELF_OFFDIAG <anphon_self_offdiag>`, :ref:`TOL_SCPH <anphon_tol_scph>`
   :ref:`WARMSTART <anphon_warmstart>`
   **&qha**
   :ref:`IALGO <anphon_ialgo>`, :ref:`KMESH_INTERPOLATE <anphon_qha_kmesh_interpolate>`, :ref:`KMESH_QHA <anphon_qha_kmesh_qha>`, :ref:`LOWER_TEMP <anphon_qha_lower_temp>`
   :ref:`QHA_SCHEME <anphon_qha_scheme>`, :ref:`RELAX_STR <anphon_qha_relax_str>`, :ref:`RESTART_QHA <anphon_restart_qha>`, :ref:`SELF_OFFDIAG <anphon_self_offdiag>`
   **&relax**
   :ref:`ADD_HESS_DIAG <anphon_add_hess_diag>`, :ref:`ALPHA_STDECENT <anphon_alpha_stdecent>`, :ref:`CELL_CONV_TOL <anphon_cell_conv_tol>`, :ref:`CELL_GRADIENT_CONV_TOL <anphon_cell_gradient_conv_tol>`
   :ref:`COOLING_U0_INDEX <anphon_cooling_u0_index>`, :ref:`COOLING_U0_THR <anphon_cooling_u0_thr>`, :ref:`COORD_CONV_TOL <anphon_coord_conv_tol>`, :ref:`GDIIS_CONTROL <anphon_gdiis_control>`
   :ref:`GDIIS_PLAIN <anphon_gdiis_plain>`, :ref:`GRADIENT_CONV_TOL <anphon_gradient_conv_tol>`, :ref:`MAX_STR_ITER <anphon_max_str_iter>`, :ref:`MIXBETA_CELL <anphon_mixbeta_cell>`
   :ref:`MIXBETA_COORD <anphon_mixbeta_coord>`, :ref:`RELAX_ALGO <anphon_relax_algo>`, :ref:`RENORM_2TO1ST <anphon_renorm_2to1st>`, :ref:`RENORM_34TO1ST <anphon_renorm_34to1st>`
   :ref:`RENORM_3TO2ND <anphon_renorm_3to2nd>`, :ref:`SET_INIT_STR <anphon_set_init_str>`, :ref:`STAT_PRESSURE <anphon_stat_pressure>`, :ref:`STRAIN_IFC_DIR <anphon_strain_ifc_dir>`
   **&analysis**
   :ref:`ANIME <anphon_anime>`, :ref:`ANIME_CELLSIZE <anphon_anime_cellsize>`, :ref:`ANIME_FORMAT <anphon_anime_format>`, :ref:`ANIME_FRAMES <anphon_anime_frames>`
   :ref:`DIELEC <anphon_dielec>`, :ref:`DOS <anphon_dos>`, :ref:`FC2_EWALD <anphon_fc2_ewald>`, :ref:`GRUNEISEN <anphon_gruneisen>`
   :ref:`KS_INPUT <anphon_ks_input>`, :ref:`PDOS <anphon_pdos>`, :ref:`PRINTEVAL <anphon_printeval>`, :ref:`PRINTEVEC <anphon_printevec>`
   :ref:`PRINTMSD <anphon_printmsd>`, :ref:`PRINTPR <anphon_printpr>`, :ref:`PRINTV3 <anphon_printv3>`, :ref:`PRINTV4 <anphon_printv4>`
   :ref:`PRINTVEL <anphon_printvel>`, :ref:`PRINTXSF <anphon_printxsf>`, :ref:`PROJECTION_AXES <anphon_projection_axes>`, :ref:`QUARTIC <anphon_quartic>`
   :ref:`REALPART <anphon_realpart>`, :ref:`SELF_ENERGY <anphon_self_energy>`, :ref:`SELF_W <anphon_self_w>`, :ref:`SHIFT_UCORR <anphon_shift_ucorr>`
   :ref:`SPS <anphon_sps>`, :ref:`TDOS <anphon_tdos>`, :ref:`UCORR <anphon_ucorr>`, :ref:`ZMODE <anphon_zmode>`
   **&kappa**
   :ref:`ADAPTIVE_FACTOR <anphon_adaptive_factor>`, :ref:`EPSILON_4PH <anphon_epsilon_4ph>`, :ref:`IBTE_MIXING <anphon_ibte_mixing>`, :ref:`INCLUDE_4PH <anphon_include_4ph>`
   :ref:`INTERPOLATOR <anphon_interpolator>`, :ref:`ISMEAR_4PH <anphon_ismear_4ph>`, :ref:`ISOFACT <anphon_isofact>`, :ref:`ISOTOPE <anphon_isotope>`
   :ref:`ISOTOPE_INSCATTERING <anphon_isotope_inscattering>`, :ref:`ITERATIVE <anphon_iterative>`, :ref:`ITER_THRESHOLD <anphon_iter_threshold>`, :ref:`KAPPA_COHERENT <anphon_kappa_coherent>`
   :ref:`KAPPA_SPEC <anphon_kappa_spec>`
   :ref:`KMESH_COARSE <anphon_kmesh_coarse>`, :ref:`LEN_BOUNDARY <anphon_len_boundary>`, :ref:`MAX_CYCLE <anphon_max_cycle>`, :ref:`MIN_CYCLE <anphon_min_cycle>`
   :ref:`RESTART <anphon_restart>`, :ref:`RESTART_4PH <anphon_restart_4ph>`, :ref:`SOLVER <anphon_solver>`, :ref:`WRITE_INTERPOL <anphon_write_interpol>`




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

* **MODE**-tag = phonons | kappa

 ========= ==============================================================
  phonons  | Calculate phonon dispersion relation, phonon DOS, 
           | Gr\ |umulaut_u|\ neisen parameters etc.

  kappa    | Calculate phonon lifetimes and lattice thermal conductivity
           | based on the Boltzmann transport equation (BTE). The solver
           | level (RTA, IBTE, ...) is chosen by the :ref:`SOLVER <anphon_solver>`
           | tag of the &kappa field. ``MODE = RTA`` is a deprecated alias.

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

.. _anphon_prec_ewald:

* PREC_EWALD-tag : Accuracy of the Ewald summation used for ``NONANALYTIC = 3``

 :Default: 1.0e-12
 :Type: Double
 :Description: Read only when ``NONANALYTIC = 3``. The convergence parameter and the real- and reciprocal-space cutoffs of the Ewald summation of the dipole-dipole interaction are chosen automatically so that the neglected terms are of the order of ``PREC_EWALD``. The value must satisfy 0 < ``PREC_EWALD`` < 1.

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
               (DOS and three-phonon scattering rates). With ``ISMEAR = 2``, the
               Gaussian width is chosen automatically for each pair of phonon
               modes from the group-velocity difference and the :math:`k`-mesh
               spacing,

               .. math::
                  \sigma_{jj'}(\boldsymbol{q}) = \alpha \sqrt{\frac{1}{12} \sum_{u=1}^{3}
                  \left[ \left( \boldsymbol{v}_{j} - \boldsymbol{v}_{j'} \right) \cdot
                  \frac{\boldsymbol{G}_{u}}{N_{u}} \right]^{2} },

               following the adaptive-broadening scheme of Yates *et al.*
               [Phys. Rev. B **75**, 195121 (2007)], where
               :math:`\boldsymbol{G}_{u}/N_{u}` are the mesh steps along the three
               reciprocal lattice vectors and the prefactor :math:`\alpha` is set by
               :ref:`ADAPTIVE_FACTOR <anphon_adaptive_factor>`. ``EPSILON`` is not
               used in this case, and no manual width convergence test is needed.

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

.. _anphon_nbands:

* NBANDS-tag : Number of phonon branches to be written in output files

 :Default: :math:`3N`, where :math:`N` is the number of atoms in the primitive cell
 :Type: Integer
 :Description: When ``NBANDS`` is given, only the lowest ``NBANDS`` phonon branches are written in the frequency output files, such as ``PREFIX``.bands and the eigenvalue files created with :ref:`PRINTEVAL <anphon_printeval>`.

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
 :Description: This variable is used only when ``MODE = kappa``.

 .. Note::

  ``TRISYM = 1`` can reduce the computational cost, but phonon linewidth stored to the file
  ``PREFIX``.result needs to be averaged at points of degeneracy.
  For that purpose, a subsidiary program ``analyzer.py`` should be used.

````

.. _anphon_file_format:

* FILE_FORMAT-tag : Format of the restart/state files

 ====== =========================================================================
  h5     Unified crash-safe HDF5 files (``PREFIX``.kappa.h5 for ``MODE = kappa``;
         ``PREFIX``.scph.h5 / ``PREFIX``.qha.h5 for ``MODE = SCPH`` / ``QHA``)
  text   Legacy text files (``PREFIX``.result, ``PREFIX``.scph_dymat, etc.)
 ====== =========================================================================

 :Default: h5
 :Type: String
 :Description: The text format is kept for one transition release. Human-readable
               outputs (``PREFIX``.kl, ``PREFIX``.scph_thermo, ``PREFIX``.scph_dfc2,
               ``PREFIX``.V0, ...) are written in both modes. The
               eigenvalue/eigenvector outputs of :ref:`PRINTEVAL <anphon_printeval>`
               and :ref:`PRINTEVEC <anphon_printevec>` also follow this tag: ``h5``
               writes the schema-stamped \*.eval.hdf5 / \*.evec.hdf5 files, ``text``
               the plain-text variants.

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
               as usual. In ``MODE = kappa``, this tag also switches
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

.. _anphon_verbosity:

* VERBOSITY-tag : Level of the standard output

 :Default: 1
 :Type: Integer
 :Description: ``VERBOSITY = 0`` suppresses the echo of the input variables and the per-iteration log of the SCPH solver. ``VERBOSITY >= 2`` keeps the per-iteration SCPH log even inside the structural optimization loop (``RELAX_STR != 0``), where it is compacted by default.

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

.. _anphon_imix:

* IMIX-tag = 0 | 1

 === ===============================================================================
  0   Simple linear mixing with the mixing parameter ``MIXALPHA``
  1   DIIS (Pulay) mixing
 === ===============================================================================

 :Default: 1
 :Type: Integer
 :Description: Mixing algorithm used in the SCPH iteration. With ``IMIX = 1``, a few simple-mixing steps are performed first and the subsequent iterations are accelerated by the DIIS method; :ref:`MIXALPHA <anphon_mixalpha>` is used as the underlying mixing parameter and must be in (0, 1] in this case. ``IMIX = 0`` restores the plain simple mixing.

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

.. _anphon_restart_qha:

* RESTART_QHA-tag = 0 | 1

 === ==============================================================
  0   Perform a QHA calculation from scratch
  1   Skip the QHA optimization by loading a precalculated result
 === ==============================================================

 :Default: 1 if the file ``PREFIX.qha.h5``, or the pair of the legacy files ``PREFIX.renorm_harm_dymat`` and ``PREFIX.V0``, exists in the working directory; 0 otherwise
 :Type: Integer
 :Description: With the default ``FILE_FORMAT = h5``, the restart data are read preferentially from the unified state file ``PREFIX``.qha.h5. When only the legacy text files are found, they are loaded instead and migrated into ``PREFIX``.qha.h5 once. Temperatures whose structural optimization did not converge are refused unless :ref:`ALLOW_UNCONVERGED <anphon_allow_unconverged>` is set.

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

.. _anphon_alpha_stdecent:
.. _anphon_alpha_steepest_decent:

* ALPHA_STDECENT-tag: Coefficient of steepest descent in structural optimization

 :Default: 1.0e4
 :Type: Double

 :Description: :math:`\alpha` coefficient in structural optimization with the steepest-descent algorithm.
               The unit is [:math:`m_e a_B^2/(2\text{Ry})`].
               This option is used only when ``RELAX_ALGO = 1``.
               (This tag was formerly documented as ``ALPHA_STEEPEST_DECENT``; the tag name accepted by the parser is ``ALPHA_STDECENT``.)

````

.. _anphon_gdiis_plain:

* GDIIS_PLAIN-tag = 0 | 1

 === ==============================================================
  0   Use the controlled GDIIS (with step-acceptance tests)
  1   Use the plain GDIIS (without step-acceptance tests)
 === ==============================================================

 :Default: 0
 :Type: Integer

 :Description: This option is used only when ``RELAX_ALGO = 3``. By default, each GDIIS step is subjected to the step-acceptance criteria of the controlled GDIIS method by Farkas and Schlegel (step-length cap, coefficient/extrapolation cap, and near-singularity rejection with error-vector rescaling), which makes the optimization more robust. Set ``GDIIS_PLAIN = 1`` to switch back to the regular GDIIS update.

````

.. _anphon_gdiis_control:

* GDIIS_CONTROL-tag = 0 | 1

 :Default: 1
 :Type: Integer

 :Description: Deprecated. This tag was formerly used to enable the controlled GDIIS, which is now the default; a warning is printed when the tag is given. Use :ref:`GDIIS_PLAIN <anphon_gdiis_plain>` = 1 to disable the controlled GDIIS instead. When both tags are given, ``GDIIS_PLAIN`` takes precedence.

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

.. _anphon_gradient_conv_tol:

* GRADIENT_CONV_TOL-tag: Threshold of convergence for the residual force in structural optimization.

 :Default: 0.0 (disabled)
 :Type: Double

 :Description: When ``GRADIENT_CONV_TOL > 0``, the structural optimization at each temperature is declared converged only if the norm of the residual force acting on the internal coordinates is smaller than this value, in addition to the step-size criterion :ref:`COORD_CONV_TOL <anphon_coord_conv_tol>`. This guards against false convergence at a non-stationary point, where the step can be small although the gradient is not, which may happen with the GDIIS optimizer (``RELAX_ALGO = 3``).

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

.. _anphon_cell_gradient_conv_tol:

* CELL_GRADIENT_CONV_TOL-tag: Threshold of convergence for the residual cell gradient (stress) in structural optimization.

 :Default: 0.0 (disabled)
 :Type: Double

 :Description: When ``CELL_GRADIENT_CONV_TOL > 0`` and the cell shape is relaxed (``RELAX_STR = 2``), convergence additionally requires the norm of the gradient of the free energy with respect to the strain tensor (including the term due to :ref:`STAT_PRESSURE <anphon_stat_pressure>`) to be smaller than this value. Its units differ from those of :ref:`GRADIENT_CONV_TOL <anphon_gradient_conv_tol>`, hence the separate threshold.

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

.. _anphon_printeval:

* PRINTEVAL-tag = 0 | 1

 === ===================================================================
  0   Do not print phonon frequencies to separate files
  1   Print phonon frequencies (eigenvalues) to files
 === ===================================================================

 :Default: 0
 :Type: Integer
 :Description: When ``PRINTEVAL = 1``, the phonon frequencies are saved in ``PREFIX``.eval (*KPMODE* = 0), ``PREFIX``.band.eval (*KPMODE* = 1), and ``PREFIX``.mesh.eval (*KPMODE* = 2). The file format follows :ref:`FILE_FORMAT <anphon_file_format>`: the default ``h5`` writes the corresponding \*.eval.hdf5 files (schema ``alamode:eigenvalues``) instead of the text files; ``FILE_FORMAT = text`` (or a build without HDF5 support) writes the plain-text files.

````

.. _anphon_printevec:

* PRINTEVEC-tag = 0 | 1

 === ===================================================================
  0   Do not print phonon eigenvectors
  1   Print phonon eigenvectors in the ``PREFIX``.evec file
 === ===================================================================

 :Default: 0
 :Type: Integer
 :Description: The file format follows :ref:`FILE_FORMAT <anphon_file_format>`: the default ``h5`` writes ``PREFIX``\ [.band|.mesh].evec.hdf5 (schema ``alamode:eigenvectors``); ``FILE_FORMAT = text`` (or a build without HDF5 support) writes the plain-text ``PREFIX``\ [.band|.mesh].evec files.

````

.. _anphon_projection_axes:

* PROJECTION_AXES-tag = e1x e1y e1z [, e2x e2y e2z]

 :Default: None
 :Type: Array of doubles (one or two vectors separated by a comma)
 :Description: When given, the eigenvectors of degenerate phonon modes are transformed within each degenerate subspace so that they align with the specified direction(s), which are given in Cartesian coordinates. Up to two axes can be given, separated by a comma; additional entries are ignored with a warning. This option affects the outputs that depend on eigenvectors, such as :ref:`PRINTEVEC <anphon_printevec>` and :ref:`ANIME <anphon_anime>`.

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

.. _anphon_dos:

* DOS-tag = 0 | 1

 === ===================================================================
  0   Do not compute the phonon DOS
  1   The total phonon DOS is computed and stored in ``PREFIX``.dos
 === ===================================================================

 :Default: 1
 :Type: Integer
 :Description: This flag is available only when ``MODE = phonons`` and *KPMODE* = 2. Set ``DOS = 0`` to skip the DOS calculation. See also :ref:`PDOS <anphon_pdos>`.

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

.. _anphon_dielec:

* DIELEC-tag = 0 | 1

 === =========================================================================
  0   Do nothing
  1   Compute the frequency-dependent dielectric function
 === =========================================================================

 :Default: 0
 :Type: Integer
 :Description: When ``DIELEC = 1``, the phonon contribution to the frequency-dependent dielectric function :math:`\epsilon(\omega)` is computed from the Born effective charges and the zone-center phonon modes and saved in ``PREFIX``.dielec (``MODE = phonons``). ``BORNINFO`` must be given. When ``MODE = SCPH`` (``QHA``), the dielectric function is computed with the temperature-renormalized phonon frequencies and saved in ``PREFIX``.scph_dielec (``PREFIX``.qha_dielec).

````

.. _anphon_quartic:

* QUARTIC-tag = 0 | 1 | 2

 === ====================================================================================
  0   Quartic force constants are not used.
  1   | Read the quartic force constants and enable the quartic terms.
      | ``QUARTIC = 1`` is required when ``MODE = SCPH`` or ``MODE = QHA``.
  2   | In the mode analysis (:ref:`KS_INPUT <anphon_ks_input>`), additionally compute
      | the contribution of a specific four-phonon diagram to the linewidth (expensive).
 === ====================================================================================

 :Default: 0
 :Type: Integer
 :Description: When ``QUARTIC = 1``, the quartic force constants are read from ``FCSFILE`` (or ``FC4FILE``). In the mode analysis with ``REALPART = 1``, the frequency shift due to the loop diagram of the quartic anharmonicity is then computed in addition to the third-order terms; The former role of ``QUARTIC`` as the switch of four-phonon scattering in ``MODE = kappa`` is deprecated in favor of :ref:`INCLUDE_4PH <anphon_include_4ph>`; ``QUARTIC > 0`` without ``INCLUDE_4PH`` still enables it, with a warning.

````

.. _anphon_ks_input:

* KS_INPUT-tag : File containing a list of phonon modes to be analyzed

 :Default: None
 :Type: String
 :Description: When ``MODE = kappa`` and ``KS_INPUT`` is given, the mode analysis of the phonon modes listed in the specified file is performed *instead of* the thermal conductivity calculation. The first line of the file gives the number of entries, and each of the following lines contains the fractional coordinates of a :math:`k` point and a branch index (1-based) as ``k1 k2 k3 s``. Each :math:`k` point must be a point of the grid given in the ``&kpoint`` field (*KPMODE* = 2). The quantities to be computed are selected by :ref:`SELF_ENERGY <anphon_self_energy>`, :ref:`REALPART <anphon_realpart>`, :ref:`SELF_W <anphon_self_w>`, :ref:`PRINTV3 <anphon_printv3>`, and :ref:`PRINTV4 <anphon_printv4>`.

````

.. _anphon_self_energy:

* SELF_ENERGY-tag = 0 | 1

 === =========================================================================
  0   Do nothing
  1   | Compute the temperature dependence of the phonon linewidth for
      | the modes given in ``KS_INPUT``
 === =========================================================================

 :Default: 0
 :Type: Integer
 :Description: For each mode listed in :ref:`KS_INPUT <anphon_ks_input>`, the linewidth (FWHM = :math:`2\Gamma`) due to three-phonon interactions is computed as a function of temperature and saved in ``PREFIX``.Gamma.[number]. With ``QUARTIC = 2``, contributions of specific four-phonon diagrams are appended. When ``REALPART = 1``, the frequency shift is also computed and saved in ``PREFIX``.Shift.[number].

````

.. _anphon_realpart:

* REALPART-tag = 0 | 1

 :Default: 0
 :Type: Integer
 :Description: When ``REALPART = 1`` is used with :ref:`KS_INPUT <anphon_ks_input>` and ``SELF_ENERGY = 1``, the real part of the phonon self-energy --- the frequency shift due to the tadpole and bubble diagrams of the cubic anharmonicity and, if ``QUARTIC = 1``, the loop diagram of the quartic anharmonicity --- is computed and saved in ``PREFIX``.Shift.[number]. This option works only with ``ISMEAR = 0``.

````

.. _anphon_self_w:

* SELF_W-tag = 0 | 1

 :Default: 0
 :Type: Integer
 :Description: When ``SELF_W = 1`` is used with :ref:`KS_INPUT <anphon_ks_input>`, the frequency dependence of the three-phonon (bubble) self-energy is computed at each temperature for the modes listed in ``KS_INPUT`` and saved in ``PREFIX``.Self.[number], from which spectral functions can be obtained. This option works only with the tetrahedron method (``ISMEAR = -1``).

````

.. _anphon_printv3:

* PRINTV3-tag = 0 | 1 | 2

 === ===================================================================
  0   Do nothing
  1   Print :math:`|V_{3}|^{2}` matrix elements in ``PREFIX``.V3.[number]
  2   Print :math:`\Phi_{3}` matrix elements in ``PREFIX``.Phi3.[number]
 === ===================================================================

 :Default: 0
 :Type: Integer
 :Description: For each mode listed in :ref:`KS_INPUT <anphon_ks_input>`, the three-phonon matrix elements involving the mode are printed. ``PRINTV3 = 1`` prints the squared elements :math:`|V_{3}(-qj; q'j', q''j'')|^{2}` in units of cm\ :sup:`-2`, while ``PRINTV3 = 2`` prints the complex elements :math:`\Phi_{3}(qj; q'j', q''j'')` in units of Ry/(u\ :sup:`1/2` Bohr)\ :sup:`3`.

````

.. _anphon_printv4:

* PRINTV4-tag = 0 | 1 | 2

 === ===================================================================
  0   Do nothing
  1   Print :math:`|V_{4}|^{2}` matrix elements in ``PREFIX``.V4.[number]
  2   Print :math:`\Phi_{4}` matrix elements in ``PREFIX``.Phi4.[number]
 === ===================================================================

 :Default: 0
 :Type: Integer
 :Description: Same as :ref:`PRINTV3 <anphon_printv3>`, but for the four-phonon matrix elements :math:`|V_{4}(-qj; q_1 j_1, q_2 j_2, q_3 j_3)|^{2}` (in cm\ :sup:`-2`) and :math:`\Phi_{4}` (in Ry/(u\ :sup:`1/2` Bohr)\ :sup:`4`). The quartic force constants are required (``QUARTIC = 1``).

````

.. _anphon_fc2_ewald:

* FC2_EWALD-tag = 0 | 1

 === ====================================================================================
  0   Do nothing
  1   | Print the dipole-dipole and short-range components of the harmonic
      | force constants in ``PREFIX``.fc2_ewald
 === ====================================================================================

 :Default: 0
 :Type: Integer
 :Description: Available only when ``NONANALYTIC = 3``. For each pair of atoms, the original harmonic force constant, its dipole-dipole (Ewald) part, and the short-range remainder are printed, which is useful for checking the long-range separation of the harmonic force constants.

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

"&kappa"-field (Read only when ``MODE = kappa``)
++++++++++++++++++++++++++++++++++++++++++++++++

.. _anphon_solver:

* SOLVER-tag : Solver level of the Boltzmann transport equation

 ====== ========================================================================
  RTA    Relaxation-time approximation (single-mode approximation)
  IBTE   Iterative solution of the linearized BTE (result in ``PREFIX``.kl_iter)
  VBTE   Variational solution by preconditioned conjugate gradients
         (result in ``PREFIX``.kl_iter)
  DBTE   Direct solution: dense eigendecomposition of the collision kernel
         (diagnostic; small meshes; result in ``PREFIX``.kl_iter)
 ====== ========================================================================

 :Default: RTA
 :Type: String
 :Description: Case insensitive. ``SOLVER = IBTE`` replaces the deprecated
               ``ITERATIVE = 1`` tag; the iteration is controlled by
               :ref:`MIN_CYCLE <anphon_min_cycle>`, :ref:`MAX_CYCLE <anphon_max_cycle>`,
               :ref:`ITER_THRESHOLD <anphon_iter_threshold>`, and
               :ref:`IBTE_MIXING <anphon_ibte_mixing>`. With the default
               ``FILE_FORMAT = h5``, the per-temperature results are stored in
               the ``/iterativebte`` group of ``PREFIX``.kappa.h5 and an
               interrupted temperature sweep restarts from the missing
               temperatures (``RESTART = 0`` forces a recomputation).
               Temperatures whose iteration hits ``MAX_CYCLE`` or diverges are
               reported and flagged as unconverged (also in a comment line of
               ``PREFIX``.kl_iter, with the lowest-residual iterate kept); a
               restart continues their iteration from the stored deviation
               function instead of skipping them.

 ``SOLVER = VBTE`` solves the same linearized BTE as ``IBTE`` by
 preconditioned conjugate gradients on the symmetrized collision operator.
 For ``VBTE``, ``ITER_THRESHOLD`` acts on the relative residual of the linear
 system; since kappa is the value of the variational functional, its error is
 quadratic in the residual. ``MIN_CYCLE`` and ``IBTE_MIXING`` are not used by
 ``VBTE``. All non-RTA solvers share the ``/iterativebte`` restart state in
 ``PREFIX``.kappa.h5.

 ``SOLVER = DBTE`` is a diagnostic solver: it assembles the collision kernel
 on the irreducible wedge as an explicit dense matrix - restricted to the
 degeneracy-reduced, little-group-invariant basis, in the Omega normalization
 whose diagonal is :math:`1/\tau` - and computes its full eigendecomposition
 with LAPACK. It reports what the matrix-free solvers cannot access: the
 residual asymmetry of the discretized kernel, violations of positive
 semidefiniteness (a direct probe of a too-coarse mesh or an inadequate
 smearing width), the near-null part of the spectrum (scattering rates in
 cm^-1) with its overlap onto the momentum-drift directions, and the
 sensitivity of kappa to dropping the softest modes. All iterative-family
 solvers use the detailed-balance-symmetric occupation kernel
 (:math:`g_1 g_2 g_3` with :math:`g = \sqrt{n(n+1)}`, identical on the
 energy shell to the raw occupation products), so for fixed-width smearing
 the three solvers agree to their tolerances; with the tetrahedron or
 adaptive methods a small residual difference remains from the direction
 dependence of the integration weights. Memory scales with the square of the
 reduced dimension; oversized problems abort with a clear message
 (memory-distributed ELPA/ScaLAPACK and GPU MAGMA/cuSOLVER backends are
 planned behind the same interface).

 .. caution::

     ``SOLVER = IBTE``, ``VBTE``, and ``DBTE`` are pilot implementations under
     development. Please check the validity of the results carefully.

````

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

.. _anphon_kmesh_coarse:

* KMESH_COARSE-tag = k1, k2, k3

 :Default: The same :math:`k` mesh as given in the ``&kpoint`` field
 :Type: Array of integers
 :Description: :math:`k` mesh used for evaluating the four-phonon scattering rates when ``INCLUDE_4PH = 1``. Because four-phonon calculations are far more expensive than three-phonon ones, a mesh coarser than that of the ``&kpoint`` field can be given here; the computed rates are then interpolated onto the dense mesh by the method selected with :ref:`INTERPOLATOR <anphon_interpolator>`.

````

.. _anphon_ismear_4ph:

* ISMEAR_4PH-tag = -1 | 0 | 1 | 2

 === ==========================================================================
  -1  Not implemented for four-phonon scattering; automatically switched to 2
  0   Lorentzian smearing with width of ``EPSILON_4PH``
  1   Gaussian smearing with width of ``EPSILON_4PH``
  2   Adaptive Gaussian smearing
 === ==========================================================================

 :Default: 1
 :Type: Integer
 :Description: Brillouin-zone integration method for the four-phonon scattering
               rates (``INCLUDE_4PH = 1``), independent of the three-phonon
               setting :ref:`ISMEAR <anphon_ismear>`. ``ISMEAR_4PH = 2`` uses the
               same adaptive-broadening scheme as ``ISMEAR = 2``, generalized to
               the three-mode energy denominators of the four-phonon processes
               and evaluated on the (possibly coarser) mesh given by
               ``KMESH_COARSE``; the width is scaled by
               :ref:`ADAPTIVE_FACTOR <anphon_adaptive_factor>` and
               ``EPSILON_4PH`` is not used. Adaptive smearing is recommended for
               the coarse four-phonon meshes, where a single fixed width is hard
               to converge.

````

.. _anphon_epsilon_4ph:

* EPSILON_4PH-tag : Smearing width for four-phonon scattering rates in units of Kayser (cm\ :sup:`-1`)

 :Default: 10.0
 :Type: Double
 :Description: Used when ``ISMEAR_4PH = 0`` or ``1``; neglected when
               ``ISMEAR_4PH = 2``. Because the four-phonon phase space is much
               larger and the mesh usually coarser than for three-phonon
               processes, the optimal width generally differs from ``EPSILON``.

````

.. _anphon_interpolator:

* INTERPOLATOR-tag = linear | log-linear | modified-log-linear

 ==================== =================================================================
  linear               Trilinear interpolation of the scattering rates
  log-linear           Trilinear interpolation of the logarithm of the scattering rates
  modified-log-linear  | Same as log-linear, but for the acoustic branches the
                       | interpolation cells containing the :math:`\Gamma` point
                       | are avoided
 ==================== =================================================================

 :Default: log-linear
 :Type: String
 :Description: Method for interpolating the four-phonon scattering rates computed on :ref:`KMESH_COARSE <anphon_kmesh_coarse>` onto the dense :math:`k` mesh of the ``&kpoint`` field. The value is case-insensitive.

````

.. _anphon_write_interpol:

* WRITE_INTERPOL-tag = 0 | 1

 :Default: 0
 :Type: Integer
 :Description: When ``WRITE_INTERPOL = 1`` and the four-phonon scattering rates are interpolated (``INCLUDE_4PH = 1``), the interpolated linewidths on the dense :math:`k` mesh are written to ``PREFIX``.interpolated_gamma, which is useful for checking the interpolation quality.

````

.. _anphon_adaptive_factor:

* ADAPTIVE_FACTOR-tag : Scaling prefactor :math:`\alpha` of the adaptive smearing width

 :Default: 1.0
 :Type: Double
 :Description: Multiplies the automatically determined Gaussian widths of the
               adaptive smearing method, for both ``ISMEAR = 2`` (three-phonon)
               and ``ISMEAR_4PH = 2`` (four-phonon). The default is usually
               adequate; smaller values sharpen the energy conservation (and may
               require denser meshes), larger values smooth it.

````

.. _anphon_restart:

* RESTART-tag : Flag to restart the calculation when ``MODE = kappa``

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
 :Description: This flag is available when ``MODE = kappa``. For the theoretical details, please see :ref:`this page <kappa_coherent>`.

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
 :Description: This flag is available when ``MODE = kappa``.


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
 :Description: When ``MODE = kappa`` and ``ISOTOPE = 1 or 2``, phonon scatterings due to isotopes will be considered perturbatively.
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

.. _anphon_isotope_inscattering:

* ISOTOPE_INSCATTERING-tag = 0 | 1

 === =========================================================================
  0   Isotope scattering enters the iterative solvers as an RTA-level
      (diagonal) rate only
  1   The elastic isotope-disorder channel (Tamura kernel) becomes part of
      the collision operator, including its in-scattering term
 === =========================================================================

 :Default: 1
 :Type: Integer
 :Description: Used by ``SOLVER = IBTE`` and ``VBTE`` when ``ISOTOPE >= 1``
               (``SOLVER = RTA`` is unaffected). With the in-scattering term the
               isotope diagonal is the row sum of the operator entries, so the
               elastic channel annihilates constant deviation functions exactly.
               The effect on kappa is positive (in-scattering restores heat flux)
               and grows with the mass variance and towards low temperatures; for
               natural silicon it is below 0.2 %. Boundary scattering remains
               exactly diagonal (it has no in-scattering term), and the
               four-phonon channel is treated at the RTA level.

````

.. _anphon_len_boundary:

* LEN_BOUNDARY-tag : Characteristic length for boundary scattering in units of meter

 :Default: 0.0 (no boundary scattering)
 :Type: Double
 :Description: When ``LEN_BOUNDARY > 0``, a boundary scattering rate :math:`|\boldsymbol{v}_{qj}|/L` with :math:`L` = ``LEN_BOUNDARY`` is added to the phonon linewidths used for the thermal conductivity, where :math:`\boldsymbol{v}_{qj}` is the group velocity of each phonon mode.

````

.. _anphon_iterative:

* ITERATIVE-tag = 0 | 1

 === ====================================================================================
  0   Solve the BTE within the relaxation-time approximation (RTA)
  1   Solve the linearized BTE iteratively (beyond the RTA)
 === ====================================================================================

 :Default: 0
 :Type: Integer
 :Description: Deprecated alias of :ref:`SOLVER <anphon_solver>` = IBTE (a warning is printed; ``SOLVER`` wins when both are given). When enabled, the linearized phonon Boltzmann transport equation is solved self-consistently by iteration instead of the RTA, and the resulting lattice thermal conductivity is written to ``PREFIX``.kl_iter. The iteration is controlled by :ref:`MIN_CYCLE <anphon_min_cycle>`, :ref:`MAX_CYCLE <anphon_max_cycle>`, :ref:`ITER_THRESHOLD <anphon_iter_threshold>`, and :ref:`IBTE_MIXING <anphon_ibte_mixing>`.

````

.. _anphon_min_cycle:

* MIN_CYCLE-tag : Minimum number of iterations of the iterative BTE solver

 :Default: 5
 :Type: Integer
 :Description: Used when ``ITERATIVE = 1``. The convergence test starts only after ``MIN_CYCLE`` iterations have been performed.

````

.. _anphon_max_cycle:

* MAX_CYCLE-tag : Maximum number of iterations of the iterative BTE solver

 :Default: 20
 :Type: Integer
 :Description: Used when ``ITERATIVE = 1``. If the iteration does not converge within ``MAX_CYCLE`` cycles, the last value of the thermal conductivity is kept and the calculation proceeds to the next temperature.

````

.. _anphon_iter_threshold:

* ITER_THRESHOLD-tag : Convergence criterion of the iterative BTE solver

 :Default: 0.02
 :Type: Double
 :Description: Used when ``ITERATIVE = 1``. The iteration at each temperature stops when the relative change of every diagonal component of the thermal conductivity tensor between two successive iterations becomes smaller than ``ITER_THRESHOLD``.

````

.. _anphon_ibte_mixing:

* IBTE_MIXING-tag : Mixing factor of the iterative BTE solver

 :Default: 0.9
 :Type: Double
 :Description: Used when ``ITERATIVE = 1``. In each iteration, the updated nonequilibrium distribution is mixed with the previous one as :math:`f_{\mathrm{new}} \leftarrow \alpha f_{\mathrm{new}} + (1-\alpha) f_{\mathrm{old}}` with :math:`\alpha` = ``IBTE_MIXING``. Values smaller than 1 damp oscillations of slowly converging iterations.

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
