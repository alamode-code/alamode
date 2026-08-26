ANPHON: Output files
--------------------

.. _reference_output:

.. |umulaut_u|    unicode:: U+00FC


* ``PREFIX``.bands

 Phonon dispersion along given :math:`k` paths in units of cm :sup:`-1`.
 Created when ``MODE = phonons`` with **KPMODE** = 1.

* ``PREFIX``.dos

 Phonon density of states (DOS). Atom projected phonon DOSs are also printed when ``PDOS = 1``.
 Created when ``MODE = phonons`` with **KPMODE** = 2.

* ``PREFIX``.tdos

 Two-phonon density of states for all irreducible :math:`k` points.
 Created when ``MODE = phonons`` with **KPMODE** = 2 and ``TDOS = 1``.

* ``PREFIX``.thermo

 Constant volume heat capacity, vibrational entropy, internal energy, and vibrational free energy.
 Created when ``MODE = phonons`` with **KPMODE** = 2.
 When ``FE_BUBBLE = 1`` is set in the **&analysis** field, an additional bubble correction term 
 to the vibrational free energy is also calculated.

* ``PREFIX``.msd
 
 Mean-square-displacements of atoms.
 Created when ``MODE = phonons`` with **KPMODE** = 2 and ``PRINTMSD = 1``.

* ``PREFIX``.sps

 Total and mode-decomposed scattering phase space. 
 Created when ``MODE = phonons`` with **KPMODE** = 2 and ``SPS = 1``.

* ``PREFIX``.pr

 Participation ratio of every phonon mode.
 Created when ``MODE = phonons`` and ``PRINTPR = 1``.

* ``PREFIX``.apr

 Atomic participation ratio of every phonon mode.
 Created when ``MODE = phonons`` and ``PRINTPR = 1``.

* ``PREFIX``.phvel

 Phonon group velocity along given :math:`k` paths.
 Created when ``MODE = phonons`` with **KPMODE** = 1 and ``PRINTVEL = 1``.

* ``PREFIX``.phvel_all

 Magnitude of group velocity :math:`|\boldsymbol{v}|` of all phonon modes at the uniform :math:`k` grid. 
 Created when ``MODE = phonons`` with **KPMODE** = 2 and ``PRINTVEL = 1``.

* ``PREFIX``.evec, ``PREFIX``.band.evec, ``PREFIX``.mesh.evec

 Eigenvalues and eigenvectors of dynamical matrices.
 Eigenvalues are printed in Rydberg atomic units.
 Created when ``MODE = phonons`` with ``PRINTEVEC = 1``.

* ``PREFIX``.gruneisen

 Gr\ |umulaut_u|\ neisen parameters along given :math:`k` paths.
 Created when ``MODE = phonons`` with **KPMODE** = 1 and ``GRUNEISEN >= 1``.
 ``GRUNEISEN = 1`` gives the volumetric parameters
 :math:`\gamma_{\boldsymbol{q}j} = -\partial \log\omega_{\boldsymbol{q}j}/\partial \log V`.
 With ``GRUNEISEN = 2`` or ``3``, the generalized (strain-component-resolved)
 parameters :math:`\gamma_{\boldsymbol{q}j}^{\mu\nu} = -\partial \log\omega_{\boldsymbol{q}j}/\partial \varepsilon_{\mu\nu}`
 are written in a long format with one line per (:math:`k` point, branch).


* ``PREFIX``.gru_all

 Gr\ |umulaut_u|\ neisen parameters of all phonon modes at the uniform :math:`k` grid.
 Created when ``MODE = phonons`` with **KPMODE** = 2 and ``GRUNEISEN >= 1``.
 ``GRUNEISEN = 1`` gives the volumetric parameters
 :math:`\gamma_{\boldsymbol{q}j} = -\partial \log\omega_{\boldsymbol{q}j}/\partial \log V`.
 With ``GRUNEISEN = 2`` or ``3``, the generalized (strain-component-resolved)
 parameters are written with one column per strain component.


* ``PREFIX``\_+.h5, ``PREFIX``\_-.h5 (``PREFIX``\_+.xml, ``PREFIX``\_-.xml with ``FILE_FORMAT = text``)

 Estimated force constants of deformed systems, usable as ``FCSFILE``
 of subsequent calculations. Created when ``MODE = phonons`` with ``NEWFCS = 1``;
 the format follows ``FILE_FORMAT`` (HDF5 by default, the legacy XML with ``FILE_FORMAT = text``).
 The two files correspond to the strains :math:`+u` and :math:`-u` given in the ``&strain``
 field (a small isotropic strain of :math:`\pm 0.001` when the field is absent).
 With ``SUBLATTICE_RELAX = 1``, the written structures and force constants include the
 strain-induced internal (sublattice) displacements.


* ``PREFIX``.zmode

 Mode effective charges of zone-center phonon modes.
 Created when ``MODE = phonons`` with ``ZMODE = 1``.

* ``PREFIX``.irreps

 Irreducible representations (Mulliken symbols), IR/Raman activities, and,
 when ``BORNINFO`` is given, IR oscillator strengths of the zone-center
 phonon modes, together with the conjugacy classes and the raw characters
 of each phonon multiplet.
 Created when ``MODE = phonons`` with ``IRREPS = 1``.

* ``PREFIX``.axsf

 Zone-center phonon modes with directions indicated by arrows.
 This file can be visualized by XcrySDen.
 Created when ``MODE = phonons`` with ``PRINTXSF = 1``.

* ``PREFIX``.anime???.axsf and ``PREFIX``.anime???.xyz

 Files for animating phonon modes. ??? is the mode number.
 Created when ``MODE = phonons`` with a proper ``ANIME``-tag.
 If ``ANIME_FORMAT = xsf``, axsf files will be created which can be displayed by XcrySDen.
 If ``ANIME_FORMAT = xyz``, xyz files will be created which can be visualized by VMD, Jmol, etc.

````

* ``PREFIX``.kappa.h5

 Unified, crash-safe HDF5 result file of ``MODE = kappa`` (schema
 ``alamode:kappa_result``). It stores the run metadata, phonon frequencies,
 group velocities, the three-phonon (and, when ``QUARTIC = 1``, four-phonon)
 linewidths with per-mode completion flags, the isotope-scattering linewidths
 and factors (``/scattering/isotope/gamma``, ``/metadata/isotope_factors``,
 when ``ISOTOPE > 0``), and the final thermal-conductivity
 tensors: ``/kappa/kappa_peierls`` (the intraband Peierls term, i.e. the
 contents of ``PREFIX``.kl) and, when ``KAPPA_COHERENT > 0``,
 ``/kappa/kappa_coherent`` together with ``/kappa/kappa_total`` =
 ``kappa_peierls`` + ``kappa_coherent`` (the Wigner total; it is written only
 when the coherent term is actually computed, so "total" never silently means
 "Peierls only"). Every kappa dataset carries a ``scattering_processes``
 attribute naming the linewidth contributions it includes (e.g.
 ``3ph+4ph+isotope+boundary``), and the ``/kappa`` group mirrors the same
 information as machine-readable flags (``includes_isotope_scattering``,
 ``includes_boundary_scattering``, ``includes_4ph_scattering``,
 ``boundary_length``); the labels always describe the run that produced the
 tensors — changing such a setting between restarts rebuilds the group while
 keeping the computed linewidths. It is updated incrementally during the run and read back when the
 restart mode is on (``RESTART = 1`` / ``RESTART_4PH = 1``). Written with the
 default ``FILE_FORMAT = h5``; readable with h5py.

 With ``SOLVER = IBTE``, the file additionally carries an
 ``/iterativebte`` group holding the per-temperature results of the
 iterative solver: the diagonal (out-scattering) part ``Q``, the deviation
 function ``dF`` at the irreducible k points, the kappa tensor, and
 solver-convergence/completion flags. Each temperature is committed durably
 as soon as it finishes, so an interrupted temperature sweep restarts from
 the missing temperatures only (``RESTART = 0`` discards the stored
 results); when every temperature is already present, the expensive
 transition probabilities are not rebuilt at all. RTA and IBTE results
 coexist in the same file.

 When the run uses a temperature-dependent basis
 (:ref:`FC2_TEMPERATURE <anphon_fc2_temperature>` with an SCPH/QHA state
 file), the file switches to a temperature-resolved layout
 (``format_version = 2``): frequencies, group velocities, linewidths, and
 kappa are stored separately for each temperature, and runs at different
 basis temperatures accumulate into the same file — its temperature grid
 grows as new temperatures are computed, so a full kappa-vs-T sweep on top
 of SCPH/QHA ends up in a single portable file. Set
 ``TMIN = TMAX = FC2_TEMPERATURE`` in each run so that every kappa value is
 computed with the self-consistent basis of its own temperature.

* ``PREFIX``.result

 Legacy text counterpart of ``PREFIX``.kappa.h5, written when
 ``FILE_FORMAT = text``. In this file, phonon frequency, group velocity, and
 anharmonic phonon linewidths are printed. An existing file from an older run
 is imported into ``PREFIX``.kappa.h5 once (read-only) when restarting in the
 default h5 mode.

* ``PREFIX``.kl

 Lattice thermal conductivity tensor (Peierls term). Created when ``MODE = kappa``.

* ``PREFIX``.kl_iter

 Lattice thermal conductivity tensor obtained with ``SOLVER = IBTE``
 (full 3x3 tensor per temperature). Created when ``MODE = kappa`` and
 ``SOLVER = IBTE``.

* ``PREFIX``.kl_spec

 Spectra of lattice thermal conductivity. Only diagonal components are saved.
 Created when ``MODE = kappa`` and ``KAPPA_SPEC = 1``.


* ``PREFIX``.kl_coherent

 Coherent component of lattice thermal conductivity. Created when ``KAPPA_COHERENT > 0`` in ``MODE = kappa``.


* ``PREFIX``.kc_elem

 Momentum- and mode-decomposed contributions to the coherent components of lattice thermal conductivity. 
 Created when ``KAPPA_COHERENT = 2`` in ``MODE = kappa``.


* ``PREFIX``.self_isotope

 Phonon selfenergy due to isotope scatterings calculated by Tamura's formula.
 Created when ``MODE = kappa`` and ``ISOTOPE = 2``.

````

* ``PREFIX``.scph.h5 (``PREFIX``.qha.h5 for ``MODE = QHA``)

 Unified state file of an SCPH/QHA run (schema ``alamode:scph_state``),
 written atomically at the end of the run with the default
 ``FILE_FORMAT = h5``. It bundles everything the legacy
 ``PREFIX``.scph_dymat / ``PREFIX``.renorm_harm_dymat / ``PREFIX``.V0 trio
 stored (and is the preferred restart source), plus per-temperature
 convergence flags of the SCPH iteration and the structural optimization
 (``/convergence/scph``, ``/convergence/structure``; unconverged
 temperatures are refused by later calculations unless
 :ref:`ALLOW_UNCONVERGED <anphon_allow_unconverged>` is set), and the
 temperature-dependent effective harmonic force constants in the standard
 ``/ForceConstants/Order2`` layout: a later anphon run can consume them
 directly via ``FCSFILE = PREFIX.scph.h5`` together with the
 :ref:`FC2_TEMPERATURE <anphon_fc2_temperature>` tag, replacing the
 ``dfc2.py`` workflow. Readable with h5py.

* ``PREFIX``.scph_dymat

 Anharmonic dynamical matrix calculated on the :math:`k` grid defined by the ``KMESH_INTERPOLATE`` tag.
 Legacy restart file, written when ``FILE_FORMAT = text``; an existing file
 from an older run is imported into ``PREFIX``.scph.h5 once when restarting
 in the default h5 mode.

* ``PREFIX``.scph_bands

 Anharmonic phonon dispersion curves. 

* ``PREFIX``.scph_dos

 Anharmonic phonon DOS. Created when ``MODE = SCPH`` and ``DOS = 1`` with **KPMODE** = 2.


* ``PREFIX``.scph_thermo

 Constant volume heat capacity, vibrational entropy, and vibrational free energy calculated based on the self-consistent phonon calculation. 
 Created when ``MODE = SCPH`` with **KPMODE** = 2.
 
..  When ``FE_BUBBLE = 1`` is set in the **&analysis** field, an additional bubble correction term 
..  to the vibrational free energy is also calculated.

* ``PREFIX``.scph_msd

 Mean square displacement calculated within the SCPH theory. Created when ``MODE = SCPH`` and ``PRINTMSD = 1`` with **KPMODE** = 2.

* ``PREFIX``.scph_dfc2

 This file contains :math:`\Delta D(\boldsymbol{q}) = D_{\mathrm{SCPH}}(\boldsymbol{q}) - D_{\mathrm{Harmonic}}(\boldsymbol{q})`.
 For the definition, see the :ref:`formalism of the SCPH calculation <formalism_SCPH>`.

````

* ``PREFIX``.atom_disp
 
 Temperature-dependence of the atomic displacements :math:`u^{(0)}_{\alpha \mu}` in Cartesian representation. Created when ``MODE = SCPH`` and ``RELAX_STR != 0``.

* ``PREFIX``.normal_disp

 Temperature-dependence of the atomic displacement :math:`q^{(0)}_{\lambda}` in normal coordinate representation. Created when ``MODE = SCPH`` and ``RELAX_STR != 0``.

* ``PREFIX``.umn_tensor

 Temperature-dependence of the displacement gradient tensor :math:`u_{\mu \nu}`. Created when ``MODE = SCPH`` and ``RELAX_STR = 2, -1, -2``.

* ``PREFIX``.V0

 Temperature-dependent zero-th order IFC :math:`U_0`. Created when ``MODE = SCPH`` and ``RELAX_STR != 0``.
 This file is used to restart the SCPH/QHA + structural optimization.

* ``PREFIX``.renorm_harm_dymat

 Renormalization of harmonic dynamical matrix by the structure change. Created when ``MODE = SCPH`` and ``RELAX_STR != 0``.
 This file is used to restart the SCPH/QHA + structural optimization.

* step_q0.txt

 Record of atomic displacement :math:`q^{(0)}_{\lambda}` at all steps of structural optimization.

* step_u0.txt

 Record of atomic displacements :math:`u^{(0)}_{\alpha \mu}` at all steps of structural optimization.

* step_u_tensor.txt

 Record of displacement gradient tensor :math:`u_{\mu \nu}` at all steps of structural optimization.