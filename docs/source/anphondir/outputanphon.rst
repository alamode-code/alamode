ANPHON: Output files
--------------------

.. _reference_output:

.. |umulaut_u|    unicode:: U+00FC

The files are listed by ``MODE``. ``PREFIX`` is the value of the ``PREFIX`` tag.
Several files come in two formats: the HDF5 version is written by default, and the plain-text version with ``FILE_FORMAT = text``.
The layouts of the HDF5 files are described in :doc:`../hdf5_format`.

``MODE = phonons``
~~~~~~~~~~~~~~~~~~

Files written for the points of the ``&kpoint`` field. *KPMODE* 0, 1, and 2 give a list of points, band paths, and a uniform mesh, respectively.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - File
     - Written when
     - Contents
   * - ``PREFIX``.bands
     - *KPMODE* = 1
     - Phonon dispersion (cm\ :sup:`-1`) along the paths
   * - ``PREFIX``.connection
     - *KPMODE* = 1, ``BCONNECT = 2``
     - How the branches connect along the paths
   * - ``PREFIX``.dos
     - *KPMODE* = 2 (``DOS = 1``, default)
     - Phonon DOS; atom-projected DOS with ``PDOS = 1``
   * - ``PREFIX``.tdos
     - *KPMODE* = 2, ``TDOS = 1``
     - Two-phonon DOS at the irreducible :math:`k` points
   * - ``PREFIX``.longitudinal_dos
     - *KPMODE* = 2, ``LONGITUDINAL_DOS = 1``
     - Longitudinal-projected DOS
   * - ``PREFIX``.thermo
     - *KPMODE* = 2
     - Heat capacity, entropy, internal energy, and free energy vs. temperature
   * - ``PREFIX``.msd
     - *KPMODE* = 2, ``PRINTMSD = 1``
     - Mean-square displacements of atoms vs. temperature
   * - ``PREFIX``.ucorr
     - *KPMODE* = 2, ``UCORR = 1``
     - Displacement–displacement correlation function (``SHIFT_UCORR``)
   * - ``PREFIX``.sps, ``PREFIX``.sps_Bose
     - *KPMODE* = 2, ``SPS = 1`` or ``2``
     - Three-phonon scattering phase space; ``.sps_Bose`` includes the Bose factor
   * - ``PREFIX``.phvel, ``PREFIX``.phvel_all
     - ``PRINTVEL = 1`` with *KPMODE* = 1 or 2
     - Group velocities along the paths, or at every mesh point (magnitude and components)
   * - ``PREFIX``\ [.band | .mesh].eval.h5
     - ``PRINTEVAL = 1``
     - Phonon eigenvalues. The file name follows *KPMODE* 0, 1, 2 (no suffix, ``.band``, ``.mesh``);
       plain text ``PREFIX``\ [.band | .mesh].eval with ``FILE_FORMAT = text``
   * - ``PREFIX``\ [.band | .mesh].evec.h5
     - ``PRINTEVEC = 1``
     - Phonon eigenvalues and eigenvectors, named as above;
       plain text ``PREFIX``\ [.band | .mesh].evec with ``FILE_FORMAT = text``. Both can be given to ``displace.py --evec``
   * - ``PREFIX``\ [.band | .mesh].pr, .apr
     - ``PRINTPR = 1``
     - Participation ratio and atomic participation ratio of every mode, named as above
   * - ``PREFIX``\ [.band | .mesh].axsf
     - ``PRINTXSF = 1``
     - Mode directions at the given :math:`k` points, for XCrySDen, named as above
   * - ``PREFIX``.animeNNN.xyz, ``PREFIX``.animeNNN.axsf
     - ``ANIME`` (with ``ANIME_CELLSIZE``)
     - Animation of mode NNN; XYZ (default) or AXSF with ``ANIME_FORMAT``
   * - ``PREFIX``.gru_kpoints, ``PREFIX``.gruneisen, ``PREFIX``.gru_all
     - ``GRUNEISEN >= 1`` with *KPMODE* = 0, 1, or 2
     - Gr\ |umulaut_u|\ neisen parameters at the given points, along the paths, or on the mesh:
       volumetric (``GRUNEISEN = 1``) or for each strain component (``2``, ``3``)
   * - ``PREFIX``\_+.h5, ``PREFIX``\_-.h5
     - ``NEWFCS = 1``
     - Force constants of the crystal strained by :math:`\pm u`, usable as ``FCSFILE``
       (``.xml`` with ``FILE_FORMAT = text``)
   * - ``PREFIX``.dielec
     - ``DIELEC = 1`` (needs ``BORNINFO``)
     - Phonon contribution to the dielectric function :math:`\epsilon(\omega)`
   * - ``PREFIX``.zmode
     - ``ZMODE = 1`` (needs ``BORNINFO``)
     - Mode effective charges of the zone-center modes
   * - ``PREFIX``.irreps
     - ``IRREPS = 1``
     - Irreducible representations and IR/Raman activities at :math:`\Gamma`
   * - ``PREFIX``.fc2_ewald
     - ``NONANALYTIC = 3``, ``FC2_EWALD = 1`` (any ``MODE``)
     - Harmonic force constants split into the dipole–dipole and short-range parts

``MODE = kappa``
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - File
     - Written when
     - Contents
   * - ``PREFIX``.kappa.h5
     - Always (default format)
     - All results: frequencies, velocities, linewidths, and thermal conductivity. Updated during the run and used for restarting;
       read by ``analyzer.py``. See :ref:`label_hdf5_kappa`
   * - ``PREFIX``.result, ``PREFIX``.4ph.result
     - ``FILE_FORMAT = text``
     - Legacy text versions of the three- and four-phonon linewidths
   * - ``PREFIX``.kl
     - ``SOLVER = RTA``
     - Thermal conductivity tensor vs. temperature
   * - ``PREFIX``.kl3, ``PREFIX``.kl4
     - ``SOLVER = RTA``, ``INCLUDE_4PH = 1`` (instead of ``.kl``)
     - Thermal conductivity with three-phonon scattering only, and with three- and four-phonon scattering
   * - ``PREFIX``.kl_iter
     - ``SOLVER = IBTE``, ``VBTE``, or ``DBTE``
     - Thermal conductivity from the solution of the Boltzmann equation beyond the RTA
   * - ``PREFIX``.kl_spec
     - ``SOLVER = RTA``, ``KAPPA_SPEC = 1``
     - Spectrum of the thermal conductivity (diagonal components)
   * - ``PREFIX``.kl_coherent
     - ``SOLVER = RTA``, ``KAPPA_COHERENT >= 1``
     - Coherent (off-diagonal) contribution to the thermal conductivity
   * - ``PREFIX``.kc_elem
     - ``SOLVER = RTA``, ``KAPPA_COHERENT = 2``
     - Mode-resolved contributions to the coherent term
   * - ``PREFIX``.self_isotope
     - ``SOLVER = RTA``, ``ISOTOPE = 2``
     - Linewidths due to isotope scattering
   * - ``PREFIX``.interpolated_gamma
     - ``INCLUDE_4PH = 1``, ``WRITE_INTERPOL = 1``
     - Four-phonon linewidths interpolated onto the three-phonon mesh

``MODE = selfenergy``
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - File
     - Written when
     - Contents
   * - ``PREFIX``.selfenergy.h5
     - Always (default format)
     - Results of all target modes. See :ref:`label_hdf5_selfenergy`
   * - ``PREFIX``.Gamma.N
     - ``FILE_FORMAT = text``, ``LINEWIDTH = 1`` (default)
     - Linewidth :math:`2\Gamma` vs. temperature of target N
   * - ``PREFIX``.Shift.N
     - ``FILE_FORMAT = text``, ``SHIFT = 1``
     - Frequency shifts vs. temperature
   * - ``PREFIX``.Self.N
     - ``FILE_FORMAT = text``, ``SELF_W = 1``
     - Self-energy as a function of frequency
   * - ``PREFIX``.fw.N
     - ``FILE_FORMAT = text``, ``FSTATE_W = 1``
     - Frequency-resolved final-state analysis of the linewidth
   * - ``PREFIX``.spectrum
     - ``FILE_FORMAT = text``, ``INTERPOLATE = 1``
     - Spectral function :math:`A(\boldsymbol{q}, \omega)`, total and per branch
   * - ``PREFIX``.V3.N, .Phi3.N, .V4.N, .Phi4.N
     - ``PRINTV3``, ``PRINTV4`` = 1 or 2 (always text)
     - Three- and four-phonon matrix elements involving target N

In the text files, N numbers the targets on the integration mesh first and the other targets after them;
the header of each file gives the wave vector and the branch.
The same per-target files are written in ``MODE = kappa`` with the ``KS_INPUT`` analysis (``SELF_ENERGY = 1``, ``REALPART = 1``, ...).

``MODE = SCPH`` and ``MODE = QHA``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The file names start with ``PREFIX``.scph for ``MODE = SCPH`` and ``PREFIX``.qha for ``MODE = QHA``.
With ``BUBBLE = 1, 2, 3``, SCPH additionally writes the bubble-corrected versions
``PREFIX``.scph+bubble(0), ``+bubble(w)``, or ``+bubble(wQP)`` of the eval, bands, dos, msd, ucorr, and dfc2 files.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - File
     - Written when
     - Contents
   * - ``PREFIX``.scph.h5, ``PREFIX``.qha.h5
     - Always (default format)
     - Results at every temperature, used by later runs (``FC2_TEMPERATURE``, ``RELAXED_STRUCTURE``) and for restarting.
       See :ref:`label_hdf5_scph`
   * - ``PREFIX``.scph_eval, .qha_eval
     - *KPMODE* = 0
     - Frequencies vs. temperature at the given :math:`k` points
   * - ``PREFIX``.scph_bands, .qha_bands
     - *KPMODE* = 1
     - Dispersion vs. temperature
   * - ``PREFIX``.scph_dos, .qha_dos
     - *KPMODE* = 2 (``DOS = 1``, default)
     - DOS vs. temperature
   * - ``PREFIX``.scph_thermo, .qha_thermo
     - *KPMODE* = 2
     - Heat capacity, entropy, and free energy vs. temperature
   * - ``PREFIX``.scph_hessian
     - ``BUBBLE = 4`` (MODE = SCPH, ``RELAX_STR > 0``)
     - Curvature of the SCP free energy at each converged temperature: frequencies of the SCPH
       and of the free-energy curvature (static bubble + quartic ladder) along the :math:`\Gamma`
       displacements; negative values are unstable directions
   * - ``PREFIX``.scph_hessian_displace
     - ``BUBBLE = 4``, a negative curvature
     - For each unstable direction, the ``&displace`` (``DISPMODE = 1``, Cartesian, bohr) and
       ``&strain`` fields of the converged structure moved 0.05 bohr along it, with the space group
       of the displaced structure
   * - ``PREFIX``.scph_msd, .qha_msd
     - *KPMODE* = 2, ``PRINTMSD = 1``
     - Mean-square displacements vs. temperature
   * - ``PREFIX``.scph_ucorr, .qha_ucorr
     - *KPMODE* = 2, ``UCORR = 1``
     - Displacement correlation function vs. temperature
   * - ``PREFIX``.scph_dielec, .qha_dielec
     - ``DIELEC = 1``
     - Dielectric function vs. temperature
   * - ``PREFIX``.scph_dfc2, .qha_dfc2
     - Always
     - Change of the harmonic force constants :math:`\Delta\Phi_2` at each temperature
       (see the :ref:`formalism of the SCPH calculation <formalism_SCPH>`)
   * - ``PREFIX``.atom_disp, ``PREFIX``.normal_disp
     - ``RELAX_STR`` > 0
     - Atomic displacements vs. temperature, in Cartesian and normal coordinates
   * - ``PREFIX``.umn_tensor
     - ``RELAX_STR = 2`` (and QHA with ``RELAX_STR = 3``)
     - Displacement gradient tensor :math:`u_{\mu\nu}` vs. temperature
   * - step_q0.txt, step_u0.txt, step_u_tensor.txt
     - ``RELAX_STR = 1`` or ``2`` (step_u_tensor.txt: ``RELAX_STR = 2``)
     - History of the structural optimization at every step (fixed names in the working directory)
   * - ``PREFIX``.V0
     - ``RELAX_STR`` > 0
     - Potential energy :math:`U_0` vs. temperature
   * - ``PREFIX``.scph_dymat, ``PREFIX``.renorm_harm_dymat
     - ``FILE_FORMAT = text`` (``.renorm_harm_dymat``: ``RELAX_STR`` > 0)
     - Legacy restart files: the change of the dynamical matrix

The optimization files (``atom_disp``, ``step_*.txt``, ...) and the dfc2 files are written by a new calculation, not by a restart.

.. note::
   A calculation restarts automatically when the restart files of an earlier run with the same ``PREFIX`` are found
   (``PREFIX``.kappa.h5 or ``.result``, ``PREFIX``.scph.h5 or ``.scph_dymat``, ...).
   Remove them, or change ``PREFIX``, to start from scratch.
