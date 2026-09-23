HDF5 files
==========

.. _label_hdf5_format:

ALAMODE stores the force constants and the main results in HDF5 files.
HDF5 is a portable binary format organized like a file system: **groups** (folders) contain **datasets** (arrays) and other groups,
and both can carry **attributes** (small labels such as units).
The chart below shows which program writes each file and which program reads it.

.. figure:: ../img/h5layout/h5_overview.*
   :align: center
   :alt: Which programs write and read the HDF5 files of ALAMODE
   :figclass: only-light

   HDF5 files of ALAMODE and the programs that write and read them.

.. only:: html

   .. figure:: ../img/h5layout/h5_overview.dark.*
      :align: center
      :alt: Which programs write and read the HDF5 files of ALAMODE
      :figclass: only-dark

      HDF5 files of ALAMODE and the programs that write and read them.


Reading the files
-----------------

List the contents of a file with ``h5ls -r PREFIX.h5`` (from the HDF5 tools), or read it in Python with h5py::

    import h5py

    with h5py.File("PREFIX.kappa.h5", "r") as f:
        print(f.attrs["schema"])               # which kind of file this is
        T = f["metadata/temperatures"][:]      # K
        kappa = f["kappa/kappa_peierls"][:]    # (nT, 3, 3), W/mK
        print(f["kappa/kappa_peierls"].attrs["unit"])

The files share a few conventions:

* The root attributes identify the file: ``schema`` (e.g. ``alamode:kappa_result``), ``format_version``,
  ``alamode_version``, ``alamode_git_commit``, and ``created_date``.
* Physical quantities carry their unit in a ``unit`` attribute.
* Indices of atoms, coordinates (0, 1, 2 = :math:`x, y, z`), and branches start from 0.
* ``/metadata/input_variables`` records the input variables of the run that wrote the file, one attribute per tag.

In the figures below, folder-shaped boxes are groups, the boxes next to them list their datasets with the shapes
(``()`` is a scalar) and units (``[ ]``), and dashed boxes are written only under the condition shown.
The symbols for the array sizes are explained at the bottom of each figure.

.. _label_hdf5_fcs:

Force constants (PREFIX.h5)
---------------------------

Written by *alm* (``MODE = optimize``), and by *anphon* for strained cells (``NEWFCS = 1``).
*anphon* reads it through ``FCSFILE`` (or ``FC2FILE``, ``FC3FILE``, ``FC4FILE``).

.. figure:: ../img/h5layout/h5_fcs.*
   :align: center
   :alt: Layout of the force-constant file
   :figclass: only-light

   Layout of the force-constant file (schema ``alamode:force_constants``).

.. only:: html

   .. figure:: ../img/h5layout/h5_fcs.dark.*
      :align: center
      :alt: Layout of the force-constant file
      :figclass: only-dark

      Layout of the force-constant file (schema ``alamode:force_constants``).


* ``/ForceConstants/Order2``, ``Order3``, ``Order4``, ...: the harmonic, cubic, quartic, ... force constants,
  one row per force constant :math:`\Phi_{\alpha_1 \cdots \alpha_n}(i_1, \ldots, i_n)`:

  * ``force_constant_values``: the value, in Ry/bohr\ :sup:`n`.
  * ``atom_indices``: the atoms :math:`i_1, \ldots, i_n` as indices in the primitive cell;
    ``atom_indices_supercell``: the same atoms in the supercell.
  * ``coord_indices``: the Cartesian components :math:`\alpha_1, \ldots, \alpha_n`.
  * ``shift_vectors``: the vectors from the first atom to the other atoms (Cartesian, bohr).

* ``/PrimitiveCell`` and ``/SuperCell``: lattice vectors (bohr, one vector per row), fractional coordinates, elements,
  atomic kinds, and ``mapping_table``, which gives the supercell index of every primitive-cell atom in each translated cell.

.. _label_hdf5_kappa:

Thermal conductivity (PREFIX.kappa.h5)
--------------------------------------

Written by ``MODE = kappa``. The file is updated during the run, so an interrupted calculation continues from it
with ``RESTART = 1``, and ``analyzer.py`` reads it for lifetimes, mean free paths, and cumulative thermal conductivity.

.. figure:: ../img/h5layout/h5_kappa.*
   :align: center
   :alt: Layout of the thermal-conductivity file
   :figclass: only-light

   Layout of ``PREFIX``.kappa.h5 (schema ``alamode:kappa_result``).

.. only:: html

   .. figure:: ../img/h5layout/h5_kappa.dark.*
      :align: center
      :alt: Layout of the thermal-conductivity file
      :figclass: only-dark

      Layout of ``PREFIX``.kappa.h5 (schema ``alamode:kappa_result``).


* ``/kappa``: the thermal-conductivity tensors in W/mK, one :math:`3 \times 3` tensor per temperature.
  ``kappa_peierls`` is the usual (Peierls) term, the content of ``PREFIX``.kl.
  With ``KAPPA_COHERENT > 0``, ``kappa_coherent`` and their sum ``kappa_total`` are added.
  The ``scattering_processes`` attribute of each tensor names the scattering included, e.g. ``3ph+4ph+isotope``.
  With ``INCLUDE_4PH = 1``, ``kappa_3ph_only`` keeps the three-phonon-only result,
  and with ``KAPPA_SPEC = 1``, ``kappa_spec`` holds the spectrum on the frequency axis ``energy_axis``.
* ``/scattering/3ph``: phonon frequencies (cm\ :sup:`-1`) and linewidths :math:`\Gamma` (cm\ :sup:`-1`) at the irreducible :math:`k` points,
  and group velocities (m/s) on the full mesh.
  The lifetime is :math:`\tau = 1/(2\Gamma)`.
  ``gamma`` has one row per mode (:math:`k` point × branch) and one column per temperature;
  ``gamma_computed`` marks the rows already calculated, which is what makes restarting possible.
  ``equiv_knum`` and ``equiv_offsets`` list the full-mesh :math:`k` points that belong to each irreducible point.
* ``/scattering/4ph`` (``INCLUDE_4PH = 1``) has the same layout for the four-phonon linewidths,
  and ``/scattering/isotope`` (``ISOTOPE > 0``) holds the isotope-scattering linewidths.
* ``/iterativebte`` (``SOLVER = IBTE``): the result of the iterative solution of the Boltzmann equation.
* ``/metadata``: temperatures, the primitive cell, smearing settings, and the input variables.

The notebook `Analysing PREFIX.kappa.h5 <https://github.com/alamode-code/alamode/blob/2.0dev/example/notebooks/kappa_h5_analysis.ipynb>`__
reads this file and rebuilds the thermal conductivity from the lifetimes and velocities in it.

When the harmonic force constants depend on temperature (``FC2_TEMPERATURE`` with an SCPH or QHA result),
the file is *temperature resolved* (root attribute ``temperature_resolved = 1``):
frequencies and velocities gain a leading temperature axis, and runs at different temperatures accumulate in one file.

.. _label_hdf5_scph:

SCPH and QHA results (PREFIX.scph.h5, PREFIX.qha.h5)
----------------------------------------------------

Written by ``MODE = SCPH`` and ``MODE = QHA`` at the end of the run. A later run uses it through
``FC2_TEMPERATURE`` (with ``FCSFILE``, ``FC2FILE``, or ``DFC2FILE``) and ``RELAXED_STRUCTURE``, and to restart the calculation.

.. figure:: ../img/h5layout/h5_scph.*
   :align: center
   :alt: Layout of the SCPH and QHA result file
   :figclass: only-light

   Layout of ``PREFIX``.scph.h5 and ``PREFIX``.qha.h5 (schema ``alamode:scph_state``).

.. only:: html

   .. figure:: ../img/h5layout/h5_scph.dark.*
      :align: center
      :alt: Layout of the SCPH and QHA result file
      :figclass: only-dark

      Layout of ``PREFIX``.scph.h5 and ``PREFIX``.qha.h5 (schema ``alamode:scph_state``).


* ``/ForceConstants/Order2``: the harmonic force constants folded onto the supercell of ``KMESH_INTERPOLATE``;
  ``Order2_temperature_dependent/force_constant_values`` holds the effective (renormalized) values at every temperature
  and uses the index datasets of ``Order2``.
* ``/convergence``: whether the SCPH iteration (``scph``) and the structural optimization (``structure``) converged at each temperature.
  Later runs refuse unconverged temperatures unless ``ALLOW_UNCONVERGED`` is set.
* ``/structure`` (``RELAX_STR > 0``): the relaxed structure at each temperature: the displacement gradient ``u_tensor``,
  the atomic displacements ``u0`` (bohr), and the space group ``spg_label``.
  ``/provenance`` records a fingerprint of the force constants used, which ``RELAXED_STRUCTURE`` checks.
* ``/dymat``, ``V0``: the change of the dynamical matrix and the potential energy used to restart the calculation.
* ``/settings``: the temperatures and meshes of the run. The root attribute ``mode`` is ``SCPH`` or ``QHA``.

.. _label_hdf5_selfenergy:

Self-energy (PREFIX.selfenergy.h5)
----------------------------------

Written by ``MODE = selfenergy``, with one group per target mode.

.. figure:: ../img/h5layout/h5_selfenergy.*
   :align: center
   :alt: Layout of the self-energy file
   :figclass: only-light

   Layout of ``PREFIX``.selfenergy.h5.

.. only:: html

   .. figure:: ../img/h5layout/h5_selfenergy.dark.*
      :align: center
      :alt: Layout of the self-energy file
      :figclass: only-dark

      Layout of ``PREFIX``.selfenergy.h5.


* ``/targets/00001``, ``00002``, ...: one group per target (wave vector × branch), in the order of the ``&kpoint`` field and ``BRANCHES``,
  with the wave vector ``xk``, the ``branch``, the harmonic ``frequency``, and the requested results:
  ``linewidth`` (the full width :math:`2\Gamma`, per temperature), the frequency shifts ``shift_*``,
  the frequency-dependent self-energy ``self_*``, and the final-state analysis ``fstate_*``.
* ``/path``: the path coordinates when the targets lie on a band path (``KPMODE = 1``).
* ``/spectrum`` (``INTERPOLATE = 1``): the spectral function.
* ``/metadata``: temperatures, the integration mesh, and the input variables.

.. note::
   The linewidth in this file is the full width :math:`2\Gamma`,
   while ``gamma`` in ``PREFIX``.kappa.h5 is :math:`\Gamma`.

Strain couplings (PREFIX.strain.h5)
-----------------------------------

The file given as ``STRAINFILE`` for cell relaxation is made by the strain tools and is described in :ref:`label_strain_container`.
