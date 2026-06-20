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
 Created when ``MODE = phonons`` with **KPMODE** = 1 and ``GRUNEISEN = 1``.


* ``PREFIX``.gru_all

 Gr\ |umulaut_u|\ neisen parameters of all phonon modes at the uniform :math:`k` grid.
 Created when ``MODE = phonons`` with **KPMODE** = 2 and ``GRUNEISEN = 1``.


* ``PREFIX``.zmode

 Mode effective charges of zone-center phonon modes.
 Created when ``MODE = phonons`` with ``ZMODE = 1``.

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

* ``PREFIX``.result

 In this file, phonon frequency, group velocity, and anharmonic phonon linewidths are printed.
 This file is updated during thermal conductivity calculations (``MODE = RTA``).
 In addition, this file is read when the restart mode is turned on (``RESTART = 1``).

* ``PREFIX``.kl

 Lattice thermal conductivity tensor (Peierls term). Created when ``MODE = RTA``.

* ``PREFIX``.kl_spec

 Spectra of lattice thermal conductivity. Only diagonal components are saved.
 Created when ``MODE = RTA`` and ``KAPPA_SPEC = 1``.


* ``PREFIX``.kl_coherent

 Coherent component of lattice thermal conductivity. Created when ``KAPPA_COHERENT > 0`` in ``MODE = RTA``.


* ``PREFIX``.kc_elem

 Momentum- and mode-decomposed contributions to the coherent components of lattice thermal conductivity. 
 Created when ``KAPPA_COHERENT = 2`` in ``MODE = RTA``.


* ``PREFIX``.self_isotope

 Phonon selfenergy due to isotope scatterings calculated by Tamura's formula.
 Created when ``MODE = RTA`` and ``ISOTOPE = 2``.

````

* ``PREFIX``.scph_dymat

 Anharmonic dynamical matrix calculated on the :math:`k` grid defined by the ``KMESH_INTERPOLATE`` tag.
 This file is used to restart the SCPH calculation.

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