.. raw:: html

    <script type="text/javascript"><!--
      function gen_mail_to_link(name, lhs,rhs)
      {
       document.write("<br>");
       document.write(name + " (<A HREF=\"mailto");
       document.write(":" + lhs + "@");
       document.write(rhs + "\">" + lhs + "@" + rhs + "<\/A>)"); } 
       // --> </SCRIPT>


About
=====

What is ALAMODE?
-----------------

**ALAMODE** is an open source software designed for analyzing lattice anharmonicity and lattice thermal conductivity of solids. By using an external DFT package such as VASP and Quantum ESPRESSO, you can extract harmonic and anharmonic force constants straightforwardly with ALAMODE. Using the calculated anharmonic force constants, you can also estimate lattice thermal conductivity, phonon linewidth, and other anharmonic phonon properties from first principles.

Features
--------

General
^^^^^^^

* Extraction of harmonic and anharmonic force constants based on the supercell approach
* Applicable to any crystal structures and low-dimensional systems
* Accurate treatment of translational and rotational invariance
* Interface to VASP, Quantum-ESPRESSO, OpenMX, xTAPP, and LAMMPS codes
* Mainly written in C++, parallelized with MPI+OpenMP

Harmonic properties
^^^^^^^^^^^^^^^^^^^
* Phonon dispersion
* Phonon DOS, atom-projected phonon DOS
* Two-phonon DOS
* Vibrational thermodynamic functions (heat capacity, entropy, free energy)
* Mean-square displacement
* Animation and visualization of phonon modes (requires VMD or XCrysDen)
* 3-phonon scattering phase space
* Phonon-isotope scattering rate
* Participation ratio for analyzing the localization of phonon modes

Anharmonic properties
^^^^^^^^^^^^^^^^^^^^^

.. |umulaut_u|    unicode:: U+00FC

* Gr\ |umulaut_u|\ neisen parameter via cubic force constants
* Lattice thermal conductivity by BTE-RTA
* Cumulative thermal conductivity
* Phonon linewidth due to 3-phonon interactions
* Phonon frequency shift due to 3- and 4-phonon interactions
* Temperature-dependent effective potential method
* Self-consistent phonon (SCPH) calculation
* Anharmonic vibrational free-energy
* Structural optimization at finite temperatures based on SCPH or QHA.

.. Note::
   The structural optimization of the current version CANNOT be used with thermal conductivity calculations or bubble corrections.
   The standard SCPH dispersion and thermodynamic properties are correctly calculated for relaxed configurations.

.. Note::
   Tutorials for structural optimization at finite temperatures are now included in this
   documentation; see :doc:`tutorial_pages/bto_scph_relax` (SCPH-based) and
   :doc:`tutorial_pages/zno_qha_relax` (QHA-based).

Links
-----

* Download page  : https://sourceforge.net/projects/alamode
* Documentation  : https://alamode.readthedocs.io (this page)
* Git repository : https://github.com/ttadano/alamode


License
-------

.. |copy|   unicode:: U+000A9 

Copyright |copy| 2014-2026 Terumasa Tadano

This software is distributed under the MIT license.
See the LICENSE.txt file for license rights and limitations. 


How to Cite ALAMODE
-------------------

Please cite the following article when you use ALAMODE:

.. list-table:: ALAMODE papers
    :header-rows: 1

    * - Reference number
      - Features
    * - [1]_
      - General citation paper for all features of ALAMODE
    * - [2]_
      - Self-consistent phonon (SCP) method (``MODE = SCPH``)
    * - [3]_
      - Anharmonic free energies within SCP or improved SCP (ISC)
    * - [4]_
      - SCP-based structural optimization or ``PERIODIC_IMAGE_CONV = 1`` (default) in the **alm** code
    * - [5]_
      - QHA-based structural optimization

References
^^^^^^^^^^

.. [1] T\. Tadano, Y. Gohda, and S. Tsuneyuki, J. Phys.: Condens. Matter **26**\ , 225402 (2014) 
  [`Link <http://iopscience.iop.org/0953-8984/26/22/225402/>`__].

.. [2] T\. Tadano and S. Tsuneyuki, Phys. Rev. B **92**\ , 054301 (2015). 
  [`Link <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.92.054301>`__]

.. [3] Y\. Oba, T. Tadano, R. Akashi, and S. Tsuneyuki, Phys. Rev. Materials **3**\, 033601 (2019). 
  [`Link <https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.3.033601>`__]

.. [4] R\. Masuki, T. Nomoto, R. Arita, and T. Tadano, Phys. Rev. B **106**\, 224104 (2022).
  [`Link <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.106.224104>`__]

.. [5] R\. Masuki, T. Nomoto, R. Arita, and T. Tadano, Phys. Rev. B **107**\, 134119 (2023).
  [`Link <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.107.134119>`__]


Issues & Bug report
-------------------

* If you find a bug or issue related to ALAMODE, please report it at `GitHub issues <https://github.com/ttadano/alamode/issues/>`_. 
* Other questions and suggestions can be posted on the `GitHub discussion page. <https://github.com/ttadano/alamode/discussions>`_

In either case, please search the previous questions and read FAQ page of this document before asking.

Acknowledgment
--------------

This project is/was partially supported by the following projects:

* Grant-in-Aid for Scientific Research (C) (21K03424)
* Grant-in-Aid for Young Scientists (B) (16K17724) 
* Grant-in-Aid for Scientific Research on Innovative Areas 'Materials Design through Computics: Complex Correlation and Non-Equilibrium Dynamics'. (http://computics-material.jp)


Author & Contact
----------------

.. raw:: html

    <script>gen_mail_to_link('Terumasa TADANO', 'terumasa.tadano','gmail.com')</script>

| Research Center for Magnetic and Spintronic Materials (CMSM),
| National Institute for Material Science (NIMS), 
| Japan


