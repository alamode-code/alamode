Tutorial
========

.. The card grid is for the web pages; in the PDF the tutorials follow as sections.

.. only:: html

   .. grid:: 1 2 2 3
      :gutter: 3
      :class-container: tutorial-grid

      .. grid-item-card:: Silicon
         :link: tutorial_pages/silicon
         :link-type: doc
         :img-top: ../img/si_kappa.png
         :img-alt: Lattice thermal conductivity of Si

         Phonon dispersion, DOS, and lattice thermal conductivity of Si from DFT forces.

         +++
         :bdg-primary:`Si` :bdg-secondary:`harmonic + kappa`

      .. grid-item-card:: Silicon with LAMMPS
         :link: tutorial_pages/silicon_lammps
         :link-type: doc
         :img-top: ../img/Si_phband_DFT.png
         :img-alt: Phonon dispersion of Si

         Use LAMMPS with an empirical potential instead of DFT to compute the forces.

         +++
         :bdg-primary:`Si` :bdg-secondary:`LAMMPS forces`

      .. grid-item-card:: BAs: four-phonon scattering
         :link: tutorial_pages/bas_4ph
         :link-type: doc
         :img-top: ../img/BAs_kappa.png
         :img-alt: Lattice thermal conductivity of BAs with and without four-phonon scattering

         Add four-phonon scattering to the thermal conductivity of BAs and go beyond the RTA.

         +++
         :bdg-primary:`BAs` :bdg-secondary:`kappa + 4ph`

      .. grid-item-card:: PbTe (non-analytic correction)
         :link: tutorial_pages/pbte_nonanalytic_correction
         :link-type: doc
         :img-top: ../img/PbTe_harm_compare.png
         :img-alt: Phonon dispersion of PbTe with different NONANALYTIC options

         LO-TO splitting in polar materials from Born effective charges and the NONANALYTIC options.

         +++
         :bdg-primary:`PbTe` :bdg-secondary:`harmonic + NONANALYTIC`

      .. grid-item-card:: SrTiO\ :sub:`3`: self-consistent phonons
         :link: tutorial_pages/sto_scph
         :link-type: doc
         :img-top: ../img/STO_scph.png
         :img-alt: Self-consistent phonon dispersion of SrTiO3

         Stabilize soft modes at finite temperature and compute SCP thermodynamic functions.

         +++
         :bdg-primary:`SrTiO3` :bdg-secondary:`MODE = SCPH`

      .. grid-item-card:: BaTiO\ :sub:`3`: anharmonic IFCs
         :link: tutorial_pages/bto_ifc
         :link-type: doc
         :img-top: ../img/BTO_IFC_cv.png
         :img-alt: Cross-validation score of BaTiO3

         Fit anharmonic IFCs from AIMD-sampled configurations, with the regularization chosen by cross validation.

         +++
         :bdg-primary:`BaTiO3` :bdg-secondary:`anharmonic IFCs + CV`

      .. grid-item-card:: Si: anharmonic IFCs
         :link: tutorial_pages/silicon_ifc
         :link-type: doc
         :img-top: ../img/si222.png
         :img-alt: 2x2x2 supercell of Si

         Sample configurations from the harmonic potential energy surface and run the CV sets as separate jobs.

         +++
         :bdg-primary:`Si` :bdg-secondary:`anharmonic IFCs + CV`

      .. grid-item-card:: BaTiO\ :sub:`3`: SCPH structural optimization
         :link: tutorial_pages/bto_scph_relax
         :link-type: doc
         :img-top: ../img/BaTiO3_scph_relax.png
         :img-alt: Temperature dependence of atomic displacements in BaTiO3

         Follow the cubic-tetragonal phase transition through the temperature-dependent atomic positions.

         +++
         :bdg-primary:`BaTiO3` :bdg-secondary:`SCPH + RELAX_STR`

      .. grid-item-card:: ZnO: QHA structural optimization
         :link: tutorial_pages/zno_qha_relax
         :link-type: doc
         :img-top: ../img/ZnO_thermal_strain.png
         :img-alt: Thermal strain of ZnO

         Thermal expansion of wurtzite ZnO from the strain couplings within the QHA.

         +++
         :bdg-primary:`ZnO` :bdg-secondary:`QHA + RELAX_STR`

.. toctree::
   :maxdepth: 1
   :hidden:

   tutorial_pages/silicon
   tutorial_pages/silicon_lammps
   tutorial_pages/bas_4ph
   tutorial_pages/pbte_nonanalytic_correction
   tutorial_pages/sto_scph
   tutorial_pages/bto_ifc
   tutorial_pages/silicon_ifc
   tutorial_pages/bto_scph_relax
   tutorial_pages/zno_qha_relax
