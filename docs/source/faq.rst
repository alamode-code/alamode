.. _label_faq:

.. raw:: html

    <style> .red {color:red} </style>
    <style> .question {color:#cc0000; font-weight:bold} </style>


.. role:: red

.. role:: question


Frequently Asked Questions (FAQ)
================================

- :question:`The fitting error is very large (> 90%). Is it problematic? If so, how can I reduce the error?`
  
  A large fitting error can affect the accuracy of the force constants. The most likely cause is non-zero residual forces in the original supercell structure (before the atoms are displaced). Even when the structure of the primitive cell is optimized with a relatively strict convergence criterion, the atomic forces in the supercell generated from it may deviate from zero. To reduce the error associated with the residual forces, use the ``--offset`` option of :red:`extract.py` when generating the displacement-force datasets. For example, for the VASP calculator, issue
  ::

      $ python ${ALAMODE_ROOT}/tools/extract.py --VASP SPOSCAR --offset vasprun0.xml vasprun_harm*.xml > DFSET_harmonic

  Here, ``vasprun0.xml`` is the file obtained by running a VASP calculation for the original supercell (``SPOSCAR``).

  If the fitting error is still large after subtracting the offset components, please consider the following points:

  1. Give the fractional coordinates with ~15 significant digits; for example, 1/3 should be 0.33333333333333 rather than 0.33333.

  2. Check that the DFT calculations are properly converged.

  3. Use a smaller displacement magnitude.


- :question:`How small should the fitting error be?`

  It depends on the Taylor expansion potential and displacement magnitude you choose. 
  
  In a standard harmonic calculation where ``--mag=0.01`` is used in :red:`displace.py` and all harmonic interactions are considered (cutoff = None), the fitting error is usually less than 5% (~1--2% in most cases).

  In the calculation of third-order force constants with ``--mag=0.04``, the fitting error should be small as well. Indeed, in many cases, the fitting errors are much smaller (< 1%) than in the harmonic case.

  In the temperature-dependent effective potential method, the harmonic potential is fitted to displacement-force datasets sampled by ab initio molecular dynamics at finite temperature, so the fitting error tends to be much larger (> 10%).

 
- :question:`What value should I use for the cutoff radius?`

  For the harmonic term, we recommend ``None``, which includes all harmonic interactions inside the supercell. This choice hardly increases the computational cost because the number of displacement patterns does not change. Also, the harmonic dynamical matrix becomes exact at the commensurate q points only when ``None`` is selected. (Giving a very large cutoff radius has the same effect as giving ``None``.)

  For the anharmonic terms, increase the cutoff radii gradually and check the convergence of physical quantities, such as thermal conductivity and free energy, with respect to the cutoff values. In most cases, a cutoff radius of 10 Bohr is a good starting point, but a larger value may be necessary for polar materials.

- :question:`Why are the phonon dispersion curves discontinuous at the Brillouin zone boundaries?`

  This usually happens when the supercell lattice vectors are mistakenly given in the ``&cell`` field of the **anphon** code. In that case, :red:`use the primitive lattice vectors` for **anphon**.

  .. Note::

      For the ``&cell`` field of **alm**, you need to give the supercell lattice vectors.


