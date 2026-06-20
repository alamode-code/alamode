.. _label_tutorial_silicon_ifc:

.. raw:: html

    <style> .red {color:red} </style>

.. role:: red

.. |Angstrom|   unicode:: U+00C5 

Si : Anharmonic interatomic force constants (IFCs)
---------------------------------------------------

This page explains another method to calculate anharmonic interatomic force constants (IFCs) using ALAMODE.

The example input files are provided in **example/Si/anharm_IFCs**.

Let us move to the example directory.

.. code-block:: bash

  $ cd ${ALAMODE_ROOT}/example/Si/anharm_IFCs

.. _tutorial_Si_IFC_step1:

1. Calculate the harmonic IFCs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, we calculate the harmonic IFCs in the considering supercell.

The procedure is the same as explained in :ref:`Tutorial 7.1 <label_tutorial_01>`, so we briefly describe the outline here.

We first calculate the displacement patterns by **alm**.

.. code-block:: bash

  $ cd ${ALAMODE_ROOT}/example/Si/anharm_IFCs/1_harmonic
  $ ${ALAMODE_ROOT}/alm/alm si_alm_sug.in > si_alm_sug.log

Then, generate the VASP input files using the generated ``si222_harmonic.pattern_HARMONIC`` 
and the input files in the **VASP_input** directory.
After running the VASP calculation and obtaining :red:`DFSET_harmonic`
(You can skip this part because :red:`DFSET_harmonic` is provided in this tutorial), 
calculate the harmonic IFCs with 

.. code-block:: bash

  $ ${ALAMODE_ROOT}/alm/alm si_alm_opt.in > si_alm_opt.log

.. _tutorial_Si_IFC_step2:

2. Generate the displacement-force data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In :ref:`Tutorial 7.5 <label_tutorial_bto_ifc>`, we used the AIMD calculation
to generate random configurations, from which we calculated the anharmonic IFCs.

However, the low-energy region of the potential energy surface (PES) can be sampled more efficiently 
for weakly anharmonic materials or materials without imaginary harmonic frequencies.
Here, we use the harmonic IFCs and perform the independent random sampling from the 
harmonic PES at a given temperature.

In preparation, we calculate the phonon frequencies and the polarization vectors at 
commensurate :math:`q`-points.

.. code-block:: bash

  $ cd ${ALAMODE_ROOT}/example/Si/anharm_IFCs/2_generate_config
  $ python3 ${ALAMODE_ROOT}/tools/displace.py --VASP POSCAR_supercell --prim POSCAR_primitive_cell --random_normalcoord >> "si_anphon.in"

The information on the :math:`q`-points are written out in :red:`si_anphon.in` as follows.

.. code-block::

  &general
    PREFIX = si222_harmonic
    MODE   = phonons
    FCSFILE = si222_harmonic.xml

    NKD = 1; KD = Si
  /

  *****************************************************************
      displace.py --  Generator of displaced configurations
                        Version. 1.2.1
  *****************************************************************

  Output format                  : VASP POSCAR
  Structure before displacements : POSCAR_supercell
  Output file names              : disp{counter}.POSCAR
  Magnitude of displacements     : 0.02 Angstrom
  Number of atoms                : 64

  The --evec option is necessary when '--random_normalcoord'
  option is used.
  Please generate a PREFIX.evec file by using the ANPHON code
  with the following inputs and then run displace.py again with
  --evec=PREFIX.evec option:

  &cell
  1.0
    0.000000000000000   5.131551292420093   5.131551292420093
    5.131551292420093   0.000000000000000   5.131551292420093
    5.131551292420093   5.131551292420093   0.000000000000000
  /
  &kpoint
  0
    0.000000000000000    0.000000000000000    0.000000000000000
    ...

Now, delete the unnecessary part of the output and run the **anphon** calculation.

.. code-block:: bash

  $ ${ALAMODE_ROOT}/anphon/anphon si_anphon.in > si_anphon.log

The calculated phonon frequencies and the polarization vectors are stored in :red:`si222_harmonic.evec`.

With these preparations, we can generate supercells with random displacements by 

.. code-block:: bash

  $ mkdir configurations
  $ cd configurations
  $ cp ../POSCAR_primitive_cell  ../POSCAR_supercell ../si222_harmonic.evec ./
  $ python3 ${ALAMODE_ROOT}/tools/displace.py --VASP POSCAR_supercell --prim POSCAR_primitive_cell --random_normalcoord --evec si222_harmonic.evec --temp 300 --prefix randomQ_ -nd 100

Here, we generated ``-nd 100`` configurations by randomly sampling from the distribution
at ``--temp 300`` K in the harmonic PES.

Please run the DFT calculation for each generated supercell
using the VASP input in **example/Si/anharm_IFCs/1_harmonic/VASP_input**.
Then, use **extract.py** to obtain :red:`DFSET_randomQ` using the procedure explained 
in :ref:`Tutorial 7.1 <tutorial_Si_step2>`.

.. note::
  
  The imaginary frequencies are replaced by their absolute values in the random sampling.
  Thus, the procedure can be performed for the strongly anharmonic materials as well.

  However, the user must be careful whether the generated set of random configurations
  is a good dataset for calculating IFCs.

.. _tutorial_Si_IFC_step3:

3. Cross validation (CV)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this step, we explain how to run the different sets of CV calculations separately.

The calculation of different sets can be executed in parallel because they are independent of each other.
So, if you have a cluster computer with multiple cores, you can run the calculations of each CV set 
in separate jobs.
The preparation of the input files is slightly complicated, but it will be time-saving
when the computational cost of the CV calculation is significant.

The input files are :red:`si_alm_cvset1.in` to :red:`si_alm_cvset4.in` in **example/Si/anharm_IFCs/3_cv**.

The essential parts of the input file :red:`si_alm_cvset1.in` are as follows.

We have ``NDATA = 100`` displacement-force data, and we will perform CV with 4 sets.
Thus, we want to use the first 25 data (``NSTART_CV = 1``, ``NEND_CV = 25``) 
in the validation process in the calculation of the first CV set (set1).
Note that these 25 sets have to be excluded in the training process (``SKIP = 1-25``)

The input files of the other CV sets are set accordingly.
It is important that we use different ``PREFIX`` for each set because 
the result of another CV set will overwrite the output file otherwise.

.. code-block::

  &general
    PREFIX = si222_cvset1
    ...  
  /

  ...
  &optimize
    ...
    NDATA = 100
  ...
    SKIP = 1-25
    NSTART_CV = 1
    NEND_CV = 25
  /

  ...


Run the calculation with 

.. code-block:: bash

  $ cd ${ALAMODE_ROOT}/example/Si/anharm_IFCs/3_cv
  $ ${ALAMODE_ROOT}/alm/alm si_alm_cvset1.in > si_alm_cvset1.log
  $ ${ALAMODE_ROOT}/alm/alm si_alm_cvset2.in > si_alm_cvset2.log
  $ ${ALAMODE_ROOT}/alm/alm si_alm_cvset3.in > si_alm_cvset3.log
  $ ${ALAMODE_ROOT}/alm/alm si_alm_cvset4.in > si_alm_cvset4.log

After all the calculations are finished, collect the cvscore data with 

.. code-block:: bash

  $ python3 cvscore.py *cvset > si222.cvscore


.. note::
  The number of :math:`\alpha` for which the calculation is performed can differ 
  depending on the CV sets because the calculation stops in the middle due to the ``STOP_CRITERION``-tag.
  If the calculations stop at different steps, the Python script stops 
  with an error of "Inconsistent number of entries".

  In that case, please manually adjust the cvset files so that the number of entries is consistent.


The optimal amplitude of regularization (:math:`\alpha`) can be read from the last line
of :red:`si222.cvscore`.

.. code-block:: 

  #Minimum cvscore at  2.25633e-07

.. _tutorial_Si_IFC_step4:

4. Calculation of IFCs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, we calculate the IFCs of silicon in **example/Si/anharm_IFCs/4_optimize**.

The input file is :red:`si_alm_opt.in`.
Set ``CV = 0`` and set the optimal :math:`\alpha` with ``L1_ALPHA = 2.25633e-07`` in ``&optimize``-field.

Run the calculation with 

.. code-block:: bash 

  $ cd ${ALAMODE_ROOT}/example/Si/anharm_IFCs/4_optimize
  $ ${ALAMODE_ROOT}/alm/alm si_alm_opt.in > si_alm_opt.log

The calculated IFCs are written out in :red:`si222.xml` and :red:`si222.fcs`.
The fitting error is 

.. code-block::

  RESIDUAL (%): 0.524303












