.. raw:: html

    <style> .red {color:red} </style>

.. role:: red

ALM: Input files 
----------------

.. _reference_input_alm:

Format of input files
~~~~~~~~~~~~~~~~~~~~~

Each input file should consist of entry fields.
Available entry fields are 

**&general**, **&interaction**, **&cutoff**, **&cell**, **&position**, and **&optimize**.


Each entry field starts from the key label **&field** and ends at the terminate character "/". (This is equivalent to Fortran namelist.) 

For example, &general entry field of program *alm* should be given like

::

  &general
    # Comment line
    PREFIX = prefix
    MODE = optimize
  /

Multiple entries can be put in a single line. Also, characters put on the right of sharp (“#”) are neglected. Therefore, the above example is equivalent to the following::
  
  &general
    PREFIX = prefix; MODE = optimize  # Comment line
  /

Each variable must be given inside the appropriate entry field.


.. _label_inputvar_alm:

List of supported input variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
   :widths: 20, 20, 20, 20

   **&general**
   :ref:`HESSIAN <alm_hessian>`, :ref:`FC3_SHENGBTE <alm_fc3_shengbte>`, :ref:`FCSYM_BASIS <alm_fcsym_basis>`, :ref:`FC_ZERO_THR <alm_fc_zero_thr>`
   :ref:`KD <alm_kd>`, :ref:`MAGMOM <alm_magmom>`, :ref:`MODE <alm_mode>`, :ref:`NAT <alm_nat>`
   :ref:`NKD <alm_nkd>`, :ref:`NMAXSAVE <alm_nmaxsave>`, :ref:`NONCOLLINEAR <alm_noncollinear>`, :ref:`PERIODIC <alm_periodic>`
   :ref:`PREFIX <alm_prefix>`, :ref:`PRINTSYM <alm_printsym>`, :ref:`TOLERANCE <alm_tolerance>`
   **&interaction**
   :ref:`NBODY <alm_nbody>`, :ref:`NORDER <alm_norder>`
   **&optimize**
   :ref:`CHUNKSIZE <alm_chunksize>`, :ref:`CONV_TOL <alm_conv_tol>`, :ref:`CV <alm_cv>`, :ref:`CV_MINALPHA <alm_cv_minalpha>`
   :ref:`DEBIAS_OLS <alm_debias_ols>`, :ref:`DFSET <alm_dfset>`, :ref:`DFSET_CV <alm_dfset_cv>`, :ref:`ENET_DNORM <alm_enet_dnorm>`
   :ref:`FC2FIX <alm_fc2fix>`, :ref:`FC3FIX <alm_fc3fix>`, :ref:`ICONST <alm_iconst>`
   :ref:`L1_ALPHA <alm_l1_alpha>`, :ref:`L1_RATIO <alm_l1_ratio>`, :ref:`L1_SOLVER <alm_l1_solver>`, :ref:`LMODEL <alm_lmodel>`
   :ref:`MAXITER <alm_maxiter>`, :ref:`NDATA <alm_ndata>`, :ref:`NDATA_CV <alm_ndata_cv>`, :ref:`NSTART NEND <alm_nstart>`
   :ref:`NSTART_CV NEND_CV <alm_nstart_cv>`, :ref:`PERIODIC_IMAGE_CONV<alm_periodic_image_conv>`, :ref:`ROTAXIS <alm_rotaxis>`, :ref:`SKIP <alm_skip>`
   :ref:`SOLUTION_PATH <alm_solution_path>`, :ref:`SPARSE <alm_sparse>`
   :ref:`SPARSESOLVER <alm_sparsesolver>`, :ref:`STANDARDIZE <alm_standardize>`, :ref:`STOP_CRITERION <alm_stop_criterion>`
   :ref:`USE_CHOLESKY <alm_use_cholesky>`


Description of input variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"&general"-field
++++++++++++++++

.. _alm_prefix:

* **PREFIX**-tag : Job prefix to be used for names of output files

 :Default:  None
 :Type: String

````

.. _alm_mode:

* **MODE**-tag = optimize | suggest 

 =================================== ====================================================================
  optimize (:red:`>= 1.1.0`)         | Estimate harmonic and anharmonic IFCs. 
                                     | This mode requires an appropriate &optimize field.

  suggest                            | Suggests the displacement patterns necessary
                                     | to estimate harmonic and anharmonic IFCs.
 =================================== ====================================================================

 :Default: None
 :Type: String

````

.. _alm_nat:

* **NAT**-tag : Number of atoms in the supercell

 :Default: None
 :Type: Integer

````

.. _alm_nkd:

* **NKD**-tag : Number of atomic species

 :Default: None
 :Type: Integer

````

.. _alm_kd:

* **KD**-tag = Name[1], ... , Name[``NKD``]

 :Default: None
 :Type: Array of strings
 :Example: In the case of GaAs with ``NKD = 2``, it should be ``KD = Ga As``.

````

.. _alm_tolerance:

* TOLERANCE-tag : Tolerance for finding symmetry operations
  
 :Default: 1.0e-3
 :Type: Double

````

.. _alm_printsym:

* PRINTSYM-tag = 0 | 1

 === ====================================================
  0   Symmetry operations won’t be saved in “SYMM_INFO”
  1   Symmetry operations will be saved in “SYMM_INFO”
 === ====================================================

 :Default: 0
 :type: Integer

````

.. _alm_fcsym_basis:

* FCSYM_BASIS-tag = Cartesian | Lattice

 ============== ===========================================================================
  Cartesian, C   Symmetry reduction of force constant is performed in the Cartesian basis

  Lattice, L     Symmetry reduction of force constant is performed in the :math:`\boldsymbol{a}_1, \boldsymbol{a}_2, \boldsymbol{a}_3` basis
 ============== ===========================================================================

 :Default: Lattice
 :type: String
 :Description: The calculation results should not depend on the choice of ``FCSYM_BASIS`` when ``LMODEL = ols``. For other regression methods (enet, adaptive LASSO), an optimal value of the ``L1_ALPHA`` changes when you change the ``FCSYM_BASIS`` option.  
 
    In some cases, ``FCSYM_BASIS = Lattice`` is more stable and efficient. In particular, we recommend setting ``FCSYM_BASIS = Lattice`` for hexagonal systems. If a calculation with ``FCSYM_BASIS = Lattice`` is slow, please switch to ``FCSYM_BASIS = Cartesian``.
    
    For more details about the symmetry reduction of force constants, please see :ref:`here <IFC_crystal_symmetry>`.

 .. important::

     When ``FCSYM_BASIS = Lattice``, the basis of force constants saved in ``PREFIX``.fcs becomes the :math:`\boldsymbol{a}_1, \boldsymbol{a}_2, \boldsymbol{a}_3` basis. Hence, to compare the values of force constants saved in ``PREFIX``.fcs, you will have to change their basis to the Cartesian basis manually. The basis of force constants saved in ``PREFIX``.xml is Cartesian irrespective of the ``FCSYM_BASIS`` value.

     Also, imposing the constraints for rotational invariance with ``FCSYM_BASIS = Lattice`` is not supported. Therefore, if you want to apply the constraints for rotational invariance, please use ``FCSYM_BASIS = Cartesian``.

````

.. _alm_magmom:

* MAGMOM-tag : List of magnetic moments

 :Default: 0 ... 0 (``NAT`` entries when ``NONCOLLINEAR = 0``, 3x\ ``NAT`` entries when ``NONCOLLINEAR = 1``.)
 :type: Array of double
 :Example: When a supercell containts 64 atoms and the local magnetic moments of the first 32 atoms are up and those of the last 32 atoms are down, please set the ``MAGMOM`` tag as ``MAGMOM = 32*1 32*-1``. The wildcard (``*``) is available when ``NONCOLLINEAR = 0``. For the noncollinear case (``NONCOLLINEAR = 1``), the wildcard is not supported. So, please give the magnetic moment explicitly as ``MAGMOM = 0 0 1 0 0 1 0 0 1 ... 0 0 -1 0 0 -1 ...`` (3\ :math:`\times`\ ``NAT`` entries in *one line*).

 .. note::

     ``MAGMOM`` information is used only for generating space group operations. So, the values of the magnetic moment are somewhat arbitrary. For the 4\ :math:`\times` 4\ :math:`\times` 4 supercell of ferromagnetic bcc Fe (64 atoms), for instance, ``MAGMOM = 64*1`` and ``MAGMOM = 64*2`` give the same results. By contrast, ``MAGMOM = 32*1 32*2`` of course gives a different result because it breaks the symmetry of the original lattice.

````

.. _alm_noncollinear:

* NONCOLLINEAR-tag = 0 | 1

 :Default: 0 
 :type: Integer
 :Description: When ``NONCOLLINEAR = 1``, the code accepts a noncollinear magnetic structure as an input to the ``MAGMOM`` tag and uses it for generating space group operations. The spin quantization axis is fixed to the (0,0,1) direction of the Cartesian axis.

 .. caution::

     Still experimental. Please use with care.

````

.. _alm_periodic:

* PERIODIC-tag = PERIODIC[1], PERIODIC[2], PERIODIC[3] 

 ===== ====================================================
   0   | Do not consider periodic boundary conditions when
       | searching for interacting atoms.

   1   | Consider periodic boundary conditions when
       | searching for interacting atoms.
 ===== ====================================================

 :Default: 1 1 1
 :type: Array of integers
 :Description: This tag is useful for generating interacting atoms in low dimensional systems. When ``PERIODIC[i]`` is zero, periodic boundary condition is turned off along the direction of the lattice vector :math:`\boldsymbol{a}_{i}`.

````

.. _alm_nmaxsave:

* NMAXSAVE-tag : The maximum order of anharmonic force constants printed out in ``PREFIX``.xml

 :Default: min(5, ``NORDER``) 
 :Type: Integer
 :Example: If your model includes anharmonic terms up to the sixth-order (``NORDER = 5``), but you want to avoid printing out the fifth-order and sixth-order IFCs in ``PREFIX``.xml, please set ``NMAXSAVE = 3``.

````

.. _alm_hessian:

* HESSIAN-tag = 0 | 1

 ===== =====================================================================
   0    Do not save the Hessian matrix
   1    Save the entire Hessian matrix of the supercell as PREFIX.hessian.
 ===== =====================================================================

 :Default: 0
 :type: Integer

````

.. .. _alm_fc2_qefc:

.. * FC2_QEFC-tag = 0 | 1

..  ===== =====================================================================
..    0   | Do not save the second-order force constants in .fc format
..    1   | Save the second-order force constants in the Quantum ESPRESSO
..          .fc format in PREFIX.fc.
..  ===== =====================================================================

..  :Default: 0
..  :type: Integer

.. ````

.. _alm_fc3_shengbte:

* FC3_SHENGBTE-tag = 0 | 1

 ===== ==========================================================================================
   0   | Do not save the third-order force constants for ShengBTE code
   1   | Save the third-order force constants for the ShengBTE code in PREFIX.FORCE_CONSTANT_3RD.
 ===== ==========================================================================================

 :Default: 0
 :type: Integer

````

.. _alm_fc_zero_thr:

* FC_ZERO_THR-tag : Threshold value used when trimming force constants when creating PREFIX.xml
  
 :Default: 1.0e-12
 :Type: Double
 :Description: ``FC_ZERO_THR`` defines the threshold of force constants to be saved in an XML file. If the absolute value of a force constant is smaller than ``FC_ZERO_THR``, it will NOT be printed out.

 .. note::
    If the harmonic force constants are calculated using a model potential (e.g., classical FF) where the interaction becomes zero beyond a certain cutoff radius, the default value of ``FC_ZERO_THR`` may raise a warning when creating a renormalized harmonic force constant file using ``tools/dfc2``. This issue may be resolved by using a smaller ``FC_ZERO_THR``, say ``FC_ZERO_THR = 1.0e-15``. The force constants that become exactly zero due to symmetry and acoustic sum rule constraints will not be printed even when setting ``FC_ZERO_THR = 0``.

````

"&interaction"-field
++++++++++++++++++++

.. _alm_norder:


* **NORDER**-tag : The order of force constants to be calculated. Anharmonic terms up to :math:`(m+1)`\ th order will be considered with ``NORDER`` = :math:`m`.

 :Default: None
 :Type: Integer
 :Example: ``NORDER = 1`` to calculate harmonic terms only, ``NORDER = 2`` to include cubic terms as well, and so on.

````

.. _alm_nbody:


* NBODY-tag : Entry for excluding multiple-body interactions from anharmonic force constants
 
 :Default: ``NBODY`` = [2, 3, 4, ..., ``NORDER`` + 1]
 :Type: Array of integers
 :Description: This tag may be useful for excluding multi-body clusters which are supposedly less important. For example, a set of fourth-order IFCs :math:`\{\Phi_{ijkl}\}`, where :math:`i, j, k`, and :math:`l` label atoms in the supercell, can be categorized into four different subsets; **on-site**, **two-body**, **three-body**, and **four-body** terms. Neglecting the Cartesian coordinates of IFCs for simplicity, each subset contains the IFC elements shown as follows:

    =========== =========================================================================
     on-site    | :math:`\{\Phi_{iiii}\}`
     two-body   | :math:`\{\Phi_{iijj}\}`, :math:`\{\Phi_{iiij}\}` (:math:`i\neq j`)
     three-body | :math:`\{\Phi_{iijk}\}` (:math:`i\neq j, i\neq k, j \neq k`)
     four-body  | :math:`\{\Phi_{ijkl}\}` (all subscripts are different from each other)
    =========== =========================================================================    

    Since the four-body clusters are expected to be less important than the three-body and less-body clusters, you may want to exclude the four-body terms from the Taylor expansion potential because the number of such terms is huge. This can be done by setting the ``NBODY`` tag as ``NBODY = 2 3 3`` together with ``NORDER = 3``.

 :More examples: ``NORDER = 2; NBODY = 2 2`` includes harmonic and cubic IFCs but excludes three-body clusters from the cubic terms.

                 ``NORDER = 5; NBODY = 2 3 3 2 2`` includes anharmonic terms up to the sixth-order, where the four-body clusters are excluded from the fourth-order IFCs, and the multi (:math:`\geq 3`)-body clusters are excluded from the fifth- and sixth-order IFCs.

````

"&cutoff"-field
+++++++++++++++

In this entry field, one needs to specify cutoff radii of interaction for each order in units of bohr. 
In the current implementation, cutoff radii should be defined for every possible pair of atomic elements. 
For example, the cutoff entry for a harmonic calculation (``NORDER = 1``) of Si (``NKD = 1``) should be like
::

 &cutoff
  Si-Si 10.0
 /

This means that the cutoff radius of 10 :math:`a_{0}` is used for harmonic Si-Si terms. 
Please note that the first column should be two character strings, which are contained in the ``KD``-tag, 
connected by a hyphen (’-’). 

When one wants to consider cubic terms (``NORDER = 2``), please specify the cutoff radius for cubic terms in the third column as the following::

 
 &cutoff
  Si-Si 10.0 5.6 # Pair r_{2} r_{3}
 /

Instead of giving specific cutoff radii, one can write "None" as follows::

 &cutoff
  Si-Si None 5.6
 /

which means that all possible harmonic terms between Si-Si atoms will be included. 

.. caution::

  Setting 'None' for anharmonic terms can greatly increase the number of parameters and thereby increase the computational cost.

When there are more than two atomic elements, please specify the cutoff radii between every possible pair of atomic elements. In the case of MgO (``NKD = 2``), the cutoff entry should be like
::
 
 &cutoff
  Mg-Mg 8.0
  O-O 8.0
  Mg-O 10.0
 /

which can equivalently be written by using the wild card (’*’) as
::

 &cutoff
  *-* 8.0
  Mg-O 10.0 # Overwrite the cutoff radius for Mg-O harmonic interactions
 /

.. important::

  Cutoff radii specified by an earlier entry are overwritten by a new entry that comes later.

Once the cutoff radii are properly given, harmonic force constants
:math:`\Phi_{i,j}^{\mu,\nu}` satisfying :math:`r_{ij} \le r_{c}^{\mathrm{KD}[i]-\mathrm{KD}[j]}` will be searched.

In the case of cubic terms, force constants :math:`\Phi_{ijk}^{\mu\nu\lambda}` satisfying :math:`r_{ij} \le r_{c}^{\mathrm{KD}[i]-\mathrm{KD}[j]}`, :math:`r_{ik} \le r_{c}^{\mathrm{KD}[i]-\mathrm{KD}[k]}`, and
:math:`r_{jk} \le r_{c}^{\mathrm{KD}[j]-\mathrm{KD}[k]}` will be searched and determined by fitting.

````

"&cell"-field
+++++++++++++

Please give the cell parameters in this entry in units of bohr as the following::

 &cell
  a
  a11 a12 a13
  a21 a22 a23
  a31 a32 a33
 /

The cell parameters are then given by :math:`\vec{a}_{1} = a \times (a_{11}, a_{12}, a_{13})`,
:math:`\vec{a}_{2} = a \times (a_{21}, a_{22}, a_{23})`, and :math:`\vec{a}_{3} = a \times (a_{31}, a_{32}, a_{33})`.

````

"&position"-field
+++++++++++++++++

In this field, one needs to specify the atomic element and fractional coordinate of atoms in the supercell. 
Each line should be
::

  ikd xf[1] xf[2] xf[3]

where `ikd` is an integer specifying the atomic element (`ikd` = 1, ..., ``NKD``) and `xf[i]` is the
fractional coordinate of an atom. There should be ``NAT`` such lines in the &position entry field.


````

"&optimize"-field 
++++++++++++++++++

This field is necessary when ``MODE = optimize``.

.. _alm_lmodel:

* LMODEL-tag : Choice of the linear model used for estimating force constants

 =================================== ==========================
   "least-squares", "LS", "OLS",  1    Ordinary least square
   "elastic-net", "enet", 2            Elastic net
   "adaptive-lasso", 3                 Adaptive LASSO
 =================================== ==========================

 :Default: least-squares
 :Type: String
 :Description: When ``LMODEL = ols``, the force constants are estimated from the displacement-force datasets via the ordinary least-squares (OLS), which is usually sufficient to calculate harmonic and third-order force constants. 

               The elastic net (``LMODEL = enet``) or adaptive LASSO (``LMODEL = adaptive-lasso``) are useful for calculating fourth-order (and higher-order) force constants. When the elastic net or adaptive LASSO is selected, the users have to set the following related tags: ``CV``, ``L1_RATIO``, ``L1_ALPHA``, ``CV_MAXALPHA``, ``CV_MINALPHA``, ``CV_NALPHA``, ``STANDARDIZE``, ``ENET_DNORM``, ``MAXITER``, ``CONV_TOL``, ``NWRITE``, ``SOLUTION_PATH``, ``DEBIAS_OLS``, ``STOP_CRITERION``. Please note that ``STANDARDIZE`` will be effective only for the elastic net.

````

.. _alm_dfset:

* **DFSET**-tag: File name containing displacement-force datasets for training

 .. versionadded:: 1.1.0

 :Default: None
 :Type: String
 :Description: The format of ``DFSET`` can be found :ref:`here <label_format_DFSET>`

````

.. _alm_ndata:

* NDATA-tag : Number of displacement-force data sets

 :Default: None
 :Type: Integer
 :Description: If ``NDATA`` is not given, the code reads all lines of ``DFSET`` (excluding comment lines) and estimates ``NDATA`` by dividing the line number by ``NAT``. If the number of lines is not divisible by ``NAT``, an error is raised. ``DFSET`` should contain at least ``NDATA``:math:`\times` ``NAT`` lines.

````

.. _alm_nstart:

* NSTART, NEND-tags : Specifies the range of data to be used for training

 :Default: ``NSTART = 1``, ``NEND = NDATA``
 :Type: Integer
 :Example: To use the data in the range of [20:30] out of 50 entries, the tags should be ``NSTART = 20`` and ``NEND = 30``.

````

.. _alm_skip:

* SKIP-tag : Specifies the range of data to be skipped for training

 :Default: None
 :Type: Two integers connected by a hyphen
 :Description: ``SKIP`` =\ :math:`i`-:math:`j` skips the data in the range of [:math:`i`:\ :math:`j`]. The :math:`i` and :math:`j` must satisfy :math:`1\leq i \leq j \leq` ``NDATA``.  This option may be useful when doing cross-validation manually (``CV=-1``).

````

.. _alm_iconst:

* ICONST-tag = 0 | 1 | 2 | 3 | 4 | 11

 ===== =============================================================================================
   0    No constraints
   1   | Constraint for translational invariance is imposed between IFCs.
       | Available only when ``LMODEL = ols``.
  11   | Same as ``ICONST = 1`` but the constraint is imposed *algebraically* rather than numerically.
       | Select this option when ``LMODEL = enet``.
   2   | In addition to ``ICONST = 1``, constraints for rotational invariance will be
       | imposed up to (``NORDER`` + 1)th order. Available only when ``LMODEL = ols``.
   3   | In addition to ``ICONST = 2``, constraints for rotational invariance between (``NORDER`` + 1)th order
       | and (``NORDER`` + 2)th order, which are zero, will be considered.
       | Available only when ``LMODEL = ols``.
   4   | In addition to ``ICONST = 2``, constraints for the Huang invariance will be imposed.
       | Available only when ``LMODEL = ols``.
 ===== =============================================================================================

 :Default: 11
 :Type: Integer
 :Description: See :ref:`this page<constraint_IFC>` for the numerical formulae.

````

.. _alm_periodic_image_conv: 

* PERIODIC_IMAGE_CONV-tag = 0 | 1

 ===== =============================================================================================
   0    Impose the constraints on IFCs (acoustic sum rule) in the considering supercell.
   1    | Consider the periodic images when generating the constraints.
        | The resultant IFCs simultaneously satisfy the permutation symmetry, ASR, 
        | and the space group symmetry in the infinite space.
        | For more details, please see Appendix D of the `original paper <https://arxiv.org/abs/2205.08789>`_.
        | (Note that we use the term "mirror image" instead of "periodic image" in the paper.) 
 ===== =============================================================================================

 :Default: 1
 :Type: Integer

````

.. _alm_rotaxis:


* ROTAXIS-tag : Rotation axis used to estimate constraints for rotational invariance. This entry is necessary when ``ICONST = 2, 3, 4``.

 :Default: None
 :Type: String
 :Example: When one wants to consider the rotational invariance around the :math:`x`\ -axis, one should give ``ROTAXIS = x``. If one needs additional constraints for the rotation around the :math:`y`\ -axis, ``ROTAXIS`` should be ``ROTAXIS = xy``. 

````

.. _alm_fc2fix:

* FC2FIX-tag : File to which the harmonic terms are fixed upon training

 :Default: None
 :Type: String
 :Description: When the ``FC2FIX``-tag is given, harmonic force constants are fixed to the values stored in the ``FC2FIX`` file. This may be useful for optimizing cubic and higher-order terms without changing the harmonic terms. Please make sure that the number of harmonic terms in the new computational condition is the same as that in the ``FC2FIX`` file.

 .. note::

     ``FC2XML`` is a deprecated alias of ``FC2FIX`` kept for backward compatibility. It will be removed in a future release, so please use ``FC2FIX`` instead.

 .. important::

     The ``FCSYM_BASIS`` option must be the same as the one used when creating the reference harmonic force constant file (``FC2FIX``). The code raises an error when they are inconsistent.

````

.. _alm_fc3fix:

* FC3FIX-tag : File to which the cubic terms are fixed upon training

 :Default: None
 :Type: String
 :Description: Same as the ``FC2FIX``-tag, but ``FC3FIX`` is used to fix cubic force constants. (``FC3XML`` is a deprecated alias of ``FC3FIX``.)

 .. important::

     The ``FCSYM_BASIS`` option must be the same as the one used when creating the reference cubic force constant file (``FC3FIX``). The code raises an error when they are inconsistent.

````


.. _alm_sparse:

* SPARSE-tag = 0 | 1

 ===== ==============================================================
   0    Use a direct solver (SVD or QRD) to estimate force constants
   1    Use a sparse solver to estimate force constants
 ===== ==============================================================

 :Default: 0
 :Type: Integer
 :Description: When you want to calculate force constants of a large system and generate training datasets by displacing only a few atoms from equilibrium positions, the resulting sensing matrix becomes large but sparse. For such matrices, a sparse solver is expected to be more efficient than SVD or QRD in terms of both memory usage and computational time. When ``SPARSE = 1`` is set, the code uses a sparse solver implemented in Eigen3 library. You can change the solver type via ``SPARSESOLVER``. Effective when ``LMODEL = ols``.

              **When to use:** prefer ``SPARSE = 1`` for large systems whose sensing matrix is genuinely sparse (only a handful of atoms displaced per configuration). For small or dense problems the default direct solvers (``SPARSE = 0``) are usually faster, and the sparse factorization can even use more memory because of fill-in. When the sensing matrix is dense but large, consider :ref:`USE_CHOLESKY <alm_use_cholesky>` instead.

 .. note::

     With constraints imposed numerically (``ICONST = 1, 2, 3``), the sparse path assembles and factorizes the KKT system. The factorization backend is chosen at build time: Intel MKL **PARDISO** (``-DUSE_MKL_BACKEND=yes``) or Apple **Accelerate** (``-DUSE_ACCEL_BACKEND=yes``), falling back to Eigen's ``SimplicialLDLT`` otherwise. PARDISO is a parallel sparse direct solver and is the fastest option for large constrained sparse fits; see :ref:`Optional linear-algebra backends <install_backends>` on the installation page.

````

.. _alm_sparsesolver:

* SPARSESOLVER-tag : Type of the sparse solver to use

 :Default: SimplicialLDLT
 :Type: String
 :Description: Selects the Eigen3 sparse solver used for the unconstrained / algebraically constrained sparse fit. Available options are `SimplicialLDLT` (sparse Cholesky, direct), `SparseQR` (direct, most robust for ill-conditioned or rank-deficient matrices), and the iterative Krylov solvers `ConjugateGradient`, `LeastSquaresConjugateGradient`, and `BiCGSTAB`. When an iterative solver is selected, the stopping criterion is controlled by the ``CONV_TOL`` and ``MAXITER`` tags. As a rule of thumb, use `SimplicialLDLT` (the default) for well-conditioned problems, `SparseQR` when the direct factorization fails or the matrix is poorly conditioned, and the iterative solvers when even the sparse factorization is too large to store. Effective when ``LMODEL = ols`` and ``SPARSE = 1``. Note that these Eigen solvers are independent of the PARDISO/Accelerate KKT backend described under ``SPARSE``.


 .. seealso::
    Eigen documentation page: `Solving Sparse Linear Systems <https://eigen.tuxfamily.org/dox/group__TopicSparseSystems.html>`__

````

.. _alm_use_cholesky:

* USE_CHOLESKY-tag = 0 | 1

 ===== ==============================================================
   0    Use a direct solver (SVD or QRD) to estimate force constants
   1    Solve the normal equations by Cholesky decomposition
 ===== ==============================================================

 :Default: 0
 :Type: Integer
 :Description: When ``USE_CHOLESKY = 1``, the ordinary least-squares fit is obtained by forming the
              normal equations :math:`A^{\mathsf{T}} A\,\boldsymbol{\Phi} = A^{\mathsf{T}} \boldsymbol{b}`
              and factorizing :math:`A^{\mathsf{T}} A` with a Cholesky decomposition (a dense ``LLT``,
              or a sparse ``SimplicialLDLT`` when ``SPARSE = 1``). This is typically about twice as fast
              and uses noticeably less memory than the default SVD/QRD direct solvers, which makes it
              attractive for large, **well-conditioned** overdetermined fits (many more rows than
              columns). The trade-off is accuracy: forming :math:`A^{\mathsf{T}} A` squares the
              condition number of :math:`A`, so the solution degrades when the sensing matrix is
              ill-conditioned. Keep the default ``USE_CHOLESKY = 0`` (SVD/QRD) when robustness matters
              or the fit is close to rank-deficient. Effective when ``LMODEL = ols``.

 .. important::

     ``USE_CHOLESKY = 1`` cannot be combined with the numerically imposed constraints
     ``ICONST = 1, 2, 3``. When constraints are required, impose them algebraically with
     ``ICONST = 11`` instead; otherwise the code stops with an error.

````

.. _alm_chunksize:

* CHUNKSIZE-tag : Number of configurations processed per pass when building :math:`A^{\mathsf{T}} A`

 :Default: 100
 :Type: Integer
 :Description: Relevant only when ``USE_CHOLESKY = 1``. Instead of forming the full dense sensing
              matrix :math:`A` (size :math:`N_{\mathrm{rows}} \times N_{\mathrm{cols}}`) at once, the
              normal-equations matrix :math:`A^{\mathsf{T}} A` is accumulated in chunks of
              ``CHUNKSIZE`` training configurations: for each chunk only a small block of :math:`A` is
              built and added into :math:`A^{\mathsf{T}} A`. This caps the peak memory at roughly the
              larger of :math:`A^{\mathsf{T}} A` itself and one chunk's block (proportional to
              ``CHUNKSIZE`` :math:`\times\, N_{\mathrm{cols}}`), rather than the full :math:`A`.
              A smaller ``CHUNKSIZE`` lowers the peak memory but uses more passes with smaller, less
              efficient matrix products; a larger ``CHUNKSIZE`` is faster but needs more memory.
              Lowering it below the point where one chunk is already smaller than
              :math:`A^{\mathsf{T}} A` gives no further savings, since :math:`A^{\mathsf{T}} A` sets
              the memory floor. The default (100) is a reasonable balance; reduce it if the chunk
              block does not fit in memory.

````

.. _alm_maxiter:

* MAXITER-tag : Number of maximum iterations in iterative algorithms

 :Default: 10,000
 :Type: Integer
 :Description: Effective when an iterative solver is selected via ``SPARSESOLVER`` (when ``LMODEL = ols``) or when ``LMODEL = enet | adaptive-lasso``.

````

.. _alm_conv_tol:

* CONV_TOL-tag : Convergence criterion of iterative algorithms

 :Default: 1.0e-8
 :Type: Double
 :Description: When ``LMODEL = ols`` and an iterative solver is selected via ``SPARSESOLVER``, ``CONV_TOL`` value is passed to the Eigen3 function via `setTolerance()`.
               When ``LMODEL = enet | adaptive-lasso``, the L1 solver iteration stops at :math:`i`\ th iteration if :math:`\sqrt{\frac{1}{N}|\boldsymbol{\Phi}_{i} - \boldsymbol{\Phi}_{i-1}|_{2}^{2}} <` ``CONV_TOL`` is satisfied, where :math:`N` is the length of the vector :math:`\boldsymbol{\Phi}`.


 .. seealso::
    Eigen documentation page: `IterativeSolverBase <https://eigen.tuxfamily.org/dox/classEigen_1_1IterativeSolverBase.html>`__

````

.. _alm_l1_ratio:

* L1_RATIO-tag : The ratio of the L1 regularization term

 :Default: 1.0 (LASSO)
 :Type: Double
 :Description: The ``L1_RATIO`` changes the regularization term as ``L1_ALPHA`` :math:`\times` [``L1_RATIO`` :math:`|\boldsymbol{\Phi}|_{1}` + :math:`\frac{1}{2}` (1-``L1_RATIO``) :math:`|\boldsymbol{\Phi}|_{2}^{2}`]. Therefore, ``L1_RATIO = 1`` corresponds to LASSO. ``L1_RATIO`` must be ``0 < L1_ratio <= 1``. Effective when ``LMODEL = enet``. See also :ref:`here <alm_theory_enet>`.

````

.. _alm_l1_solver:

* L1_SOLVER-tag = cd | fista | admm

 ====== =============================================================================================
  cd     Use the coordinate descent solver.
  fista  Use the fast iterative shrinkage-thresholding algorithm.
  admm   Use the alternating direction method of multipliers (experimental).
 ====== =============================================================================================

 :Default: cd
 :Type: String
 :Description: Selects the iterative solver used when ``LMODEL = enet | adaptive-lasso``. All three
               solvers target the same elastic-net or adaptive-LASSO objective and are
               interchangeable. The ``cd`` solver is the coordinate descent method; it warm-starts
               along the cross-validation alpha path and is the fastest choice for that path, so it
               is the recommended default. The ``fista`` solver avoids forming the dense
               :math:`N \times N` Gram matrix and can reduce memory usage for large dense
               L1-regularized problems. The ``admm`` solver is provided mainly for research and
               cross-checking; on warm-started alpha paths it is generally slower than ``cd`` and
               offers no advantage for the typical problem sizes here.

````

.. _alm_l1_alpha:

* L1_ALPHA-tag : The coefficient of the L1 regularization term

 :Default: 0.0 
 :Type: Double
 :Description: This tag is used when ``LMODEL = enet | adaptive-lasso`` and ``CV = 0``. See also :ref:`here <alm_theory_enet>`.

````

.. _alm_cv:

* CV-tag : Cross-validation mode for elastic net 

 ===== ===================================================================================================================
   0   | Cross-validation mode is off. 
       | The elastic net optimization is solved with the given ``L1_ALPHA`` value. 
       | The force constants are written to ``PREFIX``.fcs and ``PREFIX``.xml.

  >= 2 | ``CV``-fold cross-validation is performed *automatically*. 
       | ``NDATA`` training datasets are divided into ``CV`` subsets, and ``CV`` different combinations of 
       | training-validation datasets are created internally. For each combination, the elastic net 
       | optimization is solved with the various ``L1_ALPHA`` values defined by the ``CV_MINALPHA``, 
       | ``CV_MAXALPHA``, and ``CV_NALPHA`` tags. The result of each cross-validation is stored in 
       | ``PREFIX``.cvset[1, ..., ``CV``], and their average and deviation are stored in ``PREFIX``.cvscore. 

  -1   | The cross-validation is performed *manually*.
       | The Taylor expansion potential is trained by using the training datasets in ``DFSET``, and 
       | the validation score is calculated by using the data in ``DFSET_CV`` for various ``L1_ALPHA`` values
       | defined by the ``CV_MINALPHA``, ``CV_MAXALPHA``, and ``CV_NALPHA`` tags.
       | After the calculation, the fitting and validation errors are stored in ``PREFIX``.cvset.
       | This option may be convenient for a large-scale problem since multiple optimization tasks with
       | different training-validation datasets can be done in parallel.
 ===== ===================================================================================================================

 :Default: 0
 :Type: Integer
 :Description: This tag is used when ``LMODEL = enet | adaptive-lasso``.


````

.. _alm_dfset_cv:

* DFSET_CV-tag : File name containing displacement-force datasets used for manual cross-validation

 :Default: ``DFSET_CV = DFSET``
 :Type: String
 :Description: This tag is used when ``LMODEL = enet | adaptive-lasso`` and ``CV = -1``.

````

.. _alm_ndata_cv:

* NDATA_CV-tag : Number of displacement-force validation datasets 

 :Default: None 
 :Type: Integer
 :Description: This tag is used when ``LMODEL = enet | adaptive-lasso`` and ``CV = -1``.

````

.. _alm_nstart_cv:

* NSTART_CV, NEND_CV-tags : Specifies the range of data to be used for validation

 :Default: ``NSTART_CV = 1``, ``NEND_CV = NDATA_CV``
 :Type: Integer
 :Description: This tag is used when ``LMODEL = enet | adaptive-lasso`` and ``CV = -1``.

````


.. _alm_cv_minalpha:

* CV_MINALPHA, CV_MAXALPHA, CV_NALPHA-tags : Options to specify the ``L1_ALPHA`` values used in cross-validation 

 :Default: ``CV_MAXALPHA`` is set automatically

           ``CV_MINALPHA = CV_MAXALPHA * 1.0e-6``

           ``CV_NALPHA = 50`` 
 :Type: Double, Double, Integer
 :Description: ``CV_NALPHA`` values of ``L1_ALPHA`` are generated from ``CV_MINALPHA`` to ``CV_MAXALPHA`` in logarithmic scale. When ``CV_MAXALPHA`` is not specified by user, the code automatically sets ``CV_MAXALPHA`` so that the maximum ``L1_ALPHA`` makes all coefficients zero. The default value of ``CV_MINALPHA`` is ``CV_MAXALPHA * 1.0e-6``, which is reasonable in many cases. If the minimum value of the validation score is found at ``CV_MINALPHA``, you may need to use a smaller value of ``CV_MINALPHA``. This tag is used when ``LMODEL = enet | adaptive-lasso`` and the cross-validation mode is on (``CV > 0`` or ``CV = -1``).

````

.. _alm_standardize:

* STANDARDIZE-tag = 0 | 1

 ===== =============================================================================================
   0    Do not standardize the sensing matrix
   1   | Each column of the sensing matrix is standardized in such a way that its mean value
       | becomes 0 and standard deviation becomes 1. 
 ===== =============================================================================================

 :Default: 1
 :Type: Integer
 :Description: This option influences the optimal ``L1_ALPHA`` value. So, if you change the ``STANDARDIZE`` option, you have to rerun the cross-validation. Effective only when ``LMODEL = enet``.


````

.. _alm_enet_dnorm:

* ENET_DNORM-tag : Normalization factor of atomic displacements

 :Default: 1.0
 :Type: Double
 :Description: The normalization factor of atomic displacement :math:`u_{0}` in units of bohr. When :math:`u_{0} (\neq 1)` is given, the displacement data are scaled as :math:`u_{i} \rightarrow u_{i}/u_{0}` before constructing the sensing matrix. This option influences the optimal ``L1_ALPHA`` value. So, if you change the ``ENET_DNORM`` value, you will have to rerun the cross-validation. Effective only when ``LMODEL = enet`` and ``STANDARDIZE = 0``. 

````



.. _alm_solution_path:

* SOLUTION_PATH-tag = 0 | 1

 ===== =============================================================================================
   0    Do not save the solution path.
   1    Save the solution path of each cross-validation combination in ``PREFIX``.solution_path.
 ===== =============================================================================================

 :Default: 0
 :Type: Integer
 :Description: Effective when ``LMODEL = enet | adaptive-lasso`` and the cross-validation mode is on.

````

.. _alm_debias_ols:

* DEBIAS_OLS-tag = 0 | 1

 ===== =============================================================================================
   0    Save the solution of the L1-regularized problem to ``PREFIX``.fcs and ``PREFIX``.xml.
   1    | After the solution of the elastic net or adaptive LASSO optimization problem is obtained,
        | only non-zero coefficients are collected, and the ordinary least-squares fitting is
        | solved again with the non-zero coefficients before saving the results to ``PREFIX``.fcs and
        | ``PREFIX``.xml. This might be useful to reduce the bias of the L1-regularized solution.
 ===== =============================================================================================

 :Default: 0
 :Type: Integer
 :Description: Effective when ``LMODEL = enet | adaptive-lasso`` and ``CV = 0``.


````

.. _alm_stop_criterion:

* STOP_CRITERION-tag : The scan over ``L1_ALPHA`` stops when the cross-validation score keeps increasing in ``STOP_CRITERION`` consecutive steps

 :Default: 5
 :Type: Integer
 :Description: Effective when ``LMODEL = enet | adaptive-lasso`` and the cross-validation mode is turned on (``CV > 0`` or ``CV = -1``).


````


How to make a DFSET file
~~~~~~~~~~~~~~~~~~~~~~~~

.. _label_format_DFSET:

Format of ``DFSET`` 
++++++++++++++++++++

The displacement-force data sets obtained by first-principles (or classical force-field) calculations
have to be saved to a file, say *DFSET*. Then, the force constants are estimated by setting ``DFSET =`` *DFSET* and with ``MODE = optimize``.

The *DFSET* file must contain the atomic displacements and corresponding forces in Cartesian coordinate for at least ``NDATA`` structures (displacement patterns)
in the following format: 

.. math::
  :nowrap:

    # Structure number 1 (this is just a comment line)
    \begin{eqnarray*}
     u_{x}(1) & u_{y}(1) & u_{z}(1) & f_{x}(1) & f_{y}(1) & f_{z}(1) \\
     u_{x}(2) & u_{y}(2) & u_{z}(2) & f_{x}(2) & f_{y}(2) & f_{z}(2) \\
              & \vdots   &          &          & \vdots   &          \\
     u_{x}(\mathrm{NAT}) & u_{y}(\mathrm{NAT}) & u_{z}(\mathrm{NAT}) & f_{x}(\mathrm{NAT}) & f_{y}(\mathrm{NAT}) & f_{z}(\mathrm{NAT})
    \end{eqnarray*}
    # Structure number 2
    \begin{eqnarray*}
     u_{x}(1) & u_{y}(1) & u_{z}(1) & f_{x}(1) & f_{y}(1) & f_{z}(1) \\
              & \vdots   &          &          & \vdots   &          
    \end{eqnarray*}

Here, ``NAT`` is the number of atoms in the supercell. 
The unit of displacements and forces must be **bohr** and **Ryd/bohr**, respectively.
