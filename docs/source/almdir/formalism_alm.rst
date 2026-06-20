ALM: Theoretical background
=============================

Interatomic force constants (IFCs)
----------------------------------

The starting point of the computational methodology is to approximate the potential energy of interacting atoms 
by a Taylor expansion with respect to atomic displacements by

.. math::
    :label: U_Taylor

    &U - U_{0} = \sum_{n=1}^{N} U_{n} = U_{1} + U_{2} + U_{3} + \cdots, \\
    &U_{n} = \frac{1}{n!} \sum_{\substack{\ell_{1}\kappa_{1}, \dots, \ell_{n}\kappa_{n} \\ \mu_{1},\dots, \mu_{n}}} \Phi_{\mu_{1}\dots\mu_{n}}(\ell_{1}\kappa_{1};\dots;\ell_{n}\kappa_{n}) \; u_{\mu_{1}}(\ell_{1}\kappa_{1})\cdots u_{\mu_{n}}(\ell_{n}\kappa_{n}).

Here, :math:`u_{\mu}(\ell\kappa)` is the atomic displacement of :math:`\kappa`\ th atom in the :math:`\ell`\ th unit cell along :math:`\mu`\ th direction, and :math:`\Phi_{\mu_{1}\dots\mu_{n}}(\ell_{1}\kappa_{1};\dots;\ell_{n}\kappa_{n})` is the :math:`n`\ th-order interatomic force constant (IFC).


Symmetry relationship between IFCs
----------------------------------

There are several relationships between IFCs which may be used to reduce the number of independent IFCs.

* Permutation

 IFC should be invariant under the exchange of the triplet :math:`(\ell,\kappa,\mu)`, e.g.,

 .. math::

  \Phi_{\mu_{1}\mu_{2}\mu_{3}}(\ell_{1}\kappa_{1};\ell_{2}\kappa_{2};\ell_{3}\kappa_{3})=\Phi_{\mu_{1}\mu_{3}\mu_{2}}(\ell_{1}\kappa_{1};\ell_{3}\kappa_{3};\ell_{2}\kappa_{2})=\dots. 

* Periodicity

 Since IFCs should depend on interatomic distances, they are invariant under a translation in units of the lattice vector, namely

 .. math::
 
  \Phi_{\mu_{1}\mu_{2}\dots\mu_{n}}(\ell_{1}\kappa_{1};\ell_{2}\kappa_{2};\dots;\ell_{n}\kappa_{n})=\Phi_{\mu_{1}\mu_{2}\dots\mu_{n}}(0\kappa_{1};\ell_{2}-\ell_{1}\kappa_{2};\dots;\ell_{n}-\ell_{1}\kappa_{n}). 

.. _IFC_crystal_symmetry:

* Crystal symmetry

 A crystal symmetry operation maps an atom :math:`\vec{r}(\ell\kappa)` to another equivalent atom :math:`\vec{r}(LK)` by rotation and translation.
 Since the potential energy is invariant under any crystal symmetry operations, IFCs should transform under a symmetry operation as follows:

 .. math::
     :label: ifcsym1

     \sum_{\nu_{1},\dots,\nu_{n}}\Phi_{\nu_{1}\dots\nu_{n}}(L_{1}K_{1};\dots;L_{n}K_{n}) O_{\nu_{1}\mu_{1}}\cdots O_{\nu_{n}\mu_{n}} = \Phi_{\mu_{1}\dots\mu_{n}}(\ell_{1}\kappa_{1};\dots;\ell_{n}\kappa_{n}),

 where :math:`O` is the rotational matrix of the symmetry operation. 
 Let :math:`N_{s}` be the number of symmetry operations, there are :math:`N_{s}` relationships between IFCs which may be used to find independent IFCs.

 .. Note::

   When ``FCSYM_BASIS = Cartesian``, symmetricaly irreducible set of IFCs is searched in the Cartesian coordinate where the matrix element :math:`O_{\mu\nu}` is 0 or :math:`\pm1` for all space group operations except for those of the **hexagonal** (trigonal) systems. Also, except for the hexagonal (trigonal) systems, the product :math:`O_{\nu_{1}\mu_{1}}\cdots O_{\nu_{n}\mu_{n}}` in the left hand side of equation :eq:`ifcsym1` becomes non-zero only for a specific pair of :math:`\{\nu\}` (and becomes 0 for all other :math:`\{\nu\}`\ s). Therefore, let :math:`\{\nu^{\prime}\}` be such a pair of :math:`\{\nu\}`, the equation :eq:`ifcsym1` can be reduced to

   .. math::
       :label: ifcsym2
     
       \Phi_{\nu_{1}^{\prime}\dots\nu_{n}^{\prime}}(L_{1}K_{1};\dots;L_{n}K_{n}) = s \Phi_{\mu_{1}\dots\mu_{n}}(\ell_{1}\kappa_{1};\dots;\ell_{n}\kappa_{n}),
   
   where :math:`s=\pm1`. The code employs equation :eq:`ifcsym2` instead of equation :eq:`ifcsym1` to reduce the number of IFCs. If IFCs of the left-hand side and the right-hand side of equation :eq:`ifcsym2` are equivalent and the coupling coefficient is :math:`s=-1`, the IFC is removed since it becomes zero. For **hexagonal** (trigonal) systems, there can be symmetry operations where multiple terms in the left-hand side of equation :eq:`ifcsym1` become non-zero. For such cases, equation :eq:`ifcsym1` is not used to reduce the number of IFCs. Alternatively, the corresponding symmetry relationships are imposed as constraints between IFCs in solving fitting problems.

   When ``FCSYM_BASIS = Lattice`` (default), the symmetry reduction of IFCs is performed in the lattice coordinate. In this case, all elements of the rotational matrix become either 0 or :math:`\pm1` in the :math:`\boldsymbol{a}_1, \boldsymbol{a}_2, \boldsymbol{a}_3` basis even for the **hexagonal** systems. Therefore, the numerical stability of the reduction process, particularly of the construction of rref (reduced row echelon form), is improved even when the number of numerical digits for irrational numbers given in an input file is less than sufficient (double precision).


.. _constraint_IFC:

Constraints between IFCs
------------------------

Since the potential energy is invariant under rigid translation and rotation, it may be necessary for IFCs to satisfy corresponding constraints.

The constraints for translational invariance are given by

.. math::
    :label: consttran

    \sum_{\ell_{1}\kappa_{1}}\Phi_{\mu_{1}\mu_{2}\dots\mu_{n}}(\ell_{1}\kappa_{1};\ell_{2}\kappa_{2};\dots;\ell_{n}\kappa_{n}) = 0,
  
which should be satisfied for arbitrary pairs of :math:`\ell_{2}\kappa_{2},\dots,\ell_{n}\kappa_{n}` and :math:`\mu_{1},\dots,\mu_{n}`. The code **alm** imposes equation :eq:`consttran` by default (``ICONST = 1``). 

The constraints for rotational invariance are

.. math::
    
    &\sum_{\ell^{\prime}\kappa^{\prime}} (\Phi_{\mu_{1}\dots\mu_{n}\nu}(\ell_{1}\kappa_{1};\dots;\ell_{n}\kappa_{n};\ell^{\prime}\kappa^{\prime}) r_{\mu}(\ell^{\prime}\kappa^{\prime}) 
    - \Phi_{\mu_{1}\dots\mu_{n}\mu}(\ell_{1}\kappa_{1};\dots;\ell_{n}\kappa_{n};\ell^{\prime}\kappa^{\prime}) r_{\nu}(\ell^{\prime}\kappa^{\prime})) \\
    & \hspace{10mm} + \sum_{\lambda = 1}^{n}\sum_{\mu_{\lambda}^{\prime}} \Phi_{\mu_{1}\dots\mu_{\lambda}^{\prime}\dots\mu_{n}}(\ell_{1}\kappa_{1};\dots;\ell_{\lambda}\kappa_{\lambda};\dots;\ell_{n}\kappa_{n}) (\delta_{\mu,\mu_{\lambda}}\delta_{\nu,\mu_{\lambda}^{\prime}} - \delta_{\nu,\mu_{\lambda}}\delta_{\mu,\mu_{\lambda}^{\prime}}) = 0,

which must be satisfied for arbitrary pairs of :math:`(\ell_{1}\kappa_{1},\dots,\ell_{n}\kappa_{n};\mu_{1},\dots,\mu_{n};\mu,\nu)`.
This is complicated since :math:`(n+1)`\ th-order IFCs (first line) are related to :math:`n`\ th-order IFCs (second line).

For example, the constraints for rotational invariance related to harmonic terms can be found as 

.. math::
    :label: constrot1

    &\sum_{\ell_{2}\kappa_{2}} (\Phi_{\mu_{1}\nu}(\ell_{1}\kappa_{1};\ell_{2}\kappa_{2})r_{\mu}(\ell_{2}\kappa_{2})-\Phi_{\mu_{1}\mu}(\ell_{1}\kappa_{1};\ell_{2}\kappa_{2})r_{\nu}(\ell_{2}\kappa_{2})) \notag \\
    & \hspace{10mm} + \Phi_{\nu}(\ell_{1}\kappa_{1})\delta_{\mu,\mu_{1}} - \Phi_{\mu}(\ell_{1}\kappa_{1})\delta_{\nu,\mu_{1}} = 0,

and

.. math::
    :label: constrot2 

    & \sum_{\ell_{3}\kappa_{3}} (\Phi_{\mu_{1}\mu_{2}\nu}(\ell_{1}\kappa_{1};\ell_{2}\kappa_{2};\ell_{3}\kappa_{3}) r_{\mu}(\ell_{3}\kappa_{3}) \notag
    - \Phi_{\mu_{1}\mu_{2}\mu}(\ell_{1}\kappa_{1};\ell_{2}\kappa_{2};\ell_{3}\kappa_{3}) r_{\nu}(\ell_{3}\kappa_{3})) \\
    & \hspace{10mm} 
    + \Phi_{\nu\mu_{2}}(\ell_{1}\kappa_{1};\ell_{2}\kappa_{2})\delta_{\mu,\mu_{1}} 
    - \Phi_{\mu\mu_{2}}(\ell_{1}\kappa_{1};\ell_{2}\kappa_{2})\delta_{\nu,\mu_{1}} \notag \\
    & \hspace{10mm} + \Phi_{\mu_{1}\nu}(\ell_{1}\kappa_{1};\ell_{2}\kappa_{2})\delta_{\mu,\mu_{2}}
    - \Phi_{\mu_{1}\mu}(\ell_{1}\kappa_{1};\ell_{2}\kappa_{2})\delta_{\nu,\mu_{2}} = 0.
  
When ``NORDER = 1``, equation :eq:`constrot1` will be considered if ``ICONST = 2``, whereas equation :eq:`constrot2` will be neglected. To further consider equation :eq:`constrot2`, please use ``ICONST = 3``, though it may enforce a number of harmonic IFCs to be zero since cubic terms don't exist in harmonic calculations (``NORDER = 1``).


.. _fitting_formalism:

Estimate IFCs by linear regression
----------------------------------

Basic notations
+++++++++++++++

From the symmetrically independent set of IFCs and the constraints between them for satisfying the translational and/or rotational invariance, we can construct an irreducible set of IFCs :math:`\{\Phi_{i}\}`. Let us denote a column vector comprising the :math:`N` irreducible set of IFCs as :math:`\boldsymbol{\Phi}`. Then, the Taylor expansion potential (TEP) defined by equation :eq:`U_Taylor` is written as

.. math::
    U_{\mathrm{TEP}} = \boldsymbol{b}^{T}\boldsymbol{\Phi}.

Here, :math:`\boldsymbol{b} \in \mathbb{R}^{1\times N}` is a function of atomic displacements :math:`\{u_{i}\}` defined as :math:`\boldsymbol{b} = \partial U / \partial \boldsymbol{\Phi}`. The atomic forces based on the TEP is then given as

.. math::
    :label: force_tep

    \boldsymbol{F}_{\mathrm{TEP}} = - \frac{\partial U_{\mathrm{TEP}}}{\partial \boldsymbol{u}} = - \frac{\partial \boldsymbol{b}^{T}}{\partial \boldsymbol{u}} \boldsymbol{\Phi} = A \boldsymbol{\Phi},

where :math:`A \in \mathbb{R}^{3N_{s} \times N}` with :math:`N_{s}` being the number of atoms in the supercell, 
and :math:`\boldsymbol{u}^{T} = (u_{1}^{x}, u_{1}^{y}, u_{1}^{z}, \dots, u_{N_{s}}^{x}, u_{N_{s}}^{y}, u_{N_{s}}^{z})` is the vector comprising :math:`3N_{s}` atomic displacements in the supercell. 
Note that the matrix :math:`A` and force vector :math:`\boldsymbol{F}_{\mathrm{TEP}}` depend on the atomic configuration of the supercell.
To make this point clearer, let us denote them as :math:`A(\boldsymbol{u})` and :math:`\boldsymbol{F}_{\mathrm{TEP}}(\boldsymbol{u})`.

To estimate the IFC vector :math:`\boldsymbol{\Phi}` by linear regression, it is usually necessary to consider several different displacement patterns.
Let us suppose we have :math:`N_d` displacement patterns and atomic forces for each pattern obtained by DFT.
Then, equation :eq:`force_tep` defined for each displacement pattern can be combined to a single equation as

.. math::
    \boldsymbol{\mathscr{F}}_{\mathrm{TEP}} =  \mathbb{A} \boldsymbol{\Phi},

where :math:`\boldsymbol{\mathscr{F}}^{T} = [\boldsymbol{F}^{T}(\boldsymbol{u}_{1}), \dots, \boldsymbol{F}^{T}(\boldsymbol{u}_{N_d})]` and 
:math:`\mathbb{A}^{T} = [A^{T}(\boldsymbol{u}_{1}),\dots,A^{T}(\boldsymbol{u}_{N_d})]`.


Ordinary least-squares
++++++++++++++++++++++

In the ordinary least-squares (``LMODEL = least-squares``), IFCs are estimated by solving the following problem:

.. math::
   :label: lsq

   \boldsymbol{\Phi}_{\mathrm{OLS}} = \mathop{\rm argmin}\limits_{\boldsymbol{\Phi}} \frac{1}{2N_{d}} \|\boldsymbol{\mathscr{F}}_{\mathrm{DFT}} - \boldsymbol{\mathscr{F}}_{\mathrm{TEP}} \|^{2}_{2} = \mathop{\rm argmin}\limits_{\boldsymbol{\Phi}} \frac{1}{2N_{d}}   \|\boldsymbol{\mathscr{F}}_{\mathrm{DFT}} - \mathbb{A} \boldsymbol{\Phi} \|^{2}_{2}.

Therefore, the IFCs are determined so that the residual sum of squares (RSS) is minimized. 
To determine all elements of  :math:`\boldsymbol{\Phi}_{\mathrm{OLS}}` uniquely, :math:`\mathbb{A}^{T}\mathbb{A}` must be full rank. When the fitting is successful, **alm** reports the relative fitting error :math:`\sigma` defined by

.. math::
   :label: fitting_error

   \sigma = \sqrt{\frac{\|\boldsymbol{\mathscr{F}}_{\mathrm{DFT}} - \mathbb{A} \boldsymbol{\Phi} \|^{2}_{2}}{\|\boldsymbol{\mathscr{F}}_{\mathrm{DFT}}\|_{2}^{2}}},

where the denominator is the square sum of the DFT forces.

.. _alm_theory_enet:

Elastic-net regression
++++++++++++++++++++++

In the elastic-net optimization (``LMODEL = elastic-net``), IFCs are estimated by solving the following optimization problem:

.. math::
   :label: enet

   \boldsymbol{\Phi}_{\mathrm{enet}} = \mathop{\rm argmin}\limits_{\boldsymbol{\Phi}} \frac{1}{2N_{d}}   \|\boldsymbol{\mathscr{F}}_{\mathrm{DFT}} - \mathbb{A} \boldsymbol{\Phi} \|^{2}_{2} + \alpha \beta \| \boldsymbol{\Phi}  \|_{1} + \frac{1}{2} \alpha (1-\beta) \| \boldsymbol{\Phi}  \|_{2}^{2},

where :math:`\alpha` is a hyperparameter that controls the trade-off between the sparsity and accuracy of the model, and :math:`\beta \; (0 < \beta \leq 1)` is a hyperparameter that controls the ratio of the :math:`L_{1}` and :math:`L_{2}` regularization terms. :math:`\alpha` and :math:`\beta` must be given by input tags ``L1_ALPHA`` and ``L1_RATIO``, respectively.

An optimal value of :math:`\alpha` can be estimated, for example, by cross-validation (CV). A :math:`n`\ -fold CV can be performed by setting the ``CV``-tag properly.

Adaptive LASSO [1]_ 
++++++++++++++++++++

In adaptive LASSO (``LMODEL = adaptive-lasso``), IFCs are estimated by solving the following optimization problem:

.. math::
   :label: adalasso

   \boldsymbol{\Phi}_{\mathrm{adalasso}} = \mathop{\rm argmin}\limits_{\boldsymbol{\Phi}} \frac{1}{2N_{d}}   \|\boldsymbol{\mathscr{F}}_{\mathrm{DFT}} - \mathbb{A} \boldsymbol{\Phi} \|^{2}_{2} + \alpha \sum_i w_i |\Phi_i| ,

where :math:`\alpha` is a hyperparameter given by ``L1_ALPHA``, and :math:`w_i` is the parameter-dependent weight. In ALM, we simply use :math:`w_i = 1/|\Phi_{\mathrm{OLS},i}|` with :math:`\Phi_{\mathrm{OLS},i}` being the coefficient estimator produced by an OLS fitting. Hence, in this option, the size of the training dataset must be large enough to make the matrix :math:`\mathbb{A}` *overdetermined*. The code keeps running even when :math:`\mathbb{A}` is *underdetermined*. So, please be careful.

.. note::

    The minimum size of the training dataset necessary for making :math:`\mathbb{A}` overdetermined can be roughly estimated as follows: 
    
    We assume that there are :math:`N` independent IFCs (after imposing constraints, if there is any). In this case, the number of columns of matrix :math:`\mathbb{A}` becomes :math:`N`, and :math:`\mathbb{A}` becomes overdetermined when the number of independent rows of :math:`\mathbb{A}` is :math:`N` or larger. If the training structures are generated randomly and all atoms are displaced from their original positions in each configuration, we can generate :math:`3\times N_{\mathrm{atom}}` (:math:`N_{\mathrm{atom}}` is the number of atoms in the supercell) linearly independent rows of :math:`\mathbb{A}` from one displaced configuration , i.e., one static DFT calculation. Hence, we expect that :math:`\mathbb{A}` becomes overdetermined when

    .. math::

        N_d \geq \frac{N}{3N_{\mathrm{atom}}}

    where :math:`N_d` is the number of displacement patterns in the training dataset. 

    In cross validation, the entire training dataset is divided into smaller subsets. For each subset, the above condition should be satisfied.


````

.. [1] H\. Zou, *The Adaptive Lasso and Its Oracle Properties*, J\. Am\. Stat\. Assoc\. **101**, 1418 (2006).
