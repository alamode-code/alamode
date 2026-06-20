ANPHON: Theoretical background
==============================

.. |umulaut_u|   unicode:: U+00FC
.. |umulaut_o|   unicode:: U+00F6

Dynamical matrix
----------------

The dynamical matrix is given by

.. math::
    :label: dymat 

    D_{\mu\nu}(\kappa\kappa^{\prime};\boldsymbol{q}) = \frac{1}{\sqrt{M_{\kappa}M_{\kappa^{\prime}}}}
    \sum_{\ell^{\prime}}\Phi_{\mu\nu}(\ell\kappa;\ell^{\prime}\kappa^{\prime})\exp{\left[i\boldsymbol{q}\cdot(\boldsymbol{r}(\ell^{\prime})-\boldsymbol{r}(\ell))\right]},

where :math:`M_{\kappa}` is the atomic mass of atom :math:`\kappa`.
By diagonalizing the dynamical matrix, one can obtain :math:`m` (:math:`=3N_{\kappa}`) eigenvalues :math:`\omega_{\boldsymbol{q}j}^{2}`  (:math:`j = 1, 2, \dots, m`) and corresponding eigenvectors :math:`\boldsymbol{e}_{\boldsymbol{q}j}` for each :math:`\boldsymbol{q}` point.
Here, :math:`\boldsymbol{e}_{\boldsymbol{q}j}` is a column vector consisting of atomic polarization :math:`e_{\mu}(\kappa;\boldsymbol{q}j)`.
Let :math:`D(\boldsymbol{q})` denote a matrix form of equation :eq:`dymat`, the eigenvalues may be written as

.. math::
    :label: omega2

    \omega_{\boldsymbol{q}j}^{2} = (\boldsymbol{e}_{\boldsymbol{q}j}^{*})^{\mathrm{T}} D(\boldsymbol{q})\boldsymbol{e}_{\boldsymbol{q}j}.

Next, we introduce :math:`m\times m` matrices :math:`\Lambda` and :math:`W` which are defined as 
:math:`\Lambda(\boldsymbol{q}) = \mathrm{diag} (\omega_{\boldsymbol{q}1}^{2},\dots,\omega_{\boldsymbol{q}m}^{2})` and 
:math:`W(\boldsymbol{q}) = (\boldsymbol{e}_{\boldsymbol{q}1},\dots,\boldsymbol{e}_{\boldsymbol{q}m})`, respectively. 
Then, equation :eq:`omega2` can be denoted as 

.. math::
    
    \Lambda(\boldsymbol{q}) = W^{\dagger}(\boldsymbol{q})D(\boldsymbol{q})W(\boldsymbol{q}).


Non-analytic correction
-----------------------

When one needs to capture the LO\-TO splitting near the zone-center, it is necessary to account for the long\-range correction to the dynamical matrix. 
In the **anphon** code, three different approaches are implemented: the Parlinski's method [1]_ (``NONANALYTIC = 1``), the mixed-space approach [2]_  (``NONANALYTIC = 2``), and the Ewald summation method (``NONANALYTIC = 3``). 

The first two methods are rather simple and add the non-analytic part of the dynamical matrix with different weighting factors.
The non-analytic part of the dynamical matrix is defined as

.. math::

    D_{\mu\nu}^{\mathrm{NA}}(\kappa\kappa^{\prime};\boldsymbol{q}) = \frac{1}{\sqrt{M_{\kappa}M_{\kappa^{\prime}}}}
    \frac{4\pi e^{2}}{V} \frac{(Z_{\kappa}^{*}\boldsymbol{q})_{\mu}(Z_{\kappa^{\prime}}^{*}\boldsymbol{q})_{\nu}}{\boldsymbol{q}\cdot\epsilon^{\infty}\boldsymbol{q}},

where :math:`V` is the volume of the primitive cell, :math:`Z_{\kappa}^{*}` is the Born effective charge tensor of atom :math:`\kappa`, 
and :math:`\epsilon^{\infty}` is the electronic dielectric permittivity tensor, respectively.

Parlinski's approach (``NONANALYTIC = 1``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the Parlinski's approach [1]_ , the total dynamical matrix is given by

.. math::

    D(\boldsymbol{q}) + D^{\textrm{NA}}(\boldsymbol{q})\exp{(-q^{2}/\sigma^{2})},

where :math:`\sigma` is a damping factor (``NA_SIGMA``). 
:math:`\sigma` must be chosen carefully so that the non-analytic contribution becomes negligible at Brillouin zone (BZ) boundaries. 
Too large :math:`\sigma` can result in a discontinuity of phonon dispersion at the BZ boundaries, while too small :math:`\sigma` can yield a large oscillation of the LO phonon frequencies as going away from :math:`\boldsymbol{q}=0`. So, an appropriate :math:`\sigma` value needs to be chosen in an *ad hoc* way.


Mixed-space approach (``NONANALYTIC = 2``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the mixed-space approach [2]_ , the total dynamical matrix is given by

.. math::

    D(\boldsymbol{q}) + D^{\textrm{NA}}(\boldsymbol{q})\frac{1}{N}\sum_{\ell^{\prime}}\exp{\left[i\boldsymbol{q}\cdot(\boldsymbol{r}(\ell^{\prime})-\boldsymbol{r}(\ell))\right]}.

The second term vanishes at the :math:`\boldsymbol{q}` points commensurate with the supercell except for the :math:`\Gamma` point (:math:`\boldsymbol{q} = 0`).
No adjustable parameters exist in this approach. In many cases, the mixed-space approach give reasonable dispersion curves. 
However, it can still yield a discontinuity in the phonon dispersion at BZ boundaries when the boundary point is incommensurate with the supercell size.


Ewald method (``NONANALYTIC = 3``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Ewald summation technique [10]_ is more accurate and therefore recommended in general.
In this method, we first decompose the harmonic force constants computed in a supercell into the long\-range dipole-dipole (DD) components and the short\-range ones as follows

.. math:: 

    \Phi_{\mu\nu}^{\mathrm{SR}}(\ell\kappa;\ell^{\prime}\kappa^{\prime})= \Phi_{\mu\nu}(\ell\kappa;\ell^{\prime}\kappa^{\prime}) - \Phi_{\mu\nu}^{\mathrm{DD}}(\ell\kappa;\ell^{\prime}\kappa^{\prime}).


After the short\-range components are obtained, they are used in equation :eq:`dymat` to obtain the short\-range dynamical matrix :math:`D_{\mu\nu}^{\mathrm{SR}}(\kappa\kappa^{\prime};\boldsymbol{q})`. Then, the total dynamical matrix is constructed as

.. math:: 

    D_{\mu\nu}(\kappa\kappa^{\prime};\boldsymbol{q}) = D^{\mathrm{SR}}_{\mu\nu}(\kappa\kappa^{\prime};\boldsymbol{q}) +  D^{\mathrm{DD}}_{\mu\nu}(\kappa\kappa^{\prime};\boldsymbol{q}).

The long\-range terms, :math:`\Phi_{\mu\nu}^{\mathrm{DD}}(\ell\kappa;\ell^{\prime}\kappa^{\prime})` and :math:`D^{\mathrm{DD}}_{\mu\nu}(\kappa\kappa^{\prime};\boldsymbol{q})`, can be calculated from :math:`\epsilon^{\infty}` and :math:`Z_{\kappa}^{*}` using the Ewald summation technique.

To include the non-analytic correction with ``NONANALYTIC > 0``, one also need to give ``BORNINFO``.


Group velocity
--------------

The group velocity of phonon mode :math:`\boldsymbol{q}j` is given by 

.. math::
    
    \boldsymbol{v}_{\boldsymbol{q}j} = \frac{\partial \omega_{\boldsymbol{q}j}}{\partial \boldsymbol{q}}.

To evaluate the group velocity numerically, we employ a central difference where
:math:`\boldsymbol{v}` may approximately be given by

.. math::

    \boldsymbol{v}_{\boldsymbol{q}j} \approx \frac{\omega_{\boldsymbol{q}+\Delta\boldsymbol{q}j} - \omega_{\boldsymbol{q}-\Delta\boldsymbol{q}j}}{2\Delta\boldsymbol{q}}.

If one needs to save the group velocities, please turn on the ``PRINTVEL``-tag.


Mode effective charge
---------------------

The mode effective charge is defined as 

.. math::
    
    Z^{*}_{j,\alpha} = \sum_{\kappa\beta}Z^{*}_{\kappa,\alpha\beta}\frac{e_{\beta}(\kappa;\boldsymbol{0}j)}{\sqrt{M_{\kappa}}}

where :math:`Z^{*}_{\kappa,\alpha\beta}` is the atomic Born effective charge. To compute the mode effective charges of the zone-center modes, please set ``ZMODE = 1`` in the &analysis field and supply ``BORNINFO``.


Thermodynamics functions
------------------------

The specific heat at constant volume :math:`C_{\mathrm{v}}`, the internal energy :math:`U`, 
the vibrational entropy :math:`S`, and the Helmholtz free energy :math:`F` of individual harmonic oscillator are
given as follows:

.. math::
    :nowrap:
    
    \begin{align*}
     U &= \frac{1}{N_{q}}\sum_{\boldsymbol{q},j} \hbar\omega_{\boldsymbol{q}j} \left[\frac{1}{e^{\hbar\omega_{\boldsymbol{q}j}/kT} - 1} + \frac{1}{2}\right], \\
     C_{\mathrm{v}} &= \frac{k}{N_{q}}\sum_{\boldsymbol{q},j} \left(\frac{\hbar\omega_{\boldsymbol{q}j}}{2kT}\right)^{2} \mathrm{cosech}^{2}\left(\frac{\hbar\omega_{\boldsymbol{q}j}}{2kT}\right),\\
     S &= \frac{k}{N_{q}}\sum_{\boldsymbol{q},j} \left[\frac{\hbar\omega_{\boldsymbol{q}j}}{kT} \frac{1}{e^{\hbar\omega_{\boldsymbol{q}j}/kT} - 1} 
        - \log{\left( 1 - e^{-\hbar\omega_{\boldsymbol{q}j}/kT}\right)}\right], \\
     F &= \frac{1}{N_{q}}\sum_{\boldsymbol{q},j}\left[ \frac{\hbar\omega_{\boldsymbol{q}j}}{2} + kT\log{\left( 1 - e^{-\hbar\omega_{\boldsymbol{q}j}/kT}\right)} \right].
    \end{align*}

Here, :math:`k` is the Boltzmann constant. These quantities are saved in the ``PREFIX``.thermo file.

When the self-consistent phonon mode (``MODE = SCPH``) is selected, the anharmonic free-energy 
defined by the following equation will be calculated and saved in the ``PREFIX``.scph_thermo file:

.. math::
    :nowrap:
    
    \begin{align*}
     F^{\mathrm{SCP}} &= \frac{1}{N_{q}}\sum_{\boldsymbol{q},j}\left[ \frac{\hbar\Omega_{\boldsymbol{q}j}}{2} + kT\log{\left( 1 - e^{-\hbar\Omega_{\boldsymbol{q}j}/kT}\right)} \right] \\
     & - \frac{1}{4N_{q}}\sum_{\boldsymbol{q},j}\left[ \Omega_{\boldsymbol{q}j}^{2} - (C_{\boldsymbol{q}}^{\dagger}\Lambda_{\boldsymbol{q}}^{(\mathrm{HA})}C_{\boldsymbol{q}})_{jj} \right]
     \times \frac{\hbar [1 + 2n_{\boldsymbol{q}j} ]}{2\Omega_{\boldsymbol{q}j}}.
    \end{align*}

Details of the derivation of the above expression can be found in Ref. [7]_.


Mean square displacement
------------------------


The displacement-displacement correlation function is given by

.. math::
    :nowrap:

    \begin{align}
     \left< u_{\mu}(0\kappa)u_{\nu}(\ell'\kappa') \right> & = \frac{1}{\sqrt{M_{\kappa}M_{\kappa'}}N_{q}}\sum_{\boldsymbol{q}j}
     \frac{\hbar (2n_{\boldsymbol{q}j}+1)}{2\omega_{\boldsymbol{q}j}}
     \mathrm{Re}\bigg[e_{\mu}(\kappa;\boldsymbol{q}j)e_{\nu}^{*}(\kappa';\boldsymbol{q}j) e^{-i\boldsymbol{q}\cdot\boldsymbol{r}(\ell')}  \bigg],
    \end{align}

where :math:`n_{\boldsymbol{q}j} = 1/(e^{\hbar\omega_{\boldsymbol{q}j}/kT}-1)` is the Bose-Einstein distribution function. When ``UCORR = 1``, the code prints the above correlation function in ``PREFIX``.ucorr. The vector :math:`\boldsymbol{r}(\ell')` is the lattice translation vector to the :math:`\ell'` \ th unit cell, which can be specified by the ``SHIFT_UCORR`` tag.
When ``PRINTMSD`` is turned on, the code print the mean square displacements, which are the diagonal components of the correlation function:

.. math::

    \left< u_{\mu}^{2}(0\kappa)\right> = \frac{\hbar}{M_{\kappa}N_{q}}\sum_{\boldsymbol{q},j}\frac{1}{\omega_{\boldsymbol{q}j}} |e_{\mu}(\kappa;\boldsymbol{q}j)|^{2}
    \left(n_{\boldsymbol{q}j}+\frac{1}{2}\right).


Phonon DOS
----------

When *KPMODE* = 2, the program *anphon* saves the (one) phonon density of states (DOS) to the file ``PREFIX``.dos.
The one-phonon DOS is given by

.. math::

    \mathrm{DOS}(\omega) = \frac{1}{N_{q}}\sum_{\boldsymbol{q},j}\delta(\omega - \omega_{\boldsymbol{q}j}).

If ``PDOS = 1`` is given, the program also prints the atom-projected phonon DOS which is given by

.. math::
 
    \mathrm{PDOS}(\kappa;\omega) = \frac{1}{N_{q}}\sum_{\boldsymbol{q},j}|\boldsymbol{e}(\kappa;\boldsymbol{q}j)|^{2}\delta(\omega - \omega_{\boldsymbol{q}j}).

In addition, ``TDOS``-tag is available to compute the two-phonon DOS defined by

.. math::

    \mathrm{DOS2}(\omega;\boldsymbol{q};\pm) = \frac{1}{N_{q}}\sum_{\boldsymbol{q}_{1},\boldsymbol{q}_{2}, j_{1}, j_{2}}
    \delta(\omega\pm\omega_{\boldsymbol{q}_{1}j_{1}}-\omega_{\boldsymbol{q}_{2}j_{2}})\delta_{\boldsymbol{q}\pm\boldsymbol{q}_{1},\boldsymbol{q}_{2}+\boldsymbol{G}},

where :math:`\boldsymbol{G}` is a reciprocal lattice vector. The sign :math:`\pm` correspond to absorption and emission processes, respectively. Please note that the computation of the two-phonon DOS can be expensive
especially when :math:`N_{q}` or :math:`N_{\kappa}` is large.


(Atomic) participation ratio
----------------------------

Participation ratio (PR) and atomic participation ratio (APR) defined in the following may be useful to analyze the localized nature of the phonon mode :math:`\boldsymbol{q}j`.

* Participation ratio (PR)

.. math::

    PR_{\boldsymbol{q}j} = \left(\sum_{\kappa}^{N_{\kappa}} \frac{|\boldsymbol{e}(\kappa;\boldsymbol{q}j)|^{2}}{M_{\kappa}}\right)^{2} \Bigg/
    N_{\kappa} \sum_{\kappa}^{N_{\kappa}} \frac{|\boldsymbol{e}(\kappa;\boldsymbol{q}j)|^{4}}{M_{\kappa}^{2}}

* Atomic participation ratio (APR)

.. math::

    APR_{\boldsymbol{q}j,\kappa} = \frac{|\boldsymbol{e}(\kappa;\boldsymbol{q}j)|^{2}}{M_{\kappa}} \Bigg/ \left(  N_{\kappa} \sum_{\kappa}^{N_{\kappa}} \frac{|\boldsymbol{e}(\kappa;\boldsymbol{q}j)|^{4}}{M_{\kappa}^{2}} \right)^{1/2}

For an extended eigenmode, the PR value is of order 1, whereas for a localized eigenmodes PR is of order :math:`1/N_{\kappa}` [3]_. APR is an atomic decomposition of PR that satisfies :math:`PR_{\boldsymbol{q}j} = \sum_{\kappa} (APR_{\boldsymbol{q}j,\kappa})^{2}`. To print the PR and APR, please set ``MODE = phonons`` and ``PRINTPR = 1`` in the ``&analysis`` entry field. 

Scattering phase space
-----------------------

When *KPMODE* = 2 and ``SPS = 1``, the three-phonon scattering phase space :math:`P_{3}` is calculated and saved to the file ``PREFIX``.sps. :math:`P_{3}` is defined as

.. math::
    
    P_{3}(\boldsymbol{q}j) = \frac{1}{3m^{3}} (2P_{3}^{(+)}(\boldsymbol{q}j) + P_{3}^{(-)}(\boldsymbol{q}j)),

where :math:`m` is the number of phonon branches and 

.. math::
    
    P_{3}^{(\pm)}(\boldsymbol{q}j) = \frac{1}{N_{q}}\sum_{\boldsymbol{q}_{1},\boldsymbol{q}_{2}, j_{1}, j_{2}}\delta(\omega_{\boldsymbol{q}j}\pm\omega_{\boldsymbol{q}_{1}j_{1}}-\omega_{\boldsymbol{q}_{2}j_{2}})\delta_{\boldsymbol{q}\pm\boldsymbol{q}_{1},\boldsymbol{q}_{2}+\boldsymbol{G}}.

*anphon* also print the total scattering phase space

.. math::

    P_{3} = \frac{1}{N_{q}}\sum_{\boldsymbol{q}j} P_{3}(\boldsymbol{q}j).

When ``SPS = 2``, the three-phonon scattering phase space with the occupation factor :math:`W_{3}^{(\pm)}` will be calculated and saved to the file ``PREFIX``.sps_Bose. :math:`W_{3}^{(\pm)}` is defined as

.. math::

    W_{3}^{(\pm)}(\boldsymbol{q}j) = \frac{1}{N_{q}}{\sum_{\boldsymbol{q}_{1},\boldsymbol{q}_{2}, j_{1}, j_{2}}}
    \left\{
      \begin{array}{c}
      n_{2} - n_{1} \\
      n_{1} + n_{2} + 1
      \end{array}
    \right\}
    \delta(\omega_{\boldsymbol{q}j}\pm\omega_{\boldsymbol{q}_{1}j_{1}}-\omega_{\boldsymbol{q}_{2}j_{2}})\delta_{\boldsymbol{q}\pm\boldsymbol{q}_{1},\boldsymbol{q}_{2}+\boldsymbol{G}}.

Here, :math:`n_{1}=n(\omega_{\boldsymbol{q}_{1}j_{1}})` and :math:`n_{2}=n(\omega_{\boldsymbol{q}_{2}j_{2}})` where :math:`n(\omega) = \frac{1}{e^{\hbar\omega/k_B T}-1}` is the Bose-Einstein distribution function. Since :math:`n(\omega)` is temperature dependent, :math:`W_{3}^{(\pm)}` is also temperature dependent. The file ``PREFIX``.sps_Bose contains :math:`W_{3}^{(\pm)}` for all phonon modes at various temperatures specified with ``TMIN``, ``TMAX``, and ``DT`` tags.

Gr\ |umulaut_u|\ neisen parameter
---------------------------------

The mode Gr\ |umulaut_u|\ neisen parameter, defined as :math:`\gamma_{\boldsymbol{q}j} = - \frac{\partial \log{\omega_{\boldsymbol{q}j}}}{\partial \log{V}}`, 
is calculated by

.. math::

    \gamma_{\boldsymbol{q}j}= -\frac{(\boldsymbol{e}_{\boldsymbol{q}j}^{*})^{\mathrm{T}} \delta D(\boldsymbol{q})\boldsymbol{e}_{\boldsymbol{q}j}}{6\omega_{\boldsymbol{q}j}^{2}},

where :math:`\delta D(\boldsymbol{q})` is a change in the dynamical matrix due to a volume change :math:`\delta V`, 
which is given by

.. math::
    :nowrap:

    \begin{align}
     \delta D_{\mu\nu}(\kappa\kappa^{\prime};\boldsymbol{q}) &= \frac{1}{\sqrt{M_{\kappa}M_{\kappa^{\prime}}}}
     \sum_{\ell^{\prime}}\delta\Phi_{\mu\nu}(\ell\kappa;\ell^{\prime}\kappa^{\prime})\exp{\left[i\boldsymbol{q}\cdot(\boldsymbol{r}(\ell^{\prime})-\boldsymbol{r}(\ell))\right]},\\
     \delta\Phi_{\mu\nu}(\ell\kappa;\ell^{\prime}\kappa^{\prime}) 
     &= \sum_{\ell^{\prime\prime},\kappa^{\prime\prime},\lambda}\Phi_{\mu\nu\lambda}(\ell\kappa;\ell^{\prime}\kappa^{\prime};\ell^{\prime\prime}\kappa^{\prime\prime})r_{\lambda}(\ell^{\prime\prime}\kappa^{\prime\prime}).
    \end{align}

Please set ``GRUNEISEN = 1`` and give an appropriate ``FCSFILE`` file containing cubic IFCs to print Gr\ |umulaut_u|\ neisen parameters.


Anharmonic self-energy
-----------------------

The anharmonic self-energy due to cubic anharmonicity to the lowest order is given by

.. math::
    :label: self3

    \Sigma_{\boldsymbol{q}j}(i\omega_m) &= \frac{1}{2\hbar^{2}}\sum_{\boldsymbol{q}_{1},\boldsymbol{q}_{2}}\sum_{j_{1},j_{2}}
    |V^{(3)}_{-\boldsymbol{q}j,\boldsymbol{q}_{1}j_{1},\boldsymbol{q}_{2}j_{2}}|^{2} \notag \\
    & \times \left[ \frac{n_{1}+n_{2} + 1}{i\omega_{m} + \omega_{1} + \omega_{2}} - \frac{n_{1}+n_{2} + 1}{i\omega_{m} - \omega_{1} - \omega_{2}} 
    + \frac{n_{1}-n_{2}}{i\omega_{m} - \omega_{1} + \omega_{2}} - \frac{n_{1}-n_{2}}{i\omega_{m} + \omega_{1} - \omega_{2}} \right],
    
where :math:`i\omega_{m}` is the Matsubara frequency. In equation :eq:`self3`, we simply denoted :math:`\omega_{\boldsymbol{q}_{i}j_{i}}` as :math:`\omega_{i}`. The matrix element :math:`V^{(3)}` is given by

.. math::
  
    V^{(3)}_{\boldsymbol{q}j,\boldsymbol{q}^{\prime}j^{\prime},\boldsymbol{q}^{\prime\prime}j^{\prime\prime}} 
    & = \left( \frac{\hbar}{2N_{q}}\right)^{\frac{3}{2}}
    \frac{1}{\sqrt{\omega_{\boldsymbol{q}j}\omega_{\boldsymbol{q}^{\prime}j^{\prime}}\omega_{\boldsymbol{q}^{\prime\prime}j^{\prime\prime}}}}
    \sum_{\ell,\ell^{\prime},\ell^{\prime\prime}}
    \exp{\left[\mathrm{i}(\boldsymbol{q}\cdot\boldsymbol{r}(\ell)+\boldsymbol{q}^{\prime}\cdot\boldsymbol{r}(\ell^{\prime})+\boldsymbol{q}^{\prime\prime}\cdot\boldsymbol{r}(\ell^{\prime\prime}))\right]} \notag \\
    & \times \sum_{\kappa,\kappa^{\prime},\kappa^{\prime\prime}} \frac{1}{\sqrt{M_{\kappa}M_{\kappa^{\prime}}M_{\kappa^{\prime\prime}}}}
    \sum_{\mu,\nu,\lambda}
    \Phi_{\mu\nu\lambda}(\ell\kappa;\ell^{\prime}\kappa^{\prime};\ell^{\prime\prime}\kappa^{\prime\prime}) 
    e_{\mu}(\kappa;\boldsymbol{q}j)e_{\nu}(\kappa^{\prime};\boldsymbol{q}^{\prime}j^{\prime})e_{\lambda}(\kappa^{\prime\prime};\boldsymbol{q}^{\prime\prime}j^{\prime\prime}) \; ,
    

which becomes zero unless :math:`\boldsymbol{q}+\boldsymbol{q}^{\prime}+\boldsymbol{q}^{\prime\prime}` is an integral multiple of :math:`\boldsymbol{G}=n_{1}\boldsymbol{b}_{1}+n_{2}\boldsymbol{b}_{2}+n_{3}\boldsymbol{b}_{3}`.
Phonon linewidth :math:`\Gamma_{\boldsymbol{q}j}`, which is the imaginary part of the phonon self-energy, can be obtained by the analytic continuation to the real axis (:math:`i\omega_{m}\to \omega + i0^{+}`) as

.. math::
    :label: selfmod

     \Gamma^{\mathrm{anh}}_{\boldsymbol{q}j}(\omega) &= \frac{\pi}{2\hbar^{2}}\sum_{\boldsymbol{q}_{1},\boldsymbol{q}_{2}}\sum_{j_{1},j_{2}}
     |V^{(3)}_{-\boldsymbol{q}j,\boldsymbol{q}_{1}j_{1},\boldsymbol{q}_{2}j_{2}}|^{2} \notag \\
     & \times \left[ -(n_{1}+n_{2} + 1)\delta{(\omega + \omega_{1} + \omega_{2})} + (n_{1}+n_{2} + 1) \delta{(\omega - \omega_{1} - \omega_{2})} \right. \notag \\
     & \left. \hspace{12mm} - (n_{1}-n_{2})\delta{(\omega - \omega_{1} + \omega_{2})} + (n_{1}-n_{2})\delta{(\omega + \omega_{1} - \omega_{2})} \right].

The computation of equation :eq:`selfmod` is the most expensive part of the thermal conductivity calculations.
Therefore, we employ the crystal symmetry to reduce the number of triplet pairs :math:`(\boldsymbol{q}j,\boldsymbol{q}^{\prime}j^{\prime},\boldsymbol{q}^{\prime\prime}j^{\prime\prime})` of :math:`V^{(3)}` to calculate.
To disable the reduction, please set ``TRISYM = 0``.


Isotope scattering
------------------

The effect of isotope scatterings can be considered by the mass perturbation approach proposed by S. Tamura [4]_ by the ``ISOTOPE``-tag.
The corresponding phonon linewidth is given by

.. math::

    \Gamma_{\boldsymbol{q}j}^{\mathrm{iso}}(\omega)= \frac{\pi}{4N_{q}} \omega_{\boldsymbol{q}j}^{2}\sum_{\boldsymbol{q}_{1},j_{1}}\delta(\omega-\omega_{\boldsymbol{q}_{1}j_{1}})
    \sum_{\kappa}g_{2}(\kappa)|\boldsymbol{e}^{*}(\kappa;\boldsymbol{q}_{1}\boldsymbol{j}_{1})\cdot\boldsymbol{e}(\kappa;\boldsymbol{q}\boldsymbol{j})|^{2},

where :math:`g_{2}` is a dimensionless factor given by

.. math::

    g_{2}(\kappa)=\sum_{i}f_{i}(\kappa)\left(1 - \frac{m_{i}(\kappa)}{M_{\kappa}}\right)^{2}.

Here, :math:`f_{i}` is the fraction of the :math:`i`\ th isotope of an element having mass :math:`m_i`, 
and :math:`M_{\kappa}=\sum_{i}f_{i}m_{i}(\kappa)` is the average mass, respectively.
The :math:`g_{2}` values should be provided by the ``ISOFACT``-tag.
The average mass :math:`M_{\kappa}` is substituted by the value specified in the ``MASS``-tag.

.. _kappa:

Lattice thermal conductivity (Peierls term)
-------------------------------------------

The lattice thermal conductivity tensor :math:`\kappa_{\mathrm{ph}}^{\mu\nu}(T)` is estimated within the relaxation-time approximation as

.. math::
  
  \kappa_{\mathrm{ph}}^{\mu\nu}(T) = \frac{1}{V N_{q}} \sum_{\boldsymbol{q},j}c_{\boldsymbol{q}j}(T)v_{\boldsymbol{q}j}^{\mu}v_{\boldsymbol{q}j}^{\nu}\tau_{\boldsymbol{q}j}(T),

where :math:`V` is the unit cell volume, :math:`c_{\boldsymbol{q}j} = \hbar\omega_{\boldsymbol{q}j}\partial n_{\boldsymbol{q}j}/\partial T`, and :math:`\tau_{\boldsymbol{q}j}(T)` is the phonon lifetime.
The phonon lifetime is estimated using the Matthiessen's rule as

.. math::

    \tau_{\boldsymbol{q}j}^{-1}(T) = 2 (\Gamma_{\boldsymbol{q}j}^{\mathrm{anh}}(T) + \Gamma_{\boldsymbol{q}j}^{\mathrm{iso}}).

The lattice thermal conductivity is saved in the file ``PREFIX``.kl.

The spectra of the lattice thermal conductivity :math:`\kappa_{\mathrm{ph}}^{\mu\mu}(\omega)` can also be calculated by setting ``KAPPA_SPEC = 1`` in the ``&analysis`` field. :math:`\kappa_{\mathrm{ph}}^{\mu\mu}(\omega)` is defined as 

.. math::
    \kappa_{\mathrm{ph}}^{\mu\mu}(\omega) = \frac{1}{\Omega N_{q}}\sum_{\boldsymbol{q},j}c_{\boldsymbol{q}j}v_{\boldsymbol{q}j}^{\mu}v_{\boldsymbol{q}j}^{\mu}\tau_{\boldsymbol{q}j} \delta(\omega-\omega_{\boldsymbol{q}j}).

If we integrate this quantity over :math:`\omega`, we then obtain the bulk thermal conductivity, namely :math:`\kappa_{\mathrm{ph}}^{\mu\mu} = \int_{0}^{\infty} \kappa_{\mathrm{ph}}^{\mu\mu}(\omega) \; \mathrm{d}\omega`.

.. _cumulative_kappa:

Cumulative thermal conductivity
-------------------------------

The accumulative lattice thermal conductivity :math:`\kappa_{\mathrm{ph,acc}}^{\mu\nu}(L)` is defined as

.. math::
  
  \kappa_{\mathrm{ph,acc}}^{\mu\mu}(L) = \frac{1}{V N_{q}} \sum_{\boldsymbol{q},j}c_{\boldsymbol{q}j}v_{\boldsymbol{q}j}^{\mu}v_{\boldsymbol{q}j}^{\mu}\tau_{\boldsymbol{q}j}\Theta (L-|\boldsymbol{v}_{\boldsymbol{q}j}|\tau_{\boldsymbol{q}j}),

where :math:`\Theta(x)` is the step function. This quantity can be calculated by using the script ``analyze_phonons.py`` with ``--calc cumulative`` flag. 
One can also use another definition for the accumulative thermal conductivity:

.. math::
  
  \kappa_{\mathrm{ph,acc}}^{\mu\nu}(L) = \frac{1}{V N_{q}} \sum_{\boldsymbol{q},j}c_{\boldsymbol{q}j}v_{\boldsymbol{q}j}^{\mu}v_{\boldsymbol{q}j}^{\nu}\tau_{\boldsymbol{q}j}\Theta (L-|v_{\boldsymbol{q}j}^{\mu}|\tau_{\boldsymbol{q}j}).

In this case, the contribution to the total thermal conductivity is limited only from phonon modes whose mean-free-path along the :math:`\mu`\ -direction is smaller than :math:`L`.
To calculate this, please use the ``--calc cumulative2`` flag and specify the direction :math:`\mu` by the ``--direction`` option.

.. _kappa_coherent:

Coherent component of lattice thermal conductivity
--------------------------------------------------

The coherent components of lattice thermal conductivity (see Ref. [8]_), which are associated with the band off-diagonal components of the harmonic heat-flux operator, is calculated as 

.. math::
  
  \kappa_{\mathrm{c}}^{\mu\nu}(T) = \frac{1}{V N_{q}} \sum_{\substack{\boldsymbol{q},jj'\\ j\neq j'}}\frac{c_{\boldsymbol{q}j}\omega_{\boldsymbol{q}j'} + c_{\boldsymbol{q}j'}\omega_{\boldsymbol{q}j}}{\omega_{\boldsymbol{q}j}+ \omega_{\boldsymbol{q}j'}}  v_{\boldsymbol{q}jj'}^{\mu}v_{\boldsymbol{q}j'j}^{\nu} \frac{\Gamma_{\boldsymbol{q}j}+\Gamma_{\boldsymbol{q}j'}}{(\omega_{\boldsymbol{q}j}-\omega_{\boldsymbol{q}j'})^{2}+(\Gamma_{\boldsymbol{q}j}+\Gamma_{\boldsymbol{q}j'})^2},

where :math:`c_{\boldsymbol{q}j} = \hbar\omega_{\boldsymbol{q}j}\partial n_{\boldsymbol{q}j}/\partial T` and :math:`\Gamma_{\boldsymbol{q}j}` is the total phonon linewidth (half width) of phonon mode :math:`\boldsymbol{q}j`. 

:math:`\boldsymbol{v}_{\boldsymbol{q}jj'}` is a band off-diagonal generalization of the group velocity [9]_. When ``KAPPA_COHERENT = 1 | 2`` the coherent component is calculated and saved in ``PREFIX``.kl_coherent. When ``KAPPA_COHERENT = 2``, all components of the coherent term before summation are saved in ``PREFIX``.kc_elem.


Delta function
--------------

To compute the phonon DOSs and the imaginary part of phonon self-energies,
it is necessary to evaluate the Brillouin-zone integration containing Dirac's delta function.
For that purpose, we provide 3 options through the ``ISMEAR``-tag.

When ``ISMEAR = 0``, the delta function is replaced by the Lorentzian function as

.. math::
    
    \delta(\omega) \approx \frac{1}{\pi}\frac{\epsilon^{2}}{\omega^{2}+\epsilon^{2}}.

When ``ISMEAR = 1``, the delta function is replaced by the Gaussian function as

.. math::
    
    \delta(\omega) \approx \frac{1}{\sqrt{\pi}\epsilon}\exp{(-\omega^{2}/\epsilon^{2})},

which decays faster than the Lorentzian function. 
For both cases, :math:`\epsilon` should be given by the ``EPSILON``-tag, which must be chosen carefully
to avoid any unscientific results. :math:`\epsilon` should be small enough to capture detailed phonon structures 
such as phonon DOS or energy conservation surface related to three-phonon processes, but it should be large
enough to avoid unscientific oscillations. Choosing an appropriate value for :math:`\epsilon` is not a trivial task
since it may depend on the phonon structure and the density of :math:`\boldsymbol{q}` points.

To avoid such issues, the program *anphon* employs the tetrahedron method [5]_ by default (``ISMEAR = -1``)
for numerical evaluations of Brillouin zone integration containing :math:`\delta(\omega)`.
When the tetrahedron method is used, the ``EPSILON``-tag is neglected.
We recommend using the tetrahedron method whenever possible.

.. _formalism_SCPH:

Self-consistent phonon (SCPH) calculation
-----------------------------------------

The self-consistent phonon mode (``MODE = SCPH``) computes temperature-dependent phonon frequencies by solving the following equation self-consistently [6]_:

.. math::
    :label: scph_v_iter

    V_{\boldsymbol{q}ij}^{[n]} = \omega_{\boldsymbol{q}i}^{2}\delta_{ij}+\frac{1}{2}\sum_{\boldsymbol{q}_{1},k,\ell}F_{\boldsymbol{q}\boldsymbol{q}_{1},ijk\ell}\mathcal{K}_{\boldsymbol{q}_{1},k\ell}^{[n-1]}.

Here, :math:`\omega_{\boldsymbol{q}j}` is the harmonic phonon frequency and :math:`F_{\boldsymbol{q}\boldsymbol{q}_{1},ijk\ell} = \Phi(-\boldsymbol{q}i;\boldsymbol{q}j;\boldsymbol{q}_{1}k;-\boldsymbol{q}_{1}\ell)` is the reciprocal representation of fourth-order force constants computed using the harmonic eigenvectors. The updated phonon frequency in the :math:`n`\ th iteration is obtained by diagonalizing the matrix :math:`V_{\boldsymbol{q}ij}^{[n]}` as 

.. math::

    \Lambda^{[n]}_{\boldsymbol{q}} = C^{[n]\dagger}_{\boldsymbol{q}}V^{[n]}_{\boldsymbol{q}}C^{[n]}_{\boldsymbol{q}},

where :math:`\omega_{\boldsymbol{q}j}^{[n]} = (\Lambda^{[n]}_{\boldsymbol{q}jj})^{\frac{1}{2}}` and :math:`C^{[n]}_{\boldsymbol{q}}` is the unitary matrix that transforms the harmonic phonon eigenvectors into anharmonic ones as :math:`W^{[n]}_{\boldsymbol{q}} = W_{\boldsymbol{q}}C^{[n]}_{\boldsymbol{q}}`. The matrix :math:`\mathcal{K}` in Eq. :eq:`scph_v_iter` is defined as

.. math::

    \mathcal{K}_{\boldsymbol{q},ij}^{[n]} &= \alpha K_{\boldsymbol{q},ij}^{[n]} + (1-\alpha) K_{\boldsymbol{q},ij}^{[n-1]},  \\
    K_{\boldsymbol{q},ij}^{[n]} 
    &= \sum_{k} C_{\boldsymbol{q},ik}^{[n]} C_{\boldsymbol{q},jk}^{[n]*} \frac{\hbar\big[1+2n(\omega_{\boldsymbol{q}k}^{[n]})\big]}{2\omega_{\boldsymbol{q}k}^{[n]}}
    = (C^{[n]}_{\boldsymbol{q}} Q^{[n]}_{\boldsymbol{q}} C^{[n]\dagger}_{\boldsymbol{q}})_{ij},\\
    Q_{\boldsymbol{q},ij}^{[n]} 
    &= \frac{\hbar\big[1+2n(\omega_{\boldsymbol{q}i}^{[n]})\big]}{2\omega_{\boldsymbol{q}i}^{[n]}}\delta_{ij}.

:math:`\alpha` is the mixing parameter, which can be changed via the ``MIXALPHA`` tag, and :math:`n(\omega)` is the Bose-Einstein distribution function.

When ``CLASSICAL = 1`` is given in input, the expectation value of the mean square displacement is computed using classical statistics, where the :math:`Q_{\boldsymbol{q}}` matrix is replaced by

.. math::

    Q_{\boldsymbol{q},ij}^{[n]} = \frac{k_{B}T}{(\omega_{\boldsymbol{q}i}^{[n]})^{2}} \delta_{ij}.

The SCPH equation is solved on the irreducible :math:`\boldsymbol{q}` grid defined by the ``KMESH_INTERPOLATE`` tag.
The :math:`\boldsymbol{q}_{1}` grid in Eq. :eq:`scph_v_iter`, given  by the ``KMESH_SCPH`` tag,  
can be finer than the :math:`\boldsymbol{q}` grid. After the SCPH iteration converges, the code computes the anharmonic correction to the harmonic force constant :math:`\Delta D(\boldsymbol{r}(\ell))` as follows:

.. math::
    
    &\Delta D(\boldsymbol{r}(\ell)) = \frac{1}{N_{q}}\sum_{\boldsymbol{q}} \Delta D(\boldsymbol{q}) e^{-i\boldsymbol{q}\cdot\boldsymbol{r}(\ell)}, \\
    &\Delta D(\boldsymbol{q}) = D_{\mathrm{SCPH}}(\boldsymbol{q}) - D_{\mathrm{Harmonic}}(\boldsymbol{q}), \\
    &D_{\mathrm{SCPH}}(\boldsymbol{q}) = W_{\boldsymbol{q}}C_{\boldsymbol{q}}^{[n]}\Lambda_{\boldsymbol{q}}^{[n]}C_{\boldsymbol{q}}^{[n]\dagger}W_{\boldsymbol{q}}^{\dagger}.

:math:`\Delta D(\boldsymbol{r}(\ell))` is saved in ``PREFIX.scph_dfc2``.

The most computationally expensive part is the calculation of matrix elements of :math:`F_{\boldsymbol{q}\boldsymbol{q}_{1},ijk\ell}`.
When ``SELF_OFFDIAG = 0`` (default), the code only computes the elements of :math:`F_{\boldsymbol{q}\boldsymbol{q}_{1},iikk}`. 
Therefore, the computational complexity is :math:`\mathcal{O}(N_{q}^{\mathrm{irred.}}N_{q_{1}}m^{2})`.
When ``SELF_OFFDIAG = 1``, the off-diagonal elements are also calculated, and the computational complexity is :math:`\mathcal{O}(N_{q}^{\mathrm{irred.}}N_{q_{1}}m^{4})`. 



````

.. [1] K\. Parlinski, Z. Q. Li, and Y. Kawazoe, Phys. Rev. Lett. **81**, 3298 (1998).

.. [2] Y\. Wang *et al.*, J. Phys.: Condens. Matter **22**, 202201 (2010).

.. [3] J\. Hafner and M. Krajci, J. Phys.: Condens. Matter **5**, 2489 (1993).

.. [4] S\. -I. Tamura, Phys. Rev. B **27**, 858 (1983).

.. [5] P\. E. Bl\ |umulaut_o|\ chl, O. Jepsen, and O. K. Andersen, Phys. Rev. B **49**, 16223 (1994).

.. [6] T\. Tadano and S. Tsuneyuki, Phys. Rev. B **92**, 054301 (2015).

.. [7] Y\. Oba, T. Tadano, R. Akashi, and S. Tsuneyuki, Phys. Rev. Materials **3**, 033601 (2019).

.. [8] M\. Simoncelli, N. Marzari, and F. Mauri, Nat. Phys. **15**, 809 (2019).

.. [9] P\. B. Allen and J. L. Feldman, Phys. Rev. B **48**, 12581 (1993).

.. [10] X\. Gonze and C. Lee, Phys. Rev. B **55**, 10355 (1997).
  

