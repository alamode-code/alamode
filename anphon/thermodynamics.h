/*
 phonon_thermodynamics.h

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <complex>
#include <vector>
#include "kpoint.h"
#include "ndarray.h"

namespace PHON_NS
{
class System;
class AnharmonicCore;
class SymmetryOperation;
class DymatEigenValue;

// Thermodynamic functions of the phonon gas. No Pointers base: the methods
// are (mostly static or const) functions of their explicit arguments; the
// parser fills the public config members and setup() broadcasts them.
class Thermodynamics
{
public:
    Thermodynamics();

    ~Thermodynamics();

    // Conversion constant K -> Ry (k_Boltzmann / Ryd); fixed at compile time.
    static const double T_to_Ryd;
    bool classical;
    bool calc_FE_bubble;
    NDArray<double, 1> FE_bubble;

    void setup();

    double Cv(const double omega, const double temp_in) const;

    static double Cv_classical(const double omega, const double temp_in);

    static double fB(const double omega, const double temp_in);

    static double fC(const double omega, const double temp_in);

    double Cv_tot(const double temp_in, const unsigned int nk_irred, const unsigned int ns,
                  const std::vector<std::vector<KpointList>> &kp_irred, const double *weight_k_irred,
                  const double *const *eval_in) const;

    double Cv_anharm_correction(const double temp_in, const unsigned int nk_irred, const unsigned int ns,
                                const std::vector<std::vector<KpointList>> &kp_irred, const double *weight_k_irred,
                                const double *const *eval_in, const double *const *del_eval_in) const;

    double internal_energy(const double temp_in, const unsigned int nk_irred, const unsigned int ns,
                           const std::vector<std::vector<KpointList>> &kp_irred, const double *weight_k_irred,
                           const double *const *eval_in) const;

    double vibrational_entropy(const double temp_in, const unsigned int nk_irred, const unsigned int ns,
                               const std::vector<std::vector<KpointList>> &kp_irred, const double *weight_k_irred,
                               const double *const *eval_in) const;

    double free_energy_QHA(const double temp_in, const unsigned int nk_irred, const unsigned int ns,
                           const std::vector<std::vector<KpointList>> &kp_irred, const double *weight_k_irred,
                           const double *const *eval_in) const;

    double disp2_avg(const double T_in, const unsigned int ncrd1, const unsigned int ncrd2, const unsigned int nk,
                     const unsigned int ns, const double *const *xk_in, const double *const *eval_in,
                     std::complex<double> ***evec_in, const System &system_in) const;

    double disp_corrfunc(const double T_in, const unsigned int ncrd1, const unsigned int ncrd2,
                         const double cell_shift[3], const unsigned int nk, const unsigned int ns,
                         const double *const *xk_in, const double *const *eval_in, std::complex<double> ***evec_in,
                         const System &system_in) const;

    double coth_T(double, double) const;

    void compute_free_energy_bubble(const System &system_in, const KpointMeshUniform &kmesh_dos_in,
                                    const DymatEigenValue &dymat_dos_in,
                                    const std::vector<SymmetryOperation> &symmlist_in,
                                    AnharmonicCore &anharmonic_core_in, unsigned int ns_in, int my_rank_in,
                                    int nprocs_in);

    void compute_FE_bubble(const double *const *eval, const std::complex<double> *const *const *evec,
                           double *FE_bubble_out, const System &system_in, const KpointMeshUniform &kmesh_dos_in,
                           const std::vector<SymmetryOperation> &symmlist_in, AnharmonicCore &anharmonic_core_in,
                           unsigned int ns_in, int my_rank_in, int nprocs_in) const;

    void compute_FE_bubble_SCPH(double ***eval_in, std::complex<double> ****evec_in, double *FE_bubble,
                                const System &system_in, const KpointMeshUniform &kmesh_dos_in,
                                const std::vector<SymmetryOperation> &symmlist_in, AnharmonicCore &anharmonic_core_in,
                                unsigned int ns_in, int my_rank_in, int nprocs_in) const;

    double FE_scph_correction(unsigned int, double **, std::complex<double> ***, double **, std::complex<double> ***,
                              const KpointMeshUniform &kmesh_dos_in, unsigned int ns_in, const System &system_in) const;

    double compute_FE_total(unsigned int, double, double, double v0_renorm, bool is_scph_mode) const;
};

} // namespace PHON_NS
