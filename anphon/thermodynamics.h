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
#include "pointers.h"

namespace PHON_NS
{
class Thermodynamics: protected Pointers
{
public:
    Thermodynamics(class PHON *);

    ~Thermodynamics();

    double T_to_Ryd;
    bool classical;
    bool calc_FE_bubble;
    double *FE_bubble;

    void setup();

    double Cv(const double omega, const double temp_in) const;

    static double Cv_classical(const double omega, const double temp_in);

    double fB(const double omega, const double temp_in) const;

    double fC(const double omega, const double temp_in) const;

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
                     std::complex<double> ***evec_in) const;

    double disp_corrfunc(const double T_in, const unsigned int ncrd1, const unsigned int ncrd2,
                         const double cell_shift[3], const unsigned int nk, const unsigned int ns,
                         const double *const *xk_in, const double *const *eval_in,
                         std::complex<double> ***evec_in) const;

    double coth_T(double, double) const;

    void compute_free_energy_bubble();

    void compute_FE_bubble(double **, std::complex<double> ***, double *) const;

    void compute_FE_bubble_SCPH(double ***eval_in, std::complex<double> ****evec_in, double *FE_bubble) const;

    double FE_scph_correction(unsigned int, double **, std::complex<double> ***, double **,
                              std::complex<double> ***) const;

    double compute_FE_total(unsigned int, double, double) const;
};

} // namespace PHON_NS
