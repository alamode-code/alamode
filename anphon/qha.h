/*
 qha.h

 Copyright (c) 2022 Ryota Masuki, Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <complex>
#include "anharmonic_core.h"
#include "kpoint.h"
#include "pointers.h"
#include "relaxation_types.h"
#include "scph_qha_common.h"

namespace PHON_NS
{
class Qha: protected ScphQhaCommon
{
public:
    Qha(class PHON *phon);

    ~Qha();

    void setup_qha();

    unsigned int kmesh_qha[3];
    unsigned int kmesh_interpolate[3];

    // optimization scheme used in QHA
    QhaScheme qha_scheme = QhaScheme::Standard;

    bool restart_qha;
    bool warmstart_qha;
    bool lower_temp;
    double tolerance_qha;

    void exec_qha_optimization();

    using ScphQhaCommon::ialgo;
    using ScphQhaCommon::selfenergy_offdiagonal;

private:
    void set_default_variables();

    void exec_QHA_relax_main(std::complex<double> ****, std::complex<double> ****);

    void exec_perturbative_QHA(std::complex<double> ****, std::complex<double> ****);

    void calc_del_v0_del_umn_vib(std::complex<double> *, std::complex<double> ***, double);


    void calculate_del_v1_del_umn_renorm(std::complex<double> **, double **, std::complex<double> **,
                                         std::complex<double> **, std::complex<double> **, std::complex<double> ***,
                                         std::complex<double> ***, std::complex<double> ****, double *);

    void calculate_C2_array_renorm(double **, double **, double **, double **, double ***, std::complex<double> **,
                                   std::complex<double> **, std::complex<double> ***, double *);

    void calculate_C2_array_ZSISA(double **, double **, std::complex<double> **, double **);

    void compute_ZSISA_stress(double **, std::complex<double> *, std::complex<double> ***, double **,
                              std::complex<double> *, std::complex<double> **, std::complex<double> *,
                              std::vector<int> &);

    void compute_vZSISA_stress(std::complex<double> *, double **, std::complex<double> *, std::complex<double> *,
                               double **);

    // QHA
    void compute_cmat(std::complex<double> ***, const std::complex<double> *const *const *const);

    void calc_v1_vib(std::complex<double> *, std::complex<double> ***, const double);
};
} // namespace PHON_NS
