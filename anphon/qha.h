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
#include "scph.h"

namespace PHON_NS
{
class Qha: protected Pointers
{
public:
    Qha(class PHON *phon);

    ~Qha();

    void setup_qha();

    unsigned int kmesh_qha[3];
    unsigned int kmesh_interpolate[3];

    // optimization scheme used in QHA
    int qha_scheme;

    bool restart_qha;
    bool warmstart_qha;
    bool lower_temp;
    double tolerance_qha;

    void exec_qha_optimization();

private:
    void set_default_variables();

    void deallocate_variables();

    void setup_kmesh();

    void setup_eigvecs();

    void setup_pp_interaction();

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

    KpointMeshUniform *kmesh_coarse = nullptr;
    KpointMeshUniform *kmesh_dense = nullptr;
    std::vector<int> kmap_coarse_to_dense;

    // Information of harmonic dynamical matrix
    double **omega2_harmonic;
    std::complex<double> ***evec_harmonic;
    MinimumDistList ***mindist_list_qha;

    // Information for calculating the ph-ph interaction coefficients
    std::complex<double> *phi3_reciprocal, *phi4_reciprocal;

    // Local variables for handling symmetry of dynamical matrix
    std::complex<double> ****mat_transform_sym;

    // Phase shift
    PhaseFactorStorage *phase_factor_qha;

    std::vector<Eigen::MatrixXcd> dymat_harm_short, dymat_harm_long;

    void zerofill_harmonic_dymat_renormalize(std::complex<double> ****, unsigned int);
};
} // namespace PHON_NS
