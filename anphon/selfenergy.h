/*
 selfenergy.h

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <complex>
#include <string>
#include <vector>
#include "kpoint.h"

namespace PHON_NS
{
class AnharmonicCore;
class SymmetryOperation;

// Anharmonic phonon self-energy diagrams. No Pointers base: the run-wide
// inputs are stored once by setup_selfenergy (called from
// PHON::execute_kappa), and the per-call data enter as arguments.
class Selfenergy
{
public:
    Selfenergy();

    ~Selfenergy();

    void setup_selfenergy(unsigned int ns_in, double epsilon_in, bool classical_in,
                          const std::vector<SymmetryOperation> &symmlist_in, AnharmonicCore &anharmonic_core_in,
                          int my_rank_in, int nprocs_in);

    void selfenergy_tadpole(const unsigned int N, const double *T, const double omega, const unsigned int knum,
                            const unsigned int snum, const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                            const std::complex<double> *const *const *evec_in, std::complex<double> *ret) const;

    void selfenergy_a(const unsigned int N, const double *T, const double omega, const unsigned int knum,
                      const unsigned int snum, const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                      const std::complex<double> *const *const *evec_in, std::complex<double> *ret) const;

    void selfenergy_b(const unsigned int N, const double *T, const double omega, const unsigned int knum,
                      const unsigned int snum, const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                      const std::complex<double> *const *const *evec_in, std::complex<double> *ret) const;

    void selfenergy_c(const unsigned int N, const double *T, const double omega, const unsigned int knum,
                      const unsigned int snum, const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                      const std::complex<double> *const *const *evec_in, std::complex<double> *ret) const;

    void selfenergy_c_mod(const unsigned int N, const double *T, const double omega, const unsigned int knum,
                          const unsigned int snum, const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                          const std::complex<double> *const *const *evec_in, std::complex<double> *ret) const;

    void selfenergy_d(const unsigned int N, const double *T, const double omega, const unsigned int knum,
                      const unsigned int snum, const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                      const std::complex<double> *const *const *evec_in, std::complex<double> *ret) const;

    void selfenergy_e(const unsigned int N, const double *T, const double omega, const unsigned int knum,
                      const unsigned int snum, const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                      const std::complex<double> *const *const *evec_in, std::complex<double> *ret) const;

    void selfenergy_f(const unsigned int N, const double *T, const double omega, const unsigned int knum,
                      const unsigned int snum, const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                      const std::complex<double> *const *const *evec_in, std::complex<double> *ret) const;

    void selfenergy_g(const unsigned int N, const double *T, const double omega, const unsigned int knum,
                      const unsigned int snum, const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                      const std::complex<double> *const *const *evec_in, std::complex<double> *ret) const;

    void selfenergy_h(const unsigned int N, const double *T, const double omega, const unsigned int knum,
                      const unsigned int snum, const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                      const std::complex<double> *const *const *evec_in, std::complex<double> *ret) const;

    void selfenergy_i(const unsigned int N, const double *T, const double omega, const unsigned int knum,
                      const unsigned int snum, const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                      const std::complex<double> *const *const *evec_in, std::complex<double> *ret) const;

    void selfenergy_j(const unsigned int N, const double *T, const double omega, const unsigned int knum,
                      const unsigned int snum, const KpointMeshUniform *kmesh_in, const double *const *eval_in,
                      const std::complex<double> *const *const *evec_in, std::complex<double> *ret) const;

private:
    unsigned int ns;
    double epsilon;
    bool classical;
    const std::vector<SymmetryOperation> *symmlist = nullptr;
    AnharmonicCore *anharmonic_core = nullptr;
    int my_rank;
    int nprocs;

    void mpi_reduce_complex(unsigned int, std::complex<double> *, std::complex<double> *) const;
};
} // namespace PHON_NS
