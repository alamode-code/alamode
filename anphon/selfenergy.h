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
#include "anharmonic_core.h"
#include "kpoint.h"

namespace PHON_NS
{
class AnharmonicCore;
class PhaseFactorCache;
class SymmetryOperation;

// Anharmonic phonon self-energy diagrams. No Pointers base: the run-wide
// inputs are stored once by setup_selfenergy (called from
// PHON::execute_kappa and Scph::bubble_correction), and the per-call data enter as arguments.
class Selfenergy
{
public:
    Selfenergy();

    ~Selfenergy();

    void setup_selfenergy(unsigned int ns_in, double epsilon_in, bool classical_in,
                          const std::vector<SymmetryOperation> &symmlist_in, AnharmonicCore &anharmonic_core_in,
                          int my_rank_in, int nprocs_in);

    // Off-mesh targets: external legs from (xq, omega_q, evec_q), the bubble partner
    // q - k from the shifted grid; same formulas and prefactors as the mesh kernels.
    void selfenergy_a_at(const unsigned int N, const double *T, const double omega, const double *xq,
                         const double omega_q, const std::complex<double> *evec_q, const KpointMeshUniform *kmesh_in,
                         const double *const *eval_in, const std::complex<double> *const *const *evec_in,
                         const AnharmonicCore::ShiftedGrid &sg, std::complex<double> *ret) const;

    void selfenergy_tadpole_at(const unsigned int N, const double *T, const double *xq, const double omega_q,
                               const std::complex<double> *evec_q, const KpointMeshUniform *kmesh_in,
                               const double *const *eval_in, const std::complex<double> *const *const *evec_in,
                               std::complex<double> *ret) const;

    void selfenergy_b_at(const unsigned int N, const double *T, const double *xq, const double omega_q,
                         const std::complex<double> *evec_q, const KpointMeshUniform *kmesh_in,
                         const double *const *eval_in, const std::complex<double> *const *const *evec_in,
                         std::complex<double> *ret) const;

    // Full bubble matrix in the mode basis at the mesh point knum (INTERPOLATE):
    // sig[iomega][j][j'] = (1/16 N_k) sum V3(-q j; k s1; q-k s2) V3(-q j'; k s1; q-k s2)^*
    // [f1 Omega0 + f2 Omega1] at omega + i epsilon. Reduced to rank 0.
    void bubble_matrix(const double Temp, const unsigned int knum, const KpointMeshUniform *kmesh_in,
                       const double *const *eval_in, const std::complex<double> *const *const *evec_in,
                       const unsigned int nomega, const double *omega, NDArray<std::complex<double>, 3> &sig) const;

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

    std::vector<std::complex<double>> get_bubble_selfenergy(const KpointMeshUniform *kmesh_in, const unsigned int ns_in,
                                                            const double *const *eval_in,
                                                            const std::complex<double> *const *const *evec_in,
                                                            const unsigned int knum, const unsigned int snum,
                                                            const double temp_in,
                                                            const std::vector<std::complex<double>> &omegalist,
                                                            const PhaseFactorCache *phase_cache_in) const;

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
