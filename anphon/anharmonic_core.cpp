/*
anharmonic_core.cpp

Copyright (c) 2014 Terumasa Tadano

This file is distributed under the terms of the MIT license.
Please see the file 'LICENCE.txt' in the root directory
or http://opensource.org/licenses/mit-license.php for information.
*/

#include "anharmonic_core.h"
#include <algorithm>
#include <array>
#include <boost/lexical_cast.hpp>
#include <utility>
#include <vector>
#include "constants.h"
#include "dynamical.h"
#include "error.h"
#include "fcs_phonon.h"
#include "integration.h"
#include "kpoint.h"
#include "mathfunctions.h"
#include "memory.h"
#include "mode_analysis.h"
#include "mpi_common.h"
#include "phonon_dos.h"
#include "system.h"
#include "thermodynamics.h"
#include "timer.h"
#include "write_phonons.h"

#ifdef _OPENMP

#include <omp.h>

#endif

using namespace PHON_NS;

AnharmonicCore::AnharmonicCore(PHON *phon) : Pointers(phon)
{
    set_default_variables();
}

AnharmonicCore::~AnharmonicCore()
{
    deallocate_variables();
};

void AnharmonicCore::set_default_variables()
{
    quartic_mode = 0;
    use_tuned_ver = true;
    use_triplet_symmetry = true;
    use_quartet_symmetry = true;
}

void AnharmonicCore::deallocate_variables()
{
    if (relvec_v3) {
        relvec_v3.clear();
    }
    if (relvec_v4) {
        relvec_v4.clear();
    }
    if (invmass_v3) {
        invmass_v3.clear();
    }
    if (invmass_v4) {
        invmass_v4.clear();
    }
    if (evec_index_v3) {
        evec_index_v3.clear();
    }
    if (evec_index_v4) {
        evec_index_v4.clear();
    }
    if (fcs_group_v3) {
        fcs_group_v3.clear();
    }
    if (fcs_group_v4) {
        fcs_group_v4.clear();
    }
    if (phi3_reciprocal) {
        phi3_reciprocal.clear();
    }
    if (phi4_reciprocal) {
        phi4_reciprocal.clear();
    }
    phase_storage_dos.reset();
}

void AnharmonicCore::setup()
{
    sym_permutation = true;
    use_tuned_ver = true;
    MPI_Bcast(&use_tuned_ver, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);

    if (fcs_phonon->maxorder >= 2) setup_cubic();
    if (fcs_phonon->maxorder >= 3) setup_quartic();

    if (mympi->my_rank == 0 && writes->getVerbosity() > 0 && fcs_phonon->maxorder >= 2) {
        std::cout << "  Number of distinct index groups of the anharmonic IFCs:\n";
        std::cout << "   Order 3 : " << ngroup_v3 << '\n';
        if (fcs_phonon->maxorder >= 3) std::cout << "   Order 4 : " << ngroup_v4 << '\n';
        std::cout << '\n';
    }

    if (!mode_analysis->calc_fstate_k && dos->kmesh_dos.get()) {
        phase_storage_dos = std::make_unique<PhaseFactorCache>(dos->kmesh_dos->nk_i);
        phase_storage_dos->create(use_tuned_ver);
    }
}

void AnharmonicCore::prepare_relative_vector(const std::vector<FcsArrayWithCell> &fcs_in, const int number_of_groups,
                                             std::vector<double> *fcs_group, std::vector<RelativeVector> *vec_out)
{
    double vecs[3][3];
    unsigned int icount = 0;

    const auto nsize = fcs_in.begin()->pairs.size();

    for (auto igroup = 0; igroup < number_of_groups; ++igroup) {

        unsigned int nsize_group = fcs_group[igroup].size();

        for (auto j = 0; j < nsize_group; ++j) {
            for (auto i = 0; i < nsize - 1; ++i) {
                for (auto k = 0; k < 3; ++k) {
                    // include tpi phase factor here
                    vecs[i][k] = tpi * fcs_in[icount].relvecs[i][k];
                }
            }
            if (nsize == 3) {
                vec_out[igroup].emplace_back(vecs[0], vecs[1]);
            } else if (nsize == 4) {
                vec_out[igroup].emplace_back(vecs[0], vecs[1], vecs[2]);
            }
            ++icount;
        }
    }
}

void AnharmonicCore::prepare_group_of_force_constants(const std::vector<FcsArrayWithCell> &fcs_in,
                                                      int &number_of_groups,
                                                      NDArray<std::vector<double>, 1> &fcs_group_out)
{
    // Find the number of groups which has different evecs.

    unsigned int i;

    number_of_groups = 0;

    if (fcs_in.empty()) {
        fcs_group_out.clear();
        return;
    }

    const auto nsize_pair = fcs_in.begin()->pairs.size();
    std::vector<int> arr_old(nsize_pair, -1);
    std::vector<int> arr_tmp(nsize_pair);

    for (const auto &it: fcs_in) {

        for (i = 0; i < nsize_pair; ++i) {
            arr_tmp[i] = it.pairs[i].index;
        }
        if (arr_tmp != arr_old) {
            ++number_of_groups;
            arr_old = arr_tmp;
        }
    }

    fcs_group_out.clear();
    fcs_group_out.resize(number_of_groups);

    int igroup = -1;

    arr_old.resize(nsize_pair, -1);

    for (const auto &it: fcs_in) {

        for (i = 0; i < nsize_pair; ++i) {
            arr_tmp[i] = it.pairs[i].index;
        }
        if (arr_tmp != arr_old) {
            ++igroup;
            arr_old = arr_tmp;
        }
        fcs_group_out[igroup].push_back(it.fcs_val);
    }
}

std::complex<double> AnharmonicCore::V3(const unsigned int ks[3])
{
    return V3(ks,
              dos->kmesh_dos->xk,
              dos->dymat_dos->get_eigenvalues(),
              dos->dymat_dos->get_eigenvectors(),
              this->phase_storage_dos.get());
}

std::complex<double> AnharmonicCore::V3(const unsigned int ks[3], const double *const *xk_in,
                                        const double *const *eval_in, const std::complex<double> *const *const *evec_in)
{
    return V3(ks, xk_in, eval_in, evec_in, this->phase_storage_dos.get());
}

std::complex<double> AnharmonicCore::V4(const unsigned int ks[4])
{
    return V4(ks,
              dos->kmesh_dos->xk,
              dos->dymat_dos->get_eigenvalues(),
              dos->dymat_dos->get_eigenvectors(),
              this->phase_storage_dos.get());
}

std::complex<double> AnharmonicCore::Phi3(const unsigned int ks[3])
{
    return Phi3(ks,
                dos->kmesh_dos->xk,
                dos->dymat_dos->get_eigenvalues(),
                dos->dymat_dos->get_eigenvectors(),
                this->phase_storage_dos.get());
}

std::complex<double> AnharmonicCore::Phi4(const unsigned int ks[4])
{
    return Phi4(ks,
                dos->kmesh_dos->xk,
                dos->dymat_dos->get_eigenvalues(),
                dos->dymat_dos->get_eigenvectors(),
                this->phase_storage_dos.get());
}

std::complex<double> AnharmonicCore::V3(const unsigned int ks[3], const double *const *xk_in,
                                        const double *const *eval_in, const std::complex<double> *const *const *evec_in,
                                        const PhaseFactorCache *phase_storage_in)
{
    int i;
    unsigned int kn[3], sn[3];
    const int ns = dynamical->neval;

    double omega[3];
    auto ret = std::complex<double>(0.0, 0.0);
    auto ret_re = 0.0;
    auto ret_im = 0.0;

    for (i = 0; i < 3; ++i) {
        kn[i] = ks[i] / ns;
        sn[i] = ks[i] % ns;
        omega[i] = eval_in[kn[i]][sn[i]];
    }

    // Return zero if any of the involving phonon has imaginary frequency
    if (omega[0] < eps8 || omega[1] < eps8 || omega[2] < eps8) return 0.0;

    if (kn[1] != kindex_phi3_stored[0] || kn[2] != kindex_phi3_stored[1]) {
        calc_phi3_reciprocal(xk_in[kn[1]],
                             xk_in[kn[2]],
                             ngroup_v3,
                             fcs_group_v3,
                             relvec_v3,
                             phase_storage_in,
                             phi3_reciprocal);
        kindex_phi3_stored[0] = kn[1];
        kindex_phi3_stored[1] = kn[2];
    }
#ifdef _OPENMP
#pragma omp parallel for private(ret), reduction(+ : ret_re, ret_im)
#endif
    for (i = 0; i < ngroup_v3; ++i) {
        ret = evec_in[kn[0]][sn[0]][evec_index_v3[i][0]] * evec_in[kn[1]][sn[1]][evec_index_v3[i][1]] *
              evec_in[kn[2]][sn[2]][evec_index_v3[i][2]] * invmass_v3[i] * phi3_reciprocal[i];
        ret_re += ret.real();
        ret_im += ret.imag();
    }

    return std::complex<double>(ret_re, ret_im) / std::sqrt(omega[0] * omega[1] * omega[2]);
}

std::complex<double> AnharmonicCore::V3(const unsigned int ks[3], const double *const *xk_in,
                                        const double *const *eval_in, const std::complex<double> *const *const *evec_in,
                                        std::complex<double> *phi3_work, int *kindex_work)
{
    return V3(ks, xk_in, eval_in, evec_in, this->phase_storage_dos.get(), phi3_work, kindex_work);
}

std::complex<double> AnharmonicCore::V3(const unsigned int ks[3], const double *const *xk_in,
                                        const double *const *eval_in, const std::complex<double> *const *const *evec_in,
                                        const PhaseFactorCache *phase_storage_in, std::complex<double> *phi3_work,
                                        int *kindex_work)
{
    // Thread-safe serial variant of V3 (see the header note): the caller
    // owns the per-triplet reciprocal-FC3 cache and no OpenMP region is
    // entered here, so independent threads may evaluate different triplets
    // concurrently. The members read below are immutable after setup.
    int i;
    unsigned int kn[3], sn[3];
    const int ns = dynamical->neval;

    double omega[3];

    for (i = 0; i < 3; ++i) {
        kn[i] = ks[i] / ns;
        sn[i] = ks[i] % ns;
        omega[i] = eval_in[kn[i]][sn[i]];
    }

    // Return zero if any of the involving phonon has imaginary frequency
    if (omega[0] < eps8 || omega[1] < eps8 || omega[2] < eps8) return std::complex<double>(0.0, 0.0);

    if (static_cast<int>(kn[1]) != kindex_work[0] || static_cast<int>(kn[2]) != kindex_work[1]) {
        calc_phi3_reciprocal(xk_in[kn[1]],
                             xk_in[kn[2]],
                             ngroup_v3,
                             fcs_group_v3,
                             relvec_v3,
                             phase_storage_in,
                             phi3_work,
                             false);
        kindex_work[0] = static_cast<int>(kn[1]);
        kindex_work[1] = static_cast<int>(kn[2]);
    }

    auto ret_re = 0.0;
    auto ret_im = 0.0;
    for (i = 0; i < ngroup_v3; ++i) {
        const auto ret = evec_in[kn[0]][sn[0]][evec_index_v3[i][0]] * evec_in[kn[1]][sn[1]][evec_index_v3[i][1]] *
                         evec_in[kn[2]][sn[2]][evec_index_v3[i][2]] * invmass_v3[i] * phi3_work[i];
        ret_re += ret.real();
        ret_im += ret.imag();
    }

    return std::complex<double>(ret_re, ret_im) / std::sqrt(omega[0] * omega[1] * omega[2]);
}

std::complex<double> AnharmonicCore::Phi3(const unsigned int ks[3], const double *const *xk_in,
                                          const double *const *eval_in,
                                          const std::complex<double> *const *const *evec_in,
                                          const PhaseFactorCache *phase_storage_in)
{
    int i;
    unsigned int kn[3], sn[3];
    const auto ns = dynamical->neval;

    double omega[3];
    std::complex<double> ret = std::complex<double>(0.0, 0.0);
    double ret_re = 0.0;
    double ret_im = 0.0;

    for (i = 0; i < 3; ++i) {
        kn[i] = ks[i] / ns;
        sn[i] = ks[i] % ns;
        omega[i] = eval_in[kn[i]][sn[i]];
    }

    if (kn[1] != kindex_phi3_stored[0] || kn[2] != kindex_phi3_stored[1]) {
        calc_phi3_reciprocal(xk_in[kn[1]],
                             xk_in[kn[2]],
                             ngroup_v3,
                             fcs_group_v3,
                             relvec_v3,
                             phase_storage_in,
                             phi3_reciprocal);

        kindex_phi3_stored[0] = kn[1];
        kindex_phi3_stored[1] = kn[2];
    }
#ifdef _OPENMP
#pragma omp parallel for private(ret), reduction(+ : ret_re, ret_im)
#endif
    for (i = 0; i < ngroup_v3; ++i) {
        ret = evec_in[kn[0]][sn[0]][evec_index_v3[i][0]] * evec_in[kn[1]][sn[1]][evec_index_v3[i][1]] *
              evec_in[kn[2]][sn[2]][evec_index_v3[i][2]] * invmass_v3[i] * phi3_reciprocal[i];
        ret_re += ret.real();
        ret_im += ret.imag();
    }

    return std::complex<double>(ret_re, ret_im);
}

void AnharmonicCore::calc_phi3_reciprocal(const double *xk1, const double *xk2, const int ngroup_v3_in,
                                          const std::vector<double> *fcs_group_v3_in,
                                          const std::vector<RelativeVector> *relvec_v3_in,
                                          const PhaseFactorCache *phase_storage_in, std::complex<double> *ret,
                                          const bool use_openmp)
{
    int i, j;
    double phase;
    std::complex<double> ret_in;
    unsigned int nsize_group;

    const auto tune_type_now = phase_storage_in->get_tune_type();

    if (tune_type_now == 1) {

#pragma omp parallel for private(ret_in, nsize_group, j, phase) if (use_openmp)
        for (i = 0; i < ngroup_v3_in; ++i) {

            ret_in = std::complex<double>(0.0, 0.0);
            nsize_group = fcs_group_v3_in[i].size();

            for (j = 0; j < nsize_group; ++j) {
                phase = relvec_v3_in[i][j].vecs[0][0] * xk1[0] + relvec_v3_in[i][j].vecs[0][1] * xk1[1] +
                        relvec_v3_in[i][j].vecs[0][2] * xk1[2] + relvec_v3_in[i][j].vecs[1][0] * xk2[0] +
                        relvec_v3_in[i][j].vecs[1][1] * xk2[1] + relvec_v3_in[i][j].vecs[1][2] * xk2[2];

                ret_in += fcs_group_v3_in[i][j] * phase_storage_in->get_exp_type1(phase);
            }
            ret[i] = ret_in;
        }

    } else if (tune_type_now == 2) {

        // Tuned version is used when nk1=nk2=nk3 doesn't hold.

        double phase3[3];

#pragma omp parallel for private(ret_in, nsize_group, j, phase3) if (use_openmp)
        for (i = 0; i < ngroup_v3_in; ++i) {

            ret_in = std::complex<double>(0.0, 0.0);
            nsize_group = fcs_group_v3_in[i].size();

            for (j = 0; j < nsize_group; ++j) {
                for (auto ii = 0; ii < 3; ++ii) {
                    phase3[ii] = relvec_v3_in[i][j].vecs[0][ii] * xk1[ii] + relvec_v3_in[i][j].vecs[1][ii] * xk2[ii];
                }
                ret_in += fcs_group_v3_in[i][j] * phase_storage_in->get_exp_type2(phase3);
            }
            ret[i] = ret_in;
        }
    } else {
        // Original version
#pragma omp parallel for private(ret_in, nsize_group, phase, j) if (use_openmp)
        for (i = 0; i < ngroup_v3_in; ++i) {

            ret_in = std::complex<double>(0.0, 0.0);
            nsize_group = fcs_group_v3_in[i].size();

            for (j = 0; j < nsize_group; ++j) {
                phase = relvec_v3_in[i][j].vecs[0][0] * xk1[0] + relvec_v3_in[i][j].vecs[0][1] * xk1[1] +
                        relvec_v3_in[i][j].vecs[0][2] * xk1[2] + relvec_v3_in[i][j].vecs[1][0] * xk2[0] +
                        relvec_v3_in[i][j].vecs[1][1] * xk2[1] + relvec_v3_in[i][j].vecs[1][2] * xk2[2];
                ret_in += fcs_group_v3_in[i][j] * std::exp(im * phase);
            }
            ret[i] = ret_in;
        }
    }
}

std::complex<double> AnharmonicCore::V4(const unsigned int ks[4], const double *const *xk_in,
                                        const double *const *eval_in, const std::complex<double> *const *const *evec_in,
                                        const PhaseFactorCache *phase_storage_in)
{
    int i;
    const int ns = dynamical->neval;
    unsigned int kn[4], sn[4];
    double omega[4];
    auto ret_re = 0.0;
    auto ret_im = 0.0;
    auto ret = std::complex<double>(0.0, 0.0);

    for (i = 0; i < 4; ++i) {
        kn[i] = ks[i] / ns;
        sn[i] = ks[i] % ns;
        omega[i] = eval_in[kn[i]][sn[i]];
    }
    // Return zero if any of the involving phonon has imaginary frequency
    if (omega[0] < eps8 || omega[1] < eps8 || omega[2] < eps8 || omega[3] < eps8) return 0.0;

    if (kn[1] != kindex_phi4_stored[0] || kn[2] != kindex_phi4_stored[1] || kn[3] != kindex_phi4_stored[2]) {

        calc_phi4_reciprocal(xk_in[kn[1]], xk_in[kn[2]], xk_in[kn[3]], phase_storage_in, phi4_reciprocal);

        kindex_phi4_stored[0] = kn[1];
        kindex_phi4_stored[1] = kn[2];
        kindex_phi4_stored[2] = kn[3];
    }

#ifdef _OPENMP
#pragma omp parallel for private(ret), reduction(+ : ret_re, ret_im)
#endif
    for (i = 0; i < ngroup_v4; ++i) {
        ret = evec_in[kn[0]][sn[0]][evec_index_v4[i][0]] * evec_in[kn[1]][sn[1]][evec_index_v4[i][1]] *
              evec_in[kn[2]][sn[2]][evec_index_v4[i][2]] * evec_in[kn[3]][sn[3]][evec_index_v4[i][3]] * invmass_v4[i] *
              phi4_reciprocal[i];
        ret_re += ret.real();
        ret_im += ret.imag();
    }

    return std::complex<double>(ret_re, ret_im) / std::sqrt(omega[0] * omega[1] * omega[2] * omega[3]);
}

std::complex<double> AnharmonicCore::V4(const unsigned int ks[4], const double *const *xk_in,
                                        const double *const *eval_in, const std::complex<double> *const *const *evec_in,
                                        const PhaseFactorCache *phase_storage_in, std::complex<double> *phi4_work,
                                        int *kindex_work)
{
    // Thread-safe serial variant of V4 (see the V3 counterpart): the caller
    // owns the per-quartet reciprocal-FC4 cache and no OpenMP region is
    // entered here.
    int i;
    const int ns = dynamical->neval;
    unsigned int kn[4], sn[4];
    double omega[4];

    for (i = 0; i < 4; ++i) {
        kn[i] = ks[i] / ns;
        sn[i] = ks[i] % ns;
        omega[i] = eval_in[kn[i]][sn[i]];
    }
    // Return zero if any of the involving phonon has imaginary frequency
    if (omega[0] < eps8 || omega[1] < eps8 || omega[2] < eps8 || omega[3] < eps8) {
        return std::complex<double>(0.0, 0.0);
    }

    if (static_cast<int>(kn[1]) != kindex_work[0] || static_cast<int>(kn[2]) != kindex_work[1] ||
        static_cast<int>(kn[3]) != kindex_work[2])
    {
        calc_phi4_reciprocal(xk_in[kn[1]], xk_in[kn[2]], xk_in[kn[3]], phase_storage_in, phi4_work, false);
        kindex_work[0] = static_cast<int>(kn[1]);
        kindex_work[1] = static_cast<int>(kn[2]);
        kindex_work[2] = static_cast<int>(kn[3]);
    }

    auto ret_re = 0.0;
    auto ret_im = 0.0;
    for (i = 0; i < ngroup_v4; ++i) {
        const auto ret = evec_in[kn[0]][sn[0]][evec_index_v4[i][0]] * evec_in[kn[1]][sn[1]][evec_index_v4[i][1]] *
                         evec_in[kn[2]][sn[2]][evec_index_v4[i][2]] * evec_in[kn[3]][sn[3]][evec_index_v4[i][3]] *
                         invmass_v4[i] * phi4_work[i];
        ret_re += ret.real();
        ret_im += ret.imag();
    }

    return std::complex<double>(ret_re, ret_im) / std::sqrt(omega[0] * omega[1] * omega[2] * omega[3]);
}

std::complex<double> AnharmonicCore::Phi4(const unsigned int ks[4], const double *const *xk_in,
                                          const double *const *eval_in,
                                          const std::complex<double> *const *const *evec_in,
                                          const PhaseFactorCache *phase_storage_in)
{
    int i;
    int ns = dynamical->neval;
    unsigned int kn[4], sn[4];
    double omega[4];
    double ret_re = 0.0;
    double ret_im = 0.0;
    auto ret = std::complex<double>(0.0, 0.0);

    for (i = 0; i < 4; ++i) {
        kn[i] = ks[i] / ns;
        sn[i] = ks[i] % ns;
        omega[i] = eval_in[kn[i]][sn[i]];
    }

    if (kn[1] != kindex_phi4_stored[0] || kn[2] != kindex_phi4_stored[1] || kn[3] != kindex_phi4_stored[2]) {

        calc_phi4_reciprocal(xk_in[kn[1]], xk_in[kn[2]], xk_in[kn[3]], phase_storage_in, phi4_reciprocal);

        kindex_phi4_stored[0] = kn[1];
        kindex_phi4_stored[1] = kn[2];
        kindex_phi4_stored[2] = kn[3];
    }

#ifdef _OPENMP
#pragma omp parallel for private(ret), reduction(+ : ret_re, ret_im)
#endif
    for (i = 0; i < ngroup_v4; ++i) {
        ret = evec_in[kn[0]][sn[0]][evec_index_v4[i][0]] * evec_in[kn[1]][sn[1]][evec_index_v4[i][1]] *
              evec_in[kn[2]][sn[2]][evec_index_v4[i][2]] * evec_in[kn[3]][sn[3]][evec_index_v4[i][3]] * invmass_v4[i] *
              phi4_reciprocal[i];
        ret_re += ret.real();
        ret_im += ret.imag();
    }

    return std::complex<double>(ret_re, ret_im);
}

void AnharmonicCore::calc_phi4_reciprocal(const double *xk1, const double *xk2, const double *xk3,
                                          const PhaseFactorCache *phase_storage_in, std::complex<double> *ret,
                                          const bool use_openmp)
{
    int i, j;
    double phase;
    std::complex<double> ret_in;
    unsigned int nsize_group;

    const auto tune_type_now = phase_storage_in->get_tune_type();
    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);

    if (tune_type_now == 1) {

#pragma omp parallel for private(ret_in, nsize_group, j, phase) if (use_openmp)
        for (i = 0; i < ngroup_v4; ++i) {

            ret_in = complex_zero;
            nsize_group = fcs_group_v4[i].size();

            for (j = 0; j < nsize_group; ++j) {
                phase = relvec_v4[i][j].vecs[0][0] * xk1[0] + relvec_v4[i][j].vecs[0][1] * xk1[1] +
                        relvec_v4[i][j].vecs[0][2] * xk1[2] + relvec_v4[i][j].vecs[1][0] * xk2[0] +
                        relvec_v4[i][j].vecs[1][1] * xk2[1] + relvec_v4[i][j].vecs[1][2] * xk2[2] +
                        relvec_v4[i][j].vecs[2][0] * xk3[0] + relvec_v4[i][j].vecs[2][1] * xk3[1] +
                        relvec_v4[i][j].vecs[2][2] * xk3[2];

                ret_in += fcs_group_v4[i][j] * phase_storage_in->get_exp_type1(phase);
            }
            ret[i] = ret_in;
        }

    } else if (tune_type_now == 2) {

        // Tuned version is used when nk1=nk2=nk3 doesn't hold.

        double phase3[3];

#pragma omp parallel for private(ret_in, nsize_group, j, phase3) if (use_openmp)
        for (i = 0; i < ngroup_v4; ++i) {

            ret_in = complex_zero;
            nsize_group = fcs_group_v4[i].size();

            for (j = 0; j < nsize_group; ++j) {
                for (auto ii = 0; ii < 3; ++ii) {
                    phase3[ii] = relvec_v4[i][j].vecs[0][ii] * xk1[ii] + relvec_v4[i][j].vecs[1][ii] * xk2[ii] +
                                 relvec_v4[i][j].vecs[2][ii] * xk3[ii];
                }
                ret_in += fcs_group_v4[i][j] * phase_storage_in->get_exp_type2(phase3);
            }
            ret[i] = ret_in;
        }
    } else {
        // Original version
#pragma omp parallel for private(ret_in, nsize_group, phase) if (use_openmp)
        for (i = 0; i < ngroup_v4; ++i) {

            ret_in = complex_zero;
            nsize_group = fcs_group_v4[i].size();

            for (j = 0; j < nsize_group; ++j) {
                phase = relvec_v4[i][j].vecs[0][0] * xk1[0] + relvec_v4[i][j].vecs[0][1] * xk1[1] +
                        relvec_v4[i][j].vecs[0][2] * xk1[2] + relvec_v4[i][j].vecs[1][0] * xk2[0] +
                        relvec_v4[i][j].vecs[1][1] * xk2[1] + relvec_v4[i][j].vecs[1][2] * xk2[2] +
                        relvec_v4[i][j].vecs[2][0] * xk3[0] + relvec_v4[i][j].vecs[2][1] * xk3[1] +
                        relvec_v4[i][j].vecs[2][2] * xk3[2];

                ret_in += fcs_group_v4[i][j] * std::exp(im * phase);
            }
            ret[i] = ret_in;
        }
    }
}

void AnharmonicCore::setup_cubic()
{
    // Sort force_constant[1] using the operator defined in fcs_phonons.h
    // This sorting is necessary.
    std::sort(fcs_phonon->force_constant_with_cell[1].begin(), fcs_phonon->force_constant_with_cell[1].end());

    prepare_group_of_force_constants(fcs_phonon->force_constant_with_cell[1], ngroup_v3, fcs_group_v3);

    invmass_v3.resize(ngroup_v3);
    evec_index_v3.resize(ngroup_v3, 3);
    relvec_v3.resize(ngroup_v3);
    phi3_reciprocal.resize(ngroup_v3);

    prepare_relative_vector(fcs_phonon->force_constant_with_cell[1], ngroup_v3, fcs_group_v3, relvec_v3);

    const auto invsqrt_mass_p = system->get_invsqrt_mass();

    int k = 0;
    for (auto i = 0; i < ngroup_v3; ++i) {
        for (int j = 0; j < 3; ++j) {
            evec_index_v3[i][j] = fcs_phonon->force_constant_with_cell[1][k].pairs[j].index;
        }
        invmass_v3[i] = invsqrt_mass_p[evec_index_v3[i][0] / 3] * invsqrt_mass_p[evec_index_v3[i][1] / 3] *
                        invsqrt_mass_p[evec_index_v3[i][2] / 3];
        k += fcs_group_v3[i].size();
    }
}

void AnharmonicCore::setup_quartic()
{
    std::sort(fcs_phonon->force_constant_with_cell[2].begin(), fcs_phonon->force_constant_with_cell[2].end());
    prepare_group_of_force_constants(fcs_phonon->force_constant_with_cell[2], ngroup_v4, fcs_group_v4);

    invmass_v4.resize(ngroup_v4);
    evec_index_v4.resize(ngroup_v4, 4);
    relvec_v4.resize(ngroup_v4);
    phi4_reciprocal.resize(ngroup_v4);

    prepare_relative_vector(fcs_phonon->force_constant_with_cell[2], ngroup_v4, fcs_group_v4, relvec_v4);

    const auto invsqrt_mass_p = system->get_invsqrt_mass();

    int k = 0;
    for (auto i = 0; i < ngroup_v4; ++i) {
        for (int j = 0; j < 4; ++j) {
            evec_index_v4[i][j] = fcs_phonon->force_constant_with_cell[2][k].pairs[j].index;
        }
        invmass_v4[i] = invsqrt_mass_p[evec_index_v4[i][0] / 3] * invsqrt_mass_p[evec_index_v4[i][1] / 3] *
                        invsqrt_mass_p[evec_index_v4[i][2] / 3] * invsqrt_mass_p[evec_index_v4[i][3] / 3];
        k += fcs_group_v4[i].size();
    }
}

void PhaseFactorCache::create(const bool use_tuned_ver, const bool switch_to_type2)
{
    // For accelerating function V3 and V4 by avoiding continual call of std::exp.

    if (use_tuned_ver) {

        const auto inv2pi = 1.0 / (2.0 * pi);

        for (auto i = 0; i < 3; ++i) dnk[i] = static_cast<double>(nk_grid[i]) * inv2pi;

        tune_type = 1;

        if (nk_grid[0] == nk_grid[1] && nk_grid[1] == nk_grid[2]) {
            nk_represent = nk_grid[0];
        } else if (nk_grid[0] == nk_grid[1] && nk_grid[2] == 1) {
            nk_represent = nk_grid[0];
        } else if (nk_grid[1] == nk_grid[2] && nk_grid[0] == 1) {
            nk_represent = nk_grid[1];
        } else if (nk_grid[2] == nk_grid[0] && nk_grid[1] == 1) {
            nk_represent = nk_grid[2];
        } else if (nk_grid[0] == 1 && nk_grid[1] == 1) {
            nk_represent = nk_grid[2];
        } else if (nk_grid[1] == 1 && nk_grid[2] == 1) {
            nk_represent = nk_grid[0];
        } else if (nk_grid[2] == 1 && nk_grid[0] == 1) {
            nk_represent = nk_grid[1];
        } else {
            tune_type = 2;
        }

        // Force using tune_type == 2 version
        if (switch_to_type2) tune_type = 2;

        int ii, jj, kk;

        if (tune_type == 1) {

            double phase;
            dnk_represent = static_cast<double>(nk_represent) * inv2pi;
            const auto inv_dnk_represent = 1.0 / dnk_represent;

            // Pre-calculate the phase factor exp[i 2pi * phase]
            // for different phase angles ranging from [-2pi + 2pi/nk_represent: 2pi*(nk_represent-1)/nk_represent].
            // The redundancy of the data here is intentional and helpful for accepting
            // both positive and negative modulo.
            exp_phase.resize(2 * nk_represent - 1);
#ifdef _OPENMP
#pragma omp parallel for private(phase)
#endif
            for (ii = 0; ii < 2 * nk_represent - 1; ++ii) {
                phase = static_cast<double>(ii - nk_represent + 1) * inv_dnk_represent;
                exp_phase[ii] = std::exp(im * phase);
            }

        } else if (tune_type == 2) {

            double phase[3];
            double inv_dnk[3];

            for (auto i = 0; i < 3; ++i) inv_dnk[i] = 1.0 / dnk[i];

            exp_phase3.resize(2 * nk_grid[0] - 1, 2 * nk_grid[1] - 1, 2 * nk_grid[2] - 1);
#ifdef _OPENMP
#pragma omp parallel for private(phase, jj, kk)
#endif
            for (ii = 0; ii < 2 * nk_grid[0] - 1; ++ii) {
                phase[0] = static_cast<double>(ii - nk_grid[0] + 1) * inv_dnk[0];
                for (jj = 0; jj < 2 * nk_grid[1] - 1; ++jj) {
                    phase[1] = static_cast<double>(jj - nk_grid[1] + 1) * inv_dnk[1];
                    for (kk = 0; kk < 2 * nk_grid[2] - 1; ++kk) {
                        phase[2] = static_cast<double>(kk - nk_grid[2] + 1) * inv_dnk[2];
                        exp_phase3[ii][jj][kk] = std::exp(im * (phase[0] + phase[1] + phase[2]));
                    }
                }
            }
        }
    } else {
        tune_type = 0;
    }
}

unsigned int PhaseFactorCache::get_tune_type() const
{
    return tune_type;
}

std::complex<double> PhaseFactorCache::get_exp_type1(const double phase_in) const
{
    int iloc = nint(phase_in * dnk_represent) % nk_represent + nk_represent - 1;
    return exp_phase[iloc];
}

std::complex<double> PhaseFactorCache::get_exp_type2(const double *phase3_in) const
{
    int loc[3];
    for (auto i = 0; i < 3; ++i) {
        loc[i] = nint(phase3_in[i] * dnk[i]) % nk_grid[i] + nk_grid[i] - 1;
    }
    return exp_phase3[loc[0]][loc[1]][loc[2]];
}

int AnharmonicCore::get_ngroup_fcs(const unsigned int order) const
{
    if (order == 3) return ngroup_v3;
    if (order == 4) return ngroup_v4;
    return 0;
}

const std::vector<double> *AnharmonicCore::get_fcs_group(const unsigned int order) const
{
    if (order == 3) return fcs_group_v3;
    if (order == 4) return fcs_group_v4;
    return nullptr;
}

const double *AnharmonicCore::get_invmass_factor(const unsigned int order) const
{
    if (order == 3) return invmass_v3;
    if (order == 4) return invmass_v4;
    return nullptr;
}

const int *const *AnharmonicCore::get_evec_index(const unsigned int order) const
{
    if (order == 3) return evec_index_v3;
    if (order == 4) return evec_index_v4;
    return nullptr;
}

const std::vector<RelativeVector> *AnharmonicCore::get_relvec(const unsigned int order) const
{
    if (order == 3) return relvec_v3;
    if (order == 4) return relvec_v4;
    return nullptr;
}
