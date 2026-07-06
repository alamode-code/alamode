/*
 phonon_thermodynamics.cpp

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "thermodynamics.h"
#include <complex>
#include <iostream>
#include "anharmonic_core.h"
#include "constants.h"
#include "dynamical.h"
#include "kpoint.h"
#include "memory.h"
#include "mpi_common.h"
#include "phonon_dos.h"
#include "pointers.h"
#include "progress_bar.h"
#include "relaxation.h"
#include "system.h"

using namespace PHON_NS;

Thermodynamics::Thermodynamics(PHON *phon) :
    Pointers(phon), T_to_Ryd(k_Boltzmann / Ryd), classical(false), calc_FE_bubble(false), FE_bubble(nullptr)
{}

Thermodynamics::~Thermodynamics()
{
    if (FE_bubble) {
        deallocate(FE_bubble);
    }
};

void Thermodynamics::setup()
{
    MPI_Bcast(&classical, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
}

auto Thermodynamics::Cv(const double omega, const double temp_in) const -> double
{
    // Mode specific heat at constant volume.
    // Exactly heat capacity only within the QHA.
    // In other cases, Σ_q C_v is not equal to the total heat capacity.
    if (std::abs(temp_in) < eps) return 0.0;

    const auto x = omega / (T_to_Ryd * temp_in);
    return k_Boltzmann * pow2(x / (2.0 * sinh(0.5 * x)));
}

auto Thermodynamics::Cv_classical(const double omega, const double temp_in) -> double
{
    // Just return k_B
    if (std::abs(temp_in) < eps) return 0.0;

    return k_Boltzmann;
}

auto Thermodynamics::fB(const double omega, const double temp_in) const -> double
{
    // Bose-Einstein distribution function
    if (std::abs(temp_in) < eps || omega < eps8) return 0.0;

    const auto x = omega / (T_to_Ryd * temp_in);
    return 1.0 / (std::exp(x) - 1.0);
}

auto Thermodynamics::fC(const double omega, const double temp_in) const -> double
{
    // Classical limit of Bose-Einstein distribution function
    if (std::abs(temp_in) < eps || omega < eps8) return 0.0;

    const auto x = omega / (T_to_Ryd * temp_in);
    return 1.0 / x;
}

auto Thermodynamics::Cv_tot(const double temp_in, const unsigned int nk_irred, const unsigned int ns,
                            const std::vector<std::vector<KpointList>> &kp_irred, const double *weight_k_irred,
                            const double *const *eval_in) const -> double
{
    // Total constant-volume heat capacity. Only the quasiharmonic term is included here.
    int i;
    unsigned int ik, is;
    double omega;
    auto ret = 0.0;

    const auto N = nk_irred * ns;
    int ik_irred;

    if (classical) {
#pragma omp parallel for private(ik_irred, ik, is, omega), reduction(+ : ret)
        for (i = 0; i < N; ++i) {
            ik_irred = i / ns;
            ik = kp_irred[ik_irred][0].knum;
            is = i % ns;

            omega = eval_in[ik][is];

            if (omega < eps8) continue;

            ret += Cv_classical(omega, temp_in) * weight_k_irred[ik_irred];
        }
    } else {
#pragma omp parallel for private(ik_irred, ik, is, omega), reduction(+ : ret)
        for (i = 0; i < N; ++i) {
            ik_irred = i / ns;
            ik = kp_irred[ik_irred][0].knum;
            is = i % ns;

            omega = eval_in[ik][is];

            if (omega < eps8) continue;

            ret += Cv(omega, temp_in) * weight_k_irred[ik_irred];
        }
    }

    return ret;
}

auto Thermodynamics::Cv_anharm_correction(const double temp_in, const unsigned int nk_irred, const unsigned int ns,
                                          const std::vector<std::vector<KpointList>> &kp_irred,
                                          const double *weight_k_irred, const double *const *eval_in,
                                          const double *const *del_eval_in) const -> double
{
    // Anharmonic correction to constant-volume heat capacity
    // We only use the adjacent temperature point for the numerical derivative, so the numerical accuracy
    // may not be very high. For more reliable estimate, it is recommended to fit the entropy curve with
    // respect to temperature by polynomial function and take the derivative of the fitted curve.
    int i;
    unsigned int ik, is;
    double omega, domega_dt;
    auto ret = 0.0;

    const auto N = nk_irred * ns;
    int ik_irred;

    if (classical) {
#pragma omp parallel for private(ik_irred, ik, is, omega, domega_dt), reduction(+ : ret)
        for (i = 0; i < N; ++i) {
            ik_irred = i / ns;
            ik = kp_irred[ik_irred][0].knum;
            is = i % ns;

            omega = eval_in[ik][is];
            domega_dt = del_eval_in[ik][is];

            if (omega < eps8) continue;

            ret -= Cv_classical(omega, temp_in) * (temp_in / omega) * domega_dt * weight_k_irred[ik_irred];
        }
    } else {
#pragma omp parallel for private(ik_irred, ik, is, omega, domega_dt), reduction(+ : ret)
        for (i = 0; i < N; ++i) {
            ik_irred = i / ns;
            ik = kp_irred[ik_irred][0].knum;
            is = i % ns;

            omega = eval_in[ik][is];
            domega_dt = del_eval_in[ik][is];

            if (omega < eps8) continue;

            ret -= Cv(omega, temp_in) * (temp_in / omega) * domega_dt * weight_k_irred[ik_irred];
        }
    }

    return ret;
}

auto Thermodynamics::internal_energy(const double temp_in, const unsigned int nk_irred, const unsigned int ns,
                                     const std::vector<std::vector<KpointList>> &kp_irred, const double *weight_k_irred,
                                     const double *const *eval_in) const -> double
{
    // Vibrational internal energy within QHA.
    // U = F + TS = Σ_q  0.5 * hw_q * coth(hw_q/2k_BT)] = Σ_q [hw_q (n_q + 0.5)]
    int i;
    unsigned int ik, is;
    double omega;
    auto ret = 0.0;

    const auto N = nk_irred * ns;
    int ik_irred;

    if (classical) {
#pragma omp parallel for private(ik_irred, ik, is, omega), reduction(+ : ret)
        for (i = 0; i < N; ++i) {
            ik_irred = i / ns;
            ik = kp_irred[ik_irred][0].knum;
            is = i % ns;
            omega = eval_in[ik][is];

            if (omega < eps8) continue;

            ret += T_to_Ryd * temp_in * weight_k_irred[ik_irred];
        }
        ret *= 2.0;

    } else {
#pragma omp parallel for private(ik_irred, ik, is, omega), reduction(+ : ret)
        for (i = 0; i < N; ++i) {
            ik_irred = i / ns;
            ik = kp_irred[ik_irred][0].knum;
            is = i % ns;
            omega = eval_in[ik][is];

            if (omega < eps8) continue;

            ret += omega * coth_T(omega, temp_in) * weight_k_irred[ik_irred];
        }
    }
    return ret * 0.5;
}

auto Thermodynamics::vibrational_entropy(const double temp_in, const unsigned int nk_irred, const unsigned int ns,
                                         const std::vector<std::vector<KpointList>> &kp_irred,
                                         const double *weight_k_irred, const double *const *eval_in) const -> double
{
    // Vibrational entropy correct for QHA/SCP and other quasiparticle approaches
    // S = -(∂F/∂T) = k_B * Σ_q [(n_q + 1)ln(n_q + 1) - n_q ln n_q]
    int i;
    unsigned int ik, is;
    double omega, x;
    auto ret = 0.0;

    const auto N = nk_irred * ns;
    int ik_irred;

    if (classical) {
#pragma omp parallel for private(ik_irred, ik, is, omega, x), reduction(+ : ret)
        for (i = 0; i < N; ++i) {
            ik_irred = i / ns;
            ik = kp_irred[ik_irred][0].knum;
            is = i % ns;
            omega = eval_in[ik][is];

            if (omega < eps8 || std::abs(temp_in) < eps) continue;

            x = omega / (temp_in * T_to_Ryd);
            ret += (std::log(x) - 1.0) * weight_k_irred[ik_irred];
        }

    } else {
#pragma omp parallel for private(ik_irred, ik, is, omega, x), reduction(+ : ret)
        for (i = 0; i < N; ++i) {
            ik_irred = i / ns;
            ik = kp_irred[ik_irred][0].knum;
            is = i % ns;
            omega = eval_in[ik][is];

            if (omega < eps8 || std::abs(temp_in) < eps) continue;

            x = omega / (temp_in * T_to_Ryd);
            ret += (std::log(1.0 - std::exp(-x)) - x / (std::exp(x) - 1.0)) * weight_k_irred[ik_irred];
        }
    }
    return -k_Boltzmann * ret;
}

auto Thermodynamics::free_energy_QHA(const double temp_in, const unsigned int nk_irred, const unsigned int ns,
                                     const std::vector<std::vector<KpointList>> &kp_irred, const double *weight_k_irred,
                                     const double *const *eval_in) const -> double
{
    // Vibrational free energy within QHA and QHA-like term within SCP.
    // F = Σ_q [0.5 hw_q + k_B T ln(1 - exp(-hw_q/k_BT))] (quantum)
    int i;
    unsigned int ik, is;
    double omega, x;
    auto ret = 0.0;

    const auto N = nk_irred * ns;
    int ik_irred;

    if (classical) {
#pragma omp parallel for private(ik_irred, ik, is, omega, x), reduction(+ : ret)
        for (i = 0; i < N; ++i) {
            ik_irred = i / ns;
            ik = kp_irred[ik_irred][0].knum;
            is = i % ns;
            omega = eval_in[ik][is];

            if (omega < eps8) continue;

            if (std::abs(temp_in) > eps) {
                x = omega / (temp_in * T_to_Ryd);
                ret += std::log(x) * weight_k_irred[ik_irred];
            }
        }

        return temp_in * T_to_Ryd * ret;
    }
#pragma omp parallel for private(ik_irred, ik, is, omega, x), reduction(+ : ret)
    for (i = 0; i < N; ++i) {
        ik_irred = i / ns;
        ik = kp_irred[ik_irred][0].knum;
        is = i % ns;
        omega = eval_in[ik][is];

        if (omega < eps8) continue;

        if (std::abs(temp_in) < eps) {
            ret += 0.5 * omega * weight_k_irred[ik_irred];
        } else {
            x = omega / (temp_in * T_to_Ryd);
            ret += (0.5 * x + std::log(1.0 - std::exp(-x))) * weight_k_irred[ik_irred];
        }
    }


    if (std::abs(temp_in) < eps) return ret;

    return temp_in * T_to_Ryd * ret;
}

auto Thermodynamics::disp2_avg(const double T_in, const unsigned int ncrd1, const unsigned int ncrd2,
                               const unsigned int nk, const unsigned int ns, const double *const *xk_in,
                               const double *const *eval_in, std::complex<double> ***evec_in) const -> double
{
    constexpr double cell_shift[3]{0, 0, 0};
    return disp_corrfunc(T_in, ncrd1, ncrd2, cell_shift, nk, ns, xk_in, eval_in, evec_in);
}

auto Thermodynamics::disp_corrfunc(const double T_in, const unsigned int ncrd1, const unsigned int ncrd2,
                                   const double cell_shift[3], const unsigned int nk, const unsigned int ns,
                                   const double *const *xk_in, const double *const *eval_in,
                                   std::complex<double> ***evec_in) const -> double
{
    int i;
    int const N = nk * ns;
    unsigned int ik, is;
    double omega;
    double phase;
    constexpr std::complex<double> im(0.0, 1.0);
    double ret = 0.0;

    if (classical) {
#pragma omp parallel for private(ik, is, omega, phase), reduction(+ : ret)
        for (i = 0; i < N; ++i) {
            ik = i / ns;
            is = i % ns;
            omega = eval_in[ik][is];

            if (omega < eps8) continue;

            phase =
                2.0 * pi * (xk_in[ik][0] * cell_shift[0] + xk_in[ik][1] * cell_shift[1] + xk_in[ik][2] * cell_shift[2]);

            ret += real(std::conj(evec_in[ik][is][ncrd1]) * evec_in[ik][is][ncrd2] * std::exp(phase)) * T_in *
                   T_to_Ryd / (omega * omega);
        }

    } else {
#pragma omp parallel for private(ik, is, omega, phase), reduction(+ : ret)
        for (i = 0; i < N; ++i) {
            ik = i / ns;
            is = i % ns;
            omega = eval_in[ik][is];

            if (omega < eps8) continue;

            phase =
                2.0 * pi * (xk_in[ik][0] * cell_shift[0] + xk_in[ik][1] * cell_shift[1] + xk_in[ik][2] * cell_shift[2]);

            ret += real(std::conj(evec_in[ik][is][ncrd1]) * evec_in[ik][is][ncrd2] * std::exp(im * phase)) *
                   (fB(omega, T_in) + 0.5) / omega;
        }
    }

    ret *= 1.0 / (static_cast<double>(nk) * std::sqrt(system->get_mass_super()[system->get_map_p2s(0)[ncrd1 / 3][0]] *
                                                      system->get_mass_super()[system->get_map_p2s(0)[ncrd2 / 3][0]]));

    return ret;
}

auto Thermodynamics::coth_T(const double omega, const double T) const -> double
{
    // This function returns coth(hbar*omega/2*kB*T)

    // if T = 0.0 and omega > 0, coth(hbar*omega/(2*kB*T)) = 1.0
    if (T < eps) return 1.0;

    const auto x = omega / (T_to_Ryd * T * 2.0);
    return 1.0 + 2.0 / (std::exp(2.0 * x) - 1.0);
}

auto Thermodynamics::compute_free_energy_bubble() -> void
{
    const auto NT = static_cast<unsigned int>((system->Tmax - system->Tmin) / system->dT) + 1;

    if (mympi->my_rank == 0) {
        std::cout << '\n';
        std::cout << " -----------------------------------------------------------------\n";
        std::cout << " Calculating the vibrational free energy from the Bubble diagram \n" << std::flush;
    }

    allocate(FE_bubble, NT);

    compute_FE_bubble(dos->dymat_dos->get_eigenvalues(), dos->dymat_dos->get_eigenvectors(), FE_bubble);

    if (mympi->my_rank == 0) {
        std::cout << " done!\n\n";
    }
}

auto Thermodynamics::compute_FE_bubble(double **eval, std::complex<double> ***evec, double *FE_bubble_out) const -> void
{
    // This function calculates the free energy of the bubble diagram
    double omega_sum[2];
    double nsum[2];
    const auto nk = dos->kmesh_dos->nk;
    const auto nk_irred = dos->kmesh_dos->nk_irred;
    const auto ns = dynamical->neval;
    unsigned int i0, iT;
    unsigned int arr_cubic[3];
    const auto nks0 = nk_irred * ns;
    const auto NT = static_cast<unsigned int>((system->Tmax - system->Tmin) / system->dT) + 1;
    const auto factor = -1.0 / (static_cast<double>(nk * nk) * 48.0);

    double n0, n1, n2;
    double *FE_local;
    double *FE_tmp;

    allocate(FE_local, NT);
    allocate(FE_tmp, NT);
    std::vector<KsListGroup> triplet;

    std::vector<int> vks_l;
    vks_l.clear();

    for (i0 = 0; i0 < nks0; ++i0) {
        if (i0 % mympi->nprocs == mympi->my_rank) {
            vks_l.push_back(i0);
        }
    }

    unsigned int nk_tmp;

    if (nks0 % mympi->nprocs != 0) {
        nk_tmp = nks0 / mympi->nprocs + 1;
    } else {
        nk_tmp = nks0 / mympi->nprocs;
    }
    if (vks_l.size() < nk_tmp) {
        vks_l.push_back(-1);
    }

    for (iT = 0; iT < NT; ++iT)
        FE_local[iT] = 0.0;

    for (i0 = 0; i0 < nk_tmp; ++i0) {

        if (vks_l[i0] != -1) {

            const auto ik0 = dos->kmesh_dos->kpoint_irred_all[vks_l[i0] / ns][0].knum;
            const auto is0 = vks_l[i0] % ns;

            dos->kmesh_dos->get_unique_triplet_k(vks_l[i0] / ns,
                                                 symmetry->SymmList,
                                                 anharmonic_core->use_triplet_symmetry,
                                                 true,
                                                 triplet,
                                                 1);

            const size_t npair_uniq = triplet.size();

            arr_cubic[0] = ns * ik0 + is0;

            for (iT = 0; iT < NT; ++iT)
                FE_tmp[iT] = 0.0;

            for (auto ik = 0; ik < npair_uniq; ++ik) {
                const int multi = triplet[ik].group.size();

                arr_cubic[0] = ns * ik0 + is0;

                const unsigned int ik1 = triplet[ik].group[0].ks[0];
                const unsigned int ik2 = triplet[ik].group[0].ks[1];

                for (unsigned int is1 = 0; is1 < ns; ++is1) {
                    arr_cubic[1] = ns * ik1 + is1;

                    for (unsigned int is2 = 0; is2 < ns; ++is2) {
                        arr_cubic[2] = ns * ik2 + is2;

                        const auto omega0 = eval[ik0][is0];
                        const auto omega1 = eval[ik1][is1];
                        const auto omega2 = eval[ik2][is2];

                        if (omega0 < eps8 || omega1 < eps8 || omega2 < eps8) continue;

                        omega_sum[0] = 1.0 / (omega0 + omega1 + omega2);
                        omega_sum[1] = 1.0 / (-omega0 + omega1 + omega2);

                        const auto v3_tmp = std::norm(anharmonic_core->V3(arr_cubic)) * static_cast<double>(multi);

                        for (iT = 0; iT < NT; ++iT) {
                            const auto temp = system->Tmin + static_cast<double>(iT) * system->dT;

                            if (classical) {
                                n0 = fC(omega0, temp);
                                n1 = fC(omega1, temp);
                                n2 = fC(omega2, temp);

                                nsum[0] = n0 * (n1 + n2) + n1 * n2;
                                nsum[1] = n0 * (n1 + n2) - n1 * n2;
                            } else {
                                n0 = fB(omega0, temp);
                                n1 = fB(omega1, temp);
                                n2 = fB(omega2, temp);

                                nsum[0] = (1.0 + n0) * (1.0 + n1 + n2) + n1 * n2;
                                nsum[1] = n0 * n1 - n1 * n2 + n2 * n0 + n0;
                            }

                            FE_tmp[iT] += v3_tmp * (nsum[0] * omega_sum[0] + 3.0 * nsum[1] * omega_sum[1]);
                        }
                    }
                }
            }
            const auto weight = static_cast<double>(dos->kmesh_dos->kpoint_irred_all[vks_l[i0] / ns].size());
            for (iT = 0; iT < NT; ++iT)
                FE_local[iT] += FE_tmp[iT] * weight;
        }
    }

    MPI_Allreduce(&FE_local[0], &FE_bubble_out[0], NT, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (iT = 0; iT < NT; ++iT) {
        FE_bubble_out[iT] *= factor;
    }

    deallocate(FE_local);
    deallocate(FE_tmp);
}

auto Thermodynamics::compute_FE_bubble_SCPH(double ***eval_in, std::complex<double> ****evec_in,
                                            double *FE_bubble) const -> void
{
    // This function calculates the free energy from the bubble diagram
    // at the given temperature and lattice dynamics wavefunction
    double omega_sum[2];
    double n0, n1, n2, nsum[2];
    const auto nk = dos->kmesh_dos->nk;
    const auto nk_reduced = dos->kmesh_dos->nk_irred;
    const auto ns = dynamical->neval;
    unsigned int i0, iT;
    unsigned int arr_cubic[3];
    const auto nks0 = nk_reduced * ns;
    const unsigned int NT = static_cast<unsigned int>((system->Tmax - system->Tmin) / system->dT) + 1;
    const double factor = -1.0 / (static_cast<double>(nk * nk) * 48.0);

    double *FE_local;
    double *FE_tmp;

    allocate(FE_local, NT);
    allocate(FE_tmp, NT);
    std::vector<KsListGroup> triplet;

    std::vector<int> vks_l;
    vks_l.clear();

    for (i0 = 0; i0 < nks0; ++i0) {
        if (i0 % mympi->nprocs == mympi->my_rank) {
            vks_l.push_back(i0);
        }
    }

    const auto startTime = std::chrono::system_clock::now();
    auto lastUpdate = startTime;
    const bool isConsole = isOutputToConsole();

    unsigned int nk_tmp;

    if (nks0 % mympi->nprocs != 0) {
        nk_tmp = nks0 / mympi->nprocs + 1;
    } else {
        nk_tmp = nks0 / mympi->nprocs;
    }
    if (vks_l.size() < nk_tmp) {
        vks_l.push_back(-1);
    }

    const auto nks_tmp = vks_l.size();

    for (iT = 0; iT < NT; ++iT) {
        FE_local[iT] = 0.0;
    }

    if (mympi->my_rank == 0) {
        std::cout << " Total number of modes per MPI process: " << nk_tmp << '\n';
    }

    for (i0 = 0; i0 < nk_tmp; ++i0) {

        if (vks_l[i0] != -1) {

            unsigned int const ik0 = dos->kmesh_dos->kpoint_irred_all[vks_l[i0] / ns][0].knum;
            unsigned int const is0 = vks_l[i0] % ns;

            dos->kmesh_dos->get_unique_triplet_k(vks_l[i0] / ns,
                                                 symmetry->SymmList,
                                                 anharmonic_core->use_triplet_symmetry,
                                                 true,
                                                 triplet,
                                                 1);

            const size_t npair_uniq = triplet.size();

            arr_cubic[0] = ns * ik0 + is0;

            for (iT = 0; iT < NT; ++iT)
                FE_tmp[iT] = 0.0;

            for (size_t ik = 0; ik < npair_uniq; ++ik) {
                const auto multi = static_cast<double>(triplet[ik].group.size());

                arr_cubic[0] = ns * ik0 + is0;

                const unsigned int ik1 = triplet[ik].group[0].ks[0];
                const unsigned int ik2 = triplet[ik].group[0].ks[1];

                for (unsigned int is1 = 0; is1 < ns; ++is1) {
                    arr_cubic[1] = ns * ik1 + is1;

                    for (unsigned int is2 = 0; is2 < ns; ++is2) {
                        arr_cubic[2] = ns * ik2 + is2;

                        for (iT = 0; iT < NT; ++iT) {

                            const double temp = system->Tmin + static_cast<double>(iT) * system->dT;

                            const double omega0 = eval_in[iT][ik0][is0];
                            const double omega1 = eval_in[iT][ik1][is1];
                            const double omega2 = eval_in[iT][ik2][is2];

                            if (omega0 < eps8 || omega1 < eps8 || omega2 < eps8) continue;

                            omega_sum[0] = 1.0 / (omega0 + omega1 + omega2);
                            omega_sum[1] = 1.0 / (-omega0 + omega1 + omega2);

                            const double v3_tmp =
                                std::norm(
                                    anharmonic_core->V3(arr_cubic, dos->kmesh_dos->xk, eval_in[iT], evec_in[iT])) *
                                multi;

                            if (classical) {
                                n0 = fC(omega0, temp);
                                n1 = fC(omega1, temp);
                                n2 = fC(omega2, temp);

                                nsum[0] = n0 * (n1 + n2) + n1 * n2;
                                nsum[1] = n0 * (n1 + n2) - n1 * n2;
                            } else {
                                n0 = fB(omega0, temp);
                                n1 = fB(omega1, temp);
                                n2 = fB(omega2, temp);

                                nsum[0] = (1.0 + n0) * (1.0 + n1 + n2) + n1 * n2;
                                nsum[1] = n0 * n1 - n1 * n2 + n2 * n0 + n0;
                            }

                            FE_tmp[iT] += v3_tmp * (nsum[0] * omega_sum[0] + 3.0 * nsum[1] * omega_sum[1]);
                        }
                    }
                }
            }
            const double weight = static_cast<double>(dos->kmesh_dos->kpoint_irred_all[vks_l[i0] / ns].size());
            for (iT = 0; iT < NT; ++iT) {
                FE_local[iT] += FE_tmp[iT] * weight;
            }
        }
        if (mympi->my_rank == 0) {
            auto currentTime = std::chrono::system_clock::now();
            const long long totalElapsedTime =
                std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime).count();
            const long long avgTimePerStep = (i0 == 0) ? 0 : totalElapsedTime / i0;
            const long long timeRemaining = (i0 == 0) ? 0 : avgTimePerStep * (nks_tmp - i0 - 1);
            displayProgressBar(i0, nks_tmp - 1, std::cout, timeRemaining, isConsole, "Fe-bubble");
            lastUpdate = currentTime;
            if (i0 == nk_tmp - 1) std::cout << "\n done. \n\n" << std::flush;
        }
    }

    MPI_Allreduce(&FE_local[0], &FE_bubble[0], NT, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (iT = 0; iT < NT; ++iT) {
        FE_bubble[iT] *= factor;
    }

    deallocate(FE_local);
    deallocate(FE_tmp);
}

auto Thermodynamics::FE_scph_correction(unsigned int iT, double **eval, std::complex<double> ***evec,
                                        double **eval_harm_renormalized,
                                        std::complex<double> ***evec_harm_renormalized) const -> double
{
    // The correction term to the free energy within SCPH theory.
    // This term is necessary to result in S =
    using namespace Eigen;
    const auto nk = dos->kmesh_dos->nk;
    const auto ns = dynamical->neval;
    const auto temp = system->Tmin + static_cast<double>(iT) * system->dT;
    const auto N = nk * ns;

    double ret = 0.0;

#pragma omp parallel for reduction(+ : ret)
    for (int i = 0; i < N; ++i) {
        int ik = i / ns;
        int is = i % ns;
        const auto omega = eval[ik][is];
        if (std::abs(omega) < eps8) continue;

        MatrixXcd Cmat(ns, ns);

        // calculate Cmat
        for (int js = 0; js < ns; js++) {
            for (int ks = 0; ks < ns; ks++) {
                Cmat(js, ks) = 0.0;
                for (int ls = 0; ls < ns; ls++) {
                    Cmat(js, ks) += std::conj(evec_harm_renormalized[ik][js][ls]) * evec[ik][ks][ls];
                }
            }
        }

        auto tmp_c = std::complex<double>(0.0, 0.0);
        double omega2_harm;

        for (int js = 0; js < ns; js++) {
            if (eval_harm_renormalized[ik][js] < 0.0) {
                omega2_harm = -pow2(eval_harm_renormalized[ik][js]);
            } else {
                omega2_harm = pow2(eval_harm_renormalized[ik][js]);
            }
            tmp_c += std::conj(Cmat(js, is)) * omega2_harm * Cmat(js, is);
        }

        if (thermodynamics->classical) {
            ret += (tmp_c.real() - omega * omega) * thermodynamics->fC(omega, temp) / (4.0 * omega);
        } else {
            ret += (tmp_c.real() - omega * omega) * (1.0 + 2.0 * thermodynamics->fB(omega, temp)) / (8.0 * omega);
        }
    }

    return ret / static_cast<double>(nk);
}

auto Thermodynamics::compute_FE_total(const unsigned int iT, const double fe_qha,
                                      const double dfe_scph, const double v0_renorm) const -> double
{
    double fe_total = fe_qha;
    // skip scph correction for QHA + structural optimization
    if (phon->mode == "SCPH") {
        fe_total += dfe_scph;
    }
    if (thermodynamics->calc_FE_bubble) {
        fe_total += thermodynamics->FE_bubble[iT];
    }
    // The renormalized static potential of the relaxed structure; the caller
    // passes 0.0 when no structural optimization is performed.
    fe_total += v0_renorm;

    return fe_total;
}
