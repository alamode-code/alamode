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
#include <vector>
#include "anharmonic_core.h"
#include "constants.h"
#include "dynamical.h"
#include "kpoint.h"
#include "memory.h"
#include "mpi_common.h"
#include "phonon_dos.h"
#include "progress_bar.h"
#include "relaxation.h"
#include "system.h"

using namespace PHON_NS;

Thermodynamics::Thermodynamics() : classical(false), calc_FE_bubble(false)
{}

Thermodynamics::~Thermodynamics()
{
    if (FE_bubble) {
        FE_bubble.clear();
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

const double Thermodynamics::T_to_Ryd = k_Boltzmann / Ryd;

auto Thermodynamics::fB(const double omega, const double temp_in) -> double
{
    // Bose-Einstein distribution function
    if (std::abs(temp_in) < eps || omega < eps8) return 0.0;

    const auto x = omega / (T_to_Ryd * temp_in);
    return 1.0 / (std::exp(x) - 1.0);
}

auto Thermodynamics::fC(const double omega, const double temp_in) -> double
{
    // Classical limit of Bose-Einstein distribution function
    if (std::abs(temp_in) < eps || omega < eps8) return 0.0;

    const auto x = omega / (T_to_Ryd * temp_in);
    return 1.0 / x;
}

// (2 n_B(omega) + 1)/omega, the equal-time displacement-correlation (Qmat)
// factor; classical limit 2 kB T / omega^2. Zero/negative-frequency guards
// remain the caller's responsibility.
double Thermodynamics::disp_corr_factor(const double omega, const double temp_in) const
{
    if (classical) return 2.0 * temp_in * T_to_Ryd / (omega * omega);
    return (2.0 * fB(omega, temp_in) + 1.0) / omega;
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
                               const double *const *eval_in, std::complex<double> ***evec_in,
                               const System &system_in) const -> double
{
    constexpr double cell_shift[3]{0, 0, 0};
    return disp_corrfunc(T_in, ncrd1, ncrd2, cell_shift, nk, ns, xk_in, eval_in, evec_in, system_in);
}

auto Thermodynamics::disp_corrfunc(const double T_in, const unsigned int ncrd1, const unsigned int ncrd2,
                                   const double cell_shift[3], const unsigned int nk, const unsigned int ns,
                                   const double *const *xk_in, const double *const *eval_in,
                                   std::complex<double> ***evec_in, const System &system_in) const -> double
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

    ret *=
        1.0 / (static_cast<double>(nk) * std::sqrt(system_in.get_mass_super()[system_in.get_map_p2s(0)[ncrd1 / 3][0]] *
                                                   system_in.get_mass_super()[system_in.get_map_p2s(0)[ncrd2 / 3][0]]));

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

auto Thermodynamics::compute_free_energy_bubble(const System &system_in, const KpointMeshUniform &kmesh_dos_in,
                                                const DymatEigenValue &dymat_dos_in,
                                                const std::vector<SymmetryOperation> &symmlist_in,
                                                AnharmonicCore &anharmonic_core_in, const unsigned int ns_in,
                                                const int my_rank_in, const int nprocs_in,
                                                const unsigned int verbosity) -> void
{
    const auto NT = static_cast<unsigned int>((system_in.Tmax - system_in.Tmin) / system_in.dT) + 1;

    if (my_rank_in == 0 && verbosity > 0) {
        std::cout << '\n';
        std::cout << " -----------------------------------------------------------------\n";
        std::cout << " Calculating the vibrational free energy from the Bubble diagram \n" << std::flush;
    }

    FE_bubble.resize(NT);

    compute_FE_bubble(dymat_dos_in.get_eigenvalues(),
                      dymat_dos_in.get_eigenvectors(),
                      FE_bubble,
                      system_in,
                      kmesh_dos_in,
                      symmlist_in,
                      anharmonic_core_in,
                      ns_in,
                      my_rank_in,
                      nprocs_in);

    if (my_rank_in == 0 && verbosity > 0) {
        std::cout << " done!\n\n";
    }
}

namespace
{

// Bubble free energy of one first-leg mode (k0, s0), summed over its triplets
// k0 + k1 + k2 = G and all (s1, s2), for ntemp temperatures sharing eval/evec:
// fe_out[it] = sum multi |V3|^2 [ N0 / (w0 + w1 + w2) + 3 N1 / (w1 + w2 - w0) ].
// occ[it * nk * ns + k * ns + s] holds the occupations on the full mesh.
void fe_bubble_mode(AnharmonicCore &ac, const KpointMeshUniform &kmesh, const std::vector<SymmetryOperation> &symmlist,
                    const int ns, const int ik_irred, const int is0, const double *const *eval,
                    const std::complex<double> *const *const *evec, const unsigned int ntemp, const double *occ,
                    const bool classical, double *fe_out)
{
    const int nk = kmesh.nk;
    const size_t nks = static_cast<size_t>(nk) * ns;
    const size_t ns2 = static_cast<size_t>(ns) * ns;
    for (unsigned int it = 0; it < ntemp; ++it) fe_out[it] = 0.0;

    const int knum0 = kmesh.kpoint_irred_all[ik_irred][0].knum;
    const double omega0 = eval[knum0][is0];
    if (omega0 < eps8) return;

    std::vector<KsListGroup> triplet;
    kmesh.get_unique_triplet_k(ik_irred, symmlist, ac.use_triplet_symmetry, true, triplet, 1);
    const int npair_uniq = static_cast<int>(triplet.size());

    ac.prepare_v3_mode(&kmesh, knum0, is0, evec);

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        AnharmonicCore::V3Workspace ws;
        std::vector<double> v3sq(ns2);
        std::vector<double> fe_loc(ntemp, 0.0);
        ac.v3_setup_workspace(ws, &kmesh);

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 4)
#endif
        for (int ik = 0; ik < npair_uniq; ++ik) {
            const int k1 = triplet[ik].group[0].ks[0];
            const int k2 = triplet[ik].group[0].ks[1];
            const double multi = static_cast<double>(triplet[ik].group.size());

            ac.v3sq_pairs(ws, &kmesh, k1, k2, eval, evec, v3sq.data());

            for (int is1 = 0; is1 < ns; ++is1) {
                const double omega1 = eval[k1][is1];
                if (omega1 < eps8) continue;
                for (int is2 = 0; is2 < ns; ++is2) {
                    const double omega2 = eval[k2][is2];
                    if (omega2 < eps8) continue;

                    const double v3_tmp = multi * v3sq[is1 * ns + is2];
                    const double d0 = 1.0 / (omega0 + omega1 + omega2);
                    const double d1 = 1.0 / (-omega0 + omega1 + omega2);

                    for (unsigned int it = 0; it < ntemp; ++it) {
                        const double *o = occ + it * nks;
                        const double n0 = o[knum0 * ns + is0];
                        const double n1 = o[k1 * ns + is1];
                        const double n2 = o[k2 * ns + is2];
                        double nsum0, nsum1;
                        if (classical) {
                            nsum0 = n0 * (n1 + n2) + n1 * n2;
                            nsum1 = n0 * (n1 + n2) - n1 * n2;
                        } else {
                            nsum0 = (1.0 + n0) * (1.0 + n1 + n2) + n1 * n2;
                            nsum1 = n0 * n1 - n1 * n2 + n2 * n0 + n0;
                        }
                        fe_loc[it] += v3_tmp * (nsum0 * d0 + 3.0 * nsum1 * d1);
                    }
                }
            }
        }

#ifdef _OPENMP
#pragma omp critical
#endif
        {
            for (unsigned int it = 0; it < ntemp; ++it) fe_out[it] += fe_loc[it];
        }
    }
}

// Round-robin distribution of the irreducible modes over the MPI ranks.
std::vector<int> modes_of_rank(const int nks0, const int my_rank, const int nprocs)
{
    std::vector<int> vks_l;
    for (int i0 = 0; i0 < nks0; ++i0) {
        if (i0 % nprocs == my_rank) vks_l.push_back(i0);
    }
    return vks_l;
}

} // namespace

auto Thermodynamics::compute_FE_bubble(const double *const *eval, const std::complex<double> *const *const *evec,
                                       double *FE_bubble_out, const System &system_in,
                                       const KpointMeshUniform &kmesh_dos_in,
                                       const std::vector<SymmetryOperation> &symmlist_in,
                                       AnharmonicCore &anharmonic_core_in, const unsigned int ns_in,
                                       const int my_rank_in, const int nprocs_in) const -> void
{
    // Free energy of the bubble diagram with temperature-independent phonons.
    const auto nk = kmesh_dos_in.nk;
    const int ns = static_cast<int>(ns_in);
    const auto NT = static_cast<unsigned int>((system_in.Tmax - system_in.Tmin) / system_in.dT) + 1;
    const auto factor = -1.0 / (static_cast<double>(nk * nk) * 48.0);

    std::vector<double> temps(NT);
    for (unsigned int iT = 0; iT < NT; ++iT) temps[iT] = system_in.Tmin + static_cast<double>(iT) * system_in.dT;
    std::vector<double> occ;
    AnharmonicCore::tabulate_occupations(NT, temps.data(), nk, ns, eval, classical, occ);

    std::vector<double> FE_local(NT, 0.0), FE_tmp(NT);
    for (const auto iks: modes_of_rank(kmesh_dos_in.nk_irred * ns, my_rank_in, nprocs_in)) {
        const int ik_irred = iks / ns;
        fe_bubble_mode(anharmonic_core_in,
                       kmesh_dos_in,
                       symmlist_in,
                       ns,
                       ik_irred,
                       iks % ns,
                       eval,
                       evec,
                       NT,
                       occ.data(),
                       classical,
                       FE_tmp.data());
        const auto weight = static_cast<double>(kmesh_dos_in.kpoint_irred_all[ik_irred].size());
        for (unsigned int iT = 0; iT < NT; ++iT) FE_local[iT] += FE_tmp[iT] * weight;
    }

    MPI_Allreduce(FE_local.data(), FE_bubble_out, NT, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (unsigned int iT = 0; iT < NT; ++iT) FE_bubble_out[iT] *= factor;
}

auto Thermodynamics::compute_FE_bubble_SCPH(double ***eval_in, std::complex<double> ****evec_in, double *FE_bubble,
                                            const System &system_in, const KpointMeshUniform &kmesh_dos_in,
                                            const std::vector<SymmetryOperation> &symmlist_in,
                                            AnharmonicCore &anharmonic_core_in, const unsigned int ns_in,
                                            const int my_rank_in, const int nprocs_in) const -> void
{
    // Same as compute_FE_bubble with the SCPH phonons of each temperature.
    const auto nk = kmesh_dos_in.nk;
    const int ns = static_cast<int>(ns_in);
    const auto NT = static_cast<unsigned int>((system_in.Tmax - system_in.Tmin) / system_in.dT) + 1;
    const auto factor = -1.0 / (static_cast<double>(nk * nk) * 48.0);

    std::vector<double> FE_local(NT, 0.0), occ;
    const auto modes = modes_of_rank(kmesh_dos_in.nk_irred * ns, my_rank_in, nprocs_in);
    for (unsigned int iT = 0; iT < NT; ++iT) {
        const double temp = system_in.Tmin + static_cast<double>(iT) * system_in.dT;
        AnharmonicCore::tabulate_occupations(1, &temp, nk, ns, eval_in[iT], classical, occ);
        for (const auto iks: modes) {
            const int ik_irred = iks / ns;
            double fe_tmp = 0.0;
            fe_bubble_mode(anharmonic_core_in,
                           kmesh_dos_in,
                           symmlist_in,
                           ns,
                           ik_irred,
                           iks % ns,
                           eval_in[iT],
                           evec_in[iT],
                           1,
                           occ.data(),
                           classical,
                           &fe_tmp);
            FE_local[iT] += fe_tmp * static_cast<double>(kmesh_dos_in.kpoint_irred_all[ik_irred].size());
        }
    }

    MPI_Allreduce(FE_local.data(), FE_bubble, NT, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (unsigned int iT = 0; iT < NT; ++iT) FE_bubble[iT] *= factor;
}

auto Thermodynamics::FE_scph_correction(unsigned int iT, double **eval, std::complex<double> ***evec,
                                        double **eval_harm_renormalized, std::complex<double> ***evec_harm_renormalized,
                                        const KpointMeshUniform &kmesh_dos_in, const unsigned int ns_in,
                                        const System &system_in) const -> double
{
    // The correction term to the free energy within SCPH theory.
    // This term is necessary to result in S =
    const auto nk = kmesh_dos_in.nk;
    const auto ns = ns_in;
    const auto temp = system_in.Tmin + static_cast<double>(iT) * system_in.dT;
    const auto N = nk * ns;

    double ret = 0.0;

#pragma omp parallel for reduction(+ : ret)
    for (int i = 0; i < N; ++i) {
        int ik = i / ns;
        int is = i % ns;
        const auto omega = eval[ik][is];
        if (std::abs(omega) < eps8) continue;

        // Only the overlaps of mode `is` with the renormalized-harmonic modes are
        // needed: c(js) = <e_harm(js) | e(is)>, ns^2 work per (ik, is).
        auto tmp_c = std::complex<double>(0.0, 0.0);
        double omega2_harm;

        for (int js = 0; js < ns; js++) {
            auto c = std::complex<double>(0.0, 0.0);
            for (int ls = 0; ls < ns; ls++) {
                c += std::conj(evec_harm_renormalized[ik][js][ls]) * evec[ik][is][ls];
            }
            if (eval_harm_renormalized[ik][js] < 0.0) {
                omega2_harm = -pow2(eval_harm_renormalized[ik][js]);
            } else {
                omega2_harm = pow2(eval_harm_renormalized[ik][js]);
            }
            tmp_c += std::conj(c) * omega2_harm * c;
        }

        if (classical) {
            ret += (tmp_c.real() - omega * omega) * fC(omega, temp) / (4.0 * omega);
        } else {
            ret += (tmp_c.real() - omega * omega) * (1.0 + 2.0 * fB(omega, temp)) / (8.0 * omega);
        }
    }

    return ret / static_cast<double>(nk);
}

auto Thermodynamics::compute_FE_total(const unsigned int iT, const double fe_qha, const double dfe_scph,
                                      const double v0_renorm, const bool is_scph_mode) const -> double
{
    double fe_total = fe_qha;
    // skip scph correction for QHA + structural optimization
    if (is_scph_mode) {
        fe_total += dfe_scph;
    }
    if (calc_FE_bubble) {
        fe_total += FE_bubble[iT];
    }
    // The renormalized static potential of the relaxed structure; the caller
    // passes 0.0 when no structural optimization is performed.
    fe_total += v0_renorm;

    return fe_total;
}
