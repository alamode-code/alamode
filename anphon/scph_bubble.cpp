/*
 scph_bubble.cpp

 Copyright (c) 2015 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

/*
 Functions for computing bubble diagram corrections to SCPH.
 The bubble self-energy provides higher-order anharmonic corrections
 on top of the self-consistent phonon (SCPH) calculation.

 Functions included:
 - compute_free_energy_bubble_SCPH: Compute free energy from bubble diagrams
 - bubble_correction: Calculate bubble self-energy corrections to frequencies
 - get_bubble_selfenergy: Compute bubble self-energy at specific k-point and mode
*/

#include <complex>
#include <iomanip>
#include <iostream>
#include <vector>
#include "anharmonic_core.h"
#include "constants.h"
#include "dynamical.h"
#include "error.h"
#include "integration.h"
#include "kpoint.h"
#include "memory.h"
#include "mpi_common.h"
#include "phonon_dos.h"
#include "scph.h"
#include "system.h"
#include "thermodynamics.h"
#include "write_phonons.h"

using namespace PHON_NS;

void Scph::compute_free_energy_bubble_SCPH(const unsigned int kmesh[3], std::complex<double> ****delta_dymat_scph)
{
    const auto NT = static_cast<unsigned int>((system->Tmax - system->Tmin) / system->dT) + 1;
    const auto nk_ref = dos->kmesh_dos->nk;
    const auto ns = dynamical->neval;
    double ***eval;
    std::complex<double> ****evec;

    if (mympi->my_rank == 0) {
        std::cout << '\n';
        std::cout << " -----------------------------------------------------------------\n";
        std::cout << " Calculating the vibrational free energy from the Bubble diagram \n";
        std::cout << " on top of the SCPH calculation.\n\n";
        std::cout << " This calculation requires allocation of additional memory:\n";

        size_t nsize = nk_ref * ns * ns * NT * sizeof(std::complex<double>) + nk_ref * ns * NT * sizeof(double);

        const auto nsize_dble = static_cast<double>(nsize) / 1000000000.0;
        std::cout << "  Estimated memory usage per MPI process: " << std::setw(10) << std::fixed << std::setprecision(4)
                  << nsize_dble << " GByte.\n";
        std::cout << "  To avoid possible faults associated with insufficient memory,\n"
                     "  please reduce the number of MPI processes per node and/or\n"
                     "  the number of temperature grids.\n\n";
    }

    allocate(thermodynamics->FE_bubble, NT);
    allocate(eval, NT, nk_ref, ns);
    allocate(evec, NT, nk_ref, ns, ns); // This requires lots of RAM

    for (auto iT = 0; iT < NT; ++iT) {
        dynamical->exec_interpolation(kmesh,
                                      delta_dymat_scph[iT],
                                      nk_ref,
                                      dos->kmesh_dos->xk,
                                      dos->kmesh_dos->kvec_na,
                                      eval[iT],
                                      evec[iT],
                                      dymat_harm_short,
                                      dymat_harm_long,
                                      mindist_list);
    }

    thermodynamics->compute_FE_bubble_SCPH(eval,
                                           evec,
                                           thermodynamics->FE_bubble,
                                           *system,
                                           *dos->kmesh_dos,
                                           symmetry->SymmList,
                                           *anharmonic_core,
                                           dynamical->neval,
                                           mympi->my_rank,
                                           mympi->nprocs);

    deallocate(eval);
    deallocate(evec);

    if (mympi->my_rank == 0) {
        std::cout << " done!\n\n";
    }
}

void Scph::bubble_correction(std::complex<double> ****delta_dymat_scph,
                             std::complex<double> ****delta_dymat_scph_plus_bubble)
{
    const auto NT = static_cast<unsigned int>((system->Tmax - system->Tmin) / system->dT) + 1;
    const auto ns = dynamical->neval;

    auto epsilon = integration->epsilon;
    const auto nk_irred_interpolate = kmesh_coarse->nk_irred;
    const auto nk_scph = kmesh_dense->nk;

    double **eval = nullptr;
    double ***eval_bubble = nullptr;
    std::complex<double> ***evec;
    double *real_self = nullptr;
    std::vector<std::complex<double>> omegalist;

    if (mympi->my_rank == 0) {
        std::cout << '\n';
        std::cout << " -----------------------------------------------------------------\n";
        std::cout << " Calculating the bubble self-energy \n";
        std::cout << " on top of the SCPH calculation.\n\n";
    }

    allocate(eval, nk_scph, ns);
    allocate(evec, nk_scph, ns, ns);

    if (mympi->my_rank == 0) {
        allocate(eval_bubble, NT, nk_scph, ns);
        for (auto iT = 0; iT < NT; ++iT) {
            for (auto ik = 0; ik < nk_scph; ++ik) {
                for (auto is = 0; is < ns; ++is) {
                    eval_bubble[iT][ik][is] = 0.0;
                }
            }
        }
        allocate(real_self, ns);
    }

    std::vector<int> *degeneracy_at_k;
    allocate(degeneracy_at_k, nk_scph);

    for (auto iT = 0; iT < NT; ++iT) {
        const auto temp = system->Tmin + system->dT * float(iT);

        dynamical->exec_interpolation(kmesh_interpolate,
                                      delta_dymat_scph[iT],
                                      nk_scph,
                                      kmesh_dense->xk,
                                      kmesh_dense->kvec_na,
                                      eval,
                                      evec,
                                      dymat_harm_short,
                                      dymat_harm_long,
                                      mindist_list);

        find_degeneracy(degeneracy_at_k, nk_scph, eval);

        if (mympi->my_rank == 0) std::cout << " Temperature (K) : " << std::setw(6) << temp << '\n';

        for (auto ik = 0; ik < nk_irred_interpolate; ++ik) {

            auto knum_interpolate = kmesh_coarse->kpoint_irred_all[ik][0].knum;
            auto knum = kmap_coarse_to_dense[knum_interpolate];

            if (mympi->my_rank == 0) {
                std::cout << "  Irred. k: " << std::setw(5) << ik + 1 << " (";
                for (auto m = 0; m < 3; ++m)
                    std::cout << std::setw(15) << kmesh_dense->xk[knum][m];
                std::cout << ")\n";
            }

            for (unsigned int snum = 0; snum < ns; ++snum) {

                if (eval[knum][snum] < eps8) {
                    if (mympi->my_rank == 0) real_self[snum] = 0.0;
                } else {
                    omegalist.clear();

                    if (bubble == 1) {

                        omegalist.push_back(im * epsilon);

                        auto se_bubble =
                            get_bubble_selfenergy(kmesh_dense, ns, eval, evec, knum, snum, temp, omegalist);

                        if (mympi->my_rank == 0) real_self[snum] = se_bubble[0].real();

                    } else if (bubble == 2) {

                        omegalist.push_back(eval[knum][snum] + im * epsilon);

                        auto se_bubble =
                            get_bubble_selfenergy(kmesh_dense, ns, eval, evec, knum, snum, temp, omegalist);

                        if (mympi->my_rank == 0) real_self[snum] = se_bubble[0].real();

                    } else if (bubble == 3) {

                        auto maxfreq = eval[knum][snum] + 50.0 * time_ry / Hz_to_kayser;
                        auto minfreq = eval[knum][snum] - 50.0 * time_ry / Hz_to_kayser;

                        if (minfreq < 0.0) minfreq = 0.0;

                        const auto domega = 0.1 * time_ry / Hz_to_kayser;
                        auto nomega = static_cast<unsigned int>((maxfreq - minfreq) / domega) + 1;

                        for (auto iomega = 0; iomega < nomega; ++iomega) {
                            omegalist.push_back(minfreq + static_cast<double>(iomega) * domega + im * epsilon);
                        }

                        auto se_bubble =
                            get_bubble_selfenergy(kmesh_dense, ns, eval, evec, knum, snum, temp, omegalist);

                        if (mympi->my_rank == 0) {

                            std::vector<double> nonlinear_func(nomega);
                            for (auto iomega = 0; iomega < nomega; ++iomega) {
                                nonlinear_func[iomega] = omegalist[iomega].real() * omegalist[iomega].real() -
                                                         eval[knum][snum] * eval[knum][snum] +
                                                         2.0 * eval[knum][snum] * se_bubble[iomega].real();
                            }

                            // find a root of nonlinear_func = 0 from the sign change.
                            int count_root = 0;
                            std::vector<unsigned int> root_index;

                            for (auto iomega = 0; iomega < nomega - 1; ++iomega) {
                                if (nonlinear_func[iomega] * nonlinear_func[iomega + 1] < 0.0) {
                                    ++count_root;
                                    root_index.push_back(iomega);
                                }
                            }

                            if (count_root == 0) {
                                warn("bubble_correction",
                                     "Could not find a root in the nonlinear equation at this temperature. "
                                     "Use the w=0 component.");

                                real_self[snum] = se_bubble[0].real();

                            } else {
                                if (count_root > 1) {
                                    warn("bubble_correction",
                                         "Multiple roots were found in the nonlinear equation at this temperature. "
                                         "Use the lowest-frequency solution");
                                    std::cout << "   solution found at the following frequencies:\n";
                                    for (auto iroot = 0; iroot < count_root; ++iroot) {
                                        std::cout << std::setw(15)
                                                  << writes->in_kayser(omegalist[root_index[iroot]].real());
                                    }
                                    std::cout << '\n';
                                }

                                // Instead of performing a linear interpolation (secant method) of nonlinear_func,
                                // we interpolate the bubble self-energy. Since the frequency grid is dense (0.1 cm^-1 step),
                                // this approximation should not make any problems (hopefully).

                                double omega_solution =
                                    omegalist[root_index[0] + 1].real() -
                                    nonlinear_func[root_index[0] + 1] * domega /
                                        (nonlinear_func[root_index[0] + 1] - nonlinear_func[root_index[0]]);

                                real_self[snum] =
                                    (se_bubble[root_index[0] + 1].real() - se_bubble[root_index[0]].real()) *
                                        (omega_solution - omegalist[root_index[0] + 1].real()) / domega +
                                    se_bubble[root_index[0] + 1].real();
                            }
                        }
                    }
                }
                if (mympi->my_rank == 0) {
                    std::cout << "   branch : " << std::setw(5) << snum + 1;
                    std::cout << " omega (SC1) = " << std::setw(15) << writes->in_kayser(eval[knum][snum])
                              << " (cm^-1); ";
                    std::cout << " Re[Self] = " << std::setw(15) << writes->in_kayser(real_self[snum]) << " (cm^-1)\n";
                }
            }

            if (mympi->my_rank == 0) {
                // average self energy of degenerate modes
                int ishift = 0;
                double real_self_avg = 0.0;

                for (const auto &it: degeneracy_at_k[knum]) {
                    for (auto m = 0; m < it; ++m) {
                        real_self_avg += real_self[m + ishift];
                    }
                    real_self_avg /= static_cast<double>(it);

                    for (auto m = 0; m < it; ++m) {
                        real_self[m + ishift] = real_self_avg;
                    }
                    real_self_avg = 0.0;
                    ishift += it;
                }

                for (unsigned int snum = 0; snum < ns; ++snum) {
                    eval_bubble[iT][knum][snum] =
                        eval[knum][snum] * eval[knum][snum] - 2.0 * eval[knum][snum] * real_self[snum];
                    for (auto jk = 1; jk < kmesh_coarse->kpoint_irred_all[ik].size(); ++jk) {
                        auto knum2 = kmap_coarse_to_dense[kmesh_coarse->kpoint_irred_all[ik][jk].knum];
                        eval_bubble[iT][knum2][snum] = eval_bubble[iT][knum][snum];
                    }
                }

                std::cout << '\n';
            }
        }

        if (mympi->my_rank == 0) {
            dynamical->calc_new_dymat_with_evec(delta_dymat_scph_plus_bubble[iT],
                                                eval_bubble[iT],
                                                evec,
                                                kmesh_coarse,
                                                kmap_coarse_to_dense);
        }
    }

    deallocate(eval);
    deallocate(evec);
    deallocate(degeneracy_at_k);

    if (eval_bubble) deallocate(eval_bubble);

    if (mympi->my_rank == 0) {
        std::cout << " done!\n\n";
    }
}

std::vector<std::complex<double>> Scph::get_bubble_selfenergy(const KpointMeshUniform *kmesh_in,
                                                              const unsigned int ns_in, const double *const *eval_in,
                                                              const std::complex<double> *const *const *evec_in,
                                                              const unsigned int knum, const unsigned int snum,
                                                              const double temp_in,
                                                              const std::vector<std::complex<double>> &omegalist)
{
    unsigned int arr_cubic[3];
    double xk_tmp[3];
    std::complex<double> omega_sum[2];

    double factor = 1.0 / (static_cast<double>(kmesh_in->nk) * std::pow(2.0, 4));
    const auto ns2 = ns_in * ns_in;
    const auto nks = kmesh_in->nk * ns2;

    double n1, n2;
    double f1, f2;

    auto knum_minus = kmesh_in->kindex_minus_xk[knum];
    arr_cubic[0] = ns_in * knum_minus + snum;

    std::vector<std::complex<double>> se_bubble(omegalist.size());

    const auto nomega = omegalist.size();

    std::complex<double> *ret_sum, *ret_mpi;
    allocate(ret_sum, nomega);
    allocate(ret_mpi, nomega);

    for (auto iomega = 0; iomega < nomega; ++iomega) {
        ret_sum[iomega] = std::complex<double>(0.0, 0.0);
        ret_mpi[iomega] = std::complex<double>(0.0, 0.0);
    }

    for (auto iks = mympi->my_rank; iks < nks; iks += mympi->nprocs) {

        auto ik1 = iks / ns2;
        auto is1 = (iks % ns2) / ns_in;
        auto is2 = iks % ns_in;

        for (auto m = 0; m < 3; ++m)
            xk_tmp[m] = kmesh_in->xk[knum][m] - kmesh_in->xk[ik1][m];
        auto ik2 = kmesh_in->get_knum(xk_tmp);

        double omega1 = eval_in[ik1][is1];
        double omega2 = eval_in[ik2][is2];

        arr_cubic[1] = ns_in * ik1 + is1;
        arr_cubic[2] = ns_in * ik2 + is2;

        double v3_tmp = std::norm(anharmonic_core->V3(arr_cubic, kmesh_in->xk, eval_in, evec_in, phase_factor));

        if (thermodynamics->classical) {
            n1 = thermodynamics->fC(omega1, temp_in);
            n2 = thermodynamics->fC(omega2, temp_in);
            f1 = n1 + n2;
            f2 = n2 - n1;
        } else {
            n1 = thermodynamics->fB(omega1, temp_in);
            n2 = thermodynamics->fB(omega2, temp_in);
            f1 = n1 + n2 + 1.0;
            f2 = n2 - n1;
        }
        for (auto iomega = 0; iomega < nomega; ++iomega) {
            omega_sum[0] = 1.0 / (omegalist[iomega] + omega1 + omega2) - 1.0 / (omegalist[iomega] - omega1 - omega2);
            omega_sum[1] = 1.0 / (omegalist[iomega] + omega1 - omega2) - 1.0 / (omegalist[iomega] - omega1 + omega2);
            ret_mpi[iomega] += v3_tmp * (f1 * omega_sum[0] + f2 * omega_sum[1]);
        }
    }
    for (auto iomega = 0; iomega < nomega; ++iomega) {
        ret_mpi[iomega] *= factor;
    }

    MPI_Reduce(&ret_mpi[0], &ret_sum[0], nomega, MPI_COMPLEX16, MPI_SUM, 0, MPI_COMM_WORLD);

    for (auto iomega = 0; iomega < nomega; ++iomega) {
        se_bubble[iomega] = ret_sum[iomega];
    }

    deallocate(ret_mpi);
    deallocate(ret_sum);

    return se_bubble;
}
