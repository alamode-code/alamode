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
*/

#include <complex>
#include <iomanip>
#include <iostream>
#include <vector>
#include "anharmonic_core.h"
#include "constants.h"
#include "degeneracy_utils.h"
#include "dynamical.h"
#include "error.h"
#include "integration.h"
#include "kpoint.h"
#include "memory.h"
#include "mpi_common.h"
#include "phonon_dos.h"
#include "scph.h"
#include "selfenergy.h"
#include "system.h"
#include "thermodynamics.h"

using namespace PHON_NS;

void Scph::compute_free_energy_bubble_SCPH(const unsigned int kmesh[3], std::complex<double> ****delta_dymat_scph)
{
    const auto NT = static_cast<unsigned int>((system->Tmax - system->Tmin) / system->dT) + 1;
    const auto nk_ref = dos->kmesh_dos->nk;
    const auto ns = dynamical->neval;
    NDArray<double, 3> eval;
    NDArray<std::complex<double>, 4> evec;

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

    thermodynamics->FE_bubble.resize(NT);
    eval.resize(NT, nk_ref, ns);
    evec.resize(NT, nk_ref, ns, ns); // This requires lots of RAM

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
                                           *dos->kmesh_dos.get(),
                                           symmetry->SymmList,
                                           *anharmonic_core,
                                           dynamical->neval,
                                           mympi->my_rank,
                                           mympi->nprocs);

    eval.clear();
    evec.clear();

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

    NDArray<double, 2> eval;
    NDArray<double, 3> eval_bubble;
    NDArray<std::complex<double>, 3> evec;
    NDArray<double, 1> real_self;
    std::vector<std::complex<double>> omegalist;

    if (mympi->my_rank == 0) {
        std::cout << '\n';
        std::cout << " -----------------------------------------------------------------\n";
        std::cout << " Calculating the bubble self-energy \n";
        std::cout << " on top of the SCPH calculation.\n\n";
    }

    eval.resize(nk_scph, ns);
    evec.resize(nk_scph, ns, ns);
    selfenergy->setup_selfenergy(dynamical->neval,
                                 integration->epsilon,
                                 thermodynamics->classical,
                                 symmetry->SymmList,
                                 *anharmonic_core,
                                 mympi->my_rank,
                                 mympi->nprocs);

    if (mympi->my_rank == 0) {
        eval_bubble.resize(NT, nk_scph, ns);
        for (auto iT = 0; iT < NT; ++iT) {
            for (auto ik = 0; ik < nk_scph; ++ik) {
                for (auto is = 0; is < ns; ++is) {
                    eval_bubble[iT][ik][is] = 0.0;
                }
            }
        }
        real_self.resize(ns);
    }

    NDArray<std::vector<int>, 1> degeneracy_at_k;
    degeneracy_at_k.resize(nk_scph);

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

        for (unsigned int ik = 0; ik < nk_scph; ++ik) find_degenerate_groups(ns, eval[ik], degeneracy_at_k[ik]);

        if (mympi->my_rank == 0) std::cout << " Temperature (K) : " << std::setw(6) << temp << '\n';

        for (auto ik = 0; ik < nk_irred_interpolate; ++ik) {

            auto knum_interpolate = kmesh_coarse->kpoint_irred_all[ik][0].knum;
            auto knum = kmap_coarse_to_dense[knum_interpolate];

            if (mympi->my_rank == 0) {
                std::cout << "  Irred. k: " << std::setw(5) << ik + 1 << " (";
                for (auto m = 0; m < 3; ++m) std::cout << std::setw(15) << kmesh_dense->xk[knum][m];
                std::cout << ")\n";
            }

            for (unsigned int snum = 0; snum < ns; ++snum) {

                if (eval[knum][snum] < eps8) {
                    if (mympi->my_rank == 0) real_self[snum] = 0.0;
                } else {
                    omegalist.clear();

                    if (bubble == 1) {

                        omegalist.push_back(im * epsilon);

                        auto se_bubble = selfenergy->get_bubble_selfenergy(kmesh_dense.get(),
                                                                           ns,
                                                                           eval,
                                                                           evec,
                                                                           knum,
                                                                           snum,
                                                                           temp,
                                                                           omegalist,
                                                                           phase_factor.get());

                        if (mympi->my_rank == 0) real_self[snum] = se_bubble[0].real();

                    } else if (bubble == 2) {

                        omegalist.push_back(eval[knum][snum] + im * epsilon);

                        auto se_bubble = selfenergy->get_bubble_selfenergy(kmesh_dense.get(),
                                                                           ns,
                                                                           eval,
                                                                           evec,
                                                                           knum,
                                                                           snum,
                                                                           temp,
                                                                           omegalist,
                                                                           phase_factor.get());

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

                        auto se_bubble = selfenergy->get_bubble_selfenergy(kmesh_dense.get(),
                                                                           ns,
                                                                           eval,
                                                                           evec,
                                                                           knum,
                                                                           snum,
                                                                           temp,
                                                                           omegalist,
                                                                           phase_factor.get());

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
                                        std::cout << std::setw(15) << in_kayser(omegalist[root_index[iroot]].real());
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
                    std::cout << " omega (SC1) = " << std::setw(15) << in_kayser(eval[knum][snum]) << " (cm^-1); ";
                    std::cout << " Re[Self] = " << std::setw(15) << in_kayser(real_self[snum]) << " (cm^-1)\n";
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
                                                kmesh_coarse.get(),
                                                kmap_coarse_to_dense);
        }
    }

    eval.clear();
    evec.clear();
    degeneracy_at_k.clear();

    eval_bubble.clear();

    if (mympi->my_rank == 0) {
        std::cout << " done!\n\n";
    }
}
