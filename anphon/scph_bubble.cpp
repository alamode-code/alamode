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

#include <algorithm>
#include <complex>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
#include "anharmonic_core.h"
#include "constants.h"
#include "degeneracy_utils.h"
#include "dynamical.h"
#include "error.h"
#include "fcs_phonon.h"
#include "ifc_derivative.h"
#include "integration.h"
#include "kpoint.h"
#include "memory.h"
#include "mpi_common.h"
#include "phonon_dos.h"
#include "relaxation.h"
#include "scph.h"
#include "selfenergy.h"
#include "system.h"
#include "thermodynamics.h"

using namespace PHON_NS;

void Scph::compute_free_energy_bubble_SCPH(const unsigned int kmesh[3], std::complex<double> ****delta_dymat_scph)
{
    const auto NT = system->get_num_temperature_points();
    const auto nk_ref = dos->kmesh_dos->nk;
    const auto ns = dynamical->neval;
    NDArray<double, 3> eval;
    NDArray<std::complex<double>, 4> evec;

    if (run.my_rank == 0) {
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
                                      mindist_list,
                                      fcs_phonon->force_constant_with_cell[0],
                                      *dielec,
                                      *ewald);
    }

    thermodynamics->compute_FE_bubble_SCPH(eval,
                                           evec,
                                           thermodynamics->FE_bubble,
                                           *system,
                                           *dos->kmesh_dos.get(),
                                           symmetry->SymmList,
                                           *anharmonic_core,
                                           dynamical->neval,
                                           run.my_rank,
                                           run.nprocs);

    eval.clear();
    evec.clear();

    if (run.my_rank == 0) {
        std::cout << " done!\n\n";
    }
}

void Scph::bubble_correction(std::complex<double> ****delta_dymat_scph,
                             std::complex<double> ****delta_dymat_scph_plus_bubble)
{
    const auto NT = system->get_num_temperature_points();
    const auto ns = dynamical->neval;

    auto epsilon = integration->epsilon;
    const auto nk_irred_interpolate = kmesh_coarse->nk_irred;

    // The bubble self-energy is summed on KMESH_BUBBLE; the SCPH mesh and its
    // phase cache are reused when the two meshes coincide.
    MPI_Bcast(&kmesh_bubble[0], 3, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    std::unique_ptr<KpointMeshUniform> kmesh_bubble_own;
    std::unique_ptr<PhaseFactorCache> phase_factor_own;
    std::vector<int> kmap_coarse_to_bubble_own;
    const KpointMeshUniform *kmesh_b = kmesh_dense.get();
    const PhaseFactorCache *phase_b = phase_factor.get();
    const std::vector<int> *kmap_b = &kmap_coarse_to_dense;
    if (kmesh_bubble[0] != kmesh_dense->nk_i[0] || kmesh_bubble[1] != kmesh_dense->nk_i[1] ||
        kmesh_bubble[2] != kmesh_dense->nk_i[2])
    {
        kmesh_bubble_own = std::make_unique<KpointMeshUniform>(kmesh_bubble);
        kmesh_bubble_own->setup(symmetry->SymmList,
                                system->get_primcell().reciprocal_lattice_vector,
                                symmetry->use_time_reversal && symmetry->time_reversal_sym);
        if (kpoint->get_kmap_coarse_to_dense(kmesh_coarse.get(), kmesh_bubble_own.get(), kmap_coarse_to_bubble_own) ==
            1)
        {
            exit("bubble_correction", "KMESH_BUBBLE should be an integral multiple of KMESH_INTERPOLATE");
        }
        phase_factor_own = std::make_unique<PhaseFactorCache>(kmesh_bubble_own->nk_i);
        phase_factor_own->create(true);
        kmesh_b = kmesh_bubble_own.get();
        phase_b = phase_factor_own.get();
        kmap_b = &kmap_coarse_to_bubble_own;
    }
    const auto nk_scph = kmesh_b->nk;

    NDArray<double, 2> eval;
    NDArray<double, 3> eval_bubble;
    NDArray<std::complex<double>, 3> evec;
    NDArray<double, 1> real_self;
    std::vector<std::complex<double>> omegalist;

    if (run.my_rank == 0) {
        std::cout << '\n';
        std::cout << " -----------------------------------------------------------------\n";
        std::cout << " Calculating the bubble self-energy \n";
        std::cout << " on top of the SCPH calculation.\n";
        std::cout << "  KMESH_BUBBLE: " << kmesh_b->nk_i[0] << " " << kmesh_b->nk_i[1] << " " << kmesh_b->nk_i[2]
                  << " (" << kmesh_b->nk << " points)\n\n";
    }

    eval.resize(nk_scph, ns);
    evec.resize(nk_scph, ns, ns);
    selfenergy->setup_selfenergy(dynamical->neval,
                                 integration->epsilon,
                                 thermodynamics->classical,
                                 run.my_rank,
                                 run.nprocs);

    if (run.my_rank == 0) {
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

    // Relaxed run (RELAX_STR != 0). The SCP propagators already belong to the
    // relaxed structure of each temperature: the stored correction is the full
    // SCP matrix minus the reference harmonic one, so it contains the q0 and
    // strain renormalization, and the phases exp(ik.R) of fractional k do not
    // change under the homogeneous strain. Only the cubic IFCs do, to
    // Phi3 + Phi4 : d(T). Temperatures whose SCP iteration or structure did not
    // converge keep the SCPH result without the bubble correction.
    // The recorded structure is the one the optimizer accepted, one update past
    // the structure of the last SCP solve (see ScphRelaxationModel::
    // finalize_temperature); the two differ within COORD_CONV_TOL/CELL_CONV_TOL.
    const auto relaxed = relaxation->relax_str != 0;
    std::vector<double> u_tensor_all, u0_all;
    std::vector<int> use_temp(NT, 1);
    if (relaxed) {
        const auto nt = static_cast<size_t>(NT);
        const auto nsz = static_cast<size_t>(ns);
        if (nt > static_cast<size_t>(std::numeric_limits<int>::max()) / std::max(nsz, size_t{9})) {
            exit("bubble_correction", "Too many temperatures x modes to broadcast the relaxed structures.");
        }
        if (run.my_rank == 0) {
            if (relaxed_structure.u_tensor.size() != nt * 9 || relaxed_structure.u0.size() != nt * nsz ||
                relaxed_structure_recorded.size() != nt)
            {
                exit("bubble_correction", "The relaxed structures of this run are not available.");
            }
            u_tensor_all = relaxed_structure.u_tensor;
            u0_all = relaxed_structure.u0;
            for (size_t iT = 0; iT < nt; ++iT) {
                // An unrecorded temperature was never visited by the loop, so its
                // SCP matrix does not exist either: nothing to keep or correct.
                if (!relaxed_structure_recorded[iT]) {
                    exit("bubble_correction",
                         "A temperature of the grid has no relaxed structure; the structural\n"
                         " optimization loop skipped it. Check TMIN, TMAX and DT.");
                }
                const auto conv_scph = converged_scph_temp.size() != nt || converged_scph_temp[iT];
                const auto conv_str = converged_str_temp.size() != nt || converged_str_temp[iT];
                use_temp[iT] = conv_scph && conv_str ? 1 : 0;
            }
        } else {
            u_tensor_all.resize(nt * 9);
            u0_all.resize(nt * nsz);
        }
        MPI_Bcast(u_tensor_all.data(), static_cast<int>(nt * 9), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(u0_all.data(), static_cast<int>(nt * nsz), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(use_temp.data(), static_cast<int>(NT), MPI_INT, 0, MPI_COMM_WORLD);
        if (run.my_rank == 0) {
            std::cout << "  RELAX_STR != 0: the cubic IFCs are deformed to the relaxed structure\n"
                         "  of each temperature (Phi3 + Phi4 : d).\n\n";
        }
    }

    for (auto iT = 0; iT < NT; ++iT) {
        const auto temp = system->Tmin + system->dT * float(iT);

        if (relaxed) {
            if (!use_temp[iT]) {
                if (run.my_rank == 0) {
                    std::cout << " Temperature (K) : " << std::setw(6) << temp
                              << " : skipped (SCPH or structure not converged); the SCPH result is kept.\n\n";
                    for (auto is = 0; is < ns; ++is) {
                        for (auto js = 0; js < ns; ++js) {
                            for (auto ik = 0; ik < kmesh_coarse->nk; ++ik) {
                                delta_dymat_scph_plus_bubble[iT][is][js][ik] = delta_dymat_scph[iT][is][js][ik];
                            }
                        }
                    }
                }
                continue;
            }
            const auto offset = static_cast<size_t>(iT) * ns;
            const std::vector<double> u0_T(u0_all.begin() + offset, u0_all.begin() + offset + ns);
            std::vector<FcsArrayWithCell> fc3_deformed;
            DerivativeIFC::compute_deformed_cubic_ifcs(fcs_phonon->force_constant_with_cell,
                                                       &u_tensor_all[static_cast<size_t>(iT) * 9],
                                                       u0_T,
                                                       system->get_primcell().lattice_vector,
                                                       fc3_deformed);
            anharmonic_core->replace_cubic(fc3_deformed);
        }

        dynamical->exec_interpolation(kmesh_interpolate,
                                      delta_dymat_scph[iT],
                                      nk_scph,
                                      kmesh_b->xk,
                                      kmesh_b->kvec_na,
                                      eval,
                                      evec,
                                      mindist_list,
                                      fcs_phonon->force_constant_with_cell[0],
                                      *dielec,
                                      *ewald);

        for (unsigned int ik = 0; ik < nk_scph; ++ik) find_degenerate_groups(ns, eval[ik], degeneracy_at_k[ik]);

        if (run.my_rank == 0) std::cout << " Temperature (K) : " << std::setw(6) << temp << '\n';

        for (auto ik = 0; ik < nk_irred_interpolate; ++ik) {

            auto knum_interpolate = kmesh_coarse->kpoint_irred_all[ik][0].knum;
            auto knum = (*kmap_b)[knum_interpolate];

            if (run.my_rank == 0) {
                std::cout << "  Irred. k: " << std::setw(5) << ik + 1 << " (";
                for (auto m = 0; m < 3; ++m) std::cout << std::setw(15) << kmesh_b->xk[knum][m];
                std::cout << ")\n";
            }

            for (unsigned int snum = 0; snum < ns; ++snum) {

                if (eval[knum][snum] < eps8) {
                    if (run.my_rank == 0) real_self[snum] = 0.0;
                } else {
                    omegalist.clear();

                    if (bubble == 1) {

                        omegalist.push_back(im * epsilon);

                        auto se_bubble = selfenergy->get_bubble_selfenergy(kmesh_b,
                                                                           ns,
                                                                           eval,
                                                                           evec,
                                                                           knum,
                                                                           snum,
                                                                           temp,
                                                                           omegalist,
                                                                           phase_b,
                                                                           *anharmonic_core);

                        if (run.my_rank == 0) real_self[snum] = se_bubble[0].real();

                    } else if (bubble == 2) {

                        omegalist.push_back(eval[knum][snum] + im * epsilon);

                        auto se_bubble = selfenergy->get_bubble_selfenergy(kmesh_b,
                                                                           ns,
                                                                           eval,
                                                                           evec,
                                                                           knum,
                                                                           snum,
                                                                           temp,
                                                                           omegalist,
                                                                           phase_b,
                                                                           *anharmonic_core);

                        if (run.my_rank == 0) real_self[snum] = se_bubble[0].real();

                    } else if (bubble == 3) {

                        auto maxfreq = eval[knum][snum] + 50.0 * time_ry / Hz_to_kayser;
                        auto minfreq = eval[knum][snum] - 50.0 * time_ry / Hz_to_kayser;

                        if (minfreq < 0.0) minfreq = 0.0;

                        const auto domega = 0.1 * time_ry / Hz_to_kayser;
                        auto nomega = static_cast<unsigned int>((maxfreq - minfreq) / domega) + 1;

                        for (auto iomega = 0; iomega < nomega; ++iomega) {
                            omegalist.push_back(minfreq + static_cast<double>(iomega) * domega + im * epsilon);
                        }

                        auto se_bubble = selfenergy->get_bubble_selfenergy(kmesh_b,
                                                                           ns,
                                                                           eval,
                                                                           evec,
                                                                           knum,
                                                                           snum,
                                                                           temp,
                                                                           omegalist,
                                                                           phase_b,
                                                                           *anharmonic_core);

                        if (run.my_rank == 0) {

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
                if (run.my_rank == 0) {
                    // Explicit format: the stream state left by earlier output
                    // otherwise decides the precision (two digits in some runs).
                    const auto flags = std::cout.flags();
                    const auto prec = std::cout.precision();
                    std::cout << std::scientific << std::setprecision(6);
                    std::cout << "   branch : " << std::setw(5) << snum + 1;
                    std::cout << " omega (SC1) = " << std::setw(15) << in_kayser(eval[knum][snum]) << " (cm^-1); ";
                    std::cout << " Re[Self] = " << std::setw(15) << in_kayser(real_self[snum]) << " (cm^-1)\n";
                    std::cout.flags(flags);
                    std::cout.precision(prec);
                }
            }

            if (run.my_rank == 0) {
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
                        auto knum2 = (*kmap_b)[kmesh_coarse->kpoint_irred_all[ik][jk].knum];
                        eval_bubble[iT][knum2][snum] = eval_bubble[iT][knum][snum];
                    }
                }

                std::cout << '\n';
            }
        }

        if (run.my_rank == 0) {
            dynamical->calc_new_dymat_with_evec(delta_dymat_scph_plus_bubble[iT],
                                                eval_bubble[iT],
                                                evec,
                                                kmesh_coarse.get(),
                                                *kmap_b,
                                                fcs_phonon->force_constant_with_cell[0]);
        }
    }

    // Later consumers expect the reference cubic IFCs.
    if (relaxed) anharmonic_core->replace_cubic(fcs_phonon->force_constant_with_cell[1]);

    eval.clear();
    evec.clear();
    degeneracy_at_k.clear();

    eval_bubble.clear();

    if (run.my_rank == 0) {
        std::cout << " done!\n\n";
    }
}
