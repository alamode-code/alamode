/*
 scph.cpp

 Copyright (c) 2015 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "scph.h"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <fftw3.h>
#include <iomanip>
#include <iostream>
#include <vector>
#include "anharmonic_core.h"
#include "constants.h"
#include "dielec.h"
#include "diis.h"
#include "dynamical.h"
#include "error.h"
#include "ewald.h"
#include "fcs_phonon.h"
#include "integration.h"
#include "kpoint.h"
#include "memory.h"
#include "mpi.h"
#include "mpi_common.h"
#include "parsephon.h"
#include "phonon_dos.h"
#include "relaxation.h"
#include "symmetry_core.h"
#include "system.h"
#include "thermodynamics.h"
#include "timer.h"
#include "write_phonons.h"

using namespace PHON_NS;

Scph::Scph(PHON *phon) : ScphQhaCommon(phon)
{
    set_default_variables();
}

Scph::~Scph()
{
    deallocate_variables();
}

void Scph::set_default_variables()
{
    restart_scph = false;
    warmstart_scph = false;
    lower_temp = true;
    tolerance_scph = 1.0e-10;
    mixalpha = 0.1;
    maxiter = 100;
    print_self_consistent_fc2 = false;
    selfenergy_offdiagonal = true;
    mix_anderson_ratio = 1.0;

    bubble = 0;
    compute_Cv_anharmonic = 0;

    initialize_variables();
}


void Scph::setup_scph()
{
    MPI_Bcast(&bubble, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Prepare coarse/dense q-point meshes used in SCPH iterations and postprocess.
    setup_kmesh(kmesh_scph, kmesh_interpolate, "SCPH", "KMESH_INTERPOLATE should be a integral multiple of KMESH_SCPH");
    // Allocate and cache harmonic eigenvectors/eigenvalues on the coarse mesh.
    setup_eigvecs();

    // Precompute reciprocal-space anharmonic interactions (phi3/phi4).
    const auto relax_mode = to_relaxation_str_mode(relaxation->relax_str);
    setup_pp_interaction(relax_mode != RelaxationStrMode::None || bubble > 0);
    // Build structural/symmetry data used for IFC reconstruction and matrix symmetrization.
    setup_structural_data();
}

void Scph::exec_scph()
{
    const auto ns = dynamical->neval;
    const auto Tmin = system->Tmin;
    const auto Tmax = system->Tmax;
    const auto dT = system->dT;

    std::complex<double> ****delta_dymat_scph = nullptr;
    std::complex<double> ****delta_dymat_scph_plus_bubble = nullptr;
    // change of harmonic dymat by IFC renormalization
    std::complex<double> ****delta_harmonic_dymat_renormalize = nullptr;

    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;

    MPI_Bcast(&restart_scph, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&selfenergy_offdiagonal, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ialgo, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    allocate(delta_dymat_scph, NT, ns, ns, kmesh_coarse->nk);
    allocate(delta_harmonic_dymat_renormalize, NT, ns, ns, kmesh_coarse->nk);

    zerofill_harmonic_dymat_renormalize(delta_harmonic_dymat_renormalize, NT);

    const auto relax_mode = to_relaxation_str_mode(relaxation->relax_str);

    if (relax_mode != RelaxationStrMode::None && thermodynamics->calc_FE_bubble) {
        exit("exec_scph", "Sorry, RELAX_STR!=0 can't be used with bubble correction of the free energy.");
    }
    if (relax_mode != RelaxationStrMode::None && bubble > 0) {
        exit("exec_scph", "Sorry, RELAX_STR!=0 can't be used with bubble self-energy on top of the SCPH calculation.");
    }

    if (restart_scph) {

        if (mympi->my_rank == 0) {
            std::cout << " RESTART_SCPH is true.\n";
            std::cout << " Dynamical matrix is read from file ...";
        }

        // Read anharmonic correction to the dynamical matrix from the existing file
        // SCPH calculation, no structural optimization
        if (relax_mode == RelaxationStrMode::None) {
            // Resume SCPH by loading previously saved anharmonic dynamical-matrix corrections.
            load_scph_dymat_from_file(delta_dymat_scph,
                                      input->job_title + ".scph_dymat",
                                      kmesh_dense,
                                      kmesh_coarse,
                                      dynamical->nonanalytic,
                                      selfenergy_offdiagonal);
        }
        // SCPH + structural optimization
        else
        {
            // Resume SCPH correction part.
            load_scph_dymat_from_file(delta_dymat_scph,
                                      input->job_title + ".scph_dymat",
                                      kmesh_dense,
                                      kmesh_coarse,
                                      dynamical->nonanalytic,
                                      selfenergy_offdiagonal);

            // Resume harmonic-dynamical-matrix renormalization used in structural relaxation.
            load_scph_dymat_from_file(delta_harmonic_dymat_renormalize,
                                      input->job_title + ".renorm_harm_dymat",
                                      kmesh_dense,
                                      kmesh_coarse,
                                      dynamical->nonanalytic,
                                      selfenergy_offdiagonal);
        }

        // structural optimization
        if (relax_mode != RelaxationStrMode::None) {
            // Load previously optimized static potential offset V0.
            relaxation->load_V0_from_file();
        }

    } else {
        if (relax_mode == RelaxationStrMode::None) {
            // Run standard SCPH fixed-point iteration.
            exec_scph_main(delta_dymat_scph);
        }
        // SCPH + structural optimization
        else
        {
            // Run coupled SCPH + cell/coordinate relaxation loop.
            exec_scph_relax_cell_coordinate_main(delta_dymat_scph, delta_harmonic_dymat_renormalize);
        }

        if (mympi->my_rank == 0) {
            // write dymat to file
            // write scph dynamical matrix when scph calculation is performed
            // Persist converged SCPH dynamical-matrix corrections for restart/reuse.
            store_renormalized_dymat_to_file(delta_dymat_scph,
                                             input->job_title + ".scph_dymat",
                                             kmesh_dense,
                                             kmesh_coarse,
                                             dynamical->nonanalytic,
                                             selfenergy_offdiagonal);
            // write renormalized harmonic dynamical matrix when the crystal structure is optimized
            if (relax_mode != RelaxationStrMode::None) {
                // Persist renormalized harmonic dynamical matrix and relaxation offset.
                store_renormalized_dymat_to_file(delta_harmonic_dymat_renormalize,
                                                 input->job_title + ".renorm_harm_dymat",
                                                 kmesh_dense,
                                                 kmesh_coarse,
                                                 dynamical->nonanalytic,
                                                 selfenergy_offdiagonal);
                relaxation->store_V0_to_file();
            }
            // Convert dynamical-matrix correction back to real-space FC2 and write it out.
            write_anharmonic_correction_fc2(delta_dymat_scph, NT, kmesh_coarse, mindist_list, false, 0);
        }
    }

    if (kpoint->kpoint_mode == 2) {
        if (thermodynamics->calc_FE_bubble) {
            // Evaluate bubble correction to free energy on the interpolation mesh.
            compute_free_energy_bubble_SCPH(kmesh_interpolate, delta_dymat_scph);
        }
    }

    if (bubble) {
        allocate(delta_dymat_scph_plus_bubble, NT, ns, ns, kmesh_coarse->nk);
        // Add bubble self-energy to SCPH dynamical-matrix correction.
        bubble_correction(delta_dymat_scph, delta_dymat_scph_plus_bubble);
        if (mympi->my_rank == 0) {
            // Output FC2 after including bubble self-energy contribution.
            write_anharmonic_correction_fc2(delta_dymat_scph_plus_bubble,
                                            NT,
                                            kmesh_coarse,
                                            mindist_list,
                                            false,
                                            bubble);
        }
    }

    postprocess(delta_dymat_scph,
                delta_harmonic_dymat_renormalize,
                delta_dymat_scph_plus_bubble,
                kmesh_coarse,
                mindist_list,
                false,
                bubble);

    deallocate(delta_dymat_scph);
    deallocate(delta_harmonic_dymat_renormalize);
    if (delta_dymat_scph_plus_bubble) deallocate(delta_dymat_scph_plus_bubble);
}

void ScphQhaCommon::postprocess(std::complex<double> ****delta_dymat,
                                std::complex<double> ****delta_harmonic_dymat_renormalize,
                                std::complex<double> ****delta_dymat_scph_plus_bubble,
                                const KpointMeshUniform *kmesh_coarse_in, MinimumDistList ***mindist_list_in,
                                const bool is_qha, const int bubble_in)
{
    double ***eval_update = nullptr;
    double ***eval_harm_renorm = nullptr;
    const auto ns = dynamical->neval;
    const auto Tmin = system->Tmin;
    const auto Tmax = system->Tmax;
    const auto dT = system->dT;
    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;

    unsigned int nomega_dielec;

    if (mympi->my_rank == 0) {

        std::cout << '\n';
        std::cout << " Running postprocess of SCPH/QHA (calculation of free energy, MSD, DOS)\n";
        std::cout << " The number of temperature points: " << std::setw(4) << NT << '\n';
        std::cout << "   ";

        std::complex<double> ***evec_tmp = nullptr;
        std::complex<double> ***evec_harm_renorm = nullptr;
        double **eval_gam = nullptr;
        std::complex<double> ***evec_gam = nullptr;
        double **xk_gam = nullptr;

        double **dos_update = nullptr;
        double ***pdos_update = nullptr;
        double *heat_capacity = nullptr;
        double *heat_capacity_correction = nullptr;
        double *FE_QHA = nullptr;
        double *dFE_scph = nullptr;
        double *FE_total = nullptr;
        double *entropy = nullptr;
        double **msd_update = nullptr;
        double ***ucorr_update = nullptr;
        double ****dielec_update = nullptr;
        double *omega_grid = nullptr;
        double **domega_dt = nullptr;

        if (dos->kmesh_dos) {
            allocate(eval_update, NT, dos->kmesh_dos->nk, ns);
            allocate(evec_tmp, dos->kmesh_dos->nk, ns, ns);
            allocate(eval_harm_renorm, NT, dos->kmesh_dos->nk, ns);
            allocate(evec_harm_renorm, dos->kmesh_dos->nk, ns, ns);

            if (dos->compute_dos) {
                allocate(dos_update, NT, dos->n_energy);

                if (dos->projected_dos) {
                    allocate(pdos_update, NT, ns, dos->n_energy);
                }
            }
            allocate(heat_capacity, NT);
            allocate(FE_QHA, NT);
            allocate(dFE_scph, NT);
            allocate(FE_total, NT);
            allocate(entropy, NT);

            if (writes->getPrintMSD()) {
                allocate(msd_update, NT, ns);
            }
            if (writes->getPrintUcorr()) {
                allocate(ucorr_update, NT, ns, ns);
            }
            if (compute_Cv_anharmonic) {
                allocate(heat_capacity_correction, NT);
                allocate(domega_dt, dos->kmesh_dos->nk, ns);
                if (compute_Cv_anharmonic == 1) {
                    // Use central difference to evaluate temperature derivative of
                    // anharmonic frequencies
                    heat_capacity_correction[0] = 0.0;
                    heat_capacity_correction[NT - 1] = 0.0;
                }
            }

            dynamical->precompute_dymat_harm(dos->kmesh_dos->nk,
                                             dos->kmesh_dos->xk,
                                             dos->kmesh_dos->kvec_na,
                                             dymat_harm_short,
                                             dymat_harm_long);

            if (dos->compute_dos) {
                auto emin_now = std::numeric_limits<double>::max();
                auto emax_now = std::numeric_limits<double>::min();

                double eval_tmp;
                for (auto iT = 0; iT < NT; ++iT) {
                    if (iT == 0 || (iT == NT - 1)) {
                        // Interpolate SCPH frequencies on the DOS mesh (edge temperatures for energy-grid bounds).
                        dynamical->exec_interpolation(kmesh_coarse_in->nk_i,
                                                      delta_dymat[iT],
                                                      dos->kmesh_dos->nk,
                                                      dos->kmesh_dos->xk,
                                                      dos->kmesh_dos->kvec_na,
                                                      eval_update[iT],
                                                      evec_tmp,
                                                      dymat_harm_short,
                                                      dymat_harm_long,
                                                      mindist_list_in,
                                                      true);

                        for (unsigned int j = 0; j < dos->kmesh_dos->nk_irred; ++j) {
                            for (unsigned int k = 0; k < ns; ++k) {
                                eval_tmp =
                                    writes->in_kayser(eval_update[iT][dos->kmesh_dos->kpoint_irred_all[j][0].knum][k]);
                                emin_now = std::min(emin_now, eval_tmp);
                                emax_now = std::max(emax_now, eval_tmp);
                            }
                        }
                    }
                }
                emax_now += dos->delta_e;
                dos->update_dos_energy_grid(emin_now, emax_now);
            }

            for (auto iT = 0; iT < NT; ++iT) {
                auto temperature = Tmin + dT * static_cast<double>(iT);

                // Interpolate SCPH-renormalized frequencies/eigenvectors onto DOS mesh.
                dynamical->exec_interpolation(kmesh_coarse_in->nk_i,
                                              delta_dymat[iT],
                                              dos->kmesh_dos->nk,
                                              dos->kmesh_dos->xk,
                                              dos->kmesh_dos->kvec_na,
                                              eval_update[iT],
                                              evec_tmp,
                                              dymat_harm_short,
                                              dymat_harm_long,
                                              mindist_list_in,
                                              true);

                // when is_qha = true, eval_harm_renorm is same as eval_update.
                // Interpolate renormalized harmonic branch needed for SCPH free-energy correction.
                dynamical->exec_interpolation(kmesh_coarse_in->nk_i,
                                              delta_harmonic_dymat_renormalize[iT],
                                              dos->kmesh_dos->nk,
                                              dos->kmesh_dos->xk,
                                              dos->kmesh_dos->kvec_na,
                                              eval_harm_renorm[iT],
                                              evec_harm_renorm,
                                              dymat_harm_short,
                                              dymat_harm_long,
                                              mindist_list_in,
                                              true);

                if (dos->compute_dos) {
                    // Compute total DOS from interpolated frequencies via tetrahedron integration.
                    dos->calc_dos_from_given_frequency(dos->kmesh_dos,
                                                       eval_update[iT],
                                                       dos->tetra_nodes_dos->get_ntetra(),
                                                       dos->tetra_nodes_dos->get_tetras(),
                                                       dos_update[iT]);
                }

                heat_capacity[iT] = thermodynamics->Cv_tot(temperature,
                                                           dos->kmesh_dos->nk_irred,
                                                           ns,
                                                           dos->kmesh_dos->kpoint_irred_all,
                                                           dos->kmesh_dos->weight_k.data(),
                                                           eval_update[iT]);

                FE_QHA[iT] = thermodynamics->free_energy_QHA(temperature,
                                                             dos->kmesh_dos->nk_irred,
                                                             ns,
                                                             dos->kmesh_dos->kpoint_irred_all,
                                                             dos->kmesh_dos->weight_k.data(),
                                                             eval_update[iT]);

                // when is_qha is true, this value is zero.
                dFE_scph[iT] = thermodynamics->FE_scph_correction(iT,
                                                                  eval_update[iT],
                                                                  evec_tmp,
                                                                  eval_harm_renorm[iT],
                                                                  evec_harm_renorm);

                FE_total[iT] = thermodynamics->compute_FE_total(iT, FE_QHA[iT], dFE_scph[iT]);

                entropy[iT] = thermodynamics->vibrational_entropy(temperature,
                                                                  dos->kmesh_dos->nk_irred,
                                                                  ns,
                                                                  dos->kmesh_dos->kpoint_irred_all,
                                                                  dos->kmesh_dos->weight_k.data(),
                                                                  eval_update[iT]) /
                              k_Boltzmann;

                if (writes->getPrintMSD()) {
                    double shift[3]{0.0, 0.0, 0.0};

                    for (auto is = 0; is < ns; ++is) {
                        msd_update[iT][is] = thermodynamics->disp_corrfunc(temperature,
                                                                           is,
                                                                           is,
                                                                           shift,
                                                                           dos->kmesh_dos->nk,
                                                                           ns,
                                                                           dos->kmesh_dos->xk,
                                                                           eval_update[iT],
                                                                           evec_tmp);
                    }
                }

                if (writes->getPrintUcorr()) {
                    double shift[3];
                    for (auto i = 0; i < 3; ++i)
                        shift[i] = static_cast<double>(writes->getShiftUcorr()[i]);

                    for (auto is = 0; is < ns; ++is) {
                        for (auto js = 0; js < ns; ++js) {
                            ucorr_update[iT][is][js] = thermodynamics->disp_corrfunc(temperature,
                                                                                     is,
                                                                                     js,
                                                                                     shift,
                                                                                     dos->kmesh_dos->nk,
                                                                                     ns,
                                                                                     dos->kmesh_dos->xk,
                                                                                     eval_update[iT],
                                                                                     evec_tmp);
                        }
                    }
                }

                if (compute_Cv_anharmonic == 1) {

                    if (iT >= 1 and iT <= NT - 2) {
                        get_derivative_central_diff(dT,
                                                    dos->kmesh_dos->nk,
                                                    eval_update[iT - 1],
                                                    eval_update[iT + 1],
                                                    domega_dt);

                        heat_capacity_correction[iT] =
                            thermodynamics->Cv_anharm_correction(temperature,
                                                                 dos->kmesh_dos->nk_irred,
                                                                 ns,
                                                                 dos->kmesh_dos->kpoint_irred_all,
                                                                 dos->kmesh_dos->weight_k.data(),
                                                                 eval_update[iT],
                                                                 domega_dt);
                    }
                }

                std::cout << '.' << std::flush;
                if (iT % 25 == 24) {
                    std::cout << '\n';
                    std::cout << std::setw(3);
                }
            }
            std::cout << "\n\n";

            if (dos->compute_dos) {
                writes->writePhononDos(dos_update, is_qha, 0);
            }
            writes->writeThermodynamicFunc(heat_capacity,
                                           heat_capacity_correction,
                                           FE_QHA,
                                           dFE_scph,
                                           FE_total,
                                           entropy,
                                           is_qha);
            if (writes->getPrintMSD()) {
                writes->writeMSD(msd_update, is_qha, 0);
            }
            if (writes->getPrintUcorr()) {
                writes->writeDispCorrelation(ucorr_update, is_qha, 0);
            }

            // If delta_dymat_scph_plus_bubble != nullptr, run postprocess again with
            // delta_dymat_scph_plus_bubble.
            if (bubble_in > 0) {
                std::cout << '\n';
                std::cout << "   ";

                if (dos->compute_dos) {
                    auto emin_now = std::numeric_limits<double>::max();
                    auto emax_now = std::numeric_limits<double>::min();

                    double eval_tmp;
                    for (auto iT = 0; iT < NT; ++iT) {
                        if (iT == 0 || (iT == NT - 1)) {
                            dynamical->exec_interpolation(kmesh_coarse_in->nk_i,
                                                          delta_dymat_scph_plus_bubble[iT],
                                                          dos->kmesh_dos->nk,
                                                          dos->kmesh_dos->xk,
                                                          dos->kmesh_dos->kvec_na,
                                                          eval_update[iT],
                                                          evec_tmp,
                                                          dymat_harm_short,
                                                          dymat_harm_long,
                                                          mindist_list_in,
                                                          true);

                            for (unsigned int j = 0; j < dos->kmesh_dos->nk_irred; ++j) {
                                for (unsigned int k = 0; k < ns; ++k) {
                                    eval_tmp = writes->in_kayser(
                                        eval_update[iT][dos->kmesh_dos->kpoint_irred_all[j][0].knum][k]);
                                    emin_now = std::min(emin_now, eval_tmp);
                                    emax_now = std::max(emax_now, eval_tmp);
                                }
                            }
                        }
                    }
                    emax_now += dos->delta_e;
                    dos->update_dos_energy_grid(emin_now, emax_now);
                }

                for (auto iT = 0; iT < NT; ++iT) {
                    auto temperature = Tmin + dT * static_cast<double>(iT);

                    dynamical->exec_interpolation(kmesh_coarse_in->nk_i,
                                                  delta_dymat_scph_plus_bubble[iT],
                                                  dos->kmesh_dos->nk,
                                                  dos->kmesh_dos->xk,
                                                  dos->kmesh_dos->kvec_na,
                                                  eval_update[iT],
                                                  evec_tmp,
                                                  dymat_harm_short,
                                                  dymat_harm_long,
                                                  mindist_list_in,
                                                  true);

                    if (dos->compute_dos) {
                        dos->calc_dos_from_given_frequency(dos->kmesh_dos,
                                                           eval_update[iT],
                                                           dos->tetra_nodes_dos->get_ntetra(),
                                                           dos->tetra_nodes_dos->get_tetras(),
                                                           dos_update[iT]);
                    }

                    heat_capacity[iT] = thermodynamics->Cv_tot(temperature,
                                                               dos->kmesh_dos->nk_irred,
                                                               ns,
                                                               dos->kmesh_dos->kpoint_irred_all,
                                                               dos->kmesh_dos->weight_k.data(),
                                                               eval_update[iT]);

                    if (writes->getPrintMSD()) {
                        double shift[3]{0.0, 0.0, 0.0};

                        for (auto is = 0; is < ns; ++is) {
                            msd_update[iT][is] = thermodynamics->disp_corrfunc(temperature,
                                                                               is,
                                                                               is,
                                                                               shift,
                                                                               dos->kmesh_dos->nk,
                                                                               ns,
                                                                               dos->kmesh_dos->xk,
                                                                               eval_update[iT],
                                                                               evec_tmp);
                        }
                    }

                    if (writes->getPrintUcorr()) {
                        double shift[3];
                        for (auto i = 0; i < 3; ++i)
                            shift[i] = static_cast<double>(writes->getShiftUcorr()[i]);

                        for (auto is = 0; is < ns; ++is) {
                            for (auto js = 0; js < ns; ++js) {
                                ucorr_update[iT][is][js] = thermodynamics->disp_corrfunc(temperature,
                                                                                         is,
                                                                                         js,
                                                                                         shift,
                                                                                         dos->kmesh_dos->nk,
                                                                                         ns,
                                                                                         dos->kmesh_dos->xk,
                                                                                         eval_update[iT],
                                                                                         evec_tmp);
                            }
                        }
                    }

                    std::cout << '.' << std::flush;
                    if (iT % 25 == 24) {
                        std::cout << '\n';
                        std::cout << std::setw(3);
                    }
                }
                std::cout << "\n\n";

                if (dos->compute_dos) {
                    writes->writePhononDos(dos_update, false, bubble_in);
                }
                if (writes->getPrintMSD()) {
                    writes->writeMSD(msd_update, false, bubble_in);
                }
                if (writes->getPrintUcorr()) {
                    writes->writeDispCorrelation(ucorr_update, false, bubble_in);
                }
            }
            deallocate(eval_update);
            eval_update = nullptr;
            deallocate(evec_tmp);
            evec_tmp = nullptr;
        }

        if (kpoint->kpoint_general) {
            allocate(eval_update, NT, kpoint->kpoint_general->nk, ns);
            allocate(evec_tmp, kpoint->kpoint_general->nk, ns, ns);

            for (auto iT = 0; iT < NT; ++iT) {
                dynamical->exec_interpolation(kmesh_coarse_in->nk_i,
                                              delta_dymat[iT],
                                              kpoint->kpoint_general->nk,
                                              kpoint->kpoint_general->xk,
                                              kpoint->kpoint_general->kvec_na,
                                              eval_update[iT],
                                              evec_tmp,
                                              dymat_harm_short,
                                              dymat_harm_short,
                                              mindist_list_in);
            }

            writes->writePhononEnergies(kpoint->kpoint_general->nk, eval_update, is_qha, 0);

            if (bubble_in > 0) {
                for (auto iT = 0; iT < NT; ++iT) {
                    dynamical->exec_interpolation(kmesh_coarse_in->nk_i,
                                                  delta_dymat_scph_plus_bubble[iT],
                                                  kpoint->kpoint_general->nk,
                                                  kpoint->kpoint_general->xk,
                                                  kpoint->kpoint_general->kvec_na,
                                                  eval_update[iT],
                                                  evec_tmp,
                                                  dymat_harm_short,
                                                  dymat_harm_long,
                                                  mindist_list_in);
                }
                writes->writePhononEnergies(kpoint->kpoint_general->nk, eval_update, false, bubble_in);
            }
            deallocate(eval_update);
            deallocate(evec_tmp);
            eval_update = nullptr;
            evec_tmp = nullptr;
        }

        if (kpoint->kpoint_bs) {
            allocate(eval_update, NT, kpoint->kpoint_bs->nk, ns);
            allocate(evec_tmp, kpoint->kpoint_bs->nk, ns, ns);

            dynamical->precompute_dymat_harm(kpoint->kpoint_bs->nk,
                                             kpoint->kpoint_bs->xk,
                                             kpoint->kpoint_bs->kvec_na,
                                             dymat_harm_short,
                                             dymat_harm_long);

            for (auto iT = 0; iT < NT; ++iT) {
                dynamical->exec_interpolation(kmesh_coarse_in->nk_i,
                                              delta_dymat[iT],
                                              kpoint->kpoint_bs->nk,
                                              kpoint->kpoint_bs->xk,
                                              kpoint->kpoint_bs->kvec_na,
                                              eval_update[iT],
                                              evec_tmp,
                                              dymat_harm_short,
                                              dymat_harm_long,
                                              mindist_list_in,
                                              true);
            }

            writes->writePhononBands(kpoint->kpoint_bs->nk, kpoint->kpoint_bs->kaxis, eval_update, is_qha, 0);

            if (bubble_in > 0) {
                for (auto iT = 0; iT < NT; ++iT) {
                    dynamical->exec_interpolation(kmesh_coarse_in->nk_i,
                                                  delta_dymat_scph_plus_bubble[iT],
                                                  kpoint->kpoint_bs->nk,
                                                  kpoint->kpoint_bs->xk,
                                                  kpoint->kpoint_bs->kvec_na,
                                                  eval_update[iT],
                                                  evec_tmp,
                                                  dymat_harm_short,
                                                  dymat_harm_long,
                                                  mindist_list_in,
                                                  true);
                }
                writes->writePhononBands(kpoint->kpoint_bs->nk,
                                         kpoint->kpoint_bs->kaxis,
                                         eval_update,
                                         false,
                                         bubble_in);
            }
            deallocate(eval_update);
            deallocate(evec_tmp);
            eval_update = nullptr;
            evec_tmp = nullptr;
        }

        if (dielec->calc_dielectric_constant) {
            omega_grid = dielec->get_omega_grid(nomega_dielec);
            allocate(dielec_update, NT, nomega_dielec, 3, 3);
            allocate(eval_gam, 1, ns);
            allocate(evec_gam, 1, ns, ns);
            allocate(xk_gam, 1, 3);
            for (auto i = 0; i < 3; ++i)
                xk_gam[0][i] = 0.0;

            for (auto iT = 0; iT < NT; ++iT) {
                dynamical->exec_interpolation(kmesh_coarse_in->nk_i,
                                              delta_dymat[iT],
                                              1,
                                              xk_gam,
                                              xk_gam,
                                              eval_gam,
                                              evec_gam,
                                              dymat_harm_short,
                                              dymat_harm_long,
                                              mindist_list_in);

                for (auto is = 0; is < ns; ++is) {
                    if (eval_gam[0][is] < 0.0) {
                        eval_gam[0][is] = -pow2(eval_gam[0][is]);
                    } else {
                        eval_gam[0][is] = pow2(eval_gam[0][is]);
                    }
                }
                dielec->compute_dielectric_function(nomega_dielec,
                                                    omega_grid,
                                                    eval_gam[0],
                                                    evec_gam[0],
                                                    dielec_update[iT]);
            }
            writes->writeDielecFunc(dielec_update, is_qha);
        }

        if (eval_update) deallocate(eval_update);
        if (evec_tmp) deallocate(evec_tmp);

        if (dos_update) deallocate(dos_update);
        if (pdos_update) deallocate(pdos_update);
        if (heat_capacity) deallocate(heat_capacity);
        if (heat_capacity_correction) deallocate(heat_capacity_correction);
        if (FE_QHA) deallocate(FE_QHA);
        if (dFE_scph) deallocate(dFE_scph);
        if (FE_total) deallocate(FE_total);
        if (entropy) deallocate(entropy);
        if (dielec_update) deallocate(dielec_update);

        if (eval_gam) deallocate(eval_gam);
        if (evec_gam) deallocate(evec_gam);
        if (xk_gam) deallocate(xk_gam);
    }
}


void Scph::exec_scph_main(std::complex<double> ****dymat_anharm)
{
    int ik, is;
    const auto nk = kmesh_dense->nk;
    const auto nk_interpolate = kmesh_coarse->nk;
    const auto ns = dynamical->neval;
    const auto nk_irred_interpolate = kmesh_coarse->nk_irred;
    const auto Tmin = system->Tmin;
    const auto Tmax = system->Tmax;
    const auto dT = system->dT;
    double ***omega2_anharm;
    std::complex<double> ***evec_anharm_tmp;
    std::complex<double> ***v3_array_all;
    std::complex<double> ***v4_array_all;

    std::complex<double> **delta_v2_renorm;

    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;

    // Compute matrix element of 4-phonon interaction

    allocate(omega2_anharm, NT, nk, ns);
    allocate(evec_anharm_tmp, nk, ns, ns);
    allocate(v4_array_all, nk_irred_interpolate * nk, ns * ns, ns * ns);

    // delta_v2_renorm is zero when structural optimization is not performed
    allocate(delta_v2_renorm, nk_interpolate, ns * ns);
    for (ik = 0; ik < nk_interpolate; ik++) {
        for (is = 0; is < ns * ns; is++) {
            delta_v2_renorm[ik][is] = 0.0;
        }
    }

    const auto relax_mode = to_relaxation_str_mode(relaxation->relax_str);

    // Calculate v4 array.
    // This operation is the most expensive part of the calculation.
    if (selfenergy_offdiagonal & (ialgo == 1)) {
        compute_V4_elements_mpi_over_band(v4_array_all,
                                          omega2_harmonic,
                                          evec_harmonic,
                                          selfenergy_offdiagonal,
                                          kmesh_coarse,
                                          kmesh_dense,
                                          kmap_coarse_to_dense,
                                          phase_factor,
                                          phi4_reciprocal);
    } else {
        compute_V4_elements_mpi_over_kpoint(v4_array_all,
                                            omega2_harmonic,
                                            evec_harmonic,
                                            selfenergy_offdiagonal,
                                            relax_mode != RelaxationStrMode::None,
                                            kmesh_coarse,
                                            kmesh_dense,
                                            kmap_coarse_to_dense,
                                            phase_factor,
                                            phi4_reciprocal);
    }

    if (relax_mode != RelaxationStrMode::None) {
        allocate(v3_array_all, nk, ns, ns * ns);

        compute_V3_elements_mpi_over_kpoint(v3_array_all,
                                            omega2_harmonic,
                                            evec_harmonic,
                                            selfenergy_offdiagonal,
                                            kmesh_coarse,
                                            kmesh_dense,
                                            phase_factor,
                                            phi3_reciprocal);
    }

    if (mympi->my_rank == 0) {
        std::vector<double> vec_temp;
        std::complex<double> ***cmat_convert;
        allocate(cmat_convert, nk, ns, ns);

        vec_temp.clear();

        dynamical->precompute_dymat_harm(kmesh_dense->nk,
                                         kmesh_dense->xk,
                                         kmesh_dense->kvec_na,
                                         dymat_harm_short,
                                         dymat_harm_long);

        if (lower_temp) {
            for (int i = NT - 1; i >= 0; --i) {
                vec_temp.push_back(Tmin + static_cast<double>(i) * dT);
            }
        } else {
            for (int i = 0; i < NT; ++i) {
                vec_temp.push_back(Tmin + static_cast<double>(i) * dT);
            }
        }

        auto converged_prev = false;

        for (const double temp: vec_temp) {
            const auto iT = static_cast<unsigned int>((temp - Tmin) / dT);

            // Initialize phonon eigenvectors with harmonic values

            for (ik = 0; ik < nk; ++ik) {
                for (is = 0; is < ns; ++is) {
                    for (int js = 0; js < ns; ++js) {
                        evec_anharm_tmp[ik][is][js] = evec_harmonic[ik][is][js];
                    }
                }
            }

            if (converged_prev) {
                if (lower_temp) {
                    for (ik = 0; ik < nk; ++ik) {
                        for (is = 0; is < ns; ++is) {
                            omega2_anharm[iT][ik][is] = omega2_anharm[iT + 1][ik][is];
                        }
                    }
                } else {
                    for (ik = 0; ik < nk; ++ik) {
                        for (is = 0; is < ns; ++is) {
                            omega2_anharm[iT][ik][is] = omega2_anharm[iT - 1][ik][is];
                        }
                    }
                }
            }

            compute_anharmonic_frequency(v4_array_all,
                                         omega2_anharm[iT],
                                         evec_anharm_tmp,
                                         temp,
                                         converged_prev,
                                         cmat_convert,
                                         selfenergy_offdiagonal,
                                         delta_v2_renorm,
                                         writes->getVerbosity());


            dynamical->calc_new_dymat_with_evec(dymat_anharm[iT],
                                                omega2_anharm[iT],
                                                evec_anharm_tmp,
                                                kmesh_coarse,
                                                kmap_coarse_to_dense);

            if (!warmstart_scph) converged_prev = false;
        }

        deallocate(cmat_convert);
    }

    mpi_bcast_complex(dymat_anharm, NT, kmesh_coarse->nk, ns);

    deallocate(omega2_anharm);
    deallocate(v4_array_all);
    deallocate(evec_anharm_tmp);
    deallocate(delta_v2_renorm);
    if (relax_mode != RelaxationStrMode::None) {
        deallocate(v3_array_all);
    }
}


// relax internal coordinate and lattice
void Scph::exec_scph_relax_cell_coordinate_main(std::complex<double> ****dymat_anharm,
                                                std::complex<double> ****delta_harmonic_dymat_renormalize)
{
    using namespace Eigen;

    int is, js;
    int i1;
    int iat1, ixyz1, ixyz2;
    std::string str_tmp;

    const auto nk = kmesh_dense->nk;
    const auto nk_interpolate = kmesh_coarse->nk;
    const auto ns = dynamical->neval;
    const auto nk_irred_interpolate = kmesh_coarse->nk_irred;
    const auto Tmin = system->Tmin;
    const auto Tmax = system->Tmax;
    const auto dT = system->dT;
    double ***omega2_anharm;
    std::complex<double> ***evec_anharm_tmp;
    // renormalization of harmonic dynamical matrix
    std::complex<double> **delta_v2_renorm;
    std::complex<double> **delta_v2_with_umn;
    double ***omega2_harm_renorm;
    std::complex<double> ***evec_harm_renorm_tmp;
    // k-space IFCs at the reference and updated structures
    std::complex<double> *v1_ref, *v1_renorm, *v1_with_umn;
    std::complex<double> ***v3_ref, ***v3_renorm, ***v3_with_umn;
    std::complex<double> ***v4_ref; //, ***v4_renorm, ***v4_with_umn;
    double v0_ref, v0_renorm, v0_with_umn;
    v0_ref = 0.0; // set original ground state energy as zero

    // elastic constants
    double *C1_array;
    double **C2_array;
    double ***C3_array;

    // strain-derivative of k-space IFCs
    DelVStrainData del_v_strain;

    std::complex<double> *del_v0_del_umn_renorm;

    // atomic forces and stress tensor at finite temperatures
    std::complex<double> *v1_SCP;
    std::complex<double> *del_v0_del_umn_SCP;

    // structure optimization
    int i_str_loop, i_temp_loop;

    // structure update
    double du0;
    double du_tensor;
    std::vector<int> harm_optical_modes(ns - 3);

    // cell optimization
    double pvcell = 0.0; // pressure * v_{cell,reference} [Ry]
    pvcell = relaxation->stat_pressure * system->get_primcell().volume * std::pow(Bohr_in_Angstrom, 3) *
             1.0e-30;      // in 10^9 J = GJ
    pvcell *= 1.0e9 / Ryd; // in Ry

    const auto relax_mode = to_relaxation_str_mode(relaxation->relax_str);

    // temperature grid
    std::vector<double> vec_temp;
    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;


    allocate(omega2_anharm, NT, nk, ns);
    allocate(evec_anharm_tmp, nk, ns, ns);

    allocate(delta_v2_renorm, nk_interpolate, ns * ns);
    allocate(delta_v2_with_umn, nk_interpolate, ns * ns);
    allocate(omega2_harm_renorm, NT, nk, ns);
    allocate(evec_harm_renorm_tmp, nk, ns, ns);

    allocate(v1_ref, ns);
    allocate(v1_with_umn, ns);
    allocate(v1_renorm, ns);

    RelaxationStructureState structure_state;
    structure_state.resize(ns);
    auto &q0 = structure_state.q0;
    auto &u0 = structure_state.u0;
    auto &u_tensor = structure_state.u_tensor;
    auto &eta_tensor = structure_state.eta_tensor;

    allocate(v1_SCP, ns);
    allocate(del_v0_del_umn_renorm, 9);
    allocate(del_v0_del_umn_SCP, 9);

    // assume that the atomic forces are zero at initial structure
    for (is = 0; is < ns; is++) {
        v1_ref[is] = 0.0;
    }
    // compute IFC renormalization by lattice relaxation
    std::cout << " RELAX_STR = " << to_int(relax_mode) << ": ";
    if (relax_mode == RelaxationStrMode::CoordinatesOnly) {
        std::cout << "Set zeros in derivatives of k-space IFCs by strain.\n\n";
    }
    if (relax_mode == RelaxationStrMode::CoordinatesAndCell) {
        std::cout << "Calculating derivatives of k-space IFCs by strain.\n\n";
    }

    del_v_strain.resize(nk, ns);

    // This function precomputes the 1st, 2nd, and 3rd order derivatives of v1
    // 1st and 2nd order derivatives of v2, and 1st order derivative of v3.
    relaxation->compute_del_v_strain(kmesh_coarse,
                                     kmesh_dense,
                                     del_v_strain,
                                     omega2_harmonic,
                                     evec_harmonic,
                                     relax_mode,
                                     mindist_list,
                                     phase_factor);

    allocate(v4_ref, nk_irred_interpolate * kmesh_dense->nk, ns * ns, ns * ns);

    // initialize optimizer
    relaxation->create_optimizer(ns);

    // Compute matrix element of 4-phonon interaction
    // This operation is the most expensive part of the calculation.
    if (selfenergy_offdiagonal & (ialgo == 1)) {

        compute_V4_elements_mpi_over_band(v4_ref,
                                          omega2_harmonic,
                                          evec_harmonic,
                                          selfenergy_offdiagonal,
                                          kmesh_coarse,
                                          kmesh_dense,
                                          kmap_coarse_to_dense,
                                          phase_factor,
                                          phi4_reciprocal);
    } else {
        compute_V4_elements_mpi_over_kpoint(v4_ref,
                                            omega2_harmonic,
                                            evec_harmonic,
                                            selfenergy_offdiagonal,
                                            relax_mode != RelaxationStrMode::None,
                                            kmesh_coarse,
                                            kmesh_dense,
                                            kmap_coarse_to_dense,
                                            phase_factor,
                                            phi4_reciprocal);
    }

    allocate(v3_ref, nk, ns, ns * ns);
    allocate(v3_renorm, nk, ns, ns * ns);
    allocate(v3_with_umn, nk, ns, ns * ns);

    compute_V3_elements_mpi_over_kpoint(v3_ref,
                                        omega2_harmonic,
                                        evec_harmonic,
                                        selfenergy_offdiagonal,
                                        kmesh_coarse,
                                        kmesh_dense,
                                        phase_factor,
                                        phi3_reciprocal);


    // get indices of optical modes at Gamma point
    js = 0;
    for (is = 0; is < ns; is++) {
        if (std::fabs(omega2_harmonic[0][is]) < eps10) {
            continue;
        }
        harm_optical_modes[js] = is;
        js++;
    }
    if (js != ns - 3) {
        exit("exec_scph_relax_cell_coordinate_main", "The number of detected optical modes is not ns-3.");
    }

    if (mympi->my_rank == 0) {
        int ik;

        dynamical->precompute_dymat_harm(kmesh_dense->nk,
                                         kmesh_dense->xk,
                                         kmesh_dense->kvec_na,
                                         dymat_harm_short,
                                         dymat_harm_long);


        std::complex<double> ***cmat_convert;
        allocate(cmat_convert, nk, ns, ns);

        vec_temp.clear();

        if (lower_temp) {
            for (int i = NT - 1; i >= 0; --i) {
                vec_temp.push_back(Tmin + static_cast<double>(i) * dT);
            }
        } else {
            for (int i = 0; i < NT; ++i) {
                vec_temp.push_back(Tmin + static_cast<double>(i) * dT);
            }
        }

        auto converged_prev = false;
        auto str_diverged = 0;

        allocate(C1_array, 9);
        allocate(C2_array, 9, 9);
        allocate(C3_array, 9, 9, 9);

        relaxation->set_elastic_constants(C1_array, C2_array, C3_array);

        // output files of structural optimization
        std::ofstream fout_step_q0, fout_step_u0;
        std::ofstream fout_q0, fout_u0;

        // cell optimization
        std::ofstream fout_step_u_tensor, fout_u_tensor;

        fout_step_q0.open("step_q0.txt");
        fout_step_u0.open("step_u0.txt");
        fout_q0.open(input->job_title + ".normal_disp");
        fout_u0.open(input->job_title + ".atom_disp");

        // if the unit cell is relaxed
        if (relax_mode == RelaxationStrMode::CoordinatesAndCell) {
            fout_step_u_tensor.open("step_u_tensor.txt");
            fout_u_tensor.open(input->job_title + ".umn_tensor");
        }

        relaxation->write_resfile_header(fout_q0, fout_u0, fout_u_tensor);

        i_temp_loop = -1;

        std::cout << " Start structural optimization.\n";

        if (relax_mode == RelaxationStrMode::CoordinatesOnly) {
            std::cout << "  Internal coordinates are relaxed.\n";
            std::cout << "  Shape of the unit cell is fixed.\n\n";
        } else if (relax_mode == RelaxationStrMode::CoordinatesAndCell) {
            std::cout << "  Internal coordinates and shape of the unit cell are relaxed.\n\n";
        }

        for (double temp: vec_temp) {
            i_temp_loop++;
            auto iT = static_cast<unsigned int>((temp - Tmin) / dT);

            std::cout << " ----------------------------------------------------------------\n";
            std::cout << " Temperature = " << temp << " K\n";
            std::cout << " Temperature index : " << std::setw(4) << i_temp_loop << "/" << std::setw(4) << NT << "\n\n";

            // Initialize phonon eigenvectors with harmonic values

            for (ik = 0; ik < nk; ++ik) {
                for (is = 0; is < ns; ++is) {
                    for (js = 0; js < ns; ++js) {
                        evec_anharm_tmp[ik][is][js] = evec_harmonic[ik][is][js];
                    }
                }
            }
            if (converged_prev) {
                if (lower_temp) {
                    for (ik = 0; ik < nk; ++ik) {
                        for (is = 0; is < ns; ++is) {
                            omega2_anharm[iT][ik][is] = omega2_anharm[iT + 1][ik][is];
                        }
                    }
                } else {
                    for (ik = 0; ik < nk; ++ik) {
                        for (is = 0; is < ns; ++is) {
                            omega2_anharm[iT][ik][is] = omega2_anharm[iT - 1][ik][is];
                        }
                    }
                }
            }

            relaxation->set_init_structure_atT(structure_state,
                                               converged_prev,
                                               str_diverged,
                                               i_temp_loop,
                                               omega2_harmonic,
                                               evec_harmonic);


            std::cout << " Initial atomic displacements [Bohr] : \n";
            for (iat1 = 0; iat1 < system->get_primcell().number_of_atoms; iat1++) {
                std::cout << " ";
                for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                    relaxation->get_xyz_string(ixyz1, str_tmp);
                    std::cout << std::setw(10) << ("u_{" + std::to_string(iat1) + "," + str_tmp + "}");
                }
                std::cout << " :";
                for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                    std::cout << std::scientific << std::setw(15) << std::setprecision(6) << u0[iat1 * 3 + ixyz1];
                }
                std::cout << '\n';
            }
            std::cout << '\n';

            if (relax_mode == RelaxationStrMode::CoordinatesAndCell) {
                std::cout << " Initial strain (displacement gradient tensor u_{mu nu}) : \n";
                for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                    std::cout << " ";
                    for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                        std::cout << std::scientific << std::setw(15) << std::setprecision(6) << u_tensor[ixyz1][ixyz2];
                    }
                    std::cout << '\n';
                }
                std::cout << '\n';
            }

            relaxation->write_stepresfile_header_atT(fout_step_q0, fout_step_u0, fout_step_u_tensor, temp);

            relaxation->write_stepresfile(structure_state, 0, fout_step_q0, fout_step_u0, fout_step_u_tensor);

            std::cout << " ----------------------------------------------------------------\n";

            std::cout << " Start structural optimization at " << temp << " K.";

            for (i_str_loop = 0; i_str_loop < relaxation->max_str_iter; i_str_loop++) {

                std::cout << "\n\n Structure loop :" << std::setw(5) << i_str_loop + 1;

                // get eta tensor
                relaxation->calculate_eta_tensor(eta_tensor, u_tensor);

                // calculate IFCs under strain
                relaxation->renormalize_v0_from_umn(v0_with_umn,
                                                    v0_ref,
                                                    eta_tensor,
                                                    C1_array,
                                                    C2_array,
                                                    C3_array,
                                                    u_tensor,
                                                    pvcell);

                // std::cout << "u_tensor\n";
                // for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                //     for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                //         std::cout << std::scientific << std::setw(15) << std::setprecision(6) << u_tensor[ixyz1][ixyz2]
                //                   << " ";
                //     }
                //     std::cout << '\n';
                // }

                relaxation->renormalize_v1_from_umn(v1_with_umn, v1_ref, del_v_strain, u_tensor);


                relaxation->renormalize_v2_from_umn(kmesh_coarse,
                                                    kmap_coarse_to_dense,
                                                    delta_v2_with_umn,
                                                    del_v_strain,
                                                    u_tensor);
                relaxation
                    ->renormalize_v3_from_umn(kmesh_coarse, kmesh_dense, v3_with_umn, v3_ref, del_v_strain, u_tensor);

                //                for (ik = 0; ik < nk_irred_interpolate * nk; ik++) {
                //                    for (is = 0; is < ns * ns; is++) {
                //                        for (is1 = 0; is1 < ns * ns; is1++) {
                //                            v4_with_umn[ik][is][is1] = v4_ref[ik][is][is1];
                //                        }
                //                    }
                //                }

                //renormalize IFC
                // TODO: check whether bug exists here
                relaxation->renormalize_v1_from_q0(omega2_harmonic,
                                                   kmesh_coarse,
                                                   kmesh_dense,
                                                   v1_renorm,
                                                   v1_with_umn,
                                                   delta_v2_with_umn,
                                                   v3_with_umn,
                                                   v4_ref,
                                                   q0);
                relaxation->renormalize_v2_from_q0(evec_harmonic,
                                                   kmesh_coarse,
                                                   kmesh_dense,
                                                   kmap_coarse_to_dense,
                                                   mat_transform_sym,
                                                   delta_v2_renorm,
                                                   delta_v2_with_umn,
                                                   v3_with_umn,
                                                   v4_ref,
                                                   q0);
                relaxation->renormalize_v3_from_q0(kmesh_dense, kmesh_coarse, v3_renorm, v3_with_umn, v4_ref, q0);
                relaxation->renormalize_v0_from_q0(omega2_harmonic,
                                                   kmesh_dense,
                                                   v0_renorm,
                                                   v0_with_umn,
                                                   v1_with_umn,
                                                   delta_v2_with_umn,
                                                   v3_with_umn,
                                                   v4_ref,
                                                   q0);

                // calculate PES gradient by strain
                if (relax_mode == RelaxationStrMode::CoordinatesOnly) {
                    for (i1 = 0; i1 < 9; i1++) {
                        del_v0_del_umn_renorm[i1] = 0.0;
                    }
                } else if (relax_mode == RelaxationStrMode::CoordinatesAndCell) {
                    calculate_del_v0_del_umn_renorm(del_v0_del_umn_renorm,
                                                    C1_array,
                                                    C2_array,
                                                    C3_array,
                                                    eta_tensor,
                                                    u_tensor,
                                                    del_v_strain,
                                                    q0,
                                                    pvcell,
                                                    kmesh_dense);

                    // for (i1 = 0; i1 < 9; i1++) {
                    //     std::cout << " del_v0_del_umn_renorm[" << i1 << "] = " << std::scientific << std::setw(15)
                    //               << std::setprecision(6) << del_v0_del_umn_renorm[i1] << '\n';
                    // }
                }


                // copy v4_ref to v4_renorm
                //                for (ik = 0; ik < nk_irred_interpolate * kmesh_dense->nk; ik++) {
                //                    for (is1 = 0; is1 < ns * ns; is1++) {
                //                        for (is2 = 0; is2 < ns * ns; is2++) {
                //                            v4_renorm[ik][is1][is2] = v4_ref[ik][is1][is2];
                //                        }
                //                    }
                //                }

                // solve SCP equation
                compute_anharmonic_frequency(v4_ref,
                                             omega2_anharm[iT],
                                             evec_anharm_tmp,
                                             temp,
                                             converged_prev,
                                             cmat_convert,
                                             selfenergy_offdiagonal,
                                             delta_v2_renorm,
                                             writes->getVerbosity());

                dynamical->calc_new_dymat_with_evec(dymat_anharm[iT],
                                                    omega2_anharm[iT],
                                                    evec_anharm_tmp,
                                                    kmesh_coarse,
                                                    kmap_coarse_to_dense);

                // calculate SCP force
                compute_anharmonic_v1_array(v1_SCP,
                                            v1_renorm,
                                            v3_renorm,
                                            cmat_convert,
                                            omega2_anharm[iT],
                                            temp,
                                            kmesh_dense);

                // std::cout << std::setw(15) << "v1_with_umn";
                // std::cout << std::setw(15) << "v1_renorm";
                // std::cout << std::setw(15) << "v1_SCP\n";
                // for (auto ii = 0; ii < ns; ++ii) {
                //     std::cout << std::setw(15) << std::setprecision(6) << std::scientific << v1_with_umn[ii];
                //     std::cout << std::setw(15) << std::setprecision(6) << std::scientific << v1_renorm[ii];
                //     std::cout << std::setw(15) << std::setprecision(6) << std::scientific << v1_SCP[ii] << '\n';
                // }

                // calculate SCP stress tensor
                if (relax_mode == RelaxationStrMode::CoordinatesOnly) {
                    for (i1 = 0; i1 < 9; i1++) {
                        del_v0_del_umn_SCP[i1] = 0.0;
                    }
                } else if (relax_mode == RelaxationStrMode::CoordinatesAndCell) {
                    compute_anharmonic_del_v0_del_umn(del_v0_del_umn_SCP,
                                                      del_v0_del_umn_renorm,
                                                      del_v_strain,
                                                      u_tensor,
                                                      q0,
                                                      cmat_convert,
                                                      omega2_anharm[iT],
                                                      temp,
                                                      kmesh_dense);
                }

                // for (i1 = 0; i1 < 9; i1++) {
                //     std::cout << " del_v0_del_umn_SCP[" << i1 << "] = " << std::scientific << std::setw(15)
                //               << std::setprecision(6) << del_v0_del_umn_SCP[i1] << '\n';
                // }

                relaxation->update_cell_coordinate(structure_state,
                                                   v1_SCP,
                                                   omega2_anharm[iT],
                                                   del_v0_del_umn_SCP,
                                                   C2_array,
                                                   cmat_convert,
                                                   harm_optical_modes,
                                                   omega2_harmonic,
                                                   evec_harmonic);
                du0 = structure_state.du0;
                du_tensor = structure_state.du_tensor;

                // for (i1 = 0; i1 < ns; i1++) {
                //     std::cout << " q0[" << i1 << "] = " << std::scientific << std::setw(15) << std::setprecision(6)
                //               << q0[i1] << '\n';
                // }

                relaxation->write_stepresfile(structure_state,
                                              i_str_loop + 1,
                                              fout_step_q0,
                                              fout_step_u0,
                                              fout_step_u_tensor);

                relaxation->check_str_divergence(str_diverged, structure_state);

                if (str_diverged) {
                    converged_prev = false;
                    std::cout << " The crystal structure diverged.";
                    std::cout << " Break from the structure loop.\n";
                    break;
                }

                // Residual gradient norms over the optimized degrees of freedom (the same
                // gradients the optimizer acts on). A small step (du0/du_tensor) does not by
                // itself imply a small gradient for the GDIIS optimizer (relax_algo == 3), so
                // these are printed for diagnostics and, when the corresponding tolerance is
                // > 0, also required for convergence to guard against false convergence at a
                // non-stationary point. The coordinate force (gradient w.r.t. q0) and the
                // cell gradient (stress conjugate to the strain tensor) have different units,
                // so they are checked separately (cf. COORD_CONV_TOL vs CELL_CONV_TOL).
                double grad_norm = 0.0;
                for (is = 0; is < ns - 3; is++) {
                    const double f = v1_SCP[harm_optical_modes[is]].real();
                    grad_norm += f * f;
                }
                grad_norm = std::sqrt(grad_norm);

                double cell_grad_norm = 0.0;
                if (relax_mode == RelaxationStrMode::CoordinatesAndCell) {
                    for (i1 = 0; i1 < 3; i1++) {
                        const double fd = del_v0_del_umn_SCP[i1 * 3 + i1].real();
                        const int j1 = (i1 + 1) % 3;
                        const int j2 = (i1 + 2) % 3;
                        const double fs = del_v0_del_umn_SCP[j1 * 3 + j2].real();
                        cell_grad_norm += fd * fd + fs * fs;
                    }
                    cell_grad_norm = std::sqrt(cell_grad_norm);
                }

                // check convergence
                std::cout << " du0 =" << std::scientific << std::setw(15) << std::setprecision(6) << du0 << " [Bohr]";

                std::cout << " du_tensor =" << std::scientific << std::setw(15) << std::setprecision(6) << du_tensor
                          << '\n';

                std::cout << " |residual force| =" << std::scientific << std::setw(15) << std::setprecision(6)
                          << grad_norm;
                if (relax_mode == RelaxationStrMode::CoordinatesAndCell) {
                    std::cout << " |residual stress| =" << std::scientific << std::setw(15) << std::setprecision(6)
                              << cell_grad_norm;
                }
                std::cout << '\n';

                const bool step_converged =
                    (du0 < relaxation->coord_conv_tol && du_tensor < relaxation->cell_conv_tol);
                const bool force_converged =
                    (relaxation->gradient_conv_tol <= 0.0) || (grad_norm < relaxation->gradient_conv_tol);
                const bool cell_force_converged = (relax_mode != RelaxationStrMode::CoordinatesAndCell) ||
                                                  (relaxation->cell_gradient_conv_tol <= 0.0) ||
                                                  (cell_grad_norm < relaxation->cell_gradient_conv_tol);

                if (step_converged && force_converged && cell_force_converged) {
                    std::cout << "\n\n du0 is smaller than COORD_CONV_TOL = " << std::scientific << std::setw(15)
                              << std::setprecision(6) << relaxation->coord_conv_tol << '\n';
                    if (relax_mode == RelaxationStrMode::CoordinatesAndCell) {
                        std::cout << " du_tensor is smaller than CELL_CONV_TOL = " << std::scientific << std::setw(15)
                                  << std::setprecision(6) << relaxation->cell_conv_tol << '\n';
                    }
                    if (relaxation->gradient_conv_tol > 0.0) {
                        std::cout << " |residual force| is smaller than GRADIENT_CONV_TOL = " << std::scientific
                                  << std::setw(15) << std::setprecision(6) << relaxation->gradient_conv_tol << '\n';
                    }
                    if (relax_mode == RelaxationStrMode::CoordinatesAndCell &&
                        relaxation->cell_gradient_conv_tol > 0.0) {
                        std::cout << " |residual stress| is smaller than CELL_GRADIENT_CONV_TOL = " << std::scientific
                                  << std::setw(15) << std::setprecision(6) << relaxation->cell_gradient_conv_tol << '\n';
                    }
                    std::cout << " Structural optimization converged in " << i_str_loop + 1 << "-th loop.\n";
                    std::cout << " break structural loop.\n\n";
                    break;
                }

            } // close structure loop

            std::cout << " ----------------------------------------------------------------\n";
            std::cout << " Final atomic displacements [Bohr] at " << temp << " K\n";
            for (iat1 = 0; iat1 < system->get_primcell().number_of_atoms; iat1++) {
                std::cout << " ";
                for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                    relaxation->get_xyz_string(ixyz1, str_tmp);
                    std::cout << std::setw(10) << ("u_{" + std::to_string(iat1) + "," + str_tmp + "}");
                }
                std::cout << " :";
                for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                    std::cout << std::scientific << std::setw(15) << std::setprecision(6) << u0[iat1 * 3 + ixyz1];
                }
                std::cout << '\n';
            }
            std::cout << '\n';

            if (relax_mode == RelaxationStrMode::CoordinatesAndCell) {
                std::cout << " Final strain (displacement gradient tensor u_{mu nu}) : \n";
                for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                    std::cout << " ";
                    for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                        std::cout << std::scientific << std::setw(15) << std::setprecision(6) << u_tensor[ixyz1][ixyz2];
                    }
                    std::cout << '\n';
                }
            }
            if (i_temp_loop == NT - 1) {
                std::cout << " ----------------------------------------------------------------\n\n";
            } else {
                std::cout << '\n';
            }

            // record zero-th order term of PES
            relaxation->V0[iT] = v0_renorm;

            // print obtained structure
            relaxation->calculate_u0(q0, u0, omega2_harmonic, evec_harmonic);

            relaxation->write_resfile_atT(structure_state, temp, fout_q0, fout_u0, fout_u_tensor);

            if (!warmstart_scph) converged_prev = false;

            // get renormalization of harmonic dymat
            dynamical->compute_renormalized_harmonic_frequency(omega2_harm_renorm[iT],
                                                               evec_harm_renorm_tmp,
                                                               delta_v2_renorm,
                                                               omega2_harmonic,
                                                               evec_harmonic,
                                                               kmesh_coarse,
                                                               kmesh_dense,
                                                               kmap_coarse_to_dense,
                                                               mat_transform_sym,
                                                               mindist_list,
                                                               writes->getVerbosity());

            dynamical->calc_new_dymat_with_evec(delta_harmonic_dymat_renormalize[iT],
                                                omega2_harm_renorm[iT],
                                                evec_harm_renorm_tmp,
                                                kmesh_coarse,
                                                kmap_coarse_to_dense);

        } // close temperature loop

        // output files of structural optimization
        fout_step_q0.close();
        fout_step_u0.close();
        fout_q0.close();
        fout_u0.close();

        if (relax_mode == RelaxationStrMode::CoordinatesAndCell) {
            fout_step_u_tensor.close();
            fout_u_tensor.close();
        }

        deallocate(cmat_convert);

        deallocate(C1_array);
        deallocate(C2_array);
        deallocate(C3_array);
    }

    mpi_bcast_complex(dymat_anharm, NT, kmesh_coarse->nk, ns);
    mpi_bcast_complex(delta_harmonic_dymat_renormalize, NT, kmesh_coarse->nk, ns);

    deallocate(omega2_anharm);
    deallocate(evec_anharm_tmp);
    deallocate(delta_v2_renorm);
    deallocate(delta_v2_with_umn);

    deallocate(omega2_harm_renorm);
    deallocate(evec_harm_renorm_tmp);

    deallocate(v1_ref);
    deallocate(v1_with_umn);
    deallocate(v1_renorm);
    deallocate(v3_ref);
    deallocate(v3_renorm);
    deallocate(v3_with_umn);
    deallocate(v4_ref);
    //    deallocate(v4_renorm);
    //    deallocate(v4_with_umn);

    deallocate(del_v0_del_umn_renorm);
    deallocate(v1_SCP);
    deallocate(del_v0_del_umn_SCP);
}


//void Scph::calculate_force_in_real_space(const std::complex<double> *const v1_renorm,
//                                         double *force_array)
//{
//    int natmin = system->natmin;
//    auto ns = dynamical->neval;
//    int is, iatm, ixyz;
//    double force[3] = {0.0, 0.0, 0.0};
//
//    for (iatm = 0; iatm < natmin; iatm++) {
//        for (ixyz = 0; ixyz < 3; ixyz++) {
//            force[ixyz] = 0.0;
//            for (is = 0; is < ns; is++) {
//                force[ixyz] -= evec_harmonic[0][is][iatm * 3 + ixyz].real() *
//                               std::sqrt(system->mass[system->map_p2s[iatm][0]]) * v1_renorm[is].real();
//            }
//            force_array[iatm * 3 + ixyz] = force[ixyz];
//        }
//    }
//}


void Scph::find_degeneracy(std::vector<int> *degeneracy_out, const unsigned int nk_in, double **eval_in) const
{
    // eval is omega^2 in atomic unit

    const auto ns = dynamical->neval;
    const auto tol_omega = 1.0e-7;

    for (unsigned int ik = 0; ik < nk_in; ++ik) {

        degeneracy_out[ik].clear();

        auto omega_prev = eval_in[ik][0];
        auto ideg = 1;

        for (unsigned int is = 1; is < ns; ++is) {
            const auto omega_now = eval_in[ik][is];

            if (std::abs(omega_now - omega_prev) < tol_omega) {
                ++ideg;
            } else {
                degeneracy_out[ik].push_back(ideg);
                ideg = 1;
                omega_prev = omega_now;
            }
        }
        degeneracy_out[ik].push_back(ideg);
    }
}


void Scph::initialize_scph_iteration(const double temp, const bool flag_converged, double **omega2_prev,
                                     const unsigned int verbosity, Eigen::MatrixXd &omega_now,
                                     Eigen::MatrixXd &omega2_HA, std::vector<Eigen::MatrixXcd> &evec_initial,
                                     std::vector<Eigen::MatrixXcd> &evec_initial_adjoint,
                                     std::complex<double> ***cmat_convert) const
{
    using namespace Eigen;
    constexpr auto complex_one = std::complex<double>(1.0, 0.0);
    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);

    const auto nk = kmesh_dense->nk;
    const auto ns = dynamical->neval;

    std::cout << " Temperature = " << temp << " K\n";

    // Set initial values
    for (unsigned int ik = 0; ik < nk; ++ik) {
        for (unsigned int is = 0; is < ns; ++is) {

            if (flag_converged) {
                if (omega2_prev[ik][is] < 0.0 && std::abs(omega2_prev[ik][is]) > 1.0e-16 && verbosity > 0) {
                    std::cout << "Warning : Large negative frequency detected\n";
                }
                omega_now(ik, is) = std::sqrt(std::abs(omega2_prev[ik][is]));
            } else {
                omega_now(ik, is) = std::sqrt(std::abs(omega2_harmonic[ik][is]));
            }

            omega2_HA(ik, is) = omega2_harmonic[ik][is];

            for (unsigned int js = 0; js < ns; ++js) {
                // transpose evec so that evec_initial can be used as is.
                evec_initial[ik](js, is) = evec_harmonic[ik][is][js];

                if (!flag_converged) {
                    // Initialize Cmat with identity matrix
                    if (is == js) {
                        cmat_convert[ik][is][js] = complex_one;
                    } else {
                        cmat_convert[ik][is][js] = complex_zero;
                    }
                }
            }
        }
        // Compute adjoint once after the entire matrix is filled
        evec_initial_adjoint[ik] = evec_initial[ik].adjoint();
    }
}

void Scph::setup_harmonic_dynamical_matrices(const Eigen::MatrixXd &omega2_HA,
                                             const std::vector<Eigen::MatrixXcd> &evec_initial,
                                             std::complex<double> **delta_v2_renorm,
                                             std::vector<Eigen::MatrixXcd> &Fmat0,
                                             std::complex<double> ***dymat_q_HA) const
{
    using namespace Eigen;
    const auto ns = dynamical->neval;
    const auto nk_irred_interpolate = kmesh_coarse->nk_irred;

    MatrixXcd Dymat(ns, ns);

    // Set initial harmonic dymat and eigenvalues
    for (unsigned int ik = 0; ik < nk_irred_interpolate; ++ik) {
        const auto knum_interpolate = kmesh_coarse->kpoint_irred_all[ik][0].knum;
        const auto knum = kmap_coarse_to_dense[knum_interpolate];

        Dymat = omega2_HA.row(knum).asDiagonal();
        auto evec_tmp = evec_initial[knum];

        // Harmonic dynamical matrix
        Dymat = evec_tmp * Dymat * evec_tmp.adjoint();
        dynamical->symmetrize_dynamical_matrix(ik, kmesh_coarse, mat_transform_sym, Dymat);

        for (unsigned int is = 0; is < ns; ++is) {
            for (unsigned int js = 0; js < ns; ++js) {
                dymat_q_HA[is][js][knum_interpolate] = Dymat(is, js);
            }
        }

        // Harmonic Fmat
        Fmat0[ik] = omega2_HA.row(knum).asDiagonal();
        for (unsigned int is = 0; is < ns; ++is) {
            for (unsigned int js = 0; js < ns; ++js) {
                Fmat0[ik](is, js) += delta_v2_renorm[knum_interpolate][is * ns + js];
            }
        }
    }

    dynamical->replicate_dymat_for_all_kpoints(kmesh_coarse, mat_transform_sym, dymat_q_HA);
}

void Scph::compute_qmat_and_dmat(const Eigen::MatrixXd &omega_now, const double temp,
                                 std::complex<double> ***cmat_convert,
                                 std::vector<Eigen::MatrixXcd> &dmat_convert) const
{
    using namespace Eigen;

    const auto nk = kmesh_dense->nk;
    const auto ns = dynamical->neval;

    MatrixXcd Qmat(ns, ns);
    MatrixXcd Cmat(ns, ns), Dmat(ns, ns);

    for (unsigned int ik = 0; ik < nk; ++ik) {
        Qmat.setZero();
        for (unsigned int is = 0; is < ns; ++is) {
            const auto omega1 = omega_now(ik, is);
            if (std::abs(omega1) > eps8) {
                // Note that the missing factor 2 in the denominator of Qmat is
                // already considered in the v4_array_all.
                if (thermodynamics->classical) {
                    Qmat(is, is) = std::complex<double>(2.0 * temp * thermodynamics->T_to_Ryd / (omega1 * omega1), 0.0);
                } else {
                    const auto n1 = thermodynamics->fB(omega1, temp);
                    Qmat(is, is) = std::complex<double>((2.0 * n1 + 1.0) / omega1, 0.0);
                }
            }
        }

        for (unsigned int is = 0; is < ns; ++is) {
            for (unsigned int js = 0; js < ns; ++js) {
                Cmat(is, js) = cmat_convert[ik][is][js];
            }
        }

        Dmat = Cmat * Qmat * Cmat.adjoint();
        dmat_convert[ik] = Dmat;
    }
}

void Scph::update_fmat_with_v4(const std::vector<Eigen::MatrixXcd> &Fmat0,
                               std::complex<double> *const *const *v4_array_all,
                               const std::vector<Eigen::MatrixXcd> &dmat_convert, const bool offdiag,
                               const unsigned int ik_irred, Eigen::MatrixXcd &Fmat) const
{
    using namespace Eigen;
    const auto nk = kmesh_dense->nk;
    const auto ns = dynamical->neval;
    const auto ns2 = ns * ns;

    // Fmat harmonic
    Fmat = Fmat0[ik_irred];

    // Anharmonic correction to Fmat
    if (!offdiag) {
#pragma omp parallel for
        for (int is = 0; is < ns; ++is) {
            for (unsigned int jk = 0; jk < nk; ++jk) {
                for (unsigned int ks = 0; ks < ns; ++ks) {
                    Fmat(is, is) +=
                        v4_array_all[nk * ik_irred + jk][(ns + 1) * is][(ns + 1) * ks] * dmat_convert[jk](ks, ks);
                }
            }
        }
    } else {
#pragma omp parallel for
        for (int ijs = 0; ijs < ns2; ++ijs) {
            auto is = ijs / ns;
            auto js = ijs % ns;
            for (unsigned int jk = 0; jk < nk; ++jk) {
                for (unsigned int ks = 0; ks < ns; ++ks) {
                    for (unsigned int ls = 0; ls < ns; ++ls) {
                        Fmat(is, js) += v4_array_all[nk * ik_irred + jk][ijs][ns * ks + ls] * dmat_convert[jk](ks, ls);
                    }
                }
            }
        }
    }
}

void Scph::diagonalize_and_symmetrize(const Eigen::MatrixXcd &Fmat, const std::vector<Eigen::MatrixXcd> &evec_initial,
                                      std::complex<double> ***v4_array_all, const unsigned int ik_irred,
                                      const unsigned int knum, const unsigned int knum_interpolate,
                                      const bool flag_converged, double **omega2_out, const unsigned int verbosity,
                                      int &icount, Eigen::VectorXd &eval_tmp, std::complex<double> ***dymat_q) const
{
    using namespace Eigen;
    const auto ns = dynamical->neval;
    const auto nk = kmesh_dense->nk;

    SelfAdjointEigenSolver<MatrixXcd> saes;
    saes.compute(Fmat);
    eval_tmp = saes.eigenvalues();

    for (unsigned int is = 0; is < ns; ++is) {

        double omega2_tmp = eval_tmp(is);

        if (omega2_tmp < 0.0 && std::abs(omega2_tmp) > 1.0e-16) {

            if (verbosity > 1) {
                std::cout << " Detect imaginary : ";
                std::cout << "  knum = " << knum + 1 << " is = " << is + 1 << '\n';
                for (int j = 0; j < 3; ++j) {
                    std::cout << "  xk = " << std::setw(15) << kmesh_dense->xk[knum][j];
                }
                std::cout << '\n';
            }

            if (v4_array_all[nk * ik_irred + knum][(ns + 1) * is][(ns + 1) * is].real() > 0.0) {
                if (verbosity > 1) {
                    std::cout << "  onsite V4 is positive\n\n";
                }

                if (flag_converged) {
                    ++icount;
                    eval_tmp(is) = omega2_out[knum][is] * std::pow(0.99, icount);
                } else {
                    ++icount;
                    eval_tmp(is) = -eval_tmp(is) * std::pow(0.99, icount);
                }
            } else {
                if (verbosity > 1) {
                    std::cout << "  onsite V4 is negative\n\n";
                }
                eval_tmp(is) = std::abs(omega2_tmp);
            }
        }
    }

    // New eigenvector matrix E_{new}= E_{old} * C
    const auto mat_tmp = evec_initial[knum] * saes.eigenvectors();
    MatrixXcd Dymat = mat_tmp * eval_tmp.asDiagonal() * mat_tmp.adjoint();

    dynamical->symmetrize_dynamical_matrix(ik_irred, kmesh_coarse, mat_transform_sym, Dymat);
    for (unsigned int is = 0; is < ns; ++is) {
        for (unsigned int js = 0; js < ns; ++js) {
            dymat_q[is][js][knum_interpolate] = Dymat(is, js);
        }
    }
}

void Scph::interpolate_to_dense_mesh(std::complex<double> ***dymat_q,
                                     const std::complex<double> *const *const *dymat_q_HA,
                                     const std::vector<Eigen::MatrixXcd> &evec_initial,
                                     Eigen::MatrixXd &eval_interpolate, std::vector<Eigen::MatrixXcd> &evec_new,
                                     std::complex<double> ***cmat_convert, Eigen::MatrixXd &omega_now) const
{
    using namespace Eigen;
    const auto nk = kmesh_dense->nk;
    const auto ns = dynamical->neval;
    const auto nk_interpolate = kmesh_coarse->nk;
    const auto nk1 = kmesh_interpolate[0];
    const auto nk2 = kmesh_interpolate[1];
    const auto nk3 = kmesh_interpolate[2];

    std::complex<double> ***dymat_r_new;
    allocate(dymat_r_new, ns, ns, nk_interpolate);

    dynamical->replicate_dymat_for_all_kpoints(kmesh_coarse, mat_transform_sym, dymat_q);

    // Subtract harmonic contribution from the dynamical matrix
    for (unsigned int ik = 0; ik < nk_interpolate; ++ik) {
        for (unsigned int is = 0; is < ns; ++is) {
            for (unsigned int js = 0; js < ns; ++js) {
                dymat_q[is][js][ik] -= dymat_q_HA[is][js][ik];
            }
        }
    }

    // IMPORTANT: FFTW plan must be created inside the loop for each (is,js) pair
    // because the plan is tied to specific memory addresses. Creating it once
    // outside with nullptr or fftw_execute_dft() with different pointers
    // causes incorrect results due to memory alignment assumptions in the plan.
    for (unsigned int is = 0; is < ns; ++is) {
        for (unsigned int js = 0; js < ns; ++js) {
            fftw_plan plan = fftw_plan_dft_3d(nk1,
                                              nk2,
                                              nk3,
                                              reinterpret_cast<fftw_complex *>(dymat_q[is][js]),
                                              reinterpret_cast<fftw_complex *>(dymat_r_new[is][js]),
                                              FFTW_FORWARD,
                                              FFTW_ESTIMATE);
            fftw_execute(plan);
            fftw_destroy_plan(plan);

            for (unsigned int ik = 0; ik < nk_interpolate; ++ik)
                dymat_r_new[is][js][ik] /= static_cast<double>(nk_interpolate);
        }
    }

    // Create temporary C-style arrays for exec_interpolation
    double **eval_temp;
    std::complex<double> ***evec_temp;
    allocate(eval_temp, nk, ns);
    allocate(evec_temp, nk, ns, ns);

    dynamical->exec_interpolation(kmesh_interpolate,
                                  dymat_r_new,
                                  nk,
                                  kmesh_dense->xk,
                                  kmesh_dense->kvec_na,
                                  eval_temp,
                                  evec_temp,
                                  dymat_harm_short,
                                  dymat_harm_long,
                                  mindist_list,
                                  true,
                                  true);

    MatrixXcd evec_tmp(ns, ns);
    MatrixXcd Cmat(ns, ns);

    for (unsigned int ik = 0; ik < nk; ++ik) {
        // Copy eigenvalues from temp array to Eigen matrix
        for (unsigned int is = 0; is < ns; ++is) {
            eval_interpolate(ik, is) = eval_temp[ik][is];
        }

        // Copy eigenvectors from temp array to Eigen matrix and transpose
        for (unsigned int is = 0; is < ns; ++is) {
            for (unsigned int js = 0; js < ns; ++js) {
                evec_tmp(is, js) = evec_temp[ik][js][is];
                evec_new[ik](is, js) = evec_temp[ik][is][js];
            }
        }

        Cmat = evec_initial[ik].adjoint() * evec_tmp;

        for (unsigned int is = 0; is < ns; ++is) {
            omega_now(ik, is) = eval_interpolate(ik, is);
            for (unsigned int js = 0; js < ns; ++js) {
                cmat_convert[ik][is][js] = Cmat(is, js);
            }
        }
    }

    deallocate(eval_temp);
    deallocate(evec_temp);
    deallocate(dymat_r_new);
}


bool Scph::check_convergence(const Eigen::MatrixXd &omega_now, const Eigen::MatrixXd &omega_old, const double conv_tol,
                             const unsigned int verbosity, const int iloop, double &diff) const
{
    const auto nk_interpolate = kmesh_coarse->nk;
    const auto ns = dynamical->neval;

    if (iloop == 0) {
        if (verbosity > 0) {
            std::cout << "  SCPH ITER " << std::setw(5) << iloop + 1 << " :  DIFF = N/A\n";
        }
        return false;
    }

    diff = 0.0;

    for (unsigned int ik = 0; ik < nk_interpolate; ++ik) {
        const auto knum = kmap_coarse_to_dense[ik];
        for (unsigned int is = 0; is < ns; ++is) {
            diff += pow2(omega_now(knum, is) - omega_old(knum, is));
        }
    }
    diff /= static_cast<double>(nk_interpolate * ns);
    if (verbosity > 0) {
        std::cout << "  SCPH ITER " << std::setw(5) << iloop + 1 << " : ";
        std::cout << " DIFF = " << std::scientific << std::setw(15) << std::sqrt(diff) << '\n';
    }

    if (std::sqrt(diff) < conv_tol) {
        auto has_negative = false;

        for (unsigned int ik = 0; ik < nk_interpolate; ++ik) {
            const auto knum = kmap_coarse_to_dense[ik];
            for (unsigned int is = 0; is < ns; ++is) {
                if (omega_now(knum, is) < 0.0 && std::abs(omega_now(knum, is)) > eps8) {
                    has_negative = true;
                    break;
                }
            }
        }
        if (!has_negative) {
            if (verbosity > 0) std::cout << "  DIFF < SCPH_TOL : break SCPH loop\n";
            return true;
        }
        if (verbosity > 0) std::cout << "  DIFF < SCPH_TOL but a negative frequency is detected.\n";
    }

    return false;
}

void Scph::compute_anharmonic_frequency(std::complex<double> ***v4_array_all, double **omega2_out,
                                        std::complex<double> ***evec_anharm_scph, const double temp,
                                        bool &flag_converged, std::complex<double> ***cmat_convert, const bool offdiag,
                                        std::complex<double> **delta_v2_renorm, const unsigned int verbosity)
{
    // This is the main function of the SCPH equation.
    // The detailed algorithm can be found in PRB 92, 054301 (2015).
    // Eigen3 library is used for the compact notation of matrix-matrix products.

    using namespace Eigen;

    int ik;
    unsigned int is, js;
    const auto nk = kmesh_dense->nk;
    const auto ns = dynamical->neval;
    unsigned int knum;
    const auto nk_interpolate = kmesh_coarse->nk;
    const auto nk_irred_interpolate = kmesh_coarse->nk_irred;
    int iloop;

    MatrixXd omega_now(nk, ns), omega_old(nk, ns);
    MatrixXd omega2_HA(nk, ns);
    VectorXd eval_tmp(ns);
    MatrixXcd Fmat(ns, ns);

    double diff, diff_prev = 1.0e10;
    const double conv_tol = tolerance_scph;
    double alpha = mixalpha;

    // Use Eigen types for better memory management and performance
    MatrixXd eval_interpolate(nk, ns);
    std::vector<MatrixXcd> evec_new;

    std::complex<double> ***dymat_q, ***dymat_q_HA;

    std::vector<MatrixXcd> dmat_convert, dmat_convert_old;
    std::vector<MatrixXcd> evec_initial, evec_initial_adjoint;
    std::vector<MatrixXcd> Fmat0;

    // Reserve and construct Eigen matrices efficiently
    dmat_convert.reserve(nk);
    dmat_convert_old.reserve(nk);
    evec_initial.reserve(nk);
    evec_initial_adjoint.reserve(nk);
    evec_new.reserve(nk);

    for (ik = 0; ik < nk; ++ik) {
        dmat_convert.emplace_back(ns, ns);
        dmat_convert_old.emplace_back(ns, ns);
        evec_initial.emplace_back(ns, ns);
        evec_initial_adjoint.emplace_back(ns, ns);
        evec_new.emplace_back(ns, ns);
    }

    Fmat0.reserve(nk_irred_interpolate);
    for (ik = 0; ik < nk_irred_interpolate; ++ik) {
        Fmat0.emplace_back(ns, ns);
    }

    // dymat arrays still use C-style for FFTW compatibility (old method)
    allocate(dymat_q, ns, ns, nk_interpolate);
    allocate(dymat_q_HA, ns, ns, nk_interpolate);

    const auto T_in = temp;

    // Initialize iteration
    initialize_scph_iteration(T_in,
                              flag_converged,
                              omega2_out,
                              verbosity,
                              omega_now,
                              omega2_HA,
                              evec_initial,
                              evec_initial_adjoint,
                              cmat_convert);

    // Setup harmonic dynamical matrices
    setup_harmonic_dynamical_matrices(omega2_HA, evec_initial, delta_v2_renorm, Fmat0, dymat_q_HA);

    // This can be done outside the function
    dynamical->precompute_dymat_harm(kmesh_dense->nk,
                                     kmesh_dense->xk,
                                     kmesh_dense->kvec_na,
                                     dymat_harm_short,
                                     dymat_harm_long);

    int icount = 0;

    // Main loop
    for (iloop = 0; iloop < maxiter; ++iloop) {

        // Compute Qmat and Dmat from current frequencies
        compute_qmat_and_dmat(omega_now, T_in, cmat_convert, dmat_convert);

        // Mixing dmat
        if (iloop > 0) {
            // if (diff > diff_prev) {
            //     alpha = std::max(alpha * 0.7, 0.02);
            // } else {
            //     alpha = std::min(alpha * 1.1, 1.0);
            // }
            for (ik = 0; ik < nk; ++ik) {
                dmat_convert[ik] = alpha * dmat_convert[ik] + (1.0 - alpha) * dmat_convert_old[ik];
            }
        }

        // TODO: This loop may be parallelized by MPI in the future.
        for (ik = 0; ik < nk_irred_interpolate; ++ik) {

            const unsigned int knum_interpolate = kmesh_coarse->kpoint_irred_all[ik][0].knum;
            knum = kmap_coarse_to_dense[knum_interpolate];

            // Update Fmat with V4 contribution
            update_fmat_with_v4(Fmat0, v4_array_all, dmat_convert, offdiag, ik, Fmat);

            // Diagonalize and symmetrize
            diagonalize_and_symmetrize(Fmat,
                                       evec_initial,
                                       v4_array_all,
                                       ik,
                                       knum,
                                       knum_interpolate,
                                       flag_converged,
                                       omega2_out,
                                       verbosity,
                                       icount,
                                       eval_tmp,
                                       dymat_q);

        } // close loop ik

        // Interpolate to dense mesh using FFT-based method (correct Fourier interpolation)
        interpolate_to_dense_mesh(dymat_q,
                                  dymat_q_HA,
                                  evec_initial,
                                  eval_interpolate,
                                  evec_new,
                                  cmat_convert,
                                  omega_now);

        // Check convergence on the coarse k points
        if (check_convergence(omega_now, omega_old, conv_tol, verbosity, iloop, diff)) {
            break;
        }

        // Save current omega for next iteration's comparison
        omega_old = omega_now;

        for (ik = 0; ik < nk; ++ik) {
            dmat_convert_old[ik] = dmat_convert[ik];
        }
        diff_prev = diff;
    } // end loop iteration

    if (std::sqrt(diff) < conv_tol) {
        if (verbosity > 0) {
            std::cout << " Temp = " << T_in;
            std::cout << " : convergence achieved in " << std::setw(5) << iloop + 1 << " iterations.\n";
        }
        flag_converged = true;
    } else {
        if (verbosity > 0) {
            std::cout << "Temp = " << T_in;
            std::cout << " : not converged.\n";
        }
        flag_converged = false;
    }

    for (ik = 0; ik < nk; ++ik) {
        for (is = 0; is < ns; ++is) {
            if (eval_interpolate(ik, is) < 0.0) {
                if (std::abs(eval_interpolate(ik, is)) <= eps10) {
                    omega2_out[ik][is] = 0.0;
                } else {
                    omega2_out[ik][is] = -pow2(eval_interpolate(ik, is));
                }
            } else {
                omega2_out[ik][is] = pow2(eval_interpolate(ik, is));
            }
            for (js = 0; js < ns; ++js) {
                evec_anharm_scph[ik][is][js] = evec_new[ik](is, js);
            }
        }
    }

    if (verbosity > 1) {
        std::cout << "New eigenvalues\n";
        for (ik = 0; ik < nk_interpolate; ++ik) {
            knum = kmap_coarse_to_dense[ik];
            for (is = 0; is < ns; ++is) {
                std::cout << " ik_interpolate = " << std::setw(5) << ik + 1;
                std::cout << " is = " << std::setw(5) << is + 1;
                std::cout << " omega2 = " << std::setw(15) << omega2_out[knum][is] << '\n';
            }
            std::cout << '\n';
        }
    }

    // Eigen matrices are automatically deallocated
    deallocate(dymat_q);
    deallocate(dymat_q_HA);
}


void Scph::compute_anharmonic_frequency_diis_perkpoint(std::complex<double> ***v4_array_all, double **omega2_out,
                                                       std::complex<double> ***evec_anharm_scph, const double temp,
                                                       bool &flag_converged, std::complex<double> ***cmat_convert,
                                                       const bool offdiag, std::complex<double> **delta_v2_renorm,
                                                       const unsigned int verbosity)
{
    // SCPH with per-k-point DIIS using GDIIS_PerKpoint class
    // Each k-point has independent DIIS history and coefficients

    using namespace Eigen;

    int ik;
    unsigned int is, js;
    const auto nk = kmesh_dense->nk;
    const auto ns = dynamical->neval;
    unsigned int knum;
    const auto nk_interpolate = kmesh_coarse->nk;
    const auto nk_irred_interpolate = kmesh_coarse->nk_irred;
    int iloop;

    MatrixXd omega_now(nk, ns), omega_old(nk, ns);
    MatrixXd omega2_HA(nk, ns);
    VectorXd eval_tmp(ns);
    MatrixXcd Fmat(ns, ns);

    double diff = 1.0;
    const double conv_tol = tolerance_scph;

    MatrixXd eval_interpolate(nk, ns);
    std::vector<MatrixXcd> evec_new;

    std::complex<double> ***dymat_q, ***dymat_q_HA;

    std::vector<MatrixXcd> dmat_convert, dmat_convert_old, dmat_input;
    std::vector<MatrixXcd> evec_initial, evec_initial_adjoint;
    std::vector<MatrixXcd> Fmat0;

    dmat_convert.reserve(nk);
    dmat_convert_old.reserve(nk);
    dmat_input.reserve(nk);
    evec_initial.reserve(nk);
    evec_initial_adjoint.reserve(nk);
    evec_new.reserve(nk);

    for (ik = 0; ik < nk; ++ik) {
        dmat_convert.emplace_back(ns, ns);
        dmat_convert_old.emplace_back(ns, ns);
        dmat_input.emplace_back(ns, ns);
        evec_initial.emplace_back(ns, ns);
        evec_initial_adjoint.emplace_back(ns, ns);
        evec_new.emplace_back(ns, ns);
    }

    Fmat0.reserve(nk_irred_interpolate);
    for (ik = 0; ik < nk_irred_interpolate; ++ik) {
        Fmat0.emplace_back(ns, ns);
    }

    allocate(dymat_q, ns, ns, nk_interpolate);
    allocate(dymat_q_HA, ns, ns, nk_interpolate);

    const auto T_in = temp;

    initialize_scph_iteration(T_in,
                              flag_converged,
                              omega2_out,
                              verbosity,
                              omega_now,
                              omega2_HA,
                              evec_initial,
                              evec_initial_adjoint,
                              cmat_convert);

    setup_harmonic_dynamical_matrices(omega2_HA, evec_initial, delta_v2_renorm, Fmat0, dymat_q_HA);

    dynamical->precompute_dymat_harm(kmesh_dense->nk,
                                     kmesh_dense->xk,
                                     kmesh_dense->kvec_na,
                                     dymat_harm_short,
                                     dymat_harm_long);

    // Initialize per-k-point DIIS mixer
    const int diis_history = std::min(3, static_cast<int>(maxiter / 3));
    GDIIS_PerKpoint gdiis_mixer_perkpoint(nk, diis_history, mixalpha, verbosity);

    int icount = 0;
    bool use_diis = false;
    int diis_fail_count = 0;

    // Main SCPH iteration with per-k-point DIIS
    for (iloop = 0; iloop < maxiter; ++iloop) {

        // Step 1: Use current D-matrix to build F and diagonalize
        for (ik = 0; ik < nk_irred_interpolate; ++ik) {
            const unsigned int knum_interpolate = kmesh_coarse->kpoint_irred_all[ik][0].knum;
            knum = kmap_coarse_to_dense[knum_interpolate];

            update_fmat_with_v4(Fmat0, v4_array_all, dmat_convert, offdiag, ik, Fmat);
            diagonalize_and_symmetrize(Fmat,
                                       evec_initial,
                                       v4_array_all,
                                       ik,
                                       knum,
                                       knum_interpolate,
                                       flag_converged,
                                       omega2_out,
                                       verbosity,
                                       icount,
                                       eval_tmp,
                                       dymat_q);
        }

        // Step 2: Interpolate to get new eigenvalues (output from using dmat_convert)
        interpolate_to_dense_mesh(dymat_q,
                                  dymat_q_HA,
                                  evec_initial,
                                  eval_interpolate,
                                  evec_new,
                                  cmat_convert,
                                  omega_now);

        // Step 3: Check convergence before mixing
        if (check_convergence(omega_now, omega_old, conv_tol, verbosity, iloop, diff)) {
            break;
        }

        // Save current omega for next iteration's comparison
        omega_old = omega_now;

        // Step 4: Push to DIIS - the D we USED and the ω we GOT (input-output pair!)
        if (iloop >= 2 && diis_fail_count < 3) {
            use_diis = true;
        }

        if (use_diis) {
            gdiis_mixer_perkpoint.push(dmat_convert, omega_now); // Correct: D_input → ω_output
        }

        // Step 5: Compute what D should be based on new eigenvalues
        compute_qmat_and_dmat(omega_now, T_in, cmat_convert, dmat_input);

        // Step 6: Mix - try per-k-point DIIS or fall back to simple mixing
        std::vector<MatrixXcd> dmat_mixed(nk);

        if (iloop > 0) {
            // Default: simple mixing as fallback
            for (ik = 0; ik < nk; ++ik) {
                dmat_mixed[ik] = mixalpha * dmat_input[ik] + (1.0 - mixalpha) * dmat_convert[ik];
            }
        } else {
            dmat_mixed = dmat_input;
        }

        // Try per-k-point DIIS extrapolation if ready
        if (use_diis && gdiis_mixer_perkpoint.is_ready()) {
            bool diis_success = gdiis_mixer_perkpoint.extrapolate(dmat_mixed);
            if (diis_success) {
                diis_fail_count = 0;
                if (verbosity > 1) {
                    std::cout << "  Per-k-point DIIS extrapolation successful at iteration " << iloop + 1 << "\n";
                }
            } else {
                // Fallback already set above
                if (verbosity > 0) {
                    std::cout << "  Per-k-point DIIS failed at iteration " << iloop + 1 << ", using simple mixing\n";
                }
                diis_fail_count++;
                if (diis_fail_count >= 3) {
                    use_diis = false;
                    gdiis_mixer_perkpoint.clear();
                    if (verbosity > 0) {
                        std::cout << "  Disabling per-k-point DIIS after 3 failures\n";
                    }
                }
            }
        }

        // Step 7: Update dmat_convert for next iteration
        for (ik = 0; ik < nk; ++ik) {
            dmat_convert[ik] = dmat_mixed[ik];
        }
    }

    if (std::sqrt(diff) < conv_tol) {
        if (verbosity > 0) {
            std::cout << " Temp = " << T_in << " : convergence achieved in " << std::setw(5) << iloop + 1
                      << " iterations";
            if (use_diis) {
                std::cout << " (with per-k-point DIIS)";
                auto stats = gdiis_mixer_perkpoint.get_success_stats();
                int total_success = 0;
                for (int s: stats)
                    total_success += s;
                if (verbosity > 1) {
                    std::cout << "\n  Total DIIS successes: " << total_success;
                }
            }
            std::cout << ".\n";
        }
        flag_converged = true;
    } else {
        if (verbosity > 0) {
            std::cout << "Temp = " << T_in << " : not converged.\n";
        }
        flag_converged = false;
    }

    for (ik = 0; ik < nk; ++ik) {
        for (is = 0; is < ns; ++is) {
            if (eval_interpolate(ik, is) < 0.0) {
                if (std::abs(eval_interpolate(ik, is)) <= eps10) {
                    omega2_out[ik][is] = 0.0;
                } else {
                    omega2_out[ik][is] = -pow2(eval_interpolate(ik, is));
                }
            } else {
                omega2_out[ik][is] = pow2(eval_interpolate(ik, is));
            }
            for (js = 0; js < ns; ++js) {
                evec_anharm_scph[ik][is][js] = evec_new[ik](is, js);
            }
        }
    }

    deallocate(dymat_q);
    deallocate(dymat_q_HA);
}

void Scph::get_permutation_matrix(const int ns, std::complex<double> **cmat_in, Eigen::MatrixXd &permutation_matrix)
{
    std::vector<int> has_visited(ns, 0);
    permutation_matrix = Eigen::MatrixXd::Zero(ns, ns);

    for (auto is = 0; is < ns; ++is) {

        if (has_visited[is]) continue;
        auto iloc = -1;
        auto maxelem = 0.0;

        for (auto js = 0; js < ns; ++js) {
            const auto cnorm = std::abs(cmat_in[js][is]);
            if (cnorm > maxelem && (has_visited[js] == 0)) {
                iloc = js;
                maxelem = cnorm;
            }
        }
        if (iloc == -1) {
            exit("get_permutation_matrix", "Band index mapping failed.");
        } else {
            permutation_matrix(is, iloc) = 1.0;
            permutation_matrix(iloc, is) = 1.0;
            has_visited[iloc] = 1;
            has_visited[is] = 1;
        }
    }
}

void Scph::update_frequency(const double temperature_in, const Eigen::MatrixXd &omega2_in,
                            const std::vector<Eigen::MatrixXcd> &Fmat0, const std::vector<Eigen::MatrixXcd> &evec0,
                            std::complex<double> ***dymat0, std::complex<double> ***v4_array_all,
                            std::complex<double> ***cmat_convert, std::complex<double> ***dymat_out,
                            std::complex<double> ***evec_out, const bool offdiag, Eigen::MatrixXd &omega2_out)
{
    // From the given omega2_in, compute new omega2_out.
    // The order of the eigenvalues may be changed, so you will need to take care of it outside this function,
    // especially when computing the difference.
    using namespace Eigen;
    const auto nk = kmesh_dense->nk;
    const auto ns = dynamical->neval;
    const auto ns2 = ns * ns;
    const auto nk_interpolate = kmesh_coarse->nk;
    std::vector<MatrixXcd> dmat(nk);
    VectorXcd Kmat(ns);
    MatrixXcd Cmat(ns, ns), Dmat(ns, ns);
    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);
    std::complex<double> ***dymat_q;
    std::complex<double> ***dymat_r_new;

    allocate(dymat_q, ns, ns, nk_interpolate);
    allocate(dymat_r_new, ns, ns, nk_interpolate);

    for (auto ik = 0; ik < nk; ++ik) {
        for (auto is = 0; is < ns; ++is) {
            auto omega_tmp = omega2_in(ik, is);
            if (std::abs(omega_tmp) < eps15) {
                Kmat(is) = complex_zero;
                std::cout << "Kmat is zero for " << std::setw(4) << ik << std::setw(5) << is << " omega = " << omega_tmp
                          << '\n';
            } else {
                // Note that the missing factor 2 in the denominator of Qmat is
                // already considered in the v4_array_all.
                if (thermodynamics->classical) {
                    Kmat(is) =
                        std::complex<double>(2.0 * temperature_in * thermodynamics->T_to_Ryd / (std::abs(omega_tmp)),
                                             0.0);
                } else {
                    const auto omega1 = std::sqrt(std::abs(omega_tmp));
                    auto n1 = thermodynamics->fB(omega1, temperature_in);
                    Kmat(is) = std::complex<double>((2.0 * n1 + 1.0) / omega1, 0.0);
                }
            }
        }

        for (auto is = 0; is < ns; ++is) {
            for (auto js = 0; js < ns; ++js) {
                Cmat(is, js) = cmat_convert[ik][is][js];
            }
        }

        // C * K * C^{\dagger}
        Dmat = Cmat * Kmat.asDiagonal() * Cmat.adjoint();
        dmat[ik] = Dmat.eval();
    }

    const auto nk_irred_interpolate = kmesh_coarse->nk_irred;

    MatrixXcd mat_tmp(ns, ns);
    SelfAdjointEigenSolver<MatrixXcd> saes;

    MatrixXcd Dymat(ns, ns);
    MatrixXcd Fmat(ns, ns);

    for (auto ik = 0; ik < nk_irred_interpolate; ++ik) {

        const auto knum_interpolate = kmesh_coarse->kpoint_irred_all[ik][0].knum;
        const auto knum = kmap_coarse_to_dense[knum_interpolate];

        // Fmat harmonic
        Fmat = Fmat0[ik];

        // Anharmonic correction to Fmat
        if (!offdiag) {
#pragma omp parallel for
            for (int is = 0; is < ns; ++is) {
                for (auto jk = 0; jk < nk; ++jk) {
                    for (auto ks = 0; ks < ns; ++ks) {
                        Fmat(is, is) += v4_array_all[nk * ik + jk][(ns + 1) * is][(ns + 1) * ks] * dmat[jk](ks, ks);
                    }
                }
            }
        } else {
#pragma omp parallel for
            for (int ijs = 0; ijs < ns2; ++ijs) {
                auto is = ijs / ns;
                auto js = ijs % ns;
                for (auto jk = 0; jk < nk; ++jk) {
                    for (auto ks = 0; ks < ns; ++ks) {
                        for (unsigned int ls = 0; ls < ns; ++ls) {
                            Fmat(is, js) += v4_array_all[nk * ik + jk][ijs][ns * ks + ls] * dmat[jk](ks, ls);
                        }
                    }
                }
            }
        }

        // Diagonalize the new (pseudo) dynamical matrix
        saes.compute(Fmat);

        // New eigenvector matrix E_{new}= E_{old} * C
        mat_tmp = evec0[knum] * saes.eigenvectors();
        // Construct the new (true) dynamical matrix
        Dymat = mat_tmp * saes.eigenvalues().asDiagonal() * mat_tmp.adjoint();
        dynamical->symmetrize_dynamical_matrix(ik, kmesh_coarse, mat_transform_sym, Dymat);

        for (auto is = 0; is < ns; ++is) {
            for (auto js = 0; js < ns; ++js) {
                dymat_out[is][js][knum_interpolate] = Dymat(is, js);
            }
        }
    } // close loop ik

    dynamical->replicate_dymat_for_all_kpoints(kmesh_coarse, mat_transform_sym, dymat_out);

    // Subtract harmonic contribution to the dynamical matrix
    for (auto ik = 0; ik < nk_interpolate; ++ik) {
        for (auto is = 0; is < ns; ++is) {
            for (auto js = 0; js < ns; ++js) {
                dymat_out[is][js][ik] -= dymat0[is][js][ik];
            }
        }
    }

    const auto nk1 = kmesh_interpolate[0];
    const auto nk2 = kmesh_interpolate[1];
    const auto nk3 = kmesh_interpolate[2];

    for (auto is = 0; is < ns; ++is) {
        for (auto js = 0; js < ns; ++js) {
            fftw_plan plan = fftw_plan_dft_3d(nk1,
                                              nk2,
                                              nk3,
                                              reinterpret_cast<fftw_complex *>(dymat_out[is][js]),
                                              reinterpret_cast<fftw_complex *>(dymat_r_new[is][js]),
                                              FFTW_FORWARD,
                                              FFTW_ESTIMATE);
            fftw_execute(plan);
            fftw_destroy_plan(plan);

            for (auto ik = 0; ik < nk_interpolate; ++ik)
                dymat_r_new[is][js][ik] /= static_cast<double>(nk_interpolate);
        }
    }

    double **eval_tmp;

    allocate(eval_tmp, nk, ns);

    dynamical->exec_interpolation(kmesh_interpolate,
                                  dymat_r_new,
                                  nk,
                                  kmesh_dense->xk,
                                  kmesh_dense->kvec_na,
                                  eval_tmp,
                                  evec_out,
                                  dymat_harm_short,
                                  dymat_harm_long,
                                  mindist_list,
                                  false,
                                  false);

    MatrixXcd evec_tmp(ns, ns);

    for (auto ik = 0; ik < nk; ++ik) {
        for (auto is = 0; is < ns; ++is) {
            for (auto js = 0; js < ns; ++js) {
                evec_tmp(is, js) = evec_out[ik][js][is];
            }
        }

        Cmat = evec0[ik].adjoint() * evec_tmp;

        for (auto is = 0; is < ns; ++is) {
            omega2_out(ik, is) = eval_tmp[ik][is];

            for (auto js = 0; js < ns; ++js) {
                cmat_convert[ik][is][js] = Cmat(is, js);
            }
        }
    }
    deallocate(dymat_q);
    deallocate(dymat_r_new);
}
