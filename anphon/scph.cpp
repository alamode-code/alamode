/*
 scph.cpp

 Copyright (c) 2015 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "mpi_common.h"
#include "scph.h"
#include "dynamical.h"
#include "kpoint.h"
#include "anharmonic_core.h"
#include "dielec.h"
#include "ewald.h"
#include "memory.h"
#include "thermodynamics.h"
#include "write_phonons.h"
#include "constants.h"
#include "system.h"
#include "error.h"
#include "mathfunctions.h"
#include "integration.h"
#include "parsephon.h"
#include "phonon_dos.h"
#include "symmetry_core.h"
#include "fcs_phonon.h"
#include "relaxation.h"
#include <iostream>
#include <iomanip>
#include <complex>
#include <algorithm>
#include <fftw3.h>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include "timer.h"
#include <cmath>
#include <cstdlib>
#include <vector>

using namespace PHON_NS;

Scph::Scph(PHON *phon) : Pointers(phon)
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

    evec_harmonic = nullptr;
    omega2_harmonic = nullptr;
    mat_transform_sym = nullptr;
    mindist_list_scph = nullptr;

    bubble = 0;
    phi3_reciprocal = nullptr;
    phi4_reciprocal = nullptr;
    compute_Cv_anharmonic = 0;

    kmesh_coarse = nullptr;
    kmesh_dense = nullptr;

    phase_factor_scph = nullptr;
}

void Scph::deallocate_variables()
{
    if (mindist_list_scph) {
        deallocate(mindist_list_scph);
    }
    if (evec_harmonic) {
        deallocate(evec_harmonic);
    }
    if (omega2_harmonic) {
        deallocate(omega2_harmonic);
    }
    if (mat_transform_sym) {
        deallocate(mat_transform_sym);
    }
    if (phi3_reciprocal) {
        deallocate(phi3_reciprocal);
    }
    if (phi4_reciprocal) {
        deallocate(phi4_reciprocal);
    }

    delete kmesh_coarse;
    delete kmesh_dense;
    delete phase_factor_scph;
}

void Scph::setup_scph()
{
    MPI_Bcast(&bubble, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    setup_kmesh();
    setup_eigvecs();
    system->get_minimum_distances(kmesh_coarse->nk_i, mindist_list_scph);
    setup_pp_interaction();
    dynamical->get_symmetry_gamma_dynamical(kmesh_coarse,
                                            system->get_primcell().number_of_atoms,
                                            system->get_primcell().x_fractional,
                                            symmetry->SymmListWithMap,
                                            mat_transform_sym);
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

    const auto relax_str = relaxation->relax_str;

    if (relax_str != 0 && thermodynamics->calc_FE_bubble) {
        exit("exec_scph",
             "Sorry, RELAX_STR!=0 can't be used with bubble correction of the free energy.");
    }
    if (relax_str != 0 && bubble > 0) {
        exit("exec_scph",
             "Sorry, RELAX_STR!=0 can't be used with bubble self-energy on top of the SCPH calculation.");
    }

    if (restart_scph) {

        if (mympi->my_rank == 0) {
            std::cout << " RESTART_SCPH is true.\n";
            std::cout << " Dynamical matrix is read from file ...";
        }

        // Read anharmonic correction to the dynamical matrix from the existing file
        // SCPH calculation, no structural optimization
        if (relax_str == 0) {
            load_scph_dymat_from_file(delta_dymat_scph,
                                      input->job_title + ".scph_dymat",
                                      kmesh_dense, kmesh_coarse,
                                      dynamical->nonanalytic,
                                      selfenergy_offdiagonal);
        }
            // SCPH + structural optimization
        else if (phon->mode == "SCPH" && relax_str != 0) {
            load_scph_dymat_from_file(delta_dymat_scph, input->job_title + ".scph_dymat",
                                      kmesh_dense, kmesh_coarse,
                                      dynamical->nonanalytic,
                                      selfenergy_offdiagonal);

            load_scph_dymat_from_file(delta_harmonic_dymat_renormalize, input->job_title + ".renorm_harm_dymat",
                                      kmesh_dense, kmesh_coarse,
                                      dynamical->nonanalytic,
                                      selfenergy_offdiagonal);
        }

        // structural optimization
        if (relax_str != 0) {
            relaxation->load_V0_from_file();
        }

    } else {
        if (relax_str == 0) {
            exec_scph_main(delta_dymat_scph);
        }
            // SCPH + structural optimization
        else if (phon->mode == "SCPH" && relax_str != 0) {
            exec_scph_relax_cell_coordinate_main(delta_dymat_scph, delta_harmonic_dymat_renormalize);
        }

        if (mympi->my_rank == 0) {
            // write dymat to file
            // write scph dynamical matrix when scph calculation is performed
            if (phon->mode == "SCPH") {
                store_scph_dymat_to_file(delta_dymat_scph, input->job_title + ".scph_dymat",
                                         kmesh_dense, kmesh_coarse, dynamical->nonanalytic,
                                         selfenergy_offdiagonal);
            }
            // write renormalized harmonic dynamical matrix when the crystal structure is optimized
            if (relax_str != 0) {
                store_scph_dymat_to_file(delta_harmonic_dymat_renormalize, input->job_title + ".renorm_harm_dymat",
                                         kmesh_dense, kmesh_coarse, dynamical->nonanalytic,
                                         selfenergy_offdiagonal);
                relaxation->store_V0_to_file();
            }
            write_anharmonic_correction_fc2(delta_dymat_scph, NT,
                                            kmesh_coarse, mindist_list_scph,
                                            false, 0);
        }
    }

    if (kpoint->kpoint_mode == 2) {
        if (thermodynamics->calc_FE_bubble) {
            compute_free_energy_bubble_SCPH(kmesh_interpolate,
                                            delta_dymat_scph);
        }
    }

    if (bubble) {
        allocate(delta_dymat_scph_plus_bubble, NT, ns, ns, kmesh_coarse->nk);
        bubble_correction(delta_dymat_scph,
                          delta_dymat_scph_plus_bubble);
        if (mympi->my_rank == 0) {
            write_anharmonic_correction_fc2(delta_dymat_scph_plus_bubble, NT,
                                            kmesh_coarse, mindist_list_scph,
                                            false,
                                            bubble);
        }
    }

    postprocess(delta_dymat_scph,
                delta_harmonic_dymat_renormalize,
                delta_dymat_scph_plus_bubble,
                kmesh_coarse,
                mindist_list_scph,
                false, bubble);

    deallocate(delta_dymat_scph);
    deallocate(delta_harmonic_dymat_renormalize);
    if (delta_dymat_scph_plus_bubble) deallocate(delta_dymat_scph_plus_bubble);

}

void Scph::postprocess(std::complex<double> ****delta_dymat,
                       std::complex<double> ****delta_harmonic_dymat_renormalize,
                       std::complex<double> ****delta_dymat_scph_plus_bubble,
                       const KpointMeshUniform *kmesh_coarse_in,
                       MinimumDistList ***mindist_list_in,
                       const bool is_qha,
                       const int bubble_in)
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

                FE_total[iT] = thermodynamics->compute_FE_total(iT,
                                                                FE_QHA[iT],
                                                                dFE_scph[iT]);

                if (writes->getPrintMSD()) {
                    double shift[3]{0.0, 0.0, 0.0};

                    for (auto is = 0; is < ns; ++is) {
                        msd_update[iT][is] = thermodynamics->disp_corrfunc(temperature,
                                                                           is, is,
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
                    for (auto i = 0; i < 3; ++i) shift[i] = static_cast<double>(writes->getShiftUcorr()[i]);

                    for (auto is = 0; is < ns; ++is) {
                        for (auto js = 0; js < ns; ++js) {
                            ucorr_update[iT][is][js] = thermodynamics->disp_corrfunc(temperature,
                                                                                     is, js,
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
                        get_derivative_central_diff(dT, dos->kmesh_dos->nk,
                                                    eval_update[iT - 1],
                                                    eval_update[iT + 1],
                                                    domega_dt);

                        heat_capacity_correction[iT] = thermodynamics->Cv_anharm_correction(temperature,
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
                                           FE_total, is_qha);
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
                                                                               is, is,
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
                        for (auto i = 0; i < 3; ++i) shift[i] = static_cast<double>(writes->getShiftUcorr()[i]);

                        for (auto is = 0; is < ns; ++is) {
                            for (auto js = 0; js < ns; ++js) {
                                ucorr_update[iT][is][js] = thermodynamics->disp_corrfunc(temperature,
                                                                                         is, js,
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

            writes->writePhononEnergies(kpoint->kpoint_general->nk,
                                        eval_update,
                                        is_qha, 0);

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
                writes->writePhononEnergies(kpoint->kpoint_general->nk,
                                            eval_update, false,
                                            bubble_in);
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

            writes->writePhononBands(kpoint->kpoint_bs->nk,
                                     kpoint->kpoint_bs->kaxis,
                                     eval_update, is_qha, 0);

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
                                         eval_update, false,
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
            for (auto i = 0; i < 3; ++i) xk_gam[0][i] = 0.0;

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
                        eval_gam[0][is] = -std::pow(eval_gam[0][is], 2.0);
                    } else {
                        eval_gam[0][is] = std::pow(eval_gam[0][is], 2.0);
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
        if (dielec_update) deallocate(dielec_update);

        if (eval_gam) deallocate(eval_gam);
        if (evec_gam) deallocate(evec_gam);
        if (xk_gam) deallocate(xk_gam);
    }
}

void Scph::load_scph_dymat_from_file(std::complex<double> ****dymat_out,
                                     std::string filename_dymat,
                                     const KpointMeshUniform *kmesh_dense_in,
                                     const KpointMeshUniform *kmesh_coarse_in,
                                     const unsigned int nonanalytic_in,
                                     const bool selfenergy_offdiagonal_in)
{
    const auto ns = dynamical->neval;
    const auto Tmin = system->Tmin;
    const auto Tmax = system->Tmax;
    const auto dT = system->dT;
    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;
    std::vector<double> Temp_array(NT);

    for (int i = 0; i < NT; ++i) {
        Temp_array[i] = Tmin + dT * static_cast<double>(i);
    }

    if (mympi->my_rank == 0) {
        double temp;
        std::ifstream ifs_dymat;
        auto file_dymat = filename_dymat;
        bool consider_offdiag_tmp;
        unsigned int nk_interpolate_ref[3];
        unsigned int nk_scph_tmp[3];
        double Tmin_tmp, Tmax_tmp, dT_tmp;
        double dymat_real, dymat_imag;
        std::string str_dummy;
        int nonanalytic_tmp;


        ifs_dymat.open(file_dymat.c_str(), std::ios::in);

        if (!ifs_dymat) {
            exit("load_scph_dymat_from_file",
                 "Cannot open scph_dymat file");
        }

        // Read computational settings from file and check the consistency.
        ifs_dymat >> nk_interpolate_ref[0] >> nk_interpolate_ref[1] >> nk_interpolate_ref[2];
        ifs_dymat >> nk_scph_tmp[0] >> nk_scph_tmp[1] >> nk_scph_tmp[2];
        ifs_dymat >> Tmin_tmp >> Tmax_tmp >> dT_tmp;
        ifs_dymat >> nonanalytic_tmp >> consider_offdiag_tmp;

        if (nk_interpolate_ref[0] != kmesh_coarse_in->nk_i[0] ||
            nk_interpolate_ref[1] != kmesh_coarse_in->nk_i[1] ||
            nk_interpolate_ref[2] != kmesh_coarse_in->nk_i[2]) {
            exit("load_scph_dymat_from_file",
                 "The number of KMESH_INTERPOLATE is not consistent");
        }
        if (nk_scph_tmp[0] != kmesh_dense_in->nk_i[0] ||
            nk_scph_tmp[1] != kmesh_dense_in->nk_i[1] ||
            nk_scph_tmp[2] != kmesh_dense_in->nk_i[2]) {
            exit("load_scph_dymat_from_file",
                 "The number of KMESH_SCPH is not consistent");
        }
        if (nonanalytic_tmp != nonanalytic_in) {
            warn("load_scph_dymat_from_file",
                 "The NONANALYTIC tag is not consistent");
        }
        if (consider_offdiag_tmp != selfenergy_offdiagonal_in) {
            exit("load_scph_dymat_from_file",
                 "The SELF_OFFDIAG tag is not consistent");
        }

        // Check if the precalculated data for the given temperature range exists
        const auto NT_ref = static_cast<unsigned int>((Tmax_tmp - Tmin_tmp) / dT_tmp) + 1;
        std::vector<double> Temp_array_ref(NT_ref);
        for (int i = 0; i < NT_ref; ++i) {
            Temp_array_ref[i] = Tmin_tmp + dT_tmp * static_cast<double>(i);
        }
        std::vector<int> flag_load(NT_ref);
        for (int i = 0; i < NT_ref; ++i) {
            flag_load[i] = 0;
            for (int j = 0; j < NT; ++j) {
                if (std::abs(Temp_array_ref[i] - Temp_array[j]) < eps6) {
                    flag_load[i] = 1;
                    break;
                }
            }
        }
        int icount = 0;
        for (int iT = 0; iT < NT_ref; ++iT) {
            ifs_dymat >> str_dummy >> temp;
            for (int is = 0; is < ns; ++is) {
                for (int js = 0; js < ns; ++js) {
                    for (int ik = 0; ik < kmesh_coarse_in->nk; ++ik) {
                        ifs_dymat >> dymat_real >> dymat_imag;
                        if (flag_load[iT]) {
                            dymat_out[icount][is][js][ik]
                                    = std::complex<double>(dymat_real, dymat_imag);
                        }
                    }
                }
            }
            if (flag_load[iT]) icount += 1;
        }

        ifs_dymat.close();

        if (icount != NT) {
            exit("load_scph_dymat_from_file",
                 "The temperature information is not consistent");
        }
        std::cout << " done.\n";
    }
    // Broadcast to all MPI threads
    mpi_bcast_complex(dymat_out, NT, kmesh_coarse_in->nk, ns);
}

void Scph::store_scph_dymat_to_file(const std::complex<double> *const *const *const *dymat_in,
                                    std::string filename_dymat,
                                    const KpointMeshUniform *kmesh_dense_in,
                                    const KpointMeshUniform *kmesh_coarse_in,
                                    const unsigned int nonanalytic_in,
                                    const bool selfenergy_offdiagonal_in)
{
    int i;
    const auto ns = dynamical->neval;
    const auto Tmin = system->Tmin;
    const auto Tmax = system->Tmax;
    const auto dT = system->dT;
    std::ofstream ofs_dymat;
    // auto file_dymat = input->job_title + ".scph_dymat";
    auto file_dymat = filename_dymat;

    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;

    ofs_dymat.open(file_dymat.c_str(), std::ios::out);

    if (!ofs_dymat) {
        exit("store_scph_dymat_to_file",
             "Cannot open scph_dymat file");
    }
    for (i = 0; i < 3; ++i) {
        ofs_dymat << std::setw(5) << kmesh_coarse_in->nk_i[i];
    }
    ofs_dymat << '\n';
    for (i = 0; i < 3; ++i) {
        ofs_dymat << std::setw(5) << kmesh_dense_in->nk_i[i];
    }
    ofs_dymat << '\n';
    ofs_dymat << std::setw(10) << Tmin;
    ofs_dymat << std::setw(10) << Tmax;
    ofs_dymat << std::setw(10) << dT << '\n';
    ofs_dymat << std::setw(5) << nonanalytic_in;
    ofs_dymat << std::setw(5) << selfenergy_offdiagonal_in << '\n';

    for (auto iT = 0; iT < NT; ++iT) {
        const auto temp = Tmin + static_cast<double>(iT) * dT;
        ofs_dymat << "# " << temp << '\n';
        for (auto is = 0; is < ns; ++is) {
            for (auto js = 0; js < ns; ++js) {
                for (auto ik = 0; ik < kmesh_coarse_in->nk; ++ik) {
                    ofs_dymat << std::setprecision(15)
                              << std::setw(25) << dymat_in[iT][is][js][ik].real();
                    ofs_dymat << std::setprecision(15)
                              << std::setw(25) << dymat_in[iT][is][js][ik].imag();
                    ofs_dymat << '\n';
                }
            }
        }
    }
    ofs_dymat.close();
    std::cout << "  " << std::setw(input->job_title.length() + 12) << std::left << file_dymat;
    std::cout << " : Anharmonic dynamical matrix (restart file)\n";
}


void Scph::zerofill_harmonic_dymat_renormalize(std::complex<double> ****delta_harmonic_dymat_renormalize,
                                               unsigned int NT)
{
    const auto ns = dynamical->neval;
    static auto complex_zero = std::complex<double>(0.0, 0.0);
    int iT, is1, is2, ik;

    for (iT = 0; iT < NT; iT++) {
        for (is1 = 0; is1 < ns; is1++) {
            for (is2 = 0; is2 < ns; is2++) {
                for (ik = 0; ik < kmesh_coarse->nk; ik++) {
                    delta_harmonic_dymat_renormalize[iT][is1][is2][ik] = complex_zero;
                }
            }
        }
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

    std::vector<double> vec_temp;

    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;

    // Compute matrix element of 4-phonon interaction

    allocate(omega2_anharm, NT, nk, ns);
    allocate(evec_anharm_tmp, nk, ns, ns);
    allocate(v4_array_all, nk_irred_interpolate * nk,
             ns * ns, ns * ns);

    // delta_v2_renorm is zero when structural optimization is not performed
    allocate(delta_v2_renorm, nk_interpolate, ns * ns);
    for (ik = 0; ik < nk_interpolate; ik++) {
        for (is = 0; is < ns * ns; is++) {
            delta_v2_renorm[ik][is] = 0.0;
        }
    }

    const auto relax_str = relaxation->relax_str;

    // Calculate v4 array.
    // This operation is the most expensive part of the calculation.
    if (selfenergy_offdiagonal & (ialgo == 1)) {
        compute_V4_elements_mpi_over_band(v4_array_all,
                                          omega2_harmonic,
                                          evec_harmonic,
                                          selfenergy_offdiagonal,
                                          kmesh_coarse,
                                          kmesh_dense,
                                          kmap_interpolate_to_scph,
                                          phase_factor_scph,
                                          phi4_reciprocal);
    } else {
        compute_V4_elements_mpi_over_kpoint(v4_array_all,
                                            omega2_harmonic,
                                            evec_harmonic,
                                            selfenergy_offdiagonal,
                                            relax_str,
                                            kmesh_coarse,
                                            kmesh_dense,
                                            kmap_interpolate_to_scph,
                                            phase_factor_scph,
                                            phi4_reciprocal);
    }

    if (relax_str) {
        allocate(v3_array_all, nk, ns, ns * ns);

        compute_V3_elements_mpi_over_kpoint(v3_array_all,
                                            omega2_harmonic,
                                            evec_harmonic,
                                            selfenergy_offdiagonal,
                                            kmesh_coarse,
                                            kmesh_dense,
                                            phase_factor_scph,
                                            phi3_reciprocal);
    }

    if (mympi->my_rank == 0) {

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

        for (double temp: vec_temp) {
            auto iT = static_cast<unsigned int>((temp - Tmin) / dT);

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
//
//            compute_anharmonic_frequency2(v4_array_all,
//                                          omega2_anharm[iT],
//                                          evec_anharm_tmp,
//                                          temp,
//                                          converged_prev,
//                                          cmat_convert,
//                                          selfenergy_offdiagonal,
//                                          writes->getVerbosity());

            dynamical->calc_new_dymat_with_evec(dymat_anharm[iT],
                                                omega2_anharm[iT],
                                                evec_anharm_tmp,
                                                kmesh_coarse,
                                                kmap_interpolate_to_scph);

            if (!warmstart_scph) converged_prev = false;
        }

        deallocate(cmat_convert);

    }

    mpi_bcast_complex(dymat_anharm, NT, kmesh_coarse->nk, ns);

    deallocate(omega2_anharm);
    deallocate(v4_array_all);
    deallocate(evec_anharm_tmp);
    deallocate(delta_v2_renorm);
    if (relax_str) {
        deallocate(v3_array_all);
    }
}


// relax internal coordinate and lattice
void Scph::exec_scph_relax_cell_coordinate_main(std::complex<double> ****dymat_anharm,
                                                std::complex<double> ****delta_harmonic_dymat_renormalize)
{
    using namespace Eigen;

    int ik, is, js;
    int is1, is2;
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
    std::complex<double> ***v4_ref, ***v4_renorm, ***v4_with_umn;
    double v0_ref, v0_renorm, v0_with_umn;
    v0_ref = 0.0; // set original ground state energy as zero

    // elastic constants
    double *C1_array;
    double **C2_array;
    double ***C3_array;

    // strain-derivative of k-space IFCs
    // (calculated by real-space IFC renormalization or finite-difference method)
    std::complex<double> **del_v1_del_umn;
    std::complex<double> **del2_v1_del_umn2;
    std::complex<double> **del3_v1_del_umn3;
    std::complex<double> ***del_v2_del_umn;
    std::complex<double> ***del2_v2_del_umn2;
    std::complex<double> ****del_v3_del_umn;

    std::complex<double> *del_v0_del_umn_renorm;

    // atomic forces and stress tensor at finite temperatures
    std::complex<double> *v1_SCP;
    std::complex<double> *del_v0_del_umn_SCP;

    // structure optimization
    int i_str_loop, i_temp_loop;
    double *q0, *u0;
    double **u_tensor, **eta_tensor;

    // structure update
    double du0;
    double du_tensor;
    double *delta_q0, *delta_u0;
    double *delta_umn;
    std::vector<int> harm_optical_modes(ns - 3);

    // cell optimization
    double pvcell = 0.0; // pressure * v_{cell,reference} [Ry]
    pvcell = relaxation->stat_pressure * system->get_primcell().volume * std::pow(Bohr_in_Angstrom, 3) *
             1.0e-30; // in 10^9 J = GJ
    pvcell *= 1.0e9 / Ryd; // in Ry

    const auto relax_str = relaxation->relax_str;

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

    allocate(q0, ns);
    allocate(u0, ns);
    allocate(u_tensor, 3, 3);
    allocate(eta_tensor, 3, 3);

    allocate(delta_q0, ns);
    allocate(delta_u0, ns);
    allocate(delta_umn, 6);

    allocate(v1_SCP, ns);
    allocate(del_v0_del_umn_renorm, 9);
    allocate(del_v0_del_umn_SCP, 9);

    allocate(v4_ref, nk_irred_interpolate * kmesh_dense->nk,
             ns * ns, ns * ns);
    allocate(v4_renorm, nk_irred_interpolate * kmesh_dense->nk,
             ns * ns, ns * ns);
    allocate(v4_with_umn, nk_irred_interpolate * kmesh_dense->nk,
             ns * ns, ns * ns);

    // initialize optimizer
    if (relaxation->relax_str == 1 & relaxation->relax_algo == 2){
        relaxation->optimizer = new Newton_Optimizer(relaxation->mixbeta_coord);
    }
    else if (relaxation->relax_str == 2 & relaxation->relax_algo == 2){
        relaxation->optimizer = new CellCoord_Newton_Optimizer(relaxation->mixbeta_cell, relaxation->mixbeta_coord);
        std::cout << "typeid(*optimizer).name(): " << typeid(*(relaxation->optimizer)).name() << std::endl;
    }
    else if (relaxation->relax_str == 1 & relaxation->relax_algo == 3){
        Eigen::MatrixXd H_init = Eigen::MatrixXd::Identity(ns-3, ns-3);
        relaxation->optimizer = new FarkasIII_Optimizer(6, H_init);
    }
    else if (relaxation->relax_str == 2 & relaxation->relax_algo == 3){
        Eigen::MatrixXd H_init = Eigen::MatrixXd::Identity(ns+3, ns+3);
        relaxation->optimizer = new FarkasIII_Optimizer(6, H_init);
    }

    // Compute matrix element of 4-phonon interaction
    // This operation is the most expensive part of the calculation.
    if (selfenergy_offdiagonal & (ialgo == 1)) {

        compute_V4_elements_mpi_over_band(v4_ref,
                                          omega2_harmonic,
                                          evec_harmonic,
                                          selfenergy_offdiagonal,
                                          kmesh_coarse,
                                          kmesh_dense,
                                          kmap_interpolate_to_scph,
                                          phase_factor_scph,
                                          phi4_reciprocal);
    } else {
        compute_V4_elements_mpi_over_kpoint(v4_ref,
                                            omega2_harmonic,
                                            evec_harmonic,
                                            selfenergy_offdiagonal,
                                            relax_str,
                                            kmesh_coarse,
                                            kmesh_dense,
                                            kmap_interpolate_to_scph,
                                            phase_factor_scph,
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
                                        phase_factor_scph,
                                        phi3_reciprocal);

    // assume that the atomic forces are zero at initial structure
    for (is = 0; is < ns; is++) {
        v1_ref[is] = 0.0;
    }
    // compute IFC renormalization by lattice relaxation
    std::cout << " RELAX_STR = " << relax_str << ": ";
    if (relax_str == 1) {
        std::cout << "Set zeros in derivatives of k-space IFCs by strain.\n\n";
    }
    if (relax_str == 2) {
        std::cout << "Calculating derivatives of k-space IFCs by strain.\n\n";
    }

    allocate(del_v1_del_umn, 9, ns);
    allocate(del2_v1_del_umn2, 81, ns);
    allocate(del3_v1_del_umn3, 729, ns);
    allocate(del_v2_del_umn, 9, nk, ns * ns);
    allocate(del2_v2_del_umn2, 81, nk, ns * ns);
    allocate(del_v3_del_umn, 9, nk, ns, ns * ns);

    relaxation->compute_del_v_strain(kmesh_coarse, kmesh_dense,
                                     del_v1_del_umn, del2_v1_del_umn2, del3_v1_del_umn3,
                                     del_v2_del_umn, del2_v2_del_umn2, del_v3_del_umn,
                                     omega2_harmonic,
                                     evec_harmonic, relax_str,
                                     mindist_list_scph,
                                     phase_factor_scph);


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
        exit("exec_scph_relax_cell_coordinate_main",
             "The number of detected optical modes is not ns-3.");
    }

    if (mympi->my_rank == 0) {

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
        if (relax_str == 2) {
            fout_step_u_tensor.open("step_u_tensor.txt");
            fout_u_tensor.open(input->job_title + ".umn_tensor");
        }

        relaxation->write_resfile_header(fout_q0, fout_u0, fout_u_tensor);

        i_temp_loop = -1;

        std::cout << " Start structural optimization.\n";

        if (relax_str == 1) {
            std::cout << "  Internal coordinates are relaxed.\n";
            std::cout << "  Shape of the unit cell is fixed.\n\n";
        } else if (relax_str == 2) {
            std::cout << "  Internal coordinates and shape of the unit cell are relaxed.\n\n";
        }

        for (double temp: vec_temp) {
            i_temp_loop++;
            auto iT = static_cast<unsigned int>((temp - Tmin) / dT);

            std::cout << " ----------------------------------------------------------------\n";
            std::cout << " Temperature = " << temp << " K\n";
            std::cout << " Temperature index : " << std::setw(4) << i_temp_loop << "/" << std::setw(4) << NT
                      << "\n\n";

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

            relaxation->set_init_structure_atT(q0, u_tensor, u0,
                                               converged_prev, str_diverged,
                                               i_temp_loop, omega2_harmonic, evec_harmonic);


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

            if (relax_str == 2) {
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

            relaxation->write_stepresfile(q0, u_tensor, u0, 0,
                                          fout_step_q0, fout_step_u0, fout_step_u_tensor);

            std::cout << " ----------------------------------------------------------------\n";

            std::cout << " Start structural optimization at " << temp << " K.";

            for (i_str_loop = 0; i_str_loop < relaxation->max_str_iter; i_str_loop++) {

                std::cout << "\n\n Structure loop :" << std::setw(5) << i_str_loop + 1;

                // get eta tensor
                relaxation->calculate_eta_tensor(eta_tensor, u_tensor);

                // calculate IFCs under strain
                relaxation->renormalize_v0_from_umn(v0_with_umn, v0_ref, eta_tensor,
                                                    C1_array, C2_array, C3_array, u_tensor, pvcell);

                relaxation->renormalize_v1_from_umn(v1_with_umn, v1_ref,
                                                    del_v1_del_umn, del2_v1_del_umn2, del3_v1_del_umn3,
                                                    u_tensor);

                relaxation->renormalize_v2_from_umn(kmesh_coarse, kmesh_dense, kmap_interpolate_to_scph,
                                                    delta_v2_with_umn, del_v2_del_umn, del2_v2_del_umn2, u_tensor);
                relaxation->renormalize_v3_from_umn(kmesh_coarse, kmesh_dense, v3_with_umn, v3_ref, del_v3_del_umn,
                                                    u_tensor);

                for (ik = 0; ik < nk_irred_interpolate * nk; ik++) {
                    for (is = 0; is < ns * ns; is++) {
                        for (is1 = 0; is1 < ns * ns; is1++) {
                            v4_with_umn[ik][is][is1] = v4_ref[ik][is][is1];
                        }
                    }
                }

                //renormalize IFC
                relaxation->renormalize_v1_from_q0(omega2_harmonic, kmesh_coarse, kmesh_dense,
                                                   v1_renorm, v1_with_umn, delta_v2_with_umn, v3_with_umn, v4_with_umn,
                                                   q0);
                relaxation->renormalize_v2_from_q0(evec_harmonic, kmesh_coarse, kmesh_dense, kmap_interpolate_to_scph,
                                                   mat_transform_sym,
                                                   delta_v2_renorm, delta_v2_with_umn, v3_with_umn, v4_with_umn, q0);
                relaxation->renormalize_v3_from_q0(kmesh_dense, kmesh_coarse, v3_renorm, v3_with_umn,
                                                   v4_with_umn, q0);
                relaxation->renormalize_v0_from_q0(omega2_harmonic, kmesh_dense,
                                                   v0_renorm, v0_with_umn, v1_with_umn, delta_v2_with_umn, v3_with_umn,
                                                   v4_with_umn,
                                                   q0);

                // calculate PES gradient by strain
                if (relax_str == 1) {
                    for (i1 = 0; i1 < 9; i1++) {
                        del_v0_del_umn_renorm[i1] = 0.0;
                    }
                } else if (relax_str == 2) {
                    calculate_del_v0_del_umn_renorm(del_v0_del_umn_renorm,
                                                    C1_array, C2_array, C3_array,
                                                    eta_tensor, u_tensor,
                                                    del_v1_del_umn, del2_v1_del_umn2, del3_v1_del_umn3,
                                                    del_v2_del_umn, del2_v2_del_umn2, del_v3_del_umn,
                                                    q0, pvcell, kmesh_dense);
                }


                // copy v4_ref to v4_renorm
                for (ik = 0; ik < nk_irred_interpolate * kmesh_dense->nk; ik++) {
                    for (is1 = 0; is1 < ns * ns; is1++) {
                        for (is2 = 0; is2 < ns * ns; is2++) {
                            v4_renorm[ik][is1][is2] = v4_ref[ik][is1][is2];
                        }
                    }
                }

                // solve SCP equation
                compute_anharmonic_frequency(v4_renorm,
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
                                                    kmap_interpolate_to_scph);

                // calculate SCP force
                compute_anharmonic_v1_array(v1_SCP, v1_renorm, v3_renorm, cmat_convert, omega2_anharm[iT], temp,
                                            kmesh_dense);

                // calculate SCP stress tensor
                if (relax_str == 1) {
                    for (i1 = 0; i1 < 9; i1++) {
                        del_v0_del_umn_SCP[i1] = 0.0;
                    }
                } else if (relax_str == 2) {
                    compute_anharmonic_del_v0_del_umn(del_v0_del_umn_SCP,
                                                      del_v0_del_umn_renorm, del_v2_del_umn, del2_v2_del_umn2,
                                                      del_v3_del_umn,
                                                      u_tensor, q0, cmat_convert, omega2_anharm[iT], temp,
                                                      kmesh_dense);
                }

                relaxation->update_cell_coordinate(q0, u0, u_tensor,
                                                   v1_SCP, omega2_anharm[iT],
                                                   del_v0_del_umn_SCP, C2_array,
                                                   cmat_convert,
                                                   harm_optical_modes,
                                                   delta_q0, delta_u0, delta_umn,
                                                   du0, du_tensor,
                                                   omega2_harmonic,
                                                   evec_harmonic);

                relaxation->write_stepresfile(q0, u_tensor, u0, i_str_loop + 1,
                                              fout_step_q0, fout_step_u0, fout_step_u_tensor);

                relaxation->check_str_divergence(str_diverged,
                                                 q0, u0, u_tensor);

                if (str_diverged) {
                    converged_prev = false;
                    std::cout << " The crystal structure diverged.";
                    std::cout << " Break from the structure loop.\n";
                    break;
                }

                // check convergence
                std::cout << " du0 =" << std::scientific << std::setw(15) << std::setprecision(6) << du0 << " [Bohr]";

                std::cout << " du_tensor =" << std::scientific << std::setw(15) << std::setprecision(6) << du_tensor
                          << '\n';

                if (du0 < relaxation->coord_conv_tol && du_tensor < relaxation->cell_conv_tol) {
                    std::cout << "\n\n du0 is smaller than COORD_CONV_TOL = " << std::scientific << std::setw(15)
                              << std::setprecision(6) << relaxation->coord_conv_tol << '\n';
                    if (relax_str == 2) {
                        std::cout << " du_tensor is smaller than CELL_CONV_TOL = " << std::scientific << std::setw(15)
                                  << std::setprecision(6) << relaxation->cell_conv_tol << '\n';
                    }
                    std::cout << " Structural optimization converged in " << i_str_loop + 1 << "-th loop.\n";
                    std::cout << " break structural loop.\n\n";
                    break;
                }

            }// close structure loop

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

            if (relax_str == 2) {
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

            relaxation->write_resfile_atT(q0, u_tensor, u0, temp, fout_q0, fout_u0, fout_u_tensor);

            if (!warmstart_scph) converged_prev = false;

            // get renormalization of harmonic dymat
            dynamical->compute_renormalized_harmonic_frequency(omega2_harm_renorm[iT],
                                                               evec_harm_renorm_tmp,
                                                               delta_v2_renorm,
                                                               omega2_harmonic,
                                                               evec_harmonic,
                                                               kmesh_coarse,
                                                               kmesh_dense,
                                                               kmap_interpolate_to_scph,
                                                               mat_transform_sym,
                                                               mindist_list_scph,
                                                               writes->getVerbosity());

            dynamical->calc_new_dymat_with_evec(delta_harmonic_dymat_renormalize[iT],
                                                omega2_harm_renorm[iT],
                                                evec_harm_renorm_tmp,
                                                kmesh_coarse,
                                                kmap_interpolate_to_scph);

        } // close temperature loop

        // output files of structural optimization
        fout_step_q0.close();
        fout_step_u0.close();
        fout_q0.close();
        fout_u0.close();

        if (relax_str == 2) {
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
    deallocate(v4_renorm);
    deallocate(v4_with_umn);


    deallocate(del_v1_del_umn);
    deallocate(del2_v1_del_umn2);
    deallocate(del3_v1_del_umn3);
    deallocate(del_v2_del_umn);
    deallocate(del2_v2_del_umn2);
    deallocate(del_v3_del_umn);

    deallocate(del_v0_del_umn_renorm);
    deallocate(v1_SCP);
    deallocate(del_v0_del_umn_SCP);

    deallocate(q0);
    deallocate(u0);
    deallocate(u_tensor);
    deallocate(eta_tensor);

    deallocate(delta_q0);
    deallocate(delta_u0);
    deallocate(delta_umn);
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

void Scph::compute_V3_elements_mpi_over_kpoint(std::complex<double> ***v3_out,
                                               double **omega2_harmonic_in,
                                               const std::complex<double> *const *const *evec_in,
                                               const bool self_offdiag,
                                               const KpointMeshUniform *kmesh_coarse_in,
                                               const KpointMeshUniform *kmesh_dense_in,
                                               const PhaseFactorStorage *phase_storage_in,
                                               std::complex<double> *phi3_reciprocal_inout)
{
    // Calculate the matrix elements of quartic terms in reciprocal space.
    // This is the most expensive part of the SCPH calculation.

    auto ns = dynamical->neval;
    auto ns2 = ns * ns;
    auto ns3 = ns * ns * ns;
    unsigned int is, js, ks;
    unsigned int **ind;
    unsigned int i, j;

    size_t js2_1, js2_2;
    size_t is2, js2, ks2;

    std::complex<double> ret;
    long int ii;

    const auto nk_scph = kmesh_dense_in->nk;
    const auto ngroup_v3 = anharmonic_core->get_ngroup_fcs(3);
    const auto factor = std::pow(0.5, 2) / static_cast<double>(nk_scph);
    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);
    std::complex<double> *v3_array_at_kpair;
    std::complex<double> ***v3_mpi;

    std::complex<double> **v3_tmp0, **v3_tmp1, **v3_tmp2, **v3_tmp3;

    if (mympi->my_rank == 0) {
        if (self_offdiag) {
            std::cout << " SELF_OFFDIAG = 1: Calculating all components of v3_array ... ";
        } else {
            std::cout << " SELF_OFFDIAG = 0: Calculating diagonal components of v3_array ... ";
        }
    }

    allocate(v3_array_at_kpair, ngroup_v3);
    allocate(ind, ngroup_v3, 3);
    allocate(v3_mpi, nk_scph, ns, ns2);

    allocate(v3_tmp0, ns, ns2);
    allocate(v3_tmp1, ns, ns2);
    allocate(v3_tmp2, ns, ns2);
    allocate(v3_tmp3, ns, ns2);

    for (unsigned int ik = mympi->my_rank; ik < nk_scph; ik += mympi->nprocs) {

        anharmonic_core->calc_phi3_reciprocal(kmesh_dense_in->xk[ik],
                                              kmesh_dense_in->xk[kmesh_dense_in->kindex_minus_xk[ik]],
                                              anharmonic_core->get_ngroup_fcs(3),
                                              anharmonic_core->get_fcs_group(3),
                                              anharmonic_core->get_relvec(3),
                                              phase_storage_in,
                                              phi3_reciprocal_inout);

#pragma omp parallel for private(j)
        for (ii = 0; ii < ngroup_v3; ++ii) {
            v3_array_at_kpair[ii] = phi3_reciprocal_inout[ii] * anharmonic_core->get_invmass_factor(3)[ii];
            for (j = 0; j < 3; ++j) ind[ii][j] = anharmonic_core->get_evec_index(3)[ii][j];
        }

#pragma omp parallel for private(is)
        for (ii = 0; ii < ns; ++ii) {
            for (is = 0; is < ns2; ++is) {
                v3_mpi[ik][ii][is] = complex_zero;
                v3_out[ik][ii][is] = complex_zero;
            }
        }

        if (self_offdiag) {

            // All matrix elements will be calculated when considering the off-diagonal
            // elements of the phonon self-energy (i.e., when considering polarization mixing).

            // initialize temporary matrices
#pragma omp parallel for private(js)
            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns2; ++js) {
                    v3_tmp0[is][js] = complex_zero;
                    v3_tmp1[is][js] = complex_zero;
                    v3_tmp2[is][js] = complex_zero;
                    v3_tmp3[is][js] = complex_zero;
                }
            }

            // copy v3 in (alpha,mu) representation to the temporary matrix
#pragma omp parallel for private(is, js)
            for (ii = 0; ii < ngroup_v3; ++ii) {

                is = ind[ii][0];
                js = ind[ii][1] * ns + ind[ii][2];
                v3_tmp0[is][js] = v3_array_at_kpair[ii];
            }

            // transform the first index
#pragma omp parallel for private(js2_1, is, js, ks, js2_2, is2, js2, ks2)
            for (ii = 0; ii < ns3; ++ii) {
                is = ii / ns2;
                js2_1 = ii % ns2;

                for (is2 = 0; is2 < ns; ++is2) {
                    v3_tmp1[is][js2_1] += v3_tmp0[is2][js2_1]
                                          * evec_in[0][is][is2];
                }
            }

            // transform the second index
#pragma omp parallel for private(js2_1, is, js, ks, js2_2, is2, js2, ks2)
            for (ii = 0; ii < ns3; ++ii) {
                is = ii / ns2;
                js2_1 = ii % ns2;
                js = js2_1 / ns; // second index
                ks = js2_1 % ns; // third index

                for (js2 = 0; js2 < ns; ++js2) {
                    js2_2 = js2 * ns + ks;
                    v3_tmp2[is][js2_1] += v3_tmp1[is][js2_2]
                                          * evec_in[ik][js][js2];
                }
            }

            // transform the third index
#pragma omp parallel for private(js2_1, is, js, ks, js2_2, is2, js2, ks2)
            for (ii = 0; ii < ns3; ++ii) {
                is = ii / ns2;
                js2_1 = ii % ns2;
                js = js2_1 / ns; // third index
                ks = js2_1 % ns; // fourth index

                for (ks2 = 0; ks2 < ns; ++ks2) {
                    js2_2 = js * ns + ks2;
                    v3_tmp3[is][js2_1] += v3_tmp2[is][js2_2]
                                          * std::conj(evec_in[ik][ks][ks2]);
                }
            }

            // copy to the final matrix
#pragma omp parallel for private(is, js2_1)
            for (ii = 0; ii < ns3; ++ii) {
                is = ii / ns2;
                js2_1 = ii % ns2;

                v3_mpi[ik][is][js2_1] = factor * v3_tmp3[is][js2_1];
            }

        } else {

            // Only diagonal elements will be computed when neglecting the polarization mixing.

            if (ik == 0) {
#pragma omp parallel for private(is, js, ks, ret, i)
                for (ii = 0; ii < ns3; ++ii) {
                    is = ii / ns2;
                    js = (ii - ns2 * is) / ns;
                    ks = ii % ns;

                    ret = std::complex<double>(0.0, 0.0);

                    for (i = 0; i < ngroup_v3; ++i) {

                        ret += v3_array_at_kpair[i]
                               * evec_in[0][is][ind[i][0]]
                               * evec_in[ik][js][ind[i][1]]
                               * std::conj(evec_in[ik][ks][ind[i][2]]);
                    }

                    v3_mpi[ik][is][ns * js + ks] = factor * ret;
                }
            } else {

#pragma omp parallel for private(is, js, ret, i)
                for (ii = 0; ii < ns2; ++ii) {
                    is = ii / ns;
                    js = ii % ns;

                    ret = std::complex<double>(0.0, 0.0);

                    for (i = 0; i < ngroup_v3; ++i) {

                        ret += v3_array_at_kpair[i]
                               * evec_in[0][is][ind[i][0]]
                               * evec_in[ik][js][ind[i][1]]
                               * std::conj(evec_in[ik][js][ind[i][2]]);
                    }

                    v3_mpi[ik][is][(ns + 1) * js] = factor * ret;
                }
            }
        }
    }

    deallocate(v3_array_at_kpair);
    deallocate(ind);
#ifdef MPI_CXX_DOUBLE_COMPLEX
    MPI_Allreduce(&v3_mpi[0][0][0], &v3_out[0][0][0],
                  static_cast<int>(nk_scph) * ns3,
                  MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
#else
                                                                                                                            MPI_Allreduce(&v3_mpi[0][0][0], &v3_out[0][0][0], static_cast<int>(nk_scph) * ns3,
                  MPI_COMPLEX16, MPI_SUM, MPI_COMM_WORLD);
#endif

    deallocate(v3_mpi);
    deallocate(v3_tmp0);
    deallocate(v3_tmp1);
    deallocate(v3_tmp2);
    deallocate(v3_tmp3);


    zerofill_elements_acoustic_at_gamma(omega2_harmonic_in, v3_out, 3,
                                        kmesh_dense_in->nk, kmesh_coarse_in->nk_irred);

    if (mympi->my_rank == 0) {
        std::cout << " done !\n";
        timer->print_elapsed();
    }
}

// This function should be merged with void Scph::compute_V3_elements_mpi_over_kpoint
// after merged with dev2.0 because the implementation is redundant.
void Scph::compute_V3_elements_for_given_IFCs(std::complex<double> ***v3_out,
                                              double **omega2_harmonic_in,
                                              const int ngroup_v3_in,
                                              std::vector<double> *fcs_group_v3_in,
                                              std::vector<RelativeVector> *relvec_v3_in,
                                              double *invmass_v3_in,
                                              int **evec_index_v3_in,
                                              const std::complex<double> *const *const *evec_in,
                                              const bool self_offdiag,
                                              const KpointMeshUniform *kmesh_coarse_in,
                                              const KpointMeshUniform *kmesh_dense_in,
                                              const PhaseFactorStorage *phase_storage_in)
{
    auto ns = dynamical->neval;
    auto ns2 = ns * ns;
    auto ns3 = ns * ns * ns;
    unsigned int is, js, ks;
    unsigned int **ind;
    unsigned int i, j;
    size_t js2_1, js2_2;
    size_t is2, js2, ks2;

    std::complex<double> ret;
    long int ii;

    const auto nk_scph = kmesh_dense_in->nk;
    const auto factor = std::pow(0.5, 2) / static_cast<double>(nk_scph);
    static auto complex_zero = std::complex<double>(0.0, 0.0);
    std::complex<double> *v3_array_at_kpair;
    std::complex<double> ***v3_mpi;
    std::complex<double> *phi3_reciprocal_tmp;

    std::complex<double> **v3_tmp0, **v3_tmp1, **v3_tmp2, **v3_tmp3;

    allocate(phi3_reciprocal_tmp, ngroup_v3_in);
    allocate(v3_array_at_kpair, ngroup_v3_in);
    allocate(ind, ngroup_v3_in, 3);
    allocate(v3_mpi, nk_scph, ns, ns2);

    allocate(v3_tmp0, ns, ns2);
    allocate(v3_tmp1, ns, ns2);
    allocate(v3_tmp2, ns, ns2);
    allocate(v3_tmp3, ns, ns2);

    for (unsigned int ik = mympi->my_rank; ik < nk_scph; ik += mympi->nprocs) {

        anharmonic_core->calc_phi3_reciprocal(kmesh_dense_in->xk[ik],
                                              kmesh_dense_in->xk[kmesh_dense_in->kindex_minus_xk[ik]],
                                              ngroup_v3_in,
                                              fcs_group_v3_in,
                                              relvec_v3_in,
                                              phase_storage_in,
                                              phi3_reciprocal_tmp);

#ifdef _OPENMP
#pragma omp parallel for private(j)
#endif
        for (ii = 0; ii < ngroup_v3_in; ++ii) {
            v3_array_at_kpair[ii] = phi3_reciprocal_tmp[ii] * invmass_v3_in[ii];
            for (j = 0; j < 3; ++j) ind[ii][j] = evec_index_v3_in[ii][j];
        }

#pragma omp parallel for private(is)
        for (ii = 0; ii < ns; ++ii) {
            for (is = 0; is < ns2; ++is) {
                v3_mpi[ik][ii][is] = complex_zero;
                v3_out[ik][ii][is] = complex_zero;
            }
        }

        if (self_offdiag) {

            // All matrix elements will be calculated when considering the off-diagonal
            // elements of the phonon self-energy (i.e., when considering polarization mixing).

            // initialize temporary matrices
#pragma omp parallel for private(js)
            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns2; ++js) {
                    v3_tmp0[is][js] = complex_zero;
                    v3_tmp1[is][js] = complex_zero;
                    v3_tmp2[is][js] = complex_zero;
                    v3_tmp3[is][js] = complex_zero;
                }
            }

            // copy v3 in (alpha,mu) representation to the temporary matrix
#pragma omp parallel for private(is, js)
            for (ii = 0; ii < ngroup_v3_in; ++ii) {

                is = ind[ii][0];
                js = ind[ii][1] * ns + ind[ii][2];
                v3_tmp0[is][js] = v3_array_at_kpair[ii];
            }

            // transform the first index
#pragma omp parallel for private(js2_1, is, js, ks, js2_2, is2, js2, ks2)
            for (ii = 0; ii < ns3; ++ii) {
                is = ii / ns2;
                js2_1 = ii % ns2;

                for (is2 = 0; is2 < ns; ++is2) {
                    v3_tmp1[is][js2_1] += v3_tmp0[is2][js2_1]
                                          * evec_in[0][is][is2];
                }
            }

            // transform the second index
#pragma omp parallel for private(js2_1, is, js, ks, js2_2, is2, js2, ks2)
            for (ii = 0; ii < ns3; ++ii) {
                is = ii / ns2;
                js2_1 = ii % ns2;
                js = js2_1 / ns; // second index
                ks = js2_1 % ns; // third index

                for (js2 = 0; js2 < ns; ++js2) {
                    js2_2 = js2 * ns + ks;
                    v3_tmp2[is][js2_1] += v3_tmp1[is][js2_2]
                                          * evec_in[ik][js][js2];
                }
            }

            // transform the third index
#pragma omp parallel for private(js2_1, is, js, ks, js2_2, is2, js2, ks2)
            for (ii = 0; ii < ns3; ++ii) {
                is = ii / ns2;
                js2_1 = ii % ns2;
                js = js2_1 / ns; // third index
                ks = js2_1 % ns; // fourth index

                for (ks2 = 0; ks2 < ns; ++ks2) {
                    js2_2 = js * ns + ks2;
                    v3_tmp3[is][js2_1] += v3_tmp2[is][js2_2]
                                          * std::conj(evec_in[ik][ks][ks2]);
                }
            }

            // copy to the final matrix
#pragma omp parallel for private(is, js2_1)
            for (ii = 0; ii < ns3; ++ii) {
                is = ii / ns2;
                js2_1 = ii % ns2;

                v3_mpi[ik][is][js2_1] = factor * v3_tmp3[is][js2_1];
            }

        } else {

            // Only diagonal elements will be computed when neglecting the polarization mixing.

            if (ik == 0) {
#pragma omp parallel for private(is, js, ks, ret, i)
                for (ii = 0; ii < ns3; ++ii) {
                    is = ii / ns2;
                    js = (ii - ns2 * is) / ns;
                    ks = ii % ns;

                    ret = std::complex<double>(0.0, 0.0);

                    for (i = 0; i < ngroup_v3_in; ++i) {

                        ret += v3_array_at_kpair[i]
                               * evec_in[0][is][ind[i][0]]
                               * evec_in[ik][js][ind[i][1]]
                               * std::conj(evec_in[ik][ks][ind[i][2]]);
                    }

                    v3_mpi[ik][is][ns * js + ks] = factor * ret;
                }
            } else {

#pragma omp parallel for private(is, js, ret, i)
                for (ii = 0; ii < ns2; ++ii) {
                    is = ii / ns;
                    js = ii % ns;

                    ret = std::complex<double>(0.0, 0.0);

                    for (i = 0; i < ngroup_v3_in; ++i) {

                        ret += v3_array_at_kpair[i]
                               * evec_in[0][is][ind[i][0]]
                               * evec_in[ik][js][ind[i][1]]
                               * std::conj(evec_in[ik][js][ind[i][2]]);
                    }

                    v3_mpi[ik][is][(ns + 1) * js] = factor * ret;
                }
            }
        }
    }

    deallocate(v3_array_at_kpair);
    deallocate(ind);
#ifdef MPI_CXX_DOUBLE_COMPLEX
    MPI_Allreduce(&v3_mpi[0][0][0], &v3_out[0][0][0],
                  static_cast<int>(nk_scph) * ns3,
                  MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
#else
                                                                                                                            MPI_Allreduce(&v3_mpi[0][0][0], &v3_out[0][0][0], static_cast<int>(nk_scph) * ns3,
                  MPI_COMPLEX16, MPI_SUM, MPI_COMM_WORLD);
#endif

    deallocate(v3_mpi);
    deallocate(v3_tmp0);
    deallocate(v3_tmp1);
    deallocate(v3_tmp2);
    deallocate(v3_tmp3);

    zerofill_elements_acoustic_at_gamma(omega2_harmonic_in, v3_out, 3,
                                        kmesh_dense_in->nk, kmesh_coarse_in->nk_irred);
}


void Scph::compute_V4_elements_mpi_over_kpoint(std::complex<double> ***v4_out,
                                               double **omega2_harmonic_in,
                                               std::complex<double> ***evec_in,
                                               const bool self_offdiag,
                                               const bool relax,
                                               const KpointMeshUniform *kmesh_coarse_in,
                                               const KpointMeshUniform *kmesh_dense_in,
                                               const std::vector<int> &kmap_coarse_to_dense,
                                               const PhaseFactorStorage *phase_storage_in,
                                               std::complex<double> *phi4_reciprocal_inout)
{
    // Calculate the matrix elements of quartic terms in reciprocal space.
    // This is the most expensive part of the SCPH calculation.

    const size_t nk_reduced_interpolate = kmesh_coarse_in->nk_irred;
    const size_t ns = dynamical->neval;
    const size_t ns2 = ns * ns;
    const size_t ns3 = ns * ns * ns;
    const size_t ns4 = ns * ns * ns * ns;
    size_t is, js, ks, ls;
    size_t is2_1, js2_1, is2_2, js2_2;
    size_t is2, js2, ks2, ls2;
    unsigned int **ind;
    unsigned int j;
    long int ii;

    const auto nk_scph = kmesh_dense_in->nk;
    const auto ngroup_v4 = anharmonic_core->get_ngroup_fcs(4);
    const auto factor = std::pow(0.5, 2) / static_cast<double>(nk_scph);
    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);
    std::complex<double> *v4_array_at_kpair;
    std::complex<double> ***v4_mpi;
    std::complex<double> ***evec_conj;

    std::complex<double> **v4_tmp0, **v4_tmp1, **v4_tmp2, **v4_tmp3, **v4_tmp4;

    const size_t nk2_prod = nk_reduced_interpolate * nk_scph;

    if (mympi->my_rank == 0) {
        if (self_offdiag) {
            std::cout << " SELF_OFFDIAG = 1: Calculating all components of v4_array ... " << std::flush;
        } else {
            std::cout << " SELF_OFFDIAG = 0: Calculating diagonal components of v4_array ... " << std::flush;
        }
    }

    allocate(v4_array_at_kpair, ngroup_v4);
    allocate(ind, ngroup_v4, 4);
    allocate(v4_mpi, nk2_prod, ns2, ns2);
    allocate(evec_conj, kmesh_dense_in->nk, ns, ns);

    allocate(v4_tmp0, ns2, ns2);
    allocate(v4_tmp1, ns2, ns2);
    allocate(v4_tmp2, ns2, ns2);
    allocate(v4_tmp3, ns2, ns2);
    allocate(v4_tmp4, ns2, ns2);

    const long int nks2 = kmesh_dense_in->nk * ns2;

#pragma omp parallel for private(is, js)
    for (long int iks = 0; iks < nks2; ++iks) {
        size_t ik = iks / ns2;
        is = (iks - ik * ns2) / ns;
        js = iks % ns;
        evec_conj[ik][is][js] = std::conj(evec_in[ik][is][js]);
    }

    for (size_t ik_prod = mympi->my_rank; ik_prod < nk2_prod; ik_prod += mympi->nprocs) {
        const auto ik = ik_prod / nk_scph;
        const auto jk = ik_prod % nk_scph;

        const unsigned int knum = kmap_coarse_to_dense[kmesh_coarse_in->kpoint_irred_all[ik][0].knum];

        anharmonic_core->calc_phi4_reciprocal(kmesh_dense_in->xk[knum],
                                              kmesh_dense_in->xk[jk],
                                              kmesh_dense_in->xk[kmesh_dense_in->kindex_minus_xk[jk]],
                                              phase_storage_in,
                                              phi4_reciprocal_inout);

#pragma omp parallel for private(j)
        for (ii = 0; ii < ngroup_v4; ++ii) {
            v4_array_at_kpair[ii] = phi4_reciprocal_inout[ii] * anharmonic_core->get_invmass_factor(4)[ii];
            for (j = 0; j < 4; ++j) ind[ii][j] = anharmonic_core->get_evec_index(4)[ii][j];
        }

#pragma omp parallel for private(is, js)
        for (ii = 0; ii < ns4; ++ii) {
            is = ii / ns2;
            js = ii % ns2;
            v4_mpi[ik_prod][is][js] = complex_zero;
            v4_out[ik_prod][is][js] = complex_zero;
        }

        // initialize temporary matrices
#pragma omp parallel for private(js)
        for (is = 0; is < ns2; ++is) {
            for (js = 0; js < ns2; ++js) {
                v4_tmp0[is][js] = complex_zero;
                v4_tmp1[is][js] = complex_zero;
                v4_tmp2[is][js] = complex_zero;
                v4_tmp3[is][js] = complex_zero;
                v4_tmp4[is][js] = complex_zero;
            }
        }

        if (self_offdiag || relaxation->relax_str) {

            // All matrix elements will be calculated when considering the off-diagonal
            // elements of the phonon self-energy (loop diagram).


            // copy v4 in (alpha,mu) representation to the temporary matrix
#pragma omp parallel for private(is, js)
            for (ii = 0; ii < ngroup_v4; ++ii) {

                is = ind[ii][0] * ns + ind[ii][1];
                js = ind[ii][2] * ns + ind[ii][3];
                v4_tmp0[is][js] = v4_array_at_kpair[ii];
            }

            // transform the first index
#pragma omp parallel for private(is2_1, js2_1, is, js, ks, ls, is2_2, js2_2, is2, js2, ks2, ls2)
            for (ii = 0; ii < ns4; ++ii) {
                is2_1 = ii / ns2;
                js2_1 = ii % ns2;
                is = is2_1 / ns; // first index
                js = is2_1 % ns; // second index

                for (is2 = 0; is2 < ns; ++is2) {
                    is2_2 = is2 * ns + js;
                    v4_tmp1[is2_1][js2_1] += v4_tmp0[is2_2][js2_1]
                                             * evec_conj[knum][is][is2];
                }
            }
            // transform the second index
#pragma omp parallel for private(is2_1, js2_1, is, js, ks, ls, is2_2, js2_2, is2, js2, ks2, ls2)
            for (ii = 0; ii < ns4; ++ii) {
                is2_1 = ii / ns2;
                js2_1 = ii % ns2;
                is = is2_1 / ns; // first index
                js = is2_1 % ns; // second index

                for (js2 = 0; js2 < ns; ++js2) {
                    is2_2 = is * ns + js2;
                    v4_tmp2[is2_1][js2_1] += v4_tmp1[is2_2][js2_1]
                                             * evec_in[knum][js][js2];
                }
            }
            // transform the third index
#pragma omp parallel for private(is2_1, js2_1, is, js, ks, ls, is2_2, js2_2, is2, js2, ks2, ls2)
            for (ii = 0; ii < ns4; ++ii) {
                is2_1 = ii / ns2;
                js2_1 = ii % ns2;
                ks = js2_1 / ns; // third index
                ls = js2_1 % ns; // fourth index

                for (ks2 = 0; ks2 < ns; ++ks2) {
                    js2_2 = ks2 * ns + ls;
                    v4_tmp3[is2_1][js2_1] += v4_tmp2[is2_1][js2_2]
                                             * evec_in[jk][ks][ks2];
                }
            }

            // transform the fourth index
#pragma omp parallel for private(is2_1, js2_1, is, js, ks, ls, is2_2, js2_2, is2, js2, ks2, ls2)
            for (ii = 0; ii < ns4; ++ii) {
                is2_1 = ii / ns2;
                js2_1 = ii % ns2;
                ks = js2_1 / ns; // third index
                ls = js2_1 % ns; // fourth index

                for (ls2 = 0; ls2 < ns; ++ls2) {
                    js2_2 = ks * ns + ls2;
                    v4_tmp4[is2_1][js2_1] += v4_tmp3[is2_1][js2_2]
                                             * evec_conj[jk][ls][ls2];
                }
            }

            // copy to the final matrix
            for (ii = 0; ii < ns4; ++ii) {
                is2_1 = ii / ns2;
                js2_1 = ii % ns2;

                v4_mpi[ik_prod][is2_1][js2_1] = factor * v4_tmp4[is2_1][js2_1];
            }

        } else {

            // copy v4 in (alpha,mu) representation to the temporary matrix
#pragma omp parallel for private(is, js)
            for (ii = 0; ii < ngroup_v4; ++ii) {

                is = ind[ii][0] * ns + ind[ii][1];
                js = ind[ii][2] * ns + ind[ii][3];
                v4_tmp0[is][js] = v4_array_at_kpair[ii];
            }

            // transform the first and the second index
#pragma omp parallel for private(is, js, ks, is2_1, is2_2)
            for (ii = 0; ii < ns3; ++ii) {
                is = ii / ns2;
                is2_1 = ii % ns2;
                for (is2_2 = 0; is2_2 < ns2; ++is2_2) {
                    // is2_2 = js*ns+ks
                    js = is2_2 / ns;
                    ks = is2_2 % ns;

                    v4_tmp1[(ns + 1) * is][is2_1] += v4_tmp0[is2_2][is2_1]
                                                     * evec_conj[knum][is][js]
                                                     * evec_in[knum][is][ks];

                }
            }
#pragma omp parallel for private(is, js, ks, ls, is2_2)
            // transform the third and the fourth index
            for (is2_1 = 0; is2_1 < ns2; ++is2_1) {
                is = is2_1 / ns;
                js = is2_1 % ns;
                for (is2_2 = 0; is2_2 < ns2; ++is2_2) {
                    ks = is2_2 / ns;
                    ls = is2_2 % ns;

                    v4_tmp2[(ns + 1) * is][(ns + 1) * js] += v4_tmp1[(ns + 1) * is][is2_2]
                                                             * evec_in[jk][js][ks]
                                                             * evec_conj[jk][js][ls];

                }
            }
            // copy to the final matrix
#pragma omp parallel for private(is, js)
            for (ii = 0; ii < ns2; ++ii) {
                is = ii / ns;
                js = ii % ns;

                v4_mpi[ik_prod][(ns + 1) * is][(ns + 1) * js] = factor * v4_tmp2[(ns + 1) * is][(ns + 1) * js];
            }
        }
    }


    deallocate(evec_conj);
    deallocate(v4_array_at_kpair);
    deallocate(ind);

    deallocate(v4_tmp0);
    deallocate(v4_tmp1);
    deallocate(v4_tmp2);
    deallocate(v4_tmp3);
    deallocate(v4_tmp4);

// Now, communicate the calculated data.
// When the data count is larger than 2^31-1, split it.

    long maxsize = 1;
    maxsize = (maxsize << 31) - 1;

    const size_t count = nk2_prod * ns4;
    const size_t count_sub = ns4;

    if (count <= maxsize) {
#ifdef MPI_CXX_DOUBLE_COMPLEX
        MPI_Allreduce(&v4_mpi[0][0][0], &v4_out[0][0][0], count,
                      MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
#else
                                                                                                                                MPI_Allreduce(&v4_mpi[0][0][0], &v4_out[0][0][0], count,
                      MPI_COMPLEX16, MPI_SUM, MPI_COMM_WORLD);
#endif
    } else if (count_sub <= maxsize) {
        for (size_t ik_prod = 0; ik_prod < nk2_prod; ++ik_prod) {
#ifdef MPI_CXX_DOUBLE_COMPLEX
            MPI_Allreduce(&v4_mpi[ik_prod][0][0], &v4_out[ik_prod][0][0],
                          count_sub,
                          MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
#else
                                                                                                                                    MPI_Allreduce(&v4_mpi[ik_prod][0][0], &v4_out[ik_prod][0][0],
                          count_sub,
                          MPI_COMPLEX16, MPI_SUM, MPI_COMM_WORLD);
#endif
        }
    } else {
        for (size_t ik_prod = 0; ik_prod < nk2_prod; ++ik_prod) {
            for (is = 0; is < ns2; ++is) {
#ifdef MPI_CXX_DOUBLE_COMPLEX
                MPI_Allreduce(&v4_mpi[ik_prod][is][0], &v4_out[ik_prod][is][0],
                              ns2,
                              MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
#else
                                                                                                                                        MPI_Allreduce(&v4_mpi[ik_prod][is][0], &v4_out[ik_prod][is][0],
                              ns2,
                              MPI_COMPLEX16, MPI_SUM, MPI_COMM_WORLD);
#endif
            }
        }
    }

    deallocate(v4_mpi);

    zerofill_elements_acoustic_at_gamma(omega2_harmonic_in, v4_out, 4,
                                        kmesh_dense_in->nk, kmesh_coarse_in->nk_irred);

    if (mympi->my_rank == 0) {
        std::cout << " done !\n";
        timer->print_elapsed();
    }
}

void Scph::compute_V4_elements_mpi_over_band(std::complex<double> ***v4_out,
                                             double **omega2_harmonic_in,
                                             std::complex<double> ***evec_in,
                                             const bool self_offdiag,
                                             const KpointMeshUniform *kmesh_coarse_in,
                                             const KpointMeshUniform *kmesh_dense_in,
                                             const std::vector<int> &kmap_coarse_to_dense,
                                             const PhaseFactorStorage *phase_storage_in,
                                             std::complex<double> *phi4_reciprocal_inout)
{
    // Calculate the matrix elements of quartic terms in reciprocal space.
    // This is the most expensive part of the SCPH calculation.

    size_t ik_prod;
    const size_t nk_reduced_interpolate = kmesh_coarse_in->nk_irred;
    const size_t ns = dynamical->neval;
    const size_t ns2 = ns * ns;
    const size_t ns4 = ns * ns * ns * ns;
    int is, js, ks, ls;
    size_t is2_1, js2_1, is2_2;
    size_t is2;
    int is4_1;
    unsigned int knum;
    unsigned int **ind;
    unsigned int i, j;
    long int *nset_mpi;

    const auto nk_scph = kmesh_dense_in->nk;
    const auto ngroup_v4 = anharmonic_core->get_ngroup_fcs(4);
    auto factor = std::pow(0.5, 2) / static_cast<double>(nk_scph);
    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);
    std::complex<double> *v4_array_at_kpair;
    std::complex<double> ***v4_mpi;

    std::complex<double> **v4_tmp0, **v4_tmp1, **v4_tmp2, **v4_tmp3, **v4_tmp4;

    std::vector<int> ik_vec, jk_vec, is_vec;

    auto nk2_prod = nk_reduced_interpolate * nk_scph;

    if (mympi->my_rank == 0) {
        if (self_offdiag) {
            std::cout << " IALGO = 1 : Use different algorithm efficient when nbands >> nk_3ph\n";
            std::cout << " SELF_OFFDIAG = 1: Calculating all components of v4_array ... \n";
        } else {
            exit("compute_V4_elements_mpi_over_kpoint",
                 "This function can be used only when SELF_OFFDIAG = 1");
        }
    }

    allocate(nset_mpi, mympi->nprocs);

    const long int nset_tot = nk2_prod * ns;
    long int nset_each = nset_tot / mympi->nprocs;
    const long int nres = nset_tot - nset_each * mympi->nprocs;

    for (i = 0; i < mympi->nprocs; ++i) {
        nset_mpi[i] = nset_each;
        if (nres > i) {
            nset_mpi[i] += 1;
        }
    }

    MPI_Bcast(&nset_mpi[0], mympi->nprocs, MPI_LONG, 0, MPI_COMM_WORLD);
    long int nstart = 0;
    for (i = 0; i < mympi->my_rank; ++i) {
        nstart += nset_mpi[i];
    }
    long int nend = nstart + nset_mpi[mympi->my_rank];
    nset_each = nset_mpi[mympi->my_rank];
    deallocate(nset_mpi);

    ik_vec.clear();
    jk_vec.clear();
    is_vec.clear();

    long int icount = 0;
    for (ik_prod = 0; ik_prod < nk2_prod; ++ik_prod) {
        for (is = 0; is < ns; ++is) {
            // if (is < js && relax_str == 0) continue;

            if (icount >= nstart && icount < nend) {
                ik_vec.push_back(ik_prod / nk_scph);
                jk_vec.push_back(ik_prod % nk_scph);
                is_vec.push_back(is);
            }
            ++icount;
        }
    }

    allocate(v4_array_at_kpair, ngroup_v4);
    allocate(ind, ngroup_v4, 4);
    allocate(v4_mpi, nk2_prod, ns2, ns2);
    allocate(v4_tmp0, ns2, ns2);
    allocate(v4_tmp1, ns, ns2);
    allocate(v4_tmp2, ns, ns2);
    allocate(v4_tmp3, ns, ns2);
    allocate(v4_tmp4, ns, ns2);

    for (ik_prod = 0; ik_prod < nk2_prod; ++ik_prod) {
#pragma omp parallel for private (js)
        for (is = 0; is < ns2; ++is) {
            for (js = 0; js < ns2; ++js) {
                v4_mpi[ik_prod][is][js] = complex_zero;
                v4_out[ik_prod][is][js] = complex_zero;
            }
        }
    }

    int ik_old = -1;
    int jk_old = -1;

    if (mympi->my_rank == 0) {
        std::cout << " Total number of sets to compute : " << nset_each << '\n';
    }

    for (long int ii = 0; ii < nset_each; ++ii) {

        auto ik_now = ik_vec[ii];
        auto jk_now = jk_vec[ii];
        auto is_now = is_vec[ii];

        if (!(ik_now == ik_old && jk_now == jk_old)) {

            // Update v4_array_at_kpair and ind

            knum = kmap_coarse_to_dense[kmesh_coarse_in->kpoint_irred_all[ik_now][0].knum];

            anharmonic_core->calc_phi4_reciprocal(kmesh_dense_in->xk[knum],
                                                  kmesh_dense_in->xk[jk_now],
                                                  kmesh_dense_in->xk[kmesh_dense_in->kindex_minus_xk[jk_now]],
                                                  phase_storage_in,
                                                  phi4_reciprocal_inout);

#ifdef _OPENMP
#pragma omp parallel for private(j)
#endif
            for (i = 0; i < ngroup_v4; ++i) {
                v4_array_at_kpair[i] = phi4_reciprocal_inout[i] * anharmonic_core->get_invmass_factor(4)[i];
                for (j = 0; j < 4; ++j) ind[i][j] = anharmonic_core->get_evec_index(4)[i][j];
            }
            ik_old = ik_now;
            jk_old = jk_now;

            for (is4_1 = 0; is4_1 < ns4; is4_1++) {
                is2_1 = is4_1 / ns2;
                js2_1 = is4_1 % ns2;
                v4_tmp0[is2_1][js2_1] = complex_zero;
            }

            for (i = 0; i < ngroup_v4; ++i) {

                is = ind[i][0] * ns + ind[i][1];
                js = ind[i][2] * ns + ind[i][3];
                v4_tmp0[is][js] = v4_array_at_kpair[i];
            }

        }

        ik_prod = ik_now * nk_scph + jk_now;
        // int is_prod = ns * is_now + js_now;


        // initialize temporary matrices
#pragma omp parallel for private(js)
        for (is = 0; is < ns; ++is) {
            for (js = 0; js < ns2; ++js) {
                v4_tmp1[is][js] = complex_zero;
                v4_tmp2[is][js] = complex_zero;
                v4_tmp3[is][js] = complex_zero;
                v4_tmp4[is][js] = complex_zero;
            }
        }

        // transform the first index
#pragma omp parallel for private(is2_1, is, js, ks, ls)
        for (is2_2 = 0; is2_2 < ns2; ++is2_2) {
            ks = is2_2 / ns;
            ls = is2_2 % ns;

            for (is2_1 = 0; is2_1 < ns2; ++is2_1) {
                is = is2_1 / ns;
                js = is2_1 % ns;

                v4_tmp1[js][is2_2] += v4_tmp0[is2_1][is2_2]
                                      * std::conj(evec_in[knum][is_now][is]);
            }
        }

        // transform the second index
#pragma omp parallel for private(is2_1, is, js, ks, ls)
        for (is2_2 = 0; is2_2 < ns2; ++is2_2) {
            ks = is2_2 / ns;
            ls = is2_2 % ns;

            for (is2_1 = 0; is2_1 < ns2; ++is2_1) {
                is = is2_1 / ns;
                js = is2_1 % ns;

                v4_tmp2[is][is2_2] += v4_tmp1[js][is2_2] * evec_in[knum][is][js];

            }
        }


        // transform the third index
#pragma omp parallel for private(is2_2, is, js, ks, ls)
        for (is2_1 = 0; is2_1 < ns2; ++is2_1) {
            is = is2_1 / ns;
            js = is2_1 % ns;

            for (is2_2 = 0; is2_2 < ns2; ++is2_2) {
                ks = is2_2 / ns;
                ls = is2_2 % ns;

                v4_tmp3[is][ks * ns + js] += v4_tmp2[is][ls * ns + js] * evec_in[jk_now][ks][ls];
            }
        }

        // transform the fourth index
#pragma omp parallel for private(is2_2, is, js, ks, ls)
        for (is2_1 = 0; is2_1 < ns2; ++is2_1) {
            is = is2_1 / ns;
            js = is2_1 % ns;

            for (is2_2 = 0; is2_2 < ns2; ++is2_2) {
                ks = is2_2 / ns;
                ls = is2_2 % ns;

                v4_tmp4[is][js * ns + ks] += v4_tmp3[is][js * ns + ls] * std::conj(evec_in[jk_now][ks][ls]);
            }
        }

        // copy to the final matrix
#pragma omp parallel for private(is, js)
        for (is2_1 = 0; is2_1 < ns2; ++is2_1) {
            is = is2_1 / ns;
            js = is2_1 % ns;

            for (is2 = 0; is2 < ns; is2++) {
                v4_mpi[ik_prod][is_now * ns + is2][is2_1] = factor * v4_tmp4[is2][is2_1];
            }
        }

        if (mympi->my_rank == 0) {
            std::cout << " SET " << ii + 1 << " done. \n";
        }

    } // loop over nk2_prod*ns

    deallocate(v4_array_at_kpair);
    deallocate(ind);

// Now, communicate the calculated data.
// When the data count is larger than 2^31-1, split it.

    long maxsize = 1;
    maxsize = (maxsize << 31) - 1;

    const size_t count = nk2_prod * ns4;
    const size_t count_sub = ns4;

    if (mympi->my_rank == 0) {
        std::cout << "Communicating v4_array over MPI ..." << std::flush;
    }
    if (count <= maxsize) {
#ifdef MPI_CXX_DOUBLE_COMPLEX
        MPI_Allreduce(&v4_mpi[0][0][0], &v4_out[0][0][0], count,
                      MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
#else
                                                                                                                                MPI_Allreduce(&v4_mpi[0][0][0], &v4_out[0][0][0], count,
                      MPI_COMPLEX16, MPI_SUM, MPI_COMM_WORLD);
#endif
    } else if (count_sub <= maxsize) {
        for (size_t ik_prod = 0; ik_prod < nk2_prod; ++ik_prod) {
#ifdef MPI_CXX_DOUBLE_COMPLEX
            MPI_Allreduce(&v4_mpi[ik_prod][0][0], &v4_out[ik_prod][0][0],
                          count_sub,
                          MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
#else
                                                                                                                                    MPI_Allreduce(&v4_mpi[ik_prod][0][0], &v4_out[ik_prod][0][0],
                          count_sub,
                          MPI_COMPLEX16, MPI_SUM, MPI_COMM_WORLD);
#endif
        }
    } else {
        for (size_t ik_prod = 0; ik_prod < nk2_prod; ++ik_prod) {
            for (is = 0; is < ns2; ++is) {
#ifdef MPI_CXX_DOUBLE_COMPLEX
                MPI_Allreduce(&v4_mpi[ik_prod][is][0], &v4_out[ik_prod][is][0],
                              ns2,
                              MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
#else
                                                                                                                                        MPI_Allreduce(&v4_mpi[ik_prod][is][0], &v4_out[ik_prod][is][0],
                              ns2,
                              MPI_COMPLEX16, MPI_SUM, MPI_COMM_WORLD);
#endif
            }
        }
    }
    if (mympi->my_rank == 0) {
        std::cout << "done.\n";
    }

    deallocate(v4_mpi);
    deallocate(v4_tmp0);
    deallocate(v4_tmp1);
    deallocate(v4_tmp2);
    deallocate(v4_tmp3);
    deallocate(v4_tmp4);

    zerofill_elements_acoustic_at_gamma(omega2_harmonic_in, v4_out, 4,
                                        kmesh_dense_in->nk, kmesh_coarse_in->nk_irred);

    if (mympi->my_rank == 0) {
        std::cout << " done !\n";
        timer->print_elapsed();
    }
}

void Scph::zerofill_elements_acoustic_at_gamma(double **omega2,
                                               std::complex<double> ***v_elems,
                                               const int fc_order,
                                               const unsigned int nk_dense_in,
                                               const unsigned int nk_irred_coarse_in) const
{
    // Set V3 or V4 elements involving acoustic modes at Gamma point
    // exactly zero.

    int jk;
    int is, js, ks, ls;
    const auto ns = dynamical->neval;
    bool *is_acoustic;
    allocate(is_acoustic, ns);
    int nacoustic;
    auto threshould = 1.0e-24;
    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);

    if (!(fc_order == 3 || fc_order == 4)) {
        exit("zerofill_elements_acoustic_at_gamma",
             "The fc_order must be either 3 or 4.");
    }

    do {
        nacoustic = 0;
        for (is = 0; is < ns; ++is) {
            if (std::abs(omega2[0][is]) < threshould) {
                is_acoustic[is] = true;
                ++nacoustic;
            } else {
                is_acoustic[is] = false;
            }
        }
        if (nacoustic > 3) {
            exit("zerofill_elements_acoustic_at_gamma",
                 "Could not assign acoustic modes at Gamma.");
        }
        threshould *= 2.0;
    } while (nacoustic < 3);


    if (fc_order == 3) {

        // Set V3 to zeros so as to avoid mixing with gamma acoustic modes
        // jk = 0;
        for (is = 0; is < ns; ++is) {
            for (ks = 0; ks < ns; ++ks) {
                for (ls = 0; ls < ns; ++ls) {
                    if (is_acoustic[ks] || is_acoustic[ls]) {
                        v_elems[0][is][ns * ks + ls] = complex_zero;
                    }
                }
            }
        }

        // ik = 0;
        for (jk = 0; jk < nk_dense_in; ++jk) {
            for (is = 0; is < ns; ++is) {
                if (is_acoustic[is]) {
                    for (ks = 0; ks < ns; ++ks) {
                        for (ls = 0; ls < ns; ++ls) {
                            v_elems[jk][is][ns * ks + ls] = complex_zero;
                        }
                    }
                }
            }
        }

    } else if (fc_order == 4) {
        // Set V4 to zeros so as to avoid mixing with gamma acoustic modes
        // jk = 0;
        for (int ik = 0; ik < nk_irred_coarse_in; ++ik) {
            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns; ++js) {
                    for (ks = 0; ks < ns; ++ks) {
                        for (ls = 0; ls < ns; ++ls) {
                            if (is_acoustic[ks] || is_acoustic[ls]) {
                                v_elems[nk_dense_in * ik][ns * is + js][ns * ks + ls] = complex_zero;
                            }
                        }
                    }
                }
            }
        }
        // ik = 0;
        for (jk = 0; jk < nk_dense_in; ++jk) {
            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns; ++js) {
                    if (is_acoustic[is] || is_acoustic[js]) {
                        for (ks = 0; ks < ns; ++ks) {
                            for (ls = 0; ls < ns; ++ls) {
                                v_elems[jk][ns * is + js][ns * ks + ls] = complex_zero;
                            }
                        }
                    }
                }
            }
        }
    }

    deallocate(is_acoustic);
}

void Scph::calculate_del_v0_del_umn_renorm(std::complex<double> *del_v0_del_umn_renorm,
                                           double *C1_array,
                                           double **C2_array,
                                           double ***C3_array,
                                           double **eta_tensor,
                                           double **u_tensor,
                                           std::complex<double> **del_v1_del_umn,
                                           std::complex<double> **del2_v1_del_umn2,
                                           std::complex<double> **del3_v1_del_umn3,
                                           std::complex<double> ***del_v2_del_umn,
                                           std::complex<double> ***del2_v2_del_umn2,
                                           std::complex<double> ****del_v3_del_umn,
                                           double *q0,
                                           double pvcell,
                                           const KpointMeshUniform *kmesh_dense_in)
{

    int ns = dynamical->neval;
    int nk = kmesh_dense_in->nk;
    double **del_eta_del_u;
    double *del_v0_del_eta;
    double *del_v0_strain_with_strain;

    std::complex<double> **del_v1_del_umn_with_umn;
    std::complex<double> ***del_v2_del_umn_with_umn;

    allocate(del_eta_del_u, 9, 9);
    allocate(del_v0_del_eta, 9);
    allocate(del_v0_strain_with_strain, 9);

    allocate(del_v1_del_umn_with_umn, 9, ns);
    allocate(del_v2_del_umn_with_umn, 9, ns, ns);

    double factor = 1.0 / 6.0 * 4.0 * nk;
    int i1, i2, i3, ixyz1, ixyz2, ixyz3, ixyz4;
    int is1, is2, is3;



    // calculate the derivative of eta_tensor by u_tensor
    for (i1 = 0; i1 < 9; i1++) {
        ixyz1 = i1 / 3;
        ixyz2 = i1 % 3;
        for (i2 = 0; i2 < 9; i2++) {
            ixyz3 = i2 / 3;
            ixyz4 = i2 % 3;

            del_eta_del_u[i1][i2] = 0.0;

            if (ixyz1 == ixyz3 && ixyz2 == ixyz4) {
                del_eta_del_u[i1][i2] += 0.5;
            }
            if (ixyz2 == ixyz3 && ixyz1 == ixyz4) {
                del_eta_del_u[i1][i2] += 0.5;
            }
            if (ixyz1 == ixyz3) {
                del_eta_del_u[i1][i2] += 0.5 * u_tensor[ixyz2][ixyz4];
            }
            if (ixyz2 == ixyz3) {
                del_eta_del_u[i1][i2] += 0.5 * u_tensor[ixyz1][ixyz4];
            }
        }
    }

    // calculate del_v0_del_eta
    for (i1 = 0; i1 < 9; i1++) {
        del_v0_del_eta[i1] = C1_array[i1];
        for (i2 = 0; i2 < 9; i2++) {
            del_v0_del_eta[i1] += C2_array[i1][i2] * eta_tensor[i2 / 3][i2 % 3];
            for (i3 = 0; i3 < 9; i3++) {
                del_v0_del_eta[i1] +=
                        0.5 * C3_array[i1][i2][i3] * eta_tensor[i2 / 3][i2 % 3] * eta_tensor[i3 / 3][i3 % 3];
            }
        }
    }

    // calculate contribution from v0 without atomic displacements
    for (i1 = 0; i1 < 9; i1++) {
        del_v0_strain_with_strain[i1] = 0.0;
        for (i2 = 0; i2 < 9; i2++) {
            del_v0_strain_with_strain[i1] += del_eta_del_u[i2][i1] * del_v0_del_eta[i2];
        }
    }

    // add pV term
    double F_tensor[3][3]; // F_{mu nu} = delta_{mu nu} + u_{mu nu}
    for (i1 = 0; i1 < 3; i1++) {
        for (i2 = 0; i2 < 3; i2++) {
            F_tensor[i1][i2] = u_tensor[i1][i2];
        }
        F_tensor[i1][i1] += 1.0;
    }
    for (i1 = 0; i1 < 9; i1++) {
        is1 = i1 / 3;
        is2 = i1 % 3;
        ixyz1 = (is1 + 1) % 3;
        ixyz2 = (is1 + 2) % 3;
        ixyz3 = (is2 + 1) % 3;
        ixyz4 = (is2 + 2) % 3;

        del_v0_strain_with_strain[i1] += pvcell * (F_tensor[ixyz1][ixyz3] * F_tensor[ixyz2][ixyz4] -
                                                   F_tensor[ixyz1][ixyz4] * F_tensor[ixyz2][ixyz3]);
    }

    // calculate del_v1_del_umn
    for (i1 = 0; i1 < 9; i1++) {
        for (is1 = 0; is1 < ns; is1++) {
            del_v1_del_umn_with_umn[i1][is1] = del_v1_del_umn[i1][is1];
            for (i2 = 0; i2 < 9; i2++) {
                del_v1_del_umn_with_umn[i1][is1] += del2_v1_del_umn2[i1 * 9 + i2][is1] * u_tensor[i2 / 3][i2 % 3];
                for (i3 = 0; i3 < 9; i3++) {
                    del_v1_del_umn_with_umn[i1][is1] +=
                            0.5 * del3_v1_del_umn3[i1 * 81 + i2 * 9 + i3][is1] * u_tensor[i2 / 3][i2 % 3] *
                            u_tensor[i3 / 3][i3 % 3];
                }
            }
        }
    }

    // calculate del_v2_del_umn
    for (i1 = 0; i1 < 9; i1++) {
        for (is1 = 0; is1 < ns; is1++) {
            for (is2 = 0; is2 < ns; is2++) {
                del_v2_del_umn_with_umn[i1][is1][is2] = del_v2_del_umn[i1][0][is1 * ns + is2];
                for (i2 = 0; i2 < 9; i2++) {
                    del_v2_del_umn_with_umn[i1][is1][is2] +=
                            del2_v2_del_umn2[i1 * 9 + i2][0][is1 * ns + is2] * u_tensor[i2 / 3][i2 % 3];
                }
            }
        }
    }

    // calculate del_v0_del_umn_renorm
    for (i1 = 0; i1 < 9; i1++) {
        del_v0_del_umn_renorm[i1] = del_v0_strain_with_strain[i1];
        for (is1 = 0; is1 < ns; is1++) {
            del_v0_del_umn_renorm[i1] += del_v1_del_umn_with_umn[i1][is1] * q0[is1];
            for (is2 = 0; is2 < ns; is2++) {
                del_v0_del_umn_renorm[i1] += 0.5 * del_v2_del_umn_with_umn[i1][is1][is2] * q0[is1] * q0[is2];
                for (is3 = 0; is3 < ns; is3++) {
                    del_v0_del_umn_renorm[i1] +=
                            factor * del_v3_del_umn[i1][0][is1][is2 * ns + is3] * q0[is1] * q0[is2] * q0[is3];
                }
            }
        }
    }


    deallocate(del_eta_del_u);
    deallocate(del_v0_del_eta);
    deallocate(del_v0_strain_with_strain);
    deallocate(del_v1_del_umn_with_umn);
    deallocate(del_v2_del_umn_with_umn);

    return;
}


void Scph::compute_anharmonic_v1_array(std::complex<double> *v1_SCP,
                                       std::complex<double> *v1_renorm,
                                       std::complex<double> ***v3_renorm,
                                       std::complex<double> ***cmat_convert,
                                       double **omega2_anharm_T,
                                       const double T_in,
                                       const KpointMeshUniform *kmesh_dense_in)
{
    using namespace Eigen;

    int is, js, js1, js2, ik;
    double n1, omega1_tmp;
    std::complex<double> Qtmp;
    const auto ns = dynamical->neval;
    int nk_scph = kmesh_dense_in->nk;

    int count_zero;

    MatrixXcd Cmat(ns, ns);
    MatrixXcd v3mat_original_mode(ns, ns), v3mat_tmp(ns, ns);

    // get gradient of the BO surface
    for (is = 0; is < ns; is++) {
        v1_SCP[is] = v1_renorm[is];
    }

    // calculate SCP renormalization
    for (is = 0; is < ns; is++) {
        for (ik = 0; ik < nk_scph; ik++) {
            // unitary transform phi3 to SCP mode
            for (js1 = 0; js1 < ns; js1++) {
                for (js2 = 0; js2 < ns; js2++) {
                    Cmat(js2, js1) = cmat_convert[ik][js1][js2]; // transpose
                    v3mat_original_mode(js1, js2) = v3_renorm[ik][is][js1 * ns + js2];
                }
            }
            v3mat_tmp = Cmat * v3mat_original_mode * Cmat.adjoint();

            // update v1_SCP
            count_zero = 0;
            for (js = 0; js < ns; js++) {
                omega1_tmp = std::sqrt(std::fabs(omega2_anharm_T[ik][js]));
                if (std::abs(omega1_tmp) < eps8) {
                    Qtmp = 0.0;
                    count_zero++;
                } else {
                    if (thermodynamics->classical) {
                        Qtmp = std::complex<double>(2.0 * T_in * thermodynamics->T_to_Ryd / (omega1_tmp * omega1_tmp),
                                                    0.0);
                    } else {
                        n1 = thermodynamics->fB(omega1_tmp, T_in);
                        Qtmp = std::complex<double>((2.0 * n1 + 1.0) / omega1_tmp, 0.0);
                    }
                }

                v1_SCP[is] += v3mat_tmp(js, js) * Qtmp;
            }
            if (ik == 0 && count_zero != 3) {
                std::cout << "Warning in compute_anharmonic_v1_array : ";
                std::cout << count_zero << " acoustic modes are detected in Gamma point.\n\n";
            } else if (ik != 0 && count_zero != 0) {
                std::cout << "Warning in compute_anharmonic_v1_array : ";
                std::cout << count_zero << " zero frequencies are detected in non-Gamma point (ik = " << ik << ").\n\n";
            }
        }
    }

}

void Scph::compute_anharmonic_del_v0_del_umn(std::complex<double> *del_v0_del_umn_SCP,
                                             std::complex<double> *del_v0_del_umn_renorm,
                                             std::complex<double> ***del_v2_del_umn,
                                             std::complex<double> ***del2_v2_del_umn2,
                                             std::complex<double> ****del_v3_del_umn,
                                             double **u_tensor,
                                             double *q0,
                                             std::complex<double> ***cmat_convert,
                                             double **omega2_anharm_T,
                                             const double T_in,
                                             const KpointMeshUniform *kmesh_dense_in)
{

    using namespace Eigen;

    int nk = kmesh_dense_in->nk;
    int ns = dynamical->neval;
    double factor = 4.0 * static_cast<double>(nk);
    double factor2 = 1.0 / factor;

    std::complex<double> ***del_v2_del_umn_renorm;
    allocate(del_v2_del_umn_renorm, 9, nk, ns * ns);

    int i1, i2;
    int ik;
    int is1, is2, is3, js, js1, js2;
    double n1, omega1_tmp;
    std::complex<double> Qtmp;
    int count_zero;

    MatrixXcd Cmat(ns, ns);
    MatrixXcd del_v2_strain_mat_original_mode(ns, ns), del_v2_strain_mat(ns, ns);

    // calculate del_v2_del_umn_renorm
    for (i1 = 0; i1 < 9; i1++) {
        for (ik = 0; ik < nk; ik++) {
            for (is1 = 0; is1 < ns; is1++) {
                for (is2 = 0; is2 < ns; is2++) {
                    del_v2_del_umn_renorm[i1][ik][is1 * ns + is2] = del_v2_del_umn[i1][ik][is1 * ns + is2];
                    // renormalization by strain
                    for (i2 = 0; i2 < 9; i2++) {
                        del_v2_del_umn_renorm[i1][ik][is1 * ns + is2] +=
                                del2_v2_del_umn2[i1 * 9 + i2][ik][is1 * ns + is2] * u_tensor[i2 / 3][i2 % 3];
                    }
                    // renormalization by displace
                    for (is3 = 0; is3 < ns; is3++) {
                        del_v2_del_umn_renorm[i1][ik][is1 * ns + is2] +=
                                factor * del_v3_del_umn[i1][ik][is3][is2 * ns + is1] * q0[is3];
                    }
                }
            }
        }
    }

    // potential energy term
    for (i1 = 0; i1 < 9; i1++) {
        del_v0_del_umn_SCP[i1] = del_v0_del_umn_renorm[i1];
    }

    // SCP renormalization
    for (i1 = 0; i1 < 9; i1++) {
        for (ik = 0; ik < nk; ik++) {
            // unitary transform the derivative of harmonic IFCs to SCP mode
            for (js1 = 0; js1 < ns; js1++) {
                for (js2 = 0; js2 < ns; js2++) {
                    Cmat(js1, js2) = cmat_convert[ik][js1][js2];
                    del_v2_strain_mat_original_mode(js1, js2) = del_v2_del_umn_renorm[i1][ik][js1 * ns + js2];
                }
            }
            del_v2_strain_mat = Cmat.adjoint() * del_v2_strain_mat_original_mode * Cmat;

            // update del_v0_del_umn_SCP
            count_zero = 0;
            for (js = 0; js < ns; js++) {
                omega1_tmp = std::sqrt(std::fabs(omega2_anharm_T[ik][js]));
                if (std::abs(omega1_tmp) < eps8) {
                    Qtmp = 0.0;
                    count_zero++;
                } else {
                    if (omega2_anharm_T[ik][js] < 0.0) {
                        std::cout
                                << "Warning in compute_anharmonic_del_v0_del_umn: squared SCP frequency is negative. ik = "
                                << ik << '\n';
                    }
                    if (thermodynamics->classical) {
                        Qtmp = std::complex<double>(2.0 * T_in * thermodynamics->T_to_Ryd / (omega1_tmp * omega1_tmp),
                                                    0.0);
                    } else {
                        n1 = thermodynamics->fB(omega1_tmp, T_in);
                        Qtmp = std::complex<double>((2.0 * n1 + 1.0) / omega1_tmp, 0.0);
                    }
                }

                del_v0_del_umn_SCP[i1] += factor2 * del_v2_strain_mat(js, js) * Qtmp;
            }
        }
    }

    deallocate(del_v2_del_umn_renorm);
}


void Scph::setup_kmesh()
{
    // Setup k points for SCPH equation
    MPI_Bcast(&kmesh_scph[0], 3, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&kmesh_interpolate[0], 3, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    kmesh_coarse = new KpointMeshUniform(kmesh_interpolate);
    kmesh_dense = new KpointMeshUniform(kmesh_scph);
    kmesh_coarse->setup(symmetry->SymmList, system->get_primcell().reciprocal_lattice_vector, true);
    kmesh_dense->setup(symmetry->SymmList, system->get_primcell().reciprocal_lattice_vector, true);

    if (mympi->my_rank == 0) {
//        if (verbosity > 0) {
        std::cout << " Setting up the SCPH calculations ...\n\n";
        std::cout << "  Gamma-centered uniform grid with the following mesh density:\n";
        std::cout << "  nk1:" << std::setw(5) << kmesh_scph[0] << '\n';
        std::cout << "  nk2:" << std::setw(5) << kmesh_scph[1] << '\n';
        std::cout << "  nk3:" << std::setw(5) << kmesh_scph[2] << "\n\n";
        std::cout << "  Number of k points : " << kmesh_dense->nk << '\n';
        std::cout << "  Number of irreducible k points : " << kmesh_dense->nk_irred << "\n\n";
        std::cout << "  Fourier interpolation from reciprocal to real space\n";
        std::cout << "  will be performed with the following mesh density:\n";
        std::cout << "  nk1:" << std::setw(5) << kmesh_interpolate[0] << '\n';
        std::cout << "  nk2:" << std::setw(5) << kmesh_interpolate[1] << '\n';
        std::cout << "  nk3:" << std::setw(5) << kmesh_interpolate[2] << "\n\n";
        std::cout << "  Number of k points : " << kmesh_coarse->nk << '\n';
        std::cout << "  Number of irreducible k points : "
                  << kmesh_coarse->nk_irred << '\n';
//        }
    }

    auto info_mapping = kpoint->get_kmap_coarse_to_dense(kmesh_coarse,
                                                         kmesh_dense,
                                                         kmap_interpolate_to_scph);
    if (info_mapping == 1) {
        exit("setup_kmesh",
             "KMESH_INTERPOLATE should be a integral multiple of KMESH_SCPH");
    }

    kmesh_coarse->setup_kpoint_symmetry(symmetry->SymmListWithMap);
}

void Scph::setup_eigvecs()
{
    const auto ns = dynamical->neval;

    if (mympi->my_rank == 0) {
        std::cout << '\n'
                  << " Diagonalizing dynamical matrices for all k points ... ";
    }

    allocate(evec_harmonic, kmesh_dense->nk, ns, ns);
    allocate(omega2_harmonic, kmesh_dense->nk, ns);

    // Calculate phonon eigenvalues and eigenvectors for all k-points for scph

#pragma omp parallel for
    for (int ik = 0; ik < kmesh_dense->nk; ++ik) {

        dynamical->eval_k(kmesh_dense->xk[ik],
                          kmesh_dense->kvec_na[ik],
                          fcs_phonon->force_constant_with_cell[0],
                          omega2_harmonic[ik],
                          evec_harmonic[ik], true);

        for (auto is = 0; is < ns; ++is) {
            if (std::abs(omega2_harmonic[ik][is]) < eps) {
                omega2_harmonic[ik][is] = 1.0e-30;
            }
        }
    }

    if (mympi->my_rank == 0) {
        std::cout << "done !\n";
    }
}

void Scph::setup_pp_interaction()
{
    // Prepare information for calculating ph-ph interaction coefficients.

    const auto relax_str = relaxation->relax_str;

    if (mympi->my_rank == 0) {
        if (relax_str || bubble > 0) {
            std::cout << " Preparing for calculating V3 & V4  ...";
        } else {
            std::cout << " Preparing for calculating V4  ...";
        }
    }

    if (anharmonic_core->quartic_mode != 1) {
        exit("setup_pp_interaction",
             "quartic_mode should be 1 for SCPH");
    }

    // Setup for V3 if relax_str = True.
    if (relax_str || bubble > 0) {
        allocate(phi3_reciprocal, anharmonic_core->get_ngroup_fcs(3));
    }
    allocate(phi4_reciprocal, anharmonic_core->get_ngroup_fcs(4));

    phase_factor_scph = new PhaseFactorStorage(kmesh_dense->nk_i);
    phase_factor_scph->create(true);

    if (mympi->my_rank == 0) {
        std::cout << " done!\n";
    }
}

void Scph::find_degeneracy(std::vector<int> *degeneracy_out,
                           const unsigned int nk_in,
                           double **eval_in) const
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


void Scph::compute_anharmonic_frequency(std::complex<double> ***v4_array_all,
                                        double **omega2_out,
                                        std::complex<double> ***evec_anharm_scph,
                                        const double temp,
                                        bool &flag_converged,
                                        std::complex<double> ***cmat_convert,
                                        const bool offdiag,
                                        std::complex<double> **delta_v2_renorm,
                                        const unsigned int verbosity)
{
    // This is the main function of the SCPH equation.
    // The detailed algorithm can be found in PRB 92, 054301 (2015).
    // Eigen3 library is used for the compact notation of matrix-matrix products.

    using namespace Eigen;

    int ik, jk;
    unsigned int is, js, ks;
    const auto nk = kmesh_dense->nk;
    const auto ns = dynamical->neval;
    const auto ns2 = ns * ns;
    unsigned int knum, knum_interpolate;
    const auto nk_interpolate = kmesh_coarse->nk;
    const auto nk_irred_interpolate = kmesh_coarse->nk_irred;
    const auto nk1 = kmesh_interpolate[0];
    const auto nk2 = kmesh_interpolate[1];
    const auto nk3 = kmesh_interpolate[2];
    int iloop;

    MatrixXd omega_now(nk, ns), omega_old(nk, ns);
    MatrixXd omega2_HA(nk, ns);
    MatrixXcd mat_tmp(ns, ns), evec_tmp(ns, ns);

    VectorXd eval_tmp(ns);
    MatrixXcd Dymat(ns, ns);
    MatrixXcd Fmat(ns, ns);
    MatrixXcd Qmat = MatrixXcd::Zero(ns, ns);
    MatrixXcd Cmat(ns, ns), Dmat(ns, ns);

    double diff;
    double conv_tol = tolerance_scph;
    double alpha = mixalpha;

    double **eval_interpolate;
    bool has_negative;

    std::complex<double> ***evec_new;
    std::complex<double> ***dymat_r_new;
    std::complex<double> ***dymat_q, ***dymat_q_HA;

    std::vector<MatrixXcd> dmat_convert, dmat_convert_old;
    std::vector<MatrixXcd> evec_initial, evec_initial_adjoint;
    std::vector<MatrixXcd> Fmat0;

    dmat_convert.resize(nk);
    dmat_convert_old.resize(nk);
    evec_initial.resize(nk);
    evec_initial_adjoint.resize(nk); // just for minior optimization (may not be very effective)
    Fmat0.resize(nk_irred_interpolate);

    for (ik = 0; ik < nk; ++ik) {
        dmat_convert[ik].resize(ns, ns);
        dmat_convert_old[ik].resize(ns, ns);
        evec_initial[ik].resize(ns, ns);
    }
    for (ik = 0; ik < nk_irred_interpolate; ++ik) {
        Fmat0[ik].resize(ns, ns);
    }

    constexpr auto complex_one = std::complex<double>(1.0, 0.0);
    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);

    SelfAdjointEigenSolver<MatrixXcd> saes;

    allocate(eval_interpolate, nk, ns);
    allocate(evec_new, nk, ns, ns);
    allocate(dymat_r_new, ns, ns, nk_interpolate);
    allocate(dymat_q, ns, ns, nk_interpolate);
    allocate(dymat_q_HA, ns, ns, nk_interpolate);

    const auto T_in = temp;

    std::cout << " Temperature = " << T_in << " K\n";

    // Set initial values

    for (ik = 0; ik < nk; ++ik) {
        for (is = 0; is < ns; ++is) {

            if (flag_converged) {
                if (omega2_out[ik][is] < 0.0 && std::abs(omega2_out[ik][is]) > 1.0e-16 && verbosity > 0) {
                    std::cout << "Warning : Large negative frequency detected\n";
                }

                if (omega2_out[ik][is] < 0.0) {
                    omega_now(ik, is) = std::sqrt(-omega2_out[ik][is]);
                } else {
                    omega_now(ik, is) = std::sqrt(omega2_out[ik][is]);
                }
            } else {
                if (omega2_harmonic[ik][is] < 0.0) {
                    omega_now(ik, is) = std::sqrt(-omega2_harmonic[ik][is]);
                } else {
                    omega_now(ik, is) = std::sqrt(omega2_harmonic[ik][is]);
                }
            }

            omega2_HA(ik, is) = omega2_harmonic[ik][is];

            for (js = 0; js < ns; ++js) {
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
            evec_initial_adjoint[ik] = evec_initial[ik].adjoint();
        }
    }

    // Set initial harmonic dymat and eigenvalues

    for (ik = 0; ik < nk_irred_interpolate; ++ik) {
        knum_interpolate = kmesh_coarse->kpoint_irred_all[ik][0].knum;
        knum = kmap_interpolate_to_scph[knum_interpolate];

        Dymat = omega2_HA.row(knum).asDiagonal();
        evec_tmp = evec_initial[knum];

        // Harmonic dynamical matrix
        Dymat = evec_tmp * Dymat.eval() * evec_tmp.adjoint();

        dynamical->symmetrize_dynamical_matrix(ik, kmesh_coarse,
                                               mat_transform_sym,
                                               Dymat);

        for (is = 0; is < ns; ++is) {
            for (js = 0; js < ns; ++js) {
                dymat_q_HA[is][js][knum_interpolate] = Dymat(is, js);
            }
        }

        // Harmonic Fmat
        Fmat0[ik] = omega2_HA.row(knum).asDiagonal();
        for (is = 0; is < ns; ++is) {
            for (js = 0; js < ns; ++js) {
                Fmat0[ik](is, js) += delta_v2_renorm[knum_interpolate][is * ns + js];
            }
        }
    } // close loop ik


    dynamical->replicate_dymat_for_all_kpoints(kmesh_coarse, mat_transform_sym,
                                               dymat_q_HA);

    // This can be done outside the function
    dynamical->precompute_dymat_harm(kmesh_dense->nk,
                                     kmesh_dense->xk,
                                     kmesh_dense->kvec_na,
                                     dymat_harm_short,
                                     dymat_harm_long);

    int icount = 0;

    // Main loop
    for (iloop = 0; iloop < maxiter; ++iloop) {
        for (ik = 0; ik < nk; ++ik) {
            for (is = 0; is < ns; ++is) {
                auto omega1 = omega_now(ik, is);
                if (std::abs(omega1) < eps8) {
                    Qmat(is, is) = complex_zero;
                } else {
                    // Note that the missing factor 2 in the denominator of Qmat is
                    // already considered in the v4_array_all.
                    if (thermodynamics->classical) {
                        Qmat(is, is) = std::complex<double>(2.0 * T_in * thermodynamics->T_to_Ryd / (omega1 * omega1),
                                                            0.0);
                    } else {
                        auto n1 = thermodynamics->fB(omega1, T_in);
                        Qmat(is, is) = std::complex<double>((2.0 * n1 + 1.0) / omega1, 0.0);
                    }
                }
            }

            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns; ++js) {
                    Cmat(is, js) = cmat_convert[ik][is][js];
                }
            }

            Dmat = Cmat * Qmat * Cmat.adjoint();
            dmat_convert[ik] = Dmat;
        }

        // Mixing dmat
        if (iloop > 0) {
            for (ik = 0; ik < nk; ++ik) {
                dmat_convert[ik] = alpha * dmat_convert[ik].eval() + (1.0 - alpha) * dmat_convert_old[ik];
            }
        }

        // TODO: This loop may be parallelized by MPI in the future.
        for (ik = 0; ik < nk_irred_interpolate; ++ik) {

            knum_interpolate = kmesh_coarse->kpoint_irred_all[ik][0].knum;
            knum = kmap_interpolate_to_scph[knum_interpolate];

            // Fmat harmonic
            Fmat = Fmat0[ik];

            // Anharmonic correction to Fmat
            if (!offdiag) {
#pragma omp parallel for private(jk, ks)
                for (is = 0; is < ns; ++is) {
                    for (jk = 0; jk < nk; ++jk) {
                        for (ks = 0; ks < ns; ++ks) {
                            Fmat(is, is) += v4_array_all[nk * ik + jk][(ns + 1) * is][(ns + 1) * ks]
                                            * dmat_convert[jk](ks, ks);
                        }
                    }
                }
            } else {
#pragma omp parallel for private(is, js, jk, ks)
                for (int ijs = 0; ijs < ns2; ++ijs) {
                    is = ijs / ns;
                    js = ijs % ns;
                    for (jk = 0; jk < nk; ++jk) {
                        for (ks = 0; ks < ns; ++ks) {
                            for (unsigned int ls = 0; ls < ns; ++ls) {
                                Fmat(is, js) += v4_array_all[nk * ik + jk][ijs][ns * ks + ls]
                                                * dmat_convert[jk](ks, ls);
                            }
                        }
                    }
                }
            }

            saes.compute(Fmat);
            eval_tmp = saes.eigenvalues();

            for (is = 0; is < ns; ++is) {

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

                    if (v4_array_all[nk * ik + knum][(ns + 1) * is][(ns + 1) * is].real() > 0.0) {
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
            mat_tmp = evec_initial[knum] * saes.eigenvectors();
            Dymat = mat_tmp * eval_tmp.asDiagonal() * mat_tmp.adjoint();

#ifdef _DEBUG2
                                                                                                                                    Dymat_sym = Dymat;
            symmetrize_dynamical_matrix(ik, Dymat_sym);
            std::complex<double> **dymat_exact;
            allocate(dymat_exact, ns, ns);
            std::cout << "ik = " << ik + 1 << '\n';
            std::cout << "Dymat" << '\n';
            std::cout << Dymat << '\n';
            std::cout << "Dymat_sym" << '\n';
            std::cout << Dymat_sym << '\n';
            dynamical->calc_analytic_k(xk_interpolate[knum_interpolate], fcs_phonon->fc2_ext, dymat_exact);
            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns; ++js) {
                    Dymat_sym(is,js) = dymat_exact[is][js];
                }
            }
            std::cout << "Dymat_exact" << '\n';
            std::cout << Dymat_sym << '\n';
            deallocate(dymat_exact);jjj

#endif
            dynamical->symmetrize_dynamical_matrix(ik, kmesh_coarse, mat_transform_sym,
                                                   Dymat);
            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns; ++js) {
                    dymat_q[is][js][knum_interpolate] = Dymat(is, js);
                }
            }
        } // close loop ik
        dynamical->replicate_dymat_for_all_kpoints(kmesh_coarse,
                                                   mat_transform_sym,
                                                   dymat_q);
#ifdef _DEBUG2
                                                                                                                                for (ik = 0; ik < nk_interpolate; ++ik) {

            knum = kmap_interpolate_to_scph[ik];

            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns; ++js) {
                    Dymat(is,js) = dymat_q[is][js][ik];
                }
            }

            saes.compute(Dymat);
            eval_tmp = saes.eigenvalues();

            for (is = 0; is < ns; ++is) {
                eval_orig(is) = omega2_harmonic(knum,is);
            }

            std::cout << " ik = " << std::setw(4) << ik + 1 << " : ";
            for (i = 0; i < 3; ++i)  std::cout << std::setw(15) << xk_scph[knum][i];
            std::cout << '\n';

            for (is = 0; is < ns; ++is) {
                std::cout << std::setw(15) << eval_tmp(is);
                std::cout << std::setw(15) << eval_orig(is);
                std::cout << std::setw(15) << eval_tmp(is) - eval_orig(is) << '\n';
            }

        }
#endif

        // Subtract harmonic contribution to the dynamical matrix
        for (ik = 0; ik < nk_interpolate; ++ik) {
            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns; ++js) {
                    dymat_q[is][js][ik] -= dymat_q_HA[is][js][ik];
                }
            }
        }

        for (is = 0; is < ns; ++is) {
            for (js = 0; js < ns; ++js) {
                fftw_plan plan = fftw_plan_dft_3d(nk1, nk2, nk3,
                                                  reinterpret_cast<fftw_complex *>(dymat_q[is][js]),
                                                  reinterpret_cast<fftw_complex *>(dymat_r_new[is][js]),
                                                  FFTW_FORWARD, FFTW_ESTIMATE);
                fftw_execute(plan);
                fftw_destroy_plan(plan);

                for (ik = 0; ik < nk_interpolate; ++ik)
                    dymat_r_new[is][js][ik] /= static_cast<double>(nk_interpolate);
            }
        }

        dynamical->exec_interpolation(kmesh_interpolate,
                                      dymat_r_new,
                                      nk,
                                      kmesh_dense->xk,
                                      kmesh_dense->kvec_na,
                                      eval_interpolate,
                                      evec_new,
                                      dymat_harm_short,
                                      dymat_harm_long,
                                      mindist_list_scph,
                                      true, true);

        for (ik = 0; ik < nk; ++ik) {
            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns; ++js) {
                    evec_tmp(is, js) = evec_new[ik][js][is];
                }
            }


//            Cmat = evec_initial[ik].adjoint() * evec_tmp;
            Cmat = evec_initial_adjoint[ik] * evec_tmp;

            for (is = 0; is < ns; ++is) {
                omega_now(ik, is) = eval_interpolate[ik][is];
                for (js = 0; js < ns; ++js) {
                    cmat_convert[ik][is][js] = Cmat(is, js);
                }
            }
        }

        if (iloop == 0) {
            if (verbosity > 0) {
                std::cout << "  SCPH ITER " << std::setw(5) << iloop + 1 << " :  DIFF = N/A\n";
            }

            omega_old = omega_now;

        } else {

            diff = 0.0;

            for (ik = 0; ik < nk_interpolate; ++ik) {
                knum = kmap_interpolate_to_scph[ik];
                for (is = 0; is < ns; ++is) {
                    diff += std::pow(omega_now(knum, is) - omega_old(knum, is), 2.0);
                }
            }
            diff /= static_cast<double>(nk_interpolate * ns);
            if (verbosity > 0) {
                std::cout << "  SCPH ITER " << std::setw(5) << iloop + 1 << " : ";
                std::cout << " DIFF = " << std::setw(15) << std::sqrt(diff) << '\n';
            }

            omega_old = omega_now;

            if (std::sqrt(diff) < conv_tol) {
                has_negative = false;

                for (ik = 0; ik < nk_interpolate; ++ik) {
                    knum = kmap_interpolate_to_scph[ik];
                    for (is = 0; is < ns; ++is) {
                        if (omega_now(knum, is) < 0.0 && std::abs(omega_now(knum, is)) > eps8) {
                            has_negative = true;
                            break;
                        }
                    }
                }
                if (!has_negative) {
                    if (verbosity > 0) std::cout << "  DIFF < SCPH_TOL : break SCPH loop\n";
                    break;
                }
                if (verbosity > 0) std::cout << "  DIFF < SCPH_TOL but a negative frequency is detected.\n";
            }
        }

        for (ik = 0; ik < nk; ++ik) {
            dmat_convert_old[ik] = dmat_convert[ik];
        }
    } // end loop iteration

    if (std::sqrt(diff) < conv_tol) {
        if (verbosity > 0) {
            std::cout << " Temp = " << T_in;
            std::cout << " : convergence achieved in " << std::setw(5)
                      << iloop + 1 << " iterations.\n";
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
            if (eval_interpolate[ik][is] < 0.0) {
                if (std::abs(eval_interpolate[ik][is]) <= eps10) {
                    omega2_out[ik][is] = 0.0;
                } else {
                    omega2_out[ik][is] = -std::pow(eval_interpolate[ik][is], 2.0);
                }
            } else {
                omega2_out[ik][is] = std::pow(eval_interpolate[ik][is], 2.0);
            }
            for (js = 0; js < ns; ++js) {
                evec_anharm_scph[ik][is][js] = evec_new[ik][is][js];
            }
        }
    }

    if (verbosity > 1) {
        std::cout << "New eigenvalues\n";
        for (ik = 0; ik < nk_interpolate; ++ik) {
            knum = kmap_interpolate_to_scph[ik];
            for (is = 0; is < ns; ++is) {
                std::cout << " ik_interpolate = " << std::setw(5) << ik + 1;
                std::cout << " is = " << std::setw(5) << is + 1;
                std::cout << " omega2 = " << std::setw(15) << omega2_out[knum][is] << '\n';
            }
            std::cout << '\n';
        }
    }

    deallocate(eval_interpolate);
    deallocate(evec_new);
    deallocate(dymat_r_new);
    deallocate(dymat_q);
    deallocate(dymat_q_HA);
}


void Scph::compute_anharmonic_frequency2(std::complex<double> ***v4_array_all,
                                         double **omega2_anharm,
                                         std::complex<double> ***evec_anharm_scph,
                                         const double temp,
                                         bool &flag_converged,
                                         std::complex<double> ***cmat_convert,
                                         const bool offdiag,
                                         const unsigned int verbosity)
{
    using namespace Eigen;
    constexpr auto complex_one = std::complex<double>(1.0, 0.0);
    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);

    int ik;
    unsigned int is, js;
    const auto nk = kmesh_dense->nk;
    const auto ns = dynamical->neval;
    unsigned int knum, knum_interpolate;
    const auto nk_interpolate = kmesh_coarse->nk;
    const auto nk_irred_interpolate = kmesh_coarse->nk_irred;

    MatrixXd omega2_in(nk, ns), omega2_out(nk, ns);
    MatrixXd omega2_HA(nk, ns);
    MatrixXd omega2_in_sorted(nk, ns);
    std::vector<MatrixXd> permutation1, permutation2, permutation3;
    MatrixXcd Dymat(ns, ns);
    MatrixXd permutation;

    const double conv_tol = tolerance_scph;
    double alpha = mixalpha;

    std::complex<double> ***evec_new;
    std::complex<double> ***dymat_q, ***dymat_q_HA;

//    std::vector<MatrixXcd> dmat_convert;
    std::vector<MatrixXcd> evec_initial;
    std::vector<MatrixXcd> Fmat0;

//    dmat_convert.resize(nk);
    evec_initial.resize(nk);
    Fmat0.resize(nk_irred_interpolate);

    for (ik = 0; ik < nk; ++ik) {
//        dmat_convert[ik].resize(ns, ns);
        evec_initial[ik].resize(ns, ns);
    }
    for (ik = 0; ik < nk_irred_interpolate; ++ik) {
        Fmat0[ik].resize(ns, ns);
    }


    allocate(evec_new, nk, ns, ns);
    allocate(dymat_q, ns, ns, nk_interpolate);
    allocate(dymat_q_HA, ns, ns, nk_interpolate);

    const auto T_in = temp;

    std::vector<unsigned int> is_acoustic(ns);

    std::cout << " Temperature = " << T_in << " K\n";


    const int imix_algo = 1;

    // Set initial values

    for (ik = 0; ik < nk; ++ik) {
        for (is = 0; is < ns; ++is) {

            if (flag_converged) {
                if (omega2_anharm[ik][is] < 0.0 && std::abs(omega2_anharm[ik][is]) > 1.0e-16 && verbosity > 0) {
                    std::cout << "Warning : Large negative frequency detected\n";
                }

                if (omega2_anharm[ik][is] < 0.0) {
                    omega2_in(ik, is) = -omega2_anharm[ik][is];
                } else {
                    omega2_in(ik, is) = omega2_anharm[ik][is];
                }
            } else {
                if (omega2_harmonic[ik][is] < 0.0) {
                    omega2_in(ik, is) = -omega2_harmonic[ik][is];
                } else {
                    omega2_in(ik, is) = omega2_harmonic[ik][is];
                }
            }

            omega2_HA(ik, is) = omega2_harmonic[ik][is];

            for (js = 0; js < ns; ++js) {
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
    }

    int nacoustic;
    auto threshould = 1.0e-24;
    do {
        nacoustic = 0;
        for (is = 0; is < ns; ++is) {
            if (std::abs(omega2_harmonic[0][is]) < threshould) {
                is_acoustic[is] = 1;
                ++nacoustic;
            } else {
                is_acoustic[is] = 0;
            }
        }
        if (nacoustic > 3) {
            exit("compute_anharmonic_frequency2",
                 "Could not assign acoustic modes at Gamma.");
        }
        threshould *= 2.0;
    } while (nacoustic < 3);

    for (is = 0; is < ns; ++is) {
        if (is_acoustic[is]) {
            omega2_in(0, is) = 0.0;
            omega2_HA(0, is) = 0.0;
        }
    }

    // Set initial harmonic dymat and eigenvalues
    for (ik = 0; ik < nk_irred_interpolate; ++ik) {
        knum_interpolate = kmesh_coarse->kpoint_irred_all[ik][0].knum;
        knum = kmap_interpolate_to_scph[knum_interpolate];
        Dymat = omega2_HA.row(knum).asDiagonal();

        // Harmonic dynamical matrix
        Dymat = evec_initial[knum] * Dymat.eval() * evec_initial[knum].adjoint();
        dynamical->symmetrize_dynamical_matrix(ik, kmesh_coarse,
                                               mat_transform_sym,
                                               Dymat);
        for (is = 0; is < ns; ++is) {
            for (js = 0; js < ns; ++js) {
                dymat_q_HA[is][js][knum_interpolate] = Dymat(is, js);
            }
        }

        // Harmonic Fmat
        Fmat0[ik] = omega2_HA.row(knum).asDiagonal();
    } // close loop ik
    dynamical->replicate_dymat_for_all_kpoints(kmesh_coarse,
                                               mat_transform_sym,
                                               dymat_q_HA);

    // Main loop
    int iloop = 0;

    permutation1.resize(nk);
    permutation2.resize(nk);
    permutation3.resize(nk);

    for (ik = 0; ik < nk; ++ik) {
        get_permutation_matrix(ns, cmat_convert[ik], permutation);
        permutation1[ik] = permutation;
    }


    update_frequency(T_in,
                     omega2_in,
                     Fmat0,
                     evec_initial,
                     dymat_q_HA,
                     v4_array_all,
                     cmat_convert,
//                     dmat_convert,
                     dymat_q,
                     evec_new,
                     1.0,
                     offdiag,
                     omega2_out);

    for (ik = 0; ik < nk; ++ik) {
        get_permutation_matrix(ns, cmat_convert[ik], permutation);
        permutation2[ik] = permutation;
    }


    if (verbosity > 0) {
        std::cout << "  SCPH ITER " << std::setw(5) << iloop + 1 << " :  DIFF = N/A\n";
    }

    double diff;

    if (imix_algo == 0) {

        omega2_in = omega2_out;

        for (iloop = 1; iloop < maxiter; ++iloop) {

            update_frequency(T_in,
                             omega2_in,
                             Fmat0,
                             evec_initial,
                             dymat_q_HA,
                             v4_array_all,
                             cmat_convert,
//                             dmat_convert,
                             dymat_q,
                             evec_new,
                             1.0,
                             offdiag,
                             omega2_out);

            omega2_out = alpha * omega2_out + (1.0 - alpha) * omega2_in;

            diff = 0.0;
            double diff2 = 0.0;
            double omega_in, omega_out;

            for (ik = 0; ik < nk_interpolate; ++ik) {
                knum = kmap_interpolate_to_scph[ik];
                for (is = 0; is < ns; ++is) {
                    if (omega2_out(knum, is) >= 0.0) {
                        omega_out = std::sqrt(omega2_out(knum, is));
                    } else {
                        omega_out = -std::sqrt(-omega2_out(knum, is));
                    }
                    if (omega2_in(knum, is) >= 0.0) {
                        omega_in = std::sqrt(omega2_in(knum, is));
                    } else {
                        omega_in = -std::sqrt(-omega2_in(knum, is));
                    }
                    diff += (omega_out - omega_in) * (omega_out - omega_in);
                }
                diff2 += (omega2_out.row(knum) - omega2_in.row(knum)).norm();
            }
            diff /= static_cast<double>(nk_interpolate * ns);
            diff2 /= static_cast<double>(nk_interpolate * ns);
            diff = std::sqrt(diff);
            diff2 = std::sqrt(diff2);
            if (verbosity > 0) {
                std::cout << "  SCPH ITER " << std::setw(5) << iloop + 1 << " : ";
                std::cout << " DIFF = " << std::setw(15) << diff << '\n';
            }

            if (diff < conv_tol) {
                auto has_negative = false;

                for (ik = 0; ik < nk_interpolate; ++ik) {
                    knum = kmap_interpolate_to_scph[ik];
                    for (is = 0; is < ns; ++is) {
                        if (omega2_out(knum, is) < 0.0 && std::abs(omega2_out(knum, is)) > eps15) {
                            has_negative = true;
                            break;
                        }
                    }
                }
                if (!has_negative) {
                    if (verbosity > 0) std::cout << "  DIFF < SCPH_TOL : break SCPH loop\n";
                    break;
                }
                if (verbosity > 0) std::cout << "  DIFF < SCPH_TOL but a negative frequency is detected.\n";
            }

            omega2_in = omega2_out;
//        omega2_in = alpha * omega2_out + (1.0 - alpha) * omega2_in;

        } // end loop iteration

    } else if (imix_algo == 1) {

        MatrixXd omega2_prev(nk, ns), res_now(nk, ns), res_prev(nk, ns);
        MatrixXd omega2_tmp(nk, ns);
        VectorXd beta1_k(nk);
        MatrixXd beta1_ks(nk, ns);

        std::cout << "omega2_in = " << omega2_in.row(0) << '\n';
        std::cout << "omega2_out = " << omega2_out.row(0) << '\n';

        omega2_prev = omega2_in; // x_{i-1}
        omega2_in = omega2_out; // x_{i}

        double beta1, beta2;

        for (iloop = 1; iloop < maxiter; ++iloop) {

            update_frequency(T_in,
                             omega2_in,
                             Fmat0,
                             evec_initial,
                             dymat_q_HA,
                             v4_array_all,
                             cmat_convert,
//                             dmat_convert,
                             dymat_q,
                             evec_new,
                             1.0,
                             offdiag,
                             omega2_out);

            for (ik = 0; ik < nk; ++ik) {
                get_permutation_matrix(ns, cmat_convert[ik], permutation);
                permutation3[ik] = permutation;
            }

//            std::cout << "omega2_prev = " << omega2_prev.row(0)<< '\n';
//            std::cout << "omega2_in   = " << omega2_in.row(0)  << '\n';
//            std::cout << "omega2_out  = " << omega2_out.row(0) << '\n';

//            std::cout << "omega2_prev perm = " << omega2_prev.row(0) * permutation1[0] << '\n';
//            std::cout << "omega2_in perm   = " << omega2_in.row(0) * permutation2[0] << '\n';
//            std::cout << "omega2_out perm  = " << omega2_out.row(0)  * permutation3[0] << '\n';

//            std::cout << "permutation1:\n" << permutation1[0] << '\n';
//            std::cout << "permutation2:\n" << permutation2[0] << '\n';
//            std::cout << "permutation3:\n" << permutation3[0] << '\n';

            res_now = omega2_out - omega2_in; // (f(x_i) - x_i)
            res_prev = omega2_in - omega2_prev; // f(x_{i-1}) - x_{i-1} = x_{i} - x_{i-1}
//
//            for (ik = 0; ik < nk; ++ik) {
//                res_now.row(ik) = omega2_out.row(ik) * permutation3[ik] - omega2_in.row(ik) * permutation2[ik];
//                res_prev.row(ik) = omega2_in.row(ik) * permutation2[ik] - omega2_prev.row(ik) * permutation1[ik];
//            }
            std::cout << "res_prev = " << res_prev.row(0) << '\n';
            std::cout << "rev_now  = " << res_now.row(0) << '\n';

            beta1 = 0.0;
            beta2 = 0.0;
            auto denom = 0.0;
            auto numerator = 0.0;
            auto numerator2 = 0.0;
            for (ik = 0; ik < nk_interpolate; ++ik) {
                knum = kmap_interpolate_to_scph[ik];
                for (is = 0; is < ns; ++is) {
                    numerator += (res_now(knum, is) - res_prev(knum, is)) * res_now(knum, is);
                    numerator2 += (res_prev(knum, is) - res_now(knum, is)) * res_prev(knum, is);
                    denom += std::pow(res_now(knum, is) - res_prev(knum, is), 2.0);
                }
            }

            beta1 = numerator / denom;
            beta2 = numerator2 / denom;

            std::cout << "Beta1 = " << beta1 << " Beta2 = " << beta2 << " Beta1 + Beta2 = " << beta1 + beta2 << '\n';

//            if (iloop < 3) beta1 = 1.0;
            beta1 = 1.0;

            omega2_out = beta1 * (omega2_prev + alpha * res_prev) +
                         (1.0 - beta1) * (omega2_in + alpha * res_now);
//            for (ik = 0; ik < nk; ++ik) {
//                omega2_out.row(ik) = (beta1 * (omega2_prev.row(ik) * permutation1[ik] + alpha * res_prev.row(ik))
//                                      + (1.0 - beta1) * (omega2_in.row(ik) * permutation2[ik] + alpha * res_now.row(ik))) *
//                                     permutation3[ik];
//            }

            std::cout << "omega2_out (after_mix) = " << omega2_out.row(0) << '\n';
            diff = 0.0;
            double diff2 = 0.0;
            double omega_in, omega_out;

            for (ik = 0; ik < nk_interpolate; ++ik) {
                knum = kmap_interpolate_to_scph[ik];
                for (is = 0; is < ns; ++is) {
                    if (omega2_out(knum, is) >= 0.0) {
                        omega_out = std::sqrt(omega2_out(knum, is));
                    } else {
                        omega_out = -std::sqrt(-omega2_out(knum, is));
                    }
                    if (omega2_in(knum, is) >= 0.0) {
                        omega_in = std::sqrt(omega2_in(knum, is));
                    } else {
                        omega_in = -std::sqrt(-omega2_in(knum, is));
                    }
                    diff += (omega_out - omega_in) * (omega_out - omega_in);
                }
                diff2 += (omega2_out.row(knum) - omega2_in.row(knum)).norm();
            }
            diff /= static_cast<double>(nk_interpolate * ns);
            diff2 /= static_cast<double>(nk_interpolate * ns);
            diff = std::sqrt(diff);
            diff2 = std::sqrt(diff2);
            if (verbosity > 0) {
                std::cout << "  SCPH ITER " << std::setw(5) << iloop + 1 << " : ";
                std::cout << " DIFF = " << std::setw(15) << diff << '\n';
            }

            if (diff < conv_tol) {
                auto has_negative = false;

                for (ik = 0; ik < nk_interpolate; ++ik) {
                    knum = kmap_interpolate_to_scph[ik];
                    for (is = 0; is < ns; ++is) {
                        if (omega2_out(knum, is) < 0.0 && std::abs(omega2_out(knum, is)) > eps15) {
                            has_negative = true;
                            break;
                        }
                    }
                }
                if (!has_negative) {
                    if (verbosity > 0) std::cout << "  DIFF < SCPH_TOL : break SCPH loop\n";
                    break;
                }
                if (verbosity > 0) std::cout << "  DIFF < SCPH_TOL but a negative frequency is detected.\n";
            }

            omega2_prev = omega2_in;
            omega2_in = omega2_out;

            permutation1 = permutation2;
            permutation2 = permutation3;
//        omega2_in = alpha * omega2_out + (1.0 - alpha) * omega2_in;

        } // end loop iteration
    }

    if (diff < conv_tol) {
        if (verbosity > 0) {
            std::cout << " Temp = " << T_in;
            std::cout << " : convergence achieved in " << std::setw(5)
                      << iloop + 1 << " iterations.\n";
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
            if (omega2_out(ik, is) < 0.0) {
                if (std::abs(omega2_out(ik, is)) <= eps10) {
                    omega2_anharm[ik][is] = 0.0;
                } else {
                    omega2_anharm[ik][is] = omega2_out(ik, is);
                }
            } else {
                omega2_anharm[ik][is] = omega2_out(ik, is);
            }
            for (js = 0; js < ns; ++js) {
                evec_anharm_scph[ik][is][js] = evec_new[ik][is][js];
            }
        }
    }

    if (verbosity > 1) {
        std::cout << "New eigenvalues\n";
        for (ik = 0; ik < nk_interpolate; ++ik) {
            knum = kmap_interpolate_to_scph[ik];
            for (is = 0; is < ns; ++is) {
                std::cout << " ik_interpolate = " << std::setw(5) << ik + 1;
                std::cout << " is = " << std::setw(5) << is + 1;
                std::cout << " omega2 = " << std::setw(15) << omega2_anharm[knum][is] << '\n';
            }
            std::cout << '\n';
        }
    }

    deallocate(evec_new);
    deallocate(dymat_q);
    deallocate(dymat_q_HA);
}


void Scph::get_permutation_matrix(const int ns,
                                  std::complex<double> **cmat_in,
                                  Eigen::MatrixXd &permutation_matrix)
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

void Scph::update_frequency(const double temperature_in,
                            const Eigen::MatrixXd &omega2_in,
                            const std::vector<Eigen::MatrixXcd> &Fmat0,
                            const std::vector<Eigen::MatrixXcd> &evec0,
                            std::complex<double> ***dymat0,
                            std::complex<double> ***v4_array_all,
                            std::complex<double> ***cmat_convert,
//                            std::vector<Eigen::MatrixXcd> &dmat,
                            std::complex<double> ***dymat_out,
                            std::complex<double> ***evec_out,
                            const double alpha,
                            const bool offdiag,
                            Eigen::MatrixXd &omega2_out)
{
    using namespace Eigen;
    const auto nk = kmesh_dense->nk;
    const auto ns = dynamical->neval;
    const auto ns2 = ns * ns;
    const auto nk_interpolate = kmesh_coarse->nk;
    std::vector<MatrixXcd> dmat(nk);
    VectorXcd Kmat(ns);
    MatrixXcd Cmat(ns, ns), Dmat(ns, ns);
    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);
    std::complex<double> ***dymat_r_new;

    allocate(dymat_r_new, ns, ns, nk_interpolate);


    for (auto ik = 0; ik < nk; ++ik) {
        for (auto is = 0; is < ns; ++is) {
            auto omega_tmp = omega2_in(ik, is);
            if (std::abs(omega_tmp) < eps15) {
//            if (omega_tmp < eps15) {
                Kmat(is) = complex_zero;
                std::cout << "Kmat is zero for " << std::setw(4) << ik << std::setw(5) << is << " omega = " << omega_tmp
                          << '\n';
            } else {
                // Note that the missing factor 2 in the denominator of Qmat is
                // already considered in the v4_array_all.
                if (thermodynamics->classical) {
                    Kmat(is) = std::complex<double>(
                            2.0 * temperature_in * thermodynamics->T_to_Ryd / (std::abs(omega_tmp)),
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

        Dmat = Cmat * Kmat.asDiagonal() * Cmat.adjoint();
        // if iloop > 0
        dmat[ik] = Dmat.eval();
//        dmat[ik] = alpha * Dmat.eval() + (1.0 - alpha) * dmat[ik];
    }

    const auto nk_irred_interpolate = kmesh_coarse->nk_irred;

    MatrixXcd mat_tmp(ns, ns);
    SelfAdjointEigenSolver<MatrixXcd> saes;

    MatrixXcd Dymat(ns, ns);
    MatrixXcd Fmat(ns, ns);

    for (auto ik = 0; ik < nk_irred_interpolate; ++ik) {

        const auto knum_interpolate = kmesh_coarse->kpoint_irred_all[ik][0].knum;
        const auto knum = kmap_interpolate_to_scph[knum_interpolate];

        // Fmat harmonic
        Fmat = Fmat0[ik];

        // Anharmonic correction to Fmat
        if (!offdiag) {
#pragma omp parallel for
            for (int is = 0; is < ns; ++is) {
                for (auto jk = 0; jk < nk; ++jk) {
                    for (auto ks = 0; ks < ns; ++ks) {
                        Fmat(is, is) += v4_array_all[nk * ik + jk][(ns + 1) * is][(ns + 1) * ks]
                                        * dmat[jk](ks, ks);
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
                            Fmat(is, js) += v4_array_all[nk * ik + jk][ijs][ns * ks + ls]
                                            * dmat[jk](ks, ls);
                        }
                    }
                }
            }
        }

        saes.compute(Fmat);

        // New eigenvector matrix E_{new}= E_{old} * C
        mat_tmp = evec0[knum] * saes.eigenvectors();
        Dymat = mat_tmp * saes.eigenvalues().asDiagonal() * mat_tmp.adjoint();
        dynamical->symmetrize_dynamical_matrix(ik, kmesh_coarse,
                                               mat_transform_sym,
                                               Dymat);

        for (auto is = 0; is < ns; ++is) {
            for (auto js = 0; js < ns; ++js) {
                dymat_out[is][js][knum_interpolate] = Dymat(is, js);
            }
        }
    } // close loop ik

    dynamical->replicate_dymat_for_all_kpoints(kmesh_coarse, mat_transform_sym,
                                               dymat_out);

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
            fftw_plan plan = fftw_plan_dft_3d(nk1, nk2, nk3,
                                              reinterpret_cast<fftw_complex *>(dymat_out[is][js]),
                                              reinterpret_cast<fftw_complex *>(dymat_r_new[is][js]),
                                              FFTW_FORWARD, FFTW_ESTIMATE);
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
                                  mindist_list_scph,
                                  false,
                                  false);

    Eigen::MatrixXcd evec_tmp(ns, ns);

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
    deallocate(eval_tmp);
    deallocate(dymat_r_new);
}


void Scph::compute_free_energy_bubble_SCPH(const unsigned int kmesh[3],
                                           std::complex<double> ****delta_dymat_scph)
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

        size_t nsize = nk_ref * ns * ns * NT * sizeof(std::complex<double>)
                       + nk_ref * ns * NT * sizeof(double);

        const auto nsize_dble = static_cast<double>(nsize) / 1000000000.0;
        std::cout << "  Estimated memory usage per MPI process: " << std::setw(10)
                  << std::fixed << std::setprecision(4) << nsize_dble << " GByte.\n";
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
                                      mindist_list_scph);
    }

    thermodynamics->compute_FE_bubble_SCPH(eval, evec, thermodynamics->FE_bubble);

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
                                      mindist_list_scph);

        find_degeneracy(degeneracy_at_k,
                        nk_scph,
                        eval);

        if (mympi->my_rank == 0) std::cout << " Temperature (K) : " << std::setw(6) << temp << '\n';

        for (auto ik = 0; ik < nk_irred_interpolate; ++ik) {

            auto knum_interpolate = kmesh_coarse->kpoint_irred_all[ik][0].knum;
            auto knum = kmap_interpolate_to_scph[knum_interpolate];

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

                        auto se_bubble = get_bubble_selfenergy(kmesh_dense,
                                                               ns,
                                                               eval,
                                                               evec,
                                                               knum,
                                                               snum,
                                                               temp,
                                                               omegalist);

                        if (mympi->my_rank == 0) real_self[snum] = se_bubble[0].real();

                    } else if (bubble == 2) {

                        omegalist.push_back(eval[knum][snum] + im * epsilon);

                        auto se_bubble = get_bubble_selfenergy(kmesh_dense,
                                                               ns,
                                                               eval,
                                                               evec,
                                                               knum,
                                                               snum,
                                                               temp,
                                                               omegalist);

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

                        auto se_bubble = get_bubble_selfenergy(kmesh_dense,
                                                               ns,
                                                               eval,
                                                               evec,
                                                               knum,
                                                               snum,
                                                               temp,
                                                               omegalist);

                        if (mympi->my_rank == 0) {

                            std::vector<double> nonlinear_func(nomega);
                            for (auto iomega = 0; iomega < nomega; ++iomega) {
                                nonlinear_func[iomega] = omegalist[iomega].real() * omegalist[iomega].real()
                                                         - eval[knum][snum] * eval[knum][snum]
                                                         + 2.0 * eval[knum][snum] * se_bubble[iomega].real();
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

                                double omega_solution = omegalist[root_index[0] + 1].real()
                                                        - nonlinear_func[root_index[0] + 1]
                                                          * domega / (nonlinear_func[root_index[0] + 1] -
                                                                      nonlinear_func[root_index[0]]);

                                real_self[snum] = (se_bubble[root_index[0] + 1].real()
                                                   - se_bubble[root_index[0]].real())
                                                  * (omega_solution - omegalist[root_index[0] + 1].real()) / domega
                                                  + se_bubble[root_index[0] + 1].real();
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
                    eval_bubble[iT][knum][snum] = eval[knum][snum] * eval[knum][snum]
                                                  - 2.0 * eval[knum][snum] * real_self[snum];
                    for (auto jk = 1; jk < kmesh_coarse->kpoint_irred_all[ik].size(); ++jk) {
                        auto knum2 = kmap_interpolate_to_scph[kmesh_coarse->kpoint_irred_all[ik][jk].knum];
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
                                                kmap_interpolate_to_scph);
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
                                                              const unsigned int ns_in,
                                                              const double *const *eval_in,
                                                              const std::complex<double> *const *const *evec_in,
                                                              const unsigned int knum,
                                                              const unsigned int snum,
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

        for (auto m = 0; m < 3; ++m) xk_tmp[m] = kmesh_in->xk[knum][m] - kmesh_in->xk[ik1][m];
        auto ik2 = kmesh_in->get_knum(xk_tmp);

        double omega1 = eval_in[ik1][is1];
        double omega2 = eval_in[ik2][is2];

        arr_cubic[1] = ns_in * ik1 + is1;
        arr_cubic[2] = ns_in * ik2 + is2;

        double v3_tmp = std::norm(anharmonic_core->V3(arr_cubic,
                                                      kmesh_in->xk,
                                                      eval_in,
                                                      evec_in,
                                                      phase_factor_scph));

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


void Scph::write_anharmonic_correction_fc2(std::complex<double> ****delta_dymat,
                                           const unsigned int NT,
                                           const KpointMeshUniform *kmesh_coarse_in,
                                           MinimumDistList ***mindist_list_in,
                                           const bool is_qha,
                                           const int type)
{
    unsigned int i, j;
    const auto Tmin = system->Tmin;
    const auto dT = system->dT;
    double ***delta_fc2;
    Eigen::MatrixXd xtmp;
    const auto ns = dynamical->neval;
    unsigned int is, js, icell;
    unsigned int iat, jat;

    std::string file_fc2;
    std::ofstream ofs_fc2;

    if (is_qha) {
        file_fc2 = input->job_title + ".qha_dfc2";
    } else {
        if (type == 0) {
            file_fc2 = input->job_title + ".scph_dfc2";
        } else if (type == 1) {
            file_fc2 = input->job_title + ".scph+bubble(0)_dfc2";
        } else if (type == 2) {
            file_fc2 = input->job_title + ".scph+bubble(w)_dfc2";
        } else if (type == 3) {
            file_fc2 = input->job_title + ".scph+bubble(wQP)_dfc2";
        }
    }

    ofs_fc2.open(file_fc2.c_str(), std::ios::out);
    if (!ofs_fc2)
        exit("write_anharmonic_correction_fc2",
             "Cannot open file_fc2");

    const auto ncell = kmesh_coarse_in->nk_i[0] * kmesh_coarse_in->nk_i[1] * kmesh_coarse_in->nk_i[2];

    allocate(delta_fc2, ns, ns, ncell);

    xtmp.resize(system->get_primcell().number_of_atoms, 3);
    //allocate(xtmp, system->get_primcell().number_of_atoms, 3);

    ofs_fc2.precision(10);

    for (i = 0; i < system->get_primcell().number_of_atoms; ++i) {
        for (j = 0; j < 3; ++j) {
            xtmp(i, j) = system->get_supercell(0).x_fractional(system->get_map_p2s(0)[i][0], j);
        }
    }
    xtmp = xtmp * system->get_supercell(0).lattice_vector.transpose();
    xtmp = xtmp * system->get_primcell().lattice_vector.inverse().transpose();

    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            ofs_fc2 << std::setw(20) << system->get_primcell().lattice_vector(j, i);
        }
        ofs_fc2 << '\n';
    }
    ofs_fc2 << std::setw(5) << system->get_primcell().number_of_atoms << std::setw(5)
            << system->get_primcell().number_of_elems << '\n';
    for (i = 0; i < system->get_primcell().number_of_elems; ++i) {
        ofs_fc2 << std::setw(5) << system->symbol_kd[i];
    }
    ofs_fc2 << '\n';

    for (i = 0; i < system->get_primcell().number_of_atoms; ++i) {
        for (j = 0; j < 3; ++j) {
            ofs_fc2 << std::setw(20) << xtmp(i, j);
        }
        ofs_fc2 << std::setw(5) << system->get_primcell().kind[i] + 1 << '\n';
    }

    //deallocate(xtmp);

    for (unsigned int iT = 0; iT < NT; ++iT) {
        const auto temp = Tmin + dT * static_cast<double>(iT);

        ofs_fc2 << "# Temp = " << temp << '\n';

        for (is = 0; is < ns; ++is) {
            iat = is / 3;

            for (js = 0; js < ns; ++js) {
                jat = js / 3;

                for (icell = 0; icell < ncell; ++icell) {
                    delta_fc2[is][js][icell]
                            = delta_dymat[iT][is][js][icell].real()
                              * std::sqrt(system->get_mass_super()[system->get_map_p2s(0)[iat][0]]
                                          * system->get_mass_super()[system->get_map_p2s(0)[jat][0]]);
                }

            }
        }

        for (icell = 0; icell < ncell; ++icell) {

            for (is = 0; is < ns; ++is) {
                iat = is / 3;
                const auto icrd = is % 3;

                for (js = 0; js < ns; ++js) {
                    jat = js / 3;
                    const auto jcrd = js % 3;

                    const auto nmulti = mindist_list_in[iat][jat][icell].shift.size();

                    for (auto it = mindist_list_in[iat][jat][icell].shift.cbegin();
                         it != mindist_list_in[iat][jat][icell].shift.cend(); ++it) {

                        ofs_fc2 << std::setw(4) << (*it).sx;
                        ofs_fc2 << std::setw(4) << (*it).sy;
                        ofs_fc2 << std::setw(4) << (*it).sz;
                        ofs_fc2 << std::setw(5) << iat << std::setw(3) << icrd;
                        ofs_fc2 << std::setw(4) << jat << std::setw(3) << jcrd;
                        ofs_fc2 << std::setprecision(15) << std::setw(25)
                                << delta_fc2[is][js][icell] / static_cast<double>(nmulti) << '\n';

                    }

                }
            }
        }

        ofs_fc2 << '\n';
    }

    deallocate(delta_fc2);

    ofs_fc2.close();
    std::cout << "  " << std::setw(input->job_title.length() + 12) << std::left << file_fc2;

    if (is_qha) {
        std::cout << " : Anharmonic corrections to the second-order IFCs (QHA)\n";
    } else {
        if (type == 0) {
            std::cout << " : Anharmonic corrections to the second-order IFCs (SCPH)\n";
        } else if (type == 1) {
            std::cout << " : Anharmonic corrections to the second-order IFCs (SCPH+Bubble(0))\n";
        } else if (type == 2) {
            std::cout << " : Anharmonic corrections to the second-order IFCs (SCPH+Bubble(w))\n";
        } else if (type == 3) {
            std::cout << " : Anharmonic corrections to the second-order IFCs (SCPH+Bubble(wQP))\n";
        }
    }
}

void Scph::mpi_bcast_complex(std::complex<double> ****data,
                             const unsigned int NT,
                             const unsigned int nk,
                             const unsigned int ns)
{
    const int _NT = static_cast<int>(NT);
    const int _nk = static_cast<int>(nk);
    const int _ns = static_cast<int>(ns);

#ifdef MPI_CXX_DOUBLE_COMPLEX
    MPI_Bcast(&data[0][0][0][0], _NT * _nk * _ns * _ns,
              MPI_CXX_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
#elif defined MPI_DOUBLE_COMPLEX
                                                                                                                            MPI_Bcast(&data[0][0][0][0], _NT * _nk * _ns * _ns, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
#else
    MPI_Bcast(&data[0][0][0][0], _NT * _nk * _ns * _ns, MPI_COMPLEX16, 0, MPI_COMM_WORLD);
#endif
}

void Scph::get_derivative_central_diff(const double delta_t,
                                       const unsigned int nk,
                                       double **omega0,
                                       double **omega2,
                                       double **domega_dt)
{
    const auto ns = dynamical->neval;
    const auto inv_dt = 1.0 / (2.0 * delta_t);
    for (auto ik = 0; ik < nk; ++ik) {
        for (auto is = 0; is < ns; ++is) {
            domega_dt[ik][is] = (omega2[ik][is] - omega0[ik][is]) * inv_dt;
            //    std::cout << "domega_dt = " << domega_dt[ik][is] << '\n';
        }
    }
}







