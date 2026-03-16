/*
qha.cpp

Copyright (c) 2022 Ryota Masuki, Terumasa Tadano

This file is distributed under the terms of the MIT license.
Please see the file 'LICENCE.txt' in the root directory
or http://opensource.org/licenses/mit-license.php for information.
*/

#include "qha.h"
#include <Eigen/Core>
#include <iomanip>
#include "constants.h"
#include "dynamical.h"
#include "error.h"
#include "mathfunctions.h"
#include "mpi_common.h"
#include "parsephon.h"
#include "relaxation.h"
#include "system.h"
#include "thermodynamics.h"
#include "timer.h"
#include "write_phonons.h"

using namespace PHON_NS;

Qha::Qha(PHON *phon) : ScphQhaCommon(phon)
{
    set_default_variables();
}

Qha::~Qha()
{
    deallocate_variables();
}

void Qha::set_default_variables()
{
    restart_qha = false;
    qha_scheme = 0;

    initialize_variables();
}


void Qha::setup_qha()
{
    setup_kmesh(kmesh_qha,
                kmesh_interpolate,
                "QHA",
                "KMESH_QHA should be a integral multiple of KMESH_INTERPOLATE.");    
    setup_eigvecs();
    const auto relax_str = relaxation->relax_str;
    setup_pp_interaction(relax_str > 0);

    setup_structural_data();
}


void Qha::exec_qha_optimization()
{
    const auto ns = dynamical->neval;
    const auto Tmin = system->Tmin;
    const auto Tmax = system->Tmax;
    const auto dT = system->dT;
    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;

    std::complex<double> ****delta_dymat_qha = nullptr;
    std::complex<double> ****delta_harmonic_dymat_renormalize = nullptr;
    allocate(delta_dymat_qha, NT, ns, ns, kmesh_coarse->nk);
    allocate(delta_harmonic_dymat_renormalize, NT, ns, ns, kmesh_coarse->nk);

    const auto relax_str = relaxation->relax_str;

    zerofill_harmonic_dymat_renormalize(delta_harmonic_dymat_renormalize, NT);

    if (restart_qha) {

        if (mympi->my_rank == 0) {
            std::cout << " RESTART_QHA is true.\n";
            std::cout << " Dynamical matrix is read from file ...";
        }

        load_scph_dymat_from_file(delta_dymat_qha,
                                  input->job_title + ".renorm_harm_dymat",
                                  kmesh_dense,
                                  kmesh_coarse,
                                  dynamical->nonanalytic,
                                  true);
        load_scph_dymat_from_file(delta_harmonic_dymat_renormalize,
                                  input->job_title + ".renorm_harm_dymat",
                                  kmesh_dense,
                                  kmesh_coarse,
                                  dynamical->nonanalytic,
                                  true);

        // structural optimization
        if (relax_str != 0) {
            relaxation->load_V0_from_file();
        }
    } else {

        // QHA + structural optimization
        if (relax_str == 1 || relax_str == 2) {
            exec_QHA_relax_main(delta_dymat_qha, delta_harmonic_dymat_renormalize);
        }
        // lowest-order QHA
        else if (relax_str == 3)
        {
            exec_perturbative_QHA(delta_dymat_qha, delta_harmonic_dymat_renormalize);
        }

        if (mympi->my_rank == 0) {
            // write dymat to file
            // write renormalized harmonic dynamical matrix when the crystal structure is optimized
            if (relax_str != 0) {
                store_scph_dymat_to_file(delta_harmonic_dymat_renormalize,
                                         input->job_title + ".renorm_harm_dymat",
                                         kmesh_dense,
                                         kmesh_coarse,
                                         dynamical->nonanalytic,
                                         true);
                relaxation->store_V0_to_file();
            }
            write_anharmonic_correction_fc2(delta_dymat_qha, NT, kmesh_coarse, mindist_list, true);
        }
    }

    postprocess(delta_dymat_qha,
                delta_harmonic_dymat_renormalize,
                delta_dymat_qha,
                kmesh_coarse,
                mindist_list,
                true,
                0);

    deallocate(delta_dymat_qha);
    deallocate(delta_harmonic_dymat_renormalize);
}

void Qha::exec_QHA_relax_main(std::complex<double> ****dymat_anharm,
                              std::complex<double> ****delta_harmonic_dymat_renormalize)
{
    using namespace Eigen;

    int ik, is, js;
    int is1, is2, i1;
    int iat1, ixyz1, ixyz2;
    std::string str_tmp;
    static auto complex_zero = std::complex<double>(0.0, 0.0);

    const auto nk = kmesh_dense->nk;
    const auto nk_interpolate = kmesh_coarse->nk;
    const auto ns = dynamical->neval;
    const auto nk_irred_interpolate = kmesh_coarse->nk_irred;
    const auto Tmin = system->Tmin;
    const auto Tmax = system->Tmax;
    const auto dT = system->dT;
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
    double **C2_array_ZSISA;

    // strain-derivative of k-space IFCs
    // (calculated by real-space IFC renormalization or finite-difference method)
    std::complex<double> **del_v1_del_umn;
    std::complex<double> **del2_v1_del_umn2;
    std::complex<double> **del3_v1_del_umn3;
    std::complex<double> ***del_v2_del_umn;
    std::complex<double> ***del2_v2_del_umn2;
    std::complex<double> ****del_v3_del_umn;

    std::complex<double> *del_v0_del_umn_renorm;
    std::complex<double> **del_v1_del_umn_renorm;
    double **C2_array_renorm;

    // atomic forces and stress tensor at finite temperatures
    std::complex<double> *v1_QHA;
    std::complex<double> *del_v0_del_umn_QHA;
    std::complex<double> *del_v0_del_umn_ZSISA;
    std::complex<double> *del_v0_del_umn_vZSISA;

    double **delq_delu_ZSISA;

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
    //    double pvcell = 0.0; // pressure * v_{cell,reference} [Ry]
    auto pvcell = relaxation->stat_pressure * system->get_primcell().volume * std::pow(Bohr_in_Angstrom, 3) *
                  1.0e-30; // in 10^9 J = GJ
    pvcell *= 1.0e9 / Ryd; // in Ry

    // temperature grid
    std::vector<double> vec_temp;
    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;

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

    allocate(v1_QHA, ns);
    allocate(del_v0_del_umn_renorm, 9);
    allocate(del_v0_del_umn_QHA, 9);
    allocate(del_v0_del_umn_ZSISA, 9);
    allocate(del_v0_del_umn_vZSISA, 9);
    allocate(del_v1_del_umn_renorm, 9, ns);

    allocate(delq_delu_ZSISA, ns, 9);
    allocate(C2_array_renorm, 9, 9);

    // Compute matrix element of 4-phonon interaction
    allocate(v4_ref, nk_irred_interpolate * kmesh_dense->nk, ns * ns, ns * ns);
    allocate(v4_renorm, nk_irred_interpolate * kmesh_dense->nk, ns * ns, ns * ns);

    allocate(v4_with_umn, nk_irred_interpolate * kmesh_dense->nk, ns * ns, ns * ns);

    relaxation->create_optimizer(ns);


    // Calculate v4 array.
    // This operation is the most expensive part of the calculation.
    if (use_band_parallel_v4()) {
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
                                            true,
                                            relaxation->relax_str,
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

    // assume that the atomic forces are zero at initial structure
    for (is = 0; is < ns; is++) {
        v1_ref[is] = 0.0;
    }

    // compute IFC renormalization by lattice relaxation
    if (mympi->my_rank == 0) {
        std::cout << " RELAX_STR = " << relaxation->relax_str << ": ";
        if (relaxation->relax_str == 1) {
            std::cout << "Set zeros in derivatives of k-space IFCs by strain.\n\n";
        }
        if (relaxation->relax_str == 2) {
            std::cout << "Calculating derivatives of k-space IFCs by strain.\n\n";
        }
    }

    allocate(del_v1_del_umn, 9, ns);
    allocate(del2_v1_del_umn2, 81, ns);
    allocate(del3_v1_del_umn3, 729, ns);
    allocate(del_v2_del_umn, 9, nk, ns * ns);
    allocate(del2_v2_del_umn2, 81, nk, ns * ns);
    allocate(del_v3_del_umn, 9, nk, ns, ns * ns);

    relaxation->compute_del_v_strain(kmesh_coarse,
                                     kmesh_dense,
                                     del_v1_del_umn,
                                     del2_v1_del_umn2,
                                     del3_v1_del_umn3,
                                     del_v2_del_umn,
                                     del2_v2_del_umn2,
                                     del_v3_del_umn,
                                     omega2_harmonic,
                                     evec_harmonic,
                                     relaxation->relax_str,
                                     mindist_list,
                                     phase_factor);


    // get indices of optical modes at Gamma point
    js = 0;
    for (is = 0; is < ns; is++) {
        if (std::fabs(omega2_harmonic[0][is]) < eps8) {
            continue;
        }
        harm_optical_modes[js] = is;
        js++;
    }
    if (js != ns - 3) {
        exit("exec_scph_relax_cell_coordinate_main", "The number of detected optical modes is not ns-3.");
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
        allocate(C2_array_ZSISA, 9, 9);

        relaxation->set_elastic_constants(C1_array, C2_array, C3_array);

        // output files of structural optimization
        std::ofstream fout_step_q0, fout_step_u0;
        std::ofstream fout_q0, fout_u0;
        std::ofstream fout_step_u_tensor, fout_u_tensor;

        fout_step_q0.open("step_q0.txt");
        fout_step_u0.open("step_u0.txt");
        fout_q0.open(input->job_title + ".normal_disp");
        fout_u0.open(input->job_title + ".atom_disp");
        // if the unit cell is relaxed
        if (relaxation->relax_str == 2) {
            fout_step_u_tensor.open("step_u_tensor.txt");
            fout_u_tensor.open(input->job_title + ".umn_tensor");
        }

        relaxation->write_resfile_header(fout_q0, fout_u0, fout_u_tensor);

        i_temp_loop = -1;

        std::cout << " Start structural optimization.\n";
        if (relaxation->relax_str == 1) {
            std::cout << "  Internal coordinates are relaxed.\n";
            std::cout << "  Shape of the unit cell is fixed.\n\n";
        } else if (relaxation->relax_str == 2) {
            std::cout << "  Internal coordinates and shape of the unit cell are relaxed.\n\n";
        }


        for (double temp: vec_temp) {
            i_temp_loop++;
            auto iT = static_cast<unsigned int>((temp - Tmin) / dT);

            std::cout << " ----------------------------------------------------------------\n";
            std::cout << " Temperature = " << temp << " K\n";
            std::cout << " Temperature index : " << std::setw(4) << i_temp_loop << "/" << std::setw(4) << NT << "\n\n";

            relaxation->set_init_structure_atT(q0,
                                               u_tensor,
                                               u0,
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

            if (relaxation->relax_str == 2) {
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

            relaxation->write_stepresfile(q0, u_tensor, u0, 0, fout_step_q0, fout_step_u0, fout_step_u_tensor);

            std::cout << " ----------------------------------------------------------------\n";

            std::cout << " Start structural optimization at " << temp << " K.\n";

            for (i_str_loop = 0; i_str_loop < relaxation->max_str_iter; i_str_loop++) {

                std::cout << " Structure loop :" << std::setw(5) << i_str_loop + 1;

                // get eta tensor
                relaxation->calculate_eta_tensor(eta_tensor, u_tensor);

                // std::cout << "Eta tensor:\n";
                // for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                //     std::cout << " ";
                //     for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                //         std::cout << std::scientific << std::setw(15) << std::setprecision(6) << eta_tensor[ixyz1][ixyz2];
                //     }
                //     std::cout << '\n';
                // }

                // calculate IFCs under strain
                relaxation->renormalize_v0_from_umn(v0_with_umn,
                                                    v0_ref,
                                                    eta_tensor,
                                                    C1_array,
                                                    C2_array,
                                                    C3_array,
                                                    u_tensor,
                                                    pvcell);

                relaxation->renormalize_v1_from_umn(v1_with_umn,
                                                    v1_ref,
                                                    del_v1_del_umn,
                                                    del2_v1_del_umn2,
                                                    del3_v1_del_umn3,
                                                    u_tensor);

                relaxation->renormalize_v2_from_umn(kmesh_coarse,
                                                    kmap_coarse_to_dense,
                                                    delta_v2_with_umn,
                                                    del_v2_del_umn,
                                                    del2_v2_del_umn2,
                                                    u_tensor);

                relaxation
                    ->renormalize_v3_from_umn(kmesh_coarse, kmesh_dense, v3_with_umn, v3_ref, del_v3_del_umn, u_tensor);

                for (ik = 0; ik < nk_irred_interpolate * nk; ik++) {
                    for (is = 0; is < ns * ns; is++) {
                        for (is1 = 0; is1 < ns * ns; is1++) {
                            v4_with_umn[ik][is][is1] = v4_ref[ik][is][is1];
                        }
                    }
                }

                // std::cout << "V0 with strain = " << v0_with_umn << '\n';
                // std::cout << "V1_with_umn\n";
                // for (i1 = 0; i1 < ns; ++i1) {
                //     std::cout << "v1_with_umn[" << i1 << "] = " << v1_with_umn[i1] << '\n';
                // }
                // std::cout << "delta_v2_with_umn\n";
                // for (i1 = 0; i1 < nk_interpolate; ++i1) {
                //     for (size_t i2 = 0; i2 < ns * ns; ++i2) {
                //         if (std::abs(delta_v2_with_umn[i1][i2]) > eps8) {
                //             std::cout << "delta_v2_with_umn[" << i1 << "][" << i2 << "] = " << delta_v2_with_umn[i1][i2] << '\n';
                //         }
                //     }
                // }

                //renormalize IFC
                relaxation->renormalize_v1_from_q0(omega2_harmonic,
                                                   kmesh_coarse,
                                                   kmesh_dense,
                                                   v1_renorm,
                                                   v1_with_umn,
                                                   delta_v2_with_umn,
                                                   v3_with_umn,
                                                   v4_with_umn,
                                                   q0);
                relaxation->renormalize_v2_from_q0(evec_harmonic,
                                                   kmesh_coarse,
                                                   kmesh_dense,
                                                   kmap_coarse_to_dense,
                                                   mat_transform_sym,
                                                   delta_v2_renorm,
                                                   delta_v2_with_umn,
                                                   v3_with_umn,
                                                   v4_with_umn,
                                                   q0);
                relaxation->renormalize_v3_from_q0(kmesh_dense, kmesh_coarse, v3_renorm, v3_with_umn, v4_with_umn, q0);
                relaxation->renormalize_v0_from_q0(omega2_harmonic,
                                                   kmesh_dense,
                                                   v0_renorm,
                                                   v0_with_umn,
                                                   v1_with_umn,
                                                   delta_v2_with_umn,
                                                   v3_with_umn,
                                                   v4_with_umn,
                                                   q0);

                // copy v4_ref to v4_renorm
                for (ik = 0; ik < nk_irred_interpolate * kmesh_dense->nk; ik++) {
                    for (is1 = 0; is1 < ns * ns; is1++) {
                        for (is2 = 0; is2 < ns * ns; is2++) {
                            v4_renorm[ik][is1][is2] = v4_ref[ik][is1][is2];
                        }
                    }
                }

                if (relaxation->relax_str == 1) {
                    for (i1 = 0; i1 < 9; i1++) {
                        del_v0_del_umn_renorm[i1] = 0.0;
                    }
                    for (i1 = 0; i1 < 9; i1++) {
                        for (is1 = 0; is1 < ns; is1++) {
                            del_v1_del_umn_renorm[i1][is1] = complex_zero;
                        }
                    }

                } else if (relaxation->relax_str == 2) {
                    // calculate renormalized stress tensor
                    calculate_del_v0_del_umn_renorm(del_v0_del_umn_renorm,
                                                    C1_array,
                                                    C2_array,
                                                    C3_array,
                                                    eta_tensor,
                                                    u_tensor,
                                                    del_v1_del_umn,
                                                    del2_v1_del_umn2,
                                                    del3_v1_del_umn3,
                                                    del_v2_del_umn,
                                                    del2_v2_del_umn2,
                                                    del_v3_del_umn,
                                                    q0,
                                                    pvcell,
                                                    kmesh_dense);
                    // std::cout << "\n";
                    // for (i1 = 0; i1 < 9; ++i1) {
                    //     std::cout << "del_v0_del_umn_renorm[" << i1 << "] = " << del_v0_del_umn_renorm[i1] << '\n';
                    // }
                    // calculate renormalized strain-force coupling for ZSISA and v-ZSISA.
                    calculate_del_v1_del_umn_renorm(del_v1_del_umn_renorm,
                                                    u_tensor,
                                                    del_v1_del_umn,
                                                    del2_v1_del_umn2,
                                                    del3_v1_del_umn3,
                                                    del_v2_del_umn,
                                                    del2_v2_del_umn2,
                                                    del_v3_del_umn,
                                                    q0);
                    // std::cout << "\n";
                    // for (i1 = 0; i1 < 9; ++i1) {
                    //     for (is1 = 0; is1 < ns; ++is1) {
                    //         std::cout << "del_v1_del_umn_renorm[" << i1 << "][" << is1 << "] = " << del_v1_del_umn_renorm[i1][is1] << '\n';
                    //     }
                    // }
                }

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
                // delta_harmonic_dymat_renormalize is copied to dymat_anharm after structure convergence,
                // which is required for postprocess.

                compute_cmat(cmat_convert, evec_harm_renorm_tmp);

                // The same functions (compute_anharmonic_v1_array, compute_anharmonic_del_v0_del_umn) as
                // in Scph::exec_scph_relax_cell_coordinate_main can be used for
                // calculating the finite-temperature forces and stress tensor.
                // This is because we truncate the Taylor expansion of PES at the fourth order (?)
                compute_anharmonic_v1_array(v1_QHA,
                                            v1_renorm,
                                            v3_renorm,
                                            cmat_convert,
                                            omega2_harm_renorm[iT],
                                            temp,
                                            kmesh_dense);

                if (relaxation->relax_str == 1) {
                    for (i1 = 0; i1 < 9; i1++) {
                        del_v0_del_umn_QHA[i1] = complex_zero;
                    }
                } else if (relaxation->relax_str == 2) {
                    compute_anharmonic_del_v0_del_umn(del_v0_del_umn_QHA,
                                                      del_v0_del_umn_renorm,
                                                      del_v2_del_umn,
                                                      del2_v2_del_umn2,
                                                      del_v3_del_umn,
                                                      u_tensor,
                                                      q0,
                                                      cmat_convert,
                                                      omega2_harm_renorm[iT],
                                                      temp,
                                                      kmesh_dense);

                    compute_ZSISA_stress(delq_delu_ZSISA,
                                         del_v0_del_umn_ZSISA,
                                         cmat_convert,
                                         omega2_harm_renorm[iT],
                                         del_v0_del_umn_QHA,
                                         del_v1_del_umn_renorm,
                                         v1_QHA,
                                         harm_optical_modes);

                    // qha_scheme == 1 : ZSISA
                    if (qha_scheme == 1) {
                        // overwrite v1_QHA by zero-temperature first-order IFCs.
                        for (is = 0; is < ns; is++) {
                            v1_QHA[is] = v1_renorm[is];
                        }
                        // overwrite finite-temperature stress tensor
                        for (i1 = 0; i1 < 9; i1++) {
                            del_v0_del_umn_QHA[i1] = del_v0_del_umn_ZSISA[i1];
                        }
                    }

                    // calculate renormalized second-order elastic constants
                    calculate_C2_array_renorm(C2_array_renorm,
                                              u_tensor,
                                              eta_tensor,
                                              C2_array,
                                              C3_array,
                                              del2_v1_del_umn2,
                                              del3_v1_del_umn3,
                                              del2_v2_del_umn2,
                                              q0);

                    calculate_C2_array_ZSISA(C2_array_ZSISA, C2_array_renorm, del_v1_del_umn_renorm, delq_delu_ZSISA);

                    compute_vZSISA_stress(del_v0_del_umn_vZSISA,
                                          C2_array_ZSISA,
                                          del_v0_del_umn_renorm,
                                          del_v0_del_umn_ZSISA,
                                          u_tensor);

                    // qha_scheme == 2 : v-ZSISA
                    // overwrite finite-temperature force and stress tensor
                    if (qha_scheme == 2) {
                        for (is = 0; is < ns; is++) {
                            v1_QHA[is] = v1_renorm[is];
                        }

                        for (i1 = 0; i1 < 9; i1++) {
                            del_v0_del_umn_QHA[i1] = del_v0_del_umn_vZSISA[i1];
                        }
                    }
                }

                relaxation->update_cell_coordinate(q0,
                                                   u0,
                                                   u_tensor,
                                                   v1_QHA,
                                                   omega2_harm_renorm[iT],
                                                   del_v0_del_umn_QHA,
                                                   C2_array,
                                                   cmat_convert,
                                                   harm_optical_modes,
                                                   delta_q0,
                                                   delta_u0,
                                                   delta_umn,
                                                   du0,
                                                   du_tensor,
                                                   omega2_harmonic,
                                                   evec_harmonic);

                relaxation->write_stepresfile(q0,
                                              u_tensor,
                                              u0,
                                              i_str_loop + 1,
                                              fout_step_q0,
                                              fout_step_u0,
                                              fout_step_u_tensor);

                relaxation->check_str_divergence(str_diverged, q0, u0, u_tensor);

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
                    if (relaxation->relax_str == 2) {
                        std::cout << " du_tensor is smaller than CELL_CONV_TOL = " << std::scientific << std::setw(15)
                                  << std::setprecision(6) << relaxation->cell_conv_tol << '\n';
                    }
                    std::cout << " Structural optimization converged in " << i_str_loop + 1 << "-th loop.\n\n";
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

            if (relaxation->relax_str == 2) {
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

            // copy delta_harmonic_dymat_renormalize to dymat_anharm
            // This process is required for postprocess.
            for (is1 = 0; is1 < ns; is1++) {
                for (is2 = 0; is2 < ns; is2++) {
                    for (ik = 0; ik < kmesh_coarse->nk; ik++) {
                        dymat_anharm[iT][is1][is2][ik] = delta_harmonic_dymat_renormalize[iT][is1][is2][ik];
                    }
                }
            }

            // print obtained structure
            relaxation->calculate_u0(q0, u0, omega2_harmonic, evec_harmonic);

            relaxation->write_resfile_atT(q0, u_tensor, u0, temp, fout_q0, fout_u0, fout_u_tensor);
        }

        // Output files of structural optimization
        fout_step_q0.close();
        fout_step_u0.close();
        fout_q0.close();
        fout_u0.close();
        if (relaxation->relax_str == 2) {
            fout_step_u_tensor.close();
            fout_u_tensor.close();
        }

        deallocate(cmat_convert);

        deallocate(C1_array);
        deallocate(C2_array);
        deallocate(C3_array);
        deallocate(C2_array_ZSISA);
    }

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

    deallocate(q0);
    deallocate(u0);
    deallocate(u_tensor);
    deallocate(eta_tensor);

    deallocate(delta_q0);
    deallocate(delta_u0);
    deallocate(delta_umn);

    deallocate(v1_QHA);
    deallocate(del_v1_del_umn_renorm);
    deallocate(del_v0_del_umn_QHA);
    deallocate(del_v0_del_umn_ZSISA);
    deallocate(del_v0_del_umn_vZSISA);
    deallocate(del_v0_del_umn_renorm);

    deallocate(delq_delu_ZSISA);

    deallocate(C2_array_renorm);
}


void Qha::exec_perturbative_QHA(std::complex<double> ****dymat_anharm,
                                std::complex<double> ****delta_harmonic_dymat_renormalize)
{
    using namespace Eigen;

    int ik, is, js, ik1, is1, is2;
    int ixyz1, ixyz2, ixyz3;
    int itmp1, itmp2, itmp3, itmp4;
    static auto complex_zero = std::complex<double>(0.0, 0.0);

    const auto nk = kmesh_dense->nk;
    const auto nk_interpolate = kmesh_coarse->nk;
    const auto ns = dynamical->neval;
    const auto nk_irred_interpolate = kmesh_coarse->nk_irred;
    const auto Tmin = system->Tmin;
    const auto Tmax = system->Tmax;
    const auto dT = system->dT;

    // renormalization of harmonic dynamical matrix
    std::complex<double> **delta_v2_renorm;
    std::complex<double> **delta_v2_with_umn;
    double ***omega2_harm_renorm;
    std::complex<double> ***evec_harm_renorm_tmp;
    // original and renormalized IFCs
    std::complex<double> *v1_ref, *v1_renorm, *v1_with_umn;
    std::complex<double> ***v3_ref;         // We fix cubic IFCs in perturbative QHA.
    std::complex<double> ***v4_array_dummy; // We set quartic IFCs as zero.

    // elastic constants
    double *C1_array;
    double **C2_array;
    double ***C3_array;

    // force and stress tensor from F_vib
    std::complex<double> *v1_vib;
    std::complex<double> *del_v0_del_umn_vib;

    // strain-derivative of k-space IFCs
    // (calculated by real-space IFC renormalization or finite-difference method)
    std::complex<double> **del_v1_del_umn;
    std::complex<double> **del2_v1_del_umn2;
    std::complex<double> **del3_v1_del_umn3_dummy;
    std::complex<double> ***del_v2_del_umn;
    std::complex<double> ***del2_v2_del_umn2_dummy;
    std::complex<double> ****del_v3_del_umn_dummy;

    // IFC renormalization
    double v0_with_umn, v0_renorm;

    // structural optimization
    int i_temp_loop;
    double *q0, *u0;
    double **u_tensor, **eta_tensor;
    std::vector<int> harm_optical_modes(ns - 3);

    // temperature grid
    std::vector<double> vec_temp;
    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;

    MatrixXcd elastic_mat_tmp(ns - 3 + 6, ns - 3 + 6); // optical phonons + independent strain
    VectorXcd q0_umn(ns - 3 + 6), del_Fvib_q0_umn(ns - 3 + 6);

    allocate(omega2_harm_renorm, NT, nk, ns);
    allocate(evec_harm_renorm_tmp, nk, ns, ns);
    allocate(delta_v2_renorm, nk_interpolate, ns * ns);
    allocate(delta_v2_with_umn, nk_interpolate, ns * ns);

    allocate(v1_ref, ns);
    allocate(v1_with_umn, ns);
    allocate(v1_renorm, ns);

    allocate(v1_vib, ns);
    allocate(del_v0_del_umn_vib, 9);

    allocate(q0, ns);
    allocate(u0, ns);
    allocate(u_tensor, 3, 3);
    allocate(eta_tensor, 3, 3);

    allocate(v4_array_dummy, nk_irred_interpolate * kmesh_dense->nk, ns * ns, ns * ns);

    for (ik1 = 0; ik1 < nk_irred_interpolate * kmesh_dense->nk; ik1++) {
        for (is1 = 0; is1 < ns * ns; is1++) {
            for (is2 = 0; is2 < ns * ns; is2++) {
                v4_array_dummy[ik1][is1][is2] = complex_zero;
            }
        }
    }

    allocate(v3_ref, nk, ns, ns * ns);

    compute_V3_elements_mpi_over_kpoint(v3_ref,
                                        omega2_harmonic,
                                        evec_harmonic,
                                        selfenergy_offdiagonal,
                                        kmesh_coarse,
                                        kmesh_dense,
                                        phase_factor,
                                        phi3_reciprocal);

    // assume that the atomic forces are zero at initial structure
    for (is = 0; is < ns; is++) {
        v1_ref[is] = 0.0;
    }

    allocate(del_v1_del_umn, 9, ns);
    allocate(del2_v1_del_umn2, 81, ns);
    allocate(del_v2_del_umn, 9, nk, ns * ns);

    allocate(del3_v1_del_umn3_dummy, 729, ns);
    allocate(del2_v2_del_umn2_dummy, 81, nk, ns * ns);
    allocate(del_v3_del_umn_dummy, 9, nk, ns, ns * ns);

    relaxation->compute_del_v_strain(kmesh_coarse,
                                     kmesh_dense,
                                     del_v1_del_umn,
                                     del2_v1_del_umn2,
                                     nullptr,
                                     del_v2_del_umn,
                                     nullptr,
                                     nullptr,
                                     omega2_harmonic,
                                     evec_harmonic,
                                     relaxation->relax_str,
                                     mindist_list,
                                     phase_factor);

    // set dummy variables as zero.
    // These are used for the preparation for the postprocess
    for (ixyz1 = 0; ixyz1 < 729; ixyz1++) {
        for (is1 = 0; is1 < ns; is1++) {
            del3_v1_del_umn3_dummy[ixyz1][is1] = complex_zero;
        }
    }
    for (ixyz1 = 0; ixyz1 < 81; ixyz1++) {
        for (ik1 = 0; ik1 < nk; ik1++) {
            for (is1 = 0; is1 < ns * ns; is1++) {
                del2_v2_del_umn2_dummy[ixyz1][ik1][is1] = complex_zero;
            }
        }
    }
    for (ixyz1 = 0; ixyz1 < 9; ixyz1++) {
        for (ik1 = 0; ik1 < nk; ik1++) {
            for (is1 = 0; is1 < ns; is1++) {
                for (is2 = 0; is2 < ns * ns; is2++) {
                    del_v3_del_umn_dummy[ixyz1][ik1][is1][is2] = complex_zero;
                }
            }
        }
    }

    allocate(C1_array, 9);
    allocate(C2_array, 9, 9);
    allocate(C3_array, 9, 9, 9);

    // get indices of optical modes at Gamma point
    js = 0;
    for (is = 0; is < ns; is++) {
        if (std::fabs(omega2_harmonic[0][is]) < eps8) {
            continue;
        }
        harm_optical_modes[js] = is;
        js++;
    }
    if (js != ns - 3) {
        exit("exec_scph_relax_cell_coordinate_main", "The number of detected optical modes is not ns-3.");
    }

    if (mympi->my_rank == 0) {

        dynamical->precompute_dymat_harm(kmesh_dense->nk,
                                         kmesh_dense->xk,
                                         kmesh_dense->kvec_na,
                                         dymat_harm_short,
                                         dymat_harm_long);


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

        // set elastic constants
        relaxation->set_elastic_constants(C1_array, C2_array, C3_array);

        // output files of structural optimization
        std::ofstream fout_q0, fout_u0, fout_u_tensor;

        fout_q0.open(input->job_title + ".normal_disp");
        fout_u0.open(input->job_title + ".atom_disp");
        fout_u_tensor.open(input->job_title + ".umn_tensor");

        relaxation->write_resfile_header(fout_q0, fout_u0, fout_u_tensor);

        i_temp_loop = -1;

        std::cout << " Start QHA calculation.\n";
        std::cout
            << " Internal coordinates and shape of the unit cell are calculated by lowest-order perturbation theory ...\n\n";

        for (double temp: vec_temp) {
            i_temp_loop++;
            auto iT = static_cast<unsigned int>((temp - Tmin) / dT);

            calc_v1_vib(v1_vib, v3_ref, temp);
            calc_del_v0_del_umn_vib(del_v0_del_umn_vib, del_v2_del_umn, temp);

            // calculate matrix
            // elastic_mat_tmp
            for (itmp1 = 0; itmp1 < ns + 3; itmp1++) {
                for (itmp2 = 0; itmp2 < ns + 3; itmp2++) {
                    elastic_mat_tmp(itmp1, itmp2) = complex_zero;
                }
            }

            for (is1 = 0; is1 < ns - 3; is1++) {
                elastic_mat_tmp(is1, is1) = omega2_harmonic[0][harm_optical_modes[is1]];
            }

            for (is1 = 0; is1 < ns - 3; is1++) {
                is2 = harm_optical_modes[is1];
                for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                    elastic_mat_tmp(is1, ns - 3 + ixyz1) = del_v1_del_umn[ixyz1 * 3 + ixyz1][is2];
                    elastic_mat_tmp(ns - 3 + ixyz1, is1) = elastic_mat_tmp(is1, ns - 3 + ixyz1);
                }

                for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                    ixyz2 = (ixyz1 + 1) % 3;
                    ixyz3 = (ixyz1 + 2) % 3;

                    elastic_mat_tmp(is1, ns + ixyz1) =
                        del_v1_del_umn[ixyz2 * 3 + ixyz3][is2] + del_v1_del_umn[ixyz3 * 3 + ixyz2][is2];
                    elastic_mat_tmp(ns + ixyz1, is1) = elastic_mat_tmp(is1, ns + ixyz1);
                }
            }

            for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                    itmp1 = (ixyz1 + 1) % 3;
                    itmp2 = (ixyz1 + 2) % 3;
                    itmp3 = (ixyz2 + 1) % 3;
                    itmp4 = (ixyz2 + 2) % 3;

                    elastic_mat_tmp(ns - 3 + ixyz1, ns - 3 + ixyz2) = C2_array[ixyz1 * 4][ixyz2 * 4];

                    elastic_mat_tmp(ns - 3 + ixyz1, ns + ixyz2) =
                        C2_array[ixyz1 * 4][itmp3 * 3 + itmp4] + C2_array[ixyz1 * 4][itmp4 * 3 + itmp3];
                    elastic_mat_tmp(ns + ixyz1, ns - 3 + ixyz2) =
                        C2_array[itmp1 * 3 + itmp2][ixyz2 * 4] + C2_array[itmp2 * 3 + itmp1][ixyz2 * 4];

                    elastic_mat_tmp(ns + ixyz1, ns + ixyz2) = C2_array[itmp1 * 3 + itmp2][itmp3 * 3 + itmp4] +
                                                              C2_array[itmp2 * 3 + itmp1][itmp3 * 3 + itmp4] +
                                                              C2_array[itmp1 * 3 + itmp2][itmp4 * 3 + itmp3] +
                                                              C2_array[itmp2 * 3 + itmp1][itmp4 * 3 + itmp3];
                }
            }

            for (is1 = 0; is1 < ns - 3; is1++) {
                is2 = harm_optical_modes[is1];
                del_Fvib_q0_umn(is1) = -v1_vib[is2];
            }
            for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                ixyz2 = (ixyz1 + 1) % 3;
                ixyz3 = (ixyz1 + 2) % 3;

                del_Fvib_q0_umn(ns - 3 + ixyz1) = -del_v0_del_umn_vib[ixyz1 * 3 + ixyz1];
                del_Fvib_q0_umn(ns + ixyz1) =
                    -del_v0_del_umn_vib[ixyz2 * 3 + ixyz3] + del_v0_del_umn_vib[ixyz3 * 3 + ixyz2];
            }
            q0_umn = elastic_mat_tmp.colPivHouseholderQr().solve(del_Fvib_q0_umn);

            for (is1 = 0; is1 < ns; is1++) {
                q0[is1] = 0.0;
            }
            for (is1 = 0; is1 < ns - 3; is1++) {
                is2 = harm_optical_modes[is1];
                q0[is2] = q0_umn(is1).real();
            }
            relaxation->calculate_u0(q0, u0, omega2_harmonic, evec_harmonic);

            for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
                ixyz2 = (ixyz1 + 1) % 3;
                ixyz3 = (ixyz1 + 2) % 3;

                u_tensor[ixyz1][ixyz1] = q0_umn(ns - 3 + ixyz1).real();
                u_tensor[ixyz2][ixyz3] = q0_umn(ns + ixyz1).real();
                u_tensor[ixyz3][ixyz2] = q0_umn(ns + ixyz1).real();
            }

            // print obtained structure
            relaxation->calculate_u0(q0, u0, omega2_harmonic, evec_harmonic);

            relaxation->write_resfile_atT(q0, u_tensor, u0, temp, fout_q0, fout_u0, fout_u_tensor);

            // calculate renormalized IFCs for postprocess
            // Note that the cubic IFCs are fixed at the reference values in perturbative QHA.

            // renormalization by strain
            relaxation->calculate_eta_tensor(eta_tensor, u_tensor);
            relaxation->renormalize_v0_from_umn(v0_with_umn,
                                                0.0,
                                                eta_tensor,
                                                C1_array,
                                                C2_array,
                                                C3_array,
                                                u_tensor,
                                                0.0); // pressure is limited to zero

            relaxation->renormalize_v1_from_umn(v1_with_umn,
                                                v1_ref,
                                                del_v1_del_umn,
                                                del2_v1_del_umn2,
                                                del3_v1_del_umn3_dummy,
                                                u_tensor);

            relaxation->renormalize_v2_from_umn(kmesh_coarse,
                                                kmap_coarse_to_dense,
                                                delta_v2_with_umn,
                                                del_v2_del_umn,
                                                del2_v2_del_umn2_dummy,
                                                u_tensor);

            // renormalization by displacements
            relaxation->renormalize_v1_from_q0(omega2_harmonic,
                                               kmesh_coarse,
                                               kmesh_dense,
                                               v1_renorm,
                                               v1_with_umn,
                                               delta_v2_with_umn,
                                               v3_ref,
                                               v4_array_dummy,
                                               q0);

            relaxation->renormalize_v2_from_q0(evec_harmonic,
                                               kmesh_coarse,
                                               kmesh_dense,
                                               kmap_coarse_to_dense,
                                               mat_transform_sym,
                                               delta_v2_renorm,
                                               delta_v2_with_umn,
                                               v3_ref,
                                               v4_array_dummy,
                                               q0);

            relaxation->renormalize_v0_from_q0(omega2_harmonic,
                                               kmesh_dense,
                                               v0_renorm,
                                               v0_with_umn,
                                               v1_with_umn,
                                               delta_v2_with_umn,
                                               v3_ref,
                                               v4_array_dummy,
                                               q0);

            relaxation->V0[iT] = v0_renorm;

            // calculate renormalizations of harmonic IFCs, which is stored in delta_harmonic_dymat_renormalize
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

            // copy delta_harmonic_dymat_renormalize to dymat_anharm
            for (is1 = 0; is1 < ns; is1++) {
                for (is2 = 0; is2 < ns; is2++) {
                    for (ik = 0; ik < kmesh_coarse->nk; ik++) {
                        dymat_anharm[iT][is1][is2][ik] = delta_harmonic_dymat_renormalize[iT][is1][is2][ik];
                    }
                }
            }
        }

        fout_q0.close();
        fout_u0.close();
        fout_u_tensor.close();
    }

    deallocate(del_v0_del_umn_vib);

    deallocate(v1_vib);
    deallocate(v1_ref);
    deallocate(v1_with_umn);
    deallocate(v1_renorm);

    deallocate(omega2_harm_renorm);
    deallocate(evec_harm_renorm_tmp);
    deallocate(delta_v2_renorm);
    deallocate(delta_v2_with_umn);

    deallocate(q0);
    deallocate(u0);
    deallocate(u_tensor);
    deallocate(eta_tensor);

    deallocate(v4_array_dummy);
    deallocate(v3_ref);

    deallocate(del_v1_del_umn);
    deallocate(del2_v1_del_umn2);
    deallocate(del_v2_del_umn);
    deallocate(del3_v1_del_umn3_dummy);
    deallocate(del2_v2_del_umn2_dummy);
    deallocate(del_v3_del_umn_dummy);

    deallocate(C1_array);
    deallocate(C2_array);
    deallocate(C3_array);
}

void Qha::calc_del_v0_del_umn_vib(std::complex<double> *del_v0_del_umn_vib, std::complex<double> ***del_v2_del_umn,
                                  double T_in)
{
    const auto nk = kmesh_dense->nk;
    const auto ns = dynamical->neval;

    static auto complex_zero = std::complex<double>(0.0, 0.0);

    int ixyz1, ixyz2, is, ik;
    std::complex<double> Qtmp;
    double omega1_tmp, n1;

    double factor = 0.25 / static_cast<double>(nk);

    for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
        for (ixyz2 = 0; ixyz2 < 3; ixyz2++) {
            del_v0_del_umn_vib[ixyz1 * 3 + ixyz2] = complex_zero;

            for (is = 0; is < ns; is++) {
                for (ik = 0; ik < nk; ik++) {
                    omega1_tmp = std::sqrt(std::fabs(omega2_harmonic[ik][is]));

                    if (omega2_harmonic[ik][is] < 0 && omega1_tmp > eps8) {
                        std::cout << "Warning : Negative frequency is detected in perturbative QHA.\n";
                    }

                    if (std::abs(omega1_tmp) < eps8) {
                        Qtmp = 0.0;
                    } else {
                        if (thermodynamics->classical) {
                            Qtmp =
                                std::complex<double>(2.0 * T_in * thermodynamics->T_to_Ryd / (omega1_tmp * omega1_tmp),
                                                     0.0);
                        } else {
                            n1 = thermodynamics->fB(omega1_tmp, T_in);
                            Qtmp = std::complex<double>((2.0 * n1 + 1.0) / omega1_tmp, 0.0);
                        }
                    }

                    del_v0_del_umn_vib[ixyz1 * 3 + ixyz2] +=
                        factor * del_v2_del_umn[ixyz1 * 3 + ixyz2][ik][is * ns + is] * Qtmp;
                }
            }
        }
    }
}

void Qha::calculate_del_v1_del_umn_renorm(std::complex<double> **del_v1_del_umn_renorm, double **u_tensor,
                                          std::complex<double> **del_v1_del_umn,
                                          std::complex<double> **del2_v1_del_umn2,
                                          std::complex<double> **del3_v1_del_umn3,
                                          std::complex<double> ***del_v2_del_umn,
                                          std::complex<double> ***del2_v2_del_umn2,
                                          std::complex<double> ****del_v3_del_umn, double *q0)
{
    int ns = dynamical->neval;
    int nk = kmesh_dense->nk;

    std::complex<double> ***del_v2_strain_tmp;
    allocate(del_v2_strain_tmp, 9, ns, ns);

    double factor = 0.5 * 4.0 * nk;
    int i1, i2, i3, ixyz1, ixyz2;
    int is1, is2, is3;

    const auto complex_zero = std::complex<double>(0.0, 0.0);

    // initialize
    for (i1 = 0; i1 < 9; i1++) {
        for (is1 = 0; is1 < ns; is1++) {
            del_v1_del_umn_renorm[i1][is1] = complex_zero;
        }
    }

    // calculate renormalization from strain
    for (i1 = 0; i1 < 9; i1++) {
        for (is1 = 0; is1 < ns; is1++) {
            del_v1_del_umn_renorm[i1][is1] = del_v1_del_umn[i1][is1];

            for (i2 = 0; i2 < 9; i2++) {
                del_v1_del_umn_renorm[i1][is1] += del2_v1_del_umn2[i1 * 9 + i2][is1] * u_tensor[i2 / 3][i2 % 3];
                for (i3 = 0; i3 < 9; i3++) {
                    del_v1_del_umn_renorm[i1][is1] += 0.5 * del3_v1_del_umn3[i1 * 81 + i2 * 9 + i3][is1] *
                                                      u_tensor[i2 / 3][i2 % 3] * u_tensor[i3 / 3][i3 % 3];
                }
            }
        }
    }

    // first order in internal coordinate
    // preparation
    for (i1 = 0; i1 < 9; i1++) {
        for (is1 = 0; is1 < ns; is1++) {
            for (is2 = 0; is2 < ns; is2++) {
                del_v2_strain_tmp[i1][is1][is2] = del_v2_del_umn[i1][0][is1 * ns + is2];

                for (i2 = 0; i2 < 9; i2++) {
                    del_v2_strain_tmp[i1][is1][is2] +=
                        del2_v2_del_umn2[i1 * 9 + i2][0][is1 * ns + is2] * u_tensor[i2 / 3][i2 % 3];
                }
            }
        }
    }

    // add to del_v1_del_umn_renorm
    for (i1 = 0; i1 < 9; i1++) {
        for (is1 = 0; is1 < ns; is1++) {
            for (is2 = 0; is2 < ns; is2++) {
                del_v1_del_umn_renorm[i1][is1] += del_v2_strain_tmp[i1][is1][is2] * q0[is2];
            }
        }
    }

    // second order in internal coordinate
    for (i1 = 0; i1 < 9; i1++) {
        for (is1 = 0; is1 < ns; is1++) {

            for (is2 = 0; is2 < ns; is2++) {
                for (is3 = 0; is3 < ns; is3++) {
                    del_v1_del_umn_renorm[i1][is1] +=
                        factor * del_v3_del_umn[i1][0][is1][is2 * ns + is3] * q0[is2] * q0[is3];
                }
            }
        }
    }


    // symmetrize with respect to the interchange of indices of strain tensor
    for (is1 = 0; is1 < ns; is1++) {
        for (ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            for (ixyz2 = ixyz1 + 1; ixyz2 < 3; ixyz2++) {
                del_v1_del_umn_renorm[ixyz1 * 3 + ixyz2][is1] = 0.5 * (del_v1_del_umn_renorm[ixyz1 * 3 + ixyz2][is1] +
                                                                       del_v1_del_umn_renorm[ixyz2 * 3 + ixyz1][is1]);

                del_v1_del_umn_renorm[ixyz2 * 3 + ixyz1][is1] = del_v1_del_umn_renorm[ixyz1 * 3 + ixyz2][is1];
            }
        }
    }

    deallocate(del_v2_strain_tmp);
}


void Qha::calculate_C2_array_renorm(double **C2_array_renorm, double **u_tensor, double **eta_tensor, double **C2_array,
                                    double ***C3_array, std::complex<double> **del2_v1_del_umn2,
                                    std::complex<double> **del3_v1_del_umn3, std::complex<double> ***del2_v2_del_umn2,
                                    double *q0)
{
    int ns = dynamical->neval;
    double **del_eta_del_u;
    allocate(del_eta_del_u, 9, 9);

    int i1, i2, i3, i4, ixyz1, ixyz2, ixyz3, ixyz4;
    int is1, is2;

    double **C2_array_with_strain_eta;
    allocate(C2_array_with_strain_eta, 9, 9);


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

    for (i1 = 0; i1 < 9; i1++) {
        for (i2 = 0; i2 < 9; i2++) {
            C2_array_with_strain_eta[i1][i2] = C2_array[i1][i2];
            for (i3 = 0; i3 < 9; i3++) {
                C2_array_with_strain_eta[i1][i2] += C3_array[i1][i2][i3] * eta_tensor[i3 / 3][i3 % 3];
            }
        }
    }

    for (i1 = 0; i1 < 9; i1++) {
        for (i2 = 0; i2 < 9; i2++) {
            C2_array_renorm[i1][i2] = 0.0;

            for (i3 = 0; i3 < 9; i3++) {
                for (i4 = 0; i4 < 9; i4++) {
                    C2_array_renorm[i1][i2] +=
                        C2_array_with_strain_eta[i3][i4] * del_eta_del_u[i3][i1] * del_eta_del_u[i4][i2];
                }
            }
        }
    }

    for (i1 = 0; i1 < 9; i1++) {
        for (i2 = 0; i2 < 9; i2++) {
            for (is1 = 0; is1 < ns; is1++) {
                C2_array_renorm[i1][i2] += del2_v1_del_umn2[i1 * 9 + i2][is1].real() * q0[is1];
            }

            for (is1 = 0; is1 < ns; is1++) {
                for (is2 = 0; is2 < ns; is2++) {
                    C2_array_renorm[i1][i2] +=
                        0.5 * del2_v2_del_umn2[i1 * 9 + i2][0][is1 * ns + is2].real() * q0[is1] * q0[is2];
                }
            }

            for (is1 = 0; is1 < ns; is1++) {
                for (i3 = 0; i3 < 9; i3++) {
                    C2_array_renorm[i1][i2] +=
                        del3_v1_del_umn3[i1 * 81 + i2 * 9 + i3][is1].real() * q0[is1] * u_tensor[i3 / 3][i3 % 3];
                }
            }
        }
    }

    deallocate(C2_array_with_strain_eta);
    deallocate(del_eta_del_u);
}

void Qha::calculate_C2_array_ZSISA(double **C2_array_ZSISA, double **C2_array_renorm,
                                   std::complex<double> **del_v1_del_umn_renorm, double **delq_delu_ZSISA)
{
    // calculate ZSISA second-order elastic constants,
    // which is the elastic constants with ZSISA internal coordinates.

    int i1, i2, is;
    int ns = dynamical->neval;

    for (i1 = 0; i1 < 9; i1++) {
        for (i2 = 0; i2 < 9; i2++) {
            C2_array_ZSISA[i1][i2] = C2_array_renorm[i1][i2];
            for (is = 0; is < ns; is++) {
                C2_array_ZSISA[i1][i2] += del_v1_del_umn_renorm[i1][is].real() * delq_delu_ZSISA[is][i2];
            }
        }
    }
}


void Qha::compute_ZSISA_stress(double **delq_delu_ZSISA_out, std::complex<double> *del_v0_del_umn_ZSISA_out,
                               std::complex<double> ***cmat_convert, double **omega2_harm_renorm_in,
                               std::complex<double> *del_v0_del_umn_QHA, std::complex<double> **del_v1_del_umn_renorm,
                               std::complex<double> *v1_QHA, std::vector<int> &harm_optical_modes)
{
    using namespace Eigen;
    int i1;
    int is, js;

    const auto ns = dynamical->neval;


    MatrixXcd Cmat(ns, ns), v2_mat_full(ns, ns);
    MatrixXcd v2_mat_optical(ns - 3, ns - 3);
    VectorXcd vec_del_V1_strain(ns - 3);
    VectorXcd vec_delq_delu_ZSISA(ns - 3);


    // calculate delq_delu_ZSISA_out
    for (i1 = 0; i1 < 9; i1++) {

        for (is = 0; is < ns; is++) {
            for (js = 0; js < ns; js++) {
                Cmat(js, is) = cmat_convert[0][is][js]; // transpose
                v2_mat_full(is, js) = 0.0;
            }
            v2_mat_full(is, is) = omega2_harm_renorm_in[0][is];
        }
        v2_mat_full = Cmat.adjoint() * v2_mat_full * Cmat;

        for (is = 0; is < ns - 3; is++) {
            for (js = 0; js < ns - 3; js++) {
                v2_mat_optical(is, js) = v2_mat_full(harm_optical_modes[is], harm_optical_modes[js]);
            }
        }

        for (is = 0; is < ns - 3; is++) {
            vec_del_V1_strain(is) = del_v1_del_umn_renorm[i1][harm_optical_modes[is]];
        }

        vec_delq_delu_ZSISA = -1.0 * v2_mat_optical.colPivHouseholderQr().solve(vec_del_V1_strain);

        for (is = 0; is < ns; is++) {
            delq_delu_ZSISA_out[is][i1] = 0.0;
        }
        for (is = 0; is < ns - 3; is++) {
            delq_delu_ZSISA_out[harm_optical_modes[is]][i1] = vec_delq_delu_ZSISA(is).real();
        }
    }

    // calculate ZSISA stress tensor
    for (i1 = 0; i1 < 9; i1++) {

        del_v0_del_umn_ZSISA_out[i1] = del_v0_del_umn_QHA[i1];

        // add correction to QHA stress tensor
        for (is = 0; is < ns - 3; is++) {
            del_v0_del_umn_ZSISA_out[i1] +=
                v1_QHA[harm_optical_modes[is]] * delq_delu_ZSISA_out[harm_optical_modes[is]][i1];
        }
    }
}

void Qha::compute_vZSISA_stress(std::complex<double> *del_v0_del_umn_vZSISA, double **C2_array_ZSISA,
                                std::complex<double> *del_v0_del_umn_renorm, std::complex<double> *del_v0_del_umn_ZSISA,
                                double **u_tensor)
{
    using namespace Eigen;

    int i1, i2;
    int is1, is2, ixyz1, ixyz2, ixyz3, ixyz4;
    int itmp1, itmp2, itmp3, itmp4, itmp5, itmp6;

    double F_tensor[3][3]; // F_{mu nu} = delta_{mu nu} + u_{mu nu}
    double ddetF_dumn[9], u_tilde[9], delta_umn_vZSISA[9];
    VectorXcd ddetF_dumn_vec(6), delta_umn_vZSISA_vec(6);
    double factor_tmp;
    double deltaF_vZSISA, deltaU_vZSISA;

    MatrixXcd C2_mat_tmp(6, 6);

    // calculate ddetF_dumn = (d det(I+u))/(du)
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

        ddetF_dumn[i1] =
            F_tensor[ixyz1][ixyz3] * F_tensor[ixyz2][ixyz4] - F_tensor[ixyz1][ixyz4] * F_tensor[ixyz2][ixyz3];
    }


    factor_tmp = 0.0;
    for (i1 = 0; i1 < 9; i1++) {
        factor_tmp += ddetF_dumn[i1] * ddetF_dumn[i1];
    }
    factor_tmp = 1.0 / std::sqrt(factor_tmp);

    for (i1 = 0; i1 < 9; i1++) {
        u_tilde[i1] = factor_tmp * ddetF_dumn[i1];
    }

    // calcualte delta_umn_vZSISA
    for (itmp1 = 0; itmp1 < 3; itmp1++) {
        ddetF_dumn_vec(itmp1) = ddetF_dumn[itmp1 * 3 + itmp1];

        itmp2 = (itmp1 + 1) % 3;
        itmp3 = (itmp1 + 2) % 3;
        ddetF_dumn_vec(itmp1 + 3) = ddetF_dumn[itmp2 * 3 + itmp3];
    }

    for (itmp1 = 0; itmp1 < 3; itmp1++) {
        for (itmp2 = 0; itmp2 < 3; itmp2++) {
            C2_mat_tmp(itmp1, itmp2) = C2_array_ZSISA[itmp1 * 3 + itmp1][itmp2 * 3 + itmp2];
        }
    }
    for (itmp1 = 0; itmp1 < 3; itmp1++) {
        for (itmp2 = 0; itmp2 < 3; itmp2++) {
            itmp3 = (itmp2 + 1) % 3;
            itmp4 = (itmp2 + 2) % 3;
            C2_mat_tmp(itmp1, itmp2 + 3) = 2.0 * C2_array_ZSISA[itmp1 * 3 + itmp1][itmp3 * 3 + itmp4];
            C2_mat_tmp(itmp2 + 3, itmp1) = C2_array_ZSISA[itmp3 * 3 + itmp4][itmp1 * 3 + itmp1];
        }
    }
    for (itmp1 = 0; itmp1 < 3; itmp1++) {
        for (itmp2 = 0; itmp2 < 3; itmp2++) {
            itmp3 = (itmp1 + 1) % 3;
            itmp4 = (itmp1 + 2) % 3;
            itmp5 = (itmp2 + 1) % 3;
            itmp6 = (itmp2 + 2) % 3;
            C2_mat_tmp(itmp1 + 3, itmp2 + 3) = 2.0 * C2_array_ZSISA[itmp3 * 3 + itmp4][itmp5 * 3 + itmp6];
        }
    }

    // solve linear equation for delta_umn_vZSISA
    delta_umn_vZSISA_vec = C2_mat_tmp.colPivHouseholderQr().solve(ddetF_dumn_vec);
    for (itmp1 = 0; itmp1 < 3; itmp1++) {
        itmp3 = (itmp1 + 1) % 3;
        itmp4 = (itmp1 + 2) % 3;

        delta_umn_vZSISA[itmp1 * 3 + itmp1] = delta_umn_vZSISA_vec(itmp1).real();
        delta_umn_vZSISA[itmp3 * 3 + itmp4] = delta_umn_vZSISA_vec(itmp1 + 3).real();
        delta_umn_vZSISA[itmp3 + itmp4 * 3] = delta_umn_vZSISA_vec(itmp1 + 3).real();
    }

    // normalize delta_umn_vZSISA
    factor_tmp = 0.0;
    for (i1 = 0; i1 < 9; i1++) {
        factor_tmp += u_tilde[i1] * delta_umn_vZSISA[i1];
    }
    factor_tmp = 1.0 / factor_tmp;

    for (i1 = 0; i1 < 9; i1++) {
        delta_umn_vZSISA[i1] *= factor_tmp;
    }

    // calculate delta_q0_vZSISA

    deltaF_vZSISA = 0.0;
    deltaU_vZSISA = 0.0;
    for (i1 = 0; i1 < 9; i1++) {
        deltaF_vZSISA += delta_umn_vZSISA[i1] * del_v0_del_umn_ZSISA[i1].real();
        deltaU_vZSISA += u_tilde[i1] * del_v0_del_umn_renorm[i1].real();
    }

    for (i1 = 0; i1 < 9; i1++) {
        del_v0_del_umn_vZSISA[i1] = del_v0_del_umn_renorm[i1] + u_tilde[i1] * (deltaF_vZSISA - deltaU_vZSISA);
    }
}


void Qha::calc_v1_vib(std::complex<double> *v1_vib, std::complex<double> ***v3_ref, const double T_in)
{
    const auto nk = kmesh_dense->nk;
    const auto ns = dynamical->neval;

    static auto complex_zero = std::complex<double>(0.0, 0.0);

    int is1, is2, ik;
    std::complex<double> Qtmp;
    double omega1_tmp, n1;


    for (is1 = 0; is1 < ns; is1++) {
        v1_vib[is1] = complex_zero;

        for (is2 = 0; is2 < ns; is2++) {
            for (ik = 0; ik < nk; ik++) {
                omega1_tmp = std::sqrt(std::fabs(omega2_harmonic[ik][is2]));

                if (omega2_harmonic[ik][is2] < 0 && omega1_tmp > eps8) {
                    std::cout << "Warning : Negative frequency is detected in perturbative QHA.\n";
                }

                if (std::abs(omega1_tmp) < eps8) {
                    Qtmp = 0.0;
                } else {
                    if (thermodynamics->classical) {
                        Qtmp = std::complex<double>(2.0 * T_in * thermodynamics->T_to_Ryd / (omega1_tmp * omega1_tmp),
                                                    0.0);
                    } else {
                        n1 = thermodynamics->fB(omega1_tmp, T_in);
                        Qtmp = std::complex<double>((2.0 * n1 + 1.0) / omega1_tmp, 0.0);
                    }
                }

                v1_vib[is1] += v3_ref[ik][is1][is2 * ns + is2] * Qtmp;
            }
        }
    }
}

void Qha::compute_cmat(std::complex<double> ***cmat_convert, const std::complex<double> *const *const *const evec_new)
{

    using namespace Eigen;

    const auto nk = kmesh_dense->nk;
    const auto ns = dynamical->neval;

    int ik, is, js;
    MatrixXcd evec_mat_original(ns, ns), evec_mat_QHA(ns, ns), Cmat;

    for (ik = 0; ik < nk; ++ik) {

        for (is = 0; is < ns; ++is) {
            for (js = 0; js < ns; ++js) {
                evec_mat_original(is, js) = evec_harmonic[ik][js][is];
                evec_mat_QHA(is, js) = evec_new[ik][js][is];
            }
        }

        Cmat = evec_mat_original.adjoint() * evec_mat_QHA;

        for (is = 0; is < ns; ++is) {
            for (js = 0; js < ns; ++js) {
                cmat_convert[ik][is][js] = Cmat(is, js);
            }
        }
    }
}
