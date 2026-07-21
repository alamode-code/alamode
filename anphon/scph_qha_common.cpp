/*
 scph_qha_common.cpp

 Copyright (c) 2026

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "scph_qha_common.h"
#include <algorithm>
#include <complex>
#include <fstream>
#include <numeric>
#include <iomanip>
#include <iostream>
#include <vector>
#include "constants.h"
#include "dielec.h"
#include "interpolation.h"
#include "phonon_dos.h"
#include "relaxation.h"
#include "thermodynamics.h"
#include "write_phonons.h"

using namespace PHON_NS;

ScphQhaCommon::ScphQhaCommon(class PHON *phon) : Pointers(phon)
{}

ScphQhaCommon::~ScphQhaCommon() = default;

void ScphQhaCommon::build_cmat_at_k(const unsigned int ns, const Eigen::MatrixXcd &evec_ref_mat,
                                    const std::complex<double> *const *evec_new_at_k, std::complex<double> **cmat_out)
{
    Eigen::MatrixXcd evec_new_mat(ns, ns);

    for (unsigned int is = 0; is < ns; ++is) {
        for (unsigned int js = 0; js < ns; ++js) {
            evec_new_mat(is, js) = evec_new_at_k[js][is];
        }
    }

    Eigen::MatrixXcd Cmat = evec_ref_mat.adjoint() * evec_new_mat;

    for (unsigned int is = 0; is < ns; ++is) {
        for (unsigned int js = 0; js < ns; ++js) {
            cmat_out[is][js] = Cmat(is, js);
        }
    }
}

std::vector<bool> ScphQhaCommon::classify_acoustic_modes_from_cmat(
    const std::complex<double> *const *cmat_at_gamma) const
{
    const auto ns = dynamical->neval;

    std::vector<double> overlap(ns, 0.0);
    for (auto is = 0; is < ns; ++is) {
        if (!is_acoustic_gamma_harm[is]) continue;
        for (auto js = 0; js < ns; ++js) {
            overlap[js] += std::norm(cmat_at_gamma[is][js]);
        }
    }

    // Majority overlap with the harmonic acoustic subspace marks a mode as acoustic.
    // A threshold is used instead of picking the three largest overlaps on purpose:
    // when a soft optical mode becomes numerically degenerate with the acoustic modes
    // during the SCPH iteration, the eigensolver may return arbitrarily mixed columns,
    // and a fixed count could then assign a mostly-translational column as "optical"
    // (whose then huge 1/omega occupation factor would destabilize the iteration).
    // With a threshold, every mostly-translational column is excluded, transiently
    // mixed columns resolve themselves once the degeneracy is lifted, and in the
    // clean (non-degenerate) case exactly the three translational modes are flagged.
    std::vector<bool> is_acoustic(ns, false);
    for (auto js = 0; js < ns; ++js) {
        is_acoustic[js] = overlap[js] > 0.5;
    }

    return is_acoustic;
}

void ScphQhaCommon::initialize_variables()
{
    kmap_coarse_to_dense.clear();
    dymat_harm_short.clear();
    dymat_harm_long.clear();
    ialgo = 0;
    selfenergy_offdiagonal = true;
}

void ScphQhaCommon::deallocate_variables()
{
    mindist_list.clear();
    evec_harmonic.clear();
    omega2_harmonic.clear();
    phi3_reciprocal.clear();
    phi4_reciprocal.clear();
    mat_transform_sym.clear();

    kmesh_coarse.reset();
    kmesh_dense.reset();
    phase_factor.reset();

    kmap_coarse_to_dense.clear();
    dymat_harm_short.clear();
    dymat_harm_long.clear();
}

void ScphQhaCommon::setup_kmesh(unsigned int kmesh_dense_input[3], unsigned int kmesh_coarse_input[3],
                                const char *mode_name, const char *mapping_error_message)
{
    MPI_Bcast(&kmesh_dense_input[0], 3, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&kmesh_coarse_input[0], 3, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    kmesh_coarse = std::make_unique<KpointMeshUniform>(kmesh_coarse_input);
    kmesh_dense = std::make_unique<KpointMeshUniform>(kmesh_dense_input);
    kmesh_coarse->setup(symmetry->SymmList, system->get_primcell().reciprocal_lattice_vector, true);
    kmesh_dense->setup(symmetry->SymmList, system->get_primcell().reciprocal_lattice_vector, true);

    if (mympi->my_rank == 0 && writes->getVerbosity() > 0) {
        std::cout << " Setting up the " << mode_name << " calculations ...\n\n";
        std::cout << "  Gamma-centered uniform grid with the following mesh density:\n";
        std::cout << "  nk1:" << std::setw(5) << kmesh_dense_input[0] << '\n';
        std::cout << "  nk2:" << std::setw(5) << kmesh_dense_input[1] << '\n';
        std::cout << "  nk3:" << std::setw(5) << kmesh_dense_input[2] << "\n\n";
        std::cout << "  Number of k points : " << kmesh_dense->nk << '\n';
        std::cout << "  Number of irreducible k points : " << kmesh_dense->nk_irred << "\n\n";
        std::cout << "  Fourier interpolation from reciprocal to real space\n";
        std::cout << "  will be performed with the following mesh density:\n";
        std::cout << "  nk1:" << std::setw(5) << kmesh_coarse_input[0] << '\n';
        std::cout << "  nk2:" << std::setw(5) << kmesh_coarse_input[1] << '\n';
        std::cout << "  nk3:" << std::setw(5) << kmesh_coarse_input[2] << "\n\n";
        std::cout << "  Number of k points : " << kmesh_coarse->nk << '\n';
        std::cout << "  Number of irreducible k points : " << kmesh_coarse->nk_irred << '\n';
    }

    const auto info_mapping =
        kpoint->get_kmap_coarse_to_dense(kmesh_coarse.get(), kmesh_dense.get(), kmap_coarse_to_dense);
    if (info_mapping == 1) {
        exit("setup_kmesh", mapping_error_message);
    }

    kmesh_coarse->setup_kpoint_symmetry(symmetry->SymmListWithMap);
}

void ScphQhaCommon::setup_eigvecs()
{
    const auto ns = dynamical->neval;

    if (mympi->my_rank == 0 && writes->getVerbosity() > 0) {
        std::cout << '\n' << " Diagonalizing dynamical matrices for all k points ... ";
    }

    evec_harmonic.clear();
    omega2_harmonic.clear();

    evec_harmonic.resize(kmesh_dense->nk, ns, ns);
    omega2_harmonic.resize(kmesh_dense->nk, ns);

    for (int ik = 0; ik < kmesh_dense->nk; ++ik) {
        dynamical->eval_k(kmesh_dense->xk[ik],
                          kmesh_dense->kvec_na[ik],
                          fcs_phonon->force_constant_with_cell[0],
                          omega2_harmonic[ik],
                          evec_harmonic[ik],
                          true);

        for (auto is = 0; is < ns; ++is) {
            if (std::abs(omega2_harmonic[ik][is]) < eps) {
                omega2_harmonic[ik][is] = 1.0e-30;
            }
        }
    }

    // Assign the three acoustic modes at Gamma from the eigenvectors (projection onto the
    // rigid-translation subspace). The result replaces the frequency-magnitude criteria that
    // could misclassify a soft optical mode collapsing to zero frequency.
    const double xk_gamma[3] = {0.0, 0.0, 0.0};
    ik_gamma_dense = kmesh_dense->get_knum(xk_gamma);
    if (ik_gamma_dense < 0) {
        exit("setup_eigvecs", "Gamma point not found in the dense k mesh.");
    }
    is_acoustic_gamma_harm = dynamical->detect_acoustic_modes_at_gamma(evec_harmonic[ik_gamma_dense]);

    if (mympi->my_rank == 0 && writes->getVerbosity() > 0) {
        std::cout << "done !\n";
    }
}

void ScphQhaCommon::setup_structural_data()
{
    mindist_list.clear();
    mat_transform_sym.clear();

    system->get_minimum_distances(kmesh_coarse->nk_i, mindist_list);
    get_symmetry_gamma_dynamical(kmesh_coarse.get(),
                                 system->get_primcell().number_of_atoms,
                                 dynamical->neval,
                                 system->get_primcell().x_fractional,
                                 symmetry->SymmListWithMap,
                                 mat_transform_sym);
}

void ScphQhaCommon::setup_pp_interaction(const bool prepare_v3)
{
    if (mympi->my_rank == 0 && writes->getVerbosity() > 0) {
        if (prepare_v3) {
            std::cout << " Preparing for calculating V3 & V4  ...";
        } else {
            std::cout << " Preparing for calculating V4  ...";
        }
    }

    if (anharmonic_core->quartic_mode != 1) {
        exit("setup_pp_interaction", "quartic_mode should be 1 for SCPH");
    }

    phi3_reciprocal.clear();
    phi4_reciprocal.clear();

    if (prepare_v3) {
        phi3_reciprocal.resize(anharmonic_core->get_ngroup_fcs(3));
    }
    phi4_reciprocal.resize(anharmonic_core->get_ngroup_fcs(4));

    phase_factor = std::make_unique<PhaseFactorCache>(kmesh_dense->nk_i);
    phase_factor->create(true);

    if (mympi->my_rank == 0 && writes->getVerbosity() > 0) {
        std::cout << " done!\n";
    }
}

void ScphQhaCommon::zerofill_harmonic_dymat_renormalize(std::complex<double> ****delta_harmonic_dymat_renormalize,
                                                        const unsigned int NT) const
{
    const auto ns = dynamical->neval;
    const auto complex_zero = std::complex<double>(0.0, 0.0);

    for (unsigned int iT = 0; iT < NT; ++iT) {
        for (unsigned int is1 = 0; is1 < ns; ++is1) {
            for (unsigned int is2 = 0; is2 < ns; ++is2) {
                for (unsigned int ik = 0; ik < kmesh_coarse->nk; ++ik) {
                    delta_harmonic_dymat_renormalize[iT][is1][is2][ik] = complex_zero;
                }
            }
        }
    }
}

bool ScphQhaCommon::use_band_parallel_v4() const
{
    return ialgo == 1;
}

void ScphQhaCommon::load_scph_dymat_from_file(std::complex<double> ****dymat_out, std::string filename_dymat,
                                              const KpointMeshUniform *kmesh_dense_in,
                                              const KpointMeshUniform *kmesh_coarse_in,
                                              const unsigned int nonanalytic_in, const bool selfenergy_offdiagonal_in)
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
            exit("load_scph_dymat_from_file", "Cannot open scph_dymat file");
        }

        // Read computational settings from file and check the consistency.
        ifs_dymat >> nk_interpolate_ref[0] >> nk_interpolate_ref[1] >> nk_interpolate_ref[2];
        ifs_dymat >> nk_scph_tmp[0] >> nk_scph_tmp[1] >> nk_scph_tmp[2];
        ifs_dymat >> Tmin_tmp >> Tmax_tmp >> dT_tmp;
        ifs_dymat >> nonanalytic_tmp >> consider_offdiag_tmp;

        if (nk_interpolate_ref[0] != kmesh_coarse_in->nk_i[0] || nk_interpolate_ref[1] != kmesh_coarse_in->nk_i[1] ||
            nk_interpolate_ref[2] != kmesh_coarse_in->nk_i[2])
        {
            exit("load_scph_dymat_from_file", "The number of KMESH_INTERPOLATE is not consistent");
        }
        if (nk_scph_tmp[0] != kmesh_dense_in->nk_i[0] || nk_scph_tmp[1] != kmesh_dense_in->nk_i[1] ||
            nk_scph_tmp[2] != kmesh_dense_in->nk_i[2])
        {
            exit("load_scph_dymat_from_file", "The number of KMESH_SCPH is not consistent");
        }
        if (nonanalytic_tmp != nonanalytic_in) {
            warn("load_scph_dymat_from_file", "The NONANALYTIC tag is not consistent");
        }
        if (consider_offdiag_tmp != selfenergy_offdiagonal_in) {
            exit("load_scph_dymat_from_file", "The SELF_OFFDIAG tag is not consistent");
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
                            dymat_out[icount][is][js][ik] = std::complex<double>(dymat_real, dymat_imag);
                        }
                    }
                }
            }
            if (flag_load[iT]) icount += 1;
        }

        ifs_dymat.close();

        if (icount != NT) {
            exit("load_scph_dymat_from_file", "The temperature information is not consistent");
        }
        if (writes->getVerbosity() > 0) std::cout << " done.\n";
    }
    // Broadcast to all MPI threads
    mpi_bcast_complex(dymat_out, NT, kmesh_coarse_in->nk, ns);
}

void ScphQhaCommon::store_renormalized_dymat_to_file(const std::complex<double> *const *const *const *dymat_in,
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
    // auto file_dymat = phon->job_title + ".scph_dymat";
    auto file_dymat = filename_dymat;

    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;

    ofs_dymat.open(file_dymat.c_str(), std::ios::out);

    if (!ofs_dymat) {
        exit("store_scph_dymat_to_file", "Cannot open scph_dymat file");
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
                    ofs_dymat << std::setprecision(15) << std::setw(25) << dymat_in[iT][is][js][ik].real();
                    ofs_dymat << std::setprecision(15) << std::setw(25) << dymat_in[iT][is][js][ik].imag();
                    ofs_dymat << '\n';
                }
            }
        }
    }
    ofs_dymat.close();
    if (writes->getVerbosity() > 0) {
        std::cout << "  " << std::setw(phon->job_title.length() + 12) << std::left << file_dymat;
        std::cout << " : Anharmonic dynamical matrix (restart file)\n";
    }
}

void ScphQhaCommon::mpi_bcast_complex(std::complex<double> ****data, const unsigned int NT, const unsigned int nk,
                                      const unsigned int ns)
{
    const int _NT = static_cast<int>(NT);
    const int _nk = static_cast<int>(nk);
    const int _ns = static_cast<int>(ns);

#ifdef MPI_CXX_DOUBLE_COMPLEX
    MPI_Bcast(&data[0][0][0][0], _NT * _nk * _ns * _ns, MPI_CXX_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
#else
    MPI_Bcast(&data[0][0][0][0], _NT * _nk * _ns * _ns, MPI_COMPLEX16, 0, MPI_COMM_WORLD);
#endif
}

void ScphQhaCommon::postprocess(std::complex<double> ****delta_dymat,
                                std::complex<double> ****delta_harmonic_dymat_renormalize,
                                std::complex<double> ****delta_dymat_scph_plus_bubble,
                                const KpointMeshUniform *kmesh_coarse_in, MinimumDistList ***mindist_list_in,
                                const bool is_qha, const int bubble_in)
{
    NDArray<double, 3> eval_update;
    NDArray<double, 3> eval_harm_renorm;
    const auto ns = dynamical->neval;
    const auto Tmin = system->Tmin;
    const auto Tmax = system->Tmax;
    const auto dT = system->dT;
    const auto NT = static_cast<unsigned int>((Tmax - Tmin) / dT) + 1;

    unsigned int nomega_dielec;

    if (mympi->my_rank == 0) {

        if (writes->getVerbosity() > 0) {
            std::cout << '\n';
            std::cout << " Running postprocess of SCPH/QHA (calculation of free energy, MSD, DOS)\n";
            std::cout << " The number of temperature points: " << std::setw(4) << NT << '\n';
            std::cout << "   ";
        }

        NDArray<std::complex<double>, 3> evec_tmp;
        NDArray<std::complex<double>, 3> evec_harm_renorm;
        NDArray<double, 2> eval_gam;
        NDArray<std::complex<double>, 3> evec_gam;
        NDArray<double, 2> xk_gam;

        NDArray<double, 2> dos_update;
        NDArray<double, 3> pdos_update;
        NDArray<double, 1> heat_capacity;
        NDArray<double, 1> heat_capacity_correction;
        NDArray<double, 1> FE_QHA;
        NDArray<double, 1> dFE_scph;
        NDArray<double, 1> FE_total;
        NDArray<double, 1> entropy;
        NDArray<double, 2> msd_update;
        NDArray<double, 3> ucorr_update;
        NDArray<double, 4> dielec_update;
        const double *omega_grid = nullptr;
        NDArray<double, 2> domega_dt;

        if (dos->kmesh_dos.get()) {
            eval_update.resize(NT, dos->kmesh_dos->nk, ns);
            evec_tmp.resize(dos->kmesh_dos->nk, ns, ns);
            eval_harm_renorm.resize(NT, dos->kmesh_dos->nk, ns);
            evec_harm_renorm.resize(dos->kmesh_dos->nk, ns, ns);

            if (dos->compute_dos) {
                dos_update.resize(NT, dos->n_energy);

                if (dos->projected_dos) {
                    pdos_update.resize(NT, ns, dos->n_energy);
                }
            }
            heat_capacity.resize(NT);
            FE_QHA.resize(NT);
            dFE_scph.resize(NT);
            FE_total.resize(NT);
            entropy.resize(NT);

            if (writes->getPrintMSD()) {
                msd_update.resize(NT, ns);
            }
            if (writes->getPrintUcorr()) {
                ucorr_update.resize(NT, ns, ns);
            }
            if (compute_Cv_anharmonic) {
                heat_capacity_correction.resize(NT);
                domega_dt.resize(dos->kmesh_dos->nk, ns);
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
                                eval_tmp = in_kayser(eval_update[iT][dos->kmesh_dos->kpoint_irred_all[j][0].knum][k]);
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
                    dos->calc_dos_from_given_frequency(dos->kmesh_dos.get(),
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
                                                                  evec_harm_renorm,
                                                                  *dos->kmesh_dos.get(),
                                                                  ns,
                                                                  *system);

                FE_total[iT] = thermodynamics->compute_FE_total(iT,
                                                                FE_QHA[iT],
                                                                dFE_scph[iT],
                                                                relaxation->relax_str != 0 ? V0[iT] : 0.0,
                                                                !is_qha);

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
                                                                           evec_tmp,
                                                                           *system);
                    }
                }

                if (writes->getPrintUcorr()) {
                    double shift[3];
                    for (auto i = 0; i < 3; ++i) shift[i] = static_cast<double>(writes->getShiftUcorr()[i]);

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
                                                                                     evec_tmp,
                                                                                     *system);
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

                if (writes->getVerbosity() > 0) {
                    std::cout << '.' << std::flush;
                    if (iT % 25 == 24) {
                        std::cout << '\n';
                        std::cout << std::setw(3);
                    }
                }
            }
            if (writes->getVerbosity() > 0) std::cout << "\n\n";

            if (dos->compute_dos) {
                writes->writePhononDos(dos_update, is_qha, 0);
            }
            writes->writeThermodynamicFunc(heat_capacity,
                                           heat_capacity_correction,
                                           FE_QHA,
                                           dFE_scph,
                                           FE_total,
                                           entropy,
                                           relaxation->relax_str != 0 ? V0.data() : nullptr,
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
                if (writes->getVerbosity() > 0) {
                    std::cout << '\n';
                    std::cout << "   ";
                }

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
                                    eval_tmp =
                                        in_kayser(eval_update[iT][dos->kmesh_dos->kpoint_irred_all[j][0].knum][k]);
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
                        dos->calc_dos_from_given_frequency(dos->kmesh_dos.get(),
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
                                                                               evec_tmp,
                                                                               *system);
                        }
                    }

                    if (writes->getPrintUcorr()) {
                        double shift[3];
                        for (auto i = 0; i < 3; ++i) shift[i] = static_cast<double>(writes->getShiftUcorr()[i]);

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
                                                                                         evec_tmp,
                                                                                         *system);
                            }
                        }
                    }

                    if (writes->getVerbosity() > 0) {
                        std::cout << '.' << std::flush;
                        if (iT % 25 == 24) {
                            std::cout << '\n';
                            std::cout << std::setw(3);
                        }
                    }
                }
                if (writes->getVerbosity() > 0) std::cout << "\n\n";

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
            eval_update.clear();
            evec_tmp.clear();
        }

        if (kpoint->kpoint_general.get()) {
            eval_update.resize(NT, kpoint->kpoint_general->nk, ns);
            evec_tmp.resize(kpoint->kpoint_general->nk, ns, ns);

            for (auto iT = 0; iT < NT; ++iT) {
                // The short/long harmonic dymats are ignored here because
                // use_precomputed_dymat is left false (they were precomputed on
                // the DOS mesh, not on these k points); still pass the correct
                // pair to keep this call consistent with the other branches.
                dynamical->exec_interpolation(kmesh_coarse_in->nk_i,
                                              delta_dymat[iT],
                                              kpoint->kpoint_general->nk,
                                              kpoint->kpoint_general->xk,
                                              kpoint->kpoint_general->kvec_na,
                                              eval_update[iT],
                                              evec_tmp,
                                              dymat_harm_short,
                                              dymat_harm_long,
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
            eval_update.clear();
            evec_tmp.clear();
        }

        if (kpoint->kpoint_bs.get()) {
            eval_update.resize(NT, kpoint->kpoint_bs->nk, ns);
            evec_tmp.resize(kpoint->kpoint_bs->nk, ns, ns);

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
            eval_update.clear();
            evec_tmp.clear();
        }

        if (dielec->calc_dielectric_constant) {
            omega_grid = dielec->get_omega_grid(nomega_dielec);
            dielec_update.resize(NT, nomega_dielec, 3, 3);
            eval_gam.resize(1, ns);
            evec_gam.resize(1, ns, ns);
            xk_gam.resize(1, 3);
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

        eval_update.clear();
        evec_tmp.clear();

        dos_update.clear();
        pdos_update.clear();
        heat_capacity.clear();
        heat_capacity_correction.clear();
        FE_QHA.clear();
        dFE_scph.clear();
        FE_total.clear();
        entropy.clear();
        dielec_update.clear();

        eval_gam.clear();
        evec_gam.clear();
        xk_gam.clear();
    }
}

void ScphQhaCommon::renormalize_ifcs_at_structure(StructuralOptWorkspace &ws)
{
    auto &q0 = ws.structure_state.q0;
    auto &u_tensor = ws.structure_state.u_tensor;
    auto &eta_tensor = ws.structure_state.eta_tensor;

    // get eta tensor
    relaxation->calculate_eta_tensor(eta_tensor, u_tensor);

    // calculate IFCs under strain
    relaxation->renormalize_v0_from_umn(ws.v0_with_umn,
                                        ws.v0_ref,
                                        eta_tensor,
                                        ws.C1_array,
                                        ws.C2_array,
                                        ws.C3_array,
                                        u_tensor,
                                        ws.pvcell);

    relaxation->renormalize_v1_from_umn(ws.v1_with_umn, ws.v1_ref, *ws.del_v_strain, u_tensor);

    relaxation->renormalize_v2_from_umn(kmesh_coarse.get(),
                                        kmap_coarse_to_dense,
                                        ws.delta_v2_with_umn,
                                        *ws.del_v_strain,
                                        u_tensor);
    relaxation->renormalize_v3_from_umn(kmesh_coarse.get(),
                                        kmesh_dense.get(),
                                        ws.v3_with_umn,
                                        ws.v3_ref,
                                        *ws.del_v_strain,
                                        u_tensor);

    // Renormalize the IFCs by the internal displacement q0 (exact Taylor
    // recentering of the quartic PES). The strain-renormalized v1..v3
    // (_with_umn) enter here; v4 enters through ws.v4_for_renorm because its
    // strain renormalization would require d(v4)/du IFC data, which
    // del_v_strain does not include (it stops at d(v3)/du) -- within this
    // truncation the strain-renormalized v4 equals the reference v4.
    relaxation->renormalize_v1_from_q0(omega2_harmonic,
                                       kmesh_coarse.get(),
                                       kmesh_dense.get(),
                                       ws.v1_renorm,
                                       ws.v1_with_umn,
                                       ws.delta_v2_with_umn,
                                       ws.v3_with_umn,
                                       ws.v4_for_renorm,
                                       q0);
    relaxation->renormalize_v2_from_q0(evec_harmonic,
                                       kmesh_coarse.get(),
                                       kmesh_dense.get(),
                                       kmap_coarse_to_dense,
                                       mat_transform_sym,
                                       ws.delta_v2_renorm,
                                       ws.delta_v2_with_umn,
                                       ws.v3_with_umn,
                                       ws.v4_for_renorm,
                                       q0);
    relaxation->renormalize_v3_from_q0(kmesh_dense.get(),
                                       kmesh_coarse.get(),
                                       ws.v3_renorm,
                                       ws.v3_with_umn,
                                       ws.v4_for_renorm,
                                       q0);
    relaxation->renormalize_v0_from_q0(omega2_harmonic,
                                       kmesh_dense.get(),
                                       ws.v0_renorm,
                                       ws.v0_with_umn,
                                       ws.v1_with_umn,
                                       ws.delta_v2_with_umn,
                                       ws.v3_with_umn,
                                       ws.v4_for_renorm,
                                       q0);

    // calculate PES gradient by strain
    if (ws.relax_mode == RelaxationStrMode::CoordinatesOnly) {
        for (auto i1 = 0; i1 < 9; i1++) {
            ws.del_v0_del_umn_renorm[i1] = 0.0;
        }
    } else if (ws.relax_mode == RelaxationStrMode::CoordinatesAndCell) {
        calculate_del_v0_del_umn_renorm(ws.del_v0_del_umn_renorm,
                                        ws.C1_array,
                                        ws.C2_array,
                                        ws.C3_array,
                                        eta_tensor,
                                        u_tensor,
                                        *ws.del_v_strain,
                                        q0,
                                        ws.pvcell,
                                        kmesh_dense.get());
    }
}

void ScphQhaCommon::print_initial_structure(const RelaxationStructureState &state,
                                            const RelaxationStrMode relax_mode) const
{
    if (writes->getVerbosity() == 0) return;

    std::string str_tmp;

    std::cout << " Initial atomic displacements [Bohr] : \n";
    for (auto iat1 = 0; iat1 < system->get_primcell().number_of_atoms; iat1++) {
        std::cout << " ";
        for (auto ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            relaxation->get_xyz_string(ixyz1, str_tmp);
            std::cout << std::setw(10) << ("u_{" + std::to_string(iat1) + "," + str_tmp + "}");
        }
        std::cout << " :";
        for (auto ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            std::cout << std::scientific << std::setw(15) << std::setprecision(6) << state.u0[iat1 * 3 + ixyz1];
        }
        std::cout << '\n';
    }
    std::cout << '\n';

    if (relax_mode == RelaxationStrMode::CoordinatesAndCell) {
        std::cout << " Initial strain (displacement gradient tensor u_{mu nu}) : \n";
        for (auto ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            std::cout << " ";
            for (auto ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                std::cout << std::scientific << std::setw(15) << std::setprecision(6) << state.u_tensor[ixyz1][ixyz2];
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }
}

void ScphQhaCommon::setup_structural_opt_buffers(StructuralOptWorkspace &ws)
{
    const auto nk = kmesh_dense->nk;
    const auto nk_interpolate = kmesh_coarse->nk;
    const auto ns = dynamical->neval;
    const auto nk_irred_interpolate = kmesh_coarse->nk_irred;

    ws.delta_v2_renorm.resize(nk_interpolate, ns * ns);
    ws.delta_v2_with_umn.resize(nk_interpolate, ns * ns);

    ws.v1_ref.resize(ns);
    ws.v1_with_umn.resize(ns);
    ws.v1_renorm.resize(ns);

    ws.structure_state.resize(ns);

    ws.del_v0_del_umn_renorm.resize(9);

    // assume that the atomic forces are zero at the initial structure
    for (auto is = 0; is < ns; is++) {
        ws.v1_ref[is] = 0.0;
    }

    if (mympi->my_rank == 0 && writes->getVerbosity() > 0) {
        std::cout << " RELAX_STR = " << to_int(ws.relax_mode) << ": ";
        if (ws.relax_mode == RelaxationStrMode::CoordinatesOnly) {
            std::cout << "Set zeros in derivatives of k-space IFCs by strain.\n\n";
        }
        if (ws.relax_mode == RelaxationStrMode::CoordinatesAndCell) {
            std::cout << "Calculating derivatives of k-space IFCs by strain.\n\n";
        }
    }

    ws.del_v_strain->resize(nk, ns);

    // Precompute the strain derivatives of v1 (1st..3rd order), v2 (1st and
    // 2nd order), and v3 (1st order).
    relaxation->compute_del_v_strain(kmesh_coarse.get(),
                                     kmesh_dense.get(),
                                     *ws.del_v_strain,
                                     omega2_harmonic,
                                     evec_harmonic,
                                     ws.relax_mode,
                                     mindist_list,
                                     phase_factor.get());

    ws.v4_ref.resize(nk_irred_interpolate * nk, ns * ns, ns * ns);

    // initialize optimizer
    relaxation->create_optimizer(ns);

    // Compute matrix elements of the 4-phonon interaction.
    // This operation is the most expensive part of the calculation.
    if (selfenergy_offdiagonal && use_band_parallel_v4()) {
        compute_V4_elements_mpi_over_band(ws.v4_ref,
                                          evec_harmonic,
                                          selfenergy_offdiagonal,
                                          kmesh_coarse.get(),
                                          kmesh_dense.get(),
                                          kmap_coarse_to_dense,
                                          phase_factor.get(),
                                          phi4_reciprocal);
    } else {
        compute_V4_elements_mpi_over_kpoint(ws.v4_ref,
                                            evec_harmonic,
                                            selfenergy_offdiagonal,
                                            ws.relax_mode != RelaxationStrMode::None,
                                            kmesh_coarse.get(),
                                            kmesh_dense.get(),
                                            kmap_coarse_to_dense,
                                            phase_factor.get(),
                                            phi4_reciprocal);
    }

    ws.v3_ref.resize(nk, ns, ns * ns);
    ws.v3_renorm.resize(nk, ns, ns * ns);
    ws.v3_with_umn.resize(nk, ns, ns * ns);

    compute_V3_elements_mpi_over_kpoint(ws.v3_ref,
                                        evec_harmonic,
                                        selfenergy_offdiagonal,
                                        kmesh_coarse.get(),
                                        kmesh_dense.get(),
                                        phase_factor.get(),
                                        phi3_reciprocal);

    // get indices of optical modes at the Gamma point (the complement of the
    // eigenvector-based acoustic assignment)
    ws.harm_optical_modes.resize(ns - 3);
    auto js = 0;
    for (auto is = 0; is < ns; is++) {
        if (is_acoustic_gamma_harm[is]) {
            continue;
        }
        ws.harm_optical_modes[js] = is;
        js++;
    }
    if (js != ns - 3) {
        exit("setup_structural_opt_buffers", "The number of detected optical modes is not ns-3.");
    }
}

void ScphQhaCommon::compute_and_print_step_gradients(const StructuralOptWorkspace &ws,
                                                     const std::complex<double> *v1_eff,
                                                     const std::complex<double> *del_v0_del_umn_eff, const double du0,
                                                     const double du_tensor, const std::string &spg_label,
                                                     std::vector<StructOptStepRecord> &step_history, double &grad_norm,
                                                     double &cell_grad_norm) const
{
    const auto ns = dynamical->neval;

    // Residual gradient norms over the optimized degrees of freedom (the same
    // gradients the optimizer acts on). A small step (du0/du_tensor) does not
    // by itself imply a small gradient for the GDIIS optimizer
    // (relax_algo == 3), so these are printed for diagnostics and, when the
    // corresponding tolerance is > 0, also required for convergence (SCPH) to
    // guard against false convergence at a non-stationary point. The
    // coordinate force (gradient w.r.t. q0) and the cell gradient (stress
    // conjugate to the strain tensor) have different units, so they are
    // checked separately (cf. COORD_CONV_TOL vs CELL_CONV_TOL).
    grad_norm = 0.0;
    for (auto is = 0; is < ns - 3; is++) {
        const double f = v1_eff[ws.harm_optical_modes[is]].real();
        grad_norm += f * f;
    }
    grad_norm = std::sqrt(grad_norm);

    cell_grad_norm = -1.0;
    if (ws.relax_mode == RelaxationStrMode::CoordinatesAndCell) {
        cell_grad_norm = 0.0;
        for (auto i1 = 0; i1 < 3; i1++) {
            const double fd = del_v0_del_umn_eff[i1 * 3 + i1].real();
            const int j1 = (i1 + 1) % 3;
            const int j2 = (i1 + 2) % 3;
            const double fs = del_v0_del_umn_eff[j1 * 3 + j2].real();
            cell_grad_norm += fd * fd + fs * fs;
        }
        cell_grad_norm = std::sqrt(cell_grad_norm);
    }

    if (writes->getVerbosity() > 0) {
        std::cout << " du0 =" << std::scientific << std::setw(15) << std::setprecision(6) << du0 << " [Bohr]";
        std::cout << " du_tensor =" << std::scientific << std::setw(15) << std::setprecision(6) << du_tensor << '\n';
        std::cout << " |residual force| =" << std::scientific << std::setw(15) << std::setprecision(6) << grad_norm;
        if (ws.relax_mode == RelaxationStrMode::CoordinatesAndCell) {
            std::cout << " |residual stress| =" << std::scientific << std::setw(15) << std::setprecision(6)
                      << cell_grad_norm;
        }
        std::cout << '\n';
    }

    step_history.push_back({true, du0, du_tensor, grad_norm, cell_grad_norm, spg_label});
}

void ScphQhaCommon::print_final_structure(const RelaxationStructureState &state, const RelaxationStrMode relax_mode,
                                          const double temp, const bool last_temperature) const
{
    if (writes->getVerbosity() == 0) return;

    std::string str_tmp;

    std::cout << " ----------------------------------------------------------------\n";
    std::cout << " Final atomic displacements [Bohr] at " << temp << " K\n";
    for (auto iat1 = 0; iat1 < system->get_primcell().number_of_atoms; iat1++) {
        std::cout << " ";
        for (auto ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            relaxation->get_xyz_string(ixyz1, str_tmp);
            std::cout << std::setw(10) << ("u_{" + std::to_string(iat1) + "," + str_tmp + "}");
        }
        std::cout << " :";
        for (auto ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            std::cout << std::scientific << std::setw(15) << std::setprecision(6) << state.u0[iat1 * 3 + ixyz1];
        }
        std::cout << '\n';
    }
    std::cout << '\n';

    if (relax_mode == RelaxationStrMode::CoordinatesAndCell) {
        std::cout << " Final strain (displacement gradient tensor u_{mu nu}) : \n";
        for (auto ixyz1 = 0; ixyz1 < 3; ixyz1++) {
            std::cout << " ";
            for (auto ixyz2 = 0; ixyz2 < 3; ixyz2++) {
                std::cout << std::scientific << std::setw(15) << std::setprecision(6) << state.u_tensor[ixyz1][ixyz2];
            }
            std::cout << '\n';
        }
    }

    std::cout << "\n Final structure at " << temp << " K :\n";
    relaxation->print_structure_and_symmetry(state, nullptr);

    if (last_temperature) {
        std::cout << " ----------------------------------------------------------------\n\n";
    } else {
        std::cout << '\n';
    }
}

void ScphQhaCommon::run_structural_optimization_loop(IRelaxationModel &model, StructuralOptLoopContext &ctx)
{
    auto i_temp_loop = -1;

    for (double temp: ctx.vec_temp) {
        i_temp_loop++;
        auto iT = static_cast<unsigned int>((temp - ctx.Tmin) / ctx.dT);

        if (writes->getVerbosity() > 0) {
            std::cout << "\n ================================================================\n";
            std::cout << "  Temperature = " << temp << " K    (" << std::setw(4) << i_temp_loop + 1 << " of "
                      << std::setw(4) << ctx.NT << ")\n";
            std::cout << " ================================================================\n\n";
        }

        model.before_init_structure(iT, static_cast<unsigned int>(i_temp_loop), temp, ctx.converged_prev);

        relaxation->set_init_structure_atT(ctx.structure_state,
                                           ctx.converged_prev,
                                           ctx.str_diverged,
                                           i_temp_loop,
                                           omega2_harmonic,
                                           evec_harmonic);

        model.after_init_structure(iT, temp);

        print_initial_structure(ctx.structure_state, ctx.relax_mode);

        relaxation->write_stepresfile_header_atT(ctx.fout_step_q0, ctx.fout_step_u0, ctx.fout_step_u_tensor, temp);

        relaxation->write_stepresfile(ctx.structure_state,
                                      0,
                                      ctx.fout_step_q0,
                                      ctx.fout_step_u0,
                                      ctx.fout_step_u_tensor);

        if (writes->getVerbosity() > 0) std::cout << " Start structural optimization at " << temp << " K.\n";

        // per-step records for the optimization-history table printed below
        std::vector<StructOptStepRecord> step_history;

        bool converged_this_temp = false;
        int i_str_loop;
        for (i_str_loop = 0; i_str_loop < relaxation->max_str_iter; i_str_loop++) {

            if (writes->getVerbosity() > 0) {
                std::cout << "\n ----------------------------------------------------------------\n";
                std::cout << "  Structure opt. step " << std::setw(4) << i_str_loop + 1 << " of "
                          << relaxation->max_str_iter << "    (T = " << temp << " K)\n";
                std::cout << " ----------------------------------------------------------------\n";
            }

            const auto status = model.do_structure_step(iT, temp, i_str_loop, step_history);
            switch (status) {
            case StructOptStepStatus::Continue:
                break;
            case StructOptStepStatus::SolverFailedRetry:
                continue;
            case StructOptStepStatus::Converged:
            case StructOptStepStatus::Diverged:
            case StructOptStepStatus::Aborted:
                goto structure_loop_done;
            }
        }

    structure_loop_done:
        model.after_structure_loop(iT, temp, i_str_loop, converged_this_temp);

        Relaxation::print_optimization_history(step_history,
                                               temp,
                                               ctx.relax_mode == RelaxationStrMode::CoordinatesAndCell,
                                               model.history_has_scp_column(),
                                               writes->getVerbosity());

        print_final_structure(ctx.structure_state, ctx.relax_mode, temp, i_temp_loop == static_cast<int>(ctx.NT) - 1);

        model.record_v0(iT);

        // print obtained structure
        relaxation->calculate_u0(ctx.structure_state.q0, ctx.structure_state.u0, omega2_harmonic, evec_harmonic);

        relaxation->write_resfile_atT(ctx.structure_state, temp, ctx.fout_q0, ctx.fout_u0, ctx.fout_u_tensor);

        model.finalize_temperature(iT, temp, converged_this_temp, ctx.converged_prev);
    }

    model.print_run_summary();
}
