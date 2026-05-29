/*
 scph_qha_common.cpp

 Copyright (c) 2026

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "scph_qha_common.h"
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include "parsephon.h"

using namespace PHON_NS;

ScphQhaCommon::ScphQhaCommon(class PHON *phon) : Pointers(phon)
{}

ScphQhaCommon::~ScphQhaCommon() = default;

void ScphQhaCommon::initialize_variables()
{
    evec_harmonic = nullptr;
    omega2_harmonic = nullptr;
    phi3_reciprocal = nullptr;
    phi4_reciprocal = nullptr;
    kmesh_coarse = nullptr;
    kmesh_dense = nullptr;
    mindist_list = nullptr;
    phase_factor = nullptr;
    mat_transform_sym = nullptr;
    kmap_coarse_to_dense.clear();
    dymat_harm_short.clear();
    dymat_harm_long.clear();
    ialgo = 0;
    selfenergy_offdiagonal = true;
}

void ScphQhaCommon::deallocate_variables()
{
    if (mindist_list) {
        deallocate(mindist_list);
    }
    if (evec_harmonic) {
        deallocate(evec_harmonic);
    }
    if (omega2_harmonic) {
        deallocate(omega2_harmonic);
    }
    if (phi3_reciprocal) {
        deallocate(phi3_reciprocal);
    }
    if (phi4_reciprocal) {
        deallocate(phi4_reciprocal);
    }
    if (mat_transform_sym) {
        deallocate(mat_transform_sym);
    }

    delete kmesh_coarse;
    kmesh_coarse = nullptr;
    delete kmesh_dense;
    kmesh_dense = nullptr;
    delete phase_factor;
    phase_factor = nullptr;

    kmap_coarse_to_dense.clear();
    dymat_harm_short.clear();
    dymat_harm_long.clear();
}

void ScphQhaCommon::setup_kmesh(unsigned int kmesh_dense_input[3], unsigned int kmesh_coarse_input[3],
                                const char *mode_name, const char *mapping_error_message)
{
    MPI_Bcast(&kmesh_dense_input[0], 3, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&kmesh_coarse_input[0], 3, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    delete kmesh_coarse;
    delete kmesh_dense;
    kmesh_coarse = new KpointMeshUniform(kmesh_coarse_input);
    kmesh_dense = new KpointMeshUniform(kmesh_dense_input);
    kmesh_coarse->setup(symmetry->SymmList, system->get_primcell().reciprocal_lattice_vector, true);
    kmesh_dense->setup(symmetry->SymmList, system->get_primcell().reciprocal_lattice_vector, true);

    if (mympi->my_rank == 0) {
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

    const auto info_mapping = kpoint->get_kmap_coarse_to_dense(kmesh_coarse, kmesh_dense, kmap_coarse_to_dense);
    if (info_mapping == 1) {
        exit("setup_kmesh", mapping_error_message);
    }

    kmesh_coarse->setup_kpoint_symmetry(symmetry->SymmListWithMap);
}

void ScphQhaCommon::setup_eigvecs()
{
    const auto ns = dynamical->neval;

    if (mympi->my_rank == 0) {
        std::cout << '\n' << " Diagonalizing dynamical matrices for all k points ... ";
    }

    if (evec_harmonic) {
        deallocate(evec_harmonic);
    }
    if (omega2_harmonic) {
        deallocate(omega2_harmonic);
    }

    allocate(evec_harmonic, kmesh_dense->nk, ns, ns);
    allocate(omega2_harmonic, kmesh_dense->nk, ns);

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

    if (mympi->my_rank == 0) {
        std::cout << "done !\n";
    }
}

void ScphQhaCommon::setup_structural_data()
{
    if (mindist_list) {
        deallocate(mindist_list);
    }
    if (mat_transform_sym) {
        deallocate(mat_transform_sym);
    }

    system->get_minimum_distances(kmesh_coarse->nk_i, mindist_list);
    dynamical->get_symmetry_gamma_dynamical(kmesh_coarse,
                                            system->get_primcell().number_of_atoms,
                                            system->get_primcell().x_fractional,
                                            symmetry->SymmListWithMap,
                                            mat_transform_sym);
}

void ScphQhaCommon::setup_pp_interaction(const bool prepare_v3)
{
    if (mympi->my_rank == 0) {
        if (prepare_v3) {
            std::cout << " Preparing for calculating V3 & V4  ...";
        } else {
            std::cout << " Preparing for calculating V4  ...";
        }
    }

    if (anharmonic_core->quartic_mode != 1) {
        exit("setup_pp_interaction", "quartic_mode should be 1 for SCPH");
    }

    if (phi3_reciprocal) {
        deallocate(phi3_reciprocal);
    }
    if (phi4_reciprocal) {
        deallocate(phi4_reciprocal);
    }

    if (prepare_v3) {
        allocate(phi3_reciprocal, anharmonic_core->get_ngroup_fcs(3));
    }
    allocate(phi4_reciprocal, anharmonic_core->get_ngroup_fcs(4));

    delete phase_factor;
    phase_factor = new PhaseFactorStorage(kmesh_dense->nk_i);
    phase_factor->create(true);

    if (mympi->my_rank == 0) {
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
        std::cout << " done.\n";
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
    // auto file_dymat = input->job_title + ".scph_dymat";
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
    std::cout << "  " << std::setw(input->job_title.length() + 12) << std::left << file_dymat;
    std::cout << " : Anharmonic dynamical matrix (restart file)\n";
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
