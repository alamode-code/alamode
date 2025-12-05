/*
 scph_io.cpp

 Copyright (c) 2015 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

/*
 Functions for loading/saving and writing SCPH data.
 These handle file I/O operations for dynamical matrices, force constants,
 and restart files.

 Functions included:
 - load_scph_dymat_from_file: Load SCPH dynamical matrices from restart file
 - store_scph_dymat_to_file: Save SCPH dynamical matrices to restart file
 - zerofill_harmonic_dymat_renormalize: Initialize harmonic renormalization array
 - write_anharmonic_correction_fc2: Write anharmonic force constant corrections
 - mpi_bcast_complex: MPI broadcast utility for complex arrays
*/

#include "scph.h"
#include "constants.h"
#include "dynamical.h"
#include "error.h"
#include "kpoint.h"
#include "memory.h"
#include "mpi_common.h"
#include "parsephon.h"
#include "system.h"
#include "write_phonons.h"
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace PHON_NS;

void Scph::load_scph_dymat_from_file(std::complex<double> ****dymat_out, std::string filename_dymat,
                                     const KpointMeshUniform *kmesh_dense_in, const KpointMeshUniform *kmesh_coarse_in,
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

void Scph::store_scph_dymat_to_file(const std::complex<double> *const *const *const *dymat_in,
                                    std::string filename_dymat, const KpointMeshUniform *kmesh_dense_in,
                                    const KpointMeshUniform *kmesh_coarse_in, const unsigned int nonanalytic_in,
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


void Scph::zerofill_harmonic_dymat_renormalize(std::complex<double> ****delta_harmonic_dymat_renormalize,
                                               unsigned int NT) const
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

void Scph::write_anharmonic_correction_fc2(std::complex<double> ****delta_dymat, const unsigned int NT,
                                           const KpointMeshUniform *kmesh_coarse_in, MinimumDistList ***mindist_list_in,
                                           const bool is_qha, const int type)
{
    // Output anharmonically-renormalized IFC to file

    unsigned int i, j;
    const auto Tmin = system->Tmin;
    const auto dT = system->dT;
    double ***delta_fc2;
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
    if (!ofs_fc2) exit("write_anharmonic_correction_fc2", "Cannot open file_fc2");

    const auto ncell = kmesh_coarse_in->nk_i[0] * kmesh_coarse_in->nk_i[1] * kmesh_coarse_in->nk_i[2];

    allocate(delta_fc2, ns, ns, ncell);

    ofs_fc2.precision(10);
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
            ofs_fc2 << std::setw(20) << system->get_primcell().x_fractional(i, j);
        }
        ofs_fc2 << std::setw(5) << system->get_primcell().kind[i] + 1 << '\n';
    }

    for (unsigned int iT = 0; iT < NT; ++iT) {
        const auto temp = Tmin + dT * static_cast<double>(iT);

        ofs_fc2 << "# Temp = " << temp << '\n';

        for (is = 0; is < ns; ++is) {
            iat = is / 3;

            for (js = 0; js < ns; ++js) {
                jat = js / 3;

                for (icell = 0; icell < ncell; ++icell) {
                    delta_fc2[is][js][icell] = delta_dymat[iT][is][js][icell].real() *
                                               std::sqrt(system->get_mass_prim()[iat] * system->get_mass_prim()[jat]);
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
                         it != mindist_list_in[iat][jat][icell].shift.cend();
                         ++it)
                    {

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

void Scph::mpi_bcast_complex(std::complex<double> ****data, const unsigned int NT, const unsigned int nk,
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

