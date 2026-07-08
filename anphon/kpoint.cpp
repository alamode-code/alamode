/*
kpoint.cpp

Copyright (c) 2014 Terumasa Tadano

This file is distributed under the terms of the MIT license.
Please see the file 'LICENCE.txt' in the root directory
or http://opensource.org/licenses/mit-license.php for information.
*/

#include "kpoint.h"
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include "constants.h"
#include "error.h"
#include "mathfunctions.h"
#include "memory.h"
#include "mpi_common.h"
#include "niggli_wrapper.h"
#include "phonon_dos.h"
#include "symmetry_core.h"
#include "system.h"
#include "timer.h"

using namespace PHON_NS;

Kpoint::Kpoint(PHON *phon) : Pointers(phon)
{
    set_default_variables();
}

Kpoint::~Kpoint()
{
    deallocate_variables();
}

void Kpoint::set_default_variables()
{}

void Kpoint::deallocate_variables()
{
    kpoint_bs.reset();
    kpoint_general.reset();
}

void Kpoint::kpoint_setups(const std::string mode)
{
    MPI_Bcast(&kpoint_mode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (mympi->my_rank == 0) {
        std::cout << '\n';
        std::cout << " ==========\n";
        std::cout << "  K points \n";
        std::cout << " ==========\n\n";
    }

    switch (kpoint_mode) {
    case 0:

        if (mympi->my_rank == 0) {
            std::cout << "  KPMODE = 0 : Calculation on given k points\n";
        }

        setup_kpoint_given(kpInp, system->get_primcell().reciprocal_lattice_vector);

        if (mympi->my_rank == 0) {
            std::cout << "  Number of k points : " << kpoint->kpoint_general->nk << "\n\n";
            std::cout << "  List of k points : " << '\n';
            for (auto i = 0; i < kpoint->kpoint_general->nk; ++i) {
                std::cout << std::setw(5) << i + 1 << ":";
                for (auto j = 0; j < 3; ++j) {
                    std::cout << std::setw(15) << kpoint->kpoint_general->xk[i][j];
                }
                std::cout << '\n';
            }
            std::cout << '\n';
        }

        break;

    case 1:

        if (mympi->my_rank == 0) {
            std::cout << "  KPMODE = 1: Band structure calculation\n";
        }

        setup_kpoint_band(kpInp, system->get_primcell().reciprocal_lattice_vector);
        if (mympi->my_rank == 0) {
            std::cout << "  Number of paths : " << kpInp.size() << "\n\n";
            std::cout << "  List of k paths : " << '\n';

            for (auto i = 0; i < kpInp.size(); ++i) {
                std::cout << std::setw(4) << i + 1 << ":";
                std::cout << std::setw(3) << kpInp[i].kpelem[0];
                std::cout << " (";
                for (int k = 0; k < 3; ++k) {
                    std::cout << std::setprecision(4) << std::setw(8) << std::atof(kpInp[i].kpelem[k + 1].c_str());
                }
                std::cout << ")";
                std::cout << std::setw(3) << kpInp[i].kpelem[4];
                std::cout << " (";
                for (int k = 0; k < 3; ++k) {
                    std::cout << std::setprecision(4) << std::setw(8) << std::atof(kpInp[i].kpelem[k + 5].c_str());
                }
                std::cout << ")";
                std::cout << std::setw(4) << kpInp[i].kpelem[8] << '\n';
            }
            std::cout << '\n';
            std::cout << "  Number of k points : " << kpoint_bs->nk << "\n\n";
        }

        break;

    case 2:

        if (mympi->my_rank == 0) {
            std::cout << "  KPMODE = 2: Uniform grid\n";
        }

        unsigned int nk_tmp[3];
        nk_tmp[0] = 0;
        nk_tmp[1] = 0;
        nk_tmp[2] = 0;
        if (mympi->my_rank == 0) {
            for (auto i = 0; i < 3; ++i) {
                const auto nk_in = std::atoi(kpInp[0].kpelem[i].c_str());
                if (nk_in < 1) {
                    exit("kpoint_setups", "Each k-point mesh dimension (KPMODE=2) must be a positive integer.");
                }
                nk_tmp[i] = nk_in;
            }
        }
        MPI_Bcast(&nk_tmp[0], 3, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        dos->create_kmesh_dos(nk_tmp,
                              symmetry->SymmList,
                              system->get_primcell().reciprocal_lattice_vector,
                              symmetry->time_reversal_sym);

        if (mympi->my_rank == 0) {
            std::cout << "  Gamma-centered uniform grid with the following mesh density: \n";
            std::cout << "  nk1:" << std::setw(4) << dos->kmesh_dos->nk_i[0] << '\n';
            std::cout << "  nk2:" << std::setw(4) << dos->kmesh_dos->nk_i[1] << '\n';
            std::cout << "  nk3:" << std::setw(4) << dos->kmesh_dos->nk_i[2] << "\n\n";
            std::cout << "  Number of k points : " << dos->kmesh_dos->nk << '\n';
            std::cout << "  Number of irreducible k points : " << dos->kmesh_dos->nk_irred << "\n\n";
            std::cout << "  List of irreducible k points (reciprocal coordinate, weight) : \n";

            for (auto i = 0; i < dos->kmesh_dos->nk_irred; ++i) {
                std::cout << "  " << std::setw(5) << i + 1 << ":";
                for (auto j = 0; j < 3; ++j) {
                    std::cout << std::setprecision(5) << std::setw(14) << std::scientific
                              << dos->kmesh_dos->kpoint_irred_all[i][0].kval[j];
                }
                std::cout << std::setprecision(6) << std::setw(11) << std::fixed << dos->kmesh_dos->weight_k[i] << '\n';
            }
            std::cout << '\n';
        }

        break;

    default:
        exit("setup_kpoints", "This cannot happen.");
    }
}

void Kpoint::setup_kpoint_given(const std::vector<KpointInp> &kpinfo, const Eigen::Matrix3d &rlavec_p)
{
    int i;
    NDArray<double, 2> k;
    NDArray<double, 2> kdirec;
    unsigned int n = kpinfo.size();

    MPI_Bcast(&n, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    k.resize(n, 3);
    kdirec.resize(n, 3);

    if (mympi->my_rank == 0) {
        int j = 0;
        for (const auto &it: kpinfo) {
            for (i = 0; i < 3; ++i) {
                k[j][i] = std::atof(it.kpelem[i].c_str());
                // For the manual k-point mode, we shift the kdirec vector the first BZ.
                // With this treatment, the non-analytic correction becomes zero
                // at arbitrary G points, e.g., q=(0,0,0), (1,0,0), (0,1,0) ....
                kdirec[j][i] = k[j][i] - static_cast<double>(nint(k[j][i]));
            }

            rotvec(kdirec[j], kdirec[j], rlavec_p, 'T');

            const auto norm = kdirec[j][0] * kdirec[j][0] + kdirec[j][1] * kdirec[j][1] + kdirec[j][2] * kdirec[j][2];

            if (norm > eps) {
                for (i = 0; i < 3; ++i) kdirec[j][i] /= std::sqrt(norm);
            }

            ++j;
        }
    }

    MPI_Bcast(&k[0][0], 3 * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&kdirec[0][0], 3 * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    kpoint_general = std::make_unique<KpointGeneral>(n, k, kdirec);

    k.clear();
    kdirec.clear();
}

void Kpoint::setup_kpoint_band(const std::vector<KpointInp> &kpinfo, const Eigen::Matrix3d &rlavec_p)
{
    int j, k;

    NDArray<double, 2> xk_tmp;
    NDArray<double, 2> kdirec_tmp;
    NDArray<double, 1> axis_tmp;
    unsigned int n = 0;

    if (mympi->my_rank == 0) {

        NDArray<std::string, 2> kp_symbol;
        NDArray<unsigned int, 1> nk_path;
        NDArray<double, 2> k_start;
        NDArray<double, 2> k_end;

        const auto npath = kpinfo.size();

        kp_symbol.resize(npath, 2);
        k_start.resize(npath, 3);
        k_end.resize(npath, 3);
        nk_path.resize(npath);

        n = 0;
        int i = 0;

        for (const auto &it: kpinfo) {
            kp_symbol[i][0] = it.kpelem[0];
            kp_symbol[i][1] = it.kpelem[4];

            for (j = 0; j < 3; ++j) {
                k_start[i][j] = std::atof(it.kpelem[j + 1].c_str());
                k_end[i][j] = std::atof(it.kpelem[j + 5].c_str());
            }
            const auto nk_path_in = std::atoi(it.kpelem[8].c_str());
            if (nk_path_in < 2) {
                exit("setup_kpoint_band",
                     "The number of points along a band-structure segment must be an integer >= 2.");
            }
            nk_path[i] = nk_path_in;
            n += nk_path[i];
            ++i;
        }

        xk_tmp.resize(n, 3);
        kdirec_tmp.resize(n, 3);
        axis_tmp.resize(n);

        unsigned int ik = 0;
        double direc_tmp[3], tmp[3];

        for (i = 0; i < npath; ++i) {
            for (j = 0; j < 3; ++j) {
                direc_tmp[j] = k_end[i][j] - k_start[i][j];
            }

            rotvec(direc_tmp, direc_tmp, rlavec_p, 'T');
            auto norm = pow2(direc_tmp[0]) + pow2(direc_tmp[1]) + pow2(direc_tmp[2]);
            norm = std::sqrt(norm);

            if (norm > eps) {
                for (j = 0; j < 3; ++j) direc_tmp[j] /= norm;
            }

            for (j = 0; j < nk_path[i]; ++j) {
                for (k = 0; k < 3; ++k) {
                    xk_tmp[ik][k] = k_start[i][k] + (k_end[i][k] - k_start[i][k]) * static_cast<double>(j) /
                                                        static_cast<double>(nk_path[i] - 1);

                    kdirec_tmp[ik][k] = direc_tmp[k];
                }

                if (ik == 0) {
                    axis_tmp[ik] = 0.0;
                } else {
                    if (j == 0) {
                        axis_tmp[ik] = axis_tmp[ik - 1];
                    } else {
                        for (k = 0; k < 3; ++k) tmp[k] = xk_tmp[ik][k] - xk_tmp[ik - 1][k];
                        rotvec(tmp, tmp, rlavec_p, 'T');
                        axis_tmp[ik] =
                            axis_tmp[ik - 1] + std::sqrt(tmp[0] * tmp[0] + tmp[1] * tmp[1] + tmp[2] * tmp[2]);
                    }
                }
                ++ik;
            }
        }
        nk_path.clear();
        k_start.clear();
        k_end.clear();
    }

    MPI_Bcast(&n, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    if (mympi->my_rank > 0) {
        xk_tmp.resize(n, 3);
        kdirec_tmp.resize(n, 3);
        axis_tmp.resize(n);
    }

    MPI_Bcast(&xk_tmp[0][0], 3 * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&kdirec_tmp[0][0], 3 * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&axis_tmp[0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    kpoint_bs = std::make_unique<KpointBandStructure>(n, xk_tmp, kdirec_tmp, axis_tmp);

    xk_tmp.clear();
    kdirec_tmp.clear();
    axis_tmp.clear();
}

void KpointMeshUniform::setup(const std::vector<SymmetryOperation> &symmlist, const Eigen::Matrix3d &rlavec_p,
                              const bool time_reversal_symmetry, const bool niggli_reduce_in)
{
    const bool usesym = true;

    niggli_reduced = niggli_reduce_in;

    if (niggli_reduced) {
        gen_kmesh_niggli(symmlist, rlavec_p, usesym, time_reversal_symmetry);
    } else {
        gen_kmesh(symmlist, rlavec_p, usesym, time_reversal_symmetry);
    }

    for (auto i = 0; i < nk; ++i) {
        for (auto j = 0; j < 3; ++j) kvec_na[i][j] = xk[i][j];

        rotvec(&kvec_na[i][0], &kvec_na[i][0], rlavec_p, 'T');
        const auto norm = kvec_na[i][0] * kvec_na[i][0] + kvec_na[i][1] * kvec_na[i][1] + kvec_na[i][2] * kvec_na[i][2];

        if (norm > eps) {
            for (auto j = 0; j < 3; ++j) kvec_na[i][j] /= std::sqrt(norm);
        }
    }

    nk_irred = kpoint_irred_all.size();
    weight_k.resize(nk_irred);
    for (auto i = 0; i < nk_irred; ++i) {
        weight_k[i] = static_cast<double>(kpoint_irred_all[i].size()) / static_cast<double>(nk);
    }
    gen_nkminus();

    kmap_to_irreducible.resize(nk);
    for (auto i = 0; i < nk_irred; ++i) {
        for (auto j = 0; j < kpoint_irred_all[i].size(); ++j) {
            kmap_to_irreducible[kpoint_irred_all[i][j].knum] = i;
        }
    }
    // Compute small group of every irreducible k points for later use
    small_group_of_k.resize(nk_irred);
    set_small_groups_k_irred(usesym, symmlist);
}

void KpointMeshUniform::gen_kmesh(const std::vector<SymmetryOperation> &symmlist, const Eigen::Matrix3d &rlavec_p,
                                  const bool usesym, const bool time_reversal_symmetry)
{
    // Generates the uniform grid points in the reciprocal space
    // Search the periodic images that minimize |q+G|
    // TODO: save all q+G having the same Euclidean distance from origin for later use

    unsigned int ik;
    NDArray<double, 2> xkr;
    unsigned int nsym;
    Eigen::Matrix3d rlat;

    // transpose the input matrix for reciprocal lattice
    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            rlat(i, j) = rlavec_p(j, i);
        }
    }
    std::vector<std::vector<int>> gvec_shift;
    std::vector<int> gvec_tmp(3);

    gvec_tmp[0] = 0;
    gvec_tmp[1] = 0;
    gvec_tmp[2] = 0;
    gvec_shift.emplace_back(gvec_tmp);

    for (auto ia = -1; ia <= 1; ++ia) {
        for (auto ja = -1; ja <= 1; ++ja) {
            for (auto ka = -1; ka <= 1; ++ka) {

                if (ia == 0 && ja == 0 && ka == 0) continue;

                gvec_tmp[0] = ia;
                gvec_tmp[1] = ja;
                gvec_tmp[2] = ka;
                gvec_shift.emplace_back(gvec_tmp);
            }
        }
    }

    xkr.resize(nk, 3);
    for (unsigned int ix = 0; ix < nk_i[0]; ++ix) {
        for (unsigned int iy = 0; iy < nk_i[1]; ++iy) {
            for (unsigned int iz = 0; iz < nk_i[2]; ++iz) {
                ik = iz + iy * nk_i[2] + ix * nk_i[2] * nk_i[1];
                xkr[ik][0] = static_cast<double>(ix) / static_cast<double>(nk_i[0]);
                xkr[ik][1] = static_cast<double>(iy) / static_cast<double>(nk_i[1]);
                xkr[ik][2] = static_cast<double>(iz) / static_cast<double>(nk_i[2]);
            }
        }
    }

    Eigen::MatrixXd xkr_periodic(3, 27);
    std::vector<double> distances(27);
    for (ik = 0; ik < nk; ++ik) {
        for (auto icell = 0; icell < 27; ++icell) {
            for (auto i = 0; i < 3; ++i) {
                xkr_periodic(i, icell) = xkr[ik][i] + static_cast<double>(gvec_shift[icell][i]);
            }
        }
        xkr_periodic = rlat * xkr_periodic;

        std::vector<int> idx(27);
        std::iota(idx.begin(), idx.end(), 0);

        for (auto icell = 0; icell < 27; ++icell) {
            auto norm = std::sqrt(xkr_periodic(0, icell) * xkr_periodic(0, icell) +
                                  xkr_periodic(1, icell) * xkr_periodic(1, icell) +
                                  xkr_periodic(2, icell) * xkr_periodic(2, icell));
            distances[icell] = norm;
        }
        // find the periodic image having the minimum distance from (0, 0, 0)
        std::stable_sort(idx.begin(), idx.end(), [&distances](size_t i1, size_t i2) {
            return distances[i1] < distances[i2];
        });

        // Select the first periodic image that gives the shortest |q+G|
        for (auto i = 0; i < 3; ++i) {
            xkr[ik][i] += static_cast<double>(gvec_shift[idx[0]][i]);
        }
    }

    if (usesym) {
        nsym = symmlist.size();
    } else {
        nsym = 1;
    }
    reduce_kpoints(nsym, symmlist, time_reversal_symmetry, xkr);

    for (ik = 0; ik < nk; ++ik) {
        for (unsigned int i = 0; i < 3; ++i) {
            xk[ik][i] = xkr[ik][i];
        }
    }

    xkr.clear();
}

void KpointMeshUniform::gen_kmesh_niggli(const std::vector<SymmetryOperation> &symmlist,
                                         const Eigen::Matrix3d &rlavec_p, const bool usesym,
                                         const bool time_reversal_symmetry)
{
    // Generates the uniform grid points in the reciprocal space
    // Search the periodic images that minimize |q+G|.
    // Niggli reduction is done for searching Gs that minimize |q+G| exhaustively
    // from the centering cell and the surrounding 26 periodic images.
    // TODO: save all q+G having the same Euclidean distance from origin for later use

    unsigned int ik;
    unsigned int nsym;
    Eigen::Matrix3d rlat, rlat_reduced, c_matrix, c_matrix_inv;

    // transpose the input matrix for reciprocal lattice
    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            rlat(i, j) = rlavec_p(j, i);
        }
    }
    auto success = niggli_reduction(rlat, rlat_reduced, c_matrix);
    if (!success) {
        exit("gen_kmesh_niggli", "Niggli reduction failed");
    }

    c_matrix_inv = c_matrix.inverse();

    std::vector<std::vector<int>> gvec_shift;
    std::vector<int> gvec_tmp(3);

    gvec_tmp[0] = 0;
    gvec_tmp[1] = 0;
    gvec_tmp[2] = 0;
    gvec_shift.emplace_back(gvec_tmp);

    for (auto ia = -1; ia <= 1; ++ia) {
        for (auto ja = -1; ja <= 1; ++ja) {
            for (auto ka = -1; ka <= 1; ++ka) {

                if (ia == 0 && ja == 0 && ka == 0) continue;

                gvec_tmp[0] = ia;
                gvec_tmp[1] = ja;
                gvec_tmp[2] = ka;
                gvec_shift.emplace_back(gvec_tmp);
            }
        }
    }


    Eigen::MatrixXd xkr(3, nk);

    // Generate regular mesh in the original basis
    for (unsigned int ix = 0; ix < nk_i[0]; ++ix) {
        for (unsigned int iy = 0; iy < nk_i[1]; ++iy) {
            for (unsigned int iz = 0; iz < nk_i[2]; ++iz) {
                ik = iz + iy * nk_i[2] + ix * nk_i[2] * nk_i[1];
                xkr(0, ik) = static_cast<double>(ix) / static_cast<double>(nk_i[0]);
                xkr(1, ik) = static_cast<double>(iy) / static_cast<double>(nk_i[1]);
                xkr(2, ik) = static_cast<double>(iz) / static_cast<double>(nk_i[2]);
            }
        }
    }
    // Convert to the reduced basis
    xkr = c_matrix_inv * xkr;

    // Move back to the 0<=q<1 segment
    for (ik = 0; ik < nk; ++ik) {
        for (auto i = 0; i < 3; ++i) {
            xkr(i, ik) = std::fmod(xkr(i, ik), 1.0);
            if (xkr(i, ik) < 0.0) xkr(i, ik) += 1.0;
        }
    }

    // Find the G-vector that minimizes |q+G| in the reduced basis
    Eigen::MatrixXd xkr_periodic(3, 27);
    std::vector<double> distances(27);
    for (ik = 0; ik < nk; ++ik) {
        for (auto icell = 0; icell < 27; ++icell) {
            for (auto i = 0; i < 3; ++i) {
                xkr_periodic(i, icell) = xkr(i, ik) + static_cast<double>(gvec_shift[icell][i]);
            }
        }
        xkr_periodic = rlat_reduced * xkr_periodic;

        std::vector<int> idx(27);
        std::iota(idx.begin(), idx.end(), 0);

        for (auto icell = 0; icell < 27; ++icell) {
            auto norm = std::sqrt(xkr_periodic(0, icell) * xkr_periodic(0, icell) +
                                  xkr_periodic(1, icell) * xkr_periodic(1, icell) +
                                  xkr_periodic(2, icell) * xkr_periodic(2, icell));
            distances[icell] = norm;
        }
        // find the periodic image having the minimum distance from (0, 0, 0)
        std::stable_sort(idx.begin(), idx.end(), [&distances](size_t i1, size_t i2) {
            return distances[i1] < distances[i2];
        });

        // Select the first G that minimizes |q+G|
        for (auto i = 0; i < 3; ++i) {
            xkr(i, ik) += static_cast<double>(gvec_shift[idx[0]][i]);
        }
    }

    // Move back to the original basis
    xkr = c_matrix * xkr;

    NDArray<double, 2> xkr_arr;
    xkr_arr.resize(nk, 3);

    for (ik = 0; ik < nk; ++ik) {
        for (auto i = 0; i < 3; ++i) {
            xkr_arr[ik][i] = xkr(i, ik);
        }
    }

    if (usesym) {
        nsym = symmlist.size();
    } else {
        nsym = 1;
    }
    reduce_kpoints(nsym, symmlist, time_reversal_symmetry, xkr_arr);

    for (ik = 0; ik < nk; ++ik) {
        for (unsigned int i = 0; i < 3; ++i) {
            xk[ik][i] = xkr_arr[ik][i];
        }
    }

    xkr_arr.clear();
}

void KpointMeshUniform::reduce_kpoints(const unsigned int nsym, const std::vector<SymmetryOperation> &symmlist,
                                       const bool time_reversal_symmetry, const double *const *xkr)
{
    unsigned int ik;
    unsigned int i, j;
    int isym;

    NDArray<bool, 1> k_found;

    std::vector<KpointList> k_group;
    std::vector<double> ktmp;

    double srot[3][3];
    double xk_sym[3], xk_orig[3];
    double srot_inv[3][3], srot_inv_t[3][3];

    NDArray<double, 3> symop_k;

    symop_k.resize(nsym, 3, 3);

    for (isym = 0; isym < nsym; ++isym) {

        for (i = 0; i < 3; ++i) {
            for (j = 0; j < 3; ++j) {
                srot[i][j] = static_cast<double>(symmlist[isym].rotation(i, j));
            }
        }

        invmat3(srot_inv, srot);
        transpose3(srot_inv_t, srot_inv);

        for (i = 0; i < 3; ++i) {
            for (j = 0; j < 3; ++j) {
                symop_k[isym][i][j] = srot_inv_t[i][j];
            }
        }
    }

    kpoint_irred_all.clear();

    k_found.resize(nk);

    for (ik = 0; ik < nk; ++ik) k_found[ik] = false;

    for (ik = 0; ik < nk; ++ik) {

        if (k_found[ik]) continue;

        k_group.clear();

        for (i = 0; i < 3; ++i) xk_orig[i] = xkr[ik][i];

        for (isym = 0; isym < nsym; ++isym) {

            rotvec(xk_sym, xk_orig, symop_k[isym]);

            for (i = 0; i < 3; ++i) xk_sym[i] = xk_sym[i] - nint(xk_sym[i]);

            int nloc = get_knum(xk_sym);

            if (nloc == -1) {

                //     exit("reduce_kpoints", "Cannot find the kpoint");

            } else {

                if (!k_found[nloc]) {
                    k_found[nloc] = true;
                    ktmp.clear();
                    ktmp.push_back(xk_sym[0]);
                    ktmp.push_back(xk_sym[1]);
                    ktmp.push_back(xk_sym[2]);

                    k_group.emplace_back(nloc, ktmp);
                }
            }

            // Time-reversal symmetry

            if (0) {

                for (i = 0; i < 3; ++i) xk_sym[i] *= -1.0;

                nloc = get_knum(xk_sym);

                if (nloc == -1) {

                    //     exit("reduce_kpoints", "Cannot find the kpoint");

                } else {

                    if (!k_found[nloc]) {
                        k_found[nloc] = true;
                        ktmp.clear();
                        ktmp.push_back(xk_sym[0]);
                        ktmp.push_back(xk_sym[1]);
                        ktmp.push_back(xk_sym[2]);

                        k_group.emplace_back(nloc, ktmp);
                    }
                }
            }
        }
        kpoint_irred_all.push_back(k_group);
    }

    k_found.clear();
    symop_k.clear();
}

void KpointMeshUniform::gen_nkminus()
{
    kindex_minus_xk.resize(nk);
    double minus_xk[3];

    for (unsigned int ik = 0; ik < nk; ++ik) {

        for (auto i = 0; i < 3; ++i) minus_xk[i] = -xk[ik][i];

        const auto ik_minus = get_knum(minus_xk);

        if (ik_minus == -1) {
            exit("gen_nkminus", "-xk doesn't exist on the mesh point.");
        }

        if (ik_minus < ik) continue;

        kindex_minus_xk[ik] = ik_minus;
        kindex_minus_xk[ik_minus] = ik;
    }
}

void KpointMeshUniform::set_small_groups_k_irred(const bool usesym, const std::vector<SymmetryOperation> &symmlist)
{
    small_group_of_k.resize(nk_irred);
    for (auto ik = 0; ik < nk_irred; ++ik) {
        small_group_of_k[ik] = get_small_group_of_k(kpoint_irred_all[ik][0].knum, usesym, symmlist);
    }
}

std::vector<int> KpointMeshUniform::get_small_group_of_k(const unsigned int ik, const bool usesym,
                                                         const std::vector<SymmetryOperation> &symmlist) const
{
    std::vector<int> small_group;
    small_group.clear();
    unsigned int nsym;
    if (usesym) {
        nsym = symmlist.size();
    } else {
        nsym = 1;
    }
    for (auto isym = 0; isym < nsym; ++isym) {
        const auto ksym = knum_sym(ik, symmlist[isym].rotation);
        if (ksym == ik) {
            small_group.push_back(isym);
        }
    }
    return small_group;
}

int KpointMeshUniform::knum_sym(const unsigned int ik, const Eigen::Matrix3i &rot) const
{
    // Returns kpoint index of S(symop_num)*xk[ik_in]
    // Works only for gamma-centered mesh calculations
    int i;

    double srot[3][3];
    double srot_inv[3][3], srot_inv_t[3][3];
    double xk_orig[3], xk_sym[3];

    for (i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            srot[i][j] = static_cast<double>(rot(i, j));
        }
    }

    invmat3(srot_inv, srot);
    transpose3(srot_inv_t, srot_inv);

    for (i = 0; i < 3; ++i) xk_orig[i] = xk[ik][i];

    rotvec(xk_sym, xk_orig, srot_inv_t);
    for (i = 0; i < 3; ++i) {
        xk_sym[i] = xk_sym[i] - nint(xk_sym[i]);
    }

    return get_knum(xk_sym);
}

int KpointMeshUniform::get_knum(const double xk[3]) const
{
    int i;
    double diff[3];
    double dnk[3];

    for (i = 0; i < 3; ++i) dnk[i] = static_cast<double>(nk_i[i]);
    for (i = 0; i < 3; ++i) diff[i] = static_cast<double>(nint(xk[i] * dnk[i])) - xk[i] * dnk[i];

    const auto norm = std::sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);

    if (norm >= eps12) return -1;

    const int iloc = nint(xk[0] * dnk[0] + 2.0 * dnk[0]) % nk_i[0];
    const int jloc = nint(xk[1] * dnk[1] + 2.0 * dnk[1]) % nk_i[1];
    const int kloc = nint(xk[2] * dnk[2] + 2.0 * dnk[2]) % nk_i[2];

    return kloc + nk_i[2] * jloc + nk_i[1] * nk_i[2] * iloc;
}

void Kpoint::mpi_broadcast_kplane_vector(const unsigned int nplane, std::vector<KpointPlane> *&kp_plane) const
{
    int j;
    NDArray<int, 2> naxis;
    NDArray<double, 2> xk_plane;

    for (int i = 0; i < nplane; ++i) {
        int nkp = kp_plane[i].size();

        MPI_Bcast(&nkp, 1, MPI_INT, 0, MPI_COMM_WORLD);

        naxis.resize(nkp, 2);
        xk_plane.resize(nkp, 3);

        if (mympi->my_rank == 0) {
            for (j = 0; j < nkp; ++j) {
                naxis[j][0] = kp_plane[i][j].n[0];
                naxis[j][1] = kp_plane[i][j].n[1];
                xk_plane[j][0] = kp_plane[i][j].k[0];
                xk_plane[j][1] = kp_plane[i][j].k[1];
                xk_plane[j][2] = kp_plane[i][j].k[2];
            }
        }

        MPI_Bcast(&naxis[0][0], 2 * nkp, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&xk_plane[0][0], 3 * nkp, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (mympi->my_rank > 0) {
            for (j = 0; j < nkp; ++j) {
                kp_plane[i].emplace_back(xk_plane[j], naxis[j]);
            }
        }
        naxis.clear();
        xk_plane.clear();
    }
}

int Kpoint::get_knum(const double xk[3], const unsigned int nk[3]) const
{
    int i;
    double diff[3];
    double dnk[3];

    for (i = 0; i < 3; ++i) dnk[i] = static_cast<double>(nk[i]);
    for (i = 0; i < 3; ++i) diff[i] = static_cast<double>(nint(xk[i] * dnk[i])) - xk[i] * dnk[i];

    const auto norm = std::sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);

    if (norm >= eps12) return -1;

    const int iloc = nint(xk[0] * dnk[0] + 2.0 * dnk[0]) % nk[0];
    const int jloc = nint(xk[1] * dnk[1] + 2.0 * dnk[1]) % nk[1];
    const int kloc = nint(xk[2] * dnk[2] + 2.0 * dnk[2]) % nk[2];

    return kloc + nk[2] * jloc + nk[1] * nk[2] * iloc;
}

void Kpoint::get_symmetrization_matrix_at_k(const double *xk_in, std::vector<int> &sym_list, double S_avg[3][3]) const
{
    int i, j;
    double srot[3][3];
    double srot_inv[3][3], srot_inv_t[3][3];
    double xk_orig[3], xk_sym[3];
    double xk_diff[3];

    sym_list.clear();

    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            S_avg[i][j] = 0.0;
        }
    }

    for (i = 0; i < 3; ++i) xk_orig[i] = xk_in[i];

    for (auto isym = 0; isym < symmetry->nsym; ++isym) {

        for (i = 0; i < 3; ++i) {
            for (j = 0; j < 3; ++j) {
                srot[i][j] = static_cast<double>(symmetry->SymmList[isym].rotation(i, j));
            }
        }

        invmat3(srot_inv, srot);
        transpose3(srot_inv_t, srot_inv);
        rotvec(xk_sym, xk_orig, srot_inv_t);

        for (i = 0; i < 3; ++i) {
            xk_sym[i] = xk_sym[i] - nint(xk_sym[i]);
            const auto diff = xk_sym[i] - xk_orig[i];
            xk_diff[i] = diff - nint(diff);
        }

        if (std::sqrt(pow2(xk_diff[0]) + pow2(xk_diff[1]) + pow2(xk_diff[2])) < eps10) {
            sym_list.push_back(isym);

            for (i = 0; i < 3; ++i) {
                for (j = 0; j < 3; ++j) {
                    S_avg[i][j] += srot_inv_t[i][j];
                }
            }
        }
    }

    if (sym_list.empty()) {
        static bool warning_issued = false;
        if (!warning_issued) {
            warn("get_symmetrization_matrix_at_k", "No small-group operation found. Identity symmetrizer is used.");
            warning_issued = true;
        }
        for (i = 0; i < 3; ++i) {
            S_avg[i][i] = 1.0;
        }
    } else {
        for (i = 0; i < 3; ++i) {
            for (j = 0; j < 3; ++j) {
                S_avg[i][j] /= static_cast<double>(sym_list.size());
            }
        }
    }
}

void Kpoint::get_commensurate_kpoints(const Eigen::Matrix3d &lavec_super, const Eigen::Matrix3d &lavec_prim,
                                      std::vector<std::vector<double>> &klist) const
{
    int i, j;
    Eigen::Matrix3d inv_lavec_super;
    Eigen::Matrix3d convmat;

    inv_lavec_super = lavec_super.inverse();
    convmat = (inv_lavec_super * lavec_prim).transpose();

    const auto det = convmat.determinant();
    const auto nkmax = static_cast<int>(std::ceil(1.0 / det));
    const auto tol = 1.0e-6;
    const auto max_denom = 10000; // for safety
    int k;

    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {

            const auto frac = std::abs(convmat(i, j));

            if (frac < tol) {
                convmat(i, j) = 0.0;
            } else {
                auto found_denom = false;
                for (k = 1; k < max_denom; ++k) {
                    if (std::abs(frac * static_cast<double>(k) - 1.0) < tol) {
                        found_denom = true;
                        break;
                    }
                }
                if (found_denom) {
                    const auto sign = (0.0 < convmat(i, j)) - (convmat(i, j) < 0.0);
                    convmat(i, j) = static_cast<double>(sign) / static_cast<double>(k);
                } else {
                    exit("get_commensurate_kpoints", "The denominator of the conversion matrix > 10000");
                }
            }
        }
    }

    const auto nmax_cell = static_cast<int>(nkmax) / 2 + 1;
    std::vector<int> signs({-1, 1});
    std::vector<std::vector<int>> comb;
    comb.clear();
    for (i = 0; i < nmax_cell; ++i) {
        for (j = 0; j < nmax_cell; ++j) {
            for (k = 0; k < nmax_cell; ++k) {
                for (const auto &sx: signs) {
                    for (const auto &sy: signs) {
                        for (const auto &sz: signs) {
                            comb.push_back({i * sx, j * sy, k * sz});
                        }
                    }
                }
            }
        }
    }

    double qtmp[3], qdiff[3];

    for (const auto &p: comb) {
        for (i = 0; i < 3; ++i) qtmp[i] = p[i];
        rotvec(qtmp, qtmp, convmat);

        for (i = 0; i < 3; ++i) {
            qtmp[i] = std::fmod(qtmp[i], 1.0);
            if (qtmp[i] >= 0.5) {
                qtmp[i] -= 1.0;
            } else if (qtmp[i] < -0.5) {
                qtmp[i] += 1.0;
            }
        }

        auto new_entry = true;

        for (const auto &elem: klist) {

            for (i = 0; i < 3; ++i) {
                qdiff[i] = std::fmod(qtmp[i] - elem[i], 1.0);
                if (qdiff[i] >= 0.5) {
                    qdiff[i] -= 1.0;
                } else if (qdiff[i] < -0.5) {
                    qdiff[i] += 1.0;
                }
            }

            const auto norm = std::sqrt(qdiff[0] * qdiff[0] + qdiff[1] * qdiff[1] + qdiff[2] * qdiff[2]);

            if (norm < tol) {
                new_entry = false;
                break;
            }
        }
        if (new_entry) klist.push_back({qtmp[0], qtmp[1], qtmp[2]});

        if (klist.size() == nkmax) break;
    }
}

void KpointMeshUniform::get_unique_triplet_k(const int ik, const std::vector<SymmetryOperation> &symmlist,
                                             const bool use_triplet_symmetry, const bool use_permutation_symmetry,
                                             std::vector<KsListGroup> &triplet, const int sign) const
{
    // This function returns the irreducible set of (k2, k3) satisfying the momentum conservation.
    // When sign = -1 (default), pairs satisfying - k1 + k2 + k3 = G are returned.
    // When sign =  1, pairs satisfying k1 + k2 + k3 = G are returned.
    //

    int i;
    unsigned int num_group_k;
    int ks_in[2];
    const auto knum = kpoint_irred_all[ik][0].knum;
    NDArray<bool, 1> flag_found;
    std::vector<KsList> kslist;
    double xk0[3], xk1[3], xk2[3];

    flag_found.resize(nk);

    if (use_triplet_symmetry) {
        num_group_k = small_group_of_k[ik].size();
    } else {
        num_group_k = 1;
    }

    for (i = 0; i < 3; ++i) xk0[i] = xk[knum][i];
    for (i = 0; i < nk; ++i) flag_found[i] = false;

    triplet.clear();

    for (auto ik1 = 0; ik1 < nk; ++ik1) {

        for (i = 0; i < 3; ++i) xk1[i] = xk[ik1][i];

        if (sign == -1) {
            for (i = 0; i < 3; ++i) xk2[i] = xk0[i] - xk1[i];
        } else if (sign == 1) {
            for (i = 0; i < 3; ++i) xk2[i] = -xk0[i] - xk1[i];
        } else {
            //exit("get_unituq_triplet_k", "Invalid sign");
        }

        const auto ik2 = get_knum(xk2);

        kslist.clear();

        if (ik1 > ik2 && use_permutation_symmetry) continue;

        // Add symmety-connected triplets to kslist
        for (auto isym = 0; isym < num_group_k; ++isym) {

            ks_in[0] = knum_sym(ik1, symmlist[small_group_of_k[ik][isym]].rotation);
            ks_in[1] = knum_sym(ik2, symmlist[small_group_of_k[ik][isym]].rotation);

            if (ks_in[0] == -1 || ks_in[1] == -1) {
                exit("get_unique_triplet_k",
                     "Cannot find the kpoint after rotation.\n"
                     " This indicates that the input kmesh grid is not compatible with the lattice symmetry.");
            }

            if (!flag_found[ks_in[0]]) {
                kslist.emplace_back(2, ks_in, small_group_of_k[ik][isym]);
                flag_found[ks_in[0]] = true;
            }

            if (ks_in[0] != ks_in[1] && use_permutation_symmetry && !flag_found[ks_in[1]]) {
                const auto tmp = ks_in[0];
                ks_in[0] = ks_in[1];
                ks_in[1] = tmp;

                kslist.emplace_back(2, ks_in, small_group_of_k[ik][isym]);
                flag_found[ks_in[0]] = true;
            }
        }
        if (!kslist.empty()) {
            triplet.emplace_back(kslist);
        }
    }

    flag_found.clear();
}

void KpointMeshUniform::get_unique_quartet_k(const int ik, const std::vector<SymmetryOperation> &symmlist,
                                             const bool use_quartet_symmetry, const bool use_permutation_symmetry,
                                             std::vector<KsListGroup> &quartet, const int sign) const
{
    // This function returns the irreducible set of (k2, k3, k4) satisfying the momentum conservation.
    // When sign = -1 (default), pairs satisfying - k1 + k2 + k3 + k4 = G are returned.
    // When sign =  1, pairs satisfying k1 + k2 + k3 + k4 = G are returned.
    //

    int i;
    unsigned int num_group_k;
    std::vector<int> ks_in(3);
    const auto knum = kpoint_irred_all[ik][0].knum;
    NDArray<bool, 2> flag_found;
    std::vector<KsList> kslist;
    double xk0[3], xk1[3], xk2[3], xk3[3];

    flag_found.resize(nk, nk);

    if (use_quartet_symmetry) {
        num_group_k = small_group_of_k[ik].size();
    } else {
        num_group_k = 1;
    }

    for (i = 0; i < 3; ++i) xk0[i] = xk[knum][i];
    for (i = 0; i < nk; ++i) {
        for (int j = 0; j < nk; ++j) {
            flag_found[i][j] = false;
        }
    }

    quartet.clear();

    for (int ik1 = 0; ik1 < nk; ++ik1) {

        for (i = 0; i < 3; ++i) xk1[i] = xk[ik1][i];

        int ik2_start = 0;
        if (use_permutation_symmetry) ik2_start = ik1;

        for (int ik2 = ik2_start; ik2 < nk; ++ik2) {
            for (i = 0; i < 3; ++i) xk2[i] = this->xk[ik2][i];

            if (sign == -1) {
                for (i = 0; i < 3; ++i) xk3[i] = xk0[i] - xk1[i] - xk2[i];
            } else {
                for (i = 0; i < 3; ++i) xk3[i] = -xk0[i] - xk1[i] - xk2[i];
            }

            int ik3 = get_knum(xk3);

            kslist.clear();

            if (ik3 > ik2 && use_permutation_symmetry) continue;

            for (int isym = 0; isym < num_group_k; ++isym) {

                ks_in[0] = knum_sym(ik1, symmlist[small_group_of_k[ik][isym]].rotation);
                ks_in[1] = knum_sym(ik2, symmlist[small_group_of_k[ik][isym]].rotation);
                ks_in[2] = knum_sym(ik3, symmlist[small_group_of_k[ik][isym]].rotation);

                if (ks_in[0] == -1 || ks_in[1] == -1 || ks_in[2] == -1) {
                    exit("get_unique_quartet_k",
                         "Cannot find the kpoint after rotation.\n"
                         " This indicates that the input kmesh grid is not compatible with the lattice symmetry.");
                }

                if (!flag_found[ks_in[0]][ks_in[1]]) {
                    kslist.emplace_back(3, &ks_in[0], small_group_of_k[ik][isym]);
                    flag_found[ks_in[0]][ks_in[1]] = true;
                }

                if (use_permutation_symmetry) {
                    std::sort(ks_in.begin(), ks_in.end());
                    do {
                        if (!flag_found[ks_in[0]][ks_in[1]]) {
                            kslist.emplace_back(3, &ks_in[0], small_group_of_k[ik][isym]);
                            flag_found[ks_in[0]][ks_in[1]] = true;
                        }
                    } while (std::next_permutation(ks_in.begin(), ks_in.end()));
                }
            }

            if (!kslist.empty()) {
                quartet.emplace_back(kslist);
            }
        }
    }

    flag_found.clear();
}

void KpointMeshUniform::setup_kpoint_symmetry(const std::vector<SymmetryOperationWithMapping> &symmlist)
{
    double k[3], k_minus[3], Sk[3];
    double S_cart[3][3], S_frac[3][3], S_frac_inv[3][3], S_recip[3][3];

    symop_minus_at_k.resize(nk_irred);
    kpoint_map_symmetry.resize(nk);

    std::vector<int> flag(nk, 0);

    for (auto ik = 0; ik < nk_irred; ++ik) {
        symop_minus_at_k[ik].clear();
        const auto knum = kpoint_irred_all[ik][0].knum;
        for (auto icrd = 0; icrd < 3; ++icrd) {
            k[icrd] = xk[knum][icrd];
            k_minus[icrd] = -k[icrd];
        }

        const auto knum_minus = kindex_minus_xk[knum];

        unsigned int isym = 0;
        for (const auto &it: symmlist) {
            for (auto icrd = 0; icrd < 3; ++icrd) {
                for (auto jcrd = 0; jcrd < 3; ++jcrd) {
                    S_cart[icrd][jcrd] = it.rot[3 * icrd + jcrd];
                    S_frac[icrd][jcrd] = it.rot_real[3 * icrd + jcrd];
                    S_recip[icrd][jcrd] = it.rot_reciprocal[3 * icrd + jcrd];
                }
            }
            invmat3(S_frac_inv, S_frac);
            rotvec(Sk, k, S_recip);

            for (double &x: Sk) x = x - nint(x);
            const auto knum_sym = get_knum(Sk);

            if (knum_sym == -1) {
                exit("setup_kpoint_symmetry", "Cannot find the kpoint");
            }

            if (knum_sym == knum_minus) symop_minus_at_k[ik].emplace_back(isym);

            if (!flag[knum_sym]) {
                kpoint_map_symmetry[knum_sym].symmetry_op = isym;
                kpoint_map_symmetry[knum_sym].knum_irred_orig = ik;
                kpoint_map_symmetry[knum_sym].knum_orig = knum;
                flag[knum_sym] = 1;
            }
            ++isym;
        }
    }

    for (auto ik = 0; ik < nk_irred; ++ik) {

        const auto knum = kpoint_irred_all[ik][0].knum;
        for (auto icrd = 0; icrd < 3; ++icrd) {
            k[icrd] = xk[knum][icrd];
            k_minus[icrd] = -k[icrd];
        }

        const auto knum_minus = get_knum(k_minus);

        if (!flag[knum_minus]) {
            kpoint_map_symmetry[knum_minus].symmetry_op = -1;
            kpoint_map_symmetry[knum_minus].knum_irred_orig = ik;
            kpoint_map_symmetry[knum_minus].knum_orig = knum;
            flag[knum_minus] = 1;
        }
    }
}

int Kpoint::get_kmap_coarse_to_dense(const KpointMeshUniform *kmesh_coarse, const KpointMeshUniform *kmesh_dense,
                                     std::vector<int> &kmap)
{
    int info_mapping = 0;

    kmap.resize(kmesh_coarse->nk);
    double xtmp[3];

    for (auto ik = 0; ik < kmesh_coarse->nk; ++ik) {
        for (auto i = 0; i < 3; ++i) xtmp[i] = kmesh_coarse->xk[ik][i];

        const auto loc = kmesh_dense->get_knum(xtmp);

        if (loc == -1) {
            info_mapping = 1;
        }

        kmap[ik] = loc;
    }
    return info_mapping;
}
