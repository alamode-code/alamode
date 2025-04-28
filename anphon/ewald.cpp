/*
 ewald.cpp

 Copyright (c) 2015 Tatsuro Nishimoto
 Copyright (c) 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "mpi_common.h"
#include "ewald.h"
#include "constants.h"
#include "dielec.h"
#include "dynamical.h"
#include "error.h"
#include "memory.h"
#include "parsephon.h"
#include "system.h"
#include "niggli_wrapper.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <complex>
#include <boost/math/special_functions/erf.hpp>
#include <Eigen/Cholesky>

using namespace PHON_NS;

Ewald::Ewald(PHON *phon) : Pointers(phon)
{
    set_default_variables();
}

Ewald::~Ewald()
{
    deallocate_variables();
}

void Ewald::set_default_variables()
{
    is_longrange = false;
    print_fc2_ewald = false;
    file_longrange = "";
    prec_ewald = 1.0e-15;
    rate_ab = 1.0;
    multiplicity = nullptr;
    Born_charge = nullptr;
    distall_ewald = nullptr;
    force_permutation_sym = true;
}

void Ewald::deallocate_variables()
{
    if (multiplicity) {
        deallocate(multiplicity);
    }
    if (Born_charge) {
        deallocate(Born_charge);
    }
    if (distall_ewald) {
        deallocate(distall_ewald);
    }
}

void Ewald::init()
{
    MPI_Bcast(&is_longrange, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&prec_ewald, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rate_ab, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (is_longrange) {
        int nsize[3] = {1, 1, 1};

        const auto nat_tmp = system->get_supercell(0).number_of_atoms;
        const auto natmin_tmp = system->get_primcell().number_of_atoms;
        allocate(multiplicity, nat_tmp, nat_tmp);
        allocate(Born_charge, natmin_tmp, 3, 3);
        allocate(distall_ewald, nat_tmp, nat_tmp);

        get_pairs_of_minimum_distance(nat_tmp, nsize, system->get_supercell(0).x_fractional);

        for (int i = 0; i < natmin_tmp; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                    Born_charge[i][j][k] = dielec->get_borncharge()[i][j][k];
                }
            }
        }

        prepare_Ewald(dielec->get_dielec_tensor());
        prepare_G();
        compute_ewald_fcs();
    }
}

void Ewald::prepare_Ewald(const Eigen::Matrix3d &dielectric)
{
    int icrd;

    double lavec_norm[3], rlavec_norm[3];
    Eigen::Vector3d e_lavec, e_rlavec;
    double lavec_enorm, rlavec_enorm, lavec_min[2], rlavec_min[2];
    Eigen::Matrix3d epsilon_mat, invepsilon_mat;

    if (mympi->my_rank == 0) {
        std::cout << '\n';
        std::cout << "  Preparing for the Ewald summation ...\n\n";
    }

    const double p = -std::log(prec_ewald);

    for (icrd = 0; icrd < 3; ++icrd) {
        for (int jcrd = 0; jcrd < 3; ++jcrd) {
            epsilon[icrd][jcrd] = dielectric(icrd, jcrd);
            //            epsilon_mat(icrd, jcrd) = epsilon[icrd][jcrd];
        }
    }
    epsilon_mat = dielectric;

    // perform Cholesky decomposition assuming that epsilon is symmetric positive definite.
    // If the off-diagonal elements of epsilon is negligible, this assumption is valid.
    Eigen::LLT<Eigen::Matrix3d> llt(epsilon_mat);
    Eigen::Matrix3d L = llt.matrixL();

    std::cout << "original epsilon matrix:\n";
    std::cout << epsilon_mat << "\n\n";
    std::cout << "L matrix:\n";
    std::cout << L << "\n\n";
    std::cout << "L * L^T:\n";
    std::cout << L * L.transpose() << "\n\n";


    // Calculating convergence parameters
    invmat3(epsilon_inv, epsilon);
    invepsilon_mat = epsilon_mat.inverse();

    const auto lavec_s_tmp = system->get_supercell(0).lattice_vector;
    const auto rlavec_s_tmp = system->get_supercell(0).reciprocal_lattice_vector.transpose();

    Eigen::Matrix3d lavec_s_niggli, c_matrix;

    auto success = niggli_reduction(lavec_s_tmp, lavec_s_niggli, c_matrix);
    std::cout << "success = " << success << "\n";
    // assert(success == 0);

    std::cout << "Original lattice vectors:\n";
    std::cout << lavec_s_tmp << "\n\n";
    std::cout << "Niggli reduced lattice vectors:\n";
    std::cout << lavec_s_niggli << "\n\n";
    std::cout << "Transformation matrix:\n";
    std::cout << c_matrix << "\n\n";


    // For calculating Coulombic (dipole-dipole) FCs
    for (icrd = 0; icrd < 3; ++icrd) {
        lavec_norm[icrd] = lavec_s_tmp.col(icrd).norm();
        rlavec_norm[icrd] = rlavec_s_tmp.col(icrd).norm();

        lavec_enorm = std::sqrt(lavec_s_tmp.col(icrd).dot(invepsilon_mat * lavec_s_tmp.col(icrd)));
        rlavec_enorm = std::sqrt(rlavec_s_tmp.col(icrd).dot(epsilon_mat * rlavec_s_tmp.col(icrd)));

        if (lavec_enorm < lavec_min[0] || icrd == 0) {
            lavec_min[0] = lavec_enorm;
            lavec_min[1] = lavec_norm[icrd];
        }
        if (rlavec_enorm < rlavec_min[0] || icrd == 0) {
            rlavec_min[0] = rlavec_enorm;
            rlavec_min[1] = rlavec_norm[icrd];
        }
    }


    Lmax_sub = std::sqrt(2.0 * p / (lavec_min[0] * rlavec_min[0] * std::pow(rate_ab, 1.0 / 3.0))) * lavec_min[1];
    Gmax_sub = std::sqrt(2.0 * p / (lavec_min[0] * rlavec_min[0]) * std::pow(rate_ab, 1.0 / 3.0)) * rlavec_min[1];
    lambda_sub = std::sqrt(rlavec_min[0] / (2.0 * lavec_min[0]) * std::pow(rate_ab, 1.0 / 3.0));

    for (icrd = 0; icrd < 3; ++icrd) {
        nl_sub[icrd] = static_cast<int>(Lmax_sub / lavec_norm[icrd]) + 1;
        ng_sub[icrd] = static_cast<int>(Gmax_sub / rlavec_norm[icrd]) + 1;
    }
    num_l_sub = (2 * nl_sub[0] + 1) * (2 * nl_sub[1] + 1) * (2 * nl_sub[2] + 1);
    num_g_sub = (2 * ng_sub[0] + 1) * (2 * ng_sub[1] + 1) * (2 * ng_sub[2] + 1);


    // For calculating Coulombic (dipole-dipole) dynamical matrix
    const auto lavec_p_tmp = system->get_primcell().lattice_vector;

    Eigen::Matrix3d lavec_p_niggli, c_matrix_p;

    success = niggli_reduction(lavec_p_tmp, lavec_p_niggli, c_matrix_p);
    std::cout << "success = " << success << "\n";
    // assert(success == 0);

    std::cout << "Original lattice vectors:\n";
    std::cout << lavec_p_tmp << "\n\n";
    std::cout << "Niggli reduced lattice vectors:\n";
    std::cout << lavec_p_niggli << "\n\n";
    std::cout << "Transformation matrix:\n";
    std::cout << c_matrix_p << "\n\n";


    const auto rlavec_p_tmp = system->get_primcell().reciprocal_lattice_vector.transpose();
    for (icrd = 0; icrd < 3; ++icrd) {
        lavec_norm[icrd] = lavec_p_tmp.col(icrd).norm();
        rlavec_norm[icrd] = rlavec_p_tmp.col(icrd).norm();

        lavec_enorm = std::sqrt(lavec_p_tmp.col(icrd).dot(invepsilon_mat * lavec_p_tmp.col(icrd)));
        rlavec_enorm = std::sqrt(rlavec_p_tmp.col(icrd).dot(epsilon_mat * rlavec_p_tmp.col(icrd)));

        if (lavec_enorm < lavec_min[0] || icrd == 0) {
            lavec_min[0] = lavec_enorm;
            lavec_min[1] = lavec_norm[icrd];
        }
        if (rlavec_enorm < rlavec_min[0] || icrd == 0) {
            rlavec_min[0] = rlavec_enorm;
            rlavec_min[1] = rlavec_norm[icrd];
        }
    }

    Lmax = std::sqrt(2.0 * p / (lavec_min[0] * rlavec_min[0] * std::pow(rate_ab, 1.0 / 3.0))) * lavec_min[1];
    Gmax = std::sqrt(2.0 * p / (lavec_min[0] * rlavec_min[0]) * std::pow(rate_ab, 1.0 / 3.0)) * rlavec_min[1];
    lambda = std::sqrt(rlavec_min[0] / (2.0 * lavec_min[0]) * std::pow(rate_ab, 1.0 / 3.0));

    for (icrd = 0; icrd < 3; ++icrd) {
        nl[icrd] = static_cast<int>(Lmax / lavec_norm[icrd]) + 1;
        ng[icrd] = static_cast<int>(Gmax / rlavec_norm[icrd]) + 1;
    }
    num_l = (2 * nl[0] + 1) * (2 * nl[1] + 1) * (2 * nl[2] + 1);
    num_g = (2 * ng[0] + 1) * (2 * ng[1] + 1) * (2 * ng[2] + 1);

    det_epsilon = epsilon_mat.determinant();

    if (mympi->my_rank == 0) {
        std::cout << "  Inverse dielectric tensor : \n";
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                std::cout << std::setw(15) << epsilon_inv[i][j];
            }
            std::cout << '\n';
        }
        std::cout << '\n';

        std::cout << "  Determinant of epsilon: " << std::setw(15) << det_epsilon
                << "\n\n";

        std::cout << "  Parameters for the Ewald summation :\n";
        std::cout << "  - Force constant\n";
        std::cout << "    Lambda : " << std::setw(15) << lambda_sub << '\n';
        std::cout << "    Lmax   : " << std::setw(15) << Lmax_sub << '\n';
        std::cout << "    Gmax   : " << std::setw(15) << Gmax_sub << '\n';
        std::cout << "    Maximum number of real-space cells : "
                << std::setw(3) << nl_sub[0] << "x" << std::setw(3) << nl_sub[1] << "x" << std::setw(3) << nl_sub[2]
                << '\n';
        std::cout << "    Maximum number of reciprocal cells : "
                << std::setw(3) << ng_sub[0] << "x" << std::setw(3) << ng_sub[1] << "x" << std::setw(3) << ng_sub[2]
                << '\n';
        std::cout << '\n';
        std::cout << "  - Dynamical matrix\n";
        std::cout << "    Lambda : " << std::setw(15) << lambda << '\n';
        std::cout << "    Lmax   : " << std::setw(15) << Lmax << '\n';
        std::cout << "    Gmax   : " << std::setw(15) << Gmax << '\n';
        std::cout << "    Maximum number of real-space cells : "
                << std::setw(3) << nl[0] << "x" << std::setw(3) << nl[1] << "x" << std::setw(3) << nl[2] << '\n';
        std::cout << "    Maximum number of reciprocal cells : "
                << std::setw(3) << ng[0] << "x" << std::setw(3) << ng[1] << "x" << std::setw(3) << ng[2] << '\n';
        std::cout << '\n';
    }
}

void Ewald::prepare_G()
{
    // Accumulate reciprocal lattice vectors

    int ix, iy, iz;
    double g_tmp[3], gnorm;

    G_vector_sub.clear();
    G_vector.clear();

    for (ix = -ng_sub[0]; ix <= ng_sub[0]; ++ix) {
        for (iy = -ng_sub[1]; iy <= ng_sub[1]; ++iy) {
            for (iz = -ng_sub[2]; iz <= ng_sub[2]; ++iz) {
                if (ix == 0 && iy == 0 && iz == 0) continue;
                g_tmp[0] = static_cast<double>(ix);
                g_tmp[1] = static_cast<double>(iy);
                g_tmp[2] = static_cast<double>(iz);
                rotvec(g_tmp, g_tmp, system->get_supercell(0).reciprocal_lattice_vector, 'T');
                gnorm = std::sqrt(g_tmp[0] * g_tmp[0] + g_tmp[1] * g_tmp[1] + g_tmp[2] * g_tmp[2]);
                if (gnorm <= Gmax_sub) {
                    G_vector_sub.emplace_back(g_tmp);
                }
            }
        }
    }

    for (ix = -ng[0]; ix <= ng[0]; ++ix) {
        for (iy = -ng[1]; iy <= ng[1]; ++iy) {
            for (iz = -ng[2]; iz <= ng[2]; ++iz) {
                if (ix == 0 && iy == 0 && iz == 0) continue;
                g_tmp[0] = static_cast<double>(ix);
                g_tmp[1] = static_cast<double>(iy);
                g_tmp[2] = static_cast<double>(iz);
                rotvec(g_tmp, g_tmp, system->get_primcell().reciprocal_lattice_vector, 'T');
                gnorm = std::sqrt(g_tmp[0] * g_tmp[0] + g_tmp[1] * g_tmp[1] + g_tmp[2] * g_tmp[2]);
                if (gnorm <= Gmax) {
                    G_vector.emplace_back(g_tmp);
                }
            }
        }
    }
}

void Ewald::get_pairs_of_minimum_distance(const int nat,
                                          const int nsize[3],
                                          const Eigen::MatrixXd &xf) const
{
    // Get pairs and multiplicities

    int icell = 0;
    int iat, jat;
    double dist_tmp;
    double ***xcrd; // fractional coordinate

    const auto ncell = (2 * nsize[0] + 1) * (2 * nsize[1] + 1) * (2 * nsize[2] + 1);

    allocate(xcrd, ncell, nat, 3);

    for (iat = 0; iat < nat; ++iat) {
        for (int icrd = 0; icrd < 3; ++icrd) {
            xcrd[0][iat][icrd] = xf(iat, icrd);
        }
        rotvec(xcrd[0][iat], xcrd[0][iat], system->get_supercell(0).lattice_vector);
    }

    for (int isize = -nsize[0]; isize <= nsize[0]; ++isize) {
        for (int jsize = -nsize[1]; jsize <= nsize[1]; ++jsize) {
            for (int ksize = -nsize[2]; ksize <= nsize[2]; ++ksize) {
                if (isize == 0 && jsize == 0 && ksize == 0) continue;
                ++icell;
                for (iat = 0; iat < nat; ++iat) {
                    xcrd[icell][iat][0] = xf(iat, 0) + static_cast<double>(isize);
                    xcrd[icell][iat][1] = xf(iat, 1) + static_cast<double>(jsize);
                    xcrd[icell][iat][2] = xf(iat, 2) + static_cast<double>(ksize);
                    rotvec(xcrd[icell][iat], xcrd[icell][iat], system->get_supercell(0).lattice_vector);
                }
            }
        }
    }

    for (iat = 0; iat < nat; ++iat) {
        for (jat = 0; jat < nat; ++jat) {
            for (icell = 0; icell < ncell; ++icell) {
                dist_tmp = std::sqrt(std::pow(xcrd[0][iat][0] - xcrd[icell][jat][0], 2.0)
                                     + std::pow(xcrd[0][iat][1] - xcrd[icell][jat][1], 2.0)
                                     + std::pow(xcrd[0][iat][2] - xcrd[icell][jat][2], 2.0));

                distall_ewald[iat][jat].emplace_back(icell, dist_tmp);
            }
            std::sort(distall_ewald[iat][jat].begin(), distall_ewald[iat][jat].end());
        }
    }
    double dist_hold = -1.0;
    for (iat = 0; iat < nat; ++iat) {
        for (jat = 0; jat < nat; ++jat) {
            multiplicity[iat][jat] = 0;

            for (auto it = distall_ewald[iat][jat].begin();
                 it != distall_ewald[iat][jat].end(); ++it) {
                if (it == distall_ewald[iat][jat].begin()) dist_hold = (*it).dist;
                dist_tmp = (*it).dist;
                if (std::abs(dist_tmp - dist_hold) < 1.0e-3) multiplicity[iat][jat] += 1;
            }
        }
    }
    deallocate(xcrd);
}

void Ewald::compute_ewald_fcs()
{
    int j;
    int iat, jat;
    int icrd, jcrd;
    int atm_s;
    int nat = system->get_supercell(0).number_of_atoms;
    int natmin = system->get_primcell().number_of_atoms;
    double **fc_ewald_short, **fc_ewald_long;
    std::string file_fcs_ewald = input->job_title + ".fc2_ewald";

    if (mympi->my_rank == 0) {
        std::cout << " Calculating long-range (dipole-dipole) FCs in the supercell ...";
    }

    const auto map_p2s = system->get_map_p2s(0);
    const auto map_s2p = system->get_map_s2p(0);

    Eigen::MatrixXd fcs_ewald(3 * natmin, 3 * nat);
    Eigen::MatrixXd fcs_total(3 * natmin, 3 * nat);
    Eigen::MatrixXd fcs_other(3 * natmin, 3 * nat);

    allocate(fc_ewald_short, 3, 3);
    allocate(fc_ewald_long, 3, 3);

    for (iat = 0; iat < natmin; ++iat) {
        atm_s = map_p2s[iat][0];
        for (jat = 0; jat < nat; ++jat) {
            calc_short_term_ewald_fcs(atm_s, jat, fc_ewald_short);
            calc_long_term_ewald_fcs(atm_s, jat, fc_ewald_long);

            for (icrd = 0; icrd < 3; ++icrd) {
                for (jcrd = 0; jcrd < 3; ++jcrd) {
                    fcs_ewald(3 * iat + icrd, 3 * jat + jcrd)
                            = fc_ewald_short[icrd][jcrd] + fc_ewald_long[icrd][jcrd];
                }
            }
        }
    }

    deallocate(fc_ewald_short);
    deallocate(fc_ewald_long);

    fcs_total.setZero();

    for (const auto &it: fcs_phonon->force_constant_with_cell[0]) {
        fcs_total(it.pairs[0].index, 3 * it.atoms_s[1] + it.coords[1]) += it.fcs_val;
    }
    fcs_other = fcs_total - fcs_ewald;

    std::vector<Eigen::Vector3d> relvecs, relvecs_vel;
    Eigen::Vector3d relvec_tmp, relvec_tmp2;
    std::vector<AtomCellSuper> pairs_tmp(2);
    std::vector<unsigned int> atom_super(2);

    const auto cell_tmp = system->get_supercell(0);
    const auto xf_image = dynamical->get_xrs_image();

    for (iat = 0; iat < natmin; ++iat) {
        atm_s = map_p2s[iat][0];

        for (jat = 0; jat < nat; ++jat) {
            int nmulti = multiplicity[atm_s][jat];

            pairs_tmp[0].tran = 0;
            pairs_tmp[1].tran = map_s2p[jat].tran_num;
            pairs_tmp[0].cell_s = 0;

            atom_super[0] = atm_s;
            atom_super[1] = jat;

            for (int icell = 0; icell < nmulti; ++icell) {
                pairs_tmp[1].cell_s = distall_ewald[atm_s][jat][icell].cell;

                for (j = 0; j < 3; ++j) {
                    relvec_tmp2[j] = cell_tmp.x_fractional(jat, j)
                                     + xf_image[pairs_tmp[1].cell_s][j]
                                     - cell_tmp.x_fractional(atm_s, j);

                    relvec_tmp[j] = cell_tmp.x_fractional(jat, j)
                                    + xf_image[pairs_tmp[1].cell_s][j]
                                    - cell_tmp.x_fractional(map_p2s[map_s2p[jat].atom_num][0], j);
                }

                relvec_tmp = system->get_primcell().lattice_vector.inverse() * cell_tmp.lattice_vector * relvec_tmp;
                relvec_tmp2 = system->get_primcell().lattice_vector.inverse() * cell_tmp.lattice_vector * relvec_tmp2;

                relvecs.clear();
                relvecs_vel.clear();

                relvecs.emplace_back(relvec_tmp);
                relvecs_vel.emplace_back(relvec_tmp2);

                for (icrd = 0; icrd < 3; ++icrd) {
                    for (jcrd = 0; jcrd < 3; ++jcrd) {
                        const auto fcs_val = fcs_other(3 * iat + icrd, 3 * jat + jcrd) / static_cast<double>(nmulti);

                        pairs_tmp[0].index = 3 * iat + icrd;
                        pairs_tmp[1].index = 3 * map_s2p[jat].atom_num + jcrd;

                        fc2_without_dipole.emplace_back(fcs_val,
                                                        pairs_tmp,
                                                        atom_super,
                                                        relvecs,
                                                        relvecs_vel);
                    }
                }
            }
        }
    }

    if (mympi->my_rank == 0) {
        if (print_fc2_ewald) {
            std::ofstream ofs_fcs_ewald;

            ofs_fcs_ewald.open(file_fcs_ewald.c_str(), std::ios::out);
            if (!ofs_fcs_ewald) exit("compute_ewald_fcs", "cannot open file PREFIX.fcs_ewald");

            ofs_fcs_ewald << "# Harmonic force constants\n";
            ofs_fcs_ewald << "# atom1, xyz1, atom2, xyz2, fc2 original, fc2 dipole-dipole, fc2_orig - fc2_dipole\n";

            for (iat = 0; iat < natmin; ++iat) {
                for (icrd = 0; icrd < 3; ++icrd) {
                    for (jat = 0; jat < nat; ++jat) {
                        for (jcrd = 0; jcrd < 3; ++jcrd) {
                            ofs_fcs_ewald << std::setw(5) << iat + 1;
                            ofs_fcs_ewald << std::setw(5) << icrd + 1;
                            ofs_fcs_ewald << std::setw(5) << jat + 1;
                            ofs_fcs_ewald << std::setw(5) << jcrd + 1;
                            ofs_fcs_ewald << std::setw(15) << fcs_total(3 * iat + icrd, 3 * jat + jcrd);
                            ofs_fcs_ewald << std::setw(15) << fcs_ewald(3 * iat + icrd, 3 * jat + jcrd);
                            ofs_fcs_ewald << std::setw(15) << fcs_other(3 * iat + icrd, 3 * jat + jcrd);
                            ofs_fcs_ewald << '\n';
                        }
                    }
                }
            }
            ofs_fcs_ewald.close();
        }
    }

    if (mympi->my_rank == 0) {
        std::cout << " done.\n";
        if (print_fc2_ewald) {
            std::cout << '\n';
            std::cout << " FC2_EWALD = 1: Dipole-dipole and short-ranged components of harmonic \n";
            std::cout << "                force constants are printed in " << file_fcs_ewald << '\n';
        }
    }
}


void Ewald::calc_short_term_ewald_fcs(const int iat,
                                      const int jat,
                                      double **fc_l_out)
{
    // Real lattice sum part for FCs
    // iat : atom index in the supercell (should be in the center primitive cell)
    // jat : atom index in the supercell

    int icrd, jcrd;
    int icell, jcell, kcell;
    int kat;
    double xnorm;
    Eigen::Vector3d x_tmp, trans;
    std::vector<std::vector<double> > func_L(3, std::vector<double>(3, 0.0));

    for (icrd = 0; icrd < 3; ++icrd) {
        for (jcrd = 0; jcrd < 3; ++jcrd) {
            fc_l_out[icrd][jcrd] = 0.0;
        }
    }

    const auto x_frac_super = system->get_supercell(0).x_fractional.transpose();
    const auto lavec = system->get_supercell(0).lattice_vector;

    if (iat == jat) {
        // k = k', where k labels atoms in the supercell

        for (icell = -nl_sub[0]; icell <= nl_sub[0]; ++icell) {
            for (jcell = -nl_sub[1]; jcell <= nl_sub[1]; ++jcell) {
                for (kcell = -nl_sub[2]; kcell <= nl_sub[2]; ++kcell) {
                    if (icell == 0 && jcell == 0 && kcell == 0) {
                        // l'' = l = 0

                        for (kat = 0; kat < system->get_supercell(0).number_of_atoms; ++kat) {
                            if (kat == iat) continue; // k'' = k

                            x_tmp = lavec * (x_frac_super.col(iat) - x_frac_super.col(kat));
                            xnorm = x_tmp.norm();

                            if (xnorm < Lmax_sub) {
                                calc_realspace_sum(iat, kat, x_tmp.data(), lambda_sub, func_L);

                                if (force_permutation_sym) {
                                    for (icrd = 0; icrd < 3; ++icrd) {
                                        for (jcrd = 0; jcrd < 3; ++jcrd) {
                                            fc_l_out[icrd][jcrd] += 0.5 * (func_L[icrd][jcrd]
                                                                           + func_L[jcrd][icrd]);
                                        }
                                    }
                                } else {
                                    for (icrd = 0; icrd < 3; ++icrd) {
                                        for (jcrd = 0; jcrd < 3; ++jcrd) {
                                            fc_l_out[icrd][jcrd] += func_L[icrd][jcrd];
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        // l'' != 0

                        trans[0] = static_cast<double>(icell);
                        trans[1] = static_cast<double>(jcell);
                        trans[2] = static_cast<double>(kcell);

                        for (kat = 0; kat < system->get_supercell(0).number_of_atoms; ++kat) {
                            x_tmp = lavec * (x_frac_super.col(iat) - x_frac_super.col(kat) - trans);
                            xnorm = x_tmp.norm();

                            if (xnorm < Lmax_sub) {
                                calc_realspace_sum(iat, kat, x_tmp.data(), lambda_sub, func_L);

                                if (force_permutation_sym) {
                                    for (icrd = 0; icrd < 3; ++icrd) {
                                        for (jcrd = 0; jcrd < 3; ++jcrd) {
                                            fc_l_out[icrd][jcrd] += 0.5 * (func_L[icrd][jcrd]
                                                                           + func_L[jcrd][icrd]);
                                        }
                                    }
                                } else {
                                    for (icrd = 0; icrd < 3; ++icrd) {
                                        for (jcrd = 0; jcrd < 3; ++jcrd) {
                                            fc_l_out[icrd][jcrd] += func_L[icrd][jcrd];
                                        }
                                    }
                                }
                            }
                        }
                        x_tmp = lavec * (x_frac_super.col(iat) - x_frac_super.col(jat) - trans);
                        xnorm = x_tmp.norm();

                        if (xnorm < Lmax_sub) {
                            calc_realspace_sum(iat, jat, x_tmp.data(), lambda_sub, func_L);

                            for (icrd = 0; icrd < 3; ++icrd) {
                                for (jcrd = 0; jcrd < 3; ++jcrd) {
                                    fc_l_out[icrd][jcrd] -= func_L[icrd][jcrd];
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        // case of i != j (k != k')

        for (icell = -nl_sub[0]; icell <= nl_sub[0]; ++icell) {
            for (jcell = -nl_sub[1]; jcell <= nl_sub[1]; ++jcell) {
                for (kcell = -nl_sub[2]; kcell <= nl_sub[2]; ++kcell) {
                    trans[0] = static_cast<double>(icell);
                    trans[1] = static_cast<double>(jcell);
                    trans[2] = static_cast<double>(kcell);

                    x_tmp = lavec * (x_frac_super.col(iat) - x_frac_super.col(jat) - trans);
                    xnorm = x_tmp.norm();

                    if (xnorm < Lmax_sub) {
                        calc_realspace_sum(iat, jat, x_tmp.data(), lambda_sub, func_L);

                        for (icrd = 0; icrd < 3; ++icrd) {
                            for (jcrd = 0; jcrd < 3; ++jcrd) {
                                fc_l_out[icrd][jcrd] -= func_L[icrd][jcrd];
                            }
                        }
                    }
                }
            }
        }
    }
}

void Ewald::calc_long_term_ewald_fcs(const int iat,
                                     const int jat,
                                     double **fc_g_out)
{
    // Reciprocal lattice sum part for FCs

    int i;
    int icrd, jcrd;
    int acrd, bcrd;
    double gnorm2;
    double x_tmp[3], g_tmp[3], epsilon_gvector[3];
    double common_tmp;

    for (icrd = 0; icrd < 3; ++icrd) {
        for (jcrd = 0; jcrd < 3; ++jcrd) {
            fc_g_out[icrd][jcrd] = 0.0;
        }
    }
    double volume = system->get_supercell(0).volume;

    int ikd = system->get_map_s2p(0)[iat].atom_num;
    int jkd = system->get_map_s2p(0)[jat].atom_num;

    double factor = 4.0 * pi / volume;

    if (iat == jat) {
        for (const auto &it: G_vector_sub) {
            for (i = 0; i < 3; ++i) g_tmp[i] = it.vec[i];
            rotvec(epsilon_gvector, g_tmp, epsilon);
            gnorm2 = g_tmp[0] * epsilon_gvector[0]
                     + g_tmp[1] * epsilon_gvector[1]
                     + g_tmp[2] * epsilon_gvector[2];

            for (int kat = 0; kat < system->get_supercell(0).number_of_atoms; ++kat) {
                int kkd = system->get_map_s2p(0)[kat].atom_num;

                for (i = 0; i < 3; ++i) {
                    x_tmp[i] = system->get_supercell(0).x_fractional(iat, i)
                               - system->get_supercell(0).x_fractional(kat, i);
                }
                rotvec(x_tmp, x_tmp, system->get_supercell(0).lattice_vector);

                common_tmp = factor * std::exp(-0.25 * gnorm2 / std::pow(lambda_sub, 2.0)) / gnorm2
                             * std::cos(g_tmp[0] * x_tmp[0]
                                        + g_tmp[1] * x_tmp[1]
                                        + g_tmp[2] * x_tmp[2]);

                for (icrd = 0; icrd < 3; ++icrd) {
                    for (jcrd = 0; jcrd < 3; ++jcrd) {
                        for (acrd = 0; acrd < 3; ++acrd) {
                            for (bcrd = 0; bcrd < 3; ++bcrd) {
                                if (force_permutation_sym) {
                                    fc_g_out[icrd][jcrd] -= g_tmp[acrd] * g_tmp[bcrd] * common_tmp
                                            *
                                            (Born_charge[ikd][acrd][icrd] * Born_charge[kkd][bcrd][jcrd]
                                             +
                                             Born_charge[jkd][acrd][jcrd] *
                                             Born_charge[kkd][bcrd][icrd]);
                                } else {
                                    fc_g_out[icrd][jcrd] -= g_tmp[acrd] * g_tmp[bcrd] * common_tmp * 2.0
                                            * (Born_charge[ikd][acrd][icrd] *
                                               Born_charge[kkd][bcrd][jcrd]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (const auto &it: G_vector_sub) {
        for (i = 0; i < 3; ++i) g_tmp[i] = it.vec[i];
        rotvec(epsilon_gvector, g_tmp, epsilon);
        gnorm2 = g_tmp[0] * epsilon_gvector[0]
                 + g_tmp[1] * epsilon_gvector[1]
                 + g_tmp[2] * epsilon_gvector[2];

        for (i = 0; i < 3; ++i) {
            x_tmp[i] = system->get_supercell(0).x_fractional(iat, i)
                       - system->get_supercell(0).x_fractional(jat, i);
        }
        rotvec(x_tmp, x_tmp, system->get_supercell(0).lattice_vector);

        common_tmp = 2.0 * factor * std::exp(-0.25 * gnorm2 / std::pow(lambda_sub, 2.0)) / gnorm2
                     * std::cos(g_tmp[0] * x_tmp[0] + g_tmp[1] * x_tmp[1] + g_tmp[2] * x_tmp[2]);

        for (icrd = 0; icrd < 3; ++icrd) {
            for (jcrd = 0; jcrd < 3; ++jcrd) {
                for (acrd = 0; acrd < 3; ++acrd) {
                    for (bcrd = 0; bcrd < 3; ++bcrd) {
                        fc_g_out[icrd][jcrd] += g_tmp[acrd] * g_tmp[bcrd] * common_tmp
                                * Born_charge[ikd][acrd][icrd] * Born_charge[jkd][bcrd][jcrd];
                    }
                }
            }
        }
    }
}

void Ewald::add_longrange_matrix(const double *xk_in,
                                 const double *kvec_in,
                                 std::complex<double> **dymat_k_out)
{
    int icrd, jcrd, iat, jat;
    const int natmin = system->get_primcell().number_of_atoms;
    const long natmin2 = natmin * natmin;
    int neval = 3 * system->get_primcell().number_of_atoms;
    double xk[3];

    // Move input xk back to the -0.5 <= xk < 0.5 range to avoid zero division.
    // Also, this is necessary to make the phonon dispersion periodic in the reciprocal lattice.
    for (auto i = 0; i < 3; ++i) {
        xk[i] = xk_in[i] - static_cast<double>(nint(xk_in[i]));
    }

    rotvec(xk, xk, system->get_primcell().reciprocal_lattice_vector, 'T');

    for (int i = 0; i < neval; ++i) {
        for (int j = 0; j < neval; ++j) {
            dymat_k_out[i][j] = std::complex<double>(0.0, 0.0);
        }
    }

#pragma omp parallel
    {
        std::complex<double> **dymat_tmp_l, **dymat_tmp_g;

        allocate(dymat_tmp_l, 3, 3);
        allocate(dymat_tmp_g, 3, 3);

#pragma omp for private(iat, jat, icrd, jcrd)
        for (long i = 0; i < natmin2; ++i) {
            iat = i / natmin;
            jat = i % natmin;
            calc_short_term_dynamical_matrix(iat, jat, xk, dymat_tmp_l);
            calc_long_term_dynamical_matrix(iat, jat, xk, kvec_in, dymat_tmp_g);
            for (icrd = 0; icrd < 3; ++icrd) {
                for (jcrd = 0; jcrd < 3; ++jcrd) {
                    dymat_k_out[3 * iat + icrd][3 * jat + jcrd] = dymat_tmp_l[icrd][jcrd]
                                                                  + dymat_tmp_g[icrd][jcrd];
                }
            }
        }

        deallocate(dymat_tmp_l);
        deallocate(dymat_tmp_g);
    }
}

void Ewald::calc_short_term_dynamical_matrix(const int iat,
                                             const int jat,
                                             double *xk_in,
                                             std::complex<double> **mat_out)
{
    // Real lattice sum part for a dynamical matrix
    // iat : atom index in the primitive cell
    // jat : atom index in the primitive cell

    int i;
    int icrd, jcrd, kat;
    int atm_s3;
    int icell, jcell, kcell;
    double xnorm, phase;
    double x_tmp[3], trans[3];
    std::vector<std::vector<double> > func_L(3, std::vector<double>(3, 0.0));

    // Substitute quantities into variables
    int atm_s1 = system->get_map_p2s(0)[iat][0];
    int atm_s2 = system->get_map_p2s(0)[jat][0];

    for (i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            mat_out[i][j] = std::complex<double>(0.0, 0.0);
        }
    }

    if (iat == jat) {
        // k = k'

        for (icell = -nl[0]; icell <= nl[0]; ++icell) {
            for (jcell = -nl[1]; jcell <= nl[1]; ++jcell) {
                for (kcell = -nl[2]; kcell <= nl[2]; ++kcell) {
                    trans[0] = static_cast<double>(icell);
                    trans[1] = static_cast<double>(jcell);
                    trans[2] = static_cast<double>(kcell);
                    rotvec(trans, trans, system->get_primcell().lattice_vector);

                    // Lattice vector = 0
                    if (icell == 0 && jcell == 0 && kcell == 0) {
                        for (kat = 0; kat < system->get_primcell().number_of_atoms; ++kat) {
                            if (kat == iat) continue;

                            atm_s3 = system->get_map_p2s(0)[kat][0];
                            for (i = 0; i < 3; ++i) {
                                x_tmp[i] = system->get_supercell(0).x_fractional(atm_s1, i)
                                           - system->get_supercell(0).x_fractional(atm_s3, i);
                            }
                            rotvec(x_tmp, x_tmp, system->get_supercell(0).lattice_vector);
                            xnorm = std::sqrt(x_tmp[0] * x_tmp[0]
                                              + x_tmp[1] * x_tmp[1]
                                              + x_tmp[2] * x_tmp[2]);

                            if (xnorm < Lmax) {
                                calc_realspace_sum(atm_s1, atm_s3, x_tmp, lambda, func_L);

                                if (force_permutation_sym) {
                                    for (icrd = 0; icrd < 3; ++icrd) {
                                        for (jcrd = 0; jcrd < 3; ++jcrd) {
                                            mat_out[icrd][jcrd] += 0.5 * (func_L[icrd][jcrd]
                                                                          + func_L[jcrd][icrd]);
                                        }
                                    }
                                } else {
                                    for (icrd = 0; icrd < 3; ++icrd) {
                                        for (jcrd = 0; jcrd < 3; ++jcrd) {
                                            mat_out[icrd][jcrd] += func_L[icrd][jcrd];
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        for (kat = 0; kat < system->get_primcell().number_of_atoms; ++kat) {
                            atm_s3 = system->get_map_p2s(0)[kat][0];
                            for (i = 0; i < 3; ++i) {
                                x_tmp[i] = system->get_supercell(0).x_fractional(atm_s1, i)
                                           - system->get_supercell(0).x_fractional(atm_s3, i);
                            }
                            rotvec(x_tmp, x_tmp, system->get_supercell(0).lattice_vector);
                            for (i = 0; i < 3; ++i) {
                                x_tmp[i] -= trans[i];
                            }
                            xnorm = std::sqrt(x_tmp[0] * x_tmp[0]
                                              + x_tmp[1] * x_tmp[1]
                                              + x_tmp[2] * x_tmp[2]);

                            if (xnorm < Lmax) {
                                calc_realspace_sum(atm_s1, atm_s3, x_tmp, lambda, func_L);

                                if (force_permutation_sym) {
                                    for (icrd = 0; icrd < 3; ++icrd) {
                                        for (jcrd = 0; jcrd < 3; ++jcrd) {
                                            mat_out[icrd][jcrd] += 0.5 * (func_L[icrd][jcrd]
                                                                          + func_L[jcrd][icrd]);
                                        }
                                    }
                                } else {
                                    for (icrd = 0; icrd < 3; ++icrd) {
                                        for (jcrd = 0; jcrd < 3; ++jcrd) {
                                            mat_out[icrd][jcrd] += func_L[icrd][jcrd];
                                        }
                                    }
                                }
                            }
                        }

                        for (i = 0; i < 3; ++i) {
                            x_tmp[i] = system->get_supercell(0).x_fractional(atm_s1, i)
                                       - system->get_supercell(0).x_fractional(atm_s2, i);
                        }
                        rotvec(x_tmp, x_tmp, system->get_supercell(0).lattice_vector);
                        for (i = 0; i < 3; ++i) {
                            x_tmp[i] -= trans[i];
                        }
                        xnorm = std::sqrt(x_tmp[0] * x_tmp[0]
                                          + x_tmp[1] * x_tmp[1]
                                          + x_tmp[2] * x_tmp[2]);

                        if (xnorm < Lmax) {
                            calc_realspace_sum(atm_s1, atm_s2, x_tmp, lambda, func_L);
                            phase = xk_in[0] * trans[0] + xk_in[1] * trans[1] + xk_in[2] * trans[2];

                            for (icrd = 0; icrd < 3; ++icrd) {
                                for (jcrd = 0; jcrd < 3; ++jcrd) {
                                    mat_out[icrd][jcrd] -= func_L[icrd][jcrd] * std::exp(im * phase);
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        for (icell = -nl[0]; icell <= nl[0]; ++icell) {
            for (jcell = -nl[1]; jcell <= nl[1]; ++jcell) {
                for (kcell = -nl[2]; kcell <= nl[2]; ++kcell) {
                    trans[0] = static_cast<double>(icell);
                    trans[1] = static_cast<double>(jcell);
                    trans[2] = static_cast<double>(kcell);
                    rotvec(trans, trans, system->get_primcell().lattice_vector);

                    for (i = 0; i < 3; ++i) {
                        x_tmp[i] = system->get_supercell(0).x_fractional(atm_s1, i)
                                   - system->get_supercell(0).x_fractional(atm_s2, i);
                    }
                    rotvec(x_tmp, x_tmp, system->get_supercell(0).lattice_vector);
                    for (i = 0; i < 3; ++i) {
                        x_tmp[i] -= trans[i];
                    }
                    xnorm = std::sqrt(x_tmp[0] * x_tmp[0]
                                      + x_tmp[1] * x_tmp[1]
                                      + x_tmp[2] * x_tmp[2]);

                    if (xnorm < Lmax) {
                        calc_realspace_sum(atm_s1, atm_s2, x_tmp, lambda, func_L);
                        phase = xk_in[0] * trans[0] + xk_in[1] * trans[1] + xk_in[2] * trans[2];

                        for (icrd = 0; icrd < 3; ++icrd) {
                            for (jcrd = 0; jcrd < 3; ++jcrd) {
                                mat_out[icrd][jcrd] -= func_L[icrd][jcrd] * std::exp(im * phase);
                            }
                        }
                    }
                }
            }
        }
    }

    const auto mi = system->get_mass_super()[atm_s1];
    const auto mj = system->get_mass_super()[atm_s2];
    for (icrd = 0; icrd < 3; ++icrd) {
        for (jcrd = 0; jcrd < 3; ++jcrd) {
            mat_out[icrd][jcrd] /= std::sqrt(mi * mj);
        }
    }
}

void Ewald::calc_long_term_dynamical_matrix(const int iat,
                                            const int jat,
                                            const double *xk_in,
                                            const double *kvec_in,
                                            std::complex<double> **mat_out)
{
    // Reciprocal lattice sum part for a dynamical matrix

    int i, j;
    int icrd, jcrd, acrd, bcrd;
    double vec[3], e_kvec[3];
    double tmp;

    int atm_s1 = system->get_map_p2s(0)[iat][0];
    int atm_s2 = system->get_map_p2s(0)[jat][0];
    double mi = system->get_mass_super()[atm_s1];
    double mj = system->get_mass_super()[atm_s2];
    double vol_p = system->get_primcell().volume;
    for (i = 0; i < 3; ++i) {
        vec[i] = system->get_supercell(0).x_fractional(atm_s1, i)
                 - system->get_supercell(0).x_fractional(atm_s2, i);
    }
    rotvec(vec, vec, system->get_supercell(0).lattice_vector);
    double phase = xk_in[0] * vec[0] + xk_in[1] * vec[1] + xk_in[2] * vec[2];
    rotvec(e_kvec, xk_in, epsilon);

    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            mat_out[i][j] = std::complex<double>(0.0, 0.0);
        }
    }
    double kd = xk_in[0] * e_kvec[0] + xk_in[1] * e_kvec[1] + xk_in[2] * e_kvec[2];

    std::complex<double> exp_phase = std::exp(im * phase);

    if (std::sqrt(kd) > eps10) {
        // constant analytic term

        for (icrd = 0; icrd < 3; ++icrd) {
            for (jcrd = 0; jcrd < 3; ++jcrd) {
                tmp = 0.0;

                for (acrd = 0; acrd < 3; ++acrd) {
                    for (bcrd = 0; bcrd < 3; ++bcrd) {
                        tmp += xk_in[acrd] * xk_in[bcrd]
                                * Born_charge[iat][acrd][icrd] * Born_charge[jat][bcrd][jcrd];
                    }
                }
                mat_out[icrd][jcrd] += 2.0 * tmp / kd * exp_phase
                        * std::exp(-0.25 * kd / std::pow(lambda, 2.0));
            }
        }
    } else {
        // Treat non-analytic term

        double kdirec[3], e_kdirec[3];
        for (i = 0; i < 3; ++i) kdirec[i] = kvec_in[i];
        rotvec(e_kdirec, kdirec, epsilon);
        double norm = kdirec[0] * e_kdirec[0] + kdirec[1] * e_kdirec[1] + kdirec[2] * e_kdirec[2];

        if (norm > eps) {
            for (icrd = 0; icrd < 3; ++icrd) {
                for (jcrd = 0; jcrd < 3; ++jcrd) {
                    tmp = 0.0;

                    for (acrd = 0; acrd < 3; ++acrd) {
                        for (bcrd = 0; bcrd < 3; ++bcrd) {
                            tmp += kdirec[acrd] * kdirec[bcrd]
                                    * Born_charge[iat][acrd][icrd] * Born_charge[jat][bcrd][jcrd];
                        }
                    }
                    mat_out[icrd][jcrd] += 2.0 * tmp / norm * exp_phase;
                }
            }
        }
    }

    double g[3], gk[3], vecl[3], g_tmp[3], gk_tmp[3];
    double common;

    for (auto &it: G_vector) {
        for (int l = 0; l < 3; ++l) {
            g[l] = it.vec[l];
            gk[l] = g[l] + xk_in[l];
        }

        if (iat == jat) {
            rotvec(g_tmp, g, epsilon);
            double gd = g[0] * g_tmp[0] + g[1] * g_tmp[1] + g[2] * g_tmp[2];
            common = std::exp(-0.25 * gd / std::pow(lambda, 2.0)) / gd;

            for (int kat = 0; kat < system->get_primcell().number_of_atoms; ++kat) {
                int atm_s3 = system->get_map_p2s(0)[kat][0];

                for (i = 0; i < 3; ++i) {
                    vecl[i] = system->get_supercell(0).x_fractional(atm_s1, i)
                              - system->get_supercell(0).x_fractional(atm_s3, i);
                }
                rotvec(vecl, vecl, system->get_supercell(0).lattice_vector);
                double phase_g1 = g[0] * vecl[0] + g[1] * vecl[1] + g[2] * vecl[2];
                exp_phase = std::exp(im * phase_g1);

                for (icrd = 0; icrd < 3; ++icrd) {
                    for (jcrd = 0; jcrd < 3; ++jcrd) {
                        tmp = 0.0;

                        for (acrd = 0; acrd < 3; ++acrd) {
                            for (bcrd = 0; bcrd < 3; ++bcrd) {
                                if (force_permutation_sym) {
                                    tmp += g[acrd] * g[bcrd]
                                            * (Born_charge[iat][acrd][icrd] * Born_charge[kat][bcrd][jcrd]
                                               + Born_charge[jat][acrd][jcrd] * Born_charge[kat][bcrd][icrd]);
                                } else {
                                    tmp += g[acrd] * g[bcrd] * 2.0
                                            * (Born_charge[iat][acrd][icrd] * Born_charge[kat][bcrd][jcrd]);
                                }
                            }
                        }
                        mat_out[icrd][jcrd] -= tmp * common * exp_phase;
                    }
                }
            }
        }

        rotvec(gk_tmp, gk, epsilon);
        double gkd = gk[0] * gk_tmp[0] + gk[1] * gk_tmp[1] + gk[2] * gk_tmp[2];
        double phase_g2 = gk[0] * vec[0] + gk[1] * vec[1] + gk[2] * vec[2];

        common = 2.0 * std::exp(-0.25 * gkd / std::pow(lambda, 2.0)) / gkd;
        exp_phase = std::exp(im * phase_g2);

        for (icrd = 0; icrd < 3; ++icrd) {
            for (jcrd = 0; jcrd < 3; ++jcrd) {
                tmp = 0.0;

                for (acrd = 0; acrd < 3; ++acrd) {
                    for (bcrd = 0; bcrd < 3; ++bcrd) {
                        tmp += gk[acrd] * gk[bcrd]
                                * Born_charge[iat][acrd][icrd] * Born_charge[jat][bcrd][jcrd];
                    }
                }
                mat_out[icrd][jcrd] += tmp * common * exp_phase;
            }
        }
    }

    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            mat_out[i][j] *= 4.0 * pi / (vol_p * std::sqrt(mi * mj));
        }
    }
}

void Ewald::calc_realspace_sum(const int iat,
                               const int jat,
                               const double xdist[3],
                               const double lambda_in,
                               std::vector<std::vector<double> > &ret)
{
    // iat : atom index in the supercell
    // jat : atom index in the supercell
    // xdist: distance between the two atomic sites
    unsigned int icrd, jcrd;
    const double lambda3 = std::pow(lambda_in, 3.0);
    Eigen::Matrix3d hmat_tmp;

    calc_anisotropic_hmat(lambda_in, xdist, hmat_tmp);

    const auto ikd = system->get_map_s2p(0)[iat].atom_num;
    const auto jkd = system->get_map_s2p(0)[jat].atom_num;

    for (icrd = 0; icrd < 3; ++icrd) {
        for (jcrd = 0; jcrd < 3; ++jcrd) {
            ret[icrd][jcrd] = 0.0;
        }
    }

    for (icrd = 0; icrd < 3; ++icrd) {
        for (jcrd = 0; jcrd < 3; ++jcrd) {
            double tmp = 0.0;
            for (unsigned int acrd = 0; acrd < 3; ++acrd) {
                for (unsigned int bcrd = 0; bcrd < 3; ++bcrd) {
                    tmp += hmat_tmp(acrd, bcrd)
                            * Born_charge[ikd][acrd][icrd]
                            * Born_charge[jkd][bcrd][jcrd];
                }
            }
            ret[icrd][jcrd] = 2.0 * tmp * lambda3; // factor 2 due to e^2 = 2 in the Rydberg unit
        }
    }
}

void Ewald::calc_anisotropic_hmat(const double lambda_in,
                                  const double *x,
                                  Eigen::Matrix3d &hmat_out) const
{
    // Compute H_ab(0\kappa;\ell'\kappa')
    double common_tmp[2];
    double x_tmp[3], y_tmp[3];

    hmat_out.setZero();

    for (int i = 0; i < 3; ++i) {
        y_tmp[i] = x[i] * lambda_in;
    }

    rotvec(x_tmp, y_tmp, epsilon_inv);

    double const yd = std::sqrt(x_tmp[0] * y_tmp[0] + x_tmp[1] * y_tmp[1] + x_tmp[2] * y_tmp[2]);
    if (yd == 0.0) {
        exit("ewald->calc_anisotropic_hmat", "components of hmat diverge.");
    }
    double const yd_inv = 1.0 / yd;
    double const yd2 = std::pow(yd, 2.0);
    double const yd2_inv = yd_inv * yd_inv;
    const double erfc_y = boost::math::erfc(yd);
    const double exp_y2 = std::exp(-yd2);
    const double two_over_sqrtpi = 2.0 / std::sqrt(pi);

    common_tmp[0] = (3.0 * yd_inv * yd2_inv * erfc_y + two_over_sqrtpi * (3.0 * yd2_inv + 2.0) * exp_y2)
                    * yd2_inv / std::sqrt(det_epsilon);
    common_tmp[1] = (yd_inv * yd2_inv * erfc_y + two_over_sqrtpi * yd2_inv * exp_y2) / std::sqrt(det_epsilon);

    for (int icrd = 0; icrd < 3; ++icrd) {
        for (int jcrd = 0; jcrd < 3; ++jcrd) {
            hmat_out(icrd, jcrd) = x_tmp[icrd] * x_tmp[jcrd] * common_tmp[0]
                                   - epsilon_inv[icrd][jcrd] * common_tmp[1];
        }
    }
}
