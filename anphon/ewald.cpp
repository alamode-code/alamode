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
    if (mympi->my_rank == 0) {
        std::cout << '\n';
        std::cout << "  Preparing for the Ewald summation ...\n\n";
    }

    // exp(-p) = prec_ewald
    const double p = -std::log(prec_ewald);

    for (int icrd = 0; icrd < 3; ++icrd) {
        for (int jcrd = 0; jcrd < 3; ++jcrd) {
            epsilon[icrd][jcrd] = dielectric(icrd, jcrd);
        }
    }
    epsilon_mat = dielectric;
    // Symmetrize the dielectric tensor
    epsilon_mat = 0.5 * (epsilon_mat + epsilon_mat.transpose());
    invepsilon_mat = epsilon_mat.inverse();

    // Calculating convergence parameters
    invmat3(epsilon_inv, epsilon);

    const auto lavec_s_tmp = system->get_supercell(0).lattice_vector;
    const auto rlavec_s_tmp = system->get_supercell(0).reciprocal_lattice_vector.transpose();
    const auto lavec_p_tmp = system->get_primcell().lattice_vector;
    const auto rlavec_p_tmp = system->get_primcell().reciprocal_lattice_vector.transpose();

    get_lambda_and_lgmax(lavec_s_tmp, rlavec_s_tmp,
                         epsilon_mat, invepsilon_mat, p,
                         lambda_fcs,
                         Lmax_fcs, Gmax_fcs,
                         nl_fcs, ng_fcs);


    get_lambda_and_lgmax(lavec_p_tmp, rlavec_p_tmp,
                         epsilon_mat, invepsilon_mat, p,
                         lambda_dymat,
                         Lmax_dymat, Gmax_dymat,
                         nl_dymat, ng_dymat);

    det_epsilon = epsilon_mat.determinant();

    if (mympi->my_rank == 0) {
        std::cout << "  Inverse dielectric tensor : \n";
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                std::cout << std::setw(15) << invepsilon_mat(i, j);
            }
            std::cout << '\n';
        }
        std::cout << '\n';

        std::cout << "  Determinant of epsilon: " << std::setw(15) << det_epsilon
                << "\n\n";

        std::cout << "  Parameters for the Ewald summation :\n";
        std::cout << "  - Force constant\n";
        std::cout << "    Lambda : " << std::setw(15) << lambda_fcs << '\n';
        std::cout << "    Lmax   : " << std::setw(15) << Lmax_fcs << '\n';
        std::cout << "    Gmax   : " << std::setw(15) << Gmax_fcs << '\n';
        std::cout << "    Maximum number of real-space cells : "
                << std::setw(3) << nl_fcs[0] << "x" << std::setw(3) << nl_fcs[1] << "x" << std::setw(3) << nl_fcs[2]
                << '\n';
        std::cout << "    Maximum number of reciprocal cells : "
                << std::setw(3) << ng_fcs[0] << "x" << std::setw(3) << ng_fcs[1] << "x" << std::setw(3) << ng_fcs[2]
                << '\n';
        std::cout << '\n';
        std::cout << "  - Dynamical matrix\n";
        std::cout << "    Lambda : " << std::setw(15) << lambda_dymat << '\n';
        std::cout << "    Lmax   : " << std::setw(15) << Lmax_dymat << '\n';
        std::cout << "    Gmax   : " << std::setw(15) << Gmax_dymat << '\n';
        std::cout << "    Maximum number of real-space cells : "
                << std::setw(3) << nl_dymat[0] << "x"
                << std::setw(3) << nl_dymat[1] << "x"
                << std::setw(3) << nl_dymat[2] << '\n';
        std::cout << "    Maximum number of reciprocal cells : "
                << std::setw(3) << ng_dymat[0] << "x"
                << std::setw(3) << ng_dymat[1] << "x"
                << std::setw(3) << ng_dymat[2] << '\n';
        std::cout << '\n';
    }
}

void Ewald::get_lambda_and_lgmax(const Eigen::Matrix3d &lavec,
                                 const Eigen::Matrix3d &rlavec,
                                 const Eigen::Matrix3d &epsilon,
                                 const Eigen::Matrix3d &epsilon_inv,
                                 const double &p,
                                 double &lambda_out,
                                 double &Lmax_out, double &Gmax_out,
                                 Eigen::Vector3i &lsize, Eigen::Vector3i &gsize)
{
    const Eigen::Matrix3d metric_tensor_real = lavec.transpose() * epsilon_inv * lavec;
    const Eigen::Matrix3d metric_tensor_reciprocal = rlavec.transpose() * epsilon * rlavec;

    // Cholesky decomposition M = LL^T
    Eigen::LLT<Eigen::Matrix3d> llt_real(metric_tensor_real);
    Eigen::LLT<Eigen::Matrix3d> llt_reciprocal(metric_tensor_reciprocal);

    const Eigen::Matrix3d L_real = llt_real.matrixL();
    const Eigen::Matrix3d L_reciprocal = llt_reciprocal.matrixL();

    // L^{-T}
    const Eigen::Matrix3d inv_trans_L_real = L_real.inverse().transpose();
    const Eigen::Matrix3d inv_trans_L_reciprocal = L_reciprocal.inverse().transpose();

    double sum1 = 0.0;
    double sum2 = 0.0;
    double prod1 = 1.0;
    double prod2 = 1.0;
    for (size_t i = 0; i < 3; ++i) {
        sum1 = 0.0;
        sum2 = 0.0;
        for (size_t j = 0; j < 3; ++j) {
            sum1 += abs(inv_trans_L_real(i, j));
            sum2 += abs(inv_trans_L_reciprocal(i, j));
        }
        prod1 *= sum1;
        prod2 *= sum2;
    }

    lambda_out = std::pow((prod1 / (8.0 * prod2) * rate_ab), 1.0 / 6.0);
    Lmax_out = std::sqrt(p) / lambda_out;
    Gmax_out = 2.0 * lambda_out * std::sqrt(p);

    for (size_t i = 0; i < 3; ++i) {
        sum1 = 0.0;
        sum2 = 0.0;
        for (size_t j = 0; j < 3; ++j) {
            sum1 += abs(inv_trans_L_real(i, j));
            sum2 += abs(inv_trans_L_reciprocal(i, j));
        }
        lsize[i] = std::ceil(std::sqrt(p) / lambda_out * sum1);
        gsize[i] = std::ceil(2.0 * lambda_out * std::sqrt(p) * sum2);
    }
}


void Ewald::prepare_G()
{
    // Accumulate reciprocal lattice vectors

    int ix, iy, iz;
    Eigen::Vector3d g_tmp;
    double gnorm;

    G_vectors_fcs.clear();
    G_vectors_dymat.clear();

    const auto rlavec = system->get_supercell(0).reciprocal_lattice_vector.transpose();
    const auto rlavec_p = system->get_primcell().reciprocal_lattice_vector.transpose();

    const double gmax2_fcs = Gmax_fcs * Gmax_fcs;
    const double gmax2 = Gmax_dymat * Gmax_dymat;

    for (ix = -ng_fcs[0]; ix <= ng_fcs[0]; ++ix) {
        for (iy = -ng_fcs[1]; iy <= ng_fcs[1]; ++iy) {
            for (iz = -ng_fcs[2]; iz <= ng_fcs[2]; ++iz) {
                if (ix == 0 && iy == 0 && iz == 0) continue;
                g_tmp[0] = static_cast<double>(ix);
                g_tmp[1] = static_cast<double>(iy);
                g_tmp[2] = static_cast<double>(iz);

                g_tmp = rlavec * g_tmp;
                gnorm = g_tmp.dot(epsilon_mat * g_tmp);
                if (gnorm <= gmax2_fcs) {
                    G_vectors_fcs.emplace_back(g_tmp);
                }
            }
        }
    }

    for (ix = -ng_dymat[0]; ix <= ng_dymat[0]; ++ix) {
        for (iy = -ng_dymat[1]; iy <= ng_dymat[1]; ++iy) {
            for (iz = -ng_dymat[2]; iz <= ng_dymat[2]; ++iz) {
                if (ix == 0 && iy == 0 && iz == 0) continue;
                g_tmp[0] = static_cast<double>(ix);
                g_tmp[1] = static_cast<double>(iy);
                g_tmp[2] = static_cast<double>(iz);
                g_tmp = rlavec_p * g_tmp;
                gnorm = g_tmp.dot(epsilon_mat * g_tmp);
                if (gnorm <= gmax2) {
                    G_vectors_dymat.emplace_back(g_tmp);
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
    const auto nat = system->get_supercell(0).number_of_atoms;
    const auto natmin = system->get_primcell().number_of_atoms;
    double **fc_ewald_real_space_sum, **fc_ewald_reciprocal_space_sum;
    const std::string file_fcs_ewald = input->job_title + ".fc2_ewald";

    if (mympi->my_rank == 0) {
        std::cout << " Calculating long-range (dipole-dipole) FCs in the supercell ...";
    }

    const auto map_p2s = system->get_map_p2s(0);
    const auto map_s2p = system->get_map_s2p(0);

    Eigen::MatrixXd fcs_ewald(3 * natmin, 3 * nat);
    Eigen::MatrixXd fcs_total(3 * natmin, 3 * nat);
    Eigen::MatrixXd fcs_other(3 * natmin, 3 * nat);

    allocate(fc_ewald_real_space_sum, 3, 3);
    allocate(fc_ewald_reciprocal_space_sum, 3, 3);

    for (iat = 0; iat < natmin; ++iat) {
        atm_s = map_p2s[iat][0];
        for (jat = 0; jat < nat; ++jat) {
            calc_real_space_sum_ewald_fcs(atm_s, jat, fc_ewald_real_space_sum);
            calc_reciprocal_space_sum_ewald_fcs(atm_s, jat, fc_ewald_reciprocal_space_sum);

            for (icrd = 0; icrd < 3; ++icrd) {
                for (jcrd = 0; jcrd < 3; ++jcrd) {
                    fcs_ewald(3 * iat + icrd, 3 * jat + jcrd)
                            = fc_ewald_real_space_sum[icrd][jcrd]
                              + fc_ewald_reciprocal_space_sum[icrd][jcrd];
                }
            }
        }
    }

    deallocate(fc_ewald_real_space_sum);
    deallocate(fc_ewald_reciprocal_space_sum);

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
            const auto nmulti = multiplicity[atm_s][jat];

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


void Ewald::calc_real_space_sum_ewald_fcs(const int iat,
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

    const double lmax2 = Lmax_fcs * Lmax_fcs;

    const auto x_frac_super = system->get_supercell(0).x_fractional.transpose();
    const auto lavec = system->get_supercell(0).lattice_vector;

    if (iat == jat) {
        // k = k', where k labels atoms in the supercell

        for (icell = -nl_fcs[0]; icell <= nl_fcs[0]; ++icell) {
            for (jcell = -nl_fcs[1]; jcell <= nl_fcs[1]; ++jcell) {
                for (kcell = -nl_fcs[2]; kcell <= nl_fcs[2]; ++kcell) {
                    if (icell == 0 && jcell == 0 && kcell == 0) {
                        // l'' = l = 0

                        for (kat = 0; kat < system->get_supercell(0).number_of_atoms; ++kat) {
                            if (kat == iat) continue; // k'' = k

                            x_tmp = lavec * (x_frac_super.col(iat) - x_frac_super.col(kat));
                            xnorm = x_tmp.dot(invepsilon_mat * x_tmp);

                            if (xnorm < lmax2) {
                                calc_realspace_sum(iat, kat, x_tmp.data(), lambda_fcs, func_L);

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
                            xnorm = x_tmp.dot(invepsilon_mat * x_tmp);

                            if (xnorm < lmax2) {
                                calc_realspace_sum(iat, kat, x_tmp.data(), lambda_fcs, func_L);

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
                        xnorm = x_tmp.dot(invepsilon_mat * x_tmp);

                        if (xnorm < lmax2) {
                            calc_realspace_sum(iat, jat, x_tmp.data(), lambda_fcs, func_L);

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

        for (icell = -nl_fcs[0]; icell <= nl_fcs[0]; ++icell) {
            for (jcell = -nl_fcs[1]; jcell <= nl_fcs[1]; ++jcell) {
                for (kcell = -nl_fcs[2]; kcell <= nl_fcs[2]; ++kcell) {
                    trans[0] = static_cast<double>(icell);
                    trans[1] = static_cast<double>(jcell);
                    trans[2] = static_cast<double>(kcell);

                    x_tmp = lavec * (x_frac_super.col(iat) - x_frac_super.col(jat) - trans);
                    xnorm = x_tmp.dot(invepsilon_mat * x_tmp);

                    if (xnorm < lmax2) {
                        calc_realspace_sum(iat, jat, x_tmp.data(), lambda_fcs, func_L);

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

void Ewald::calc_reciprocal_space_sum_ewald_fcs(const int iat,
                                                const int jat,
                                                double **fc_g_out)
{
    // Reciprocal lattice sum part for FCs

    int i;
    int icrd, jcrd;
    int acrd, bcrd;
    double gnorm2;
    Eigen::Vector3d x_tmp, g_tmp, epsilon_gvector;
    double common_tmp;

    for (icrd = 0; icrd < 3; ++icrd) {
        for (jcrd = 0; jcrd < 3; ++jcrd) {
            fc_g_out[icrd][jcrd] = 0.0;
        }
    }
    const auto volume = system->get_supercell(0).volume;
    const auto ikd = system->get_map_s2p(0)[iat].atom_num;
    const auto jkd = system->get_map_s2p(0)[jat].atom_num;
    const auto factor = 4.0 * pi / volume;
    const auto x_frac_super = system->get_supercell(0).x_fractional.transpose();
    const auto lavec = system->get_supercell(0).lattice_vector;


    if (iat == jat) {
        for (const auto &it: G_vectors_fcs) {
            g_tmp = it.vec;
            epsilon_gvector = epsilon_mat * g_tmp;
            gnorm2 = g_tmp.dot(epsilon_gvector);

            for (int kat = 0; kat < system->get_supercell(0).number_of_atoms; ++kat) {
                const auto kkd = system->get_map_s2p(0)[kat].atom_num;

                x_tmp = lavec * (x_frac_super.col(iat) - x_frac_super.col(kat));

                common_tmp = factor * std::exp(-0.25 * gnorm2 / std::pow(lambda_fcs, 2.0)) / gnorm2
                             * std::cos(g_tmp.dot(x_tmp));

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

    for (const auto &it: G_vectors_fcs) {
        g_tmp = it.vec;
        epsilon_gvector = epsilon_mat * g_tmp;
        gnorm2 = g_tmp.dot(epsilon_gvector);
        x_tmp = lavec * (x_frac_super.col(iat) - x_frac_super.col(jat));
        common_tmp = 2.0 * factor * std::exp(-0.25 * gnorm2 / std::pow(lambda_fcs, 2.0)) / gnorm2
                     * std::cos(g_tmp.dot(x_tmp));

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
    const auto natmin = system->get_primcell().number_of_atoms;
    const long natmin2 = static_cast<long>(natmin * natmin);
    const auto neval = 3 * system->get_primcell().number_of_atoms;
    Eigen::Vector3d xk, kvec;

    // Move input xk back to the -0.5 <= xk < 0.5 range to avoid zero division.
    // Also, this is necessary to make the phonon dispersion periodic in the reciprocal lattice.
    for (auto i = 0; i < 3; ++i) {
        xk[i] = xk_in[i] - static_cast<double>(nint(xk_in[i]));
        kvec[i] = kvec_in[i];
    }

    xk = system->get_primcell().reciprocal_lattice_vector.transpose() * xk;

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
            calc_short_term_dynamical_matrix(iat, jat, xk.data(), dymat_tmp_l);
            calc_long_term_dynamical_matrix(iat, jat, xk, kvec, dymat_tmp_g);
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

    int icrd, jcrd;
    int icell, jcell, kcell;
    double xnorm, phase;
    Eigen::Vector3d x_tmp, trans;
    std::vector<std::vector<double> > func_L(3, std::vector<double>(3, 0.0));

    // Substitute quantities into variables
    const auto atm_s1 = system->get_map_p2s(0)[iat][0];
    const auto atm_s2 = system->get_map_p2s(0)[jat][0];
    const auto x_frac_super = system->get_supercell(0).x_fractional.transpose();
    const auto lavec_s = system->get_supercell(0).lattice_vector;
    const auto lavec_p = system->get_primcell().lattice_vector;

    const double lmax2 = Lmax_dymat * Lmax_dymat;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            mat_out[i][j] = std::complex<double>(0.0, 0.0);
        }
    }

    if (iat == jat) {
        int atm_s3;
        int kat;
        // k = k'

        for (icell = -nl_dymat[0]; icell <= nl_dymat[0]; ++icell) {
            for (jcell = -nl_dymat[1]; jcell <= nl_dymat[1]; ++jcell) {
                for (kcell = -nl_dymat[2]; kcell <= nl_dymat[2]; ++kcell) {
                    trans[0] = static_cast<double>(icell);
                    trans[1] = static_cast<double>(jcell);
                    trans[2] = static_cast<double>(kcell);
                    trans = lavec_p * trans;

                    // Lattice vector = 0
                    if (icell == 0 && jcell == 0 && kcell == 0) {
                        for (kat = 0; kat < system->get_primcell().number_of_atoms; ++kat) {
                            if (kat == iat) continue;

                            atm_s3 = system->get_map_p2s(0)[kat][0];
                            x_tmp = lavec_s * (x_frac_super.col(atm_s1) - x_frac_super.col(atm_s3));
                            xnorm = x_tmp.dot(invepsilon_mat * x_tmp);

                            if (xnorm < lmax2) {
                                calc_realspace_sum(atm_s1, atm_s3, x_tmp.data(), lambda_dymat, func_L);

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
                            x_tmp = lavec_s * (x_frac_super.col(atm_s1) - x_frac_super.col(atm_s3)) - trans;
                            xnorm = x_tmp.dot(invepsilon_mat * x_tmp);

                            if (xnorm < lmax2) {
                                calc_realspace_sum(atm_s1, atm_s3, x_tmp.data(), lambda_dymat, func_L);

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

                        x_tmp = lavec_s * (x_frac_super.col(atm_s1) - x_frac_super.col(atm_s2)) - trans;
                        xnorm = x_tmp.dot(invepsilon_mat * x_tmp);

                        if (xnorm < lmax2) {
                            calc_realspace_sum(atm_s1, atm_s2, x_tmp.data(), lambda_dymat, func_L);
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
        for (icell = -nl_dymat[0]; icell <= nl_dymat[0]; ++icell) {
            for (jcell = -nl_dymat[1]; jcell <= nl_dymat[1]; ++jcell) {
                for (kcell = -nl_dymat[2]; kcell <= nl_dymat[2]; ++kcell) {
                    trans[0] = static_cast<double>(icell);
                    trans[1] = static_cast<double>(jcell);
                    trans[2] = static_cast<double>(kcell);
                    trans = lavec_p * trans;

                    x_tmp = lavec_s * (x_frac_super.col(atm_s1) - x_frac_super.col(atm_s2)) - trans;
                    xnorm = x_tmp.dot(invepsilon_mat * x_tmp);

                    if (xnorm < lmax2) {
                        calc_realspace_sum(atm_s1, atm_s2, x_tmp.data(), lambda_dymat, func_L);
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
                                            const Eigen::Vector3d &xk_in,
                                            const Eigen::Vector3d &kvec_in,
                                            std::complex<double> **mat_out)
{
    // Reciprocal lattice sum part for a dynamical matrix

    int i, j;
    int icrd, jcrd, acrd, bcrd;
    Eigen::Vector3d vec, e_kvec;
    double tmp;

    const auto atm_s1 = system->get_map_p2s(0)[iat][0];
    const auto atm_s2 = system->get_map_p2s(0)[jat][0];
    const double mi = system->get_mass_super()[atm_s1];
    const double mj = system->get_mass_super()[atm_s2];
    const double vol_p = system->get_primcell().volume;

    const auto x_frac_super = system->get_supercell(0).x_fractional.transpose();
    const auto lavec = system->get_supercell(0).lattice_vector;

    vec = lavec * (x_frac_super.col(atm_s1) - x_frac_super.col(atm_s2));
    e_kvec = epsilon_mat * xk_in;

    for (i = 0; i < 3; ++i) {
        for (j = 0; j < 3; ++j) {
            mat_out[i][j] = std::complex<double>(0.0, 0.0);
        }
    }

    double phase = xk_in.dot(vec);
    double kd = xk_in.dot(e_kvec);

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
                        * std::exp(-0.25 * kd / std::pow(lambda_dymat, 2.0));
            }
        }
    } else {
        // Treat non-analytic term

        Eigen::Vector3d kdirec, e_kdirec;
        kdirec = kvec_in;
        e_kdirec = epsilon_mat * kdirec;
        double norm = kdirec.dot(e_kdirec);

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

    Eigen::Vector3d g, gk, vecl, g_tmp, gk_tmp;
    double common;

    for (auto &it: G_vectors_dymat) {
        for (int l = 0; l < 3; ++l) {
            g[l] = it.vec[l];
            gk[l] = g[l] + xk_in[l];
        }

        if (iat == jat) {
            g_tmp = epsilon_mat * g;
            const auto gd = g.dot(g_tmp);

            common = std::exp(-0.25 * gd / std::pow(lambda_dymat, 2.0)) / gd;

            for (int kat = 0; kat < system->get_primcell().number_of_atoms; ++kat) {
                int atm_s3 = system->get_map_p2s(0)[kat][0];

                vecl = lavec * (x_frac_super.col(atm_s1) - x_frac_super.col(atm_s3));

                exp_phase = std::exp(im * g.dot(vecl));

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

        gk_tmp = epsilon_mat * gk;

        const double gkd = gk.dot(gk_tmp);
        const double phase_g2 = gk.dot(vec);

        common = 2.0 * std::exp(-0.25 * gkd / std::pow(lambda_dymat, 2.0)) / gkd;
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
