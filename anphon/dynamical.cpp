/*
 dynamical.cpp

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "dynamical.h"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include "constants.h"
#include "dielec.h"
#include "error.h"
#include "ewald.h"
#include "fcs_phonon.h"
#include "kpoint.h"
#include "mathfunctions.h"
#include "memory.h"
#include "mpi_common.h"
#include "phonon_dos.h"
#include "symmetry_core.h"
#include "system.h"
#include "timer.h"
#include "write_phonons.h"


using namespace PHON_NS;

Dynamical::Dynamical(PHON *phon) : Pointers(phon)
{
    set_default_variables();
}

Dynamical::~Dynamical()
{
    deallocate_variables();
}

void Dynamical::set_default_variables()
{
    neval = 0;
    eigenvectors = true;
    print_eigenvectors = false;
    nonanalytic = 0;
    participation_ratio = false;
    band_connection = 0;
    na_sigma = 0.0;
    UPLO = 'U';

    index_bconnect = nullptr;
    is_imaginary = nullptr;
    xshift_s = nullptr;
    dymat = nullptr;
    mindist_list = nullptr;
    dymat_band = nullptr;
    dymat_general = nullptr;
}

void Dynamical::deallocate_variables()
{
    if (index_bconnect) {
        deallocate(index_bconnect);
    }
    if (is_imaginary) {
        deallocate(is_imaginary);
    }
    if (xshift_s) {
        deallocate(xshift_s);
    }
    if (dymat) {
        deallocate(dymat);
    }
    if (mindist_list) {
        deallocate(mindist_list);
    }

    delete dymat_band;
    delete dymat_general;
}

void DymatEigenValue::set_eigenvalues(const unsigned int n, double **eval_in)
{
    if (n <= this->nk) {
        for (unsigned int i = 0; i < n; ++i) {
            for (unsigned int j = 0; j < ns; ++j) {
                eval[i][j] = eval_in[i][j];
            }
        }
    } else {
        exit("set_eigenvalues",
             "the number of kpoint is larger than the one"
             "used in the constructor.");
    }
}

void DymatEigenValue::set_eigenvectors(const unsigned int n, std::complex<double> ***evec_in)
{
    if (!this->is_stored_eigvec) {
        exit("set_eigenvectors", "the array for the eigenvector is not allocated.");
    }
    if (n > this->nk) {
        exit("set_eigenvectors",
             "the number of kpoint is larger than "
             "the one used in the constructor.");
    }
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < ns; ++j) {
            for (unsigned int k = 0; k < ns; ++k) {
                evec[i][j][k] = evec_in[i][j][k];
            }
        }
    }
}

void DymatEigenValue::set_eigenvals_and_eigenvecs(const unsigned int n, double **eval_in,
                                                  std::complex<double> ***evec_in)
{
    this->set_eigenvalues(n, eval_in);
    if (this->is_stored_eigvec) {
        this->set_eigenvectors(n, evec_in);
    }
}

double **DymatEigenValue::get_eigenvalues() const
{
    return this->eval;
}

std::complex<double> ***DymatEigenValue::get_eigenvectors() const
{
    return this->evec;
}

void Dynamical::setup_dynamical()
{
    neval = 3 * system->get_primcell().number_of_atoms;

    if (mympi->my_rank == 0) {
        std::cout << '\n';
        std::cout << " ==================\n";
        std::cout << "  Dynamical matrix \n";
        std::cout << " ==================\n";
        if (nonanalytic == 0) {
            std::cout << '\n';
            std::cout << "  NONANALYTIC = 0 : No non-analytic correction. \n\n";
        } else if (nonanalytic == 1) {
            std::cout << '\n';
            std::cout << "  NONANALYTIC = 1 : Non-analytic part of the dynamical matrix will be included \n";
            std::cout << "                    by the Parlinski's method.\n";
            std::cout << "                    The damping factor for the non-analytic term : " << na_sigma << "\n\n";
        } else if (nonanalytic == 2) {
            std::cout << '\n';
            std::cout << "  NONANALYTIC = 2 : Non-analytic part of the dynamical matrix will be included \n";
            std::cout << "                    by the mixed-space approach.\n\n";
        } else if (nonanalytic == 3) {
            std::cout << '\n';
            std::cout << "  NONANALYTIC = 3 : Non-analytic part of the dynamical matrix will be included \n";
            std::cout << "                    by the Ewald method.\n\n";
        }

        if (!projection_directions.empty()) {
            std::cout << "\n\n";
            std::cout << "  PROJECTION_AXES of eigenvectors for degenerate modes:\n";
            auto axis_num = 1;
            for (const auto &it: projection_directions) {
                std::cout << "   Axis " << std::setw(2) << axis_num << ':';
                for (const auto &it2: it) {
                    std::cout << std::setw(15) << it2;
                }
                std::cout << '\n';
                ++axis_num;
            }
        }
    }

    allocate(xshift_s, 27, 3);

    for (auto i = 0; i < 3; ++i)
        xshift_s[0][i] = 0.0;
    auto icell = 0;

    for (auto ix = -1; ix <= 1; ++ix) {
        for (auto iy = -1; iy <= 1; ++iy) {
            for (auto iz = -1; iz <= 1; ++iz) {
                if (ix == 0 && iy == 0 && iz == 0) continue;

                ++icell;

                xshift_s[icell][0] = static_cast<double>(ix);
                xshift_s[icell][1] = static_cast<double>(iy);
                xshift_s[icell][2] = static_cast<double>(iz);
            }
        }
    }

    if (mympi->my_rank == 0) eigenvectors = true;

    MPI_Bcast(&eigenvectors, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nonanalytic, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&band_connection, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    if (kpoint->kpoint_bs) {
        dymat_band = new DymatEigenValue(eigenvectors, false, kpoint->kpoint_bs->nk, neval);
    }

    if (kpoint->kpoint_general) {
        dymat_general = new DymatEigenValue(eigenvectors, false, kpoint->kpoint_general->nk, neval);
    }

    // Bcast projection_directions
    unsigned int nsize_proj = projection_directions.size();
    MPI_Bcast(&nsize_proj, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    for (auto i = 0; i < nsize_proj; ++i) {
        double vec[3];
        std::vector<double> vec2(3);

        if (mympi->my_rank == 0) {
            for (auto j = 0; j < 3; ++j) {
                vec[j] = projection_directions[i][j];
            }
        }
        MPI_Bcast(&vec[0], 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (mympi->my_rank > 0) {
            for (auto j = 0; j < 3; ++j)
                vec2[j] = vec[j];
            projection_directions.push_back(vec2);
        }
    }

    if (nonanalytic) {
        MPI_Bcast(&na_sigma, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        allocate(mindist_list,
                 system->get_primcell().number_of_atoms,
                 system->get_supercell(0).number_of_atoms); // should use fc2 cell?
        prepare_mindist_list(mindist_list);
    }

    if (mympi->my_rank == 0) {
        std::cout << '\n';
    }
}

void Dynamical::prepare_mindist_list(std::vector<int> **mindist_out) const
{
    unsigned int i, j;
    const unsigned int nneib = 27;

    double ***xcrd;

    const auto scell_tmp = system->get_supercell(0);
    const auto pcell_tmp = system->get_primcell();
    const auto nat = scell_tmp.number_of_atoms;
    const auto natmin = pcell_tmp.number_of_atoms;

    std::vector<DistWithCell> **distall;

    allocate(distall, natmin, nat);
    allocate(xcrd, nneib, nat, 3);

    for (i = 0; i < nat; ++i) {
        for (j = 0; j < 3; ++j) {
            xcrd[0][i][j] = scell_tmp.x_fractional(i, j);
        }
    }
    auto icell = 0;
    for (auto isize = -1; isize <= 1; ++isize) {
        for (auto jsize = -1; jsize <= 1; ++jsize) {
            for (auto ksize = -1; ksize <= 1; ++ksize) {

                if (isize == 0 && jsize == 0 && ksize == 0) continue;

                ++icell;
                for (i = 0; i < nat; ++i) {
                    xcrd[icell][i][0] = scell_tmp.x_fractional(i, 0) + static_cast<double>(isize);
                    xcrd[icell][i][1] = scell_tmp.x_fractional(i, 1) + static_cast<double>(jsize);
                    xcrd[icell][i][2] = scell_tmp.x_fractional(i, 2) + static_cast<double>(ksize);
                }
            }
        }
    }

    for (icell = 0; icell < nneib; ++icell) {
        for (i = 0; i < nat; ++i) {
            rotvec(xcrd[icell][i], xcrd[icell][i], scell_tmp.lattice_vector);
        }
    }

    for (i = 0; i < natmin; ++i) {
        // TODO: replace map_p2s
        const auto iat = system->get_map_p2s(0)[i][0];
        for (j = 0; j < nat; ++j) {
            distall[i][j].clear();
            for (icell = 0; icell < nneib; ++icell) {

                double dist_tmp = distance(xcrd[0][iat], xcrd[icell][j]);
                distall[i][j].emplace_back(icell, dist_tmp);
            }
            std::sort(distall[i][j].begin(), distall[i][j].end());
        }
    }

    // Construct pairs of minimum distance.

    for (i = 0; i < natmin; ++i) {
        for (j = 0; j < nat; ++j) {
            mindist_out[i][j].clear();

            const auto dist_min = distall[i][j][0].dist;
            for (auto it = distall[i][j].begin(); it != distall[i][j].end(); ++it) {
                if (std::abs((*it).dist - dist_min) < 1.0e-3) {
                    mindist_out[i][j].push_back((*it).cell);
                }
            }
        }
    }

    deallocate(distall);
    deallocate(xcrd);
}

double Dynamical::distance(double *x1, double *x2)
{
    return std::sqrt(pow2(x1[0] - x2[0]) + pow2(x1[1] - x2[1]) + pow2(x1[2] - x2[2]));
}

void Dynamical::eval_k(const double *xk_in, const double *kvec_in, const std::vector<FcsArrayWithCell> &fc2,
                       double *eval_out, std::complex<double> **evec_out, const bool require_evec) const
{
    // Calculate phonon energy for the specific k-point given in fractional basis

    unsigned int i, j;
    std::complex<double> **dymat_k;

    allocate(dymat_k, neval, neval);

    calc_analytic_k(xk_in, fc2, dymat_k);

    if (nonanalytic) {

        // Add non-analytic correction

        std::complex<double> **dymat_na_k;

        allocate(dymat_na_k, neval, neval);

        if (nonanalytic == 1) {
            calc_nonanalytic_k(xk_in, kvec_in, dymat_na_k);
        } else if (nonanalytic == 2) {
            calc_nonanalytic_k2(xk_in, kvec_in, dymat_na_k);
        }

        for (i = 0; i < neval; ++i) {
            for (j = 0; j < neval; ++j) {
                dymat_k[i][j] += dymat_na_k[i][j];
            }
        }
        deallocate(dymat_na_k);
    }

    // Force the dynamical matrix be real when k point is
    // zone-center or zone-boundaries.

    if (std::sqrt(pow2(std::fmod(xk_in[0], 0.5)) + pow2(std::fmod(xk_in[1], 0.5)) + pow2(std::fmod(xk_in[2], 0.5))) <
        eps)
    {

        for (i = 0; i < neval; ++i) {
            for (j = 0; j < neval; ++j) {
                dymat_k[i][j] = std::complex<double>(dymat_k[i][j].real(), 0.0);
            }
        }
    }

    char JOBZ;
    int INFO;
    double *RWORK;
    std::complex<double> *WORK;

    int LWORK = (2 * neval - 1) * 10;
    allocate(RWORK, 3 * neval - 2);
    allocate(WORK, LWORK);

    std::complex<double> *amat;
    allocate(amat, neval * neval);

    unsigned int k = 0;
    int n = dynamical->neval;

    for (j = 0; j < neval; ++j) {
        for (i = 0; i < neval; ++i) {
            amat[k++] = dymat_k[i][j];
        }
    }

    deallocate(dymat_k);

    if (require_evec) {
        JOBZ = 'V';
    } else {
        JOBZ = 'N';
    }

    // Perform diagonalization
    zheev_(&JOBZ, &UPLO, &n, amat, &n, eval_out, WORK, &LWORK, RWORK, &INFO);
    if (INFO != 0) {
        exit("Dynamical::diagonalize", "zheev failed to diagonalize the dynamical matrix (INFO != 0).");
    }

    if (eigenvectors && require_evec) {
        k = 0;
        // Here we transpose the matrix evec_out so that
        // evec_out[i] becomes phonon eigenvector of i-th mode.
        for (j = 0; j < neval; ++j) {
            for (i = 0; i < neval; ++i) {
                evec_out[j][i] = amat[k++];
            }
        }
    }

    deallocate(RWORK);
    deallocate(WORK);
    deallocate(amat);
}

void Dynamical::eval_k_ewald(const double *xk_in, const double *kvec_in, const std::vector<FcsArrayWithCell> &fc2_in,
                             double *eval_out, std::complex<double> **evec_out, const bool require_evec) const
{
    //
    // Calculate phonon energy for the specific k-point given in fractional basis
    // Contributions from dipole-dipole interactions should be absent in 'fc2_in'.
    //
    unsigned int i, j;
    int icrd, jcrd;
    std::complex<double> **dymat_k, **mat_longrange;

    allocate(dymat_k, neval, neval);
    allocate(mat_longrange, neval, neval);

    calc_analytic_k(xk_in, fc2_in, dymat_k);

    // Calculate Coulombic contributions including long-range interactions
    ewald->add_longrange_matrix(xk_in, kvec_in, mat_longrange);

    const auto nat_prim = system->get_primcell().number_of_atoms;
    // Add calculated dynamical matrix of Coulomb parts
    for (i = 0; i < nat_prim; ++i) {
        for (icrd = 0; icrd < 3; ++icrd) {
            for (j = 0; j < nat_prim; ++j) {
                for (jcrd = 0; jcrd < 3; ++jcrd) {
                    dymat_k[3 * i + icrd][3 * j + jcrd] += mat_longrange[3 * i + icrd][3 * j + jcrd];
                }
            }
        }
    }

    // Check acoustic sum rule
    if (xk_in[0] == 0.0 && xk_in[1] == 0.0 && xk_in[2] == 0.0) {
        for (i = 0; i < nat_prim; ++i) {
            for (icrd = 0; icrd < 3; ++icrd) {
                for (jcrd = 0; jcrd < 3; ++jcrd) {
                    auto check = std::complex<double>(0.0, 0.0);
                    for (j = 0; j < nat_prim; ++j) {
                        const auto mass = system->get_mass_prim()[i] * system->get_mass_prim()[j];
                        check += std::sqrt(mass) * dymat_k[3 * i + icrd][3 * j + jcrd];
                    }

                    if (std::abs(check) > eps8) {
                        std::cout << "(" << 3 * i + icrd << "," << jcrd << "): " << check << '\n';
                        warn("ewald->eval_k_ewald", "Acoustic sum rule is broken.");
                    }
                }
            }
        }
    }

    char JOBZ;
    int INFO;
    double *RWORK;
    std::complex<double> *WORK;

    int LWORK = (2 * neval - 1) * 10;
    allocate(RWORK, 3 * neval - 2);
    allocate(WORK, LWORK);

    std::complex<double> *amat;
    allocate(amat, neval * neval);

    unsigned int k = 0;
    int n = dynamical->neval;
    for (j = 0; j < neval; ++j) {
        for (i = 0; i < neval; ++i) {
            amat[k++] = dymat_k[i][j];
        }
    }

    deallocate(dymat_k);

    if (require_evec) {
        JOBZ = 'V';
    } else {
        JOBZ = 'N';
    }

    // Perform diagonalization
    zheev_(&JOBZ, &UPLO, &n, amat, &n, eval_out, WORK, &LWORK, RWORK, &INFO);
    if (INFO != 0) {
        exit("Dynamical::diagonalize", "zheev failed to diagonalize the dynamical matrix (INFO != 0).");
    }

    if (eigenvectors && require_evec) {
        k = 0;
        // Here we transpose the matrix evec_out so that
        // evec_out[i] becomes phonon eigenvector of i-th mode.
        for (j = 0; j < neval; ++j) {
            for (i = 0; i < neval; ++i) {
                evec_out[j][i] = amat[k++];
            }
        }
    }

    deallocate(RWORK);
    deallocate(WORK);
    deallocate(amat);
}

void Dynamical::calc_analytic_k(const double *xk_in, const std::vector<FcsClassExtent> &fc2_in,
                                std::complex<double> **dymat_out) const
{
    int i;
    Eigen::Vector3d vec;
    Eigen::Matrix3d convmat =
        system->get_primcell().reciprocal_lattice_vector * system->get_supercell(0).lattice_vector;

    const auto xf_tmp = system->get_supercell(0).x_fractional;

    for (i = 0; i < neval; ++i) {
        for (auto j = 0; j < neval; ++j) {
            dymat_out[i][j] = std::complex<double>(0.0, 0.0);
        }
    }

    for (const auto &it: fc2_in) {

        const auto atm1_p = it.atm1;
        const auto atm2_s = it.atm2;
        const auto xyz1 = it.xyz1;
        const auto xyz2 = it.xyz2;
        const auto icell = it.cell_s;

        const auto atm1_s = system->get_map_p2s(0)[atm1_p][0];
        const auto atm2_p = system->get_map_s2p(0)[atm2_s].atom_num;

        for (i = 0; i < 3; ++i) {
            vec[i] = xf_tmp(atm2_s, i) + xshift_s[icell][i] - xf_tmp(system->get_map_p2s(0)[atm2_p][0], i);
        }

        vec = convmat * vec;

        const auto phase = vec[0] * xk_in[0] + vec[1] * xk_in[1] + vec[2] * xk_in[2];

        dymat_out[3 * atm1_p + xyz1][3 * atm2_p + xyz2] +=
            it.fcs_val * std::exp(im * phase) /
            std::sqrt(system->get_mass_super()[atm1_s] * system->get_mass_super()[atm2_s]);
    }
}

void Dynamical::calc_analytic_k(const double *xk_in, const std::vector<FcsArrayWithCell> &fc2_in,
                                std::complex<double> **dymat_out) const
{
    for (auto i = 0; i < neval; ++i) {
        for (auto j = 0; j < neval; ++j) {
            dymat_out[i][j] = std::complex<double>(0.0, 0.0);
        }
    }

    const auto invsqrt_mass = system->get_invsqrt_mass();

    for (const auto &it: fc2_in) {
        const auto phase =
            tpi * (it.relvecs[0][0] * xk_in[0] + it.relvecs[0][1] * xk_in[1] + it.relvecs[0][2] * xk_in[2]);
        dymat_out[it.pairs[0].index][it.pairs[1].index] += it.fcs_val * std::exp(im * phase) *
                                                           invsqrt_mass[it.pairs[0].index / 3] *
                                                           invsqrt_mass[it.pairs[1].index / 3];
    }
}

void Dynamical::calc_nonanalytic_k(const double *xk_in, const double *kvec_na_in,
                                   std::complex<double> **dymat_na_out) const
{
    // Calculate the non-analytic part of dynamical matrices
    // by Parlinski's method.

    unsigned int i, j;
    unsigned int iat, jat;
    const auto pcell = system->get_primcell();
    const auto nat_prim = pcell.number_of_atoms;
    Eigen::Vector3d kepsilon, kvec_na_vec;
    double kz1[3], kz2[3];
    double born_tmp[3][3];
    Eigen::Vector3d xk_tmp, xdiff;

    for (i = 0; i < neval; ++i) {
        for (j = 0; j < neval; ++j) {
            dymat_na_out[i][j] = std::complex<double>(0.0, 0.0);
        }
    }

    auto dielec_tmp = dielec->get_dielec_tensor();
    const auto borncharge = dielec->get_borncharge();

    for (i = 0; i < 3; ++i) {
        kvec_na_vec[i] = kvec_na_in[i];
    }
    const auto denom = kvec_na_vec.dot(dielec_tmp * kvec_na_vec);

    if (denom > eps) {

        for (iat = 0; iat < nat_prim; ++iat) {
            const auto atm_p1 = system->get_map_p2s(0)[iat][0];

            for (i = 0; i < 3; ++i) {
                for (j = 0; j < 3; ++j) {
                    born_tmp[i][j] = borncharge[iat][i][j];
                }
            }

            rotvec(kz1, kvec_na_in, born_tmp, 'T');

            for (jat = 0; jat < nat_prim; ++jat) {
                const auto atm_p2 = system->get_map_p2s(0)[jat][0];

                for (i = 0; i < 3; ++i) {
                    for (j = 0; j < 3; ++j) {
                        born_tmp[i][j] = borncharge[jat][i][j];
                    }
                }

                rotvec(kz2, kvec_na_in, born_tmp, 'T');

                for (i = 0; i < 3; ++i) {
                    for (j = 0; j < 3; ++j) {

                        dymat_na_out[3 * iat + i][3 * jat + j] =
                            kz1[i] * kz2[j] /
                            (denom * std::sqrt(system->get_mass_super()[atm_p1] * system->get_mass_super()[atm_p2]));
                    }
                }
            }
        }
    }
    // Move input xk back to the -0.5 <= xk < 0.5 range to
    // make the phonon dispersion periodic in the reciprocal lattice.
    // For the moment, comment out here as it gives strange values for group velocities.
    // Need to use Niggli reduction for this to work properly.
    //    for (i = 0; i < 3; ++i) {
    //        xk_tmp[i] = xk_in[i] - static_cast<double>(nint(xk_in[i]));
    //    }

    for (i = 0; i < 3; ++i)
        xk_tmp[i] = xk_in[i];
    xk_tmp = pcell.reciprocal_lattice_vector.transpose() * xk_tmp;
    const auto norm2 = xk_tmp.squaredNorm();

    const auto factor = 8.0 * pi / system->get_primcell().volume * std::exp(-norm2 / pow2(na_sigma));

    for (i = 0; i < neval; ++i) {
        for (j = 0; j < neval; ++j) {
            dymat_na_out[i][j] *= factor;
        }
    }

    // Multiply an additional phase factor for the non-analytic term.

    const auto xf_tmp = system->get_supercell(0).x_fractional;
    const auto convmat = pcell.reciprocal_lattice_vector * system->get_supercell(0).lattice_vector;

    for (iat = 0; iat < nat_prim; ++iat) {
        for (jat = 0; jat < nat_prim; ++jat) {

            for (i = 0; i < 3; ++i) {
                xdiff[i] = xf_tmp(system->get_map_p2s(0)[iat][0], i) - xf_tmp(system->get_map_p2s(0)[jat][0], i);
            }

            xdiff = convmat * xdiff;

            const double phase = xk_tmp[0] * xdiff[0] + xk_tmp[1] * xdiff[1] + xk_tmp[2] * xdiff[2];

            for (i = 0; i < 3; ++i) {
                for (j = 0; j < 3; ++j) {
                    dymat_na_out[3 * iat + i][3 * jat + j] *= exp(im * phase);
                }
            }
        }
    }
}

void Dynamical::calc_nonanalytic_k2(const double *xk_in, const double *kvec_na_in,
                                    std::complex<double> **dymat_na_out) const
{
    // Calculate the non-analytic part of dynamical matrices
    // by the mixed-space approach.

    unsigned int i, j;
    const auto natmin = system->get_primcell().number_of_atoms;
    Eigen::Vector3d kvec_na_vec;
    double kz1[3], kz2[3];
    double born_tmp[3][3];
    Eigen::Vector3d vec;
    Eigen::Matrix3d convmat =
        system->get_primcell().reciprocal_lattice_vector * system->get_supercell(0).lattice_vector;

    const auto xf_tmp = system->get_supercell(0).x_fractional;

    for (i = 0; i < neval; ++i) {
        for (j = 0; j < neval; ++j) {
            dymat_na_out[i][j] = std::complex<double>(0.0, 0.0);
        }
    }

    auto dielec_tmp = dielec->get_dielec_tensor();
    const auto borncharge = dielec->get_borncharge();

    for (i = 0; i < 3; ++i) {
        kvec_na_vec[i] = kvec_na_in[i];
    }
    const auto denom = kvec_na_vec.dot(dielec_tmp * kvec_na_vec);


    if (denom > eps) {

        for (unsigned int iat = 0; iat < natmin; ++iat) {
            unsigned int atm_p1 = system->get_map_p2s(0)[iat][0];

            for (i = 0; i < 3; ++i) {
                for (j = 0; j < 3; ++j) {
                    born_tmp[i][j] = borncharge[iat][i][j];
                }
            }

            rotvec(kz1, kvec_na_in, born_tmp, 'T');

            for (unsigned int jat = 0; jat < natmin; ++jat) {
                unsigned int atm_p2 = system->get_map_p2s(0)[jat][0];

                for (i = 0; i < 3; ++i) {
                    for (j = 0; j < 3; ++j) {
                        born_tmp[i][j] = borncharge[jat][i][j];
                    }
                }

                rotvec(kz2, kvec_na_in, born_tmp, 'T');

                std::complex<double> exp_phase = std::complex<double>(0.0, 0.0);

                for (i = 0; i < system->get_map_p2s(0)[0].size(); ++i) {

                    std::complex<double> exp_phase_tmp = std::complex<double>(0.0, 0.0);
                    unsigned int atm_s2 = system->get_map_p2s(0)[jat][i];

                    // Average over periodic images

                    for (j = 0; j < mindist_list[iat][atm_s2].size(); ++j) {
                        unsigned int cell = mindist_list[iat][atm_s2][j];

                        for (unsigned int k = 0; k < 3; ++k) {
                            vec[k] = xf_tmp(system->get_map_p2s(0)[jat][i], k) + xshift_s[cell][k] - xf_tmp(atm_p2, k);
                        }


                        //                        rotvec(vec, vec, system->get_supercell(0).lattice_vector);
                        //                        rotvec(vec, vec, system->get_primcell().reciprocal_lattice_vector);

                        vec = convmat * vec;
                        double phase = vec[0] * xk_in[0] + vec[1] * xk_in[1] + vec[2] * xk_in[2];

                        exp_phase_tmp += std::exp(im * phase);
                    }
                    exp_phase += exp_phase_tmp / static_cast<double>(mindist_list[iat][atm_s2].size());
                }
                exp_phase /= static_cast<double>(system->get_map_p2s(0)[0].size());

                for (i = 0; i < 3; ++i) {
                    for (j = 0; j < 3; ++j) {
                        dymat_na_out[3 * iat + i][3 * jat + j] =
                            kz1[i] * kz2[j] /
                            (denom * std::sqrt(system->get_mass_super()[atm_p1] * system->get_mass_super()[atm_p2])) *
                            exp_phase;
                    }
                }
            }
        }
    }

    const auto factor = 8.0 * pi / system->get_primcell().volume;

    for (i = 0; i < neval; ++i) {
        for (j = 0; j < neval; ++j) {
            dymat_na_out[i][j] *= factor;
        }
    }
}

void Dynamical::diagonalize_dynamical_all()
{
    unsigned int nk;

    if (mympi->my_rank == 0) {
        std::cout << '\n' << " Diagonalizing dynamical matrices for all k points ... ";
    }
    double **eval_tmp;
    std::complex<double> ***evec_tmp;
    // k points for general mode (manual entry)

    if (kpoint->kpoint_general) {
        nk = kpoint->kpoint_general->nk;
        allocate(eval_tmp, nk, neval);
        if (eigenvectors) {
            allocate(evec_tmp, nk, neval, neval);
        } else {
            allocate(evec_tmp, nk, 1, 1);
        }

        get_eigenvalues_dymat(nk,
                              kpoint->kpoint_general->xk,
                              kpoint->kpoint_general->kvec_na,
                              fcs_phonon->force_constant_with_cell[0],
                              ewald->fc2_without_dipole,
                              eigenvectors,
                              eval_tmp,
                              evec_tmp);

        if (!projection_directions.empty()) {
            if (mympi->my_rank == 0) {

                for (auto ik = 0; ik < nk; ++ik) {
                    project_degenerate_eigenvectors(system->get_primcell().lattice_vector,
                                                    fcs_phonon->force_constant_with_cell[0],
                                                    kpoint->kpoint_general->xk[ik],
                                                    projection_directions,
                                                    evec_tmp[ik]);
                }
            }

            MPI_Bcast(&evec_tmp[0][0][0], nk * neval * neval, MPI_CXX_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
        }

        dymat_general->set_eigenvals_and_eigenvecs(nk, eval_tmp, evec_tmp);
        deallocate(eval_tmp);
        deallocate(evec_tmp);
    }

    // k points for band structure
    if (kpoint->kpoint_bs) {
        nk = kpoint->kpoint_bs->nk;
        allocate(eval_tmp, nk, neval);
        if (eigenvectors) {
            allocate(evec_tmp, nk, neval, neval);
        } else {
            allocate(evec_tmp, nk, 1, 1);
        }
        get_eigenvalues_dymat(nk,
                              kpoint->kpoint_bs->xk,
                              kpoint->kpoint_bs->kvec_na,
                              fcs_phonon->force_constant_with_cell[0],
                              ewald->fc2_without_dipole,
                              eigenvectors,
                              eval_tmp,
                              evec_tmp);

        if (!projection_directions.empty()) {
            if (mympi->my_rank == 0) {
                for (auto ik = 0; ik < nk; ++ik) {
                    project_degenerate_eigenvectors(system->get_primcell().lattice_vector,
                                                    fcs_phonon->force_constant_with_cell[0],
                                                    kpoint->kpoint_bs->xk[ik],
                                                    projection_directions,
                                                    evec_tmp[ik]);
                }
            }

            MPI_Bcast(&evec_tmp[0][0][0], nk * neval * neval, MPI_CXX_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
        }

        dymat_band->set_eigenvals_and_eigenvecs(nk, eval_tmp, evec_tmp);

        deallocate(eval_tmp);
        deallocate(evec_tmp);
    }

    // k points for dos
    if (dos->kmesh_dos) {
        nk = dos->kmesh_dos->nk;
        allocate(eval_tmp, nk, neval);
        if (eigenvectors) {
            allocate(evec_tmp, nk, neval, neval);
        } else {
            allocate(evec_tmp, nk, 1, 1);
        }
        get_eigenvalues_dymat(nk,
                              dos->kmesh_dos->xk,
                              dos->kmesh_dos->kvec_na,
                              fcs_phonon->force_constant_with_cell[0],
                              ewald->fc2_without_dipole,
                              eigenvectors,
                              eval_tmp,
                              evec_tmp);

        if (!projection_directions.empty()) {
            if (mympi->my_rank == 0) {
                for (auto ik = 0; ik < nk; ++ik) {
                    project_degenerate_eigenvectors(system->get_primcell().lattice_vector,
                                                    fcs_phonon->force_constant_with_cell[0],
                                                    dos->kmesh_dos->xk[ik],
                                                    projection_directions,
                                                    evec_tmp[ik]);
                }
            }

            MPI_Bcast(&evec_tmp[0][0][0], nk * neval * neval, MPI_CXX_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
        }

        dos->dymat_dos->set_eigenvals_and_eigenvecs(nk, eval_tmp, evec_tmp);
        deallocate(eval_tmp);
        deallocate(evec_tmp);
    }

    if (band_connection > 0 && kpoint->kpoint_bs) {
        allocate(index_bconnect, kpoint->kpoint_bs->nk, neval);
        connect_band_by_eigen_similarity(kpoint->kpoint_bs->nk, dymat_band->get_eigenvectors(), index_bconnect);
    }

    if (mympi->my_rank == 0) {
        std::cout << "done!\n";
    }

    if (dos->kmesh_dos && phon->mode == "RTA") {
        detect_imaginary_branches(*dos->kmesh_dos, dos->dymat_dos->get_eigenvalues());
    }
}

void Dynamical::get_eigenvalues_dymat(const unsigned int nk_in, const double *const *xk_in,
                                      const double *const *kvec_na_in, const std::vector<FcsArrayWithCell> &fc2,
                                      const std::vector<FcsArrayWithCell> &fc2_without_dipole_in,
                                      const bool require_evec, double **eval_ret, std::complex<double> ***evec_ret)
{
    if (nk_in <= 0) {
        exit("get_eigenvalues_dymat", "The number of k points must be larger than 0.");
    }

    // Calculate phonon eigenvalues and eigenvectors for all k-points
    // We should not use OpenMP parallelization for this part because
    // LAPACK is called inside each function, which also used thread parallelization.
    for (int ik = 0; ik < nk_in; ++ik) {
        if (nonanalytic == 3) {
            eval_k_ewald(&xk_in[ik][0],
                         &kvec_na_in[ik][0],
                         fc2_without_dipole_in,
                         eval_ret[ik],
                         evec_ret[ik],
                         require_evec);
        } else {
            eval_k(&xk_in[ik][0], &kvec_na_in[ik][0], fc2, eval_ret[ik], evec_ret[ik], require_evec);
        }
        // Phonon energy is the square-root of the eigenvalue
        for (unsigned int is = 0; is < neval; ++is) {
            eval_ret[ik][is] = freq(eval_ret[ik][is]);
        }
    }
}

void Dynamical::modify_eigenvectors() const
{
    bool *flag_done;
    unsigned int ik;
    unsigned int is, js;
    std::complex<double> ***evec_tmp;

    const auto nk = dos->kmesh_dos->nk;
    const auto ns = neval;

    /*   if (mympi->my_rank == 0) {
           std::cout << " **********      NOTICE      ********** " << '\n';
           std::cout << " For the brevity of the calculation, " << '\n';
           std::cout << " phonon eigenvectors will be modified" << '\n';
           std::cout << " so that e_{-ks}^{mu} = (e_{ks}^{mu})^{*}. " << '\n';
       }*/

    allocate(flag_done, nk);
    allocate(evec_tmp, nk, ns, ns);

    for (ik = 0; ik < nk; ++ik) {
        for (is = 0; is < ns; ++is) {
            for (js = 0; js < ns; ++js) {
                evec_tmp[ik][is][js] = dos->dymat_dos->get_eigenvectors()[ik][is][js];
            }
        }
    }

    for (ik = 0; ik < nk; ++ik)
        flag_done[ik] = false;

    for (ik = 0; ik < nk; ++ik) {

        if (!flag_done[ik]) {

            const auto nk_inv = dos->kmesh_dos->kindex_minus_xk[ik];

            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns; ++js) {
                    evec_tmp[nk_inv][is][js] = std::conj(evec_tmp[ik][is][js]);
                }
            }

            flag_done[ik] = true;
            flag_done[nk_inv] = true;
        }
    }

    deallocate(flag_done);
    dos->dymat_dos->set_eigenvectors(nk, evec_tmp);

    deallocate(evec_tmp);

    MPI_Barrier(MPI_COMM_WORLD);
    //if (mympi->my_rank == 0) {
    //    std::cout << " done !" << '\n';
    //    std::cout << " **************************************" << '\n';
    //}
}

void Dynamical::project_degenerate_eigenvectors(const Eigen::Matrix3d &lavec_p,
                                                const std::vector<FcsArrayWithCell> &fc2_in, double *xk_in,
                                                const std::vector<std::vector<double>> &project_directions,
                                                std::complex<double> **evec_out) const
{
    int i, j;
    const auto ns = this->neval;

    //
    // The projector is given in the real space Cartesian coordinate.
    // Let's transform the basis into the crystal coordinate and normalize the norm to unity.
    //
    std::vector<std::vector<double>> directions;
    std::vector<double> vec(3);
    for (const auto &it: project_directions) {
        for (i = 0; i < 3; ++i) {
            vec[i] = it[i];
        }
        rotvec(&vec[0], &vec[0], lavec_p, 'T');

        auto norm = 0.0;
        for (i = 0; i < 3; ++i) {
            norm += vec[i] * vec[i];
        }
        norm = std::sqrt(norm);
        for (i = 0; i < 3; ++i)
            vec[i] = vec[i] / norm;

        directions.push_back(vec);
    }

    //
    // Diagonalize dymat at xk_in and get degeneracy information.
    //
    Eigen::MatrixXcd dymat(ns, ns);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> saes;

    std::complex<double> **dymat_tmp;

    allocate(dymat_tmp, ns, ns);

    calc_analytic_k(xk_in, fc2_in, dymat_tmp);

    if (std::sqrt(pow2(std::fmod(xk_in[0], 0.5)) + pow2(std::fmod(xk_in[1], 0.5)) + pow2(std::fmod(xk_in[2], 0.5))) <
        eps)
    {

        for (i = 0; i < neval; ++i) {
            for (j = 0; j < neval; ++j) {
                dymat_tmp[i][j] = std::complex<double>(dymat_tmp[i][j].real(), 0.0);
            }
        }
    }

    for (i = 0; i < ns; ++i) {
        for (j = 0; j < ns; ++j) {
            dymat(i, j) = dymat_tmp[i][j];
        }
    }
    deallocate(dymat_tmp);

    saes.compute(dymat);

    auto eval_orig = saes.eigenvalues();
    auto evec_orig = saes.eigenvectors();

    //
    // Construct degeneracy information
    //
    const double tol_omega = 1.0e-14; // Approximately equal to (0.01 cm^{-1})^2

    std::vector<int> degeneracy_at_k;

    degeneracy_at_k.clear();
    double omega_prev = eval_orig[0];
    int ideg = 1;

    for (j = 1; j < ns; ++j) {
        double omega_now = eval_orig[j];

        if (std::abs(omega_now - omega_prev) < tol_omega) {
            ++ideg;
        } else {
            degeneracy_at_k.push_back(ideg);
            ideg = 1;
            omega_prev = omega_now;
        }
    }
    degeneracy_at_k.push_back(ideg);

    const auto ndirec = directions.size();

    //
    // For each degenerate subset, apply the perturbation field in the direction
    // defined by "directions".
    //
    int ishift = 0;
    const double dk = 1.0e-3; // Small value may be preferable.
    Eigen::MatrixXcd evec_new(ns, ns);

    for (int iset: degeneracy_at_k) {

        if (iset == 1) {
            // Non degenerate case. just copy the original eigenvector

            evec_new.block(0, ishift, ns, 1) = evec_orig.block(0, ishift, ns, 1);

        } else if (iset == 2) {
            // Doubly degenerate case.

            if (ndirec == 0) {
                evec_new.block(0, ishift, ns, 2) = evec_orig.block(0, ishift, ns, 2);
            } else {

                Eigen::MatrixXcd evec_sub = evec_orig.block(0, ishift, ns, 2);
                auto is_lifted = transform_eigenvectors(xk_in, directions[0], dk, evec_sub);

                if (is_lifted == 0 and ndirec >= 2) {
                    evec_sub = evec_orig.block(0, ishift, ns, 2);
                    is_lifted = transform_eigenvectors(xk_in, directions[1], dk, evec_sub);

                    if (is_lifted == 0 and ndirec >= 3) {
                        evec_sub = evec_orig.block(0, ishift, ns, 2);
                        is_lifted = transform_eigenvectors(xk_in, directions[2], dk, evec_sub);
                    }
                }

                if (is_lifted == 0) {
                    std::cout << " xk = ";
                    for (i = 0; i < 3; ++i)
                        std::cout << std::setw(15) << xk_in[i];
                    std::cout << '\n';
                    std::cout << " All projections did not lift the two-fold degeneracy.\n"
                                 " Try another projection!\n";

                    evec_new.block(0, ishift, ns, 2) = evec_orig.block(0, ishift, ns, 2);
                } else {
                    evec_new.block(0, ishift, ns, 2) = evec_sub.block(0, 0, ns, 2);
                }
            }

        } else if (iset == 3) {
            // Triply degenerate case

            if (ndirec == 0) {

                evec_new.block(0, ishift, ns, 3) = evec_orig.block(0, ishift, ns, 3);

            } else if (ndirec == 1) {

                Eigen::MatrixXcd evec_sub = evec_orig.block(0, ishift, ns, 3);
                auto is_lifted = transform_eigenvectors(xk_in, directions[0], dk, evec_sub);

                evec_new.block(0, ishift, ns, 3) = evec_sub.block(0, 0, ns, 3);
                if (is_lifted == 0) {
                    std::cout << " xk = ";
                    for (i = 0; i < 3; ++i)
                        std::cout << std::setw(15) << xk_in[i];
                    std::cout << '\n';
                    std::cout << " The first projection did not lift the two-fold degeneracy.\n"
                                 " Try another projection!\n";
                }

            } else if (ndirec >= 2) {

                Eigen::MatrixXcd evec_sub = evec_orig.block(0, ishift, ns, 3);
                auto is_lifted1 = transform_eigenvectors(xk_in, directions[0], dk, evec_sub);

                Eigen::MatrixXcd evec_sub2 = evec_sub.block(0, 1, ns, 2);
                auto is_lifted2 = transform_eigenvectors(xk_in, directions[1], dk, evec_sub2);

                evec_new.block(0, ishift, ns, 1) = evec_sub.block(0, 0, ns, 1);
                evec_new.block(0, ishift + 1, ns, 2) = evec_sub2.block(0, 0, ns, 2);

                if (is_lifted1 == 0 || is_lifted2 == 0) {
                    std::cout << " xk = ";
                    for (i = 0; i < 3; ++i)
                        std::cout << std::setw(15) << xk_in[i];
                    std::cout << '\n';
                    std::cout << " The given projections did not lift the three-fold degeneracy.\n"
                                 " Try another set of projections!\n";
                }
            }

        } else {
            std::cout << iset << '\n';
            exitall("project_degenerate_eigenvectors", "This should not happen.");
        }

        ishift += iset;
    }

#ifdef _DEBUG
    std::cout << "Check if the original dynamical matrix can be recovered\n";
    std::cout << evec_new * eval_orig.asDiagonal() * evec_new.adjoint() - dymat << '\n';
#endif

    for (i = 0; i < ns; ++i) {
        for (j = 0; j < ns; ++j) {
            evec_out[i][j] = evec_new(j, i);
        }
    }
}

int Dynamical::transform_eigenvectors(double *xk_in, std::vector<double> perturb_direction, const double dk,
                                      Eigen::MatrixXcd &evec_sub) const
{
    int i;
    double xk_shift[3], xk_shift_minus[3];
    const auto ns = this->neval;
    const auto tol_ediff = dk * dk * 1.0e-2;
    std::complex<double> **dymat_dq, **dymat_dq_minus;

    int is_lifted = 0;

    Eigen::MatrixXcd ddymat(ns, ns);

    for (i = 0; i < 3; ++i) {
        xk_shift[i] = xk_in[i] + perturb_direction[i] * dk;
        xk_shift_minus[i] = xk_in[i] - perturb_direction[i] * dk;
    }

    allocate(dymat_dq, ns, ns);
    allocate(dymat_dq_minus, ns, ns);
    calc_analytic_k(xk_shift, fcs_phonon->force_constant_with_cell[0], dymat_dq);
    calc_analytic_k(xk_shift_minus, fcs_phonon->force_constant_with_cell[0], dymat_dq_minus);

    for (auto is = 0; is < ns; ++is) {
        for (auto js = 0; js < ns; ++js) {
            // This treatment helps to avoid unwanted small imaginary components
            // in the perturbation matrix.
            ddymat(is, js) = (dymat_dq[is][js] + dymat_dq_minus[is][js]) / (2.0 * dk);
        }
    }
    deallocate(dymat_dq);
    deallocate(dymat_dq_minus);

    // The perturbation matrix (the size is ndeg x ndeg)
    auto pertmat = evec_sub.adjoint() * ddymat * evec_sub;

    // Diagonalize
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> saes;

    saes.compute(pertmat);
    auto eval_pert = saes.eigenvalues();
    auto Umat = saes.eigenvectors();

    // Transform the eigenvectors and keep it for later.
    // auto should not be used here.
    Eigen::MatrixXcd evec_tmp = evec_sub * Umat;

    const auto ndeg = eval_pert.size();

    if (ndeg == 2) {

        if (std::abs(eval_pert[0] - eval_pert[1]) > tol_ediff * std::abs(eval_pert[1])) {
            // Degeneracy is lifted!
            //
            // We assume that the applied perturbation increases the energy.
            // If this is the case, the eigenvector along the perturbed direction should be
            // the second one, whose energy is higher than the first one.
            //
            // swap the order of the eigenvectors
            evec_sub.block(0, 0, ns, 1) = evec_tmp.block(0, 1, ns, 1);
            evec_sub.block(0, 1, ns, 1) = evec_tmp.block(0, 0, ns, 1);
            is_lifted = 1;
        }

    } else if (ndeg == 3) {

        if (std::abs(eval_pert[0] - eval_pert[1]) > tol_ediff * std::abs(eval_pert[1])) {
            // Degeneracy is lifted!
            //
            // When the states split as (3) --> (1, 2),
            //
            evec_sub.block(0, 0, ns, 1) = evec_tmp.block(0, 0, ns, 1);
            evec_sub.block(0, 1, ns, 2) = evec_tmp.block(0, 1, ns, 2);
            is_lifted = 1;
        } else if (std::abs(eval_pert[1] - eval_pert[2]) > tol_ediff * std::abs(eval_pert[2])) {
            // Degeneracy is lifted!
            //
            // When the states split as (3) --> (2, 1),
            //
            evec_sub.block(0, 0, ns, 1) = evec_tmp.block(0, 2, ns, 1);
            evec_sub.block(0, 1, ns, 2) = evec_tmp.block(0, 0, ns, 2);
            is_lifted = 1;
        }
    }

    return is_lifted;
}

double Dynamical::fold(const double x) const
{
    return x - static_cast<double>(nint(x));
}

double Dynamical::freq(const double x) const
{
    // Special treatment to avoid the divergence of computation.
    if (std::abs(x) < eps) return eps15;

    if (x > 0.0) return std::sqrt(x);

    return -std::sqrt(-x);
}

void Dynamical::calc_participation_ratio_all(const unsigned int nk_in,
                                             const std::complex<double> *const *const *evec_in, double **ret,
                                             double ***ret_all) const
{
    const auto ns = dynamical->neval;
    const auto natmin = system->get_primcell().number_of_atoms;

    double *atomic_pr;

    allocate(atomic_pr, natmin);

    for (auto ik = 0; ik < nk_in; ++ik) {
        for (auto is = 0; is < ns; ++is) {
            calc_atomic_participation_ratio(evec_in[ik][is], atomic_pr);

            auto sum = 0.0;

            for (auto iat = 0; iat < natmin; ++iat) {
                sum += atomic_pr[iat];
                ret_all[ik][is][iat] = atomic_pr[iat];
            }

            ret[ik][is] = sum * sum;
        }
    }

    deallocate(atomic_pr);
}

void Dynamical::calc_atomic_participation_ratio(const std::complex<double> *evec_in, double *ret) const
{
    unsigned int iat;
    const auto natmin = system->get_primcell().number_of_atoms;

    for (iat = 0; iat < natmin; ++iat)
        ret[iat] = 0.0;

    for (iat = 0; iat < natmin; ++iat) {
        ret[iat] = (std::norm(evec_in[3 * iat]) + std::norm(evec_in[3 * iat + 1]) + std::norm(evec_in[3 * iat + 2])) /
                   system->get_mass_super()[system->get_map_p2s(0)[iat][0]];
    }

    auto sum = 0.0;

    for (iat = 0; iat < natmin; ++iat)
        sum += ret[iat] * ret[iat];

    for (iat = 0; iat < natmin; ++iat)
        ret[iat] /= std::sqrt(static_cast<double>(natmin) * sum);
}

void Dynamical::connect_band_by_eigen_similarity(const unsigned int nk_in, std::complex<double> ***evec,
                                                 int **index_sorted) const
{
    int ik, is, js;
    const auto ns = neval;
    std::vector<int> index;
    std::complex<double> **evec_tmp;
    std::vector<std::vector<double>> abs_similarity;
    std::complex<double> dprod;
    std::vector<int> found;

    allocate(evec_tmp, ns, ns);

    for (ik = 0; ik < nk_in; ++ik) {
        for (is = 0; is < ns; ++is) {
            index_sorted[ik][is] = 0;
        }
    }

    index.resize(ns);
    found.resize(ns);
    abs_similarity.resize(ns);
    for (is = 0; is < ns; ++is) {
        abs_similarity[is].resize(ns);
    }

    for (int i = 0; i < ns; ++i)
        index[i] = i;

    for (ik = 0; ik < nk_in; ++ik) {

        if (ik == 0) {
            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns; ++js) {
                    if (is == js) {
                        abs_similarity[is][js] = 1.0;
                    } else {
                        abs_similarity[is][js] = 0.0;
                    }
                }
            }
        } else {
#ifdef _OPENMP
#pragma omp parallel for private(js, dprod)
#endif
            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns; ++js) {
                    dprod = std::complex<double>(0.0, 0.0);
                    for (int ks = 0; ks < ns; ++ks) {
                        dprod += std::conj(evec[ik][is][ks]) * evec_tmp[js][ks];
                    }
                    abs_similarity[is][js] = std::abs(dprod);
                }
            }
        }

        for (auto &v: found)
            v = 0;

        for (is = 0; is < ns; ++is) {

            // Argsort abs_similarity[is] (use C++11 lambda)
            iota(index.begin(), index.end(), 0);
            std::sort(index.begin(), index.end(), [&abs_similarity, is](int i1, int i2) {
                return abs_similarity[is][i1] > abs_similarity[is][i2];
            });

            int loc = index[0];
            index_sorted[ik][loc] = is;
            found[loc] = 1;
            for (js = 0; js < ns; ++js)
                abs_similarity[js][loc] = -1.0;
            for (js = 0; js < ns; ++js) {
                evec_tmp[loc][js] = evec[ik][is][js];
            }
        }

        if (std::any_of(found.begin(), found.end(), [](int i1) { return i1 == 0; })) {
            exit("connect_band_by_eigen_similarity", "Could not identify the connection.");
        }
    }
    deallocate(evec_tmp);
}

void Dynamical::detect_imaginary_branches(const KpointMeshUniform &kmesh_in, double **eval_in)
{
    int ik, is;
    const auto nk = kmesh_in.nk;
    const auto nk_irred = kmesh_in.nk_irred;
    const auto ns = dynamical->neval;
    const auto nks = ns * nk;
    int knum;
    double omega;

    auto is_anyof_imaginary = false;
    if (mympi->my_rank == 0) {

        allocate(is_imaginary, nk_irred, ns);

        for (ik = 0; ik < nk_irred; ++ik) {
            for (is = 0; is < ns; ++is) {
                knum = kmesh_in.kpoint_irred_all[ik][0].knum;
                omega = eval_in[knum][is];

                if (omega < 0.0) {
                    is_imaginary[ik][is] = true;
                    is_anyof_imaginary = true;
                } else {
                    is_imaginary[ik][is] = false;
                }
            }
        }

        if (is_anyof_imaginary) {
            int count = 0;
            std::cout << '\n';
            std::cout << " WARNING: Imaginary frequency detected at the following branches:" << '\n';
            for (ik = 0; ik < nk_irred; ++ik) {
                for (is = 0; is < ns; ++is) {
                    if (is_imaginary[ik][is]) {
                        const int ndup = kmesh_in.kpoint_irred_all[ik].size();
                        count += ndup;
                        for (auto i = 0; i < ndup; ++i) {
                            knum = kmesh_in.kpoint_irred_all[ik][i].knum;
                            omega = eval_in[knum][is];
                            for (int j = 0; j < 3; ++j) {
                                std::cout << std::setw(15) << kmesh_in.xk[knum][j];
                            }
                            std::cout << std::setw(4) << is + 1 << " :" << std::setw(10) << std::fixed
                                      << writes->in_kayser(omega) << " (cm^-1)" << '\n';
                            std::cout << std::scientific;
                        }
                    }
                }
            }
            std::cout << std::setw(5) << count << " imaginary branches out of " << std::setw(5) << nks
                      << " total branches.\n\n";
            std::cout << " Phonon-phonon scattering rate and thermal conductivity involving these\n";
            std::cout << " imaginary branches will be treated as zero in the following calculations.\n";
            std::cout << " If imaginary branches are acoustic phonons at Gamma point (0, 0, 0), \n";
            std::cout << " you can safely ignore this message.\n\n" << std::flush;
        }
    }
}

void Dynamical::set_projection_directions(const std::vector<std::vector<double>> projections_in)
{
    projection_directions = projections_in;
}

std::vector<std::vector<double>> Dynamical::get_projection_directions() const
{
    return projection_directions;
}

void Dynamical::r2q(const double *xk_in, const unsigned int nx, const unsigned int ny, const unsigned int nz,
                    const unsigned int ns, MinimumDistList ***mindist_list_in, std::complex<double> ***dymat_r_in,
                    std::complex<double> **dymat_k_out)
{
    const auto ncell = nx * ny * nz;
    const auto ns2 = ns * ns;

    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);

#pragma omp parallel for
    for (int ij = 0; ij < ns2; ++ij) {
        const auto i = ij / ns;
        const auto j = ij % ns;
        const auto iat = i / 3;
        const auto jat = j / 3;

        dymat_k_out[i][j] = complex_zero;

        for (auto icell = 0; icell < ncell; ++icell) {
            auto exp_phase = complex_zero;
            // This operation is necessary for the Hermiticity of the dynamical matrix.
            for (const auto &it: mindist_list_in[iat][jat][icell].shift) {
                auto phase = 2.0 * pi *
                             (static_cast<double>(it.sx) * xk_in[0] + static_cast<double>(it.sy) * xk_in[1] +
                              static_cast<double>(it.sz) * xk_in[2]);

                exp_phase += std::exp(im * phase);
            }
            exp_phase /= static_cast<double>(mindist_list_in[iat][jat][icell].shift.size());
            dymat_k_out[i][j] += dymat_r_in[i][j][icell] * exp_phase;
        }
    }
}

void Dynamical::precompute_dymat_harm(const unsigned int nk_in, double **xk_in, double **kvec_in,
                                      std::vector<Eigen::MatrixXcd> &dymat_short,
                                      std::vector<Eigen::MatrixXcd> &dymat_long) const
{
    const auto ns = neval;
    dymat_short.clear();
    dymat_long.clear();

    dymat_short.resize(nk_in);
    if (nonanalytic) {
        dymat_long.resize(nk_in);
    }

    Eigen::MatrixXcd mat_tmp_eigen(ns, ns);

    std::complex<double> **mat_tmp;
    allocate(mat_tmp, ns, ns);

    for (auto ik = 0; ik < nk_in; ++ik) {
        if (nonanalytic == 3) {
            calc_analytic_k(xk_in[ik], ewald->fc2_without_dipole, mat_tmp);
        } else {
            calc_analytic_k(xk_in[ik], fcs_phonon->force_constant_with_cell[0], mat_tmp);
        }

        for (auto is = 0; is < ns; ++is) {
            for (auto js = 0; js < ns; ++js) {
                mat_tmp_eigen(is, js) = mat_tmp[is][js];
            }
        }
        dymat_short[ik] = mat_tmp_eigen;
    }

    if (nonanalytic) {

        for (auto ik = 0; ik < nk_in; ++ik) {
            if (nonanalytic == 1) {
                calc_nonanalytic_k(xk_in[ik], kvec_in[ik], mat_tmp);
            } else if (nonanalytic == 2) {
                calc_nonanalytic_k2(xk_in[ik], kvec_in[ik], mat_tmp);

            } else if (nonanalytic == 3) {
                ewald->add_longrange_matrix(xk_in[ik], kvec_in[ik], mat_tmp);
            }
            for (auto is = 0; is < ns; ++is) {
                for (auto js = 0; js < ns; ++js) {
                    mat_tmp_eigen(is, js) = mat_tmp[is][js];
                }
            }
            dymat_long[ik] = mat_tmp_eigen;
        }
    }

    deallocate(mat_tmp);
}


void Dynamical::fourier_dymat_k_to_r(const unsigned int nk1, const unsigned int nk2, const unsigned int nk3,
                                     const unsigned int ns, const std::complex<double> *const *const *dymat_k,
                                     std::complex<double> ***dymat_r)
{
    // Forward DFT (k -> r), including the 1/N normalization, of all (is, js)
    // components of a coarse-mesh dynamical-matrix array at once:
    //   dymat_r[is][js][r] = (1/N) sum_k dymat_k[is][js][k] e^{-2 pi i k.r}.
    //
    // This used to be done with one FFTW plan per (is, js) pair, re-created at
    // every call: fftw_execute() always transforms the arrays its plan was
    // created with, and fftw_execute_dft() on different arrays carries strict
    // alignment requirements, so the plan could not be hoisted safely. The
    // coarse mesh is tiny, so plan creation dominated the transform itself.
    // The explicit DFT matrix below reproduces the FFTW_FORWARD convention
    // (row-major multi-index, negative exponent) exactly, is alignment-free,
    // and applies to all ns^2 components as a single matrix product; the arrays
    // from allocate() are contiguous, with (is, js) blocks of length
    // nk1*nk2*nk3 each.

    using namespace Eigen;

    const auto nk_coarse = nk1 * nk2 * nk3;

    MatrixXcd dft_matrix(nk_coarse, nk_coarse);
    for (unsigned int ir = 0; ir < nk_coarse; ++ir) {
        const auto r1 = ir / (nk2 * nk3);
        const auto r2 = (ir / nk3) % nk2;
        const auto r3 = ir % nk3;
        for (unsigned int ik = 0; ik < nk_coarse; ++ik) {
            const auto k1 = ik / (nk2 * nk3);
            const auto k2 = (ik / nk3) % nk2;
            const auto k3 = ik % nk3;
            const auto phase = -2.0 * pi
                               * (static_cast<double>(r1 * k1) / static_cast<double>(nk1)
                                  + static_cast<double>(r2 * k2) / static_cast<double>(nk2)
                                  + static_cast<double>(r3 * k3) / static_cast<double>(nk3));
            dft_matrix(ir, ik) = std::complex<double>(std::cos(phase), std::sin(phase));
        }
    }
    // 1/N normalization previously applied after the FFT
    dft_matrix /= static_cast<double>(nk_coarse);

    Map<const MatrixXcd> dymat_k_flat(dymat_k[0][0], nk_coarse, ns * ns);
    Map<MatrixXcd> dymat_r_flat(dymat_r[0][0], nk_coarse, ns * ns);
    dymat_r_flat.noalias() = dft_matrix * dymat_k_flat;
}

void Dynamical::compute_renormalized_harmonic_frequency(
    double **omega2_out, std::complex<double> ***evec_harm_renormalized, std::complex<double> **delta_v2_renorm,
    const double *const *omega2_harmonic, const std::complex<double> *const *const *evec_harmonic,
    const KpointMeshUniform *kmesh_coarse, const KpointMeshUniform *kmesh_dense,
    const std::vector<int> &kmap_interpolate_to_scph, std::complex<double> ****mat_transform_sym,
    MinimumDistList ***mindist_list, const unsigned int verbosity)
{
    using namespace Eigen;

    int ik;
    int is, js;
    const auto nk = kmesh_dense->nk;
    const auto nk_interpolate = kmesh_coarse->nk;
    const auto ns = dynamical->neval;
    unsigned int knum, knum_interpolate;
    const auto nk_irred_interpolate = kmesh_coarse->nk_irred;
    const auto nk1 = kmesh_coarse->nk_i[0];
    const auto nk2 = kmesh_coarse->nk_i[1];
    const auto nk3 = kmesh_coarse->nk_i[2];

    MatrixXcd evec_tmp(ns, ns);

    MatrixXcd Dymat(ns, ns);
    MatrixXcd Fmat(ns, ns);

    double **eval_interpolate;

    std::complex<double> ***dymat_new, ***dymat_harmonic_without_renormalize;
    std::complex<double> ***dymat_q;

    constexpr auto complex_zero = std::complex<double>(0.0, 0.0);

    SelfAdjointEigenSolver<MatrixXcd> saes;

    allocate(eval_interpolate, nk, ns);
    allocate(dymat_new, ns, ns, nk_interpolate);
    allocate(dymat_q, ns, ns, nk_interpolate);
    allocate(dymat_harmonic_without_renormalize, nk_interpolate, ns, ns);

    // Set initial harmonic dymat without IFC renormalization

    for (ik = 0; ik < nk_interpolate; ++ik) {
        calc_analytic_k(kmesh_coarse->xk[ik],
                        fcs_phonon->force_constant_with_cell[0],
                        dymat_harmonic_without_renormalize[ik]);
    }

    for (ik = 0; ik < nk_irred_interpolate; ++ik) {

        knum_interpolate = kmesh_coarse->kpoint_irred_all[ik][0].knum;
        knum = kmap_interpolate_to_scph[knum_interpolate];

        // calculate Fmat
        for (is = 0; is < ns; ++is) {
            for (js = 0; js < ns; ++js) {
                if (is == js) {
                    Fmat(is, js) = omega2_harmonic[knum][is];
                } else {
                    Fmat(is, js) = complex_zero;
                }
                Fmat(is, js) += delta_v2_renorm[knum_interpolate][is * ns + js];
            }
        }

        for (is = 0; is < ns; ++is) {
            for (js = 0; js < ns; ++js) {
                evec_tmp(is, js) = evec_harmonic[knum][js][is]; // transpose
            }
        }

        Dymat = evec_tmp * Fmat * evec_tmp.adjoint();

        symmetrize_dynamical_matrix(ik, kmesh_coarse, mat_transform_sym, Dymat);
        for (is = 0; is < ns; ++is) {
            for (js = 0; js < ns; ++js) {
                dymat_q[is][js][knum_interpolate] = Dymat(is, js);
            }
        }
    }

    replicate_dymat_for_all_kpoints(kmesh_coarse, mat_transform_sym, dymat_q);

    // Subtract harmonic contribution to the dynamical matrix
    for (ik = 0; ik < nk_interpolate; ++ik) {
        for (is = 0; is < ns; ++is) {
            for (js = 0; js < ns; ++js) {
                dymat_q[is][js][ik] -= dymat_harmonic_without_renormalize[ik][is][js];
            }
        }
    }

    fourier_dymat_k_to_r(nk1, nk2, nk3, ns, dymat_q, dymat_new);

    std::vector<Eigen::MatrixXcd> dymat_short, dymat_long;

    exec_interpolation(kmesh_coarse->nk_i,
                       dymat_new,
                       nk,
                       kmesh_dense->xk,
                       kmesh_dense->kvec_na,
                       eval_interpolate,
                       evec_harm_renormalized,
                       dymat_short,
                       dymat_long,
                       mindist_list);

    for (ik = 0; ik < nk; ++ik) {
        for (is = 0; is < ns; ++is) {
            if (eval_interpolate[ik][is] < 0.0) {
                if (std::abs(eval_interpolate[ik][is]) <= eps10) {
                    omega2_out[ik][is] = 0.0;
                } else {
                    omega2_out[ik][is] = -pow2(eval_interpolate[ik][is]);
                }
            } else {
                omega2_out[ik][is] = pow2(eval_interpolate[ik][is]);
            }
        }
    }


    deallocate(dymat_q);
    deallocate(dymat_new);

    deallocate(eval_interpolate);
    deallocate(dymat_harmonic_without_renormalize);
}


void Dynamical::symmetrize_dynamical_matrix(const unsigned int ik, const KpointMeshUniform *kmesh_coarse,
                                            std::complex<double> ****mat_transform_sym, Eigen::MatrixXcd &dymat) const
{
    // Symmetrize the dynamical matrix of given index ik.
    using namespace Eigen;
    unsigned int i, isym;
    unsigned int is, js;
    const auto ns = dynamical->neval;
    MatrixXcd dymat_sym = MatrixXcd::Zero(ns, ns);
    MatrixXcd dymat_tmp(ns, ns), gamma(ns, ns);

    const auto nsym_small = kmesh_coarse->small_group_of_k[ik].size();
    const auto nsym_minus = kmesh_coarse->symop_minus_at_k[ik].size();

    for (i = 0; i < nsym_minus; ++i) {
        isym = kmesh_coarse->symop_minus_at_k[ik][i];

        for (is = 0; is < ns; ++is) {
            for (js = 0; js < ns; ++js) {
                gamma(is, js) = mat_transform_sym[ik][isym][is][js];
            }
        }

        // Eq. (3.35) of Maradudin & Vosko
        dymat_tmp = gamma * dymat * gamma.transpose().conjugate();
        dymat_sym += dymat_tmp.conjugate();
    }

    for (i = 0; i < nsym_small; ++i) {
        isym = kmesh_coarse->small_group_of_k[ik][i];

        for (is = 0; is < ns; ++is) {
            for (js = 0; js < ns; ++js) {
                gamma(is, js) = mat_transform_sym[ik][isym][is][js];
            }
        }

        // Eq. (3.14) of Maradudin & Vosko
        dymat_tmp = gamma * dymat * gamma.transpose().conjugate();
        dymat_sym += dymat_tmp;
    }

    dymat = dymat_sym / static_cast<double>(nsym_small + nsym_minus);
}


void Dynamical::replicate_dymat_for_all_kpoints(const KpointMeshUniform *kmesh_coarse,
                                                std::complex<double> ****mat_transform_sym,
                                                std::complex<double> ***dymat_inout) const
{
    using namespace Eigen;
    unsigned int i;
    unsigned int is, js;
    const auto ns = neval;
    MatrixXcd dymat_tmp(ns, ns), gamma(ns, ns), dymat(ns, ns);

    std::complex<double> ***dymat_all;

    allocate(dymat_all, ns, ns, kmesh_coarse->nk);

    for (i = 0; i < kmesh_coarse->nk; ++i) {

        const auto ik_irred = kmesh_coarse->kpoint_map_symmetry[i].knum_irred_orig;
        const auto ik_orig = kmesh_coarse->kpoint_map_symmetry[i].knum_orig;
        const auto isym = kmesh_coarse->kpoint_map_symmetry[i].symmetry_op;

        if (isym >= 0) {
            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns; ++js) {
                    gamma(is, js) = mat_transform_sym[ik_irred][isym][is][js];
                    dymat(is, js) = dymat_inout[is][js][ik_orig];
                }
            }
            dymat_tmp = gamma * dymat * gamma.transpose().conjugate();
            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns; ++js) {
                    dymat_all[is][js][i] = dymat_tmp(is, js);
                }
            }
        }
    }

    // When the point group operation S_ which transforms k into -k, i.e., (S_)k = -k,
    // does not exist for k, we simply set D(k)=D(-k)^{*}.
    // (This should hold even when the time-reversal symmetry breaks.)
    for (i = 0; i < kmesh_coarse->nk; ++i) {
        const auto ik_orig = kmesh_coarse->kpoint_map_symmetry[i].knum_orig;
        const auto isym = kmesh_coarse->kpoint_map_symmetry[i].symmetry_op;
        if (isym == -1) {
            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns; ++js) {
                    dymat_all[is][js][i] = std::conj(dymat_all[is][js][ik_orig]);
                }
            }
        }
    }

    for (is = 0; is < ns; ++is) {
        for (js = 0; js < ns; ++js) {
            for (i = 0; i < kmesh_coarse->nk; ++i) {
                dymat_inout[is][js][i] = dymat_all[is][js][i];
            }
        }
    }
    deallocate(dymat_all);
}

void Dynamical::replicate_dymat_for_all_kpoints(const KpointMeshUniform *kmesh_coarse,
                                                std::complex<double> ****mat_transform_sym,
                                                std::vector<Eigen::MatrixXcd> &dymat_inout) const
{
    // Eigen version: works directly with vector of MatrixXcd
    // Avoids temporary C-style array allocation

    using namespace Eigen;
    const auto ns = neval;
    const auto nk = kmesh_coarse->nk;

    MatrixXcd gamma(ns, ns);

    // Temporary storage for replicated dynamical matrices
    std::vector<MatrixXcd> dymat_all;
    dymat_all.reserve(nk);
    for (unsigned int i = 0; i < nk; ++i) {
        dymat_all.emplace_back(ns, ns);
    }

    // Apply symmetry operations to replicate for all k-points
    for (unsigned int i = 0; i < nk; ++i) {

        const auto ik_irred = kmesh_coarse->kpoint_map_symmetry[i].knum_irred_orig;
        const auto ik_orig = kmesh_coarse->kpoint_map_symmetry[i].knum_orig;
        const auto isym = kmesh_coarse->kpoint_map_symmetry[i].symmetry_op;

        if (isym >= 0) {
            // Copy transformation matrix and dynamical matrix
            for (unsigned int is = 0; is < ns; ++is) {
                for (unsigned int js = 0; js < ns; ++js) {
                    gamma(is, js) = mat_transform_sym[ik_irred][isym][is][js];
                }
            }

            // Apply symmetry transformation: D'(k) = Γ D(k) Γ^†
            dymat_all[i] = gamma * dymat_inout[ik_orig] * gamma.adjoint();
            ;
        }
    }

    // Handle time-reversal symmetry: D(k) = D(-k)^*
    // When point group operation S_ which transforms k into -k doesn't exist,
    // we set D(k) = D(-k)^* (holds even when time-reversal symmetry breaks)
    for (unsigned int i = 0; i < nk; ++i) {
        const auto ik_orig = kmesh_coarse->kpoint_map_symmetry[i].knum_orig;
        const auto isym = kmesh_coarse->kpoint_map_symmetry[i].symmetry_op;

        if (isym == -1) {
            dymat_all[i] = dymat_all[ik_orig].conjugate();
        }
    }

    // Copy results back to input/output vector
    for (unsigned int i = 0; i < nk; ++i) {
        dymat_inout[i] = dymat_all[i];
    }
}


void Dynamical::exec_interpolation(const unsigned int kmesh_orig[3], std::complex<double> ***dymat_r,
                                   const unsigned int nk_dense, double **xk_dense, double **kvec_dense,
                                   double **eval_out, std::complex<double> ***evec_out,
                                   const std::vector<Eigen::MatrixXcd> &dymat_short,
                                   const std::vector<Eigen::MatrixXcd> &dymat_long, MinimumDistList ***mindist_list_in,
                                   const bool use_precomputed_dymat, const bool return_sqrt) const
{
    unsigned int i, j, is;
    const auto ns = dynamical->neval;
    const auto nk1 = kmesh_orig[0];
    const auto nk2 = kmesh_orig[1];
    const auto nk3 = kmesh_orig[2];

    double *eval_real = nullptr;
    std::complex<double> **mat_tmp = nullptr;

    std::vector<double> eval_vec(ns);

    allocate(mat_tmp, ns, ns);
    allocate(eval_real, ns);

    if (use_precomputed_dymat) {

        for (int ik = 0; ik < nk_dense; ++ik) {

            r2q(xk_dense[ik], nk1, nk2, nk3, ns, mindist_list_in, dymat_r, mat_tmp);

            for (i = 0; i < ns; ++i) {
                for (j = 0; j < ns; ++j) {
                    mat_tmp[i][j] += dymat_short[ik](i, j);
                }
            }

            if (nonanalytic) {
                for (i = 0; i < ns; ++i) {
                    for (j = 0; j < ns; ++j) {
                        mat_tmp[i][j] += dymat_long[ik](i, j);
                    }
                }
            }
            diagonalize_interpolated_matrix(mat_tmp, eval_real, evec_out[ik], true);

            if (return_sqrt) {
                for (is = 0; is < ns; ++is) {
                    const auto eval_tmp = eval_real[is];
                    if (eval_tmp < 0.0) {
                        eval_vec[is] = -std::sqrt(-eval_tmp);
                    } else {
                        eval_vec[is] = std::sqrt(eval_tmp);
                    }
                }
                for (is = 0; is < ns; ++is)
                    eval_out[ik][is] = eval_vec[is];
            } else {
                for (is = 0; is < ns; ++is)
                    eval_out[ik][is] = eval_real[is];
            }
        }

    } else {

        std::complex<double> **mat_harmonic = nullptr;
        std::complex<double> **mat_harmonic_na = nullptr;

        allocate(mat_harmonic, ns, ns);
        if (nonanalytic) {
            allocate(mat_harmonic_na, ns, ns);
        }

        for (int ik = 0; ik < nk_dense; ++ik) {
            if (nonanalytic == 3) {
                calc_analytic_k(xk_dense[ik], ewald->fc2_without_dipole, mat_harmonic);
            } else {
                calc_analytic_k(xk_dense[ik], fcs_phonon->force_constant_with_cell[0], mat_harmonic);
            }
            r2q(xk_dense[ik], nk1, nk2, nk3, ns, mindist_list_in, dymat_r, mat_tmp);
            for (i = 0; i < ns; ++i) {
                for (j = 0; j < ns; ++j) {
                    mat_tmp[i][j] += mat_harmonic[i][j];
                }
            }
            if (nonanalytic) {
                if (nonanalytic == 1) {
                    calc_nonanalytic_k(xk_dense[ik], kvec_dense[ik], mat_harmonic_na);
                } else if (nonanalytic == 2) {
                    calc_nonanalytic_k2(xk_dense[ik], kvec_dense[ik], mat_harmonic_na);
                } else if (nonanalytic == 3) {
                    ewald->add_longrange_matrix(xk_dense[ik], kvec_dense[ik], mat_harmonic_na);
                }
                for (i = 0; i < ns; ++i) {
                    for (j = 0; j < ns; ++j) {
                        mat_tmp[i][j] += mat_harmonic_na[i][j];
                    }
                }
            }
            diagonalize_interpolated_matrix(mat_tmp, eval_real, evec_out[ik], true);

            if (return_sqrt) {
                for (is = 0; is < ns; ++is) {
                    const auto eval_tmp = eval_real[is];
                    if (eval_tmp < 0.0) {
                        eval_vec[is] = -std::sqrt(-eval_tmp);
                    } else {
                        eval_vec[is] = std::sqrt(eval_tmp);
                    }
                }
                for (is = 0; is < ns; ++is)
                    eval_out[ik][is] = eval_vec[is];
            } else {
                for (is = 0; is < ns; ++is)
                    eval_out[ik][is] = eval_real[is];
            }
        }
        if (mat_harmonic) deallocate(mat_harmonic);
        if (mat_harmonic_na) deallocate(mat_harmonic_na);
    }

    if (eval_real) deallocate(eval_real);
    if (mat_tmp) deallocate(mat_tmp);
}


void Dynamical::diagonalize_interpolated_matrix(std::complex<double> **mat_in, double *eval_out,
                                                std::complex<double> **evec_out, const bool require_evec) const
{
    unsigned int i, j;
    char JOBZ;
    int INFO;
    double *RWORK;
    std::complex<double> *amat;
    std::complex<double> *WORK;

    int ns = dynamical->neval;

    int LWORK = (2 * ns - 1) * 10;
    allocate(RWORK, 3 * ns - 2);
    allocate(WORK, LWORK);

    if (require_evec) {
        JOBZ = 'V';
    } else {
        JOBZ = 'N';
    }

    char UPLO = 'U';

    allocate(amat, ns * ns);

    unsigned int k = 0;
    for (j = 0; j < ns; ++j) {
        for (i = 0; i < ns; ++i) {
            amat[k++] = mat_in[i][j];
        }
    }

    zheev_(&JOBZ, &UPLO, &ns, amat, &ns, eval_out, WORK, &LWORK, RWORK, &INFO);
    if (INFO != 0) {
        exit("Dynamical::diagonalize_interpolated_matrix",
             "zheev failed to diagonalize the interpolated matrix (INFO != 0).");
    }

    k = 0;

    if (require_evec) {
        // Here we transpose the matrix evec_out so that
        // evec_out[i] becomes phonon eigenvector of i-th mode.
        for (j = 0; j < ns; ++j) {
            for (i = 0; i < ns; ++i) {
                evec_out[j][i] = amat[k++];
            }
        }
    }

    deallocate(amat);
    deallocate(WORK);
    deallocate(RWORK);
}


void Dynamical::calc_new_dymat_with_evec(std::complex<double> ***dymat_out, double **omega2_in,
                                         std::complex<double> ***evec_in, const KpointMeshUniform *kmesh_coarse,
                                         const std::vector<int> &kmap_interpolate_to_scph)
{
    std::complex<double> *polarization_matrix, *mat_tmp;
    std::complex<double> *eigval_matrix, *dmat;
    std::complex<double> *beta;
    std::complex<double> ***dymat_q, **dymat_harmonic;
    std::complex<double> im(0.0, 1.0);

    unsigned int ik, is, js;
    int ns = neval;

    const unsigned int ns2 = ns * ns;

    auto alpha = std::complex<double>(1.0, 0.0);

    char TRANSA[] = "N";
    char TRANSB[] = "C";

    allocate(polarization_matrix, ns2);
    allocate(mat_tmp, ns2);
    allocate(eigval_matrix, ns2);
    allocate(beta, ns);
    allocate(dmat, ns2);
    allocate(dymat_q, ns, ns, kmesh_coarse->nk);
    allocate(dymat_harmonic, ns, ns);

    for (is = 0; is < ns; ++is)
        beta[is] = std::complex<double>(0.0, 0.0);

    for (ik = 0; ik < kmesh_coarse->nk; ++ik) {

        const auto knum = kmap_interpolate_to_scph[ik];

        // create eigval matrix

        for (is = 0; is < ns2; ++is)
            eigval_matrix[is] = std::complex<double>(0.0, 0.0);

        unsigned int m = 0;
        for (is = 0; is < ns; ++is) {
            for (js = 0; js < ns; ++js) {
                if (is == js) {
                    eigval_matrix[m] = omega2_in[knum][is];
                }
                ++m;
            }
        }

        // create polarization matrix

        m = 0;

        for (is = 0; is < ns; ++is) {
            for (js = 0; js < ns; ++js) {
                polarization_matrix[m++] = evec_in[knum][is][js];
            }
        }

        zgemm_cpx(TRANSA, TRANSB, &ns, &ns, &ns, &alpha, eigval_matrix, &ns, polarization_matrix, &ns, beta, mat_tmp, &ns);
        zgemm_cpx(TRANSA, TRANSA, &ns, &ns, &ns, &alpha, polarization_matrix, &ns, mat_tmp, &ns, beta, dmat, &ns);

        m = 0;

        for (js = 0; js < ns; ++js) {
            for (is = 0; is < ns; ++is) {
                dymat_q[is][js][ik] = dmat[m];
                ++m;
            }
        }


        // Subtract harmonic contribution
        dynamical->calc_analytic_k(kmesh_coarse->xk[ik], fcs_phonon->force_constant_with_cell[0], dymat_harmonic);

        for (is = 0; is < ns; ++is) {
            for (js = 0; js < ns; ++js) {
                dymat_q[is][js][ik] -= dymat_harmonic[is][js];
            }
        }
    }

    deallocate(polarization_matrix);
    deallocate(mat_tmp);
    deallocate(eigval_matrix);
    deallocate(beta);
    deallocate(dmat);
    deallocate(dymat_harmonic);

    const auto nk1 = kmesh_coarse->nk_i[0];
    const auto nk2 = kmesh_coarse->nk_i[1];
    const auto nk3 = kmesh_coarse->nk_i[2];

    std::vector<std::vector<double>> xk_dup;

    int icell = 0;

    for (int ix = 0; ix < nk1; ++ix) {
        for (int iy = 0; iy < nk2; ++iy) {
            for (int iz = 0; iz < nk3; ++iz) {

                for (is = 0; is < ns; ++is) {
                    for (js = 0; js < ns; ++js) {
                        dymat_out[is][js][icell] = std::complex<double>(0.0, 0.0);
                    }
                }

                for (ik = 0; ik < kmesh_coarse->nk; ++ik) {

                    duplicate_xk_boundary(kmesh_coarse->xk[ik], xk_dup);

                    auto cexp_phase = std::complex<double>(0.0, 0.0);

                    for (const auto &i: xk_dup) {

                        auto phase = 2.0 * pi *
                                     (i[0] * static_cast<double>(ix) + i[1] * static_cast<double>(iy) +
                                      i[2] * static_cast<double>(iz));
                        cexp_phase += std::exp(-im * phase);
                    }
                    cexp_phase /= static_cast<double>(xk_dup.size());

                    for (is = 0; is < ns; ++is) {
                        for (js = 0; js < ns; ++js) {
                            dymat_out[is][js][icell] += dymat_q[is][js][ik] * cexp_phase;
                        }
                    }
                }
                for (is = 0; is < ns; ++is) {
                    for (js = 0; js < ns; ++js) {
                        dymat_out[is][js][icell] /= static_cast<double>(kmesh_coarse->nk);
                    }
                }

                ++icell;
            }
        }
    }

    deallocate(dymat_q);
}

void Dynamical::duplicate_xk_boundary(double *xk_in, std::vector<std::vector<double>> &vec_xk)
{
    int i;
    int n[3];
    double sign[3];
    std::vector<double> vec_tmp;

    vec_xk.clear();

    for (i = 0; i < 3; ++i) {
        if (std::abs(std::abs(xk_in[i]) - 0.5) < eps) {
            n[i] = 2;
        } else {
            n[i] = 1;
        }
    }

    for (i = 0; i < n[0]; ++i) {
        sign[0] = 1.0 - 2.0 * static_cast<double>(i);
        for (int j = 0; j < n[1]; ++j) {
            sign[1] = 1.0 - 2.0 * static_cast<double>(j);
            for (int k = 0; k < n[2]; ++k) {
                sign[2] = 1.0 - 2.0 * static_cast<double>(k);

                vec_tmp.clear();
                for (int l = 0; l < 3; ++l) {
                    vec_tmp.push_back(sign[l] * xk_in[l]);
                }
                vec_xk.push_back(vec_tmp);
            }
        }
    }
}

void Dynamical::get_symmetry_gamma_dynamical(KpointMeshUniform *kmesh_in, const unsigned int natmin_in,
                                             const Eigen::MatrixXd &x_fractional_in,
                                             const std::vector<SymmetryOperationWithMapping> &symmlist,
                                             std::complex<double> ****&mat_transform_sym) const
{
    // Construct the transformation matrix for the dynamical matrix.

    unsigned int ik;
    unsigned int is, js;
    unsigned int icrd, jcrd;
    double x1[3], x2[3], k[3], xtmp[3];
    double S_cart[3][3], S_frac[3][3], S_frac_inv[3][3];
    std::complex<double> **gamma_tmp;

    const auto natmin = natmin_in;
    const auto ns = neval;
    const auto nk_irred_interpolate = kmesh_in->nk_irred;

    const auto nsym = symmlist.size();

    allocate(gamma_tmp, ns, ns);

    if (mat_transform_sym) deallocate(mat_transform_sym);
    allocate(mat_transform_sym, nk_irred_interpolate, nsym, ns, ns);

    for (ik = 0; ik < nk_irred_interpolate; ++ik) {

        const auto knum = kmesh_in->kpoint_irred_all[ik][0].knum;
        for (icrd = 0; icrd < 3; ++icrd) {
            k[icrd] = kmesh_in->xk[knum][icrd];
        }

        unsigned int isym = 0;

        for (const auto &it: symmlist) {

            for (icrd = 0; icrd < 3; ++icrd) {
                for (jcrd = 0; jcrd < 3; ++jcrd) {
                    S_cart[icrd][jcrd] = it.rot[3 * icrd + jcrd];
                    S_frac[icrd][jcrd] = it.rot_real[3 * icrd + jcrd];
                }
            }

            invmat3(S_frac_inv, S_frac);

            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns; ++js) {
                    gamma_tmp[is][js] = std::complex<double>(0.0, 0.0);
                }
            }

            for (unsigned int jat = 0; jat < natmin; ++jat) {
                const auto iat = it.mapping[jat];

                // Fractional coordinates of x1 and x2
                for (icrd = 0; icrd < 3; ++icrd) {
                    x1[icrd] = x_fractional_in(iat, icrd);
                    x2[icrd] = x_fractional_in(jat, icrd);
                }

                rotvec(xtmp, x1, S_frac_inv);
                for (icrd = 0; icrd < 3; ++icrd) {
                    xtmp[icrd] = xtmp[icrd] - x2[icrd];
                }

                auto phase = 2.0 * pi * (k[0] * xtmp[0] + k[1] * xtmp[1] + k[2] * xtmp[2]);

                for (icrd = 0; icrd < 3; ++icrd) {
                    for (jcrd = 0; jcrd < 3; ++jcrd) {
                        gamma_tmp[3 * iat + icrd][3 * jat + jcrd] = S_cart[icrd][jcrd] * std::exp(im * phase);
                    }
                }
            }

            for (is = 0; is < ns; ++is) {
                for (js = 0; js < ns; ++js) {
                    mat_transform_sym[ik][isym][is][js] = gamma_tmp[is][js];
                }
            }

            ++isym;
        }
    }

    deallocate(gamma_tmp);
}


double **Dynamical::get_xrs_image() const
{
    return xshift_s;
}
