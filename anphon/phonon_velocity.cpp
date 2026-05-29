/*
phonon_velocity.cpp

Copyright (c) 2014, 2015, 2016 Terumasa Tadano

This file is distributed under the terms of the MIT license.
Please see the file 'LICENCE.txt' in the root directory 
or http://opensource.org/licenses/mit-license.php for information.
*/

#include "phonon_velocity.h"
#include <complex>
#include <iomanip>
#include "constants.h"
#include "dynamical.h"
#include "error.h"
#include "ewald.h"
#include "fcs_phonon.h"
#include "kpoint.h"
#include "mathfunctions.h"
#include "memory.h"
#include "mpi_common.h"
#include "phonon_dos.h"
#include "system.h"

using namespace PHON_NS;

PhononVelocity::PhononVelocity(PHON *phon) : Pointers(phon)
{
    set_default_variables();
}

PhononVelocity::~PhononVelocity()
{
    deallocate_variables();
}

void PhononVelocity::set_default_variables()
{
    print_velocity = false;
    velmat = nullptr;

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
}

void PhononVelocity::deallocate_variables()
{
    if (xshift_s) {
        deallocate(xshift_s);
    }
    if (velmat) {
        deallocate(velmat);
    }
}

void PhononVelocity::setup_velocity()
{
    MPI_Bcast(&print_velocity, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
}

void PhononVelocity::get_phonon_group_velocity_bandstructure(const KpointBandStructure *kpoint_bs_in,
                                                             const Eigen::Matrix3d &lavec_p,
                                                             const Eigen::Matrix3d &rlavec_p,
                                                             const std::vector<FcsArrayWithCell> &fc2_in,
                                                             const std::vector<FcsArrayWithCell> &fc2_without_dipole,
                                                             double **phvel_out) const
{
    unsigned int i;
    unsigned int idiff;
    const auto nk = kpoint_bs_in->nk;
    const auto n = dynamical->neval;
    double **xk_shift;
    double *xk_tmp;
    double **omega_shift, *omega_tmp;

    const auto h = 1.0e-4;

    std::complex<double> **evec_tmp;

    allocate(evec_tmp, 1, 1);

    //    if (mympi->my_rank == 0) {
    //        std::cout << " Calculating group velocities of phonon along given k path ... ";
    //    }

    const unsigned int ndiff = 2;
    allocate(xk_shift, ndiff, 3);
    allocate(omega_shift, ndiff, n);
    allocate(omega_tmp, ndiff);

    allocate(xk_tmp, 3);

    for (unsigned int ik = 0; ik < nk; ++ik) {

        // Represent the given kpoint in Cartesian coordinate
        rotvec(xk_tmp, kpoint_bs_in->xk[ik], rlavec_p, 'T');

        //        if (ndiff == 2) {
        // central difference
        // f'(x) =~ f(x+h)-f(x-h)/2h

        for (i = 0; i < 3; ++i) {
            xk_shift[0][i] = xk_tmp[i] - h * kpoint_bs_in->kvec_na[ik][i];
            xk_shift[1][i] = xk_tmp[i] + h * kpoint_bs_in->kvec_na[ik][i];
        }

        //        } else {
        //            exit("get_phonon_group_velocity_bandstructure",
        //                        "ndiff > 2 is not supported yet.");
        //        }

        for (idiff = 0; idiff < ndiff; ++idiff) {

            // Move back to fractional basis

            rotvec(xk_shift[idiff], xk_shift[idiff], lavec_p, 'T');
            for (i = 0; i < 3; ++i)
                xk_shift[idiff][i] /= 2.0 * pi;

            if (dynamical->nonanalytic == 3) {
                dynamical->eval_k_ewald(xk_shift[idiff],
                                        kpoint_bs_in->kvec_na[ik],
                                        fc2_without_dipole,
                                        omega_shift[idiff],
                                        evec_tmp,
                                        false);
            } else {
                dynamical
                    ->eval_k(xk_shift[idiff], kpoint_bs_in->kvec_na[ik], fc2_in, omega_shift[idiff], evec_tmp, false);
            }
        }

        for (i = 0; i < n; ++i) {
            for (idiff = 0; idiff < ndiff; ++idiff) {
                omega_tmp[idiff] = dynamical->freq(omega_shift[idiff][i]);
            }
            phvel_out[ik][i] = diff(omega_tmp, ndiff, h);
        }
    }
    deallocate(omega_tmp);
    deallocate(omega_shift);
    deallocate(xk_shift);
    deallocate(xk_tmp);

    deallocate(evec_tmp);

    //    if (mympi->my_rank == 0) {
    //        std::cout << "done!" << '\n';
    //    }
}

void PhononVelocity::get_phonon_group_velocity_mesh(const KpointMeshUniform &kmesh_in, const Eigen::Matrix3d &lavec_p,
                                                    const bool irreducible_only, double ***phvel3_out) const
{
    // This routine computes the group velocities for the given uniform k mesh.
    const auto nk = kmesh_in.nk;
    const auto nk_irred = kmesh_in.nk_irred;
    const auto ns = dynamical->neval;

    double **vel;

    allocate(vel, ns, 3);

    if (irreducible_only) {
        for (unsigned int i = 0; i < nk_irred; ++i) {
            phonon_vel_k(&kmesh_in.xk[kmesh_in.kpoint_irred_all[i][0].knum][0], vel);

            for (unsigned int j = 0; j < ns; ++j) {
                rotvec(vel[j], vel[j], lavec_p);
                for (unsigned int k = 0; k < 3; ++k) {
                    vel[j][k] /= 2.0 * pi;
                    phvel3_out[i][j][k] = vel[j][k];
                }
            }
        }
    } else {
        for (unsigned int i = 0; i < nk; ++i) {
            phonon_vel_k(&kmesh_in.xk[i][0], vel);

            for (unsigned int j = 0; j < ns; ++j) {
                rotvec(vel[j], vel[j], lavec_p);
                for (unsigned int k = 0; k < 3; ++k) {
                    vel[j][k] /= 2.0 * pi;
                    phvel3_out[i][j][k] = vel[j][k];
                }
            }
        }
    }
    deallocate(vel);
}

void PhononVelocity::get_phonon_group_velocity_mesh_mpi(const KpointMeshUniform &kmesh_in,
                                                        const Eigen::Matrix3d &lavec_p, double ***phvel3_out) const
{
    // This routine computes the group velocities for the given uniform k mesh
    // using MPI parallelization.
    const auto nk = kmesh_in.nk;
    const auto ns = dynamical->neval;

    double **vel;
    double ***phvel3_loc = nullptr;
    int *displs = nullptr;
    int *sendcount = nullptr;
    int *recvcount = nullptr;
    std::vector<int> nk_proc;
    std::vector<int> ik_begin_proc, ik_end_proc;

    //    if (mympi->my_rank == 0) {
    //        std::cout << " Calculating group velocities of phonons on uniform grid ... ";
    //    }

    allocate(sendcount, mympi->nprocs);
    allocate(recvcount, mympi->nprocs);
    nk_proc.resize(mympi->nprocs);

    auto nk_loc = nk / mympi->nprocs;
    auto nk_res = nk - nk_loc * mympi->nprocs;

    for (auto i = 0; i < mympi->nprocs; ++i) {
        nk_proc[i] = nk_loc;
        if (i < nk_res) ++nk_proc[i];
        sendcount[i] = 3 * ns * nk_proc[i];
        recvcount[i] = sendcount[i];
    }

    if (mympi->my_rank == 0) {
        allocate(displs, mympi->nprocs);
        displs[0] = 0;
        for (auto i = 1; i < mympi->nprocs; ++i) {
            displs[i] = displs[i - 1] + recvcount[i - 1];
        }
    }

    ik_begin_proc.resize(mympi->nprocs);
    ik_end_proc.resize(mympi->nprocs);
    ik_begin_proc[0] = 0;
    ik_end_proc[0] = nk_proc[0];
    for (auto i = 1; i < mympi->nprocs; ++i) {
        ik_begin_proc[i] = ik_end_proc[i - 1];
        ik_end_proc[i] = ik_begin_proc[i] + nk_proc[i];
    }

    std::vector<int> klist_proc;
    for (auto ik = ik_begin_proc[mympi->my_rank]; ik < ik_end_proc[mympi->my_rank]; ++ik) {
        klist_proc.push_back(ik);
    }

    nk_loc = klist_proc.size();

    allocate(phvel3_loc, nk_loc, ns, 3);
    allocate(vel, ns, 3);

    for (unsigned int i = 0; i < nk_loc; ++i) {
        phonon_vel_k(&kmesh_in.xk[klist_proc[i]][0], vel);
        //        phonon_vel_k2(kpoint->xk[i],
        //                      dynamical->eval_phonon[i],
        //                      dynamical->evec_phonon[i],
        //                      vel);

        for (unsigned int j = 0; j < ns; ++j) {
            rotvec(vel[j], vel[j], lavec_p);
            for (unsigned int k = 0; k < 3; ++k) {
                vel[j][k] /= 2.0 * pi;
                phvel3_loc[i][j][k] = vel[j][k];
            }
        }
    }

    deallocate(vel);

    MPI_Gatherv(nk_loc > 0 ? &phvel3_loc[0][0][0] : nullptr,
                sendcount[mympi->my_rank],
                MPI_DOUBLE,
                mympi->my_rank == 0 ? &phvel3_out[0][0][0] : nullptr,
                mympi->my_rank == 0 ? &recvcount[0] : nullptr,
                mympi->my_rank == 0 ? &displs[0] : nullptr,
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

    deallocate(phvel3_loc);
    deallocate(sendcount);
    deallocate(recvcount);
    if (displs) deallocate(displs);

    //    if (mympi->my_rank == 0) {
    //        std::cout << "done!" << '\n';
    //    }
}

void PhononVelocity::calc_phonon_velmat_mesh(std::complex<double> ****velmat_out) const
{
    const auto nk = dos->kmesh_dos->nk;
    const auto ns = dynamical->neval;

    std::complex<double> ****velmat_loc = nullptr;
    int *displs = nullptr;
    int *sendcount = nullptr;
    int *recvcount = nullptr;
    std::vector<int> nk_proc;
    std::vector<int> ik_begin_proc, ik_end_proc;

    const auto factor = Bohr_in_Angstrom * 1.0e-10 / (time_ry * 2.0 * pi);

    if (mympi->my_rank == 0) {
        std::cout << " Calculating group velocity matrix of phonons on uniform grid ... ";
    }

    allocate(sendcount, mympi->nprocs);
    allocate(recvcount, mympi->nprocs);
    nk_proc.resize(mympi->nprocs);

    auto nk_loc = nk / mympi->nprocs;
    auto nk_res = nk - nk_loc * mympi->nprocs;

    for (auto i = 0; i < mympi->nprocs; ++i) {
        nk_proc[i] = nk_loc;
        if (i < nk_res) ++nk_proc[i];
        sendcount[i] = 3 * ns * ns * nk_proc[i];
        recvcount[i] = sendcount[i];
    }

    if (mympi->my_rank == 0) {
        allocate(displs, mympi->nprocs);
        displs[0] = 0;
        for (auto i = 1; i < mympi->nprocs; ++i) {
            displs[i] = displs[i - 1] + recvcount[i - 1];
        }
    }

    ik_begin_proc.resize(mympi->nprocs);
    ik_end_proc.resize(mympi->nprocs);
    ik_begin_proc[0] = 0;
    ik_end_proc[0] = nk_proc[0];
    for (auto i = 1; i < mympi->nprocs; ++i) {
        ik_begin_proc[i] = ik_end_proc[i - 1];
        ik_end_proc[i] = ik_begin_proc[i] + nk_proc[i];
    }

    std::vector<int> klist_proc;
    for (auto ik = ik_begin_proc[mympi->my_rank]; ik < ik_end_proc[mympi->my_rank]; ++ik) {
        klist_proc.push_back(ik);
    }

    nk_loc = klist_proc.size();

    allocate(velmat_loc, nk_loc, ns, ns, 3);

    for (auto i = 0; i < nk_loc; ++i) {
        auto knum = klist_proc[i];
        velocity_matrix_analytic(dos->kmesh_dos->xk[knum],
                                 fcs_phonon->force_constant_with_cell[0],
                                 dos->dymat_dos->get_eigenvalues()[knum],
                                 dos->dymat_dos->get_eigenvectors()[knum],
                                 velmat_loc[i]);

        double symmetrizer_k[3][3];
        std::vector<int> smallgroup_k;
        kpoint->get_symmetrization_matrix_at_k(dos->kmesh_dos->xk[knum], smallgroup_k, symmetrizer_k);

        for (auto j = 0; j < ns; ++j) {
            for (auto k = 0; k < ns; ++k) {
                rotvec(velmat_loc[i][j][k], velmat_loc[i][j][k], symmetrizer_k, 'T');
                rotvec(velmat_loc[i][j][k], velmat_loc[i][j][k], system->get_primcell().lattice_vector);
                for (auto mu = 0; mu < 3; ++mu) {
                    velmat_loc[i][j][k][mu] *= factor;
                }
            }
        }

        //        std::cout << "k = " << i << '\n';
        //        std::cout << dos->kmesh_dos->xk[i][0] << "  " << dos->kmesh_dos->xk[i][1] << " " << dos->kmesh_dos->xk[i][2] << '\n';
        //        for (auto mu = 0; mu < 3; ++mu) {
        //            std::cout << "mu = " << mu << '\n';
        //
        //            std::cout << "Diagonal:\n";
        //
        //            for (auto j = 0; j < ns; ++j) {
        //                std::cout << std::setw(20) << velmat_loc[i][j][j][mu] << '\n';
        //            }
        //
        //            std::cout << "Full:\n";
        //            for (auto j = 0; j < ns; ++j) {
        //                for (auto k = 0; k < ns; ++k) {
        //                    std::cout << std::setw(20) << velmat_loc[i][j][k][mu].real()
        //                                << std::setw(15) << velmat_loc[i][j][k][mu].imag();
        //                }
        //                std::cout << '\n';
        //            }
        //            std::cout << '\n';
        //        }
        //        std::cout << '\n';
    }

#ifdef MPI_CXX_DOUBLE_COMPLEX
    const auto mpi_complex_type = MPI_CXX_DOUBLE_COMPLEX;
#else
    const auto mpi_complex_type = MPI_COMPLEX16;
#endif

    MPI_Gatherv(nk_loc > 0 ? &velmat_loc[0][0][0][0] : nullptr,
                sendcount[mympi->my_rank],
                mpi_complex_type,
                mympi->my_rank == 0 ? &velmat_out[0][0][0][0] : nullptr,
                mympi->my_rank == 0 ? &recvcount[0] : nullptr,
                mympi->my_rank == 0 ? &displs[0] : nullptr,
                mpi_complex_type,
                0,
                MPI_COMM_WORLD);

    deallocate(velmat_loc);
    deallocate(sendcount);
    deallocate(recvcount);
    if (displs) deallocate(displs);

    if (mympi->my_rank == 0) {
        std::cout << "done!\n";
    }
}

void PhononVelocity::phonon_vel_k(const double *xk_in, double **vel_out) const
{
    unsigned int j;
    unsigned int idiff;
    const auto n = dynamical->neval;
    double **xk_shift;
    std::complex<double> **evec_tmp;
    double **omega_shift, *omega_tmp;
    double **kvec_na_tmp;
    const auto h = 1.0e-4;

    const unsigned int ndiff = 2;

    allocate(omega_shift, ndiff, n);
    allocate(xk_shift, ndiff, 3);
    allocate(omega_tmp, ndiff);
    allocate(evec_tmp, 1, 1);
    allocate(kvec_na_tmp, 2, 3);

    for (unsigned int i = 0; i < 3; ++i) {

        for (j = 0; j < 3; ++j) {
            xk_shift[0][j] = xk_in[j];
            xk_shift[1][j] = xk_in[j];
        }

        xk_shift[0][i] -= h;
        xk_shift[1][i] += h;

        // kvec_na_tmp for nonalaytic term
        for (j = 0; j < 3; ++j) {
            kvec_na_tmp[0][j] = xk_shift[0][j];
            kvec_na_tmp[1][j] = xk_shift[1][j];
        }
        rotvec(kvec_na_tmp[0], kvec_na_tmp[0], system->get_primcell().reciprocal_lattice_vector, 'T');
        rotvec(kvec_na_tmp[1], kvec_na_tmp[1], system->get_primcell().reciprocal_lattice_vector, 'T');

        auto norm = std::sqrt(kvec_na_tmp[0][0] * kvec_na_tmp[0][0] + kvec_na_tmp[0][1] * kvec_na_tmp[0][1] +
                              kvec_na_tmp[0][2] * kvec_na_tmp[0][2]);

        if (norm > eps) {
            for (j = 0; j < 3; ++j)
                kvec_na_tmp[0][j] /= norm;
        }
        norm = std::sqrt(kvec_na_tmp[1][0] * kvec_na_tmp[1][0] + kvec_na_tmp[1][1] * kvec_na_tmp[1][1] +
                         kvec_na_tmp[1][2] * kvec_na_tmp[1][2]);

        if (norm > eps) {
            for (j = 0; j < 3; ++j)
                kvec_na_tmp[1][j] /= norm;
        }

        for (idiff = 0; idiff < ndiff; ++idiff) {

            if (dynamical->nonanalytic == 3) {
                dynamical->eval_k_ewald(xk_shift[idiff],
                                        kvec_na_tmp[idiff],
                                        ewald->fc2_without_dipole,
                                        omega_shift[idiff],
                                        evec_tmp,
                                        false);
            } else {
                dynamical->eval_k(xk_shift[idiff],
                                  kvec_na_tmp[idiff],
                                  fcs_phonon->force_constant_with_cell[0],
                                  omega_shift[idiff],
                                  evec_tmp,
                                  false);
            }
        }

        for (j = 0; j < n; ++j) {
            for (idiff = 0; idiff < ndiff; ++idiff) {
                omega_tmp[idiff] = dynamical->freq(omega_shift[idiff][j]);
            }
            vel_out[j][i] = diff(omega_tmp, ndiff, h);
        }
    }

    deallocate(xk_shift);
    deallocate(omega_shift);
    deallocate(omega_tmp);
    deallocate(evec_tmp);
    deallocate(kvec_na_tmp);
}

double PhononVelocity::diff(const double *f, const unsigned int n, const double h) const
{
    auto df = 0.0;

    if (n == 2) {
        df = (f[1] - f[0]) / (2.0 * h);
    } else {
        exit("diff", "Numerical differentiation of n > 2 is not supported yet.");
    }

    return df;
}

void PhononVelocity::phonon_vel_k2(const double *xk_in, const double *omega_in, std::complex<double> **evec_in,
                                   double **vel_out) const
{
    unsigned int i, j, l, m;
    unsigned int icrd;
    const auto nmode = 3 * system->get_primcell().number_of_atoms;

    std::complex<double> ***ddyn;
    std::complex<double> ctmp;
    std::complex<double> **vel_tmp;
    std::complex<double> ***mat_tmp;
    std::complex<double> czero(0.0, 0.0);
    std::vector<int> smallgroup_k;
    double **eval_tmp;

    if (dynamical->nonanalytic) {
        exit("phonon_vel_k2",
             "Sorry. Analytic calculation of "
             "group velocity is not supported for NONANALYTIC>0.");
    }

    allocate(ddyn, 3, nmode, nmode);
    allocate(vel_tmp, 3, nmode);
    calc_derivative_dynmat_k(xk_in, fcs_phonon->force_constant_with_cell[0], ddyn);

    const auto do_diagonalize = false;

    if (do_diagonalize) {
        // Detect degeneracy at the given k
        double tol_omega = 1.0e-7; // Approximately equal to 0.01 cm^{-1}

        std::vector<int> degeneracy_at_k;

        degeneracy_at_k.clear();

        double omega_prev = omega_in[0];
        int ideg = 1;

        for (i = 1; i < nmode; ++i) {
            double omega_now = omega_in[i];

            if (std::abs(omega_now - omega_prev) < tol_omega) {
                ++ideg;
            } else {
                degeneracy_at_k.push_back(ideg);
                ideg = 1;
                omega_prev = omega_now;
            }
        }
        degeneracy_at_k.push_back(ideg);

        int is = 0;

        for (i = 0; i < degeneracy_at_k.size(); ++i) {
            ideg = degeneracy_at_k[i];

            if (ideg == 1) {

                // When the branch is non-degenerate, the velocity can be calculated
                // from the diagonal element of e^{*} * DDYN * e.

                for (icrd = 0; icrd < 3; ++icrd) {
                    vel_tmp[icrd][is] = czero;

                    for (l = 0; l < nmode; ++l) {
                        ctmp = czero;
                        for (m = 0; m < nmode; ++m) {
                            ctmp += ddyn[icrd][l][m] * evec_in[is][m];
                        }
                        vel_tmp[icrd][is] += std::conj(evec_in[is][l]) * ctmp;
                    }
                    vel_tmp[icrd][is] /= 2.0 * omega_in[is];
                }

            } else if (ideg > 1) {

                // When the branch is degenerated with two or more branches,
                // we have to construct a MxM matrix and diagonalize it to obtain
                // group velocities.

                allocate(mat_tmp, 3, ideg, ideg);
                allocate(eval_tmp, 3, ideg);

                for (icrd = 0; icrd < 3; ++icrd) {

                    for (j = 0; j < ideg; ++j) {
                        for (unsigned int k = 0; k < ideg; ++k) {
                            mat_tmp[icrd][j][k] = czero;

                            for (l = 0; l < nmode; ++l) {
                                ctmp = czero;
                                for (m = 0; m < nmode; ++m) {
                                    ctmp += ddyn[icrd][l][m] * evec_in[j + is][m];
                                }
                                mat_tmp[icrd][j][k] += std::conj(evec_in[k + is][l]) * ctmp;
                            }
                        }
                    }
                    // Diagonalize the matrix here

                    diagonalize_hermite_mat(ideg, mat_tmp[icrd], eval_tmp[icrd]);

                    for (j = 0; j < ideg; ++j) {
                        vel_tmp[icrd][j + is] = eval_tmp[icrd][j] / (2.0 * omega_in[j + is]);
                    }
                }

                deallocate(mat_tmp);
                deallocate(eval_tmp);

            } else {
                exit("phonon_vel_k2", "This cannot happen.");
            }

            is += ideg;
        }
    } else {

        for (icrd = 0; icrd < 3; ++icrd) {

            for (j = 0; j < nmode; ++j) {
                vel_tmp[icrd][j] = czero;

                for (l = 0; l < nmode; ++l) {
                    ctmp = czero;
                    for (m = 0; m < nmode; ++m) {
                        ctmp += ddyn[icrd][l][m] * evec_in[j][m];
                    }
                    vel_tmp[icrd][j] += std::conj(evec_in[j][l]) * ctmp;
                }
            }
            for (j = 0; j < nmode; ++j) {
                vel_tmp[icrd][j] /= 2.0 * omega_in[j];
            }
        }
    }

    for (icrd = 0; icrd < 3; ++icrd) {
        for (i = 0; i < nmode; ++i) {
            vel_out[i][icrd] = vel_tmp[icrd][i].real();
        }
    }

    if (ddyn) {
        deallocate(ddyn);
    }
    if (vel_tmp) {
        deallocate(vel_tmp);
    }

    double symmetrizer_k[3][3];

    kpoint->get_symmetrization_matrix_at_k(xk_in, smallgroup_k, symmetrizer_k);

    // std::cout << "symmetrizer_k" << '\n';
    // for (i = 0; i < 3; ++i) {
    //     for (j = 0; j < 3; ++j) {
    //         std::cout << std::setw(15) << symmetrizer_k[i][j];
    //     }
    //     std::cout << '\n';
    // }
    // std::cout << '\n';

    for (i = 0; i < nmode; ++i) {
        rotvec(vel_out[i], vel_out[i], symmetrizer_k, 'T');
    }
}

void PhononVelocity::calc_derivative_dynmat_k(const double *xk_in, const std::vector<FcsArrayWithCell> &fc2_in,
                                              std::complex<double> ***ddyn_out) const
{
    unsigned int i, j, k;

    const auto nmode = dynamical->neval;

    for (k = 0; k < 3; ++k) {
        for (i = 0; i < nmode; ++i) {
            for (j = 0; j < nmode; ++j) {
                ddyn_out[k][i][j] = std::complex<double>(0.0, 0.0);
            }
        }
    }

    const auto invsqrt_mass = system->get_invsqrt_mass();

    for (const auto &it: fc2_in) {

        const auto phase =
            tpi * (it.relvecs[0][0] * xk_in[0] + it.relvecs[0][1] * xk_in[1] + it.relvecs[0][2] * xk_in[2]);

        for (k = 0; k < 3; ++k) {
            // For the diagonal components, this should be fine,
            // whereas it.relvecs_vel should be used for computing the off diagonal elements.
            ddyn_out[k][it.pairs[0].index][it.pairs[1].index] +=
                it.fcs_val * std::exp(im * phase) * tpi * it.relvecs[0][k] * invsqrt_mass[it.pairs[0].index / 3] *
                invsqrt_mass[it.pairs[1].index / 3];
        }
    }

    for (k = 0; k < 3; ++k) {
        for (i = 0; i < nmode; ++i) {
            for (j = 0; j < nmode; ++j) {
                ddyn_out[k][i][j] *= std::complex<double>(0.0, 1.0);
            }
        }
    }
}

void PhononVelocity::diagonalize_hermite_mat(const int n, std::complex<double> **mat_in, double *eval_out) const
{
    std::complex<double> *mat_1D;
    int LWORK = (2 * n - 1) * 10;
    int INFO;
    std::complex<double> *WORK;
    double *RWORK;
    char JOBZ = 'N';
    char UPLO = 'U';
    int n_ = n;

    allocate(mat_1D, n * n);
    allocate(RWORK, 3 * n - 2);
    allocate(WORK, LWORK);

    int k = 0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            mat_1D[k++] = mat_in[i][j];
        }
    }

    zheev_(&JOBZ, &UPLO, &n_, mat_1D, &n_, eval_out, WORK, &LWORK, RWORK, &INFO);
    if (INFO != 0) {
        exit("PhononVelocity::diagonalize_hermite_mat",
             "zheev failed to diagonalize the Hermitian matrix (INFO != 0).");
    }

    deallocate(RWORK);
    deallocate(WORK);
    deallocate(mat_1D);
}

void PhononVelocity::velocity_matrix_analytic(const double *xk_in, const std::vector<FcsArrayWithCell> &fc2_in,
                                              const double *omega_in, std::complex<double> **evec_in,
                                              std::complex<double> ***velmat_out) const
{
    // Use Allen's definition
    // Only the analytic part of the dynamical matrix will be considered.
    // Non-analytic part must be treated seperately.

    unsigned int i, j, k;

    const auto nmode = dynamical->neval;

    std::complex<double> ***ddymat;

    allocate(ddymat, nmode, nmode, 3);

    for (i = 0; i < nmode; ++i) {
        for (j = 0; j < nmode; ++j) {
            for (k = 0; k < 3; ++k) {
                velmat_out[i][j][k] = std::complex<double>(0.0, 0.0);
                ddymat[i][j][k] = std::complex<double>(0.0, 0.0);
            }
        }
    }

    const auto invsqrt_mass = system->get_invsqrt_mass();

    for (const auto &it: fc2_in) {
        const auto phase =
            tpi * (it.relvecs[0][0] * xk_in[0] + it.relvecs[0][1] * xk_in[1] + it.relvecs[0][2] * xk_in[2]);

        for (k = 0; k < 3; ++k) {
            ddymat[it.pairs[0].index][it.pairs[1].index][k] +=
                it.fcs_val * std::exp(im * phase) * tpi * it.relvecs_velocity[0][k] *
                invsqrt_mass[it.pairs[0].index / 3] * invsqrt_mass[it.pairs[1].index / 3];
        }
    }

    unsigned int ii, jj;

    for (i = 0; i < nmode; ++i) {
        for (j = 0; j < nmode; ++j) {
            for (ii = 0; ii < nmode; ++ii) {
                for (jj = 0; jj < nmode; ++jj) {
                    for (k = 0; k < 3; ++k) {
                        velmat_out[i][j][k] += std::conj(evec_in[i][ii]) * ddymat[ii][jj][k] * evec_in[j][jj];
                    }
                }
            }
        }
    }

    for (i = 0; i < nmode; ++i) {
        for (j = 0; j < nmode; ++j) {
            for (k = 0; k < 3; ++k) {
                velmat_out[i][j][k] *= 0.5 * im / std::sqrt(omega_in[i] * omega_in[j]);
            }
        }
    }
}
