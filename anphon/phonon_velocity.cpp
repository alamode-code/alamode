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
#include "cell_shift_table.h"
#include "constants.h"
#include "dense_hermitian_eigen.h"
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
#include "write_phonons.h"

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

    build_27cell_shift_table(xshift_s);
}

void PhononVelocity::deallocate_variables()
{}

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
    NDArray<double, 2> xk_shift;
    NDArray<double, 1> xk_tmp;
    NDArray<double, 2> omega_shift;
    NDArray<double, 1> omega_tmp;

    const auto h = 1.0e-4;

    NDArray<std::complex<double>, 2> evec_tmp;

    evec_tmp.resize(1, 1);

    const unsigned int ndiff = 2;
    xk_shift.resize(ndiff, 3);
    omega_shift.resize(ndiff, n);
    omega_tmp.resize(ndiff);

    xk_tmp.resize(3);

    for (unsigned int ik = 0; ik < nk; ++ik) {

        // Represent the given kpoint in Cartesian coordinate
        rotvec(xk_tmp, kpoint_bs_in->xk[ik], rlavec_p, 'T');

        // central difference
        // f'(x) =~ f(x+h)-f(x-h)/2h
        for (i = 0; i < 3; ++i) {
            xk_shift[0][i] = xk_tmp[i] - h * kpoint_bs_in->kvec_na[ik][i];
            xk_shift[1][i] = xk_tmp[i] + h * kpoint_bs_in->kvec_na[ik][i];
        }

        for (idiff = 0; idiff < ndiff; ++idiff) {

            // Move back to fractional basis

            rotvec(xk_shift[idiff], xk_shift[idiff], lavec_p, 'T');
            for (i = 0; i < 3; ++i) xk_shift[idiff][i] /= 2.0 * pi;

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
    omega_tmp.clear();
    omega_shift.clear();
    xk_shift.clear();
    xk_tmp.clear();

    evec_tmp.clear();
}

void PhononVelocity::get_phonon_group_velocity_mesh(const KpointMeshUniform &kmesh_in, const Eigen::Matrix3d &lavec_p,
                                                    const bool irreducible_only, double ***phvel3_out) const
{
    // This routine computes the group velocities for the given uniform k mesh.
    const auto nk = kmesh_in.nk;
    const auto nk_irred = kmesh_in.nk_irred;
    const auto ns = dynamical->neval;

    NDArray<double, 2> vel;

    vel.resize(ns, 3);

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
    vel.clear();
}

void PhononVelocity::get_phonon_group_velocity_mesh_mpi(const KpointMeshUniform &kmesh_in,
                                                        const Eigen::Matrix3d &lavec_p, double ***phvel3_out) const
{
    // This routine computes the group velocities for the given uniform k mesh
    // using MPI parallelization.
    const auto nk = kmesh_in.nk;
    const auto ns = dynamical->neval;

    NDArray<double, 2> vel;
    NDArray<double, 3> phvel3_loc;
    NDArray<int, 1> displs;
    NDArray<int, 1> sendcount;
    NDArray<int, 1> recvcount;
    std::vector<int> nk_proc;
    std::vector<int> ik_begin_proc, ik_end_proc;

    sendcount.resize(mympi->nprocs);
    recvcount.resize(mympi->nprocs);
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
        displs.resize(mympi->nprocs);
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

    phvel3_loc.resize(nk_loc, ns, 3);
    vel.resize(ns, 3);

    for (unsigned int i = 0; i < nk_loc; ++i) {
        phonon_vel_k(&kmesh_in.xk[klist_proc[i]][0], vel);

        for (unsigned int j = 0; j < ns; ++j) {
            rotvec(vel[j], vel[j], lavec_p);
            for (unsigned int k = 0; k < 3; ++k) {
                vel[j][k] /= 2.0 * pi;
                phvel3_loc[i][j][k] = vel[j][k];
            }
        }
    }

    vel.clear();

    MPI_Gatherv(nk_loc > 0 ? &phvel3_loc[0][0][0] : nullptr,
                sendcount[mympi->my_rank],
                MPI_DOUBLE,
                mympi->my_rank == 0 ? &phvel3_out[0][0][0] : nullptr,
                mympi->my_rank == 0 ? &recvcount[0] : nullptr,
                mympi->my_rank == 0 ? &displs[0] : nullptr,
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

    phvel3_loc.clear();
    sendcount.clear();
    recvcount.clear();
    displs.clear();
}

void PhononVelocity::gather_group_velocities_mesh(const KpointMeshUniform &kmesh_in, const Eigen::Matrix3d &lavec_p,
                                                  NDArray<double, 3> &vel_out, const double unit_factor,
                                                  const bool bcast_full) const
{
    // Allocate-and-fill wrapper around get_phonon_group_velocity_mesh_mpi
    // shared by the RTA and IBTE setup paths. The gathered velocities live
    // on rank 0 only unless bcast_full is set; other ranks get a dummy
    // allocation. unit_factor is applied before the broadcast, so every
    // caller states its unit convention in one place (1.0 keeps atomic
    // units; Bohr_in_Angstrom * 1.0e-10 / time_ry converts to m/s). The
    // caller owns and deallocates vel_out.
    const auto nk = kmesh_in.nk;
    const auto neval = dynamical->neval;

    if (mympi->my_rank == 0 || bcast_full) {
        vel_out.resize(nk, neval, 3);
    } else {
        vel_out.resize(1, 1, 1);
    }

    get_phonon_group_velocity_mesh_mpi(kmesh_in, lavec_p, vel_out);

    if (mympi->my_rank == 0 && unit_factor != 1.0) {
        for (unsigned int i = 0; i < nk; ++i) {
            for (unsigned int j = 0; j < neval; ++j) {
                for (auto k = 0; k < 3; ++k) {
                    vel_out[i][j][k] *= unit_factor;
                }
            }
        }
    }

    if (bcast_full) {
        MPI_Bcast(&vel_out[0][0][0], static_cast<int>(nk * neval * 3), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
}

void PhononVelocity::calc_phonon_velmat_mesh(std::complex<double> ****velmat_out) const
{
    const auto nk = dos->kmesh_dos->nk;
    const auto ns = dynamical->neval;

    NDArray<std::complex<double>, 4> velmat_loc;
    NDArray<int, 1> displs;
    NDArray<int, 1> sendcount;
    NDArray<int, 1> recvcount;
    std::vector<int> nk_proc;
    std::vector<int> ik_begin_proc, ik_end_proc;

    const auto factor = Bohr_in_Angstrom * 1.0e-10 / (time_ry * 2.0 * pi);

    if (mympi->my_rank == 0 && writes->getVerbosity() > 0) {
        std::cout << " Calculating group velocity matrix of phonons on uniform grid ... ";
    }

    sendcount.resize(mympi->nprocs);
    recvcount.resize(mympi->nprocs);
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
        displs.resize(mympi->nprocs);
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

    velmat_loc.resize(nk_loc, ns, ns, 3);

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

    velmat_loc.clear();
    sendcount.clear();
    recvcount.clear();
    displs.clear();

    if (mympi->my_rank == 0 && writes->getVerbosity() > 0) {
        std::cout << "done!\n";
    }
}

void PhononVelocity::phonon_vel_k(const double *xk_in, double **vel_out) const
{
    unsigned int j;
    unsigned int idiff;
    const auto n = dynamical->neval;
    NDArray<double, 2> xk_shift;
    NDArray<std::complex<double>, 2> evec_tmp;
    NDArray<double, 2> omega_shift;
    NDArray<double, 1> omega_tmp;
    NDArray<double, 2> kvec_na_tmp;
    const auto h = 1.0e-4;

    const unsigned int ndiff = 2;

    omega_shift.resize(ndiff, n);
    xk_shift.resize(ndiff, 3);
    omega_tmp.resize(ndiff);
    evec_tmp.resize(1, 1);
    kvec_na_tmp.resize(2, 3);

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
            for (j = 0; j < 3; ++j) kvec_na_tmp[0][j] /= norm;
        }
        norm = std::sqrt(kvec_na_tmp[1][0] * kvec_na_tmp[1][0] + kvec_na_tmp[1][1] * kvec_na_tmp[1][1] +
                         kvec_na_tmp[1][2] * kvec_na_tmp[1][2]);

        if (norm > eps) {
            for (j = 0; j < 3; ++j) kvec_na_tmp[1][j] /= norm;
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

    xk_shift.clear();
    omega_shift.clear();
    omega_tmp.clear();
    evec_tmp.clear();
    kvec_na_tmp.clear();
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

    NDArray<std::complex<double>, 3> ddyn;
    std::complex<double> ctmp;
    NDArray<std::complex<double>, 2> vel_tmp;
    NDArray<std::complex<double>, 3> mat_tmp;
    std::complex<double> czero(0.0, 0.0);
    std::vector<int> smallgroup_k;
    NDArray<double, 2> eval_tmp;

    if (dynamical->nonanalytic) {
        exit("phonon_vel_k2",
             "Sorry. Analytic calculation of "
             "group velocity is not supported for NONANALYTIC>0.");
    }

    ddyn.resize(3, nmode, nmode);
    vel_tmp.resize(3, nmode);
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

                mat_tmp.resize(3, ideg, ideg);
                eval_tmp.resize(3, ideg);

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

                    solve_dense_hermitian(ideg, mat_tmp[icrd], eval_tmp[icrd], nullptr, false);

                    for (j = 0; j < ideg; ++j) {
                        vel_tmp[icrd][j + is] = eval_tmp[icrd][j] / (2.0 * omega_in[j + is]);
                    }
                }

                mat_tmp.clear();
                eval_tmp.clear();

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
        ddyn.clear();
    }
    if (vel_tmp) {
        vel_tmp.clear();
    }

    double symmetrizer_k[3][3];

    kpoint->get_symmetrization_matrix_at_k(xk_in, smallgroup_k, symmetrizer_k);

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

void PhononVelocity::velocity_matrix_analytic(const double *xk_in, const std::vector<FcsArrayWithCell> &fc2_in,
                                              const double *omega_in, std::complex<double> **evec_in,
                                              std::complex<double> ***velmat_out) const
{
    // Use Allen's definition
    // Only the analytic part of the dynamical matrix will be considered.
    // Non-analytic part must be treated seperately.

    unsigned int i, j, k;

    const auto nmode = dynamical->neval;

    NDArray<std::complex<double>, 3> ddymat;

    ddymat.resize(nmode, nmode, 3);

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
            if (omega_in[i] < eps8 || omega_in[j] < eps8) {
                for (k = 0; k < 3; ++k) {
                    velmat_out[i][j][k] = std::complex<double>(0.0, 0.0);
                }
                continue;
            }
            const auto inv_omega = 0.5 * im / std::sqrt(omega_in[i] * omega_in[j]);
            for (k = 0; k < 3; ++k) {
                velmat_out[i][j][k] *= inv_omega;
            }
        }
    }
}
