/*
phonon_velocity.cpp

Copyright (c) 2014, 2015, 2016 Terumasa Tadano

This file is distributed under the terms of the MIT license.
Please see the file 'LICENCE.txt' in the root directory 
or http://opensource.org/licenses/mit-license.php for information.
*/

#include "phonon_velocity.h"
#include <algorithm>
#include <complex>
#include <iomanip>
#include <limits>
#include "constants.h"
#include "degeneracy_utils.h"
#include "dense_hermitian_eigen.h"
#include "dielec.h"
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

PhononVelocity::PhononVelocity(const RunInfo &run_in, const System *system_in) : run(run_in), system(system_in)
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
}

void PhononVelocity::deallocate_variables()
{}

// Transport uses the unsymmetrized velocity matrix with nonanalytic
// connection, block-trace Peierls weights, cross-block coherent pairs,
// block boundary speeds, and matrix-diagonal PRINTVEL.
void PhononVelocity::setup_velocity()
{
    MPI_Bcast(&print_velocity, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
}

// Project the velocity-matrix diagonal onto the band-path direction,
// avoiding finite differences across sorted-band crossings. Diagonals
// within degenerate multiplets remain basis dependent; only block traces
// are invariant.
void PhononVelocity::get_phonon_group_velocity_bandstructure_velmat(
    const KpointBandStructure *kpoint_bs_in, const Eigen::Matrix3d &lavec_p, const Dynamical &dynamical,
    const std::vector<FcsArrayWithCell> &fc2_in, const Dielec &dielec, const Ewald &ewald, double **phvel_out) const
{
    const auto nk = kpoint_bs_in->nk;
    const auto ns = system->get_num_modes();

    NDArray<std::complex<double>, 3> velmat_k;
    NDArray<std::complex<double>, 2> evec_k;
    NDArray<double, 1> eval_k;
    velmat_k.resize(ns, ns, 3);
    evec_k.resize(ns, ns);
    eval_k.resize(ns);

    const auto &fc2_vel = (dynamical.nonanalytic == 3) ? ewald.fc2_without_dipole : fc2_in;

    for (auto ik = 0u; ik < nk; ++ik) {
        if (dynamical.nonanalytic == 3) {
            dynamical.eval_k_ewald(kpoint_bs_in->xk[ik],
                                   kpoint_bs_in->kvec_na[ik],
                                   ewald.fc2_without_dipole,
                                   eval_k,
                                   evec_k,
                                   true);
        } else {
            dynamical.eval_k(kpoint_bs_in->xk[ik], kpoint_bs_in->kvec_na[ik], fc2_in, eval_k, evec_k, true);
        }
        for (auto is = 0u; is < ns; ++is) eval_k[is] = dynamical.freq(eval_k[is]);

        velocity_matrix_analytic(kpoint_bs_in->xk[ik], fc2_vel, eval_k, evec_k, velmat_k);
        add_nonanalytic_velocity_matrix(kpoint_bs_in->xk[ik],
                                        eval_k,
                                        evec_k,
                                        dynamical,
                                        dielec,
                                        ewald,
                                        velmat_k,
                                        kpoint_bs_in->kvec_na[ik]);

        for (auto is = 0u; is < ns; ++is) {
            double v[3];
            for (auto j = 0; j < 3; ++j) v[j] = velmat_k[is][is][j].real();
            rotvec(v, v, lavec_p);
            auto vproj = 0.0;
            for (auto j = 0; j < 3; ++j) vproj += (v[j] / (2.0 * pi)) * kpoint_bs_in->kvec_na[ik][j];
            phvel_out[ik][is] = vproj;
        }
    }
    velmat_k.clear();
    evec_k.clear();
    eval_k.clear();
}

void PhononVelocity::get_phonon_group_velocity_mesh(const KpointMeshUniform &kmesh_in, const Eigen::Matrix3d &lavec_p,
                                                    const Dynamical &dynamical,
                                                    const std::vector<FcsArrayWithCell> &fc2_in, const Ewald &ewald,
                                                    double ***phvel3_out) const
{
    // This routine computes the group velocities for the given uniform k mesh.
    const auto nk = kmesh_in.nk;
    const auto ns = system->get_num_modes();

    NDArray<double, 2> vel;

    vel.resize(ns, 3);

    for (unsigned int i = 0; i < nk; ++i) {
        phonon_vel_k(&kmesh_in.xk[i][0], dynamical, fc2_in, ewald, vel);

        for (unsigned int j = 0; j < ns; ++j) {
            rotvec(vel[j], vel[j], lavec_p);
            for (unsigned int k = 0; k < 3; ++k) {
                vel[j][k] /= 2.0 * pi;
                phvel3_out[i][j][k] = vel[j][k];
            }
        }
    }
    vel.clear();
}

void PhononVelocity::get_phonon_group_velocity_mesh_mpi(const KpointMeshUniform &kmesh_in,
                                                        const Eigen::Matrix3d &lavec_p, const Dynamical &dynamical,
                                                        const std::vector<FcsArrayWithCell> &fc2_in, const Ewald &ewald,
                                                        double ***phvel3_out) const
{
    // This routine computes the group velocities for the given uniform k mesh
    // using MPI parallelization.
    const auto nk = kmesh_in.nk;
    const auto ns = system->get_num_modes();

    NDArray<double, 2> vel;
    NDArray<double, 3> phvel3_loc;
    NDArray<int, 1> displs;
    NDArray<int, 1> sendcount;
    NDArray<int, 1> recvcount;
    std::vector<int> nk_proc;
    std::vector<int> ik_begin_proc, ik_end_proc;

    sendcount.resize(run.nprocs);
    recvcount.resize(run.nprocs);
    nk_proc.resize(run.nprocs);

    auto nk_loc = nk / run.nprocs;
    auto nk_res = nk - nk_loc * run.nprocs;

    for (auto i = 0; i < run.nprocs; ++i) {
        nk_proc[i] = nk_loc;
        if (i < nk_res) ++nk_proc[i];
        sendcount[i] = 3 * ns * nk_proc[i];
        recvcount[i] = sendcount[i];
    }

    if (run.my_rank == 0) {
        displs.resize(run.nprocs);
        displs[0] = 0;
        for (auto i = 1; i < run.nprocs; ++i) {
            displs[i] = displs[i - 1] + recvcount[i - 1];
        }
    }

    ik_begin_proc.resize(run.nprocs);
    ik_end_proc.resize(run.nprocs);
    ik_begin_proc[0] = 0;
    ik_end_proc[0] = nk_proc[0];
    for (auto i = 1; i < run.nprocs; ++i) {
        ik_begin_proc[i] = ik_end_proc[i - 1];
        ik_end_proc[i] = ik_begin_proc[i] + nk_proc[i];
    }

    std::vector<int> klist_proc;
    for (auto ik = ik_begin_proc[run.my_rank]; ik < ik_end_proc[run.my_rank]; ++ik) {
        klist_proc.push_back(ik);
    }

    nk_loc = klist_proc.size();

    phvel3_loc.resize(nk_loc, ns, 3);
    vel.resize(ns, 3);

    for (unsigned int i = 0; i < nk_loc; ++i) {
        phonon_vel_k(&kmesh_in.xk[klist_proc[i]][0], dynamical, fc2_in, ewald, vel);

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
                sendcount[run.my_rank],
                MPI_DOUBLE,
                run.my_rank == 0 ? &phvel3_out[0][0][0] : nullptr,
                run.my_rank == 0 ? &recvcount[0] : nullptr,
                run.my_rank == 0 ? &displs[0] : nullptr,
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

    phvel3_loc.clear();
    sendcount.clear();
    recvcount.clear();
    displs.clear();
}

void PhononVelocity::gather_group_velocities_mesh(const KpointMeshUniform &kmesh_in, const Eigen::Matrix3d &lavec_p,
                                                  const Dynamical &dynamical,
                                                  const std::vector<FcsArrayWithCell> &fc2_in, const Ewald &ewald,
                                                  NDArray<double, 3> &vel_out, const double unit_factor,
                                                  const bool bcast_full) const
{
    // Allocate and gather velocities on rank 0, or all ranks if bcast_full.
    // Apply unit_factor before broadcasting: 1.0 keeps atomic units;
    // Bohr_in_Angstrom * 1.0e-10 / time_ry gives m/s.
    // Other ranks receive dummy storage; the caller deallocates vel_out.
    const auto nk = kmesh_in.nk;
    const auto neval = system->get_num_modes();

    if (run.my_rank == 0 || bcast_full) {
        vel_out.resize(nk, neval, 3);
    } else {
        vel_out.resize(1, 1, 1);
    }

    get_phonon_group_velocity_mesh_mpi(kmesh_in, lavec_p, dynamical, fc2_in, ewald, vel_out);

    if (run.my_rank == 0 && unit_factor != 1.0) {
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

// Fill PRINTVEL from the analytic velocity-matrix diagonal, without
// elementwise symmetrization. Units match get_phonon_group_velocity_mesh
// (Cartesian, divided by 2 pi, no SI factor). Compute the full matrix
// but retain only its diagonal. Adaptive smearing keeps finite differences.
void PhononVelocity::get_phonon_group_velocity_mesh_velmat(const KpointMeshUniform &kmesh_in,
                                                           const Eigen::Matrix3d &lavec_p, const Dynamical &dynamical,
                                                           const std::vector<FcsArrayWithCell> &fc2_in,
                                                           const Dielec &dielec, const Ewald &ewald,
                                                           double ***phvel3_out) const
{
    const auto nk = kmesh_in.nk;
    const auto ns = system->get_num_modes();

    // Self-contained (diagonalizes for itself) so it does not depend on the mesh's
    // eigenpairs having been filled or on the mesh being kmesh_dos.
    NDArray<std::complex<double>, 3> velmat_k;
    NDArray<std::complex<double>, 2> evec_k;
    NDArray<double, 1> eval_k;
    velmat_k.resize(ns, ns, 3);
    evec_k.resize(ns, ns);
    eval_k.resize(ns);

    const auto &fc2_vel = (dynamical.nonanalytic == 3) ? ewald.fc2_without_dipole : fc2_in;

    for (auto ik = 0u; ik < nk; ++ik) {
        double kvec[3];
        for (auto j = 0; j < 3; ++j) kvec[j] = kmesh_in.xk[ik][j];
        rotvec(kvec, kvec, system->get_primcell().reciprocal_lattice_vector, 'T');
        const auto norm = std::sqrt(kvec[0] * kvec[0] + kvec[1] * kvec[1] + kvec[2] * kvec[2]);
        if (norm > eps) {
            for (auto j = 0; j < 3; ++j) kvec[j] /= norm;
        }

        if (dynamical.nonanalytic == 3) {
            dynamical.eval_k_ewald(kmesh_in.xk[ik], kvec, ewald.fc2_without_dipole, eval_k, evec_k, true);
        } else {
            dynamical.eval_k(kmesh_in.xk[ik], kvec, fc2_in, eval_k, evec_k, true);
        }
        for (auto is = 0u; is < ns; ++is) eval_k[is] = dynamical.freq(eval_k[is]);

        velocity_matrix_analytic(kmesh_in.xk[ik], fc2_vel, eval_k, evec_k, velmat_k);
        add_nonanalytic_velocity_matrix(kmesh_in.xk[ik], eval_k, evec_k, dynamical, dielec, ewald, velmat_k);

        for (auto is = 0u; is < ns; ++is) {
            double v[3];
            for (auto j = 0; j < 3; ++j) v[j] = velmat_k[is][is][j].real();
            rotvec(v, v, lavec_p);
            for (auto j = 0; j < 3; ++j) phvel3_out[ik][is][j] = v[j] / (2.0 * pi);
        }
    }
    velmat_k.clear();
    evec_k.clear();
    eval_k.clear();
}

// Gather k-distributed contiguous records (stride elements per k) to rank 0 in chunks
// whose element counts fit an int. A single MPI_Gatherv with count nk*ns*ns*3 overflows
// 32-bit counts for large systems (e.g. 20^3 mesh x 300 branches = 2.16e9 elements).
template <class T>
static void gather_k_records(const T *local, const std::vector<int> &nk_proc, const int my_rank, const int nprocs,
                             const size_t stride, const MPI_Datatype type, T *out)
{
    // k offsets and chunk endpoints in size_t: only the per-chunk MPI counts and
    // displacements are ever narrowed to int, and those are bounded below INT_MAX/2.
    std::vector<size_t> kbeg(nprocs + 1, 0);
    for (auto r = 0; r < nprocs; ++r) kbeg[r + 1] = kbeg[r] + static_cast<size_t>(nk_proc[r]);
    const auto nk = kbeg[nprocs];

    const auto max_elems = static_cast<size_t>(std::numeric_limits<int>::max()) / 2;
    if (stride > max_elems) exit("gather_k_records", "A single k-point record exceeds the MPI count limit.");
    const auto chunk_k = std::max<size_t>(1, max_elems / stride);

    std::vector<int> cnt(nprocs), dsp(nprocs);
    for (size_t k0 = 0; k0 < nk; k0 += chunk_k) {
        const auto k1 = std::min(nk, k0 + chunk_k);
        for (auto r = 0; r < nprocs; ++r) {
            const auto lo = std::max(k0, kbeg[r]);
            const auto hi = std::min(k1, kbeg[r + 1]);
            const auto n = hi > lo ? hi - lo : 0;
            cnt[r] = static_cast<int>(n * stride);
            dsp[r] = n > 0 ? static_cast<int>((lo - k0) * stride) : 0;
        }
        const auto lo_me = std::max(k0, kbeg[my_rank]);
        const auto off_me = (lo_me > kbeg[my_rank] ? lo_me - kbeg[my_rank] : 0) * stride;
        MPI_Gatherv(cnt[my_rank] > 0 ? local + off_me : nullptr,
                    cnt[my_rank],
                    type,
                    my_rank == 0 ? out + k0 * stride : nullptr,
                    my_rank == 0 ? cnt.data() : nullptr,
                    my_rank == 0 ? dsp.data() : nullptr,
                    type,
                    0,
                    MPI_COMM_WORLD);
    }
}

void PhononVelocity::calc_phonon_velmat_mesh(const KpointMeshUniform &kmesh_in, const DymatEigenValue &dymat_in,
                                             const Dynamical &dynamical, const std::vector<FcsArrayWithCell> &fc2_in,
                                             const Dielec &dielec, const Ewald &ewald,
                                             NDArray<std::complex<double>, 4> *velmat_out,
                                             NDArray<double, 4> *velblock_out) const
{
    // velmat_out  : full velocity matrix [nk][ns][ns][3] on rank 0 (needed only by the
    //               coherent term; nullptr to skip -- it is the memory hog).
    // velblock_out: per-branch block-summed diad [nk][ns][3][3] on rank 0,
    //                   velblock[k][is][a][b] = sum_{js in D(is)} Re(V^a_{is js} V^b_{js is}),
    //               so that summing it over the branches of one degenerate block gives
    //               the basis-invariant Tr(P V^a P V^b P). This is all the Peierls term
    //               and the boundary speed need, and it is O(nk ns) in memory, so the
    //               default RTA run never stores the full matrix.
    if (!velmat_out && !velblock_out) return;

    const auto nk = kmesh_in.nk;
    const auto ns = system->get_num_modes();
    const auto factor = Bohr_in_Angstrom * 1.0e-10 / (time_ry * 2.0 * pi);

    if (run.my_rank == 0 && run.verbosity > 0) {
        std::cout << " Calculating group velocity matrix of phonons on uniform grid ... ";
    }

    // k distribution
    std::vector<int> nk_proc(run.nprocs);
    if (nk > static_cast<unsigned int>(std::numeric_limits<int>::max())) {
        exit("calc_phonon_velmat_mesh", "Number of k points exceeds the supported range.");
    }
    const auto nk_int = static_cast<int>(nk);
    auto nk_loc = nk_int / run.nprocs;
    const auto nk_res = nk_int - nk_loc * run.nprocs;
    for (auto i = 0; i < run.nprocs; ++i) nk_proc[i] = nk_loc + (i < nk_res ? 1 : 0);
    auto ik_begin = 0;
    for (auto i = 0; i < run.my_rank; ++i) ik_begin += nk_proc[i];
    nk_loc = nk_proc[run.my_rank];

    NDArray<std::complex<double>, 3> vk;         // one k point, discarded after use
    NDArray<std::complex<double>, 4> velmat_loc; // only if the full matrix is wanted
    NDArray<double, 4> velblock_loc;
    vk.resize(ns, ns, 3);
    if (velmat_out) velmat_loc.resize(std::max(nk_loc, 1), ns, ns, 3);
    if (velblock_out) velblock_loc.resize(std::max(nk_loc, 1), ns, 3, 3);

    const auto &fc2_vel = (dynamical.nonanalytic == 3) ? ewald.fc2_without_dipole : fc2_in;
    const auto eval_all = dymat_in.get_eigenvalues();
    const auto evec_all = dymat_in.get_eigenvectors();
    const auto tol_cm = transport_block_tol_cm();
    std::vector<int> blk_lo, blk_hi;

    for (auto i = 0; i < nk_loc; ++i) {
        const auto knum = ik_begin + i;

        // For NONANALYTIC = 3 the eigenproblem is solved with the dipole-free force
        // constants plus an Ewald long-range matrix, so the velocity matrix has to be
        // built from the same decomposition.
        velocity_matrix_analytic(kmesh_in.xk[knum], fc2_vel, eval_all[knum], evec_all[knum], vk);
        add_nonanalytic_velocity_matrix(kmesh_in.xk[knum],
                                        eval_all[knum],
                                        evec_all[knum],
                                        dynamical,
                                        dielec,
                                        ewald,
                                        vk);

        for (auto j = 0u; j < ns; ++j) {
            for (auto k = 0u; k < ns; ++k) {
                rotvec(vk[j][k], vk[j][k], system->get_primcell().lattice_vector);
                for (auto mu = 0; mu < 3; ++mu) vk[j][k][mu] *= factor;
            }
        }

        if (velmat_out) {
            for (auto j = 0u; j < ns; ++j) {
                for (auto k = 0u; k < ns; ++k) {
                    for (auto mu = 0; mu < 3; ++mu) velmat_loc[i][j][k][mu] = vk[j][k][mu];
                }
            }
        }

        if (velblock_out) {
            // Blocks taken at the REPRESENTATIVE of this k's star, from the same shared
            // partition the coherent term uses to exclude same-block pairs, so the two
            // partitions are identical by construction.
            const auto irr = kmesh_in.kmap_to_irreducible[knum];
            const auto krep = kmesh_in.kpoint_irred_all[irr][0].knum;
            transport_block_bounds(ns, eval_all[krep], tol_cm, blk_lo, blk_hi);

            for (auto is = 0u; is < ns; ++is) {
                for (auto a = 0; a < 3; ++a) {
                    for (auto b = 0; b < 3; ++b) {
                        auto acc = 0.0;
                        for (auto js = blk_lo[is]; js < blk_hi[is]; ++js) {
                            acc += (vk[is][js][a] * vk[js][is][b]).real();
                        }
                        velblock_loc[i][is][a][b] = acc;
                    }
                }
            }
        }
    }
    vk.clear();

#ifdef MPI_CXX_DOUBLE_COMPLEX
    const auto mpi_complex_type = MPI_CXX_DOUBLE_COMPLEX;
#else
    const auto mpi_complex_type = MPI_COMPLEX16;
#endif

    if (velmat_out) {
        gather_k_records<std::complex<double>>(nk_loc > 0 ? &velmat_loc[0][0][0][0] : nullptr,
                                               nk_proc,
                                               run.my_rank,
                                               run.nprocs,
                                               static_cast<size_t>(ns) * ns * 3,
                                               mpi_complex_type,
                                               run.my_rank == 0 ? &(*velmat_out)[0][0][0][0] : nullptr);
        velmat_loc.clear();
    }
    if (velblock_out) {
        gather_k_records<double>(nk_loc > 0 ? &velblock_loc[0][0][0][0] : nullptr,
                                 nk_proc,
                                 run.my_rank,
                                 run.nprocs,
                                 static_cast<size_t>(ns) * 9,
                                 MPI_DOUBLE,
                                 run.my_rank == 0 ? &(*velblock_out)[0][0][0][0] : nullptr);
        velblock_loc.clear();
    }

    if (run.my_rank == 0 && run.verbosity > 0) {
        std::cout << "done!\n";
    }
}

void PhononVelocity::phonon_vel_k(const double *xk_in, const Dynamical &dynamical,
                                  const std::vector<FcsArrayWithCell> &fc2_in, const Ewald &ewald,
                                  double **vel_out) const
{
    unsigned int j;
    unsigned int idiff;
    const auto n = system->get_num_modes();
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

            if (dynamical.nonanalytic == 3) {
                dynamical.eval_k_ewald(xk_shift[idiff],
                                       kvec_na_tmp[idiff],
                                       ewald.fc2_without_dipole,
                                       omega_shift[idiff],
                                       evec_tmp,
                                       false);
            } else {
                dynamical.eval_k(xk_shift[idiff], kvec_na_tmp[idiff], fc2_in, omega_shift[idiff], evec_tmp, false);
            }
        }

        for (j = 0; j < n; ++j) {
            for (idiff = 0; idiff < ndiff; ++idiff) {
                omega_tmp[idiff] = dynamical.freq(omega_shift[idiff][j]);
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

// Central-difference D_na for NONANALYTIC methods 1/2/3. No unique
// direction-independent gradient exists at Gamma; no directional limit
// is implemented. Convert from the eigenproblem's cell-phase convention
// to the displacement-aware velocity convention using
//   (grad_a D_na)_ij = (d_a D_na)_ij + i 2pi (t_j - t_i)_a (D_na)_ij,
// where t is the primitive fractional position. The connection term is
// needed even for diagonal velocities: commutator cancellation applies
// to the full dynamical matrix, not D_na alone.
void PhononVelocity::add_nonanalytic_velocity_matrix(const double *xk_in, const double *omega_in,
                                                     const std::complex<double> *const *evec_in,
                                                     const Dynamical &dynamical, const Dielec &dielec,
                                                     const Ewald &ewald, std::complex<double> ***velmat_inout,
                                                     const double *kvec_fixed) const
{
    if (dynamical.nonanalytic == 0) return;

    const auto nmode = system->get_num_modes();
    const auto h = 1.0e-4;

    // Drop the nonanalytic velocity at Gamma, where no direction-independent
    // gradient exists. A central difference can be nonzero there;
    // a directional limit is not implemented.
    if (std::abs(xk_in[0]) < eps && std::abs(xk_in[1]) < eps && std::abs(xk_in[2]) < eps) {
        if (run.my_rank == 0 && run.verbosity > 0) {
            static auto warned_gamma = false;
            if (!warned_gamma) {
                warned_gamma = true;
                warn("add_nonanalytic_velocity_matrix",
                     "Velocity at Gamma with NONANALYTIC != 0 is convention dependent; "
                     "the nonanalytic contribution is set to zero there.");
            }
        }
        return;
    }

    NDArray<std::complex<double>, 2> dna_plus, dna_minus;
    NDArray<std::complex<double>, 3> ddna;
    dna_plus.resize(nmode, nmode);
    dna_minus.resize(nmode, nmode);
    ddna.resize(nmode, nmode, 3);

    for (auto i = 0u; i < nmode; ++i) {
        for (auto j = 0u; j < nmode; ++j) {
            for (auto k = 0; k < 3; ++k) ddna[i][j][k] = std::complex<double>(0.0, 0.0);
        }
    }

    double xk_shift[2][3], kvec[2][3];

    for (auto idir = 0; idir < 3; ++idir) {
        for (auto j = 0; j < 3; ++j) {
            xk_shift[0][j] = xk_in[j];
            xk_shift[1][j] = xk_in[j];
        }
        xk_shift[0][idir] -= h;
        xk_shift[1][idir] += h;

        for (auto ishift = 0; ishift < 2; ++ishift) {
            if (kvec_fixed) {
                // Hold the band-segment direction kvec_na fixed so the derivative uses
                // the same nonanalytic matrix as the eigenproblem.
                for (auto j = 0; j < 3; ++j) kvec[ishift][j] = kvec_fixed[j];
                continue;
            }
            for (auto j = 0; j < 3; ++j) kvec[ishift][j] = xk_shift[ishift][j];
            rotvec(kvec[ishift], kvec[ishift], system->get_primcell().reciprocal_lattice_vector, 'T');
            const auto norm = std::sqrt(kvec[ishift][0] * kvec[ishift][0] + kvec[ishift][1] * kvec[ishift][1] +
                                        kvec[ishift][2] * kvec[ishift][2]);
            if (norm > eps) {
                for (auto j = 0; j < 3; ++j) kvec[ishift][j] /= norm;
            }
        }

        auto &dst_minus = dna_minus;
        auto &dst_plus = dna_plus;

        for (auto i = 0u; i < nmode; ++i) {
            for (auto j = 0u; j < nmode; ++j) {
                dst_minus[i][j] = std::complex<double>(0.0, 0.0);
                dst_plus[i][j] = std::complex<double>(0.0, 0.0);
            }
        }

        if (dynamical.nonanalytic == 3) {
            ewald.add_longrange_matrix(xk_shift[0], kvec[0], dst_minus);
            ewald.add_longrange_matrix(xk_shift[1], kvec[1], dst_plus);
        } else {
            dynamical.calc_nonanalytic_k(xk_shift[0], kvec[0], dielec, dst_minus);
            dynamical.calc_nonanalytic_k(xk_shift[1], kvec[1], dielec, dst_plus);
        }

        for (auto i = 0u; i < nmode; ++i) {
            for (auto j = 0u; j < nmode; ++j) {
                ddna[i][j][idir] = (dst_plus[i][j] - dst_minus[i][j]) / (2.0 * h);
            }
        }
    }

    // Connection term: derivative of the sublattice phase that calc_nonanalytic_k_*
    // has already folded into D_na. Needs D_na at xk_in itself, not at the shifts.
    NDArray<std::complex<double>, 2> dna0;
    dna0.resize(nmode, nmode);
    for (auto i = 0u; i < nmode; ++i) {
        for (auto j = 0u; j < nmode; ++j) dna0[i][j] = std::complex<double>(0.0, 0.0);
    }

    double kvec0[3];
    if (kvec_fixed) {
        for (auto j = 0; j < 3; ++j) kvec0[j] = kvec_fixed[j];
    } else {
        for (auto j = 0; j < 3; ++j) kvec0[j] = xk_in[j];
        rotvec(kvec0, kvec0, system->get_primcell().reciprocal_lattice_vector, 'T');
        const auto norm0 = std::sqrt(kvec0[0] * kvec0[0] + kvec0[1] * kvec0[1] + kvec0[2] * kvec0[2]);
        if (norm0 > eps) {
            for (auto j = 0; j < 3; ++j) kvec0[j] /= norm0;
        }
    }

    if (dynamical.nonanalytic == 3) {
        ewald.add_longrange_matrix(xk_in, kvec0, dna0);
    } else {
        dynamical.calc_nonanalytic_k(xk_in, kvec0, dielec, dna0);
    }

    const auto &xf_prim = system->get_primcell().x_fractional;

    for (auto i = 0u; i < nmode; ++i) {
        for (auto j = 0u; j < nmode; ++j) {
            for (auto k = 0; k < 3; ++k) {
                const auto dt = xf_prim(j / 3, k) - xf_prim(i / 3, k);
                ddna[i][j][k] += im * tpi * dt * dna0[i][j];
            }
        }
    }
    dna0.clear();

    // Project onto the eigenvectors at xk_in and apply Allen's normalisation.
    // ddna is d(D_na)/d(q_fractional) already, so no extra factor of i here.
    // E^H ddna E per direction: O(ns^3).
    {
        Eigen::MatrixXcd E(nmode, nmode);
        for (auto i = 0u; i < nmode; ++i) {
            for (auto j = 0u; j < nmode; ++j) E(j, i) = evec_in[i][j];
        }
        Eigen::MatrixXcd Dk(nmode, nmode);
        for (auto k = 0; k < 3; ++k) {
            for (auto i = 0u; i < nmode; ++i) {
                for (auto j = 0u; j < nmode; ++j) Dk(i, j) = ddna[i][j][k];
            }
            const Eigen::MatrixXcd Vk = E.adjoint() * (Dk * E);
            for (auto i = 0u; i < nmode; ++i) {
                for (auto j = 0u; j < nmode; ++j) {
                    if (omega_in[i] < eps8 || omega_in[j] < eps8) continue;
                    velmat_inout[i][j][k] += Vk(i, j) * (0.5 / std::sqrt(omega_in[i] * omega_in[j]));
                }
            }
        }
    }

    dna_plus.clear();
    dna_minus.clear();
    ddna.clear();
}

void PhononVelocity::velocity_matrix_analytic(const double *xk_in, const std::vector<FcsArrayWithCell> &fc2_in,
                                              const double *omega_in, const std::complex<double> *const *evec_in,
                                              std::complex<double> ***velmat_out) const
{
    // Use Allen's definition
    // Only the analytic part of the dynamical matrix will be considered.
    // Non-analytic part must be treated seperately.

    unsigned int i, j, k;

    const auto nmode = system->get_num_modes();

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

    // Project onto the eigenvectors as E^H (dD/dq) E with two matrix products per
    // direction: O(ns^3), instead of the O(ns^4) explicit four-index sum.
    {
        Eigen::MatrixXcd E(nmode, nmode);
        for (i = 0; i < nmode; ++i) {
            for (j = 0; j < nmode; ++j) E(j, i) = evec_in[i][j]; // column i = eigenvector i
        }
        Eigen::MatrixXcd Dk(nmode, nmode);
        for (k = 0; k < 3; ++k) {
            for (i = 0; i < nmode; ++i) {
                for (j = 0; j < nmode; ++j) Dk(i, j) = ddymat[i][j][k];
            }
            const Eigen::MatrixXcd Vk = E.adjoint() * (Dk * E);
            for (i = 0; i < nmode; ++i) {
                for (j = 0; j < nmode; ++j) velmat_out[i][j][k] = Vk(i, j);
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
