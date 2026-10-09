/*
 phonon_velocity.h

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <Eigen/Core>
#include <complex>
#include <vector>
#include "fcs_phonon.h"
#include "kpoint.h"
#include "ndarray.h"
#include "phonon.h"
#include "velocity_symmetry.h"

namespace PHON_NS
{
class DymatEigenValue;

class PhononVelocity
{
public:
    PhononVelocity(const RunInfo &run, const System *system);

    ~PhononVelocity();

    // Copies the space-group operations that reduce the k mesh (Symmetry::SymmListWithMap,
    // the distorted cell's group for RELAXED_STRUCTURE / field runs) for the little-group
    // symmetrization of the mesh velocities.
    void setup_velocity(const std::vector<SymmetryOperationWithMapping> &symmlist, const bool time_reversal);

    void phonon_vel_k(const double *, const Dynamical &dynamical, const std::vector<FcsArrayWithCell> &fc2_in,
                      const Dielec &dielec, const Ewald &ewald, double **) const;

    // Raw finite-difference velocities (no little-group averaging), for the adaptive
    // smearing widths only. Transport velocities: gather_group_velocities_mesh.
    void get_phonon_group_velocity_mesh(const KpointMeshUniform &kmesh_in, const Eigen::Matrix3d &lavec_p,
                                        const Dynamical &dynamical, const std::vector<FcsArrayWithCell> &fc2_in,
                                        const Dielec &dielec, const Ewald &ewald, double ***phvel3_out) const;

    void get_phonon_group_velocity_mesh_velmat(const KpointMeshUniform &kmesh_in, const Dynamical &dynamical,
                                               const std::vector<FcsArrayWithCell> &fc2_in, const Dielec &dielec,
                                               const Ewald &ewald, double ***phvel3_out) const;

    void get_phonon_group_velocity_mesh_mpi(const KpointMeshUniform &kmesh_in, const Eigen::Matrix3d &lavec_p,
                                            const Dynamical &dynamical, const std::vector<FcsArrayWithCell> &fc2_in,
                                            const Dielec &dielec, const Ewald &ewald, double ***phvel3_out) const;

    void gather_group_velocities_mesh(const KpointMeshUniform &kmesh_in, const Eigen::Matrix3d &lavec_p,
                                      const Dynamical &dynamical, const std::vector<FcsArrayWithCell> &fc2_in,
                                      const Dielec &dielec, const Ewald &ewald, NDArray<double, 3> &vel_out,
                                      const double unit_factor, const bool bcast_full) const;

    // velmat_out (full matrix, coherent term only) and velblock_out (per-branch
    // block-summed diad for the Peierls term / boundary speed) may each be nullptr.
    void calc_phonon_velmat_mesh(const KpointMeshUniform &kmesh_in, const DymatEigenValue &dymat_in,
                                 const Dynamical &dynamical, const std::vector<FcsArrayWithCell> &fc2_in,
                                 const Dielec &dielec, const Ewald &ewald, NDArray<std::complex<double>, 4> *velmat_out,
                                 NDArray<double, 4> *velblock_out) const;

    void get_phonon_group_velocity_bandstructure_velmat(const KpointBandStructure *kpoint_bs_in,
                                                        const Dynamical &dynamical,
                                                        const std::vector<FcsArrayWithCell> &fc2_in,
                                                        const Dielec &dielec, const Ewald &ewald,
                                                        double **phvel_out) const;

    // Velocity operator dD~/dq (Cartesian components, atomic basis, without the 1/(2 pi)).
    // kvec_fixed: hold the nonanalytic direction fixed (band paths, where the eigenproblem
    // uses the segment direction). nullptr = radial, as on a mesh.
    void velocity_operator(const double *xk_in, const std::vector<FcsArrayWithCell> &fc2_in, const Dynamical &dynamical,
                           const Dielec &dielec, const Ewald &ewald, Eigen::MatrixXcd (&m_out)[3],
                           const double *kvec_fixed = nullptr) const;

    // velmat_out[i][j][mu] = <e_i| M^mu |e_j> / (2 sqrt(w_i w_j)), zero where a frequency vanishes.
    void project_velocity_operator(const Eigen::MatrixXcd (&m)[3], const double *omega_in,
                                   const std::complex<double> *const *evec_in,
                                   std::complex<double> ***velmat_out) const;

    // Little group of the mesh point xk_in (fractional), with the time-reversal partners.
    std::vector<velocity_symmetry::LittleGroupOp> little_group(const double *xk_in) const;

    bool print_velocity;

private:
    double diff(const double *, unsigned int, double) const;

    void set_default_variables();

    void add_nonanalytic_velocity_operator(const double *xk_in, const Dynamical &dynamical, const Dielec &dielec,
                                           const Ewald &ewald, Eigen::MatrixXcd (&m_frac)[3],
                                           const double *kvec_fixed) const;

    void symmetrize_mode_velocities(const double *xk_in, const Dynamical &dynamical,
                                    const std::vector<FcsArrayWithCell> &fc2_in, const Dielec &dielec,
                                    const Ewald &ewald, double **vel) const;

    void deallocate_variables();

private:
    // Collaborators (non-owning; owned by PHON, which outlives this object).
    const RunInfo &run;
    const System *system;

    // Owned copies of the k-mesh symmetry operations, set by setup_velocity.
    std::vector<Eigen::Matrix3d> rot_frac;
    std::vector<std::vector<unsigned int>> atom_mapping;
    bool time_reversal = false;
};
} // namespace PHON_NS
