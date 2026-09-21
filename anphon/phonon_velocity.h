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

namespace PHON_NS
{
class PhononVelocity
{
public:
    PhononVelocity(const RunInfo &run, const System *system, const Fcs_phonon *fcs_phonon, const Ewald *ewald,
                   const Dynamical *dynamical, const Dos *dos);

    ~PhononVelocity();

    void setup_velocity();

    void phonon_vel_k(const double *, double **) const;

    void get_phonon_group_velocity_mesh(const KpointMeshUniform &kmesh_in, const Eigen::Matrix3d &lavec_p,
                                        double ***phvel3_out) const;

    void get_phonon_group_velocity_mesh_velmat(const KpointMeshUniform &kmesh_in, const Eigen::Matrix3d &lavec_p,
                                               double ***phvel3_out) const;

    void get_phonon_group_velocity_mesh_mpi(const KpointMeshUniform &kmesh_in, const Eigen::Matrix3d &lavec_p,
                                            double ***phvel3_out) const;

    void gather_group_velocities_mesh(const KpointMeshUniform &kmesh_in, const Eigen::Matrix3d &lavec_p,
                                      NDArray<double, 3> &vel_out, const double unit_factor,
                                      const bool bcast_full) const;

    // velmat_out (full matrix, coherent term only) and velblock_out (per-branch
    // block-summed diad for the Peierls term / boundary speed) may each be nullptr.
    void calc_phonon_velmat_mesh(NDArray<std::complex<double>, 4> *velmat_out, NDArray<double, 4> *velblock_out) const;

    void get_phonon_group_velocity_bandstructure_velmat(const KpointBandStructure *kpoint_bs_in,
                                                        const Eigen::Matrix3d &lavec_p,
                                                        const std::vector<FcsArrayWithCell> &fc2_in,
                                                        double **phvel_out) const;

    // kvec_fixed: hold the nonanalytic direction fixed (band paths, where the
    // eigenproblem uses the segment direction). nullptr = radial, as on a mesh.
    void add_nonanalytic_velocity_matrix(const double *xk_in, const double *omega_in,
                                         const std::complex<double> *const *evec_in,
                                         std::complex<double> ***velmat_inout,
                                         const double *kvec_fixed = nullptr) const;

    void velocity_matrix_analytic(const double *xk_in, const std::vector<FcsArrayWithCell> &fc2_in,
                                  const double *omega_in, const std::complex<double> *const *evec_in,
                                  std::complex<double> ***velmat_out) const;

    bool print_velocity;

private:
    double diff(const double *, unsigned int, double) const;

    void set_default_variables();

    void deallocate_variables();

private:
    // Collaborators (non-owning; owned by PHON, which outlives this object).
    const RunInfo &run;
    const System *system;
    const Fcs_phonon *fcs_phonon;
    const Ewald *ewald;
    const Dynamical *dynamical;
    const Dos *dos;
};
} // namespace PHON_NS
