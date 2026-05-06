/*
 scph_qha_common.h

 Copyright (c) 2026

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "anharmonic_core.h"
#include "constants.h"
#include "dynamical.h"
#include "error.h"
#include "fcs_phonon.h"
#include "kpoint.h"
#include "memory.h"
#include "mpi.h"
#include "mpi_common.h"
#include "pointers.h"
#include "system.h"
#include "symmetry_core.h"

namespace PHON_NS
{
class ScphQhaCommon: protected Pointers
{
public:
    explicit ScphQhaCommon(class PHON *phon);

    ~ScphQhaCommon() override;

protected:
    // Shared state between Scph and Qha.
    KpointMeshUniform *kmesh_coarse = nullptr;
    KpointMeshUniform *kmesh_dense = nullptr;
    std::vector<int> kmap_coarse_to_dense;

    std::complex<double> *phi3_reciprocal = nullptr;
    std::complex<double> *phi4_reciprocal = nullptr;
    PhaseFactorStorage *phase_factor = nullptr;

    double **omega2_harmonic = nullptr;
    std::complex<double> ***evec_harmonic = nullptr;
    MinimumDistList ***mindist_list = nullptr;
    std::complex<double> ****mat_transform_sym = nullptr;

    std::vector<Eigen::MatrixXcd> dymat_harm_short;
    std::vector<Eigen::MatrixXcd> dymat_harm_long;
    int compute_Cv_anharmonic = 0;
    unsigned int ialgo = 0;
    bool selfenergy_offdiagonal = true;

    void initialize_variables();

    void deallocate_variables();

    void setup_kmesh(unsigned int kmesh_dense_input[3], unsigned int kmesh_coarse_input[3],
                     const char *mode_name, const char *mapping_error_message);

    void setup_eigvecs();

    void setup_structural_data();

    void setup_pp_interaction(const bool prepare_v3);

    void zerofill_harmonic_dymat_renormalize(std::complex<double> ****delta_harmonic_dymat_renormalize,
                                             const unsigned int NT) const;

    void load_scph_dymat_from_file(std::complex<double> ****dymat_out, std::string filename_dymat,
                                   const KpointMeshUniform *kmesh_dense_in, const KpointMeshUniform *kmesh_coarse_in,
                                   const unsigned int nonanalytic_in, const bool selfenergy_offdiagonal_in);

    void store_renormalized_dymat_to_file(const std::complex<double> *const *const *const *dymat_in,
                                  std::string filename_dymat, const KpointMeshUniform *kmesh_dense_in,
                                  const KpointMeshUniform *kmesh_coarse_in, const unsigned int nonanalytic_in,
                                  const bool selfenergy_offdiagonal_in);

    static void mpi_bcast_complex(std::complex<double> ****data, const unsigned int NT, const unsigned int nk,
                                  const unsigned int ns);

    void write_anharmonic_correction_fc2(std::complex<double> ****delta_dymat, unsigned int NT,
                                         const KpointMeshUniform *kmesh_coarse_in, MinimumDistList ***mindist_list_in,
                                         bool is_qha = false, int type = 0);

    void postprocess(std::complex<double> ****delta_dymat, std::complex<double> ****delta_harmonic_dymat_renormalize,
                     std::complex<double> ****delta_dymat_scph_plus_bubble,
                     const KpointMeshUniform *kmesh_coarse_in, MinimumDistList ***mindist_list_in,
                     bool is_qha = false, int bubble_in = 0);

    void compute_V4_elements_mpi_over_kpoint(std::complex<double> ***v4_out, double **omega2_harmonic_in,
                                             std::complex<double> ***evec_in, bool self_offdiag, bool relax,
                                             const KpointMeshUniform *kmesh_coarse_in,
                                             const KpointMeshUniform *kmesh_dense_in,
                                             const std::vector<int> &kmap_coarse_to_dense,
                                             const PhaseFactorStorage *phase_storage_in,
                                             std::complex<double> *phi4_reciprocal_inout);

    void compute_V4_elements_mpi_over_band(std::complex<double> ***v4_out, double **omega2_harmonic_in,
                                           std::complex<double> ***evec_in, bool self_offdiag,
                                           const KpointMeshUniform *kmesh_coarse_in,
                                           const KpointMeshUniform *kmesh_dense_in,
                                           const std::vector<int> &kmap_coarse_to_scph,
                                           const PhaseFactorStorage *phase_storage_in,
                                           std::complex<double> *phi4_reciprocal_inout);

    void compute_V3_elements_mpi_over_kpoint(std::complex<double> ***v3_out, double **omega2_harmonic_in,
                                             const std::complex<double> *const *const *evec_in, bool self_offdiag,
                                             const KpointMeshUniform *kmesh_coarse_in,
                                             const KpointMeshUniform *kmesh_dense_in,
                                             const PhaseFactorStorage *phase_storage_in,
                                             std::complex<double> *phi3_reciprocal_inout);

    void calculate_del_v0_del_umn_renorm(std::complex<double> *del_v0_del_umn_renorm, double *C1_array,
                                         double **C2_array, double ***C3_array, 
                                         std::array<std::array<double, 3>, 3> &eta_tensor,
                                         const std::array<std::array<double, 3>, 3> &u_tensor,
                                         std::complex<double> **del_v1_del_umn, std::complex<double> **del2_v1_del_umn2,
                                         std::complex<double> **del3_v1_del_umn3,
                                         std::complex<double> ***del_v2_del_umn,
                                         std::complex<double> ***del2_v2_del_umn2,
                                         std::complex<double> ****del_v3_del_umn, const std::vector<double> &q0,
                                         double pvcell,
                                         const KpointMeshUniform *kmesh_dense_in);

    void compute_anharmonic_v1_array(std::complex<double> *v1_SCP, std::complex<double> *v1_renorm,
                                     std::complex<double> ***v3_renorm, std::complex<double> ***cmat_convert,
                                     double **omega2_anharm_T, double T_in, const KpointMeshUniform *kmesh_dense_in);

    void compute_anharmonic_del_v0_del_umn(std::complex<double> *del_v0_del_umn_SCP,
                                           std::complex<double> *del_v0_del_umn_renorm,
                                           std::complex<double> ***del_v2_del_umn,
                                           std::complex<double> ***del2_v2_del_umn2,
                                           std::complex<double> ****del_v3_del_umn,
                                           const std::array<std::array<double, 3>, 3> &u_tensor,
                                           const std::vector<double> &q0,
                                           std::complex<double> ***cmat_convert, double **omega2_anharm_T,
                                           double T_in, const KpointMeshUniform *kmesh_dense_in);

    void get_derivative_central_diff(double delta_t, unsigned int nk, double **omega0, double **omega2,
                                     double **domega_dt);

    void zerofill_elements_acoustic_at_gamma(double **omega2, std::complex<double> ***v_elems, int fc_order,
                                             unsigned int nk_dense_in, unsigned int nk_irred_coarse_in) const;

    bool use_band_parallel_v4() const;
};
} // namespace PHON_NS
