/*
 relaxation.h

 Copyright (c) 2022 Ryota Masuki, Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include "pointers.h"
#include <complex>
#include "kpoint.h"
#include "fcs_phonon.h"
#include "scph.h"
#include "gruneisen.h"
#include "optimizers.h"

namespace PHON_NS {
class Relaxation : protected Pointers {
public:

    Relaxation(class PHON *phon);

    ~Relaxation();

    int relax_str;

    // initial strain and displacement
    double init_u_tensor[3][3]{{0.0}};
    std::vector<double> init_u0;

    // zero-th order term of the potential energy surface
    std::vector<double> V0;

    // variables related to structural optimization
    int relax_algo;
    int max_str_iter;
    double coord_conv_tol;
    double mixbeta_coord;
    double alpha_steepest_decent;
    double cell_conv_tol;
    double mixbeta_cell;

    int set_init_str;
    int cooling_u0_index; // used if set_init_str is 3
    double cooling_u0_thr; // used if set_init_str is 3
    double add_hess_diag;
    double stat_pressure;

    int renorm_3to2nd;
    int renorm_2to1st;
    int renorm_34to1st;
    std::string strain_IFC_dir;

    Optimizer * optimizer;

    void setup_relaxation();

    void compute_del_v_strain(const KpointMeshUniform *kmesh_coarse,
                              const KpointMeshUniform *kmesh_dense,
                              std::complex<double> **del_v1_del_umn,
                              std::complex<double> **del2_v1_del_umn2,
                              std::complex<double> **del3_v1_del_umn3,
                              std::complex<double> ***del_v2_del_umn,
                              std::complex<double> ***del2_v2_del_umn2,
                              std::complex<double> ****del_v3_del_umn,
                              double **omega2_harmonic,
                              std::complex<double> ***evec_harmonic,
                              int relax_str,
                              MinimumDistList ***mindist_list,
                              const PhaseFactorStorage *phase_storage_in);

    void setInitialDistortion(const double (*u_tensor_in)[3]);

    void load_V0_from_file();

    void store_V0_to_file();

    void set_init_structure_atT(double *q0,
                                double **u_tensor,
                                double *u0,
                                bool &converged_prev,
                                int &str_diverged,
                                const int i_temp_loop,
                                double **omega2_harmonic,
                                std::complex<double> ***evec_harmonic);


    void set_elastic_constants(double *C1_array,
                               double **C2_array,
                               double ***C3_array);

    void renormalize_v0_from_umn(double &,
                                 double,
                                 double **,
                                 double *,
                                 double **,
                                 double ***,
                                 double **,
                                 const double);

    void renormalize_v1_from_umn(std::complex<double> *,
                                 const std::complex<double> *const,
                                 const std::complex<double> *const *const,
                                 const std::complex<double> *const *const,
                                 const std::complex<double> *const *const,
                                 const double *const *const);

    void renormalize_v2_from_umn(const KpointMeshUniform *kmesh_coarse,
                                 const KpointMeshUniform *kmesh_dense,
                                 const std::vector<int> &kmap_coarse_to_dense,
                                 std::complex<double> **,
                                 std::complex<double> ***,
                                 std::complex<double> ***,
                                 double **);

    void renormalize_v3_from_umn(const KpointMeshUniform *kmesh_coarse,
                                 const KpointMeshUniform *kmesh_dense,
                                 std::complex<double> ***,
                                 std::complex<double> ***,
                                 std::complex<double> ****,
                                 double **);

    void renormalize_v1_from_q0(double **omega2_harmonic,
                                const KpointMeshUniform *kmesh_coarse,
                                const KpointMeshUniform *kmesh_dense,
                                std::complex<double> *,
                                std::complex<double> *,
                                std::complex<double> **,
                                std::complex<double> ***,
                                std::complex<double> ***,
                                double *);

    void renormalize_v2_from_q0(std::complex<double> ***evec_harmonic,
                                const KpointMeshUniform *kmesh_coarse,
                                const KpointMeshUniform *kmesh_dense,
                                const std::vector<int> &kmap_coarse_to_dense,
                                std::complex<double> ****mat_transform_sym,
                                std::complex<double> **delta_v2_renorm,
                                std::complex<double> **delta_v2_array_original,
                                std::complex<double> ***v3_ref,
                                std::complex<double> ***v4_ref,
                                double *q0);

    void renormalize_v3_from_q0(const KpointMeshUniform *kmesh_dense,
                                const KpointMeshUniform *kmesh_coarse,
                                std::complex<double> ***,
                                std::complex<double> ***,
                                std::complex<double> ***,
                                double *);

    void renormalize_v0_from_q0(double **omega2_harmonic,
                                const KpointMeshUniform *kmesh_dense,
                                double &,
                                double,
                                std::complex<double> *,
                                std::complex<double> **,
                                std::complex<double> ***,
                                std::complex<double> ***,
                                double *);

    void calculate_u0(const double *const q0, double *const u0,
                      double **omega2_harmonic,
                      std::complex<double> ***evec_harmonic);

    void update_cell_coordinate(double *,
                                double *,
                                double **,
                                const std::complex<double> *const,
                                const double *const *const,
                                const std::complex<double> *const,
                                const double *const *const,
                                const std::complex<double> *const *const *const,
                                const std::vector<int> &,
                                double *,
                                double *,
                                double *,
                                double &,
                                double &,
                                double **omega2_harmonic,
                                std::complex<double> ***evec_harmonic);

    void check_str_divergence(int &diverged,
                              const double *const q0,
                              const double *const u0,
                              const double *const *const u_tensor);


    void write_resfile_header(std::ofstream &fout_q0,
                              std::ofstream &fout_u0,
                              std::ofstream &fout_u_tensor);

    void write_resfile_atT(const double *const q0,
                           const double *const *const u_tensor,
                           const double *const u0,
                           const double temperature,
                           std::ofstream &fout_q0,
                           std::ofstream &fout_u0,
                           std::ofstream &fout_u_tensor);

    void write_stepresfile_header_atT(std::ofstream &fout_step_q0,
                                      std::ofstream &fout_step_u0,
                                      std::ofstream &fout_step_u_tensor,
                                      const double temp);

    void write_stepresfile(const double *const q0,
                           const double *const *const u_tensor,
                           const double *const u0,
                           const int i_str_loop,
                           std::ofstream &fout_step_q0,
                           std::ofstream &fout_step_u0,
                           std::ofstream &fout_step_u_tensor);

    int get_xyz_string(const int, std::string &);

    void calculate_eta_tensor(double **,
                              const double *const *const);


private:

    void set_default_variables();

    void deallocate_variables();

    void read_C1_array(double *const);

    void read_elastic_constants(double *const *const,
                                double *const *const *const);

    void set_initial_q0(double *const q0, std::complex<double> ***evec_harmonic);


    void set_initial_strain(double *const *const);


    void compute_del_v1_del_umn(std::complex<double> **,
                                const std::complex<double> *const *const *const);

    void compute_del2_v1_del_umn2(std::complex<double> **,
                                  const std::complex<double> *const *const *const);

    void compute_del3_v1_del_umn3(std::complex<double> **,
                                  const std::complex<double> *const *const *const);


    void compute_del_v2_del_umn(std::complex<double> ***,
                                const std::complex<double> *const *const *const,
                                const unsigned int nk,
                                const unsigned int nk_interpolate,
                                double **xk_in);

    void compute_del2_v2_del_umn2(std::complex<double> ***,
                                  const std::complex<double> *const *const *const,
                                  const unsigned int nk,
                                  double **xk_in);

    void compute_del_v3_del_umn(std::complex<double> ****del_v3_del_umn,
                                double **omega2_harmonic,
                                const std::complex<double> *const *const *const evec_harmonic,
                                const KpointMeshUniform *kmesh_coarse_in,
                                const KpointMeshUniform *kmesh_dense_in,
                                const PhaseFactorStorage *phase_storage_in);

    void calculate_delv2_delumn_finite_difference(double **omega2_harmonic,
                                                  const std::complex<double> *const *const *const,
                                                  std::complex<double> ***,
                                                  const KpointMeshUniform *kmesh_coarse,
                                                  const KpointMeshUniform *kmesh_dense,
                                                  MinimumDistList ***mindist_list);

    void read_del_v2_del_umn_in_kspace(double **omega2_harmonic,
                                       const std::complex<double> *const *const *const,
                                       std::complex<double> ***,
                                       const unsigned int nk,
                                       const unsigned int nk_interpolate);

    void calculate_delv1_delumn_finite_difference(std::complex<double> **,
                                                  const std::complex<double> *const *const *const);

    void compute_del_v_strain_in_real_space1(const std::vector<FcsArrayWithCell> &fcs_in,
                                             std::vector<FcsArrayWithCell> &delta_fcs,
                                             const int ixyz1,
                                             const int ixyz2,
                                             const int periodic_image_mode);

    void compute_del_v_strain_in_real_space2(const std::vector<FcsArrayWithCell> &,
                                             std::vector<FcsArrayWithCell> &,
                                             const int,
                                             const int,
                                             const int,
                                             const int,
                                             const int);


    void make_supercell_mapping_by_symmetry_operations(int **);

    void make_inverse_translation_mapping(int **);


};
}



