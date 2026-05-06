/*
 relaxation.h

 Copyright (c) 2022 Ryota Masuki, Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <Eigen/Core>
#include <complex>
#include <memory>
#include "kpoint.h"
#include "optimizers.h"
#include "pointers.h"
#include "relaxation_types.h"
#include "scph.h"

namespace PHON_NS
{

class DerivativeIFC;

class DelVStrainData
// TODO: implement the class for derivative of V by umn
{
public:
    Eigen::MatrixXcd del_v1;                           // del_v1_del_umn
    Eigen::MatrixXcd del2_v1;                          // del2_v1_del_umn2
    Eigen::MatrixXcd del3_v1;                          // del3_v1_del_umn3
    std::vector<Eigen::MatrixXcd> del_v2;              // del_v2_del_umn
    std::vector<Eigen::MatrixXcd> del2_v2;             // del2_v2_del_umn2
    std::vector<std::vector<Eigen::MatrixXcd>> del_v3; // del_v3_del_umn

    DelVStrainData() = default;
    ~DelVStrainData() = default;

    void resize(const int nk, const int nmode)
    {
        del_v1.resize(9, nmode);
        del2_v1.resize(81, nmode);
        del3_v1.resize(729, nmode);
    }
};

class Relaxation: protected Pointers
{
public:
    Relaxation(class PHON *phon);

    ~Relaxation();

    int relax_str;

    // initial strain and displacement
    double init_u_tensor[3][3]{{0.0}};
    std::vector<double> init_u0;

    // zeroth order term of the potential energy surface
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
    int cooling_u0_index;  // used if set_init_str is 3
    double cooling_u0_thr; // used if set_init_str is 3
    double add_hess_diag;
    double stat_pressure;

    int renorm_3to2nd;
    int renorm_2to1st;
    int renorm_34to1st;
    std::string strain_IFC_dir;

    std::unique_ptr<Optimizer> optimizer;
    std::unique_ptr<DerivativeIFC> derivative_ifc;

    void create_optimizer(const size_t num_modes);

    void setup_relaxation();

    void compute_del_v_strain(const KpointMeshUniform *kmesh_coarse, const KpointMeshUniform *kmesh_dense,
                              std::complex<double> **del_v1_del_umn, std::complex<double> **del2_v1_del_umn2,
                              std::complex<double> **del3_v1_del_umn3, std::complex<double> ***del_v2_del_umn,
                              std::complex<double> ***del2_v2_del_umn2, std::complex<double> ****del_v3_del_umn,
                              double **omega2_harmonic, std::complex<double> ***evec_harmonic,
                              RelaxationStrMode relax_mode,
                              MinimumDistList ***mindist_list, const PhaseFactorStorage *phase_storage_in);

    void setInitialDistortion(const double (*u_tensor_in)[3]);

    void load_V0_from_file();

    void store_V0_to_file() const;

    void set_init_structure_atT(double *q0, double **u_tensor, double *u0, bool &converged_prev, int &str_diverged,
                                const int i_temp_loop, double **omega2_harmonic,
                                std::complex<double> ***evec_harmonic) const;


    void set_elastic_constants(double *C1_array, double **C2_array, double ***C3_array) const;

    static void renormalize_v0_from_umn(double &, double, double **, double *, double **, double ***, double **,
                                        const double);

    void renormalize_v1_from_umn(std::complex<double> *, const std::complex<double> *const,
                                 const std::complex<double> *const *const, const std::complex<double> *const *const,
                                 const std::complex<double> *const *const, const double *const *const) const;

    void renormalize_v2_from_umn(const KpointMeshUniform *kmesh_coarse, const std::vector<int> &kmap_coarse_to_dense,
                                 std::complex<double> **, std::complex<double> ***, std::complex<double> ***,
                                 double **) const;

    void renormalize_v3_from_umn(const KpointMeshUniform *kmesh_coarse, const KpointMeshUniform *kmesh_dense,
                                 std::complex<double> ***, std::complex<double> ***, std::complex<double> ****,
                                 double **) const;

    void renormalize_v1_from_q0(double **omega2_harmonic, const KpointMeshUniform *kmesh_coarse,
                                const KpointMeshUniform *kmesh_dense, std::complex<double> *, std::complex<double> *,
                                std::complex<double> **, std::complex<double> ***, std::complex<double> ***,
                                double *) const;

    void renormalize_v2_from_q0(std::complex<double> ***evec_harmonic, const KpointMeshUniform *kmesh_coarse,
                                const KpointMeshUniform *kmesh_dense, const std::vector<int> &kmap_coarse_to_dense,
                                std::complex<double> ****mat_transform_sym, std::complex<double> **delta_v2_renorm,
                                std::complex<double> **delta_v2_array_original, std::complex<double> ***v3_ref,
                                std::complex<double> ***v4_ref, double *q0) const;

    void renormalize_v3_from_q0(const KpointMeshUniform *kmesh_dense, const KpointMeshUniform *kmesh_coarse,
                                std::complex<double> ***, std::complex<double> ***, std::complex<double> ***,
                                double *) const;

    void renormalize_v0_from_q0(double **omega2_harmonic, const KpointMeshUniform *kmesh_dense, double &, double,
                                std::complex<double> *, std::complex<double> **, std::complex<double> ***,
                                std::complex<double> ***, double *) const;

    void calculate_u0(const double *const q0, double *const u0, double **omega2_harmonic,
                      std::complex<double> ***evec_harmonic) const;

    void update_cell_coordinate(double *, double *, double **, const std::complex<double> *const,
                                const double *const *const, const std::complex<double> *const,
                                const double *const *const, const std::complex<double> *const *const *const,
                                const std::vector<int> &, double *, double *, double *, double &, double &,
                                double **omega2_harmonic, std::complex<double> ***evec_harmonic) const;

    void check_str_divergence(int &diverged, const double *const q0, const double *const u0,
                              const double *const *const u_tensor) const;


    void write_resfile_header(std::ofstream &fout_q0, std::ofstream &fout_u0, std::ofstream &fout_u_tensor) const;

    void write_resfile_atT(const double *const q0, const double *const *const u_tensor, const double *const u0,
                           const double temperature, std::ofstream &fout_q0, std::ofstream &fout_u0,
                           std::ofstream &fout_u_tensor) const;

    void write_stepresfile_header_atT(std::ofstream &fout_step_q0, std::ofstream &fout_step_u0,
                                      std::ofstream &fout_step_u_tensor, const double temp) const;

    void write_stepresfile(const double *const q0, const double *const *const u_tensor, const double *const u0,
                           const int i_str_loop, std::ofstream &fout_step_q0, std::ofstream &fout_step_u0,
                           std::ofstream &fout_step_u_tensor) const;

    static int get_xyz_string(const int, std::string &);

    static void calculate_eta_tensor(double **, const double *const *const);

private:
    void set_default_variables();

    void deallocate_variables();

    static void read_C1_array(double *const);

    void read_elastic_constants(double *const *const, double *const *const *const) const;

    void set_initial_q0(double *const q0, std::complex<double> ***evec_harmonic) const;


    void set_initial_strain(double *const *const) const;
};
} // namespace PHON_NS
