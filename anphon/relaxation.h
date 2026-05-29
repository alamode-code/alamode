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
{
public:
    using MatrixXcdRowMajor = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    MatrixXcdRowMajor del_v1;                           // [9][ns]
    MatrixXcdRowMajor del2_v1;                          // [81][ns]
    MatrixXcdRowMajor del3_v1;                          // [729][ns]
    std::vector<MatrixXcdRowMajor> del_v2;              // [9][nk][ns*ns]
    std::vector<MatrixXcdRowMajor> del2_v2;             // [81][nk][ns*ns]
    std::vector<std::vector<MatrixXcdRowMajor>> del_v3; // [9][nk][ns][ns*ns]

    DelVStrainData() = default;
    ~DelVStrainData() = default;

    void resize(const int nk, const int nmode)
    {
        nk_ = nk;
        nmode_ = nmode;
        const auto nmode2 = nmode * nmode;

        del_v1.resize(9, nmode);
        del2_v1.resize(81, nmode);
        del3_v1.resize(729, nmode);

        del_v2.resize(9);
        for (auto &mat: del_v2) {
            mat.resize(nk, nmode2);
        }

        del2_v2.resize(81);
        for (auto &mat: del2_v2) {
            mat.resize(nk, nmode2);
        }

        del_v3.resize(9);
        for (auto &per_strain: del_v3) {
            per_strain.resize(nk);
            for (auto &mat: per_strain) {
                mat.resize(nmode, nmode2);
            }
        }

        build_pointer_views();
    }

    int nk() const
    {
        return nk_;
    }
    int nmode() const
    {
        return nmode_;
    }

    std::complex<double> **del_v1_raw()
    {
        return del_v1_rows_.data();
    }
    std::complex<double> *const *del_v1_raw() const
    {
        return del_v1_rows_.data();
    }
    std::complex<double> **del2_v1_raw()
    {
        return del2_v1_rows_.data();
    }
    std::complex<double> *const *del2_v1_raw() const
    {
        return del2_v1_rows_.data();
    }
    std::complex<double> **del3_v1_raw()
    {
        return del3_v1_rows_.data();
    }
    std::complex<double> *const *del3_v1_raw() const
    {
        return del3_v1_rows_.data();
    }
    std::complex<double> ***del_v2_raw()
    {
        return del_v2_ptrs_.data();
    }
    std::complex<double> **const *del_v2_raw() const
    {
        return del_v2_ptrs_.data();
    }
    std::complex<double> ***del2_v2_raw()
    {
        return del2_v2_ptrs_.data();
    }
    std::complex<double> **const *del2_v2_raw() const
    {
        return del2_v2_ptrs_.data();
    }
    std::complex<double> ****del_v3_raw()
    {
        return del_v3_ptrs_.data();
    }
    std::complex<double> ***const *del_v3_raw() const
    {
        return del_v3_ptrs_.data();
    }

private:
    int nk_{0};
    int nmode_{0};

    std::vector<std::complex<double> *> del_v1_rows_;
    std::vector<std::complex<double> *> del2_v1_rows_;
    std::vector<std::complex<double> *> del3_v1_rows_;

    std::vector<std::vector<std::complex<double> *>> del_v2_rows_;
    std::vector<std::complex<double> **> del_v2_ptrs_;

    std::vector<std::vector<std::complex<double> *>> del2_v2_rows_;
    std::vector<std::complex<double> **> del2_v2_ptrs_;

    std::vector<std::vector<std::vector<std::complex<double> *>>> del_v3_rows_;
    std::vector<std::vector<std::complex<double> **>> del_v3_kptrs_;
    std::vector<std::complex<double> ***> del_v3_ptrs_;

    void build_pointer_views()
    {
        const auto nmode2 = nmode_ * nmode_;

        del_v1_rows_.resize(9);
        for (int i = 0; i < 9; ++i) {
            del_v1_rows_[i] = del_v1.data() + static_cast<std::size_t>(i) * nmode_;
        }

        del2_v1_rows_.resize(81);
        for (int i = 0; i < 81; ++i) {
            del2_v1_rows_[i] = del2_v1.data() + static_cast<std::size_t>(i) * nmode_;
        }

        del3_v1_rows_.resize(729);
        for (int i = 0; i < 729; ++i) {
            del3_v1_rows_[i] = del3_v1.data() + static_cast<std::size_t>(i) * nmode_;
        }

        del_v2_rows_.resize(9);
        del_v2_ptrs_.resize(9);
        for (int i = 0; i < 9; ++i) {
            del_v2_rows_[i].resize(nk_);
            for (int ik = 0; ik < nk_; ++ik) {
                del_v2_rows_[i][ik] = del_v2[i].data() + static_cast<std::size_t>(ik) * nmode2;
            }
            del_v2_ptrs_[i] = del_v2_rows_[i].data();
        }

        del2_v2_rows_.resize(81);
        del2_v2_ptrs_.resize(81);
        for (int i = 0; i < 81; ++i) {
            del2_v2_rows_[i].resize(nk_);
            for (int ik = 0; ik < nk_; ++ik) {
                del2_v2_rows_[i][ik] = del2_v2[i].data() + static_cast<std::size_t>(ik) * nmode2;
            }
            del2_v2_ptrs_[i] = del2_v2_rows_[i].data();
        }

        del_v3_rows_.resize(9);
        del_v3_kptrs_.resize(9);
        del_v3_ptrs_.resize(9);
        for (int i = 0; i < 9; ++i) {
            del_v3_rows_[i].resize(nk_);
            del_v3_kptrs_[i].resize(nk_);
            for (int ik = 0; ik < nk_; ++ik) {
                del_v3_rows_[i][ik].resize(nmode_);
                for (int is = 0; is < nmode_; ++is) {
                    del_v3_rows_[i][ik][is] = del_v3[i][ik].data() + static_cast<std::size_t>(is) * nmode2;
                }
                del_v3_kptrs_[i][ik] = del_v3_rows_[i][ik].data();
            }
            del_v3_ptrs_[i] = del_v3_kptrs_[i].data();
        }
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
                              DelVStrainData &del_v_strain, double **omega2_harmonic,
                              std::complex<double> ***evec_harmonic, RelaxationStrMode relax_mode,
                              MinimumDistList ***mindist_list, const PhaseFactorStorage *phase_storage_in);

    void setInitialDistortion(const double (*u_tensor_in)[3]);

    void load_V0_from_file();

    void store_V0_to_file() const;

    void set_init_structure_atT(RelaxationStructureState &structure_state, bool &converged_prev, int &str_diverged,
                                const int i_temp_loop, double **omega2_harmonic,
                                std::complex<double> ***evec_harmonic) const;


    void set_elastic_constants(double *C1_array, double **C2_array, double ***C3_array) const;

    static void renormalize_v0_from_umn(double &v0_with_umn, double v0_ref,
                                        std::array<std::array<double, 3>, 3> &eta_tensor, double *C1_array,
                                        double **C2_array, double ***C3_array,
                                        const std::array<std::array<double, 3>, 3> &u_tensor, const double pvcell);

    void renormalize_v1_from_umn(std::complex<double> *, const std::complex<double> *const, const DelVStrainData &,
                                 const std::array<std::array<double, 3>, 3> &) const;

    void renormalize_v2_from_umn(const KpointMeshUniform *kmesh_coarse, const std::vector<int> &kmap_coarse_to_dense,
                                 std::complex<double> **, const DelVStrainData &,
                                 const std::array<std::array<double, 3>, 3> &) const;

    void renormalize_v3_from_umn(const KpointMeshUniform *kmesh_coarse, const KpointMeshUniform *kmesh_dense,
                                 std::complex<double> ***, std::complex<double> ***, const DelVStrainData &,
                                 const std::array<std::array<double, 3>, 3> &) const;

    void renormalize_v1_from_q0(double **omega2_harmonic, const KpointMeshUniform *kmesh_coarse,
                                const KpointMeshUniform *kmesh_dense, std::complex<double> *, std::complex<double> *,
                                std::complex<double> **, std::complex<double> ***, std::complex<double> ***,
                                const std::vector<double> &) const;

    void renormalize_v2_from_q0(std::complex<double> ***evec_harmonic, const KpointMeshUniform *kmesh_coarse,
                                const KpointMeshUniform *kmesh_dense, const std::vector<int> &kmap_coarse_to_dense,
                                std::complex<double> ****mat_transform_sym, std::complex<double> **delta_v2_renorm,
                                std::complex<double> **delta_v2_array_original, std::complex<double> ***v3_ref,
                                std::complex<double> ***v4_ref, const std::vector<double> &) const;

    void renormalize_v3_from_q0(const KpointMeshUniform *kmesh_dense, const KpointMeshUniform *kmesh_coarse,
                                std::complex<double> ***, std::complex<double> ***, std::complex<double> ***,
                                const std::vector<double> &) const;

    void renormalize_v0_from_q0(double **omega2_harmonic, const KpointMeshUniform *kmesh_dense, double &, double,
                                std::complex<double> *, std::complex<double> **, std::complex<double> ***,
                                std::complex<double> ***, const std::vector<double> &) const;

    void calculate_u0(const double *const q0, double *const u0, double **omega2_harmonic,
                      std::complex<double> ***evec_harmonic) const;
    void calculate_u0(const std::vector<double> &q0, std::vector<double> &u0, double **omega2_harmonic,
                      std::complex<double> ***evec_harmonic) const;

    void update_cell_coordinate(RelaxationStructureState &, const std::complex<double> *const,
                                const double *const *const, const std::complex<double> *const,
                                const double *const *const, const std::complex<double> *const *const *const,
                                const std::vector<int> &, double **omega2_harmonic,
                                std::complex<double> ***evec_harmonic) const;

    void check_str_divergence(int &diverged, const RelaxationStructureState &structure_state) const;


    void write_resfile_header(std::ofstream &fout_q0, std::ofstream &fout_u0, std::ofstream &fout_u_tensor) const;

    void write_resfile_atT(const RelaxationStructureState &structure_state, const double temperature,
                           std::ofstream &fout_q0, std::ofstream &fout_u0, std::ofstream &fout_u_tensor) const;

    void write_stepresfile_header_atT(std::ofstream &fout_step_q0, std::ofstream &fout_step_u0,
                                      std::ofstream &fout_step_u_tensor, const double temp) const;

    void write_stepresfile(const RelaxationStructureState &structure_state, const int i_str_loop,
                           std::ofstream &fout_step_q0, std::ofstream &fout_step_u0,
                           std::ofstream &fout_step_u_tensor) const;

    static int get_xyz_string(const int, std::string &);

    static void calculate_eta_tensor(std::array<std::array<double, 3>, 3> &,
                                     const std::array<std::array<double, 3>, 3> &);

private:
    void set_default_variables();

    void deallocate_variables();

    static void read_C1_array(double *const);

    void read_elastic_constants(double *const *const, double *const *const *const) const;

    void set_initial_q0(std::vector<double> &q0, std::complex<double> ***evec_harmonic) const;


    void set_initial_strain(std::array<std::array<double, 3>, 3> &u_tensor) const;
};
} // namespace PHON_NS
