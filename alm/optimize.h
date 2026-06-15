/*
 optimize.h

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <vector>
#include "constraint.h"
#include "fcs.h"
#include "files.h"
#include "symmetry.h"
#include "timer.h"

using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor>;


namespace ALM_NS
{
class OptimizerControl
{
public:
    // General optimization options
    int linear_model;         // 1 : least-squares, 2 : elastic net, 3 : adaptive lasso (experimental)
    int use_sparse_solver;    // 0: No, 1: Yes
    std::string sparsesolver; // Method name of Eigen sparse solver
    int use_cholesky;         // 0: No, 1: Yes
    int chunk_size;           // chunk size used for the decomposed computation of (A^T A)
    int maxnum_iteration;
    double tolerance_iteration;
    int output_frequency;

    // Options related to L1-regularized optimization
    int standardize;
    double displacement_normalization_factor;
    int debiase_after_l1opt;

    // Energy-difference loss term: weight w in  ||A_F θ − b_F||² + w² ||Ã_E θ − Ẽ_ref||².
    // 0 disables the term (default; behavior identical to the force-only fit).
    double efit_weight;

    // cross-validation related variables
    int cross_validation; // 0 : No CV mode, -1 or > 0: CV mode
    double l1_alpha;      // L1-regularization coefficient
    double l1_alpha_min;
    double l1_alpha_max;
    double l1_alpha_min_ratio; // l1_alpha_min = l1_alpha_max * l1_alpha_min_ratio
    int num_l1_alpha;
    double l1_ratio; // l1_ratio = 1 for LASSO; 0 < l1_ratio < 1 for Elastic net
    int save_solution_path;
    int stop_criterion; // If stop_criterion > 0,
    // the solution path calculation stops when the validation error
    // increases for `stop_criterion` times consecutively.

    // convention to assign IFCs to periodic images
    int periodic_image_conv;


    OptimizerControl()
    {
        linear_model = 1;
        use_sparse_solver = 0;
        sparsesolver = "SimplicialLDLT";
        maxnum_iteration = 10000;
        tolerance_iteration = 1.0e-8;
        output_frequency = 1000;
        standardize = 1;
        displacement_normalization_factor = 1.0;
        debiase_after_l1opt = 0;
        efit_weight = 0.0;
        cross_validation = 0;
        l1_alpha = 0.0;
        l1_alpha_min = -1.0; // Recommended l1_alpha_max * 1e-6
        l1_alpha_max = -1.0; // Use recommended value
        l1_alpha_min_ratio = 1.0e-6;
        l1_ratio = 1.0;
        num_l1_alpha = 50;
        save_solution_path = 0;
        stop_criterion = 5;
        periodic_image_conv = 1;
        use_cholesky = 0;
        chunk_size = 100;
    }

    ~OptimizerControl() = default;

    OptimizerControl(const OptimizerControl &obj) = default;

    OptimizerControl &operator=(const OptimizerControl &obj) = default;
};

class SensingMatrix
{
    // A class storing matrix information necessary for linear algebra solvers
public:
    std::vector<double> amat_dense;      // Sensing matrix A (dense)
    std::vector<double> bvec;            // vector b
    std::vector<double> original_forces; // stored to compute the relative errors
    SpMat amat_sparse;                   // Sensing matrix A (sparse form)
};

class Optimize
{
public:
    Optimize();

    ~Optimize();

    auto optimize_main(const std::unique_ptr<Symmetry> &symmetry, std::unique_ptr<Constraint> &constraint,
                       const std::unique_ptr<Fcs> &fcs, const int maxorder, const std::string &file_prefix,
                       const std::vector<std::string> &str_order, const int verbosity,
                       const DispForceFile &filedata_train, const DispForceFile &filedata_validation,
                       const int output_maxorder, std::unique_ptr<Timer> &timer) -> int;

    auto set_u_train(const std::vector<std::vector<double>> &u_train_in) -> void;

    auto set_f_train(const std::vector<std::vector<double>> &f_train_in) -> void;

    auto set_e_train(const std::vector<double> &e_train_in) -> void;

    // Phase 2b: build the centered, constraint-compacted energy sensing matrix and target for the
    // training configs. amat_energy_out is row-major [M_E * ncols_compact]; evec_out length M_E
    // holds w·(centered (E_ref + e_rhs)). Columns of amat_energy_out are already ·w and centered.
    auto build_energy_matrix(const std::unique_ptr<Symmetry> &symmetry, const std::unique_ptr<Fcs> &fcs,
                             const std::unique_ptr<Constraint> &constraint, const int maxorder,
                             const size_t ncols_compact, std::vector<double> &amat_energy_out,
                             std::vector<double> &evec_out, const int verbosity) const -> void;

    auto set_validation_data(const std::vector<std::vector<double>> &u_validation_in,
                             const std::vector<std::vector<double>> &f_validation_in) -> void;

    [[nodiscard]] auto get_u_train() const -> std::vector<std::vector<double>>;

    [[nodiscard]] auto get_f_train() const -> std::vector<std::vector<double>>;

    [[nodiscard]] auto get_number_of_data() const -> size_t;

    auto get_matrix_elements_unified(const int maxorder, std::unique_ptr<SensingMatrix> &matrix_out,
                                     const std::vector<std::vector<double>> &u_in,
                                     const std::vector<std::vector<double>> &f_in,
                                     const std::unique_ptr<Symmetry> &symmetry, const std::unique_ptr<Fcs> &fcs,
                                     const std::unique_ptr<Constraint> &constraint, const bool compact,
                                     const bool sparse, const bool return_ata, const int verbosity = 0) const -> void;

    auto set_fcs_values(const int maxorder, double *fc_in, std::vector<size_t> *nequiv,
                        const std::unique_ptr<Constraint> &constraint) -> void;

    [[nodiscard]] auto get_number_of_rows_sensing_matrix() const -> size_t;

    [[nodiscard]] auto get_params() const -> double *;

    auto set_optimizer_control(const OptimizerControl &) -> void;

    [[nodiscard]] auto get_optimizer_control() const -> OptimizerControl;

    [[nodiscard]] auto get_cv_l1_alpha() const -> double;

private:
    double *params;
    double cv_l1_alpha; // stores alpha at minimum CV

    std::vector<std::vector<double>> u_train, f_train;
    std::vector<std::vector<double>> u_validation, f_validation;
    std::vector<double> e_train;  // reference (DFT) total-supercell energies, Ry, one per training config

    OptimizerControl optcontrol;

    auto set_default_variables() -> void;

    auto deallocate_variables() -> void;

    auto data_multiplier(const std::vector<std::vector<double>> &data_in, std::vector<std::vector<double>> &data_out,
                         const std::unique_ptr<Symmetry> &symmetry) const -> void;

    static auto inprim_index(const int n, const std::unique_ptr<Symmetry> &symmetry) -> int;

    auto least_squares(const int maxorder, const size_t N, const size_t N_new, const size_t M, const int verbosity,
                       const std::unique_ptr<Symmetry> &symmetry, const std::unique_ptr<Fcs> &fcs,
                       const std::unique_ptr<Constraint> &constraint, std::vector<double> &param_out) const -> int;

    auto compressive_sensing(const std::string &job_prefix, const int maxorder, const size_t N_new, const size_t M,
                             const std::unique_ptr<Symmetry> &symmetry, const std::vector<std::string> &str_order,
                             const std::unique_ptr<Fcs> &fcs, std::unique_ptr<Constraint> &constraint,
                             const int verbosity, std::vector<double> &param_out) -> int;

    auto crossvalidation(const std::string &job_prefix, const int maxorder, const std::unique_ptr<Fcs> &fcs,
                         const std::unique_ptr<Symmetry> &symmetry, const std::unique_ptr<Constraint> &constraint,
                         const int verbosity) -> double;

    auto run_manual_cv(const std::string &job_prefix, const int maxorder, const std::unique_ptr<Fcs> &fcs,
                       const std::unique_ptr<Symmetry> &symmetry, const std::unique_ptr<Constraint> &constraint,
                       const int verbosity) const -> double;

    auto run_auto_cv(const std::string &job_prefix, const int maxorder, const std::unique_ptr<Fcs> &fcs,
                     const std::unique_ptr<Symmetry> &symmetry, const std::unique_ptr<Constraint> &constraint,
                     const int verbosity) -> double;

    auto write_cvresult_to_file(const std::string &file_out, const std::vector<double> &alphas,
                                const std::vector<double> &training_error, const std::vector<double> &validation_error,
                                const std::vector<std::vector<int>> &nonzeros) const -> void;

    auto write_cvscore_to_file(const std::string &file_out, const std::vector<double> &alphas,
                               const std::vector<double> &terr_mean, const std::vector<double> &terr_std,
                               const std::vector<double> &verr_mean, const std::vector<double> &verr_std,
                               const int ialpha_minimum, const size_t nsets) const -> void;

    auto set_errors_of_cvscore(std::vector<double> &terr_mean, std::vector<double> &terr_std,
                               std::vector<double> &verr_mean, std::vector<double> &verr_std,
                               const std::vector<std::vector<double>> &training_error_accum,
                               const std::vector<std::vector<double>> &validation_error_accum) const -> void;

    static auto get_ialpha_at_minimum_validation_error(const std::vector<double> &validation_error) -> int;

    auto optimize_with_given_l1alpha(const int maxorder, const size_t M, const size_t N_new,
                                     const std::unique_ptr<Fcs> &fcs, const std::unique_ptr<Symmetry> &symmetry,
                                     const std::unique_ptr<Constraint> &constraint, const int verbosity,
                                     std::vector<double> &param_out) const -> void;

    auto run_least_squares_with_nonzero_coefs(const Eigen::MatrixXd &A_in, const Eigen::VectorXd &b_in,
                                              const Eigen::VectorXd &factor_std, std::vector<double> &params_inout,
                                              const int verbosity) const -> void;

    static auto get_number_of_zero_coefs(const int maxorder, const std::unique_ptr<Constraint> &constraint,
                                         const Eigen::VectorXd &x, std::vector<int> &nzeros) -> void;

    auto get_standardizer(const Eigen::MatrixXd &Amat, Eigen::VectorXd &mean, Eigen::VectorXd &dev,
                          Eigen::VectorXd &factor_std, Eigen::VectorXd &scale_beta) const -> void;

    auto apply_standardizer(Eigen::MatrixXd &Amat, const Eigen::VectorXd &mean,
                            const Eigen::VectorXd &dev) const -> void;

    [[nodiscard]] auto get_estimated_max_alpha(const Eigen::MatrixXd &Amat,
                                               const Eigen::VectorXd &bvec) const -> double;

    static auto apply_scaler_displacement(std::vector<std::vector<double>> &u_inout, const double normalization_factor,
                                          const bool scale_back = false) -> void;

    static auto apply_scaler_constraint(const int maxorder, const double normalization_factor,
                                        const std::unique_ptr<Constraint> &constraint,
                                        const bool scale_back = false) -> void;

    static auto apply_scaler_force_constants(const int maxorder, const double normalization_factor,
                                             const std::unique_ptr<Constraint> &constraint,
                                             std::vector<double> &param_inout) -> void;

    auto apply_scalers(const int maxorder, const std::unique_ptr<Constraint> &constraint) -> void;

    auto finalize_scalers(const int maxorder, const std::unique_ptr<Constraint> &constraint) -> void;

    static auto apply_basis_converter(std::vector<std::vector<double>> &u_multi, Eigen::Matrix3d cmat) -> void;

    static auto apply_basis_converter_amat(const int natmin3, const int ncols, double **amat_orig_tmp,
                                           Eigen::Matrix3d cmat) -> void;

    auto fit_algebraic_constraints(const size_t N, const size_t M, double *amat, const double *bvec,
                                   std::vector<double> &param_out, const double fnorm, const int maxorder,
                                   const std::unique_ptr<Fcs> &fcs, const std::unique_ptr<Constraint> &constraint,
                                   const int verbosity) const -> int;

    auto solve_normal_equation(const size_t N, double *amat, double *bvec, std::vector<double> &param_out,
                               const double fnorm, const int maxorder, const std::unique_ptr<Fcs> &fcs,
                               const std::unique_ptr<Constraint> &constraint, const int verbosity,
                               const bool algebraic_constraint) const -> int;


    auto get_matrix_elements2(const int maxorder, const size_t ncycle, const size_t nrows, const size_t ncols,
                              const size_t ncols_compact, std::unique_ptr<SensingMatrix> &matrix_out,
                              const std::vector<std::vector<double>> &u_multi,
                              const std::vector<std::vector<double>> &f_multi,
                              const std::vector<std::vector<double>> &gamma_precomputed,
                              const std::unique_ptr<Symmetry> &symmetry, const std::unique_ptr<Fcs> &fcs,
                              const std::unique_ptr<Constraint> &constraint, const bool sparse) const -> void;

    auto get_matrix_elements_normal_equation2(
        const int maxorder, const size_t ncycle, const size_t nrows, const size_t ncols, const size_t ncols_compact,
        std::unique_ptr<SensingMatrix> &matrix_out, const std::vector<std::vector<double>> &u_multi,
        const std::vector<std::vector<double>> &f_multi, const std::vector<std::vector<double>> &gamma_precomputed,
        const std::unique_ptr<Symmetry> &symmetry, const std::unique_ptr<Fcs> &fcs,
        const std::unique_ptr<Constraint> &constraint, const bool sparse) const -> void;

    static auto fill_bvec(const size_t natmin, const size_t irow, const std::vector<std::vector<int>> &index_mapping,
                          const std::vector<double> &f_sub, std::vector<double> &bvec) -> void;

    static auto fill_amat(const int maxorder, const size_t natmin, const size_t ncols, const std::vector<double> &u_sub,
                          const std::vector<std::vector<double>> &gamma_precomputed,
                          const std::unique_ptr<Symmetry> &symmetry, const std::unique_ptr<Fcs> &fcs,
                          double **&amat_orig) -> void;

    // Phase 1 (energy term): build a single energy row for one displacement image.
    // energy_row[iparam] += gamma_energy_precomputed * prod_{j=0..order+1} u_sub[elems[j]]
    // (product over ALL order+2 indices; cf. fill_amat which drops elems[0] for the force).
    static auto fill_amat_energy(const int maxorder, const size_t ncols, const std::vector<double> &u_sub,
                                 const std::vector<std::vector<double>> &gamma_energy_precomputed,
                                 const std::unique_ptr<Fcs> &fcs, std::vector<double> &energy_row) -> void;

    // Per-entry energy multiplicity factor = gamma(n,arr)/n (NOT 1/denom; the two coincide only
    // for diagonal clusters). Matches tools/taylor.py (E_order *= 1/order). Caller multiplies by
    // fc_table.sign, exactly as for the force gamma table.
    [[nodiscard]] auto gamma_energy(const int n, const int *arr) const -> double;

    // Self-contained verification (env ALM_ENERGY_SELFTEST): checks the energy-row builder
    // against the trusted force builder via the per-order Euler identity
    // E_order == -(1/n) * sum_a u_a F_a(theta).  Returns true on PASS. Does not touch the fit path.
    auto run_energy_selftest(const std::unique_ptr<Symmetry> &symmetry, const std::unique_ptr<Fcs> &fcs,
                             const std::unique_ptr<Constraint> &constraint, const int maxorder,
                             const int verbosity) const -> bool;

    static auto project_constraints(const int maxorder, const size_t natmin, const size_t irow,
                                    const std::unique_ptr<Fcs> &fcs, const std::unique_ptr<Constraint> &constraint,
                                    double **amat_orig, double **&amat_mod, std::vector<double> &bvec_mod) -> void;

    // Phase 2: project one full-basis energy row (e_full, length ncols) into the constraint-compacted
    // basis (e_compact, length ncols_compact), moving the FC2FIX-fixed-coefficient energy onto the
    // RHS scalar e_rhs. Single-row analogue of project_constraints' const_fix/index_bimap/const_relate.
    static auto project_energy_row(const int maxorder, const std::unique_ptr<Fcs> &fcs,
                                   const std::unique_ptr<Constraint> &constraint,
                                   const std::vector<double> &e_full, std::vector<double> &e_compact,
                                   double &e_rhs) -> void;

    auto run_eigen_sparse_solver(const SpMat &sp_mat, const Eigen::VectorXd &sp_bvec, std::vector<double> &param_out,
                                 const double fnorm, const int maxorder, const std::unique_ptr<Fcs> &fcs,
                                 const std::unique_ptr<Constraint> &constraint, const std::string &solver_type,
                                 const int verbosity) const -> int;

    auto recover_original_forceconstants(const int maxorder, const std::vector<double> &param_in,
                                         std::vector<double> &param_out, const std::vector<size_t> *nequiv,
                                         const std::unique_ptr<Constraint> &constraint) const -> void;

    [[nodiscard]] auto factorial(const int) const -> int;

    auto gamma(const int, const int *) const -> double;

    auto coordinate_descent(const int M, const int N, const double alpha, const int warm_start, Eigen::VectorXd &x,
                            const Eigen::MatrixXd &A, const Eigen::VectorXd &b, const Eigen::VectorXd &grad0,
                            bool *has_prod, Eigen::MatrixXd &Prod, Eigen::VectorXd &grad, const double fnorm,
                            const Eigen::VectorXd &scale_beta, const int verbosity) const -> void;

    auto solution_path(const int maxorder, Eigen::MatrixXd &A, Eigen::VectorXd &b, Eigen::MatrixXd &A_validation,
                       Eigen::VectorXd &b_validation, const double fnorm, const double fnorm_validation,
                       const std::string &file_coef, const int verbosity, const std::unique_ptr<Constraint> &constraint,
                       const std::vector<double> &alphas, std::vector<double> &training_error,
                       std::vector<double> &validation_error, std::vector<std::vector<int>> &nonzeros) const -> void;

    static auto compute_alphas(const double l1_alpha_max, const double l1_alpha_min, const int num_l1_alpha,
                               std::vector<double> &alphas) -> void;
};

inline auto shrink(const double x, const double a) -> double
{
    const auto xabs = std::abs(x);
    const auto sign = static_cast<double>((0.0 < x) - (x < 0.0));
    return sign * std::max<double>(xabs - a, 0.0);
}

} // namespace ALM_NS
