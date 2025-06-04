/*
 optimize.h

 Copyright (c) 2014, 2015, 2016 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <vector>
#include "files.h"
#include "constraint.h"
#include "symmetry.h"
#include "fcs.h"
#include "timer.h"
#include <Eigen/Dense>
#include <Eigen/SparseCore>

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

    // cross-validation related variables
    int cross_validation; // 0 : No CV mode, -1 or > 0: CV mode
    double l1_alpha;      // L1-regularization coefficient
    double l1_alpha_min;
    double l1_alpha_max;
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
        cross_validation = 0;
        l1_alpha = 0.0;
        l1_alpha_min = -1.0; // Recommended l1_alpha_max * 1e-6
        l1_alpha_max = -1.0; // Use recommended value
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

    int optimize_main(const std::unique_ptr<Symmetry> &symmetry,
                      std::unique_ptr<Constraint> &constraint,
                      std::unique_ptr<Fcs> &fcs,
                      const int maxorder,
                      const std::string &file_prefix,
                      const std::vector<std::string> &str_order,
                      const int verbosity,
                      const DispForceFile &filedata_train,
                      const DispForceFile &filedata_validation,
                      const int output_maxorder,
                      std::unique_ptr<Timer> &timer);

    void set_u_train(const std::vector<std::vector<double>> &u_train_in);

    void set_f_train(const std::vector<std::vector<double>> &f_train_in);

    void set_validation_data(const std::vector<std::vector<double>> &u_validation_in,
                             const std::vector<std::vector<double>> &f_validation_in);

    [[nodiscard]] std::vector<std::vector<double>> get_u_train() const;

    [[nodiscard]] std::vector<std::vector<double>> get_f_train() const;

    [[nodiscard]] size_t get_number_of_data() const;

    void get_matrix_elements_unified(const int maxorder,
                                     std::unique_ptr<SensingMatrix> &matrix_out,
                                     const std::vector<std::vector<double>> &u_in,
                                     const std::vector<std::vector<double>> &f_in,
                                     const std::unique_ptr<Symmetry> &symmetry,
                                     const std::unique_ptr<Fcs> &fcs,
                                     const std::unique_ptr<Constraint> &constraint,
                                     const bool compact,
                                     const bool sparse,
                                     const bool return_ata,
                                     const int verbosity = 0) const;

    void set_fcs_values(const int maxorder,
                        double *fc_in,
                        std::vector<size_t> *nequiv,
                        const std::unique_ptr<Constraint> &constraint);

    [[nodiscard]] size_t get_number_of_rows_sensing_matrix() const;

    [[nodiscard]] double *get_params() const;

    void set_optimizer_control(const OptimizerControl &);

    [[nodiscard]] OptimizerControl get_optimizer_control() const;

    [[nodiscard]] double get_cv_l1_alpha() const;

private:
    double *params;
    double cv_l1_alpha; // stores alpha at minimum CV

    std::vector<std::vector<double>> u_train, f_train;
    std::vector<std::vector<double>> u_validation, f_validation;

    OptimizerControl optcontrol;

    void set_default_variables();

    void deallocate_variables();

    void data_multiplier(const std::vector<std::vector<double>> &data_in,
                         std::vector<std::vector<double>> &data_out,
                         const std::unique_ptr<Symmetry> &symmetry) const;

    static int inprim_index(const int n,
                            const std::unique_ptr<Symmetry> &symmetry);

    int least_squares(const int maxorder,
                      const size_t N,
                      const size_t N_new,
                      const size_t M,
                      const int verbosity,
                      const std::unique_ptr<Symmetry> &symmetry,
                      const std::unique_ptr<Fcs> &fcs,
                      const std::unique_ptr<Constraint> &constraint,
                      std::vector<double> &param_out);

    int compressive_sensing(const std::string &job_prefix,
                            const int maxorder,
                            const size_t N_new,
                            const size_t M,
                            const std::unique_ptr<Symmetry> &symmetry,
                            const std::vector<std::string> &str_order,
                            const std::unique_ptr<Fcs> &fcs,
                            std::unique_ptr<Constraint> &constraint,
                            const int verbosity,
                            std::vector<double> &param_out);

    double crossvalidation(const std::string job_prefix,
                           const int maxorder,
                           const std::unique_ptr<Fcs> &fcs,
                           const std::unique_ptr<Symmetry> &symmetry,
                           const std::unique_ptr<Constraint> &constraint,
                           const int verbosity);

    double run_manual_cv(const std::string &job_prefix,
                         const int maxorder,
                         const std::unique_ptr<Fcs> &fcs,
                         const std::unique_ptr<Symmetry> &symmetry,
                         const std::unique_ptr<Constraint> &constraint,
                         const int verbosity);

    double run_auto_cv(const std::string &job_prefix,
                       const int maxorder,
                       const std::unique_ptr<Fcs> &fcs,
                       const std::unique_ptr<Symmetry> &symmetry,
                       const std::unique_ptr<Constraint> &constraint,
                       const int verbosity);

    void write_cvresult_to_file(const std::string &file_out,
                                const std::vector<double> &alphas,
                                const std::vector<double> &training_error,
                                const std::vector<double> &validation_error,
                                const std::vector<std::vector<int>> &nonzeros) const;

    void write_cvscore_to_file(const std::string &file_out,
                               const std::vector<double> &alphas,
                               const std::vector<double> &terr_mean,
                               const std::vector<double> &terr_std,
                               const std::vector<double> &verr_mean,
                               const std::vector<double> &verr_std,
                               const int ialpha_minimum,
                               const size_t nsets) const;

    void set_errors_of_cvscore(std::vector<double> &terr_mean,
                               std::vector<double> &terr_std,
                               std::vector<double> &verr_mean,
                               std::vector<double> &verr_std,
                               const std::vector<std::vector<double>> &training_error_accum,
                               const std::vector<std::vector<double>> &validation_error_accum) const;

    int get_ialpha_at_minimum_validation_error(const std::vector<double> &validation_error) const;

    void optimize_with_given_l1alpha(const int maxorder,
                                     const size_t M,
                                     const size_t N_new,
                                     const std::unique_ptr<Fcs> &fcs,
                                     const std::unique_ptr<Symmetry> &symmetry,
                                     const std::unique_ptr<Constraint> &constraint,
                                     const int verbosity,
                                     std::vector<double> &param_out) const;

    void run_least_squares_with_nonzero_coefs(const Eigen::MatrixXd &A_in,
                                              const Eigen::VectorXd &b_in,
                                              const Eigen::VectorXd &factor_std,
                                              std::vector<double> &params_inout,
                                              const int verbosity) const;

    static void get_number_of_zero_coefs(const int maxorder,
                                         const std::unique_ptr<Constraint> &constraint,
                                         const Eigen::VectorXd &x,
                                         std::vector<int> &nzeros);

    void get_standardizer(const Eigen::MatrixXd &Amat,
                          Eigen::VectorXd &mean,
                          Eigen::VectorXd &dev,
                          Eigen::VectorXd &factor_std,
                          Eigen::VectorXd &scale_beta) const;

    void apply_standardizer(Eigen::MatrixXd &Amat,
                            const Eigen::VectorXd &mean,
                            const Eigen::VectorXd &dev) const;

    [[nodiscard]] double get_estimated_max_alpha(const Eigen::MatrixXd &Amat,
                                                 const Eigen::VectorXd &bvec) const;

    static void apply_scaler_displacement(std::vector<std::vector<double>> &u_inout,
                                          const double normalization_factor,
                                          const bool scale_back = false);

    static void apply_scaler_constraint(const int maxorder,
                                        const double normalization_factor,
                                        const std::unique_ptr<Constraint> &constraint,
                                        const bool scale_back = false);

    static void apply_scaler_force_constants(const int maxorder,
                                             const double normalization_factor,
                                             const std::unique_ptr<Constraint> &constraint,
                                             std::vector<double> &param_inout);

    void apply_scalers(const int maxorder,
                       const std::unique_ptr<Constraint> &constraint);

    void finalize_scalers(const int maxorder,
                          const std::unique_ptr<Constraint> &constraint);

    static void apply_basis_converter(std::vector<std::vector<double>> &u_multi,
                                      Eigen::Matrix3d cmat);

    static void apply_basis_converter_amat(const int natmin3,
                                           const int ncols,
                                           double **amat_orig_tmp,
                                           Eigen::Matrix3d cmat);

    static int fit_without_constraints(const size_t N,
                                       const size_t M,
                                       double *amat,
                                       const double *bvec,
                                       double *param_out,
                                       const int verbosity);

    int fit_algebraic_constraints(const size_t N,
                                  const size_t M,
                                  double *amat,
                                  const double *bvec,
                                  std::vector<double> &param_out,
                                  const double fnorm,
                                  const int maxorder,
                                  const std::unique_ptr<Fcs> &fcs,
                                  const std::unique_ptr<Constraint> &constraint,
                                  const int verbosity) const;

    int solve_normal_equation(const size_t N,
                              double *amat,
                              double *bvec,
                              std::vector<double> &param_out,
                              const double fnorm,
                              const int maxorder,
                              const std::unique_ptr<Fcs> &fcs,
                              const std::unique_ptr<Constraint> &constraint,
                              const int verbosity,
                              const bool algebraic_constraint) const;


    void get_matrix_elements2(const int maxorder,
                              const size_t ncycle,
                              const size_t nrows,
                              const size_t ncols,
                              const size_t ncols_compact,
                              std::unique_ptr<SensingMatrix> &matrix_out,
                              const std::vector<std::vector<double>> &u_multi,
                              const std::vector<std::vector<double>> &f_multi,
                              const std::vector<std::vector<double>> &gamma_precomputed,
                              const std::unique_ptr<Symmetry> &symmetry,
                              const std::unique_ptr<Fcs> &fcs,
                              const std::unique_ptr<Constraint> &constraint,
                              const bool sparse) const;

    void get_matrix_elements_normal_equation2(const int maxorder,
                                              const size_t ncycle,
                                              const size_t nrows,
                                              const size_t ncols,
                                              const size_t ncols_compact,
                                              std::unique_ptr<SensingMatrix> &matrix_out,
                                              const std::vector<std::vector<double>> &u_multi,
                                              const std::vector<std::vector<double>> &f_multi,
                                              const std::vector<std::vector<double>> &gamma_precomputed,
                                              const std::unique_ptr<Symmetry> &symmetry,
                                              const std::unique_ptr<Fcs> &fcs,
                                              const std::unique_ptr<Constraint> &constraint,
                                              const bool sparse) const;

    static void fill_bvec(const size_t natmin,
                          const size_t irow,
                          const std::vector<std::vector<int>> &index_mapping,
                          const std::vector<double> &f_sub,
                          std::vector<double> &bvec);

    static void fill_amat(const int maxorder,
                          const size_t natmin,
                          const size_t ncols,
                          const std::vector<double> &u_sub,
                          const std::vector<std::vector<double>> &gamma_precomputed,
                          const std::unique_ptr<Symmetry> &symmetry,
                          const std::unique_ptr<Fcs> &fcs,
                          double **&amat_orig);

    static void project_constraints(const int maxorder,
                                    const size_t natmin,
                                    const size_t irow,
                                    const std::unique_ptr<Fcs> &fcs,
                                    const std::unique_ptr<Constraint> &constraint,
                                    double **amat_orig,
                                    double **&amat_mod,
                                    std::vector<double> &bvec_mod);

    int run_eigen_sparse_solver(const SpMat &sp_mat,
                                const Eigen::VectorXd &sp_bvec,
                                std::vector<double> &param_out,
                                const double fnorm,
                                const int maxorder,
                                const std::unique_ptr<Fcs> &fcs,
                                const std::unique_ptr<Constraint> &constraint,
                                const std::string solver_type,
                                const int verbosity) const;

    void recover_original_forceconstants(const int maxorder,
                                         const std::vector<double> &param_in,
                                         std::vector<double> &param_out,
                                         const std::vector<size_t> *nequiv,
                                         const std::unique_ptr<Constraint> &constraint) const;

    [[nodiscard]] int factorial(const int) const;

    static int rankQRD(const size_t m,
                       const size_t n,
                       double *mat,
                       const double tolerance);

    double gamma(const int,
                 const int *) const;

    void coordinate_descent(const int M,
                            const int N,
                            const double alpha,
                            const int warm_start,
                            Eigen::VectorXd &x,
                            const Eigen::MatrixXd &A,
                            const Eigen::VectorXd &b,
                            const Eigen::VectorXd &grad0,
                            bool *has_prod,
                            Eigen::MatrixXd &Prod,
                            Eigen::VectorXd &grad,
                            const double fnorm,
                            const Eigen::VectorXd &scale_beta,
                            const int verbosity) const;

    void solution_path(const int maxorder,
                       Eigen::MatrixXd &A,
                       Eigen::VectorXd &b,
                       Eigen::MatrixXd &A_validation,
                       Eigen::VectorXd &b_validation,
                       const double fnorm,
                       const double fnorm_validation,
                       const std::string &file_coef,
                       const int verbosity,
                       const std::unique_ptr<Constraint> &constraint,
                       const std::vector<double> &alphas,
                       std::vector<double> &training_error,
                       std::vector<double> &validation_error,
                       std::vector<std::vector<int>> &nonzeros) const;

    static void compute_alphas(const double l1_alpha_max,
                               const double l1_alpha_min,
                               const int num_l1_alpha,
                               std::vector<double> &alphas);
};

inline double shrink(const double x,
                     const double a)
{
    const auto xabs = std::abs(x);
    const auto sign = static_cast<double>((0.0 < x) - (x < 0.0));
    return sign * std::max<double>(xabs - a, 0.0);
}

}
