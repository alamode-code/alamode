/*
 optimize.cpp

 Copyright (c) 2014-2018 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "optimize.h"
#include <boost/algorithm/string.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "constraint.h"
#include "error.h"
#include "fcs.h"
#include "files.h"
#include "input_parser.h"
#include "lapack_wrapper.h"
#include "least_squares.h"
#include "memory.h"
#include "symmetry.h"
#include "timer.h"

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <Eigen/SparseQR>
#include <omp.h>
#include "logger.h"

using namespace ALM_NS;

Optimize::Optimize()
{
    set_default_variables();
}

Optimize::~Optimize()
{
    deallocate_variables();
}

void Optimize::set_default_variables()
{
    params = nullptr;
    cv_l1_alpha = 0.0;
}

void Optimize::deallocate_variables()
{
    if (params) {
        deallocate(params);
    }
}

auto Optimize::optimize_main(const std::unique_ptr<Symmetry> &symmetry, std::unique_ptr<Constraint> &constraint,
                             const std::unique_ptr<Fcs> &fcs, const int maxorder, const std::string &file_prefix,
                             const std::vector<std::string> &str_order, const int verbosity,
                             const DispForceFile &filedata_train, const DispForceFile &filedata_validation,
                             const int output_maxorder, std::unique_ptr<Timer> &timer) -> int
{
    timer->start_clock("optimize");

    const auto ndata_used =
        filedata_train.nend - filedata_train.nstart + 1 - filedata_train.skip_e + filedata_train.skip_s;
    const auto ndata_used_validation = filedata_validation.nend - filedata_validation.nstart + 1;
    auto info_fitting = 0;
    const auto M = get_number_of_rows_sensing_matrix();
    size_t nparams = 0;
    size_t nparams_irred = 0;
    for (auto i = 0; i < maxorder; ++i) {
        nparams += fcs->get_nequiv()[i].size();
    }

    if (constraint->get_constraint_algebraic()) {
        for (auto i = 0; i < maxorder; ++i) {
            nparams_irred += constraint->get_index_bimap(i).size();
        }
    }

    if (verbosity > 0) {
        std::vector<std::string> str_linearmodel{"least-squares", "elastic-net", "adaptive-lasso"};
        std::cout << " ==============\n";
        std::cout << "  OPTIMIZATION \n";
        std::cout << " ==============\n\n";
        std::cout << "  LMODEL = " << str_linearmodel[optcontrol.linear_model - 1] << "\n\n";
        if (!filedata_train.filename.empty()) {
            std::cout << "  Training data file (DFSET) : " << filedata_train.filename << "\n\n";
            std::cout << "  NSTART = " << filedata_train.nstart << "; NEND = " << filedata_train.nend << '\n';
            if (filedata_train.skip_s < filedata_train.skip_e) {
                std::cout << ": SKIP = " << filedata_train.skip_s << "-" << filedata_train.skip_e - 1 << '\n';
            }
            std::cout << "  " << ndata_used << " entries will be used for training.\n\n";
        }

        if (optcontrol.cross_validation == -1) {
            std::cout << "  CV = -1 : Manual cross-validation mode is selected\n";
            if (!filedata_validation.filename.empty()) {
                std::cout << "  Validation data file (DFSET_CV) : " << filedata_validation.filename << "\n\n";
                std::cout << "  NSTART_CV = " << filedata_validation.nstart
                          << "; NEND_CV = " << filedata_validation.nend << '\n';
                std::cout << "  " << ndata_used_validation << " entries will be used for validation.\n\n";
            }
        }
        std::cout << "  Total Number of Parameters : " << nparams << '\n';
        if (constraint->get_constraint_algebraic()) {
            std::cout << "  Total Number of Free Parameters : " << nparams_irred << '\n';
        }
        std::cout << '\n';
    }

    // Run optimization and get force constants

    std::vector<double> fcs_tmp(nparams, 0.0);
    if (optcontrol.linear_model == 1) {
        // Use ordinary least-squares

        info_fitting =
            least_squares(maxorder, nparams, nparams_irred, M, verbosity, symmetry, fcs, constraint, fcs_tmp);

    } else if (optcontrol.linear_model == 2 or optcontrol.linear_model == 3) {

        // Use elastic net or adaptive lasso

        if (!constraint->get_constraint_algebraic()) {
            exit("optimize_main", "Sorry, ICONST = 10 or ICONST = 11 must be used when using elastic net.");
        }

        if (optcontrol.linear_model == 3) {
            if (optcontrol.standardize) {
                if (verbosity > 0) {
                    warn("optimize_main",
                         "STANDARDIZE = 1 in adaptive LASSO is meaningless.\n"
                         " Switch to STANDARDIZE = 0.");
                }
                optcontrol.standardize = 0;
            }

            if (std::abs(optcontrol.displacement_normalization_factor - 1.0) > eps) {
                if (verbosity > 0) {
                    warn("optimize_main",
                         "DNORM_BASIS != 1.0 should be avoided in adaptive LASSO.\n"
                         " Switch to DNORM_BASIS = 1.0.");
                }
                optcontrol.displacement_normalization_factor = 1.0;
            }

            if (std::abs(optcontrol.l1_ratio - 1.0) > eps) {
                if (verbosity > 0) {
                    warn("optimize_main", "L1_RATIO != 1.0 should be avoided in adaptive LASSO");
                }
            }
        }

        info_fitting = compressive_sensing(file_prefix,
                                           maxorder,
                                           nparams_irred,
                                           M,
                                           symmetry,
                                           str_order,
                                           fcs,
                                           constraint,
                                           verbosity,
                                           fcs_tmp);
    }

    if (info_fitting == 0) {
        // I should copy fcs_tmp to parameters in the Fcs class?
        // Copy force constants to public variable "params"
        if (params) {
            deallocate(params);
        }
        allocate(params, nparams);
        for (auto i = 0; i < nparams; ++i)
            params[i] = fcs_tmp[i];

        // Set calculated force constants in FCS class
        auto maxorder_min = std::min(maxorder, output_maxorder);
        fcs->set_forceconstant_cartesian(maxorder_min, params);
    } else {
        exit("optimize_main",
             "Fitting failed. \n"
             " Please check the input parameters and try other optimization algorithms.");
    }

    fcs_tmp.clear();
    fcs_tmp.shrink_to_fit();

    if (verbosity > 0) {
        std::cout << '\n';
        timer->print_elapsed();
        std::cout << " -------------------------------------------------------------------\n\n";
    }

    timer->stop_clock("optimize");

    return info_fitting;
}

auto Optimize::least_squares(const int maxorder, const size_t N, const size_t N_new, const size_t M,
                             const int verbosity, const std::unique_ptr<Symmetry> &symmetry,
                             const std::unique_ptr<Fcs> &fcs, const std::unique_ptr<Constraint> &constraint,
                             std::vector<double> &param_out) const -> int
{
    auto info_fitting = 0;

    const bool sparse = optcontrol.use_sparse_solver;
    const bool compact = (constraint->get_constraint_algebraic() == 1);
    const bool return_ata = (optcontrol.use_cholesky == 1);

    auto matrix_out = std::make_unique<SensingMatrix>();

    if (return_ata && (!compact) && constraint->get_exist_constraint()) {
        exit("least_squares",
             "The combination of ICONST=1, 2, 3 and USE_CHOLESKY=1 is not supported.\n"
             " Please use ICONST=10, 11, 12 instead.");
    }

    LOG_IF(verbosity, 1, " compact= ", compact, '\n');
    LOG_IF(verbosity, 1, " sparse = ", sparse, '\n');
    LOG_IF(verbosity, 1, " return_ata = ", return_ata, "\n\n");

    get_matrix_elements_unified(maxorder,
                                matrix_out,
                                u_train,
                                f_train,
                                symmetry,
                                fcs,
                                constraint,
                                compact,
                                sparse,
                                return_ata,
                                (verbosity > 0));
    auto fnorm = 0.0;
    for (const auto &it: matrix_out->original_forces) {
        fnorm += it * it;
    }
    fnorm = std::sqrt(fnorm);

    LOG_IF(verbosity, 1, "Sensing matrix is generated.\n");

    if (return_ata) {
        if (sparse) {
            const Eigen::VectorXd sp_bvec =
                Eigen::Map<Eigen::VectorXd>(matrix_out->bvec.data(), matrix_out->bvec.size());
            if (verbosity > 0) {
                std::cout << "  Now, start fitting ...\n";
            }

            const Eigen::SimplicialLDLT<SpMat> ldlt(matrix_out->amat_sparse);
            Eigen::VectorXd x = ldlt.solve(sp_bvec);

            const auto nparams = x.size();
            std::vector<double> param_irred(nparams);

            for (auto i = 0; i < nparams; ++i) {
                param_irred[i] = x(i);
            }
            if (compact) {
                // Recover full reducible force constants
                recover_original_forceconstants(maxorder, param_irred, param_out, fcs->get_nequiv(), constraint);
            } else {
                param_out = param_irred;
            }

        } else {
            // Solve the normal equation (A^T A)x=A^T b using dense datatype
            // Cholesky decomposition
            info_fitting = solve_normal_equation(matrix_out->bvec.size(),
                                                 matrix_out->amat_dense.data(),
                                                 matrix_out->bvec.data(),
                                                 param_out,
                                                 fnorm,
                                                 maxorder,
                                                 fcs,
                                                 constraint,
                                                 verbosity,
                                                 compact);
        }

    } else {

        if (sparse) {
            Eigen::VectorXd sp_bvec = Eigen::Map<Eigen::VectorXd>(matrix_out->bvec.data(), matrix_out->bvec.size());
            if (verbosity > 0) {
                std::cout << "  Now, start fitting ...\n";
            }

            if (compact) {
                info_fitting = run_eigen_sparse_solver(matrix_out->amat_sparse,
                                                       sp_bvec,
                                                       param_out,
                                                       fnorm,
                                                       maxorder,
                                                       fcs,
                                                       constraint,
                                                       optcontrol.sparsesolver,
                                                       verbosity);
            } else {
                Eigen::VectorXd x, lambda;

                solveGQRSparse(matrix_out->amat_sparse,
                               sp_bvec,
                               constraint->get_const_mat_sparse(),
                               constraint->get_const_rhs_vec(),
                               x,
                               lambda,
                               verbosity);

                auto res = sp_bvec - matrix_out->amat_sparse * x;
                const auto res2norm = res.squaredNorm();
                const auto nparams = x.size();
                std::vector<double> param_irred(nparams);

                for (auto i = 0; i < nparams; ++i) {
                    param_irred[i] = x(i);
                }

                // Recover reducible set of force constants

                if (constraint->get_constraint_algebraic()) {
                    recover_original_forceconstants(maxorder, param_irred, param_out, fcs->get_nequiv(), constraint);
                } else {
                    param_out.resize(nparams, 0.0);
                    for (size_t i = 0; i < nparams; ++i) {
                        param_out[i] = param_irred[i];
                    }
                }

                if (verbosity > 0) {
                    std::cout << "  Residual sum of squares for the solution: " << sqrt(res2norm) << '\n';
                    std::cout << "  Fitting error (%) : " << sqrt(res2norm / (fnorm * fnorm)) * 100.0 << '\n';
                }
            }

        } else {
            // Use a direct solver for a dense matrix

            if (compact) {
                // Perform singular value decomposition to solve
                // min||Ax-b||^{2}_{2}
                info_fitting = fit_algebraic_constraints(N_new,
                                                         M,
                                                         matrix_out->amat_dense.data(),
                                                         matrix_out->bvec.data(),
                                                         param_out,
                                                         fnorm,
                                                         maxorder,
                                                         fcs,
                                                         constraint,
                                                         verbosity);
            } else if (constraint->get_exist_constraint()) {

                info_fitting = least_squares_with_constraints_gqr(N,
                                                                  M,
                                                                  constraint->get_number_of_constraints(),
                                                                  matrix_out->amat_dense.data(),
                                                                  matrix_out->bvec.data(),
                                                                  param_out.data(),
                                                                  constraint->get_const_mat(),
                                                                  constraint->get_const_rhs(),
                                                                  verbosity);
            } else {
                // Perform fitting with SVD
                info_fitting = least_squares_svd(N,
                                                 M,
                                                 matrix_out->amat_dense.data(),
                                                 matrix_out->bvec.data(),
                                                 param_out.data(),
                                                 verbosity);
            }
        }
    }
    return info_fitting;
}


auto Optimize::compressive_sensing(const std::string &job_prefix, const int maxorder, const size_t N_new,
                                   const size_t M, const std::unique_ptr<Symmetry> &symmetry,
                                   const std::vector<std::string> &str_order, const std::unique_ptr<Fcs> &fcs,
                                   std::unique_ptr<Constraint> &constraint, const int verbosity,
                                   std::vector<double> &param_out) -> int
{
    // Perform compressive sensing analysis of the linear model either based on
    // the elastic net or adaptive lasso.

    int info_fitting;

    std::vector<double> param_tmp(N_new, 0.0);

    // Scale displacements if DNORM != 1.0 and the data is not standardized.
    // This rule is not applied when the adaptive lasso is selected.
    const int scale_displacement = std::abs(optcontrol.displacement_normalization_factor - 1.0) > eps &&
                                   (optcontrol.standardize == 0) && (optcontrol.linear_model == 2);

    if (optcontrol.cross_validation == 0) {

        if (scale_displacement) {
            apply_scalers(maxorder, constraint);
        }

        // Optimize with a given L1 coefficient (l1_alpha)
        optimize_with_given_l1alpha(maxorder, M, N_new, fcs, symmetry, constraint, verbosity, param_tmp);

        if (verbosity > 0) {
            size_t iparam = 0;
            std::vector<int> nzero_cs(maxorder);

            for (auto i = 0; i < maxorder; ++i) {
                nzero_cs[i] = 0;
                for (const auto &it: constraint->get_index_bimap(i)) {
                    const auto inew = it.left + iparam;
                    if (std::abs(param_tmp[inew]) < eps) ++nzero_cs[i];
                }
                iparam += constraint->get_index_bimap(i).size();
            }

            for (auto order = 0; order < maxorder; ++order) {
                std::cout << "  Number of non-zero " << std::setw(9) << str_order[order]
                          << " FCs : " << constraint->get_index_bimap(order).size() - nzero_cs[order] << '\n';
            }
            std::cout << '\n';
        }

        // Scale back force constants

        if (scale_displacement) {
            apply_scaler_force_constants(maxorder, optcontrol.displacement_normalization_factor, constraint, param_tmp);
            finalize_scalers(maxorder, constraint);
        }

        recover_original_forceconstants(maxorder, param_tmp, param_out, fcs->get_nequiv(), constraint);
        info_fitting = 0;

    } else {
        // Run cross validation (manually or automatically) to
        // get a L1 alpha that gives the minimum CV score
        if (scale_displacement) {
            apply_scalers(maxorder, constraint);
        }

        // cv_l1_alpha is a private variable of Optimize class.
        cv_l1_alpha = crossvalidation(job_prefix, maxorder, fcs, symmetry, constraint, verbosity);
        if (scale_displacement) {
            finalize_scalers(maxorder, constraint);
        }

        info_fitting = 0;
    }

    return info_fitting;
}


auto Optimize::crossvalidation(const std::string &job_prefix, const int maxorder, const std::unique_ptr<Fcs> &fcs,
                               const std::unique_ptr<Symmetry> &symmetry, const std::unique_ptr<Constraint> &constraint,
                               const int verbosity) -> double
{
    // Cross-validation mode:
    // Returns alpha giving minimum CV score


    if (verbosity > 0) {
        std::vector<std::string> str_linearmodel{"Elastic-net", "Adaptive LASSO"};
        std::cout << " " << str_linearmodel[optcontrol.linear_model - 2];
        std::cout << "  cross-validation with the following parameters:\n";
        std::cout << "   L1_RATIO = " << optcontrol.l1_ratio << '\n';
        std::cout << "   CV = " << std::setw(15) << optcontrol.cross_validation << '\n';
        if (optcontrol.l1_alpha_min > 0) {
            std::cout << "   CV_MINALPHA = " << std::setw(15) << optcontrol.l1_alpha_min;
        } else {
            std::cout << "   CV_MINALPHA = CV_MAXALPHA * CV_MINALPHA_RATIO (" << std::setw(10)
                      << optcontrol.l1_alpha_min_ratio << ")";
        }
        if (optcontrol.l1_alpha_max > 0) {
            std::cout << "  CV_MAXALPHA = " << std::setw(15) << optcontrol.l1_alpha_max << '\n';
        } else {
            std::cout << " CV_MAXALPHA = (Use recommended value)\n";
        }
        std::cout << "   CV_NALPHA = " << std::setw(5) << optcontrol.num_l1_alpha << '\n';
        std::cout << "   CONV_TOL = " << std::setw(15) << optcontrol.tolerance_iteration << '\n';
        std::cout << "   MAXITER = " << std::setw(5) << optcontrol.maxnum_iteration << '\n';
        std::cout << "   STOP_CRITERION = " << std::setw(5) << optcontrol.stop_criterion << '\n';
        std::cout << "   ENET_DNORM = " << std::setw(15) << optcontrol.displacement_normalization_factor << '\n';
        std::cout << '\n';

        if (optcontrol.linear_model == 2) {
            if (optcontrol.standardize) {
                std::cout << "  STANDARDIZE = 1 : Standardization will be performed for matrix A and vector b.\n";
                std::cout << "                    The ENET_DNORM-tag will be neglected.\n\n";
            } else {
                std::cout << "  STANDARDIZE = 0 : No standardization of matrix A and vector b.\n";
                std::cout << "                    Columns of matrix A will be scaled by the ENET_DNORM value.\n\n";
            }
        }

        if (optcontrol.cross_validation == -1) {
            std::cout << "  CV = -1: Manual CV mode.\n";
            std::cout << "           Validation data is read from DFSET_CV\n";
        } else if (optcontrol.cross_validation > 0) {
            std::cout << "  CV > 0: Automatic CV mode.\n";
        } else {
            exit("crossvalidation", "This cannot happen.");
        }
        std::cout << '\n';
    }


    // Returns alpha at minimum CV
    if (optcontrol.cross_validation == -1) {
        return run_manual_cv(job_prefix, maxorder, fcs, symmetry, constraint, verbosity);
    } else {
        return run_auto_cv(job_prefix, maxorder, fcs, symmetry, constraint, verbosity);
    }
}

auto Optimize::run_manual_cv(const std::string &job_prefix, const int maxorder, const std::unique_ptr<Fcs> &fcs,
                             const std::unique_ptr<Symmetry> &symmetry, const std::unique_ptr<Constraint> &constraint,
                             const int verbosity) const -> double
{
    // Manual CV mode where the test data is read from the user-defined file.
    // Indeed, the test data is already read in the input_parser and stored in u_validation and f_validation.

    std::vector<double> alphas, training_error, validation_error;
    std::vector<std::vector<int>> nonzeros;
    double fnorm, fnorm_validation;

    size_t N_new = 0;
    if (constraint->get_constraint_algebraic()) {
        for (auto i = 0; i < maxorder; ++i) {
            N_new += constraint->get_index_bimap(i).size();
        }
    }

    std::unique_ptr<SensingMatrix> matrix_train = std::make_unique<SensingMatrix>();
    std::unique_ptr<SensingMatrix> matrix_validation = std::make_unique<SensingMatrix>();

    get_matrix_elements_unified(maxorder,
                                matrix_train,
                                u_train,
                                f_train,
                                symmetry,
                                fcs,
                                constraint,
                                true,
                                false,
                                false);

    get_matrix_elements_unified(maxorder,
                                matrix_validation,
                                u_validation,
                                f_validation,
                                symmetry,
                                fcs,
                                constraint,
                                true,
                                false,
                                false);

    Eigen::MatrixXd A =
        Eigen::Map<Eigen::MatrixXd>(matrix_train->amat_dense.data(), matrix_train->amat_dense.size() / N_new, N_new);
    Eigen::VectorXd b = Eigen::Map<Eigen::VectorXd>(matrix_train->bvec.data(), matrix_train->bvec.size());
    Eigen::MatrixXd A_validation = Eigen::Map<Eigen::MatrixXd>(matrix_validation->amat_dense.data(),
                                                               matrix_validation->amat_dense.size() / N_new,
                                                               N_new);
    Eigen::VectorXd b_validation =
        Eigen::Map<Eigen::VectorXd>(matrix_validation->bvec.data(), matrix_validation->bvec.size());

    fnorm = 0.0;
    for (const auto &it: matrix_train->original_forces) {
        fnorm += it * it;
    }
    fnorm = std::sqrt(fnorm);

    fnorm_validation = 0.0;
    for (const auto &it: matrix_validation->original_forces) {
        fnorm_validation += it * it;
    }
    fnorm_validation = std::sqrt(fnorm_validation);

    if (optcontrol.linear_model == 3) {
        // Merge training and validation sets and run OLS once to get the weight in adalasso.
        Eigen::MatrixXd A_merged(A.rows() + A_validation.rows(), N_new);
        Eigen::VectorXd b_merged(b.size() + b_validation.size());
        A_merged << A, A_validation;
        b_merged << b, b_validation;

        // Use QR decomposition with pivoting to check if system is well-determined
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_solver(A_merged);

        // Check if the system is underdetermined or rank-deficient
        const auto rank = qr_solver.rank();

        if (rank < N_new) {
            std::string error_msg = "Adaptive lasso failed: The least squares problem is rank-deficient.\n"
                                    "  Matrix rank = " +
                                    std::to_string(rank) + ", Number of parameters = " + std::to_string(N_new) +
                                    "\n"
                                    "  This typically occurs when there are too few training data\n"
                                    "  or when some parameters cannot be uniquely determined.\n"
                                    "  Please try one of the following:\n"
                                    "  - Increase the amount of training data (DFSET)\n"
                                    "  - Use LMODEL = 1 (least-squares) or 2 (elastic-net) instead\n"
                                    "  - Reduce the interaction cutoff distance\n";
            ALM_NS::exit("optimize_main", error_msg.c_str());
        }

        Eigen::VectorXd x_ols = qr_solver.solve(b_merged);
        Eigen::VectorXd weight_adalasso = x_ols.cwiseAbs();

        // Check if any weights are too small (could cause numerical issues)
        const double min_weight_threshold = 1.0e-10;
        int n_zero_weights = 0;
        for (int i = 0; i < weight_adalasso.size(); ++i) {
            if (weight_adalasso[i] < min_weight_threshold) {
                n_zero_weights++;
            }
        }

        if (n_zero_weights > 0) {
            if (verbosity > 0) {
                std::cout << "  WARNING: Adaptive lasso detected " << n_zero_weights << " near-zero weights (< "
                          << min_weight_threshold << ").\n";
                std::cout << "  This may indicate an underdetermined or ill-conditioned problem.\n";
                std::cout << "  The optimization will continue but results may be unreliable.\n";
                std::cout << "  Consider using LMODEL = 2 (elastic-net) instead.\n\n";
            }
        }

        A = A * weight_adalasso.asDiagonal();
        A_validation = A_validation * weight_adalasso.asDiagonal();
    }

    const auto estimated_max_alpha = get_estimated_max_alpha(A, b);

    if (verbosity > 0) {
        std::cout << "  Recommended CV_MAXALPHA = " << estimated_max_alpha << "\n\n";
    }

    const auto file_coef = job_prefix + ".solution_path";
    const auto file_cv = job_prefix + ".cvset";

    if (optcontrol.l1_alpha_max > 0) {
        compute_alphas(optcontrol.l1_alpha_max, optcontrol.l1_alpha_min, optcontrol.num_l1_alpha, alphas);
    } else {
        if (optcontrol.l1_alpha_min > 0) {
            compute_alphas(estimated_max_alpha, optcontrol.l1_alpha_min, optcontrol.num_l1_alpha, alphas);
        } else {
            compute_alphas(estimated_max_alpha,
                           estimated_max_alpha * optcontrol.l1_alpha_min_ratio,
                           optcontrol.num_l1_alpha,
                           alphas);
        }
    }

    solution_path(maxorder,
                  A,
                  b,
                  A_validation,
                  b_validation,
                  fnorm,
                  fnorm_validation,
                  file_coef,
                  verbosity,
                  constraint,
                  alphas,
                  training_error,
                  validation_error,
                  nonzeros);

    write_cvresult_to_file(file_cv, alphas, training_error, validation_error, nonzeros);

    const auto ialpha = get_ialpha_at_minimum_validation_error(validation_error);

    if (verbosity > 0) {
        std::cout << "  The manual CV has been done.\n";
        std::cout << "  Minimum validation error at alpha = " << alphas[ialpha] << '\n';
        std::cout << "  The CV result is saved in " << file_cv << '\n';

        if (ialpha == optcontrol.num_l1_alpha - 1) {
            warn("run_manual_cv",
                 "The minimum validation score occurs at CV_MINALPHA.\n"
                 " Please use a smaller CV_MINALPHA or CV_MINALPHA_RATIO to suppress this message.");
        }
    }

    return alphas[ialpha];
}

auto Optimize::run_auto_cv(const std::string &job_prefix, const int maxorder, const std::unique_ptr<Fcs> &fcs,
                           const std::unique_ptr<Symmetry> &symmetry, const std::unique_ptr<Constraint> &constraint,
                           const int verbosity) -> double
{
    // Automatic CV mode.

    size_t N_new = 0;
    if (constraint->get_constraint_algebraic()) {
        for (auto i = 0; i < maxorder; ++i) {
            N_new += constraint->get_index_bimap(i).size();
        }
    }

    const auto nstructures = u_train.size();
    const auto nsets = optcontrol.cross_validation;

    if (nsets > nstructures) {
        exit("run_auto_cv", "The input CV is larger than the total number of training data.");
    }

    std::vector<int> ndata_block(nsets, nstructures / nsets);
    for (auto iset = 0; iset < nsets; ++iset) {
        if (nstructures - nsets * (nstructures / nsets) > iset) {
            ++ndata_block[iset];
        }
    }

    std::vector<std::vector<double>> u_train_tmp, u_validation_tmp;
    std::vector<std::vector<double>> f_train_tmp, f_validation_tmp;

    std::vector<double> alphas, training_error, validation_error;
    std::vector<std::vector<int>> nonzeros;
    std::vector<std::vector<double>> training_error_accum, validation_error_accum;
    double fnorm, fnorm_validation, estimated_max_alpha;
    Eigen::VectorXd weight_adalasso;

    std::unique_ptr<SensingMatrix> matrix_train = std::make_unique<SensingMatrix>();
    std::unique_ptr<SensingMatrix> matrix_validation = std::make_unique<SensingMatrix>();

    auto ishift = 0;

    if (verbosity > 0) {
        std::cout << "  Start " << nsets << "-fold CV with " << u_train.size() << " Datasets\n\n";
    }

    if (optcontrol.linear_model == 3) {
        get_matrix_elements_unified(maxorder,
                                    matrix_train,
                                    u_train,
                                    f_train,
                                    symmetry,
                                    fcs,
                                    constraint,
                                    true,
                                    false,
                                    false);

        Eigen::MatrixXd A_full = Eigen::Map<Eigen::MatrixXd>(matrix_train->amat_dense.data(),
                                                             matrix_train->amat_dense.size() / N_new,
                                                             N_new);
        Eigen::VectorXd b_full = Eigen::Map<Eigen::VectorXd>(matrix_train->bvec.data(), matrix_train->bvec.size());

        // Use QR decomposition with pivoting to check if system is well-determined
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_solver(A_full);

        // Check if the system is underdetermined or rank-deficient
        const auto rank = qr_solver.rank();

        if (rank < N_new) {
            std::string error_msg = "Adaptive lasso failed in CV: The least squares problem is rank-deficient.\n"
                                    "  Matrix rank = " +
                                    std::to_string(rank) + ", Number of parameters = " + std::to_string(N_new) +
                                    "\n"
                                    "  This typically occurs when there are too few training data\n"
                                    "  or when some parameters cannot be uniquely determined.\n"
                                    "  Please try one of the following:\n"
                                    "  - Increase the amount of training data (DFSET)\n"
                                    "  - Use LMODEL = 1 (least-squares) or 2 (elastic-net) instead\n"
                                    "  - Reduce the interaction cutoff distance\n";
            ALM_NS::exit("optimize_main", error_msg.c_str());
        }

        Eigen::VectorXd x_ols = qr_solver.solve(b_full);
        weight_adalasso = x_ols.cwiseAbs();

        // Check if any weights are too small (could cause numerical issues)
        const double min_weight_threshold = 1.0e-10;
        int n_zero_weights = 0;
        for (int i = 0; i < weight_adalasso.size(); ++i) {
            if (weight_adalasso[i] < min_weight_threshold) {
                n_zero_weights++;
            }
        }

        if (n_zero_weights > 0 && verbosity > 0) {
            std::cout << "  WARNING: Adaptive lasso detected " << n_zero_weights << " near-zero weights (< "
                      << min_weight_threshold << ").\n";
            std::cout << "  This may indicate an underdetermined or ill-conditioned problem.\n\n";
        }
    }

    if (optcontrol.l1_alpha_max <= 0) {
        estimated_max_alpha = 0;
        for (auto iset = 0; iset < nsets; ++iset) {
            const auto istart_validation = ishift;
            const auto iend_validation = istart_validation + ndata_block[iset];

            u_train_tmp.clear();
            f_train_tmp.clear();
            u_validation_tmp.clear();
            f_validation_tmp.clear();

            for (auto idata = 0; idata < nstructures; ++idata) {
                if (idata >= istart_validation && idata < iend_validation) {
                    u_validation_tmp.emplace_back(u_train[idata]);
                    f_validation_tmp.emplace_back(f_train[idata]);
                } else {
                    u_train_tmp.emplace_back(u_train[idata]);
                    f_train_tmp.emplace_back(f_train[idata]);
                }
            }
            ishift += ndata_block[iset];

            get_matrix_elements_unified(maxorder,
                                        matrix_train,
                                        u_train_tmp,
                                        f_train_tmp,
                                        symmetry,
                                        fcs,
                                        constraint,
                                        true,
                                        false,
                                        false);

            Eigen::MatrixXd A = Eigen::Map<Eigen::MatrixXd>(matrix_train->amat_dense.data(),
                                                            matrix_train->amat_dense.size() / N_new,
                                                            N_new);
            Eigen::VectorXd b = Eigen::Map<Eigen::VectorXd>(matrix_train->bvec.data(), matrix_train->bvec.size());
            if (optcontrol.linear_model == 3) A = A * weight_adalasso.asDiagonal();
            const auto this_estimated_max_alpha = get_estimated_max_alpha(A, b);

            if (verbosity > 0) {
                std::cout << "  Recommended CV_MAXALPHA (" << std::setw(3) << iset + 1
                          << ") = " << this_estimated_max_alpha << '\n';
            }

            if (this_estimated_max_alpha > estimated_max_alpha) {
                estimated_max_alpha = this_estimated_max_alpha;
            }
        }
        ishift = 0;
    }

    for (auto iset = 0; iset < nsets; ++iset) {

        if (verbosity > 0) {
            std::cout << '\n';
            std::cout << "  SET : " << std::setw(3) << iset + 1 << '\n';
        }
        const auto istart_validation = ishift;
        const auto iend_validation = istart_validation + ndata_block[iset];

        u_train_tmp.clear();
        f_train_tmp.clear();
        u_validation_tmp.clear();
        f_validation_tmp.clear();

        for (auto idata = 0; idata < nstructures; ++idata) {
            if (idata >= istart_validation && idata < iend_validation) {
                u_validation_tmp.emplace_back(u_train[idata]);
                f_validation_tmp.emplace_back(f_train[idata]);
            } else {
                u_train_tmp.emplace_back(u_train[idata]);
                f_train_tmp.emplace_back(f_train[idata]);
            }
        }
        ishift += ndata_block[iset];

        get_matrix_elements_unified(maxorder,
                                    matrix_train,
                                    u_train_tmp,
                                    f_train_tmp,
                                    symmetry,
                                    fcs,
                                    constraint,
                                    true,
                                    false,
                                    false);

        get_matrix_elements_unified(maxorder,
                                    matrix_validation,
                                    u_validation_tmp,
                                    f_validation_tmp,
                                    symmetry,
                                    fcs,
                                    constraint,
                                    true,
                                    false,
                                    false);

        Eigen::MatrixXd A = Eigen::Map<Eigen::MatrixXd>(matrix_train->amat_dense.data(),
                                                        matrix_train->amat_dense.size() / N_new,
                                                        N_new);
        Eigen::VectorXd b = Eigen::Map<Eigen::VectorXd>(matrix_train->bvec.data(), matrix_train->bvec.size());
        Eigen::MatrixXd A_validation = Eigen::Map<Eigen::MatrixXd>(matrix_validation->amat_dense.data(),
                                                                   matrix_validation->amat_dense.size() / N_new,
                                                                   N_new);
        Eigen::VectorXd b_validation =
            Eigen::Map<Eigen::VectorXd>(matrix_validation->bvec.data(), matrix_validation->bvec.size());


        fnorm = 0.0;
        for (const auto &it: matrix_train->original_forces) {
            fnorm += it * it;
        }
        fnorm = std::sqrt(fnorm);

        fnorm_validation = 0.0;
        for (const auto &it: matrix_validation->original_forces) {
            fnorm_validation += it * it;
        }
        fnorm_validation = std::sqrt(fnorm_validation);

        if (optcontrol.linear_model == 3) {
            A = A * weight_adalasso.asDiagonal();
            A_validation = A_validation * weight_adalasso.asDiagonal();
        }

        if (verbosity > 0) {
            std::cout << "  Recommended CV_MAXALPHA = " << get_estimated_max_alpha(A, b) << "\n\n";
        }

        const auto file_coef = job_prefix + ".solution_path" + std::to_string(iset + 1);
        const auto file_cv = job_prefix + ".cvset" + std::to_string(iset + 1);

        if (optcontrol.l1_alpha_max > 0) {
            compute_alphas(optcontrol.l1_alpha_max, optcontrol.l1_alpha_min, optcontrol.num_l1_alpha, alphas);
        } else {
            if (optcontrol.l1_alpha_min > 0) {
                compute_alphas(estimated_max_alpha, optcontrol.l1_alpha_min, optcontrol.num_l1_alpha, alphas);
            } else {
                compute_alphas(estimated_max_alpha,
                               estimated_max_alpha * optcontrol.l1_alpha_min_ratio,
                               optcontrol.num_l1_alpha,
                               alphas);
            }
        }

        solution_path(maxorder,
                      A,
                      b,
                      A_validation,
                      b_validation,
                      fnorm,
                      fnorm_validation,
                      file_coef,
                      verbosity,
                      constraint,
                      alphas,
                      training_error,
                      validation_error,
                      nonzeros);

        if (!job_prefix.empty()) {
            write_cvresult_to_file(file_cv, alphas, training_error, validation_error, nonzeros);
        }

        if (verbosity > 0) {
            auto ialpha = get_ialpha_at_minimum_validation_error(validation_error);
            std::cout << "  SET " << std::setw(3) << iset + 1 << " has been finished.\n";
            std::cout << "  Minimum validation error at alpha = " << alphas[ialpha] << '\n';
            if (!job_prefix.empty()) {
                std::cout << "  The CV result is saved in " << file_cv << "\n\n";
            }

            if (ialpha == optcontrol.num_l1_alpha - 1) {
                warn("run_auto_cv",
                     "The minimum validation score occurs at CV_MINALPHA.\n"
                     " Please use a smaller CV_MINALPHA to suppress this message.");
            }
            std::cout << "  ---------------------------------------------------\n";
        }

        training_error_accum.emplace_back(training_error);
        validation_error_accum.emplace_back(validation_error);
    }

    std::vector<double> terr_mean, terr_std;
    std::vector<double> verr_mean, verr_std;

    const auto nalphas = alphas.size();

    terr_mean.resize(nalphas);
    terr_std.resize(nalphas);
    verr_mean.resize(nalphas);
    verr_std.resize(nalphas);

    set_errors_of_cvscore(terr_mean, terr_std, verr_mean, verr_std, training_error_accum, validation_error_accum);
    const auto ialpha_minimum = get_ialpha_at_minimum_validation_error(verr_mean);

    if (!job_prefix.empty()) {
        const auto file_cvscore = job_prefix + ".cvscore";
        write_cvscore_to_file(file_cvscore, alphas, terr_mean, terr_std, verr_mean, verr_std, ialpha_minimum, nsets);

        if (verbosity > 0) {
            std::cout << " Average and standard deviation of the CV error are\n";
            std::cout << " saved in " << file_cvscore << '\n';
            std::cout << " Minimum CVSCORE at alpha = " << alphas[ialpha_minimum] << "\n\n";

            if (ialpha_minimum == optcontrol.num_l1_alpha - 1) {
                warn("run_auto_cv",
                     "The minimum CVSCORE occurs at CV_MINALPHA.\n"
                     " It is highly recommended to use CV_MINALPHA or CV_MINALPHA_RATIO.");
            }
        }
    }

    return alphas[ialpha_minimum];
}


auto Optimize::write_cvresult_to_file(const std::string &file_out, const std::vector<double> &alphas,
                                      const std::vector<double> &training_error,
                                      const std::vector<double> &validation_error,
                                      const std::vector<std::vector<int>> &nonzeros) const -> void
{
    std::vector<std::string> str_linearmodel{"Elastic-net", "Adaptive LASSO"};
    std::ofstream ofs_cv;
    ofs_cv.open(file_out.c_str(), std::ios::out);
    ofs_cv << "# Algorithm : Coordinate descent" << '\n';
    ofs_cv << "# Linear model : " << str_linearmodel[optcontrol.linear_model - 2] << '\n';
    ofs_cv << "# L1_RATIO = " << optcontrol.l1_ratio << '\n';
    ofs_cv << "# ENET_DNORM = " << std::setw(15) << optcontrol.displacement_normalization_factor << '\n';
    ofs_cv << "# STANDARDIZE = " << optcontrol.standardize << '\n';
    ofs_cv << "# CONV_TOL = " << std::setw(15) << optcontrol.tolerance_iteration << '\n';
    ofs_cv << "# L1 ALPHA, Fitting error, Validation error, Num. zero IFCs (2nd, 3rd, ...) \n";

    const auto maxorder = nonzeros[0].size();
    for (auto ialpha = 0; ialpha < training_error.size(); ++ialpha) {
        ofs_cv << std::setw(15) << alphas[ialpha];
        ofs_cv << std::setw(15) << training_error[ialpha];
        ofs_cv << std::setw(15) << validation_error[ialpha];
        for (auto i = 0; i < maxorder; ++i) {
            ofs_cv << std::setw(10) << nonzeros[ialpha][i];
        }
        ofs_cv << '\n';
    }
    ofs_cv.close();
}

auto Optimize::write_cvscore_to_file(const std::string &file_out, const std::vector<double> &alphas,
                                     const std::vector<double> &terr_mean, const std::vector<double> &terr_std,
                                     const std::vector<double> &verr_mean, const std::vector<double> &verr_std,
                                     const int ialpha_minimum, const size_t nsets) const -> void
{
    const auto nalphas = alphas.size();
    const auto n_terr = terr_mean.size();
    std::vector<std::string> str_linearmodel{"Elastic-net", "Adaptive LASSO"};
    std::ofstream ofs_cv;
    ofs_cv.open(file_out.c_str(), std::ios::out);
    ofs_cv << "# Algorithm : Coordinate descent\n";
    ofs_cv << "# Linear model : " << str_linearmodel[optcontrol.linear_model - 2] << '\n';
    ofs_cv << "# L1_RATIO = " << optcontrol.l1_ratio << '\n';
    ofs_cv << "# ENET_DNORM = " << std::setw(15) << optcontrol.displacement_normalization_factor << '\n';
    ofs_cv << "# STANDARDIZE = " << optcontrol.standardize << '\n';
    ofs_cv << "# CONV_TOL = " << std::setw(15) << optcontrol.tolerance_iteration << '\n';
    ofs_cv << "# " << nsets << "-fold cross-validation scores\n";
    ofs_cv << "# L1 ALPHA, Fitting error (mean, std), Validation error (mean, std) \n";

    const auto nsize = std::min(nalphas, n_terr);
    for (size_t ialpha = 0; ialpha < nsize; ++ialpha) {
        ofs_cv << std::setw(15) << alphas[ialpha];
        ofs_cv << std::setw(15) << terr_mean[ialpha];
        ofs_cv << std::setw(15) << terr_std[ialpha];
        ofs_cv << std::setw(15) << verr_mean[ialpha];
        ofs_cv << std::setw(15) << verr_std[ialpha];
        ofs_cv << '\n';
    }

    ofs_cv << "# Minimum CVSCORE at alpha = " << alphas[ialpha_minimum] << '\n';
    ofs_cv.close();
}

auto Optimize::set_errors_of_cvscore(std::vector<double> &terr_mean, std::vector<double> &terr_std,
                                     std::vector<double> &verr_mean, std::vector<double> &verr_std,
                                     const std::vector<std::vector<double>> &training_error_accum,
                                     const std::vector<std::vector<double>> &validation_error_accum) const -> void
{
    const auto nsets = training_error_accum.size();
    const auto nalphas = terr_mean.size();

    double sum_t, sum2_t;
    double sum_v, sum2_v;
    const auto factor = 1.0 / static_cast<double>(nsets);

    // The length of the training_error array may be different between different subsets
    // Let's use the shortest one to compute the mean and std.
    auto nmax_common = nalphas;
    for (auto iset = 0; iset < nsets; ++iset) {
        nmax_common = std::min(nmax_common, validation_error_accum[iset].size());
    }
    terr_mean.resize(nmax_common);
    terr_std.resize(nmax_common);
    verr_mean.resize(nmax_common);
    verr_std.resize(nmax_common);

    for (size_t ialpha = 0; ialpha < nmax_common; ++ialpha) {
        sum_t = 0.0;
        sum2_t = 0.0;
        sum_v = 0.0;
        sum2_v = 0.0;
        for (size_t iset = 0; iset < nsets; ++iset) {
            sum_t += training_error_accum[iset][ialpha];
            sum2_t += training_error_accum[iset][ialpha] * training_error_accum[iset][ialpha];
            sum_v += validation_error_accum[iset][ialpha];
            sum2_v += validation_error_accum[iset][ialpha] * validation_error_accum[iset][ialpha];
        }
        sum_t *= factor;
        sum2_t *= factor;
        sum_v *= factor;
        sum2_v *= factor;
        terr_mean[ialpha] = sum_t;
        terr_std[ialpha] = std::sqrt(sum2_t - sum_t * sum_t);
        verr_mean[ialpha] = sum_v;
        verr_std[ialpha] = std::sqrt(sum2_v - sum_v * sum_v);
    }
}

auto Optimize::get_ialpha_at_minimum_validation_error(const std::vector<double> &validation_error) -> int
{
    return std::distance(validation_error.begin(), std::min_element(validation_error.begin(), validation_error.end()));
}

auto Optimize::solution_path(const int maxorder, Eigen::MatrixXd &A, Eigen::VectorXd &b, Eigen::MatrixXd &A_validation,
                             Eigen::VectorXd &b_validation, const double fnorm, const double fnorm_validation,
                             const std::string &file_coef, const int verbosity,
                             const std::unique_ptr<Constraint> &constraint, const std::vector<double> &alphas,
                             std::vector<double> &training_error, std::vector<double> &validation_error,
                             std::vector<std::vector<int>> &nonzeros) const -> void
{
    int initialize_mode;
    int ncount_verr_consecutive_increase = 0;
    std::ofstream ofs_coef;

    std::vector<double> params_tmp;
    std::vector<int> nzero_lasso(maxorder);

    bool *has_prod;

    Eigen::MatrixXd Prod;
    Eigen::VectorXd grad0, grad, x;
    Eigen::VectorXd scale_beta, scale_beta_enet;
    Eigen::VectorXd factor_std;
    Eigen::VectorXd fdiff, fdiff_validation;
    Eigen::VectorXd mean, dev;

    size_t N_new = A.cols();
    size_t M = A.rows();
    size_t M_validation = A_validation.rows();

    Prod.setZero(N_new, N_new);
    grad0.resize(N_new);
    grad.resize(N_new);
    x.setZero(N_new);
    scale_beta.resize(N_new);
    scale_beta_enet.resize(N_new);
    factor_std.resize(N_new);
    fdiff.resize(M);
    fdiff_validation.resize(M_validation);

    allocate(has_prod, N_new);

    for (size_t i = 0; i < N_new; ++i) {
        has_prod[i] = false;
    }

    if (optcontrol.save_solution_path) {
        ofs_coef.open(file_coef.c_str(), std::ios::out);
        ofs_coef << "# L1 ALPHA, coefficients\n";
        params_tmp.resize(N_new);
    }

    if (optcontrol.standardize) {
        get_standardizer(A, mean, dev, factor_std, scale_beta);
        apply_standardizer(A, mean, dev);
        apply_standardizer(A_validation, mean, dev);
    } else {
        get_standardizer(A, mean, dev, factor_std, scale_beta);
    }

    training_error.clear();
    validation_error.clear();
    nonzeros.clear();

    // Start iteration

    grad0 = A.transpose() * b;
    grad = grad0;

    if (verbosity == 1) std::cout << std::setw(3);

    for (size_t ialpha = 0; ialpha < alphas.size(); ++ialpha) {

        const auto l1_alpha = alphas[ialpha];

        if (ialpha == 0) {
            initialize_mode = 0;
        } else {
            initialize_mode = 1;
        }

#pragma omp parallel for
        for (auto i = 0; i < N_new; ++i) {
            scale_beta_enet(i) = 1.0 / (1.0 / scale_beta(i) + (1.0 - optcontrol.l1_ratio) * l1_alpha);
        }

        coordinate_descent(M,
                           N_new,
                           l1_alpha,
                           initialize_mode,
                           x,
                           A,
                           b,
                           grad0,
                           has_prod,
                           Prod,
                           grad,
                           fnorm,
                           scale_beta_enet,
                           verbosity);

        double correction_intercept = 0.0;
        for (size_t i = 0; i < N_new; ++i) {
            correction_intercept += x(i) * mean(i) * factor_std(i);
        }
        fdiff = A * x - b + correction_intercept * Eigen::VectorXd::Ones(M);
        fdiff_validation = A_validation * x - b_validation + correction_intercept * Eigen::VectorXd::Ones(M_validation);
        const auto res1 = fdiff.dot(fdiff) / (fnorm * fnorm);
        const auto res2 = fdiff_validation.dot(fdiff_validation) / (fnorm_validation * fnorm_validation);

        get_number_of_zero_coefs(maxorder, constraint, x, nzero_lasso);

        training_error.push_back(std::sqrt(res1));
        validation_error.push_back(std::sqrt(res2));
        nonzeros.push_back(nzero_lasso);

        if (optcontrol.save_solution_path) {
            ofs_coef << std::setw(15) << l1_alpha;

            for (auto i = 0; i < N_new; ++i)
                params_tmp[i] = x[i];

            apply_scaler_force_constants(maxorder,
                                         optcontrol.displacement_normalization_factor,
                                         constraint,
                                         params_tmp);

            for (auto i = 0; i < N_new; ++i) {
                ofs_coef << std::setw(15) << params_tmp[i];
            }
            ofs_coef << '\n';
        }

        if (verbosity == 1) {
            std::cout << '.' << std::flush;
            if (ialpha % 25 == 24) {
                std::cout << '\n';
                std::cout << std::setw(3);
            }
        }

        if (optcontrol.stop_criterion > 0) {
            const auto nsize_now = validation_error.size();
            if (nsize_now > 1) {
                if (validation_error[nsize_now - 1] > validation_error[nsize_now - 2]) {
                    ncount_verr_consecutive_increase += 1;
                } else {
                    ncount_verr_consecutive_increase = 0;
                }
            }
            if (ncount_verr_consecutive_increase >= optcontrol.stop_criterion) {
                break;
            }
        }
    }

    if (verbosity == 1) std::cout << '\n';

    if (verbosity == 1 && (alphas.size() > validation_error.size())) {
        std::cout << "  STOP_CRITERION is satisfied: The solution path calculation has stopped.\n";
    }

    if (optcontrol.save_solution_path) {
        ofs_coef.close();
        params_tmp.clear();
        params_tmp.shrink_to_fit();
    }
    deallocate(has_prod);
}

auto Optimize::compute_alphas(const double l1_alpha_max, const double l1_alpha_min, const int num_l1_alpha,
                              std::vector<double> &alphas) -> void
{
    alphas.resize(num_l1_alpha);

    for (auto ialpha = 0; ialpha < num_l1_alpha; ++ialpha) {

        const auto l1_alpha =
            l1_alpha_min * std::pow(l1_alpha_max / l1_alpha_min,
                                    static_cast<double>(num_l1_alpha - ialpha - 1) / static_cast<double>(num_l1_alpha));

        alphas[ialpha] = l1_alpha;
    }
}

auto Optimize::optimize_with_given_l1alpha(const int maxorder, const size_t M, const size_t N_new,
                                           const std::unique_ptr<Fcs> &fcs, const std::unique_ptr<Symmetry> &symmetry,
                                           const std::unique_ptr<Constraint> &constraint, const int verbosity,
                                           std::vector<double> &param_out) const -> void
{
    // Start Elastic-net or adaptive lasso optimization
    int i;
    bool *has_prod;

    Eigen::MatrixXd A, Prod;
    Eigen::VectorXd b, grad0, grad, x;
    Eigen::VectorXd scale_beta, factor_std;
    Eigen::VectorXd fdiff;
    Eigen::VectorXd mean, dev;
    Eigen::VectorXd weight_adalasso;


    std::unique_ptr<SensingMatrix> matrix_train = std::make_unique<SensingMatrix>();
    get_matrix_elements_unified(maxorder,
                                matrix_train,
                                u_train,
                                f_train,
                                symmetry,
                                fcs,
                                constraint,
                                true,
                                false,
                                false);

    double fnorm = 0.0;
    for (const auto &it: matrix_train->original_forces) {
        fnorm += it * it;
    }
    fnorm = std::sqrt(fnorm);

    A = Eigen::Map<Eigen::MatrixXd>(matrix_train->amat_dense.data(), M, N_new);
    b = Eigen::Map<Eigen::VectorXd>(matrix_train->bvec.data(), M);

    if (optcontrol.linear_model == 3) {
        // Use QR decomposition with pivoting to check if system is well-determined
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_solver(A);

        // Check if the system is underdetermined or rank-deficient
        const auto rank = qr_solver.rank();

        if (rank < N_new) {
            std::string error_msg = "Adaptive lasso failed: The least squares problem is rank-deficient.\n"
                                    "  Matrix rank = " +
                                    std::to_string(rank) + ", Number of parameters = " + std::to_string(N_new) +
                                    "\n"
                                    "  Number of data points = " +
                                    std::to_string(M) +
                                    "\n"
                                    "  This typically occurs when there are too few training data\n"
                                    "  or when some parameters cannot be uniquely determined.\n"
                                    "  Please try one of the following:\n"
                                    "  - Increase the amount of training data (DFSET)\n"
                                    "  - Use LMODEL = 1 (least-squares) or 2 (elastic-net) instead\n"
                                    "  - Reduce the interaction cutoff distance\n";
            ALM_NS::exit("optimize_elasticnet", error_msg.c_str());
        }

        Eigen::VectorXd x_ols = qr_solver.solve(b);
        weight_adalasso = x_ols.cwiseAbs();

        // Check if any weights are too small (could cause numerical issues)
        const double min_weight_threshold = 1.0e-10;
        int n_zero_weights = 0;
        double max_weight = weight_adalasso.maxCoeff();

        for (int i = 0; i < weight_adalasso.size(); ++i) {
            if (weight_adalasso[i] < min_weight_threshold) {
                n_zero_weights++;
            }
        }

        if (n_zero_weights > 0) {
            if (verbosity > 0) {
                std::cout << "  WARNING: Adaptive lasso detected " << n_zero_weights << " near-zero weights (< "
                          << min_weight_threshold << ").\n";
                std::cout << "  Maximum weight = " << max_weight << "\n";
                std::cout << "  This may indicate an underdetermined or ill-conditioned problem.\n";
                std::cout << "  The optimization will continue but results may be unreliable.\n";
                std::cout << "  Consider using LMODEL = 2 (elastic-net) instead.\n\n";
            }
        }

        A = A * weight_adalasso.asDiagonal();
    }

    Prod.setZero(N_new, N_new);
    grad0.resize(N_new);
    grad.resize(N_new);
    x.setZero(N_new);
    scale_beta.resize(N_new);
    factor_std.resize(N_new);
    fdiff.resize(M);

    allocate(has_prod, N_new);

    for (i = 0; i < N_new; ++i) {
        has_prod[i] = false;
    }

    if (verbosity > 0) {
        if (optcontrol.linear_model == 2) {
            std::cout << "  Elastic-net minimization with the following parameters:" << '\n';
            std::cout << "   L1_RATIO = " << optcontrol.l1_ratio << '\n';
            std::cout << "   ENET_DNORM = " << std::setw(15) << optcontrol.displacement_normalization_factor << '\n';
            if (optcontrol.standardize) {
                std::cout << "  STANDARDIZE = 1 : Standardization will be performed for matrix A and vector b.\n";
                std::cout << "                    The ENET_DNORM-tag will be neglected.\n\n";
            } else {
                std::cout << "  STANDARDIZE = 0 : No standardization of matrix A and vector b.\n";
                std::cout << "                    Columns of matrix A will be scaled by the ENET_DNORM value.\n\n";
            }
        } else if (optcontrol.linear_model == 3) {
            std::cout << "  Adaptive LASSO optimization with the following parameters:\n";
        }
        std::cout << "   L1_ALPHA = " << std::setw(15) << optcontrol.l1_alpha << '\n';
        std::cout << "   CONV_TOL = " << std::setw(15) << optcontrol.tolerance_iteration << '\n';
        std::cout << "   MAXITER = " << std::setw(5) << optcontrol.maxnum_iteration << '\n';
        std::cout << '\n';
    }

    get_standardizer(A, mean, dev, factor_std, scale_beta);

    // Standardize if necessary only for the elastic net
    if (optcontrol.standardize && optcontrol.linear_model == 2) {
        apply_standardizer(A, mean, dev);
    }
    grad0 = A.transpose() * b;
    grad = grad0;

#pragma omp parallel for
    for (i = 0; i < N_new; ++i) {
        scale_beta(i) = 1.0 / (1.0 / scale_beta(i) + (1.0 - optcontrol.l1_ratio) * optcontrol.l1_alpha);
    }

    // Coordinate Descent Method
    coordinate_descent(M,
                       N_new,
                       optcontrol.l1_alpha,
                       0,
                       x,
                       A,
                       b,
                       grad0,
                       has_prod,
                       Prod,
                       grad,
                       fnorm,
                       scale_beta,
                       verbosity);

    if (optcontrol.linear_model == 2) {
        for (i = 0; i < N_new; ++i) {
            param_out[i] = x[i] * factor_std[i];
        }
    } else if (optcontrol.linear_model == 3) {
        for (i = 0; i < N_new; ++i) {
            param_out[i] = x[i] * factor_std[i] * weight_adalasso[i];
        }
    }

    if (verbosity > 0) {
        double correction_intercept = 0.0;
        for (size_t i = 0; i < N_new; ++i) {
            correction_intercept += x(i) * mean(i) * factor_std(i);
        }
        fdiff = A * x - b + correction_intercept * Eigen::VectorXd::Ones(M);
        const auto res1 = fdiff.dot(fdiff) / (fnorm * fnorm);
        std::cout << "  RESIDUAL (%): " << std::sqrt(res1) * 100.0 << '\n';
    }

    deallocate(has_prod);

    if (optcontrol.debiase_after_l1opt) {
        if (optcontrol.linear_model == 2) {
            run_least_squares_with_nonzero_coefs(A, b, factor_std, param_out, verbosity);
        } else if (optcontrol.linear_model == 3) {
            // The columns of A are scaled by the adaptive lasso weights,
            // so the OLS solution must be scaled back by them as well.
            const Eigen::VectorXd factor_adalasso = factor_std.cwiseProduct(weight_adalasso);
            run_least_squares_with_nonzero_coefs(A, b, factor_adalasso, param_out, verbosity);
        }
    }
}


auto Optimize::run_least_squares_with_nonzero_coefs(const Eigen::MatrixXd &A_in, const Eigen::VectorXd &b_in,
                                                    const Eigen::VectorXd &factor_std,
                                                    std::vector<double> &params_inout,
                                                    const int verbosity) const -> void
{
    // Perform OLS fitting to the features selected by LASSO for reducing the bias.

    if (verbosity > 0) {
        std::cout << " DEBIAS_OLS = 1: Attempt to reduce the bias of LASSO by performing OLS fitting\n";
        std::cout << "                 with features selected by LASSO.\n";
    }

    const auto N_new = A_in.cols();
    const auto M = A_in.rows();

    std::vector<int> nonzero_index, zero_index;

    for (auto i = 0; i < N_new; ++i) {
        if (std::abs(params_inout[i]) >= eps) {
            nonzero_index.push_back(i);
        } else {
            zero_index.push_back(i);
        }
    }

    const auto N_nonzero = nonzero_index.size();
    Eigen::MatrixXd A_nonzero(M, N_nonzero);

    for (auto i = 0; i < N_nonzero; ++i) {
        A_nonzero.col(i) = A_in.col(nonzero_index[i]);
    }
    Eigen::VectorXd x_nonzero = A_nonzero.colPivHouseholderQr().solve(b_in);

    for (auto i = 0; i < N_new; ++i)
        params_inout[i] = 0.0;
    for (auto i = 0; i < N_nonzero; ++i) {
        params_inout[nonzero_index[i]] = x_nonzero[i] * factor_std[nonzero_index[i]];
    }
}

auto Optimize::get_number_of_zero_coefs(const int maxorder, const std::unique_ptr<Constraint> &constraint,
                                        const Eigen::VectorXd &x, std::vector<int> &nzeros) -> void
{
    // Count the number of zero parameters
    size_t iparam = 0;
    nzeros.resize(maxorder);
    for (auto i = 0; i < maxorder; ++i) {
        nzeros[i] = 0;
        for (const auto &it: constraint->get_index_bimap(i)) {
            const auto inew = it.left + iparam;
            if (std::abs(x[inew]) < eps) ++nzeros[i];
        }
        iparam += constraint->get_index_bimap(i).size();
    }
}


auto Optimize::get_standardizer(const Eigen::MatrixXd &Amat, Eigen::VectorXd &mean, Eigen::VectorXd &dev,
                                Eigen::VectorXd &factor_std, Eigen::VectorXd &scale_beta) const -> void
{
    const auto nrows = Amat.rows();
    const auto ncols = Amat.cols();

    if (mean.size() != ncols) mean.resize(ncols);
    if (dev.size() != ncols) dev.resize(ncols);
    if (factor_std.size() != ncols) factor_std.resize(ncols);
    if (scale_beta.size() != ncols) scale_beta.resize(ncols);

    const auto inv_nrows = 1.0 / static_cast<double>(nrows);
    double sum1, sum2;

    if (optcontrol.standardize) {
        for (auto j = 0; j < ncols; ++j) {
            sum1 = Amat.col(j).sum() * inv_nrows;
            sum2 = Amat.col(j).dot(Amat.col(j)) * inv_nrows;
            mean(j) = sum1;
            dev(j) = std::sqrt(sum2 - sum1 * sum1);
            factor_std(j) = 1.0 / dev(j);
            scale_beta(j) = 1.0;
        }
    } else {
        for (auto j = 0; j < ncols; ++j) {
            sum2 = Amat.col(j).dot(Amat.col(j)) * inv_nrows;
            mean(j) = 0.0;
            dev(j) = 1.0;
            factor_std(j) = 1.0;
            scale_beta(j) = 1.0 / sum2;
        }
    }
}

auto Optimize::apply_standardizer(Eigen::MatrixXd &Amat, const Eigen::VectorXd &mean,
                                  const Eigen::VectorXd &dev) const -> void
{
    const auto ncols = Amat.cols();
    const auto nrows = Amat.rows();
    if (mean.size() != ncols || dev.size() != ncols) {
        exit("apply_standardizer", "The number of colums is inconsistent.");
    }

    for (auto i = 0; i < nrows; ++i) {
        for (auto j = 0; j < ncols; ++j) {
            Amat(i, j) = (Amat(i, j) - mean(j)) / dev(j);
        }
    }
}

auto Optimize::get_estimated_max_alpha(const Eigen::MatrixXd &Amat, const Eigen::VectorXd &bvec) const -> double
{
    const auto ncols = Amat.cols();
    const auto nrows = Amat.rows();
    Eigen::MatrixXd C = Amat;

    Eigen::VectorXd mean = Eigen::VectorXd::Zero(Amat.cols());
    Eigen::VectorXd dev = Eigen::VectorXd::Ones(Amat.cols());

    if (optcontrol.standardize) {
        Eigen::VectorXd factor_std, scale_beta;
        factor_std.resize(Amat.cols());
        scale_beta.resize(Amat.cols());
        get_standardizer(Amat, mean, dev, factor_std, scale_beta);
    }

    for (auto i = 0; i < nrows; ++i) {
        for (auto j = 0; j < ncols; ++j) {
            C(i, j) = (C(i, j) - mean(j)) / dev(j);
        }
    }

    C = C.transpose() * bvec;
    auto max_alpha = 0.0;

    for (auto i = 0; i < ncols; ++i) {
        max_alpha = std::max<double>(max_alpha, std::abs(C(i)));
    }
    max_alpha /= static_cast<double>(nrows);

    return max_alpha;
}

auto Optimize::apply_scaler_displacement(std::vector<std::vector<double>> &u_inout, const double normalization_factor,
                                         const bool scale_back) -> void
{
    const auto nrows = u_inout.size();
    const auto ncols = u_inout[0].size();

    if (scale_back) {
        for (auto i = 0; i < nrows; ++i) {
            for (auto j = 0; j < ncols; ++j) {
                u_inout[i][j] *= normalization_factor;
            }
        }
    } else {
        const auto inv_scale_factor = 1.0 / normalization_factor;
        for (auto i = 0; i < nrows; ++i) {
            for (auto j = 0; j < ncols; ++j) {
                u_inout[i][j] *= inv_scale_factor;
            }
        }
    }
}

auto Optimize::apply_scaler_constraint(const int maxorder, const double normalization_factor,
                                       const std::unique_ptr<Constraint> &constraint, const bool scale_back) -> void
{
    if (scale_back) {
        for (auto i = 0; i < maxorder; ++i) {
            const auto scale_factor = 1.0 / std::pow(normalization_factor, i + 1);
            for (auto j = 0; j < constraint->get_const_fix(i).size(); ++j) {
                const auto scaled_val = constraint->get_const_fix(i)[j].val_to_fix * scale_factor;
                constraint->set_const_fix_val_to_fix(i, j, scaled_val);
            }
        }
    } else {
        for (auto i = 0; i < maxorder; ++i) {
            const auto scale_factor = std::pow(normalization_factor, i + 1);
            for (auto j = 0; j < constraint->get_const_fix(i).size(); ++j) {
                const auto scaled_val = constraint->get_const_fix(i)[j].val_to_fix * scale_factor;
                constraint->set_const_fix_val_to_fix(i, j, scaled_val);
            }
        }
    }
}

auto Optimize::apply_scaler_force_constants(const int maxorder, const double normalization_factor,
                                            const std::unique_ptr<Constraint> &constraint,
                                            std::vector<double> &param_inout) -> void
{
    auto k = 0;
    for (auto i = 0; i < maxorder; ++i) {
        const auto scale_factor = 1.0 / std::pow(normalization_factor, i + 1);

        for (auto j = 0; j < constraint->get_index_bimap(i).size(); ++j) {
            param_inout[k] *= scale_factor;
            ++k;
        }
    }
}

auto Optimize::apply_scalers(const int maxorder, const std::unique_ptr<Constraint> &constraint) -> void
{
    apply_scaler_displacement(u_train, optcontrol.displacement_normalization_factor);
    apply_scaler_constraint(maxorder, optcontrol.displacement_normalization_factor, constraint);

    if (optcontrol.cross_validation == -1) {
        apply_scaler_displacement(u_validation, optcontrol.displacement_normalization_factor);
    }
}

auto Optimize::finalize_scalers(const int maxorder, const std::unique_ptr<Constraint> &constraint) -> void
{
    apply_scaler_displacement(u_train, optcontrol.displacement_normalization_factor, true);
    apply_scaler_constraint(maxorder, optcontrol.displacement_normalization_factor, constraint, true);
    if (optcontrol.cross_validation == -1) {
        apply_scaler_displacement(u_validation, optcontrol.displacement_normalization_factor, true);
    }
}

auto Optimize::apply_basis_converter(std::vector<std::vector<double>> &u_multi, Eigen::Matrix3d cmat) -> void
{
    // Convert the basis of displacements from Cartesian to fractional
    const auto nrows = u_multi.size();
    const auto ncols = u_multi[0].size();
    size_t i, j;
    Eigen::Vector3d vec_tmp;

    const auto nat = ncols / 3;
    for (i = 0; i < nrows; ++i) {
        for (j = 0; j < nat; ++j) {
            for (int k = 0; k < 3; ++k) {
                vec_tmp(k) = u_multi[i][3 * j + k];
            }
            vec_tmp = cmat * vec_tmp;
            for (int k = 0; k < 3; ++k) {
                u_multi[i][3 * j + k] = vec_tmp(k);
            }
        }
    }
}

auto Optimize::apply_basis_converter_amat(const int natmin3, const int ncols, double **amat_orig_tmp,
                                          Eigen::Matrix3d cmat) -> void
{
    const auto natmin = natmin3 / 3;
    Eigen::Vector3d vec_tmp;
    const Eigen::Matrix3d cmat_t = cmat.transpose();

    for (auto icol = 0; icol < ncols; ++icol) {
        for (auto iat = 0; iat < natmin; ++iat) {
            for (auto i = 0; i < 3; ++i) {
                vec_tmp(i) = amat_orig_tmp[3 * iat + i][icol];
            }
            vec_tmp = cmat_t * vec_tmp;
            for (auto i = 0; i < 3; ++i) {
                amat_orig_tmp[3 * iat + i][icol] = vec_tmp(i);
            }
        }
    }
}


auto Optimize::set_u_train(const std::vector<std::vector<double>> &u_train_in) -> void
{
    u_train.clear();
    u_train = u_train_in;
    u_train.shrink_to_fit();
}

auto Optimize::set_f_train(const std::vector<std::vector<double>> &f_train_in) -> void
{
    f_train.clear();
    f_train = f_train_in;
    f_train.shrink_to_fit();
}

auto Optimize::set_validation_data(const std::vector<std::vector<double>> &u_validation_in,
                                   const std::vector<std::vector<double>> &f_validation_in) -> void
{
    u_validation.clear();
    f_validation.clear();
    u_validation = u_validation_in;
    f_validation = f_validation_in;
    u_validation.shrink_to_fit();
    f_validation.shrink_to_fit();
}

auto Optimize::get_u_train() const -> std::vector<std::vector<double>>
{
    return u_train;
}

auto Optimize::get_f_train() const -> std::vector<std::vector<double>>
{
    return f_train;
}

auto Optimize::get_number_of_data() const -> size_t
{
    return u_train.size();
}

auto Optimize::set_fcs_values(const int maxorder, double *fc_in, std::vector<size_t> *nequiv,
                              const std::unique_ptr<Constraint> &constraint) -> void
{
    // fc_in: irreducible set of force constants
    // fc_length: dimension of params (can differ from that of fc_in)

    size_t i;

    size_t N = 0;
    size_t Nirred = 0;
    for (i = 0; i < maxorder; ++i) {
        N += nequiv[i].size();
        Nirred += constraint->get_index_bimap(i).size();
    }

    std::vector<double> param_in(Nirred, 0.0);
    std::vector<double> param_out(N, 0.0);

    for (i = 0; i < Nirred; ++i) {
        param_in[i] = fc_in[i];
    }
    recover_original_forceconstants(maxorder, param_in, param_out, nequiv, constraint);
    if (params) {
        deallocate(params);
    }
    allocate(params, N);
    for (i = 0; i < N; ++i) {
        params[i] = param_out[i];
    }
}

auto Optimize::get_number_of_rows_sensing_matrix() const -> size_t
{
    return u_train.size() * u_train[0].size();
}


auto Optimize::fit_algebraic_constraints(const size_t N, const size_t M, double *amat, const double *bvec,
                                         std::vector<double> &param_out, const double fnorm, const int maxorder,
                                         const std::unique_ptr<Fcs> &fcs, const std::unique_ptr<Constraint> &constraint,
                                         const int verbosity) const -> int
{
    int i;
    int nrhs = 1, nrank, INFO, M_tmp, N_tmp;
    auto rcond = -1.0;
    double *WORK, *S, *fsum2;

    if (verbosity > 0) {
        std::cout << "  Entering fitting routine: SVD with constraints considered algebraically.\n";
    }

    auto LMIN = std::min<int>(M, N);
    auto LMAX = std::max<int>(M, N);

    auto LWORK = 3 * LMIN + std::max<int>(2 * LMIN, LMAX);
    LWORK = 2 * LWORK;

    allocate(WORK, LWORK);
    allocate(S, LMIN);
    allocate(fsum2, LMAX);

    for (i = 0; i < M; ++i) {
        fsum2[i] = bvec[i];
    }
    for (i = M; i < LMAX; ++i)
        fsum2[i] = 0.0;

    if (verbosity > 0) std::cout << "  SVD has started ... " << std::flush;

    // Fitting with singular value decomposition
    // M_tmp and N_tmp are prepared to cast N and M to (non-const) int.
    M_tmp = M;
    N_tmp = N;
    dgelss_(&M_tmp, &N_tmp, &nrhs, amat, &M_tmp, fsum2, &LMAX, S, &rcond, &nrank, WORK, &LWORK, &INFO);

    deallocate(WORK);
    deallocate(S);

    if (verbosity > 0) {
        std::cout << "finished !\n\n";
        std::cout << "  RANK of the matrix = " << nrank << '\n' << std::flush;
    }

    if (nrank < N) {
        std::cout << " **************************************************************************\n";
        std::cout << "  WARNING : Rank deficient                                                 \n\n";
        std::cout << "  Force constants could not be determined uniquely because                 \n";
        std::cout << "  the sensing matrix is not full rank.                                     \n";
        std::cout << "  You may need to reduce the cutoff radii and/or increase the number of    \n";
        std::cout << "  training datasets.                                                       \n";
        std::cout << " **************************************************************************\n";
    }

    if (nrank == N && verbosity > 0) {
        auto f_residual = 0.0;
        for (i = N; i < M; ++i) {
            f_residual += pow2(fsum2[i]);
        }
        std::cout << '\n';
        std::cout << "  Residual sum of squares for the solution: " << sqrt(f_residual) << '\n';
        std::cout << "  Fitting error (%) : " << sqrt(f_residual / (fnorm * fnorm)) * 100.0 << '\n';
    }

    if (INFO == 0) {
        std::vector<double> param_irred(N, 0.0);
        for (i = 0; i < LMIN; ++i)
            param_irred[i] = fsum2[i];
        deallocate(fsum2);

        // Recover reducible set of force constants

        recover_original_forceconstants(maxorder, param_irred, param_out, fcs->get_nequiv(), constraint);
    }

    return INFO;
}

auto Optimize::solve_normal_equation(const size_t N, double *amat, double *bvec, std::vector<double> &param_out,
                                     const double fnorm, const int maxorder, const std::unique_ptr<Fcs> &fcs,
                                     const std::unique_ptr<Constraint> &constraint, const int verbosity,
                                     const bool algebraic_constraint) const -> int
{
    if (verbosity > 0) {
        std::cout << "  Entering fitting routine: Solve normal equation (A^T A)x= (A^T b) by Cholesky.\n" << std::flush;
    }
    int info;

    char uplo = 'L';
    int N_ = N;
    int m = 1;

    dpotrf_(&uplo, &N_, amat, &N_, &info);
    dpotrs_(&uplo, &N_, &m, amat, &N_, bvec, &N_, &info);

    if (verbosity > 0) std::cout << " finished. \n" << std::flush;

    if (info == 0) {
        if (algebraic_constraint) {

            std::vector<double> param_irred(N, 0.0);
            for (auto i = 0; i < N; ++i)
                param_irred[i] = bvec[i];
            // Recover reducible set of force constants

            recover_original_forceconstants(maxorder, param_irred, param_out, fcs->get_nequiv(), constraint);
        } else {
            param_out.resize(N, 0.0);
            for (size_t i = 0; i < N; ++i) {
                param_out[i] = bvec[i];
            }
        }
    }
    return info;
}

auto Optimize::get_matrix_elements_unified(const int maxorder, std::unique_ptr<SensingMatrix> &matrix_out,
                                           const std::vector<std::vector<double>> &u_in,
                                           const std::vector<std::vector<double>> &f_in,
                                           const std::unique_ptr<Symmetry> &symmetry, const std::unique_ptr<Fcs> &fcs,
                                           const std::unique_ptr<Constraint> &constraint, const bool compact,
                                           const bool sparse, const bool return_ata, const int verbosity) const -> void
{
    // Construct the matrix and vector necessary for estimating force constants.
    // The computed results are stored in matrix_out, and the updated variables in matrix_out
    // changes depending on the input options (compact, sparse, return_ata).
    //
    // compact: If true, the matrix A and vector b are projected to the null space of the constraint matrix.
    // return_ata: If true, compute (A^T A) and (A^T b) instead of A and b for solving the normal equation.
    //             If false, compute A and b for solving the least-square problem.
    // sparse: If true, store the matrix A in sparse form and save it in matrix_out->amat_sparse.
    //         If false, store the matrix A in dense form and save if in matrix_out->amat_dense.

    if (u_in.size() != f_in.size()) {
        exit("get_matrix_elements_unified", "The lengths of displacement array and force array are diferent.");
    }

    size_t i, j;
    const auto natmin = symmetry->get_nat_trueprim();
    const auto ndata_fit = u_in.size();
    const auto ncycle = ndata_fit * symmetry->get_ntran();
    const auto nrows = ndata_fit * u_in[0].size(); // length of the flattened displacement array

    size_t ncols = 0;
    size_t ncols_new = 0;
    for (i = 0; i < maxorder; ++i) {
        ncols += fcs->get_nequiv()[i].size();
    }

    if (compact) {
        for (i = 0; i < maxorder; ++i) {
            ncols_new += constraint->get_index_bimap(i).size();
        }
    } else {
        ncols_new = ncols;
    }

    std::vector<std::vector<double>> u_multi, f_multi;
    data_multiplier(u_in, u_multi, symmetry);
    data_multiplier(f_in, f_multi, symmetry);

    if (fcs->get_forceconstant_basis() == "Lattice") {
        apply_basis_converter(u_multi, fcs->get_basis_conversion_matrix());
    }

    std::vector<int> ind_tmp(maxorder + 1);
    // Precompute the product of gamma and sign,
    // which does not change durint the iteration over the training data
    std::vector<std::vector<double>> gamma_precomputed(maxorder);
    for (auto order = 0; order < maxorder; ++order) {
        auto ii = 0;

        gamma_precomputed[order].resize(fcs->get_fc_table()[order].size(), 0.0);

        for (const auto &iter: fcs->get_nequiv()[order]) {
            for (i = 0; i < iter; ++i) {
                ind_tmp[0] = fcs->get_fc_table()[order][ii].elems[0];
                for (j = 1; j < order + 2; ++j) {
                    ind_tmp[j] = fcs->get_fc_table()[order][ii].elems[j];
                }
                gamma_precomputed[order][ii] = gamma(order + 2, ind_tmp.data()) * fcs->get_fc_table()[order][ii].sign;
                ++ii;
            }
        }
    }

    matrix_out->original_forces.resize(nrows, 0.0);

    if (compact) {
        if (return_ata) {
            if (sparse) {
                if (verbosity > 0) {
                    std::cout << "  Calculate the sensing matrix A using sparse data type\n";
                    std::cout << "  This is more memory efficient when the input displacements are sparse\n";
                    std::cout << "  Directly construct (A^T A) and (A^T b)\n";
                }

                matrix_out->amat_sparse.resize(ncols_new, ncols_new);
                matrix_out->bvec.resize(ncols_new, 0.0);

                get_matrix_elements_normal_equation2(maxorder,
                                                     ncycle,
                                                     nrows,
                                                     ncols,
                                                     ncols_new,
                                                     matrix_out,
                                                     u_multi,
                                                     f_multi,
                                                     gamma_precomputed,
                                                     symmetry,
                                                     fcs,
                                                     constraint,
                                                     true);
            } else {
                if (verbosity > 0) {
                    std::cout << "  Calculate the sensing matrix A using dense data type\n";
                    std::cout << "  Directly construct (A^T A) and (A^T b)\n";
                    const auto memory_full =
                        static_cast<float>(nrows * ncols_new * 8) / static_cast<float>(1024 * 1024 * 1024);
                    const auto memory_chunk = static_cast<float>(optcontrol.chunk_size * ncols_new * 8) /
                                              static_cast<float>(1024 * 1024 * 1024);
                    const auto memory_ata = static_cast<float>(ncols_new * ncols_new * 8) / (1024 * 1024 * 1024);
                    std::cout << "  At least " << std::fixed << std::setprecision(3)
                              << std::max(memory_ata, memory_chunk) << " GiB of memory will be allocated.\n";
                }

                matrix_out->amat_dense.resize(ncols_new * ncols_new, 0.0);
                matrix_out->bvec.resize(ncols_new, 0.0);

                get_matrix_elements_normal_equation2(maxorder,
                                                     ncycle,
                                                     nrows,
                                                     ncols,
                                                     ncols_new,
                                                     matrix_out,
                                                     u_multi,
                                                     f_multi,
                                                     gamma_precomputed,
                                                     symmetry,
                                                     fcs,
                                                     constraint,
                                                     false);
            }

        } else {
            // return A and b
            if (sparse) {
                if (verbosity > 0) {
                    std::cout << "  Calculate the sensing matrix A using sparse data type\n";
                    std::cout << "  This is more memory efficient when the input displacements are sparse\n";
                }

                matrix_out->amat_sparse.resize(nrows, ncols_new);
                matrix_out->bvec.resize(nrows, 0.0);

                get_matrix_elements2(maxorder,
                                     ncycle,
                                     nrows,
                                     ncols,
                                     ncols_new,
                                     matrix_out,
                                     u_multi,
                                     f_multi,
                                     gamma_precomputed,
                                     symmetry,
                                     fcs,
                                     constraint,
                                     true);


            } else {

                if (verbosity > 0) {
                    std::cout << "  Calculate the sensing matrix A using dense data type\n";
                    std::cout << "  At least " << std::fixed << std::setprecision(3)
                              << static_cast<float>(nrows * ncols_new * 8) / static_cast<float>(1024 * 1024 * 1024)
                              << " GiB of memory will be allocated.\n";
                }

                matrix_out->amat_dense.resize(nrows * ncols_new, 0.0);
                matrix_out->bvec.resize(nrows, 0.0);

                get_matrix_elements2(maxorder,
                                     ncycle,
                                     nrows,
                                     ncols,
                                     ncols_new,
                                     matrix_out,
                                     u_multi,
                                     f_multi,
                                     gamma_precomputed,
                                     symmetry,
                                     fcs,
                                     constraint,
                                     false);
            }
        }

    } else {

        if (return_ata) {

            if (sparse) {

                if (verbosity > 0) {
                    std::cout << "  Calculate the sensing matrix A using sparse data type\n";
                    std::cout << "  Directly construct (A^T A) and (A^T b)\n";
                }

                matrix_out->amat_sparse.resize(ncols, ncols);
                matrix_out->bvec.resize(ncols, 0.0);

                get_matrix_elements_normal_equation2(maxorder,
                                                     ncycle,
                                                     nrows,
                                                     ncols,
                                                     ncols,
                                                     matrix_out,
                                                     u_multi,
                                                     f_multi,
                                                     gamma_precomputed,
                                                     symmetry,
                                                     fcs,
                                                     constraint,
                                                     true);


            } else {
                if (verbosity > 0) {
                    std::cout << "  Calculate the sensing matrix A using dense data type\n";
                    std::cout << "  Directly construct (A^T A) and (A^T b)\n";
                    const auto memory_full =
                        static_cast<float>(nrows * ncols * 8) / static_cast<float>(1024 * 1024 * 1024);
                    const auto memory_chunk =
                        static_cast<float>(optcontrol.chunk_size * ncols * 8) / static_cast<float>(1024 * 1024 * 1024);
                    const auto memory_ata = static_cast<float>(ncols * ncols * 8) / (1024 * 1024 * 1024);
                    std::cout << "  At least " << std::fixed << std::setprecision(3)
                              << std::max(memory_ata, memory_chunk) << " GiB of memory will be allocated.\n";
                }

                matrix_out->amat_dense.resize(ncols * ncols, 0.0);
                matrix_out->bvec.resize(ncols, 0.0);

                get_matrix_elements_normal_equation2(maxorder,
                                                     ncycle,
                                                     nrows,
                                                     ncols,
                                                     ncols,
                                                     matrix_out,
                                                     u_multi,
                                                     f_multi,
                                                     gamma_precomputed,
                                                     symmetry,
                                                     fcs,
                                                     constraint,
                                                     false);
            }

        } else {
            if (sparse) {
                if (verbosity > 0) {
                    std::cout << "  Calculate the sensing matrix A using sparse data type\n";
                    std::cout << "  This is more memory efficient when the input displacements are sparse\n";
                }

                matrix_out->amat_sparse.resize(nrows, ncols);
                matrix_out->bvec.resize(nrows, 0.0);
                get_matrix_elements2(maxorder,
                                     ncycle,
                                     nrows,
                                     ncols,
                                     ncols_new,
                                     matrix_out,
                                     u_multi,
                                     f_multi,
                                     gamma_precomputed,
                                     symmetry,
                                     fcs,
                                     constraint,
                                     true);
            } else {
                if (verbosity > 0) {
                    std::cout << "  Calculate the sensing matrix A using dense data type\n";
                    std::cout << "  At least " << std::fixed << std::setprecision(3)
                              << static_cast<float>(nrows * ncols * 8) / static_cast<float>(1024 * 1024 * 1024)
                              << " GiB of memory will be allocated.\n";
                }

                matrix_out->amat_dense.resize(nrows * ncols, 0.0);
                matrix_out->bvec.resize(nrows, 0.0);
                get_matrix_elements2(maxorder,
                                     ncycle,
                                     nrows,
                                     ncols,
                                     ncols_new,
                                     matrix_out,
                                     u_multi,
                                     f_multi,
                                     gamma_precomputed,
                                     symmetry,
                                     fcs,
                                     constraint,
                                     false);
            }
        }
    }
}


auto Optimize::get_matrix_elements2(const int maxorder, const size_t ncycle, const size_t nrows, const size_t ncols,
                                    const size_t ncols_compact, std::unique_ptr<SensingMatrix> &matrix_out,
                                    const std::vector<std::vector<double>> &u_multi,
                                    const std::vector<std::vector<double>> &f_multi,
                                    const std::vector<std::vector<double>> &gamma_precomputed,
                                    const std::unique_ptr<Symmetry> &symmetry, const std::unique_ptr<Fcs> &fcs,
                                    const std::unique_ptr<Constraint> &constraint, const bool sparse) const -> void
{
    size_t i, j;
    long irow;
    const auto natmin = symmetry->get_nat_trueprim();
    const auto natmin3 = 3 * natmin;

    typedef Eigen::Triplet<double, size_t> T;
    std::vector<T> nonzero_entries;

    std::vector<double> bvec_orig(nrows, 0.0);
    std::vector<double> bvec_correction(nrows, 0.0);

#ifdef _OPENMP
#pragma omp parallel private(irow, i, j)
#endif
    {
        size_t idata;
        double **amat_orig_tmp;
        double **amat_mod_tmp = nullptr;
        std::vector<T> nonzero_omp;

        allocate(amat_orig_tmp, natmin3, ncols);

        if (constraint->get_constraint_algebraic()) {
            allocate(amat_mod_tmp, natmin3, ncols_compact);
        }

#ifdef _OPENMP
#pragma omp for
#endif
        for (irow = 0; irow < ncycle; ++irow) {

            // generate r.h.s vector B
            fill_bvec(natmin, irow, symmetry->get_map_trueprim_to_super(), f_multi[irow], bvec_orig);

            // generate l.h.s. matrix A
            fill_amat(maxorder, natmin, ncols, u_multi[irow], gamma_precomputed, symmetry, fcs, amat_orig_tmp);

            // When the force constants are defined in the fractional coordinate,
            // we need to multiply the basis_conversion_matrix to get atomic forces
            // in the Cartesian coordinate.
            if (fcs->get_forceconstant_basis() == "Lattice") {
                apply_basis_converter_amat(natmin3, ncols, amat_orig_tmp, fcs->get_basis_conversion_matrix());
            }

            idata = natmin3 * irow;

            if (constraint->get_constraint_algebraic()) {
                // Project constraints
                project_constraints(maxorder,
                                    natmin,
                                    irow,
                                    fcs,
                                    constraint,
                                    amat_orig_tmp,
                                    amat_mod_tmp,
                                    bvec_correction);

                if (sparse) {
                    for (i = 0; i < natmin3; ++i) {
                        for (j = 0; j < ncols_compact; ++j) {
                            if (std::abs(amat_mod_tmp[i][j]) > eps) {
                                nonzero_omp.emplace_back(idata + i, j, amat_mod_tmp[i][j]);
                            }
                        }
                    }
                } else {
                    for (i = 0; i < natmin3; ++i) {
                        for (j = 0; j < ncols_compact; ++j) {
                            // Transpose here for later use of lapack without transpose
                            matrix_out->amat_dense[natmin3 * ncycle * j + i + idata] = amat_mod_tmp[i][j];
                        }
                    }
                }

            } else {

                if (sparse) {
                    for (i = 0; i < natmin3; ++i) {
                        for (j = 0; j < ncols; ++j) {
                            if (std::abs(amat_orig_tmp[i][j]) > eps) {
                                nonzero_omp.emplace_back(idata + i, j, amat_orig_tmp[i][j]);
                            }
                        }
                    }
                } else {
                    for (i = 0; i < natmin3; ++i) {
                        for (j = 0; j < ncols; ++j) {
                            // Transpose here for later use of lapack without transpose
                            matrix_out->amat_dense[natmin3 * ncycle * j + i + idata] = amat_orig_tmp[i][j];
                        }
                    }
                }
            }
        }
        deallocate(amat_orig_tmp);
        if (amat_mod_tmp) deallocate(amat_mod_tmp);

        if (sparse) {
#pragma omp critical
            {
                for (const auto &it: nonzero_omp) {
                    nonzero_entries.emplace_back(it);
                }
            }
        }
    }
    for (i = 0; i < nrows; ++i) {
        matrix_out->bvec[i] = bvec_orig[i] + bvec_correction[i];
        matrix_out->original_forces[i] = bvec_orig[i];
    }

    if (sparse) {
        matrix_out->amat_sparse.setFromTriplets(nonzero_entries.begin(), nonzero_entries.end());
        matrix_out->amat_sparse.makeCompressed();
    }
}

auto Optimize::get_matrix_elements_normal_equation2(
    const int maxorder, const size_t ncycle, const size_t nrows, const size_t ncols, const size_t ncols_compact,
    std::unique_ptr<SensingMatrix> &matrix_out, const std::vector<std::vector<double>> &u_multi,
    const std::vector<std::vector<double>> &f_multi, const std::vector<std::vector<double>> &gamma_precomputed,
    const std::unique_ptr<Symmetry> &symmetry, const std::unique_ptr<Fcs> &fcs,
    const std::unique_ptr<Constraint> &constraint, const bool sparse) const -> void
{
    typedef Eigen::Triplet<double, size_t> T;

    long irow;

    const auto ndata_fit = u_multi.size() / symmetry->get_ntran();
    const auto natmin = symmetry->get_nat_trueprim();
    const auto natmin3 = 3 * natmin;
    const auto nat3 = u_multi[0].size();

    auto ndata_subset = optcontrol.chunk_size;
    auto nsubset = ndata_fit / ndata_subset;
    if (nsubset * ndata_subset < ndata_fit) {
        ++nsubset;
    }

    // std::cout << "nsubset = " << nsubset << '\n';

    std::vector<double> bvec_orig(nrows, 0.0);
    std::vector<double> bvec_subset;

    if (sparse) {
        matrix_out->amat_sparse.resize(ncols_compact, ncols_compact);
    }

    // std::cout << "sparse = " << sparse << '\n';
    // std::cout << "size of ata_subset = " << ncols_compact << " x " << ncols_compact << '\n';
    // std::cout << std::flush;


    for (size_t isub = 0; isub < nsubset; ++isub) {

        const size_t istart = isub * ndata_subset;
        size_t iend = (isub + 1) * ndata_subset;
        if (iend > ndata_fit) {
            iend = ndata_fit;
        }

        auto nrows_now = (iend - istart) * nat3;

        // std::cout << "istart = " << istart << '\n';
        // std::cout << "iend = " << iend << '\n';

        const long istart_cycle = istart * symmetry->get_ntran();
        const long iend_cycle = iend * symmetry->get_ntran();

        Eigen::MatrixXd amat_subset(nrows_now, ncols_compact);
        Eigen::MatrixXd amat_subset_transpose(ncols_compact, nrows_now);

        bvec_subset.resize(nrows_now, 0.0);
        std::vector<T> nonzero_entries;

        amat_subset.setZero();

#pragma omp parallel private(irow)
        {
            std::vector<int> ind;
            size_t ii, jj;
            size_t idata;
            double **amat_orig_tmp;
            double **amat_mod_tmp;

            std::vector<T> nonzero_omp;

            ind.resize(maxorder + 1, 0);
            allocate(amat_orig_tmp, natmin3, ncols);
            allocate(amat_mod_tmp, natmin3, ncols_compact);

            // std::cout << "OK" << std::flush;
            // std::cout << "istart_cycle = " << istart_cycle << '\n';
            // std::cout << "iend_cycle = " << iend_cycle << '\n';

#pragma omp for
            for (irow = istart_cycle; irow < iend_cycle; ++irow) {
                idata = natmin3 * (irow - istart_cycle);

                // std::cout << "irow = " << irow << '\n';

                // generate r.h.s vector B
                fill_bvec(natmin,
                          irow,
                          symmetry->get_map_trueprim_to_super(),
                          f_multi[irow],
                          matrix_out->original_forces);
                fill_bvec(natmin,
                          irow - istart_cycle,
                          symmetry->get_map_trueprim_to_super(),
                          f_multi[irow],
                          bvec_subset);

                // generate l.h.s. matrix A
                fill_amat(maxorder, natmin, ncols, u_multi[irow], gamma_precomputed, symmetry, fcs, amat_orig_tmp);


                // When the force constants are defined in the fractional coordinate,
                // we need to multiply the basis_conversion_matrix to obtain atomic forces
                // in the Cartesian coordinate.
                if (fcs->get_forceconstant_basis() == "Lattice") {
                    apply_basis_converter_amat(natmin3, ncols, amat_orig_tmp, fcs->get_basis_conversion_matrix());
                }

                if (constraint->get_constraint_algebraic()) {

                    // Project constraints
                    // This operation increases non-zero entries of the originally sparse matrix.
                    // The resulting matrix manipulation At * A will be more expensive.
                    project_constraints(maxorder,
                                        natmin,
                                        irow - istart_cycle,
                                        fcs,
                                        constraint,
                                        amat_orig_tmp,
                                        amat_mod_tmp,
                                        bvec_subset);

                    if (sparse) {
                        for (ii = 0; ii < natmin3; ++ii) {
                            for (jj = 0; jj < ncols_compact; ++jj) {
                                if (std::abs(amat_mod_tmp[ii][jj]) > eps6) {
                                    nonzero_omp.emplace_back(idata + ii, jj, amat_mod_tmp[ii][jj]);
                                }
                            }
                        }
                    } else {
                        for (ii = 0; ii < natmin3; ++ii) {
                            for (jj = 0; jj < ncols_compact; ++jj) {
                                // Transpose here for later use of lapack without transpose
                                amat_subset(idata + ii, jj) = amat_mod_tmp[ii][jj];
                            }
                        }
                    }

                } else {

                    if (sparse) {
                        for (ii = 0; ii < natmin3; ++ii) {
                            for (jj = 0; jj < ncols; ++jj) {
                                if (std::abs(amat_orig_tmp[ii][jj]) > eps) {
                                    nonzero_omp.emplace_back(idata + ii, jj, amat_orig_tmp[ii][jj]);
                                }
                            }
                        }
                    } else {
                        for (ii = 0; ii < natmin3; ++ii) {
                            for (jj = 0; jj < ncols; ++jj) {
                                // Transpose here for later use of lapack without transpose
                                amat_subset(idata + ii, jj) = amat_orig_tmp[ii][jj];
                            }
                        }
                    }
                }
            }

            deallocate(amat_orig_tmp);
            deallocate(amat_mod_tmp);

            if (sparse) {
#pragma omp critical
                {
                    for (const auto &it: nonzero_omp) {
                        nonzero_entries.emplace_back(it);
                    }
                }
                nonzero_omp.clear();
            }
            // std::cout << "nonzero_entries.size() = " << nonzero_entries.size() << '\n';
        }

        if (sparse) {
            SpMat amat_subset_sparse, amat_subset_transpose_sparse, atb_tmp;
            amat_subset_sparse.resize(nrows_now, ncols_compact);
            // amat_subset_transpose_sparse.resize(ncols_compact, nrows_now);

            // std::cout << "Memory allocated for amat_subset_sparse =" << amat_subset_sparse.size() << '\n';

            amat_subset_sparse.setFromTriplets(nonzero_entries.begin(), nonzero_entries.end());
            amat_subset_sparse.makeCompressed();

            // std::cout << "Elements filled" << amat_subset_sparse.nonZeros() << '\n';

            SpMat At = amat_subset_sparse.transpose();

            // matrix_out->amat_sparse.reserve(matrix_out->amat_sparse.size() + AtA.nonZeros());

            // std::cout << "Computed transpose:\n";

            matrix_out->amat_sparse += At * amat_subset_sparse;
            // std::cout << "Computed A^T A\n";
            Eigen::VectorXd bvec_subset2 = Eigen::Map<Eigen::VectorXd>(bvec_subset.data(), bvec_subset.size());
            Eigen::VectorXd atb_tmp2 = At * bvec_subset2;

            for (size_t i = 0; i < ncols_compact; ++i) {
                matrix_out->bvec[i] += atb_tmp2[i];
            }

        } else {
            amat_subset_transpose = amat_subset.transpose();
            Eigen::MatrixXd ata_subset = amat_subset_transpose * amat_subset; // This is memory intensive
            Eigen::VectorXd atb_subset =
                amat_subset_transpose * Eigen::Map<Eigen::VectorXd>(bvec_subset.data(), bvec_subset.size());

            for (size_t i = 0; i < ncols_compact; ++i) {
                for (size_t j = 0; j < ncols_compact; ++j) {
                    matrix_out->amat_dense[i * ncols_compact + j] += ata_subset(i, j);
                }
                matrix_out->bvec[i] += atb_subset(i);
            }
        }
    }
}


auto Optimize::fill_bvec(const size_t natmin, const size_t irow, const std::vector<std::vector<int>> &index_mapping,
                         const std::vector<double> &f_sub, std::vector<double> &bvec) -> void
{
    const auto natmin3 = natmin * 3;
    for (auto i = 0; i < natmin; ++i) {
        for (auto j = 0; j < 3; ++j) {
            bvec[3 * i + j + natmin3 * irow] = f_sub[3 * index_mapping[i][0] + j];
        }
    }
}

auto Optimize::fill_amat(const int maxorder, const size_t natmin, const size_t ncols, const std::vector<double> &u_sub,
                         const std::vector<std::vector<double>> &gamma_precomputed,
                         const std::unique_ptr<Symmetry> &symmetry, const std::unique_ptr<Fcs> &fcs,
                         double **&amat_orig) -> void
{
    const auto natmin3 = natmin * 3;

    for (auto i = 0; i < natmin3; ++i) {
        for (auto j = 0; j < ncols; ++j) {
            amat_orig[i][j] = 0.0;
        }
    }

    size_t iparam = 0;

    double amat_tmp;
    int k;

    for (int order = 0; order < maxorder; ++order) {

        size_t mm = 0;

        for (const auto &iter: fcs->get_nequiv()[order]) {
            for (auto i = 0; i < iter; ++i) {
                amat_tmp = 1.0;
                for (int j = 1; j < order + 2; ++j) {
                    amat_tmp *= u_sub[fcs->get_fc_table()[order][mm].elems[j]];
                }
                k = inprim_index(fcs->get_fc_table()[order][mm].elems[0], symmetry);
                amat_orig[k][iparam] -= gamma_precomputed[order][mm] * amat_tmp;
                ++mm;
            }
            ++iparam;
        }
    }
}

auto Optimize::project_constraints(const int maxorder, const size_t natmin, const size_t irow,
                                   const std::unique_ptr<Fcs> &fcs, const std::unique_ptr<Constraint> &constraint,
                                   double **amat_orig, double **&amat_mod, std::vector<double> &bvec_mod) -> void
{
    // Convert the full matrix and vector into a smaller irreducible form
    // by using constraint information.

    size_t ishift = 0;
    size_t iparam = 0;
    size_t inew, iold;
    const auto natmin3 = 3 * natmin;
    const auto idata = natmin3 * irow;

    for (int order = 0; order < maxorder; ++order) {

        for (const auto fix: constraint->get_const_fix(order)) {

            for (size_t j = 0; j < natmin3; ++j) {
                bvec_mod[j + idata] -= fix.val_to_fix * amat_orig[j][ishift + fix.p_index_target];
            }
        }

        for (const auto &it: constraint->get_index_bimap(order)) {
            inew = it.left + iparam;
            iold = it.right + ishift;

            for (size_t j = 0; j < natmin3; ++j) {
                amat_mod[j][inew] = amat_orig[j][iold];
            }
        }

        for (size_t i = 0; i < constraint->get_const_relate(order).size(); ++i) {

            iold = constraint->get_const_relate(order)[i].p_index_target + ishift;

            for (size_t j = 0; j < constraint->get_const_relate(order)[i].alpha.size(); ++j) {

                // This part can issue an error when the constraint matrix is deviate from rref.
                //                        const auto right_value =  constraint->get_const_relate(order)[i].p_index_orig[j];
                //                        std::cout << "right = " << right_value << '\n'<< std::flush;
                //                        if (constraint->get_index_bimap(order).right.find(right_value) == constraint->get_index_bimap(order).right.end()) {
                //                            std::cout << "The key not found \n" << '\n' << std::flush;
                //                            std::exit(1);
                //                        }
                inew = constraint->get_index_bimap(order).right.at(
                           constraint->get_const_relate(order)[i].p_index_orig[j]) +
                       iparam;

                for (size_t k = 0; k < natmin3; ++k) {
                    amat_mod[k][inew] -= amat_orig[k][iold] * constraint->get_const_relate(order)[i].alpha[j];
                }
            }
        }

        ishift += fcs->get_nequiv()[order].size();
        iparam += constraint->get_index_bimap(order).size();
    }
}

auto Optimize::recover_original_forceconstants(const int maxorder, const std::vector<double> &param_in,
                                               std::vector<double> &param_out, const std::vector<size_t> *nequiv,
                                               const std::unique_ptr<Constraint> &constraint) const -> void
{
    // Expand the given force constants into the larger sets
    // by using the constraint matrix.

    size_t i, j, k;
    size_t ishift = 0;
    size_t iparam = 0;
    double tmp;
    size_t inew, iold;

    size_t nparams = 0;

    for (i = 0; i < maxorder; ++i)
        nparams += nequiv[i].size();

    param_out.resize(nparams, 0.0);

    for (i = 0; i < maxorder; ++i) {
        for (j = 0; j < constraint->get_const_fix(i).size(); ++j) {
            param_out[constraint->get_const_fix(i)[j].p_index_target + ishift] =
                constraint->get_const_fix(i)[j].val_to_fix;
        }

        for (const auto &it: constraint->get_index_bimap(i)) {
            inew = it.left + iparam;
            iold = it.right + ishift;

            param_out[iold] = param_in[inew];
        }

        for (j = 0; j < constraint->get_const_relate(i).size(); ++j) {
            tmp = 0.0;

            for (k = 0; k < constraint->get_const_relate(i)[j].alpha.size(); ++k) {
                tmp += constraint->get_const_relate(i)[j].alpha[k] *
                       param_out[constraint->get_const_relate(i)[j].p_index_orig[k] + ishift];
            }
            param_out[constraint->get_const_relate(i)[j].p_index_target + ishift] = -tmp;
        }

        ishift += nequiv[i].size();
        iparam += constraint->get_index_bimap(i).size();
    }
}


auto Optimize::data_multiplier(const std::vector<std::vector<double>> &data_in,
                               std::vector<std::vector<double>> &data_out,
                               const std::unique_ptr<Symmetry> &symmetry) const -> void
{
    const auto nat = symmetry->get_nat_trueprim() * symmetry->get_ntran();
    const auto ndata_used = data_in.size();
    const auto ntran = symmetry->get_ntran();

    data_out.resize(ntran * ndata_used, std::vector<double>(3 * nat));

    auto idata = 0;
    for (auto i = 0; i < ndata_used; ++i) {

        for (auto itran = 0; itran < symmetry->get_ntran(); ++itran) {
            for (auto j = 0; j < nat; ++j) {
                const auto n_mapped = symmetry->get_map_sym()[j][symmetry->get_symnum_tran()[itran]];
                for (auto k = 0; k < 3; ++k) {
                    data_out[idata][3 * n_mapped + k] = data_in[i][3 * j + k];
                }
            }
            ++idata;
        }
    }
}

auto Optimize::inprim_index(const int n, const std::unique_ptr<Symmetry> &symmetry) -> int
{
    auto in = -1;
    const auto atmn = n / 3;
    const auto crdn = n % 3;

    for (size_t i = 0; i < symmetry->get_nat_trueprim(); ++i) {
        if (symmetry->get_map_trueprim_to_super()[i][0] == atmn) {
            in = 3 * i + crdn;
            break;
        }
    }
    return in;
}

auto Optimize::gamma(const int n, const int *arr) const -> double
{
    std::vector<int> arr_tmp(n);
    std::vector<int> nsame(n);
    int i;

    for (i = 0; i < n; ++i) {
        arr_tmp[i] = arr[i];
        nsame[i] = 0;
    }

    const auto ind_front = arr[0];
    auto nsame_to_front = 1;

    insort(n, arr_tmp.data());

    auto nuniq = 1;
    auto iuniq = 0;

    nsame[0] = 1;

    for (i = 1; i < n; ++i) {
        if (arr_tmp[i] == arr_tmp[i - 1]) {
            ++nsame[iuniq];
        } else {
            ++nsame[++iuniq];
            ++nuniq;
        }

        if (arr[i] == ind_front) ++nsame_to_front;
    }

    auto denom = 1;

    for (i = 0; i < nuniq; ++i) {
        denom *= factorial(nsame[i]);
    }

    return static_cast<double>(nsame_to_front) / static_cast<double>(denom);
}


auto Optimize::get_params() const -> double *
{
    return params;
}

auto Optimize::factorial(const int n) const -> int
{
    if (n == 1 || n == 0) {
        return 1;
    }
    return n * factorial(n - 1);
}


auto Optimize::run_eigen_sparse_solver(const SpMat &sp_mat, const Eigen::VectorXd &sp_bvec,
                                       std::vector<double> &param_out, const double fnorm, const int maxorder,
                                       const std::unique_ptr<Fcs> &fcs, const std::unique_ptr<Constraint> &constraint,
                                       const std::string &solver_type, const int verbosity) const -> int
{
    if (verbosity > 0) {
        std::cout << "  Solve least-squares problem by Eigen " + solver_type + ".\n";
    }

    Eigen::VectorXd x;

    auto info_eigen = least_squares_eigen_sparse_solver(sp_mat,
                                                        sp_bvec,
                                                        x,
                                                        solver_type,
                                                        optcontrol.tolerance_iteration,
                                                        optcontrol.maxnum_iteration);

    auto res = sp_bvec - sp_mat * x;
    const auto res2norm = res.squaredNorm();
    const auto nparams = x.size();
    std::vector<double> param_irred(nparams);

    for (auto i = 0; i < nparams; ++i) {
        param_irred[i] = x(i);
    }

    // Recover reducible set of force constants

    if (constraint->get_constraint_algebraic()) {
        recover_original_forceconstants(maxorder, param_irred, param_out, fcs->get_nequiv(), constraint);
    } else {
        param_out.resize(nparams, 0.0);
        for (size_t i = 0; i < nparams; ++i) {
            param_out[i] = param_irred[i];
        }
    }

    if (verbosity > 0) {
        std::cout << "  Residual sum of squares for the solution: " << sqrt(res2norm) << '\n';
        std::cout << "  Fitting error (%) : " << sqrt(res2norm / (fnorm * fnorm)) * 100.0 << '\n';
    }

    return info_eigen;
}


auto Optimize::set_optimizer_control(const OptimizerControl &optcontrol_in) -> void
{
    // Check the validity of the options before copying it.

    if (optcontrol_in.cross_validation < -1) {
        exit("set_optimizer_control", "cross_validation must be -1, 0, or larger");
    }
    if (optcontrol_in.linear_model == 2) {
        if (optcontrol_in.l1_ratio <= eps || optcontrol_in.l1_ratio > 1.0) {
            exit("set_optimizer_control", "L1_RATIO must be 0 < L1_RATIO <= 1.");
        }

        if (optcontrol_in.cross_validation >= 1 || optcontrol_in.cross_validation == -1) {
            if (optcontrol_in.l1_alpha_max > 0) {
                if (optcontrol_in.l1_alpha_min >= optcontrol_in.l1_alpha_max) {
                    exit("set_optimizer_control", "L1_ALPHA_MIN must be smaller than L1_ALPHA_MAX.");
                }
            }
        }
    }

    optcontrol = optcontrol_in;
}

auto Optimize::get_optimizer_control() const -> OptimizerControl
{
    return optcontrol;
}

auto Optimize::get_cv_l1_alpha() const -> double
{
    return cv_l1_alpha;
}

auto Optimize::coordinate_descent(const int M, const int N, const double alpha, const int warm_start,
                                  Eigen::VectorXd &x, const Eigen::MatrixXd &A, const Eigen::VectorXd &b,
                                  const Eigen::VectorXd &grad0, bool *has_prod, Eigen::MatrixXd &Prod,
                                  Eigen::VectorXd &grad, const double fnorm, const Eigen::VectorXd &scale_beta,
                                  const int verbosity) const -> void
{
    int i, j;
    double diff{0.0};
    Eigen::VectorXd beta(N), delta(N);
    Eigen::VectorXd res(N);
    bool do_print_log;

    if (warm_start) {
        for (i = 0; i < N; ++i)
            beta(i) = x(i);
    } else {
        for (i = 0; i < N; ++i)
            beta(i) = 0.0;
        grad = grad0;
    }

    if (verbosity > 1) {
        std::cout << "-----------------------------------------------------------------\n";
        std::cout << "  L1_ALPHA = " << std::setw(15) << alpha << '\n';
    }

    const auto Minv = 1.0 / static_cast<double>(M);
    const auto alphlambda = alpha * optcontrol.l1_ratio;

    auto iloop = 0;

    if (optcontrol.standardize) {
        while (iloop < optcontrol.maxnum_iteration) {
            do_print_log = !((iloop + 1) % optcontrol.output_frequency) && (verbosity > 1);

            if (do_print_log) {
                std::cout << "   Coordinate Descent : " << std::setw(5) << iloop + 1 << '\n';
            }
            delta = beta;
            for (i = 0; i < N; ++i) {
                beta(i) = shrink(Minv * grad(i) + beta(i), alphlambda);
                delta(i) -= beta(i);
                if (std::abs(delta(i)) > 0.0) {
                    if (!has_prod[i]) {
#pragma omp parallel for
                        for (j = 0; j < N; ++j) {
                            Prod(j, i) = A.col(j).dot(A.col(i));
                        }
                        has_prod[i] = true;
                    }
                    grad = grad + Prod.col(i) * delta(i);
                }
            }
            ++iloop;
            diff = 0.0;
#pragma omp parallel for reduction(+ : diff)
            for (i = 0; i < N; ++i) {
                diff += delta(i) * delta(i);
            }
            diff = std::sqrt(diff / static_cast<double>(N));
            //diff = std::sqrt(delta.dot(delta) / static_cast<double>(N));

            if (diff < optcontrol.tolerance_iteration) break;

            if (do_print_log) {
                std::cout << "    1: ||u_{k}-u_{k-1}||_2     = " << std::setw(15) << diff << std::setw(15)
                          << diff * std::sqrt(static_cast<double>(N) / beta.dot(beta)) << '\n';
                auto tmp = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : tmp)
#endif
                for (i = 0; i < N; ++i) {
                    tmp += std::abs(beta(i));
                }
                std::cout << "    2: ||u_{k}||_1             = " << std::setw(15) << tmp << '\n';
                res = A * beta - b;
                tmp = res.dot(res);
                std::cout << "    3: ||Au_{k}-f||_2          = " << std::setw(15) << std::sqrt(tmp) << std::setw(15)
                          << std::sqrt(tmp / (fnorm * fnorm)) << '\n';
                std::cout << '\n';
            }
        }
    } else {
        // Non-standardized version. Needs additional operations

        Eigen::VectorXd inv_scale_beta(N);

        for (i = 0; i < N; ++i)
            inv_scale_beta(i) = 1.0 / scale_beta(i);

        while (iloop < optcontrol.maxnum_iteration) {
            do_print_log = !((iloop + 1) % optcontrol.output_frequency) && (verbosity > 1);

            if (do_print_log) {
                std::cout << "   Coordinate Descent : " << std::setw(5) << iloop + 1 << '\n';
            }
            delta = beta;
            for (i = 0; i < N; ++i) {
                beta(i) = shrink(Minv * grad(i) + beta(i) * inv_scale_beta(i), alphlambda) * scale_beta(i);
                delta(i) -= beta(i);
                if (std::abs(delta(i)) > 0.0) {
                    if (!has_prod[i]) {
#pragma omp parallel for
                        for (j = 0; j < N; ++j) {
                            Prod(j, i) = A.col(j).dot(A.col(i));
                        }
                        has_prod[i] = true;
                    }
                    grad = grad + Prod.col(i) * delta(i);
                }
            }
            ++iloop;
            diff = 0.0;
#pragma omp parallel for reduction(+ : diff)
            for (i = 0; i < N; ++i) {
                diff += delta(i) * delta(i);
            }
            diff = std::sqrt(diff / static_cast<double>(N));
            //diff = std::sqrt(delta.dot(delta) / static_cast<double>(N));

            if (diff < optcontrol.tolerance_iteration) break;

            if (do_print_log) {
                std::cout << "    1: ||u_{k}-u_{k-1}||_2     = " << std::setw(15) << diff << std::setw(15)
                          << diff * std::sqrt(static_cast<double>(N) / beta.dot(beta)) << '\n';
                auto tmp = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : tmp)
#endif
                for (i = 0; i < N; ++i) {
                    tmp += std::abs(beta(i));
                }
                std::cout << "    2: ||u_{k}||_1             = " << std::setw(15) << tmp << '\n';
                res = A * beta - b;
                tmp = res.dot(res);
                std::cout << "    3: ||Au_{k}-f||_2          = " << std::setw(15) << std::sqrt(tmp) << std::setw(15)
                          << std::sqrt(tmp / (fnorm * fnorm)) << '\n';
                std::cout << '\n';
            }
        }
    }

    if (verbosity > 1) {
        if (iloop >= optcontrol.maxnum_iteration) {
            std::cout << "WARNING: Convergence NOT achieved within " << optcontrol.maxnum_iteration
                      << " coordinate descent iterations.\n";
        } else {
            std::cout << "  Convergence achieved in " << iloop << " iterations.\n";
        }
        const auto param2norm = beta.dot(beta);
        if (std::abs(param2norm) < eps) {
            std::cout << "    1': ||u_{k}-u_{k-1}||_2     = " << std::setw(15) << 0.0 << std::setw(15) << 0.0 << '\n';
        } else {
            std::cout << "    1': ||u_{k}-u_{k-1}||_2     = " << std::setw(15) << diff << std::setw(15)
                      << diff * std::sqrt(static_cast<double>(N) / param2norm) << '\n';
        }
        double tmp = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : tmp)
#endif
        for (i = 0; i < N; ++i) {
            tmp += std::abs(beta(i));
        }
        std::cout << "    2': ||u_{k}||_1             = " << std::setw(15) << tmp << '\n';
        res = A * beta - b;
        tmp = res.dot(res);
        std::cout << "    3': ||Au_{k}-f||_2          = " << std::setw(15) << std::sqrt(tmp) << std::setw(15)
                  << std::sqrt(tmp / (fnorm * fnorm)) << '\n';
        std::cout << '\n';
    }

    for (i = 0; i < N; ++i)
        x[i] = beta(i);
}
