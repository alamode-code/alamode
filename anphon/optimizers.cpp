#include "optimizers.h"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <iomanip>
#include "timer.h"

using namespace PHON_NS;


Newton_Optimizer::Newton_Optimizer(double mixbeta) : mixbeta(mixbeta)
{}

void Newton_Optimizer::update_state(const int dim, const std::vector<double> &grad_vec, std::vector<double> &state_vec,
                                    const std::vector<std::vector<double>> &hessian, std::vector<double> &delta)
{

    Eigen::MatrixXd hessian_matrix(dim, dim);
    Eigen::VectorXd grad(dim);
    Eigen::VectorXd delta_tmp(dim);

    for (int i = 0; i < dim; ++i) {
        grad(i) = grad_vec[i];
        for (int j = 0; j < dim; ++j) {
            hessian_matrix(i, j) = hessian[i][j];
        }
    }

    delta_tmp = hessian_matrix.colPivHouseholderQr().solve(grad);

    // std::cout << "Hessian matrix:\n" << hessian_matrix << std::endl;
    // std::cout << "Eigenvalue of Hessian matrix: \n" << hessian_matrix.eigenvalues().real() << std::endl;
    // std::cout << "gradient:\n" << grad << std::endl;
    // std::cout << "-H^{-1} * grad:\n" << delta_tmp << std::endl;
    // std::cout << "H^{-1} \n" << hessian_matrix.inverse() << std::endl;


    for (int i = 0; i < dim; ++i) {
        delta[i] = -1.0 * mixbeta * delta_tmp(i);
        state_vec[i] = state_vec[i] + delta[i];
    }
}


SteepestDescent_Optimizer::SteepestDescent_Optimizer(double alpha) : alpha(alpha)
{}

void SteepestDescent_Optimizer::update_state(const int dim, const std::vector<double> &grad_vec,
                                             std::vector<double> &state_vec,
                                             const std::vector<std::vector<double>> &hessian,
                                             std::vector<double> &delta)
{
    // Steepest descent: take a fixed step along the negative gradient.
    // The Hessian is not used.
    (void)hessian;
    for (int i = 0; i < dim; ++i) {
        delta[i] = -alpha * grad_vec[i];
        state_vec[i] = state_vec[i] + delta[i];
    }
}


CellCoord_Newton_Optimizer::CellCoord_Newton_Optimizer(double mixbeta_cell, double mixbeta_coord)
{
    this->mixbeta_cell = mixbeta_cell;
    this->mixbeta_coord = mixbeta_coord;

    cell_optimizer = std::make_unique<Newton_Optimizer>(mixbeta_cell);
    coord_optimizer = std::make_unique<Newton_Optimizer>(mixbeta_coord);
}

void CellCoord_Newton_Optimizer::update_state(const int dim, const std::vector<double> &grad_vec,
                                              std::vector<double> &state_vec,
                                              const std::vector<std::vector<double>> &hessian,
                                              std::vector<double> &delta)
{

    std::vector<double> grad_cell(6);
    std::vector<double> state_cell(6);
    std::vector<double> delta_cell(6);
    std::vector<std::vector<double>> hessian_cell(6, std::vector<double>(6));

    std::vector<double> grad_coord(dim - 6);
    std::vector<double> state_coord(dim - 6);
    std::vector<double> delta_coord(dim - 6);
    std::vector<std::vector<double>> hessian_coord(dim - 6, std::vector<double>(dim - 6));


    for (int i = 0; i < dim - 6; ++i) {
        grad_coord[i] = grad_vec[i];
        state_coord[i] = state_vec[i];
        for (int j = 0; j < dim - 6; ++j) {
            hessian_coord[i][j] = hessian[i][j];
        }
    }

    for (int i = 0; i < 6; ++i) {
        grad_cell[i] = grad_vec[i + dim - 6];
        state_cell[i] = state_vec[i + dim - 6];
        for (int j = 0; j < 6; ++j) {
            hessian_cell[i][j] = hessian[i + dim - 6][j + dim - 6];
        }
    }

    coord_optimizer->update_state(dim - 6, grad_coord, state_coord, hessian_coord, delta_coord);
    cell_optimizer->update_state(6, grad_cell, state_cell, hessian_cell, delta_cell);

    for (int i = 0; i < dim - 6; ++i) {
        delta[i] = delta_coord[i];
        state_vec[i] = state_coord[i];
    }

    for (int i = 0; i < 6; ++i) {
        delta[i + dim - 6] = delta_cell[i];
        state_vec[i + dim - 6] = state_cell[i];
    }
}

void FarkasIII_Optimizer::update_state(const int dim, const std::vector<double> &grad_vec,
                                       std::vector<double> &state_vec, const std::vector<std::vector<double>> &hessian,
                                       std::vector<double> &delta)
{
    Eigen::VectorXd state_tmp(dim);
    Eigen::VectorXd grad_tmp(dim);
    Eigen::VectorXd updated_state(dim);

    for (int i = 0; i < dim; ++i) {
        state_tmp(i) = state_vec[i];
        grad_tmp(i) = grad_vec[i];
    }

    if (initialize_flag == 1) {
        set_inverse_Hessian(dim, hessian);
        initialize_history();
        initialize_flag = 0;
    }

    updated_state = gdiis_control ? this->update_controlled(state_tmp, grad_tmp) : this->update(state_tmp, grad_tmp);

    // write answer
    for (int i = 0; i < dim; ++i) {
        delta[i] = updated_state(i) - state_vec[i];
        state_vec[i] = updated_state[i];
    }
}

void FarkasIII_Optimizer::set_inverse_Hessian(const int dim, const std::vector<std::vector<double>> &hessian)
{
    Eigen::MatrixXd H_tmp(dim, dim);
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            H_tmp(i, j) = hessian[i][j];
        }
    }
    H = H_tmp.inverse();

    // Capture the smallest curvature of the initial (ADD_HESS_DIAG-regularized) Hessian. The
    // controlled-GDIIS RFO level-shift uses it as the minimum-curvature floor each step.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H_tmp);
    curvature_floor = es.eigenvalues().minCoeff();
    if (!(curvature_floor > 0.0)) curvature_floor = eps8;
    // std::cout << "Inverse Hessian matrix initialized." << std::endl;
    // std::cout << H << std::endl;
}

void FarkasIII_Optimizer::initialize_history()
{
    points.clear();
    residuals.clear();
    gradients.clear();
}

double FarkasIII_Optimizer::angle_threshold(const int n_vectors)
{
    // Farkas-Schlegel cut-offs for cos(theta); 0 for 10 or more vectors.
    switch (n_vectors) {
    case 2: return 0.97;
    case 3: return 0.84;
    case 4: return 0.71;
    case 5: return 0.67;
    case 6: return 0.62;
    case 7: return 0.56;
    case 8: return 0.49;
    case 9: return 0.41;
    default: return 0.0;
    }
}

Eigen::MatrixXd FarkasIII_Optimizer::effective_inverse_Hessian() const
{
    // H stores the inverse Hessian. Build the RFO level-shifted effective inverse Hessian, i.e.
    // (H_direct + lambda*I)^{-1}, with lambda >= 0 chosen so the smallest direct-Hessian
    // curvature becomes at least curvature_floor. This bounds the step along near-zero-curvature
    // (soft) modes and turns negative-curvature directions into controlled descent steps
    // (Farkas-Schlegel eqn 7, H_eff = H_direct + lambda*I). H need NOT be positive definite here
    // (a strongly soft initial Hessian can be indefinite).
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
    const Eigen::VectorXd &nu = es.eigenvalues(); // eigenvalues of the inverse Hessian
    const Eigen::MatrixXd &U = es.eigenvectors();
    const int n = static_cast<int>(nu.size());

    // direct-Hessian curvatures mu_i = 1/nu_i (sign preserved), and their minimum.
    Eigen::VectorXd mu(n);
    double mu_min = 0.0;
    for (int i = 0; i < n; ++i) {
        const double nui = nu(i);
        if (nui > eps15 || nui < -eps15) {
            mu(i) = 1.0 / nui; // keeps the sign: negative nu -> negative (soft/unstable) curvature
        } else {
            // |nu_i| <= eps15: a machine-noise-scale inverse eigenvalue. Treat it as a very stiff
            // positive mode (its direct curvature is enormous either way); a near-zero negative
            // nu_i thus has its sign dropped, which is harmless at this scale.
            mu(i) = 1.0 / eps15;
        }
        if (i == 0 || mu(i) < mu_min) mu_min = mu(i);
    }

    // uniform level shift so that min(mu_i) + lambda >= curvature_floor (> 0)
    const double lambda = (mu_min < curvature_floor) ? (curvature_floor - mu_min) : 0.0;

    Eigen::VectorXd nu_eff(n);
    for (int i = 0; i < n; ++i) {
        double denom = mu(i) + lambda;
        if (denom < eps15) denom = eps15; // guard against round-off at the floor
        nu_eff(i) = 1.0 / denom;          // shifted inverse curvature (always positive)
    }
    return U * nu_eff.asDiagonal() * U.transpose();
}

Eigen::VectorXd FarkasIII_Optimizer::update_controlled(const Eigen::VectorXd &point,
                                                       const Eigen::VectorXd &gradient)
{
    // --- history window ---
    if (static_cast<int>(points.size()) == max_vectors) {
        points.erase(points.begin());
        gradients.erase(gradients.begin());
    }
    points.push_back(point);
    gradients.push_back(gradient);
    const int size = static_cast<int>(points.size());

    // --- BFGS update of the inverse Hessian (same as regular GDIIS) ---
    if (size > 1) {
        const int dim = static_cast<int>(gradient.size());
        Eigen::VectorXd s = point - point_old;
        Eigen::VectorXd y = gradient - gradient_old;
        const double ys = y.dot(s);
        if (ys > 1e-12) {
            Eigen::MatrixXd I = Eigen::MatrixXd::Identity(dim, dim);
            Eigen::MatrixXd A = I - y * s.transpose() / ys;
            H = A.transpose() * H * A + s * s.transpose() / ys;
        }
    }
    gradient_old = gradient;
    point_old = point;

    // --- RFO level-shifted effective inverse Hessian, used for both the reference step and the
    //     error vectors (Farkas-Schlegel eqns 3, 7) ---
    const Eigen::MatrixXd H_eff = effective_inverse_Hessian();

    const Eigen::VectorXd diff_REF = -H_eff * gradient; // reference (quasi-Newton) step
    const Eigen::VectorXd point_BFGS = point + diff_REF;
    const double norm_REF = diff_REF.norm();

    if (size == 1) return point_BFGS;

    // error vectors e_i = -H_eff * g_i, recomputed with the current effective Hessian
    std::vector<Eigen::VectorXd> errvec(size);
    for (int i = 0; i < size; ++i) errvec[i] = -H_eff * gradients[i];

    // --- adaptive subspace: grow from the most recent point, keep the last acceptable step ---
    Eigen::VectorXd best_step = point_BFGS; // fallback: reference step (eqn 7)
    int discard_count = 0;                  // number of oldest points to permanently discard

    for (int m = 2; m <= size; ++m) {
        const int i0 = size - m; // use the m most recent points: indices [i0, size)

        // (d, part 1) rescale error vectors so the smallest has unit length (eqn 9)
        double emin = errvec[i0].norm();
        for (int i = i0 + 1; i < size; ++i) emin = std::min(emin, errvec[i].norm());
        const double escale = (emin > eps15) ? 1.0 / emin : 1.0;

        Eigen::MatrixXd B(m + 1, m + 1);
        B.setZero();
        for (int a = 0; a < m; ++a) {
            for (int b = 0; b < m; ++b) {
                B(a, b) = (escale * escale) * errvec[i0 + a].dot(errvec[i0 + b]);
            }
        }
        for (int a = 0; a < m; ++a) B(a, m) = B(m, a) = 1.0;
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(m + 1);
        rhs(m) = 1.0;
        const Eigen::VectorXd coeffs = B.colPivHouseholderQr().solve(rhs);

        Eigen::VectorXd result_point = Eigen::VectorXd::Zero(point.size());
        Eigen::VectorXd result_resid = Eigen::VectorXd::Zero(point.size());
        for (int a = 0; a < m; ++a) {
            result_point += coeffs(a) * points[i0 + a];
            result_resid += coeffs(a) * errvec[i0 + a];
        }
        const Eigen::VectorXd cand = result_point + result_resid;
        const Eigen::VectorXd diff_DIIS = cand - point;
        const double norm_DIIS = diff_DIIS.norm();

        // (a) angle criterion (eqn 8)
        const double cos_angle = diff_DIIS.dot(diff_REF) / (norm_DIIS * norm_REF);
        if (cos_angle < 0.0) {
            // angle > 90 degrees: permanently discard the offending point (oldest in the
            // subset) and all earlier points, then stop growing the subspace.
            discard_count = i0 + 1;
            break;
        }
        bool ok = (cos_angle >= angle_threshold(m));

        // (b) step-length criterion: GDIIS step <= 10x the reference step
        if (norm_DIIS > 10.0 * norm_REF) ok = false;

        // (c) extrapolation criterion: sum of positive coefficients <= 15
        double sum_pos = 0.0;
        for (int a = 0; a < m; ++a) {
            if (coeffs(a) > 0.0) sum_pos += coeffs(a);
        }
        if (sum_pos > 15.0) ok = false;

        // (d, part 2) numerical-stability criterion: reject a near-singular DIIS matrix
        const double r2 = (escale * escale) * result_resid.squaredNorm();
        if (!(coeffs.head(m).norm() <= 1.0e8 * r2)) ok = false;

        if (ok) {
            best_step = cand; // last acceptable GDIIS step
        } else {
            break; // adding this point made the step unacceptable; keep the last acceptable one
        }
    }

    // --- permanent discard of the offending point and all earlier ones ---
    if (discard_count > 0) {
        points.erase(points.begin(), points.begin() + discard_count);
        gradients.erase(gradients.begin(), gradients.begin() + discard_count);
    }

    return best_step;
}
