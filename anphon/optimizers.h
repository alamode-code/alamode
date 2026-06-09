/*
 relaxation.h

 Copyright (c) 2025 Takumi Chida, Ryota Masuki, Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <Eigen/Core>
#include <fftw3.h>
#include <iomanip>
#include "constants.h"
#include "dynamical.h"
#include "error.h"
#include "gruneisen.h"
#include "mathfunctions.h"
#include "mpi_common.h"
#include "parsephon.h"
#include "scph.h"
#include "system.h"
#include "timer.h"

using namespace PHON_NS;

class Optimizer
{
public:
    Optimizer() = default;

    virtual ~Optimizer() = default;

    virtual void update_state(const int dim, const std::vector<double> &grad_vec, std::vector<double> &state_vec,
                              const std::vector<std::vector<double>> &hessian, std::vector<double> &delta) {};

    int initialize_flag = 0; // flag to run initialization
};

class Newton_Optimizer: public Optimizer
{
public:
    double mixbeta = 1.0;

    Newton_Optimizer() = default;

    ~Newton_Optimizer() = default;

    explicit Newton_Optimizer(double mixbeta);

    void update_state(const int dim, const std::vector<double> &grad_vec, std::vector<double> &state_vec,
                      const std::vector<std::vector<double>> &hessian, std::vector<double> &delta);
};

class SteepestDescent_Optimizer: public Optimizer
{
public:
    double alpha = 1.0;

    SteepestDescent_Optimizer() = default;

    ~SteepestDescent_Optimizer() = default;

    explicit SteepestDescent_Optimizer(double alpha);

    void update_state(const int dim, const std::vector<double> &grad_vec, std::vector<double> &state_vec,
                      const std::vector<std::vector<double>> &hessian, std::vector<double> &delta);
};

class CellCoord_Newton_Optimizer: public Optimizer
{
public:
    double mixbeta_cell = 1.0;
    double mixbeta_coord = 1.0;
    std::unique_ptr<Newton_Optimizer> cell_optimizer;
    std::unique_ptr<Newton_Optimizer> coord_optimizer;

    CellCoord_Newton_Optimizer() = default;

    ~CellCoord_Newton_Optimizer() = default;

    CellCoord_Newton_Optimizer(double mixbeta_cell, double mixbeta_coord);

    void update_state(const int dim, const std::vector<double> &grad_vec, std::vector<double> &state_vec,
                      const std::vector<std::vector<double>> &hessian, std::vector<double> &delta);
};

class FarkasIII_Optimizer: public Optimizer
{
public:
    FarkasIII_Optimizer(int max_vectors, const Eigen::MatrixXd &H_ini) : max_vectors(max_vectors), H(H_ini)
    {
        gradient_old = Eigen::VectorXd::Zero(H_ini.size());
        threshold_angle = 0.0; // We set angle threshold to be zero when we store more than 9 vectors
        if (max_vectors == 2) threshold_angle = 0.97;
        if (max_vectors == 3) threshold_angle = 0.84;
        if (max_vectors == 4) threshold_angle = 0.71;
        if (max_vectors == 5) threshold_angle = 0.67;
        if (max_vectors == 6) threshold_angle = 0.62;
        if (max_vectors == 7) threshold_angle = 0.56;
        if (max_vectors == 8) threshold_angle = 0.49;
        if (max_vectors == 9) threshold_angle = 0.41;
    }

    void update_state(const int dim, const std::vector<double> &grad_vec, std::vector<double> &state_vec,
                      const std::vector<std::vector<double>> &hessian, std::vector<double> &delta);


    Eigen::VectorXd update(const Eigen::VectorXd &point, const Eigen::VectorXd &gradient)
    {
        if (points.size() == max_vectors) {
            points.erase(points.begin());
            residuals.erase(residuals.begin());
        }
        points.push_back(point);
        int const size = points.size();
        if (size > 1) {
            int const dim = gradient.size();
            Eigen::VectorXd s = point - point_old;
            Eigen::VectorXd y = gradient - gradient_old;
            double ys = y.dot(s);

            if (ys > 1e-12) {
                // Update when the condition is satisfied
                Eigen::MatrixXd I = Eigen::MatrixXd::Identity(dim, dim);
                Eigen::MatrixXd A = I - y * s.transpose() / ys;
                H = A.transpose() * H * A + s * s.transpose() / ys;
            }
        }
        threshold_angle = 0.0; // We set angle threshold to be zero when we store more than 9 vectors
        if (size == 2) threshold_angle = 0.97;
        if (size == 3) threshold_angle = 0.84;
        if (size == 4) threshold_angle = 0.71;
        if (size == 5) threshold_angle = 0.67;
        if (size == 6) threshold_angle = 0.62;
        if (size == 7) threshold_angle = 0.56;
        if (size == 8) threshold_angle = 0.49;
        if (size == 9) threshold_angle = 0.41;

        gradient_old = gradient;
        point_old = point;

        Eigen::VectorXd diff_GRAD = -0.1 * gradient; // 0.1 is used for the gradient method (it is not used in BFGS)
        Eigen::VectorXd point_GRAD = point + diff_GRAD;
        // residuals.push_back(diff_GRAD);

        Eigen::VectorXd diff_BFGS = -H * gradient;
        residuals.push_back(diff_BFGS);
        Eigen::VectorXd point_BFGS = point + diff_BFGS;

        if (size == 1) return point_BFGS;

        Eigen::MatrixXd B(size + 1, size + 1);
        B.setZero();
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                B(i, j) = residuals[i].dot(residuals[j]);
            }
        }
        for (int i = 0; i < size; ++i) {
            B(i, size) = B(size, i) = 1.0;
        }
        B(size, size) = 0.0;
        Eigen::VectorXd rhs(size + 1);
        rhs.setZero();
        rhs(size) = 1.0;
        Eigen::VectorXd coeffs = B.colPivHouseholderQr().solve(rhs);
        Eigen::VectorXd result_point = Eigen::VectorXd::Zero(points[0].size());
        Eigen::VectorXd result_residual = Eigen::VectorXd::Zero(residuals[0].size());
        for (int i = 0; i < size; ++i) {
            result_point += coeffs(i) * points[i];
            result_residual += coeffs(i) * residuals[i];
        }
        Eigen::VectorXd point_DIIS = result_point + result_residual;
        Eigen::VectorXd diff_REF = point_BFGS - point;
        Eigen::VectorXd diff_DIIS = point_DIIS - point;
        const double norm_DIIS = diff_DIIS.norm();
        const double norm_REF = diff_REF.norm();
        // Fall back to the plain BFGS step when the DIIS direction deviates too much from it.
        // If either step has (near-)zero length the cosine is undefined (0/0 -> NaN); fall
        // back to BFGS in that case too, so a spurious near-zero DIIS extrapolation at a
        // non-stationary point cannot be accepted and reported as a (false) converged step.
        if (norm_DIIS < eps15 || norm_REF < eps15) {
            point_DIIS = point_BFGS;
        } else {
            const double cos_angle = diff_DIIS.dot(diff_REF) / (norm_DIIS * norm_REF);
            if (!(cos_angle >= threshold_angle)) { // the negated form also catches NaN
                point_DIIS = point_BFGS;
            }
        }

        // return point_GRAD; // gradient method
        // return point_BFGS; // BFGS method
        return point_DIIS; //
    }


    void set_inverse_Hessian(const int dim, const std::vector<std::vector<double>> &hessian);

    void initialize_history();

private:
    int max_vectors;
    Eigen::VectorXd point_old;
    Eigen::VectorXd gradient_old;
    double threshold_angle;
    Eigen::MatrixXd H;
    std::vector<Eigen::VectorXd> points;
    std::vector<Eigen::VectorXd> residuals;
};
