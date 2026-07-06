/*
 optimizers.h

 Copyright (c) 2025 Takumi Chida, Ryota Masuki, Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <Eigen/Core>
#include <memory>
#include <vector>

namespace PHON_NS
{
// Geometry optimizers for the SCPH/QHA structural relaxation. Distinct from
// the GDIIS fixed-point mixer of the SCP inner iteration (diis.h): these act
// on the structural degrees of freedom between lattice-dynamics solves.
class Optimizer
{
public:
    Optimizer() = default;

    virtual ~Optimizer() = default;

    virtual void update_state(const int dim, const std::vector<double> &grad_vec, std::vector<double> &state_vec,
                              const std::vector<std::vector<double>> &hessian, std::vector<double> &delta) {};

    // Request re-initialization of any internal history before the next
    // update_state call (used when the structure is reset, e.g. at the start
    // of a new temperature point or after a divergence).
    void reset()
    {
        initialize_flag = 1;
    }

protected:
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
    FarkasIII_Optimizer(int max_vectors, const Eigen::MatrixXd &H_ini, bool gdiis_control);

    void update_state(const int dim, const std::vector<double> &grad_vec, std::vector<double> &state_vec,
                      const std::vector<std::vector<double>> &hessian, std::vector<double> &delta);

    // Regular GDIIS with a BFGS-updated inverse Hessian and the size-dependent
    // angle acceptance criterion.
    Eigen::VectorXd update(const Eigen::VectorXd &point, const Eigen::VectorXd &gradient);

    // Farkas-Schlegel "controlled GDIIS" (Phys. Chem. Chem. Phys. 2002, 4, 11): recomputes the
    // error vectors with a per-step RFO level-shifted Hessian, grows the DIIS subspace from the
    // most recent point keeping the last acceptable step, applies the four acceptance criteria
    // (a) angle, (b) step length, (c) coefficient sum, (d) near-singularity, and permanently
    // discards points when the angle exceeds 90 degrees. Enabled by default; disabled by GDIIS_PLAIN = 1.
    Eigen::VectorXd update_controlled(const Eigen::VectorXd &point, const Eigen::VectorXd &gradient);

    void set_inverse_Hessian(const int dim, const std::vector<std::vector<double>> &hessian);

    void initialize_history();

private:
    // RFO level-shifted effective inverse Hessian: returns (H_direct + lambda*I)^{-1} (the member
    // H stores the inverse Hessian). Used by update_controlled() for the reference step and the
    // error vectors.
    Eigen::MatrixXd effective_inverse_Hessian() const;

    // Farkas-Schlegel angle cut-off for cos(theta) as a function of the subspace size.
    static double angle_threshold(int n_vectors);

    int max_vectors;
    bool gdiis_control; // false: regular GDIIS (update); true: controlled GDIIS (update_controlled)
    Eigen::VectorXd point_old;
    Eigen::VectorXd gradient_old;
    double threshold_angle;
    double curvature_floor = 0.0; // min curvature of the initial (regularized) Hessian
    Eigen::MatrixXd H;
    std::vector<Eigen::VectorXd> points;
    std::vector<Eigen::VectorXd> residuals; // regular GDIIS (update)
    std::vector<Eigen::VectorXd> gradients; // controlled GDIIS (update_controlled)
};
} // namespace PHON_NS
