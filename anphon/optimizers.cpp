#include "optimizers.h"
#include "timer.h"
#include <iomanip>
#include <Eigen/Core>

using namespace PHON_NS;


Newton_Optimizer::Newton_Optimizer(double mixbeta) : mixbeta(mixbeta) {}

void Newton_Optimizer::update_state(const int dim,
                                    const std::vector<double> &grad_vec,
                                    std::vector<double> &state_vec,
                                    const std::vector<std::vector<double>> &hessian,
                                    std::vector<double> &delta)
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

    std::cout << "Hessian matrix:\n" << hessian_matrix << std::endl;
    std::cout << "Eigenvalue of Hessian matrix: \n" << hessian_matrix.eigenvalues().real() << std::endl;
    std::cout << "gradient:\n" << grad << std::endl;
    std::cout << "-H^{-1} * grad:\n" << delta_tmp << std::endl;
    std::cout << "H^{-1} \n" << hessian_matrix.inverse() << std::endl;


    for (int i = 0; i < dim; ++i) {
        delta[i] = -1.0 * mixbeta * delta_tmp(i);
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

void CellCoord_Newton_Optimizer::update_state(const int dim,
                                              const std::vector<double> &grad_vec,
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

void FarkasIII_Optimizer::update_state(const int dim,
                                       const std::vector<double> &grad_vec,
                                       std::vector<double> &state_vec,
                                       const std::vector<std::vector<double>> &hessian,
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

    updated_state = this->update(state_tmp, grad_tmp);

    // write answer
    for (int i = 0; i < dim; ++i) {
        delta[i] = updated_state(i) - state_vec[i];
        state_vec[i] = updated_state[i];
    }

}

void FarkasIII_Optimizer::set_inverse_Hessian(const int dim,
                                              const std::vector<std::vector<double>> &hessian)
{
    Eigen::MatrixXd H_tmp(dim, dim);
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            H_tmp(i, j) = hessian[i][j];
        }
    }
    H = H_tmp.inverse();
    // std::cout << "Inverse Hessian matrix initialized." << std::endl;
    // std::cout << H << std::endl;

}

void FarkasIII_Optimizer::initialize_history()
{
    points.clear();
    residuals.clear();
}