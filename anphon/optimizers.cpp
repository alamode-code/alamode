#include "mpi_common.h"
#include "optimizers.h"
#include "dynamical.h"
#include "gruneisen.h"
#include "system.h"
#include "constants.h"
#include "scph.h"
#include "parsephon.h"
#include "error.h"
#include "timer.h"
#include "mathfunctions.h"
#include <fftw3.h>
#include <iomanip>
#include <Eigen/Core>
#include <iomanip>


using namespace PHON_NS;

Optimizer::Optimizer() {}

Optimizer::~Optimizer() {}

Newton_Optimizer::Newton_Optimizer() {}

Newton_Optimizer::~Newton_Optimizer() {}

Newton_Optimizer::Newton_Optimizer(double mixbeta) {
    this->mixbeta = mixbeta;
    std::cout << "mixbeta = " << mixbeta << std::endl;
}

void Newton_Optimizer::update_state(const int dim,
                                    const std::vector<double> &grad_vec,
                                    std::vector<double> &state_vec,
                                    const std::vector<std::vector<double>> &hessian,
                                    std::vector<double> &delta) {

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

    for (int i = 0; i < dim; ++i) {
        delta[i] = -1.0 * mixbeta * delta_tmp(i);
        state_vec[i] = state_vec[i] + delta[i];
    }
}