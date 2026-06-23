// accelerate_solver.cpp
//
// Translation unit isolating <Eigen/AccelerateSupport> (and thus <Accelerate/Accelerate.h>) from the
// hand-rolled Fortran BLAS/LAPACK prototypes in include/blas_wrapper.h / include/lapack_wrapper.h.
// See accelerate_solver.h for the rationale. This file intentionally includes NEITHER wrapper.
#include "accelerate_solver.h"

#ifdef USE_ACCEL_BACKEND
#include <Eigen/AccelerateSupport>

bool solve_kkt_accelerate_ldlt(const Eigen::SparseMatrix<double> &K, const Eigen::VectorXd &rhs,
                               Eigen::VectorXd &sol)
{
    // The KKT matrix is symmetric indefinite (saddle-point); Accelerate's sparse LDL^T handles it.
    Eigen::AccelerateLDLT<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Symmetric> ldlt(K);
    if (ldlt.info() != Eigen::Success) {
        return false;
    }
    sol = ldlt.solve(rhs);
    return ldlt.info() == Eigen::Success;
}
#endif
