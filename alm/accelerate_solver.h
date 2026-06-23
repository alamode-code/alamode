// accelerate_solver.h
//
// Apple Accelerate sparse KKT solver, deliberately isolated in its own translation unit.
//
// Eigen's <Eigen/AccelerateSupport> pulls in <Accelerate/Accelerate.h>, whose Fortran BLAS/LAPACK
// prototypes (dgemm_, dgesdd_, dgeqp3_, ...) collide with the hand-rolled prototypes in
// include/blas_wrapper.h and include/lapack_wrapper.h. Keeping the Accelerate include out of every
// TU that also includes those wrappers (least_squares.cpp in particular) avoids a hard redeclaration
// conflict. Only this header/source pair includes the Accelerate Eigen module.
#pragma once

#ifdef USE_ACCEL_BACKEND
#include <Eigen/SparseCore>

// Solve the symmetric-indefinite KKT system K x = rhs with Apple Accelerate's sparse LDL^T
// factorization. Returns true on success (factorization and solve both reported Eigen::Success).
bool solve_kkt_accelerate_ldlt(const Eigen::SparseMatrix<double> &K, const Eigen::VectorXd &rhs,
                               Eigen::VectorXd &sol);
#endif
