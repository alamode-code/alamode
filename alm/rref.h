#pragma once

#include "fcs.h"

auto rref(const size_t nrows, const size_t ncols, double **mat, size_t &nrank, const double tolerance = 1.0e-12)
    -> void;

auto rref(std::vector<std::vector<double>> &mat, const double tolerance = 1.0e-12) -> void;

auto rref_sparse(const size_t ncols, ConstraintSparseForm &sp_constraint, const double tolerance = 1.0e-12) -> void;

// void rref_sparse2(const size_t ncols,
//                   ConstraintSparseForm &sp_constraint,
//                   const double tolerance = 1.0e-12);
//
// void rref_sparse_fullpivot(const size_t ncols,
//                            ConstraintSparseForm &sp_constraint,
//                            const double tolerance = 1.0e-12);
