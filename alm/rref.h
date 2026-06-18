#pragma once

#include "fcs.h"

auto rref(const size_t nrows, const size_t ncols, double **mat, size_t &nrank,
          const double tolerance = 1.0e-12) -> void;

auto rref(std::vector<std::vector<double>> &mat, const double tolerance = 1.0e-12) -> void;

auto rref_sparse(const size_t ncols, ConstraintSparseForm &sp_constraint, const double tolerance = 1.0e-12) -> void;

// Reduced row echelon form with partial (maximum-magnitude) row pivoting.
// Produces the same echelon structure as rref_sparse (identical pivot columns and reduced
// coefficients up to round-off) but is numerically more stable, because it never divides by a
// small accepted pivot. The output is consumed unchanged by Constraint::get_mapping_constraint.
auto rref_sparse_pivot(const size_t ncols, ConstraintSparseForm &sp_constraint,
                       const double tolerance = 1.0e-12) -> void;

// void rref_sparse2(const size_t ncols,
//                   ConstraintSparseForm &sp_constraint,
//                   const double tolerance = 1.0e-12);
//
// void rref_sparse_fullpivot(const size_t ncols,
//                            ConstraintSparseForm &sp_constraint,
//                            const double tolerance = 1.0e-12);
