#pragma once

#include "fcs.h"

void rref(const size_t nrows,
          const size_t ncols,
          double **mat,
          size_t &nrank,
          const double tolerance = 1.0e-12);

void rref(std::vector<std::vector<double>> &mat,
          const double tolerance = 1.0e-12);

void rref_sparse(const size_t ncols,
                 ConstraintSparseForm &sp_constraint,
                 const double tolerance = 1.0e-12);

void rref_sparse2(const size_t ncols,
                  ConstraintSparseForm &sp_constraint,
                  const double tolerance = 1.0e-12);

void rref_sparse_fullpivot(const size_t ncols,
                           ConstraintSparseForm &sp_constraint,
                           const double tolerance = 1.0e-12);
