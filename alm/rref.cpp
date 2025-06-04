#include "rref.h"
#include "constraint.h"
#include <vector>
#include <algorithm>

void rref(const size_t nrows,
          const size_t ncols,
          double **mat,
          size_t &nrank,
          const double tolerance)
{
    // Return the reduced row echelon form (rref) of matrix mat.
    // In addition, rank of the matrix is estimated.

    size_t jcol;
    double tmp;

    nrank = 0;

    size_t icol = 0;

    for (size_t irow = 0; irow < nrows; ++irow) {

        auto pivot = irow;

        while (std::abs(mat[pivot][icol]) < tolerance) {
            ++pivot;

            if (pivot == nrows) {
                pivot = irow;
                ++icol;

                if (icol == ncols) break;
            }
        }

        if (icol == ncols) break;

        if (std::abs(mat[pivot][icol]) > tolerance) ++nrank;

        if (pivot != irow) {
            //#pragma omp parallel for private(tmp)
            for (jcol = icol; jcol < ncols; ++jcol) {
                tmp = mat[pivot][jcol];
                mat[pivot][jcol] = mat[irow][jcol];
                mat[irow][jcol] = tmp;
            }
        }

        tmp = mat[irow][icol];
        tmp = 1.0 / tmp;
        //#pragma omp parallel for
        for (jcol = icol; jcol < ncols; ++jcol) {
            mat[irow][jcol] *= tmp;
        }

        for (auto jrow = 0; jrow < nrows; ++jrow) {
            if (jrow == irow) continue;

            tmp = mat[jrow][icol];
            //#pragma omp parallel for
            for (jcol = icol; jcol < ncols; ++jcol) {
                mat[jrow][jcol] -= tmp * mat[irow][jcol];
            }
        }
    }
}


void rref(std::vector<std::vector<double>> &mat,
          const double tolerance)
{
    // Return the reduced row echelon form (rref) of matrix mat.
    // In addition, rank of the matrix is estimated.

    size_t jcol;
    double tmp;

    size_t nrank = 0;
    size_t icol = 0;

    const auto nrows = mat.size();
    const auto ncols = mat[0].size();

    for (size_t irow = 0; irow < nrows; ++irow) {

        auto pivot = irow;

        while (std::abs(mat[pivot][icol]) < tolerance) {
            ++pivot;

            if (pivot == nrows) {
                pivot = irow;
                ++icol;

                if (icol == ncols) break;
            }
        }

        if (icol == ncols) break;

        if (std::abs(mat[pivot][icol]) > tolerance) ++nrank;

        if (pivot != irow) {
            for (jcol = icol; jcol < ncols; ++jcol) {
                tmp = mat[pivot][jcol];
                mat[pivot][jcol] = mat[irow][jcol];
                mat[irow][jcol] = tmp;
            }
        }

        tmp = mat[irow][icol];
        tmp = 1.0 / tmp;
        for (jcol = icol; jcol < ncols; ++jcol) {
            mat[irow][jcol] *= tmp;
        }

        for (size_t jrow = 0; jrow < nrows; ++jrow) {
            if (jrow == irow) continue;

            tmp = mat[jrow][icol];
            for (jcol = icol; jcol < ncols; ++jcol) {
                mat[jrow][jcol] -= tmp * mat[irow][jcol];
            }
        }
    }

    mat.erase(mat.begin() + nrank, mat.end());
    mat.shrink_to_fit();
}


void rref_sparse(const size_t ncols,
                 ConstraintSparseForm &sp_constraint,
                 const double tolerance)
{
    // This function is somewhat sensitive to the numerical accuracy.
    // The loss of numerical digits can lead to instability.
    // Column ordering may improve the stability, but I'm not sure.
    // Smaller tolerance is preferable.

    const auto nrows = sp_constraint.size();
    size_t jrow;

    // This parameter controls the stability and performance.
    // Smaller value is more stable but a little more costly.
    //    double zero_criterion = tolerance * 1.0e-3;
    constexpr auto zero_criterion = eps10;

    size_t nrank = 0;
    size_t icol = 0;
    MapConstraintElement::iterator it_other;
    MapConstraintElement::iterator it_elem;

    for (size_t irow = 0; irow < nrows; ++irow) {

        auto pivot = irow;

        while (true) {
            it_elem = sp_constraint[pivot].find(icol);
            if (it_elem != sp_constraint[pivot].end()) {
                if (std::abs(it_elem->second) >= tolerance) {
                    break;
                }
            }

            ++pivot;
            if (pivot == nrows) {
                pivot = irow;
                ++icol;

                if (icol == ncols) break;
            }
        }

        if (icol == ncols) break;

        if (std::abs(it_elem->second) >= tolerance) ++nrank;

        if (pivot != irow) {
            std::iter_swap(sp_constraint.begin() + irow,
                           sp_constraint.begin() + pivot);
        }

        const double division_factor = 1.0 / it_elem->second;
        for (auto &[fst, snd]: sp_constraint[irow]) {
            snd *= division_factor;
        }

        for (jrow = 0; jrow < nrows; ++jrow) {
            if (jrow == irow) continue;

            it_elem = sp_constraint[jrow].find(icol);
            if (it_elem == sp_constraint[jrow].end()) continue;
            const double scaling_factor = it_elem->second;

            // Subtract irow elements from jrow
            for (const auto &[fst, snd]: sp_constraint[irow]) {
                // This part might be speeded up by using std::map::lower_bound
                // when the datatype of sp_constraint[irow] is std::map.
                if (fst < icol) {
                    continue;
                }
                it_other = sp_constraint[jrow].find(fst);
                if (it_other != sp_constraint[jrow].end()) {
                    it_other->second -= scaling_factor * snd;
                    // Delete zero elements and remove from the map.
                    // A smaller threshold is used for better stability.
                    if (std::abs(it_other->second) < zero_criterion) {
                        sp_constraint[jrow].erase(it_other);
                    }
                } else {
                    sp_constraint[jrow][fst] = -scaling_factor * snd;
                }
            }
            // Make sure to erase the icol element from the target row if it exists.
            // When the original pivot element is large, the element after subtraction can sometimes be 
            // larger than the tolerance value because of the loss of significant digits.
            it_other = sp_constraint[jrow].find(icol);
            if (it_other != sp_constraint[jrow].end()) {
                sp_constraint[jrow].erase(it_other);
            }
        }
    }

    // Erase all elements smaller than the tolerance value
    for (jrow = 0; jrow < nrows; ++jrow) {
        it_other = sp_constraint[jrow].begin();
        while (it_other != sp_constraint[jrow].end()) {
            if (std::abs(it_other->second) <= tolerance) {
                sp_constraint[jrow].erase(it_other++);
            } else {
                ++it_other;
            }
        }
    }

    // Remove emptry entries from the sp_constraint vector
    sp_constraint.erase(std::remove_if(sp_constraint.begin(),
                                       sp_constraint.end(),
                                       [](const MapConstraintElement &obj) {
                                           return obj.empty();
                                       }),
                        sp_constraint.end());
    sp_constraint.shrink_to_fit();
}
