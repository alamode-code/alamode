/*
 constraint.cpp

 Copyright (c) 2014-2022 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "constraint.h"
#include <Eigen/Dense>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/bimap.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <chrono>
#include <highfive/H5Easy.hpp>
#include <iomanip>
#include <iostream>
#include <map>
#include <unordered_set>
#include "cluster.h"
#include "combination.h"
#include "error.h"
#include "fcs.h"
#include "hdf5_parser.h"
#include "least_squares.h"
#include "logger.h"
#include "memory.h"
#include "rref.h"
#include "svd.h"
#include "symmetry.h"
#include "system.h"
#include "timer.h"
#include "xml_parser.h"

using namespace ALM_NS;

namespace
{
// With STRESS, column col_rhs of the harmonic rows holds the constant part of the Huang rows. The
// reductions may pivot on it only when a row has lost all its FC coefficients, i.e. reads 0 = const:
// the constraints are inconsistent. Every reduced set that contains this column is checked here.
void check_huang_rhs_rows(const ConstraintSparseForm &rows, const size_t col_rhs, const char *where)
{
    for (const auto &row: rows) {
        if (row.find(col_rhs) != row.end() && row.size() == 1) {
            exit(where,
                 "The Huang constraints with STRESS are inconsistent with the other constraints\n"
                 " (a reduced constraint reads 0 = const). Check STRESS and the cutoff radii.\n"
                 " With ICONST = 14, nearly dependent rotational constraints can also cause this;\n"
                 " try ICONST = 4 or a larger TOL_CONST.");
        }
    }
}

// The QR row selection copies the block to a dense matrix (dgeqp3) unless built with SuiteSparse;
// the size limit (~400 MB) is that of the test in select_independent_rows.
bool qr_affordable(const size_t nrows, const size_t ncols, const int verbosity, const char *what)
{
#ifdef USE_SUITESPARSE_BACKEND
    constexpr bool dense_qr = false;
#else
    constexpr bool dense_qr = true;
#endif
    if (!dense_qr || static_cast<double>(nrows) * static_cast<double>(ncols) <= 5.0e7) return true;
    if (verbosity > 0) {
        std::cout << "  Note: " << what << " (" << nrows << " x " << ncols
                  << ") too large for the dense QR; reduced by RREF.\n";
    }
    return false;
}

// Independent rows of C x = d. The rows are selected as before by the pivoted QR with the
// machine-precision tolerance (rank_machine), unless the scale-free rank is smaller: with each row
// scaled to unit norm, R_kk is the sine of the angle between a row and the span of the rows before it,
// and rows with R_kk <= tolerance (TOL_CONST) are treated as dependent. Such nearly dependent rows
// (e.g., rotational constraints made independent only by a small internal relaxation) would otherwise
// be imposed exactly and spoil the fit; the algebraic path drops them with TOL_CONST as well.
// The test only removes rows, so rows of a tiny norm are never promoted. Rows are not removed if the
// kept rows then violate the dropped ones (nonzero d, i.e. STRESS), or if the matrix is too large.
void select_independent_rows(const Eigen::SparseMatrix<double> &C, const Eigen::VectorXd &d, const int verbosity,
                             const double tolerance, const bool test_near_dependence,
                             Eigen::SparseMatrix<double> &C_red, Eigen::VectorXd &d_red, int &rank, int &rank_machine)
{
    get_independent_rows_lapack_sparse(C, d, verbosity, rank_tolerance_auto, C_red, d_red, rank);
    rank_machine = rank;
    if (rank == 0 || !test_near_dependence) return;

    // The test runs on the selected rows only; the others are dependent to machine precision.
    const Eigen::SparseMatrix<double, Eigen::RowMajor> C_row = C_red;
    const auto nr = static_cast<int>(C_row.rows()), nc = static_cast<int>(C_row.cols());
    // ponytail: dense rank-revealing QR, limited to ~400 MB; no sparse rank-revealing QR is available.
    if (static_cast<double>(nr) * nc > 5.0e7) {
        if (verbosity > 0) {
            std::cout << "  Note: the constraint matrix is too large for the test of nearly dependent\n"
                         "        constraints; all constraints independent to machine precision are kept.\n\n";
        }
        return;
    }
    const Eigen::MatrixXd c_dense = C_red;
    Eigen::MatrixXd unit_rows = c_dense;
    for (int i = 0; i < nr; ++i) unit_rows.row(i).normalize();
    Eigen::MatrixXd a_t = unit_rows.transpose(); // column i = unit row i
    int rank_unit;
    std::vector<int> pivots;
    if (find_independent_rows_dense(nr, nc, a_t.data(), tolerance, rank_unit, pivots) != 0) return;
    if (rank_unit >= rank) return;

    if (d_red.cwiseAbs().maxCoeff() > 0.0) {
        // Keep the dropped rows unless the kept rows satisfy them: |c_i x - d_i| <= 10 * TOL_CONST |c_i| |x|
        // for the minimum-norm solution x of the kept rows (a dropped row has sine <= TOL_CONST to them).
        Eigen::MatrixXd c_kept(rank_unit, nc);
        Eigen::VectorXd d_kept(rank_unit);
        for (int k = 0; k < rank_unit; ++k) {
            c_kept.row(k) = c_dense.row(pivots[k]);
            d_kept[k] = d_red[pivots[k]];
        }
        const Eigen::VectorXd x = Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd>(c_kept).solve(d_kept);
        const Eigen::VectorXd resid = c_dense * x - d_red;
        for (int i = 0; i < nr; ++i) {
            if (std::abs(resid[i]) > 10.0 * tolerance * c_dense.row(i).norm() * x.norm()) {
                if (verbosity > 0) {
                    std::cout << "  Note: " << rank - rank_unit << " nearly dependent constraint(s) are kept: their\n"
                              << "        right-hand side (STRESS) is not satisfied by the other constraints.\n\n";
                }
                return;
            }
        }
    }

    // Keep the original (unscaled) rows of the scale-free selection.
    std::vector<Eigen::Triplet<double>> tri;
    const Eigen::VectorXd d_sel = d_red;
    d_red.resize(rank_unit);
    for (int k = 0; k < rank_unit; ++k) {
        d_red[k] = d_sel[pivots[k]];
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(C_row, pivots[k]); it; ++it) {
            tri.emplace_back(k, it.col(), it.value());
        }
    }
    C_red.resize(rank_unit, nc);
    C_red.setFromTriplets(tri.begin(), tri.end());
    C_red.makeCompressed();
    if (verbosity > 0) {
        std::cout << "  " << rank - rank_unit << " nearly dependent constraint(s) were removed"
                  << " (TOL_CONST = " << tolerance << ").\n\n";
    }
    rank = rank_unit;
}

// Algebraic map of the coordinate-preserving path: the scale-free row selection of
// select_independent_rows (column nparam holds the constant part -d of C x = d), then the elimination
// rref_threshold_pivot. pivot_col[k] is the parameter eliminated by row k.
void reduce_for_mapping(const size_t nparam, ConstraintSparseForm &rows, const double tolerance,
                        const bool test_near_dependence, const int verbosity, std::vector<size_t> &pivot_col,
                        const char *where)
{
    if (!qr_affordable(rows.size(), nparam, verbosity, "constraints of the mapping")) {
        // Echelon form with the first entry of each row as its pivot (pivot_col empty).
        pivot_col.clear();
        rref_sparse_pivot(nparam + 1, rows, tolerance);
        return;
    }
    const auto inconsistent = [where] {
        exit(where,
             "The Huang constraints with STRESS are inconsistent with the other constraints.\n"
             " Nearly dependent rotational constraints can cause this; try ICONST = 4 or a\n"
             " larger TOL_CONST.");
    };
    std::vector<Eigen::Triplet<double>> tri;
    Eigen::VectorXd d = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(rows.size()));
    for (size_t r = 0; r < rows.size(); ++r) {
        for (const auto &[c, v]: rows[r]) {
            if (c < nparam) {
                tri.emplace_back(r, c, v);
            } else {
                d[r] = -v;
            }
        }
    }
    Eigen::SparseMatrix<double> C(static_cast<Eigen::Index>(rows.size()), static_cast<Eigen::Index>(nparam)), C_red;
    C.setFromTriplets(tri.begin(), tri.end());
    C.makeCompressed();
    Eigen::VectorXd d_red;
    int rank, rank_machine;
    select_independent_rows(C, d, verbosity, tolerance, test_near_dependence, C_red, d_red, rank, rank_machine);

    if (rows.size() > 0 && d.cwiseAbs().maxCoeff() > 0.0) {
        // The selection uses C only: rows dependent in C with a different constant part are inconsistent.
        const auto dmax = d.cwiseAbs().maxCoeff();
        auto cmax = 0.0;
        for (const auto &t: tri) cmax = std::max(cmax, std::abs(t.value()));
        auto tri_aug = tri;
        for (Eigen::Index r = 0; r < d.size(); ++r) {
            if (d[r] != 0.0) tri_aug.emplace_back(r, nparam, d[r] * cmax / dmax);
        }
        Eigen::SparseMatrix<double> C_aug(C.rows(), C.cols() + 1), mat_dummy;
        C_aug.setFromTriplets(tri_aug.begin(), tri_aug.end());
        C_aug.makeCompressed();
        Eigen::VectorXd rhs_dummy;
        int rank_aug;
        get_independent_rows_lapack_sparse(C_aug, d, 0, rank_tolerance_auto, mat_dummy, rhs_dummy, rank_aug);
        if (rank_aug != rank_machine) inconsistent();
    }

    const Eigen::SparseMatrix<double, Eigen::RowMajor> C_row = C_red;
    rows.assign(rank, MapConstraintElement());
    for (int k = 0; k < rank; ++k) {
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(C_row, k); it; ++it) {
            rows[k][it.col()] = it.value();
        }
        if (d_red[k] != 0.0) rows[k][nparam] = -d_red[k];
    }

    size_t ndropped;
    bool rhs_left;
    rref_threshold_pivot(nparam, rows, tolerance, pivot_col, ndropped, rhs_left);
    if (rhs_left) inconsistent();
    if (ndropped > 0 && verbosity > 0) {
        std::cout << "  Note: " << ndropped << " constraint(s) became dependent in the elimination"
                  << " and were removed.\n\n";
    }
}
} // namespace

Constraint::Constraint()
{
    set_default_variables();
}

Constraint::~Constraint()
{
    deallocate_variables();
}

auto Constraint::set_default_variables() -> void
{
    constraint_mode = 11;
    rotation_axis = "";
    fix_harmonic = false;
    fix_cubic = false;
    constraint_algebraic = 1;
    fc2_file = "";
    fc3_file = "";
    const_mat = nullptr;
    const_rhs = nullptr;
    index_bimap = nullptr;
    number_of_constraints = 0;
    tolerance_constraint = 1.0e-6; // same default as TOL_CONST of the CLI
    status_constraint_subset["symmetry"] = 0;
    status_constraint_subset["translation"] = -1;
    status_constraint_subset["rotation"] = -1;
    status_constraint_subset["rotation_extra"] = -1;
    status_constraint_subset["fix2"] = -1;
    status_constraint_subset["fix3"] = -1;
    status_constraint_subset["huang"] = -1;
    // Default reduction backend: coord_factorization (partial-pivot RREF). Stable replacement
    // for the legacy rref; ALGO_REDUCTION = 1 restores rref. Kept in sync with input_parser.
    algo_reduction = ReductionAlgo::coord_factorization;
}

auto Constraint::deallocate_variables() -> void
{
    if (index_bimap) {
        deallocate(index_bimap);
    }
    if (const_mat) {
        deallocate(const_mat);
    }
    if (const_rhs) {
        deallocate(const_rhs);
    }
}

auto Constraint::setup(const std::unique_ptr<System> &system, const std::unique_ptr<Fcs> &fcs,
                       const std::unique_ptr<Cluster> &cluster, const std::unique_ptr<Symmetry> &symmetry,
                       const int linear_model, const int periodic_image_conv, const int verbosity,
                       std::unique_ptr<Timer> &timer) -> void
{
    timer->start_clock("constraint");

    if (verbosity > 0) {
        std::cout << " ============\n";
        std::cout << "  CONSTRAINT \n";
        std::cout << " ============\n\n";
    }

    constraint_mode = constraint_mode % 10;

    if (linear_model >= 2) {
        constraint_algebraic = 1;
    }

    switch (constraint_mode) {
    case 0: // do nothing
        impose_inv_T = false;
        impose_inv_R = false;
        impose_inv_Huang = false;
        set_constraint_flag("translation", 0);
        set_constraint_flag("rotation", 0);
        set_constraint_flag("rotation_extra", 0);
        set_constraint_flag("huang", 0);
        if (verbosity > 0) {
            std::cout << "  ICONST = 0: Constraint for translational/rotational invariance\n";
            std::cout << "              will NOT be considered.\n";
        }
        break;
    case 1:
        impose_inv_T = true;
        impose_inv_R = false;
        impose_inv_Huang = false;
        set_constraint_flag("translation", 1);
        set_constraint_flag("rotation", 0);
        set_constraint_flag("rotation_extra", 0);
        set_constraint_flag("huang", 0);
        if (verbosity > 0) {
            std::cout << "  ICONST = 1: Constraints for translational invariance\n";
            std::cout << "              will be considered.\n";
        }
        break;
    case 2:
        impose_inv_T = true;
        impose_inv_R = true;
        exclude_last_R = true;
        impose_inv_Huang = false;
        set_constraint_flag("translation", 1);
        set_constraint_flag("rotation", 1);
        set_constraint_flag("rotation_extra", 0);
        set_constraint_flag("huang", 0);
        if (verbosity > 0) {
            std::cout << "  ICONST = 2: Constraints for translational and rotational invariance\n";
            std::cout << "              will be considered. Axis of rotation is " << rotation_axis << '\n';
            std::cout << "              Rotational invariance of the maximum order will be neglected\n";
        }
        break;
    case 3:
        impose_inv_T = true;
        impose_inv_R = true;
        exclude_last_R = false;
        impose_inv_Huang = false;
        set_constraint_flag("translation", 1);
        set_constraint_flag("rotation", 1);
        set_constraint_flag("rotation_extra", 1);
        set_constraint_flag("huang", 0);
        if (verbosity > 0) {
            std::cout << "  ICONST = 3: Constraints for translational and rotational invariance\n";
            std::cout << "              will be considered. Axis of rotation is " << rotation_axis << '\n';
            std::cout << "              The rotational invariance of the maximum order assumes that the\n";
            std::cout << "              next-order IFCs are zero. If they are not negligible, this over-constrains\n";
            std::cout << "              the IFCs; compare the fitting error with ICONST = 2 (or 12).\n";
        }
        break;
    case 4:
        impose_inv_T = true;
        impose_inv_R = true;
        impose_inv_Huang = true;
        set_constraint_flag("translation", 1);
        set_constraint_flag("rotation", 1);
        set_constraint_flag("rotation_extra", 0);
        set_constraint_flag("huang", 1);
        if (verbosity > 0) {
            std::cout << "  ICONST = 4: Constraints for translational, rotational, and Huang invariance\n";
            std::cout << "              will be considered. Axis of rotation is " << rotation_axis << '\n';
        }
        break;
    default:
        exit("Constraint::setup", "invalid constraint_mode", constraint_mode);
        break;
    }

    if (fcs->get_forceconstant_basis() == "Lattice" && impose_inv_R) {
        exit("Constraint::setup()",
             "Sorry, rotational invariance with FCSYM_BASIS = Lattice is "
             "not supported.\n Use FCSYM_BASIS = Cartesian instead.");
    }
    if (verbosity > 0) std::cout << '\n';

    huang_rhs = false;
    if (stress_given) {
        if (impose_inv_Huang) {
            setup_huang_stress(symmetry, verbosity);
        } else {
            warn("Constraint::setup", "STRESS is used only with ICONST = 4 or 14 (Huang invariance); ignored.");
        }
    }

    if (fix_harmonic) {
        if (verbosity > 0) {
            std::cout << "  FC2FIX is given : Harmonic force constants will be \n";
            std::cout << "                    fixed to the values given in " << fc2_file << "\n\n";
        }
        std::vector<std::vector<int>> intpair_fix;
        std::vector<double> values_fix;
        get_forceconstants_from_file(0, symmetry, fcs, fc2_file, intpair_fix, values_fix);

        set_forceconstants_to_fix(intpair_fix, values_fix);
    }

    fix_cubic = fix_cubic & (cluster->get_maxorder() > 1);
    if (fix_cubic) {
        if (verbosity > 0) {
            std::cout << "  FC3FIX is given : Cubic force constants will be \n";
            std::cout << "                    fixed to the values given in " << fc3_file << "\n\n";
        }
        std::vector<std::vector<int>> intpair_fix;
        std::vector<double> values_fix;
        get_forceconstants_from_file(1, symmetry, fcs, fc3_file, intpair_fix, values_fix);

        set_forceconstants_to_fix(intpair_fix, values_fix);
    }

    status_constraint_subset["symmetry"] = 0;

    update_constraint_matrix(system, symmetry, cluster, fcs, verbosity, periodic_image_conv, algo_reduction);

    if (verbosity > 0) {
        print_constraint_information(cluster);
        timer->print_elapsed();
        std::cout << " -------------------------------------------------------------------" << '\n';
        std::cout << '\n';
    }

    timer->stop_clock("constraint");
}

auto Constraint::update_constraint_symmetry(const size_t nat, const int maxorder,
                                            const std::unique_ptr<Symmetry> &symmetry,
                                            const std::unique_ptr<Cluster> &cluster, const std::unique_ptr<Fcs> &fcs,
                                            const int verbosity, const ReductionAlgo algo_in) -> void
{

    if (const_symmetry.size() != maxorder) const_symmetry.resize(maxorder);

    if (status_constraint_subset["symmetry"] == 0) {
        generate_symmetry_constraint(nat, symmetry, cluster, fcs, verbosity, algo_in);

        status_constraint_subset["symmetry"] = 1;
    }
}

auto Constraint::update_constraint_translation(const Cell &supercell, const int maxorder,
                                               const std::unique_ptr<Symmetry> &symmetry,
                                               const std::unique_ptr<Cluster> &cluster, const std::unique_ptr<Fcs> &fcs,
                                               const int periodic_image_conv, const int verbosity,
                                               const ReductionAlgo algo_in) -> void
{
    if (const_translation.size() != maxorder) const_translation.resize(maxorder);

    if (status_constraint_subset["translation"] == -1) {
        for (auto order = 0; order < maxorder; ++order) {
            const_translation[order].clear();
            const_translation[order].shrink_to_fit();
        }
    }
    if (status_constraint_subset["translation"] == 0) {
        generate_translational_constraint(supercell, symmetry, cluster, fcs, periodic_image_conv, verbosity, algo_in);
        status_constraint_subset["translation"] = 1;
    }
}


auto Constraint::update_constraint_rotation(const std::unique_ptr<System> &system, const int maxorder,
                                            const std::unique_ptr<Symmetry> &symmetry,
                                            const std::unique_ptr<Cluster> &cluster, const std::unique_ptr<Fcs> &fcs,
                                            const int periodic_image_conv, const int verbosity,
                                            const ReductionAlgo algo_in) -> void
{
    if (const_rotation_self.size() != maxorder) const_rotation_self.resize(maxorder);
    if (const_rotation_cross.size() != maxorder) const_rotation_cross.resize(maxorder);

    if (status_constraint_subset["rotation"] == -1 or status_constraint_subset["rotation_extra"] == -1) {
        for (auto order = 0; order < maxorder; ++order) {
            const_rotation_self[order].clear();
            const_rotation_cross[order].clear();
            const_rotation_self[order].shrink_to_fit();
            const_rotation_cross[order].shrink_to_fit();
        }
    }

    if (status_constraint_subset["rotation"] == 0 or status_constraint_subset["rotation_extra"] == 0) {
        generate_rotational_constraint(system, symmetry, cluster, fcs, verbosity, tolerance_constraint, algo_in);

        if (status_constraint_subset["rotation"] == 0) status_constraint_subset["rotation"] = 1;
        if (status_constraint_subset["rotation_extra"] == 0) status_constraint_subset["rotation_extra"] = 1;
    }
}

auto Constraint::update_constraint_huang(const std::unique_ptr<System> &system,
                                         const std::unique_ptr<Symmetry> &symmetry,
                                         const std::unique_ptr<Cluster> &cluster, const std::unique_ptr<Fcs> &fcs,
                                         const int verbosity, const ReductionAlgo algo_in) -> void
{
    if (const_huang.size() != 1) const_huang.resize(1);
    if (status_constraint_subset["huang"] == -1) {
        const_huang[0].clear();
        const_huang[0].shrink_to_fit();
    }

    if (status_constraint_subset["huang"] == 0) {
        // Implement a function to compute the huang constraint

        generate_huang_constraint(system->get_supercell(),
                                  symmetry,
                                  cluster,
                                  fcs,
                                  system->get_x_image(),
                                  verbosity,
                                  algo_in);
        status_constraint_subset["huang"] = 1;
    }
}

auto Constraint::update_constraint_fix(const int maxorder, const std::unique_ptr<Symmetry> &symmetry,
                                       const std::unique_ptr<Fcs> &fcs) -> void
{
    if (const_fix.size() != maxorder) const_fix.resize(maxorder);
    if (status_constraint_subset["fix2"] == -1 or status_constraint_subset["fix3"] == -1) {
        for (auto order = 0; order < maxorder; ++order) {
            const_fix[order].clear();
            const_fix[order].shrink_to_fit();
        }
    }
    if (status_constraint_subset["fix2"] == 0 or status_constraint_subset["fix3"] == 0) {
        generate_fix_constraint(symmetry, fcs);
    }
}


auto Constraint::update_constraint_matrix(const std::unique_ptr<System> &system,
                                          const std::unique_ptr<Symmetry> &symmetry,
                                          const std::unique_ptr<Cluster> &cluster, const std::unique_ptr<Fcs> &fcs,
                                          const int verbosity, const int periodic_image_conv,
                                          const ReductionAlgo algo_in) -> void
{
    const auto maxorder = cluster->get_maxorder();

    // Use rank-revealing QR for ICONST = 1/2/3: fixed absolute pivot tolerances
    // can mistake round-off for independent constraints and corrupt the fit.
    // For ICONST >= 10, use the configured backend for the elimination map.
    const auto algo = constraint_algebraic ? algo_in : ReductionAlgo::qrd;

    // const_symmetry is updated.
    update_constraint_symmetry(system->get_supercell().number_of_atoms,
                               maxorder,
                               symmetry,
                               cluster,
                               fcs,
                               verbosity,
                               algo);

    // const_translation is updated.
    update_constraint_translation(system->get_supercell(),
                                  maxorder,
                                  symmetry,
                                  cluster,
                                  fcs,
                                  periodic_image_conv,
                                  verbosity,
                                  algo);

    // const_rotation_self and const_rotation_cross are updated.
    update_constraint_rotation(system, maxorder, symmetry, cluster, fcs, periodic_image_conv, verbosity, algo);

    // const_huang is updated.
    update_constraint_huang(system, symmetry, cluster, fcs, verbosity, algo);

    // const_fix is updated.
    update_constraint_fix(maxorder, symmetry, fcs);

    if (const_self.size() != maxorder) const_self.resize(maxorder);

    for (auto order = 0; order < maxorder; ++order) {
        const_self[order].clear();
        const_self[order].shrink_to_fit();
    }

    // Merge intra-order constraints and do reduction
    // This part needs to be updated if the huang constraint is considered under finite strain.
    const auto t_reduce_start = std::chrono::steady_clock::now();

    for (auto order = 0; order < maxorder; ++order) {
        // With FC2FIX/FC3FIX, const_self[order] is never used (the merged matrix and the
        // algebraic mapping both skip orders with const_fix), so skip the merge and reduction.
        if (!const_fix[order].empty()) continue;

        // Extra column for the inhomogeneous part of the Huang rows (STRESS).
        const auto nparam = fcs->get_nequiv()[order].size() + (order == 0 && huang_rhs ? 1 : 0);

        auto nlen_const =
            const_translation[order].size() + const_rotation_self[order].size() + const_symmetry[order].size();

        if (order == 0) {
            nlen_const += const_huang[order].size();
        }

        const_self[order].reserve(nlen_const);

        const_self[order].insert(const_self[order].end(), const_symmetry[order].begin(), const_symmetry[order].end());

        const_self[order].insert(const_self[order].end(),
                                 const_translation[order].begin(),
                                 const_translation[order].end());

        if (order == 0) {
            const_self[order].insert(const_self[order].end(), const_huang[0].begin(), const_huang[0].end());
        }

        const_self[order].insert(const_self[order].end(),
                                 const_rotation_self[order].begin(),
                                 const_rotation_self[order].end());

        if (algo == ReductionAlgo::rref) {
            rref_sparse(nparam, const_self[order], tolerance_constraint);
        } else if (algo == ReductionAlgo::qrd) {
            int rank;
            get_independent_rows_lapack_sparse(nparam, const_self[order], verbosity, rank_tolerance_auto, rank);
        } else if (algo == ReductionAlgo::coord_factorization) {
            // Independent rows only (algebraic path); the elimination map is built in reduce_for_mapping.
            // A block too large for the dense QR is reduced to echelon form instead.
            int rank;
            if (qr_affordable(const_self[order].size(), nparam, verbosity, "merged constraints")) {
                get_independent_rows_lapack_sparse(nparam, const_self[order], 0, rank_tolerance_auto, rank);
            } else if (!rref_sparse_pivot_two_stage(nparam,
                                                    const_self[order],
                                                    const_symmetry[order].size(),
                                                    tolerance_constraint) &&
                       verbosity > 0)
            {
                std::cout << "  Note: borderline pivots near TOL_CONST for order " << order + 1
                          << "; used the single-pass reduction.\n";
            }
        } else if (algo == ReductionAlgo::spqr_map) {
            rref_sparse_spqr(nparam, const_self[order], tolerance_constraint);
        }

        if (order == 0 && huang_rhs) check_huang_rhs_rows(const_self[0], nparam - 1, "update_constraint_matrix");
    }

    // Only non-algebraic solvers need the merged constraint matrix.
    // For ICONST >= 10, use the per-order elimination map below and avoid
    // the costly merged-matrix rank reduction.
    if (!constraint_algebraic) {
        size_t nparams = 0;
        for (auto order = 0; order < maxorder; ++order) {
            nparams += fcs->get_nequiv()[order].size();
        }
        build_constraint_matrix_sparse(maxorder, fcs->get_nequiv(), nparams, verbosity);
        number_of_constraints = const_mat_sparse.rows();
        // Drop any stale dense copy; it is rebuilt lazily for the dense solver only.
        deallocate(const_mat);
        deallocate(const_rhs);
    }

    if (const_relate.size() != maxorder) const_relate.resize(maxorder);

    if (index_bimap) {
        deallocate(index_bimap);
        index_bimap = nullptr;
    }
    allocate(index_bimap, maxorder);
    for (auto order = 0; order < maxorder; ++order) {
        index_bimap[order].clear();
        const_relate[order].clear();
        const_relate[order].shrink_to_fit();
    }

    if (constraint_algebraic) {
        // The mapping needs reduced row echelon form; only qrd / none still need reduction.
        if (algo != ReductionAlgo::rref && algo != ReductionAlgo::coord_factorization &&
            algo != ReductionAlgo::spqr_map)
        {
            for (auto order = 0; order < maxorder; ++order) {
                if (!const_fix[order].empty()) continue;
                const auto nparam = fcs->get_nequiv()[order].size() + (order == 0 && huang_rhs ? 1 : 0);
                rref_sparse(nparam, const_self[order], tolerance_constraint);
                if (order == 0 && huang_rhs) {
                    check_huang_rhs_rows(const_self[0], nparam - 1, "update_constraint_matrix");
                }
            }
        }

        // Inter-order rotational constraints are kept only between two non-fixed orders,
        // as in build_constraint_matrix_sparse. If any survive, eliminate globally.
        cross_order_algebraic = false;
        for (auto order = 1; order < maxorder; ++order) {
            if (!const_rotation_cross[order].empty() && const_fix[order - 1].empty() && const_fix[order].empty()) {
                cross_order_algebraic = true;
            }
        }

        if (cross_order_algebraic) {
            get_mapping_constraint_cross_order(maxorder, fcs->get_nequiv(), algo, verbosity);
        } else if (algo == ReductionAlgo::coord_factorization) {
            std::vector<std::vector<size_t>> pivots(maxorder);
            for (auto order = 0; order < maxorder; ++order) {
                if (!const_fix[order].empty()) continue;
                reduce_for_mapping(fcs->get_nequiv()[order].size(),
                                   const_self[order],
                                   tolerance_constraint,
                                   impose_inv_R || impose_inv_Huang,
                                   verbosity,
                                   pivots[order],
                                   "get_mapping_constraint");
            }
            get_mapping_constraint(maxorder,
                                   fcs->get_nequiv(),
                                   const_self.data(),
                                   const_fix.data(),
                                   const_relate.data(),
                                   index_bimap,
                                   pivots.data());
        } else {
            get_mapping_constraint(maxorder,
                                   fcs->get_nequiv(),
                                   const_self.data(),
                                   const_fix.data(),
                                   const_relate.data(),
                                   index_bimap);
        }

        // Count one constraint per fixed or related parameter for reporting.
        number_of_constraints = 0;
        for (auto order = 0; order < maxorder; ++order) {
            number_of_constraints += const_fix[order].size() + const_relate[order].size();
        }
    }

    const auto t_reduce_end = std::chrono::steady_clock::now();
    if (verbosity > 0) {
        const auto elapsed_reduce = std::chrono::duration<double>(t_reduce_end - t_reduce_start).count();
        // Save/restore cout formatting: std::fixed/std::setprecision are sticky and would
        // otherwise alter the formatting of all subsequent floating-point output.
        const auto saved_flags = std::cout.flags();
        const auto saved_prec = std::cout.precision();
        std::cout << "  Constraint reduction (merge + rank reduction + mapping) took " << std::fixed
                  << std::setprecision(3) << elapsed_reduce << " sec.\n\n";
        std::cout.flags(saved_flags);
        std::cout.precision(saved_prec);
    }
}

auto Constraint::print_constraint_information(const std::unique_ptr<Cluster> &cluster) const -> void
{
    const auto maxorder = cluster->get_maxorder();
    auto extra_constraint_from_symmetry = false;
    for (auto order = 0; order < cluster->get_maxorder(); ++order) {
        if (!const_symmetry[order].empty()) extra_constraint_from_symmetry = true;
    }

    const auto exist_constraint = get_exist_constraint();

    if (exist_constraint) {
        int order;

        if (impose_inv_T || impose_inv_R) {
            std::cout << "  Number of constraints [T-inv, R-inv (self), R-inv (cross)]:\n";
            for (order = 0; order < maxorder; ++order) {
                std::cout << "   " << std::setw(8) << cluster->get_ordername(order);
                std::cout << " " << std::setw(6) << const_translation[order].size();
                std::cout << std::setw(5) << const_rotation_self[order].size();
                std::cout << std::setw(5) << const_rotation_cross[order].size();
                std::cout << '\n';
            }
            std::cout << '\n';
        }

        if (extra_constraint_from_symmetry) {
            std::cout << "  There are constraints from crystal symmetry.\n";
            std::cout << "  The number of such constraints for each order:\n";
            for (order = 0; order < maxorder; ++order) {
                std::cout << "   " << std::setw(8) << cluster->get_ordername(order);
                std::cout << " " << std::setw(6) << const_symmetry[order].size();
                std::cout << '\n';
            }
            std::cout << '\n';
        }

        if (extra_constraint_from_symmetry) {
            std::cout << "  Constraints of T-inv, R-inv (self), and those from crystal symmetry are merged.\n";
        } else {
            std::cout << "  Constraints of T-inv and R-inv (self) are merged.\n";
        }
        std::cout << "  If there are redundant constraints, they are removed in this process.\n\n";
        std::cout << "  Number of inequivalent constraints (self, cross) : \n";

        for (order = 0; order < maxorder; ++order) {
            std::cout << "   " << std::setw(8) << cluster->get_ordername(order);
            std::cout << " " << std::setw(6) << const_self[order].size();
            std::cout << std::setw(5) << const_rotation_cross[order].size();
            std::cout << '\n';
        }
        std::cout << '\n';

        if (constraint_algebraic) {
            std::cout << "  ICONST >= 10 : Constraints will be considered algebraically.\n\n";

            if (cross_order_algebraic) {
                std::cout << "  Inter-order constraints for rotational invariance are included:\n"
                          << "  higher-order FCs are expressed through lower-order ones where possible.\n\n";
            }
            for (order = 1; order < maxorder; ++order) {
                // const_fix also holds parameters mapped to zero, so test the FCxFIX flags.
                const auto touches_fixed = (order == 1 && fix_harmonic) || (order <= 2 && fix_cubic);
                if (!const_rotation_cross[order].empty() && touches_fixed) {
                    std::cout << "  WARNING : Inter-order rotational constraints between "
                              << cluster->get_ordername(order - 1) << " and " << cluster->get_ordername(order)
                              << " are neglected (FC2FIX/FC3FIX).\n";
                }
            }

            for (order = 0; order < maxorder; ++order) {
                std::cout << "  Number of free" << std::setw(9) << cluster->get_ordername(order)
                          << " FCs : " << index_bimap[order].size() << '\n';
            }
            std::cout << '\n';
        } else {
            std::cout << "  Total number of constraints = " << number_of_constraints << "\n\n";
        }
    }
}


auto Constraint::build_constraint_matrix_sparse(const int maxorder, const std::vector<size_t> *nequiv,
                                                const size_t nparams, const int verbosity) -> void
{
    // Create a sparse matrix by merging const_self, const_cross, and const_fix
    // These constraints need to be constructed beforehand.

    int order;

    size_t nshift = 0;

    using tri = Eigen::Triplet<double, size_t>;
    std::vector<tri> triplets;

    size_t icount = 0;
    std::vector<std::pair<size_t, double>> rhs_huang; // (row, rhs) from the STRESS column of order 0
    for (order = 0; order < maxorder; ++order) {
        const auto nelems = nequiv[order].size();
        if (const_fix[order].empty()) {
            for (auto &p: const_self[order]) {
                for (const auto &[fst, snd]: p) {
                    if (fst == nelems) {
                        rhs_huang.emplace_back(icount, -snd / huang_rhs_scale);
                    } else {
                        triplets.emplace_back(icount, nshift + fst, snd);
                    }
                }
                ++icount;
            }
        }
        nshift += nelems;
    }

    // Inter-order constraints
    size_t nshift2 = 0;
    for (order = 0; order < maxorder; ++order) {
        if (order > 0) {
            if (const_fix[order - 1].empty() && const_fix[order].empty()) {
                for (auto &p: const_rotation_cross[order]) {
                    for (const auto &[fst, snd]: p) {
                        triplets.emplace_back(icount, nshift2 + fst, snd);
                    }
                    ++icount;
                }
            }
            nshift2 += nequiv[order - 1].size();
        }
    }

    auto nconst_so_far = icount;
    if (fix_harmonic) nconst_so_far += nequiv[0].size();
    if (fix_cubic) nconst_so_far += nequiv[1].size();

    std::vector<double> const_rhs_tmp(nconst_so_far, 0.0);
    for (const auto &[row, val]: rhs_huang) const_rhs_tmp[row] = val;

    if (fix_harmonic) {
        for (const auto &p: const_fix[0]) {
            triplets.emplace_back(icount, p.p_index_target, 1.0);
            const_rhs_tmp[icount] = p.val_to_fix;
            ++icount;
        }
    }

    if (fix_cubic && maxorder > 1) {
        const auto ishift2 = nequiv[0].size();

        for (const auto &p: const_fix[1]) {
            triplets.emplace_back(icount, p.p_index_target + ishift2, 1.0);
            const_rhs_tmp[icount] = p.val_to_fix;
            ++icount;
        }
    }

    Eigen::SparseMatrix<double> const_mat_tmp;
    const_mat_tmp.resize(nconst_so_far, nparams);
    const_mat_tmp.setFromTriplets(triplets.begin(), triplets.end());
    const_mat_tmp.makeCompressed();
    Eigen::VectorXd const_rhs_tmp2 = Eigen::Map<Eigen::VectorXd>(const_rhs_tmp.data(), const_rhs_tmp.size());

    LOG_IF(verbosity, 1, "Constraint matrix is build in sparse format.\n");

    // Translational invariance alone (ICONST = 1) has exact coefficients; skip the near-dependence test.
    int rank, rank_machine;
    select_independent_rows(const_mat_tmp,
                            const_rhs_tmp2,
                            verbosity,
                            tolerance_constraint,
                            impose_inv_R || impose_inv_Huang,
                            const_mat_sparse,
                            const_rhs_vec,
                            rank,
                            rank_machine);

    if (!rhs_huang.empty()) {
        // The row selection above uses C only. With STRESS, also require rank [C | d] == rank C, so that
        // dependent rows with different right-hand sides are not dropped silently. d is rescaled to the
        // magnitude of C (column scaling does not change the rank).
        const auto dmax = const_rhs_tmp2.cwiseAbs().maxCoeff();
        auto cmax = 0.0;
        for (const auto &t: triplets) cmax = std::max(cmax, std::abs(t.value()));
        auto triplets_aug = triplets;
        for (size_t i = 0; i < nconst_so_far; ++i) {
            if (const_rhs_tmp[i] != 0.0) triplets_aug.emplace_back(i, nparams, const_rhs_tmp[i] * cmax / dmax);
        }
        Eigen::SparseMatrix<double> const_mat_aug(nconst_so_far, nparams + 1), mat_dummy;
        const_mat_aug.setFromTriplets(triplets_aug.begin(), triplets_aug.end());
        const_mat_aug.makeCompressed();
        Eigen::VectorXd rhs_dummy;
        int rank_aug;
        // Machine-precision ranks: a TOL_CONST rank of [C | d] is not a consistency test. Rows dropped as
        // nearly dependent are checked against d in select_independent_rows.
        get_independent_rows_lapack_sparse(const_mat_aug,
                                           const_rhs_tmp2,
                                           0,
                                           rank_tolerance_auto,
                                           mat_dummy,
                                           rhs_dummy,
                                           rank_aug);
        if (rank_aug != rank_machine) {
            exit("build_constraint_matrix_sparse",
                 "The Huang constraints with STRESS are inconsistent with the other constraints\n"
                 " (rank [C | d] > rank C). Check STRESS, the cutoff radii, and ICONST.");
        }
    }

    LOG_IF(verbosity, 1, "Reduction of constraint matrix is completed.\n");
}


auto Constraint::build_constraint_matrix_dense() const -> void
{
    const auto num_const = const_mat_sparse.rows();
    const auto num_param = const_mat_sparse.cols();

    deallocate(const_mat);
    deallocate(const_rhs);
    allocate(const_mat, num_const, num_param);
    allocate(const_rhs, num_const);
    if (num_const == 0) return;

    std::fill(const_mat[0], const_mat[0] + static_cast<size_t>(num_const) * num_param, 0.0);
    for (int k = 0; k < const_mat_sparse.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(const_mat_sparse, k); it; ++it) {
            const_mat[it.row()][it.col()] = it.value();
        }
    }
    for (int i = 0; i < num_const; ++i) const_rhs[i] = const_rhs_vec[i];
}


auto Constraint::get_mapping_constraint(const int nmax, const std::vector<size_t> *nequiv,
                                        const ConstraintSparseForm *const_in,
                                        std::vector<ConstraintTypeFix> *const_fix_out,
                                        std::vector<ConstraintTypeRelate> *const_relate_out,
                                        boost::bimap<size_t, size_t> *index_bimap_out,
                                        const std::vector<size_t> *pivots_in) const -> void
{
    // If const_fix_out[order] is not empty as input, it assumes that fix_forceconstant[order] is true.
    // In this case, const_fix_out[order] is not updated.
    // pivots_in[order][k]: the parameter eliminated by row k of const_in[order]; without pivots_in (or
    // if pivots_in[order] is empty), the rows are in reduced row echelon form and the first entry is the pivot.

    int order;
    size_t i;

    std::vector<ConstraintDoubleElement> ConstVec;

    for (order = 0; order < nmax; ++order) {
        if (const_fix_out[order].empty()) {
            size_t p_index_target;
            std::vector<double> alpha_tmp;
            std::vector<size_t> p_index_tmp;

            for (auto irow = const_in[order].size(); irow-- > 0;) {
                const auto p = const_in[order].begin() + static_cast<std::ptrdiff_t>(irow);
                alpha_tmp.clear();
                p_index_tmp.clear();

#ifndef _USE_MAP_FOR_CONSTRAINT
                ConstVec.clear();
                ConstVec.reserve(p->size());
                for (const auto &[fst, snd]: (*p)) {
                    ConstVec.emplace_back(fst, snd);
                }
                std::sort(ConstVec.begin(), ConstVec.end());

                p_index_target = pivots_in && !pivots_in[order].empty() ? pivots_in[order][irow] : ConstVec[0].col;

                for (const auto &elem: ConstVec) {
                    if (elem.col == p_index_target) continue;
                    alpha_tmp.push_back(elem.val);
                    p_index_tmp.push_back(elem.col);
                }

#else
                p_index_target = pivots_in && !pivots_in[order].empty() ? pivots_in[order][irow] : p->begin()->first;
                for (const auto &p2: (*p)) {
                    if (p2.first == p_index_target) continue;
                    alpha_tmp.push_back(p2.second);
                    p_index_tmp.push_back(p2.first);
                }
#endif

                if (p_index_target >= nequiv[order].size()) {
                    exit("get_mapping_constraint",
                         "A constraint has no free FC to eliminate (inconsistent Huang constraints with STRESS).");
                }

                // The STRESS column (index nequiv[order].size(), the largest) is the constant part.
                auto offset = 0.0;
                if (!p_index_tmp.empty() && p_index_tmp.back() == nequiv[order].size()) {
                    offset = -alpha_tmp.back() / huang_rhs_scale;
                    alpha_tmp.pop_back();
                    p_index_tmp.pop_back();
                }

                if (!alpha_tmp.empty()) {
                    const_relate_out[order].emplace_back(p_index_target,
                                                         alpha_tmp,
                                                         p_index_tmp,
                                                         std::vector<int>(p_index_tmp.size(), order),
                                                         offset);
                } else {
                    const_fix_out[order].emplace_back(p_index_target, offset);
                }
            }
        }
    }

    std::vector<int> *has_constraint;
    allocate(has_constraint, nmax);
    size_t nparam;
    for (order = 0; order < nmax; ++order) {
        nparam = nequiv[order].size();
        has_constraint[order].resize(nparam, 0);

        for (i = 0; i < const_fix_out[order].size(); ++i) {
            has_constraint[order][const_fix_out[order][i].p_index_target] = 1;
        }

        for (i = 0; i < const_relate_out[order].size(); ++i) {
            has_constraint[order][const_relate_out[order][i].p_index_target] = 2;
        }
    }
    for (order = 0; order < nmax; ++order) {
        for (const auto &rel: const_relate_out[order]) {
            for (size_t j = 0; j < rel.p_index_orig.size(); ++j) {
                if (has_constraint[rel.order_orig[j]][rel.p_index_orig[j]] != 0) {
                    exit("get_mapping_constraint", "A constraint relates a parameter to a dependent one.");
                }
            }
        }
    }

    for (order = 0; order < nmax; ++order) {
        nparam = nequiv[order].size();

        size_t icount = 0;
        for (i = 0; i < nparam; ++i) {
            if (has_constraint[order][i] == 0) {
                index_bimap_out[order].insert(boost::bimap<size_t, size_t>::value_type(icount, i));
                ++icount;
            }
        }
    }

    deallocate(has_constraint);
}

auto Constraint::get_mapping_constraint_cross_order(const int maxorder, const std::vector<size_t> *nequiv,
                                                    const ReductionAlgo algo, const int verbosity) -> void
{
    // Global elimination over all non-fixed orders including the inter-order rotational
    // constraints. Columns are permuted so that the highest order comes first; the pivots
    // (dependent parameters) are therefore taken from the highest order possible, and
    // higher-order FCs are expressed through lower-order ones (with coord_factorization,
    // whose pivots must also be large, only preferably).
    std::vector<size_t> poff(maxorder, 0);
    for (auto order = maxorder - 2; order >= 0; --order) poff[order] = poff[order + 1] + nequiv[order + 1].size();
    const auto ncols = poff[0] + nequiv[0].size();
    // The STRESS column of order 0 (index nequiv[0].size()) maps to ncols, past all parameters.
    const auto ncols_red = ncols + (huang_rhs ? 1 : 0);

    ConstraintSparseForm const_all;
    MapConstraintElement row;
    for (auto order = 0; order < maxorder; ++order) {
        if (!const_fix[order].empty()) continue;
        for (const auto &p: const_self[order]) {
            row.clear();
            for (const auto &[fst, snd]: p) row[poff[order] + fst] = snd;
            const_all.emplace_back(row);
        }
    }
    for (auto order = 1; order < maxorder; ++order) {
        if (!const_fix[order - 1].empty() || !const_fix[order].empty()) continue;
        // Columns of const_rotation_cross[order]: [0, n_{order-1}) -> order-1, then order.
        const auto nprev = nequiv[order - 1].size();
        for (const auto &p: const_rotation_cross[order]) {
            row.clear();
            for (const auto &[fst, snd]: p) {
                row[fst < nprev ? poff[order - 1] + fst : poff[order] + fst - nprev] = snd;
            }
            const_all.emplace_back(row);
        }
    }

    std::vector<size_t> pivots;
    if (algo == ReductionAlgo::coord_factorization) {
        reduce_for_mapping(ncols,
                           const_all,
                           tolerance_constraint,
                           impose_inv_R || impose_inv_Huang,
                           verbosity,
                           pivots,
                           "get_mapping_constraint_cross_order");
    } else if (algo == ReductionAlgo::spqr_map) {
        rref_sparse_spqr(ncols_red, const_all, tolerance_constraint);
    } else {
        rref_sparse(ncols_red, const_all, tolerance_constraint);
    }

    const auto to_order_index = [&](const size_t col) {
        int order = 0;
        while (col < poff[order]) ++order;
        return std::make_pair(order, col - poff[order]);
    };

    std::vector<ConstraintDoubleElement> ConstVec;
    size_t nreverse = 0;
    for (auto irow = const_all.size(); irow-- > 0;) {
        const auto p = const_all.begin() + static_cast<std::ptrdiff_t>(irow);
        ConstVec.clear();
        for (const auto &[fst, snd]: (*p)) ConstVec.emplace_back(fst, snd);
        std::sort(ConstVec.begin(), ConstVec.end());

        auto offset = 0.0;
        if (ConstVec.back().col == ncols) {
            if (ConstVec.size() == 1) {
                exit("get_mapping_constraint_cross_order",
                     "The Huang constraints with STRESS are inconsistent with the other constraints.\n"
                     " Nearly dependent rotational constraints can cause this; try ICONST = 4 or a\n"
                     " larger TOL_CONST.");
            }
            offset = -ConstVec.back().val / huang_rhs_scale;
            ConstVec.pop_back();
        }

        const auto target = pivots.empty() ? ConstVec[0].col : pivots[irow];
        const auto [order_t, index_t] = to_order_index(target);
        std::vector<double> alpha_tmp;
        std::vector<size_t> p_index_tmp;
        std::vector<int> order_tmp;
        for (const auto &elem: ConstVec) {
            if (elem.col == target) continue;
            const auto [o, idx] = to_order_index(elem.col);
            alpha_tmp.push_back(elem.val);
            p_index_tmp.push_back(idx);
            order_tmp.push_back(o);
        }
        if (std::any_of(order_tmp.begin(), order_tmp.end(), [order_t = order_t](const int o) { return o > order_t; })) {
            ++nreverse;
        }
        if (alpha_tmp.empty()) {
            const_fix[order_t].emplace_back(index_t, offset);
        } else {
            const_relate[order_t].emplace_back(index_t, alpha_tmp, p_index_tmp, order_tmp, offset);
        }
    }

    if (verbosity > 0 && !pivots.empty()) {
        std::cout << "  Relations expressing a lower-order FC through higher-order ones: " << nreverse << "\n\n";
    }

    // Build index_bimap from const_fix / const_relate (no rows left to map).
    const std::vector<ConstraintSparseForm> no_rows(maxorder);
    get_mapping_constraint(maxorder, nequiv, no_rows.data(), const_fix.data(), const_relate.data(), index_bimap);
}

auto Constraint::ready_all_constraints() const -> bool
{
    for (const auto &[fst, snd]: status_constraint_subset) {
        if (snd == 0) return false;
    }

    return true;
}

auto Constraint::set_reduction_algorithm(const int ialgo_reduction) -> void
{
    if (ialgo_reduction == 0) {
        algo_reduction = ReductionAlgo::none;
    } else if (ialgo_reduction == 1) {
        algo_reduction = ReductionAlgo::rref;
    } else if (ialgo_reduction == 2) {
        algo_reduction = ReductionAlgo::qrd;
    } else if (ialgo_reduction == 3) {
        algo_reduction = ReductionAlgo::coord_factorization;
    } else if (ialgo_reduction == 4) {
#ifndef USE_SUITESPARSE_BACKEND
        exit("set_reduction_algorithm", "ALGO_REDUCTION = 4 requires ALM built with USE_SUITESPARSE_BACKEND.");
#endif
        algo_reduction = ReductionAlgo::spqr_map;
    } else {
        exit("set_reduction_algorithm", "unsupported ialgo_reduction");
    }
}

auto Constraint::get_reduction_algorithm() const -> ReductionAlgo
{
    return algo_reduction;
}

auto Constraint::get_constraint_mode() const -> int
{
    return constraint_mode;
}

auto Constraint::set_constraint_mode(const int constraint_mode_in) -> void
{
    constraint_mode = constraint_mode_in;
}

auto Constraint::get_number_of_constraints() const -> size_t
{
    return number_of_constraints;
}

auto Constraint::get_fc_file(const int order) const -> std::string
{
    switch (order) {
    case 2:
        return fc2_file;
    case 3:
        return fc3_file;
    default:
        return "";
    }
}

// Basis of the FC2FIX (order = 2) or FC3FIX (order = 3) file when it is a fixed XML file; empty
// otherwise. HDF5 files store Cartesian IFCs and work with either FCSYM_BASIS.
auto Constraint::get_fc_basis_in_xml(const int order) const -> std::string
{
    const auto &file = order == 2 ? fc2_file : fc3_file;
    if (!(order == 2 ? fix_harmonic : fix_cubic) || file.empty()) return "";
    const auto ext = file.substr(file.find_last_of('.') + 1);
    if (ext != "xml" && ext != "XML") return "";
    using namespace boost::property_tree;
    ptree pt;
    try {
        read_xml(file, pt);
    } catch (std::exception &) {
        return ""; // reported later by parse_forceconstants_from_xml
    }
    const auto basis = get_value_from_xml(pt,
                                          order == 2 ? "Data.ForceConstants.HarmonicUnique.Basis"
                                                     : "Data.ForceConstants.CubicUnique.Basis",
                                          0);
    return basis.empty() ? "Cartesian" : basis;
}

auto Constraint::set_fc_file(const int order, const std::string &fc_file) -> void
{
    switch (order) {
    case 2:
        fc2_file = fc_file;
        break;
    case 3:
        fc3_file = fc_file;
        break;
    default:
        break;
    }
}

auto Constraint::get_fix_harmonic() const -> bool
{
    return fix_harmonic;
}

auto Constraint::set_fix_harmonic(const bool fix_harmonic_in) -> void
{
    fix_harmonic = fix_harmonic_in;
}

auto Constraint::get_fix_cubic() const -> bool
{
    return fix_cubic;
}

auto Constraint::set_fix_cubic(const bool fix_cubic_in) -> void
{
    fix_cubic = fix_cubic_in;
}

auto Constraint::set_constraint_algebraic(const int constraint_algebraic_in) -> void
{
    constraint_algebraic = constraint_algebraic_in;
}

auto Constraint::get_constraint_algebraic() const -> int
{
    return constraint_algebraic;
}

auto Constraint::get_const_mat() const -> double **
{
    if (!const_mat) build_constraint_matrix_dense();
    return const_mat;
}

auto Constraint::get_const_rhs() const -> double *
{
    if (!const_rhs) build_constraint_matrix_dense();
    return const_rhs;
}

auto Constraint::get_const_mat_sparse() const -> const Eigen::SparseMatrix<double> &
{
    return const_mat_sparse;
}

auto Constraint::get_const_rhs_vec() const -> const Eigen::VectorXd &
{
    return const_rhs_vec;
}

auto Constraint::get_tolerance_constraint() const -> double
{
    return tolerance_constraint;
}

auto Constraint::set_tolerance_constraint(const double tol) -> void
{
    tolerance_constraint = tol;
}

auto Constraint::get_exist_constraint() const -> bool
{
    if (constraint_algebraic) {
        if (!const_self.empty()) {
            const auto n = const_self.size();

            for (auto i = 0; i < n; ++i) {
                if (!const_self[i].empty()) return true;
            }
        }

        if (!const_fix.empty()) {
            const auto n = const_fix.size();
            for (auto i = 0; i < n; ++i) {
                if (!const_fix[i].empty()) return true;
            }
        }

        if (!const_relate.empty()) {
            const auto n = const_relate.size();
            for (auto i = 0; i < n; ++i) {
                if (!const_relate[i].empty()) return true;
            }
        }
    } else {
        if (number_of_constraints > 0) return true;
    }

    return false;
}

auto Constraint::get_rotation_axis() const -> std::string
{
    return rotation_axis;
}

auto Constraint::set_rotation_axis(const std::string &rotation_axis_in) -> void
{
    rotation_axis = rotation_axis_in;
}

auto Constraint::get_const_symmetry(const int order) const -> const ConstraintSparseForm &
{
    return const_symmetry[order];
}

auto Constraint::get_const_fix(const int order) const -> const std::vector<ConstraintTypeFix> &
{
    return const_fix[order];
}

auto Constraint::set_const_fix_val_to_fix(const int order, const size_t idx, const double val) -> void
{
    const_fix[order][idx].val_to_fix = val;
}

auto Constraint::get_const_relate(const int order) const -> const std::vector<ConstraintTypeRelate> &
{
    return const_relate[order];
}

auto Constraint::set_const_relate_alpha(const int order, const size_t irel, const size_t j, const double val) -> void
{
    const_relate[order][irel].alpha[j] = val;
}

auto Constraint::set_const_relate_offset(const int order, const size_t irel, const double val) -> void
{
    const_relate[order][irel].offset = val;
}

auto Constraint::set_stress(const Eigen::Matrix3d &stress_in) -> void
{
    stress_input_gpa = stress_in;
    stress_given = true;
}

auto Constraint::unset_stress() -> void
{
    stress_given = false;
}

auto Constraint::setup_huang_stress(const std::unique_ptr<Symmetry> &symmetry, const int verbosity) -> void
{
    // Symmetrize the Cauchy stress: (s + s^T)/2, then average R s R^T over the space-group
    // operations of the supercell (pure translations repeat each point-group element equally).
    const Eigen::Matrix3d sym_part = 0.5 * (stress_input_gpa + stress_input_gpa.transpose());
    Eigen::Matrix3d sigma = Eigen::Matrix3d::Zero();
    const auto &symm = symmetry->get_symmetry_data("super");
    for (const auto &op: symm) {
        if (!(op.rotation_cart.transpose() * op.rotation_cart).isIdentity(1.0e-6)) {
            exit("setup_huang_stress", "Non-orthogonal Cartesian rotation in the symmetry operations.");
        }
        sigma += op.rotation_cart * sym_part * op.rotation_cart.transpose();
    }
    sigma /= static_cast<double>(symm.size());

    const auto norm_in = stress_input_gpa.norm();
    const auto removed = (stress_input_gpa - sigma).norm();
    if (verbosity > 0) {
        const auto saved_flags = std::cout.flags();
        const auto saved_prec = std::cout.precision();
        std::cout << "  STRESS is given: Huang invariance is imposed for a reference structure\n";
        std::cout << "  under the Cauchy stress (GPa, tension positive)\n";
        std::cout << std::scientific << std::setprecision(6);
        for (auto i = 0; i < 3; ++i) {
            std::cout << "   input      :";
            for (auto j = 0; j < 3; ++j) std::cout << std::setw(15) << stress_input_gpa(i, j);
            std::cout << "    symmetrized:";
            for (auto j = 0; j < 3; ++j) std::cout << std::setw(15) << sigma(i, j);
            std::cout << '\n';
        }
        std::cout << "  |input - symmetrized| = " << removed << " GPa\n\n";
        std::cout.flags(saved_flags);
        std::cout.precision(saved_prec);
    }
    if (removed > 1.0e-2 * norm_in) {
        warn("setup_huang_stress",
             "The symmetry-breaking part of STRESS exceeds 1% of |STRESS| and is discarded.\n"
             " Check the stress tensor and the structure.");
    }

    // A hydrostatic stress keeps the Huang rows homogeneous (identical to no STRESS).
    const Eigen::Matrix3d dev = sigma - sigma.trace() / 3.0 * Eigen::Matrix3d::Identity();
    huang_rhs = dev.norm() > 1.0e-10 * sigma.norm();
    if (!huang_rhs && verbosity > 0) {
        std::cout << "  The symmetrized stress is hydrostatic: the Huang constraints are unchanged.\n\n";
    }
    // GPa -> Ry/bohr^3
    const auto bohr_m = Bohr_in_Angstrom * 1.0e-10;
    stress_huang_ry_bohr = sigma * (1.0e9 * bohr_m * bohr_m * bohr_m / Ryd);
}

auto Constraint::get_index_bimap(const int order) const -> const boost::bimap<size_t, size_t> &
{
    return index_bimap[order];
}

auto Constraint::set_constraint_flag(const std::string &const_name, const int use_constraint) -> void
{
    auto it = status_constraint_subset.find(const_name);

    if (it != status_constraint_subset.end()) {
        if (use_constraint == 0) {
            it->second = -1;
        } else {
            it->second = 0;
        }
    } else {
        exit("set_constraint_flag", "Invalid constraint name");
    }
}

auto Constraint::generate_symmetry_constraint(const size_t nat, const std::unique_ptr<Symmetry> &symmetry,
                                              const std::unique_ptr<Cluster> &cluster, const std::unique_ptr<Fcs> &fcs,
                                              const int verbosity, const ReductionAlgo algo_in) -> void
{
    // Create constraint matrices arising from the crystal symmetry.
    // This function clears and updates const_symmetry.

    const auto maxorder = cluster->get_maxorder();
    auto has_constraint_from_symm = false;

    if (fcs->get_forceconstant_basis() == "Cartesian") {
        for (auto isym = 0; isym < symmetry->get_nsym(); ++isym) {
            if (!symmetry->get_symmetry_data()[isym].compatible_with_cartesian) {
                has_constraint_from_symm = true;
                break;
            }
        }
    } else {
        for (auto isym = 0; isym < symmetry->get_nsym(); ++isym) {
            if (!symmetry->get_symmetry_data()[isym].compatible_with_lattice) {
                has_constraint_from_symm = true;
                break;
            }
        }
    }

    has_constraint_from_symm = has_constraint_from_symm & (verbosity > 0);

    if (has_constraint_from_symm) {
        std::cout << "  Generating constraints from crystal symmetry\n";
        if (fcs->get_forceconstant_basis() == "Lattice") {
            std::cout << "  in crystallographic (fractional) coordinates ...\n";
        } else {
            std::cout << "  in Cartesian coordinates ...\n";
        }
    }

    for (auto order = 0; order < maxorder; ++order) {
        if (has_constraint_from_symm) {
            std::cout << "   " << std::setw(8) << cluster->get_ordername(order);
        }

        if (fcs->get_forceconstant_basis() == "Lattice") {
            Fcs::get_constraint_symmetry_in_integer(nat,
                                                    symmetry,
                                                    order,
                                                    fcs->get_forceconstant_basis(),
                                                    fcs->get_fc_table()[order],
                                                    fcs->get_nequiv()[order].size(),
                                                    tolerance_constraint,
                                                    const_symmetry[order],
                                                    algo_in);
        } else {
            Fcs::get_constraint_symmetry(nat,
                                         symmetry,
                                         order,
                                         fcs->get_forceconstant_basis(),
                                         fcs->get_fc_table()[order],
                                         fcs->get_nequiv()[order].size(),
                                         tolerance_constraint,
                                         const_symmetry[order],
                                         algo_in);
        }

        if (has_constraint_from_symm) {
            std::cout << " done.\n";
        }
    }

    if (has_constraint_from_symm) {
        std::cout << "  Finished !\n\n";
    }
}


auto Constraint::generate_translational_constraint(const Cell &supercell, const std::unique_ptr<Symmetry> &symmetry,
                                                   const std::unique_ptr<Cluster> &cluster,
                                                   const std::unique_ptr<Fcs> &fcs, const int periodic_image_conv,
                                                   const int verbosity, const ReductionAlgo algo_in) -> void
{
    // Create constraint matrix for the translational invariance (aka acoustic sum rule).
    const auto maxorder = cluster->get_maxorder();

    if (const_translation.empty()) {
        const_translation.resize(maxorder);
    }

    if (status_constraint_subset["translation"] == -1) return;

    if (verbosity > 0) {
        std::cout << "  Generating constraints for translational invariance ...\n";
    }

    for (auto order = 0; order < maxorder; ++order) {
        if (verbosity > 0) std::cout << "   " << std::setw(8) << cluster->get_ordername(order) << " ...";

        const_translation[order].clear();

        const auto nparams = fcs->get_nequiv()[order].size();

        if (nparams == 0) {
            if (verbosity > 0) std::cout << "  No parameters! Skipped.\n";
            continue;
        }

        if (periodic_image_conv == 0 || order == 0) {
            get_constraint_translation(supercell,
                                       symmetry,
                                       cluster,
                                       fcs,
                                       order,
                                       fcs->get_fc_table()[order],
                                       fcs->get_nequiv()[order].size(),
                                       const_translation[order],
                                       algo_in);
        }
        // make translation constraint for each periodic image combination
        // if periodic_image_conv == 0 or order == 0, there is no need to impose additional ASR constraints.
        else
        {
            // if(periodic_image_conv > 0 && order > 0)
            get_constraint_translation_for_periodic_images(supercell,
                                                           symmetry,
                                                           cluster,
                                                           order,
                                                           fcs->get_fc_table()[order],
                                                           fcs->get_nequiv()[order].size(),
                                                           const_translation[order],
                                                           algo_in);
        }

        if (verbosity > 0) std::cout << " done.\n" << std::flush;
    }
    if (verbosity > 0) std::cout << "  Finished !\n\n";
}


auto Constraint::get_constraint_translation(const Cell &supercell, const std::unique_ptr<Symmetry> &symmetry,
                                            const std::unique_ptr<Cluster> &cluster, const std::unique_ptr<Fcs> &fcs,
                                            const int order, const std::vector<FcProperty> &fc_table,
                                            const size_t nparams, ConstraintSparseForm &const_out,
                                            const ReductionAlgo algo_in) const -> void
{
    // Generate equality constraint for the acoustic sum rule.

    int i, j;
    int iat, jat, icrd, jcrd;
    int idata;
    int loc_nonzero;

    int *ind = nullptr;
    int *intarr = nullptr;
    int *intarr_copy = nullptr;
    int **xyzcomponent = nullptr;

    int ixyz;
    const auto natmin = symmetry->get_nat_trueprim();
    const auto nat = supercell.number_of_atoms;

    unsigned int isize;

    std::vector<int> data;
    std::unordered_set<FcProperty> list_found;
    std::unordered_set<FcProperty>::iterator iter_found;
    std::vector<std::vector<int>> data_vec;
    std::vector<int> const_now;

    using ConstEntry = std::vector<ConstraintIntegerElement>;
    std::vector<ConstEntry> constraint_all;

    ConstEntry const_tmp;

    if (order < 0) return;

    if (nparams == 0) return;

    allocate(ind, order + 2);

    // Create force constant table for search

    list_found.clear();

    for (const auto &p: fc_table) {
        for (i = 0; i < order + 2; ++i) {
            ind[i] = p.elems[i];
        }
        if (list_found.find(FcProperty(order + 2, p.sign, ind, p.mother)) != list_found.end()) {
            exit("get_constraint_translation", "Duplicate interaction list found");
        }
        list_found.insert(FcProperty(order + 2, p.sign, ind, p.mother));
    }

    deallocate(ind);
    ind = nullptr;

    // Generate xyz component for each order

    const auto nxyz = static_cast<int>(std::pow(static_cast<double>(3), order + 1));
    allocate(xyzcomponent, nxyz, order + 1);
    fcs->get_xyzcomponent(order + 1, xyzcomponent);

    allocate(intarr, order + 2);
    allocate(intarr_copy, order + 2);

    const_now.resize(nparams);

    for (i = 0; i < natmin; ++i) {
        iat = symmetry->get_map_trueprim_to_super()[i][0];

        // Generate atom pairs for each order

        if (order == 0) {
            for (icrd = 0; icrd < 3; ++icrd) {
                intarr[0] = 3 * iat + icrd;

                for (jcrd = 0; jcrd < 3; ++jcrd) {
                    // Reset the temporary array for another constraint
                    for (j = 0; j < nparams; ++j) const_now[j] = 0;

                    for (jat = 0; jat < 3 * nat; jat += 3) {
                        intarr[1] = jat + jcrd;

                        iter_found = list_found.find(FcProperty(order + 2, 1.0, intarr, 1));

                        //  If found an IFC
                        if (iter_found != list_found.end()) {
                            // Round the coefficient to integer
                            const_now[iter_found->mother] += nint(iter_found->sign);
                        }
                    }
                    // Add to the constraint list
                    if (!is_allzero(const_now, loc_nonzero)) {
                        if (const_now[loc_nonzero] < 0) {
                            for (j = 0; j < nparams; ++j) const_now[j] *= -1;
                        }
                        const_tmp.clear();
                        for (j = 0; j < nparams; ++j) {
                            if (std::abs(const_now[j]) > 0) {
                                const_tmp.emplace_back(j, const_now[j]);
                            }
                        }
                        constraint_all.emplace_back(const_tmp);
                    }
                }
            }
        } else {
            // Anharmonic cases

            auto intlist(cluster->get_atoms_in_cutoff(order, i));
            std::sort(intlist.begin(), intlist.end());

            data_vec.clear();
            // Generate data_vec that contains possible interacting clusters.
            // Each cluster contains (order + 1) atoms, and the last atom index
            // will be treated separately below.
            CombinationWithRepetition<int> g2(intlist.begin(), intlist.end(), order);
            do {
                data = g2.now();

                intarr[0] = iat;

                for (isize = 0; isize < data.size(); ++isize) {
                    intarr[isize + 1] = data[isize];
                }

                if (cluster->satisfy_nbody_rule(order + 1, intarr, order)) {
                    if (cluster->is_incutoff(order + 1, intarr, order, supercell.kind)) {
                        // Add to list if the atoms interact with each other.
                        data_vec.push_back(data);
                    }
                }
            } while (g2.next());

            const auto ndata = data_vec.size();

            // Use openmp for acceleration if possible
#ifdef _OPENMP
#pragma omp parallel
#endif
            {
                int *intarr_omp = nullptr;
                int *intarr_copy_omp = nullptr;

                allocate(intarr_omp, order + 2);
                allocate(intarr_copy_omp, order + 2);

                std::vector<int> data_omp;
                std::vector<int> const_now_omp;

                ConstEntry const_tmp_omp;
                std::vector<ConstEntry> constraint_list_omp;

                const_now_omp.resize(nparams);
#ifdef _OPENMP
#pragma omp for private(isize, ixyz, jcrd, j, jat, iter_found, loc_nonzero), nowait
#endif
                for (idata = 0; idata < ndata; ++idata) {
                    data_omp = data_vec[idata];

                    intarr_omp[0] = iat;
                    for (isize = 0; isize < data_omp.size(); ++isize) {
                        intarr_omp[isize + 1] = data_omp[isize];
                    }

                    // Loop for xyz component
                    for (ixyz = 0; ixyz < nxyz; ++ixyz) {
                        // Loop for the xyz index of the last atom
                        for (jcrd = 0; jcrd < 3; ++jcrd) {
                            // Reset the temporary array for another constraint
                            for (j = 0; j < nparams; ++j) const_now_omp[j] = 0;

                            // Loop for the last atom index
                            for (jat = 0; jat < 3 * nat; jat += 3) {
                                intarr_omp[order + 1] = jat / 3;

                                if (cluster->satisfy_nbody_rule(order + 2, intarr_omp, order)) {
                                    for (j = 0; j < order + 1; ++j) {
                                        intarr_copy_omp[j] = 3 * intarr_omp[j] + xyzcomponent[ixyz][j];
                                    }
                                    intarr_copy_omp[order + 1] = jat + jcrd;

                                    sort_tail(order + 2, intarr_copy_omp);

                                    iter_found = list_found.find(FcProperty(order + 2, 1.0, intarr_copy_omp, 1));
                                    if (iter_found != list_found.end()) {
                                        const_now_omp[iter_found->mother] += nint(iter_found->sign);
                                    }
                                }
                            } // close loop jat

                            // Add the constraint to the private array
                            if (!is_allzero(const_now_omp, loc_nonzero)) {
                                if (const_now_omp[loc_nonzero] < 0) {
                                    for (j = 0; j < nparams; ++j) const_now_omp[j] *= -1;
                                }

                                const_tmp_omp.clear();
                                for (j = 0; j < nparams; ++j) {
                                    if (std::abs(const_now_omp[j]) > 0) {
                                        const_tmp_omp.emplace_back(j, const_now_omp[j]);
                                    }
                                }
                                if (const_tmp_omp.empty()) {
                                    std::cout << "This cannot happen\n";
                                }
                                constraint_list_omp.emplace_back(const_tmp_omp);
                            }
                        }
                    }
                } // close idata (openmp main loop)

                if (intarr_omp) {
                    deallocate(intarr_omp);
                    intarr_omp = nullptr;
                }
                if (intarr_copy_omp) {
                    deallocate(intarr_copy_omp);
                    intarr_copy_omp = nullptr;
                }

                // Merge vectors
#pragma omp critical
                {
                    for (const auto &it: constraint_list_omp) {
                        constraint_all.emplace_back(it);
                    }
                }
                constraint_list_omp.clear();
            } // close openmp

            intlist.clear();
        } // close if
    } // close loop i

    if (xyzcomponent) {
        deallocate(xyzcomponent);
        xyzcomponent = nullptr;
    }
    if (intarr) {
        deallocate(intarr);
        intarr = nullptr;
    }
    if (intarr_copy) {
        deallocate(intarr_copy);
        intarr_copy = nullptr;
    }

    std::sort(constraint_all.begin(), constraint_all.end());
    constraint_all.erase(std::unique(constraint_all.begin(), constraint_all.end()), constraint_all.end());

    MapConstraintElement const_tmp2;
    auto division_factor = 1.0;
    int counter;
    const_out.clear();

    for (const auto &it: constraint_all) {
        const_tmp2.clear();
        counter = 0;
        for (const auto &it2: it) {
            if (counter == 0) {
                division_factor = 1.0 / it2.val;
            }
            const_tmp2[it2.col] = it2.val * division_factor;
            ++counter;
        }
        const_out.emplace_back(const_tmp2);
    }
    constraint_all.clear();
    if (algo_in == ReductionAlgo::rref) {
        rref_sparse(nparams, const_out, eps8);
    } else if (algo_in == ReductionAlgo::qrd) {
        // verbosity 0: this per-subset reduction has no verbosity in scope and its QR diagnostics are
        // debug noise; the main reduction reporting happens in update_constraint_matrix.
        int rank;
        auto info = get_independent_rows_lapack_sparse(nparams, const_out, 0, rank_tolerance_auto, rank);
    }
}

void Constraint::get_constraint_translation_for_periodic_images(
    const Cell &supercell, const std::unique_ptr<Symmetry> &symmetry, const std::unique_ptr<Cluster> &cluster,
    const int order, const std::vector<FcProperty> &fc_table, const size_t nparams, ConstraintSparseForm &const_out,
    const ReductionAlgo algo_in) const
{
    // Generate equality constraint for the acoustic sum rule.

    int i, j;
    int iat, jat, jcrd;
    int idata;
    int loc_nonzero;

    int *ind;
    int *intarr, *intarr_copy;
    int **xyzcomponent;

    int ixyz;
    const auto natmin = symmetry->get_nat_trueprim();
    const auto nat = supercell.number_of_atoms;

    // generate combinations of periodic images
    //long int n_mirror_images = nint(std::pow(static_cast<double>(27), order));

    unsigned int isize;

    std::vector<int> data;
    std::unordered_set<FcProperty> list_found;
    std::unordered_set<FcProperty>::iterator iter_found;
    std::vector<std::vector<int>> data_vec;
    std::vector<double> const_now;

    using ConstEntry = std::vector<ConstraintDoubleElement>;
    std::vector<ConstEntry> constraint_all;

    ConstEntry const_tmp;

    if (order < 0) return;

    if (nparams == 0) return;

    allocate(ind, order + 2);

    // Create force constant table for search
    list_found.clear();

    for (const auto &p: fc_table) {
        for (i = 0; i < order + 2; ++i) {
            ind[i] = p.elems[i];
        }
        if (list_found.find(FcProperty(order + 2, p.sign, ind, p.mother)) != list_found.end()) {
            exit("get_constraint_translation", "Duplicate interaction list found");
        }
        list_found.insert(FcProperty(order + 2, p.sign, ind, p.mother));
    }

    deallocate(ind);

    // Generate xyz component for each order

    const auto nxyz = static_cast<int>(std::pow(static_cast<double>(3), order + 1));
    allocate(xyzcomponent, nxyz, order + 1);
    Fcs::get_xyzcomponent(order + 1, xyzcomponent);

    allocate(intarr, order + 2);
    allocate(intarr_copy, order + 2);

    const_now.resize(nparams);

    for (i = 0; i < natmin; ++i) {
        iat = symmetry->get_map_trueprim_to_super()[i][0];

        // Generate atom pairs for each order

        if (order == 0) {
            continue; // there is no new translational invariance
        } else {
            // Anharmonic cases

            auto intlist(cluster->get_atoms_in_cutoff(order, i));
            std::sort(intlist.begin(), intlist.end());

            data_vec.clear();
            // Generate data_vec that contains possible interacting clusters.
            // Each cluster contains (order + 1) atoms, and the last atom index
            // will be treated separately below.
            CombinationWithRepetition<int> g2(intlist.begin(), intlist.end(), order);
            do {
                data = g2.now();

                intarr[0] = iat;

                for (isize = 0; isize < data.size(); ++isize) {
                    intarr[isize + 1] = data[isize];
                }

                if (cluster->satisfy_nbody_rule(order + 1, intarr, order)) {
                    if (cluster->is_incutoff(order + 1, intarr, order, supercell.kind)) {
                        // Add to list if the atoms interact with each other.
                        data_vec.push_back(data);
                    }
                }
            } while (g2.next());

            const auto ndata = data_vec.size();

            // Use openmp for acceleration if possible
#ifdef _OPENMP
#pragma omp parallel
#endif
            {
                int *intarr_omp, *intarr_copy_omp;

                double weight;

                allocate(intarr_omp, order + 2);
                allocate(intarr_copy_omp, order + 2);

                std::vector<int> data_omp;
                std::vector<int> atom_tmp;
                std::vector<int> sort_table, sort_table_tmp;
                std::vector<std::vector<double>> consts_now_omp;

                std::vector<long long int> periodic_images_found;

                ConstEntry const_tmp_omp;
                std::vector<ConstEntry> constraint_list_omp;
                long long int i_periodic_images;
                long long int i_tmp, j_tmp, i_tmp2;
                long int i_mi_tmp;

#ifdef _OPENMP
#pragma omp for private(isize, ixyz, jcrd, j, jat, iter_found, loc_nonzero), nowait
#endif
                for (idata = 0; idata < ndata; ++idata) {
                    data_omp = data_vec[idata];

                    intarr_omp[0] = iat;
                    for (isize = 0; isize < data_omp.size(); ++isize) {
                        intarr_omp[isize + 1] = data_omp[isize];
                    }

                    // Loop for xyz component
                    for (ixyz = 0; ixyz < nxyz; ++ixyz) {
                        // Loop for the xyz index of the last atom
                        for (jcrd = 0; jcrd < 3; ++jcrd) {
                            // Reset the temporary array for another constraint
                            consts_now_omp.clear();
                            periodic_images_found.clear();

                            // Loop for the last atom index
                            for (jat = 0; jat < 3 * nat; jat += 3) {
                                intarr_omp[order + 1] = jat / 3;
                                atom_tmp = data_omp;
                                atom_tmp.push_back(jat / 3);
                                // sort atom_tmp and get corresponding sort_table
                                sort_table_tmp.resize(atom_tmp.size());
                                for (i_tmp = 0; i_tmp < atom_tmp.size(); i_tmp++) {
                                    sort_table_tmp[i_tmp] = i_tmp;
                                }
                                for (i_tmp = 0; i_tmp < atom_tmp.size(); i_tmp++) {
                                    for (j_tmp = i_tmp + 1; j_tmp < atom_tmp.size(); j_tmp++) {
                                        if (atom_tmp[i_tmp] > atom_tmp[j_tmp]) {
                                            // swap atom numbers
                                            i_tmp2 = atom_tmp[i_tmp];
                                            atom_tmp[i_tmp] = atom_tmp[j_tmp];
                                            atom_tmp[j_tmp] = i_tmp2;
                                            // write on sort table
                                            i_tmp2 = sort_table_tmp[i_tmp];
                                            sort_table_tmp[i_tmp] = sort_table_tmp[j_tmp];
                                            sort_table_tmp[j_tmp] = i_tmp2;
                                        }
                                    }
                                }
                                // make sort table
                                sort_table.resize(atom_tmp.size());
                                for (i_tmp = 0; i_tmp < atom_tmp.size(); i_tmp++) {
                                    sort_table[sort_table_tmp[i_tmp]] = i_tmp;
                                }

                                if (cluster->satisfy_nbody_rule(order + 2, intarr_omp, order)) {
                                    for (j = 0; j < order + 1; ++j) {
                                        intarr_copy_omp[j] = 3 * intarr_omp[j] + xyzcomponent[ixyz][j];
                                    }
                                    intarr_copy_omp[order + 1] = jat + jcrd;

                                    sort_tail(order + 2, intarr_copy_omp);

                                    iter_found = list_found.find(FcProperty(order + 2, 1.0, intarr_copy_omp, 1));

                                    auto cluster_found =
                                        cluster->get_interaction_cluster(order, i).find(InteractionCluster(atom_tmp));

                                    if (iter_found != list_found.end()) {
                                        if (cluster_found == cluster->get_interaction_cluster(order, i).end()) {
                                            std::cout << "Warning: cluster corresponding to the IFC is NOT found.\n";
                                        } else {
                                            // get weight
                                            weight = 1.0 / static_cast<double>((cluster_found->cell).size());
                                            for (auto cellvec: cluster_found->cell) {
                                                // get number of the combination of the cell
                                                i_periodic_images = 0;

                                                for (i_tmp = 0; i_tmp < order; i_tmp++) {
                                                    i_periodic_images *= 27;
                                                    i_periodic_images += cellvec[sort_table[i_tmp]];
                                                }

                                                // check if the same periodic image has already been found.
                                                for (i_mi_tmp = 0; i_mi_tmp < periodic_images_found.size(); i_mi_tmp++)
                                                {
                                                    if (periodic_images_found[i_mi_tmp] == i_periodic_images) {
                                                        break;
                                                    }
                                                }
                                                // if not found
                                                if (i_mi_tmp == periodic_images_found.size()) {
                                                    periodic_images_found.push_back(i_periodic_images);
                                                    consts_now_omp.push_back(std::vector<double>(nparams, 0.0));
                                                }

                                                // add to the constraint
                                                consts_now_omp[i_mi_tmp][iter_found->mother] +=
                                                    weight * iter_found->sign;
                                            }
                                        }
                                    }
                                }
                            } // close loop jat

                            // Add the constraint to the private array
                            for (i_mi_tmp = 0; i_mi_tmp < periodic_images_found.size(); i_mi_tmp++) {
                                if (!is_allzero(consts_now_omp[i_mi_tmp], eps8, loc_nonzero, 0)) {
                                    if (consts_now_omp[i_mi_tmp][loc_nonzero] < 0) {
                                        for (j = 0; j < nparams; ++j) consts_now_omp[i_mi_tmp][j] *= -1.0;
                                    }

                                    const_tmp_omp.clear();
                                    for (j = 0; j < nparams; ++j) {
                                        if (std::abs(consts_now_omp[i_mi_tmp][j]) > 0) {
                                            const_tmp_omp.emplace_back(j, consts_now_omp[i_mi_tmp][j]);
                                        }
                                    }
                                    if (const_tmp_omp.empty()) {
                                        std::cout << "This cannot happen\n";
                                    }
                                    constraint_list_omp.emplace_back(const_tmp_omp);
                                }
                            }
                        }
                    }
                } // close idata (openmp main loop)

                deallocate(intarr_omp);
                deallocate(intarr_copy_omp);

                // Merge vectors
#pragma omp critical
                {
                    for (const auto &it: constraint_list_omp) {
                        constraint_all.emplace_back(it);
                    }
                }
                constraint_list_omp.clear();
            } // close openmp

            intlist.clear();
        } // close if
    } // close loop i

    deallocate(xyzcomponent);
    deallocate(intarr);
    deallocate(intarr_copy);

    std::sort(constraint_all.begin(), constraint_all.end());
    constraint_all.erase(std::unique(constraint_all.begin(), constraint_all.end()), constraint_all.end());

    MapConstraintElement const_tmp2;
    auto division_factor = 1.0;
    int counter;
    const_out.clear();

    for (const auto &it: constraint_all) {
        const_tmp2.clear();
        counter = 0;
        for (const auto &it2: it) {
            if (counter == 0) {
                division_factor = 1.0 / it2.val;
            }
            const_tmp2[it2.col] = it2.val * division_factor;
            ++counter;
        }
        const_out.emplace_back(const_tmp2);
    }
    constraint_all.clear();
    // if (algo_in == ReductionAlgo::rref) {
    //     rref_sparse(nparams, const_out, eps8);
    // } else if (algo_in == ReductionAlgo::qrd) {
    //     int rank;
    //     auto info = get_independent_rows_lapack_sparse(nparams, const_out, 1, eps12, rank);
    //     std::cout << "rank = " << rank << "\n";
    // }
}

auto Constraint::generate_rotational_constraint(const std::unique_ptr<System> &system,
                                                const std::unique_ptr<Symmetry> &symmetry,
                                                const std::unique_ptr<Cluster> &cluster,
                                                const std::unique_ptr<Fcs> &fcs, const int verbosity,
                                                const double tolerance, const ReductionAlgo algo_in) -> void
{
    // Create constraints for the rotational invariance
    const auto maxorder = cluster->get_maxorder();

    if (const_rotation_self.empty()) {
        const_rotation_self.resize(maxorder);
    }
    if (const_rotation_cross.empty()) {
        const_rotation_cross.resize(maxorder);
    }

    if (status_constraint_subset["rotation"] == -1 and status_constraint_subset["rotation_extra"] == -1) return;


    if (verbosity > 0) std::cout << "  Generating constraints for rotational invariance ...\n";


    int order;
    bool valid_rotation_axis[3][3];
    std::unordered_set<FcProperty> list_found;
    std::unordered_set<FcProperty> list_found_last;

    using ConstEntry = std::vector<ConstraintDoubleElement>;
    std::vector<ConstEntry> *const_self_vec, *const_cross_vec;

    allocate(const_self_vec, maxorder);
    allocate(const_cross_vec, maxorder);

    setup_rotation_axis(valid_rotation_axis);

    std::vector<size_t> nparams;
    nparams.resize(maxorder);

    for (order = 0; order < maxorder; ++order) {
        nparams[order] = fcs->get_nequiv()[order].size();

        const_rotation_self[order].clear();
        const_rotation_cross[order].clear();

        if (order == 0) {
            if (verbosity > 0) {
                std::cout << "   Constraints between " << std::setw(8) << "1st-order IFCs (which are zero) and "
                          << std::setw(8) << cluster->get_ordername(order) << " ...";
            }
        } else {
            if (verbosity > 0) {
                std::cout << "   Constraints between " << std::setw(8) << cluster->get_ordername(order - 1) << " and "
                          << std::setw(8) << cluster->get_ordername(order) << " ...";
            }
        }

        const_self_vec[order].clear();
        const_cross_vec[order].clear();

        if (order > 0) {
            list_found_last = list_found;
        }

        list_found.clear();

        // Accumulate sets of non-zero force constants.
        for (auto p = fcs->get_fc_table()[order].begin(); p != fcs->get_fc_table()[order].end(); ++p) {
            list_found.insert(FcProperty(order + 2, p->sign, p->elems.data(), p->mother));
        }

        set_rotation_constraints(system,
                                 symmetry,
                                 cluster,
                                 fcs,
                                 order,
                                 valid_rotation_axis,
                                 list_found,
                                 list_found_last,
                                 tolerance,
                                 const_self_vec,
                                 const_cross_vec);

        set_rotation_constraints_extra(system,
                                       symmetry,
                                       cluster,
                                       fcs,
                                       order,
                                       valid_rotation_axis,
                                       list_found,
                                       tolerance,
                                       const_self_vec,
                                       const_cross_vec);

        if (verbosity > 0) std::cout << " done.\n" << std::flush;
    } // order

    int counter;
    MapConstraintElement const_copy;
    auto division_factor = 1.0;

    for (order = 0; order < maxorder; ++order) {
        // Sort & unique
        std::sort(const_self_vec[order].begin(), const_self_vec[order].end());
        const_self_vec[order].erase(std::unique(const_self_vec[order].begin(), const_self_vec[order].end()),
                                    const_self_vec[order].end());
        std::sort(const_cross_vec[order].begin(), const_cross_vec[order].end());
        const_cross_vec[order].erase(std::unique(const_cross_vec[order].begin(), const_cross_vec[order].end()),
                                     const_cross_vec[order].end());

        // Copy to the return variable
        for (const auto &it: const_self_vec[order]) {
            const_copy.clear();
            counter = 0;
            for (const auto &it2: it) {
                if (counter == 0) {
                    division_factor = 1.0 / it2.val;
                }
                const_copy[it2.col] = it2.val * division_factor;
                ++counter;
            }
            const_rotation_self[order].emplace_back(const_copy);
        }
        const_self_vec[order].clear();

        for (const auto &it: const_cross_vec[order]) {
            const_copy.clear();
            counter = 0;
            for (const auto &it2: it) {
                if (counter == 0) {
                    division_factor = 1.0 / it2.val;
                }
                const_copy[it2.col] = it2.val * division_factor;
                ++counter;
            }
            const_rotation_cross[order].emplace_back(const_copy);
        }
        const_cross_vec[order].clear();

        //  Perform rref
        if (algo_in == ReductionAlgo::rref) {
            rref_sparse(nparams[order], const_rotation_self[order], eps6);
            if (order > 0) {
                rref_sparse(nparams[order - 1] + nparams[order], const_rotation_cross[order], eps6);
            }
        } else if (algo_in == ReductionAlgo::qrd || algo_in == ReductionAlgo::coord_factorization) {
            // Keep a QR-selected subset of the original rows. An RREF with the absolute eps6 pivot
            // and drop tolerances here adds spurious constraints (e.g., 11 extra of 1641 inter-order
            // rows for Si NORDER = 3), and an RREF of the selected rows alters them by dropping
            // small coefficients. For the algebraic mapping, the later stages build the echelon form.
            // A block too large for the dense QR is reduced by the RREF (coord_factorization only).
            const auto reduce = [&](const size_t ncols, ConstraintSparseForm &rows, const char *what) {
                int rank;
                if (algo_in == ReductionAlgo::qrd || qr_affordable(rows.size(), ncols, verbosity, what)) {
                    get_independent_rows_lapack_sparse(ncols, rows, verbosity, rank_tolerance_auto, rank);
                } else {
                    rref_sparse_pivot(ncols, rows, eps6);
                }
            };
            reduce(nparams[order], const_rotation_self[order], "rotational constraints");
            if (order > 0) {
                reduce(nparams[order - 1] + nparams[order],
                       const_rotation_cross[order],
                       "inter-order rotational constraints");
            }
        }
    }

    if (verbosity > 0) std::cout << "  Finished !\n\n" << std::flush;


    deallocate(const_self_vec);
    deallocate(const_cross_vec);
}

auto Constraint::set_rotation_constraints(const std::unique_ptr<System> &system,
                                          const std::unique_ptr<Symmetry> &symmetry,
                                          const std::unique_ptr<Cluster> &cluster, const std::unique_ptr<Fcs> &fcs,
                                          const int order, const bool valid_rotation_axis[3][3],
                                          const std::unordered_set<FcProperty> &list_found,
                                          const std::unordered_set<FcProperty> &list_found_last, const double tolerance,
                                          std::vector<std::vector<ConstraintDoubleElement>> *const_self_vec,
                                          std::vector<std::vector<ConstraintDoubleElement>> *const_cross_vec) -> void
{
    const auto natmin = symmetry->get_nat_trueprim();
    const auto maxorder = cluster->get_maxorder();

    using ConstEntry = std::vector<ConstraintDoubleElement>;

    std::vector<size_t> nparams;
    for (int i = 0; i < maxorder; ++i) {
        nparams.push_back(fcs->get_nequiv()[i].size());
    }
    const size_t nparam_sub = (order == 0) ? nparams[order] : nparams[order] + nparams[order - 1];

    int nxyz{0};
    int **xyzcomponent = nullptr;

    if (order > 0) {
        nxyz = static_cast<int>(pow(static_cast<double>(3), order));
        allocate(xyzcomponent, nxyz, order);
        Fcs::get_xyzcomponent(order, xyzcomponent);
    }

    // Each atom in the primitive cell is processed independently. Results are kept per atom
    // and appended in atom order afterwards: the caller's sort/unique is not canonical
    // (tolerance-based equality), so the serial order must be preserved for reproducibility.
    std::vector<std::vector<ConstEntry>> self_out(natmin), lower_out(natmin), cross_out(natmin);

#pragma omp parallel
    {
        int iat, jat;
        int icrd;

        CombinationWithRepetition<int> g;
        Eigen::Vector3d vec_for_rot;

        int ixyz;

        int mu_lambda, lambda;
        int levi_factor;

        // Dense accumulator plus the list of touched entries, so that clearing and scanning
        // a candidate costs O(#contributions) instead of O(nparam_sub).
        std::vector<double> arr_constraint;
        std::vector<char> is_touched;
        std::vector<size_t> touched;
        const auto accumulate = [&](const size_t idx, const double val) {
            if (!is_touched[idx]) {
                is_touched[idx] = 1;
                touched.push_back(idx);
            }
            arr_constraint[idx] += val;
        };
        const auto clear_accumulator = [&] {
            for (const auto idx: touched) {
                arr_constraint[idx] = 0.0;
                is_touched[idx] = 0;
            }
            touched.clear();
        };
        // Same as is_allzero restricted to [ibegin, iend), over the sorted touched list.
        const auto first_nonzero = [&](const size_t ibegin, const size_t iend) -> long {
            for (const auto idx: touched) {
                if (idx >= ibegin && idx < iend && std::abs(arr_constraint[idx]) > tolerance) return idx;
            }
            return -1;
        };
        // Append entries in [ibegin, iend) with |value| >= tolerance, column-shifted by -ibegin.
        const auto append_entries = [&](const size_t ibegin, const size_t iend, ConstEntry &out) {
            for (const auto idx: touched) {
                if (idx >= ibegin && idx < iend && std::abs(arr_constraint[idx]) >= tolerance) {
                    out.emplace_back(idx - ibegin, arr_constraint[idx]);
                }
            }
        };

        std::vector<int> atom_tmp;

        std::vector<int> interaction_list;


        std::vector<int> interaction_index, interaction_atom, interaction_tmp;
        interaction_index.resize(order + 2);
        interaction_atom.resize(order + 2);
        interaction_tmp.resize(order + 2);

        ConstEntry const_tmp;

        arr_constraint.assign(nparam_sub, 0.0);
        is_touched.assign(nparam_sub, 0);

#pragma omp for schedule(dynamic)
        for (int i = 0; i < natmin; ++i) {
            iat = symmetry->get_map_trueprim_to_super()[i][0];

            interaction_atom[0] = iat;

            if (order == 0) {
                auto interaction_list_now(cluster->get_atoms_in_cutoff(order, i));
                std::sort(interaction_list_now.begin(), interaction_list_now.end());

                // Special treatment for harmonic force constants

                for (icrd = 0; icrd < 3; ++icrd) {
                    interaction_index[0] = 3 * iat + icrd;

                    for (int mu = 0; mu < 3; ++mu) {
                        for (int nu = 0; nu < 3; ++nu) {
                            if (!valid_rotation_axis[mu][nu]) continue;

                            // Clear history

                            clear_accumulator();

                            for (auto &iter_list: interaction_list_now) {
                                jat = iter_list;


                                atom_tmp.clear();
                                atom_tmp.push_back(jat);
                                const auto iter_cluster =
                                    cluster->get_interaction_cluster(order, i).find(InteractionCluster(atom_tmp));

                                // Compute vec_for_rot
                                vec_for_rot.setZero();
                                if (iter_cluster != cluster->get_interaction_cluster(order, i).end()) {
                                    const auto nsize_equiv = iter_cluster->cell.size();
                                    for (int j = 0; j < nsize_equiv; ++j) {
                                        for (auto k = 0; k < 3; ++k) {
                                            vec_for_rot[k] += system->get_x_image()[iter_cluster->cell[j][0]](jat, k);
                                        }
                                    }
                                    // Take average
                                    vec_for_rot /= static_cast<double>(nsize_equiv);
                                }

                                interaction_index[1] = 3 * jat + mu;
                                auto iter_found = list_found.find(FcProperty(order + 2, 1.0, &interaction_index[0], 1));

                                if (iter_found != list_found.end()) {
                                    accumulate(iter_found->mother, iter_found->sign * vec_for_rot[nu]);
                                }

                                // Exchange mu <--> nu and repeat.
                                // Note that the sign is inverted (+ --> -) in the summation

                                interaction_index[1] = 3 * jat + nu;
                                iter_found = list_found.find(FcProperty(order + 2, 1.0, &interaction_index[0], 1));
                                if (iter_found != list_found.end()) {
                                    accumulate(iter_found->mother, -iter_found->sign * vec_for_rot[mu]);
                                }
                            }

                            std::sort(touched.begin(), touched.end());
                            const auto loc_nonzero = first_nonzero(0, nparam_sub);
                            if (loc_nonzero >= 0) {
                                // Add to constraint list
                                if (arr_constraint[loc_nonzero] < 0.0) {
                                    for (const auto idx: touched) arr_constraint[idx] *= -1.0;
                                }
                                const_tmp.clear();
                                append_entries(0, nparam_sub, const_tmp);
                                self_out[i].emplace_back(const_tmp);
                            }
                        } // nu
                    } // mu
                }
            } else {
                // Constraint between different orders

                auto interaction_list_now(cluster->get_atoms_in_cutoff(order, i));
                auto interaction_list_old(cluster->get_atoms_in_cutoff(order - 1, i));
                std::sort(interaction_list_now.begin(), interaction_list_now.end());
                std::sort(interaction_list_old.begin(), interaction_list_old.end());

                for (icrd = 0; icrd < 3; ++icrd) {
                    interaction_index[0] = 3 * iat + icrd;

                    const CombinationWithRepetition<int> g_now(interaction_list_now.begin(),
                                                               interaction_list_now.end(),
                                                               order);
                    const CombinationWithRepetition<int> g_old(interaction_list_old.begin(),
                                                               interaction_list_old.end(),
                                                               order);

                    // m    -th order --> (m-1)-th order
                    // (m-1)-th order -->     m-th order
                    // 2-different directions to find all constraints

                    for (unsigned int direction = 0; direction < 2; ++direction) {
                        if (direction == 0) {
                            g = g_now;
                            interaction_list = interaction_list_now;
                        } else {
                            g = g_old;
                            interaction_list = interaction_list_old;
                        }

                        // Loop for the interacting pairs

                        do {
                            auto data = g.now();

                            for (size_t idata = 0; idata < data.size(); ++idata) {
                                interaction_atom[idata + 1] = data[idata];
                            }

                            // Every FC looked up below involves the atoms {iat, data...}; if this set
                            // already violates NBODY for both orders, no lookup can succeed.
                            if (!cluster->satisfy_nbody_rule(order + 1, &interaction_atom[0], order) &&
                                !cluster->satisfy_nbody_rule(order + 1, &interaction_atom[0], order - 1))
                            {
                                continue;
                            }

                            for (ixyz = 0; ixyz < nxyz; ++ixyz) {
                                for (int j = 0; j < order; ++j)
                                    interaction_index[j + 1] = 3 * interaction_atom[j + 1] + xyzcomponent[ixyz][j];

                                for (int mu = 0; mu < 3; ++mu) {
                                    for (int nu = 0; nu < 3; ++nu) {
                                        if (!valid_rotation_axis[mu][nu]) continue;

                                        // Search for a new constraint below

                                        clear_accumulator();

                                        // Loop for m_{N+1}, a_{N+1}
                                        for (auto &iter_list: interaction_list) {
                                            jat = iter_list;

                                            interaction_atom[order + 1] = jat;
                                            if (!cluster->is_incutoff(order + 2,
                                                                      &interaction_atom[0],
                                                                      order,
                                                                      system->get_supercell().kind))
                                            {
                                                continue;
                                            }

                                            atom_tmp.clear();

                                            for (int j = 1; j < order + 2; ++j) {
                                                atom_tmp.push_back(interaction_atom[j]);
                                            }
                                            std::sort(atom_tmp.begin(), atom_tmp.end());

                                            const auto iter_cluster = cluster->get_interaction_cluster(order, i).find(
                                                InteractionCluster(atom_tmp));
                                            if (iter_cluster != cluster->get_interaction_cluster(order, i).end()) {
                                                int iloc = -1;

                                                for (int j = 0; j < atom_tmp.size(); ++j) {
                                                    if (atom_tmp[j] == jat) {
                                                        iloc = j;
                                                        break;
                                                    }
                                                }

                                                if (iloc == -1) {
                                                    exit("generate_rotational_constraint", "This cannot happen.");
                                                }

                                                for (int j = 0; j < 3; ++j) vec_for_rot[j] = 0.0;

                                                const auto nsize_equiv = (*iter_cluster).cell.size();

                                                for (int j = 0; j < nsize_equiv; ++j) {
                                                    for (auto k = 0; k < 3; ++k) {
                                                        vec_for_rot[k] +=
                                                            system->get_x_image()[(*iter_cluster).cell[j][iloc]](jat,
                                                                                                                 k);
                                                    }
                                                }

                                                for (int j = 0; j < 3; ++j) {
                                                    vec_for_rot[j] /= static_cast<double>(nsize_equiv);
                                                }
                                            }


                                            // sum_{jat} (Phi_{..nu}(..;jat) r_mu(jat) - Phi_{..mu}(..;jat) r_nu(jat))
                                            // + (Levi-Civita terms of the "order-1"th order) = 0,
                                            // see docs/source/almdir/formalism_alm.rst.

                                            interaction_index[order + 1] = 3 * jat + mu;
                                            for (int j = 0; j < order + 2; ++j)
                                                interaction_tmp[j] = interaction_index[j];

                                            sort_tail(order + 2, interaction_tmp.data());

                                            auto iter_found =
                                                list_found.find(FcProperty(order + 2, 1.0, interaction_tmp.data(), 1));
                                            if (iter_found != list_found.end()) {
                                                accumulate(nparams[order - 1] + iter_found->mother,
                                                           -iter_found->sign * vec_for_rot[nu]);
                                            }

                                            // Exchange mu <--> nu and repeat again.

                                            interaction_index[order + 1] = 3 * jat + nu;
                                            for (int j = 0; j < order + 2; ++j)
                                                interaction_tmp[j] = interaction_index[j];

                                            sort_tail(order + 2, &interaction_tmp[0]);

                                            iter_found =
                                                list_found.find(FcProperty(order + 2, 1.0, &interaction_tmp[0], 1));
                                            if (iter_found != list_found.end()) {
                                                accumulate(nparams[order - 1] + iter_found->mother,
                                                           iter_found->sign * vec_for_rot[mu]);
                                            }
                                        }

                                        for (lambda = 0; lambda < order + 1; ++lambda) {
                                            mu_lambda = interaction_index[lambda] % 3;

                                            for (int jcrd = 0; jcrd < 3; ++jcrd) {
                                                for (int j = 0; j < order + 1; ++j) {
                                                    interaction_tmp[j] = interaction_index[j];
                                                }

                                                interaction_tmp[lambda] = 3 * interaction_atom[lambda] + jcrd;

                                                levi_factor = 0;

                                                for (int j = 0; j < 3; ++j) {
                                                    levi_factor +=
                                                        levi_civita(j, mu, nu) * levi_civita(j, mu_lambda, jcrd);
                                                }

                                                if (levi_factor == 0) continue;

                                                sort_tail(order + 1, &interaction_tmp[0]);

                                                auto iter_found = list_found_last.find(
                                                    FcProperty(order + 1, 1.0, &interaction_tmp[0], 1));
                                                if (iter_found != list_found_last.end()) {
                                                    accumulate(iter_found->mother,
                                                               iter_found->sign * static_cast<double>(levi_factor));
                                                }
                                            }
                                        }

                                        std::sort(touched.begin(), touched.end());
                                        const auto loc_nonzero = first_nonzero(0, nparam_sub);
                                        if (loc_nonzero >= 0) {
                                            // A Candidate for another constraint found !
                                            // Add to the appropriate set

                                            if (arr_constraint[loc_nonzero] < 0.0) {
                                                for (const auto idx: touched) arr_constraint[idx] *= -1.0;
                                            }
                                            const auto nlower = nparams[order - 1];

                                            const_tmp.clear();

                                            if (first_nonzero(nlower, nparam_sub) < 0) {
                                                // If all elements of the "order"th order is zero,
                                                // the constraint is intraorder of the "order-1"th order.
                                                append_entries(0, nlower, const_tmp);
                                                lower_out[i].emplace_back(const_tmp);
                                            } else if (first_nonzero(0, nlower) < 0) {
                                                // If all elements of the "order-1"th order is zero,
                                                // the constraint is intraorder of the "order"th order.
                                                append_entries(nlower, nparam_sub, const_tmp);
                                                self_out[i].emplace_back(const_tmp);
                                            } else {
                                                // If nonzero elements exist in both of the "order-1" and "order",
                                                // the constraint is intrerorder.
                                                append_entries(0, nparam_sub, const_tmp);
                                                cross_out[i].emplace_back(const_tmp);
                                            }
                                        }
                                    } // nu
                                } // mu
                            } // ixyz
                        } while (g.next());
                    } // direction
                } // icrd
            }
        } // iat
    } // omp parallel

    for (int i = 0; i < natmin; ++i) {
        for (auto &c: self_out[i]) const_self_vec[order].emplace_back(std::move(c));
        if (order > 0) {
            for (auto &c: lower_out[i]) const_self_vec[order - 1].emplace_back(std::move(c));
            for (auto &c: cross_out[i]) const_cross_vec[order].emplace_back(std::move(c));
        }
    }

    if (xyzcomponent) deallocate(xyzcomponent);

    status_constraint_subset["rotation"] = 1;
}

auto Constraint::set_rotation_constraints_extra(
    const std::unique_ptr<System> &system, const std::unique_ptr<Symmetry> &symmetry,
    const std::unique_ptr<Cluster> &cluster, const std::unique_ptr<Fcs> &fcs, const int order,
    const bool valid_rotation_axis[3][3], const std::unordered_set<FcProperty> &list_found, const double tolerance,
    std::vector<std::vector<ConstraintDoubleElement>> *const_self_vec,
    std::vector<std::vector<ConstraintDoubleElement>> *const_cross_vec) -> void
{
    const auto maxorder = cluster->get_maxorder();

    // Additional constraint for the last order.
    // All IFCs over maxorder-th order are neglected.
    if (order != (maxorder - 1) or status_constraint_subset["rotation_extra"] == -1) return;

    int j;
    const auto natmin = symmetry->get_nat_trueprim();

    int iat;
    int icrd;
    int mu, nu;

    CombinationWithRepetition<int> g;

    int ixyz, nxyz{0};
    int loc_nonzero;

    int mu_lambda, lambda;
    int levi_factor;

    std::vector<double> arr_constraint;
    std::vector<double> arr_constraint_self;

    std::vector<int> atom_tmp;
    std::set<InteractionCluster>::iterator iter_cluster;

    std::vector<size_t> nparams;
    std::vector<int> interaction_index, interaction_atom, interaction_tmp;

    interaction_index.resize(order + 2);
    interaction_atom.resize(order + 2);
    interaction_tmp.resize(order + 2);

    typedef std::vector<ConstraintDoubleElement> ConstEntry;
    ConstEntry const_tmp;

    for (int i = 0; i < maxorder; ++i) {
        nparams.push_back(fcs->get_nequiv()[i].size());
    }

    arr_constraint_self.resize(nparams[order]);

    int **xyzcomponent = nullptr;

    nxyz = static_cast<int>(pow(static_cast<double>(3), order + 1));
    allocate(xyzcomponent, nxyz, order + 1);
    Fcs::get_xyzcomponent(order + 1, xyzcomponent);


    for (int i = 0; i < natmin; ++i) {
        iat = symmetry->get_map_trueprim_to_super()[i][0];

        interaction_atom[0] = iat;

        auto interaction_list_now(cluster->get_atoms_in_cutoff(order, i));
        std::sort(interaction_list_now.begin(), interaction_list_now.end());

        for (icrd = 0; icrd < 3; ++icrd) {
            interaction_index[0] = 3 * interaction_atom[0] + icrd;

            CombinationWithRepetition<int> g_now(interaction_list_now.begin(), interaction_list_now.end(), order + 1);
            do {
                auto data = g_now.now();

                for (auto idata = 0; idata < data.size(); ++idata) interaction_atom[idata + 1] = data[idata];

                for (ixyz = 0; ixyz < nxyz; ++ixyz) {
                    for (j = 0; j < order + 1; ++j)
                        interaction_index[j + 1] = 3 * interaction_atom[j + 1] + xyzcomponent[ixyz][j];

                    for (mu = 0; mu < 3; ++mu) {
                        for (nu = 0; nu < 3; ++nu) {
                            if (!valid_rotation_axis[mu][nu]) continue;

                            for (j = 0; j < nparams[order]; ++j) arr_constraint_self[j] = 0.0;

                            for (lambda = 0; lambda < order + 2; ++lambda) {
                                mu_lambda = interaction_index[lambda] % 3;

                                for (int jcrd = 0; jcrd < 3; ++jcrd) {
                                    for (j = 0; j < order + 2; ++j) interaction_tmp[j] = interaction_index[j];

                                    interaction_tmp[lambda] = 3 * interaction_atom[lambda] + jcrd;

                                    levi_factor = 0;
                                    for (j = 0; j < 3; ++j) {
                                        levi_factor += levi_civita(j, mu, nu) * levi_civita(j, mu_lambda, jcrd);
                                    }

                                    if (levi_factor == 0) continue;

                                    sort_tail(order + 2, &interaction_tmp[0]);

                                    auto iter_found =
                                        list_found.find(FcProperty(order + 2, 1.0, &interaction_tmp[0], 1));
                                    if (iter_found != list_found.end()) {
                                        arr_constraint_self[(*iter_found).mother] +=
                                            (*iter_found).sign * static_cast<double>(levi_factor);
                                    }
                                } // jcrd
                            } // lambda

                            if (!is_allzero(arr_constraint_self, tolerance, loc_nonzero)) {
                                if (arr_constraint_self[loc_nonzero] < 0.0) {
                                    for (j = 0; j < nparams[order]; ++j) arr_constraint_self[j] *= -1.0;
                                }
                                const_tmp.clear();
                                for (j = 0; j < nparams[order]; ++j) {
                                    if (std::abs(arr_constraint_self[j]) >= tolerance) {
                                        const_tmp.emplace_back(j, arr_constraint_self[j]);
                                    }
                                }
                                const_self_vec[order].emplace_back(const_tmp);
                            }
                        } // nu
                    } // mu
                } // ixyz
            } while (g_now.next());
        } // icrd
    } // iat
    deallocate(xyzcomponent);
}


auto Constraint::levi_civita(const int i, const int j, const int k) -> int
{
    return (j - i) * (k - i) * (k - j) / 2;
}


auto Constraint::setup_rotation_axis(bool flag[3][3]) -> void
{
    for (auto mu = 0; mu < 3; ++mu) {
        for (auto nu = 0; nu < 3; ++nu) {
            if (mu == nu) {
                flag[mu][nu] = false;
            } else {
                flag[mu][nu] = true;
            }
        }
    }
    std::sort(rotation_axis.begin(), rotation_axis.end());

    if (rotation_axis == "x") {
        flag[0][1] = false;
        flag[1][0] = false;
        flag[0][2] = false;
        flag[2][0] = false;
    } else if (rotation_axis == "y") {
        flag[0][1] = false;
        flag[1][0] = false;
        flag[1][2] = false;
        flag[2][1] = false;
    } else if (rotation_axis == "z") {
        flag[0][2] = false;
        flag[2][0] = false;
        flag[1][2] = false;
        flag[2][1] = false;
    } else if (rotation_axis == "xy") {
        flag[0][1] = false;
        flag[1][0] = false;
    } else if (rotation_axis == "yz") {
        flag[1][2] = false;
        flag[2][1] = false;
    } else if (rotation_axis == "xz") {
        flag[0][2] = false;
        flag[2][0] = false;
    } else if (rotation_axis == "xyz") {
        // do nothing
    } else {
        exit("setup_rotation_axis", "Invalid ROTAXIS. Use a combination of x, y and z (e.g. xyz).");
    }
}


auto Constraint::get_forceconstants_from_file(const int order, const std::unique_ptr<Symmetry> &symmetry,
                                              const std::unique_ptr<Fcs> &fcs, const std::string &file_to_fix,
                                              std::vector<std::vector<int>> &intpair_fcs,
                                              std::vector<double> &fcs_values) -> void
{
    const auto file_extension = file_to_fix.substr(file_to_fix.find_last_of('.') + 1);

    if (file_extension == "xml" || file_extension == "XML") {
        parse_forceconstants_from_xml(order, symmetry, fcs, file_to_fix, intpair_fcs, fcs_values);
    } else if (file_extension == "h5" || file_extension == "hdf5") {
        check_h5_readable(file_to_fix, order == 0 ? "FC2FIX" : "FC3FIX");
        parse_forceconstants_from_h5(order, symmetry, fcs, file_to_fix, intpair_fcs, fcs_values);
    } else {
        exit("get_forceconstants_from_file", "unacceptable extension type");
    }
}

auto Constraint::generate_huang_constraint(const Cell &supercell, const std::unique_ptr<Symmetry> &symmetry,
                                           const std::unique_ptr<Cluster> &cluster, const std::unique_ptr<Fcs> &fcs,
                                           const std::vector<Eigen::MatrixXd> &x_image, const int verbosity,
                                           const ReductionAlgo algo_in) -> void
{
    // Create constraint matrix for the Huang constraints.

    if (const_huang.empty()) {
        const_huang.resize(1);
    }

    if (status_constraint_subset["huang"] == -1) return;

    if (verbosity > 0) {
        std::cout << "  Generating constraints for Huang invariance ...";
    }
    const_huang[0].clear();
    if (verbosity > 0) std::cout << " done.\n" << std::flush;
    const auto nparams = fcs->get_nequiv()[0].size();
    if (nparams == 0) {
        if (verbosity > 0) std::cout << "  No parameters! Skipped.\n";
        return;
    }
    std::unordered_set<FcProperty> list_found;

    list_found.clear();

    // Accumulate sets of non-zero force constants.
    for (auto &p: fcs->get_fc_table()[0]) {
        list_found.insert(FcProperty(2, p.sign, p.elems.data(), p.mother));
    }


    const auto natmin = symmetry->get_nat_trueprim();
    const auto nat = supercell.number_of_atoms;
    // Supercell index of each true-primitive atom (the first atom of the FC2 entries in fc_table).
    std::vector<int> map_tp2s(natmin);
    for (size_t iat = 0; iat < natmin; ++iat) map_tp2s[iat] = symmetry->get_map_trueprim_to_super()[iat][0];

    int pair1[2], pair2[2];
    std::vector<int> atom_tmp;
    std::vector<std::vector<Eigen::Matrix3d>> relvec_tensor;
    Eigen::Vector3d vec_tmp;
    std::vector<double> const_now;

    relvec_tensor.resize(natmin, std::vector<Eigen::Matrix3d>(nat));
    using ConstEntry = std::vector<ConstraintDoubleElement>;
    ConstEntry const_tmp;
    std::vector<ConstEntry> const_list;

    // Construct relative vector products considering periodic images
    for (size_t iat = 0; iat < natmin; ++iat) {
        const auto iat_s = symmetry->get_map_trueprim_to_super()[iat][0];

        for (size_t jat = 0; jat < nat; ++jat) {
            atom_tmp.clear();
            atom_tmp.push_back(jat);

            relvec_tensor[iat][jat].setZero();

            const auto iter_cluster = cluster->get_interaction_cluster(0, iat).find(InteractionCluster(atom_tmp));

            if (iter_cluster != cluster->get_interaction_cluster(0, iat).end()) {
                const auto nsize_equiv = iter_cluster->cell.size();
                // const auto nsize_equiv = 1;
                for (int j = 0; j < nsize_equiv; ++j) {
                    for (auto k = 0; k < 3; ++k) {
                        vec_tmp[k] = x_image[iter_cluster->cell[j][0]](jat, k) - x_image[0](iat_s, k);
                    }
                    for (auto k = 0; k < 3; ++k) {
                        for (auto m = 0; m < 3; ++m) {
                            relvec_tensor[iat][jat](k, m) += vec_tmp[k] * vec_tmp[m];
                        }
                    }
                }
                relvec_tensor[iat][jat] /= static_cast<double>(nsize_equiv);
            }
        }
    }

    const_now.resize(nparams);
    const_list.clear();

    int loc_nonzero;

    // With STRESS, each row reads L = 2 Omega (sigma_mu1nu1 delta_mu2nu2 - sigma_mu2nu2 delta_mu1nu1),
    // Omega the true primitive cell volume; -rhs is stored at column nparams.
    const auto omega = supercell.volume * static_cast<double>(natmin) / static_cast<double>(nat);
    const auto rhs_max = huang_rhs ? 2.0 * omega * stress_huang_ry_bohr.cwiseAbs().maxCoeff() : 0.0; // bound of |rhs|
    if (huang_rhs) huang_rhs_scale = 1.0e4 / rhs_max;

    for (size_t mu1 = 0; mu1 < 3; ++mu1) {
        for (size_t nu1 = 0; nu1 < 3; ++nu1) {
            for (size_t mu2 = 0; mu2 < 3; ++mu2) {
                for (size_t nu2 = 0; nu2 < 3; ++nu2) {

                    if (mu1 == mu2 && nu1 == nu2) {
                        // Skip the case where mu1 == mu2 and nu1 == nu2
                        continue;
                    }

                    // Reset the temporary array for the current constraint
                    for (size_t j = 0; j < nparams; ++j) {
                        const_now[j] = 0.0;
                    }

                    for (size_t iat = 0; iat < natmin; ++iat) {
                        pair1[0] = 3 * map_tp2s[iat] + mu1; // (0k1;alpha)
                        pair2[0] = 3 * map_tp2s[iat] + mu2; // (0k1;gamma)

                        for (size_t jat = 0; jat < nat; ++jat) {
                            pair1[1] = 3 * jat + nu1; // (l2k2;beta)
                            pair2[1] = 3 * jat + nu2; // (l2k2;delta)

                            auto iter_found = list_found.find(FcProperty(2, 1.0, &pair1[0], 1));

                            auto iter_found2 = list_found.find(FcProperty(2, 1.0, &pair2[0], 1));

                            if (iter_found != list_found.end()) {
                                const_now[iter_found->mother] += iter_found->sign * relvec_tensor[iat][jat](mu2, nu2);
                            }
                            if (iter_found2 != list_found.end()) {
                                const_now[iter_found2->mother] -= iter_found2->sign * relvec_tensor[iat][jat](mu1, nu1);
                            }
                        }
                    }

                    auto rhs = 0.0;
                    if (huang_rhs) {
                        rhs = 2.0 * omega *
                              (stress_huang_ry_bohr(mu1, nu1) * static_cast<double>(mu2 == nu2) -
                               stress_huang_ry_bohr(mu2, nu2) * static_cast<double>(mu1 == nu1));
                    }

                    const_tmp.clear();
                    if (!is_allzero(const_now, eps8, loc_nonzero, 0)) {
                        if (const_now[loc_nonzero] < 0.0) {
                            for (size_t j = 0; j < nparams; ++j) {
                                const_now[j] *= -1.0;
                            }
                            rhs *= -1.0;
                        }
                        for (size_t j = 0; j < nparams; ++j) {
                            if (std::abs(const_now[j]) >= eps8) {
                                const_tmp.emplace_back(j, const_now[j]);
                            }
                        }
                        if (rhs != 0.0) const_tmp.emplace_back(nparams, -rhs * huang_rhs_scale);
                        const_list.emplace_back(const_tmp);
                    } else if (std::abs(rhs) > eps8 * rhs_max) {
                        const std::string xyz = "xyz";
                        const auto msg = "The Huang constraint [" + xyz.substr(mu1, 1) + xyz.substr(nu1, 1) + "," +
                                         xyz.substr(mu2, 1) + xyz.substr(nu2, 1) +
                                         "] has no harmonic force constants but a nonzero STRESS term,\n"
                                         " so it cannot be satisfied. Increase the harmonic cutoff radius.";
                        exit("generate_huang_constraint", msg.c_str());
                    }
                }
            }
        }
    }
    MapConstraintElement const_copy;

    auto division_factor = 1.0;
    for (const auto &it: const_list) {
        auto counter = 0;
        const_copy.clear();
        for (const auto &it2: it) {
            if (counter == 0) {
                division_factor = 1.0 / it2.val;
            }
            const_copy[it2.col] = it2.val * division_factor;
            ++counter;
        }
        const_huang[0].emplace_back(const_copy);
    }

    const auto ncols = nparams + (huang_rhs ? 1 : 0);
    if (algo_in == ReductionAlgo::rref) {
        rref_sparse(ncols, const_huang[0], eps8);
    } else if (algo_in == ReductionAlgo::qrd) {
        int rank;
        auto info = get_independent_rows_lapack_sparse(ncols, const_huang[0], verbosity, rank_tolerance_auto, rank);
    }
    if (huang_rhs) check_huang_rhs_rows(const_huang[0], nparams, "generate_huang_constraint");
}

auto Constraint::parse_forceconstants_from_xml(const int order, const std::unique_ptr<Symmetry> &symmetry,
                                               const std::unique_ptr<Fcs> &fcs, const std::string &file_to_fix,
                                               std::vector<std::vector<int>> &intpair_fcs,
                                               std::vector<double> &fcs_values) -> void
{
    using namespace boost::property_tree;
    ptree pt;

    try {
        read_xml(file_to_fix, pt);
    } catch (std::exception &e) {
        std::string str_error;
        if (order == 0) {
            str_error = "Cannot open file FC2FIX ( " + file_to_fix + " )";
        } else if (order == 1) {
            str_error = "Cannot open file FC3FIX ( " + file_to_fix + " )";
        }
        exit("fix_forceconstants_to_file", str_error.c_str());
    }

    const auto version_from_file = get_value_from_xml(pt, "Data.ALM_version");

    std::vector<std::string> version_array;
    boost::split(version_array, version_from_file, boost::is_any_of("."));
    std::vector<int> version_array_int;
    for (const auto &it: version_array) {
        version_array_int.emplace_back(std::stoi(it));
    }
    if ((version_array_int[0] <= 1) && (version_array_int[1] <= 4)) {
        exit("fit_forceconstants_to_file",
             "FCSXML files generated by older versions (<=1.4.2) do not have compatibility with\n"
             " a newer version for fixing force constants. Please use a newer version (>=1.5) and"
             " regenerate the FC2FIX or FC3FIX files");
    }

    const auto nat_ref = boost::lexical_cast<size_t>(get_value_from_xml(pt, "Data.Structure.NumberOfAtoms"));
    const auto ntran_ref = boost::lexical_cast<size_t>(get_value_from_xml(pt, "Data.Symmetry.NumberOfTranslations"));
    const auto natmin_ref = nat_ref / ntran_ref;

    if (natmin_ref != symmetry->get_nat_trueprim()) {
        exit("fix_forceconstants_to_file", "The number of atoms in the primitive cell is not consistent.");
    }

    const auto nfcs = fcs->get_nequiv()[order].size();

    if (order == 0) {
        const auto nfcs_ref =
            boost::lexical_cast<size_t>(get_value_from_xml(pt, "Data.ForceConstants.HarmonicUnique.NFC2"));

        if (nfcs_ref != nfcs) {
            exit("fix_forceconstants_to_file", "The number of harmonic force constants is not consistent.");
        }

        auto preferred_basis_ref =
            boost::lexical_cast<std::string>(get_value_from_xml(pt, "Data.ForceConstants.HarmonicUnique.Basis", 0));

        if (preferred_basis_ref.empty()) preferred_basis_ref = "Cartesian";

        if (preferred_basis_ref != fcs->get_forceconstant_basis()) {
            exit("fix_forceconstants_to_file",
                 ("The basis of harmonic force constants is not consistent: the file uses FCSYM_BASIS = " +
                  preferred_basis_ref + ", but the current run uses " + fcs->get_forceconstant_basis() +
                  ".\n The default FCSYM_BASIS changed from Lattice to Cartesian in ALAMODE 2.0."
                  "\n Use FCSYM_BASIS = " +
                  preferred_basis_ref + " (symmetrization_basis in Python) or an .h5 file.")
                     .c_str());
        }
    } else if (order == 1) {
        const auto nfcs_ref =
            boost::lexical_cast<size_t>(get_value_from_xml(pt, "Data.ForceConstants.CubicUnique.NFC3"));

        if (nfcs_ref != nfcs) {
            exit("fix_forceconstants_to_file", "The number of cubic force constants is not consistent.");
        }

        auto preferred_basis_ref =
            boost::lexical_cast<std::string>(get_value_from_xml(pt, "Data.ForceConstants.CubicUnique.Basis", 0));
        if (preferred_basis_ref.empty()) preferred_basis_ref = "Cartesian";

        if (preferred_basis_ref != fcs->get_forceconstant_basis()) {
            exit("fix_forceconstants_to_file",
                 ("The basis of cubic force constants is not consistent: the file uses FCSYM_BASIS = " +
                  preferred_basis_ref + ", but the current run uses " + fcs->get_forceconstant_basis() +
                  ".\n The default FCSYM_BASIS changed from Lattice to Cartesian in ALAMODE 2.0."
                  "\n Use FCSYM_BASIS = " +
                  preferred_basis_ref + " (symmetrization_basis in Python) or an .h5 file.")
                     .c_str());
        }
    }

    intpair_fcs.resize(nfcs, std::vector<int>(order + 2));
    fcs_values.resize(nfcs);
    std::vector<std::vector<int>> intpairs_to_fix;

    intpairs_to_fix.resize(nfcs, std::vector<int>(2));

    int counter = 0;
    if (order == 0) {
        BOOST_FOREACH (const ptree::value_type &child_, pt.get_child("Data.ForceConstants.HarmonicUnique")) {
            if (child_.first == "FC2") {
                const auto &child = child_.second;
                const auto str_intpair = child.get<std::string>("<xmlattr>.pairs");
                const auto str_multiplicity = child.get<std::string>("<xmlattr>.multiplicity");

                std::istringstream is(str_intpair);
                is >> intpair_fcs[counter][0] >> intpair_fcs[counter][1];
                fcs_values[counter] = boost::lexical_cast<double>(child.data());
                ++counter;
            }
        }
    } else if (order == 1) {
        BOOST_FOREACH (const ptree::value_type &child_, pt.get_child("Data.ForceConstants.CubicUnique")) {
            if (child_.first == "FC3") {
                const auto &child = child_.second;
                const auto str_intpair = child.get<std::string>("<xmlattr>.pairs");
                const auto str_multiplicity = child.get<std::string>("<xmlattr>.multiplicity");

                std::istringstream is(str_intpair);
                is >> intpair_fcs[counter][0] >> intpair_fcs[counter][1] >> intpair_fcs[counter][2];
                fcs_values[counter] = boost::lexical_cast<double>(child.data());
                ++counter;
            }
        }
    }
}

auto Constraint::parse_forceconstants_from_h5(const int order, const std::unique_ptr<Symmetry> &symmetry,
                                              const std::unique_ptr<Fcs> &fcs, const std::string &file_to_fix,
                                              std::vector<std::vector<int>> &intpair_fcs,
                                              std::vector<double> &fcs_values) -> void
{
    const H5Easy::File file(file_to_fix, H5Easy::File::ReadOnly);

    const std::string celltype = "SuperCell";

    std::vector<std::vector<int>> mapping_table;

    get_mapping_table_from_h5(file, celltype, mapping_table);

    const auto nat_trueprim_file = mapping_table.size();

    if (nat_trueprim_file != symmetry->get_nat_trueprim()) {
        exit("parse_forceconstants_from_h5", "The number of atoms in the true primitive cell is not consistent.");
    }

    Eigen::MatrixXi atom_indices_file, atom_indices_super_file;
    Eigen::MatrixXi coord_indices_file;
    Eigen::MatrixXd shift_vectors_file;
    Eigen::ArrayXd fcs_values_file;

    get_force_constants_from_h5(file,
                                order,
                                atom_indices_file,
                                atom_indices_super_file,
                                coord_indices_file,
                                shift_vectors_file,
                                fcs_values_file);

    // fcs_values_file are force constants in the Cartesian basis.
    // They need to be converted to the lattice basis if the FCSYM_BASIS = Lattice.

    std::vector<ForceConstantTable> fc_cart, fc_cart_unique, fc_cart_copy;
    std::vector<int> flatten_array(order + 2);

    fc_cart.clear();
    for (auto i = 0; i < atom_indices_super_file.rows(); ++i) {
        for (auto j = 0; j < order + 2; ++j) {
            flatten_array[j] = atom_indices_super_file(i, j) * 3 + coord_indices_file(i, j);
        }
        fc_cart.emplace_back(fcs_values_file[i], flatten_array);
    }

    // Merge force constant entries having the same flatten arrays
    std::sort(fc_cart.begin(), fc_cart.end());

    fc_cart_unique.clear();
    fc_cart_copy.clear();
    for (auto i = 0; i < order + 2; ++i) flatten_array[i] = -1;
    double fc_sum = 0.0;
    for (const auto &it: fc_cart) {
        if (flatten_array == it.flattenarray) {
            fc_sum += it.fc_value;
        } else {
            if (std::abs(fc_sum) > 0.0) {
                fc_cart_copy.emplace_back(fc_sum, flatten_array);
            }
            fc_sum = it.fc_value;
            flatten_array = it.flattenarray;
        }
    }
    if (std::abs(fc_sum) > 0.0) {
        fc_cart_copy.emplace_back(fc_sum, flatten_array);
    }
    fc_cart.clear();

    std::copy_if(fc_cart_copy.begin(),
                 fc_cart_copy.end(),
                 std::back_inserter(fc_cart_unique),
                 [](const ForceConstantTable &obj) { return obj.is_ascending_order; });

    std::vector<int> index_tmp(order + 2);

    if (fcs->get_forceconstant_basis() == "Lattice") {
        std::vector<ForceConstantTable> fc_lattice;
        fcs->change_basis_force_constants(fc_cart_unique, fc_lattice, 1);

        for (const auto &it: fc_lattice) {
            if (it.is_ascending_order) {
                for (auto i = 0; i < order + 2; ++i) {
                    index_tmp[i] = it.flattenarray[i];
                }
                intpair_fcs.emplace_back(index_tmp);
                fcs_values.emplace_back(it.fc_value);
            }
        }
    } else {
        for (const auto &it: fc_cart_unique) {
            for (auto i = 0; i < order + 2; ++i) {
                index_tmp[i] = it.flattenarray[i];
            }
            intpair_fcs.emplace_back(index_tmp);
            fcs_values.emplace_back(it.fc_value);
        }
    }
}

auto Constraint::set_forceconstants_to_fix(const std::vector<std::vector<int>> &intpair_fix,
                                           const std::vector<double> &values_fix) -> void
{
    const auto nelems = intpair_fix[0].size();
    const auto order = nelems - 2;

    if (order == 0) {
        intpair_fix_fc2 = intpair_fix;
        values_fix_fc2 = values_fix;
        status_constraint_subset["fix2"] = 0;
    } else if (order == 1) {
        intpair_fix_fc3 = intpair_fix;
        values_fix_fc3 = values_fix;
        status_constraint_subset["fix3"] = 0;
    } else {
        exit("fit_forceconstants", "Currently, only harmonic and cubic terms can be fixed.");
    }
}

auto Constraint::generate_fix_constraint(const std::unique_ptr<Symmetry> &symmetry,
                                         const std::unique_ptr<Fcs> &fcs) -> void
{
    if (status_constraint_subset["fix2"] == 0) {
        constexpr auto order = 0;
        auto intpair_to_fix = intpair_fix_fc2;

        Fcs::translate_forceconstant_index_to_centercell(symmetry, intpair_to_fix);
        std::set<ForceConstantTable> fc_fix_table;

        const auto nfcs = intpair_to_fix.size();

        for (auto i = 0; i < nfcs; ++i) {
            fc_fix_table.insert(ForceConstantTable(values_fix_fc2[i], intpair_to_fix[i]));
        }

        size_t ihead = 0;

        double sign;
        std::set<ForceConstantTable>::iterator it_found;

        const_fix[order].clear();
        const_fix[order].shrink_to_fit();

        for (unsigned int ui = 0; ui < fcs->get_nequiv()[order].size(); ++ui) {
            size_t const mother = fcs->get_fc_table()[order][ihead].mother;
            bool found_element = false;

            for (auto j = 0; j < fcs->get_nequiv()[order][ui]; ++j) {
                std::vector<int> const index_tmp = fcs->get_fc_table()[order][ihead + j].elems;

                it_found = fc_fix_table.find(ForceConstantTable(0.0, index_tmp));

                if (it_found != fc_fix_table.end()) {
                    found_element = true;
                    sign = fcs->get_fc_table()[order][ihead + j].sign;
                    break;
                }
            }

            if (found_element) {
                const_fix[order].emplace_back(mother, sign * it_found->fc_value);
            } else {
                const_fix[order].emplace_back(mother, 0.0);
            }

            ihead += fcs->get_nequiv()[order][ui];
        }
        status_constraint_subset["fix2"] = 1;
    }

    if (status_constraint_subset["fix3"] == 0 and const_fix.size() > 1) {
        constexpr auto order = 1;
        auto intpair_to_fix = intpair_fix_fc3;

        Fcs::translate_forceconstant_index_to_centercell(symmetry, intpair_to_fix);

        const auto nfcs = intpair_to_fix.size();

        std::set<ForceConstantTable> fc_fix_table;

        for (auto i = 0; i < nfcs; ++i) {
            fc_fix_table.insert(ForceConstantTable(values_fix_fc3[i], intpair_to_fix[i]));
        }

        size_t ihead = 0;

        double sign;
        std::set<ForceConstantTable>::iterator it_found;

        const_fix[order].clear();
        const_fix[order].shrink_to_fit();

        for (unsigned int ui = 0; ui < fcs->get_nequiv()[order].size(); ++ui) {
            size_t const mother = fcs->get_fc_table()[order][ihead].mother;
            bool found_element = false;

            for (auto j = 0; j < fcs->get_nequiv()[order][ui]; ++j) {
                std::vector<int> const index_tmp = fcs->get_fc_table()[order][ihead + j].elems;
                it_found = fc_fix_table.find(ForceConstantTable(0.0, index_tmp));

                if (it_found != fc_fix_table.end()) {
                    found_element = true;
                    sign = fcs->get_fc_table()[order][ihead + j].sign;
                    break;
                }
            }

            if (found_element) {
                const_fix[order].emplace_back(mother, sign * it_found->fc_value);
            } else {
                const_fix[order].emplace_back(mother, 0.0);
            }
            ihead += fcs->get_nequiv()[order][ui];
        }

        status_constraint_subset["fix3"] = 1;
    }
}

auto Constraint::is_allzero(const std::vector<int> &vec, int &loc) -> bool
{
    loc = -1;
    for (auto i = 0; i < vec.size(); ++i) {
        if (std::abs(vec[i]) > 0) {
            loc = i;
            return false;
        }
    }
    return true;
}

auto Constraint::is_allzero(const std::vector<double> &vec, const double tol, int &loc, const int nshift) -> bool
{
    loc = -1;
    const auto n = vec.size();
    for (auto i = nshift; i < n; ++i) {
        if (std::abs(vec[i]) > tol) {
            loc = i;
            return false;
        }
    }
    return true;
}
