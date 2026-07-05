/*
 collision_operator.h

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <array>
#include <vector>
#include <Eigen/Dense>
#include "kpoint.h"
#include "pointers.h"

namespace PHON_NS
{
// Distributed three-phonon collision operator on the irreducible wedge,
// shared by the iterative-family BTE solvers (SOLVER = IBTE today; a
// variational/CG solver can reuse it later). The irreducible k points are
// distributed round-robin over the MPI ranks; L_emitt/L_absorb store the
// transition probabilities of the local rows, and a symmetry expansion
// table maps wedge values of Cartesian vector fields back onto the full
// grid. Diagonal add-ons beyond 3ph (isotope, boundary, 4ph) remain the
// solver's responsibility.
class CollisionOperator: protected Pointers
{
public:
    CollisionOperator(class PHON *);

    ~CollisionOperator();

    // Distribute the wedge over MPI ranks, enumerate the scattering
    // triplets of the local k points and build the symmetry table.
    void setup();

    // Transition probabilities of the local rows (smearing or tetrahedron,
    // following integration->ismear).
    void build_L();

    // Include the elastic isotope-disorder channel (Tamura kernel) in the
    // operator: its in-scattering enters calc_W_at and its diagonal is the
    // row sum, so the channel conserves the constant mode exactly. Must be
    // set before build_L().
    void set_isotope_channel(const bool flag)
    {
        with_isotope = flag;
    }

    bool has_isotope_channel() const
    {
        return with_isotope;
    }

    // Add the isotope diagonal 2 n(n+1) sum_j w(row,j) to the local rows of
    // q_inout ([nklocal][ns]).
    void add_isotope_diagonal(const double *const *fb, double **q_inout) const;

    // Diagonal (out-scattering) part at the local wedge points from the
    // equilibrium occupations n; q1 is [nklocal][ns].
    void calc_Q_from_L(const double *const *n, double **q1) const;

    // In-scattering action at local wedge point ikl for all (s1, xyz);
    // Wks_out is [ns][3]. Thread-safe for concurrent calls on distinct ikl.
    void calc_W_at(const int ikl, const double *const *fb, const double *const *const *dF, double **Wks_out) const;

    // Reconstruct a full-grid vector field from its wedge values:
    // dF_full[p] = expand_mat[p] . dF_ir[rep(p)], with dF_ir laid out as
    // [(irreducible k * ns + s) * 3 + xyz].
    void reconstruct_full_from_wedge(const double *dF_ir, double ***dF_full) const;

    // Project a wedge vector field onto the little-group-invariant subspace
    // (physical fields live there; the collision operator is well defined
    // and self-adjoint only on it).
    void project_onto_littlegroup(double *dF_ir) const;

    // Assemble this rank's wedge rows of the dense operator in the
    // multiplicity-symmetrized metric, S = M^{1/2} (Q_diag + W) M^{-1/2}
    // with M = diag(star multiplicity): slab is row-major
    // [nklocal*ns*3][nk_irred*ns*3], zero-initialized by the caller.
    // Assembly runs over the stored L entries (3ph and, when enabled,
    // isotope), i.e. it costs one operator application. Used by SOLVER =
    // DBTE.
    void assemble_dense_rows(const double *const *fb, const double *const *Qfin_loc, double *slab) const;

    int get_nklocal() const
    {
        return nklocal;
    }

    const std::vector<int> &get_local_irred_ks() const
    {
        return nk_l;
    }

    long get_kplength_total() const
    {
        return static_cast<long>(kplength_emitt) + static_cast<long>(kplength_absorb);
    }

private:
    int kplength_emitt;
    int kplength_absorb;
    int nk_3ph, nklocal, ns, ns2;

    bool use_triplet_symmetry;
    bool sym_permutation;

    double ***L_absorb; // L q0 + q1 -> q2
    double ***L_emitt;  // L q0 -> q1 + q2

    // Elastic isotope-disorder kernel: sparse partners of each local wedge
    // row with the temperature-independent value w = (pi/4N) w1 w2 g2
    // |<e2|e1>|^2 delta(w1-w2); rowsum feeds the diagonal.
    struct IsotopeEntry
    {
        int knum;
        int snum;
        double val;
    };

    bool with_isotope;
    std::vector<std::vector<IsotopeEntry>> L_iso; // [ikl * ns + s]
    std::vector<double> L_iso_rowsum;

    void build_L_isotope();

    std::vector<std::vector<KsListGroup>> localnk_triplets_emitt;
    std::vector<std::vector<KsListGroup>> localnk_triplets_absorb;

    std::vector<int> nk_l;

    // Flattened (local irreducible k, triplet) index built in
    // get_triplets(): the row of triplet j of local k point ik in L is
    // offset_*[ik] + j, and pairs_* lists all rows as (ik, j) for loops
    // that parallelize over the flat index directly.
    std::vector<std::array<int, 2>> pairs_emitt, pairs_absorb;
    std::vector<int> offset_emitt, offset_absorb;

    // Cartesian rotation (with the time-reversal sign folded in) mapping
    // the wedge representative onto each full-grid point.
    std::vector<Eigen::Matrix3d> expand_mat;

    // Average over the little group of each wedge representative (with the
    // time-reversal sign): projector onto invariant Cartesian vectors.
    std::vector<Eigen::Matrix3d> littlegroup_proj;

    void get_triplets();

    void build_expansion_table();

    void setup_L_smear();

    void setup_L_tetra();
};
} // namespace PHON_NS
