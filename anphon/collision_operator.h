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

namespace PHON_NS
{
class System;
class Symmetry;
class Integration;
class AnharmonicCore;
class TetraNodes;
class DymatEigenValue;
// Distributed three-phonon collision operator on the irreducible wedge,
// shared by the iterative-family BTE solvers (SOLVER = IBTE today; a
// variational/CG solver can reuse it later). The irreducible k points are
// distributed round-robin over the MPI ranks; L_emitt/L_absorb store the
// transition probabilities of the local rows, and a symmetry expansion
// table maps wedge values of Cartesian vector fields back onto the full
// grid. Diagonal add-ons beyond 3ph (isotope, boundary, 4ph) remain the
// solver's responsibility.
// All dependencies are explicit constructor arguments (no Pointers base):
// the operator is constructed by Iterativebte::setup, after setup_base(),
// when every input already exists.
class CollisionOperator
{
public:
    CollisionOperator(const KpointMeshUniform &kmesh_dos_in, const TetraNodes &tetra_nodes_dos_in,
                      const DymatEigenValue &dymat_dos_in, const System &system_in, const Symmetry &symmetry_in,
                      Integration &integration_in, AnharmonicCore &anharmonic_core_in, unsigned int ns_in,
                      int my_rank_in, int nprocs_in);

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
    // set before build_L(). isotope_factor_in points at the per-species
    // mass-variance factors (Isotope::isotope_factor) and must stay alive
    // for the lifetime of this object.
    void set_isotope_channel(const bool flag, const double *isotope_factor_in = nullptr)
    {
        with_isotope = flag;
        isotope_factor = isotope_factor_in;
    }

    bool has_isotope_channel() const
    {
        return with_isotope;
    }

    // Add the isotope diagonal 2 g1 sum_j g2_j w(row,j) to the local rows
    // of q_inout ([nklocal][ns]) - the row sum of the in-scattering entries,
    // so the elastic channel annihilates constant fields exactly.
    void add_isotope_diagonal(const double *const *sqrt_occ, double **q_inout) const;

    // The occupation factors enter through the detailed-balance-symmetric
    // kernel: every three-phonon product n1 n2 (n3+1) / n1 (n2+1)(n3+1) is
    // replaced by g1 g2 g3 with g = sqrt(n(n+1)) = 1/(2 sinh(bw/2)), to
    // which it is identical on the energy shell. This makes the operator
    // exactly symmetric off shell as well (up to the direction dependence
    // of tetrahedron/adaptive weights), so the three solvers agree on the
    // same discretized problem. Callers pass the precomputed table
    // sqrt_occ[k][s] = sqrt(n(n+1)).

    // Diagonal (out-scattering) part at the local wedge points; q1 is
    // [nklocal][ns].
    void calc_Q_from_L(const double *const *sqrt_occ, double **q1) const;

    // In-scattering action at local wedge point ikl for all (s1, xyz);
    // Wks_out is [ns][3]. Thread-safe for concurrent calls on distinct ikl.
    void calc_W_at(const int ikl, const double *const *sqrt_occ, const double *const *const *dF,
                   double **Wks_out) const;

    // Reconstruct a full-grid vector field from its wedge values:
    // dF_full[p] = expand_mat[p] . dF_ir[rep(p)], with dF_ir laid out as
    // [(irreducible k * ns + s) * 3 + xyz].
    void reconstruct_full_from_wedge(const double *dF_ir, double ***dF_full) const;

    // Project a wedge vector field onto the little-group-invariant subspace
    // (physical fields live there; the collision operator is well defined
    // and self-adjoint only on it).
    void project_onto_littlegroup(double *dF_ir) const;

    const Eigen::Matrix3d &get_littlegroup_projector(const int ik) const
    {
        return littlegroup_proj[ik];
    }

    // Assemble this rank's wedge rows of the dense operator in the
    // multiplicity-symmetrized metric, S = M^{1/2} (Q_diag + W) M^{-1/2}
    // with M = diag(star multiplicity): slab is row-major
    // [nklocal*ns*3][nk_irred*ns*3], zero-initialized by the caller.
    // Assembly runs over the stored L entries (3ph and, when enabled,
    // isotope), i.e. it costs one operator application. Used by SOLVER =
    // DBTE.
    void assemble_dense_rows(const double *const *sqrt_occ, const double *const *Qfin_loc, double *slab) const;

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
    const KpointMeshUniform &kmesh_dos_;
    const TetraNodes &tetra_nodes_dos_;
    const DymatEigenValue &dymat_dos_;
    const System &system_;
    const Symmetry &symmetry_;
    Integration &integration_;         // smearing settings and weight kernels
    AnharmonicCore &anharmonic_core_;  // V3 evaluation
    const int my_rank_;
    const int nprocs_;

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
    const double *isotope_factor = nullptr; // per-species mass variance (set_isotope_channel)
    std::vector<std::vector<IsotopeEntry>> L_iso; // [ikl * ns + s]

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
