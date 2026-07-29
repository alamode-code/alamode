/*
 mode_symmetry.cpp

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "mode_symmetry.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <string>
#include <vector>

#include "degeneracy_utils.h"
#include "dielec.h"
#include "dynamical.h"
#include "error.h"
#include "ewald.h"
#include "fcs_phonon.h"
#include "mpi_common.h"
#include "ndarray.h"
#include "pointgroup_data.h"
#include "spglib.h"
#include "symmetry_core.h"
#include "system.h"

using namespace PHON_NS;

namespace
{

// Apply the Gamma-point representation matrix of a space-group operation to
// a 3N vector: (T(S) v)_{3*mapping[j]+a} = sum_b S_ab v_{3j+b}.  At Gamma
// all Bloch phases are unity, so T(S) is the real block permutation-rotation
// matrix; fractional translations drop out entirely.
void apply_symmetry_gamma(const std::vector<double> &rot9, const std::vector<unsigned int> &mapping,
                          const std::complex<double> *v, std::complex<double> *w)
{
    const auto natmin = mapping.size();
    for (std::size_t jat = 0; jat < natmin; ++jat) {
        const auto iat = mapping[jat];
        for (auto a = 0; a < 3; ++a) {
            std::complex<double> s(0.0, 0.0);
            for (auto b = 0; b < 3; ++b) {
                s += rot9[3 * a + b] * v[3 * jat + b];
            }
            w[3 * iat + a] = s;
        }
    }
}

std::string kind_name(const pointgroup::OpKind kind)
{
    switch (kind) {
    case pointgroup::OpKind::E:
        return "E";
    case pointgroup::OpKind::C2:
        return "C2";
    case pointgroup::OpKind::C3:
        return "C3";
    case pointgroup::OpKind::C4:
        return "C4";
    case pointgroup::OpKind::C6:
        return "C6";
    case pointgroup::OpKind::I:
        return "i";
    case pointgroup::OpKind::Sigma:
        return "sigma";
    case pointgroup::OpKind::S6:
        return "S6";
    case pointgroup::OpKind::S4:
        return "S4";
    case pointgroup::OpKind::S3:
        return "S3";
    }
    return "?";
}

// Normalize the small set of Hermann-Mauguin spelling variants that differ
// between spglib versions and settings.
std::string normalize_hm(std::string s)
{
    s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
    if (s == "m3") {
        return "m-3";
    }
    if (s == "m3m") {
        return "m-3m";
    }
    if (s == "-4m2") {
        return "-42m";
    }
    if (s == "-62m") {
        return "-6m2";
    }
    if (s == "3m1" || s == "31m") {
        return "3m";
    }
    if (s == "-3m1" || s == "-31m") {
        return "-3m";
    }
    return s;
}

std::string format_decomposition(const pointgroup::PointGroup &pg, const std::vector<int> &n_mu,
                                 const std::vector<std::string> &names)
{
    std::string str;
    for (auto mu = 0; mu < pg.nclass; ++mu) {
        if (n_mu[mu] <= 0) {
            continue;
        }
        if (!str.empty()) {
            str += " + ";
        }
        if (n_mu[mu] > 1) {
            str += std::to_string(n_mu[mu]);
        }
        str += names[mu];
    }
    if (str.empty()) {
        str = "(none)";
    }
    return str;
}

} // namespace

ModeSymmetry::ModeSymmetry(PHON *phon) : Pointers(phon)
{}

ModeSymmetry::~ModeSymmetry() = default;

void ModeSymmetry::setup()
{
    MPI_Bcast(&print_irreps, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);

    if (print_irreps) {
        const auto &spin = system->get_spin_prim();
        if (spin.lspin && spin.noncollinear) {
            exitall("ModeSymmetry::setup",
                    "IRREPS = 1 is not supported for noncollinear magnetic systems:\n"
                    " the symmetry operations found for such systems can be antiunitary\n"
                    " (coupled with time reversal), and magnetic corepresentations are not\n"
                    " implemented. Please turn off IRREPS.");
        }
    }
}

void ModeSymmetry::analyze_irreps_at_gamma()
{
    if (mympi->my_rank != 0) {
        return;
    }

    result_ = GammaIrrepResult{};
    auto &warnings = result_.warnings;

    const auto ns = static_cast<int>(dynamical->neval);
    const auto &symops = symmetry->SymmListWithMap;
    const auto nsym = static_cast<int>(symops.size());

    if (nsym == 0 || static_cast<int>(symmetry->SymmList.size()) != nsym) {
        warnings.emplace_back("symmetry operation tables are unavailable; analysis skipped.");
        return;
    }

    // SymmList and SymmListWithMap are index-aligned by construction
    // (gensym_withmap appends one entry per SymmList entry); verify defensively
    // because the class algebra below uses SymmList while the representation
    // matrices use SymmListWithMap.
    std::vector<Eigen::Matrix3i> rot_latt;
    std::vector<Eigen::Matrix3d> rot_cart;
    rot_latt.reserve(nsym);
    rot_cart.reserve(nsym);
    for (const auto &op: symmetry->SymmList) {
        rot_latt.push_back(op.rotation);
        rot_cart.push_back(op.rotation_cart);
    }
    for (auto isym = 0; isym < nsym; ++isym) {
        for (auto a = 0; a < 3; ++a) {
            for (auto b = 0; b < 3; ++b) {
                if (std::abs(symops[isym].rot[3 * a + b] - rot_cart[isym](a, b)) > 1.0e-8) {
                    warnings.emplace_back("internal error: SymmList and SymmListWithMap are not "
                                          "index-aligned; analysis skipped.");
                    return;
                }
            }
        }
    }

    // The rotation parts must be unique: one operation per point-group element.
    // Duplicates mean the cell handed to anphon is not primitive.
    for (auto i = 0; i < nsym; ++i) {
        for (auto j = i + 1; j < nsym; ++j) {
            if (rot_latt[i] == rot_latt[j]) {
                warnings.emplace_back("duplicate rotation parts detected: the input cell is not "
                                      "a primitive cell. Mulliken labels and activity flags are "
                                      "not assigned.");
                return;
            }
        }
    }

    // ------------------------------------------------------------------
    // Eigenvectors of the analytic dynamical matrix at Gamma.
    // With kvec = 0 the directional nonanalytic term vanishes for
    // NONANALYTIC = 1/2; NONANALYTIC = 3 must go through the Ewald path
    // with the dipole-free force constants (plain eval_k would add an
    // uninitialized nonanalytic matrix in that mode).
    // ------------------------------------------------------------------
    std::vector<double> eval_raw(ns), omega(ns);
    NDArray<std::complex<double>, 2> evec;
    evec.resize(ns, ns);
    {
        double xk[3] = {0.0, 0.0, 0.0};
        if (dynamical->nonanalytic == 3) {
            dynamical->eval_k_ewald(xk, xk, ewald->fc2_without_dipole, eval_raw.data(), evec, true);
        } else {
            dynamical->eval_k(xk, xk, fcs_phonon->force_constant_with_cell[0], eval_raw.data(), evec, true);
        }
    }
    for (auto is = 0; is < ns; ++is) {
        omega[is] = dynamical->freq(eval_raw[is]);
    }

    // ------------------------------------------------------------------
    // Degenerate multiplets with symmetry closure.
    // ------------------------------------------------------------------
    std::vector<int> degeneracy;
    find_degenerate_groups(ns, omega.data(), degeneracy);

    struct Cluster
    {
        int start = 0;
        int size = 0;
    };
    std::vector<Cluster> clusters;
    {
        int offset = 0;
        for (const auto d: degeneracy) {
            clusters.push_back({offset, d});
            offset += d;
        }
    }

    std::vector<std::complex<double>> work(ns);

    // Symmetry leakage of a cluster: Frobenius norm ||(1 - P) T(S) P||_F,
    // i.e. with orthonormal eigenvectors and unitary T(S) the square root of
    // sum_i (1 - sum_j |<e_j| T e_i>|^2), maximized over the operations.
    auto cluster_leakage = [&](const Cluster &cl) {
        double max_leak = 0.0;
        for (auto isym = 0; isym < nsym; ++isym) {
            double frob2 = 0.0;
            for (auto i = cl.start; i < cl.start + cl.size; ++i) {
                apply_symmetry_gamma(symops[isym].rot, symops[isym].mapping, evec[i], work.data());
                double sum = 0.0;
                for (auto j = cl.start; j < cl.start + cl.size; ++j) {
                    std::complex<double> c(0.0, 0.0);
                    for (auto k = 0; k < ns; ++k) {
                        c += std::conj(evec[j][k]) * work[k];
                    }
                    sum += std::norm(c);
                }
                frob2 += std::max(0.0, 1.0 - sum);
            }
            max_leak = std::max(max_leak, std::sqrt(frob2));
        }
        return max_leak;
    };

    constexpr double leak_tol = 1.0e-4;
    bool closure_ok = true;

    for (auto iter = 0; iter <= ns; ++iter) {
        int bad = -1;
        double worst = leak_tol;
        for (std::size_t ic = 0; ic < clusters.size(); ++ic) {
            const auto leak = cluster_leakage(clusters[ic]);
            if (leak > worst) {
                worst = leak;
                bad = static_cast<int>(ic);
            }
        }
        if (bad < 0) {
            break;
        }
        if (clusters.size() == 1) {
            closure_ok = false;
            break;
        }
        // Merge the leaking cluster with the frequency-adjacent neighbor.
        const auto last = static_cast<int>(clusters.size()) - 1;
        int partner;
        if (bad == 0) {
            partner = 1;
        } else if (bad == last) {
            partner = last - 1;
        } else {
            const auto gap_prev =
                omega[clusters[bad].start] - omega[clusters[bad - 1].start + clusters[bad - 1].size - 1];
            const auto gap_next = omega[clusters[bad + 1].start] - omega[clusters[bad].start + clusters[bad].size - 1];
            partner = (gap_prev <= gap_next) ? bad - 1 : bad + 1;
        }
        const auto lo = std::min(bad, partner);
        const auto hi = std::max(bad, partner);
        clusters[lo].size += clusters[hi].size;
        clusters.erase(clusters.begin() + hi);
        warnings.emplace_back("nearly degenerate modes were merged into one multiplet to obtain "
                              "a symmetry-invariant subspace.");
        if (iter == ns) {
            closure_ok = false;
        }
    }
    if (!closure_ok) {
        warnings.emplace_back("could not build symmetry-closed multiplets (the harmonic force "
                              "constants may break the crystal symmetry; consider FCSYM/HESSIAN "
                              "symmetrization of the input). Labels and activity flags are "
                              "reported as approximate.");
    }

    const auto ngroup = static_cast<int>(clusters.size());

    // ------------------------------------------------------------------
    // Characters per multiplet and per operation.
    // ------------------------------------------------------------------
    std::vector<std::vector<std::complex<double>>> chi_op(ngroup, std::vector<std::complex<double>>(nsym));
    for (auto isym = 0; isym < nsym; ++isym) {
        for (auto ig = 0; ig < ngroup; ++ig) {
            std::complex<double> chi(0.0, 0.0);
            for (auto i = clusters[ig].start; i < clusters[ig].start + clusters[ig].size; ++i) {
                apply_symmetry_gamma(symops[isym].rot, symops[isym].mapping, evec[i], work.data());
                for (auto k = 0; k < ns; ++k) {
                    chi += std::conj(evec[i][k]) * work[k];
                }
            }
            chi_op[ig][isym] = chi;
        }
    }

    // Route (b): activity from the table-free projections onto the vector
    // representation and its symmetric square.  Needs only the operations.
    std::vector<double> trace_op(nsym), trace_sq_op(nsym);
    for (auto isym = 0; isym < nsym; ++isym) {
        trace_op[isym] = rot_cart[isym].trace();
        trace_sq_op[isym] = (rot_cart[isym] * rot_cart[isym]).trace();
    }

    // ------------------------------------------------------------------
    // Point-group identification: operation-count fingerprint cross-checked
    // against spglib's spg_get_pointgroup on the same rotation parts.
    // ------------------------------------------------------------------
    auto labels_ok = true;

    const auto pg_number = pointgroup::identify_point_group_fingerprint(rot_cart);
    if (pg_number < 1) {
        warnings.emplace_back("the rotation parts do not form a crystallographic point group; "
                              "Mulliken labels are not assigned.");
        labels_ok = false;
    }

    if (labels_ok) {
        auto rots_spg = std::make_unique<int[][3][3]>(nsym);
        for (auto isym = 0; isym < nsym; ++isym) {
            for (auto a = 0; a < 3; ++a) {
                for (auto b = 0; b < 3; ++b) {
                    rots_spg[isym][a][b] = rot_latt[isym](a, b);
                }
            }
        }
        char symbol[6];
        int trans_mat[3][3];
        const auto ptg = spg_get_pointgroup(symbol, trans_mat, rots_spg.get(), nsym);
        if (ptg < 1) {
            warnings.emplace_back("spglib could not identify the point group; Mulliken labels "
                                  "are not assigned.");
            labels_ok = false;
        } else {
            const auto hm_spglib = normalize_hm(std::string(symbol));
            const auto hm_table = normalize_hm(pointgroup::pg_table[pg_number - 1].international);
            if (hm_spglib != hm_table) {
                warnings.emplace_back("point-group identification mismatch (fingerprint: " + hm_table +
                                      ", spglib: " + hm_spglib + "); Mulliken labels are not assigned.");
                labels_ok = false;
            }
        }
    }

    // ------------------------------------------------------------------
    // Merged conjugacy classes (exact integer arithmetic) and operation
    // classification.
    // ------------------------------------------------------------------
    auto classes = pointgroup::conjugacy_classes_merged(rot_latt);
    if (classes.empty()) {
        warnings.emplace_back("the symmetry operations do not close under multiplication; "
                              "analysis skipped.");
        return;
    }
    const auto ncl = static_cast<int>(classes.size());

    std::vector<pointgroup::OpInfo> ops_info(nsym);
    for (auto isym = 0; isym < nsym; ++isym) {
        if (!pointgroup::classify_op(rot_cart[isym], ops_info[isym])) {
            warnings.emplace_back("a symmetry operation could not be classified; analysis skipped.");
            return;
        }
    }

    std::vector<int> nelem_of_class(ncl);
    for (auto ic = 0; ic < ncl; ++ic) {
        nelem_of_class[ic] = static_cast<int>(classes[ic].size());
    }

    // ------------------------------------------------------------------
    // Class -> character-table-column matching.  The reference frame for
    // convention-dependent labels comes from the spglib standardized cell
    // when available (see Symmetry); otherwise those labels are flagged.
    // ------------------------------------------------------------------
    const pointgroup::PointGroup *pg = nullptr;
    pointgroup::MatchResult match;
    if (labels_ok) {
        pg = &pointgroup::pg_table[pg_number - 1];
        Eigen::Vector3d zhat = Eigen::Vector3d::Zero();
        Eigen::Vector3d xhat = Eigen::Vector3d::Zero();
        if (symmetry->has_spg_dataset) {
            // Authoritative conventional frame from the spglib dataset:
            // L_conv = L_input * P^-1 with P the dataset transformation
            // matrix (basis change only, expressed directly in the Cartesian
            // frame of the calculation).  Sanity-checked against the
            // operations: when a unique principal axis (order >= 3) exists,
            // the conventional c axis must be parallel to it.
            const Eigen::Matrix3d &lavec = system->get_primcell().lattice_vector;
            const Eigen::Matrix3d &pmat = symmetry->spg_transformation_matrix;
            if (std::abs(pmat.determinant()) > 1.0e-8) {
                const Eigen::Matrix3d lconv = lavec * pmat.inverse();
                zhat = lconv.col(2).normalized();
                xhat = lconv.col(0).normalized();

                Eigen::Vector3d z_ops = Eigen::Vector3d::Zero();
                auto best_order = 2;
                auto unique_axis = true;
                for (const auto &op: ops_info) {
                    if (op.kind == pointgroup::OpKind::E || op.kind == pointgroup::OpKind::I) {
                        continue;
                    }
                    if (op.order > best_order) {
                        best_order = op.order;
                        z_ops = op.axis;
                        unique_axis = true;
                    } else if (op.order == best_order && best_order > 2 && std::abs(op.axis.dot(z_ops)) < 1.0 - 1.0e-6)
                    {
                        unique_axis = false;
                    }
                }
                if (unique_axis && best_order > 2 && std::abs(zhat.dot(z_ops)) < 1.0 - 1.0e-4) {
                    warnings.emplace_back("the standardized-cell frame is inconsistent with "
                                          "the symmetry-operation axes; convention-dependent "
                                          "labels are not resolved.");
                    zhat.setZero();
                    xhat.setZero();
                }
            }
        }
        match = pointgroup::match_classes_to_columns(*pg, classes, ops_info, zhat, xhat);
        if (!match.ok) {
            for (const auto &w: match.warnings) {
                warnings.push_back(w);
            }
            warnings.emplace_back("conjugacy classes could not be matched to the character "
                                  "table; Mulliken labels are not assigned.");
            labels_ok = false;
        } else {
            for (const auto &w: match.warnings) {
                warnings.push_back(w);
            }
            if (!match.convention_resolved) {
                warnings.emplace_back("the conventional reference frame is unavailable or "
                                      "degenerate: labels that depend on the axis convention "
                                      "(primed classes, B1/B2-type symbols) follow an arbitrary "
                                      "but deterministic internal choice.");
            }
            result_.axis_convention_note = match.note;
        }
    }

    // ------------------------------------------------------------------
    // Per-class characters (class-function consistency), vector-rep traces,
    // and the total vibrational representation.
    // ------------------------------------------------------------------
    std::vector<std::vector<double>> chi_class(ngroup, std::vector<double>(ncl));
    std::vector<double> max_spread(ngroup, 0.0), max_imag(ngroup, 0.0);
    for (auto ig = 0; ig < ngroup; ++ig) {
        for (auto ic = 0; ic < ncl; ++ic) {
            double avg = 0.0;
            for (const auto im: classes[ic]) {
                avg += chi_op[ig][im].real();
                max_imag[ig] = std::max(max_imag[ig], std::abs(chi_op[ig][im].imag()));
            }
            avg /= static_cast<double>(classes[ic].size());
            for (const auto im: classes[ic]) {
                max_spread[ig] = std::max(max_spread[ig], std::abs(chi_op[ig][im].real() - avg));
            }
            chi_class[ig][ic] = avg;
        }
        if (max_imag[ig] > 1.0e-6) {
            warnings.emplace_back("characters have a nonzero imaginary part (mode group " + std::to_string(ig + 1) +
                                  "); results may be unreliable.");
        }
        if (max_spread[ig] > 1.0e-4) {
            warnings.emplace_back("characters are not constant on conjugacy classes (mode group " +
                                  std::to_string(ig + 1) +
                                  "): the harmonic force constants "
                                  "break the crystal symmetry. Labels are suppressed.");
            labels_ok = false;
        }
    }

    std::vector<double> trace_class(ncl), chi_total_class(ncl);
    for (auto ic = 0; ic < ncl; ++ic) {
        const auto rep = classes[ic][0];
        trace_class[ic] = trace_op[rep];
        int natfix = 0;
        const auto &mapping = symops[rep].mapping;
        for (std::size_t j = 0; j < mapping.size(); ++j) {
            if (mapping[j] == j) {
                ++natfix;
            }
        }
        chi_total_class[ic] = static_cast<double>(natfix) * trace_op[rep];
    }

    // Completeness cross-check: the multiplet characters must sum to the
    // character of the full vibrational representation.
    for (auto ic = 0; ic < ncl; ++ic) {
        double chi_sum = 0.0;
        for (auto ig = 0; ig < ngroup; ++ig) {
            chi_sum += chi_class[ig][ic];
        }
        if (std::abs(chi_sum - chi_total_class[ic]) > 1.0e-3 * static_cast<double>(ns)) {
            warnings.emplace_back("the multiplet characters do not sum to the total "
                                  "vibrational character (class " +
                                  std::to_string(ic + 1) + "); results may be unreliable.");
        }
    }

    // ------------------------------------------------------------------
    // Representation-level validation of the class->column assignment (the
    // hybrid scheme of the plan): among all bijections consistent with the
    // (kind, nelem) buckets, keep those whose decompositions of Gamma_V,
    // Gamma_total, and every symmetry-clean multiplet are non-negative
    // integers with consistent dimensions and Gamma_optic >= 0; the
    // geometrically chosen assignment must be a survivor.
    // ------------------------------------------------------------------
    if (labels_ok && pg) {
        auto assignment_ok = [&](const std::vector<int> &a2c) {
            const auto n_vec = pointgroup::decompose_representation(*pg, a2c, nelem_of_class, trace_class);
            const auto n_tot = pointgroup::decompose_representation(*pg, a2c, nelem_of_class, chi_total_class);
            for (auto mu = 0; mu < pg->nclass; ++mu) {
                const auto iv = std::lround(n_vec[mu]);
                const auto it = std::lround(n_tot[mu]);
                if (std::abs(n_vec[mu] - iv) > 1.0e-3 || std::abs(n_tot[mu] - it) > 1.0e-3 || iv < 0 || it < 0 ||
                    it < iv)
                {
                    return false;
                }
            }
            // Multiplet-level checks only when the clusters are trustworthy.
            for (auto ig = 0; ig < ngroup; ++ig) {
                if (!closure_ok || max_spread[ig] > 1.0e-4) {
                    continue;
                }
                const auto n_mu = pointgroup::decompose_representation(*pg, a2c, nelem_of_class, chi_class[ig]);
                auto dimsum = 0;
                for (auto mu = 0; mu < pg->nclass; ++mu) {
                    const auto im = std::lround(n_mu[mu]);
                    if (std::abs(n_mu[mu] - im) > 1.0e-3 || im < 0) {
                        return false;
                    }
                    dimsum += static_cast<int>(im) * pg->irreps[mu].dim;
                }
                if (dimsum != static_cast<int>(clusters[ig].size)) {
                    return false;
                }
            }
            return true;
        };

        const auto candidates = pointgroup::enumerate_consistent_assignments(*pg, classes, ops_info);
        std::vector<const std::vector<int> *> survivors;
        for (const auto &cand: candidates) {
            if (assignment_ok(cand)) {
                survivors.push_back(&cand);
            }
        }
        if (survivors.empty()) {
            warnings.emplace_back("no class-to-character-table assignment yields consistent "
                                  "integer decompositions; Mulliken labels are not assigned.");
            labels_ok = false;
        } else if (!assignment_ok(match.class_to_col)) {
            if (survivors.size() == 1) {
                warnings.emplace_back("the geometrically chosen class assignment failed the "
                                      "representation validation; the unique consistent "
                                      "assignment is used instead.");
                match.class_to_col = *survivors[0];
                match.convention_resolved = false;
            } else {
                warnings.emplace_back("the geometrically chosen class assignment failed the "
                                      "representation validation and several alternatives "
                                      "remain; Mulliken labels are not assigned.");
                labels_ok = false;
            }
        }
    }

    // Display names: when the axis convention could not be resolved, mark the
    // irreps that are distinguished only by convention-dependent classes.
    std::vector<std::string> irrep_names;
    if (labels_ok && pg) {
        irrep_names.resize(pg->nclass);
        std::vector<char> ambiguous_col(pg->nclass, 0);
        if (!match.convention_resolved) {
            for (const auto col: match.ambiguous_cols) {
                ambiguous_col[col] = 1;
            }
        }
        std::vector<char> uncertain(pg->nclass, 0);
        for (auto mu = 0; mu < pg->nclass; ++mu) {
            for (auto nu = mu + 1; nu < pg->nclass; ++nu) {
                auto same_outside = true;
                for (auto col = 0; col < pg->nclass; ++col) {
                    if (!ambiguous_col[col] && pg->chartab[mu * pg->nclass + col] != pg->chartab[nu * pg->nclass + col])
                    {
                        same_outside = false;
                        break;
                    }
                }
                if (same_outside) {
                    uncertain[mu] = uncertain[nu] = 1;
                }
            }
        }
        for (auto mu = 0; mu < pg->nclass; ++mu) {
            irrep_names[mu] = pg->irreps[mu].mulliken;
            if (uncertain[mu]) {
                irrep_names[mu] += "?";
            }
        }
    }

    // ------------------------------------------------------------------
    // Acoustic content: overlap with the mass-weighted rigid translations
    // t_alpha[3j+beta] = delta_ab sqrt(m_j / M).
    // ------------------------------------------------------------------
    const auto &mass = system->get_mass_prim();
    double mass_total = 0.0;
    for (const auto m: mass) {
        mass_total += m;
    }

    auto acoustic_content = [&](const Cluster &cl) {
        double content = 0.0;
        for (auto alpha = 0; alpha < 3; ++alpha) {
            for (auto i = cl.start; i < cl.start + cl.size; ++i) {
                std::complex<double> ovl(0.0, 0.0);
                for (std::size_t j = 0; j < mass.size(); ++j) {
                    ovl += std::sqrt(mass[j] / mass_total) * evec[i][3 * j + alpha];
                }
                content += std::norm(ovl);
            }
        }
        return content;
    };

    // ------------------------------------------------------------------
    // Assemble the per-multiplet results.
    // ------------------------------------------------------------------
    const auto order = nsym;
    result_.groups.resize(ngroup);

    // Per-branch acoustic detector, used as a diagnostic cross-check of the
    // subspace-content criterion (its per-eigenvector projections are
    // basis-dependent inside degenerate clusters, so it is not authoritative).
    const auto acoustic_flags = dynamical->detect_acoustic_modes_at_gamma(evec, 0.9, false);

    for (auto ig = 0; ig < ngroup; ++ig) {
        auto &grp = result_.groups[ig];
        const auto &cl = clusters[ig];

        for (auto i = cl.start; i < cl.start + cl.size; ++i) {
            grp.mode_indices.push_back(i);
        }
        double omega_mean = 0.0;
        for (auto i = cl.start; i < cl.start + cl.size; ++i) {
            omega_mean += omega[i];
        }
        grp.omega = omega_mean / static_cast<double>(cl.size);
        grp.characters = chi_class[ig];
        grp.max_imag_char = max_imag[ig];
        grp.max_class_spread = max_spread[ig];

        grp.acoustic_content = acoustic_content(cl);
        grp.is_acoustic = grp.acoustic_content > static_cast<double>(cl.size) - 0.1;
        if (grp.acoustic_content > 0.1 && !grp.is_acoustic) {
            warnings.emplace_back("acoustic and optical modes mix within mode group " + std::to_string(ig + 1) +
                                  " (acoustic content " + std::to_string(grp.acoustic_content) + " of dimension " +
                                  std::to_string(cl.size) + "); its acoustic assignment is ambiguous.");
        }
        auto n_flagged = 0;
        for (auto i = cl.start; i < cl.start + cl.size; ++i) {
            if (acoustic_flags[i]) {
                ++n_flagged;
            }
        }
        if (n_flagged != std::lround(grp.acoustic_content)) {
            warnings.emplace_back("diagnostic: the per-branch acoustic detector (" + std::to_string(n_flagged) +
                                  " branches) disagrees with "
                                  "the translation-subspace content (" +
                                  std::to_string(grp.acoustic_content) + ") for mode group " + std::to_string(ig + 1) +
                                  ".");
        }

        // Route (b): projection activity (table-free, authoritative).
        double n_ir = 0.0, n_raman = 0.0;
        for (auto isym = 0; isym < nsym; ++isym) {
            n_ir += chi_op[ig][isym].real() * trace_op[isym];
            n_raman += chi_op[ig][isym].real() * 0.5 * (trace_op[isym] * trace_op[isym] + trace_sq_op[isym]);
        }
        n_ir /= static_cast<double>(order);
        n_raman /= static_cast<double>(order);

        const auto integral_proj = std::abs(n_ir - std::lround(n_ir)) < 1.0e-3 &&
                                   std::abs(n_raman - std::lround(n_raman)) < 1.0e-3 && n_ir > -1.0e-3 &&
                                   n_raman > -1.0e-3;
        grp.n_ir_proj = n_ir;
        grp.n_raman_proj = n_raman;
        grp.ir_active = n_ir > 0.5;
        grp.raman_active = n_raman > 0.5;
        grp.activity_known = closure_ok && integral_proj && max_spread[ig] <= 1.0e-4;
        if (!integral_proj) {
            warnings.emplace_back("non-integral activity projection for mode group " + std::to_string(ig + 1) +
                                  "; activity flags are approximate.");
        }

        // Irrep assignment (route (a)) when the table matching succeeded and
        // this multiplet is a symmetry-clean invariant subspace.
        if (labels_ok && pg && closure_ok && max_spread[ig] <= 1.0e-4) {
            const auto n_mu =
                pointgroup::decompose_representation(*pg, match.class_to_col, nelem_of_class, chi_class[ig]);
            std::vector<int> n_int(pg->nclass, 0);
            auto integral = true;
            auto dimsum = 0;
            for (auto mu = 0; mu < pg->nclass; ++mu) {
                n_int[mu] = static_cast<int>(std::lround(n_mu[mu]));
                if (n_int[mu] < 0 || std::abs(n_mu[mu] - n_int[mu]) > 1.0e-3) {
                    integral = false;
                }
                dimsum += n_int[mu] * pg->irreps[mu].dim;
            }
            if (!integral || dimsum != cl.size) {
                grp.irrep_label = "??";
                grp.activity_known = false;
                warnings.emplace_back("mode group " + std::to_string(ig + 1) +
                                      " does not decompose into integer irrep multiplicities.");
            } else {
                std::string label;
                auto ir_table = false, raman_table = false;
                for (auto mu = 0; mu < pg->nclass; ++mu) {
                    for (auto rep = 0; rep < n_int[mu]; ++rep) {
                        if (!label.empty()) {
                            label += "(+)";
                        }
                        label += irrep_names[mu];
                    }
                    if (n_int[mu] > 0) {
                        ir_table = ir_table || pg->irreps[mu].ir_active;
                        raman_table = raman_table || pg->irreps[mu].raman_active;
                    }
                }
                grp.irrep_label = label;

                // Routes (a) and (b) must agree; disagreement indicates an
                // internal inconsistency, so suppress the labels.
                if (grp.activity_known && (ir_table != grp.ir_active || raman_table != grp.raman_active)) {
                    warnings.emplace_back("internal error: table-based and projection-based "
                                          "activities disagree for mode group " +
                                          std::to_string(ig + 1) + "; labels are suppressed.");
                    labels_ok = false;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // IR oscillator strengths from Born effective charges (when loaded).
    // S_ab(lambda) = sum_{nu in lambda} Z*_mode[nu][a] Z*_mode[nu][b] is
    // invariant under rotations within the degenerate subspace.
    // ------------------------------------------------------------------
    result_.has_borncharge = dielec->has_borncharge();
    if (result_.has_borncharge) {
        std::vector<std::vector<std::complex<double>>> zstar_mode(ns, std::vector<std::complex<double>>(3));
        dielec->compute_mode_effective_charge(zstar_mode, evec);
        for (auto ig = 0; ig < ngroup; ++ig) {
            auto &grp = result_.groups[ig];
            // S_ab = Re sum_nu Z_nu,a conj(Z_nu,b): invariant under both
            // eigenvector phase choices and unitary mixing in the multiplet.
            Eigen::Matrix3d s_tensor = Eigen::Matrix3d::Zero();
            for (const auto nu: grp.mode_indices) {
                for (auto a = 0; a < 3; ++a) {
                    for (auto b = 0; b < 3; ++b) {
                        s_tensor(a, b) += (zstar_mode[nu][a] * std::conj(zstar_mode[nu][b])).real();
                    }
                }
            }
            grp.ir_strength = s_tensor;
            grp.has_ir_strength = true;
            if (grp.activity_known && !grp.is_acoustic && !grp.ir_active && s_tensor.trace() > 1.0e-6) {
                warnings.emplace_back("mode group " + std::to_string(ig + 1) +
                                      " is IR-inactive by symmetry but has a nonzero "
                                      "oscillator strength: the Born effective charges are "
                                      "inconsistent with the crystal symmetry (consider "
                                      "BORNSYM = 1).");
            }
        }
    }

    // ------------------------------------------------------------------
    // Class list and total decompositions.
    // ------------------------------------------------------------------
    result_.classes.resize(ncl);
    for (auto ic = 0; ic < ncl; ++ic) {
        auto &ci = result_.classes[ic];
        ci.nelem = nelem_of_class[ic];
        if (labels_ok && pg) {
            ci.label = pg->classes[match.class_to_col[ic]].label;
            if (!match.convention_resolved &&
                std::find(match.ambiguous_cols.begin(), match.ambiguous_cols.end(), match.class_to_col[ic]) !=
                    match.ambiguous_cols.end())
            {
                ci.label += "?";
            }
        } else {
            ci.label = (nelem_of_class[ic] > 1 ? std::to_string(nelem_of_class[ic]) : "") +
                       kind_name(ops_info[classes[ic][0]].kind);
        }
        for (const auto im: classes[ic]) {
            const auto kind = ops_info[im].kind;
            if (kind != pointgroup::OpKind::E && kind != pointgroup::OpKind::I) {
                ci.axes.push_back(ops_info[im].axis);
            }
        }
    }

    if (labels_ok && pg) {
        const auto n_total =
            pointgroup::decompose_representation(*pg, match.class_to_col, nelem_of_class, chi_total_class);
        const auto n_vec = pointgroup::decompose_representation(*pg, match.class_to_col, nelem_of_class, trace_class);
        std::vector<int> n_total_int(pg->nclass), n_vec_int(pg->nclass), n_optic(pg->nclass);
        auto integral = true;
        for (auto mu = 0; mu < pg->nclass; ++mu) {
            n_total_int[mu] = static_cast<int>(std::lround(n_total[mu]));
            n_vec_int[mu] = static_cast<int>(std::lround(n_vec[mu]));
            n_optic[mu] = n_total_int[mu] - n_vec_int[mu];
            if (std::abs(n_total[mu] - n_total_int[mu]) > 1.0e-3 || std::abs(n_vec[mu] - n_vec_int[mu]) > 1.0e-3 ||
                n_total_int[mu] < 0 || n_optic[mu] < 0)
            {
                integral = false;
            }
        }
        if (integral) {
            result_.decomp_total = format_decomposition(*pg, n_total_int, irrep_names);
            result_.decomp_acoustic = format_decomposition(*pg, n_vec_int, irrep_names);
            result_.decomp_optic = format_decomposition(*pg, n_optic, irrep_names);
        } else {
            warnings.emplace_back("the total vibrational representation does not decompose "
                                  "into non-negative integer multiplicities; decomposition "
                                  "strings are suppressed.");
            labels_ok = false;
        }
    }

    if (labels_ok && pg) {
        result_.pg_number = pg_number;
        result_.pg_schoenflies = pg->schoenflies;
        result_.pg_international = pg->international;
        if (symmetry->has_spg_dataset) {
            result_.spg_symbol = symmetry->spg_symbol + " (#" + std::to_string(symmetry->spg_number) + ")";
        }
    }

    // A validation failure detected after labels were stored must not leave
    // stale Mulliken strings or decomposition strings behind.
    if (!labels_ok) {
        for (auto &grp: result_.groups) {
            grp.irrep_label.clear();
        }
        result_.decomp_total.clear();
        result_.decomp_acoustic.clear();
        result_.decomp_optic.clear();
        for (auto ic = 0; ic < ncl; ++ic) {
            result_.classes[ic].label = (nelem_of_class[ic] > 1 ? std::to_string(nelem_of_class[ic]) : "") +
                                        kind_name(ops_info[classes[ic][0]].kind);
        }
    }

    result_.available = labels_ok && closure_ok;
}
