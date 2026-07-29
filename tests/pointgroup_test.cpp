/*
 pointgroup_test.cpp

 Standalone unit test for anphon/pointgroup_data.h (the character-table
 subsystem of the IRREPS feature).  Not part of the regular build; compile
 and run with e.g.

   clang++ -std=c++17 -I<eigen3> -I../anphon tests/pointgroup_test.cpp \
       -o pointgroup_test && ./pointgroup_test

 For every one of the 32 crystallographic point groups the test
   1. generates the group by closure from Cartesian generator matrices,
   2. checks the operation-count fingerprint identification,
   3. converts to an (integer) lattice basis and builds the R~R^-1-merged
      conjugacy classes with exact integer arithmetic,
   4. matches classes to character-table columns with the geometric axis
      conventions,
   5. verifies the table algebra: chi(E) = dim, weighted row orthogonality
      sum_c w_c chi_mu chi_nu = |G| norm_mu delta_munu, the weighted
      dimension sum sum_mu dim^2/norm = |G|, the regular-representation
      multiplicities dim/norm, and non-negative integral vector-rep
      decomposition,
   6. verifies the stored IR/Raman activity flags against the projection
      formulas n_IR ~ sum_R chi_mu(R) tr R and
      n_Raman ~ sum_R chi_mu(R) (tr(R)^2 + tr(R^2))/2 evaluated on the
      generated matrices.
 It then repeats identification + matching for globally rotated copies of
 selected groups (orientation independence), an axis-permuted D2 setting,
 and checks that fingerprints are pairwise distinct and that the matcher
 flags unresolved conventions when no reference axis is supplied.
*/

#include "pointgroup_data.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Dense>

using namespace PHON_NS::pointgroup;

namespace
{

int nfail = 0;

void check(const bool cond, const std::string &msg)
{
    if (!cond) {
        std::printf("FAIL: %s\n", msg.c_str());
        ++nfail;
    }
}

// --- generators (Cartesian) -------------------------------------------------

Eigen::Matrix3d mat3(const double a00, const double a01, const double a02,
                     const double a10, const double a11, const double a12,
                     const double a20, const double a21, const double a22)
{
    Eigen::Matrix3d m;
    m << a00, a01, a02, a10, a11, a12, a20, a21, a22;
    return m;
}

const double S3 = std::sqrt(3.0) / 2.0;

const Eigen::Matrix3d g_inv = -Eigen::Matrix3d::Identity();
const Eigen::Matrix3d g_c2z = mat3(-1, 0, 0, 0, -1, 0, 0, 0, 1);
const Eigen::Matrix3d g_c2y = mat3(-1, 0, 0, 0, 1, 0, 0, 0, -1);
const Eigen::Matrix3d g_c2x = mat3(1, 0, 0, 0, -1, 0, 0, 0, -1);
const Eigen::Matrix3d g_c4z = mat3(0, -1, 0, 1, 0, 0, 0, 0, 1);
const Eigen::Matrix3d g_c3z = mat3(-0.5, -S3, 0, S3, -0.5, 0, 0, 0, 1);
const Eigen::Matrix3d g_c6z = mat3(0.5, -S3, 0, S3, 0.5, 0, 0, 0, 1);
const Eigen::Matrix3d g_mz = mat3(1, 0, 0, 0, 1, 0, 0, 0, -1);
const Eigen::Matrix3d g_my = mat3(1, 0, 0, 0, -1, 0, 0, 0, 1);
const Eigen::Matrix3d g_s4z = g_mz * g_c4z;
const Eigen::Matrix3d g_c3d = mat3(0, 0, 1, 1, 0, 0, 0, 1, 0);

struct GroupSpec
{
    int number;
    const char *name;
    std::vector<Eigen::Matrix3d> generators;
    bool hexagonal_lattice;
};

std::vector<GroupSpec> group_specs()
{
    return {
        {1, "C1", {}, false},
        {2, "Ci", {g_inv}, false},
        {3, "C2", {g_c2z}, false},
        {4, "Cs", {g_mz}, false},
        {5, "C2h", {g_c2z, g_inv}, false},
        {6, "D2", {g_c2z, g_c2x}, false},
        {7, "C2v", {g_c2z, g_my}, false},
        {8, "D2h", {g_c2z, g_c2x, g_inv}, false},
        {9, "C4", {g_c4z}, false},
        {10, "S4", {g_s4z}, false},
        {11, "C4h", {g_c4z, g_inv}, false},
        {12, "D4", {g_c4z, g_c2x}, false},
        {13, "C4v", {g_c4z, g_my}, false},
        {14, "D2d", {g_s4z, g_c2x}, false},
        {15, "D4h", {g_c4z, g_c2x, g_inv}, false},
        {16, "C3", {g_c3z}, true},
        {17, "C3i", {g_c3z, g_inv}, true},
        {18, "D3", {g_c3z, g_c2x}, true},
        {19, "C3v", {g_c3z, g_my}, true},
        {20, "D3d", {g_c3z, g_c2x, g_inv}, true},
        {21, "C6", {g_c6z}, true},
        {22, "C3h", {g_c3z, g_mz}, true},
        {23, "C6h", {g_c6z, g_inv}, true},
        {24, "D6", {g_c6z, g_c2x}, true},
        {25, "C6v", {g_c6z, g_my}, true},
        {26, "D3h", {g_c3z, g_c2x, g_mz}, true},
        {27, "D6h", {g_c6z, g_c2x, g_inv}, true},
        {28, "T", {g_c2z, g_c2x, g_c3d}, false},
        {29, "Th", {g_c2z, g_c2x, g_c3d, g_inv}, false},
        {30, "O", {g_c4z, g_c3d}, false},
        {31, "Td", {g_s4z, g_c3d}, false},
        {32, "Oh", {g_c4z, g_c3d, g_inv}, false},
    };
}

// Close a set of generators into a full group (matrix entries are rounded to
// a 1e-8 grid for identification).
std::vector<Eigen::Matrix3d> close_group(const std::vector<Eigen::Matrix3d> &gens)
{
    auto key_of = [](const Eigen::Matrix3d &m) {
        std::array<long long, 9> key{};
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                key[3 * i + j] = std::llround(m(i, j) * 1.0e8);
            }
        }
        return key;
    };

    std::vector<Eigen::Matrix3d> ops{Eigen::Matrix3d::Identity()};
    std::map<std::array<long long, 9>, int> seen;
    seen[key_of(ops[0])] = 0;

    bool grew = true;
    while (grew) {
        grew = false;
        const auto n = ops.size();
        for (std::size_t i = 0; i < n; ++i) {
            for (const auto &g: gens) {
                const Eigen::Matrix3d prod = g * ops[i];
                const auto key = key_of(prod);
                if (seen.find(key) == seen.end()) {
                    seen[key] = static_cast<int>(ops.size());
                    ops.push_back(prod);
                    grew = true;
                }
            }
        }
        if (ops.size() > 48) {
            std::printf("FAIL: group closure exceeded 48 elements\n");
            std::exit(1);
        }
    }
    return ops;
}

Eigen::Matrix3d lattice_for(const bool hexagonal)
{
    if (!hexagonal) return Eigen::Matrix3d::Identity();
    Eigen::Matrix3d a;
    // Columns: a1, a2, c of a hexagonal cell.
    a << 1.0, -0.5, 0.0,
         0.0, S3 * 2.0 / 2.0, 0.0,
         0.0, 0.0, 1.0;
    return a;
}

// Convert Cartesian ops to the (integer) lattice basis.
bool to_lattice_basis(const std::vector<Eigen::Matrix3d> &ops_cart, const Eigen::Matrix3d &lavec,
                      std::vector<Eigen::Matrix3i> &ops_latt)
{
    const Eigen::Matrix3d ainv = lavec.inverse();
    ops_latt.clear();
    for (const auto &r: ops_cart) {
        const Eigen::Matrix3d t = ainv * r * lavec;
        Eigen::Matrix3i ti;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                const auto rounded = std::lround(t(i, j));
                if (std::abs(t(i, j) - static_cast<double>(rounded)) > 1.0e-8) return false;
                ti(i, j) = static_cast<int>(rounded);
            }
        }
        ops_latt.push_back(ti);
    }
    return true;
}

// Full verification of one oriented setting of one group.
void run_group_checks(const GroupSpec &spec, const Eigen::Matrix3d &orient, const std::string &tag)
{
    const auto &pg = pg_table[spec.number - 1];
    check(pg.number == spec.number, tag + ": table numbering");

    // 1. group closure in the requested orientation
    std::vector<Eigen::Matrix3d> gens;
    gens.reserve(spec.generators.size());
    for (const auto &g: spec.generators) {
        gens.push_back(orient * g * orient.transpose());
    }
    const auto ops = close_group(gens);
    check(static_cast<int>(ops.size()) == pg.order, tag + ": group order");

    // 2. fingerprint identification
    check(identify_point_group_fingerprint(ops) == spec.number, tag + ": fingerprint id");

    // 3. integer lattice basis + merged conjugacy classes
    const Eigen::Matrix3d lavec = orient * lattice_for(spec.hexagonal_lattice);
    std::vector<Eigen::Matrix3i> ops_latt;
    check(to_lattice_basis(ops, lavec, ops_latt), tag + ": integer lattice conversion");
    const auto classes = conjugacy_classes_merged(ops_latt);
    check(static_cast<int>(classes.size()) == pg.nclass, tag + ": number of merged classes");
    if (static_cast<int>(classes.size()) != pg.nclass) return;

    // 4. class -> column matching
    std::vector<OpInfo> info(ops.size());
    for (std::size_t i = 0; i < ops.size(); ++i) {
        check(classify_op(ops[i], info[i]), tag + ": classify_op");
    }
    const Eigen::Vector3d zhat = orient * Eigen::Vector3d(0, 0, 1);
    const Eigen::Vector3d xhat = orient * Eigen::Vector3d(1, 0, 0);
    const auto match = match_classes_to_columns(pg, classes, info, zhat, xhat);
    check(match.ok, tag + ": matcher ok");
    check(match.convention_resolved, tag + ": convention resolved");
    if (!match.ok) return;

    const auto ncl = pg.nclass;
    std::vector<int> nelem(ncl);
    std::vector<double> traces(ncl);
    for (int ic = 0; ic < ncl; ++ic) {
        nelem[ic] = static_cast<int>(classes[ic].size());
        traces[ic] = ops[classes[ic][0]].trace();
        const auto col = match.class_to_col[ic];
        check(pg.classes[col].kind == info[classes[ic][0]].kind, tag + ": matched kind");
        check(pg.classes[col].nelem == nelem[ic], tag + ": matched class size");
        // All members of a merged class share the trace (real class function).
        for (const auto im: classes[ic]) {
            check(std::abs(ops[im].trace() - traces[ic]) < 1.0e-8, tag + ": class trace consistency");
        }
    }

    // 5. table algebra
    int e_col = -1;
    for (int ic = 0; ic < ncl; ++ic) {
        if (pg.classes[ic].kind == OpKind::E) e_col = ic;
    }
    check(e_col >= 0, tag + ": E column exists");

    int dimsum = 0;
    for (int mu = 0; mu < ncl; ++mu) {
        check(pg.chartab[mu * ncl + e_col] == pg.irreps[mu].dim, tag + ": chi(E) = dim");
        check(pg.irreps[mu].dim * pg.irreps[mu].dim % pg.irreps[mu].norm == 0,
              tag + ": dim^2 divisible by norm");
        dimsum += pg.irreps[mu].dim * pg.irreps[mu].dim / pg.irreps[mu].norm;
    }
    check(dimsum == pg.order, tag + ": sum dim^2/norm = |G|");

    for (int mu = 0; mu < ncl; ++mu) {
        for (int nu = 0; nu < ncl; ++nu) {
            int s = 0;
            for (int ic = 0; ic < ncl; ++ic) {
                s += pg.classes[ic].nelem * pg.chartab[mu * ncl + ic] * pg.chartab[nu * ncl + ic];
            }
            const int expected = (mu == nu) ? pg.order * pg.irreps[mu].norm : 0;
            check(s == expected, tag + ": weighted row orthogonality (" + pg.irreps[mu].mulliken
                  + ", " + pg.irreps[nu].mulliken + ")");
        }
    }

    // Regular representation: n_mu = dim/norm.
    {
        std::vector<double> chi_reg(ncl, 0.0);
        for (int ic = 0; ic < ncl; ++ic) {
            if (match.class_to_col[ic] == e_col) chi_reg[ic] = static_cast<double>(pg.order);
        }
        const auto n_reg = decompose_representation(pg, match.class_to_col, nelem, chi_reg);
        for (int mu = 0; mu < ncl; ++mu) {
            const double expected = static_cast<double>(pg.irreps[mu].dim) / pg.irreps[mu].norm;
            check(std::abs(n_reg[mu] - expected) < 1.0e-8, tag + ": regular-rep multiplicity of "
                  + pg.irreps[mu].mulliken);
        }
    }

    // Vector rep decomposes integrally.
    check(validate_assignment_by_vector_rep(pg, match.class_to_col, nelem, traces),
          tag + ": vector rep integral");

    // 6. activity flags == projection formulas, evaluated on the matrices
    std::vector<int> class_of_op(ops.size(), -1);
    for (int ic = 0; ic < ncl; ++ic) {
        for (const auto im: classes[ic]) class_of_op[im] = ic;
    }
    for (int mu = 0; mu < ncl; ++mu) {
        double n_ir = 0.0, n_raman = 0.0;
        for (std::size_t j = 0; j < ops.size(); ++j) {
            const auto chi = static_cast<double>(
                pg.chartab[mu * ncl + match.class_to_col[class_of_op[j]]]);
            const double tr = ops[j].trace();
            const double tr2 = (ops[j] * ops[j]).trace();
            n_ir += chi * tr;
            n_raman += chi * 0.5 * (tr * tr + tr2);
        }
        n_ir /= static_cast<double>(pg.order);
        n_raman /= static_cast<double>(pg.order);
        check(n_ir > -1.0e-8, tag + ": n_IR non-negative for " + pg.irreps[mu].mulliken);
        check(n_raman > -1.0e-8, tag + ": n_Raman non-negative for " + pg.irreps[mu].mulliken);
        check(std::abs(n_ir - std::lround(n_ir)) < 1.0e-8,
              tag + ": n_IR integral for " + pg.irreps[mu].mulliken);
        check(std::abs(n_raman - std::lround(n_raman)) < 1.0e-8,
              tag + ": n_Raman integral for " + pg.irreps[mu].mulliken);
        check((n_ir > 1.0e-8) == pg.irreps[mu].ir_active,
              tag + ": IR flag of " + pg.irreps[mu].mulliken);
        check((n_raman > 1.0e-8) == pg.irreps[mu].raman_active,
              tag + ": Raman flag of " + pg.irreps[mu].mulliken);
    }
}

} // namespace

int main()
{
    const auto specs = group_specs();

    // Pairwise-distinct fingerprints.
    for (int i = 0; i < 32; ++i) {
        for (int j = i + 1; j < 32; ++j) {
            check(fingerprint_of_table(pg_table[i]) != fingerprint_of_table(pg_table[j]),
                  std::string("fingerprint uniqueness: ") + pg_table[i].schoenflies + " vs "
                  + pg_table[j].schoenflies);
        }
    }

    // Standard orientation, all 32 groups.
    for (const auto &spec: specs) {
        run_group_checks(spec, Eigen::Matrix3d::Identity(), spec.name);
    }

    // Globally rotated settings for the convention-sensitive groups.
    const Eigen::Matrix3d rot =
        (Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ())
         * Eigen::AngleAxisd(0.7, Eigen::Vector3d::UnitY())
         * Eigen::AngleAxisd(1.1, Eigen::Vector3d::UnitX())).toRotationMatrix();
    for (const auto &spec: specs) {
        switch (spec.number) {
            case 6: case 7: case 8: case 12: case 13: case 14: case 15:
            case 18: case 19: case 20: case 24: case 25: case 26: case 27:
            case 29: case 31: case 32:
                run_group_checks(spec, rot, std::string(spec.name) + " (rotated)");
                break;
            default:
                break;
        }
    }

    // Axis-permuted D2 setting: principal axis along x.
    {
        GroupSpec d2_perm{6, "D2 (permuted)", {g_c2x, g_c2y}, false};
        const Eigen::Matrix3d perm =
            (Eigen::Matrix3d() << 0, 0, 1, 1, 0, 0, 0, 1, 0).finished(); // maps z->x
        run_group_checks(d2_perm, perm, "D2 (axis-permuted)");
    }

    // Physical convention anchors: sigma_v = mirrors whose NORMALS lie along
    // the conventional <100> star (International-Tables position-2 mirrors).
    // In C6v/D6h the mirror with normal || x must land in the sigma_v column
    // (this is what makes the silent wurtzite modes B1); in D4h the mirror
    // with normal || y is sigma_v (both readings coincide in tetragonal).
    {
        struct Anchor
        {
            int pg_index;
            std::vector<Eigen::Matrix3d> gens;
            Eigen::Matrix3d probe; // mirror whose class label is checked
            const char *expected;
            bool hexagonal;
        };
        const Eigen::Matrix3d g_mx = mat3(-1, 0, 0, 0, 1, 0, 0, 0, 1);
        const std::vector<Anchor> anchors = {
            {25, {g_c6z, g_my}, g_mx, "3sigma_v", true},   // C6v
            {25, {g_c6z, g_my}, g_my, "3sigma_d", true},   // C6v
            {27, {g_c6z, g_c2x, g_inv}, g_mx, "3sigma_v", true},  // D6h
            {13, {g_c4z, g_my}, g_my, "2sigma_v", false},  // C4v
            {15, {g_c4z, g_c2x, g_inv}, g_my, "2sigma_v", false}, // D4h
        };
        for (const auto &anchor: anchors) {
            const auto &pg = pg_table[anchor.pg_index - 1];
            const auto ops = close_group(anchor.gens);
            std::vector<Eigen::Matrix3i> ops_latt;
            check(to_lattice_basis(ops, lattice_for(anchor.hexagonal), ops_latt),
                  "anchor lattice conversion");
            const auto classes = conjugacy_classes_merged(ops_latt);
            std::vector<OpInfo> info(ops.size());
            for (std::size_t i = 0; i < ops.size(); ++i) {
                classify_op(ops[i], info[i]);
            }
            const auto match = match_classes_to_columns(pg, classes, info,
                                                        Eigen::Vector3d(0, 0, 1),
                                                        Eigen::Vector3d(1, 0, 0));
            check(match.ok && match.convention_resolved,
                  std::string(pg.schoenflies) + " anchor: matcher resolved");
            int probe_op = -1;
            for (std::size_t i = 0; i < ops.size(); ++i) {
                if ((ops[i] - anchor.probe).norm() < 1.0e-8) probe_op = static_cast<int>(i);
            }
            check(probe_op >= 0, std::string(pg.schoenflies) + " anchor: probe op found");
            int probe_class = -1;
            for (std::size_t ic = 0; ic < classes.size(); ++ic) {
                for (const auto im: classes[ic]) {
                    if (im == probe_op) probe_class = static_cast<int>(ic);
                }
            }
            check(probe_class >= 0 && std::string(pg.classes[match.class_to_col[probe_class]].label)
                  == anchor.expected,
                  std::string(pg.schoenflies) + " anchor: probe mirror lands in "
                  + anchor.expected);
        }
    }

    // Without a reference in-plane axis the convention-dependent columns of
    // D4h must be flagged unresolved.
    {
        const auto &pg = pg_table[14]; // D4h
        const auto ops = close_group({g_c4z, g_c2x, g_inv});
        std::vector<Eigen::Matrix3i> ops_latt;
        to_lattice_basis(ops, Eigen::Matrix3d::Identity(), ops_latt);
        const auto classes = conjugacy_classes_merged(ops_latt);
        std::vector<OpInfo> info(ops.size());
        for (std::size_t i = 0; i < ops.size(); ++i) classify_op(ops[i], info[i]);
        const auto match = match_classes_to_columns(pg, classes, info,
                                                    Eigen::Vector3d(0, 0, 1),
                                                    Eigen::Vector3d::Zero());
        check(match.ok, "D4h no-xref: matcher ok");
        check(!match.convention_resolved, "D4h no-xref: convention flagged unresolved");
        check(!match.ambiguous_cols.empty(), "D4h no-xref: ambiguous columns reported");
    }

    if (nfail == 0) {
        std::printf("All pointgroup_data.h checks passed.\n");
        return 0;
    }
    std::printf("%d check(s) FAILED.\n", nfail);
    return 1;
}
