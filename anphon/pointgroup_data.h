/*
 pointgroup_data.h

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

/*
 Character tables and point-group utilities for the Gamma-point irrep
 analysis (IRREPS tag).

 The tables use physically irreducible *real* representations: the
 complex-conjugate 1D irrep pairs of C3, C4, C6, S4, S6, C3h, C4h, C6h,
 T, and Th are merged into 2-dimensional E-type irreps with summed
 characters (the appropriate reduction for time-reversal-symmetric
 phonons), and conjugacy classes are additionally merged under
 R ~ R^-1 (real characters satisfy chi(R) = chi(R^-1)).  With both
 normalizations the character matrix is square and integer for every
 crystallographic point group.  A merged irrep carries a character norm
 <chi,chi> = 2*|G| instead of |G|; the Frobenius-Schur factor is stored
 per irrep as IrrepDesc::norm and every projection divides by it:

   n_mu(lambda) = (1/(|G| * norm_mu)) sum_R chi_lambda(R) chi_mu(R)

 The classic identities become weighted accordingly:
   sum_mu dim_mu^2 / norm_mu = |G|,
   sum_c w_c chi_mu(c) chi_nu(c) = |G| * norm_mu * delta_mu,nu.

 All table content is derived from elementary group theory and is
 machine-verified by the standalone unit test (orthogonality, weighted
 dimension sum, activity flags against the projection formulas, and
 characters against rotation-matrix traces of generated groups).
 Standard references for conventions (Mulliken symbols, class labels):
 F. A. Cotton, "Chemical Applications of Group Theory"; C. J. Bradley
 and A. P. Cracknell, "The Mathematical Theory of Symmetry in Solids".

 Axis conventions for the convention-dependent labels (B1/B2-type):
   - z^ = principal axis (axis of the rotation of highest order, proper
     or improper; in D2d this is the S4 axis).
   - x^ = reference in-plane direction (conventional a axis projected
     perpendicular to z^), supplied by the caller.
   - The "primary star" is the orbit of x^ under the proper rotations
     about z^ (the conventional <100> in-plane directions).
   - A rotation class tagged PerpPrimary (C2') has its axes along the
     primary star; PerpSecondary (C2'') along the in-plane diagonals.
   - A mirror class tagged PerpPrimary (sigma_v) has its NORMALS along
     the primary star (the International-Tables position-2 mirrors,
     m perpendicular to <100>; in D6h this is the i*C2' class), matching
     the standard labels, e.g. the silent wurtzite modes being B1 in
     C6v.  PerpSecondary (sigma_d) normals lie along the diagonals.
     In the tetragonal groups the two readings coincide (the sigma_v
     planes both contain one <100> axis and have their normal along the
     other).
   - AxisX/AxisY/AxisZ (D2, C2v, C2h setting helpers): a rotation class
     has its axis along that direction; a mirror class has its NORMAL
     along that direction.
*/

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace PHON_NS
{
namespace pointgroup
{

enum class OpKind : int
{
    E = 0, C2, C3, C4, C6, I, Sigma, S6, S4, S3
};

constexpr int NKIND = 10;

enum class AxisTag : int
{
    None = 0, Principal, PerpPrimary, PerpSecondary, AxisZ, AxisX, AxisY
};

struct ClassDesc
{
    const char *label;
    OpKind kind;
    int nelem;
    AxisTag tag;
};

struct IrrepDesc
{
    const char *mulliken;
    int dim;   // physical dimension
    int norm;  // Frobenius-Schur merge count: 1, or 2 for a merged conjugate pair
    bool ir_active;
    bool raman_active;
};

struct PointGroup
{
    int number;                // spglib point-group number, 1..32
    const char *schoenflies;
    const char *international;
    int order;
    int nclass;                // == number of irreps (tables are square)
    const ClassDesc *classes;
    const IrrepDesc *irreps;
    const int8_t *chartab;     // [nclass * nclass], row-major, rows = irreps
};

// ---------------------------------------------------------------------------
// Table data.  Column order of chartab rows follows the classes array of the
// same group.  Activity flags: ir = irrep appears in the vector rep Gamma_V,
// raman = appears in the symmetric square [Gamma_V x Gamma_V].
// ---------------------------------------------------------------------------

namespace detail
{

using K = OpKind;
using T = AxisTag;

// 1: C1 (1)
inline const ClassDesc cls_C1[] = {{"E", K::E, 1, T::None}};
inline const IrrepDesc irr_C1[] = {{"A", 1, 1, true, true}};
inline const int8_t chi_C1[] = {1};

// 2: Ci (-1)
inline const ClassDesc cls_Ci[] = {{"E", K::E, 1, T::None}, {"i", K::I, 1, T::None}};
inline const IrrepDesc irr_Ci[] = {{"Ag", 1, 1, false, true},
                                   {"Au", 1, 1, true, false}};
inline const int8_t chi_Ci[] = {1, 1,
                                1, -1};

// 3: C2 (2)
inline const ClassDesc cls_C2[] = {{"E", K::E, 1, T::None}, {"C2", K::C2, 1, T::Principal}};
inline const IrrepDesc irr_C2[] = {{"A", 1, 1, true, true},
                                   {"B", 1, 1, true, true}};
inline const int8_t chi_C2[] = {1, 1,
                                1, -1};

// 4: Cs (m)
inline const ClassDesc cls_Cs[] = {{"E", K::E, 1, T::None}, {"sigma_h", K::Sigma, 1, T::Principal}};
inline const IrrepDesc irr_Cs[] = {{"A'", 1, 1, true, true},
                                   {"A''", 1, 1, true, true}};
inline const int8_t chi_Cs[] = {1, 1,
                                1, -1};

// 5: C2h (2/m)
inline const ClassDesc cls_C2h[] = {{"E", K::E, 1, T::None},
                                    {"C2", K::C2, 1, T::Principal},
                                    {"i", K::I, 1, T::None},
                                    {"sigma_h", K::Sigma, 1, T::Principal}};
inline const IrrepDesc irr_C2h[] = {{"Ag", 1, 1, false, true},
                                    {"Bg", 1, 1, false, true},
                                    {"Au", 1, 1, true, false},
                                    {"Bu", 1, 1, true, false}};
inline const int8_t chi_C2h[] = {1, 1, 1, 1,
                                 1, -1, 1, -1,
                                 1, 1, -1, -1,
                                 1, -1, -1, 1};

// 6: D2 (222)
inline const ClassDesc cls_D2[] = {{"E", K::E, 1, T::None},
                                   {"C2(z)", K::C2, 1, T::AxisZ},
                                   {"C2(y)", K::C2, 1, T::AxisY},
                                   {"C2(x)", K::C2, 1, T::AxisX}};
inline const IrrepDesc irr_D2[] = {{"A", 1, 1, false, true},
                                   {"B1", 1, 1, true, true},
                                   {"B2", 1, 1, true, true},
                                   {"B3", 1, 1, true, true}};
inline const int8_t chi_D2[] = {1, 1, 1, 1,
                                1, 1, -1, -1,
                                1, -1, 1, -1,
                                1, -1, -1, 1};

// 7: C2v (mm2).  sigma_v(xz) has normal y, sigma_v'(yz) has normal x.
inline const ClassDesc cls_C2v[] = {{"E", K::E, 1, T::None},
                                    {"C2", K::C2, 1, T::AxisZ},
                                    {"sigma_v(xz)", K::Sigma, 1, T::AxisY},
                                    {"sigma_v'(yz)", K::Sigma, 1, T::AxisX}};
inline const IrrepDesc irr_C2v[] = {{"A1", 1, 1, true, true},
                                    {"A2", 1, 1, false, true},
                                    {"B1", 1, 1, true, true},
                                    {"B2", 1, 1, true, true}};
inline const int8_t chi_C2v[] = {1, 1, 1, 1,
                                 1, 1, -1, -1,
                                 1, -1, 1, -1,
                                 1, -1, -1, 1};

// 8: D2h (mmm).  Mirrors tagged by their normals.
inline const ClassDesc cls_D2h[] = {{"E", K::E, 1, T::None},
                                    {"C2(z)", K::C2, 1, T::AxisZ},
                                    {"C2(y)", K::C2, 1, T::AxisY},
                                    {"C2(x)", K::C2, 1, T::AxisX},
                                    {"i", K::I, 1, T::None},
                                    {"sigma(xy)", K::Sigma, 1, T::AxisZ},
                                    {"sigma(xz)", K::Sigma, 1, T::AxisY},
                                    {"sigma(yz)", K::Sigma, 1, T::AxisX}};
inline const IrrepDesc irr_D2h[] = {{"Ag", 1, 1, false, true},
                                    {"B1g", 1, 1, false, true},
                                    {"B2g", 1, 1, false, true},
                                    {"B3g", 1, 1, false, true},
                                    {"Au", 1, 1, false, false},
                                    {"B1u", 1, 1, true, false},
                                    {"B2u", 1, 1, true, false},
                                    {"B3u", 1, 1, true, false}};
inline const int8_t chi_D2h[] = {1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, -1, -1, 1, 1, -1, -1,
                                 1, -1, 1, -1, 1, -1, 1, -1,
                                 1, -1, -1, 1, 1, -1, -1, 1,
                                 1, 1, 1, 1, -1, -1, -1, -1,
                                 1, 1, -1, -1, -1, -1, 1, 1,
                                 1, -1, 1, -1, -1, 1, -1, 1,
                                 1, -1, -1, 1, -1, 1, 1, -1};

// 9: C4 (4)
inline const ClassDesc cls_C4[] = {{"E", K::E, 1, T::None},
                                   {"C2", K::C2, 1, T::Principal},
                                   {"2C4", K::C4, 2, T::Principal}};
inline const IrrepDesc irr_C4[] = {{"A", 1, 1, true, true},
                                   {"B", 1, 1, false, true},
                                   {"E", 2, 2, true, true}};
inline const int8_t chi_C4[] = {1, 1, 1,
                                1, 1, -1,
                                2, -2, 0};

// 10: S4 (-4)
inline const ClassDesc cls_S4[] = {{"E", K::E, 1, T::None},
                                   {"C2", K::C2, 1, T::Principal},
                                   {"2S4", K::S4, 2, T::Principal}};
inline const IrrepDesc irr_S4[] = {{"A", 1, 1, false, true},
                                   {"B", 1, 1, true, true},
                                   {"E", 2, 2, true, true}};
inline const int8_t chi_S4[] = {1, 1, 1,
                                1, 1, -1,
                                2, -2, 0};

// 11: C4h (4/m)
inline const ClassDesc cls_C4h[] = {{"E", K::E, 1, T::None},
                                    {"2C4", K::C4, 2, T::Principal},
                                    {"C2", K::C2, 1, T::Principal},
                                    {"i", K::I, 1, T::None},
                                    {"2S4", K::S4, 2, T::Principal},
                                    {"sigma_h", K::Sigma, 1, T::Principal}};
inline const IrrepDesc irr_C4h[] = {{"Ag", 1, 1, false, true},
                                    {"Bg", 1, 1, false, true},
                                    {"Eg", 2, 2, false, true},
                                    {"Au", 1, 1, true, false},
                                    {"Bu", 1, 1, false, false},
                                    {"Eu", 2, 2, true, false}};
inline const int8_t chi_C4h[] = {1, 1, 1, 1, 1, 1,
                                 1, -1, 1, 1, -1, 1,
                                 2, 0, -2, 2, 0, -2,
                                 1, 1, 1, -1, -1, -1,
                                 1, -1, 1, -1, 1, -1,
                                 2, 0, -2, -2, 0, 2};

// 12: D4 (422)
inline const ClassDesc cls_D4[] = {{"E", K::E, 1, T::None},
                                   {"2C4", K::C4, 2, T::Principal},
                                   {"C2", K::C2, 1, T::Principal},
                                   {"2C2'", K::C2, 2, T::PerpPrimary},
                                   {"2C2''", K::C2, 2, T::PerpSecondary}};
inline const IrrepDesc irr_D4[] = {{"A1", 1, 1, false, true},
                                   {"A2", 1, 1, true, false},
                                   {"B1", 1, 1, false, true},
                                   {"B2", 1, 1, false, true},
                                   {"E", 2, 1, true, true}};
inline const int8_t chi_D4[] = {1, 1, 1, 1, 1,
                                1, 1, 1, -1, -1,
                                1, -1, 1, 1, -1,
                                1, -1, 1, -1, 1,
                                2, 0, -2, 0, 0};

// 13: C4v (4mm)
inline const ClassDesc cls_C4v[] = {{"E", K::E, 1, T::None},
                                    {"2C4", K::C4, 2, T::Principal},
                                    {"C2", K::C2, 1, T::Principal},
                                    {"2sigma_v", K::Sigma, 2, T::PerpPrimary},
                                    {"2sigma_d", K::Sigma, 2, T::PerpSecondary}};
inline const IrrepDesc irr_C4v[] = {{"A1", 1, 1, true, true},
                                    {"A2", 1, 1, false, false},
                                    {"B1", 1, 1, false, true},
                                    {"B2", 1, 1, false, true},
                                    {"E", 2, 1, true, true}};
inline const int8_t chi_C4v[] = {1, 1, 1, 1, 1,
                                 1, 1, 1, -1, -1,
                                 1, -1, 1, 1, -1,
                                 1, -1, 1, -1, 1,
                                 2, 0, -2, 0, 0};

// 14: D2d (-42m).  Standard setting: C2' along x,y; sigma_d diagonal.
inline const ClassDesc cls_D2d[] = {{"E", K::E, 1, T::None},
                                    {"2S4", K::S4, 2, T::Principal},
                                    {"C2", K::C2, 1, T::Principal},
                                    {"2C2'", K::C2, 2, T::PerpPrimary},
                                    {"2sigma_d", K::Sigma, 2, T::PerpSecondary}};
inline const IrrepDesc irr_D2d[] = {{"A1", 1, 1, false, true},
                                    {"A2", 1, 1, false, false},
                                    {"B1", 1, 1, false, true},
                                    {"B2", 1, 1, true, true},
                                    {"E", 2, 1, true, true}};
inline const int8_t chi_D2d[] = {1, 1, 1, 1, 1,
                                 1, 1, 1, -1, -1,
                                 1, -1, 1, 1, -1,
                                 1, -1, 1, -1, 1,
                                 2, 0, -2, 0, 0};

// 15: D4h (4/mmm)
inline const ClassDesc cls_D4h[] = {{"E", K::E, 1, T::None},
                                    {"2C4", K::C4, 2, T::Principal},
                                    {"C2", K::C2, 1, T::Principal},
                                    {"2C2'", K::C2, 2, T::PerpPrimary},
                                    {"2C2''", K::C2, 2, T::PerpSecondary},
                                    {"i", K::I, 1, T::None},
                                    {"2S4", K::S4, 2, T::Principal},
                                    {"sigma_h", K::Sigma, 1, T::Principal},
                                    {"2sigma_v", K::Sigma, 2, T::PerpPrimary},
                                    {"2sigma_d", K::Sigma, 2, T::PerpSecondary}};
inline const IrrepDesc irr_D4h[] = {{"A1g", 1, 1, false, true},
                                    {"A2g", 1, 1, false, false},
                                    {"B1g", 1, 1, false, true},
                                    {"B2g", 1, 1, false, true},
                                    {"Eg", 2, 1, false, true},
                                    {"A1u", 1, 1, false, false},
                                    {"A2u", 1, 1, true, false},
                                    {"B1u", 1, 1, false, false},
                                    {"B2u", 1, 1, false, false},
                                    {"Eu", 2, 1, true, false}};
inline const int8_t chi_D4h[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, -1, -1, 1, 1, 1, -1, -1,
                                 1, -1, 1, 1, -1, 1, -1, 1, 1, -1,
                                 1, -1, 1, -1, 1, 1, -1, 1, -1, 1,
                                 2, 0, -2, 0, 0, 2, 0, -2, 0, 0,
                                 1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
                                 1, 1, 1, -1, -1, -1, -1, -1, 1, 1,
                                 1, -1, 1, 1, -1, -1, 1, -1, -1, 1,
                                 1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
                                 2, 0, -2, 0, 0, -2, 0, 2, 0, 0};

// 16: C3 (3)
inline const ClassDesc cls_C3[] = {{"E", K::E, 1, T::None},
                                   {"2C3", K::C3, 2, T::Principal}};
inline const IrrepDesc irr_C3[] = {{"A", 1, 1, true, true},
                                   {"E", 2, 2, true, true}};
inline const int8_t chi_C3[] = {1, 1,
                                2, -1};

// 17: C3i / S6 (-3)
inline const ClassDesc cls_S6[] = {{"E", K::E, 1, T::None},
                                   {"2C3", K::C3, 2, T::Principal},
                                   {"i", K::I, 1, T::None},
                                   {"2S6", K::S6, 2, T::Principal}};
inline const IrrepDesc irr_S6[] = {{"Ag", 1, 1, false, true},
                                   {"Eg", 2, 2, false, true},
                                   {"Au", 1, 1, true, false},
                                   {"Eu", 2, 2, true, false}};
inline const int8_t chi_S6[] = {1, 1, 1, 1,
                                2, -1, 2, -1,
                                1, 1, -1, -1,
                                2, -1, -2, 1};

// 18: D3 (32)
inline const ClassDesc cls_D3[] = {{"E", K::E, 1, T::None},
                                   {"2C3", K::C3, 2, T::Principal},
                                   {"3C2", K::C2, 3, T::PerpPrimary}};
inline const IrrepDesc irr_D3[] = {{"A1", 1, 1, false, true},
                                   {"A2", 1, 1, true, false},
                                   {"E", 2, 1, true, true}};
inline const int8_t chi_D3[] = {1, 1, 1,
                                1, 1, -1,
                                2, -1, 0};

// 19: C3v (3m)
inline const ClassDesc cls_C3v[] = {{"E", K::E, 1, T::None},
                                    {"2C3", K::C3, 2, T::Principal},
                                    {"3sigma_v", K::Sigma, 3, T::PerpPrimary}};
inline const IrrepDesc irr_C3v[] = {{"A1", 1, 1, true, true},
                                    {"A2", 1, 1, false, false},
                                    {"E", 2, 1, true, true}};
inline const int8_t chi_C3v[] = {1, 1, 1,
                                 1, 1, -1,
                                 2, -1, 0};

// 20: D3d (-3m)
inline const ClassDesc cls_D3d[] = {{"E", K::E, 1, T::None},
                                    {"2C3", K::C3, 2, T::Principal},
                                    {"3C2", K::C2, 3, T::PerpPrimary},
                                    {"i", K::I, 1, T::None},
                                    {"2S6", K::S6, 2, T::Principal},
                                    {"3sigma_d", K::Sigma, 3, T::PerpSecondary}};
inline const IrrepDesc irr_D3d[] = {{"A1g", 1, 1, false, true},
                                    {"A2g", 1, 1, false, false},
                                    {"Eg", 2, 1, false, true},
                                    {"A1u", 1, 1, false, false},
                                    {"A2u", 1, 1, true, false},
                                    {"Eu", 2, 1, true, false}};
inline const int8_t chi_D3d[] = {1, 1, 1, 1, 1, 1,
                                 1, 1, -1, 1, 1, -1,
                                 2, -1, 0, 2, -1, 0,
                                 1, 1, 1, -1, -1, -1,
                                 1, 1, -1, -1, -1, 1,
                                 2, -1, 0, -2, 1, 0};

// 21: C6 (6)
inline const ClassDesc cls_C6[] = {{"E", K::E, 1, T::None},
                                   {"2C6", K::C6, 2, T::Principal},
                                   {"2C3", K::C3, 2, T::Principal},
                                   {"C2", K::C2, 1, T::Principal}};
inline const IrrepDesc irr_C6[] = {{"A", 1, 1, true, true},
                                   {"B", 1, 1, false, false},
                                   {"E1", 2, 2, true, true},
                                   {"E2", 2, 2, false, true}};
inline const int8_t chi_C6[] = {1, 1, 1, 1,
                                1, -1, 1, -1,
                                2, 1, -1, -2,
                                2, -1, -1, 2};

// 22: C3h (-6)
inline const ClassDesc cls_C3h[] = {{"E", K::E, 1, T::None},
                                    {"2C3", K::C3, 2, T::Principal},
                                    {"sigma_h", K::Sigma, 1, T::Principal},
                                    {"2S3", K::S3, 2, T::Principal}};
inline const IrrepDesc irr_C3h[] = {{"A'", 1, 1, false, true},
                                    {"A''", 1, 1, true, false},
                                    {"E'", 2, 2, true, true},
                                    {"E''", 2, 2, false, true}};
inline const int8_t chi_C3h[] = {1, 1, 1, 1,
                                 1, 1, -1, -1,
                                 2, -1, 2, -1,
                                 2, -1, -2, 1};

// 23: C6h (6/m)
inline const ClassDesc cls_C6h[] = {{"E", K::E, 1, T::None},
                                    {"2C6", K::C6, 2, T::Principal},
                                    {"2C3", K::C3, 2, T::Principal},
                                    {"C2", K::C2, 1, T::Principal},
                                    {"i", K::I, 1, T::None},
                                    {"2S3", K::S3, 2, T::Principal},
                                    {"2S6", K::S6, 2, T::Principal},
                                    {"sigma_h", K::Sigma, 1, T::Principal}};
inline const IrrepDesc irr_C6h[] = {{"Ag", 1, 1, false, true},
                                    {"Bg", 1, 1, false, false},
                                    {"E1g", 2, 2, false, true},
                                    {"E2g", 2, 2, false, true},
                                    {"Au", 1, 1, true, false},
                                    {"Bu", 1, 1, false, false},
                                    {"E1u", 2, 2, true, false},
                                    {"E2u", 2, 2, false, false}};
inline const int8_t chi_C6h[] = {1, 1, 1, 1, 1, 1, 1, 1,
                                 1, -1, 1, -1, 1, -1, 1, -1,
                                 2, 1, -1, -2, 2, 1, -1, -2,
                                 2, -1, -1, 2, 2, -1, -1, 2,
                                 1, 1, 1, 1, -1, -1, -1, -1,
                                 1, -1, 1, -1, -1, 1, -1, 1,
                                 2, 1, -1, -2, -2, -1, 1, 2,
                                 2, -1, -1, 2, -2, 1, 1, -2};

// 24: D6 (622)
inline const ClassDesc cls_D6[] = {{"E", K::E, 1, T::None},
                                   {"2C6", K::C6, 2, T::Principal},
                                   {"2C3", K::C3, 2, T::Principal},
                                   {"C2", K::C2, 1, T::Principal},
                                   {"3C2'", K::C2, 3, T::PerpPrimary},
                                   {"3C2''", K::C2, 3, T::PerpSecondary}};
inline const IrrepDesc irr_D6[] = {{"A1", 1, 1, false, true},
                                   {"A2", 1, 1, true, false},
                                   {"B1", 1, 1, false, false},
                                   {"B2", 1, 1, false, false},
                                   {"E1", 2, 1, true, true},
                                   {"E2", 2, 1, false, true}};
inline const int8_t chi_D6[] = {1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, -1, -1,
                                1, -1, 1, -1, 1, -1,
                                1, -1, 1, -1, -1, 1,
                                2, 1, -1, -2, 0, 0,
                                2, -1, -1, 2, 0, 0};

// 25: C6v (6mm)
inline const ClassDesc cls_C6v[] = {{"E", K::E, 1, T::None},
                                    {"2C6", K::C6, 2, T::Principal},
                                    {"2C3", K::C3, 2, T::Principal},
                                    {"C2", K::C2, 1, T::Principal},
                                    {"3sigma_v", K::Sigma, 3, T::PerpPrimary},
                                    {"3sigma_d", K::Sigma, 3, T::PerpSecondary}};
inline const IrrepDesc irr_C6v[] = {{"A1", 1, 1, true, true},
                                    {"A2", 1, 1, false, false},
                                    {"B1", 1, 1, false, false},
                                    {"B2", 1, 1, false, false},
                                    {"E1", 2, 1, true, true},
                                    {"E2", 2, 1, false, true}};
inline const int8_t chi_C6v[] = {1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, -1, -1,
                                 1, -1, 1, -1, 1, -1,
                                 1, -1, 1, -1, -1, 1,
                                 2, 1, -1, -2, 0, 0,
                                 2, -1, -1, 2, 0, 0};

// 26: D3h (-6m2)
inline const ClassDesc cls_D3h[] = {{"E", K::E, 1, T::None},
                                    {"2C3", K::C3, 2, T::Principal},
                                    {"3C2", K::C2, 3, T::PerpPrimary},
                                    {"sigma_h", K::Sigma, 1, T::Principal},
                                    {"2S3", K::S3, 2, T::Principal},
                                    {"3sigma_v", K::Sigma, 3, T::PerpPrimary}};
inline const IrrepDesc irr_D3h[] = {{"A1'", 1, 1, false, true},
                                    {"A2'", 1, 1, false, false},
                                    {"E'", 2, 1, true, true},
                                    {"A1''", 1, 1, false, false},
                                    {"A2''", 1, 1, true, false},
                                    {"E''", 2, 1, false, true}};
inline const int8_t chi_D3h[] = {1, 1, 1, 1, 1, 1,
                                 1, 1, -1, 1, 1, -1,
                                 2, -1, 0, 2, -1, 0,
                                 1, 1, 1, -1, -1, -1,
                                 1, 1, -1, -1, -1, 1,
                                 2, -1, 0, -2, 1, 0};

// 27: D6h (6/mmm).  Following the International-Tables pairing, the
// sigma_v class is i * C2' (mirror normals along the C2' axes, i.e. the
// conventional <100> directions) and sigma_d is i * C2''.
inline const ClassDesc cls_D6h[] = {{"E", K::E, 1, T::None},
                                    {"2C6", K::C6, 2, T::Principal},
                                    {"2C3", K::C3, 2, T::Principal},
                                    {"C2", K::C2, 1, T::Principal},
                                    {"3C2'", K::C2, 3, T::PerpPrimary},
                                    {"3C2''", K::C2, 3, T::PerpSecondary},
                                    {"i", K::I, 1, T::None},
                                    {"2S3", K::S3, 2, T::Principal},
                                    {"2S6", K::S6, 2, T::Principal},
                                    {"sigma_h", K::Sigma, 1, T::Principal},
                                    {"3sigma_v", K::Sigma, 3, T::PerpPrimary},
                                    {"3sigma_d", K::Sigma, 3, T::PerpSecondary}};
inline const IrrepDesc irr_D6h[] = {{"A1g", 1, 1, false, true},
                                    {"A2g", 1, 1, false, false},
                                    {"B1g", 1, 1, false, false},
                                    {"B2g", 1, 1, false, false},
                                    {"E1g", 2, 1, false, true},
                                    {"E2g", 2, 1, false, true},
                                    {"A1u", 1, 1, false, false},
                                    {"A2u", 1, 1, true, false},
                                    {"B1u", 1, 1, false, false},
                                    {"B2u", 1, 1, false, false},
                                    {"E1u", 2, 1, true, false},
                                    {"E2u", 2, 1, false, false}};
inline const int8_t chi_D6h[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1,
                                 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
                                 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1,
                                 2, 1, -1, -2, 0, 0, 2, 1, -1, -2, 0, 0,
                                 2, -1, -1, 2, 0, 0, 2, -1, -1, 2, 0, 0,
                                 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1,
                                 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1,
                                 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1,
                                 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1,
                                 2, 1, -1, -2, 0, 0, -2, -1, 1, 2, 0, 0,
                                 2, -1, -1, 2, 0, 0, -2, 1, 1, -2, 0, 0};

// 28: T (23)
inline const ClassDesc cls_T[] = {{"E", K::E, 1, T::None},
                                  {"8C3", K::C3, 8, T::None},
                                  {"3C2", K::C2, 3, T::None}};
inline const IrrepDesc irr_T[] = {{"A", 1, 1, false, true},
                                  {"E", 2, 2, false, true},
                                  {"T", 3, 1, true, true}};
inline const int8_t chi_T[] = {1, 1, 1,
                               2, -1, 2,
                               3, 0, -1};

// 29: Th (m-3)
inline const ClassDesc cls_Th[] = {{"E", K::E, 1, T::None},
                                   {"8C3", K::C3, 8, T::None},
                                   {"3C2", K::C2, 3, T::None},
                                   {"i", K::I, 1, T::None},
                                   {"8S6", K::S6, 8, T::None},
                                   {"3sigma_h", K::Sigma, 3, T::None}};
inline const IrrepDesc irr_Th[] = {{"Ag", 1, 1, false, true},
                                   {"Eg", 2, 2, false, true},
                                   {"Tg", 3, 1, false, true},
                                   {"Au", 1, 1, false, false},
                                   {"Eu", 2, 2, false, false},
                                   {"Tu", 3, 1, true, false}};
inline const int8_t chi_Th[] = {1, 1, 1, 1, 1, 1,
                                2, -1, 2, 2, -1, 2,
                                3, 0, -1, 3, 0, -1,
                                1, 1, 1, -1, -1, -1,
                                2, -1, 2, -2, 1, -2,
                                3, 0, -1, -3, 0, 1};

// 30: O (432)
inline const ClassDesc cls_O[] = {{"E", K::E, 1, T::None},
                                  {"8C3", K::C3, 8, T::None},
                                  {"6C2", K::C2, 6, T::None},
                                  {"6C4", K::C4, 6, T::None},
                                  {"3C2", K::C2, 3, T::None}};
inline const IrrepDesc irr_O[] = {{"A1", 1, 1, false, true},
                                  {"A2", 1, 1, false, false},
                                  {"E", 2, 1, false, true},
                                  {"T1", 3, 1, true, false},
                                  {"T2", 3, 1, false, true}};
inline const int8_t chi_O[] = {1, 1, 1, 1, 1,
                               1, 1, -1, -1, 1,
                               2, -1, 0, 0, 2,
                               3, 0, -1, 1, -1,
                               3, 0, 1, -1, -1};

// 31: Td (-43m)
inline const ClassDesc cls_Td[] = {{"E", K::E, 1, T::None},
                                   {"8C3", K::C3, 8, T::None},
                                   {"3C2", K::C2, 3, T::None},
                                   {"6S4", K::S4, 6, T::None},
                                   {"6sigma_d", K::Sigma, 6, T::None}};
inline const IrrepDesc irr_Td[] = {{"A1", 1, 1, false, true},
                                   {"A2", 1, 1, false, false},
                                   {"E", 2, 1, false, true},
                                   {"T1", 3, 1, false, false},
                                   {"T2", 3, 1, true, true}};
inline const int8_t chi_Td[] = {1, 1, 1, 1, 1,
                                1, 1, 1, -1, -1,
                                2, -1, 2, 0, 0,
                                3, 0, -1, 1, -1,
                                3, 0, -1, -1, 1};

// 32: Oh (m-3m)
inline const ClassDesc cls_Oh[] = {{"E", K::E, 1, T::None},
                                   {"8C3", K::C3, 8, T::None},
                                   {"6C2", K::C2, 6, T::None},
                                   {"6C4", K::C4, 6, T::None},
                                   {"3C2", K::C2, 3, T::None},
                                   {"i", K::I, 1, T::None},
                                   {"6S4", K::S4, 6, T::None},
                                   {"8S6", K::S6, 8, T::None},
                                   {"3sigma_h", K::Sigma, 3, T::None},
                                   {"6sigma_d", K::Sigma, 6, T::None}};
inline const IrrepDesc irr_Oh[] = {{"A1g", 1, 1, false, true},
                                   {"A2g", 1, 1, false, false},
                                   {"Eg", 2, 1, false, true},
                                   {"T1g", 3, 1, false, false},
                                   {"T2g", 3, 1, false, true},
                                   {"A1u", 1, 1, false, false},
                                   {"A2u", 1, 1, false, false},
                                   {"Eu", 2, 1, false, false},
                                   {"T1u", 3, 1, true, false},
                                   {"T2u", 3, 1, false, false}};
inline const int8_t chi_Oh[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, -1, -1, 1, 1, -1, 1, 1, -1,
                                2, -1, 0, 0, 2, 2, 0, -1, 2, 0,
                                3, 0, -1, 1, -1, 3, 1, 0, -1, -1,
                                3, 0, 1, -1, -1, 3, -1, 0, -1, 1,
                                1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
                                1, 1, -1, -1, 1, -1, 1, -1, -1, 1,
                                2, -1, 0, 0, 2, -2, 0, 1, -2, 0,
                                3, 0, -1, 1, -1, -3, -1, 0, 1, 1,
                                3, 0, 1, -1, -1, -3, 1, 0, 1, -1};

} // namespace detail

inline const PointGroup pg_table[32] = {
    {1, "C1", "1", 1, 1, detail::cls_C1, detail::irr_C1, detail::chi_C1},
    {2, "Ci", "-1", 2, 2, detail::cls_Ci, detail::irr_Ci, detail::chi_Ci},
    {3, "C2", "2", 2, 2, detail::cls_C2, detail::irr_C2, detail::chi_C2},
    {4, "Cs", "m", 2, 2, detail::cls_Cs, detail::irr_Cs, detail::chi_Cs},
    {5, "C2h", "2/m", 4, 4, detail::cls_C2h, detail::irr_C2h, detail::chi_C2h},
    {6, "D2", "222", 4, 4, detail::cls_D2, detail::irr_D2, detail::chi_D2},
    {7, "C2v", "mm2", 4, 4, detail::cls_C2v, detail::irr_C2v, detail::chi_C2v},
    {8, "D2h", "mmm", 8, 8, detail::cls_D2h, detail::irr_D2h, detail::chi_D2h},
    {9, "C4", "4", 4, 3, detail::cls_C4, detail::irr_C4, detail::chi_C4},
    {10, "S4", "-4", 4, 3, detail::cls_S4, detail::irr_S4, detail::chi_S4},
    {11, "C4h", "4/m", 8, 6, detail::cls_C4h, detail::irr_C4h, detail::chi_C4h},
    {12, "D4", "422", 8, 5, detail::cls_D4, detail::irr_D4, detail::chi_D4},
    {13, "C4v", "4mm", 8, 5, detail::cls_C4v, detail::irr_C4v, detail::chi_C4v},
    {14, "D2d", "-42m", 8, 5, detail::cls_D2d, detail::irr_D2d, detail::chi_D2d},
    {15, "D4h", "4/mmm", 16, 10, detail::cls_D4h, detail::irr_D4h, detail::chi_D4h},
    {16, "C3", "3", 3, 2, detail::cls_C3, detail::irr_C3, detail::chi_C3},
    {17, "C3i", "-3", 6, 4, detail::cls_S6, detail::irr_S6, detail::chi_S6},
    {18, "D3", "32", 6, 3, detail::cls_D3, detail::irr_D3, detail::chi_D3},
    {19, "C3v", "3m", 6, 3, detail::cls_C3v, detail::irr_C3v, detail::chi_C3v},
    {20, "D3d", "-3m", 12, 6, detail::cls_D3d, detail::irr_D3d, detail::chi_D3d},
    {21, "C6", "6", 6, 4, detail::cls_C6, detail::irr_C6, detail::chi_C6},
    {22, "C3h", "-6", 6, 4, detail::cls_C3h, detail::irr_C3h, detail::chi_C3h},
    {23, "C6h", "6/m", 12, 8, detail::cls_C6h, detail::irr_C6h, detail::chi_C6h},
    {24, "D6", "622", 12, 6, detail::cls_D6, detail::irr_D6, detail::chi_D6},
    {25, "C6v", "6mm", 12, 6, detail::cls_C6v, detail::irr_C6v, detail::chi_C6v},
    {26, "D3h", "-6m2", 12, 6, detail::cls_D3h, detail::irr_D3h, detail::chi_D3h},
    {27, "D6h", "6/mmm", 24, 12, detail::cls_D6h, detail::irr_D6h, detail::chi_D6h},
    {28, "T", "23", 12, 3, detail::cls_T, detail::irr_T, detail::chi_T},
    {29, "Th", "m-3", 24, 6, detail::cls_Th, detail::irr_Th, detail::chi_Th},
    {30, "O", "432", 24, 5, detail::cls_O, detail::irr_O, detail::chi_O},
    {31, "Td", "-43m", 24, 5, detail::cls_Td, detail::irr_Td, detail::chi_Td},
    {32, "Oh", "m-3m", 48, 10, detail::cls_Oh, detail::irr_Oh, detail::chi_Oh}};

// ---------------------------------------------------------------------------
// Operation classification
// ---------------------------------------------------------------------------

struct OpInfo
{
    OpKind kind = OpKind::E;
    int order = 1;               // order of the group element
    Eigen::Vector3d axis;        // rotation axis / mirror normal (unit); zero for E, i
};

inline int op_order(const OpKind kind)
{
    switch (kind) {
        case OpKind::E: return 1;
        case OpKind::C2: return 2;
        case OpKind::C3: return 3;
        case OpKind::C4: return 4;
        case OpKind::C6: return 6;
        case OpKind::I: return 2;
        case OpKind::Sigma: return 2;
        case OpKind::S6: return 6;
        case OpKind::S4: return 4;
        case OpKind::S3: return 6;
    }
    return 0;
}

// Classify a Cartesian (proper or improper) rotation matrix.  Returns false
// if the matrix is not a crystallographic point operation.
inline bool classify_op(const Eigen::Matrix3d &rot, OpInfo &info, const double tol = 1.0e-6)
{
    const auto det = rot.determinant();
    const auto tr = rot.trace();
    const auto itr = static_cast<int>(std::lround(tr));

    if (std::abs(std::abs(det) - 1.0) > tol) {
        return false;
    }
    if (std::abs(tr - static_cast<double>(itr)) > tol) {
        return false;
    }

    const bool proper = det > 0.0;

    if (proper) {
        switch (itr) {
            case 3: info.kind = OpKind::E;
                break;
            case -1: info.kind = OpKind::C2;
                break;
            case 0: info.kind = OpKind::C3;
                break;
            case 1: info.kind = OpKind::C4;
                break;
            case 2: info.kind = OpKind::C6;
                break;
            default: return false;
        }
    } else {
        switch (itr) {
            case -3: info.kind = OpKind::I;
                break;
            case 1: info.kind = OpKind::Sigma;
                break;
            case 0: info.kind = OpKind::S6;
                break;
            case -1: info.kind = OpKind::S4;
                break;
            case -2: info.kind = OpKind::S3;
                break;
            default: return false;
        }
    }
    info.order = op_order(info.kind);
    info.axis.setZero();

    if (info.kind == OpKind::E || info.kind == OpKind::I) {
        return true;
    }

    // Reduce to a proper rotation; for a mirror the axis of -rot is the
    // mirror normal, for S_n it is the S_n axis.
    const Eigen::Matrix3d prop = proper ? rot : Eigen::Matrix3d(-rot);

    if (std::abs(prop.trace() + 1.0) < tol) {
        // 180-degree rotation: axis from the dominant column of (R + 1)/2,
        // which projects onto the axis.
        const Eigen::Matrix3d proj = 0.5 * (prop + Eigen::Matrix3d::Identity());
        int imax = 0;
        proj.colwise().norm().maxCoeff(&imax);
        info.axis = proj.col(imax).normalized();
    } else {
        // Generic angle: axial vector of the antisymmetric part.
        Eigen::Vector3d v(prop(2, 1) - prop(1, 2),
                          prop(0, 2) - prop(2, 0),
                          prop(1, 0) - prop(0, 1));
        if (v.norm() < tol) {
            return false;
        }
        info.axis = v.normalized();
    }
    return true;
}

// ---------------------------------------------------------------------------
// Point-group identification by operation-count fingerprint
// ---------------------------------------------------------------------------

inline std::array<int, NKIND> fingerprint_of_table(const PointGroup &pg)
{
    std::array<int, NKIND> counts{};
    for (int ic = 0; ic < pg.nclass; ++ic) {
        counts[static_cast<int>(pg.classes[ic].kind)] += pg.classes[ic].nelem;
    }
    return counts;
}

// Returns the point-group number (1..32), or -1 when the operation multiset
// does not match any crystallographic point group.
inline int identify_point_group_fingerprint(const std::vector<Eigen::Matrix3d> &rots_cart)
{
    std::array<int, NKIND> counts{};
    OpInfo info;
    for (const auto &rot: rots_cart) {
        if (!classify_op(rot, info)) {
            return -1;
        }
        counts[static_cast<int>(info.kind)] += 1;
    }
    for (const auto &pg: pg_table) {
        if (fingerprint_of_table(pg) == counts) {
            return pg.number;
        }
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Conjugacy classes in exact integer arithmetic (lattice-basis rotations)
// ---------------------------------------------------------------------------

inline int int_det3(const Eigen::Matrix3i &m)
{
    return m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1))
           - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0))
           + m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
}

// Inverse of a unimodular integer matrix (|det| = 1), exact.
inline Eigen::Matrix3i int_inverse(const Eigen::Matrix3i &m)
{
    Eigen::Matrix3i adj;
    adj(0, 0) = m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1);
    adj(0, 1) = m(0, 2) * m(2, 1) - m(0, 1) * m(2, 2);
    adj(0, 2) = m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1);
    adj(1, 0) = m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2);
    adj(1, 1) = m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0);
    adj(1, 2) = m(0, 2) * m(1, 0) - m(0, 0) * m(1, 2);
    adj(2, 0) = m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0);
    adj(2, 1) = m(0, 1) * m(2, 0) - m(0, 0) * m(2, 1);
    adj(2, 2) = m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
    const auto det = int_det3(m);
    return (det == 1) ? adj : Eigen::Matrix3i(-adj);
}

// Geometric conjugacy classes of {rot_latt}, merged under R ~ R^-1.
// Returns lists of indices into the input vector.
inline std::vector<std::vector<int>> conjugacy_classes_merged(const std::vector<Eigen::Matrix3i> &rot_latt)
{
    const auto nsym = static_cast<int>(rot_latt.size());

    auto find_index = [&rot_latt, nsym](const Eigen::Matrix3i &m) -> int {
        for (int i = 0; i < nsym; ++i) {
            if (rot_latt[i] == m) {
                return i;
            }
        }
        return -1;
    };

    std::vector<int> class_id(nsym, -1);
    std::vector<std::vector<int>> classes;

    for (int i = 0; i < nsym; ++i) {
        if (class_id[i] >= 0) {
            continue;
        }

        const auto id = static_cast<int>(classes.size());
        std::vector<int> members;

        auto absorb = [&](const int seed) {
            for (int ig = 0; ig < nsym; ++ig) {
                const Eigen::Matrix3i conj = rot_latt[ig] * rot_latt[seed] * int_inverse(rot_latt[ig]);
                const auto idx = find_index(conj);
                if (idx < 0) { // not closed: not a group
                    return false;
                }
                if (class_id[idx] < 0) {
                    class_id[idx] = id;
                    members.push_back(idx);
                }
            }
            return true;
        };

        class_id[i] = id;
        members.push_back(i);
        if (!absorb(i)) {
            return {};
        }

        // Merge with the class of the inverse element.
        const auto inv_idx = find_index(int_inverse(rot_latt[i]));
        if (inv_idx < 0) {
            return {};
        }
        if (class_id[inv_idx] < 0) {
            class_id[inv_idx] = id;
            members.push_back(inv_idx);
        }
        if (!absorb(inv_idx)) {
            return {};
        }

        classes.push_back(std::move(members));
    }
    return classes;
}

// ---------------------------------------------------------------------------
// Class -> table-column matching
// ---------------------------------------------------------------------------

struct MatchResult
{
    bool ok = false;                 // (kind, nelem) buckets consistent and validated
    bool convention_resolved = true; // false => B1/B2-type column identities rest on an
                                     //          unresolved axis convention
    std::vector<int> class_to_col;   // numerical class index -> table column
    std::vector<int> ambiguous_cols; // columns whose identity needed the geometric rule
    std::string note;                // human-readable statement of the convention used
    std::vector<std::string> warnings;
};

namespace detail
{

// Multiplicities of all table irreps in the vector representation, from the
// per-class traces of representative operations.  Non-negative integers for
// a correct class->column assignment.
inline bool vector_rep_decomposition_is_integral(const PointGroup &pg,
                                                 const std::vector<int> &class_to_col,
                                                 const std::vector<double> &trace_per_class,
                                                 const std::vector<int> &nelem_per_class,
                                                 const double tol = 1.0e-6)
{
    const auto ncl = static_cast<int>(class_to_col.size());
    for (int mu = 0; mu < pg.nclass; ++mu) {
        double n = 0.0;
        for (int ic = 0; ic < ncl; ++ic) {
            n += static_cast<double>(nelem_per_class[ic]) * trace_per_class[ic]
                 * static_cast<double>(pg.chartab[mu * pg.nclass + class_to_col[ic]]);
        }
        n /= static_cast<double>(pg.order * pg.irreps[mu].norm);
        if (n < -tol || std::abs(n - std::lround(n)) > tol) {
            return false;
        }
    }
    return true;
}

} // namespace detail

// Match the numerically found merged classes to the table columns of pg.
//
//  ops_info:  classification of every operation (index-aligned with the
//             operation list used to build `classes`).
//  zhat_hint: principal-axis direction (zero => derive from the operations).
//  xhat_hint: reference in-plane direction, the conventional a axis projected
//             perpendicular to the principal axis (zero => the convention-
//             dependent distinctions are left unresolved and flagged).
inline MatchResult match_classes_to_columns(const PointGroup &pg,
                                            const std::vector<std::vector<int>> &classes,
                                            const std::vector<OpInfo> &ops_info,
                                            const Eigen::Vector3d &zhat_hint = Eigen::Vector3d::Zero(),
                                            const Eigen::Vector3d &xhat_hint = Eigen::Vector3d::Zero())
{
    MatchResult res;
    const auto ncl = static_cast<int>(classes.size());
    constexpr double align_tol = 1.0e-4;

    if (ncl != pg.nclass) {
        res.warnings.emplace_back("number of merged classes does not match the character table");
        return res;
    }

    // Per-class info.
    std::vector<OpKind> kind_of(ncl);
    std::vector<int> nelem_of(ncl);
    for (int ic = 0; ic < ncl; ++ic) {
        nelem_of[ic] = static_cast<int>(classes[ic].size());
        kind_of[ic] = ops_info[classes[ic][0]].kind;
        for (const auto im: classes[ic]) {
            if (ops_info[im].kind != kind_of[ic]) {
                res.warnings.emplace_back("inconsistent operation kinds inside a conjugacy class");
                return res;
            }
        }
    }

    // Principal axis.
    Eigen::Vector3d zhat = zhat_hint;
    if (zhat.norm() < 0.5) {
        int best_order = 1;
        for (const auto &op: ops_info) {
            if (op.kind == OpKind::E || op.kind == OpKind::I) {
                continue;
            }
            if (op.order > best_order) {
                best_order = op.order;
                zhat = op.axis;
            }
        }
        // Among same-order candidates prefer a proper rotation axis; for the
        // groups where the distinction matters (S4 vs C2 in D2d) the S4 axis
        // has the higher order, so the loop above already found it.
    }
    const bool has_z = zhat.norm() > 0.5;

    Eigen::Vector3d xhat = xhat_hint;
    if (has_z && xhat.norm() > 0.5) {
        xhat -= xhat.dot(zhat) * zhat; // enforce orthogonality
        if (xhat.norm() > 1.0e-8) {
            xhat.normalize();
        } else {
            xhat.setZero();
        }
    }
    const bool has_x = xhat.norm() > 0.5;

    auto axis_dot = [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) {
        return std::abs(a.dot(b));
    };

    // Primary star: the orbit of x^ under the proper rotations about z^,
    // i.e. the conventional <100> in-plane directions.  A C2' axis and a
    // sigma_v normal both lie along this star (International-Tables
    // convention: the position-2 mirrors are perpendicular to <100>).
    std::vector<Eigen::Vector3d> star;
    if (has_x && has_z) {
        auto nprin = 1;
        for (const auto &op: ops_info) {
            const auto proper = op.kind == OpKind::C2 || op.kind == OpKind::C3
                                || op.kind == OpKind::C4 || op.kind == OpKind::C6;
            if (proper && axis_dot(op.axis, zhat) > 1.0 - 1.0e-6) {
                nprin = std::max(nprin, op.order);
            }
        }
        constexpr double two_pi = 6.283185307179586;
        for (auto k = 0; k < nprin; ++k) {
            star.emplace_back(Eigen::AngleAxisd(two_pi * k / nprin, zhat) * xhat);
        }
    }

    // Distinguished directions of every member (rotation axes / mirror
    // normals) against the primary star: ~1 for C2'/sigma_v classes.
    auto star_metric = [&](const int ic) {
        double best = -1.0;
        for (const auto im: classes[ic]) {
            for (const auto &d: star) {
                best = std::max(best, axis_dot(ops_info[im].axis, d));
            }
        }
        return best;
    };

    // Bucket numerical classes and table columns by (kind, nelem); ambiguity
    // is resolved JOINTLY inside each bucket (the classes of a bucket are
    // compared against each other, never scored in isolation).
    std::vector<int> col_of_class(ncl, -1);
    std::vector<char> class_done(ncl, 0);

    for (int ic0 = 0; ic0 < ncl; ++ic0) {
        if (class_done[ic0]) {
            continue;
        }

        std::vector<int> bucket_classes, bucket_cols;
        for (int ic = 0; ic < ncl; ++ic) {
            if (kind_of[ic] == kind_of[ic0] && nelem_of[ic] == nelem_of[ic0]) {
                bucket_classes.push_back(ic);
            }
        }
        for (int col = 0; col < pg.nclass; ++col) {
            if (pg.classes[col].kind == kind_of[ic0] && pg.classes[col].nelem == nelem_of[ic0]) {
                bucket_cols.push_back(col);
            }
        }
        if (bucket_classes.size() != bucket_cols.size() || bucket_cols.empty()) {
            res.warnings.emplace_back("class/column bucket size mismatch against the character table");
            return res;
        }
        for (const auto ic: bucket_classes) {
            class_done[ic] = 1;
        }

        if (bucket_cols.size() == 1) {
            col_of_class[bucket_classes[0]] = bucket_cols[0];
            continue;
        }

        for (const auto col: bucket_cols) {
            res.ambiguous_cols.push_back(col);
        }

        auto fallback = [&]() {
            for (std::size_t k = 0; k < bucket_classes.size(); ++k) {
                col_of_class[bucket_classes[k]] = bucket_cols[k];
            }
            res.convention_resolved = false;
        };

        // Collect the column tags of this bucket.
        auto has_tag = [&](const AxisTag t) {
            for (const auto col: bucket_cols) {
                if (pg.classes[col].tag == t) {
                    return true;
                }
            }
            return false;
        };

        if (bucket_cols.size() == 2 && has_tag(AxisTag::PerpPrimary) && has_tag(AxisTag::PerpSecondary)) {
            // C2'/C2'' or sigma_v/sigma_d: the class whose distinguished
            // directions lie along the primary star is primary.
            if (star.empty()) {
                fallback();
                continue;
            }
            const int ca = bucket_classes[0], cb = bucket_classes[1];
            const double ma = star_metric(ca);
            const double mb = star_metric(cb);
            if (std::abs(ma - mb) < align_tol) {
                fallback();
                res.warnings.emplace_back("degenerate geometry while separating primed/double-primed classes");
                continue;
            }
            const int primary_class = (ma > mb) ? ca : cb;
            const int secondary_class = (ma > mb) ? cb : ca;
            for (const auto col: bucket_cols) {
                if (pg.classes[col].tag == AxisTag::PerpPrimary) {
                    col_of_class[primary_class] = col;
                } else {
                    col_of_class[secondary_class] = col;
                }
            }
            continue;
        }

        // AxisZ/AxisX/AxisY buckets (D2, C2v, D2h): every class has a single
        // member whose distinguished axis (rotation axis or mirror normal)
        // must align with the tagged direction.  Solve the small assignment
        // problem over all permutations.
        {
            bool all_axis_tags = true;
            bool need_x = false;
            for (const auto col: bucket_cols) {
                const auto t = pg.classes[col].tag;
                if (t != AxisTag::AxisZ && t != AxisTag::AxisX && t != AxisTag::AxisY) {
                    all_axis_tags = false;
                }
                if (t != AxisTag::AxisZ) {
                    need_x = true;
                }
            }
            if (!all_axis_tags || !has_z || (need_x && !has_x)) {
                fallback();
                continue;
            }
            const Eigen::Vector3d yhat = has_x ? Eigen::Vector3d(zhat.cross(xhat)) : Eigen::Vector3d::Zero();
            auto dir_of = [&](const AxisTag t) -> Eigen::Vector3d {
                if (t == AxisTag::AxisZ) {
                    return zhat;
                }
                if (t == AxisTag::AxisX) {
                    return xhat;
                }
                return yhat;
            };

            std::vector<int> order(bucket_cols.size());
            for (std::size_t k = 0; k < order.size(); ++k) {
                order[k] = static_cast<int>(k);
            }
            std::vector<int> best_perm;
            double best_total = -1.0e30;
            std::sort(order.begin(), order.end());
            do {
                double total = 0.0;
                for (std::size_t k = 0; k < order.size(); ++k) {
                    total += axis_dot(ops_info[classes[bucket_classes[k]][0]].axis,
                                      dir_of(pg.classes[bucket_cols[order[k]]].tag));
                }
                if (total > best_total) {
                    best_total = total;
                    best_perm = order;
                }
            } while (std::next_permutation(order.begin(), order.end()));

            bool aligned = true;
            for (std::size_t k = 0; k < best_perm.size(); ++k) {
                const auto d = axis_dot(ops_info[classes[bucket_classes[k]][0]].axis,
                                        dir_of(pg.classes[bucket_cols[best_perm[k]]].tag));
                if (d < 1.0 - align_tol) {
                    aligned = false;
                }
            }
            if (!aligned) {
                fallback();
                res.warnings.emplace_back("class axes do not align with the conventional frame");
                continue;
            }
            for (std::size_t k = 0; k < best_perm.size(); ++k) {
                col_of_class[bucket_classes[k]] = bucket_cols[best_perm[k]];
            }
        }
    }

    res.class_to_col = col_of_class;

    if (has_z && has_x) {
        res.note = "principal axis z || (" + std::to_string(zhat.x()) + ", " + std::to_string(zhat.y())
                   + ", " + std::to_string(zhat.z()) + "); reference in-plane axis x || ("
                   + std::to_string(xhat.x()) + ", " + std::to_string(xhat.y()) + ", "
                   + std::to_string(xhat.z()) + "). Primed/double-primed and B1/B2-type labels "
                   "refer to this frame.";
    } else if (!res.ambiguous_cols.empty()) {
        res.note = "conventional reference frame unavailable; convention-dependent labels are "
                   "not resolved.";
    }

    res.ok = true;
    return res;
}

// All class->column bijections consistent with the (kind, nelem) bucket
// structure.  The geometric matcher picks one of these; representation-level
// validation (integral, non-negative decompositions of the vector, total,
// and multiplet representations) selects among them.  Empty on bucket
// mismatch.
inline std::vector<std::vector<int>> enumerate_consistent_assignments(
    const PointGroup &pg,
    const std::vector<std::vector<int>> &classes,
    const std::vector<OpInfo> &ops_info)
{
    const auto ncl = static_cast<int>(classes.size());
    if (ncl != pg.nclass) {
        return {};
    }

    std::vector<std::vector<int>> assignments{std::vector<int>(ncl, -1)};
    std::vector<char> class_done(ncl, 0);

    for (auto ic0 = 0; ic0 < ncl; ++ic0) {
        if (class_done[ic0]) {
            continue;
        }
        const auto kind = ops_info[classes[ic0][0]].kind;
        const auto nelem = static_cast<int>(classes[ic0].size());

        std::vector<int> bucket_classes, bucket_cols;
        for (auto ic = 0; ic < ncl; ++ic) {
            if (ops_info[classes[ic][0]].kind == kind
                && static_cast<int>(classes[ic].size()) == nelem) {
                bucket_classes.push_back(ic);
                class_done[ic] = 1;
            }
        }
        for (auto col = 0; col < pg.nclass; ++col) {
            if (pg.classes[col].kind == kind && pg.classes[col].nelem == nelem) {
                bucket_cols.push_back(col);
            }
        }
        if (bucket_classes.size() != bucket_cols.size() || bucket_cols.empty()) {
            return {};
        }

        std::vector<int> perm(bucket_cols.size());
        for (std::size_t k = 0; k < perm.size(); ++k) {
            perm[k] = static_cast<int>(k);
        }
        std::vector<std::vector<int>> expanded;
        do {
            for (const auto &base: assignments) {
                auto next = base;
                for (std::size_t k = 0; k < bucket_classes.size(); ++k) {
                    next[bucket_classes[k]] = bucket_cols[perm[k]];
                }
                expanded.push_back(std::move(next));
                if (expanded.size() > 1296) {
                    return {}; // combinatorial blow-up: not a crystallographic case
                }
            }
        } while (std::next_permutation(perm.begin(), perm.end()));
        assignments = std::move(expanded);
    }
    return assignments;
}

// ---------------------------------------------------------------------------
// Decomposition helper
// ---------------------------------------------------------------------------

// Multiplicity of every table irrep in a representation given by its
// per-class characters:  n_mu = (1/(|G| norm_mu)) sum_c w_c chi(c) chi_mu(c).
inline std::vector<double> decompose_representation(const PointGroup &pg,
                                                    const std::vector<int> &class_to_col,
                                                    const std::vector<int> &nelem_per_class,
                                                    const std::vector<double> &chi_per_class)
{
    std::vector<double> n_mu(pg.nclass, 0.0);
    const auto ncl = static_cast<int>(class_to_col.size());
    for (int mu = 0; mu < pg.nclass; ++mu) {
        double n = 0.0;
        for (int ic = 0; ic < ncl; ++ic) {
            n += static_cast<double>(nelem_per_class[ic]) * chi_per_class[ic]
                 * static_cast<double>(pg.chartab[mu * pg.nclass + class_to_col[ic]]);
        }
        n_mu[mu] = n / static_cast<double>(pg.order * pg.irreps[mu].norm);
    }
    return n_mu;
}

// Validate a class->column assignment against the integrality of the
// vector-representation decomposition (cheap sanity filter; the outer
// automorphisms that swap B1/B2-type labels pass this by construction).
inline bool validate_assignment_by_vector_rep(const PointGroup &pg,
                                              const std::vector<int> &class_to_col,
                                              const std::vector<int> &nelem_per_class,
                                              const std::vector<double> &trace_per_class)
{
    return detail::vector_rep_decomposition_is_integral(pg, class_to_col, trace_per_class,
                                                        nelem_per_class);
}

} // namespace pointgroup
} // namespace PHON_NS
