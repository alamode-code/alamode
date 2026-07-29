/*
 mode_symmetry.h

 Copyright (c) 2026 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory
 or http://opensource.org/licenses/mit-license.php for information.
*/

#pragma once

#include <string>
#include <vector>

#include <Eigen/Core>

#include "pointers.h"

namespace PHON_NS
{
struct GammaClassInfo
{
    std::string label;                 // "2C4", "3sigma_v", ...
    int nelem = 0;                     // number of group elements in the (R ~ R^-1 merged) class
    std::vector<Eigen::Vector3d> axes; // rotation axes / mirror normals (Cartesian)
};

struct GammaModeGroup
{
    std::vector<int> mode_indices;  // 0-based branch indices, consecutive
    double omega = 0.0;             // signed frequency, Ry a.u. (mean over the multiplet)
    std::string irrep_label;        // "T1u", "Eg(+)A1g" for accidental degeneracy, "??" on failure
    std::vector<double> characters; // chi_lambda per merged class (real part)
    double max_imag_char = 0.0;
    double max_class_spread = 0.0;
    double acoustic_content = 0.0;  // tr(P_T P_lambda), in [0, 3]
    bool is_acoustic = false;
    bool ir_active = false;
    bool raman_active = false;
    bool activity_known = false;    // false when subspace closure / integrality failed
    double n_ir_proj = 0.0;         // raw projection weights onto Gamma_V and
    double n_raman_proj = 0.0;      // [Gamma_V x Gamma_V]_sym (approximate when
                                    // activity_known is false)
    Eigen::Matrix3d ir_strength;    // oscillator-strength tensor S_ab; valid iff has_ir_strength
    bool has_ir_strength = false;
};

struct GammaIrrepResult
{
    bool available = false;         // false => analysis aborted (labels suppressed)
    std::string pg_schoenflies;     // "Oh"
    std::string pg_international;   // "m-3m"
    int pg_number = 0;              // point-group number 1..32
    std::string spg_symbol;         // international space-group symbol (when known)
    std::vector<GammaClassInfo> classes;
    std::vector<GammaModeGroup> groups;
    std::string decomp_total, decomp_acoustic, decomp_optic;
    bool has_borncharge = false;
    std::vector<std::string> warnings;
    std::string axis_convention_note;
};

class ModeSymmetry : protected Pointers
{
public:
    ModeSymmetry(class PHON *phon);

    ~ModeSymmetry();

    bool print_irreps = false; // IRREPS

    // Collective (broadcasts the flag); call in PHON::setup_base() before
    // Dielec::init() so the Born-charge loading trigger can see the flag.
    void setup();

    // Rank 0 only: SymmListWithMap exists only on rank 0.
    void analyze_irreps_at_gamma();

    [[nodiscard]] const GammaIrrepResult &get_result() const
    {
        return result_;
    }

private:
    GammaIrrepResult result_;
};
} // namespace PHON_NS
